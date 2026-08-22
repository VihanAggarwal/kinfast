# src/kinfast/mjcf/parse.py
"""Parse a useful subset of MJCF (MuJoCo XML) into the same Robot IR that URDF
uses, so every backend (batched torch, dynamics, the compiler) works on MuJoCo
models unchanged.

MJCF semantics this parser gets right (each one is a real trap):
- `<compiler angle=...>` defaults to DEGREES; hinge ranges and euler orientations
  are converted. Slide ranges are meters and are not converted.
- Body orientation: `quat` is (w, x, y, z); `euler` follows eulerseq, default
  "xyz" meaning INTRINSIC x-y-z, i.e. R = Rx(a) @ Ry(b) @ Rz(c). (URDF rpy is
  extrinsic xyz, which is the reverse composition; they are not the same.)
- Joint `pos` is a rotation ANCHOR inside the body: the body-frame motion is
  trans(jp) @ R(q) @ trans(-jp). URDF has no anchor concept, so nonzero anchors
  are expressed by inserting synthetic intermediate links.
- `<default>` classes: joint attribute resolution is explicit attr > joint's
  class > innermost body childclass > global default.
- Joint type defaults to hinge; axis defaults to (0, 0, 1).
- `free` joints (floating bases, common in quadruped models) are treated as
  fixed to the world; a note is recorded in `robot.parse_notes`. `ball` joints
  are not supported and raise.

Unknown elements (geoms, sites, actuators, sensors, assets, includes) are
ignored: kinematics needs bodies, joints, and inertials.
"""
import math
import xml.etree.ElementTree as ET

from kinfast.ir import Robot, Link, Joint, Inertial


def _floats(text, n=None):
    vals = tuple(float(x) for x in text.split())
    if n is not None and len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return vals


# ---- rotation helpers (pure python; small and only run at parse time) ----
def _quat_to_mat(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def _axis_rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return [[1, 0, 0], [0, c, -s], [0, s, c]]
    if axis == "y":
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _euler_to_mat(angles, seq):
    """MJCF euler: lowercase letters are intrinsic (compose left to right)."""
    R = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    for ax, ang in zip(seq, angles):
        rot = _axis_rot(ax.lower(), ang)
        R = _matmul(R, rot) if ax.islower() else _matmul(rot, R)
    return R


def _mat_to_rpy(R):
    """Rotation matrix -> URDF-style extrinsic xyz rpy (what the IR stores)."""
    sy = math.sqrt(R[0][0] ** 2 + R[1][0] ** 2)
    if sy > 1e-9:
        return (math.atan2(R[2][1], R[2][2]),
                math.atan2(-R[2][0], sy),
                math.atan2(R[1][0], R[0][0]))
    return (math.atan2(-R[1][2], R[1][1]), math.atan2(-R[2][0], sy), 0.0)


class _Ctx:
    def __init__(self, root):
        comp = root.find("compiler")
        angle = comp.get("angle", "degree") if comp is not None else "degree"
        self.ang = math.pi / 180.0 if angle == "degree" else 1.0
        self.eulerseq = (comp.get("eulerseq", "xyz") if comp is not None
                         else "xyz")
        self.defaults = {}          # class name -> {attr: value} for joints
        self._collect_defaults(root.find("default"), "", {})
        self.notes = []

    def _collect_defaults(self, el, name, inherited):
        if el is None:
            return
        d = dict(inherited)
        j = el.find("joint")
        if j is not None:
            d.update(j.attrib)
        self.defaults[name] = d
        for child in el.findall("default"):
            self._collect_defaults(child, child.get("class", ""), d)

    def jattr(self, el, attr, childclass):
        v = el.get(attr)
        if v is not None:
            return v
        for cls in (el.get("class"), childclass, ""):
            if cls is not None and cls in self.defaults:
                v = self.defaults[cls].get(attr)
                if v is not None:
                    return v
        return None


def _body_orient_rpy(el, ctx):
    if el.get("quat"):
        return _mat_to_rpy(_quat_to_mat(*_floats(el.get("quat"), 4)))
    if el.get("euler"):
        ang = [a * ctx.ang for a in _floats(el.get("euler"), 3)]
        return _mat_to_rpy(_euler_to_mat(ang, ctx.eulerseq))
    if el.get("axisangle"):
        x, y, z, a = _floats(el.get("axisangle"), 4)
        n = math.sqrt(x * x + y * y + z * z) or 1.0
        a *= ctx.ang
        s = math.sin(a / 2)
        return _mat_to_rpy(_quat_to_mat(math.cos(a / 2), x / n * s, y / n * s,
                                        z / n * s))
    return (0.0, 0.0, 0.0)


def _parse_inertial(el, ctx):
    if el is None:
        return None
    mass = float(el.get("mass", 0.0))
    com = _floats(el.get("pos", "0 0 0"), 3)
    if el.get("fullinertia"):
        i = _floats(el.get("fullinertia"), 6)   # ixx iyy izz ixy ixz iyz
    elif el.get("diaginertia"):
        d = _floats(el.get("diaginertia"), 3)
        i = (d[0], d[1], d[2], 0.0, 0.0, 0.0)
    else:
        i = (0.0,) * 6
    # the inertial frame may be rotated (quat/euler): express I in the body frame
    if el.get("quat") or el.get("euler") or el.get("axisangle"):
        from kinfast.urdf.parse import rotate_inertia
        R = _euler_to_mat(_body_orient_rpy(el, ctx), "XYZ")
        i = rotate_inertia(i, R)
    return Inertial(mass, com, i)


def parse_mjcf_string(text: str) -> Robot:
    root = ET.fromstring(text)
    if root.tag != "mujoco":
        raise ValueError(f"root element must be <mujoco>, got <{root.tag}>")
    ctx = _Ctx(root)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("no <worldbody>")

    robot = Robot(root.get("model", "mjcf_robot"),
                  links={"world": Link("world")}, joints=[])
    counter = [0]

    def unique(base):
        counter[0] += 1
        return f"{base}__{counter[0]}"

    def walk(body_el, parent_link, childclass, body_i):
        name = body_el.get("name") or unique("body")
        childclass = body_el.get("childclass", childclass)
        pos = _floats(body_el.get("pos", "0 0 0"), 3)
        rpy = _body_orient_rpy(body_el, ctx)

        joints = [j for j in body_el.findall("joint")]
        if body_el.find("freejoint") is not None:
            ctx.notes.append(f"body {name}: freejoint treated as fixed base")

        # The body transform is trans(pos) @ R @ prod_i[trans(p_i) M_i trans(-p_i)].
        # Our IR joint origin is trans(xyz) @ R(rpy) with motion after it, so:
        # first joint origin: trans(pos + R@p_1) @ R    (anchor rotated by R)
        # joint i>1 origin:   trans(p_i - p_{i-1})       (previous -p folds in)
        # final body link:    fixed joint at trans(-p_last)
        R_body = _euler_to_mat(rpy, "XYZ")   # extrinsic xyz rpy -> matrix
        cur_parent = parent_link
        prev_anchor = None                    # anchor of the previous joint
        first = True

        for jel in joints:
            jtype = ctx.jattr(jel, "type", childclass) or "hinge"
            if jtype == "ball":
                raise ValueError(f"body {name}: ball joints not supported")
            if jtype == "free":
                ctx.notes.append(f"body {name}: free joint treated as fixed")
                continue
            axis = _floats(ctx.jattr(jel, "axis", childclass) or "0 0 1", 3)
            anchor = _floats(ctx.jattr(jel, "pos", childclass) or "0 0 0", 3)
            rng = ctx.jattr(jel, "range", childclass)
            if rng:
                lo, hi = _floats(rng, 2)
                if jtype == "hinge":
                    lo, hi = lo * ctx.ang, hi * ctx.ang
            else:
                lo = hi = 0.0
            jname = jel.get("name") or unique(f"{name}_joint")

            if first:
                rot_a = [sum(R_body[i][k] * anchor[k] for k in range(3))
                         for i in range(3)]
                xyz = tuple(pos[i] + rot_a[i] for i in range(3))
                jrpy = rpy
            else:
                xyz = tuple(anchor[i] - prev_anchor[i] for i in range(3))
                jrpy = (0.0, 0.0, 0.0)

            last = jel is joints[-1]
            zero_anchor = all(abs(a) < 1e-12 for a in anchor)
            child = name if (last and zero_anchor) else unique(f"{name}__jnt")
            robot.links.setdefault(child, Link(child))
            robot.joints.append(Joint(
                jname, "revolute" if jtype == "hinge" else "prismatic",
                cur_parent, child, xyz, jrpy, axis, (lo, hi),
                velocity=0.0, effort=0.0))
            cur_parent = child
            prev_anchor = anchor
            first = False

        if cur_parent != name:
            # attach the real body link: no joints at all (fixed at pos/rpy),
            # or an anchored last joint (fixed tail at -p_last)
            if name not in robot.links:
                robot.links[name] = Link(name)
            if cur_parent is parent_link:
                xyz, jrpy = pos, rpy
            else:
                xyz = tuple(-a for a in prev_anchor)
                jrpy = (0.0, 0.0, 0.0)
            robot.joints.append(Joint(unique(f"{name}_fix"), "fixed",
                                      cur_parent, name, xyz, jrpy))
        robot.links[name].inertial = _parse_inertial(body_el.find("inertial"), ctx)

        for sub in body_el.findall("body"):
            walk(sub, name, childclass, counter)

    for body in world.findall("body"):
        walk(body, "world", "", counter)

    robot.parse_notes = ctx.notes
    return robot


def parse_mjcf_file(path: str) -> Robot:
    with open(path, "r", encoding="utf-8") as f:
        return parse_mjcf_string(f.read())
