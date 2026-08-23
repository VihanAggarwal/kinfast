# src/kinfast/mjcf/emit.py
"""Emit MJCF (MuJoCo XML) from the Robot IR: a URDF -> MJCF converter.

What it reproduces, and how it is verified:
- kinematic tree, joint types/axes/limits, and inertials, emitted in radians
  with body orientations as quaternions (no euler-convention ambiguity); the
  emitted model is loaded by MuJoCo and its forward kinematics compared
  body-by-body against kinfast's own FK of the source URDF (tests).
- geometry: box/cylinder/sphere primitives (using MJCF's half-extent
  conventions) and mesh references as <asset> entries; visual geoms go in
  group 1 with contacts off, collision geoms in group 0.

Things to know: MuJoCo refuses massless moving bodies, so a body without an
inertial gets a tiny placeholder (flagged in a comment). A link that carries a
real inertia tensor but zero mass (URDFs use these for flanges and tool frames)
keeps its tensor and only borrows the placeholder mass, so the emitted mass
matrix still matches kinfast's. `package://pkg/...` mesh paths are rewritten to
the path after the package name, relative to the `meshdir` you pass. MuJoCo
loads STL/OBJ/MSH meshes; DAE is not supported by MuJoCo itself.

MuJoCo also reserves the body name "world" for the top-level worldbody. A URDF
root link called `world` is therefore not emitted as a body of its own: it is
welded to the world frame at identity anyway, so its subtree is emitted
directly under <worldbody>. A non-root link named `world` is renamed instead.
"""
import math
import re
from xml.sax.saxutils import quoteattr

from kinfast.ir import Robot, MOVABLE
from kinfast.urdf.parse import _rpy_to_mat


def _mat_to_quat(R):
    """Rotation matrix -> (w, x, y, z), MuJoCo's order."""
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s,
                (R[1][0] - R[0][1]) / s)
    if R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        return ((R[2][1] - R[1][2]) / s, 0.25 * s, (R[0][1] + R[1][0]) / s,
                (R[0][2] + R[2][0]) / s)
    if R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        return ((R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, 0.25 * s,
                (R[1][2] + R[2][1]) / s)
    s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
    return ((R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s,
            (R[1][2] + R[2][1]) / s, 0.25 * s)


def _f(vals):
    return " ".join(f"{v:.9g}" for v in vals)


def _pose_attrs(xyz, rpy):
    out = f' pos="{_f(xyz)}"'
    if any(abs(a) > 1e-12 for a in rpy):
        out += f' quat="{_f(_mat_to_quat(_rpy_to_mat(*rpy)))}"'
    return out


_PKG = re.compile(r"^package://[^/]+/")

# MuJoCo's top-level body is always named "world"; no other body may take it.
_WORLD = "world"
# Mass we hand MuJoCo for a link whose source says zero mass but gives a real
# inertia tensor (URDFs use these for flanges and tool frames). MuJoCo refuses
# a massless moving body, so it needs something positive; this is small enough
# that the mass it adds stays far below any dynamics tolerance, and it lets the
# real tensor through instead of the 1e-6 placeholder.
_INERTIA_ONLY_MASS = 1e-9


def _mesh_file(path, meshdir_strip):
    p = _PKG.sub("", path)
    if meshdir_strip and p.startswith(meshdir_strip):
        p = p[len(meshdir_strip):].lstrip("/")
    return p


def emit_mjcf(robot: Robot, geometry: bool = True, meshdir: str = "",
              meshdir_strip: str = "") -> str:
    """Robot IR -> MJCF string. geometry=False emits a kinematics+inertia-only
    model (loads anywhere, no assets needed)."""
    root = robot.root_link()
    children = {}
    for j in robot.joints:
        children.setdefault(j.parent, []).append(j)

    assets = {}      # mesh name -> (file, scale)
    lines = []

    def mesh_name(path):
        base = re.sub(r"[^A-Za-z0-9_]+", "_", _PKG.sub("", path))
        name, k = base, 1
        while name in assets and assets[name][0] != path:
            k += 1
            name = f"{base}_{k}"
        return name

    def geom_lines(link, indent):
        out = []
        for role, g in (("visual", link.visual), ("collision", link.collision)):
            if g is None or g.kind == "none":
                continue
            contact = (' contype="0" conaffinity="0" group="1"' if role == "visual"
                       else ' group="0"')
            pose = _pose_attrs(g.origin_xyz, g.origin_rpy)
            if g.kind == "mesh" and g.mesh_path:
                name = mesh_name(g.mesh_path)
                assets[name] = (g.mesh_path, g.scale)
                out.append(f'{indent}<geom type="mesh" mesh={quoteattr(name)}{pose}{contact}/>')
            elif g.kind == "box" and len(g.size) == 3:
                half = [s / 2 for s in g.size]
                out.append(f'{indent}<geom type="box" size="{_f(half)}"{pose}{contact}/>')
            elif g.kind == "cylinder" and len(g.size) == 2:
                out.append(f'{indent}<geom type="cylinder" '
                           f'size="{_f([g.size[0], g.size[1] / 2])}"{pose}{contact}/>')
            elif g.kind == "sphere" and len(g.size) == 1:
                out.append(f'{indent}<geom type="sphere" size="{_f(g.size)}"{pose}{contact}/>')
        return out

    def inertial_line(link, is_root, indent):
        inr = link.inertial
        has_mass = inr is not None and inr.mass > 0
        has_tensor = inr is not None and any(v != 0.0 for v in inr.inertia)
        if has_mass and has_tensor:
            return (f'{indent}<inertial pos="{_f(inr.com)}" mass="{inr.mass:.9g}" '
                    f'fullinertia="{_f(inr.inertia)}"/>')
        if has_mass:
            return (f'{indent}<inertial pos="{_f(inr.com)}" mass="{inr.mass:.9g}" '
                    f'diaginertia="1e-6 1e-6 1e-6"/><!-- no inertia tensor in source -->')
        if has_tensor:
            # Zero mass but a real tensor. Keep the tensor; MuJoCo will not take
            # a massless moving body, so lend it a negligible mass.
            return (f'{indent}<inertial pos="{_f(inr.com)}" mass="{_INERTIA_ONLY_MASS:.9g}" '
                    f'fullinertia="{_f(inr.inertia)}"/>'
                    f'<!-- source mass was 0; mass is a placeholder, inertia is real -->')
        if is_root:
            return None
        return (f'{indent}<inertial pos="0 0 0" mass="1e-3" diaginertia="1e-6 1e-6 1e-6"/>'
                f'<!-- placeholder: source link had no inertial -->')

    # "world" is MuJoCo's own top-level body; nothing else may be called that.
    taken = set(robot.links)

    def body_name(link_name):
        if link_name != _WORLD:
            return link_name
        alt, k = "world_link", 1
        while alt in taken:
            k += 1
            alt = f"world_link_{k}"
        taken.add(alt)
        return alt

    def body(link_name, joint, depth):
        ind = "  " * depth
        link = robot.links[link_name]
        name = body_name(link_name)
        note = ("" if name == link_name else
                f'<!-- renamed from "{link_name}": reserved by MuJoCo -->')
        if joint is None:
            lines.append(f'{ind}<body name={quoteattr(name)}>{note}')
        else:
            lines.append(f'{ind}<body name={quoteattr(name)}'
                         f'{_pose_attrs(joint.origin_xyz, joint.origin_rpy)}>{note}')
        il = inertial_line(link, joint is None, ind + "  ")
        if il:
            lines.append(il)
        if joint is not None and joint.type in MOVABLE:
            jt = "slide" if joint.type == "prismatic" else "hinge"
            lo, hi = joint.limit
            if joint.type == "continuous" or not (lo < hi):
                rng = ' limited="false"'
            else:
                rng = f' limited="true" range="{_f((lo, hi))}"'
            lines.append(f'{ind}  <joint name={quoteattr(joint.name)} type="{jt}" '
                         f'axis="{_f(joint.axis)}"{rng}/>')
        if geometry:
            lines.extend(geom_lines(link, ind + "  "))
        for cj in children.get(link_name, []):
            body(cj.child, cj, depth + 1)
        lines.append(f'{ind}</body>')

    if root == _WORLD:
        # The root is welded to the world frame at identity, so dropping its
        # body and hanging the subtree straight off <worldbody> is exact, and
        # it is the only way MuJoCo will take a root link named "world".
        lines.append('    <!-- root link "world" is MuJoCo\'s worldbody; '
                     'its subtree is emitted directly under it -->')
        if geometry:
            lines.extend(geom_lines(robot.links[root], "    "))
        for cj in children.get(root, []):
            body(cj.child, cj, 2)
    else:
        body(root, None, 2)

    head = [f'<mujoco model={quoteattr(robot.name)}>',
            '  <compiler angle="radian"'
            + (f' meshdir={quoteattr(meshdir)}' if meshdir else '') + '/>']
    if geometry and assets:
        head.append('  <asset>')
        for name, (path, scale) in assets.items():
            sc = (f' scale="{_f(scale)}"'
                  if any(abs(s - 1.0) > 1e-12 for s in scale) else '')
            head.append(f'    <mesh name={quoteattr(name)} '
                        f'file={quoteattr(_mesh_file(path, meshdir_strip))}{sc}/>')
        head.append('  </asset>')
    head.append('  <worldbody>')
    return "\n".join(head + lines + ['  </worldbody>', '</mujoco>']) + "\n"
