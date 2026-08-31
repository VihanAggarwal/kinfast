# src/kinfast/urdf/parse.py
"""Parse a (subset of) URDF XML into the Robot IR.

Robustness is the moat: unknown tags are ignored, missing optional fields get
sane defaults, and malformed numbers raise a clear error naming the joint/link.
"""
import math
import os
import re
import xml.etree.ElementTree as ET
from kinfast.ir import Robot, Link, Joint, Inertial, Geometry


def _rpy_to_mat(r, p, y):
    """URDF rpy (extrinsic xyz): R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def rotate_inertia(i6, R):
    """Express an inertia tensor given in a rotated frame in the link frame:
    I_link = R @ I @ R^T. i6 = (ixx, iyy, izz, ixy, ixz, iyz)."""
    ixx, iyy, izz, ixy, ixz, iyz = i6
    I = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    RI = [[sum(R[a][k] * I[k][b] for k in range(3)) for b in range(3)] for a in range(3)]
    J = [[sum(RI[a][k] * R[b][k] for k in range(3)) for b in range(3)] for a in range(3)]
    return (J[0][0], J[1][1], J[2][2], J[0][1], J[0][2], J[1][2])

# ROS substitution args that leak into published URDFs (e.g. Husky's
# "$(optenv HUSKY_IMU_XYZ 0.19 0 0.149)"). Real files ship like this, so we
# expand them: $(env NAME) reads the environment (error if unset),
# $(optenv NAME default...) reads the environment or falls back to the default.
_SUBST = re.compile(r"\$\(\s*(env|optenv)\s+(\S+?)((?:\s+[^\s)]+)*)\s*\)")


def _expand_substitutions(text: str) -> str:
    def repl(m):
        kind, name, default = m.group(1), m.group(2), m.group(3).strip()
        val = os.environ.get(name)
        if val is not None:
            return val
        if kind == "optenv":
            return default  # may be empty, matching ROS semantics
        raise ValueError(f"$(env {name}): environment variable not set and no default")
    return _SUBST.sub(repl, text)


def _floats(text, n, where):
    parts = text.split()
    if len(parts) != n:
        raise ValueError(f"{where}: expected {n} numbers, got {text!r}")
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise ValueError(f"{where}: non-numeric value in {text!r}")


def _parse_link(el) -> Link:
    link = Link(el.get("name"))
    inel = el.find("inertial")
    if inel is not None:
        mass_el = inel.find("mass")
        i_el = inel.find("inertia")
        origin = inel.find("origin")
        mass = float(mass_el.get("value")) if mass_el is not None else 0.0
        com = _floats(origin.get("xyz"), 3, f"link {link.name} inertial origin") \
            if origin is not None and origin.get("xyz") else (0.0, 0.0, 0.0)
        if i_el is not None:
            inertia = tuple(float(i_el.get(k, 0.0))
                            for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        else:
            inertia = (0.0,) * 6
        if origin is not None and origin.get("rpy"):
            rpy = _floats(origin.get("rpy"), 3, f"link {link.name} inertial origin")
            if any(abs(a) > 1e-12 for a in rpy):
                inertia = rotate_inertia(inertia, _rpy_to_mat(*rpy))
        link.inertial = Inertial(mass, com, inertia)
    return link


def _parse_geometry(parent):
    if parent is None:
        return None
    geo = parent.find("geometry")
    if geo is None:
        return None
    origin = parent.find("origin")
    oxyz = (_floats(origin.get("xyz", "0 0 0"), 3, "geometry origin")
            if origin is not None else (0.0, 0.0, 0.0))
    orpy = (_floats(origin.get("rpy", "0 0 0"), 3, "geometry origin")
            if origin is not None else (0.0, 0.0, 0.0))
    mesh = geo.find("mesh")
    if mesh is not None:
        scale = (_floats(mesh.get("scale"), 3, "mesh scale") if mesh.get("scale")
                 else (1.0, 1.0, 1.0))
        return Geometry("mesh", mesh.get("filename"), scale, (), oxyz, orpy)
    box = geo.find("box")
    if box is not None:
        return Geometry("box", None, (1.0, 1.0, 1.0),
                        _floats(box.get("size", "0 0 0"), 3, "box size"), oxyz, orpy)
    cyl = geo.find("cylinder")
    if cyl is not None:
        return Geometry("cylinder", None, (1.0, 1.0, 1.0),
                        (float(cyl.get("radius", 0.0)), float(cyl.get("length", 0.0))),
                        oxyz, orpy)
    sph = geo.find("sphere")
    if sph is not None:
        return Geometry("sphere", None, (1.0, 1.0, 1.0),
                        (float(sph.get("radius", 0.0)),), oxyz, orpy)
    return None


def _parse_joint(el) -> Joint:
    name = el.get("name")
    where = f"joint {name}"
    jtype = el.get("type")
    parent = el.find("parent").get("link")
    child = el.find("child").get("link")
    origin = el.find("origin")
    xyz = _floats(origin.get("xyz", "0 0 0"), 3, where) if origin is not None else (0.0, 0.0, 0.0)
    rpy = _floats(origin.get("rpy", "0 0 0"), 3, where) if origin is not None else (0.0, 0.0, 0.0)
    axis_el = el.find("axis")
    axis = _floats(axis_el.get("xyz"), 3, where) if axis_el is not None else (0.0, 0.0, 1.0)
    limit_el = el.find("limit")
    if limit_el is not None:
        lo = float(limit_el.get("lower", 0.0))
        hi = float(limit_el.get("upper", 0.0))
        vel = float(limit_el.get("velocity", 0.0))
        eff = float(limit_el.get("effort", 0.0))
    else:
        lo = hi = vel = eff = 0.0
    return Joint(name, jtype, parent, child, xyz, rpy, axis, (lo, hi), vel, eff)


def parse_urdf_string(text: str) -> Robot:
    text = _expand_substitutions(text)
    root = ET.fromstring(text)
    if root.tag != "robot":
        raise ValueError(f"root element must be <robot>, got <{root.tag}>")
    links, notes = {}, []

    def slot(el, tag):
        """None, one Geometry, or a list of them. Keeping the single case
        unwrapped means the common link is shaped exactly as it always was."""
        geoms = [g for g in (_parse_geometry(e) for e in el.findall(tag))
                 if g is not None and g.kind != "none"]
        if not geoms:
            return None
        return geoms[0] if len(geoms) == 1 else geoms

    for el in root.findall("link"):
        link = _parse_link(el)
        if link.name in links:
            notes.append(f"link {link.name}: declared more than once, "
                         "later declaration wins")
        link.visual = slot(el, "visual")
        link.collision = slot(el, "collision")
        n_col = len(el.findall("collision"))
        if n_col > 1:
            notes.append(f"link {link.name}: {n_col} collision elements")
        links[link.name] = link

    joints = []
    for el in root.findall("joint"):
        j = _parse_joint(el)
        m = el.find("mimic")
        if m is not None and m.get("joint"):
            j.mimic = (m.get("joint"),
                       float(m.get("multiplier", 1.0)),
                       float(m.get("offset", 0.0)))
        lo, hi = j.limit
        if el.find("limit") is None and j.type in ("revolute", "prismatic"):
            notes.append(f"joint {j.name}: {j.type} with no <limit>, "
                         f"assumed {lo:.4g} to {hi:.4g}")
        elif lo > hi:
            notes.append(f"joint {j.name}: limits given as {hi:.4g} to "
                         f"{lo:.4g}, which is inverted, so they were swapped")
        elif lo == hi and j.type in ("revolute", "prismatic"):
            notes.append(f"joint {j.name}: limits are both {lo:.4g}, so it "
                         "cannot move, but it still consumes a degree of freedom")
        joints.append(j)

    robot = Robot(root.get("name", "robot"), links, joints)
    robot.parse_notes.extend(notes)
    return robot


def parse_urdf_file(path: str) -> Robot:
    with open(path, "r", encoding="utf-8") as f:
        return parse_urdf_string(f.read())
