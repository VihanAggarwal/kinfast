# src/kinfast/urdf/parse.py
"""Parse a (subset of) URDF XML into the Robot IR.

Robustness is the moat: unknown tags are ignored, missing optional fields get
sane defaults, and malformed numbers raise a clear error naming the joint/link.
"""
import xml.etree.ElementTree as ET
from kinfast.ir import Robot, Link, Joint, Inertial, Geometry


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
        link.inertial = Inertial(mass, com, inertia)
    return link


def _parse_geometry(parent):
    if parent is None:
        return None
    geo = parent.find("geometry")
    if geo is None:
        return None
    mesh = geo.find("mesh")
    if mesh is not None:
        scale = _floats(mesh.get("scale"), 3, "mesh scale") if mesh.get("scale") \
            else (1.0, 1.0, 1.0)
        return Geometry("mesh", mesh.get("filename"), scale)
    for kind in ("box", "cylinder", "sphere"):
        if geo.find(kind) is not None:
            return Geometry(kind)
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
    root = ET.fromstring(text)
    if root.tag != "robot":
        raise ValueError(f"root element must be <robot>, got <{root.tag}>")
    links = {}
    for el in root.findall("link"):
        link = _parse_link(el)
        link.visual = _parse_geometry(el.find("visual"))
        link.collision = _parse_geometry(el.find("collision"))
        links[link.name] = link
    joints = [_parse_joint(el) for el in root.findall("joint")]
    return Robot(root.get("name", "robot"), links, joints)


def parse_urdf_file(path: str) -> Robot:
    with open(path, "r", encoding="utf-8") as f:
        return parse_urdf_string(f.read())
