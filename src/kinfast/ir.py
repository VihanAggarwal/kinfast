# src/kinfast/ir.py
"""Format-agnostic robot intermediate representation (pure data)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Inertial:
    mass: float = 0.0
    com: tuple = (0.0, 0.0, 0.0)
    inertia: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # ixx, iyy, izz, ixy, ixz, iyz


@dataclass
class Geometry:
    kind: str = "none"          # "mesh" | "box" | "cylinder" | "sphere" | "none"
    mesh_path: Optional[str] = None
    scale: tuple = (1.0, 1.0, 1.0)
    size: tuple = ()            # box: (x,y,z) full extents; cylinder: (radius, length); sphere: (radius,)
    origin_xyz: tuple = (0.0, 0.0, 0.0)
    origin_rpy: tuple = (0.0, 0.0, 0.0)


@dataclass
class Link:
    name: str
    inertial: Optional[Inertial] = None
    visual: Optional[Geometry] = None
    collision: Optional[Geometry] = None


@dataclass
class Joint:
    name: str
    type: str                    # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    origin_xyz: tuple = (0.0, 0.0, 0.0)
    origin_rpy: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    limit: tuple = (0.0, 0.0)    # (lower, upper) radians/metres
    velocity: float = 0.0
    effort: float = 0.0


MOVABLE = {"revolute", "continuous", "prismatic"}


@dataclass
class Robot:
    name: str
    links: dict = field(default_factory=dict)
    joints: list = field(default_factory=list)
    # World gravity vector in m/s^2. URDF has no way to express it, so it keeps
    # the default; MJCF `<option gravity="...">` overrides it.
    gravity: tuple = (0.0, 0.0, -9.81)
    # Human-readable notes from the parser about anything it had to reinterpret,
    # e.g. an MJCF free joint that kinfast pins to the world. Always a list.
    parse_notes: list = field(default_factory=list)

    def root_link(self) -> str:
        children = {j.child for j in self.joints}
        roots = [n for n in self.links if n not in children]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root link, found {roots}")
        return roots[0]

    def movable_joints(self) -> list:
        return [j for j in self.joints if j.type in MOVABLE]

    def dof(self) -> int:
        return len(self.movable_joints())
