# src/kinfast/compile.py
"""Compile a Robot IR into flat tensors for batched kinematics.

Each non-root link is reached from its parent by exactly one joint. We store,
per link: parent index, the fixed origin transform (parent -> joint frame),
the joint axis, an integer joint type, and the index into q (or -1 if fixed).
A topological order (parents before children) drives the FK sweep.
"""
from dataclasses import dataclass
import torch
from kinfast.ir import Robot, MOVABLE
from kinfast import transforms as T

_TYPE_CODE = {"fixed": 0, "revolute": 1, "continuous": 1, "prismatic": 2}


@dataclass
class CompiledChain:
    n_links: int
    dof: int
    link_names: list
    link_index: dict
    parent: torch.Tensor         # (n_links,) long, -1 for root
    joint_origin: torch.Tensor   # (n_links, 4, 4)
    joint_axis: torch.Tensor     # (n_links, 3)
    joint_type: torch.Tensor     # (n_links,) long: 0 fixed, 1 revolute, 2 prismatic
    q_index: torch.Tensor        # (n_links,) long, -1 if fixed
    topo_order: list
    lower: torch.Tensor          # (dof,)
    upper: torch.Tensor          # (dof,)
    vmax: torch.Tensor           # (dof,) joint velocity limits (0 if unspecified)
    joint_names: list            # (dof,) movable joint names, ordered by q index
    link_mass: torch.Tensor      # (n_links,)
    link_com: torch.Tensor       # (n_links, 3) COM in link frame
    link_inertia: torch.Tensor   # (n_links, 3, 3) inertia about COM in link frame
    # Bumped on a device move or an explicit invalidate_cache(). FK folds the
    # constants above into per-(device, dtype) derived tensors and keys that
    # cache on this counter plus each source tensor's storage and in-place
    # version, so a stale cache cannot outlive an edit. Treat the tensors as
    # immutable anyway; the fold is not free.
    _version: int = 0

    def to(self, device):
        for name in ("parent", "joint_origin", "joint_axis", "joint_type",
                     "q_index", "lower", "upper", "vmax", "link_mass",
                     "link_com", "link_inertia"):
            setattr(self, name, getattr(self, name).to(device))
        self._version += 1
        return self

    def invalidate_cache(self):
        """Drop cached derived constants (FK origin rotations, prismatic
        directions, movable-joint index tensors). FK already notices an
        in-place edit of a chain tensor on its own, so this is the escape hatch
        for a change it cannot see. Returns self so it can be chained."""
        self._version += 1
        return self


def compile_robot(robot: Robot, dtype=torch.float32) -> CompiledChain:
    root = robot.root_link()
    names = [root] + [n for n in robot.links if n != root]
    index = {n: i for i, n in enumerate(names)}
    n = len(names)

    parent = torch.full((n,), -1, dtype=torch.long)
    origin = torch.eye(4, dtype=dtype).repeat(n, 1, 1)
    axis = torch.zeros(n, 3, dtype=dtype)
    axis[:, 2] = 1.0
    jtype = torch.zeros(n, dtype=torch.long)
    q_index = torch.full((n,), -1, dtype=torch.long)

    link_mass = torch.zeros(n, dtype=dtype)
    link_com = torch.zeros(n, 3, dtype=dtype)
    link_inertia = torch.zeros(n, 3, 3, dtype=dtype)
    for name in names:
        i = index[name]
        inr = robot.links[name].inertial
        if inr is None:
            continue
        link_mass[i] = inr.mass
        link_com[i] = torch.tensor(inr.com, dtype=dtype)
        ixx, iyy, izz, ixy, ixz, iyz = inr.inertia
        link_inertia[i] = torch.tensor(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=dtype)

    joint_by_child = {j.child: j for j in robot.joints}
    lowers, uppers, vels, jnames = [], [], [], []
    next_q = 0
    for name in names:
        i = index[name]
        j = joint_by_child.get(name)
        if j is None:
            continue
        parent[i] = index[j.parent]
        R = T.rpy_to_matrix(torch.tensor(j.origin_rpy, dtype=dtype))
        t = torch.tensor(j.origin_xyz, dtype=dtype)
        origin[i] = T.make_transform(R, t)
        a = torch.tensor(j.axis, dtype=dtype)
        # Normalize so FK (which rotates about the unit axis) and the geometric
        # Jacobian (which reads the axis directly) agree even if the URDF gave a
        # non-unit <axis> and repair was skipped.
        axis[i] = a / a.norm().clamp_min(1e-12)
        jtype[i] = _TYPE_CODE.get(j.type, 0)
        if j.type in MOVABLE:
            q_index[i] = next_q
            next_q += 1
            lowers.append(j.limit[0])
            uppers.append(j.limit[1])
            vels.append(j.velocity)
            jnames.append(j.name)

    # topological order: BFS from root using parent pointers
    order, frontier = [], [index[root]]
    children = {p: [] for p in range(n)}
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            children[p].append(i)
    while frontier:
        node = frontier.pop(0)
        order.append(node)
        frontier.extend(children[node])

    return CompiledChain(
        n_links=n, dof=next_q, link_names=names, link_index=index,
        parent=parent, joint_origin=origin, joint_axis=axis, joint_type=jtype,
        q_index=q_index, topo_order=order,
        lower=torch.tensor(lowers, dtype=dtype),
        upper=torch.tensor(uppers, dtype=dtype),
        vmax=torch.tensor(vels, dtype=dtype),
        joint_names=jnames,
        link_mass=link_mass, link_com=link_com, link_inertia=link_inertia,
    )
