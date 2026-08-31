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
    # A joint reads its value as scale * q[q_index] + offset. Both are 1 and 0
    # for an ordinary joint. A mimic joint instead points q_index at the slot
    # of the joint that drives it and carries that relation here, so it costs
    # no degree of freedom and every consumer that indexes through q_index
    # keeps working unchanged.
    joint_scale: torch.Tensor    # (n_links,)
    joint_offset: torch.Tensor   # (n_links,)
    topo_order: list
    lower: torch.Tensor          # (dof,)
    upper: torch.Tensor          # (dof,)
    vmax: torch.Tensor           # (dof,) joint velocity limits (0 if unspecified)
    joint_names: list            # (dof,) movable joint names, ordered by q index
    link_joint_names: list       # (n_links,) joint driving each link, None for the root
    link_mass: torch.Tensor      # (n_links,)
    link_com: torch.Tensor       # (n_links, 3) COM in link frame
    link_inertia: torch.Tensor   # (n_links, 3, 3) inertia about COM in link frame
    gravity: tuple = (0.0, 0.0, -9.81)   # world gravity vector, m/s^2
    # Bumped on a device move, a dtype recompile, or an explicit
    # invalidate_cache(). FK folds the constants above into per-(device, dtype)
    # derived tensors and keys that cache on this counter plus each source
    # tensor's storage and in-place version, so a stale cache cannot outlive an
    # edit. Treat the tensors as immutable anyway; the fold is not free.
    _version: int = 0

    # tensors that carry the model's real numbers, and the ones that are
    # integer bookkeeping (never cast to a float dtype)
    _FLOAT_FIELDS = ("joint_origin", "joint_axis", "lower", "upper", "vmax",
                     "joint_scale", "joint_offset",
                     "link_mass", "link_com", "link_inertia")
    _INT_FIELDS = ("parent", "joint_type", "q_index")

    def expand_q(self, q: torch.Tensor) -> torch.Tensor:
        """(B, dof) actuated values -> (B, n_movable) per joint values.

        Every movable joint gets its own entry, in joint_names order for the
        independent ones followed by the driven ones in link order. Needed to
        talk to a tool that does not implement <mimic> and therefore expects one
        value per joint: MuJoCo and PyBullet both ignore the tag on URDF import,
        so handing them a reduced vector silently misplaces the driven joints.
        """
        movable = [i for i in range(self.n_links) if int(self.q_index[i]) >= 0]
        cols = self.q_index[movable].to(q.device)
        sc = self.joint_scale[movable].to(device=q.device, dtype=q.dtype)
        off = self.joint_offset[movable].to(device=q.device, dtype=q.dtype)
        return q[:, cols] * sc + off

    def movable_joint_names(self) -> list:
        """Names matching expand_q's columns, driven joints included."""
        return [self.link_joint_names[i] for i in range(self.n_links)
                if int(self.q_index[i]) >= 0]

    @property
    def has_mimic(self):
        """True if any joint is driven by another rather than actuated.

        Worth asking before an algorithm that assumes one joint per degree of
        freedom: on a mimic chain two joints can share a q column, and each
        contributes through its own scale factor."""
        return bool((self.joint_scale != 1).any() or (self.joint_offset != 0).any())

    @property
    def dtype(self):
        """The float dtype the constants were compiled at."""
        return self.joint_origin.dtype

    def to(self, device=None, dtype=None):
        """Move and/or cast the compiled constants, in place.

        Casting up (float32 -> float64) does NOT recover precision: the
        constants were already rounded when they were compiled. To gain real
        digits, rebuild from the IR with compile_robot(ir, dtype=...), which
        is what Robot.to(dtype=...) does. A torch.dtype may be passed
        positionally, mirroring Tensor.to.
        """
        if isinstance(device, torch.dtype):
            device, dtype = None, device
        for name in self._INT_FIELDS:
            if device is not None:
                setattr(self, name, getattr(self, name).to(device))
        for name in self._FLOAT_FIELDS:
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        # the derived-constant caches are keyed on this counter, so any move or
        # cast invalidates them
        self._version += 1
        if dtype is not None:
            object.__setattr__(self, "_fk_cache", None)
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
    joint_by_name = {j.name: j for j in robot.joints}
    link_jnames = [None] * n
    scale = torch.ones(n, dtype=dtype)
    offset = torch.zeros(n, dtype=dtype)

    def resolve_mimic(j, seen=()):
        """Collapse a chain of mimics down to (root joint, multiplier, offset).

        A joint may mimic a joint that itself mimics another, so the relation
        has to be composed rather than read once. Composing q = k2*(k1*qr + o1)
        + o2 gives k2*k1 and k2*o1 + o2."""
        if j.mimic is None:
            return j, 1.0, 0.0
        if j.name in seen:
            raise ValueError(
                f"joint {j.name} mimics itself through {' -> '.join(seen)}; "
                "a mimic cycle has no independent joint to drive it")
        src_name, mult, off = j.mimic
        src = joint_by_name.get(src_name)
        if src is None:
            raise ValueError(
                f"joint {j.name} mimics {src_name!r}, which does not exist")
        if src.type not in MOVABLE:
            raise ValueError(
                f"joint {j.name} mimics {src_name!r}, which is "
                f"{src.type} and so has nothing to drive it")
        root, k, o = resolve_mimic(src, seen + (j.name,))
        return root, mult * k, mult * o + off

    lowers, uppers, vels, jnames = [], [], [], []
    by_slot = {}
    slot_of_joint = {}
    next_q = 0
    # first pass: independent joints claim the q slots
    for name in names:
        i = index[name]
        j = joint_by_child.get(name)
        if j is None or j.type not in MOVABLE or j.mimic is not None:
            continue
        # keyed by identity, not name: a malformed file can declare two
        # joints with the same name and they still need separate slots
        slot_of_joint[id(j)] = next_q
        next_q += 1

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
        link_jnames[i] = j.name
        if j.type in MOVABLE:
            driver, k, o = resolve_mimic(j)
            slot = slot_of_joint[id(driver)]
            q_index[i] = slot
            scale[i] = k
            offset[i] = o
            if j.mimic is None:
                by_slot[slot] = [j.limit[0], j.limit[1], j.velocity, j.name]
            else:
                # the driven joint's own limit restricts what the driving one
                # may do, mapped back through q_this = k * q_root + o
                lo_m, hi_m = j.limit
                if hi_m > lo_m and k != 0.0:
                    a, b = (lo_m - o) / k, (hi_m - o) / k
                    lo_s, hi_s = (a, b) if k > 0 else (b, a)
                    cur = by_slot.setdefault(
                        slot, [driver.limit[0], driver.limit[1],
                               driver.velocity, driver.name])
                    cur[0] = max(cur[0], lo_s)
                    cur[1] = min(cur[1], hi_s)

    for s in range(next_q):
        lo, hi, v, nm = by_slot[s]
        lowers.append(lo)
        uppers.append(hi)
        vels.append(v)
        jnames.append(nm)

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
        q_index=q_index, joint_scale=scale, joint_offset=offset,
        topo_order=order,
        lower=torch.tensor(lowers, dtype=dtype),
        upper=torch.tensor(uppers, dtype=dtype),
        vmax=torch.tensor(vels, dtype=dtype),
        joint_names=jnames, link_joint_names=link_jnames,
        link_mass=link_mass, link_com=link_com, link_inertia=link_inertia,
        gravity=tuple(float(c) for c in getattr(robot, "gravity",
                                                (0.0, 0.0, -9.81))),
    )
