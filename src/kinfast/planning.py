"""Path planning in configuration space: get from one pose to another without
hitting anything.

The library already had the pieces a planner needs and no planner: inverse
kinematics to find a goal, collision distances to say what is legal, and
trajectory profiles to turn a path into motion. This module is the join.

What makes it fit kinfast rather than being a generic RRT: an edge between two
configurations is checked in ONE batched call. A planner spends nearly all of
its time asking "is this segment free", and the usual implementation walks the
segment one configuration at a time. Here the whole segment is interpolated
into a (steps, dof) tensor and handed to the collision code as a single batch,
which is the same trick the rest of the library is built on.

    checker = CollisionChecker(robot, spheres, world)
    plan = rrt_connect(robot.chain, q_start, q_goal, checker)
    print(plan.stats)                       # nodes, time, length before and after
    t, q, qd, qdd, T = plan.to_trajectory(robot)

The planner is RRT-Connect: two trees, one from each end, growing toward each
other. It is probabilistically complete, which means it finds a path when one
exists if you give it long enough, and it says so honestly when it runs out of
iterations instead of returning something half finished. Raw RRT paths wander,
so `shortcut` walks the result and replaces detours with straight segments
wherever the straight version is also free.
"""
import math
import time
from dataclasses import dataclass, field

import torch

from kinfast import config as cfg


@dataclass
class PlanStats:
    """What the planner did, so a caller can judge the answer it got."""
    solved: bool
    iterations: int
    nodes: int
    seconds: float
    edge_checks: int
    configs_checked: int
    raw_length: float = 0.0
    length: float = 0.0

    def __str__(self):
        if not self.solved:
            return (f"no path after {self.iterations} iterations, "
                    f"{self.nodes} nodes, {self.seconds * 1e3:.0f} ms")
        gain = (1 - self.length / self.raw_length) * 100 if self.raw_length else 0
        return (f"path found in {self.seconds * 1e3:.0f} ms, "
                f"{self.nodes} nodes, {self.iterations} iterations, "
                f"length {self.raw_length:.2f} -> {self.length:.2f} rad "
                f"({gain:.0f}% shorter), "
                f"{self.configs_checked:,} configurations checked in "
                f"{self.edge_checks} batched calls")


@dataclass
class Plan:
    """A sequence of configurations from start to goal, and how it was found."""
    path: torch.Tensor                      # (n, dof), waypoints in order
    stats: PlanStats
    chain: object = field(default=None, repr=False)

    def __len__(self):
        return int(self.path.shape[0])

    @property
    def solved(self):
        return self.stats.solved

    def densify(self, resolution=0.05):
        """Resample the waypoints so no two are further apart than resolution.

        Useful for drawing, and for feeding a controller that expects a stream
        rather than corners.
        """
        if len(self) < 2:
            return self.path
        out = [self.path[0:1]]
        for a, b in zip(self.path[:-1], self.path[1:]):
            steps = max(int(math.ceil(float(_dist(self.chain, a, b)) / resolution)), 1)
            s = torch.linspace(0, 1, steps + 1, dtype=self.path.dtype)[1:]
            out.append(_interp(self.chain, a, b, s))
        return torch.cat(out, dim=0)

    def to_trajectory(self, robot, amax=None, n=None):
        """Time the path with the robot's own velocity limits.

        Each leg gets a synchronized trapezoid, so the result respects the
        limits the model declares rather than a made up speed.
        """
        from kinfast.trajectory import trapezoidal
        vmax = robot.chain.vmax.clone()
        vmax[vmax <= 0] = 1.0
        if amax is None:
            amax = torch.full_like(vmax, 4.0)
        ts, qs, qds, qdds, total = [], [], [], [], 0.0
        for a, b in zip(self.path[:-1], self.path[1:]):
            per = n or max(int(float(_dist(self.chain, a, b)) / 0.02) + 2, 8)
            t, q, qd, qdd, T = trapezoidal(a, b, vmax, amax, n=per)
            ts.append(t + total)
            qs.append(q); qds.append(qd); qdds.append(qdd)
            total += float(T)
        if not ts:
            z = self.path[:1]
            return (torch.zeros(1, dtype=z.dtype), z, torch.zeros_like(z),
                    torch.zeros_like(z), 0.0)
        return (torch.cat(ts), torch.cat(qs), torch.cat(qds),
                torch.cat(qdds), total)


# ------------------------------------------------------------------ validity
class CollisionChecker:
    """Says whether configurations are legal, a whole batch at a time.

    A configuration is legal when it is inside the joint limits, the robot is
    not inside itself, and it is clear of the world by `margin`. Every one of
    those tests is already batched in the library, so asking about a thousand
    configurations costs about what asking about one does.
    """

    def __init__(self, robot, spheres=None, world=None, margin=0.0,
                 self_collision=True):
        self.robot = robot
        self.chain = robot.chain
        self.spheres = spheres
        self.world = world
        self.margin = float(margin)
        self.self_collision = self_collision and spheres is not None
        self.calls = 0                      # batched calls, not configurations
        self.configs = 0

    def __call__(self, q):
        """q (B, dof) -> (B,) bool, True where the configuration is usable."""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        self.calls += 1
        self.configs += int(q.shape[0])
        lo = self.chain.lower.to(device=q.device, dtype=q.dtype)
        hi = self.chain.upper.to(device=q.device, dtype=q.dtype)
        ok = ((q >= lo - 1e-9) & (q <= hi + 1e-9)).all(dim=-1)
        if not bool(ok.any()) or self.spheres is None:
            return ok
        if self.world is not None:
            from kinfast.collision_world import distance_to_world
            ok = ok & (distance_to_world(self.spheres, q, self.world) > self.margin)
        if self.self_collision:
            from kinfast.collision import self_distance
            d = self_distance(self.spheres, q)
            ok = ok & (torch.isinf(d) | (d > self.margin))
        return ok

    def edge(self, a, b, resolution=0.05):
        """True when every configuration between a and b is legal.

        The segment is interpolated into one tensor and checked in a single
        call. This is the operation a planner does most, so it is the one worth
        batching.
        """
        steps = max(int(math.ceil(float(_dist(self.chain, a, b)) / resolution)), 1)
        s = torch.linspace(0, 1, steps + 1, dtype=a.dtype, device=a.device)
        return bool(self(_interp(self.chain, a, b, s)).all())


def _dist(chain, a, b):
    if chain is None:
        return (b - a).norm()
    return cfg.distance(chain, a.unsqueeze(0), b.unsqueeze(0))[0]


def _interp(chain, a, b, s):
    """Configurations along a to b at fractions s, the short way around for
    continuous joints."""
    if chain is None:
        return a.unsqueeze(0) + (b - a).unsqueeze(0) * s.unsqueeze(-1)
    return cfg.interpolate(chain, a.unsqueeze(0), b.unsqueeze(0), s)


def _steer(chain, frm, to, step):
    """A configuration at most `step` away from frm, in the direction of to."""
    d = float(_dist(chain, frm, to))
    if d <= step:
        return to.clone()
    return _interp(chain, frm, to, torch.tensor([step / d], dtype=frm.dtype))[0]


# -------------------------------------------------------------------- planner
class _Tree:
    def __init__(self, root):
        self.nodes = [root]
        self.parent = [-1]

    def nearest(self, chain, q):
        stack = torch.stack(self.nodes)
        d = cfg.distance(chain, stack, q.unsqueeze(0).expand_as(stack))
        return int(torch.argmin(d))

    def add(self, q, parent):
        self.nodes.append(q)
        self.parent.append(parent)
        return len(self.nodes) - 1

    def branch(self, index):
        out = []
        while index >= 0:
            out.append(self.nodes[index])
            index = self.parent[index]
        return out[::-1]


def rrt_connect(chain, q_start, q_goal, valid, max_iters=3000, step=0.25,
                resolution=0.05, seed=0, shortcut_iters=120):
    """Plan from q_start to q_goal around whatever `valid` rejects.

    `valid` is anything callable that takes (B, dof) and returns (B,) bool; a
    CollisionChecker is the obvious one. Returns a Plan whose `stats.solved`
    says whether it worked. A failed plan still carries its statistics, which
    is usually enough to tell a hard problem from a broken one.
    """
    t0 = time.perf_counter()
    q_start = q_start.reshape(-1).clone()
    q_goal = q_goal.reshape(-1).clone()
    edge = (valid.edge if isinstance(valid, CollisionChecker)
            else lambda a, b, resolution=resolution: _edge_free(
                chain, a, b, valid, resolution))

    def stats(solved, iters, nodes):
        return PlanStats(
            solved=solved, iterations=iters, nodes=nodes,
            seconds=time.perf_counter() - t0,
            edge_checks=getattr(valid, "calls", 0),
            configs_checked=getattr(valid, "configs", 0))

    both = torch.stack([q_start, q_goal])
    ok = valid(both)
    if not bool(ok[0]):
        return Plan(q_start.unsqueeze(0), stats(False, 0, 0), chain)
    if not bool(ok[1]):
        return Plan(q_start.unsqueeze(0), stats(False, 0, 0), chain)
    if edge(q_start, q_goal, resolution):    # the easy case, worth trying first
        path = torch.stack([q_start, q_goal])
        st = stats(True, 0, 2)
        st.raw_length = st.length = float(_dist(chain, q_start, q_goal))
        return Plan(path, st, chain)

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    lo = chain.lower.to(dtype=q_start.dtype)
    hi = chain.upper.to(dtype=q_start.dtype)
    trees = [_Tree(q_start), _Tree(q_goal)]

    for it in range(1, max_iters + 1):
        a, b = trees[it % 2], trees[(it + 1) % 2]
        q_rand = lo + (hi - lo) * torch.rand(lo.shape, generator=g, dtype=lo.dtype)
        near = a.nearest(chain, q_rand)
        q_new = _steer(chain, a.nodes[near], q_rand, step)
        if not edge(a.nodes[near], q_new, resolution):
            continue
        ia = a.add(q_new, near)

        # connect: grow the other tree at this new node until it arrives or stops
        near_b = b.nearest(chain, q_new)
        cur = b.nodes[near_b]
        ib = near_b
        while True:
            nxt = _steer(chain, cur, q_new, step)
            if not edge(cur, nxt, resolution):
                break
            ib = b.add(nxt, ib)
            cur = nxt
            if float(_dist(chain, cur, q_new)) < 1e-6:
                branch_a, branch_b = a.branch(ia), b.branch(ib)
                if a is trees[0]:
                    raw = branch_a + branch_b[::-1][1:]
                else:
                    raw = branch_b + branch_a[::-1][1:]
                path = torch.stack(raw)
                st = stats(True, it, len(trees[0].nodes) + len(trees[1].nodes))
                st.raw_length = _path_length(chain, path)
                if shortcut_iters:
                    path = shortcut(chain, path, valid, iters=shortcut_iters,
                                    resolution=resolution, seed=seed)
                st.length = _path_length(chain, path)
                st.edge_checks = getattr(valid, "calls", 0)
                st.configs_checked = getattr(valid, "configs", 0)
                st.seconds = time.perf_counter() - t0
                return Plan(path, st, chain)

    return Plan(q_start.unsqueeze(0), stats(
        False, max_iters, len(trees[0].nodes) + len(trees[1].nodes)), chain)


def _edge_free(chain, a, b, valid, resolution):
    steps = max(int(math.ceil(float(_dist(chain, a, b)) / resolution)), 1)
    s = torch.linspace(0, 1, steps + 1, dtype=a.dtype, device=a.device)
    return bool(valid(_interp(chain, a, b, s)).all())


def _path_length(chain, path):
    if path.shape[0] < 2:
        return 0.0
    return float(sum(_dist(chain, a, b) for a, b in zip(path[:-1], path[1:])))


def shortcut(chain, path, valid, iters=120, resolution=0.05, seed=0):
    """Replace detours with straight segments wherever the straight one is free.

    An RRT path is made of random steps and it looks like it. Picking two
    points at random and asking whether the segment between them is legal is
    the cheapest useful smoother there is, and it can only shorten the path,
    never break it: a replacement is kept only when its own edge check passes.
    """
    if path.shape[0] < 3:
        return path
    edge = (valid.edge if isinstance(valid, CollisionChecker)
            else lambda a, b, resolution=resolution: _edge_free(
                chain, a, b, valid, resolution))
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    pts = [row for row in path]
    for _ in range(iters):
        if len(pts) < 3:
            break
        i = int(torch.randint(0, len(pts) - 2, (1,), generator=g))
        j = int(torch.randint(i + 2, len(pts), (1,), generator=g))
        if edge(pts[i], pts[j], resolution):
            pts = pts[:i + 1] + pts[j:]
    return torch.stack(pts)


def plan_to_pose(robot, q_start, target, valid, ik_restarts=8, **kw):
    """Plan to a Cartesian pose: solve inverse kinematics, then plan to it.

    The first collision free solution the batched solve returns is used, which
    is why it is worth asking for several seeds: they come back in one call and
    some of them land in obstacles.
    """
    q, _info = robot.ik(target, pos_only=kw.pop("pos_only", True),
                        restarts=ik_restarts, iters=kw.pop("ik_iters", 100))
    cand = q if q.dim() == 2 else q.unsqueeze(0)
    ok = valid(cand)
    for row, good in zip(cand, ok):
        if bool(good):
            return rrt_connect(robot.chain, q_start, row, valid, **kw)
    return Plan(q_start.reshape(1, -1),
                PlanStats(False, 0, 0, 0.0, getattr(valid, "calls", 0),
                          getattr(valid, "configs", 0)), robot.chain)
