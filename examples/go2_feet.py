# examples/go2_feet.py
"""Stance inverse kinematics for the Unitree Go2, all four feet at once.

An arm has one end effector. A quadruped has four, and they share a single
configuration vector, so a stance is not four separate IK problems glued
together: it is one task with twelve rows (four feet, three position rows
each) over the twelve leg joints. This script builds that task and solves it
with the same damped least squares step the rest of kinfast uses, batched over
a stack of body heights so every height is solved in one pass.

What it does, in order:

1. Load the Go2 MJCF. The model's base carries a `<freejoint>`, which kinfast
   pins to the world and says so in `robot.parse_notes`. That is exactly what
   you want here: with the trunk held still, solving for foot positions in the
   base frame is the standing problem, and the floating base only decides where
   that stance ends up in the world.
2. Find the four feet. The feet are the leaf links of the kinematic tree, and
   each one is labelled FL / FR / RL / RR from the sign of its leg's hip
   attachment in the base frame, so the labelling does not depend on the
   model's naming convention.
3. Find the contact point inside each foot link. On the Go2 the foot is a
   small sphere hanging 0.213 m below the knee frame, declared as a geom in the
   MJCF. kinfast's IR keeps kinematics, not collision geometry, so this script
   reads those spheres back out of the XML (resolving MuJoCo's `<default>`
   classes) and treats the sphere centre as the foot point. Without that
   offset you would be solving for the knee.
4. Solve. Targets are placed under each hip at a series of body heights, and
   one batched solve returns a configuration per height.
5. Report solve rates and save a matplotlib figure of the stance.

Everything below the CLI is batched over a leading B dimension, takes its
working dtype and device from the caller's q (or targets), and stays
differentiable: the solve loop is plain tensor math, so gradients flow from the
returned joint angles back to the foot targets. That makes it usable as a layer
inside a learned controller and not only as a one-shot script.

Usage:
  python examples/go2_feet.py --mjcf examples/assets/menagerie/unitree_go2/go2.xml
  python examples/go2_feet.py --heights 0.20 0.28 0.34 --out stance.png
"""
import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import torch

import kinfast
from kinfast.fk import fk_rp
from kinfast.jacobian import jacobian_rp

DEFAULT_MJCF = os.path.join("examples", "assets", "menagerie", "unitree_go2",
                            "go2.xml")
DEFAULT_HEIGHTS = (0.18, 0.22, 0.26, 0.30, 0.34, 0.38)


# ---------------------------------------------------------------- model bits

@dataclass
class Feet:
    """The four feet of a quadruped, in FL, FR, RL, RR order.

    labels    ("FL", "FR", "RL", "RR"), by construction.
    links     link name per label.
    link_ids  row of `fk_all` per label.
    offsets   (4, 3) contact point in each foot link's own frame.
    radii     (4,) contact sphere radius, 0 when the model does not say.
    hips      (4, 3) base-frame position of each leg's first link, constant.
    base_link, base_id  the trunk the stance is expressed in.
    """
    labels: tuple
    links: tuple
    link_ids: tuple
    offsets: torch.Tensor
    radii: torch.Tensor
    hips: torch.Tensor
    base_link: str
    base_id: int

    def cast(self, dtype, device):
        """Offsets and hips in a given working dtype and device."""
        return (self.offsets.to(dtype=dtype, device=device),
                self.hips.to(dtype=dtype, device=device))


def leaf_links(chain):
    """Link names with no child link. On a legged robot these are the feet."""
    parents = set(int(p) for p in chain.parent.tolist())
    return [name for i, name in enumerate(chain.link_names) if i not in parents]


def fixed_base_link(chain, from_link):
    """The last link above `from_link` that is rigidly attached to the root.

    Start at the root and walk down toward the foot for as long as the joints
    are fixed; the last link you reach is the one that never moves. On the Go2
    that walk passes the world link, then stops at the trunk, whose free joint
    kinfast pinned. The trunk is the frame a stance is naturally written in.
    Returns (name, index).
    """
    path = []
    i = int(from_link)
    while i >= 0:
        path.append(i)
        i = int(chain.parent[i])
    best = path[-1]                       # the root, always fixed
    for i in reversed(path):              # root first, walking down to the foot
        if int(chain.joint_type[i]) != 0:
            break
        best = i
    return chain.link_names[best], best


def _geom_defaults(root):
    """MuJoCo `<default>` classes flattened to geom attributes.

    Each class inherits its parent class's attributes and overrides them, the
    same rule kinfast's MJCF parser applies to joints. The unnamed top level
    default is stored under "".
    """
    out = {}

    def walk(el, name, inherited):
        if el is None:
            return
        attrs = dict(inherited)
        g = el.find("geom")
        if g is not None:
            attrs.update(g.attrib)
        out[name] = attrs
        for child in el.findall("default"):
            walk(child, child.get("class", ""), attrs)

    walk(root.find("default"), "", {})
    return out


def _floats(text, n):
    vals = [float(v) for v in (text or "").split()]
    return tuple(vals[:n]) + (0.0,) * max(0, n - len(vals))


def contact_spheres(mjcf_path, bodies):
    """Contact sphere per body, read straight from the MJCF.

    Returns {body_name: (pos (3,), radius)} for the bodies that have one. A
    geom is taken as a contact sphere when its resolved type is sphere (the
    MuJoCo default when neither `type` nor `mesh` is given); when a body has
    several, the lowest one wins, which is the one that touches the ground.
    Bodies with no sphere are simply absent from the result, and the caller
    falls back to the link origin.
    """
    root = ET.parse(mjcf_path).getroot()
    defaults = _geom_defaults(root)
    wanted = set(bodies)
    found = {}

    def resolve(el, childclass):
        """A geom's own class wins, then the innermost body childclass, then
        the unnamed top level default; the geom's own attributes beat them
        all. Same precedence the kinfast MJCF parser uses for joints."""
        attrs = {}
        for cls in ("", childclass, el.get("class")):
            if cls is not None and cls in defaults:
                attrs.update(defaults[cls])
        attrs.update(el.attrib)
        return attrs

    def walk(body, childclass):
        childclass = body.get("childclass", childclass)
        name = body.get("name")
        if name in wanted:
            best = None
            for g in body.findall("geom"):
                a = resolve(g, childclass)
                kind = a.get("type", "mesh" if a.get("mesh") else "sphere")
                if kind != "sphere":
                    continue
                pos = _floats(a.get("pos", "0 0 0"), 3)
                radius = _floats(a.get("size", "0"), 1)[0]
                if best is None or pos[2] < best[0][2]:
                    best = (pos, radius)
            if best is not None:
                found[name] = best
        for sub in body.findall("body"):
            walk(sub, childclass)

    world = root.find("worldbody")
    if world is not None:
        for body in world.findall("body"):
            walk(body, "")
    return found


def find_feet(robot, mjcf_path=None, links=None):
    """Identify the four feet, their contact offsets, and the trunk frame.

    `links` names the foot links explicitly; by default they are the leaf links
    of the tree, which must number four. `mjcf_path` is the file the robot was
    loaded from: when given, the contact spheres inside the foot links are read
    from it, otherwise the link origin is used as the contact point.
    """
    chain = robot.chain
    names = list(links) if links is not None else leaf_links(chain)
    if len(names) != 4:
        raise ValueError(
            f"expected 4 foot links, found {len(names)}: {names}. This script "
            "is written for quadrupeds; pass links=[...] to name the feet of "
            "another topology.")
    ids = [robot.link_id(n) for n in names]

    bases = {fixed_base_link(chain, i) for i in ids}
    if len(bases) != 1:
        raise ValueError(f"the four feet do not share one fixed trunk: {bases}")
    base_link, base_id = bases.pop()

    # Hip attachment of each leg: walk up from the foot to the child of the
    # trunk. Its origin is rigid to the trunk, so it is the same at every q.
    q = torch.zeros(1, chain.dof, dtype=torch.float64)
    world = robot.fk_all(q)[0]
    R0 = world[base_id, :3, :3]
    t0 = world[base_id, :3, 3]
    hips = []
    for i in ids:
        j = i
        while int(chain.parent[j]) != base_id:
            j = int(chain.parent[j])
            if j < 0:
                raise ValueError("foot link is not below the trunk")
        hips.append(R0.T @ (world[j, :3, 3] - t0))
    hips = torch.stack(hips)

    # Label from the hip position in the trunk frame: +x is forward, +y is left.
    order = {}
    for k, h in enumerate(hips):
        label = ("F" if h[0] >= 0 else "R") + ("L" if h[1] >= 0 else "R")
        if label in order:
            raise ValueError(
                f"two legs land in quadrant {label}; the hip layout "
                f"{hips.tolist()} does not look like a quadruped")
        order[label] = k
    labels = ("FL", "FR", "RL", "RR")
    missing = [lab for lab in labels if lab not in order]
    if missing:
        raise ValueError(f"no leg found for {missing}; hips are {hips.tolist()}")
    perm = [order[lab] for lab in labels]

    spheres = contact_spheres(mjcf_path, names) if mjcf_path else {}
    offsets = torch.zeros(4, 3, dtype=torch.float64)
    radii = torch.zeros(4, dtype=torch.float64)
    for out_k, k in enumerate(perm):
        got = spheres.get(names[k])
        if got is not None:
            offsets[out_k] = torch.tensor(got[0], dtype=torch.float64)
            radii[out_k] = got[1]
    return Feet(labels=labels,
                links=tuple(names[k] for k in perm),
                link_ids=tuple(ids[k] for k in perm),
                offsets=offsets, radii=radii,
                hips=hips[perm].contiguous(),
                base_link=base_link, base_id=base_id)


# ------------------------------------------------------------ foot kinematics

def _base_frame(rp, feet):
    """Rotation and origin of the trunk from an fk_rp result."""
    wR, wp = rp
    return wR[feet.base_id], wp[feet.base_id]


def foot_points_rp(rp, feet, offsets):
    """(B, 4, 3) foot contact points in the trunk frame, from an fk_rp result."""
    wR, wp = rp
    Rb, tb = _base_frame(rp, feet)
    pts = []
    for k, i in enumerate(feet.link_ids):
        p_world = wp[i] + torch.einsum("bij,j->bi", wR[i], offsets[k])
        pts.append(torch.einsum("bji,bj->bi", Rb, p_world - tb))
    return torch.stack(pts, dim=1)


def foot_points(robot, q, feet):
    """(B, 4, 3) foot contact points in the trunk frame for a batch of q.

    Differentiable in q, and computed in q's dtype on q's device, so a float32
    robot answers a float64 query in float64.
    """
    offsets, _ = feet.cast(q.dtype, q.device)
    return foot_points_rp(fk_rp(robot.chain, q), feet, offsets)


def foot_jacobians_rp(chain, q, rp, feet, offsets):
    """(B, 4, 3, dof) trunk-frame position Jacobians of the four contact points.

    The geometric Jacobian of a link gives the velocity of the link origin. The
    contact point sits at a fixed offset r inside the link, so its velocity is
    v + w x r, which in matrix form is Jv - skew(r) Jw with r the offset
    rotated into the world. Rotating the result into the trunk frame is a
    constant left multiply, because no movable joint lies above the trunk.
    """
    wR, _ = rp
    Rb, _ = _base_frame(rp, feet)
    cols = []
    for k, i in enumerate(feet.link_ids):
        J = jacobian_rp(chain, q, i, rp=rp)                  # (B, 6, dof)
        r = torch.einsum("bij,j->bi", wR[i], offsets[k])     # (B, 3)
        rr = r.unsqueeze(-1).expand_as(J[:, :3])
        Jp = J[:, :3] - torch.cross(rr, J[:, 3:], dim=1)     # (B, 3, dof)
        cols.append(torch.einsum("bji,bjd->bid", Rb, Jp))
    return torch.stack(cols, dim=1)


def foot_jacobians(robot, q, feet):
    """(B, 4, 3, dof) trunk-frame foot Jacobians for a batch of q."""
    offsets, _ = feet.cast(q.dtype, q.device)
    rp = fk_rp(robot.chain, q)
    return foot_jacobians_rp(robot.chain, q, rp, feet, offsets)


# ------------------------------------------------------------------- the task

def rest_posture(robot, dtype=torch.float64):
    """Midpoint of every joint range, the standard seed and rest posture.

    On the Go2 this is already a plausible crouch (hips level, thighs forward,
    knees folded), which is why it makes a good single seed. Joints with an
    infinite range rest at zero.
    """
    lo = robot.chain.lower.to(dtype)
    hi = robot.chain.upper.to(dtype)
    mid = 0.5 * (lo + hi)
    return torch.where(torch.isfinite(mid), mid, torch.zeros_like(mid))


def stance_targets(robot, feet, heights, dx=0.0, dy=0.0, q_ref=None):
    """(B, 4, 3) foot targets in the trunk frame, one row per body height.

    The x and y of each target come from the neutral pose, every joint at zero,
    which on a legged robot is the leg hanging straight down: the foot ends up
    under its own hip at the leg's natural lateral spread, which is the stance
    a standing quadruped actually wants. dx shifts each foot outward along x
    and dy outward along y, both by the leg's own sign, so a positive dy
    widens the stance rather than sliding the whole robot sideways.

    The z is -(height - radius): the trunk origin sits `height` above the
    ground and the contact sphere's centre sits one radius above it, so the
    sphere touches rather than sinking in. Pass q_ref to take the x and y from
    some other posture instead of the neutral one.
    """
    heights = torch.as_tensor(heights, dtype=torch.float64).reshape(-1)
    if q_ref is None:
        q = torch.zeros(robot.chain.dof, dtype=torch.float64)
    else:
        q = torch.as_tensor(q_ref).to(torch.float64)
    hang = foot_points(robot, q.reshape(1, -1), feet)[0]      # (4, 3)
    sign_y = torch.sign(feet.hips[:, 1].to(torch.float64))
    sign_x = torch.sign(feet.hips[:, 0].to(torch.float64))
    xy = torch.stack([hang[:, 0] + dx * sign_x, hang[:, 1] + dy * sign_y], -1)
    B = heights.shape[0]
    z = -(heights.reshape(B, 1) - feet.radii.reshape(1, 4))   # (B, 4)
    return torch.cat([xy.unsqueeze(0).expand(B, -1, -1), z.unsqueeze(-1)], -1)


def solve_stance(robot, feet, targets, q0=None, iters=80, damping=0.05,
                 step=1.0, tol=1e-4, restarts=1, generator=None):
    """Solve all four feet of a batch of stances at once.

    targets  (B, 4, 3) contact points in the trunk frame.
    q0       (B, dof) or (dof,) seed; the rest posture when omitted. Its dtype
             and device are the working dtype and device.
    restarts extra random seeds per stance, solved in the same batch and
             reduced to the best one per stance. Seed 0 is always q0, so more
             restarts can only help.
    tol      per-foot position tolerance in metres used for the solve rate.

    Returns (q (B, dof), info) with info holding per-foot errors of the
    RETURNED q, so the numbers describe what you get back:
    foot_error (B, 4), max_error (B,), solved (B, 4) bool, solve_rate float.

    The loop is one damped least squares step on the stacked 12-row task,

        dq = J^T (J J^T + lambda^2 I)^-1 e,

    clamped to the joint limits. All of it is differentiable, so gradients flow
    from q back to the targets and the seed.
    """
    chain = robot.chain
    if q0 is not None:
        q0 = torch.as_tensor(q0)
        dtype, device = q0.dtype, q0.device
    else:
        dtype, device = targets.dtype, targets.device
    targets = targets.to(dtype=dtype, device=device)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    B, dof = targets.shape[0], chain.dof
    lo = chain.lower.to(dtype=dtype, device=device)
    hi = chain.upper.to(dtype=dtype, device=device)

    rest = rest_posture(robot, dtype).to(device)
    if q0 is None:
        seed = rest.unsqueeze(0).expand(B, -1)
    elif q0.dim() == 1:
        seed = q0.to(dtype=dtype, device=device).unsqueeze(0).expand(B, -1)
    else:
        seed = q0.to(dtype=dtype, device=device)

    K = max(1, int(restarts))
    if K > 1:
        span = torch.where(torch.isfinite(hi - lo), hi - lo,
                           torch.full_like(lo, 2.0 * math.pi))
        base = torch.where(torch.isfinite(lo), lo, -0.5 * span)
        rnd = base + span * torch.rand(B * (K - 1), dof, dtype=dtype,
                                       device=device, generator=generator)
        q = torch.cat([seed, rnd], dim=0)
        tgt = targets.repeat(K, 1, 1)
    else:
        q, tgt = seed.clone(), targets

    offsets, _ = feet.cast(dtype, device)
    eye = torch.eye(12, dtype=dtype, device=device)
    lam2 = damping * damping
    for _ in range(iters):
        rp = fk_rp(chain, q)
        e = (tgt - foot_points_rp(rp, feet, offsets)).reshape(-1, 12)
        J = foot_jacobians_rp(chain, q, rp, feet, offsets).reshape(-1, 12, dof)
        JT = J.transpose(-1, -2)
        H = J @ JT + lam2 * eye
        dq = (JT @ torch.linalg.solve(H, e.unsqueeze(-1))).squeeze(-1)
        q = torch.clamp(q + step * dq, lo, hi)

    err = (tgt - foot_points_rp(fk_rp(chain, q), feet, offsets)).norm(dim=-1)
    if K > 1:
        err_k = err.view(K, B, 4)
        best = err_k.max(dim=-1).values.argmin(dim=0)          # (B,)
        q = q.view(K, B, dof).gather(
            0, best.view(1, B, 1).expand(1, B, dof)).squeeze(0)
        err = err_k.gather(0, best.view(1, B, 1).expand(1, B, 4)).squeeze(0)

    solved = err.detach() < tol
    info = {"foot_error": err.detach(),
            "max_error": err.detach().max(dim=-1).values,
            "solved": solved,
            "solve_rate": float(solved.to(torch.float64).mean()),
            "iters": iters, "restarts": K}
    return q, info


# ------------------------------------------------------------------ reporting

def format_report(heights, info):
    """The per-height table, as a list of lines (so it is testable)."""
    lines = ["", "  height   feet solved   worst foot error",
             "  " + "-" * 44]
    for b, h in enumerate(heights):
        n = int(info["solved"][b].sum())
        lines.append(f"  {float(h):.3f} m   {n}/4          "
                     f"{float(info['max_error'][b]) * 1000:8.3f} mm")
    lines.append("  " + "-" * 44)
    lines.append(f"  solve rate {info['solve_rate'] * 100:.1f}% over "
                 f"{len(heights)} heights x 4 feet")
    return lines


def plot_stance(robot, feet, q, targets, heights, path):
    """Save a side and top view of the solved stances (Agg, no display)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chain = robot.chain
    q = q.detach().to(dtype=torch.float64, device="cpu")
    robot = robot.to("cpu")
    with torch.no_grad():
        world = robot.fk_all(q)
        Rb = world[:, feet.base_id, :3, :3]
        tb = world[:, feet.base_id, :3, 3]
        pts = torch.einsum("bji,bnj->bni", Rb,
                           world[:, :, :3, 3] - tb.unsqueeze(1))
        feet_p = foot_points(robot, q, feet)
    pts = pts.numpy()
    feet_p = feet_p.numpy()
    tgt = targets.detach().to(dtype=torch.float64, device="cpu").numpy()

    # Draw only what moves: the links below the trunk. The world link sits
    # 0.445 m under the trunk here, and a segment out to it would dwarf the dog.
    legs = []
    for i in range(chain.n_links):
        p = int(chain.parent[i])
        j = p
        while j >= 0 and j != feet.base_id:
            j = int(chain.parent[j])
        if j == feet.base_id:
            legs.append((p, i))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    cmap = plt.get_cmap("viridis")
    n = max(1, len(heights) - 1)
    for b, h in enumerate(heights):
        c = cmap(b / n)
        for p, i in legs:
            ax1.plot([pts[b, p, 0], pts[b, i, 0]],
                     [pts[b, p, 2], pts[b, i, 2]], color=c, lw=1.6, alpha=0.85)
        for k, i in enumerate(feet.link_ids):   # shin: link origin to contact
            ax1.plot([pts[b, i, 0], feet_p[b, k, 0]],
                     [pts[b, i, 2], feet_p[b, k, 2]], color=c, lw=1.6,
                     alpha=0.85)
        ax1.plot(feet_p[b, :, 0], feet_p[b, :, 2], "o", color=c, ms=4)
        ax1.axhline(-float(h), color=c, ls=":", lw=0.8, alpha=0.6)
        ax1.plot([], [], color=c, lw=1.6, label=f"{float(h):.2f} m")
        poly = [0, 1, 3, 2, 0]                      # FL, FR, RR, RL, back to FL
        ax2.plot(feet_p[b, poly, 0], feet_p[b, poly, 1], "-o", color=c, ms=4,
                 lw=1.2, alpha=0.85)
    ax1.plot(tgt[:, :, 0].ravel(), tgt[:, :, 2].ravel(), "kx", ms=5,
             label="targets")
    ax1.set_xlabel("x forward (m)")
    ax1.set_ylabel("z up, trunk frame (m)")
    ax1.set_title("side view: solved leg posture per body height")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(alpha=0.25)
    for k, lab in enumerate(feet.labels):
        ax2.annotate(lab, (feet_p[0, k, 0], feet_p[0, k, 1]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set_xlabel("x forward (m)")
    ax2.set_ylabel("y left (m)")
    ax2.set_title("top view: support polygon")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.25)
    fig.suptitle(f"{robot.ir.name}: four-foot stance IK "
                 f"({len(heights)} heights solved in one batch)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------------------- the CLI

def run(path, heights=DEFAULT_HEIGHTS, dx=0.0, dy=0.0, iters=80,
        restarts=1, tol=1e-4, dtype=torch.float64, device="cpu", out=None):
    """Load, solve, report, plot. Returns (robot, feet, q, info, lines)."""
    robot = kinfast.load(path, dtype=dtype).to(device)
    feet = find_feet(robot, mjcf_path=path)
    targets = stance_targets(robot, feet, heights, dx=dx, dy=dy)
    targets = targets.to(dtype=dtype, device=device)

    lines = [f"loaded {robot.ir.name} from {path}",
             f"  {robot.dof} dof, {robot.n_links} links, "
             f"trunk frame '{feet.base_link}'"]
    for note in robot.parse_notes:
        lines.append(f"  parser note: {note}")
    for k, lab in enumerate(feet.labels):
        off = feet.offsets[k].tolist()
        lines.append(f"  {lab} foot: link {feet.links[k]!r}, contact at "
                     f"({off[0]:+.4f}, {off[1]:+.4f}, {off[2]:+.4f}) m, "
                     f"radius {float(feet.radii[k]):.3f} m")

    t0 = time.perf_counter()
    q, info = solve_stance(robot, feet, targets, iters=iters, tol=tol,
                           restarts=restarts)
    dt = time.perf_counter() - t0
    lines.append(f"  solved {targets.shape[0]} stances x 4 feet in one batch: "
                 f"{iters} iterations, {restarts} seed(s), {dt * 1000:.1f} ms")
    lines += format_report(heights, info)
    if out:
        plot_stance(robot, feet, q, targets, heights, out)
        lines.append(f"  wrote {out}")
    return robot, feet, q, info, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mjcf", default=DEFAULT_MJCF,
                    help="path to go2.xml (MuJoCo Menagerie)")
    ap.add_argument("--heights", type=float, nargs="+",
                    default=list(DEFAULT_HEIGHTS),
                    help="body heights in metres, trunk origin above ground")
    ap.add_argument("--dx", type=float, default=0.0,
                    help="shift each foot outward along x (m)")
    ap.add_argument("--dy", type=float, default=0.0,
                    help="shift each foot outward along y (m)")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="per-foot tolerance counted as solved (m)")
    ap.add_argument("--float32", action="store_true",
                    help="work in float32 instead of float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="go2_stance.png",
                    help="figure path, or '' to skip plotting")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.mjcf):
        print(f"no MJCF at {args.mjcf!r} (looked at "
              f"{os.path.abspath(args.mjcf)}).\n"
              "The Go2 model is not shipped with kinfast. Fetch the MuJoCo "
              "Menagerie copy with\n"
              "  python examples/menagerie.py --fetch\n"
              "or point --mjcf at your own go2.xml.", file=sys.stderr)
        return 2

    _, _, _, info, lines = run(
        args.mjcf, heights=args.heights, dx=args.dx, dy=args.dy,
        iters=args.iters, restarts=args.restarts, tol=args.tol,
        dtype=torch.float32 if args.float32 else torch.float64,
        device=args.device, out=args.out or None)
    print("\n".join(lines))
    return 0 if info["solve_rate"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
