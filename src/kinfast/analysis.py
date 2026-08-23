# src/kinfast/analysis.py
"""Workspace and dexterity analysis.

  manipulability      Yoshikawa's measure w = sqrt(det(J J^T)): how far the arm
                      is from a singularity (0 = singular). Batched.
  condition_number    sigma_max / sigma_min of the Jacobian: isotropy of motion.
                      Both take `rows=` to pick the task rows (planar arms).
  joint_limit_margin  normalized distance of each config from its nearest limit
                      (1 = mid-range, 0 = on a limit).
  workspace           Monte-Carlo reachable-workspace point cloud + reach stats.
"""
import math
import torch
from kinfast.fk import forward_kinematics
from kinfast.jacobian import jacobian


def manipulability(chain, q, link_index, translational: bool = True,
                   rows=None):
    """Yoshikawa manipulability w = sqrt(det(J J^T)). (B,). 0 = singular.

    The measure requires J to have full row rank over the task rows you care
    about: for a planar arm the 3-row translational Jacobian has an identically
    zero out-of-plane row, making det(JJ^T) = 0 at every configuration. Pass
    `rows` to select the task rows explicitly (e.g. rows=(0, 1) for an xy-planar
    arm). Default: the 3 linear rows (translational=True) or all 6."""
    J = jacobian(chain, q, link_index)
    if rows is not None:
        J = J[:, list(rows), :]
    elif translational:
        J = J[:, :3, :]
    JJt = J @ J.transpose(-1, -2)
    det = torch.linalg.det(JJt)
    return torch.sqrt(det.clamp_min(0.0))


def condition_number(chain, q, link_index, translational: bool = True,
                     rows=None):
    """Jacobian condition number sigma_max/sigma_min. (B,). Huge near singularity.

    Same row-selection caveat as manipulability: the ratio only means something
    when J has full row rank over the task rows you keep. For a planar arm with
    3 or more joints the 3-row translational Jacobian carries an identically
    zero out-of-plane row, so sigma_min is 0 at every configuration and the
    function reports ~1e12 everywhere. Pass `rows` to select the task rows
    explicitly (e.g. rows=(0, 1) for an xy-planar arm). Default: the 3 linear
    rows (translational=True) or all 6."""
    J = jacobian(chain, q, link_index)
    if rows is not None:
        J = J[:, list(rows), :]
    elif translational:
        J = J[:, :3, :]
    s = torch.linalg.svdvals(J)
    return s[:, 0] / s[:, -1].clamp_min(1e-12)


def joint_limit_margin(chain, q):
    """Min over joints of normalized distance to the nearest limit. (B,).
    1 = every joint mid-range, 0 = some joint sitting on a limit."""
    lo, hi = chain.lower.to(q.device), chain.upper.to(q.device)
    span = (hi - lo).clamp_min(1e-12)
    d = torch.minimum(q - lo, hi - q) / (0.5 * span)
    # An unbounded joint (inf or NaN limit) has no mid-range to normalize
    # against; inf/inf would give NaN here. Such a joint is at full margin
    # unless q actually sits past whichever bound is finite.
    violated = (q < lo) | (q > hi)
    d = torch.where(torch.isfinite(span), d, (~violated).to(d.dtype))
    return d.clamp(0.0, 1.0).min(dim=-1).values


def sampling_bounds(chain):
    """Finite (lower, upper) bounds, one per dof, for drawing random configs.

    A joint whose limit is infinite (or NaN) cannot be sampled as
    lo + (hi - lo) * u: that is -inf + inf * u, which is NaN. A revolute joint
    with an unbounded limit is sampled over one full turn, anchored at the
    finite side if there is one, so every pose it can reach is still covered.
    A prismatic joint with an unbounded limit has no natural range, so that
    raises ValueError instead of inventing one. Finite limits pass through
    untouched."""
    lo, hi = chain.lower, chain.upper
    lo_ok, hi_ok = torch.isfinite(lo), torch.isfinite(hi)
    if bool((lo_ok & hi_ok).all()):
        return lo, hi
    # joint type lives on links; q_index maps each movable link to its q slot
    q_index = chain.q_index.to(lo.device)
    jtype = chain.joint_type.to(lo.device)
    movable = q_index >= 0
    type_per_dof = torch.zeros(chain.dof, dtype=torch.long, device=lo.device)
    type_per_dof[q_index[movable]] = jtype[movable]
    bad = ~(lo_ok & hi_ok)
    prismatic_bad = bad & (type_per_dof == 2)
    if bool(prismatic_bad.any()):
        names = [chain.joint_names[i] for i in prismatic_bad.nonzero().flatten().tolist()]
        raise ValueError(
            f"prismatic joint(s) {names} have non-finite limits; "
            "joint limits must be finite to sample configurations")
    turn = 2.0 * math.pi
    lo_s = torch.where(lo_ok, lo, torch.where(hi_ok, hi - turn, torch.full_like(lo, -math.pi)))
    hi_s = torch.where(hi_ok, hi, torch.where(lo_ok, lo + turn, torch.full_like(hi, math.pi)))
    return lo_s, hi_s


def workspace(chain, link_index, n: int = 10000, seed: int = 0, device="cpu"):
    """Monte-Carlo reachable workspace of a link origin.

    Returns dict with points (n,3), max_reach, min_reach, centroid: enough for
    quick 'can this arm reach my part?' MechE questions. Revolute joints with
    infinite limits are sampled over a full turn; prismatic ones with infinite
    limits raise ValueError (see sampling_bounds).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo, hi = sampling_bounds(chain)
    lo, hi = lo.cpu(), hi.cpu()
    # sample on the CPU generator (reproducible), then move to wherever the
    # chain lives (a CUDA robot must not mix CPU randoms with device limits)
    q = lo + (hi - lo) * torch.rand(n, chain.dof, generator=g, dtype=lo.dtype)
    q = q.to(chain.lower.device if device == "cpu" else device)
    pts = forward_kinematics(chain, q)[:, link_index, :3, 3]
    r = pts.norm(dim=-1)
    return {
        "points": pts,
        "max_reach": r.max(),
        "min_reach": r.min(),
        "centroid": pts.mean(dim=0),
    }
