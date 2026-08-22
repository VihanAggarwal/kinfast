# src/kinfast/analysis.py
"""Workspace and dexterity analysis.

  manipulability      Yoshikawa's measure w = sqrt(det(J J^T)) — how far the arm
                      is from a singularity (0 = singular). Batched.
  condition_number    sigma_max / sigma_min of the Jacobian — isotropy of motion.
  joint_limit_margin  normalized distance of each config from its nearest limit
                      (1 = mid-range, 0 = on a limit).
  workspace           Monte-Carlo reachable-workspace point cloud + reach stats.
"""
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


def condition_number(chain, q, link_index, translational: bool = True):
    """Jacobian condition number sigma_max/sigma_min. (B,). inf near singularity."""
    J = jacobian(chain, q, link_index)
    if translational:
        J = J[:, :3, :]
    s = torch.linalg.svdvals(J)
    return s[:, 0] / s[:, -1].clamp_min(1e-12)


def joint_limit_margin(chain, q):
    """Min over joints of normalized distance to the nearest limit. (B,).
    1 = every joint mid-range, 0 = some joint sitting on a limit."""
    lo, hi = chain.lower.to(q.device), chain.upper.to(q.device)
    span = (hi - lo).clamp_min(1e-12)
    d = torch.minimum(q - lo, hi - q) / (0.5 * span)
    return d.clamp(0.0, 1.0).min(dim=-1).values


def workspace(chain, link_index, n: int = 10000, seed: int = 0, device="cpu"):
    """Monte-Carlo reachable workspace of a link origin.

    Returns dict with points (n,3), max_reach, min_reach, centroid — enough for
    quick 'can this arm reach my part?' MechE questions.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo, hi = chain.lower.cpu(), chain.upper.cpu()
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
