# src/kinfast/jacobian.py
"""Batched geometric Jacobian read from the FK frames.

For each movable joint on the path from the target link to the root:
  revolute:  Jv = axis_world x (p_ee - p_joint),  Jw = axis_world
  prismatic: Jv = axis_world,                     Jw = 0
Rotating about an axis leaves that axis invariant, so the axis expressed in the
joint's world frame is world[i][:3,:3] @ axis (motion-independent).
"""
import torch
from kinfast.fk import forward_kinematics
from kinfast.compile import CompiledChain


def _resolve_link(chain: CompiledChain, link_index) -> int:
    """Turn a link index into a plain non-negative int, accepting Python-style
    negatives (-1 is the last link, matching forward_kinematics(...)[:, -1])
    and raising a readable IndexError for anything out of range."""
    n = chain.n_links
    idx = int(link_index)
    if not -n <= idx < n:
        raise IndexError(
            f"link_index {link_index} out of range for {n} links "
            f"(use chain.link_index[name] to look one up)")
    return idx % n


def jacobian_rp(chain: CompiledChain, q: torch.Tensor, link_index: int,
                rp=None) -> torch.Tensor:
    """Geometric Jacobian computed from R/p lists (the fast path).
    rp: (wR, wp) from fk_rp at the same q; computed here if absent."""
    from kinfast.fk import fk_rp, _cache
    B = q.shape[0]
    device, dtype = q.device, q.dtype
    link_index = _resolve_link(chain, link_index)
    if rp is None:
        rp = fk_rp(chain, q)
    wR, wp = rp
    p_ee = wp[link_index]                             # (B, 3)
    J = torch.zeros(B, 6, chain.dof, dtype=dtype, device=device)
    # Joint axes in q's dtype, so a float64 q works on a float32-compiled chain
    # (and vice versa) the same way fk_rp does. The fk cache already holds them.
    axes = _cache(chain, device, dtype)["axes"]

    i = link_index
    while i >= 0:
        jt = int(chain.joint_type[i])
        if jt != 0:
            col = int(chain.q_index[i])
            axis_world = wR[i] @ axes[i]              # (B, 3)
            if jt == 1:  # revolute
                J[:, :3, col] = torch.cross(axis_world, p_ee - wp[i], dim=-1)
                J[:, 3:, col] = axis_world
            else:        # prismatic
                J[:, :3, col] = axis_world
        i = int(chain.parent[i])
    return J


def jacobian(chain: CompiledChain, q: torch.Tensor, link_index: int,
             world: torch.Tensor = None) -> torch.Tensor:
    """Pass `world` (from forward_kinematics at the same q) to avoid
    recomputing FK; callers like IK evaluate both per iteration."""
    if world is None:
        return jacobian_rp(chain, q, link_index)
    rp = ([world[:, i, :3, :3] for i in range(chain.n_links)],
          [world[:, i, :3, 3] for i in range(chain.n_links)])
    return jacobian_rp(chain, q, link_index, rp=rp)
