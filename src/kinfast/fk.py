# src/kinfast/fk.py
"""Batched forward kinematics: propagate homogeneous transforms down the tree.

Vectorized: joint motions for ALL links are built in one batched Rodrigues call
and composed with the fixed origins in one broadcast matmul; only the
parent-to-child sweep (inherently sequential) loops, and it is one (B,4,4)
matmul per link.
"""
import torch
from kinfast import transforms as T
from kinfast.compile import CompiledChain


def _local_transforms(chain: CompiledChain, q: torch.Tensor) -> torch.Tensor:
    """(B, n_links, 4, 4): origin @ joint_motion for every link, in one pass."""
    B = q.shape[0]
    n = chain.n_links
    device, dtype = q.device, q.dtype
    origin = chain.joint_origin
    axes = chain.joint_axis
    if origin.device != device:
        origin, axes = origin.to(device), axes.to(device)
    origin = origin.to(dtype)
    axes = axes.to(dtype)

    # per-link joint value: q[:, q_index] for movable links, 0 for fixed
    movable = (chain.q_index.to(device) >= 0)
    if chain.dof == 0:                                           # all-fixed robot
        vals = torch.zeros(B, n, dtype=dtype, device=device)
    else:
        qidx = chain.q_index.to(device).clamp_min(0)             # (n,)
        vals = q[:, qidx]                                        # (B,n)
        vals = torch.where(movable.unsqueeze(0), vals, torch.zeros_like(vals))

    jt = chain.joint_type.to(device)                             # (n,)
    rev = movable & (jt == 1)
    pris = movable & (jt == 2)

    motion = torch.eye(4, dtype=dtype, device=device).expand(B, n, 4, 4).clone()
    if bool(rev.any()):
        R = T.axis_angle_to_matrix(axes.unsqueeze(0).expand(B, n, 3), vals)  # (B,n,3,3)
        motion[:, rev, :3, :3] = R[:, rev]
    if bool(pris.any()):
        t = axes.unsqueeze(0) * vals.unsqueeze(-1)               # (B,n,3)
        motion[:, pris, :3, 3] = t[:, pris]
    return origin.unsqueeze(0) @ motion                          # (B,n,4,4)


def forward_kinematics(chain: CompiledChain, q: torch.Tensor) -> torch.Tensor:
    """q (B, dof) -> world transforms (B, n_links, 4, 4)."""
    local = _local_transforms(chain, q)
    world = [None] * chain.n_links
    for i in chain.topo_order:
        p = int(chain.parent[i])
        world[i] = local[:, i] if p < 0 else world[p] @ local[:, i]
    return torch.stack(world, dim=1)
