# src/kinfast/fk.py
"""Batched forward kinematics: propagate transforms down the tree.

Optimized for large batches: rotations and translations are carried as
(B, 3, 3) and (B, 3) instead of homogeneous 4x4s (27 multiplies per compose
instead of 64), the Rodrigues formula runs only over the revolute links, and
prismatic directions (origin rotation times axis) are constant-folded per
device. The parent-to-child sweep is inherently sequential; everything else is
one vectorized pass.
"""
import torch
from kinfast import transforms as T
from kinfast.compile import CompiledChain


def _cache(chain: CompiledChain, device, dtype):
    """Per-(device, dtype) constants: origin R/p, movable index tensors, and
    the constant prismatic direction d = origin_R @ axis."""
    key = (str(device), str(dtype))
    store = getattr(chain, "_fk_cache", None)
    if store is None:
        store = {}
        object.__setattr__(chain, "_fk_cache", store)
    if key in store:
        return store[key]
    origin = chain.joint_origin.to(device=device, dtype=dtype)
    oR = origin[:, :3, :3].contiguous()
    op = origin[:, :3, 3].contiguous()
    jt = chain.joint_type.to(device)
    qi = chain.q_index.to(device)
    movable = qi >= 0
    rev = torch.nonzero(movable & (jt == 1), as_tuple=False).flatten()
    pris = torch.nonzero(movable & (jt == 2), as_tuple=False).flatten()
    axes = chain.joint_axis.to(device=device, dtype=dtype)
    entry = {
        "oR": oR, "op": op, "axes": axes,
        "rev": rev, "rev_q": qi[rev], "rev_axes": axes[rev],
        "rev_oR": oR[rev],
        "pris": pris, "pris_q": qi[pris],
        "pris_dir": (oR[pris] @ axes[pris].unsqueeze(-1)).squeeze(-1),
        "pris_op": op[pris],
    }
    store[key] = entry
    return entry


def _local_rp(chain: CompiledChain, q: torch.Tensor):
    """Per-link local transforms as (B, n, 3, 3) rotations + (B, n, 3) offsets."""
    B = q.shape[0]
    n = chain.n_links
    c = _cache(chain, q.device, q.dtype)
    local_R = c["oR"].unsqueeze(0).expand(B, n, 3, 3)
    local_p = c["op"].unsqueeze(0).expand(B, n, 3)

    if c["rev"].numel():
        vals = q[:, c["rev_q"]]                                   # (B, r)
        R = T.axis_angle_to_matrix(
            c["rev_axes"].unsqueeze(0).expand(B, -1, 3), vals)    # (B, r, 3, 3)
        local_R = local_R.clone()
        local_R[:, c["rev"]] = c["rev_oR"].unsqueeze(0) @ R
    if c["pris"].numel():
        vals = q[:, c["pris_q"]]                                  # (B, p)
        local_p = local_p.clone()
        local_p[:, c["pris"]] = (c["pris_op"].unsqueeze(0)
                                 + c["pris_dir"].unsqueeze(0) * vals.unsqueeze(-1))
    return local_R, local_p


def fk_rp(chain: CompiledChain, q: torch.Tensor):
    """World rotations and positions as per-link lists: (wR, wp) with
    wR[i] (B, 3, 3) and wp[i] (B, 3). The hot path for IK and Jacobians,
    which never need the assembled 4x4 tensor (assembly is ~a third of FK
    time at large batch)."""
    local_R, local_p = _local_rp(chain, q)
    wR = [None] * chain.n_links
    wp = [None] * chain.n_links
    for i in chain.topo_order:
        p = int(chain.parent[i])
        if p < 0:
            wR[i], wp[i] = local_R[:, i], local_p[:, i]
        else:
            wR[i] = wR[p] @ local_R[:, i]
            wp[i] = wp[p] + (wR[p] @ local_p[:, i].unsqueeze(-1)).squeeze(-1)
    return wR, wp


def forward_kinematics(chain: CompiledChain, q: torch.Tensor) -> torch.Tensor:
    """q (B, dof) -> world transforms (B, n_links, 4, 4)."""
    B = q.shape[0]
    wR, wp = fk_rp(chain, q)
    M = torch.zeros(B, chain.n_links, 4, 4, dtype=q.dtype, device=q.device)
    M[:, :, :3, :3] = torch.stack(wR, dim=1)
    M[:, :, :3, 3] = torch.stack(wp, dim=1)
    M[:, :, 3, 3] = 1.0
    return M
