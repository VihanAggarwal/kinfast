# src/kinfast/fk.py
"""Batched forward kinematics: propagate transforms down the tree.

Optimized for large batches: rotations and translations are carried as
(B, 3, 3) and (B, 3) instead of homogeneous 4x4s (27 multiplies per compose
instead of 64), the Rodrigues formula runs only over the revolute links, and
prismatic directions (origin rotation times axis) are constant-folded per
device. The parent-to-child sweep is inherently sequential; everything else is
one vectorized pass.

Those folded constants are cached on the chain, keyed on device, dtype, and a
fingerprint of the tensors they were folded from: the chain's own _version
counter plus each source tensor's storage pointer and in-place version counter.
That means an edit such as chain.joint_origin[i, 0, 3] += 0.1 is picked up on
the next call instead of being masked by a stale entry. A CompiledChain is
still meant to be immutable; chain.invalidate_cache() forces a refold if you
manage to change the constants in a way the fingerprint cannot see.
"""
import torch
from kinfast import transforms as T
from kinfast.compile import CompiledChain


def _check_q(chain: CompiledChain, q, name="q"):
    """Reject a q that is not a floating (B, dof) tensor, before it reaches an
    indexing or linalg call that would report something unrelated."""
    if not isinstance(q, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor of shape (B, {chain.dof}), got "
            f"{type(q).__name__}")
    if not q.is_floating_point():
        raise TypeError(
            f"{name} must be a floating point tensor (float32/float64), got "
            f"dtype {q.dtype}. Cast it, e.g. {name}.to(torch.float32).")
    if q.dim() != 2 or q.shape[1] != chain.dof:
        shape = tuple(q.shape)
        hint = ""
        if q.dim() == 1 and q.shape[0] == chain.dof:
            hint = (f" Forward kinematics is always batched: pass "
                    f"{name}.unsqueeze(0) for a single configuration.")
        raise ValueError(
            f"{name} must have shape (B, {chain.dof}) for this {chain.dof}-dof "
            f"chain, got {shape}.{hint}")


_CONST_TENSORS = ("joint_origin", "joint_axis", "joint_type", "q_index")


def _fingerprint(chain: CompiledChain):
    """Cheap identity for the constants the fold below reads.

    The chain's own _version counter catches a device move or an explicit
    invalidate_cache(); each tensor's storage pointer and torch in-place
    version counter catch a direct edit of a chain tensor, so nobody has to
    remember to announce one. Both are plain attribute reads, negligible next
    to the FK sweep itself."""
    parts = [getattr(chain, "_version", 0)]
    for name in _CONST_TENSORS:
        t = getattr(chain, name)
        parts.append(t.data_ptr())
        parts.append(t._version)
    return tuple(parts)


def _cache(chain: CompiledChain, device, dtype):
    """Per-(device, dtype) constants: origin R/p, movable index tensors, and
    the constant prismatic direction d = origin_R @ axis.

    Keyed on the chain fingerprint as well as (device, dtype), so an edited
    chain tensor is refolded instead of being silently masked by a stale
    entry."""
    version = _fingerprint(chain)
    key = (str(device), str(dtype))
    cached = getattr(chain, "_fk_cache", None)
    if cached is None or cached[0] != version:
        cached = (version, {})
        object.__setattr__(chain, "_fk_cache", cached)
    store = cached[1]
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
    sc = chain.joint_scale.to(device=device, dtype=dtype)
    off = chain.joint_offset.to(device=device, dtype=dtype)
    entry = {
        "oR": oR, "op": op, "axes": axes,
        "rev": rev, "rev_q": qi[rev], "rev_axes": axes[rev],
        "rev_oR": oR[rev], "rev_scale": sc[rev], "rev_off": off[rev],
        "pris": pris, "pris_q": qi[pris],
        "pris_dir": (oR[pris] @ axes[pris].unsqueeze(-1)).squeeze(-1),
        "pris_op": op[pris], "pris_scale": sc[pris], "pris_off": off[pris],
        # an ordinary chain leaves every joint at scale 1, offset 0, and can
        # skip the affine step entirely
        "affine": bool((sc != 1).any() or (off != 0).any()),
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
        if c["affine"]:
            vals = vals * c["rev_scale"] + c["rev_off"]
        R = T.axis_angle_to_matrix(
            c["rev_axes"].unsqueeze(0).expand(B, -1, 3), vals)    # (B, r, 3, 3)
        local_R = local_R.clone()
        local_R[:, c["rev"]] = c["rev_oR"].unsqueeze(0) @ R
    if c["pris"].numel():
        vals = q[:, c["pris_q"]]                                  # (B, p)
        if c["affine"]:
            vals = vals * c["pris_scale"] + c["pris_off"]
        local_p = local_p.clone()
        local_p[:, c["pris"]] = (c["pris_op"].unsqueeze(0)
                                 + c["pris_dir"].unsqueeze(0) * vals.unsqueeze(-1))
    return local_R, local_p


def fk_rp(chain: CompiledChain, q: torch.Tensor):
    """World rotations and positions as per-link lists: (wR, wp) with
    wR[i] (B, 3, 3) and wp[i] (B, 3). The hot path for IK and Jacobians,
    which never need the assembled 4x4 tensor (assembly is ~a third of FK
    time at large batch).

    q must be a floating tensor of shape (B, dof); anything else raises
    TypeError (dtype) or ValueError (shape)."""
    _check_q(chain, q)
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
    """q (B, dof) -> world transforms (B, n_links, 4, 4).

    q must be a floating tensor of shape (B, dof); anything else raises
    TypeError (dtype) or ValueError (shape)."""
    _check_q(chain, q)
    B = q.shape[0]
    wR, wp = fk_rp(chain, q)
    M = torch.zeros(B, chain.n_links, 4, 4, dtype=q.dtype, device=q.device)
    M[:, :, :3, :3] = torch.stack(wR, dim=1)
    M[:, :, :3, 3] = torch.stack(wp, dim=1)
    M[:, :, 3, 3] = 1.0
    return M
