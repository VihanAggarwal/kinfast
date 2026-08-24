# src/kinfast/com.py
"""Whole-body center of mass and its Jacobian.

The COM is the mass-weighted average of every link's center of mass expressed
in the world frame:

    c(q) = (1 / M) * sum_i m_i * (R_i(q) c_i + p_i(q))

with M the total mass, c_i the link-frame COM and (R_i, p_i) the link's world
pose. Differentiating gives the COM Jacobian Jc = dc/dq, whose columns say how
the whole body's balance point drifts when one joint moves. Both are what you
need for balance and zero-moment-point work on legged robots, for reasoning
about a manipulator's base reaction, and for keeping a payload's COM over a
support region.

The Jacobian is assembled in one pass over the tree rather than by summing a
per-link COM Jacobian. Joint i only moves the links in its own subtree, and for
a revolute joint the contribution of that whole subtree collapses to a single
cross product:

    sum_{L in subtree(i)} m_L * (a_i x (c_L - p_i))
        = a_i x (S_i - M_i p_i)

where S_i = sum m_L c_L and M_i = sum m_L over the subtree. A prismatic joint
simply translates its subtree, contributing M_i * a_i. So one reverse sweep
over the topological order builds every (S_i, M_i), and one forward pass fills
the columns: linear in the number of links instead of quadratic.

Everything is batched over a leading B dimension, and q fixes the working dtype
and device the same way it does in fk, dynamics and analysis. Both functions
are differentiable in q.
"""
import torch

from kinfast.compile import CompiledChain
from kinfast.fk import fk_rp


def _consts(chain: CompiledChain, device, dtype):
    """Per-(device, dtype) copies of the mass constants, cached on the chain.

    Mirrors fk._cache and dynamics._consts: the chain is compiled once in some
    dtype, but callers may run it in another (float64 finite differences over a
    float32 chain, say), so the constants are cast on first use and kept.
    """
    key = (str(device), str(dtype))
    store = getattr(chain, "_com_cache", None)
    if store is None:
        store = {}
        object.__setattr__(chain, "_com_cache", store)
    if key in store:
        return store[key]
    mass = chain.link_mass.to(device=device, dtype=dtype)
    children = [[] for _ in range(chain.n_links)]
    for i in range(chain.n_links):
        p = int(chain.parent[i])
        if p >= 0:
            children[p].append(i)
    entry = {
        "mass": mass,
        "com": chain.link_com.to(device=device, dtype=dtype),
        "axis": chain.joint_axis.to(device=device, dtype=dtype),
        "total": mass.sum(),
        "massive": [i for i in range(chain.n_links)
                    if float(chain.link_mass[i]) != 0.0],
        "children": children,
    }
    store[key] = entry
    return entry


def _check_q(chain: CompiledChain, q: torch.Tensor):
    """Reject shapes that would otherwise fail deep inside FK with an opaque
    broadcasting error."""
    if q.dim() != 2 or q.shape[1] != chain.dof:
        raise ValueError(
            f"q must have shape (B, {chain.dof}) for this chain, got "
            f"{tuple(q.shape)}")


def _require_mass(consts):
    if float(consts["total"]) <= 0.0:
        raise ValueError(
            "total mass is zero, so the center of mass is undefined. Add "
            "<inertial> elements to the model (MJCF bodies without <inertial> "
            "or geoms with mass get zero mass in kinfast).")


def total_mass(chain: CompiledChain, dtype=None, device=None) -> torch.Tensor:
    """Sum of every link's mass, as a 0-dim tensor.

    Defaults to the dtype and device the chain was compiled in; pass either to
    match a working dtype (float64 for finite differences, say). Zero-mass
    links contribute nothing, so a model with no inertials returns 0 rather
    than raising, which lets a caller test for it.
    """
    m = chain.link_mass
    if dtype is not None or device is not None:
        m = m.to(dtype=dtype or m.dtype, device=device or m.device)
    return m.sum()


def _com_from_rp(consts, wR, wp):
    """Mass-weighted world COM sum, given the FK rotation/position lists."""
    acc = None
    for L in consts["massive"]:
        c_world = wp[L] + wR[L] @ consts["com"][L]        # (B, 3)
        term = consts["mass"][L] * c_world
        acc = term if acc is None else acc + term
    return acc


def com(chain: CompiledChain, q: torch.Tensor) -> torch.Tensor:
    """Whole-body center of mass in world coordinates. q (B, dof) -> (B, 3).

    Massless links are skipped, so a model whose base link carries no inertial
    costs nothing extra.
    """
    _check_q(chain, q)
    consts = _consts(chain, q.device, q.dtype)
    _require_mass(consts)
    wR, wp = fk_rp(chain, q)
    return _com_from_rp(consts, wR, wp) / consts["total"]


def com_jacobian(chain: CompiledChain, q: torch.Tensor) -> torch.Tensor:
    """Jacobian of the whole-body COM. q (B, dof) -> (B, 3, dof).

    Column j is d c / d q_j: the world-frame velocity the COM picks up per unit
    velocity of joint j. There is no angular part, because a point has no
    orientation.
    """
    _check_q(chain, q)
    consts = _consts(chain, q.device, q.dtype)
    _require_mass(consts)
    wR, wp = fk_rp(chain, q)
    B = q.shape[0]
    mass, com_local, axis = consts["mass"], consts["com"], consts["axis"]
    children = consts["children"]

    # Reverse topological sweep: parents are visited after all their children,
    # so each subtree's mass and mass-weighted COM sum are ready when needed.
    sub_sum = [None] * chain.n_links      # (B, 3) sum of m_L * c_L over subtree
    sub_mass = [None] * chain.n_links     # 0-dim scalar
    for i in reversed(chain.topo_order):
        s = mass[i] * (wp[i] + wR[i] @ com_local[i])
        m = mass[i]
        for ch in children[i]:
            s = s + sub_sum[ch]
            m = m + sub_mass[ch]
        sub_sum[i] = s
        sub_mass[i] = m

    J = torch.zeros(B, 3, chain.dof, dtype=q.dtype, device=q.device)
    for i in range(chain.n_links):
        col = int(chain.q_index[i])
        if col < 0:
            continue
        if float(sub_mass[i]) == 0.0:
            continue                      # joint carries no mass downstream
        axis_w = wR[i] @ axis[i]          # (B, 3), axis in world coordinates
        if int(chain.joint_type[i]) == 1:  # revolute
            lever = sub_sum[i] - sub_mass[i] * wp[i]
            J[:, :, col] = torch.cross(axis_w, lever, dim=-1)
        else:                              # prismatic
            J[:, :, col] = sub_mass[i] * axis_w
    return J / consts["total"]
