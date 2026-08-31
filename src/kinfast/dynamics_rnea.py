# src/kinfast/dynamics_rnea.py
"""Recursive Newton-Euler inverse dynamics and the composite-rigid-body mass matrix.

The `dynamics` module builds M, c and g out of Jacobians and autograd. That is
easy to trust but expensive: the mass matrix costs one Jacobian per link, and
the Coriolis term needs a double backward through it. This module does the same
job the way physics engines do it, with tree sweeps and no autograd:

  rnea(chain, q, qd, qdd)   O(n)    tau = M(q) qdd + c(q, qd) + g(q)
  crba(chain, q)            O(n d)  M(q), d = tree depth

Both work on the same CompiledChain, are batched over a leading B dimension,
follow the caller's dtype and device (the working dtype is q's, like the rest of
the library), and are made of plain tensor ops so they differentiate.

How RNEA works. Every quantity lives in its own link's frame, which keeps the
inertia constant and turns the recursion into small 3-vector algebra. The
forward sweep pushes velocity and acceleration from the base outward:

  w_i  = R_i^T w_p + z_i qd_i                       (revolute; z_i is the axis)
  wd_i = R_i^T wd_p + (R_i^T w_p) x (z_i qd_i) + z_i qdd_i
  a_i  = R_i^T (a_p + wd_p x p_i + w_p x (w_p x p_i))

with an extra 2 w_i x (z_i qd_i) + z_i qdd_i on a_i for a prismatic joint, since
its frame origin slides. Gravity enters as a fictitious base acceleration
a_base = -g, which is why a gravity-free call is the same sweep with that seed
set to zero. From the link's own acceleration we get the net wrench its inertia
demands (Newton for the force, Euler for the moment). The backward sweep then
sums those wrenches from the leaves inward, and the torque at a joint is the
component of the accumulated wrench along that joint's axis.

How CRBA works. M's (i, j) entry is the force felt at joint j when joint i alone
accelerates, so we take the composite inertia of the subtree hanging off link i
(one reverse sweep, parallel-axis at every step), apply it to joint i's motion
axis, and carry the resulting wrench up the ancestor chain, reading off one
column of M as we go. Two things keep that cheap. The composite inertia is kept
as (mass, first moment, inertia about the link origin) rather than a 6x6 spatial
matrix, which is the same algebra with a third of the arithmetic. And the
ancestor walks for all joints advance together, one batched step per tree level,
so the cost in python-level operations is the depth of the tree and not the
number of (joint, ancestor) pairs.
"""
import torch

from kinfast.compile import CompiledChain
from kinfast.fk import _local_rp

GRAVITY = 9.81


def _crba_plan(parent, qidx, jtype, axes, comp_mass, device, dtype):
    """Precompute the level-by-level ancestor walk CRBA uses to fill M.

    For every movable joint we list its chain of ancestors. Sorting those chains
    longest first means the joints still walking at level t are always a prefix
    of the list, so one slice keeps the batched state aligned. Each level records
    which link's transform to apply and, where the ancestor carries a joint,
    where the resulting number lands in M.
    """
    paths = []
    for i in range(len(parent)):
        if qidx[i] < 0:
            continue
        path, j = [], i
        while j >= 0:
            path.append(j)
            j = parent[j]
        paths.append(path)
    paths.sort(key=len, reverse=True)

    def lt(v):
        return torch.tensor(v, dtype=torch.long, device=device)

    heads = [p[0] for p in paths]
    cols = [qidx[i] for i in heads]
    plan = {
        "heads": heads,
        "cols": lt(cols),
        "axes": (axes[lt(heads)] if heads else axes[:0]),
        "rev": torch.tensor([jtype[i] == 1 for i in heads],
                            dtype=torch.bool, device=device),
        "mass": (comp_mass[lt(heads)] if heads else comp_mass[:0]),
        "levels": [],
    }
    depth = (max(len(p) for p in paths) - 1) if paths else 0
    for t in range(1, depth + 1):
        k = sum(1 for p in paths if len(p) > t)
        frm = [paths[a][t - 1] for a in range(k)]
        to = [paths[a][t] for a in range(k)]
        sel = [a for a in range(k) if qidx[to[a]] >= 0]
        plan["levels"].append({
            "k": k,
            "frm": lt(frm),
            "sel": lt(sel),
            "rows": lt([qidx[to[a]] for a in sel]),
            "cols": lt([cols[a] for a in sel]),
            "rev": torch.tensor([[jtype[to[a]] == 1] for a in sel],
                                dtype=torch.bool, device=device),
            "axes": (axes[lt([to[a] for a in sel])] if sel else axes[:0]),
        })
    return plan


def _consts(chain: CompiledChain, device, dtype):
    """Per-(device, dtype) inertial constants, topology as plain ints, CRBA plan.

    Cached on the chain the same way fk and dynamics cache theirs, so a hot loop
    pays the cast, the int() unpacking and the plan build once.
    """
    key = (str(device), str(dtype))
    store = getattr(chain, "_rnea_cache", None)
    if store is None:
        store = {}
        object.__setattr__(chain, "_rnea_cache", store)
    if key in store:
        return store[key]

    n = chain.n_links
    mass = chain.link_mass.to(device=device, dtype=dtype)
    com = chain.link_com.to(device=device, dtype=dtype)
    inertia_com = chain.link_inertia.to(device=device, dtype=dtype)
    axes = chain.joint_axis.to(device=device, dtype=dtype)

    # Inertia about the link frame origin (parallel axis): I_o = I_c + m (c.c E - c c^T).
    eye = torch.eye(3, device=device, dtype=dtype)
    cc = (com * com).sum(-1).reshape(n, 1, 1)
    outer = com.unsqueeze(-1) * com.unsqueeze(-2)
    inertia_origin = inertia_com + mass.reshape(n, 1, 1) * (cc * eye - outer)

    parent = [int(x) for x in chain.parent]
    jtype = [int(x) for x in chain.joint_type]
    qidx = [int(x) for x in chain.q_index]
    jscale = [float(x) for x in chain.joint_scale]
    order = list(chain.topo_order)

    # Subtree mass is purely topological, so it never has to be recomputed.
    comp_mass_f = [float(x) for x in chain.link_mass]
    for i in reversed(order):
        p = parent[i]
        if p >= 0:
            comp_mass_f[p] += comp_mass_f[i]
    comp_mass = torch.tensor(comp_mass_f, dtype=dtype, device=device)

    entry = {
        "jscale": jscale,                      # (n,) mimic factor, 1 if none
        "mass": mass,                          # (n,)
        "com": com,                            # (n,3)
        "inertia_com": inertia_com,            # (n,3,3) about COM, link frame
        "inertia_origin": inertia_origin,      # (n,3,3) about link origin
        "moment": mass.unsqueeze(-1) * com,    # (n,3) first moment h = m c
        "axes": axes,                          # (n,3)
        "eye": eye,
        "parent": parent,
        "jtype": jtype,
        "qidx": qidx,
        "order": order,
        "comp_mass_f": comp_mass_f,
        "plan": _crba_plan(parent, qidx, jtype, axes, comp_mass, device, dtype),
        "has_inertia": any(float(chain.link_mass[L]) != 0.0
                           or float(chain.link_inertia[L].abs().sum()) != 0.0
                           for L in range(n)),
    }
    store[key] = entry
    return entry


def _mv(R, v):
    """Batched matrix times vector: (..., 3, 3) x (..., 3) -> (..., 3)."""
    return (R @ v.unsqueeze(-1)).squeeze(-1)


def _cross(a, b):
    return torch.cross(a, b, dim=-1)


def _gravity_vector(chain, gravity, dtype, device):
    """Resolve the `gravity` argument into a world vector, or None for no gravity.

    True (the default) means the chain's own vector, which an MJCF `<option
    gravity>` can override; False or None turns gravity off; a scalar g keeps the
    historical (0, 0, -g) meaning; a 3-vector is used as is.
    """
    if gravity is None or gravity is False:
        return None
    if gravity is True:
        vec = tuple(getattr(chain, "gravity", (0.0, 0.0, -GRAVITY)))
    elif isinstance(gravity, torch.Tensor):
        if gravity.numel() != 3:
            raise ValueError(
                f"gravity vector must have 3 components, got {gravity.numel()}")
        return gravity.reshape(3).to(dtype=dtype, device=device)
    elif isinstance(gravity, (list, tuple)):
        if len(gravity) != 3:
            raise ValueError(
                f"gravity vector must have 3 components, got {len(gravity)}")
        vec = tuple(gravity)
    else:
        vec = (0.0, 0.0, -float(gravity))
    return torch.tensor(vec, dtype=dtype, device=device)


def _check_q(chain, q, name="q"):
    if q.dim() != 2 or q.shape[1] != chain.dof:
        raise ValueError(f"{name} must have shape (B, {chain.dof}), "
                         f"got {tuple(q.shape)}")


def _like_q(x, q):
    """Bring qd/qdd onto q's dtype and device; q fixes the working dtype."""
    return x.to(device=q.device, dtype=q.dtype)


def rnea(chain: CompiledChain, q, qd, qdd, gravity=True) -> torch.Tensor:
    """Inverse dynamics by recursive Newton-Euler: (B,dof) joint torques.

    Returns the tau that realizes acceleration qdd at state (q, qd), which is
    M(q) qdd + c(q, qd) + g(q) but computed in O(n) without ever forming M.
    Pass gravity=False for the gravity-free torque, or a scalar or 3-vector to
    override the chain's own gravity.

    Revolute and prismatic joints are both handled; fixed links still propagate
    motion and still push their inertia back down the tree, they just have no
    torque of their own.
    """
    if chain.dof == 0:
        return torch.zeros(q.shape[0], 0, dtype=q.dtype, device=q.device)
    _check_q(chain, q)
    qd, qdd = _like_q(qd, q), _like_q(qdd, q)
    _check_q(chain, qd, "qd")
    _check_q(chain, qdd, "qdd")

    c = _consts(chain, q.device, q.dtype)
    B = q.shape[0]
    n = chain.n_links
    local_R, local_p = _local_rp(chain, q)
    # Unbind once: both sweeps touch every link's local transform and re-slicing
    # inside the loop is pure overhead.
    lR = local_R.unbind(1)
    lp = local_p.unbind(1)

    g_vec = _gravity_vector(chain, gravity, q.dtype, q.device)
    zero3 = torch.zeros(B, 3, dtype=q.dtype, device=q.device)
    # The base frame is at rest; gravity is faked as an upward base acceleration.
    a_base = zero3 if g_vec is None else (-g_vec).expand(B, 3)

    parent, jtype, qidx, axes = c["parent"], c["jtype"], c["qidx"], c["axes"]
    jscale = c["jscale"]
    com, mass, Ic = c["com"], c["mass"], c["inertia_com"]

    w = [None] * n
    wd = [None] * n
    acc = [None] * n
    force = [None] * n
    moment = [None] * n

    # Forward sweep: base -> leaves, carrying velocity and acceleration.
    for i in c["order"]:
        p = parent[i]
        R = lR[i]                                    # link i frame -> parent frame
        Rt = R.transpose(-1, -2)
        pi = lp[i]                                   # link i origin in parent coords
        if p < 0:
            w_p, wd_p, a_p = zero3, zero3, a_base
        else:
            w_p, wd_p, a_p = w[p], wd[p], acc[p]

        a_child = a_p + _cross(wd_p, pi) + _cross(w_p, _cross(w_p, pi))
        w_i = _mv(Rt, w_p)
        wd_i = _mv(Rt, wd_p)
        a_i = _mv(Rt, a_child)

        col = qidx[i]
        if col >= 0:
            z = axes[i]                              # unit axis in link i coords
            # a mimic joint turns at scale times the rate of the joint driving
            # it, and reads that joint's column. scale is 1 on every ordinary
            # joint, so this costs nothing on a normal chain.
            k = jscale[i]
            zqd = z * (qd[:, col] * k).unsqueeze(-1)
            zqdd = z * (qdd[:, col] * k).unsqueeze(-1)
            if jtype[i] == 1:                        # revolute
                wd_i = wd_i + _cross(w_i, zqd) + zqdd
                w_i = w_i + zqd
            else:                                    # prismatic
                a_i = a_i + 2.0 * _cross(w_i, zqd) + zqdd

        w[i], wd[i], acc[i] = w_i, wd_i, a_i

        # Wrench this link's own inertia demands, about the link frame origin.
        ci = com[i].expand(B, 3)
        a_com = a_i + _cross(wd_i, ci) + _cross(w_i, _cross(w_i, ci))
        F = mass[i] * a_com
        Ii = Ic[i].expand(B, 3, 3)
        N = _mv(Ii, wd_i) + _cross(w_i, _mv(Ii, w_i))
        force[i] = F
        moment[i] = N + _cross(ci, F)

    # Backward sweep: leaves -> base, summing wrenches and reading off torques.
    tau_cols = [None] * chain.dof
    for i in reversed(c["order"]):
        col = qidx[i]
        if col >= 0:
            src = moment[i] if jtype[i] == 1 else force[i]
            # and it projects its wrench back onto the coordinate that drives
            # it, through the same factor, adding to whatever is already there
            contrib = jscale[i] * (src * axes[i]).sum(-1)
            tau_cols[col] = (contrib if tau_cols[col] is None
                             else tau_cols[col] + contrib)
        p = parent[i]
        if p >= 0:
            f_p = _mv(lR[i], force[i])
            n_p = _mv(lR[i], moment[i]) + _cross(lp[i], f_p)
            force[p] = force[p] + f_p
            moment[p] = moment[p] + n_p

    return torch.stack(tau_cols, dim=1)


def crba(chain: CompiledChain, q) -> torch.Tensor:
    """Joint-space mass matrix by the composite-rigid-body algorithm: (B,dof,dof).

    Same matrix as dynamics.mass_matrix, symmetric and positive definite, but
    built from subtree composite inertias instead of one Jacobian per link.
    """
    if chain.dof == 0:
        return torch.zeros(q.shape[0], 0, 0, dtype=q.dtype, device=q.device)
    if chain.has_mimic:
        # The composite-rigid-body recursion assumes one joint per column, so a
        # mimic joint sharing its driver's column would drop the driver's own
        # contribution. The Jacobian route handles that and gives the same
        # matrix, more slowly. Mimic chains are grippers and linkages, which
        # are small, so the slower route costs nothing that matters here.
        from kinfast.dynamics import mass_matrix as _jac_mass_matrix
        return _jac_mass_matrix(chain, q)
    _check_q(chain, q)
    c = _consts(chain, q.device, q.dtype)
    if not c["has_inertia"]:
        raise ValueError(
            "no link has mass or inertia, so the mass matrix is identically zero. "
            "Add <inertial> elements to the model (MJCF bodies without <inertial> "
            "or geoms with mass get zero mass in kinfast).")

    B = q.shape[0]
    n = chain.n_links
    local_R, local_p = _local_rp(chain, q)
    lR = local_R.unbind(1)
    lp = local_p.unbind(1)
    parent, comp_m, eye = c["parent"], c["comp_mass_f"], c["eye"]

    # Composite inertia of the subtree rooted at each link, in that link's frame,
    # as first moment and inertia about the link origin (mass is precomputed).
    comp_h = [c["moment"][i].expand(B, 3) for i in range(n)]
    comp_I = [c["inertia_origin"][i].expand(B, 3, 3) for i in range(n)]

    for i in reversed(c["order"]):
        p = parent[i]
        if p < 0:
            continue
        R, pv = lR[i], lp[i]
        h_r = _mv(R, comp_h[i])
        I_r = R @ comp_I[i] @ R.transpose(-1, -2)
        # Shift the rotated inertia from link i's origin to its parent's origin.
        # The skew-matrix form of the shift expands to dot and outer products,
        # which is the same result for a fraction of the arithmetic.
        m_i = comp_m[i]
        ph = (pv * h_r).sum(-1).reshape(B, 1, 1)
        pp = (pv * pv).sum(-1).reshape(B, 1, 1)
        hp = h_r.unsqueeze(-1) * pv.unsqueeze(-2)
        ppo = pv.unsqueeze(-1) * pv.unsqueeze(-2)
        comp_I[p] = (comp_I[p] + I_r + (2.0 * ph + m_i * pp) * eye
                     - hp - hp.transpose(-1, -2) - m_i * ppo)
        comp_h[p] = comp_h[p] + h_r + m_i * pv

    plan = c["plan"]
    heads = plan["heads"]
    k0 = len(heads)
    z = plan["axes"].unsqueeze(0).expand(B, k0, 3)            # (B,K,3)
    rev = plan["rev"].reshape(1, k0, 1)
    h0 = torch.stack([comp_h[i] for i in heads], dim=1)       # (B,K,3)
    I0 = torch.stack([comp_I[i] for i in heads], dim=1)       # (B,K,3,3)

    # Wrench produced by unit acceleration of each joint alone, in its own frame.
    # Revolute motion is [z; 0] so the wrench is [I z; -h x z]; prismatic motion
    # is [0; z] so it is [h x z; m z].
    hxz = _cross(h0, z)
    Iz = _mv(I0, z)
    n_f = torch.where(rev, Iz, hxz)
    f_f = torch.where(rev, -hxz, plan["mass"].reshape(1, k0, 1) * z)
    diag = torch.where(plan["rev"].reshape(1, k0),
                       (z * Iz).sum(-1),
                       plan["mass"].reshape(1, k0).expand(B, k0))

    rows = [plan["cols"]]
    cols = [plan["cols"]]
    vals = [diag]
    # One batched step per tree level: every joint still walking moves its wrench
    # up one link, and any ancestor that carries a joint contributes an entry.
    for lev in plan["levels"]:
        k = lev["k"]
        frm = lev["frm"]
        n_f, f_f = n_f[:, :k], f_f[:, :k]
        R = local_R[:, frm]                                   # (B,k,3,3)
        pv = local_p[:, frm]                                  # (B,k,3)
        f_f = _mv(R, f_f)
        n_f = _mv(R, n_f) + _cross(pv, f_f)
        sel = lev["sel"]
        if sel.numel():
            picked = torch.where(lev["rev"], n_f[:, sel], f_f[:, sel])
            vals.append((picked * lev["axes"]).sum(-1))
            rows.append(lev["rows"])
            cols.append(lev["cols"])

    M = torch.zeros(B, chain.dof, chain.dof, dtype=q.dtype, device=q.device)
    ri, ci = torch.cat(rows), torch.cat(cols)
    v = torch.cat(vals, dim=1)
    M[:, ri, ci] = v
    M[:, ci, ri] = v                    # M is symmetric; fill the mirrored half
    return M


def bias(chain: CompiledChain, q, qd, gravity=True) -> torch.Tensor:
    """Velocity and gravity terms c(q,qd) + g(q). (B,dof).

    This is RNEA with zero acceleration, which is exactly what a physics engine
    reports as its bias force. Handy for computed-torque style controllers that
    want the bias separately from M.
    """
    zeros = torch.zeros_like(q)
    return rnea(chain, q, qd, zeros, gravity=gravity)


def gravity_torque(chain: CompiledChain, q, gravity=True) -> torch.Tensor:
    """Gravity torque g(q). (B,dof). RNEA at rest."""
    zeros = torch.zeros_like(q)
    return rnea(chain, q, zeros, zeros, gravity=gravity)


def forward_dynamics(chain: CompiledChain, q, qd, tau, gravity=True) -> torch.Tensor:
    """qdd produced by tau at state (q, qd).

    Uses the standard M \\ (tau - bias) split: one RNEA call for the bias, one
    CRBA for M, then a batched solve.
    """
    qd, tau = _like_q(qd, q), _like_q(tau, q)
    M = crba(chain, q)
    b = bias(chain, q, qd, gravity=gravity)
    return torch.linalg.solve(M, (tau - b).unsqueeze(-1)).squeeze(-1)
