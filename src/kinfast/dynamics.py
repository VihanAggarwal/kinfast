# src/kinfast/dynamics.py
"""Batched rigid-body dynamics built on the kinematics layer.

Approach (correct-by-construction, each piece independently checkable):
  M(q)      mass matrix   = sum_i [ m_i Jv_i^T Jv_i + Jw_i^T (R_i Ic_i R_i^T) Jw_i ]
  g(q)      gravity torque= -sum_i Jv_i^T (m_i g_vec)          (= dU/dq)
  c(q,qd)   Coriolis/cent.= Mdot(q,qd) qd - 1/2 d/dq (qd^T M qd)   (Christoffel identity)
  tau       inverse dyn.  = M qdd + c + g
  qdd       forward dyn.  = M^{-1} (tau - c - g)

Jv_i / Jw_i are the linear/angular Jacobians of link i's center of mass. Coriolis
uses autograd of M, so no hand-derived Christoffel symbols. All batched, float ok.

Every function accepts q in any float dtype and on any device: the chain's
inertial constants are cast once per (device, dtype) and cached on the chain,
the same way fk does for the kinematic constants. The Coriolis term keeps its
own autograd graph when q or qd require grad, so losses built on coriolis,
inverse_dynamics, forward_dynamics or control.simulate differentiate
correctly; it also works under torch.no_grad().
"""
import torch
from kinfast.fk import forward_kinematics
from kinfast.compile import CompiledChain

GRAVITY = 9.81


def _consts(chain: CompiledChain, device, dtype):
    """Per-(device, dtype) copies of the inertial constants and joint axes."""
    key = (str(device), str(dtype))
    store = getattr(chain, "_dyn_cache", None)
    if store is None:
        store = {}
        object.__setattr__(chain, "_dyn_cache", store)
    if key in store:
        return store[key]
    entry = {
        "mass": chain.link_mass.to(device=device, dtype=dtype),
        "com": chain.link_com.to(device=device, dtype=dtype),
        "inertia": chain.link_inertia.to(device=device, dtype=dtype),
        "axis": chain.joint_axis.to(device=device, dtype=dtype),
        "active": [L for L in range(chain.n_links)
                   if float(chain.link_mass[L]) != 0.0
                   or float(chain.link_inertia[L].abs().sum()) != 0.0],
    }
    store[key] = entry
    return entry


def _com_jacobian(chain, world, L, consts=None):
    """Linear/angular Jacobian of link L's COM. Returns Jv,Jw (B,3,dof), R (B,3,3)."""
    B = world.shape[0]
    device, dtype = world.device, world.dtype
    if consts is None:
        consts = _consts(chain, device, dtype)
    R = world[:, L, :3, :3]
    com_h = torch.cat([consts["com"][L],
                       torch.ones(1, dtype=dtype, device=device)])
    com_world = (world[:, L] @ com_h)[:, :3]                     # (B,3)
    Jv = torch.zeros(B, 3, chain.dof, dtype=dtype, device=device)
    Jw = torch.zeros(B, 3, chain.dof, dtype=dtype, device=device)
    i = L
    while i >= 0:
        jt = int(chain.joint_type[i])
        if jt != 0:
            col = int(chain.q_index[i])
            axis_w = world[:, i, :3, :3] @ consts["axis"][i]
            p_i = world[:, i, :3, 3]
            if jt == 1:  # revolute
                Jv[:, :, col] = torch.cross(axis_w, com_world - p_i, dim=-1)
                Jw[:, :, col] = axis_w
            else:        # prismatic
                Jv[:, :, col] = axis_w
        i = int(chain.parent[i])
    return Jv, Jw, R


def _active_links(chain):
    for L in range(chain.n_links):
        if float(chain.link_mass[L]) != 0.0 or float(chain.link_inertia[L].abs().sum()) != 0.0:
            yield L


def _require_inertials(chain, consts):
    if not consts["active"]:
        raise ValueError(
            "no link has mass or inertia, so the mass matrix is identically zero. "
            "Add <inertial> elements to the model (MJCF bodies without <inertial> "
            "or geoms with mass get zero mass in kinfast).")


def mass_matrix(chain: CompiledChain, q: torch.Tensor) -> torch.Tensor:
    """(B,dof) -> (B,dof,dof) symmetric positive-definite mass matrix."""
    consts = _consts(chain, q.device, q.dtype)
    _require_inertials(chain, consts)
    world = forward_kinematics(chain, q)
    B = q.shape[0]
    M = torch.zeros(B, chain.dof, chain.dof, dtype=q.dtype, device=q.device)
    for L in consts["active"]:
        m = consts["mass"][L]
        Jv, Jw, R = _com_jacobian(chain, world, L, consts)
        Ic = R @ consts["inertia"][L] @ R.transpose(-1, -2)      # (B,3,3)
        M = M + m * (Jv.transpose(-1, -2) @ Jv) \
              + Jw.transpose(-1, -2) @ Ic @ Jw
    return M


def gravity(chain: CompiledChain, q: torch.Tensor, g: float = GRAVITY) -> torch.Tensor:
    """Generalized gravity torque g(q) = dU/dq. (B,dof)."""
    consts = _consts(chain, q.device, q.dtype)
    world = forward_kinematics(chain, q)
    B = q.shape[0]
    gq = torch.zeros(B, chain.dof, dtype=q.dtype, device=q.device)
    g_vec = torch.tensor([0.0, 0.0, -g], dtype=q.dtype, device=q.device)
    for L in consts["active"]:
        m = consts["mass"][L]
        if float(m) == 0.0:
            continue
        Jv, _, _ = _com_jacobian(chain, world, L, consts)
        gq = gq - m * (Jv.transpose(-1, -2) @ g_vec)
    return gq


def coriolis(chain: CompiledChain, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """Coriolis + centrifugal generalized force c(q,qd) = C(q,qd) qd. (B,dof).

    Built from autograd of the mass matrix:
      term1 = Mdot(q,qd) qd  (directional derivative of M along qd, times qd)
      term2 = 1/2 d/dq (qd^T M qd)
    Both are taken with create_graph when q or qd require grad, so the
    result is itself differentiable. Runs under torch.no_grad() by enabling
    grad mode locally; torch.inference_mode cannot be re-enabled from inside,
    so that case raises.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("coriolis needs autograd; call it outside "
                         "torch.inference_mode (torch.no_grad is fine)")
    if qd.dtype != q.dtype or qd.device != q.device:
        qd = qd.to(dtype=q.dtype, device=q.device)
    needs_graph = torch.is_grad_enabled() and (q.requires_grad or qd.requires_grad)
    with torch.enable_grad():
        qr = q if q.requires_grad else q.detach().requires_grad_(True)
        M = mass_matrix(chain, qr)
        Mqd = (M @ qd.unsqueeze(-1)).squeeze(-1)                   # (B,dof)
        # term1 = (d(M qd)/dq) qd via the double-backward trick: differentiate
        # v^T (M qd) w.r.t. q with a dummy v, then that w.r.t. v along qd.
        v = torch.ones_like(Mqd, requires_grad=True)
        gq, = torch.autograd.grad((Mqd * v).sum(), qr, create_graph=True)
        term1, = torch.autograd.grad((gq * qd).sum(), v, create_graph=True)
        quad = 0.5 * (qd.unsqueeze(1) @ M @ qd.unsqueeze(-1)).reshape(-1).sum()
        term2, = torch.autograd.grad(quad, qr, create_graph=True)
        out = term1 - term2
    return out if needs_graph else out.detach()


def inverse_dynamics(chain, q, qd, qdd, use_gravity=True):
    """tau to realize (q,qd,qdd). (B,dof)."""
    M = mass_matrix(chain, q)
    qdd = qdd.to(dtype=q.dtype, device=q.device)
    tau = (M @ qdd.unsqueeze(-1)).squeeze(-1) + coriolis(chain, q, qd)
    if use_gravity:
        tau = tau + gravity(chain, q)
    return tau


def forward_dynamics(chain, q, qd, tau, use_gravity=True):
    """qdd produced by tau at state (q,qd). (B,dof)."""
    M = mass_matrix(chain, q)
    tau = tau.to(dtype=q.dtype, device=q.device)
    bias = coriolis(chain, q, qd)
    if use_gravity:
        bias = bias + gravity(chain, q)
    return torch.linalg.solve(M, (tau - bias).unsqueeze(-1)).squeeze(-1)
