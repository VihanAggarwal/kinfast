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
"""
import torch
from kinfast.fk import forward_kinematics
from kinfast.compile import CompiledChain

GRAVITY = 9.81


def _com_jacobian(chain, world, L):
    """Linear/angular Jacobian of link L's COM. Returns Jv,Jw (B,3,dof), R (B,3,3)."""
    B = world.shape[0]
    device, dtype = world.device, world.dtype
    R = world[:, L, :3, :3]
    com_h = torch.cat([chain.link_com[L],
                       torch.ones(1, dtype=dtype, device=device)])
    com_world = (world[:, L] @ com_h)[:, :3]                     # (B,3)
    Jv = torch.zeros(B, 3, chain.dof, dtype=dtype, device=device)
    Jw = torch.zeros(B, 3, chain.dof, dtype=dtype, device=device)
    i = L
    while i >= 0:
        jt = int(chain.joint_type[i])
        if jt != 0:
            col = int(chain.q_index[i])
            axis_w = world[:, i, :3, :3] @ chain.joint_axis[i]
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


def mass_matrix(chain: CompiledChain, q: torch.Tensor) -> torch.Tensor:
    """(B,dof) -> (B,dof,dof) symmetric positive-definite mass matrix."""
    world = forward_kinematics(chain, q)
    B = q.shape[0]
    M = torch.zeros(B, chain.dof, chain.dof, dtype=q.dtype, device=q.device)
    for L in _active_links(chain):
        m = chain.link_mass[L]
        Jv, Jw, R = _com_jacobian(chain, world, L)
        Ic = R @ chain.link_inertia[L] @ R.transpose(-1, -2)     # (B,3,3)
        M = M + m * (Jv.transpose(-1, -2) @ Jv) \
              + Jw.transpose(-1, -2) @ Ic @ Jw
    return M


def gravity(chain: CompiledChain, q: torch.Tensor, g: float = GRAVITY) -> torch.Tensor:
    """Generalized gravity torque g(q) = dU/dq. (B,dof)."""
    world = forward_kinematics(chain, q)
    B = q.shape[0]
    gq = torch.zeros(B, chain.dof, dtype=q.dtype, device=q.device)
    g_vec = torch.tensor([0.0, 0.0, -g], dtype=q.dtype, device=q.device)
    for L in _active_links(chain):
        m = chain.link_mass[L]
        if float(m) == 0.0:
            continue
        Jv, _, _ = _com_jacobian(chain, world, L)
        gq = gq - m * (Jv.transpose(-1, -2) @ g_vec)
    return gq


def coriolis(chain: CompiledChain, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """Coriolis + centrifugal generalized force c(q,qd) = C(q,qd) qd. (B,dof)."""
    def Mfun(qq):
        return mass_matrix(chain, qq)

    q0 = q.detach()
    # term1 = Mdot(q,qd) qd : directional derivative of M along qd, contracted with qd
    _, Mdir = torch.autograd.functional.jvp(Mfun, q0, qd)         # (B,dof,dof)
    term1 = (Mdir @ qd.unsqueeze(-1)).squeeze(-1)
    # term2 = 1/2 d/dq (qd^T M(q) qd)
    qr = q0.clone().requires_grad_(True)
    M = mass_matrix(chain, qr)
    quad = 0.5 * (qd.unsqueeze(1) @ M @ qd.unsqueeze(-1)).reshape(-1).sum()
    term2, = torch.autograd.grad(quad, qr)
    return term1 - term2


def inverse_dynamics(chain, q, qd, qdd, use_gravity=True):
    """tau to realize (q,qd,qdd). (B,dof)."""
    M = mass_matrix(chain, q)
    tau = (M @ qdd.unsqueeze(-1)).squeeze(-1) + coriolis(chain, q, qd)
    if use_gravity:
        tau = tau + gravity(chain, q)
    return tau


def forward_dynamics(chain, q, qd, tau, use_gravity=True):
    """qdd produced by tau at state (q,qd). (B,dof)."""
    M = mass_matrix(chain, q)
    bias = coriolis(chain, q, qd)
    if use_gravity:
        bias = bias + gravity(chain, q)
    return torch.linalg.solve(M, (tau - bias).unsqueeze(-1)).squeeze(-1)
