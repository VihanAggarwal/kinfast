# src/kinfast/ik.py
"""Batched damped-least-squares (Levenberg-Marquardt) inverse kinematics.

dq = J^T (J J^T + lambda^2 I)^-1 e,  applied iteratively, clamped to limits.
Differentiable: the whole loop is autograd-traceable. Set pos_only=True to solve
position only (3D error) which is the robust default for the demo.
"""
import torch
from kinfast.fk import forward_kinematics
from kinfast.jacobian import jacobian
from kinfast import transforms as T
from kinfast.compile import CompiledChain


def ik(chain: CompiledChain, target: torch.Tensor, q0: torch.Tensor,
       link_index: int, iters: int = 100, damping: float = 0.05,
       step: float = 1.0, pos_only: bool = False, tol: float = 1e-4):
    device, dtype = q0.device, q0.dtype
    q = q0.clone()
    lo, hi = chain.lower.to(device), chain.upper.to(device)
    m = 3 if pos_only else 6
    eye = torch.eye(m, dtype=dtype, device=device)
    lam2 = damping * damping
    final_err = None
    for _ in range(iters):
        world = forward_kinematics(chain, q)
        T_cur = world[:, link_index]
        e = T.pose_error(T_cur, target)              # (B, 6)
        if pos_only:
            e = e[:, :3]
        J = jacobian(chain, q, link_index)           # (B, 6, dof)
        if pos_only:
            J = J[:, :3, :]
        JT = J.transpose(-1, -2)                     # (B, dof, m)
        H = J @ JT + lam2 * eye                       # (B, m, m)
        rhs = e.unsqueeze(-1)                         # (B, m, 1)
        dq = (JT @ torch.linalg.solve(H, rhs)).squeeze(-1)  # (B, dof)
        q = torch.clamp(q + step * dq, lo, hi)
        final_err = e.norm(dim=-1)
        if bool((final_err < tol).all()):
            break
    info = {"iters": iters, "final_error": final_err.detach()}
    return q, info
