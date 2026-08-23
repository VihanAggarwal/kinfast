# src/kinfast/ik.py
"""Batched damped-least-squares (Levenberg-Marquardt) inverse kinematics.

dq = J^T (J J^T + lambda^2 I)^-1 e,  applied iteratively, clamped to limits.
Differentiable: the whole loop is autograd-traceable. Set pos_only=True to solve
position only (3D error), the robust default for the demo.

Single-seed DLS gets stuck in local minima for redundant/spatial arms. Pass
restarts>1 to run K random seeds per target in ONE batch and keep the best.
Batching makes the extra seeds nearly free, and solve rates go up a lot.
"""
import torch
from kinfast.fk import forward_kinematics, fk_rp
from kinfast.jacobian import jacobian, jacobian_rp
from kinfast import transforms as T
from kinfast.compile import CompiledChain


def _solve_from_seed(chain, target, q0, link_index, iters, damping, step,
                     pos_only, tol, check_every=10):
    """Core DLS iteration from a given seed. Returns (q, final_error)."""
    device, dtype = q0.device, q0.dtype
    q = q0.clone()
    # the seed fixes the working dtype; limits and target follow it (clamp
    # would otherwise silently promote q to the limits' dtype mid-loop)
    lo = chain.lower.to(device=device, dtype=dtype)
    hi = chain.upper.to(device=device, dtype=dtype)
    target = target.to(device=device, dtype=dtype)
    m = 3 if pos_only else 6
    eye = torch.eye(m, dtype=dtype, device=device)
    lam2 = damping * damping
    final_err = torch.full((q.shape[0],), float("inf"), dtype=dtype, device=device)
    tgt_p = target[:, :3, 3]
    tgt_R = target[:, :3, :3]
    for it in range(iters):
        # R/p fast path: no 4x4 assembly, FK computed once per iteration
        rp = fk_rp(chain, q)
        wR, wp = rp
        p_err = tgt_p - wp[link_index]
        if pos_only:
            e = p_err
        else:
            w_err = T.so3_log(tgt_R @ wR[link_index].transpose(-1, -2))
            e = torch.cat([p_err, w_err], dim=-1)
        J = jacobian_rp(chain, q, link_index, rp=rp)
        if pos_only:
            J = J[:, :3, :]
        JT = J.transpose(-1, -2)
        H = J @ JT + lam2 * eye
        dq = (JT @ torch.linalg.solve(H, e.unsqueeze(-1))).squeeze(-1)
        q = torch.clamp(q + step * dq, lo, hi)
        final_err = e.norm(dim=-1)
        # the convergence check is a host sync (a stall on GPU); do it every
        # check_every iterations instead of every one
        if (it + 1) % check_every == 0 and bool((final_err < tol).all()):
            break
    return q, final_err


def ik(chain: CompiledChain, target: torch.Tensor, q0: torch.Tensor = None,
       link_index: int = None, iters: int = 100, damping: float = 0.05,
       step: float = 1.0, pos_only: bool = False, tol: float = 1e-4,
       restarts: int = 1, check_every: int = 10):
    """Batched IK. target (B,4,4) -> (q (B,dof), info).

    restarts>1 samples that many random seeds per target from the joint limits,
    solves all B*restarts in one batch, and returns the best-per-target.
    q0 is used only when restarts==1; if q0 is None a single random seed is drawn.
    """
    B = target.shape[0]
    # working dtype: the caller's q0 when given, else the target's. The chain
    # may have been compiled in either precision; its constants get cast.
    if q0 is not None and restarts <= 1:
        device, dtype = q0.device, q0.dtype
    else:
        device, dtype = target.device, target.dtype
    lo = chain.lower.to(device=device, dtype=dtype)
    hi = chain.upper.to(device=device, dtype=dtype)

    def _sample(n):
        return lo + (hi - lo) * torch.rand(n, chain.dof, dtype=dtype, device=device)

    if restarts <= 1:
        seed = q0.clone() if q0 is not None else _sample(B)
        q, err = _solve_from_seed(chain, target, seed, link_index, iters,
                                  damping, step, pos_only, tol, check_every)
        return q, {"iters": iters, "final_error": err.detach()}

    # Multi-seed: tile targets, sample K seeds each, solve together, pick best.
    tgt = target.repeat_interleave(restarts, dim=0)            # (B*K, 4, 4)
    seeds = _sample(B * restarts)
    q_all, err_all = _solve_from_seed(chain, tgt, seeds, link_index, iters,
                                      damping, step, pos_only, tol, check_every)
    err = err_all.view(B, restarts)                            # (B, K)
    best = err.argmin(dim=1)                                    # (B,)
    q_all = q_all.view(B, restarts, chain.dof)
    gather_idx = best.view(B, 1, 1).expand(B, 1, chain.dof)
    q_best = q_all.gather(1, gather_idx).squeeze(1)             # (B, dof)
    best_err = err.gather(1, best.view(B, 1)).squeeze(1)
    return q_best, {"iters": iters, "final_error": best_err.detach(),
                    "restarts": restarts}
