# src/kinfast/ik_task.py
"""Weighted task-space IK with a nullspace posture objective.

Plain `kinfast.ik` treats all six error components alike and has nothing to say
about which solution it lands on when the arm has spare freedom. Real tasks are
rarely like that. Pouring a glass cares about the tool axis but not the spin
around it, a camera cares about where it points and not at all about where it
sits along its own optical axis, and a 7-DOF arm has a whole curve of postures
that reach the same pose, most of which are ugly.

This module solves both with one iteration:

    dq = J_w^+ e_w  +  (I - J_w^+ J_w) k (q_rest - q)

`weights` is a per-component scale on the 6D error [dx dy dz, drx dry drz].
Rows are scaled in the Jacobian too, so a weight of 0 removes that direction
from the task completely (it becomes free), a small weight makes it a soft
preference, and a large one makes it dominate. Position-only IK is just
weights=(1,1,1,0,0,0), and it reduces exactly to `ik(..., pos_only=True)`.

The second term is the classic nullspace projection: it pulls the joints toward
`q_rest` using only the motion the task does not care about, so the posture
improves without moving the tool. `w_rest` is its gain; set it to 0 and you get
pure task-space DLS back.

J_w^+ here is the damped right pseudo-inverse J^T (J J^T + lambda^2 I)^-1, so
the step stays finite at singularities. That damping also means the projector is
only approximately a projector, to O(lambda^2); with the default damping the
posture term perturbs the task error far below the solve tolerance, and you can
shrink lambda if you need the priority to be stricter.

Everything is batched over a leading B dimension, runs on any device, and takes
its working dtype from the caller's q0 (falling back to the target). The loop is
plain tensor math, so gradients flow through the returned q to the target, the
seed, and the rest posture.
"""
import math

import torch
from kinfast.fk import fk_rp
from kinfast.jacobian import jacobian_rp, _resolve_link
from kinfast import transforms as T
from kinfast.compile import CompiledChain

__all__ = ["ik_task", "weighted_dls_step"]


def weighted_dls_step(J: torch.Tensor, e: torch.Tensor, weights=None,
                      damping: float = 0.05,
                      dq_null: torch.Tensor = None) -> torch.Tensor:
    """One damped-least-squares step with row weights and a nullspace term.

    J (B, m, dof), e (B, m), weights broadcastable to (B, m) or None for all
    ones, dq_null (B, dof) the raw joint-space motion you would like in the
    task's nullspace (already scaled by its gain), or None to skip it.

    Returns dq (B, dof). Pure math: no chain, no FK, so it is easy to check
    against a finite-difference Jacobian.
    """
    if weights is not None:
        J = J * weights.unsqueeze(-1)
        e = e * weights
    m = J.shape[-2]
    eye = torch.eye(m, dtype=J.dtype, device=J.device)
    H = J @ J.transpose(-1, -2) + (damping * damping) * eye
    # H is symmetric, so solve(H, J) transposed is J^T (J J^T + lam^2 I)^-1.
    pinv = torch.linalg.solve(H, J).transpose(-1, -2)          # (B, dof, m)
    dq = (pinv @ e.unsqueeze(-1)).squeeze(-1)
    if dq_null is not None:
        # (I - pinv J) dq_null, formed without ever building the dof x dof
        # projector: cheaper and better conditioned for high-DOF arms.
        taken = (pinv @ (J @ dq_null.unsqueeze(-1))).squeeze(-1)
        dq = dq + dq_null - taken
    return dq


def _link_index(chain: CompiledChain, link) -> int:
    """Accept a link name or an index (negatives count from the end)."""
    if isinstance(link, str):
        if link not in chain.link_index:
            raise KeyError(f"unknown link {link!r}; have {chain.link_names}")
        return chain.link_index[link]
    return _resolve_link(chain, link)


def _prep_weights(weights, batch: int, m: int, device, dtype):
    """Normalize `weights` to (1, m) or (batch, m) in the working dtype/device.

    Accepts a python scalar or sequence, a numpy array, or a tensor living on
    any device in any dtype; it is moved and cast to match q.
    """
    if weights is None:
        return None
    w = torch.as_tensor(weights, dtype=dtype, device=device)
    if w.dim() == 0:
        w = w.reshape(1).expand(m)
    if w.shape[-1] != m:
        raise ValueError(
            f"weights must have {m} components (position xyz then rotation "
            f"xyz), got shape {tuple(w.shape)}")
    if w.dim() == 1:
        w = w.unsqueeze(0)
    elif w.dim() != 2 or w.shape[0] not in (1, batch):
        raise ValueError(
            f"weights must be ({m},) or (B, {m}) with B={batch}, got "
            f"shape {tuple(w.shape)}")
    if bool(torch.any(w < 0)):
        raise ValueError("weights must be non-negative")
    return w


def _default_rest(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Midpoint of the joint range, the usual joint-limit-avoidance posture.
    Joints with an infinite range have no midpoint, so they rest at 0."""
    mid = 0.5 * (lo + hi)
    return torch.where(torch.isfinite(mid), mid, torch.zeros_like(mid))


def ik_task(chain: CompiledChain, target: torch.Tensor,
            q0: torch.Tensor = None, link=-1, weights=None,
            q_rest: torch.Tensor = None, w_rest: float = 0.1,
            iters: int = 100, damping: float = 0.05, step: float = 1.0,
            tol: float = 1e-4, check_every: int = 10):
    """Batched weighted task-space IK. Returns (q (B, dof), info).

    chain    compiled chain.
    target   (B, 4, 4) desired pose of `link`, or (4, 4) for a single target.
    q0       (B, dof) seed, or (dof,); random within the limits if omitted.
             Its dtype and device are the working dtype and device.
    link     link index (negatives count from the end) or link name.
    weights  per-component scale on the error [dx dy dz, drx dry drz], shape
             (6,) or (B, 6). None means all ones. Zero frees a direction.
    q_rest   posture the nullspace term pulls toward, (dof,) or (B, dof).
             Defaults to the midpoint of the joint limits.
    w_rest   gain of that pull per iteration; 0 turns the term off.
    iters    maximum iterations.
    damping  Levenberg-Marquardt lambda. Larger is slower but safer near
             singularities, and loosens the task/posture priority.
    step     scale on each joint-space step, 1.0 for the plain DLS step.
    tol      stop once every weighted error norm in the batch is below this.
    check_every  how often to test `tol`. The test is a host sync, which stalls
             a GPU, so it is deliberately not done every iteration.

    info holds the errors of the RETURNED q (one extra FK past the loop, so the
    numbers describe what you get back rather than the previous iterate):
    final_error the weighted 6D error norm, task_error the unweighted one,
    position_error, rotation_error, rest_error = ||q - q_rest||, and iters.
    """
    if target.dim() == 2:
        target = target.unsqueeze(0)
    B = target.shape[0]
    li = _link_index(chain, link)

    if q0 is not None:
        if q0.dim() == 1:
            q0 = q0.unsqueeze(0).expand(B, -1)
        device, dtype = q0.device, q0.dtype
    else:
        device, dtype = target.device, target.dtype
    lo = chain.lower.to(device=device, dtype=dtype)
    hi = chain.upper.to(device=device, dtype=dtype)
    target = target.to(device=device, dtype=dtype)

    if q0 is None:
        span = torch.where(torch.isfinite(hi - lo), hi - lo,
                           torch.full_like(lo, 2.0 * math.pi))
        base = torch.where(torch.isfinite(lo), lo, -0.5 * span)
        q = base + span * torch.rand(B, chain.dof, dtype=dtype, device=device)
    else:
        q = q0.clone()

    w = _prep_weights(weights, B, 6, device, dtype)
    use_rest = bool(w_rest) and chain.dof > 0
    if use_rest:
        if q_rest is None:
            rest = _default_rest(lo, hi).unsqueeze(0).expand(B, -1)
        else:
            rest = q_rest.to(device=device, dtype=dtype)
            if rest.dim() == 1:
                rest = rest.unsqueeze(0).expand(B, -1)

    tgt_p = target[:, :3, 3]
    tgt_R = target[:, :3, :3]

    def _err(rp):
        wR, wp = rp
        p_err = tgt_p - wp[li]
        w_err = T.so3_log(tgt_R @ wR[li].transpose(-1, -2))
        return torch.cat([p_err, w_err], dim=-1)

    used = 0
    for it in range(iters):
        rp = fk_rp(chain, q)
        e = _err(rp)
        J = jacobian_rp(chain, q, li, rp=rp)
        dq_null = w_rest * (rest - q) if use_rest else None
        dq = weighted_dls_step(J, e, w, damping, dq_null)
        q = torch.clamp(q + step * dq, lo, hi)
        used = it + 1
        if (it + 1) % check_every == 0:
            err = e if w is None else e * w
            if bool((err.norm(dim=-1) < tol).all()):
                break

    e = _err(fk_rp(chain, q))
    werr = e if w is None else e * w
    info = {
        "iters": used,
        "final_error": werr.norm(dim=-1).detach(),
        "task_error": e.norm(dim=-1).detach(),
        "position_error": e[:, :3].norm(dim=-1).detach(),
        "rotation_error": e[:, 3:].norm(dim=-1).detach(),
    }
    if use_rest:
        info["rest_error"] = (q - rest).norm(dim=-1).detach()
    return q, info
