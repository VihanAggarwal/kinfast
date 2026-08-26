# src/kinfast/ik_ccd.py
"""Cyclic coordinate descent (CCD) position inverse kinematics, batched.

CCD is the cheap, Jacobian-free cousin of damped least squares. One sweep walks
the joints from the end effector back to the root and gives each joint the
single value that best pulls the end effector onto the target, holding every
other joint fixed. Repeat the sweep until the position error is small enough.

Why it is worth having next to `kinfast.ik`:

  * no linear solve, so no damping to tune and nothing to go singular. Near a
    kinematic singularity DLS either stalls or needs a large lambda, while CCD
    keeps making progress because each joint is solved exactly.
  * every joint step is a closed form, so a sweep costs one forward kinematics
    pass plus a handful of dot products per joint.
  * limits are handled by construction: the exact angle is clamped into the
    joint range and the resulting end effector motion uses the clamped value,
    so a returned configuration is always inside the limits.

The closed form. Let a be the joint axis in world coordinates (a unit vector),
p the world position of the joint, e the current end effector position and t
the target. Rotating this joint by theta moves the end effector to
p + R(a, theta) u with u = e - p, because everything distal to the joint is
carried rigidly. Writing v = t - p and using Rodrigues,

    v . R(a, theta) u = A cos(theta) + B sin(theta) + C
    A = u . v - (a . u)(a . v),  B = (a x u) . v,  C = (a . u)(a . v)

and since |R u| does not depend on theta, minimizing |R u - v| is the same as
maximizing that expression. The maximum is at theta = atan2(B, A), which is the
angle this module applies. A prismatic joint gets the matching one liner:
sliding by s moves the end effector by s * a, so s = a . (t - e).

The sweep only needs one forward kinematics call. Joints are visited from the
end effector towards the root, and rotating a joint never moves anything
proximal to it, so the world frames of the joints still to be visited stay
valid. The end effector position is carried forward exactly by applying the
same rotation that was applied to the joint.

Everything is batched over a leading B dimension, follows the dtype and device
of the caller's seed configuration, and is autograd traceable end to end, so
you can backpropagate through a solve.
"""
import math

import torch

from kinfast.compile import CompiledChain
from kinfast.fk import fk_rp, _cache
from kinfast.jacobian import _resolve_link


def alignment_coeffs(axis: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """Coefficients of the per joint objective, v . R(axis, theta) u.

    That dot product equals A cos(theta) + B sin(theta) + C with C independent
    of theta, so (A, B) is everything the step needs. axis must be a unit
    vector; axis, u and v are (..., 3) and the returned A and B are (...,).
    """
    au = (axis * u).sum(dim=-1)
    av = (axis * v).sum(dim=-1)
    A = (u * v).sum(dim=-1) - au * av
    B = (torch.cross(axis.expand_as(u), u, dim=-1) * v).sum(dim=-1)
    return A, B


def best_alignment_angle(axis: torch.Tensor, u: torch.Tensor,
                         v: torch.Tensor) -> torch.Tensor:
    """Angle about `axis` that rotates `u` as close to `v` as possible.

    axis: (..., 3) unit vectors, u and v: (..., 3). Returns (...,) angles in
    radians in (-pi, pi], the exact maximizer of v . R(axis, theta) u and
    therefore the exact minimizer of |R u - v|. This is the whole of CCD's per
    joint step with the limits left out, exposed on its own because it is easy
    to check against a brute force scan over theta.

    Degenerate inputs (u parallel to the axis, or a zero length u or v) give
    A = B = 0 and therefore an angle of zero, which is the right answer: no
    rotation about this axis can improve the fit.
    """
    A, B = alignment_coeffs(axis, u, v)
    return torch.atan2(B, A)


_TWO_PI = 2.0 * math.pi


def _feasible_delta(A, B, theta, q, lo, hi):
    """The best rotation the joint can actually perform, given its limits.

    The unconstrained optimum theta may sit outside [lo - q, hi - q]. Clamping
    it there is the obvious move but it is not optimal, because the objective
    is 2 pi periodic: theta - 2 pi or theta + 2 pi is the same physical
    rotation and may well be reachable when theta itself is not. That case is
    common on a joint with a range near a full turn, where naive clamping
    parks the joint on its stop and the solve stalls.

    So three candidates are considered, theta shifted by -2 pi, 0 and +2 pi and
    each clamped into range, and the one with the largest objective wins. Those
    three always bracket the true maximum over the feasible interval: either a
    shift of theta lands inside it, or the maximum is at an interval endpoint
    and a clamped shift lands exactly there. When theta itself is feasible it
    is used directly, which also keeps an unlimited (continuous) joint from
    drifting by a full turn per sweep on a tie.
    """
    raw = q + theta
    feasible = (raw >= lo) & (raw <= hi)
    cands = torch.stack([torch.clamp(raw + k * _TWO_PI, lo, hi)
                         for k in (0.0, -1.0, 1.0)], dim=0) - q     # (3, B)
    score = A * torch.cos(cands) + B * torch.sin(cands)             # (3, B)
    pick = score.argmax(dim=0, keepdim=True)                        # (1, B)
    return torch.where(feasible, theta, cands.gather(0, pick).squeeze(0))


def _rotate_about(axis, u, angle):
    """Rodrigues rotation of u about the unit `axis` by `angle`, batched."""
    c = torch.cos(angle).unsqueeze(-1)
    s = torch.sin(angle).unsqueeze(-1)
    au = (axis * u).sum(dim=-1, keepdim=True)
    return u * c + torch.cross(axis, u, dim=-1) * s + axis * au * (1 - c)


def _joint_path(chain: CompiledChain, link_index: int):
    """Movable joints between `link_index` and the root, end effector first.

    Returns a list of (link, joint_type, q_index). Fixed joints are skipped.
    The walk reads chain.parent, which is a host side sync per hop, so it is
    done once per solve rather than once per sweep.
    """
    path = []
    i = link_index
    while i >= 0:
        jt = int(chain.joint_type[i])
        qi = int(chain.q_index[i])
        if jt != 0 and qi >= 0:
            path.append((i, jt, qi))
        i = int(chain.parent[i])
    return path


def _target_positions(target: torch.Tensor, device, dtype) -> torch.Tensor:
    """Accept either (B, 3) points or (B, 4, 4) poses and return (B, 3).

    CCD as implemented here solves position only, so the rotation block of a
    pose target is ignored. Taking the pose form as well means a target built
    for `kinfast.ik` can be handed straight to this solver.
    """
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"target must be a torch.Tensor of shape (B, 3) or (B, 4, 4), got "
            f"{type(target).__name__}")
    t = target.to(device=device, dtype=dtype)
    if t.dim() == 3 and t.shape[-2:] == (4, 4):
        return t[:, :3, 3]
    if t.dim() == 2 and t.shape[-1] == 3:
        return t
    raise ValueError(
        f"target must have shape (B, 3) or (B, 4, 4), got {tuple(target.shape)}")


def _sweep_from_seed(chain, tgt_p, q0, link_index, sweeps, tol, step,
                     check_every):
    """Core CCD loop from one seed. Returns (q, final position error)."""
    device, dtype = q0.device, q0.dtype
    lo = chain.lower.to(device=device, dtype=dtype)
    hi = chain.upper.to(device=device, dtype=dtype)
    # A seed outside the limits would otherwise be reported as a solution that
    # violates them, so it is pulled in before the first sweep.
    q = torch.clamp(q0, lo, hi)
    axes = _cache(chain, device, dtype)["axes"]
    path = _joint_path(chain, link_index)

    # Columns are kept as a list so each joint update is a pure functional
    # rebind rather than an in place write, which keeps autograd happy.
    cols = list(q.unbind(dim=1))
    lo_c = list(lo.unbind(dim=0))
    hi_c = list(hi.unbind(dim=0))

    err = None
    for s in range(sweeps):
        wR, wp = fk_rp(chain, torch.stack(cols, dim=1) if cols
                       else q)
        e = wp[link_index]
        for (i, jt, qi) in path:
            a = wR[i] @ axes[i]                    # (B, 3) world joint axis
            if jt == 1:                            # revolute
                p = wp[i]
                u = e - p
                A, Bc = alignment_coeffs(a, u, tgt_p - p)
                theta = torch.atan2(Bc, A)
                delta = _feasible_delta(A, Bc, theta, cols[qi],
                                        lo_c[qi], hi_c[qi])
                # scaling towards zero cannot leave the feasible interval,
                # since a delta of zero is always feasible
                new = torch.clamp(cols[qi] + step * delta, lo_c[qi], hi_c[qi])
                applied = new - cols[qi]
                cols[qi] = new
                # the clamped angle is what the arm actually moved by, so the
                # end effector is carried forward with exactly that rotation
                e = p + _rotate_about(a, u, applied)
            else:                                  # prismatic
                slide = step * ((tgt_p - e) * a).sum(dim=-1)
                new = torch.clamp(cols[qi] + slide, lo_c[qi], hi_c[qi])
                applied = new - cols[qi]
                cols[qi] = new
                e = e + a * applied.unsqueeze(-1)
        err = (tgt_p - e).norm(dim=-1)
        # the all() is a host sync, so it is spent every check_every sweeps
        # rather than every sweep, matching kinfast.ik
        if check_every > 0 and (s + 1) % check_every == 0 \
                and bool((err < tol).all()):
            break

    q_out = torch.stack(cols, dim=1) if cols else q
    if err is None:                                # sweeps <= 0
        _, wp = fk_rp(chain, q_out)
        err = (tgt_p - wp[link_index]).norm(dim=-1)
    return q_out, err


def ik_ccd(chain: CompiledChain, target: torch.Tensor, q0: torch.Tensor = None,
           link_index: int = None, sweeps: int = 100, tol: float = 1e-4,
           step: float = 1.0, restarts: int = 1, check_every: int = 10):
    """Batched position IK by cyclic coordinate descent.

    chain:       compiled robot.
    target:      (B, 3) points or (B, 4, 4) poses; only the position is used.
    q0:          (B, dof) seed. If omitted, a random seed is drawn uniformly
                 from the joint limits. The seed fixes the working dtype and
                 device, exactly as in `kinfast.ik`.
    link_index:  which link is the end effector. Negative indices count from
                 the end; None means the last link.
    sweeps:      maximum end effector to root passes.
    tol:         stop early once every batch element is within this distance.
    step:        scale on each joint's exact angle, in (0, 1]. Leave it at 1
                 for the textbook method; lowering it trades speed for a
                 smoother, less jumpy path.
    restarts:    number of random seeds per target, solved together in one
                 batch and reduced to the best. q0 is used only when
                 restarts == 1.
    check_every: how often the early stop test is evaluated.

    Returns (q (B, dof), info) where info holds the sweep budget and the final
    per target position error. The returned q always lies inside the joint
    limits.
    """
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"target must be a torch.Tensor of shape (B, 3) or (B, 4, 4), got "
            f"{type(target).__name__}")
    if q0 is not None and not isinstance(q0, torch.Tensor):
        raise TypeError(
            f"q0 must be a torch.Tensor of shape (B, {chain.dof}), got "
            f"{type(q0).__name__}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    link_index = _resolve_link(chain, -1 if link_index is None else link_index)

    if q0 is not None and restarts <= 1:
        device, dtype = q0.device, q0.dtype
    else:
        device, dtype = target.device, target.dtype
        if not dtype.is_floating_point:
            raise TypeError(
                f"target must be a floating point tensor to seed a solve, got "
                f"dtype {dtype}")

    tgt_p = _target_positions(target, device, dtype)
    B = tgt_p.shape[0]
    lo = chain.lower.to(device=device, dtype=dtype)
    hi = chain.upper.to(device=device, dtype=dtype)

    def _sample(n):
        # An unbounded (continuous) joint has infinite limits; sampling from
        # them would give inf, so those joints are seeded from [-pi, pi].
        span = torch.where(torch.isfinite(hi - lo), hi - lo,
                           torch.full_like(lo, 2 * math.pi))
        base = torch.where(torch.isfinite(lo), lo, torch.full_like(lo, -math.pi))
        return base + span * torch.rand(n, chain.dof, dtype=dtype, device=device)

    if restarts <= 1:
        if q0 is not None and (q0.dim() != 2 or q0.shape[0] != B):
            # broadcasting would otherwise turn a mismatched seed into a silent
            # cross product of seeds and targets instead of an error
            raise ValueError(
                f"q0 must have shape ({B}, {chain.dof}) to match the "
                f"{B} targets, got {tuple(q0.shape)}")
        seed = q0.clone() if q0 is not None else _sample(B)
        q, err = _sweep_from_seed(chain, tgt_p, seed, link_index, sweeps, tol,
                                  step, check_every)
        return q, {"sweeps": sweeps, "final_error": err.detach()}

    tgt_k = tgt_p.repeat_interleave(restarts, dim=0)          # (B*K, 3)
    seeds = _sample(B * restarts)
    q_all, err_all = _sweep_from_seed(chain, tgt_k, seeds, link_index, sweeps,
                                      tol, step, check_every)
    err = err_all.view(B, restarts)
    best = err.argmin(dim=1)
    q_all = q_all.view(B, restarts, chain.dof)
    q_best = q_all.gather(1, best.view(B, 1, 1).expand(B, 1, chain.dof)).squeeze(1)
    best_err = err.gather(1, best.view(B, 1)).squeeze(1)
    return q_best, {"sweeps": sweeps, "final_error": best_err.detach(),
                    "restarts": restarts}
