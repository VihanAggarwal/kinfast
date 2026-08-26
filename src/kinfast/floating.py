# src/kinfast/floating.py
"""Floating-base kinematics: a robot whose root link is free in space.

A fixed-base chain assumes the root link is bolted to the world. Mobile
manipulators, humanoids, drones with arms and free-flying inspection robots are
not: their root link has its own six degrees of freedom. This module wraps a
CompiledChain and adds those six, so the same FK / Jacobian / IK code path works
for a robot that can translate and rotate as a whole.

Configuration layout
--------------------
    q_full = [ x, y, z,  rx, ry, rz,  joint_0 ... joint_{dof-1} ]

The first three numbers are the root link's position in world coordinates. The
next three are a rotation vector (the axis times the angle, sometimes called
exponential coordinates): its direction is the rotation axis and its length is
the rotation angle in radians. A rotation vector is used rather than a
quaternion because it has exactly three numbers with no unit-norm constraint, so
the configuration vector stays a plain unconstrained vector that gradient
descent, IK and autograd can all step through without projecting back onto a
manifold. The remaining `dof` numbers are the ordinary joint values, in the same
order the compiled chain uses.

Jacobian convention
-------------------
`jacobian` returns d(pose)/d(q_full): the linear rows are the derivative of the
link's world position with respect to each configuration number, and the angular
rows are the world-frame angular velocity produced by a unit rate of change of
each configuration number. That is the convention a finite difference measures,
so every column can be checked numerically.

Because the base orientation is parameterized by a rotation vector rather than
by an angular velocity, the three base-rotation columns are not the identity:
the world angular velocity produced by moving the rotation vector at rate rdot
is `left_jacobian(r) @ rdot`. That matrix is only the identity when the base is
unrotated, which is exactly the case where a naive implementation looks correct
and is silently wrong everywhere else.

Everything here is batched over a leading dimension, follows the dtype and
device of the configuration the caller passes (the compiled chain's constants
are cast on the fly, as in the rest of the library), and is differentiable.
"""
import math

import torch

from kinfast import transforms as T
from kinfast.compile import CompiledChain
from kinfast.fk import fk_rp as _fixed_fk_rp
from kinfast.jacobian import _resolve_link
from kinfast.jacobian import jacobian_rp as _fixed_jacobian_rp

__all__ = [
    "FloatingRobot",
    "rotvec_to_matrix",
    "matrix_to_rotvec",
    "left_jacobian",
    "wrap_rotvec",
    "skew",
]


def _series_threshold(dtype) -> float:
    """Angle below which the Taylor series beats the closed form.

    The closed forms divide by powers of theta and subtract nearly equal
    numbers ((1 - cos theta) for tiny theta is catastrophic cancellation), so
    they lose relative precision as theta shrinks. The truncated series has the
    opposite behaviour. They cross near eps**(1/4), which is about 1.2e-4 in
    float64 and 1.9e-2 in float32.
    """
    return float(torch.finfo(dtype).eps) ** 0.25


def skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) vector -> (..., 3, 3) cross-product matrix, so skew(a) @ b == a x b."""
    zero = torch.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotvec_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotation vector -> (..., 3, 3) rotation matrix.

    Rodrigues with the unnormalized vector: R = I + a K + b K^2 where K is the
    cross-product matrix of r, a = sin(theta)/theta and b = (1-cos theta)/theta^2.
    Writing it this way (instead of normalizing r to an axis and an angle) keeps
    the gradient finite at r = 0, which matters because a floating base sitting
    at zero rotation is the most common configuration there is.
    """
    theta2 = (r * r).sum(dim=-1)
    thr = _series_threshold(r.dtype)
    small = theta2 < thr * thr
    # substitute a harmless value inside the branch we are not taking, so the
    # unused branch cannot produce a nan that torch.where would multiply into
    # the gradient
    theta2_safe = torch.where(small, torch.ones_like(theta2), theta2)
    theta = torch.sqrt(theta2_safe)
    a = torch.where(small,
                    1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
                    torch.sin(theta) / theta)
    b = torch.where(small,
                    0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
                    (1.0 - torch.cos(theta)) / theta2_safe)
    K = skew(r)
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(K.shape)
    return eye + a[..., None, None] * K + b[..., None, None] * (K @ K)


def matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 3) rotation vector. Inverse of
    rotvec_to_matrix on the principal branch (angle in [0, pi]).

    The angle comes from atan2(sin, cos) rather than acos(cos). Both are exact
    on paper, but acos has an infinite derivative at its endpoints, so an IK
    loop that differentiates through a converged orientation error (where the
    residual rotation is the identity) gets nan gradients from it. atan2 stays
    well behaved at both ends. Near a half turn the axis itself is genuinely
    ill conditioned, because a rotation by pi leaves the sign of its axis
    undetermined; that is a property of the parameterization, not of this code.
    """
    w = 0.5 * torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)                                   # sin(theta) * axis
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    s2 = (w * w).sum(dim=-1)
    thr = _series_threshold(R.dtype)
    small = (s2 < thr * thr) & (cos > 0)
    s2_safe = torch.where(small, torch.ones_like(s2), s2)
    s = torch.sqrt(s2_safe)
    theta = torch.atan2(s, cos)
    # theta / sin(theta), as a series in sin(theta) where the ratio is 0/0
    scale = torch.where(small,
                        1.0 + s2 / 6.0 + 3.0 * s2 * s2 / 40.0,
                        theta / s)
    return w * scale.unsqueeze(-1)


def left_jacobian(r: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotation vector -> (..., 3, 3) left Jacobian of SO(3).

    Defined by exp([r + d]) = exp([J_l(r) d]) exp([r]) to first order in d.
    In words: if the rotation vector moves at rate rdot, the body's world-frame
    angular velocity is J_l(r) @ rdot. This is what turns a derivative with
    respect to the rotation-vector parameters into an angular velocity, so it is
    exactly the base-rotation block of the floating Jacobian.
    """
    theta2 = (r * r).sum(dim=-1)
    thr = _series_threshold(r.dtype)
    small = theta2 < thr * thr
    theta2_safe = torch.where(small, torch.ones_like(theta2), theta2)
    theta = torch.sqrt(theta2_safe)
    b = torch.where(small,
                    0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
                    (1.0 - torch.cos(theta)) / theta2_safe)
    c = torch.where(small,
                    1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
                    (theta - torch.sin(theta)) / (theta2_safe * theta))
    K = skew(r)
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(K.shape)
    return eye + b[..., None, None] * K + c[..., None, None] * (K @ K)


def wrap_rotvec(r: torch.Tensor) -> torch.Tensor:
    """Rewrite a rotation vector as the shortest one describing the same
    rotation (angle folded into [0, pi]).

    The rotation itself is unchanged, so forward kinematics does not move. The
    point is conditioning: the left Jacobian becomes singular as the angle
    approaches 2*pi, and an IK loop that keeps adding to the rotation vector
    will drift there. Folding after every step keeps the parameterization in the
    well conditioned region.
    """
    theta2 = (r * r).sum(dim=-1, keepdim=True)
    big = theta2 > math.pi * math.pi
    # substituting 1 inside the branch we are not taking keeps the norm's
    # derivative away from r = 0, where d|r|/dr is 0/0; torch.where would
    # otherwise multiply that nan into the gradient of the branch we do take
    theta = torch.sqrt(torch.where(big, theta2, torch.ones_like(theta2)))
    folded = torch.remainder(theta, 2.0 * math.pi)
    folded = torch.where(folded > math.pi, folded - 2.0 * math.pi, folded)
    return torch.where(big, r * (folded / theta), r)


class FloatingRobot:
    """A compiled chain plus six free degrees of freedom at the root link.

    Construct it from a CompiledChain (whatever compile_robot or Robot.chain
    gives you); the chain is used read-only, so one chain can back several
    floating wrappers.

        fr = FloatingRobot(chain)
        q  = torch.zeros(4, fr.dof)          # 6 base + chain.dof joints
        poses = fr.fk(q)                      # (4, n_links, 4, 4)
        J = fr.jacobian(q, "tool")            # (4, 6, 6 + chain.dof)
        q_sol, info = fr.ik(targets, "tool")  # the base is allowed to move
    """

    def __init__(self, chain: CompiledChain):
        self.chain = chain

    # ------------------------------------------------------------------ shape

    @property
    def joint_dof(self) -> int:
        """Number of actuated joints (the fixed-base dof)."""
        return self.chain.dof

    @property
    def dof(self) -> int:
        """Total configuration size: six base freedoms plus the joints."""
        return 6 + self.chain.dof

    @property
    def n_links(self) -> int:
        return self.chain.n_links

    @property
    def link_names(self) -> list:
        return self.chain.link_names

    def __repr__(self):
        return (f"FloatingRobot(links={self.chain.n_links}, "
                f"joints={self.chain.dof}, dof={self.dof})")

    def to(self, device=None, dtype=None):
        """Move or cast the underlying chain in place and return self."""
        self.chain.to(device=device, dtype=dtype)
        return self

    # ------------------------------------------------------------- resolution

    def link_id(self, link) -> int:
        """Accept a link name or an index (negatives count from the end) and
        return a plain non-negative index into the chain."""
        if isinstance(link, str):
            try:
                return self.chain.link_index[link]
            except KeyError:
                raise KeyError(
                    f"unknown link {link!r}; known links: "
                    f"{list(self.chain.link_index)}")
        return _resolve_link(self.chain, link)

    def split(self, q_full: torch.Tensor):
        """(B, 6+dof) -> (translation (B,3), rotation vector (B,3), joints (B,dof))."""
        if q_full.dim() != 2:
            raise ValueError(
                f"q_full must be (B, {self.dof}); got shape {tuple(q_full.shape)}")
        if q_full.shape[1] != self.dof:
            raise ValueError(
                f"q_full must have {self.dof} columns "
                f"(3 position + 3 rotation vector + {self.chain.dof} joints); "
                f"got {q_full.shape[1]}")
        return q_full[:, 0:3], q_full[:, 3:6], q_full[:, 6:]

    def join(self, t: torch.Tensor, r: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Inverse of split: assemble a full configuration from its parts."""
        return torch.cat([t, r, q], dim=-1)

    # ---------------------------------------------------------------- forward

    def base_transform(self, q_full: torch.Tensor) -> torch.Tensor:
        """(B, 6+dof) -> (B, 4, 4) world pose of the root link."""
        t, r, _ = self.split(q_full)
        return T.make_transform(rotvec_to_matrix(r), t)

    def _frames(self, q_full: torch.Tensor):
        """Everything the Jacobian needs, computed once.

        Returns (base rotation, base translation, base rotation vector,
        base-frame link rotations, base-frame link positions, world link
        rotations, world link positions).
        """
        t, r, qj = self.split(q_full)
        Rb = rotvec_to_matrix(r)
        bR, bp = _fixed_fk_rp(self.chain, qj)
        wR = [Rb @ R for R in bR]
        wp = [t + (Rb @ p.unsqueeze(-1)).squeeze(-1) for p in bp]
        return Rb, t, r, bR, bp, wR, wp

    def fk_rp(self, q_full: torch.Tensor):
        """World rotations and positions as per-link lists (wR, wp), the same
        shape of result as kinfast.fk.fk_rp. This is the hot path: the Jacobian
        and IK use it and never assemble 4x4 matrices."""
        _, _, _, _, _, wR, wp = self._frames(q_full)
        return wR, wp

    def fk(self, q_full: torch.Tensor) -> torch.Tensor:
        """(B, 6+dof) -> (B, n_links, 4, 4) world transforms.

        Identical to composing the base transform with the fixed-base FK, which
        is what it does; the composition happens on rotation/translation pairs
        rather than 4x4s because that is cheaper at large batch.
        """
        wR, wp = self.fk_rp(q_full)
        B = q_full.shape[0]
        M = torch.zeros(B, self.chain.n_links, 4, 4,
                        dtype=q_full.dtype, device=q_full.device)
        M[:, :, :3, :3] = torch.stack(wR, dim=1)
        M[:, :, :3, 3] = torch.stack(wp, dim=1)
        M[:, :, 3, 3] = 1.0
        return M

    def link_pose(self, q_full: torch.Tensor, link) -> torch.Tensor:
        """World pose (B, 4, 4) of one link, without building the whole stack."""
        i = self.link_id(link)
        wR, wp = self.fk_rp(q_full)
        return T.make_transform(wR[i], wp[i])

    # --------------------------------------------------------------- Jacobian

    def jacobian(self, q_full: torch.Tensor, link, frames=None) -> torch.Tensor:
        """(B, 6+dof) -> (B, 6, 6+dof) derivative of the link pose.

        Rows 0:3 are d(world position)/d(q_full); rows 3:6 are the world-frame
        angular velocity per unit rate of each configuration number.

        Column blocks:
          base position   Jv = I,                       Jw = 0
          base rotation   Jv = -skew(p_link - p_base) L, Jw = L,  L = left_jacobian(r)
          joints          the fixed-base Jacobian, both blocks rotated into the
                          world by the base rotation

        Pass `frames` (from _frames at the same configuration) to reuse an FK
        pass; IK does that every iteration.
        """
        i = self.link_id(link)
        if frames is None:
            frames = self._frames(q_full)
        Rb, t, r, bR, bp, wR, wp = frames
        B = q_full.shape[0]
        dtype, device = q_full.dtype, q_full.device

        eye = torch.eye(3, dtype=dtype, device=device).expand(B, 3, 3)
        zero = torch.zeros(B, 3, 3, dtype=dtype, device=device)

        L = left_jacobian(r)
        lever = wp[i] - t                       # base origin to link, in world
        Jv_rot = -skew(lever) @ L

        # the fixed-base Jacobian is expressed in the root link's frame, which
        # is the base frame here; rotating both 3-row blocks moves it to world
        J_fixed = _fixed_jacobian_rp(self.chain, q_full[:, 6:], i, rp=(bR, bp))
        Jv_joint = Rb @ J_fixed[:, :3, :]
        Jw_joint = Rb @ J_fixed[:, 3:, :]

        Jv = torch.cat([eye, Jv_rot, Jv_joint], dim=-1)
        Jw = torch.cat([zero, L, Jw_joint], dim=-1)
        return torch.cat([Jv, Jw], dim=-2)

    # --------------------------------------------------------------- sampling

    def _joint_bounds(self, dtype, device):
        """Joint limits cast to the working dtype, with infinite limits (from
        continuous joints) replaced by +-pi for sampling purposes."""
        lo = self.chain.lower.to(device=device, dtype=dtype)
        hi = self.chain.upper.to(device=device, dtype=dtype)
        pi = torch.full_like(lo, math.pi)
        lo_s = torch.where(torch.isfinite(lo), lo, -pi)
        hi_s = torch.where(torch.isfinite(hi), hi, pi)
        return lo, hi, lo_s, hi_s

    def seed(self, target: torch.Tensor, base_noise: float = 0.5,
             rot_noise: float = 0.5, generator=None) -> torch.Tensor:
        """A starting configuration for IK, one per target pose.

        The heuristic that matters is the base: a floating base can be anywhere,
        so starting it at the origin makes every distant target look
        unreachable. We drop the base near the target position and give it
        roughly the target's orientation, then jitter both and draw the joints
        uniformly from their limits. The remaining error is on the order of the
        arm's own length, which damped least squares closes quickly.
        """
        dtype, device = target.dtype, target.device
        B = target.shape[0]
        lo, hi, lo_s, hi_s = self._joint_bounds(dtype, device)

        def rand(*shape):
            return torch.rand(*shape, dtype=dtype, device=device, generator=generator)

        t = target[:, :3, 3] + base_noise * (2.0 * rand(B, 3) - 1.0)
        r = matrix_to_rotvec(target[:, :3, :3]) + rot_noise * (2.0 * rand(B, 3) - 1.0)
        qj = lo_s + (hi_s - lo_s) * rand(B, self.chain.dof)
        return torch.cat([t, wrap_rotvec(r), qj], dim=-1)

    # --------------------------------------------------------------------- IK

    def _bounds(self, dtype, device, base_bounds):
        """Clamping bounds for the full configuration. The base is unbounded by
        default (a free base really is free); pass base_bounds as a pair of
        length-6 sequences to keep it inside a box."""
        lo, hi, _, _ = self._joint_bounds(dtype, device)
        if base_bounds is None:
            blo = torch.full((6,), -math.inf, dtype=dtype, device=device)
            bhi = torch.full((6,), math.inf, dtype=dtype, device=device)
        else:
            blo = torch.as_tensor(base_bounds[0], dtype=dtype, device=device)
            bhi = torch.as_tensor(base_bounds[1], dtype=dtype, device=device)
            if blo.shape != (6,) or bhi.shape != (6,):
                raise ValueError(
                    "base_bounds must be a pair of length-6 sequences "
                    "(x y z rx ry rz)")
        return torch.cat([blo, lo]), torch.cat([bhi, hi])

    def _solve(self, target, q0, link_index, iters, damping, step, pos_only,
               tol, check_every, base_bounds):
        """Damped least squares from one seed per target. Returns (q, error)."""
        dtype, device = q0.dtype, q0.device
        target = target.to(device=device, dtype=dtype)
        lo, hi = self._bounds(dtype, device, base_bounds)
        m = 3 if pos_only else 6
        eye = torch.eye(m, dtype=dtype, device=device)
        lam2 = damping * damping
        tgt_p = target[:, :3, 3]
        tgt_R = target[:, :3, :3]

        q = q0.clone()
        err = torch.full((q.shape[0],), float("inf"), dtype=dtype, device=device)
        for it in range(iters):
            frames = self._frames(q)
            wR, wp = frames[5], frames[6]
            e = tgt_p - wp[link_index]
            if not pos_only:
                w_err = matrix_to_rotvec(tgt_R @ wR[link_index].transpose(-1, -2))
                e = torch.cat([e, w_err], dim=-1)
            J = self.jacobian(q, link_index, frames=frames)
            if pos_only:
                J = J[:, :3, :]
            JT = J.transpose(-1, -2)
            H = J @ JT + lam2 * eye
            dq = (JT @ torch.linalg.solve(H, e.unsqueeze(-1))).squeeze(-1)
            q = torch.clamp(q + step * dq, lo, hi)
            # keep the rotation vector short so the left Jacobian stays well
            # conditioned; this does not change the rotation it encodes
            q = torch.cat([q[:, :3], wrap_rotvec(q[:, 3:6]), q[:, 6:]], dim=-1)
            err = e.norm(dim=-1)
            if (it + 1) % check_every == 0 and bool((err < tol).all()):
                break
        return q, err

    def ik(self, target: torch.Tensor, link=-1, q0: torch.Tensor = None,
           iters: int = 100, damping: float = 0.05, step: float = 1.0,
           pos_only: bool = False, tol: float = 1e-6, restarts: int = 1,
           check_every: int = 10, base_bounds=None, base_noise: float = 0.5,
           rot_noise: float = 0.5, generator=None):
        """Batched IK that is allowed to move the base. target (B,4,4) ->
        (q_full (B, 6+dof), info).

        Same damped-least-squares loop as the fixed-base solver, run over the
        full configuration. Because the base contributes an identity block to
        the linear rows and an invertible one to the angular rows, the stacked
        Jacobian has full row rank at every configuration, so the solve is far
        better behaved than fixed-base IK and targets outside the arm's own
        reach are ordinary rather than impossible.

        q0 seeds the solve when given (and restarts must be 1); otherwise seeds
        come from `seed`. restarts>1 runs that many seeds per target in one
        batch and keeps the best, exactly as kinfast.ik does. Pass base_bounds
        as (lo6, hi6) to keep the base inside a box, for instance a mobile base
        pinned to the ground plane.
        """
        if target.dim() == 2:
            target = target.unsqueeze(0)
        B = target.shape[0]
        link_index = self.link_id(link)
        if restarts < 1:
            raise ValueError("restarts must be at least 1")
        if q0 is not None and restarts > 1:
            raise ValueError(
                "q0 seeds a single solve; use restarts>1 or q0, not both")

        if restarts == 1:
            if q0 is None:
                q0 = self.seed(target, base_noise, rot_noise, generator)
            elif q0.shape != (B, self.dof):
                raise ValueError(
                    f"q0 must be ({B}, {self.dof}); got {tuple(q0.shape)}")
            q, err = self._solve(target, q0, link_index, iters, damping, step,
                                 pos_only, tol, check_every, base_bounds)
            return q, {"iters": iters, "final_error": err.detach()}

        tgt = target.repeat_interleave(restarts, dim=0)
        seeds = self.seed(tgt, base_noise, rot_noise, generator)
        q_all, err_all = self._solve(tgt, seeds, link_index, iters, damping,
                                     step, pos_only, tol, check_every,
                                     base_bounds)
        err = err_all.view(B, restarts)
        best = err.argmin(dim=1)
        q_all = q_all.view(B, restarts, self.dof)
        q_best = q_all.gather(
            1, best.view(B, 1, 1).expand(B, 1, self.dof)).squeeze(1)
        best_err = err.gather(1, best.view(B, 1)).squeeze(1)
        return q_best, {"iters": iters, "final_error": best_err.detach(),
                        "restarts": restarts}
