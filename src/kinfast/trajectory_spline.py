# src/kinfast/trajectory_spline.py
"""Splines through waypoints, in joint space and in Cartesian space.

Two things live here, and they answer two different questions.

`CubicSpline` (and the `cubic_spline` one-shot wrapper) answers "move through
this list of joint waypoints smoothly". It is the classic C2 cubic spline: each
segment between two waypoints is a cubic, the curve passes through every
waypoint exactly, and position, velocity and acceleration are all continuous
across the joins. The ends are clamped, meaning the boundary velocity is zero by
default, so the robot starts and stops at rest. Clamping is what you want for a
motion command; the alternative "natural" boundary condition zeroes the
acceleration instead and leaves the arm still moving at the last waypoint.

Solving for the spline is a small tridiagonal system in the waypoint slopes,
which we assemble densely and hand to `torch.linalg.solve`. The number of
waypoints is tiny compared to the batch, so a dense solve costs nothing and
keeps the whole thing differentiable and batched: gradients flow back to the
waypoints, and a leading batch dimension of robots or of alternative waypoint
sets is solved in one call.

`cartesian_line` answers the other question, "drag the end effector along a
straight line in space". It interpolates the pose (linear in position, shortest
arc in orientation) and solves IK at every sample, warm starting each solve from
the previous answer. Warm starting is what keeps the joint path continuous: a
cold IK solve at each sample would happily return a different branch of the
solution set and the arm would flip between samples.

Everything follows the working dtype and device of the tensors you pass in, the
same rule as the rest of the library.
"""
import torch

from kinfast import transforms as T
from kinfast.compile import CompiledChain
from kinfast.ik import ik
from kinfast.jacobian import _resolve_link


def _as_tensor(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _clamped_slopes(y: torch.Tensor, tk: torch.Tensor,
                    v0: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
    """Slopes at the knots that make the piecewise cubic C2.

    y: (..., K, D) waypoint values, tk: (K,) strictly increasing knot times,
    v0/vf: (..., D) prescribed end velocities. Returns (..., K, D).

    Each segment is a cubic Hermite in (value, slope) at its two ends, so
    matching values and first derivatives is automatic and only the second
    derivative needs work. Equating the second derivative on either side of
    interior knot i gives one linear equation per interior knot:

        h_i v_{i-1} + 2 (h_{i-1} + h_i) v_i + h_{i-1} v_{i+1}
            = 3 (h_{i-1} d_i + h_i d_{i-1})

    with h_i = t_{i+1} - t_i the segment lengths and d_i the segment secant
    slopes. The two end rows just pin v_0 and v_{K-1} to the clamped values.
    """
    K = tk.shape[0]
    dtype, device = y.dtype, y.device
    h = tk[1:] - tk[:-1]                                    # (K-1,)
    d = (y[..., 1:, :] - y[..., :-1, :]) / h.unsqueeze(-1)  # (..., K-1, D)

    A = torch.zeros(K, K, dtype=dtype, device=device)
    A[0, 0] = 1.0
    A[K - 1, K - 1] = 1.0
    if K > 2:
        i = torch.arange(1, K - 1, device=device)
        hm, hp = h[:-1], h[1:]                              # h_{i-1}, h_i
        A[i, i - 1] = hp
        A[i, i] = 2.0 * (hm + hp)
        A[i, i + 1] = hm

    rhs = torch.zeros_like(y)
    rhs[..., 0, :] = v0
    rhs[..., K - 1, :] = vf
    if K > 2:
        hm, hp = h[:-1].unsqueeze(-1), h[1:].unsqueeze(-1)
        rhs[..., 1:-1, :] = 3.0 * (hm * d[..., 1:, :] + hp * d[..., :-1, :])

    return torch.linalg.solve(A, rhs)


class CubicSpline:
    """C2 cubic spline through joint-space waypoints, clamped at both ends.

    waypoints: (..., K, D) with K >= 2 waypoints of D joints and any number of
      leading batch dimensions.
    times: (K,) strictly increasing knot times, shared across the batch. If
      omitted the knots are spread uniformly over `duration` (which itself
      defaults to one second per segment).
    v0, vf: optional (..., D) end velocities; zero (a full stop) by default.

    The working dtype and device come from `waypoints`. The spline is
    differentiable in the waypoints, the end velocities and the knot times.
    """

    def __init__(self, waypoints: torch.Tensor, times=None, duration=None,
                 v0=None, vf=None):
        if not isinstance(waypoints, torch.Tensor):
            waypoints = torch.as_tensor(waypoints, dtype=torch.get_default_dtype())
        if waypoints.dim() < 2:
            raise ValueError("waypoints must be at least (K, dof)")
        K = waypoints.shape[-2]
        if K < 2:
            raise ValueError(f"need at least 2 waypoints, got {K}")
        dtype, device = waypoints.dtype, waypoints.device

        if times is None:
            total = float(K - 1) if duration is None else float(duration)
            if total <= 0.0:
                raise ValueError(f"duration must be positive, got {duration}")
            times = torch.linspace(0.0, total, K, dtype=dtype, device=device)
        else:
            times = _as_tensor(times, dtype, device)
            if times.dim() != 1 or times.shape[0] != K:
                raise ValueError(
                    f"times must be a 1-D tensor of {K} knot times, "
                    f"got shape {tuple(times.shape)}")
            if bool((times[1:] <= times[:-1]).any()):
                raise ValueError("times must be strictly increasing")
            if duration is not None:
                raise ValueError("pass either times or duration, not both")

        zero = torch.zeros_like(waypoints[..., 0, :])
        v0 = zero if v0 is None else _as_tensor(v0, dtype, device).expand_as(zero)
        vf = zero if vf is None else _as_tensor(vf, dtype, device).expand_as(zero)

        self.waypoints = waypoints
        self.times = times
        self.slopes = _clamped_slopes(waypoints, times, v0, vf)

    @property
    def dof(self):
        return self.waypoints.shape[-1]

    @property
    def n_waypoints(self):
        return self.waypoints.shape[-2]

    @property
    def batch_shape(self):
        return tuple(self.waypoints.shape[:-2])

    @property
    def duration(self):
        return float(self.times[-1] - self.times[0])

    def evaluate(self, t):
        """Evaluate at times t (M,) -> (q, qd, qdd), each (..., M, dof).

        Times outside the knot range are extrapolated with the end segment's
        cubic rather than raising, which keeps a sampler that overshoots the
        final knot by a rounding error well behaved.
        """
        t = _as_tensor(t, self.waypoints.dtype, self.waypoints.device)
        scalar = t.dim() == 0
        t = t.reshape(-1)
        tk, y, v = self.times, self.waypoints, self.slopes
        K = tk.shape[0]

        # segment index per sample, clamped so the ends extrapolate
        idx = torch.searchsorted(tk.detach().contiguous(), t.detach().contiguous(),
                                 right=True) - 1
        idx = idx.clamp(0, K - 2)

        t0 = tk.index_select(0, idx)
        h = tk.index_select(0, idx + 1) - t0
        s = ((t - t0) / h).unsqueeze(-1)                    # (M, 1)
        hh = h.unsqueeze(-1)                                # (M, 1)

        y0 = y.index_select(-2, idx)                        # (..., M, D)
        y1 = y.index_select(-2, idx + 1)
        v0 = v.index_select(-2, idx)
        v1 = v.index_select(-2, idx + 1)

        s2, s3 = s * s, s * s * s
        # Hermite basis and its first two derivatives with respect to s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        d00 = 6 * s2 - 6 * s
        d10 = 3 * s2 - 4 * s + 1
        d01 = -6 * s2 + 6 * s
        d11 = 3 * s2 - 2 * s
        e00 = 12 * s - 6
        e10 = 6 * s - 4
        e01 = -12 * s + 6
        e11 = 6 * s - 2

        q = h00 * y0 + h01 * y1 + hh * (h10 * v0 + h11 * v1)
        qd = (d00 * y0 + d01 * y1) / hh + d10 * v0 + d11 * v1
        qdd = (e00 * y0 + e01 * y1) / (hh * hh) + (e10 * v0 + e11 * v1) / hh

        if scalar:
            return q.squeeze(-2), qd.squeeze(-2), qdd.squeeze(-2)
        return q, qd, qdd

    def sample(self, n: int = 100):
        """Sample n points evenly over the knot range.

        Returns (t (n,), q, qd, qdd) with the motion tensors shaped
        (..., n, dof). The first and last samples land exactly on the first and
        last waypoints.
        """
        if n < 2:
            raise ValueError(f"need at least 2 samples, got {n}")
        u = torch.linspace(0.0, 1.0, n, dtype=self.waypoints.dtype,
                           device=self.waypoints.device)
        t = self.times[0] + (self.times[-1] - self.times[0]) * u
        q, qd, qdd = self.evaluate(t)
        return t, q, qd, qdd


def cubic_spline(waypoints: torch.Tensor, times=None, n: int = 100,
                 duration=None, v0=None, vf=None):
    """Build a clamped C2 cubic spline and sample it in one call.

    Returns (t (n,), q, qd, qdd) with the motion tensors shaped (..., n, dof),
    matching `quintic` in `kinfast.trajectory` for the unbatched two-waypoint
    case. Build a `CubicSpline` directly when you want to evaluate at times of
    your own choosing instead of an even grid.
    """
    return CubicSpline(waypoints, times=times, duration=duration,
                       v0=v0, vf=vf).sample(n)


def interpolate_pose(T_start: torch.Tensor, T_goal: torch.Tensor, n: int = 25):
    """Straight-line pose interpolation from T_start to T_goal.

    T_start, T_goal: (4, 4) or (B, 4, 4). Returns (B, n, 4, 4), or (n, 4, 4)
    when both inputs were unbatched.

    Position moves linearly and orientation follows the shortest arc between
    the two rotations, the matrix form of slerp: the relative rotation is taken
    to its axis-angle vector once and then replayed at a fraction of the angle.
    Constant speed in both, so sample i sits exactly i/(n-1) of the way along.
    """
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")
    squeeze = T_start.dim() == 2 and T_goal.dim() == 2
    A = T_start.reshape(-1, 4, 4)
    B_ = T_goal.reshape(-1, 4, 4)
    if A.shape[0] != B_.shape[0]:
        if A.shape[0] == 1:
            A = A.expand(B_.shape[0], 4, 4)
        elif B_.shape[0] == 1:
            B_ = B_.expand(A.shape[0], 4, 4)
        else:
            raise ValueError(
                f"batch mismatch: {A.shape[0]} start poses vs {B_.shape[0]} goals")
    dtype, device = A.dtype, A.device
    B = A.shape[0]

    s = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)      # (n,)
    p0, p1 = A[:, :3, 3], B_[:, :3, 3]
    p = p0.unsqueeze(1) + (p1 - p0).unsqueeze(1) * s.view(1, n, 1)   # (B,n,3)

    R0, R1 = A[:, :3, :3], B_[:, :3, :3]
    w = T.so3_log(R0.transpose(-1, -2) @ R1)                         # (B,3) in R0's frame
    theta = w.norm(dim=-1)                                           # (B,)
    axis = w.unsqueeze(1).expand(B, n, 3)
    R = R0.unsqueeze(1) @ T.axis_angle_to_matrix(axis, theta.unsqueeze(1) * s)

    out = T.make_transform(R, p)                                     # (B,n,4,4)
    return out[0] if squeeze else out


def cartesian_line(chain: CompiledChain, link, T_start: torch.Tensor,
                   T_goal: torch.Tensor, n: int = 25, q0: torch.Tensor = None,
                   retry: bool = True, **ik_kwargs):
    """Joint path that drags `link` along a straight line from T_start to T_goal.

    chain: a CompiledChain. link: link name or index (negatives allowed).
    T_start, T_goal: (4, 4) or (B, 4, 4) world poses.
    n: number of samples along the line, endpoints included.
    q0: (B, dof) seed for the first sample. Without one the first sample is
      solved from random restarts (pass restarts=K to control how many), which
      is the expensive part; every later sample is warm started from its
      predecessor and converges in a handful of iterations.
    retry: re-solve a sample from restarts when the warm start misses `tol`, so
      one bad sample does not poison the rest of the line. Costs one host sync
      per sample; pass False in a tight GPU loop where you trust the seed.
    Extra keyword arguments go straight to `kinfast.ik.ik` (iters, damping,
    step, tol, pos_only, ...).

    Returns (q, targets, info): q (B, n, dof) joint path, targets (B, n, 4, 4)
    the poses that were asked for, info with a (B, n) tensor of final IK errors
    per sample. The batch dimension is dropped when the inputs were unbatched.

    Warm starting is the whole point. IK is many-to-one, so solving each sample
    independently can jump between elbow-up and elbow-down between neighbouring
    points; seeding from the previous answer keeps the joint path continuous and
    cheap. The returned errors are worth checking: if the line leaves the
    workspace or crosses a singularity, IK degrades quietly rather than raising.
    """
    li = chain.link_index[link] if isinstance(link, str) else _resolve_link(chain, link)
    squeeze = T_start.dim() == 2 and T_goal.dim() == 2
    targets = interpolate_pose(T_start, T_goal, n)
    if squeeze:
        targets = targets.unsqueeze(0)                     # (1, n, 4, 4)
    B = targets.shape[0]

    # working dtype follows the caller's q when there is one, else the poses
    if q0 is not None:
        device, dtype = q0.device, q0.dtype
        seed = q0.reshape(B, chain.dof).to(device=device, dtype=dtype)
    else:
        device, dtype = targets.device, targets.dtype
        seed = None
    targets = targets.to(device=device, dtype=dtype)

    restarts = int(ik_kwargs.pop("restarts", 8))
    tol = float(ik_kwargs.get("tol", 1e-4))
    qs, errs = [], []
    for i in range(n):
        tgt = targets[:, i]
        if seed is None:
            q_i, info = ik(chain, tgt, link_index=li, restarts=restarts, **ik_kwargs)
            e_i = info["final_error"]
        else:
            q_i, info = ik(chain, tgt, q0=seed, link_index=li, **ik_kwargs)
            e_i = info["final_error"]
            if retry and bool((e_i > tol).any()):
                # the warm start landed in a bad spot (a singularity, or the
                # previous sample was already wrong); pay for a restart solve
                # here so the mistake does not propagate down the rest of the
                # line, and keep whichever answer is better per batch element
                q_r, info_r = ik(chain, tgt, link_index=li, restarts=restarts,
                                 **ik_kwargs)
                e_r = info_r["final_error"]
                better = (e_r < e_i).unsqueeze(-1)
                q_i = torch.where(better, q_r, q_i)
                e_i = torch.minimum(e_i, e_r)
        qs.append(q_i)
        errs.append(e_i)
        seed = q_i

    q = torch.stack(qs, dim=1)                              # (B, n, dof)
    err = torch.stack(errs, dim=1)                          # (B, n)
    if squeeze:
        q, targets, err = q[0], targets[0], err[0]
    return q, targets, {"final_error": err, "max_error": float(err.max())}
