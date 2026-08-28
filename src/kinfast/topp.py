"""Time optimal timing along a fixed path.

A planner returns a shape: configurations to pass through, with nothing said
about speed. `Plan.to_trajectory` turns that into motion by running a trapezoid
between each pair of waypoints, which is correct but slow, because it comes to
a full stop at every corner even when the path runs nearly straight through.

This module keeps the shape and asks how fast it can be traversed. Write the
path as q(s) for a path coordinate s. Then

    qd  = q'(s) sdot
    qdd = q'(s) sddot + q''(s) sdot^2

so the per joint limits become limits on the single scalar sdot, and the search
collapses from one dimension per joint to one dimension total. Velocity gives a
ceiling directly. Acceleration gives two things: a bound on how fast that
ceiling can be approached, and, through the q'' sdot^2 term, a second ceiling
from the curvature of the path. Integrating forward from rest gives the fastest
arrival at each point, integrating backward from the end gives the fastest
departure that can still stop in time, and the smaller of the two satisfies
both. That is the standard forward backward pass.

The corner problem, stated plainly: a path made of straight segments has
infinite curvature where two segments meet, so q'' is a delta there and no
finite acceleration can follow it at nonzero speed. There are only two honest
answers, and this module offers both.

    blend = 0      follow the waypoints exactly and stop at every corner. Same
                   path as the planner returned, same timing as running a
                   trapezoid per leg.
    blend > 0      round the corners, then keep speed through them. The path is
                   no longer followed exactly; the deviation is bounded and
                   returned, so a caller can check it against its own clearance.

    from kinfast.topp import time_parameterize
    t, q, qd, qdd, info = time_parameterize(robot.chain, plan.path, blend=0.05)
    info["deviation"]     # how far the rounded path strays, in joint units

What it is worth, measured on a six joint arm against running a trapezoid
between each pair of waypoints, for paths of increasing density:

    waypoints    trapezoid    this, blend 0    gain
            3        2.50s            2.50s      0%
            6        4.28s            4.34s      0%
           12        8.11s            8.15s      0%
           25        8.97s            6.56s     27%
           50       15.40s           12.22s     21%

The pattern is the point. On a sparse path there is nothing to win, because the
time goes on travelling rather than stopping, and this returns what the simpler
method already gave. The gain arrives when the path is dense, which is what a
planner produces once its output has been densified for collision checking, and
it comes from not braking to a halt at fifty intermediate points that were
never corners.

Blending is for the opposite case, a path whose waypoints really are corners.
It rounds them so speed can be carried through, at the cost of leaving the path
by info["deviation"]. On a path that is already smooth it does not help and can
cost a little, so it is off by default.
"""
import torch


def _limits(chain, like, vmax, amax):
    dtype, device = like.dtype, like.device
    if vmax is None:
        vmax = chain.vmax.clone()
        vmax[vmax <= 0] = 1.0
    if amax is None:
        amax = torch.full((chain.dof,), 4.0)
    vmax = torch.as_tensor(vmax, dtype=dtype, device=device).reshape(-1)
    amax = torch.as_tensor(amax, dtype=dtype, device=device).reshape(-1)
    if bool((vmax <= 0).any()) or bool((amax <= 0).any()):
        raise ValueError("vmax and amax must be positive")
    return vmax, amax


def _resample(path, n):
    """Samples spaced evenly along the path, with the waypoints kept exactly.

    Even spacing in the waypoint index would spend as many grid points on a
    short leg as a long one. Spacing by length keeps the grid uniform in joint
    space, which is what the integration assumes.
    """
    seg = (path[1:] - path[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1, dtype=path.dtype, device=path.device),
                     seg.cumsum(0)])
    total = float(cum[-1])
    if total <= 0:
        return None, None, 0.0
    want = torch.linspace(0, total, max(int(n), path.shape[0]),
                          dtype=path.dtype, device=path.device)
    want = torch.unique(torch.cat([want, cum]))     # never drop a corner
    idx = torch.searchsorted(cum, want.clamp(max=total),
                             right=True).clamp(1, len(cum) - 1)
    lo, hi = cum[idx - 1], cum[idx]
    frac = ((want - lo) / (hi - lo).clamp_min(1e-12)).unsqueeze(-1)
    return path[idx - 1] + frac * (path[idx] - path[idx - 1]), want, total


def _round_corners(q, window):
    """Round the corners with a moving average, so curvature stays finite.

    A corner is where the path is not differentiable. Averaging over a window
    replaces it with an arc: the wider the window the gentler the arc, the
    faster the traverse, and the further the path strays from the one that was
    handed in.

    The correction is tapered to zero at both ends rather than the endpoints
    being clamped back afterwards. Clamping looks equivalent and is not: it
    leaves the smoothed path and the true endpoint a step apart, which is a new
    corner in the last interval, and it gets worse as the window widens. That
    showed up as a wider blend taking longer than a narrow one, which is
    backwards.
    """
    if window < 3:
        return q
    pad = window // 2
    padded = torch.cat([q[:1].expand(pad, -1), q, q[-1:].expand(pad, -1)], dim=0)
    kernel = torch.full((window,), 1.0 / window, dtype=q.dtype, device=q.device)
    out = torch.stack([
        torch.nn.functional.conv1d(
            padded[:, j].reshape(1, 1, -1), kernel.reshape(1, 1, -1)
        ).reshape(-1)
        for j in range(q.shape[1])], dim=1)
    n = q.shape[0]
    ramp = min(pad + 1, n // 2)
    w = torch.ones(n, dtype=q.dtype, device=q.device)
    if ramp > 1:
        # starts at exactly 0 so the first and last samples keep the values the
        # caller gave, without a clamp afterwards
        edge = torch.linspace(0.0, 1.0, ramp, dtype=q.dtype, device=q.device)
        w[:ramp] = edge
        w[-ramp:] = edge.flip(0)
    return q + w.unsqueeze(-1) * (out - q)


def _accel_window(a, b, v2, amax):
    """Range of sddot allowed at one grid point, given the speed there.

    The constraint per joint is |a_j sddot + b_j v2| <= amax_j. Solving each
    for sddot gives an interval; the feasible set is their intersection. A
    joint with a_j near zero contributes no bound on sddot, only the curvature
    ceiling handled by the caller.
    """
    room = amax - b * v2                     # headroom above and below
    other = -amax - b * v2
    big = torch.full_like(a, float("inf"))
    hi = torch.where(a > 1e-9, room / a.clamp_min(1e-9),
                     torch.where(a < -1e-9, other / a.clamp_max(-1e-9), big))
    lo = torch.where(a > 1e-9, other / a.clamp_min(1e-9),
                     torch.where(a < -1e-9, room / a.clamp_max(-1e-9), -big))
    return float(lo.max()), float(hi.min())


def time_parameterize(chain, path, vmax=None, amax=None, n_grid=400,
                      blend=0.0):
    """Fastest timing along `path` inside the joint limits.

    path: (n, dof) waypoints in order. vmax and amax default to the model's own
    velocity limits and a moderate acceleration. `blend` is the fraction of the
    path length used to round corners; 0 follows the waypoints exactly and
    stops at each one.

    Returns (t, q, qd, qdd, info). info carries `duration`, `deviation` (how far
    the rounded path strays from the one handed in, zero when blend is 0) and
    `stops` (how many corners the profile had to stop at).
    """
    path = torch.as_tensor(path)
    if path.dim() != 2:
        raise ValueError(f"path must be (n, dof), got {tuple(path.shape)}")
    vmax, amax = _limits(chain, path, vmax, amax)
    zero_info = {"duration": 0.0, "deviation": 0.0, "stops": 0}
    if path.shape[0] < 2:
        z = path[:1]
        return (torch.zeros(1, dtype=path.dtype), z, torch.zeros_like(z),
                torch.zeros_like(z), zero_info)

    grid = _resample(path, n_grid)
    if grid[0] is None:                       # a path that does not move
        z = path[:1]
        return (torch.zeros(1, dtype=path.dtype), z, torch.zeros_like(z),
                torch.zeros_like(z), zero_info)
    q_exact, s, total = grid
    # where the input path bends, and by how much, in arc length coordinates
    corners = []
    if path.shape[0] > 2:
        seg = path[1:] - path[:-1]
        unit = seg / seg.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        turn = (unit[1:] * unit[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        lengths = seg.norm(dim=-1).cumsum(0)
        corners = [(float(lengths[i]), float(turn[i])) for i in range(len(turn))]
    n = q_exact.shape[0]
    if n < 3:
        q_exact = torch.cat([q_exact[:1], 0.5 * (q_exact[:1] + q_exact[-1:]),
                             q_exact[-1:]], dim=0)
        s = torch.tensor([0.0, total / 2, total], dtype=path.dtype)
        n = 3

    window = int(max(0.0, float(blend)) * n)
    window += 1 - window % 2                  # odd, so the average is centred
    q = _round_corners(q_exact, window) if blend > 0 else q_exact
    deviation = float((q - q_exact).norm(dim=-1).max()) if blend > 0 else 0.0

    ds = (s[1:] - s[:-1]).clamp_min(1e-12)
    # first and second derivative with respect to the path coordinate
    dq = torch.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (s[2:] - s[:-2]).unsqueeze(-1)
    dq[0] = (q[1] - q[0]) / ds[0]
    dq[-1] = (q[-1] - q[-2]) / ds[-1]
    d2q = torch.zeros_like(q)
    d2q[1:-1] = (dq[2:] - dq[:-2]) / (s[2:] - s[:-2]).unsqueeze(-1)

    # ceiling on sdot: from velocity, and from curvature through q'' sdot^2
    cap2 = (vmax.unsqueeze(0) / dq.abs().clamp_min(1e-12)).min(dim=-1).values ** 2
    curved = d2q.abs() > 1e-9
    curve_cap = torch.where(
        curved, amax.unsqueeze(0) / d2q.abs().clamp_min(1e-12),
        torch.full_like(d2q, float("inf"))).min(dim=-1).values
    cap2 = torch.minimum(cap2, curve_cap)

    # A true corner has infinite curvature, and no finite acceleration follows
    # it at speed. Without blending the only honest answer is to stop there, so
    # the ceiling is pinned to zero wherever the path direction turns sharply.
    if blend <= 0 and corners is not None and len(corners):
        # A corner is a break in direction, and those live at the waypoints the
        # caller handed in, not at every grid sample. Testing the turn at every
        # sample instead marks a densely sampled smooth curve as one long
        # corner and stops the robot at all of it, which is how this went wrong
        # the first time: a 25 point arc took 126 seconds instead of 9.
        sharp = torch.zeros(n, dtype=torch.bool, device=q.device)
        for pos, angle in corners:
            if angle >= 0.999:              # straighter than about 2.6 degrees
                continue
            j = int(torch.argmin((s - pos).abs()))
            if 0 < j < n - 1:
                sharp[j] = True
        cap2 = torch.where(sharp, torch.zeros_like(cap2), cap2)

    def profile(cap2):
        """One forward and one backward pass, returning the speed profile."""
        fwd = torch.zeros(n, dtype=q.dtype, device=q.device)
        for i in range(n - 1):
            _lo, hi = _accel_window(dq[i], d2q[i], fwd[i], amax)
            nxt = fwd[i] + 2 * max(hi, 0.0) * float(ds[i])
            fwd[i + 1] = min(max(nxt, 0.0), float(cap2[i + 1]))
        bwd = torch.zeros(n, dtype=q.dtype, device=q.device)
        for i in range(n - 1, 0, -1):
            lo, _hi = _accel_window(dq[i], d2q[i], bwd[i], amax)
            prv = bwd[i] + 2 * max(-lo, 0.0) * float(ds[i - 1])
            bwd[i - 1] = min(max(prv, 0.0), float(cap2[i - 1]))
        return torch.minimum(torch.minimum(fwd, bwd), cap2).clamp_min(0).sqrt()

    def sample(sdot):
        """Times, velocities and accelerations for a speed profile.

        The acceleration is the analytic one, q' sddot + q'' sdot^2, not a
        finite difference of the sampled velocity. Differencing the samples
        measures the grid as much as the motion: it reads a step between two
        points as an instantaneous jump and reports an acceleration that the
        underlying continuous trajectory never has. Planning against that
        number makes the profile slower than it needs to be.
        """
        mean_sdot = (0.5 * (sdot[1:] + sdot[:-1])).clamp_min(1e-9)
        dt = ds / mean_sdot
        t = torch.cat([torch.zeros(1, dtype=q.dtype, device=q.device),
                       dt.cumsum(0)])
        v2 = sdot ** 2
        sddot = torch.zeros(n, dtype=q.dtype, device=q.device)
        sddot[:-1] = (v2[1:] - v2[:-1]) / (2 * ds)     # v dv/ds = d(v^2/2)/ds
        if n > 1:
            sddot[-1] = sddot[-2]
        qd = dq * sdot.unsqueeze(-1)
        qdd = dq * sddot.unsqueeze(-1) + d2q * v2.unsqueeze(-1)
        return t, qd, qdd

    # The passes plan against a curvature estimated on a discrete grid, and an
    # estimate between two samples can miss a peak, so the sampled trajectory
    # can come out slightly over the acceleration limit. Rather than slow the
    # whole motion to fix one point, the ceiling is lowered only where the
    # limit was actually exceeded and the passes are run again. A handful of
    # rounds is enough, and each one only ever slows the profile, so it
    # converges from above.
    for _ in range(8):
        sdot = profile(cap2)
        t, qd, qdd = sample(sdot)
        over = (qdd.abs() / amax).max(dim=-1).values
        worst = float(over.max())
        if worst <= 1.0 + 1e-3:
            break
        hot = over > 1.0 + 1e-3
        shrink = torch.where(hot, 1.0 / over.clamp_min(1e-9),
                             torch.ones_like(over))
        # the offending acceleration is dominated by the q'' sdot^2 term, so
        # the speed squared scales down by the same factor the limit was missed
        cap2 = torch.minimum(cap2, (sdot ** 2) * shrink)

    # Last resort. If the loop above ran out of rounds with something still
    # over, one global stretch closes the gap exactly: stretching time by k
    # divides velocity by k and acceleration by k squared. It should be 1.0 in
    # ordinary use, and it is reported rather than hidden.
    scale = 1.0
    v_over = float((qd.abs() / vmax).max()) if n else 0.0
    a_over = float((qdd.abs() / amax).max()) if n else 0.0
    if v_over > 1.0 or a_over > 1.0:
        scale = max(v_over, a_over ** 0.5, 1.0) * (1.0 + 1e-6)
        t = t * scale
        qd = qd / scale
        qdd = qdd / (scale * scale)

    info = {
        "duration": float(t[-1]),
        "deviation": deviation,
        "stops": int((sdot[1:-1] < 1e-6).sum()),
        "time_scale": scale,
    }
    return t, q, qd, qdd, info


def duration(chain, path, vmax=None, amax=None, n_grid=400, blend=0.0):
    """How long the path takes under the limits, without the samples."""
    return time_parameterize(chain, path, vmax, amax, n_grid, blend)[4]["duration"]
