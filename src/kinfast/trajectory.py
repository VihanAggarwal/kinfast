# src/kinfast/trajectory.py
"""Point-to-point trajectory generation under joint limits.

Two workhorse profiles, both returning sampled (q, qd, qdd) tensors:

  quintic      minimum-jerk-style smooth motion with zero boundary velocity and
               acceleration, the default for "move nicely from A to B".
  trapezoidal  time-optimal under per-joint velocity/acceleration limits, with
               all joints synchronized to finish together (the slowest joint sets
               the duration; faster joints lower their peak velocity to match).

Everything is closed-form tensor math: differentiable, batchable over sample
times, and exactly verifiable against hand-computed oracles.
"""
import torch


def quintic(q0: torch.Tensor, qf: torch.Tensor, T: float, n: int = 100):
    """Quintic point-to-point profile.

    q0, qf: (dof,) start/goal. T: duration (s). n: samples.
    Returns (t (n,), q (n,dof), qd (n,dof), qdd (n,dof)).
    Boundary conditions: q(0)=q0, q(T)=qf, qd=qdd=0 at both ends.
    """
    dtype = q0.dtype
    t = torch.linspace(0.0, T, n, dtype=dtype, device=q0.device)
    tau = (t / T).unsqueeze(-1)                       # (n,1)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    d = (qf - q0).unsqueeze(0)                        # (1,dof)
    return t, q0.unsqueeze(0) + d * s, d * ds, d * dds


def _min_time(dist, vmax, amax):
    """Per-joint minimum time for a rest-to-rest trapezoid/triangle profile."""
    triangle = 2.0 * torch.sqrt(dist / amax)
    trapezoid = dist / vmax + vmax / amax
    return torch.where(dist * amax <= vmax**2, triangle, trapezoid)


def trapezoidal(q0: torch.Tensor, qf: torch.Tensor, vmax: torch.Tensor,
                amax: torch.Tensor, n: int = 100):
    """Synchronized trapezoidal velocity profile, time-optimal under limits.

    q0, qf, vmax, amax: (dof,). Returns (t (n,), q, qd, qdd each (n,dof), T).
    The duration T is the max of per-joint minimum times; each joint's peak
    velocity is then re-solved so every joint arrives exactly at T.
    """
    dtype = q0.dtype
    delta = qf - q0
    dist = delta.abs()
    sign = torch.sign(delta)
    moving = dist > 1e-12

    T = float(_min_time(dist[moving], vmax[moving], amax[moving]).max()) if moving.any() else 1e-3

    # Peak velocity per joint so that d = v*(T - v/a):  v^2 - a*T*v + a*d = 0.
    disc = (amax * T) ** 2 - 4.0 * amax * dist
    v = (amax * T - torch.sqrt(disc.clamp_min(0.0))) / 2.0
    v = torch.where(moving, v, torch.zeros_like(v))
    t_acc = torch.where(moving, v / amax, torch.zeros_like(v))   # (dof,)

    t = torch.linspace(0.0, T, n, dtype=dtype, device=q0.device)
    tt = t.unsqueeze(-1)                                          # (n,1)
    t1 = t_acc.unsqueeze(0)                                       # (1,dof)
    t2 = T - t1
    a = amax.unsqueeze(0)
    vv = v.unsqueeze(0)

    # phase masks
    in_acc = tt < t1
    in_dec = tt >= t2
    in_cruise = ~in_acc & ~in_dec

    pos = (in_acc * (0.5 * a * tt**2)
           + in_cruise * (0.5 * a * t1**2 + vv * (tt - t1))
           + in_dec * (dist.unsqueeze(0) - 0.5 * a * (T - tt) ** 2))
    vel = in_acc * (a * tt) + in_cruise * vv + in_dec * (a * (T - tt))
    acc = in_acc * a + in_dec * (-a)

    m = moving.unsqueeze(0)
    s = sign.unsqueeze(0)
    q = q0.unsqueeze(0) + torch.where(m, s * pos, torch.zeros_like(pos))
    qd = torch.where(m, s * vel, torch.zeros_like(vel))
    qdd = torch.where(m, s * acc, torch.zeros_like(acc))
    return t, q, qd, qdd, T
