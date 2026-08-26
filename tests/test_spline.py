# tests/test_spline.py
"""Spline trajectories checked against oracles that are not this library.

The joint-space spline is validated three ways: a hand-derived closed form for
the two-waypoint case (a clamped spline through two points is exactly the
smoothstep cubic), float64 central differences for the derivative chain, and
scipy's own clamped CubicSpline where scipy is installed. The Cartesian line is
validated geometrically: forward kinematics of the solved joint path must sit on
the straight segment between the two poses, and the interpolated orientation is
checked against a hand-computed half rotation.
"""
import math

import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.trajectory_spline import (CubicSpline, cartesian_line,
                                       cubic_spline, interpolate_pose)
from kinfast.urdf.parse import parse_urdf_string
from kinfast import transforms as TR
from tests.test_spatial import SIX_DOF

F64 = torch.float64


def _waypoints(seed=0, K=6, D=4, dtype=F64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K, D, generator=g, dtype=torch.float64).to(dtype)


def _knots(dtype=F64):
    # deliberately non-uniform: uniform spacing hides h-dependent algebra bugs
    return torch.tensor([0.0, 0.4, 1.6, 2.0, 3.5, 4.1], dtype=dtype)


def _six_dof(dtype=F64):
    return compile_robot(parse_urdf_string(SIX_DOF), dtype=dtype)


# --------------------------------------------------------------------------
# interpolation property: the spline goes through the waypoints
# --------------------------------------------------------------------------

def test_spline_passes_through_every_waypoint():
    W, tk = _waypoints(), _knots()
    q, qd, qdd = CubicSpline(W, times=tk).evaluate(tk)
    assert torch.allclose(q, W, atol=1e-12)


def test_sampled_grid_lands_on_waypoints():
    """With uniform knots and a grid that includes them, samples hit exactly."""
    W = _waypoints(seed=1, K=5, D=3)
    # 5 waypoints, 4 segments, 10 samples per segment -> knots at 0,10,20,30,40
    t, q, qd, qdd = cubic_spline(W, n=41, duration=4.0)
    assert torch.allclose(t[::10], torch.linspace(0, 4, 5, dtype=F64), atol=1e-12)
    assert torch.allclose(q[::10], W, atol=1e-12)


def test_clamped_end_velocities_are_zero():
    W, tk = _waypoints(seed=2), _knots()
    t, q, qd, qdd = CubicSpline(W, times=tk).sample(257)
    assert torch.allclose(qd[0], torch.zeros(W.shape[-1], dtype=F64), atol=1e-12)
    assert torch.allclose(qd[-1], torch.zeros(W.shape[-1], dtype=F64), atol=1e-12)
    assert torch.allclose(q[0], W[0], atol=1e-12)
    assert torch.allclose(q[-1], W[-1], atol=1e-12)


def test_prescribed_end_velocities_are_honoured():
    W, tk = _waypoints(seed=3), _knots()
    v0 = torch.tensor([0.5, -1.0, 0.0, 2.0], dtype=F64)
    vf = torch.tensor([-0.25, 0.75, 1.5, 0.0], dtype=F64)
    _, qd, _ = CubicSpline(W, times=tk, v0=v0, vf=vf).evaluate(tk)
    assert torch.allclose(qd[0], v0, atol=1e-12)
    assert torch.allclose(qd[-1], vf, atol=1e-12)


# --------------------------------------------------------------------------
# hand-computed oracle: two waypoints, clamped, is the smoothstep cubic
# --------------------------------------------------------------------------

def test_two_waypoint_spline_is_smoothstep_closed_form():
    """q(s) = q0 + (q1-q0)(3s^2 - 2s^3) with s = t/T, derived by hand from the
    Hermite basis with both end slopes pinned to zero."""
    q0 = torch.tensor([1.0, -2.0], dtype=F64)
    q1 = torch.tensor([2.5, 0.5], dtype=F64)
    T = 1.3
    t, q, qd, qdd = cubic_spline(torch.stack([q0, q1]), n=64, duration=T)
    s = (t / T).unsqueeze(-1)
    d = (q1 - q0).unsqueeze(0)
    assert torch.allclose(q, q0 + d * (3 * s**2 - 2 * s**3), atol=1e-12)
    assert torch.allclose(qd, d * (6 * s - 6 * s**2) / T, atol=1e-12)
    assert torch.allclose(qdd, d * (6 - 12 * s) / T**2, atol=1e-12)


def test_straight_line_waypoints_reproduce_exactly():
    """Collinear-in-time waypoints on a cubic must be reproduced to machine
    precision: a cubic spline reproduces cubics."""
    tk = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=F64)
    # a genuine cubic in t, with slopes at the ends fed in as the clamp
    def f(t):
        return torch.stack([2 * t**3 - t**2 + 0.5 * t + 3, -t**3 + 4 * t], dim=-1)

    def df(t):
        return torch.stack([6 * t**2 - 2 * t + 0.5, -3 * t**2 + 4], dim=-1)

    sp = CubicSpline(f(tk), times=tk, v0=df(tk[0]), vf=df(tk[-1]))
    ts = torch.linspace(0.0, 3.0, 97, dtype=F64)
    q, qd, _ = sp.evaluate(ts)
    assert torch.allclose(q, f(ts), atol=1e-11)
    assert torch.allclose(qd, df(ts), atol=1e-10)


# --------------------------------------------------------------------------
# derivative chain: finite differences and C2 continuity
# --------------------------------------------------------------------------

def _away_from_knots(t, tk, dt, k=3.0):
    """Mask of samples whose central-difference stencil does not straddle a
    knot. The third derivative jumps at a knot, so a stencil that spans one is
    O(dt) accurate instead of O(dt^2) and would swamp a tight tolerance."""
    return (t.unsqueeze(-1) - tk).abs().min(dim=-1).values > k * dt


def _fd_velocity_error(sp, tk, n):
    t, q, qd, _ = sp.sample(n)
    dt = float(t[1] - t[0])
    fd = (q[2:] - q[:-2]) / (2 * dt)
    keep = _away_from_knots(t[1:-1], tk, dt)
    return float((qd[1:-1][keep] - fd[keep]).abs().max())


def test_qd_is_finite_difference_derivative_of_q():
    """qd matches a float64 central difference of q, and the gap is pure
    stencil truncation: halving the step quarters it, the O(dt^2) signature."""
    W, tk = _waypoints(seed=4), _knots()
    sp = CubicSpline(W, times=tk)
    coarse = _fd_velocity_error(sp, tk, 10001)
    fine = _fd_velocity_error(sp, tk, 20001)
    assert fine < 3e-6
    assert 3.5 < coarse / fine < 4.5
    # across the knots the stencil is only first order, so allow more there
    t, q, qd, _ = sp.sample(20001)
    dt = float(t[1] - t[0])
    assert float((qd[1:-1] - (q[2:] - q[:-2]) / (2 * dt)).abs().max()) < 1e-3


def test_qdd_is_finite_difference_derivative_of_qd():
    W, tk = _waypoints(seed=5), _knots()
    t, q, qd, qdd = CubicSpline(W, times=tk).sample(20001)
    dt = float(t[1] - t[0])
    fd = (qd[2:] - qd[:-2]) / (2 * dt)
    # the second derivative has a kink at each knot, so a couple of samples
    # straddle it; every other sample must match tightly
    err = (qdd[1:-1] - fd).abs().max(dim=-1).values
    assert float(err.median()) < 1e-8
    assert int((err > 1e-6).sum()) <= 2 * (W.shape[0] - 2)


def test_qdd_is_continuous_across_every_knot():
    """C2 is the whole point: the one-sided second derivatives at each interior
    knot must agree. A plain Hermite fit with, say, finite-difference slopes
    would pass the waypoint test and fail this one."""
    W, tk = _waypoints(seed=6), _knots()
    sp = CubicSpline(W, times=tk)
    eps = 1e-7
    left = sp.evaluate(tk[1:-1] - eps)[2]
    right = sp.evaluate(tk[1:-1] + eps)[2]
    assert torch.allclose(left, right, atol=1e-5)
    # and the jump is genuinely zero rather than merely small in scale
    scale = sp.evaluate(tk)[2].abs().max()
    assert float((left - right).abs().max()) < 1e-6 * float(scale)


def test_qdd_has_no_jumps_on_a_dense_sample():
    W, tk = _waypoints(seed=7), _knots()
    t, _, _, qdd = CubicSpline(W, times=tk).sample(4001)
    dt = float(t[1] - t[0])
    jump = (qdd[1:] - qdd[:-1]).abs().max()
    # qdd is piecewise linear, so consecutive samples differ by O(dt); a
    # discontinuity would show up as an O(1) step
    assert float(jump) < 200.0 * dt


# --------------------------------------------------------------------------
# scipy as an independent implementation
# --------------------------------------------------------------------------

def test_matches_scipy_clamped_cubic_spline():
    scipy_interp = pytest.importorskip("scipy.interpolate")
    import numpy as np
    W, tk = _waypoints(seed=8, K=7, D=3), None
    tk = torch.tensor([0.0, 0.3, 1.1, 1.15, 2.7, 3.0, 4.4], dtype=F64)
    sp = CubicSpline(W, times=tk)
    ref = scipy_interp.CubicSpline(tk.numpy(), W.numpy(), axis=0, bc_type="clamped")
    ts = np.linspace(0.0, 4.4, 211)
    q, qd, qdd = sp.evaluate(torch.tensor(ts, dtype=F64))
    assert np.abs(q.numpy() - ref(ts)).max() < 1e-12
    assert np.abs(qd.numpy() - ref(ts, 1)).max() < 1e-11
    assert np.abs(qdd.numpy() - ref(ts, 2)).max() < 1e-10


# --------------------------------------------------------------------------
# batching, dtype, differentiability
# --------------------------------------------------------------------------

def test_batched_spline_matches_per_item_splines():
    g = torch.Generator().manual_seed(9)
    W = torch.randn(5, 6, 4, generator=g, dtype=F64)      # (B=5, K=6, D=4)
    tk = _knots()
    t, q, qd, qdd = cubic_spline(W, times=tk, n=53)
    assert q.shape == (5, 53, 4) and t.shape == (53,)
    for b in range(5):
        tb, qb, qdb, qddb = cubic_spline(W[b], times=tk, n=53)
        assert torch.allclose(q[b], qb, atol=1e-12)
        assert torch.allclose(qd[b], qdb, atol=1e-12)
        assert torch.allclose(qdd[b], qddb, atol=1e-12)


def test_multiple_leading_batch_dimensions():
    g = torch.Generator().manual_seed(10)
    W = torch.randn(2, 3, 4, 2, generator=g, dtype=F64)   # (2,3) batch, K=4, D=2
    t, q, qd, qdd = cubic_spline(W, n=17, duration=2.0)
    assert q.shape == (2, 3, 17, 2)
    assert torch.allclose(q[..., 0, :], W[..., 0, :], atol=1e-12)
    assert torch.allclose(q[..., -1, :], W[..., -1, :], atol=1e-12)
    assert torch.allclose(q[1, 2], cubic_spline(W[1, 2], n=17, duration=2.0)[1],
                          atol=1e-12)


def test_dtype_follows_the_waypoints():
    W32 = _waypoints(seed=11).to(torch.float32)
    t, q, qd, qdd = cubic_spline(W32, times=_knots(torch.float32), n=33)
    assert t.dtype == q.dtype == qd.dtype == qdd.dtype == torch.float32
    # same numbers as float64 within single precision
    t64, q64, _, _ = cubic_spline(W32.double(), times=_knots(), n=33)
    assert torch.allclose(q.double(), q64, atol=1e-5)


def test_spline_is_differentiable_in_the_waypoints():
    W = _waypoints(seed=12, K=4, D=2).requires_grad_(True)
    t, q, qd, qdd = cubic_spline(W, n=25, duration=3.0)
    # penalise acceleration, the usual smoothing objective
    (qdd**2).mean().backward()
    assert W.grad is not None and torch.isfinite(W.grad).all()
    assert float(W.grad.abs().max()) > 0.0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        CubicSpline(torch.zeros(1, 3, dtype=F64))                  # one waypoint
    with pytest.raises(ValueError):
        CubicSpline(torch.zeros(3, 2, dtype=F64), times=[0.0, 1.0])  # wrong K
    with pytest.raises(ValueError):
        CubicSpline(torch.zeros(3, 2, dtype=F64), times=[0.0, 1.0, 0.5])
    with pytest.raises(ValueError):
        CubicSpline(torch.zeros(3, 2, dtype=F64), duration=0.0)


# --------------------------------------------------------------------------
# pose interpolation
# --------------------------------------------------------------------------

def test_interpolated_positions_lie_on_the_segment():
    A = torch.eye(4, dtype=F64)
    A[:3, 3] = torch.tensor([0.1, -0.2, 0.3], dtype=F64)
    B = torch.eye(4, dtype=F64)
    B[:3, 3] = torch.tensor([-0.4, 0.5, 0.9], dtype=F64)
    B[:3, :3] = TR.rpy_to_matrix(torch.tensor([0.3, -0.7, 1.1], dtype=F64))
    poses = interpolate_pose(A, B, n=11)
    s = torch.linspace(0, 1, 11, dtype=F64).unsqueeze(-1)
    want = A[:3, 3] + s * (B[:3, 3] - A[:3, 3])
    assert torch.allclose(poses[:, :3, 3], want, atol=1e-14)
    assert torch.allclose(poses[0], A, atol=1e-12)
    assert torch.allclose(poses[-1], B, atol=1e-12)


def test_orientation_midpoint_is_a_half_rotation():
    """Hand oracle: 90 degrees about z, halfway along, is 45 degrees about z."""
    A = torch.eye(4, dtype=F64)
    B = torch.eye(4, dtype=F64)
    B[:3, :3] = TR.rpy_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2], dtype=F64))
    poses = interpolate_pose(A, B, n=3)
    half = TR.rpy_to_matrix(torch.tensor([0.0, 0.0, math.pi / 4], dtype=F64))
    assert torch.allclose(poses[1, :3, :3], half, atol=1e-12)
    # rotation angle grows linearly with the sample fraction
    poses = interpolate_pose(A, B, n=9)
    ang = TR.so3_log(poses[:, :3, :3]).norm(dim=-1)
    assert torch.allclose(ang, torch.linspace(0, math.pi / 2, 9, dtype=F64), atol=1e-12)


def test_interpolate_pose_batches():
    g = torch.Generator().manual_seed(13)
    R0 = TR.rpy_to_matrix(torch.randn(3, 3, generator=g, dtype=F64))
    R1 = TR.rpy_to_matrix(torch.randn(3, 3, generator=g, dtype=F64))
    A = TR.make_transform(R0, torch.randn(3, 3, generator=g, dtype=F64))
    B = TR.make_transform(R1, torch.randn(3, 3, generator=g, dtype=F64))
    poses = interpolate_pose(A, B, n=7)
    assert poses.shape == (3, 7, 4, 4)
    for b in range(3):
        assert torch.allclose(poses[b], interpolate_pose(A[b], B[b], n=7), atol=1e-14)
    # every frame is a proper rotation
    R = poses[..., :3, :3]
    eye = torch.eye(3, dtype=F64).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-12)


# --------------------------------------------------------------------------
# cartesian_line on the 6-DOF arm
# --------------------------------------------------------------------------

def _line_deviation(chain, li, q, T_a, T_b):
    """Perpendicular distance from each solved end-effector position to the
    infinite line through the two poses, in metres."""
    p = forward_kinematics(chain, q)[:, li, :3, 3]
    p0, p1 = T_a[:3, 3], T_b[:3, 3]
    u = (p1 - p0) / (p1 - p0).norm()
    rel = p - p0
    perp = rel - (rel @ u).unsqueeze(-1) * u
    return perp.norm(dim=-1)


def test_cartesian_line_stays_within_5mm_of_the_straight_line():
    chain = _six_dof()
    li = chain.link_index["ee"]
    qa = torch.tensor([[0.30, 0.50, -0.80, 0.20, 0.60, 0.10]], dtype=F64)
    qb = torch.tensor([[-0.40, 0.90, -0.50, -0.30, 0.40, 0.50]], dtype=F64)
    T_a = forward_kinematics(chain, qa)[0, li]
    T_b = forward_kinematics(chain, qb)[0, li]
    # the two poses are 0.4 m apart, so 5 mm is a real tolerance, not a freebie
    assert float((T_b[:3, 3] - T_a[:3, 3]).norm()) > 0.3

    torch.manual_seed(0)
    q, targets, info = cartesian_line(chain, "ee", T_a, T_b, n=25, q0=qa,
                                      iters=200, damping=0.02, tol=1e-9)
    assert q.shape == (25, 6) and targets.shape == (25, 4, 4)
    dev = _line_deviation(chain, li, q, T_a, T_b)
    assert float(dev.max()) < 5e-3, f"max deviation {float(dev.max()) * 1e3:.3f} mm"
    # the endpoints really are the requested poses
    p = forward_kinematics(chain, q)[:, li, :3, 3]
    assert torch.allclose(p[0], T_a[:3, 3], atol=1e-6)
    assert torch.allclose(p[-1], T_b[:3, 3], atol=1e-6)


def test_cartesian_line_tracks_orientation_too():
    chain = _six_dof()
    li = chain.link_index["ee"]
    qa = torch.tensor([[0.20, 0.60, -0.70, 0.30, 0.50, -0.20]], dtype=F64)
    qb = torch.tensor([[-0.30, 0.80, -0.90, -0.20, 0.70, 0.40]], dtype=F64)
    T_a = forward_kinematics(chain, qa)[0, li]
    T_b = forward_kinematics(chain, qb)[0, li]
    torch.manual_seed(1)
    q, targets, info = cartesian_line(chain, li, T_a, T_b, n=21, q0=qa,
                                      iters=200, damping=0.02, tol=1e-9)
    poses = forward_kinematics(chain, q)[:, li]
    ang = TR.so3_log(poses[:, :3, :3] @ targets[:, :3, :3].transpose(-1, -2))
    assert float(ang.norm(dim=-1).max()) < 1e-4
    assert float(info["max_error"]) < 1e-6


def test_cartesian_line_joint_path_is_continuous():
    """Warm starting is what buys this: cold solves would flip branches."""
    chain = _six_dof()
    li = chain.link_index["ee"]
    qa = torch.tensor([[0.30, 0.50, -0.80, 0.20, 0.60, 0.10]], dtype=F64)
    qb = torch.tensor([[-0.40, 0.90, -0.50, -0.30, 0.40, 0.50]], dtype=F64)
    T_a = forward_kinematics(chain, qa)[0, li]
    T_b = forward_kinematics(chain, qb)[0, li]
    torch.manual_seed(2)
    q, _, _ = cartesian_line(chain, "ee", T_a, T_b, n=41, q0=qa,
                             iters=200, damping=0.02, tol=1e-9)
    assert torch.allclose(q[0], qa[0], atol=1e-12)   # already on target, so no drift
    step = (q[1:] - q[:-1]).abs().max(dim=-1).values
    assert float(step.max()) < 0.25
    length = float((q[1:] - q[:-1]).norm(dim=-1).sum())
    # refining the sampling must not lengthen the joint path: that is the
    # signature of a continuous branch. Cold solves that flip between IK
    # branches give a length that keeps growing with the sample count.
    q_fine, _, _ = cartesian_line(chain, "ee", T_a, T_b, n=161, q0=qa,
                                  iters=200, damping=0.02, tol=1e-9)
    length_fine = float((q_fine[1:] - q_fine[:-1]).norm(dim=-1).sum())
    assert abs(length_fine - length) < 0.01 * length
    # four times the samples, roughly a quarter of the step
    assert float((q_fine[1:] - q_fine[:-1]).abs().max()) < 0.08

    # and a spline through the solved knots is a usable, smooth joint plan
    t, qs, qd, qdd = cubic_spline(q, n=161, duration=4.0)
    assert torch.allclose(qs[0], q[0], atol=1e-12)
    assert torch.allclose(qd[0], torch.zeros(6, dtype=F64), atol=1e-12)


def test_cartesian_line_batched_over_targets():
    chain = _six_dof()
    li = chain.link_index["ee"]
    qa = torch.tensor([[0.30, 0.50, -0.80, 0.20, 0.60, 0.10],
                       [0.20, 0.60, -0.70, 0.30, 0.50, -0.20]], dtype=F64)
    qb = torch.tensor([[-0.40, 0.90, -0.50, -0.30, 0.40, 0.50],
                       [-0.30, 0.80, -0.90, -0.20, 0.70, 0.40]], dtype=F64)
    T_a = forward_kinematics(chain, qa)[:, li]
    T_b = forward_kinematics(chain, qb)[:, li]
    torch.manual_seed(3)
    q, targets, info = cartesian_line(chain, "ee", T_a, T_b, n=17, q0=qa,
                                      iters=200, damping=0.02, tol=1e-9)
    assert q.shape == (2, 17, 6) and targets.shape == (2, 17, 4, 4)
    assert info["final_error"].shape == (2, 17)
    for b in range(2):
        dev = _line_deviation(chain, li, q[b], T_a[b], T_b[b])
        assert float(dev.max()) < 5e-3
    # each batch element matches the same line solved on its own
    q0, _, _ = cartesian_line(chain, "ee", T_a[0], T_b[0], n=17, q0=qa[:1],
                              iters=200, damping=0.02, tol=1e-9)
    assert torch.allclose(q[0], q0, atol=1e-8)


def test_cartesian_line_finds_a_seed_without_q0():
    chain = _six_dof()
    li = chain.link_index["ee"]
    qa = torch.tensor([[0.30, 0.50, -0.80, 0.20, 0.60, 0.10]], dtype=F64)
    qb = torch.tensor([[-0.10, 0.70, -0.60, -0.10, 0.50, 0.30]], dtype=F64)
    T_a = forward_kinematics(chain, qa)[0, li]
    T_b = forward_kinematics(chain, qb)[0, li]
    torch.manual_seed(4)
    q, _, info = cartesian_line(chain, "ee", T_a, T_b, n=13, iters=300,
                                damping=0.02, tol=1e-9, restarts=16)
    assert float(info["max_error"]) < 1e-6
    dev = _line_deviation(chain, li, q, T_a, T_b)
    assert float(dev.max()) < 5e-3


def test_cartesian_line_dtype_follows_q0():
    chain = _six_dof(dtype=torch.float32)
    li = chain.link_index["ee"]
    qa = torch.tensor([[0.30, 0.50, -0.80, 0.20, 0.60, 0.10]], dtype=torch.float32)
    qb = torch.tensor([[-0.40, 0.90, -0.50, -0.30, 0.40, 0.50]], dtype=torch.float32)
    T_a = forward_kinematics(chain, qa)[0, li]
    T_b = forward_kinematics(chain, qb)[0, li]
    torch.manual_seed(5)
    q, targets, info = cartesian_line(chain, "ee", T_a, T_b, n=13, q0=qa,
                                      iters=200, damping=0.02, tol=1e-6)
    assert q.dtype == torch.float32 and targets.dtype == torch.float32
    dev = _line_deviation(chain, li, q, T_a, T_b)
    assert float(dev.max()) < 5e-3


def test_cartesian_line_reports_failure_instead_of_faking_it():
    """A goal outside the workspace must show up in the reported error rather
    than silently returning a pose that is nowhere near the line."""
    chain = _six_dof()
    li = chain.link_index["ee"]
    qa = torch.zeros(1, 6, dtype=F64)
    T_a = forward_kinematics(chain, qa)[0, li]
    T_b = T_a.clone()
    T_b[0, 3] += 5.0                     # 5 m away, far past the 1.1 m reach
    torch.manual_seed(6)
    q, _, info = cartesian_line(chain, "ee", T_a, T_b, n=9, q0=qa, iters=60,
                                restarts=2, tol=1e-9)
    assert float(info["max_error"]) > 1e-2
    assert torch.isfinite(q).all()
