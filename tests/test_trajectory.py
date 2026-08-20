# tests/test_trajectory.py
"""Trajectory profiles validated against exact hand-computed oracles and
finite-difference consistency (velocity really is the derivative of position)."""
import math
import torch
import kinfast
from kinfast.trajectory import quintic, trapezoidal
from tests.test_parse import TWO_LINK


def test_quintic_boundary_conditions():
    q0 = torch.tensor([0.0, -1.0], dtype=torch.float64)
    qf = torch.tensor([1.5, 2.0], dtype=torch.float64)
    t, q, qd, qdd = quintic(q0, qf, T=2.0, n=201)
    assert torch.allclose(q[0], q0, atol=1e-12)
    assert torch.allclose(q[-1], qf, atol=1e-12)
    for end in (0, -1):
        assert torch.allclose(qd[end], torch.zeros(2, dtype=torch.float64), atol=1e-9)
        assert torch.allclose(qdd[end], torch.zeros(2, dtype=torch.float64), atol=1e-9)


def test_quintic_velocity_is_position_derivative():
    q0 = torch.tensor([0.0], dtype=torch.float64)
    qf = torch.tensor([3.0], dtype=torch.float64)
    t, q, qd, _ = quintic(q0, qf, T=1.7, n=4001)
    dt = float(t[1] - t[0])
    qd_fd = (q[2:] - q[:-2]) / (2 * dt)          # central difference
    assert torch.allclose(qd[1:-1], qd_fd, atol=1e-5)


def test_trapezoid_hand_computed_duration():
    # d=1, vmax=1, amax=1: exactly the trapezoid boundary case, T = d/v + v/a = 2.
    q0 = torch.tensor([0.0], dtype=torch.float64)
    qf = torch.tensor([1.0], dtype=torch.float64)
    one = torch.tensor([1.0], dtype=torch.float64)
    t, q, qd, qdd, T = trapezoidal(q0, qf, one, one, n=401)
    assert abs(T - 2.0) < 1e-9
    assert torch.allclose(q[-1], qf, atol=1e-9)


def test_triangular_case_duration_and_peak():
    # d=0.25, vmax=1, amax=1: triangular; T = 2*sqrt(d/a) = 1, peak v = 0.5.
    q0 = torch.tensor([0.0], dtype=torch.float64)
    qf = torch.tensor([0.25], dtype=torch.float64)
    one = torch.tensor([1.0], dtype=torch.float64)
    t, q, qd, qdd, T = trapezoidal(q0, qf, one, one, n=401)
    assert abs(T - 1.0) < 1e-9
    assert abs(qd.max().item() - 0.5) < 1e-2
    assert torch.allclose(q[-1], qf, atol=1e-9)


def test_trapezoid_respects_limits_and_synchronizes():
    q0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    qf = torch.tensor([2.0, -0.3, 1.0], dtype=torch.float64)   # third joint: no move
    vmax = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)
    amax = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float64)
    t, q, qd, qdd, T = trapezoidal(q0, qf, vmax, amax, n=801)
    assert torch.allclose(q[0], q0, atol=1e-9)
    assert torch.allclose(q[-1], qf, atol=1e-8)                # all arrive together
    assert (qd.abs() <= vmax.unsqueeze(0) + 1e-8).all()        # velocity limits
    assert (qdd.abs() <= amax.unsqueeze(0) + 1e-8).all()       # acceleration limits
    assert torch.allclose(q[:, 2], torch.ones(801, dtype=torch.float64), atol=1e-12)


def test_trapezoid_velocity_is_position_derivative():
    q0 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    qf = torch.tensor([1.3, -0.7], dtype=torch.float64)
    vmax = torch.tensor([0.8, 1.1], dtype=torch.float64)
    amax = torch.tensor([2.0, 3.0], dtype=torch.float64)
    t, q, qd, _, T = trapezoidal(q0, qf, vmax, amax, n=8001)
    dt = float(t[1] - t[0])
    qd_fd = (q[2:] - q[:-2]) / (2 * dt)
    assert torch.allclose(qd[1:-1], qd_fd, atol=1e-3)


def test_robot_point_to_point_uses_urdf_limits():
    robot = kinfast.load_string(TWO_LINK)                      # velocity="2.6"
    q0 = torch.zeros(2)
    qf = torch.tensor([2.0, -1.5])
    t, q, qd, qdd, T = robot.point_to_point(q0, qf, n=501)
    assert torch.allclose(q[-1], qf, atol=1e-4)
    assert (qd.abs() <= 2.6 + 1e-5).all()                      # URDF vmax honored
