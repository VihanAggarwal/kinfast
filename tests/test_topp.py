# tests/test_topp.py
"""Timing along a path: the limits it promises, and the method it claims to beat."""
import pytest
import torch

import kinfast
from kinfast.topp import duration, time_parameterize
from kinfast.trajectory import trapezoidal
from tests.test_spatial import SIX_DOF


@pytest.fixture
def robot():
    return kinfast.load_string(SIX_DOF)


def _limits(robot):
    vmax = robot.chain.vmax.clone()
    vmax[vmax <= 0] = 1.0
    return vmax, torch.full_like(vmax, 4.0)


def _trapezoid_total(path, vmax, amax):
    return sum(float(trapezoidal(a, b, vmax, amax, n=32)[4])
               for a, b in zip(path[:-1], path[1:]))


def _curved_path(robot, k, seed=0):
    """A dense sampling of a smooth arc, which is what a planner path looks
    like once it has been densified for collision checking."""
    torch.manual_seed(seed)
    base = robot.random_configs(2)
    s = torch.linspace(0, 1, k).unsqueeze(-1)
    return base[0] + s * (base[1] - base[0]) + 0.25 * torch.sin(
        3 * s * 3.14159) * torch.randn(1, robot.dof)


@pytest.mark.parametrize("blend", [0.0, 0.05])
def test_limits_are_respected(robot, blend):
    """The promise of the module. The acceleration checked is the analytic one,
    which is what the trajectory has, not a difference of the samples."""
    vmax, amax = _limits(robot)
    path = _curved_path(robot, 20, seed=1)
    t, q, qd, qdd, info = time_parameterize(robot.chain, path, vmax, amax,
                                            blend=blend)
    assert bool((qd.abs() <= vmax.unsqueeze(0) * 1.001 + 1e-6).all())
    assert bool((qdd.abs() <= amax.unsqueeze(0) * 1.001 + 1e-6).all())


def test_starts_and_ends_at_rest_on_the_path(robot):
    vmax, amax = _limits(robot)
    path = _curved_path(robot, 8, seed=2)
    t, q, qd, _qdd, _info = time_parameterize(robot.chain, path, vmax, amax)
    assert float(t[0]) == 0.0
    assert bool((t[1:] > t[:-1]).all())
    assert torch.allclose(qd[0], torch.zeros(robot.dof), atol=1e-6)
    assert torch.allclose(qd[-1], torch.zeros(robot.dof), atol=1e-6)
    assert torch.allclose(q[0], path[0], atol=1e-6)
    assert torch.allclose(q[-1], path[-1], atol=1e-6)


def test_beats_stopping_at_every_point_on_a_dense_path(robot):
    """Where the module earns its place. A dense path is mostly not corners, so
    not braking at every sample is worth double digit percentages."""
    vmax, amax = _limits(robot)
    path = _curved_path(robot, 40, seed=3)
    ours = duration(robot.chain, path, vmax, amax)
    theirs = _trapezoid_total(path, vmax, amax)
    assert ours < theirs * 0.95


def test_sparse_path_is_no_worse(robot):
    """With few waypoints there is nothing to win, and the answer should land
    on the simpler method rather than under it."""
    vmax, amax = _limits(robot)
    path = _curved_path(robot, 4, seed=4)
    ours = duration(robot.chain, path, vmax, amax)
    theirs = _trapezoid_total(path, vmax, amax)
    assert ours <= theirs * 1.05


def test_dense_smooth_path_does_not_stop_everywhere(robot):
    """Regression. Testing the turn at every grid sample marked a smooth dense
    arc as one continuous corner and stopped the robot throughout it, turning a
    nine second move into one hundred and twenty six. Corners are read from the
    input waypoints now."""
    vmax, amax = _limits(robot)
    path = _curved_path(robot, 25, seed=0)
    ours = duration(robot.chain, path, vmax, amax)
    theirs = _trapezoid_total(path, vmax, amax)
    assert ours < theirs * 2.0, "the profile is stopping where there is no corner"


def test_a_real_corner_still_forces_a_stop(robot):
    """The other side of that fix: a genuine reversal has to be braked for."""
    vmax, amax = _limits(robot)
    a = torch.zeros(robot.dof)
    b = a.clone()
    b[0] = 1.0
    path = torch.stack([a, b, a])          # out and straight back
    _t, _q, _qd, _qdd, info = time_parameterize(robot.chain, path, vmax, amax)
    assert info["stops"] >= 1


def test_blending_trades_accuracy_for_speed(robot):
    """What blending actually buys, on one right angle.

    The deviation grows with the window, cleanly and monotonically, and that is
    the cost the caller has to weigh against clearance. The time is a different
    story: it goes down, but only by a couple of percent here, and not
    monotonically, because a wider window also moves grid points around. So the
    speed claim tested is the one the measurements support, that blending never
    comes out slower than not blending, rather than a smooth ordering that is
    not there.
    """
    vmax, amax = _limits(robot)
    a = torch.zeros(robot.dof)
    b, c = a.clone(), a.clone()
    b[0], c[0], c[1] = 1.0, 1.0, 1.0
    path = torch.stack([a, b, c])          # one right angle
    _t, _q, _qd, _qdd, exact = time_parameterize(robot.chain, path, vmax, amax)
    assert exact["deviation"] == 0.0       # no blending follows the path exactly

    devs, times = [], []
    for blend in (0.02, 0.05, 0.10, 0.20):
        _t, _q, _qd, _qdd, info = time_parameterize(robot.chain, path, vmax,
                                                    amax, blend=blend)
        devs.append(info["deviation"])
        times.append(info["duration"])
    assert devs == sorted(devs), devs                  # cost rises with window
    assert devs[0] > 0
    assert max(times) <= exact["duration"] * 1.001, times


def test_blend_taper_keeps_the_deviation_small(robot):
    """Regression. Smoothing and then clamping the endpoints back leaves a step
    between the smoothed path and the true endpoint, which is a fresh corner in
    the last interval and grows with the window. It made a wide blend slower
    than a narrow one and inflated the deviation by more than ten times. The
    correction is tapered to zero at the ends instead."""
    vmax, amax = _limits(robot)
    a = torch.zeros(robot.dof)
    b, c = a.clone(), a.clone()
    b[0], c[0], c[1] = 1.0, 1.0, 1.0
    path = torch.stack([a, b, c])
    _t, q, _qd, _qdd, info = time_parameterize(robot.chain, path, vmax, amax,
                                               blend=0.05)
    # the corner is rounded by a little, not by a fifth of the path
    assert info["deviation"] < 0.05
    # and the ends still land exactly where they were asked to
    assert torch.allclose(q[0], path[0], atol=1e-9)
    assert torch.allclose(q[-1], path[-1], atol=1e-9)


def test_tighter_limits_take_longer(robot):
    vmax, amax = _limits(robot)
    path = _curved_path(robot, 10, seed=5)
    quick = duration(robot.chain, path, vmax, amax)
    slow = duration(robot.chain, path, vmax * 0.5, amax * 0.5)
    assert slow > quick


def test_degenerate_paths(robot):
    vmax, amax = _limits(robot)
    one = robot.random_configs(1)
    t, _q, _qd, _qdd, _i = time_parameterize(robot.chain, one, vmax, amax)
    assert len(t) == 1 and float(t[0]) == 0.0
    same = torch.cat([one, one], dim=0)
    t2, _q2, _qd2, _qdd2, _i2 = time_parameterize(robot.chain, same, vmax, amax)
    assert torch.isfinite(t2).all()


def test_rejects_bad_input(robot):
    vmax, amax = _limits(robot)
    with pytest.raises(ValueError, match="path must be"):
        time_parameterize(robot.chain, torch.zeros(robot.dof), vmax, amax)
    with pytest.raises(ValueError, match="positive"):
        time_parameterize(robot.chain, robot.random_configs(3),
                          torch.zeros_like(vmax), amax)


def test_defaults_come_from_the_model(robot):
    path = _curved_path(robot, 6, seed=6)
    t, _q, qd, _qdd, _info = time_parameterize(robot.chain, path)
    vmax = robot.chain.vmax.clone()
    vmax[vmax <= 0] = 1.0
    assert bool((qd.abs() <= vmax.unsqueeze(0) * 1.001 + 1e-6).all())
    assert float(t[-1]) > 0


def test_times_a_planned_path(robot):
    """The intended pairing: a planned path in, a timed trajectory out."""
    from kinfast.planning import CollisionChecker, rrt_connect
    spheres = robot.sphere_model({"ee": [(0.0, 0.0, 0.0, 0.03)]})
    checker = CollisionChecker(robot, spheres, world=None, self_collision=False)
    torch.manual_seed(7)
    plan = rrt_connect(robot.chain, torch.zeros(robot.dof),
                       robot.random_configs(1)[0], checker, seed=0)
    assert plan.solved
    dense = plan.densify(0.05)
    t, q, _qd, _qdd, info = time_parameterize(robot.chain, dense)
    assert torch.allclose(q[0], dense[0], atol=1e-6)
    assert torch.allclose(q[-1], dense[-1], atol=1e-6)
    assert info["duration"] > 0
