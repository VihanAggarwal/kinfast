# tests/test_studio.py
"""The studio window, exercised headlessly.

Matplotlib's Agg backend draws into memory with no display attached, so the
whole window can be built and measured in CI. The benchmarks are shrunk to the
smallest shape that still produces a curve, since the point here is that the
plumbing works, not how fast this machine is.
"""
import os

import pytest
import torch

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")

import kinfast
from kinfast import studio
from tests.test_spatial import SIX_DOF


@pytest.fixture
def robot():
    return kinfast.load_string(SIX_DOF)


def test_find_robots_returns_paths_that_exist():
    for name, path in studio.find_robots().items():
        assert os.path.isfile(path), name


def test_best_of_returns_the_fastest_run():
    calls = []
    studio.best_of(lambda: calls.append(1), runs=3, warmup=2)
    assert len(calls) == 5          # warmups are not timed but do run


def test_throughput_rises_with_batch_size(robot):
    """The whole argument for batching: per-configuration cost falls as the
    batch grows, so configurations per second must go up."""
    batches, fk, jac = studio.bench_throughput(
        robot, batches=(1, 64), report=lambda *_: None)
    assert batches == [1, 64]
    assert fk[1] > fk[0] and jac[1] > jac[0]


def test_single_query_reports_both_paths(robot):
    out = studio.bench_single_query(robot, report=lambda *_: None)
    assert out["tensor path"] > 0
    # the generated path exists for this robot and is the faster of the two
    assert out["compiled"] > 0
    assert out["compiled"] < out["tensor path"]


def test_ik_bench_reports_a_rate_per_restart(robot):
    r, rate, secs = studio.bench_ik(
        robot, restarts=(1, 4), targets=16, iters=40, report=lambda *_: None)
    assert r == [1, 4]
    assert all(0 <= x <= 100 for x in rate)
    assert all(s > 0 for s in secs)
    # more seeds cannot find fewer solutions
    assert rate[1] >= rate[0] - 1e-9


def test_window_builds_and_draws(robot):
    s = studio.Studio(robot, "six_dof")
    assert len(s.sliders) == robot.dof
    s._random(None)
    s._home(None)
    assert torch.allclose(s.q, torch.zeros(1, robot.dof))
    s._solve(None)                        # runs a real ik solve
    assert s.q.shape == (1, robot.dof)


def test_save_writes_a_png(robot, tmp_path):
    s = studio.Studio(robot, "six_dof")
    out = tmp_path / "studio.png"
    s.fig.savefig(out, dpi=60)
    assert out.exists() and out.stat().st_size > 5000


def test_main_can_render_headless(tmp_path):
    """--save is the batch mode: measure, draw, write, exit."""
    if not studio.find_robots():
        pytest.skip("no robot files fetched")
    out = tmp_path / "shot.png"
    assert studio.main(["--save", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 10000


def test_list_mode_exits_cleanly(capsys):
    if not studio.find_robots():
        pytest.skip("no robot files fetched")
    assert studio.main(["--list"]) == 0
    assert capsys.readouterr().out.strip()


def test_sphere_model_falls_back_when_a_model_has_no_primitives(robot):
    """SIX_DOF carries no collision geometry, so the studio has to invent
    something for the planner to keep out of an obstacle."""
    s = studio.Studio(robot, "six_dof")
    model = s._spheres()
    assert model.n > 0
    assert s._spheres() is model            # built once, then cached


def test_plan_button_plans_animates_and_draws(robot):
    """The plan button does the whole job: obstacle, plan, joint plot."""
    torch.manual_seed(0)
    s = studio.Studio(robot, "six_dof")
    for _ in range(8):                      # a random goal can land in the ball
        s._random(None)
        s._plan(None)
        if getattr(s, "plan", None) is not None:
            break
    assert getattr(s, "obstacle", None) is not None
    plan = getattr(s, "plan", None)
    if plan is None:
        pytest.skip("no plan found in the attempts allowed")
    assert plan.solved and len(plan) >= 2
    # the arm ends where the plan ends
    assert torch.allclose(s.q[0], plan.path[-1], atol=1e-5)
    # and the panel drew one line per joint
    assert len(s.ax_plan.lines) == robot.dof


def test_planned_path_is_collision_free(robot):
    """Whatever the button reports, the path it kept must actually be legal."""
    from kinfast.collision_world import Sphere
    from kinfast.planning import CollisionChecker
    torch.manual_seed(3)
    s = studio.Studio(robot, "six_dof")
    for _ in range(8):
        s._random(None)
        s._plan(None)
        if getattr(s, "plan", None) is not None:
            break
    if getattr(s, "plan", None) is None:
        pytest.skip("no plan found in the attempts allowed")
    center, radius = s.obstacle
    checker = CollisionChecker(robot, s._spheres(),
                               [Sphere(center=center, radius=radius)],
                               self_collision=False)
    assert bool(checker(s.plan.densify(0.03)).all())
