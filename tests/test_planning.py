# tests/test_planning.py
"""Path planning, checked against situations with a known answer.

The fixtures are built so the right answer is obvious by construction: a wall
across the middle of a planar arm's workspace has to be gone around, a free
world has to be crossed in a straight line, and a goal buried inside an
obstacle has to be refused rather than approximated.
"""
import math

import pytest
import torch

import kinfast
from kinfast import planning
from kinfast.collision_world import Sphere
from kinfast.planning import CollisionChecker, rrt_connect, shortcut

# a planar 2R arm with unit links, so a configuration is easy to reason about
PLANAR = """
<robot name="p2r">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="ee"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2" effort="10"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2" effort="10"/></joint>
  <joint name="jf" type="fixed"><parent link="l2"/><child link="ee"/>
    <origin xyz="1 0 0"/></joint>
</robot>
"""


@pytest.fixture
def robot():
    return kinfast.load_string(PLANAR)


@pytest.fixture
def spheres(robot):
    # one sphere at the elbow and one at the tool, which is all the geometry
    # this arm needs for a planner to have something to avoid
    return robot.sphere_model({"l2": [(0.0, 0.0, 0.0, 0.08)],
                               "ee": [(0.0, 0.0, 0.0, 0.08)]})


def test_free_world_takes_the_straight_line(robot, spheres):
    """With nothing in the way the planner should not invent waypoints."""
    checker = CollisionChecker(robot, spheres, world=None, self_collision=False)
    start = torch.tensor([0.0, 0.5])
    goal = torch.tensor([1.0, -0.5])
    plan = rrt_connect(robot.chain, start, goal, checker, seed=0)
    assert plan.solved
    assert len(plan) == 2                      # start and goal, nothing between
    assert torch.allclose(plan.path[0], start)
    assert torch.allclose(plan.path[-1], goal)


def test_path_goes_around_an_obstacle(robot, spheres):
    """A ball sitting on the straight line has to be avoided, and every
    configuration on the returned path must be legal."""
    start = torch.tensor([0.4, 0.9])
    goal = torch.tensor([-0.4, -0.9])
    # place the obstacle where the tool passes when interpolating start to goal
    mid = (start + goal) / 2
    hit = robot.fk(mid.unsqueeze(0))[0, :3, 3]
    world = [Sphere(center=hit.tolist(), radius=0.22)]
    checker = CollisionChecker(robot, spheres, world, self_collision=False)

    assert not checker.edge(start, goal)       # the direct move is blocked
    plan = rrt_connect(robot.chain, start, goal, checker, seed=1, max_iters=4000)
    assert plan.solved, plan.stats
    assert len(plan) > 2                       # it had to go around
    dense = plan.densify(0.02)
    assert bool(checker(dense).all())          # every step of it is legal
    assert torch.allclose(plan.path[0], start, atol=1e-6)
    assert torch.allclose(plan.path[-1], goal, atol=1e-6)


def test_start_or_goal_in_collision_is_refused(robot, spheres):
    """A goal inside an obstacle has no path, and the planner says so instead
    of returning the closest thing it found."""
    goal = torch.tensor([0.0, 0.0])
    tool = robot.fk(goal.unsqueeze(0))[0, :3, 3]
    world = [Sphere(center=tool.tolist(), radius=0.4)]
    checker = CollisionChecker(robot, spheres, world, self_collision=False)
    plan = rrt_connect(robot.chain, torch.tensor([1.2, 0.6]), goal, checker,
                       seed=0, max_iters=200)
    assert not plan.solved
    assert "no path" in str(plan.stats)


def test_shortcut_shortens_without_breaking(robot, spheres):
    """Smoothing may only shorten a path, and may never make it illegal."""
    start = torch.tensor([0.6, 1.0])
    goal = torch.tensor([-0.6, -1.0])
    hit = robot.fk(((start + goal) / 2).unsqueeze(0))[0, :3, 3]
    world = [Sphere(center=hit.tolist(), radius=0.2)]
    checker = CollisionChecker(robot, spheres, world, self_collision=False)
    plan = rrt_connect(robot.chain, start, goal, checker, seed=2,
                       max_iters=4000, shortcut_iters=0)
    assert plan.solved
    raw = plan.path
    smoothed = shortcut(robot.chain, raw, checker, iters=200, seed=3)
    len_raw = planning._path_length(robot.chain, raw)
    len_new = planning._path_length(robot.chain, smoothed)
    assert len_new <= len_raw + 1e-9
    assert torch.allclose(smoothed[0], raw[0]) and torch.allclose(smoothed[-1], raw[-1])
    checker2 = CollisionChecker(robot, spheres, world, self_collision=False)
    dense = Plan_densify(robot, smoothed)
    assert bool(checker2(dense).all())


def Plan_densify(robot, path):
    from kinfast.planning import Plan, PlanStats
    p = Plan(path, PlanStats(True, 0, 0, 0.0, 0, 0), robot.chain)
    return p.densify(0.02)


def test_edge_check_is_one_batched_call(robot, spheres):
    """The point of doing this inside kinfast: a segment costs one call, not
    one call per configuration on it."""
    checker = CollisionChecker(robot, spheres, world=None, self_collision=False)
    a, b = torch.tensor([0.0, 0.0]), torch.tensor([1.5, -1.2])
    before = checker.calls
    checker.edge(a, b, resolution=0.01)
    assert checker.calls == before + 1         # one call
    assert checker.configs > 100               # for well over a hundred configs


def test_plan_is_deterministic_for_a_seed(robot, spheres):
    start, goal = torch.tensor([0.5, 1.1]), torch.tensor([-0.5, -1.1])
    hit = robot.fk(((start + goal) / 2).unsqueeze(0))[0, :3, 3]
    world = [Sphere(center=hit.tolist(), radius=0.2)]
    runs = []
    for _ in range(2):
        checker = CollisionChecker(robot, spheres, world, self_collision=False)
        runs.append(rrt_connect(robot.chain, start, goal, checker, seed=7,
                                max_iters=4000).path)
    assert runs[0].shape == runs[1].shape
    assert torch.allclose(runs[0], runs[1])


def test_trajectory_respects_the_velocity_limits(robot, spheres):
    """A path is corners; a trajectory is motion. Timing it must not exceed
    the limits the model declares."""
    checker = CollisionChecker(robot, spheres, world=None, self_collision=False)
    plan = rrt_connect(robot.chain, torch.tensor([0.0, 0.0]),
                       torch.tensor([1.2, -0.8]), checker, seed=0)
    t, q, qd, qdd, T = plan.to_trajectory(robot)
    assert T > 0 and q.shape[-1] == robot.dof
    vmax = robot.chain.vmax.clone()
    vmax[vmax <= 0] = 1.0
    assert bool((qd.abs() <= vmax.unsqueeze(0) + 1e-6).all())
    assert torch.allclose(q[0], plan.path[0], atol=1e-5)
    assert torch.allclose(q[-1], plan.path[-1], atol=1e-5)


def test_stats_read_like_a_report(robot, spheres):
    checker = CollisionChecker(robot, spheres, world=None, self_collision=False)
    plan = rrt_connect(robot.chain, torch.tensor([0.0, 0.2]),
                       torch.tensor([0.9, -0.4]), checker, seed=0)
    text = str(plan.stats)
    assert "path found" in text and "ms" in text
    assert plan.stats.configs_checked >= plan.stats.edge_checks
