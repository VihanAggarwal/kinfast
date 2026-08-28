# tests/test_api_wiring.py
"""Every convenience method on Robot, called once on a real model.

The methods are thin forwards to modules that have their own tests, so what is
checked here is the wiring: the right link index gets passed, the shapes come
back as documented, and nothing was exported under a name that does not exist.
"""
import pytest
import torch

import kinfast
from tests.test_spatial import SIX_DOF
from tests.test_dynamics import DYN_ARM


@pytest.fixture
def robot():
    return kinfast.load_string(SIX_DOF)


@pytest.fixture
def massive():
    return kinfast.load_string(DYN_ARM)


def test_velocity_methods(robot):
    q = robot.random_configs(3)
    qd = torch.randn(3, robot.dof)
    qdd = torch.randn(3, robot.dof)
    assert robot.twist(q, qd).shape == (3, 6)
    assert robot.acceleration(q, qd, qdd).shape == (3, 6)
    # a named link is routed to the same place as the index
    from kinfast.velocity import twist
    assert torch.allclose(robot.twist(q, qd, link="l3"),
                          twist(robot.chain, q, qd, robot.link_id("l3")))


def test_mass_distribution_methods(massive):
    q = massive.random_configs(4)
    assert massive.com(q).shape == (4, 3)
    assert massive.com_jacobian(q).shape == (4, 3, massive.dof)
    assert float(massive.total_mass) > 0


def test_analysis_methods(robot):
    q = robot.random_configs(2)
    assert robot.manipulability(q).shape == (2,)
    ell = robot.manipulability_ellipsoid(q)
    assert isinstance(ell, dict) and ell
    ws = robot.workspace(n=500)
    assert ws["points"].shape == (500, 3)


def test_reachability_method(robot):
    m = robot.reachability(n=800, voxel=0.15)
    assert m is not None


def test_fk_links_matches_full_fk(robot):
    q = robot.random_configs(3)
    want = ["l2", "ee"]
    got = robot.fk_links(q, want)
    full = robot.fk_all(q)
    assert got.shape == (3, 2, 4, 4)
    for k, name in enumerate(want):
        assert torch.allclose(got[:, k], full[:, robot.link_id(name)], atol=1e-6)


def test_reports(robot):
    text = robot.summary()
    assert isinstance(text, str) or hasattr(text, "to_markdown")
    report = robot.lint()
    assert report is not None


def test_planning_method(robot):
    from kinfast.planning import CollisionChecker
    spheres = robot.sphere_model({"ee": [(0.0, 0.0, 0.0, 0.03)]})
    checker = CollisionChecker(robot, spheres, world=None, self_collision=False)
    q0 = torch.zeros(robot.dof)
    qf = robot.random_configs(1)[0]
    plan = robot.plan(q0, qf, checker, seed=0)
    assert plan.solved
    assert torch.allclose(plan.path[0], q0, atol=1e-6)


def test_floating_method(robot):
    free = robot.floating()
    assert free is not None


def test_auto_sphere_model_needs_the_ir(robot):
    model = robot.auto_sphere_model()      # SIX_DOF has no collision geometry
    assert getattr(model, "n", 0) == 0
    bare = kinfast.Robot(robot.chain)      # built without an IR
    with pytest.raises(ValueError, match="IR"):
        bare.auto_sphere_model()


def test_parse_notes_is_defined_once(robot):
    """Two patches each added this property; only one may survive."""
    import inspect
    src = inspect.getsource(kinfast.Robot)
    assert src.count("def parse_notes") == 1
    assert robot.parse_notes == []


def test_time_path_method(robot):
    """Robot.time_path forwards to the timing module and respects the limits."""
    torch.manual_seed(0)
    path = robot.random_configs(6)
    t, q, qd, qdd, info = robot.time_path(path)
    vmax = robot.chain.vmax.clone()
    vmax[vmax <= 0] = 1.0
    assert bool((qd.abs() <= vmax.unsqueeze(0) * 1.001 + 1e-6).all())
    assert torch.allclose(q[0], path[0], atol=1e-6)
    assert torch.allclose(q[-1], path[-1], atol=1e-6)
    assert info["duration"] > 0
