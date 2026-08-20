# tests/test_robot_integration.py
import os, pytest, torch
import kinfast

PANDA = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "panda.urdf")

pytestmark = pytest.mark.skipif(not os.path.exists(PANDA),
                                reason="panda.urdf not present; see download_panda.py")

def test_panda_loads_and_solves():
    robot = kinfast.load(PANDA)
    assert robot.dof >= 7
    q = robot.random_configs(1000)
    ee = robot.fk(q)
    assert ee.shape == (1000, 4, 4)
    target = robot.fk(robot.random_configs(1000))
    q_sol, info = robot.ik(target, iters=100, pos_only=True)
    pos_err = (robot.fk(q_sol)[:, :3, 3] - target[:, :3, 3]).norm(dim=-1)
    assert (pos_err < 5e-2).float().mean() > 0.7
