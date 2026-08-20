# tests/test_robot_api.py
import torch
import kinfast
from tests.test_parse import TWO_LINK

def test_load_and_fk_via_api(tmp_path):
    p = tmp_path / "r.urdf"
    p.write_text(TWO_LINK)
    robot = kinfast.load(str(p))
    assert robot.dof == 2
    q = robot.random_configs(16)
    assert q.shape == (16, 2)
    # random configs respect limits
    assert (q >= robot.lower).all() and (q <= robot.upper).all()
    ee = robot.fk(q)
    assert ee.shape == (16, 4, 4)

def test_fk_named_link():
    robot = kinfast.load_string(TWO_LINK)
    q = robot.random_configs(4)
    frames = robot.fk_all(q)
    assert frames.shape == (4, robot.n_links, 4, 4)
