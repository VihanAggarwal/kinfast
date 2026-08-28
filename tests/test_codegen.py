# tests/test_codegen.py
"""The scalar compiler validated against the batched torch path (which is itself
cross-validated against pytorch_kinematics on a real Panda, so agreement here is
transitive agreement with an independent library), on every fixture robot AND
every real robot present in the gallery."""
import glob
import math
import os

import numpy as np
import pytest
import torch

import kinfast
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.jacobian import jacobian as torch_jacobian
from tests.test_parse import TWO_LINK
from tests.test_spatial import SIX_DOF
from tests.test_dynamics import DYN_ARM

GALLERY = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..",
                                        "examples", "assets", "gallery", "*.urdf")))


def _agree(chain, n_cfg=32, atol=1e-5):
    from kinfast.codegen import CompiledRobot
    fast = CompiledRobot(chain)
    torch.manual_seed(0)
    if chain.dof:
        q = chain.lower + (chain.upper - chain.lower) * torch.rand(n_cfg, chain.dof)
    else:
        q = torch.zeros(n_cfg, 0)
    ref = forward_kinematics(chain, q)                       # (n_cfg, n, 4, 4)
    for k in range(n_cfg):
        got = fast.fk(q[k].tolist())
        assert np.abs(got - ref[k].numpy()).max() < atol, f"config {k}"


def test_codegen_fk_fixtures():
    for urdf in (TWO_LINK, SIX_DOF, DYN_ARM):
        _agree(compile_robot(parse_urdf_string(urdf)))


@pytest.mark.parametrize("path", GALLERY,
                         ids=[os.path.basename(p) for p in GALLERY])
def test_codegen_fk_real_robots(path):
    robot = kinfast.load(path)
    _agree(robot.chain, n_cfg=16)


def test_codegen_jacobian_matches_torch():
    chain = compile_robot(parse_urdf_string(SIX_DOF))
    from kinfast.codegen import CompiledRobot
    fast = CompiledRobot(chain)
    li = chain.link_index["ee"]
    torch.manual_seed(1)
    q = chain.lower + (chain.upper - chain.lower) * torch.rand(8, chain.dof)
    ref = torch_jacobian(chain, q, li)
    for k in range(8):
        got = fast.jacobian(q[k].tolist(), li)
        assert np.abs(got - ref[k].numpy()).max() < 1e-5


def test_codegen_ik_round_trip():
    chain = compile_robot(parse_urdf_string(SIX_DOF))
    from kinfast.codegen import CompiledRobot
    fast = CompiledRobot(chain)
    li = chain.link_index["ee"]
    torch.manual_seed(2)
    q_true = chain.lower + (chain.upper - chain.lower) * torch.rand(16, chain.dof)
    targets = forward_kinematics(chain, q_true)[:, li, :3, 3]
    solved = 0
    for k in range(16):
        q, err = fast.ik(targets[k].tolist(), li, restarts=12, iters=100)
        solved += err < 5e-2
    assert solved >= 15                     # near-universal on reachable targets


def test_compile_via_robot_api():
    robot = kinfast.load_string(TWO_LINK)
    fast = robot.compile()
    T = fast.fk([0.0, 0.0], link=robot.link_id("l2"))
    assert np.allclose(T[:3, 3], [1.0, 0.0, 0.0], atol=1e-9)
