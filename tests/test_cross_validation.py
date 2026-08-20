# tests/test_cross_validation.py
"""Cross-validation against an INDEPENDENT implementation (pytorch_kinematics)
on a real robot (Franka Panda). This is the credibility test: two codebases with
no shared code must agree to float32 precision on the same URDF.

Skips cleanly when the asset or the oracle library is absent.
"""
import os
import pytest
import torch

PANDA = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "panda.urdf")
pk = pytest.importorskip("pytorch_kinematics")
pytestmark = pytest.mark.skipif(not os.path.exists(PANDA),
                                reason="panda.urdf not present (tests/fixtures/download_panda.py)")


def _setup():
    import kinfast
    robot = kinfast.load(PANDA)
    with open(PANDA, "rb") as f:
        chain = pk.build_serial_chain_from_urdf(f.read(), "panda_hand")
    pk_names = chain.get_joint_parameter_names()
    idx = [robot.q_index(n) for n in pk_names]
    return robot, chain, pk_names, idx


def test_panda_fk_matches_pytorch_kinematics():
    robot, chain, pk_names, idx = _setup()
    torch.manual_seed(0)
    n = 256
    lo, hi = robot.lower[idx], robot.upper[idx]
    q7 = lo + (hi - lo) * torch.rand(n, len(pk_names))
    qfull = torch.zeros(n, robot.dof)
    qfull[:, idx] = q7

    ours = robot.fk(qfull, link="panda_hand")
    theirs = chain.forward_kinematics(q7, end_only=True).get_matrix()
    dp = (ours[:, :3, 3] - theirs[:, :3, 3]).norm(dim=-1)
    dr = (ours[:, :3, :3] - theirs[:, :3, :3]).abs().amax(dim=(-1, -2))
    assert dp.max() < 1e-5    # float32 accumulation across 8 joints
    assert dr.max() < 1e-5


def test_panda_ik_solves_real_robot():
    robot, chain, pk_names, idx = _setup()
    torch.manual_seed(1)
    n = 200
    lo, hi = robot.lower[idx], robot.upper[idx]
    q7 = lo + (hi - lo) * torch.rand(n, len(pk_names))
    qfull = torch.zeros(n, robot.dof)
    qfull[:, idx] = q7
    target = robot.fk(qfull, link="panda_hand")

    q_sol, info = robot.ik(target, link="panda_hand", iters=100,
                           pos_only=True, restarts=8)
    err = (robot.fk(q_sol, link="panda_hand")[:, :3, 3] - target[:, :3, 3]).norm(dim=-1)
    assert (err < 5e-2).float().mean() > 0.9   # 90%+ of real-robot targets within 5cm
