# tests/test_ik.py
import torch
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.ik import ik
from tests.test_parse import TWO_LINK

def _chain():
    return compile_robot(parse_urdf_string(TWO_LINK))

def test_ik_position_roundtrip_planar():
    chain = _chain()
    li = chain.link_index["l2"]
    torch.manual_seed(0)
    q_true = (chain.lower + (chain.upper - chain.lower) * torch.rand(64, 2))
    target = forward_kinematics(chain, q_true)[:, li]     # reachable targets
    q0 = torch.zeros(64, 2)
    q_sol, info = ik(chain, target, q0, li, iters=200, damping=0.05,
                     pos_only=True)
    ee = forward_kinematics(chain, q_sol)[:, li]
    pos_err = (ee[:, :3, 3] - target[:, :3, 3]).norm(dim=-1)
    # planar 2R: at least most targets converge tightly on position
    assert (pos_err < 1e-2).float().mean() > 0.9

def test_ik_reports_convergence_and_shapes():
    chain = _chain()
    li = chain.link_index["l2"]
    target = forward_kinematics(chain, torch.zeros(10, 2))[:, li]
    q_sol, info = ik(chain, target, torch.zeros(10, 2), li, iters=50)
    assert q_sol.shape == (10, 2)
    assert "iters" in info and "final_error" in info
    assert info["final_error"].shape == (10,)

def test_ik_respects_limits():
    chain = _chain()
    li = chain.link_index["l2"]
    target = forward_kinematics(chain, torch.zeros(8, 2))[:, li]
    q_sol, _ = ik(chain, target, torch.zeros(8, 2), li, iters=100)
    assert (q_sol >= chain.lower - 1e-6).all()
    assert (q_sol <= chain.upper + 1e-6).all()
