# tests/test_fk.py
import torch, math
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from tests.test_parse import TWO_LINK

def _chain():
    return compile_robot(parse_urdf_string(TWO_LINK))

def test_fk_zero_config_planar_2r():
    # link lengths: j2 origin x=1, so l2 frame at x=1 when q=0
    chain = _chain()
    q = torch.zeros(1, 2)
    world = forward_kinematics(chain, q)  # (1, 3, 4, 4)
    l2 = chain.link_index["l2"]
    pos = world[0, l2, :3, 3]
    assert torch.allclose(pos, torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)

def test_fk_first_joint_90deg():
    # rotate j1 by +90deg about z: l2 origin (was +x=1) moves to +y=1
    chain = _chain()
    q = torch.tensor([[math.pi / 2, 0.0]])
    world = forward_kinematics(chain, q)
    l2 = chain.link_index["l2"]
    pos = world[0, l2, :3, 3]
    assert torch.allclose(pos, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)

def test_fk_batched():
    chain = _chain()
    q = torch.zeros(2048, 2)
    world = forward_kinematics(chain, q)
    assert world.shape == (2048, 3, 4, 4)

def test_fk_differentiable():
    chain = _chain()
    q = torch.zeros(4, 2, requires_grad=True)
    world = forward_kinematics(chain, q)
    world[:, chain.link_index["l2"], :3, 3].sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
