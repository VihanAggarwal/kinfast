# tests/test_jacobian.py
import torch
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast import transforms as T
from kinfast.jacobian import jacobian
from tests.test_parse import TWO_LINK

def _chain():
    return compile_robot(parse_urdf_string(TWO_LINK))

def test_jacobian_shape():
    chain = _chain()
    q = torch.zeros(8, 2)
    J = jacobian(chain, q, chain.link_index["l2"])
    assert J.shape == (8, 6, 2)

def test_jacobian_matches_finite_difference():
    chain = _chain()
    li = chain.link_index["l2"]
    q = torch.tensor([[0.3, -0.4]])
    J = jacobian(chain, q, li)  # (1,6,2)
    eps = 1e-5
    T0 = forward_kinematics(chain, q)[:, li]
    for k in range(2):
        dq = torch.zeros_like(q); dq[0, k] = eps
        T1 = forward_kinematics(chain, q + dq)[:, li]
        # linear velocity columns via finite difference of position
        dp = (T1[:, :3, 3] - T0[:, :3, 3]) / eps
        assert torch.allclose(J[0, :3, k], dp[0], atol=1e-3)
