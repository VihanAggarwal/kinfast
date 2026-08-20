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
    # Use float64 + central differences: a fair numerical comparison. A float32
    # one-sided difference is itself only accurate to ~1e-2 and cannot validate
    # the analytic Jacobian to tight tolerance (the noise is in the check, not
    # the code).
    chain = compile_robot(parse_urdf_string(TWO_LINK), dtype=torch.float64)
    li = chain.link_index["l2"]
    q = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
    J = jacobian(chain, q, li)  # (1,6,2)
    eps = 1e-6
    for k in range(2):
        dq = torch.zeros_like(q); dq[0, k] = eps
        Tp = forward_kinematics(chain, q + dq)[:, li, :3, 3]
        Tm = forward_kinematics(chain, q - dq)[:, li, :3, 3]
        dp = (Tp - Tm) / (2 * eps)  # central difference of ee position
        assert torch.allclose(J[0, :3, k], dp[0], atol=1e-6)
