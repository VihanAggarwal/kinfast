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


# Regression: jacobian used to crash on a q whose dtype differed from the
# compiled chain (the Robot default is float32, so a float64 q blew up in
# `wR[i] @ axes[i]` with a cryptic addmv dtype error) while fk accepted it.
def test_jacobian_accepts_q_dtype_different_from_chain():
    q64 = torch.tensor([[0.3, -0.4], [1.1, 0.7]], dtype=torch.float64)
    q32 = q64.to(torch.float32)
    chain32 = compile_robot(parse_urdf_string(TWO_LINK), dtype=torch.float32)
    chain64 = compile_robot(parse_urdf_string(TWO_LINK), dtype=torch.float64)
    li = chain32.link_index["l2"]
    ref = jacobian(chain64, q64, li)                 # matched dtypes, float64

    J = jacobian(chain32, q64, li)                   # float64 q on float32 chain
    assert J.dtype == torch.float64
    assert torch.allclose(J, ref, atol=1e-6)

    J = jacobian(chain64, q32, li)                   # float32 q on float64 chain
    assert J.dtype == torch.float32
    assert torch.allclose(J.to(torch.float64), ref, atol=1e-5)


def test_robot_jacobian_and_ik_accept_float64_q():
    # The user-facing path: kinfast.load_string compiles in float32 by default.
    import kinfast
    from kinfast.ik import ik
    r = kinfast.load_string(TWO_LINK)
    assert r.chain.joint_axis.dtype == torch.float32
    q = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
    li = r.chain.link_index["l2"]
    J = r.jacobian(q, "l2")
    assert J.dtype == torch.float64
    chain64 = compile_robot(parse_urdf_string(TWO_LINK), dtype=torch.float64)
    assert torch.allclose(J, jacobian(chain64, q, li), atol=1e-6)
    # ik calls jacobian_rp every iteration; it used to crash on the first one.
    target = forward_kinematics(r.chain, q)[:, li]
    q_sol, info = ik(r.chain, target, q0=q + 0.05, link_index=li,
                     iters=50, pos_only=True)
    assert q_sol.dtype == torch.float64
    assert float(info["final_error"].max()) < 1e-4


# Regression: jacobian(chain, q, -1) returned an all-zero (B,6,dof) tensor
# without complaint because the root walk `while i >= 0` exited at once for a
# negative index. Negatives now resolve like forward_kinematics(...)[:, -1],
# and out-of-range indices raise an IndexError that names the problem.
def test_jacobian_negative_index_means_last_link():
    import pytest
    chain = compile_robot(parse_urdf_string(TWO_LINK), dtype=torch.float64)
    n = chain.n_links
    q = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
    J_neg = jacobian(chain, q, -1)
    J_last = jacobian(chain, q, n - 1)
    assert torch.equal(J_neg, J_last)
    assert J_neg.abs().sum() > 0
    # Independent check: central differences of the last link's position.
    eps = 1e-6
    for k in range(chain.dof):
        dq = torch.zeros_like(q); dq[0, k] = eps
        Tp = forward_kinematics(chain, q + dq)[:, -1, :3, 3]
        Tm = forward_kinematics(chain, q - dq)[:, -1, :3, 3]
        assert torch.allclose(J_neg[0, :3, k], ((Tp - Tm) / (2 * eps))[0], atol=1e-6)
    # -n is the root (index 0); beyond that, and at n, is an error.
    assert torch.equal(jacobian(chain, q, -n), jacobian(chain, q, 0))
    for bad in (n, -n - 1, 10 * n):
        with pytest.raises(IndexError, match="link_index"):
            jacobian(chain, q, bad)
