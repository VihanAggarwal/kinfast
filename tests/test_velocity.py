# tests/test_velocity.py
"""Velocities and accelerations checked against finite differences of the
positions they are supposed to be the derivatives of.

Everything runs in float64 with central differences, so the oracle is the
definition of a derivative rather than another part of the library.
"""
import math

import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.urdf.parse import parse_urdf_string
from kinfast import transforms as T
from kinfast.velocity import acceleration, jacobian_dot_qd, link_velocities, twist
from tests.test_spatial import SIX_DOF


@pytest.fixture
def chain():
    return compile_robot(parse_urdf_string(SIX_DOF), dtype=torch.float64)


def _pose(chain, q, li):
    w = forward_kinematics(chain, q)[:, li]
    return w[:, :3, 3], w[:, :3, :3]


def test_twist_matches_finite_differences(chain):
    """J qd is the derivative of the pose along qd, linear and angular both."""
    li = chain.link_index["ee"]
    torch.manual_seed(0)
    q = torch.rand(4, chain.dof, dtype=torch.float64) - 0.5
    qd = torch.randn(4, chain.dof, dtype=torch.float64)
    v = twist(chain, q, qd, li)

    h = 1e-6
    p_p, R_p = _pose(chain, q + h * qd, li)
    p_m, R_m = _pose(chain, q - h * qd, li)
    lin = (p_p - p_m) / (2 * h)
    ang = T.so3_log(R_p @ R_m.transpose(-1, -2)) / (2 * h)
    assert torch.allclose(v[:, :3], lin, atol=1e-6)
    assert torch.allclose(v[:, 3:], ang, atol=1e-6)


def test_acceleration_matches_second_differences(chain):
    """With qdd zero the whole acceleration is the Jdot qd term, and that is
    the second derivative of position along a constant joint velocity."""
    li = chain.link_index["ee"]
    torch.manual_seed(1)
    q = torch.rand(3, chain.dof, dtype=torch.float64) - 0.5
    qd = torch.randn(3, chain.dof, dtype=torch.float64)
    zero = torch.zeros_like(qd)
    a = acceleration(chain, q, qd, zero, li)

    h = 1e-4
    p_p, _ = _pose(chain, q + h * qd, li)
    p_0, _ = _pose(chain, q, li)
    p_m, _ = _pose(chain, q - h * qd, li)
    second = (p_p - 2 * p_0 + p_m) / (h * h)
    assert torch.allclose(a[:, :3], second, atol=1e-4)


def test_acceleration_is_linear_in_qdd(chain):
    """J qdd enters linearly, so doubling qdd doubles that half of the answer."""
    li = chain.link_index["ee"]
    torch.manual_seed(2)
    q = torch.rand(2, chain.dof, dtype=torch.float64)
    qd = torch.randn(2, chain.dof, dtype=torch.float64)
    qdd = torch.randn(2, chain.dof, dtype=torch.float64)
    base = acceleration(chain, q, qd, torch.zeros_like(qdd), li)
    one = acceleration(chain, q, qd, qdd, li)
    two = acceleration(chain, q, qd, 2 * qdd, li)
    assert torch.allclose(two - base, 2 * (one - base), atol=1e-9)


def test_centripetal_term_is_not_zero_at_constant_velocity(chain):
    """A rotating link accelerates even when no joint is accelerating. If this
    ever comes back zero the Jdot term has been dropped."""
    li = chain.link_index["ee"]
    q = torch.zeros(1, chain.dof, dtype=torch.float64)
    qd = torch.ones(1, chain.dof, dtype=torch.float64)
    a = acceleration(chain, q, qd, torch.zeros_like(qd), li)
    assert float(a[:, :3].norm()) > 1e-3


def test_zero_velocity_gives_zero_twist(chain):
    li = chain.link_index["ee"]
    q = torch.rand(2, chain.dof, dtype=torch.float64)
    z = torch.zeros(2, chain.dof, dtype=torch.float64)
    assert torch.allclose(twist(chain, q, z, li), torch.zeros(2, 6, dtype=torch.float64))
    assert torch.allclose(jacobian_dot_qd(chain, q, z, li),
                          torch.zeros(2, 6, dtype=torch.float64))


def test_batch_matches_one_at_a_time(chain):
    li = chain.link_index["ee"]
    torch.manual_seed(3)
    q = torch.rand(5, chain.dof, dtype=torch.float64)
    qd = torch.randn(5, chain.dof, dtype=torch.float64)
    qdd = torch.randn(5, chain.dof, dtype=torch.float64)
    batched = acceleration(chain, q, qd, qdd, li)
    for i in range(5):
        one = acceleration(chain, q[i:i+1], qd[i:i+1], qdd[i:i+1], li)
        assert torch.allclose(batched[i], one[0], atol=1e-10)


def test_link_velocities_covers_the_tree(chain):
    li = chain.link_index["ee"]
    torch.manual_seed(4)
    q = torch.rand(3, chain.dof, dtype=torch.float64)
    qd = torch.randn(3, chain.dof, dtype=torch.float64)
    all_v = link_velocities(chain, q, qd)
    assert all_v.shape == (3, chain.n_links, 6)
    assert torch.allclose(all_v[:, li], twist(chain, q, qd, li))
    # the root cannot move
    root = chain.topo_order[0]
    assert torch.allclose(all_v[:, root], torch.zeros(3, 6, dtype=torch.float64))


def test_shape_mismatch_is_rejected(chain):
    li = chain.link_index["ee"]
    q = torch.zeros(2, chain.dof, dtype=torch.float64)
    with pytest.raises(ValueError, match="same shape"):
        twist(chain, q, torch.zeros(2, chain.dof + 1, dtype=torch.float64), li)
    with pytest.raises(ValueError, match="qdd"):
        acceleration(chain, q, torch.zeros_like(q),
                     torch.zeros(3, chain.dof, dtype=torch.float64), li)


def test_float32_chain_accepts_float64_q(chain):
    """The library convention: the dtype of q decides the working dtype."""
    f32 = compile_robot(parse_urdf_string(SIX_DOF))
    li = f32.link_index["ee"]
    q = torch.rand(2, f32.dof, dtype=torch.float64)
    qd = torch.randn(2, f32.dof, dtype=torch.float64)
    v = twist(f32, q, qd, li)
    assert v.dtype == torch.float64
