# tests/test_floating.py
"""Tests for the floating base.

The oracles here are deliberately outside kinfast wherever that is possible:

- rotation vector to matrix is checked against torch.matrix_exp of the
  cross-product matrix, which is a completely different algorithm,
- every Jacobian column (base translation, base rotation and joints alike) is
  checked against float64 central differences of the forward kinematics, with
  the angular rows read off as vee(dR/dq @ R^T) so the check never calls the
  library's own so3_log,
- one pose is worked out by hand on paper and hard coded.
"""
import math

import pytest
import torch

from kinfast import transforms as T
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.floating import (
    FloatingRobot,
    left_jacobian,
    matrix_to_rotvec,
    rotvec_to_matrix,
    skew,
    wrap_rotvec,
)
from kinfast.jacobian import jacobian as fixed_jacobian
from kinfast.urdf.parse import parse_urdf_string

# A short spatial arm: revolute about z, revolute about y through a tilted
# origin, a prismatic slide, a revolute about x through a doubly rotated
# origin, then a fixed tool. Every joint type and a non-trivial origin rotation
# are exercised, and the whole thing reaches well under a metre so that "out of
# reach" targets are easy to construct.
FLOAT_ARM = """
<robot name="float_arm">
  <link name="base_link"/>
  <link name="l1"/>
  <link name="l2"/>
  <link name="l3"/>
  <link name="l4"/>
  <link name="tool"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9" upper="2.9" effort="10" velocity="2"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.15" rpy="0.3 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="10" velocity="2"/>
  </joint>
  <joint name="j3" type="prismatic">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.3" effort="10" velocity="1"/>
  </joint>
  <joint name="j4" type="revolute">
    <parent link="l3"/><child link="l4"/>
    <origin xyz="0.15 0 0" rpy="0 0.4 0.2"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.0" upper="3.0" effort="10" velocity="2"/>
  </joint>
  <joint name="tip" type="fixed">
    <parent link="l4"/><child link="tool"/>
    <origin xyz="0.05 0 0.02" rpy="0 0 0"/>
  </joint>
</robot>
"""

# A single revolute joint, used for the hand-computed pose.
ONE_JOINT = """
<robot name="one">
  <link name="base_link"/>
  <link name="l1"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.2" upper="3.2" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def chain64():
    return compile_robot(parse_urdf_string(FLOAT_ARM), dtype=torch.float64)


def chain32():
    return compile_robot(parse_urdf_string(FLOAT_ARM), dtype=torch.float32)


def sample_configs(dof, n=6, seed=0, dtype=torch.float64):
    """A deterministic spread of full configurations, chosen to hit the cases
    that break naive implementations: an exactly zero base, a base rotated far
    past the small-angle regime, and a base rotation so tiny that the closed
    form for the left Jacobian would divide by nearly nothing."""
    g = torch.Generator().manual_seed(seed)
    qs = [
        torch.zeros(1, 6 + dof, dtype=dtype),
        torch.cat([
            torch.tensor([[0.4, -0.2, 1.1]], dtype=dtype),
            torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype),
            torch.rand(1, dof, generator=g, dtype=dtype) - 0.5,
        ], dim=1),
        torch.cat([
            torch.tensor([[-1.3, 2.0, 0.7]], dtype=dtype),
            torch.tensor([[1.2, -0.9, 1.7]], dtype=dtype),   # |r| ~ 2.3 rad
            torch.rand(1, dof, generator=g, dtype=dtype) - 0.5,
        ], dim=1),
        torch.cat([
            torch.tensor([[0.05, 0.05, -0.05]], dtype=dtype),
            torch.tensor([[1e-9, -2e-9, 3e-10]], dtype=dtype),  # series branch
            torch.rand(1, dof, generator=g, dtype=dtype) - 0.5,
        ], dim=1),
    ]
    extra = n - len(qs)
    if extra > 0:
        rnd = 2.0 * torch.rand(extra, 6 + dof, generator=g, dtype=dtype) - 1.0
        rnd[:, 3:6] *= 2.0
        qs.append(rnd)
    return torch.cat(qs, dim=0)[:max(n, len(qs))]


def fd_jacobian(fr, q, link, h=1e-6):
    """Central-difference Jacobian of the link pose with respect to q_full.

    The linear rows are the plain difference quotient of the world position.
    The angular rows come from omega_hat = (dR/dq) R^T, whose antisymmetric
    part is read off directly; no logarithm map is involved, so this is an
    oracle that shares no code with the analytic Jacobian.
    """
    B, n = q.shape
    idx = fr.link_id(link)
    R0 = fr.fk(q)[:, idx, :3, :3]
    J = torch.zeros(B, 6, n, dtype=q.dtype)
    for k in range(n):
        d = torch.zeros_like(q)
        d[:, k] = h
        Mp = fr.fk(q + d)[:, idx]
        Mm = fr.fk(q - d)[:, idx]
        J[:, :3, k] = (Mp[:, :3, 3] - Mm[:, :3, 3]) / (2 * h)
        W = ((Mp[:, :3, :3] - Mm[:, :3, :3]) / (2 * h)) @ R0.transpose(-1, -2)
        J[:, 3, k] = 0.5 * (W[:, 2, 1] - W[:, 1, 2])
        J[:, 4, k] = 0.5 * (W[:, 0, 2] - W[:, 2, 0])
        J[:, 5, k] = 0.5 * (W[:, 1, 0] - W[:, 0, 1])
    return J


# --------------------------------------------------------------- rotation math

def test_rotvec_matches_matrix_exponential():
    # torch.matrix_exp is an independent implementation of exp(skew(r)); the
    # closed form in floating.py must agree with it everywhere, including the
    # small-angle branch where the closed form switches to a Taylor series.
    r = torch.tensor([
        [0.0, 0.0, 0.0],
        [1e-12, 0.0, 0.0],
        [1e-5, -2e-5, 3e-5],
        [0.3, -0.7, 1.1],
        [2.0, 2.0, 2.0],
        [0.0, 0.0, math.pi],
    ], dtype=torch.float64)
    got = rotvec_to_matrix(r)
    want = torch.matrix_exp(skew(r))
    assert torch.allclose(got, want, atol=1e-13)


def test_rotvec_is_a_rotation_and_round_trips():
    g = torch.Generator().manual_seed(3)
    r = 2.0 * torch.rand(32, 3, generator=g, dtype=torch.float64) - 1.0
    R = rotvec_to_matrix(r)
    eye = torch.eye(3, dtype=torch.float64).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-13)
    assert torch.allclose(torch.linalg.det(R),
                          torch.ones(32, dtype=torch.float64), atol=1e-13)
    assert torch.allclose(matrix_to_rotvec(R), r, atol=1e-8)


def test_rotvec_gradient_is_finite_at_zero():
    # The reason for the series branch: a floating base at rest sits exactly at
    # r = 0, and a closed form that divides by theta produces nan gradients
    # there, which would poison every backward pass through a resting base.
    r = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    rotvec_to_matrix(r).sum().backward()
    assert torch.isfinite(r.grad).all()


def test_matrix_to_rotvec_gradient_is_finite_at_the_identity():
    # A converged IK loop evaluates the orientation error at exactly the
    # identity. acos has an infinite derivative there, which used to make the
    # whole backward pass nan; atan2 does not.
    R = torch.eye(3, dtype=torch.float64).unsqueeze(0).clone().requires_grad_(True)
    matrix_to_rotvec(R).sum().backward()
    assert torch.isfinite(R.grad).all()


def test_matrix_to_rotvec_handles_half_turns_and_tiny_angles():
    r = torch.tensor([
        [0.0, 0.0, 0.0],
        [1e-9, 0.0, 0.0],
        [1e-3, -2e-3, 5e-4],
        [0.7, -1.3, 0.2],
        [0.0, 0.0, math.pi - 1e-6],
    ], dtype=torch.float64)
    R = torch.matrix_exp(skew(r))       # independent construction
    assert torch.allclose(matrix_to_rotvec(R), r, atol=1e-9)


def test_left_jacobian_against_finite_differences():
    # Definition being checked: d/de exp(skew(r + e*u)) evaluated at e=0, times
    # R^T, is skew(J_l(r) u). The derivative is taken numerically on matrix_exp.
    h = 1e-6
    for r in [torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
              torch.tensor([1e-8, 0.0, -1e-8], dtype=torch.float64),
              torch.tensor([0.4, -1.0, 0.2], dtype=torch.float64),
              torch.tensor([1.2, -0.9, 1.7], dtype=torch.float64)]:
        R0 = torch.matrix_exp(skew(r))
        L = left_jacobian(r.unsqueeze(0))[0]
        for k in range(3):
            d = torch.zeros(3, dtype=torch.float64)
            d[k] = h
            dR = (torch.matrix_exp(skew(r + d))
                  - torch.matrix_exp(skew(r - d))) / (2 * h)
            W = dR @ R0.transpose(-1, -2)
            got = torch.tensor([0.5 * (W[2, 1] - W[1, 2]),
                                0.5 * (W[0, 2] - W[2, 0]),
                                0.5 * (W[1, 0] - W[0, 1])], dtype=torch.float64)
            assert torch.allclose(got, L[:, k], atol=1e-8)


def test_left_jacobian_is_not_identity_when_rotated():
    # Guards the exact mistake the docstring warns about: using the identity for
    # the base-rotation block looks right only at r = 0.
    assert torch.allclose(left_jacobian(torch.zeros(1, 3, dtype=torch.float64)),
                          torch.eye(3, dtype=torch.float64).unsqueeze(0),
                          atol=1e-14)
    L = left_jacobian(torch.tensor([[1.2, -0.9, 1.7]], dtype=torch.float64))
    assert (L - torch.eye(3, dtype=torch.float64)).abs().max() > 0.3


def test_wrap_rotvec_keeps_the_rotation_and_shortens_the_vector():
    r = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.3, -0.2, 0.1],
        [0.0, 0.0, 7.0],
        [3.0, -4.0, 12.0],
    ], dtype=torch.float64)
    w = wrap_rotvec(r)
    assert torch.allclose(rotvec_to_matrix(w), rotvec_to_matrix(r), atol=1e-12)
    assert bool((w.norm(dim=-1) <= math.pi + 1e-12).all())
    # already-short vectors are untouched
    assert torch.allclose(w[:2], r[:2], atol=1e-15)


# ------------------------------------------------------------------------- FK

def test_fk_equals_base_transform_times_fixed_fk():
    chain = chain64()
    fr = FloatingRobot(chain)
    q = sample_configs(chain.dof, n=8, seed=11)
    got = fr.fk(q)
    base = T.make_transform(rotvec_to_matrix(q[:, 3:6]), q[:, :3])
    want = base.unsqueeze(1) @ forward_kinematics(chain, q[:, 6:])
    assert got.shape == (q.shape[0], chain.n_links, 4, 4)
    assert torch.allclose(got, want, atol=1e-12)


def test_fk_with_identity_base_is_the_fixed_base_fk():
    chain = chain64()
    fr = FloatingRobot(chain)
    g = torch.Generator().manual_seed(5)
    qj = torch.rand(4, chain.dof, generator=g, dtype=torch.float64) - 0.5
    q = torch.cat([torch.zeros(4, 6, dtype=torch.float64), qj], dim=1)
    assert torch.allclose(fr.fk(q), forward_kinematics(chain, qj), atol=1e-14)


def test_fk_hand_computed_pose():
    # One revolute z joint whose origin is 0.1 up the z axis, driven to +90
    # degrees, on a base translated to (1, 2, 3) and yawed by +90 degrees.
    # Base rotation Rz(90) maps the joint offset (0, 0, 0.1) to itself, so the
    # link sits at (1, 2, 3.1); the rotations compose to Rz(180).
    chain = compile_robot(parse_urdf_string(ONE_JOINT), dtype=torch.float64)
    fr = FloatingRobot(chain)
    half = math.pi / 2
    q = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, half, half]], dtype=torch.float64)
    M = fr.fk(q)[0, chain.link_index["l1"]]
    want = torch.tensor([
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.1],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    assert torch.allclose(M, want, atol=1e-14)


def test_base_transform_and_link_pose_agree_with_fk():
    chain = chain64()
    fr = FloatingRobot(chain)
    q = sample_configs(chain.dof, n=5, seed=2)
    assert torch.allclose(fr.base_transform(q), fr.fk(q)[:, 0], atol=1e-14)
    assert torch.allclose(fr.link_pose(q, "tool"),
                          fr.fk(q)[:, chain.link_index["tool"]], atol=1e-14)
    assert torch.allclose(fr.link_pose(q, -1), fr.fk(q)[:, -1], atol=1e-14)


def test_fk_rows_are_independent_across_the_batch():
    chain = chain64()
    fr = FloatingRobot(chain)
    q = sample_configs(chain.dof, n=6, seed=7)
    stacked = fr.fk(q)
    for b in range(q.shape[0]):
        assert torch.allclose(fr.fk(q[b:b + 1])[0], stacked[b], atol=1e-14)


def test_split_and_join_round_trip_and_reject_bad_shapes():
    chain = chain64()
    fr = FloatingRobot(chain)
    assert fr.dof == 6 + chain.dof
    assert fr.joint_dof == chain.dof
    q = sample_configs(chain.dof, n=4, seed=1)
    t, r, qj = fr.split(q)
    assert t.shape == (4, 3) and r.shape == (4, 3) and qj.shape == (4, chain.dof)
    assert torch.allclose(fr.join(t, r, qj), q, atol=0)
    with pytest.raises(ValueError):
        fr.split(torch.zeros(2, fr.dof + 1, dtype=torch.float64))
    with pytest.raises(ValueError):
        fr.split(torch.zeros(fr.dof, dtype=torch.float64))
    with pytest.raises(KeyError):
        fr.link_id("no_such_link")
    with pytest.raises(IndexError):
        fr.link_id(chain.n_links)


# ------------------------------------------------------------------- Jacobian

def test_jacobian_every_column_matches_finite_differences():
    chain = chain64()
    fr = FloatingRobot(chain)
    q = sample_configs(chain.dof, n=8, seed=21)
    for name in chain.link_names:
        J = fr.jacobian(q, name)
        assert J.shape == (q.shape[0], 6, 6 + chain.dof)
        num = fd_jacobian(fr, q, name)
        gap = (J - num).abs()
        # report the worst column if this ever fails, so a regression names the
        # block (base translation / base rotation / joints) that broke
        worst = int(gap.amax(dim=(0, 1)).argmax())
        assert gap.max() < 1e-7, (
            f"link {name}: worst column {worst}, error {gap.max().item():.3e}")


def test_jacobian_base_blocks_have_the_expected_structure():
    chain = chain64()
    fr = FloatingRobot(chain)
    q = sample_configs(chain.dof, n=5, seed=31)
    B = q.shape[0]
    idx = chain.link_index["tool"]
    J = fr.jacobian(q, idx)
    eye = torch.eye(3, dtype=torch.float64).expand(B, 3, 3)
    # translating the base moves the link one for one and rotates nothing
    assert torch.allclose(J[:, :3, 0:3], eye, atol=1e-14)
    assert torch.allclose(J[:, 3:, 0:3], torch.zeros(B, 3, 3, dtype=torch.float64),
                          atol=1e-14)
    # the angular part of the base-rotation block is exactly the left Jacobian
    assert torch.allclose(J[:, 3:, 3:6], left_jacobian(q[:, 3:6]), atol=1e-14)
    # and its linear part is the lever arm crossed with that
    lever = fr.fk(q)[:, idx, :3, 3] - q[:, :3]
    assert torch.allclose(J[:, :3, 3:6],
                          -skew(lever) @ left_jacobian(q[:, 3:6]), atol=1e-14)


def test_joint_columns_reduce_to_the_fixed_base_jacobian():
    chain = chain64()
    fr = FloatingRobot(chain)
    g = torch.Generator().manual_seed(13)
    qj = torch.rand(4, chain.dof, generator=g, dtype=torch.float64) - 0.5
    q = torch.cat([torch.zeros(4, 6, dtype=torch.float64), qj], dim=1)
    idx = chain.link_index["tool"]
    assert torch.allclose(fr.jacobian(q, idx)[:, :, 6:],
                          fixed_jacobian(chain, qj, idx), atol=1e-14)


def test_joint_columns_are_the_fixed_jacobian_rotated_by_the_base():
    # With the base rotated, the joint block must be the fixed-base Jacobian
    # with both 3-row halves rotated into the world frame. Base translation
    # cannot matter, because a rigid translation does not change velocities.
    chain = chain64()
    fr = FloatingRobot(chain)
    g = torch.Generator().manual_seed(17)
    qj = torch.rand(3, chain.dof, generator=g, dtype=torch.float64) - 0.5
    r = torch.tensor([[0.4, -1.1, 0.7]], dtype=torch.float64).expand(3, 3)
    t = torch.tensor([[5.0, -3.0, 2.0]], dtype=torch.float64).expand(3, 3)
    q = torch.cat([t, r, qj], dim=1)
    Rb = rotvec_to_matrix(r)
    Jf = fixed_jacobian(chain, qj, chain.link_index["tool"])
    got = fr.jacobian(q, "tool")[:, :, 6:]
    assert torch.allclose(got[:, :3], Rb @ Jf[:, :3], atol=1e-14)
    assert torch.allclose(got[:, 3:], Rb @ Jf[:, 3:], atol=1e-14)


def test_jacobian_of_the_root_link_ignores_the_joints():
    chain = chain64()
    fr = FloatingRobot(chain)
    q = sample_configs(chain.dof, n=4, seed=23)
    J = fr.jacobian(q, 0)
    assert torch.allclose(J[:, :, 6:], torch.zeros_like(J[:, :, 6:]), atol=1e-14)


def test_jacobian_accepts_names_indices_and_negative_indices():
    chain = chain64()
    fr = FloatingRobot(chain)
    q = sample_configs(chain.dof, n=3, seed=29)
    a = fr.jacobian(q, "tool")
    b = fr.jacobian(q, chain.link_index["tool"])
    c = fr.jacobian(q, -1)
    assert torch.allclose(a, b, atol=0) and torch.allclose(a, c, atol=0)


def test_jacobian_agrees_with_autograd():
    # A third opinion on the linear rows: reverse-mode differentiation of the
    # link position through the whole FK stack.
    chain = chain64()
    fr = FloatingRobot(chain)
    q = sample_configs(chain.dof, n=1, seed=41)[2:3].clone().requires_grad_(True)
    idx = chain.link_index["tool"]
    J = fr.jacobian(q, idx)
    for row in range(3):
        if q.grad is not None:
            q.grad = None
        fr.fk(q)[0, idx, row, 3].backward(retain_graph=True)
        assert torch.allclose(q.grad[0], J[0, row], atol=1e-12)


# ---------------------------------------------------------------- dtype/device

def test_dtype_follows_the_configuration():
    chain = chain32()          # compiled in float32 on purpose
    fr = FloatingRobot(chain)
    q32 = sample_configs(chain.dof, n=3, seed=3, dtype=torch.float64).to(torch.float32)
    q64 = q32.to(torch.float64)
    assert fr.fk(q32).dtype == torch.float32
    assert fr.jacobian(q32, "tool").dtype == torch.float32
    # a float64 configuration on a float32 chain works and stays float64
    assert fr.fk(q64).dtype == torch.float64
    assert fr.jacobian(q64, "tool").dtype == torch.float64
    assert torch.allclose(fr.fk(q64).to(torch.float32), fr.fk(q32), atol=1e-5)


def test_float32_jacobian_is_close_to_the_float64_one():
    fr32 = FloatingRobot(chain32())
    fr64 = FloatingRobot(chain64())
    q = sample_configs(fr64.joint_dof, n=6, seed=9)
    a = fr32.jacobian(q.to(torch.float32), "tool").to(torch.float64)
    b = fr64.jacobian(q, "tool")
    assert (a - b).abs().max() < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_runs_on_cuda():
    chain = chain32().to(device="cuda")
    fr = FloatingRobot(chain)
    q = sample_configs(chain.dof, n=4, seed=6).to(torch.float32).cuda()
    assert fr.fk(q).device.type == "cuda"
    assert fr.jacobian(q, "tool").device.type == "cuda"


# ------------------------------------------------------------------------- IK

def _max_fixed_reach(chain, n=4000, seed=0):
    """How far the tool can get from the root link with the base bolted down."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = chain.lower, chain.upper
    qj = lo + (hi - lo) * torch.rand(n, chain.dof, generator=g, dtype=chain.dtype)
    p = forward_kinematics(chain, qj)[:, chain.link_index["tool"], :3, 3]
    return float(p.norm(dim=-1).max())


def test_the_arm_really_cannot_reach_the_far_targets():
    # Establishes the premise of the IK test below rather than assuming it.
    assert _max_fixed_reach(chain64()) < 1.0


def _far_targets(dtype=torch.float64):
    """Poses metres away from the origin, well outside the fixed arm's reach."""
    pos = torch.tensor([
        [3.0, -2.0, 1.5],
        [-4.0, 1.0, -2.0],
        [0.0, 6.0, 0.5],
        [2.5, 2.5, -3.5],
    ], dtype=dtype)
    rot = rotvec_to_matrix(torch.tensor([
        [0.0, 0.0, 0.0],
        [0.6, -0.4, 1.2],
        [2.0, 0.5, -1.0],
        [-1.4, 1.4, 0.3],
    ], dtype=dtype))
    return T.make_transform(rot, pos)


def _pose_gap(fr, q, target, idx):
    got = fr.fk(q)[:, idx]
    dp = (target[:, :3, 3] - got[:, :3, 3]).norm(dim=-1)
    dR = target[:, :3, :3] @ got[:, :3, :3].transpose(-1, -2)
    ang = ((dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2] - 1.0) * 0.5).clamp(-1, 1).acos()
    return dp, ang


def test_ik_reaches_full_poses_outside_the_fixed_arm_reach():
    chain = chain64()
    fr = FloatingRobot(chain)
    idx = chain.link_index["tool"]
    target = _far_targets()
    g = torch.Generator().manual_seed(0)
    q, info = fr.ik(target, "tool", iters=300, restarts=4, tol=1e-10,
                    check_every=25, generator=g)
    assert q.shape == (4, fr.dof)
    dp, ang = _pose_gap(fr, q, target, idx)
    assert float(dp.max()) < 1e-6, dp
    assert float(ang.max()) < 1e-6, ang
    assert float(info["final_error"].max()) < 1e-6


def test_ik_position_only_reaches_far_targets():
    chain = chain64()
    fr = FloatingRobot(chain)
    target = _far_targets()
    g = torch.Generator().manual_seed(1)
    q, _ = fr.ik(target, "tool", pos_only=True, iters=200, tol=1e-10,
                 check_every=25, generator=g)
    dp, _ = _pose_gap(fr, q, target, chain.link_index["tool"])
    assert float(dp.max()) < 1e-6, dp


def test_ik_respects_joint_limits_and_a_base_box():
    chain = chain64()
    fr = FloatingRobot(chain)
    # pin the base to a slab around z = 0 and let only the yaw axis turn
    blo = [-10.0, -10.0, -0.05, 0.0, 0.0, -math.pi]
    bhi = [10.0, 10.0, 0.05, 0.0, 0.0, math.pi]
    target = _far_targets()
    target[:, 2, 3] = 0.4          # keep the targets slab-reachable
    g = torch.Generator().manual_seed(2)
    q, _ = fr.ik(target, "tool", pos_only=True, iters=400, restarts=4,
                 tol=1e-10, check_every=25, base_bounds=(blo, bhi), generator=g)
    lo = torch.tensor(blo, dtype=torch.float64)
    hi = torch.tensor(bhi, dtype=torch.float64)
    assert bool((q[:, :6] >= lo - 1e-12).all()) and bool((q[:, :6] <= hi + 1e-12).all())
    assert bool((q[:, 6:] >= chain.lower - 1e-12).all())
    assert bool((q[:, 6:] <= chain.upper + 1e-12).all())
    dp, _ = _pose_gap(fr, q, target, chain.link_index["tool"])
    assert float(dp.max()) < 1e-5, dp


def test_ik_from_an_explicit_seed_and_argument_checking():
    chain = chain64()
    fr = FloatingRobot(chain)
    target = _far_targets()[:1]
    q0 = torch.zeros(1, fr.dof, dtype=torch.float64)
    q0[0, :3] = target[0, :3, 3]      # start the base at the target
    q, _ = fr.ik(target, "tool", q0=q0, iters=200, tol=1e-10, check_every=25)
    dp, ang = _pose_gap(fr, q, target, chain.link_index["tool"])
    assert float(dp.max()) < 1e-6 and float(ang.max()) < 1e-6
    with pytest.raises(ValueError):
        fr.ik(target, "tool", q0=q0, restarts=3)
    with pytest.raises(ValueError):
        fr.ik(target, "tool", q0=torch.zeros(1, fr.dof + 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        fr.ik(target, "tool", restarts=0)


def test_ik_accepts_a_single_unbatched_target():
    chain = chain64()
    fr = FloatingRobot(chain)
    target = _far_targets()[1]        # (4, 4)
    g = torch.Generator().manual_seed(4)
    q, _ = fr.ik(target, "tool", iters=300, restarts=2, tol=1e-10,
                 check_every=25, generator=g)
    assert q.shape == (1, fr.dof)
    dp, ang = _pose_gap(fr, q, target.unsqueeze(0), chain.link_index["tool"])
    assert float(dp.max()) < 1e-6 and float(ang.max()) < 1e-6


def test_ik_is_deterministic_for_a_fixed_generator_seed():
    fr = FloatingRobot(chain64())
    target = _far_targets()
    runs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(1234)
        q, _ = fr.ik(target, "tool", iters=60, restarts=3, generator=g)
        runs.append(q)
    assert torch.allclose(runs[0], runs[1], atol=0)


def test_ik_is_differentiable_in_the_target():
    # The whole loop is autograd-traceable, so a solved configuration carries a
    # gradient back to the pose that asked for it.
    fr = FloatingRobot(chain64())
    target = _far_targets()[:1].clone().requires_grad_(True)
    q0 = torch.zeros(1, fr.dof, dtype=torch.float64)
    q0[0, :3] = target[0, :3, 3].detach()
    q, _ = fr.ik(target, "tool", q0=q0, iters=30, check_every=100)
    q.sum().backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert float(target.grad.abs().max()) > 0.0


def test_seed_puts_the_base_near_the_target():
    fr = FloatingRobot(chain64())
    target = _far_targets()
    g = torch.Generator().manual_seed(8)
    q = fr.seed(target, generator=g)
    assert q.shape == (4, fr.dof)
    assert float((q[:, :3] - target[:, :3, 3]).abs().max()) <= 0.5 + 1e-12
    assert bool((q[:, 6:] >= fr.chain.lower - 1e-12).all())
    assert bool((q[:, 6:] <= fr.chain.upper + 1e-12).all())


def test_repr_and_shape_accessors():
    fr = FloatingRobot(chain64())
    assert fr.n_links == 6 and fr.joint_dof == 4 and fr.dof == 10
    assert fr.link_names[0] == "base_link"
    assert "dof=10" in repr(fr)
