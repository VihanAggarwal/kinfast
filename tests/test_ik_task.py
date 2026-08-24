# tests/test_ik_task.py
"""Weighted task-space IK with a nullspace posture term.

Oracles here are independent of the solver being tested: the step math is
checked against the other algebraic form of damped least squares built with an
explicit matrix inverse, the nullspace claim is checked against a float64
central-difference Jacobian, and the solve quality is checked by running FK on
what came back. The only self-comparison is the deliberate one, that
position-only weights reduce exactly to the existing pos_only solver.
"""
import math

import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.ik import ik
from kinfast.ik_task import ik_task, weighted_dls_step
from kinfast.urdf.parse import parse_urdf_string
from kinfast import transforms as T
from tests.test_spatial import SIX_DOF

# 7 revolute joints reaching a 6D task: one full dimension of self-motion, so
# the nullspace term has somewhere to go on a full-pose solve.
SEVEN_DOF = """
<robot name="arm7">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/>
  <link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" velocity="2" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2" effort="50"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" velocity="2" effort="50"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2" effort="50"/></joint>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" velocity="2" effort="50"/></joint>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2" effort="50"/></joint>
  <joint name="j7" type="revolute"><parent link="l6"/><child link="tool"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" velocity="2" effort="50"/></joint>
</robot>
"""


# 3 joints against a 6D task: nothing decouples, so position and orientation
# genuinely compete and the weights have to decide the outcome. The 6-DOF arm
# above is useless for that, its spherical wrist splits the two apart.
UNDER_ACTUATED = """
<robot name="arm3">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/>
  <link name="tip"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" velocity="2" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-2.2" upper="2.2" velocity="2" effort="50"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.4"/><axis xyz="1 0 0"/>
    <limit lower="-2.2" upper="2.2" velocity="2" effort="50"/></joint>
  <joint name="jt" type="fixed"><parent link="l3"/><child link="tip"/>
    <origin xyz="0 0 0.3"/></joint>
</robot>
"""


def _six(dtype=torch.float64):
    return compile_robot(parse_urdf_string(SIX_DOF), dtype=dtype)


def _seven(dtype=torch.float64):
    return compile_robot(parse_urdf_string(SEVEN_DOF), dtype=dtype)


def _three(dtype=torch.float64):
    return compile_robot(parse_urdf_string(UNDER_ACTUATED), dtype=dtype)


def _reachable(chain, n, seed, link=-1, dtype=torch.float64, margin=0.7):
    """Random configurations well inside the limits, and the poses they reach.
    Staying off the limits keeps the targets away from clamped-out directions."""
    g = torch.Generator().manual_seed(seed)
    lo = chain.lower.to(device="cpu", dtype=dtype)
    hi = chain.upper.to(device="cpu", dtype=dtype)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * margin
    u = torch.rand(n, chain.dof, generator=g, dtype=dtype)
    q = mid + half * (2.0 * u - 1.0)
    target = forward_kinematics(chain, q)[:, link]
    return q, target


def _pose_error(chain, q, link, target):
    """Independent check of what came back: recompute FK and compare."""
    ee = forward_kinematics(chain, q)[:, link]
    p = (target[:, :3, 3] - ee[:, :3, 3]).norm(dim=-1)
    r = T.so3_log(target[:, :3, :3] @ ee[:, :3, :3].transpose(-1, -2)).norm(dim=-1)
    return p, r


def _fd_jacobian(chain, q, link, h=1e-6):
    """Geometric Jacobian by float64 central differences, used as the oracle
    for the nullspace claim so the test does not lean on kinfast.jacobian."""
    B, dof = q.shape
    J = torch.zeros(B, 6, dof, dtype=q.dtype)
    for j in range(dof):
        d = torch.zeros_like(q)
        d[:, j] = h
        Tp = forward_kinematics(chain, q + d)[:, link]
        Tm = forward_kinematics(chain, q - d)[:, link]
        J[:, :3, j] = (Tp[:, :3, 3] - Tm[:, :3, 3]) / (2 * h)
        J[:, 3:, j] = T.so3_log(
            Tp[:, :3, :3] @ Tm[:, :3, :3].transpose(-1, -2)) / (2 * h)
    return J


# --------------------------------------------------------------------------
# step math
# --------------------------------------------------------------------------

def test_weighted_step_matches_normal_equation_form():
    """dq = J_w^T (J_w J_w^T + lam^2 I)^-1 e_w must equal the left-hand form
    (J^T W^2 J + lam^2 I)^-1 J^T W^2 e, built here with an explicit inverse."""
    torch.manual_seed(3)
    B, m, dof = 4, 6, 7
    J = torch.randn(B, m, dof, dtype=torch.float64)
    e = torch.randn(B, m, dtype=torch.float64)
    w = torch.tensor([2.0, 1.0, 0.5, 0.0, 3.0, 1.0], dtype=torch.float64)
    lam = 0.07
    dq = weighted_dls_step(J, e, w, damping=lam)

    W2 = torch.diag(w * w)
    A = J.transpose(-1, -2) @ W2 @ J + lam * lam * torch.eye(dof, dtype=torch.float64)
    b = J.transpose(-1, -2) @ W2 @ e.unsqueeze(-1)
    ref = torch.linalg.solve(A, b).squeeze(-1)
    assert torch.allclose(dq, ref, atol=1e-10)


def test_unweighted_step_is_plain_dls():
    """weights=None must be exactly weights=all ones, no special casing drift."""
    torch.manual_seed(4)
    J = torch.randn(3, 6, 5, dtype=torch.float64)
    e = torch.randn(3, 6, dtype=torch.float64)
    ones = torch.ones(6, dtype=torch.float64)
    a = weighted_dls_step(J, e, None, damping=0.05)
    b = weighted_dls_step(J, e, ones, damping=0.05)
    assert torch.allclose(a, b, atol=1e-12)


def test_nullspace_component_does_not_move_the_task():
    """The extra motion the posture term adds must be (nearly) invisible to the
    task, measured with a finite-difference Jacobian rather than the library's."""
    chain = _seven()
    q, _ = _reachable(chain, 5, seed=11)
    link = chain.link_index["tool"]
    Jfd = _fd_jacobian(chain, q, link)
    e = torch.randn(5, 6, dtype=torch.float64) * 0.05
    w = torch.tensor([1.0, 1.0, 1.0, 0.4, 0.4, 0.4], dtype=torch.float64)
    dq_null = torch.randn(5, chain.dof, dtype=torch.float64)
    lam = 1e-5
    task = weighted_dls_step(Jfd, e, w, damping=lam)
    both = weighted_dls_step(Jfd, e, w, damping=lam, dq_null=dq_null)
    extra = both - task
    assert extra.norm(dim=-1).min() > 1e-3          # it really did add motion
    leak = ((Jfd * w.unsqueeze(-1)) @ extra.unsqueeze(-1)).squeeze(-1)
    assert (leak.norm(dim=-1) / extra.norm(dim=-1)).max() < 1e-6


def test_zero_weight_removes_a_direction_from_the_step():
    """A zeroed row must not influence dq at all, whatever the error there is."""
    torch.manual_seed(5)
    J = torch.randn(2, 6, 6, dtype=torch.float64)
    e = torch.randn(2, 6, dtype=torch.float64)
    w = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    e2 = e.clone()
    e2[:, 3:] += 17.0
    assert torch.allclose(weighted_dls_step(J, e, w, damping=0.05),
                          weighted_dls_step(J, e2, w, damping=0.05), atol=1e-12)


# --------------------------------------------------------------------------
# solving
# --------------------------------------------------------------------------

def test_full_pose_solve_six_dof():
    chain = _six()
    link = chain.link_index["ee"]
    q_true, target = _reachable(chain, 32, seed=1)
    g = torch.Generator().manual_seed(2)
    q0 = q_true + 0.25 * torch.randn(q_true.shape, generator=g, dtype=torch.float64)
    q, info = ik_task(chain, target, q0, link, iters=200, damping=0.01,
                      w_rest=0.0, tol=1e-10)
    p, r = _pose_error(chain, q, link, target)
    assert p.max() < 1e-6
    assert r.max() < 1e-6
    # info must describe the q that came back, not the previous iterate
    assert torch.allclose(info["position_error"], p, atol=1e-12)
    assert torch.allclose(info["rotation_error"], r, atol=1e-12)
    assert info["final_error"].shape == (32,)
    assert q.shape == (32, 6)


def test_position_only_weights_reduce_to_plain_ik():
    """weights=(1,1,1,0,0,0) with no posture term is algebraically the same
    iteration as ik(..., pos_only=True), so the two must agree step for step."""
    chain = _six()
    link = chain.link_index["ee"]
    _, target = _reachable(chain, 24, seed=7)
    q0 = torch.zeros(24, 6, dtype=torch.float64)
    w = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    q_task, _ = ik_task(chain, target, q0, link, weights=w, w_rest=0.0,
                        iters=60, damping=0.05, tol=0.0)
    q_plain, _ = ik(chain, target, q0, link, iters=60, damping=0.05,
                    pos_only=True, tol=0.0)
    assert torch.allclose(q_task, q_plain, atol=1e-9)
    # and the quality is real, not just equal: FK says most seeds land on the
    # target to machine precision, and neither solver is the worse of the two.
    # A single DLS seed does get stuck on a few of these, which is expected.
    p_task, _ = _pose_error(chain, q_task, link, target)
    p_plain, _ = _pose_error(chain, q_plain, link, target)
    assert torch.allclose(p_task, p_plain, atol=1e-9)
    assert (p_task < 1e-9).float().mean() > 0.8


def test_weights_set_the_priority_on_a_conflicting_target():
    """Ask for a pose the arm cannot reach and the weights decide what gives:
    heavy position weight buys position accuracy at the cost of orientation."""
    chain = _three()
    link = chain.link_index["tip"]
    n = 24
    # Reachable positions paired with unrelated orientations: 3 joints cannot
    # have both, so something has to be sacrificed and the weights pick which.
    q_true, reach = _reachable(chain, n, seed=13)
    g = torch.Generator().manual_seed(14)
    axis = torch.randn(n, 3, generator=g, dtype=torch.float64)
    ang = torch.rand(n, generator=g, dtype=torch.float64) * 2.0 + 0.5
    bad = T.make_transform(T.axis_angle_to_matrix(axis, ang), reach[:, :3, 3])

    kw = dict(w_rest=0.0, iters=400, damping=0.01, tol=0.0)
    q_pos, _ = ik_task(chain, bad, q_true, link, weights=(50, 50, 50, 1, 1, 1), **kw)
    q_rot, _ = ik_task(chain, bad, q_true, link, weights=(1, 1, 1, 50, 50, 50), **kw)
    p_pos, r_pos = _pose_error(chain, q_pos, link, bad)
    p_rot, r_rot = _pose_error(chain, q_rot, link, bad)
    assert (p_pos < p_rot).all()
    assert (r_rot < r_pos).all()
    # weight 50 against 1 buys most of the position back: the residual is a
    # couple of centimetres instead of the ~decimetre the other weighting gets
    assert p_pos.max() < 0.02
    assert p_pos.mean() < 0.2 * p_rot.mean()
    assert r_rot.mean() < 0.8 * r_pos.mean()


def test_nullspace_pulls_toward_rest_position_only():
    """Position-only on a 6-DOF arm leaves 3 free dimensions. The posture term
    must use them, and must not spend task accuracy doing it."""
    chain = _six()
    link = chain.link_index["ee"]
    q_rest, target = _reachable(chain, 24, seed=21)
    g = torch.Generator().manual_seed(22)
    q0 = q_rest + 0.6 * torch.randn(q_rest.shape, generator=g, dtype=torch.float64)
    q0 = q0.clamp(chain.lower.to(torch.float64), chain.upper.to(torch.float64))
    w = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    kw = dict(weights=w, q_rest=q_rest, iters=400, damping=0.01, tol=0.0)
    q_off, _ = ik_task(chain, target, q0, link, w_rest=0.0, **kw)
    q_on, info = ik_task(chain, target, q0, link, w_rest=0.05, **kw)

    # Compare postures only where both runs actually solved the task, so this
    # measures the nullspace term and not the odd seed stuck in a local minimum.
    p_off, _ = _pose_error(chain, q_off, link, target)
    p_on, _ = _pose_error(chain, q_on, link, target)
    ok = (p_off < 1e-9) & (p_on < 1e-9)
    assert ok.float().mean() > 0.8    # the posture pull costs no task accuracy
    d_off = (q_off - q_rest).norm(dim=-1)[ok]
    d_on = (q_on - q_rest).norm(dim=-1)[ok]
    assert (d_on < d_off).all()
    assert d_on.mean() < 0.5 * d_off.mean()
    assert torch.allclose(info["rest_error"], (q_on - q_rest).norm(dim=-1),
                          atol=1e-12)


def test_nullspace_pulls_toward_rest_full_pose_redundant():
    """Same claim on a 7-DOF arm solving all six task dimensions."""
    chain = _seven()
    link = chain.link_index["tool"]
    q_rest, target = _reachable(chain, 24, seed=31)
    g = torch.Generator().manual_seed(32)
    q0 = q_rest + 0.35 * torch.randn(q_rest.shape, generator=g, dtype=torch.float64)
    kw = dict(q_rest=q_rest, iters=400, damping=0.01, tol=0.0)
    q_off, _ = ik_task(chain, target, q0, link, w_rest=0.0, **kw)
    q_on, _ = ik_task(chain, target, q0, link, w_rest=0.05, **kw)

    p_off, r_off = _pose_error(chain, q_off, link, target)
    p_on, r_on = _pose_error(chain, q_on, link, target)
    ok = (p_off < 1e-8) & (p_on < 1e-8) & (r_off < 1e-8) & (r_on < 1e-8)
    assert ok.float().mean() > 0.8
    d_off = (q_off - q_rest).norm(dim=-1)[ok]
    d_on = (q_on - q_rest).norm(dim=-1)[ok]
    assert (d_on < d_off).all()
    assert d_on.mean() < 0.8 * d_off.mean()


def test_default_rest_posture_is_the_limit_midpoint():
    """With no q_rest given the arm should drift toward the middle of its
    range, which is the usual joint-limit-avoidance behaviour."""
    chain = _six()
    link = chain.link_index["ee"]
    mid = 0.5 * (chain.lower + chain.upper).to(torch.float64)
    _, target = _reachable(chain, 16, seed=41)
    q0 = torch.full((16, 6), 1.4, dtype=torch.float64)
    kw = dict(weights=(1, 1, 1, 0, 0, 0), iters=400, damping=0.01, tol=0.0)
    q_off, _ = ik_task(chain, target, q0, link, w_rest=0.0, **kw)
    q_on, _ = ik_task(chain, target, q0, link, w_rest=0.05, **kw)
    assert (q_on - mid).norm(dim=-1).mean() < (q_off - mid).norm(dim=-1).mean()


def test_respects_joint_limits():
    chain = _six()
    link = chain.link_index["ee"]
    _, target = _reachable(chain, 12, seed=51, margin=1.0)
    q0 = torch.zeros(12, 6, dtype=torch.float64)
    q, _ = ik_task(chain, target, q0, link, iters=100)
    lo = chain.lower.to(torch.float64)
    hi = chain.upper.to(torch.float64)
    assert (q >= lo - 1e-12).all() and (q <= hi + 1e-12).all()


def test_early_exit_reports_fewer_iterations():
    chain = _six()
    link = chain.link_index["ee"]
    q_true, target = _reachable(chain, 8, seed=61)
    q, info = ik_task(chain, target, q_true, link, iters=500, tol=1e-6,
                      check_every=5, w_rest=0.0)
    assert info["iters"] < 500
    assert bool((info["final_error"] < 1e-4).all())


# --------------------------------------------------------------------------
# plumbing: shapes, dtypes, devices, gradients
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_working_dtype_follows_the_seed(dtype):
    """The chain is compiled in float32; the caller's q sets the precision."""
    chain = _six(dtype=torch.float32)
    link = chain.link_index["ee"]
    q_true, target = _reachable(chain, 6, seed=71, dtype=torch.float64)
    q0 = q_true.to(dtype)
    q, info = ik_task(chain, target.to(torch.float64), q0, link, iters=50,
                      weights=torch.tensor([1, 1, 1, 1, 1, 1]))
    assert q.dtype == dtype
    assert info["final_error"].dtype == dtype


@pytest.mark.parametrize("weights", [
    (1.0, 1.0, 1.0, 0.2, 0.2, 0.2),
    [1, 1, 1, 0, 0, 0],
    torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=torch.float32),
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int64),
    2.0,
])
def test_weights_accept_any_container_and_dtype(weights):
    """Weights may arrive as a tuple, list, scalar, or a tensor in some other
    dtype; they get cast to the working dtype instead of promoting q."""
    chain = _six()
    link = chain.link_index["ee"]
    q_true, target = _reachable(chain, 4, seed=81)
    q, info = ik_task(chain, target, q_true, link, weights=weights, iters=20)
    assert q.dtype == torch.float64 and q.shape == (4, 6)
    assert torch.isfinite(q).all()


def test_per_row_weights_match_separate_calls():
    """A (B, 6) weight matrix must solve each row with its own priorities."""
    chain = _six()
    link = chain.link_index["ee"]
    q_true, target = _reachable(chain, 8, seed=91)
    full = torch.ones(6, dtype=torch.float64)
    pos = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    rows = torch.stack([full if i % 2 == 0 else pos for i in range(8)])
    kw = dict(w_rest=0.0, iters=40, damping=0.02, tol=0.0)
    q_mixed, _ = ik_task(chain, target, q_true, link, weights=rows, **kw)
    q_full, _ = ik_task(chain, target, q_true, link, weights=full, **kw)
    q_pos, _ = ik_task(chain, target, q_true, link, weights=pos, **kw)
    assert torch.allclose(q_mixed[0::2], q_full[0::2], atol=1e-12)
    assert torch.allclose(q_mixed[1::2], q_pos[1::2], atol=1e-12)


def test_link_may_be_a_name_or_a_negative_index():
    chain = _six()
    q_true, target = _reachable(chain, 4, seed=101)
    kw = dict(iters=30, w_rest=0.0)
    a, _ = ik_task(chain, target, q_true, "ee", **kw)
    b, _ = ik_task(chain, target, q_true, chain.link_index["ee"], **kw)
    c, _ = ik_task(chain, target, q_true, -1, **kw)
    assert torch.allclose(a, b) and torch.allclose(a, c)
    with pytest.raises(KeyError):
        ik_task(chain, target, q_true, "no_such_link", **kw)
    with pytest.raises(IndexError):
        ik_task(chain, target, q_true, 99, **kw)


def test_single_target_and_seed_may_be_unbatched():
    chain = _six()
    link = chain.link_index["ee"]
    q_true, target = _reachable(chain, 1, seed=111)
    q, _ = ik_task(chain, target[0], q_true[0], link, iters=30, w_rest=0.0)
    assert q.shape == (1, 6)
    p, r = _pose_error(chain, q, link, target)
    assert p.max() < 1e-6 and r.max() < 1e-6


def test_random_seed_when_q0_is_omitted():
    chain = _six()
    link = chain.link_index["ee"]
    _, target = _reachable(chain, 6, seed=121)
    torch.manual_seed(0)
    q, info = ik_task(chain, target, None, link, iters=50)
    lo = chain.lower.to(torch.float64)
    hi = chain.upper.to(torch.float64)
    assert q.shape == (6, 6) and q.dtype == torch.float64
    assert (q >= lo - 1e-12).all() and (q <= hi + 1e-12).all()


def test_bad_weights_are_rejected():
    chain = _six()
    q_true, target = _reachable(chain, 2, seed=131)
    with pytest.raises(ValueError):
        ik_task(chain, target, q_true, -1, weights=(1, 1, 1))
    with pytest.raises(ValueError):
        ik_task(chain, target, q_true, -1, weights=torch.ones(3, 6, dtype=torch.float64))
    with pytest.raises(ValueError):
        ik_task(chain, target, q_true, -1, weights=(1, 1, 1, 1, 1, -1))


def test_gradients_flow_through_the_solve():
    """Two iterations, differentiated end to end: the returned q must depend on
    the target, the seed and the rest posture, and the target gradient must
    match a central difference of the same scalar loss."""
    chain = _six()
    link = chain.link_index["ee"]
    q_true, target = _reachable(chain, 3, seed=141)
    rest = torch.zeros_like(q_true, requires_grad=True)
    q0 = (q_true + 0.05).requires_grad_(True)
    tgt = target.clone().requires_grad_(True)

    def loss_of(t):
        q, _ = ik_task(chain, t, q0, link, iters=2, damping=0.05,
                       q_rest=rest, w_rest=0.1)
        return (q * q).sum()

    loss = loss_of(tgt)
    loss.backward()
    assert q0.grad is not None and torch.isfinite(q0.grad).all()
    assert rest.grad is not None and rest.grad.abs().sum() > 0
    assert tgt.grad is not None and tgt.grad.abs().sum() > 0

    h = 1e-6
    idx = (0, 0, 3)
    with torch.no_grad():
        up, dn = target.clone(), target.clone()
        up[idx] += h
        dn[idx] -= h
        fd = (loss_of(up).item() - loss_of(dn).item()) / (2 * h)
    assert math.isclose(tgt.grad[idx].item(), fd, rel_tol=1e-4, abs_tol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_runs_on_cuda_with_cpu_weights():
    chain = _six(dtype=torch.float32)
    link = chain.link_index["ee"]
    q_true, target = _reachable(chain, 8, seed=151)
    chain = chain.to("cuda")
    q0 = q_true.to(device="cuda", dtype=torch.float32)
    q, info = ik_task(chain, target.to("cuda"), q0, link,
                      weights=torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
                      q_rest=torch.zeros(6), iters=60)
    assert q.device.type == "cuda" and q.dtype == torch.float32
    assert torch.isfinite(info["final_error"]).all()
