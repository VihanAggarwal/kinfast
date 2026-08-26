# tests/test_control_task.py
"""Operational-space impedance control, checked against oracles from outside
the module under test.

The oracles used here, in rough order of strength:

- closed-form solutions of the linear second-order system the controller is
  supposed to produce. A one-joint pendulum with a constant mass matrix under
  a one-row orientation task obeys exactly 0.35 thetadd = kp e - kd thetad,
  and the inertia-shaped controller is supposed to make a planar arm's
  Cartesian error obey xdd = kp e - kd xd in every direction, so both closed
  loops are compared step by step with the textbook step response.
- float64 central differences, for the Jacobian rows and for Jdot qd.
- Newton's second law read through forward_dynamics: pushing the tip with a
  known wrench must produce the acceleration the operational-space inertia
  predicts, and a nullspace torque must produce no tip acceleration at all.
- hand-computed geometry: the planar arm's tip position, the pendulum's mass
  matrix and its gravity sag, which pin the sign conventions down without
  asking the library what they are.

Everything runs in double precision and is seeded.
"""
import math

import pytest
import torch

from kinfast import control as C
from kinfast import control_task as CT
from kinfast import dynamics as D
from kinfast import transforms as T
from kinfast.compile import compile_robot
from kinfast.fk import fk_rp
from kinfast.jacobian import jacobian
from kinfast.urdf.parse import parse_urdf_string

# ---------------------------------------------------------------- fixtures

# One revolute joint about +y carrying a 1 kg rod whose COM sits 0.5 m out
# along +x. Gravity is -z, so the arm swings in the x-z plane. Everything
# about it is hand computable: the mass matrix is the constant
# 0.10 + 1 * 0.5^2 = 0.35, the link's world rotation is exactly Ry(theta),
# and the world-y row of the Jacobian is exactly 1.
PENDULUM = """
<robot name="pendulum">
  <link name="base"/>
  <link name="rod">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.02" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="rod"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.1" upper="3.1" velocity="5" effort="50"/></joint>
</robot>
"""

# Planar 2R arm in the x-z plane with a small tip link one metre past the
# elbow, so the tip position depends on both joints and the two-row (x, z)
# task is square and well conditioned away from the stretched configuration.
PLANAR = """
<robot name="planar2r">
  <link name="base"/>
  <link name="l1">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.02" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l2">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.02" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="tip">
    <inertial><origin xyz="0 0 0"/><mass value="0.2"/>
      <inertia ixx="0.002" iyy="0.002" izz="0.002" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.1" upper="3.1" velocity="5" effort="80"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.1" upper="3.1" velocity="5" effort="80"/></joint>
  <joint name="jt" type="fixed"><parent link="l2"/><child link="tip"/>
    <origin xyz="1 0 0"/></joint>
</robot>
"""

# Six-DOF spatial arm with inertials on every link: shoulder yaw, shoulder and
# elbow pitch, then a roll-pitch-roll wrist. The link inertias are deliberately
# of the same order all the way out to the wrist, so the closed loop is not
# numerically stiff and an explicit integrator can run it at a sane step size.
ARM6 = """
<robot name="arm6">
  <link name="base"/>
  <link name="l1">
    <inertial><origin xyz="0 0 0.05"/><mass value="2.0"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.015" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l2">
    <inertial><origin xyz="0.2 0 0"/><mass value="1.5"/>
      <inertia ixx="0.02" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l3">
    <inertial><origin xyz="0.15 0 0"/><mass value="1.2"/>
      <inertia ixx="0.02" iyy="0.04" izz="0.04" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l4">
    <inertial><origin xyz="0.05 0 0"/><mass value="1.0"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l5">
    <inertial><origin xyz="0.03 0 0"/><mass value="0.8"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="ee">
    <inertial><origin xyz="0.03 0 0"/><mass value="0.8"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.0" upper="3.0" velocity="3" effort="80"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3" effort="80"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0.4 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3" effort="80"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/>
    <origin xyz="0.3 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3.0" upper="3.0" velocity="3" effort="40"/></joint>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/>
    <origin xyz="0.1 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="3" effort="40"/></joint>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="ee"/>
    <origin xyz="0.06 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3.0" upper="3.0" velocity="3" effort="40"/></joint>
</robot>
"""

F64 = torch.float64
_CACHE = {}


def _chain(urdf, dtype=F64):
    """Compile once per (robot, dtype); the closed-loop tests are slow enough
    without re-parsing XML for each of them."""
    key = (id(urdf), dtype)
    if key not in _CACHE:
        _CACHE[key] = compile_robot(parse_urdf_string(urdf), dtype=dtype)
    return _CACHE[key]


def _tip(chain, q, link):
    """World position of `link` at q. (B,3)."""
    return fk_rp(chain, q)[1][link]


def _pose(chain, q, link):
    """World pose of `link` at q. (B,4,4)."""
    wR, wp = fk_rp(chain, q)
    return T.make_transform(wR[link], wp[link])


def _step_response(e0, wn, zeta, t):
    """Error of a second-order step response at time t, starting from e0.

    Solves edd + 2 zeta wn ed + wn^2 e = 0 with e(0) = e0 and ed(0) = 0, which
    is what an impedance law with kp = wn^2 and kd = 2 zeta wn must produce
    once the plant inertia has been divided out.
    """
    if zeta < 1.0:
        wd = wn * math.sqrt(1.0 - zeta * zeta)
        return e0 * math.exp(-zeta * wn * t) * (
            math.cos(wd * t)
            + (zeta / math.sqrt(1.0 - zeta * zeta)) * math.sin(wd * t))
    return e0 * (1.0 + wn * t) * math.exp(-wn * t)


# ------------------------------------------------------- geometry and error

def test_task_error_position_is_hand_computable():
    """The planar arm's tip is at (cos a + cos b, 0, -sin a - sin b) with
    a = q1 and b = q1 + q2, so the reported error has a closed form."""
    chain = _chain(PLANAR)
    tip = chain.link_index["tip"]
    q = torch.tensor([[0.4, -0.7], [-1.2, 0.5]], dtype=F64)
    a, b = q[:, 0], q[:, 0] + q[:, 1]
    p = torch.stack([torch.cos(a) + torch.cos(b),
                     torch.zeros_like(a),
                     -torch.sin(a) - torch.sin(b)], dim=-1)
    assert torch.allclose(_tip(chain, q, tip), p, atol=1e-12)
    x_des = torch.tensor([[1.0, 0.0, 0.5], [0.2, 0.0, -0.3]], dtype=F64)
    assert torch.allclose(CT.task_error(chain, q, x_des, tip), x_des - p, atol=1e-12)


def test_task_error_orientation_is_the_rotation_that_closes_the_gap():
    """Build the target by pre-rotating the current pose by a known rotation
    vector; the reported orientation error must be that very vector."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(0)
    q = torch.rand(4, 6, dtype=F64) * 1.4 - 0.7
    axis = torch.randn(4, 3, dtype=F64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.tensor([0.4, -0.9, 1.3, 0.05], dtype=F64)
    w = axis * angle.unsqueeze(-1)
    Rw = T.axis_angle_to_matrix(axis, angle)
    cur = _pose(chain, q, ee)
    tgt = cur.clone()
    tgt[:, :3, :3] = Rw @ cur[:, :3, :3]
    err = CT.task_error(chain, q, tgt, ee)
    assert torch.allclose(err[:, :3], torch.zeros(4, 3, dtype=F64), atol=1e-12)
    assert torch.allclose(err[:, 3:], w, atol=1e-10)


def test_task_error_rows_and_broadcast():
    chain = _chain(PLANAR)
    tip = chain.link_index["tip"]
    q = torch.tensor([[0.4, -0.7], [-1.2, 0.5]], dtype=F64)
    x_des = torch.tensor([0.9, 0.0, -0.2], dtype=F64)          # (3,), broadcast
    full = CT.task_error(chain, q, x_des, tip)
    planar = CT.task_error(chain, q, x_des, tip, rows=(0, 2))
    assert full.shape == (2, 3) and planar.shape == (2, 2)
    assert torch.allclose(planar, full[:, [0, 2]], atol=1e-14)


def test_task_jacobian_matches_central_differences():
    """All six rows against float64 central differences of the pose."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(1)
    q = torch.rand(3, 6, dtype=F64) * 1.4 - 0.7
    J = CT.task_jacobian(chain, q, ee)
    h = 1e-6
    for k in range(6):
        dq = torch.zeros_like(q)
        dq[:, k] = h
        Rp, pp = fk_rp(chain, q + dq)
        Rm, pm = fk_rp(chain, q - dq)
        v_fd = (pp[ee] - pm[ee]) / (2 * h)
        w_fd = T.so3_log(Rp[ee] @ Rm[ee].transpose(-1, -2)) / (2 * h)
        assert torch.allclose(J[:, :3, k], v_fd, atol=1e-7)
        assert torch.allclose(J[:, 3:, k], w_fd, atol=1e-7)


def test_task_jacobian_row_selection():
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(2)
    q = torch.rand(2, 6, dtype=F64)
    full = jacobian(chain, q, ee)
    sel = CT.task_jacobian(chain, q, ee, rows=(2, 4))
    assert sel.shape == (2, 2, 6)
    assert torch.allclose(sel, full[:, [2, 4], :], atol=1e-14)


def test_negative_link_index_is_the_last_link():
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    assert ee == chain.n_links - 1
    torch.manual_seed(3)
    q = torch.rand(2, 6, dtype=F64)
    assert torch.allclose(CT.task_jacobian(chain, q, -1),
                          CT.task_jacobian(chain, q, ee), atol=1e-14)


# ---------------------------------------------------------------- Jdot qd

def test_jdot_qd_matches_central_differences():
    """d/dt (J qd) along qd, against float64 central differences of J(q) qd."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(4)
    q = torch.rand(3, 6, dtype=F64) * 1.4 - 0.7
    qd = torch.rand(3, 6, dtype=F64) - 0.5
    h = 1e-6
    for rows in [(0, 1, 2), None]:
        Jp = CT.task_jacobian(chain, q + h * qd, ee, rows=rows)
        Jm = CT.task_jacobian(chain, q - h * qd, ee, rows=rows)
        fd = ((Jp - Jm) @ qd.unsqueeze(-1)).squeeze(-1) / (2 * h)
        assert torch.allclose(CT.jdot_qd(chain, q, qd, ee, rows=rows), fd, atol=1e-7)


def test_jdot_qd_works_under_no_grad_and_keeps_a_graph_when_asked():
    chain = _chain(ARM6)
    torch.manual_seed(5)
    q = torch.rand(1, 6, dtype=F64)
    qd = torch.rand(1, 6, dtype=F64)
    with torch.no_grad():
        out = CT.jdot_qd(chain, q, qd, -1)
    assert not out.requires_grad

    qg = q.clone().requires_grad_(True)
    out = CT.jdot_qd(chain, qg, qd, -1)
    assert out.requires_grad
    out.sum().backward()
    assert torch.isfinite(qg.grad).all()
    assert qg.grad.abs().max() > 0


def test_jdot_qd_rejects_inference_mode():
    chain = _chain(PLANAR)
    with torch.inference_mode():
        q = torch.full((1, 2), 0.3, dtype=F64)
        with pytest.raises(ValueError, match="inference_mode"):
            CT.jdot_qd(chain, q, torch.ones(1, 2, dtype=F64), -1)


# ------------------------------------------------- operational-space inertia

def test_task_inertia_is_the_inertia_felt_at_the_tip():
    """Newton's second law in task space: at rest with gravity off, a wrench F
    applied through tau = J^T F must accelerate the tip by Lambda^-1 F."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(6)
    q = torch.rand(4, 6, dtype=F64) * 1.4 - 0.7
    J = CT.task_jacobian(chain, q, ee, rows=(0, 1, 2))
    Lam = CT.task_inertia(chain, q, ee, rows=(0, 1, 2), damping=0.0)
    F = torch.randn(4, 3, dtype=F64)
    tau = (J.transpose(-1, -2) @ F.unsqueeze(-1)).squeeze(-1)
    qdd = D.forward_dynamics(chain, q, torch.zeros_like(q), tau, use_gravity=False)
    xdd = (J @ qdd.unsqueeze(-1)).squeeze(-1)
    want = torch.linalg.solve(Lam, F.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(xdd, want, atol=1e-9)


def test_task_inertia_of_the_pendulum_is_hand_computable():
    """Parallel axis theorem: the rod's inertia about the joint is
    0.10 + 1 * 0.5^2 = 0.35, and for a one-row rotation task Lambda is
    exactly that number."""
    chain = _chain(PENDULUM)
    rod = chain.link_index["rod"]
    q = torch.tensor([[0.0], [0.7]], dtype=F64)
    M = D.mass_matrix(chain, q)
    assert torch.allclose(M[:, 0, 0], torch.full((2,), 0.35, dtype=F64), atol=1e-12)
    Lam = CT.task_inertia(chain, q, rod, rows=(4,), damping=0.0)
    assert torch.allclose(Lam[:, 0, 0], torch.full((2,), 0.35, dtype=F64), atol=1e-12)


# --------------------------------------------------------------- nullspace

def test_nullspace_projectors_annihilate_the_task():
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(7)
    q = torch.rand(3, 6, dtype=F64) * 1.4 - 0.7
    J = CT.task_jacobian(chain, q, ee, rows=(0, 1, 2))
    M = D.mass_matrix(chain, q)
    N_kin = CT.nullspace_projector(J, damping=0.0)
    assert float((J @ N_kin).abs().max()) < 1e-10
    N_dyn = CT.nullspace_projector(J, M=M, damping=0.0)
    assert float((J @ torch.linalg.solve(M, N_dyn)).abs().max()) < 1e-10


def test_nullspace_torque_produces_no_task_acceleration():
    """The physical reading of the same statement: a torque pushed through the
    dynamically consistent projector accelerates joints but not the tip."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(8)
    q = torch.rand(3, 6, dtype=F64) * 1.4 - 0.7
    J = CT.task_jacobian(chain, q, ee, rows=(0, 1, 2))
    M = D.mass_matrix(chain, q)
    N = CT.nullspace_projector(J, M=M, damping=0.0)
    tau0 = torch.randn(3, 6, dtype=F64)
    tau = (N @ tau0.unsqueeze(-1)).squeeze(-1)
    qdd = D.forward_dynamics(chain, q, torch.zeros_like(q), tau, use_gravity=False)
    assert float(qdd.abs().max()) > 1e-3                # it does move the joints
    assert float((J @ qdd.unsqueeze(-1)).abs().max()) < 1e-9


# ------------------------------------------------------------ the control law

def test_impedance_torque_is_the_written_formula():
    """tau assembled independently from jacobian(), fk_rp() and gravity()."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(9)
    q = torch.rand(3, 6, dtype=F64) * 1.4 - 0.7
    qd = torch.rand(3, 6, dtype=F64) - 0.5
    x_des = torch.rand(3, 3, dtype=F64)
    kp, kd = 250.0, 20.0
    Jv = jacobian(chain, q, ee)[:, :3, :]
    e = x_des - _tip(chain, q, ee)
    xd = (Jv @ qd.unsqueeze(-1)).squeeze(-1)
    wrench = kp * e - kd * xd
    grav = D.gravity(chain, q)
    expect = (Jv.transpose(-1, -2) @ wrench.unsqueeze(-1)).squeeze(-1) + grav
    got = CT.opspace_impedance(chain, q, qd, x_des, kp, kd, ee)
    assert torch.allclose(got, expect, atol=1e-12)

    # use_gravity=False drops exactly the gravity feedforward and nothing else
    no_g = CT.opspace_impedance(chain, q, qd, x_des, kp, kd, ee, use_gravity=False)
    assert torch.allclose(expect - no_g, grav, atol=1e-12)


def test_gain_forms_agree():
    """Scalar, per-row vector and full matrix gains must mean the same thing
    when they describe the same stiffness."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(10)
    q = torch.rand(2, 6, dtype=F64)
    qd = torch.rand(2, 6, dtype=F64) - 0.5
    x_des = torch.rand(2, 3, dtype=F64)
    a = CT.opspace_impedance(chain, q, qd, x_des, 120.0, 9.0, ee)
    b = CT.opspace_impedance(chain, q, qd, x_des,
                             torch.full((3,), 120.0, dtype=F64),
                             torch.full((3,), 9.0, dtype=F64), ee)
    c = CT.opspace_impedance(chain, q, qd, x_des,
                             120.0 * torch.eye(3, dtype=F64),
                             9.0 * torch.eye(3, dtype=F64), ee)
    assert torch.allclose(a, b, atol=1e-12)
    assert torch.allclose(a, c, atol=1e-12)


def test_anisotropic_stiffness_only_pushes_along_its_own_axis():
    """A gain matrix that is stiff in x and zero elsewhere must produce a
    wrench with no y or z component, which is the point of matrix gains."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(15)
    q = torch.rand(2, 6, dtype=F64) * 1.4 - 0.7
    qd = torch.zeros(2, 6, dtype=F64)
    x_des = _tip(chain, q, ee) + torch.tensor([[0.05, 0.05, 0.05]], dtype=F64)
    Kp = torch.zeros(3, 3, dtype=F64)
    Kp[0, 0] = 500.0
    tau = CT.opspace_impedance(chain, q, qd, x_des, Kp, 0.0, ee, use_gravity=False)
    Jv = jacobian(chain, q, ee)[:, :3, :]
    # recover the wrench from tau: J^T is full column rank here (6 joints, 3 rows)
    wrench = torch.linalg.lstsq(Jv.transpose(-1, -2), tau.unsqueeze(-1)).solution
    assert torch.allclose(wrench[:, 1:, 0], torch.zeros(2, 2, dtype=F64), atol=1e-8)
    assert torch.allclose(wrench[:, 0, 0], torch.full((2,), 500.0 * 0.05, dtype=F64),
                          atol=1e-8)


def test_feedforward_task_velocity_cancels_the_damping_term():
    """Handed the tip's actual velocity as xd_des, the damper sees zero."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(11)
    q = torch.rand(2, 6, dtype=F64)
    qd = torch.rand(2, 6, dtype=F64) - 0.5
    x_des = torch.rand(2, 3, dtype=F64)
    Jv = jacobian(chain, q, ee)[:, :3, :]
    xd = (Jv @ qd.unsqueeze(-1)).squeeze(-1)
    got = CT.opspace_impedance(chain, q, qd, x_des, 100.0, 40.0, ee, xd_des=xd)
    e = x_des - _tip(chain, q, ee)
    expect = (Jv.transpose(-1, -2) @ (100.0 * e).unsqueeze(-1)).squeeze(-1) \
        + D.gravity(chain, q)
    assert torch.allclose(got, expect, atol=1e-11)


def test_batched_matches_one_at_a_time():
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(12)
    q = torch.rand(5, 6, dtype=F64) * 1.4 - 0.7
    qd = torch.rand(5, 6, dtype=F64) - 0.5
    x_des = torch.rand(5, 3, dtype=F64)
    q_rest = torch.rand(5, 6, dtype=F64) - 0.5
    batched = CT.opspace_impedance(chain, q, qd, x_des, 150.0, 15.0, ee,
                                   inertia_shaping=True, null_kd=2.0,
                                   null_kp=4.0, q_rest=q_rest)
    for i in range(5):
        one = CT.opspace_impedance(chain, q[i:i + 1], qd[i:i + 1], x_des[i:i + 1],
                                   150.0, 15.0, ee, inertia_shaping=True,
                                   null_kd=2.0, null_kp=4.0,
                                   q_rest=q_rest[i:i + 1])
        assert torch.allclose(batched[i], one[0], atol=1e-9)


def test_single_target_broadcasts_over_the_batch():
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(14)
    q = torch.rand(4, 6, dtype=F64)
    qd = torch.zeros(4, 6, dtype=F64)
    x = torch.tensor([0.4, 0.1, 0.3], dtype=F64)
    a = CT.opspace_impedance(chain, q, qd, x, 100.0, 10.0, ee)
    b = CT.opspace_impedance(chain, q, qd, x.expand(4, 3), 100.0, 10.0, ee)
    assert torch.allclose(a, b, atol=1e-14)


def test_dtype_follows_q():
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    torch.manual_seed(13)
    q = torch.rand(2, 6, dtype=F64) * 1.4 - 0.7
    qd = torch.rand(2, 6, dtype=F64) - 0.5
    x_des = torch.rand(2, 3, dtype=F64)
    ref = CT.opspace_impedance(chain, q, qd, x_des, 100.0, 10.0, ee, null_kd=1.0)
    got = CT.opspace_impedance(chain, q.float(), qd.float(), x_des.float(),
                               100.0, 10.0, ee, null_kd=1.0)
    assert got.dtype is torch.float32
    assert torch.allclose(got.double(), ref, atol=2e-3)
    # the float32 call must not have mutated the float64 chain it borrowed
    assert chain.joint_origin.dtype is F64


def test_differentiable_through_a_short_rollout():
    """Cartesian gains are learnable: a loss on where the tip ends up after a
    few integration steps has a finite, non-zero gradient in kp."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    kp = torch.tensor(300.0, dtype=F64, requires_grad=True)
    q = torch.tensor([[0.2, 0.5, -0.9, 0.3, 0.6, -0.2]], dtype=F64)
    qd = torch.zeros_like(q)
    x_des = _tip(chain, torch.tensor([[-0.3, 0.9, -1.3, -0.2, 0.9, 0.4]], dtype=F64), ee)
    for _ in range(15):
        tau = CT.opspace_impedance(chain, q, qd, x_des, kp, 30.0, ee)
        qd = qd + 1e-3 * D.forward_dynamics(chain, q, qd, tau)
        q = q + 1e-3 * qd
    ((_tip(chain, q, ee) - x_des) ** 2).sum().backward()
    assert torch.isfinite(kp.grad) and float(kp.grad.abs()) > 1e-9


# ------------------------------------------------------------- closed loop

def test_orientation_loop_reproduces_the_scalar_plant():
    """One joint, one task row, constant inertia. The closed loop is then
    exactly 0.35 thetadd = kp e - kd thetad, so the whole stack (task error,
    Jacobian row, gravity feedforward, forward dynamics, integrator) can be
    replayed here in four lines of plain Python and must agree to round-off.
    The same trajectory is also checked, loosely, against the continuous
    closed-form step response, which is what the discrete scheme approximates.
    """
    chain = _chain(PENDULUM)
    rod = chain.link_index["rod"]
    inertia = 0.35
    wn, zeta = 10.0, 0.5
    kp = inertia * wn * wn
    kd = inertia * 2.0 * zeta * wn

    th0, e0 = 0.2, 0.6
    q0 = torch.tensor([[th0]], dtype=F64)
    R_des = T.rpy_to_matrix(torch.tensor([[0.0, th0 + e0, 0.0]], dtype=F64))
    T_des = T.make_transform(R_des, torch.zeros(1, 3, dtype=F64))

    dt, steps = 2e-3, 600
    ctrl = CT.impedance_controller(chain, T_des, kp, kd, rod, rows=(4,))
    ts, qs, _ = C.simulate(chain, q0, torch.zeros_like(q0), ctrl, dt, steps)

    th, w = th0, 0.0                       # the same semi-implicit Euler scheme
    worst_plant = worst_analytic = 0.0
    for k in range(steps):
        tau = kp * (th0 + e0 - th) - kd * w
        w += dt * tau / inertia
        th += dt * w
        worst_plant = max(worst_plant, abs(th - float(qs[k])))
        worst_analytic = max(worst_analytic, abs(
            (th0 + e0 - float(qs[k])) - _step_response(e0, wn, zeta, float(ts[k]))))
    assert worst_plant < 1e-9, worst_plant
    assert worst_analytic < 1e-2, worst_analytic
    assert abs(th0 + e0 - float(qs[-1])) < 0.01 * e0     # and it did converge


def test_gravity_feedforward_removes_the_sag():
    """With the feedforward on, sitting on the setpoint is an equilibrium: the
    torque is exactly the gravity torque and the acceleration is exactly zero.
    With it off the arm settles lower down, where the task spring balances
    gravity, and that equilibrium is hand computable."""
    chain = _chain(PENDULUM)
    rod = chain.link_index["rod"]
    kp, kd = 35.0, 7.0                                # critically damped, wn = 10
    T_des = T.make_transform(torch.eye(3, dtype=F64).expand(1, 3, 3),
                             torch.zeros(1, 3, dtype=F64))        # theta_des = 0

    at_rest = torch.zeros(1, 1, dtype=F64)
    tau = CT.opspace_impedance(chain, at_rest, at_rest, T_des, kp, kd, rod, rows=(4,))
    assert torch.allclose(tau, D.gravity(chain, at_rest), atol=1e-14)
    qdd = D.forward_dynamics(chain, at_rest, at_rest, tau)
    assert float(qdd.abs().max()) < 1e-12

    ctrl = CT.impedance_controller(chain, T_des, kp, kd, rod, rows=(4,),
                                   use_gravity=False)
    _, qs, qds = C.simulate(chain, torch.tensor([[0.5]], dtype=F64), at_rest,
                            ctrl, 2e-3, 900)
    assert float(qds[-1].abs().max()) < 1e-5                      # settled
    theta = float(qs[-1])
    # Equilibrium without the feedforward: kp (0 - theta) = tau_g(theta), and
    # for this rod tau_g = m g d cos(theta) = 9.81 * 1.0 * 0.5 * cos(theta).
    assert abs(kp * (-theta) + 9.81 * 0.5 * math.cos(theta)) < 1e-4
    assert theta > 0.1                                            # it really sagged


def test_inertia_shaping_decouples_the_cartesian_error():
    """With Lambda and the task bias in the loop, the planar arm's tip error
    must follow xdd = kp e - kd xd in both task directions, whatever the
    configuration-dependent inertia is doing underneath."""
    chain = _chain(PLANAR)
    tip = chain.link_index["tip"]
    wn, zeta = 8.0, 0.7
    kp, kd = wn * wn, 2.0 * zeta * wn
    q0 = torch.tensor([[0.5, 0.8]], dtype=F64)
    e0 = torch.tensor([[0.04, 0.0, -0.03]], dtype=F64)
    x_des = _tip(chain, q0, tip) + e0

    ctrl = CT.impedance_controller(chain, x_des, kp, kd, tip, rows=(0, 2),
                                   inertia_shaping=True, damping=0.0)
    ts, qs, _ = C.simulate(chain, q0, torch.zeros_like(q0), ctrl, 2e-3, 400,
                           record_every=20)
    for i in range(len(ts)):
        e = (x_des - _tip(chain, qs[i], tip))[0]
        t = float(ts[i])
        assert abs(float(e[0]) - _step_response(float(e0[0, 0]), wn, zeta, t)) < 6e-4
        assert abs(float(e[2]) - _step_response(float(e0[0, 2]), wn, zeta, t)) < 6e-4
        assert abs(float(e[1])) < 1e-9            # the arm never leaves its plane
    # the two directions decay in lockstep, which is what decoupling means:
    # their ratio stays the ratio they started at
    late = (x_des - _tip(chain, qs[len(ts) // 4], tip))[0]
    assert abs(float(late[0] / late[2]) - float(e0[0, 0] / e0[0, 2])) < 0.02


def test_end_effector_converges_to_a_cartesian_setpoint_and_stays():
    """The headline behaviour: a redundant 6-DOF arm driven only by the
    Cartesian impedance law reaches a reachable point to well inside a
    centimetre within two seconds of simulated time, and is still sitting on
    it, at rest, when the two seconds are up."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    q0 = torch.tensor([[0.2, 0.5, -0.9, 0.3, 0.6, -0.2]], dtype=F64)
    q_goal = torch.tensor([[-0.3, 0.9, -1.3, -0.2, 0.9, 0.4]], dtype=F64)
    x_des = _tip(chain, q_goal, ee)
    assert float((x_des - _tip(chain, q0, ee)).norm()) > 0.3      # a real move

    ctrl = CT.impedance_controller(chain, x_des, kp=600.0, kd=50.0, link_index=ee,
                                   null_kd=1.5)
    dt, steps = 4e-3, 500                                         # 2.0 s
    ts, qs, qds = C.simulate(chain, q0, torch.zeros_like(q0), ctrl, dt, steps)
    assert float(ts[-1]) == pytest.approx(2.0)

    errs = [float((x_des - _tip(chain, qs[i], ee)).norm()) for i in range(0, steps, 10)]
    assert float((x_des - _tip(chain, qs[-1], ee)).norm()) < 1e-2
    # not merely passing through: the whole second half of the run stays inside
    assert max(errs[len(errs) // 2:]) < 1e-2, errs
    # and it has actually stopped, tip and joints both
    Jv = CT.task_jacobian(chain, qs[-1], ee, rows=(0, 1, 2))
    assert float((Jv @ qds[-1].unsqueeze(-1)).norm()) < 1e-2
    assert float(qds[-1].abs().max()) < 5e-2


def test_full_pose_setpoint_converges_in_position_and_orientation():
    """The same arm with all six task rows live: position and orientation both
    close, from an initial gap of 40 cm and most of a radian."""
    chain = _chain(ARM6)
    ee = chain.link_index["ee"]
    q0 = torch.tensor([[0.2, 0.5, -0.9, 0.3, 0.6, -0.2]], dtype=F64)
    q_goal = torch.tensor([[-0.3, 0.9, -1.3, -0.2, 0.9, 0.4]], dtype=F64)
    T_des = _pose(chain, q_goal, ee)
    e0 = T.pose_error(_pose(chain, q0, ee), T_des)[0]
    assert float(e0[:3].norm()) > 0.3 and float(e0[3:].norm()) > 0.5

    kp = torch.tensor([400.0] * 3 + [40.0] * 3, dtype=F64)
    kd = torch.tensor([40.0] * 3 + [5.0] * 3, dtype=F64)
    ctrl = CT.impedance_controller(chain, T_des, kp, kd, ee)
    _, qs, qds = C.simulate(chain, q0, torch.zeros_like(q0), ctrl, 4e-3, 400)
    e = T.pose_error(_pose(chain, qs[-1], ee), T_des)[0]
    assert float(e[:3].norm()) < 1e-2
    assert float(e[3:].norm()) < 1e-2
    assert float(qds[-1].abs().max()) < 5e-2


def test_posture_spring_moves_the_nullspace_without_disturbing_the_task():
    """Servoing only the tip's x coordinate on the planar 2R leaves one
    redundant joint free. The rest posture below is chosen to sit exactly on
    the task manifold (its own tip x is the setpoint), so a posture spring in
    the nullspace should walk the arm all the way onto it while the task stays
    converged; plain nullspace damping has no reason to go anywhere near it."""
    chain = _chain(PLANAR)
    tip = chain.link_index["tip"]
    q0 = torch.tensor([[0.9, 1.3]], dtype=F64)
    x_des = _tip(chain, torch.tensor([[0.5, 1.0]], dtype=F64), tip)
    # elbow-up solution at shoulder angle 1.0 rad for the same tip x
    q1 = 1.0
    q2 = math.acos(float(x_des[0, 0]) - math.cos(q1)) - q1
    q_rest = torch.tensor([[q1, q2]], dtype=F64)
    assert abs(float(_tip(chain, q_rest, tip)[0, 0] - x_des[0, 0])) < 1e-12
    dt, steps = 2e-3, 600

    plain = CT.impedance_controller(chain, x_des, 400.0, 40.0, tip, rows=(0,),
                                    null_kd=8.0)
    _, qs_a, _ = C.simulate(chain, q0, torch.zeros_like(q0), plain, dt, steps)
    spring = CT.impedance_controller(chain, x_des, 400.0, 40.0, tip, rows=(0,),
                                     null_kd=8.0, null_kp=40.0, q_rest=q_rest)
    _, qs_b, qd_b = C.simulate(chain, q0, torch.zeros_like(q0), spring, dt, steps)

    for qs in (qs_a, qs_b):
        assert abs(float((x_des - _tip(chain, qs[-1], tip))[0, 0])) < 1e-3
    d_a = float((qs_a[-1] - q_rest).norm())
    d_b = float((qs_b[-1] - q_rest).norm())
    assert d_a > 0.5, d_a                    # damping alone stops somewhere else
    assert d_b < 0.05, d_b                   # the spring lands on the posture
    assert float(qd_b[-1].abs().max()) < 5e-2


def test_moving_target_is_tracked():
    """x_des as a function of time: with the matching velocity feedforward the
    tip follows a Cartesian ramp with a lag of a couple of millimetres during
    the initial jerk and essentially none once it is up to speed."""
    chain = _chain(PLANAR)
    tip = chain.link_index["tip"]
    q0 = torch.tensor([[0.4, 1.5]], dtype=F64)
    x0 = _tip(chain, q0, tip)
    vel = torch.tensor([[0.10, 0.0, 0.05]], dtype=F64)

    def target(t):
        return x0 + vel * t

    ctrl = CT.impedance_controller(chain, target, 900.0, 60.0, tip, rows=(0, 2),
                                   xd_des=lambda t: vel[:, [0, 2]])
    ts, qs, _ = C.simulate(chain, q0, torch.zeros_like(q0), ctrl, 2e-3, 400,
                           record_every=20)
    lags = [float((target(float(ts[i])) - _tip(chain, qs[i], tip)).norm())
            for i in range(len(ts))]
    assert max(lags) < 3e-3, lags
    assert lags[-1] < 1e-4, lags
    assert float((_tip(chain, qs[-1], tip) - x0).norm()) > 0.05   # it did move


# ------------------------------------------------------------- error paths

def test_bad_arguments_are_rejected():
    chain = _chain(ARM6)
    q = torch.zeros(2, 6, dtype=F64)
    qd = torch.zeros(2, 6, dtype=F64)
    x = torch.zeros(2, 3, dtype=F64)

    with pytest.raises(ValueError, match="position or a"):
        CT.opspace_impedance(chain, q, qd, torch.zeros(2, 5, dtype=F64), 1.0, 1.0)
    with pytest.raises(ValueError, match="batch"):
        CT.opspace_impedance(chain, q, qd, torch.zeros(3, 3, dtype=F64), 1.0, 1.0)
    with pytest.raises(ValueError, match="orientation rows"):
        CT.opspace_impedance(chain, q, qd, x, 1.0, 1.0, rows=(0, 1, 4))
    with pytest.raises(ValueError, match="kp has 4 entries"):
        CT.opspace_impedance(chain, q, qd, x, torch.ones(4, dtype=F64), 1.0)
    with pytest.raises(ValueError, match="q_rest"):
        CT.opspace_impedance(chain, q, qd, x, 1.0, 1.0, null_kp=1.0)
    with pytest.raises(ValueError, match="xd_des has"):
        CT.opspace_impedance(chain, q, qd, x, 1.0, 1.0,
                             xd_des=torch.zeros(2, 2, dtype=F64))
    with pytest.raises(ValueError, match="non-empty"):
        CT.opspace_impedance(chain, q, qd, x, 1.0, 1.0, rows=())
    with pytest.raises(IndexError):
        CT.opspace_impedance(chain, q, qd, x, 1.0, 1.0, link_index=99)
