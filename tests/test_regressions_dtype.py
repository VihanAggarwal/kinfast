# tests/test_regressions_dtype.py
"""Regression: a robot compiled in one float dtype must accept q in the other.

kinfast.load / load_string always compile float32. fk already cast the chain's
constants to q's dtype, but jacobian, ik, dynamics, control, analysis and
collision used the raw chain tensors and crashed with bare torch errors on
float64 q (and the float64-chain / float32-q direction crashed in collision).
Mixed dtypes between q and qd/qdd/tau, or between an IK target and its seed,
crashed as well. The rule is now uniform: q's dtype is the working dtype.

Oracles are hand-derived closed forms for the planar arms, not the library.
"""
import math
import torch
import kinfast
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.jacobian import jacobian
from kinfast.ik import ik
from kinfast import analysis, control
from kinfast.collision import (SphereModel, self_distance, distance_to_obstacles,
                               collision_aware_ik)
from tests.test_parse import TWO_LINK
from tests.test_dynamics import DYN_ARM

F32, F64 = torch.float32, torch.float64


def _two_link_jacobian(q1, q2, dtype):
    """Planar 2R about z. The last link's origin sits at joint 2, one unit
    out along link 1, so the ee is (c1, s1, 0) and joint 2 only adds spin."""
    s1, c1 = math.sin(q1), math.cos(q1)
    return torch.tensor([[-s1, 0.0],
                         [c1, 0.0],
                         [0.0, 0.0],
                         [0.0, 0.0],
                         [0.0, 0.0],
                         [1.0, 1.0]], dtype=dtype)


def _dyn_arm_mass(q2, dtype):
    """Double pendulum, l1=1, lc1=lc2=0.5, m=1, I(axis)=0.1 per link."""
    c2 = math.cos(q2)
    return torch.tensor([[1.7 + c2, 0.35 + 0.5 * c2],
                         [0.35 + 0.5 * c2, 0.35]], dtype=dtype)


def _dyn_arm_gravity(q1, q2, dtype, g=9.81):
    """dU/dq for COM heights z1 = -0.5 sin q1, z2 = -sin q1 - 0.5 sin(q1+q2)."""
    c1, c12 = math.cos(q1), math.cos(q1 + q2)
    return torch.tensor([-g * (1.5 * c1 + 0.5 * c12), -g * 0.5 * c12], dtype=dtype)


def test_float32_robot_accepts_float64_q_everywhere():
    torch.manual_seed(0)
    robot = kinfast.load_string(DYN_ARM)            # compiled float32
    assert robot.chain.lower.dtype == F32
    q1, q2 = 0.3, -0.7
    q = torch.tensor([[q1, q2]], dtype=F64)
    qd = torch.tensor([[0.1, 0.2]], dtype=F64)

    # kinematics
    J = robot.jacobian(q)
    assert J.dtype == F64
    M = robot.mass_matrix(q)
    assert M.dtype == F64
    assert torch.allclose(M[0], _dyn_arm_mass(q2, F64), atol=1e-6)
    gq = robot.gravity(q)
    assert gq.dtype == F64
    assert torch.allclose(gq[0], _dyn_arm_gravity(q1, q2, F64), atol=1e-5)

    # dynamics in both directions, with velocities/torques also float64
    tau = robot.inverse_dynamics(q, qd, qd)
    assert tau.dtype == F64
    qdd = robot.forward_dynamics(q, qd, tau)
    assert qdd.dtype == F64
    assert torch.allclose(qdd, qd, atol=1e-5)          # round trip

    # ik: seed drawn internally, seed given, and multi-start
    target = robot.fk(q)
    assert target.dtype == F64
    for kw in ({}, {"q0": q + 0.05}, {"restarts": 3}):
        qs, info = robot.ik(target, pos_only=True, iters=30, **kw)
        assert qs.dtype == F64
        assert info["final_error"].dtype == F64
    qs, info = robot.ik(target, q0=q + 0.05, pos_only=True, iters=50)
    assert float(info["final_error"].max()) < 1e-4

    # analysis
    ee = robot.link_id(robot.ee_link)
    assert analysis.manipulability(robot.chain, q, ee, rows=(0, 2)).dtype == F64
    assert analysis.condition_number(robot.chain, q, ee).dtype == F64
    assert analysis.joint_limit_margin(robot.chain, q).dtype == F64

    # control
    v = control.computed_torque(robot.chain, q, qd, q, qd, qd, 1.0, 0.5)
    assert v.dtype == F64
    ts, qs, qds = control.simulate(robot.chain, q, qd,
                                   lambda t, q_, qd_: control.gravity_compensation(robot.chain, q_),
                                   dt=1e-3, steps=3)
    assert qs.dtype == F64 and qds.dtype == F64 and ts.dtype == F64

    # collision
    model = robot.sphere_model({"base": [(0, 0, 0, 0.1)], "l2": [(0, 0, 0, 0.1)]})
    assert self_distance(model, q).dtype == F64
    obs_c = torch.tensor([[5.0, 0.0, 0.0]], dtype=F64)
    obs_r = torch.tensor([0.1], dtype=F64)
    assert distance_to_obstacles(model, q, obs_c, obs_r).dtype == F64


def test_float64_chain_accepts_float32_q():
    torch.manual_seed(0)
    chain = compile_robot(parse_urdf_string(TWO_LINK), dtype=F64)
    q1, q2 = 0.4, 0.9
    q = torch.tensor([[q1, q2]], dtype=F32)
    ee = chain.link_index["l2"]

    J = jacobian(chain, q, ee)
    assert J.dtype == F32
    assert torch.allclose(J[0], _two_link_jacobian(q1, q2, F32), atol=1e-5)

    # ik keeps the seed's dtype (clamp against float64 limits used to promote)
    target = forward_kinematics(chain, q)[:, ee]
    qs, info = ik(chain, target, q0=q + 0.1, link_index=ee,
                  pos_only=True, iters=30)
    assert qs.dtype == F32 and info["final_error"].dtype == F32

    # collision path on a float64 chain with float32 q
    b, l2 = chain.link_index["base"], chain.link_index["l2"]
    model = SphereModel(chain, {b: [(0, 0, 0, 0.2)], l2: [(0, 0, 0, 0.3)]})
    d = self_distance(model, torch.zeros(1, 2, dtype=F32))
    assert d.dtype == F32
    assert torch.allclose(d, torch.tensor([0.5], dtype=F32), atol=1e-6)  # 1 - 0.2 - 0.3
    obs_c = torch.tensor([[3.0, 0.0, 0.0]], dtype=F32)
    obs_r = torch.tensor([0.5], dtype=F32)
    d_obs = distance_to_obstacles(model, torch.zeros(1, 2, dtype=F32), obs_c, obs_r)
    assert d_obs.dtype == F32
    # nearest robot sphere is l2 at x=1 (r=0.3): 3 - 1 - 0.3 - 0.5 = 1.2
    assert torch.allclose(d_obs, torch.tensor([1.2], dtype=F32), atol=1e-6)
    qc, info = collision_aware_ik(model, torch.tensor([[1.0, 1.0, 0.0]], dtype=F32),
                                  torch.zeros(1, 2, dtype=F32), l2, obs_c, obs_r,
                                  iters=3, refine_iters=2)
    assert qc.dtype == F32 and info["clearance"].dtype == F32


def test_mixed_state_dtypes_follow_q():
    torch.manual_seed(0)
    for chain_dtype in (F32, F64):
        other = F64 if chain_dtype == F32 else F32
        robot = kinfast.Robot(compile_robot(parse_urdf_string(DYN_ARM), dtype=chain_dtype))
        q = torch.tensor([[0.3, -0.7]], dtype=chain_dtype)
        v = torch.tensor([[0.1, 0.2]], dtype=other)

        assert robot.inverse_dynamics(q, v, q).dtype == chain_dtype
        assert robot.inverse_dynamics(q, q, v).dtype == chain_dtype
        assert robot.forward_dynamics(q, v, q).dtype == chain_dtype
        assert robot.forward_dynamics(q, q, v).dtype == chain_dtype

        # target in the other dtype with a seed in the chain dtype: seed wins
        target = robot.fk(q).to(other)
        qs, info = robot.ik(target, q0=q, pos_only=True, iters=3)
        assert qs.dtype == chain_dtype and info["final_error"].dtype == chain_dtype
        # no seed: the target's dtype is the working dtype
        qs, _ = robot.ik(target, pos_only=True, iters=3)
        assert qs.dtype == other

        # trajectory: vmax lives on the chain, qf arrives in another dtype
        t, qs, qd, qdd, T = robot.point_to_point(q[0], v[0])
        assert qs.dtype == chain_dtype and t.dtype == chain_dtype

        # simulate with a velocity in the other dtype
        ts, qs, qds = control.simulate(robot.chain, q, v,
                                       lambda t_, q_, qd_: torch.zeros_like(q_),
                                       dt=1e-3, steps=2)
        assert qs.dtype == chain_dtype and qds.dtype == chain_dtype


def test_random_configs_dtype_argument():
    torch.manual_seed(0)
    robot = kinfast.load_string(TWO_LINK)
    assert robot.random_configs(4).dtype == F32
    q = robot.random_configs(4, dtype=F64)
    assert q.dtype == F64 and q.shape == (4, 2)
    assert (q >= -2.9).all() and (q <= 2.9).all()
