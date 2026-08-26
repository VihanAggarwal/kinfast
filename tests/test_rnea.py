# tests/test_rnea.py
"""Recursive Newton-Euler / composite-rigid-body dynamics, checked against
oracles that do not share code with it:

- closed-form torques for a single pendulum and a vertical slide, done by hand
- MuJoCo's qfrc_bias and mj_fullM on an MJCF arm with a sliding joint
- float64 finite differences for the gradient of tau w.r.t. q
- the existing Jacobian/autograd implementation in kinfast.dynamics, which is a
  genuinely different derivation (energy based) rather than a rearrangement

Everything is seeded and self-contained; the MuJoCo tests skip if it is missing.
"""
import time

import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.urdf.parse import parse_urdf_string
from kinfast import dynamics as D
from kinfast import dynamics_rnea as R

torch.manual_seed(0)

# Planar double pendulum, the same fixture tests/test_dynamics.py uses.
DYN_ARM = """
<robot name="dpend">
  <link name="base"/>
  <link name="l1">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.02" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l2">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.02" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.1" upper="3.1" velocity="5" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.1" upper="3.1" velocity="5" effort="50"/></joint>
</robot>
"""

# Revolute + prismatic mix with tilted joint frames, a fixed link in the middle,
# an off-center COM and fully populated (non-diagonal) inertia tensors. This is
# the case a revolute-only test would never exercise.
MIXED = """
<robot name="mixed">
  <link name="base"/>
  <link name="a">
    <inertial><origin xyz="0.05 -0.02 0.11" rpy="0.2 -0.1 0.3"/><mass value="3.4"/>
      <inertia ixx="0.06" iyy="0.05" izz="0.04" ixy="0.006" ixz="-0.004" iyz="0.003"/>
    </inertial>
  </link>
  <link name="b">
    <inertial><origin xyz="0.0 0.13 0.02"/><mass value="1.9"/>
      <inertia ixx="0.03" iyy="0.02" izz="0.025" ixy="-0.002" ixz="0.001" iyz="0.004"/>
    </inertial>
  </link>
  <link name="mount">
    <inertial><origin xyz="0.01 0 0.03"/><mass value="0.45"/>
      <inertia ixx="0.002" iyy="0.002" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="c">
    <inertial><origin xyz="0.2 0.01 -0.03" rpy="-0.4 0.25 0.1"/><mass value="0.85"/>
      <inertia ixx="0.008" iyy="0.011" izz="0.009" ixy="0.001" ixz="0.0007" iyz="-0.0005"/>
    </inertial>
  </link>
  <link name="d">
    <inertial><origin xyz="0 0 0.09"/><mass value="0.6"/>
      <inertia ixx="0.004" iyy="0.004" izz="0.002" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="s1" type="prismatic"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-0.3" upper="0.4" velocity="1" effort="100"/></joint>
  <joint name="r1" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0.12 0 0.2" rpy="0.3 0 -0.2"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="3" effort="60"/></joint>
  <joint name="fix" type="fixed"><parent link="b"/><child link="mount"/>
    <origin xyz="0 0.26 0" rpy="0 0.5 0"/></joint>
  <joint name="s2" type="prismatic"><parent link="mount"/><child link="c"/>
    <origin xyz="0.03 0 0.05" rpy="-0.15 0.1 0.25"/><axis xyz="1 0 0"/>
    <limit lower="-0.15" upper="0.25" velocity="1" effort="80"/></joint>
  <joint name="r2" type="revolute"><parent link="c"/><child link="d"/>
    <origin xyz="0.3 0.02 0" rpy="0 -0.35 0.2"/><axis xyz="0.3 0.6 -0.2"/>
    <limit lower="-2.5" upper="2.5" velocity="3" effort="40"/></joint>
</robot>
"""


def _chain(xml, dtype=torch.float64):
    return compile_robot(parse_urdf_string(xml), dtype=dtype)


def _sample(chain, B, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    lo = chain.lower.to(dtype)
    hi = chain.upper.to(dtype)
    q = lo + (hi - lo) * torch.rand(B, chain.dof, generator=g, dtype=dtype)
    qd = torch.randn(B, chain.dof, generator=g, dtype=dtype)
    qdd = torch.randn(B, chain.dof, generator=g, dtype=dtype)
    return q, qd, qdd


def _long_chain_urdf(n_moving=28):
    """A serial chain of n_moving+1 links alternating revolute and prismatic."""
    parts = ['<robot name="long"><link name="base"/>']
    for i in range(n_moving):
        parts.append(
            f'<link name="l{i}"><inertial>'
            f'<origin xyz="0.0{(i % 7) + 1} 0.02 0.0{(i % 5) + 1}"/>'
            f'<mass value="{0.4 + 0.05 * (i % 6):.3f}"/>'
            f'<inertia ixx="0.01" iyy="0.012" izz="0.008" '
            f'ixy="0.0005" ixz="-0.0004" iyz="0.0003"/></inertial></link>')
    for i in range(n_moving):
        parent = "base" if i == 0 else f"l{i - 1}"
        if i % 4 == 3:
            jtype, axis, lim = "prismatic", "0 0 1", (-0.1, 0.1)
        else:
            axis = ["1 0 0", "0 1 0", "0 0 1"][i % 3]
            jtype, lim = "revolute", (-1.5, 1.5)
        parts.append(
            f'<joint name="j{i}" type="{jtype}">'
            f'<parent link="{parent}"/><child link="l{i}"/>'
            f'<origin xyz="0.11 0.03 0.07" rpy="0.1 -0.05 0.08"/>'
            f'<axis xyz="{axis}"/>'
            f'<limit lower="{lim[0]}" upper="{lim[1]}" velocity="2" effort="50"/></joint>')
    parts.append("</robot>")
    return "".join(parts)


# --------------------------------------------------------------------------
# hand-computed closed forms
# --------------------------------------------------------------------------

PENDULUM = """
<robot name="pend">
  <link name="base"/>
  <link name="l1">
    <inertial><origin xyz="0.7 0 0"/><mass value="2.0"/>
      <inertia ixx="0.01" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" velocity="5" effort="50"/></joint>
</robot>
"""


def test_pendulum_closed_form():
    """tau = (I_yy + m L^2) qdd - m g L cos(q) for a rod swinging about +y."""
    chain = _chain(PENDULUM)
    m, L, Iyy, g = 2.0, 0.7, 0.05, 9.81
    q = torch.tensor([[0.0], [0.4], [-1.1], [2.3]], dtype=torch.float64)
    qdd = torch.tensor([[0.0], [1.5], [-0.7], [3.0]], dtype=torch.float64)
    qd = torch.zeros_like(q)
    tau = R.rnea(chain, q, qd, qdd)
    want = (Iyy + m * L * L) * qdd - m * g * L * torch.cos(q)
    assert torch.allclose(tau, want, atol=1e-12)
    # M is the scalar (I + m L^2) and does not depend on q.
    M = R.crba(chain, q)
    assert torch.allclose(M, torch.full_like(M, Iyy + m * L * L), atol=1e-12)


SLIDE = """
<robot name="slide">
  <link name="base"/>
  <link name="l1">
    <inertial><origin xyz="0 0 0.25"/><mass value="1.6"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="prismatic"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="50"/></joint>
</robot>
"""


def test_vertical_slide_closed_form():
    """A mass on a vertical rail: tau = m (qdd + g), independent of q and qd."""
    chain = _chain(SLIDE)
    m, g = 1.6, 9.81
    q = torch.tensor([[-0.4], [0.0], [0.9]], dtype=torch.float64)
    qd = torch.tensor([[1.3], [-2.0], [0.5]], dtype=torch.float64)
    qdd = torch.tensor([[0.0], [2.5], [-1.25]], dtype=torch.float64)
    tau = R.rnea(chain, q, qd, qdd)
    assert torch.allclose(tau, m * (qdd + g), atol=1e-12)
    assert torch.allclose(R.rnea(chain, q, qd, qdd, gravity=False), m * qdd, atol=1e-12)
    assert torch.allclose(R.crba(chain, q), torch.full((3, 1, 1), m, dtype=torch.float64),
                          atol=1e-12)


# --------------------------------------------------------------------------
# agreement with the energy-based implementation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("xml,name", [(DYN_ARM, "dyn_arm"), (MIXED, "mixed")],
                         ids=["dyn_arm", "mixed"])
def test_crba_matches_mass_matrix(xml, name):
    chain = _chain(xml)
    q, _, _ = _sample(chain, 6, seed=11)
    M_ref = D.mass_matrix(chain, q)
    M = R.crba(chain, q)
    err = (M - M_ref).abs().max().item()
    assert err < 1e-8, f"{name}: mass matrix mismatch {err:.3e}"


@pytest.mark.parametrize("xml,name", [(DYN_ARM, "dyn_arm"), (MIXED, "mixed")],
                         ids=["dyn_arm", "mixed"])
@pytest.mark.parametrize("use_gravity", [True, False])
def test_rnea_matches_inverse_dynamics(xml, name, use_gravity):
    chain = _chain(xml)
    q, qd, qdd = _sample(chain, 6, seed=12)
    ref = D.inverse_dynamics(chain, q, qd, qdd, use_gravity=use_gravity)
    tau = R.rnea(chain, q, qd, qdd, gravity=use_gravity)
    err = (tau - ref).abs().max().item()
    assert err < 1e-8, f"{name}: torque mismatch {err:.3e}"


def test_bias_and_gravity_helpers():
    chain = _chain(MIXED)
    q, qd, _ = _sample(chain, 5, seed=13)
    zero = torch.zeros_like(q)
    assert torch.allclose(R.gravity_torque(chain, q), D.gravity(chain, q), atol=1e-9)
    assert torch.allclose(R.bias(chain, q, qd),
                          D.coriolis(chain, q, qd) + D.gravity(chain, q), atol=1e-9)
    # bias with zero velocity is pure gravity
    assert torch.allclose(R.bias(chain, q, zero), R.gravity_torque(chain, q), atol=1e-12)


def test_gravity_argument_forms():
    """gravity=False, a scalar, and an explicit vector all mean what they say."""
    chain = _chain(MIXED)
    q, qd, qdd = _sample(chain, 3, seed=14)
    free = R.rnea(chain, q, qd, qdd, gravity=False)
    assert torch.allclose(R.rnea(chain, q, qd, qdd, gravity=None), free, atol=1e-14)
    assert torch.allclose(R.rnea(chain, q, qd, qdd, gravity=0.0), free, atol=1e-14)
    scalar = R.rnea(chain, q, qd, qdd, gravity=9.81)
    vector = R.rnea(chain, q, qd, qdd, gravity=(0.0, 0.0, -9.81))
    tensor = R.rnea(chain, q, qd, qdd,
                    gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64))
    default = R.rnea(chain, q, qd, qdd)
    for other in (vector, tensor, default):
        assert torch.allclose(scalar, other, atol=1e-12)
    # a sideways gravity vector must change the answer
    tilt = R.rnea(chain, q, qd, qdd, gravity=(3.0, 0.0, -9.0))
    assert (tilt - default).abs().max() > 1e-3
    with pytest.raises(ValueError):
        R.rnea(chain, q, qd, qdd, gravity=(0.0, -9.81))


def test_forward_dynamics_roundtrip():
    chain = _chain(MIXED)
    q, qd, qdd = _sample(chain, 6, seed=15)
    tau = R.rnea(chain, q, qd, qdd)
    back = R.forward_dynamics(chain, q, qd, tau)
    assert torch.allclose(back, qdd, atol=1e-9)
    # and it agrees with the existing forward dynamics
    ref = D.forward_dynamics(chain, q, qd, tau)
    assert torch.allclose(back, ref, atol=1e-8)


def test_mass_matrix_structure():
    """M from CRBA is symmetric and positive definite, and tau is linear in qdd
    with M as the coefficient (tau(qdd) - tau(0) = M qdd)."""
    chain = _chain(MIXED)
    q, qd, qdd = _sample(chain, 4, seed=16)
    M = R.crba(chain, q)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-14)
    assert (torch.linalg.eigvalsh(M) > 1e-8).all()
    lhs = R.rnea(chain, q, qd, qdd) - R.rnea(chain, q, qd, torch.zeros_like(qdd))
    assert torch.allclose(lhs, (M @ qdd.unsqueeze(-1)).squeeze(-1), atol=1e-10)


# --------------------------------------------------------------------------
# batching, dtype, differentiability
# --------------------------------------------------------------------------

def test_batch_rows_are_independent():
    chain = _chain(MIXED)
    q, qd, qdd = _sample(chain, 7, seed=17)
    tau = R.rnea(chain, q, qd, qdd)
    M = R.crba(chain, q)
    for b in range(7):
        t1 = R.rnea(chain, q[b:b + 1], qd[b:b + 1], qdd[b:b + 1])
        m1 = R.crba(chain, q[b:b + 1])
        assert torch.allclose(t1[0], tau[b], atol=1e-12)
        assert torch.allclose(m1[0], M[b], atol=1e-12)


def test_dtype_follows_q():
    """float32 q gives float32 results, and qd/qdd in another dtype are adopted
    rather than crashing."""
    chain = compile_robot(parse_urdf_string(MIXED), dtype=torch.float32)
    q, qd, qdd = _sample(chain, 3, seed=18)
    q32 = q.float()
    tau32 = R.rnea(chain, q32, qd, qdd)             # qd/qdd are float64 here
    assert tau32.dtype == torch.float32
    assert R.crba(chain, q32).dtype == torch.float32
    # float64 on the same chain still works and is close to the float32 answer
    tau64 = R.rnea(chain, q, qd, qdd)
    assert tau64.dtype == torch.float64
    rel = (tau64 - tau32.double()).abs().max() / tau64.abs().max()
    assert rel < 1e-4


def test_gradients_match_finite_differences():
    """d(sum tau)/dq and d(sum tau)/dqd from autograd vs central differences."""
    chain = _chain(MIXED)
    q, qd, qdd = _sample(chain, 2, seed=19)
    weight = torch.arange(1, chain.dof + 1, dtype=torch.float64)

    def loss(qv, qdv):
        return (R.rnea(chain, qv, qdv, qdd) * weight).sum()

    qg = q.clone().requires_grad_(True)
    qdg = qd.clone().requires_grad_(True)
    out = loss(qg, qdg)
    out.backward()

    eps = 1e-6
    for var, grad, base in ((0, qg.grad, q), (1, qdg.grad, qd)):
        fd = torch.zeros_like(base)
        for b in range(base.shape[0]):
            for k in range(chain.dof):
                d = torch.zeros_like(base)
                d[b, k] = eps
                if var == 0:
                    up, dn = loss(q + d, qd), loss(q - d, qd)
                else:
                    up, dn = loss(q, qd + d), loss(q, qd - d)
                fd[b, k] = (up - dn) / (2 * eps)
        assert torch.allclose(grad, fd, atol=1e-6), f"gradient mismatch for var {var}"


def test_crba_gradient_matches_finite_differences():
    chain = _chain(MIXED)
    q, _, _ = _sample(chain, 1, seed=20)
    w = torch.randn(chain.dof, chain.dof, generator=torch.Generator().manual_seed(3),
                    dtype=torch.float64)

    def loss(qv):
        return (R.crba(chain, qv)[0] * w).sum()

    qg = q.clone().requires_grad_(True)
    loss(qg).backward()
    eps = 1e-6
    fd = torch.zeros_like(q)
    for k in range(chain.dof):
        d = torch.zeros_like(q)
        d[0, k] = eps
        fd[0, k] = (loss(q + d) - loss(q - d)) / (2 * eps)
    assert torch.allclose(qg.grad, fd, atol=1e-6)


def test_no_grad_and_massless_errors():
    chain = _chain(MIXED)
    q, qd, qdd = _sample(chain, 2, seed=21)
    with torch.no_grad():
        tau = R.rnea(chain, q, qd, qdd)
    assert torch.isfinite(tau).all()
    with pytest.raises(ValueError):
        R.rnea(chain, q[:, :1], qd, qdd)
    bare = _chain("""
      <robot name="bare"><link name="base"/><link name="l1"/>
      <joint name="j" type="revolute"><parent link="base"/><child link="l1"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint></robot>""")
    with pytest.raises(ValueError):
        R.crba(bare, torch.zeros(1, 1, dtype=torch.float64))
    # rnea on a massless chain is legitimately zero, not an error
    assert torch.allclose(R.rnea(bare, torch.zeros(1, 1, dtype=torch.float64),
                                 torch.zeros(1, 1, dtype=torch.float64),
                                 torch.ones(1, 1, dtype=torch.float64)),
                          torch.zeros(1, 1, dtype=torch.float64))


# --------------------------------------------------------------------------
# MuJoCo oracle
# --------------------------------------------------------------------------

MJ_ARM = """
<mujoco model="rneaarm">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="l1" pos="0 0 0.3">
      <inertial pos="0.1 0.02 0" quat="0.9659258 0 0 0.2588190" mass="2.5"
                diaginertia="0.04 0.03 0.02"/>
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
      <body name="l2" pos="0.4 0 0" euler="0 0.3 0">
        <inertial pos="0.2 0 0.05" mass="1.2" diaginertia="0.02 0.015 0.008"/>
        <joint name="j2" type="hinge" axis="0 1 0" range="-2 2"/>
        <body name="l3" pos="0.35 0 0">
          <inertial pos="0.1 0 0" quat="0.7071068 0.7071068 0 0" mass="0.6"
                    diaginertia="0.004 0.003 0.001"/>
          <joint name="j3" type="slide" axis="1 0 0" range="-0.2 0.2"/>
          <body name="l4" pos="0.15 0 0" euler="0.2 0 -0.4">
            <inertial pos="0.05 0.01 0" mass="0.3" diaginertia="0.001 0.0012 0.0008"/>
            <joint name="j4" type="hinge" axis="1 0 0" range="-3 3"/>
            <body name="l5" pos="0.12 0 0.02">
              <inertial pos="0 0 0.04" mass="0.25" diaginertia="0.0008 0.0008 0.0004"/>
              <joint name="j5" type="slide" axis="0 0 1" range="-0.1 0.15"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_rnea_matches_mujoco():
    """MuJoCo computes mj_fullM and qfrc_bias; kinfast must agree, and its full
    inverse dynamics must equal M qdd + bias in MuJoCo's own numbers."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np
    from kinfast.mjcf.parse import parse_mjcf_string
    from kinfast.urdf.repair import repair

    m = mujoco.MjModel.from_xml_string(MJ_ARM)
    m.dof_armature[:] = 0.0          # kinfast has no armature concept
    d = mujoco.MjData(m)
    ir, _ = repair(parse_mjcf_string(MJ_ARM))
    chain = compile_robot(ir, dtype=torch.float64)

    addr = {}
    for j in range(m.njnt):
        addr[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = \
            (m.jnt_qposadr[j], m.jnt_dofadr[j])
    cols = [(k, addr[nm]) for k, nm in enumerate(chain.joint_names)]
    order = [va for _, (_, va) in cols]

    rng = np.random.RandomState(0)
    lo, hi = chain.lower.numpy(), chain.upper.numpy()
    for _ in range(10):
        q = lo + (hi - lo) * rng.rand(chain.dof)
        qd = rng.randn(chain.dof)
        qdd = rng.randn(chain.dof)
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        for k, (qa, va) in cols:
            d.qpos[qa] = q[k]
            d.qvel[va] = qd[k]
        mujoco.mj_forward(m, d)
        M_mj = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M_mj)
        M_mj = M_mj[np.ix_(order, order)]
        bias_mj = d.qfrc_bias[order]
        tau_mj = M_mj @ qdd + bias_mj

        qt = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
        qdt = torch.tensor(qd, dtype=torch.float64).unsqueeze(0)
        qddt = torch.tensor(qdd, dtype=torch.float64).unsqueeze(0)

        M_kf = R.crba(chain, qt)[0].numpy()
        bias_kf = R.bias(chain, qt, qdt)[0].numpy()
        tau_kf = R.rnea(chain, qt, qdt, qddt)[0].numpy()

        tol = 2e-4
        assert np.abs(M_kf - M_mj).max() < tol * max(1.0, np.abs(M_mj).max())
        assert np.abs(bias_kf - bias_mj).max() < tol * max(1.0, np.abs(bias_mj).max())
        assert np.abs(tau_kf - tau_mj).max() < tol * max(1.0, np.abs(tau_mj).max())


# --------------------------------------------------------------------------
# 29-link chain: correctness at scale, plus the speedup over kinfast.dynamics
# --------------------------------------------------------------------------

def test_long_chain_matches_reference():
    chain = _chain(_long_chain_urdf(28))
    assert chain.n_links == 29 and chain.dof == 28
    q, qd, qdd = _sample(chain, 2, seed=22)
    M_err = (R.crba(chain, q) - D.mass_matrix(chain, q)).abs().max().item()
    tau_err = (R.rnea(chain, q, qd, qdd)
               - D.inverse_dynamics(chain, q, qd, qdd)).abs().max().item()
    assert M_err < 1e-8, f"mass matrix mismatch {M_err:.3e}"
    assert tau_err < 1e-8, f"torque mismatch {tau_err:.3e}"


def _time(fn, reps):
    fn()                                  # warm the per-dtype caches
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def test_long_chain_speedup(capsys):
    """RNEA/CRBA must beat the Jacobian+autograd path on a 29-link chain."""
    chain = _chain(_long_chain_urdf(28), dtype=torch.float32)
    q, qd, qdd = _sample(chain, 8, seed=23, dtype=torch.float32)
    reps = 5
    t_old_M = _time(lambda: D.mass_matrix(chain, q), reps)
    t_new_M = _time(lambda: R.crba(chain, q), reps)
    t_old_t = _time(lambda: D.inverse_dynamics(chain, q, qd, qdd), reps)
    t_new_t = _time(lambda: R.rnea(chain, q, qd, qdd), reps)
    with capsys.disabled():
        print(f"\n29 links / 28 dof / batch 8, float32, mean of {reps} runs:")
        print(f"  mass matrix : dynamics {t_old_M * 1e3:8.2f} ms  "
              f"crba {t_new_M * 1e3:8.2f} ms  speedup {t_old_M / t_new_M:6.2f}x")
        print(f"  inverse dyn : dynamics {t_old_t * 1e3:8.2f} ms  "
              f"rnea {t_new_t * 1e3:8.2f} ms  speedup {t_old_t / t_new_t:6.2f}x")
    assert t_new_M < t_old_M
    assert t_new_t < 0.5 * t_old_t


# --------------------------------------------------------------------------
# branching trees: the backward sweep and the CRBA ancestor walk only really
# get exercised when a link has more than one child
# --------------------------------------------------------------------------

BRANCH = """
<robot name="branch">
  <link name="base"/>
  <link name="torso">
    <inertial><origin xyz="0 0 0.1"/><mass value="4.0"/>
      <inertia ixx="0.08" iyy="0.07" izz="0.05" ixy="0.004" ixz="0.002" iyz="-0.003"/>
    </inertial>
  </link>
  <link name="lift">
    <inertial><origin xyz="0.02 0 0.06" rpy="0.1 0.2 -0.3"/><mass value="1.1"/>
      <inertia ixx="0.02" iyy="0.018" izz="0.01" ixy="0.001" ixz="0" iyz="0.0005"/>
    </inertial>
  </link>
  <link name="armA1">
    <inertial><origin xyz="0.15 0 0"/><mass value="0.9"/>
      <inertia ixx="0.004" iyy="0.012" izz="0.012" ixy="0" ixz="0.0006" iyz="0"/>
    </inertial>
  </link>
  <link name="armA2">
    <inertial><origin xyz="0.1 0.01 0" rpy="0 0.4 0"/><mass value="0.5"/>
      <inertia ixx="0.002" iyy="0.005" izz="0.005" ixy="0.0002" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="armB1">
    <inertial><origin xyz="0 0.12 0"/><mass value="0.7"/>
      <inertia ixx="0.006" iyy="0.002" izz="0.006" ixy="-0.0003" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="pad">
    <inertial><origin xyz="0 0.03 0"/><mass value="0.2"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="armB2">
    <inertial><origin xyz="0 0 0.08" rpy="-0.2 0 0.15"/><mass value="0.35"/>
      <inertia ixx="0.0015" iyy="0.0015" izz="0.0008" ixy="0" ixz="0.0001" iyz="0"/>
    </inertial>
  </link>
  <link name="tail">
    <inertial><origin xyz="-0.1 0 0"/><mass value="0.8"/>
      <inertia ixx="0.003" iyy="0.009" izz="0.009" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="yaw" type="revolute"><parent link="base"/><child link="torso"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 0 1"/>
    <limit lower="-2.5" upper="2.5" velocity="3" effort="80"/></joint>
  <!-- second child of the base: two subtrees hang off the root -->
  <joint name="tailj" type="revolute"><parent link="base"/><child link="tail"/>
    <origin xyz="-0.15 0 0.05" rpy="0 0.3 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.2" upper="1.2" velocity="3" effort="30"/></joint>
  <joint name="liftj" type="prismatic"><parent link="torso"/><child link="lift"/>
    <origin xyz="0 0 0.25" rpy="0.1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-0.2" upper="0.3" velocity="1" effort="120"/></joint>
  <!-- branch A off lift -->
  <joint name="a1" type="revolute"><parent link="lift"/><child link="armA1"/>
    <origin xyz="0.1 0.05 0.1" rpy="0 -0.4 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="3" effort="40"/></joint>
  <joint name="a2" type="prismatic"><parent link="armA1"/><child link="armA2"/>
    <origin xyz="0.3 0 0" rpy="0.2 0 -0.1"/><axis xyz="1 0 0"/>
    <limit lower="-0.1" upper="0.2" velocity="1" effort="50"/></joint>
  <!-- branch B off lift, with a fixed link in the middle -->
  <joint name="b1" type="revolute"><parent link="lift"/><child link="armB1"/>
    <origin xyz="-0.08 -0.05 0.12" rpy="-0.3 0.15 0"/><axis xyz="0.5 0.5 0.7"/>
    <limit lower="-1.8" upper="1.8" velocity="3" effort="40"/></joint>
  <joint name="bfix" type="fixed"><parent link="armB1"/><child link="pad"/>
    <origin xyz="0 0.24 0" rpy="0 0.25 0"/></joint>
  <joint name="b2" type="revolute"><parent link="pad"/><child link="armB2"/>
    <origin xyz="0 0.06 0.02" rpy="0.4 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-2.2" upper="2.2" velocity="3" effort="20"/></joint>
</robot>
"""


def _columns(chain):
    """joint name -> column of q, for readable indexing in the tests below."""
    col = {}
    for link in range(chain.n_links):
        c = int(chain.q_index[link])
        if c >= 0:
            col[chain.joint_names[c]] = c
    return col


def test_branching_tree_matches_reference():
    """Two subtrees off the root, a branch point mid-arm and a fixed link."""
    chain = _chain(BRANCH)
    assert chain.dof == 7 and chain.n_links == 9
    q, qd, qdd = _sample(chain, 5, seed=31)
    M_err = (R.crba(chain, q) - D.mass_matrix(chain, q)).abs().max().item()
    assert M_err < 1e-8, f"mass matrix mismatch {M_err:.3e}"
    for use_gravity in (True, False):
        tau_err = (R.rnea(chain, q, qd, qdd, gravity=use_gravity)
                   - D.inverse_dynamics(chain, q, qd, qdd,
                                        use_gravity=use_gravity)).abs().max().item()
        assert tau_err < 1e-8, f"torque mismatch {tau_err:.3e}"


def test_branching_tree_coupling_is_zero_across_branches():
    """Joints in disjoint subtrees share no inertia, so their M entry is exactly 0.

    A CRBA that walked the wrong ancestors would quietly fill these in, and a
    comparison against the reference alone would not say which entry was wrong.
    """
    chain = _chain(BRANCH)
    q, _, _ = _sample(chain, 3, seed=32)
    M = R.crba(chain, q)
    col = _columns(chain)
    for a, b in (("a1", "b1"), ("a2", "b2"), ("a1", "tailj"), ("b2", "tailj")):
        assert M[:, col[a], col[b]].abs().max() < 1e-12, f"{a}/{b} should not couple"
    # while joints on a shared path do couple
    assert M[:, col["a1"], col["liftj"]].abs().max() > 1e-6


def test_branch_torque_is_local_to_the_branch():
    """Accelerating one branch alone produces no torque in the other branch."""
    chain = _chain(BRANCH)
    q, _, _ = _sample(chain, 2, seed=33)
    col = _columns(chain)
    zero = torch.zeros_like(q)
    qdd = torch.zeros_like(q)
    qdd[:, col["a2"]] = 1.0
    tau = R.rnea(chain, q, zero, qdd, gravity=False)
    assert tau[:, col["b1"]].abs().max() < 1e-12
    assert tau[:, col["b2"]].abs().max() < 1e-12
    assert tau[:, col["tailj"]].abs().max() < 1e-12
    assert tau[:, col["a2"]].abs().min() > 1e-6      # the accelerated joint feels it
    assert tau[:, col["liftj"]].abs().max() > 1e-6   # so does its ancestor


# --------------------------------------------------------------------------
# more MuJoCo: a branching model under a tilted gravity vector
# --------------------------------------------------------------------------

MJ_TREE = """
<mujoco model="rneatree">
  <compiler angle="radian"/>
  <option gravity="1.7 -2.4 -9.0"/>
  <worldbody>
    <body name="torso" pos="0 0 0.25">
      <inertial pos="0 0 0.08" quat="0.9238795 0.3826834 0 0" mass="3.0"
                diaginertia="0.05 0.045 0.03"/>
      <joint name="yaw" type="hinge" axis="0 0 1" range="-2.5 2.5"/>
      <body name="lift" pos="0 0 0.2" euler="0.1 0 0">
        <inertial pos="0.02 0 0.05" mass="1.1" diaginertia="0.02 0.018 0.01"/>
        <joint name="liftj" type="slide" axis="0 0 1" range="-0.2 0.3"/>
        <body name="armA1" pos="0.1 0.05 0.1" euler="0 -0.4 0.2">
          <inertial pos="0.15 0 0" mass="0.9" diaginertia="0.004 0.012 0.012"/>
          <joint name="a1" type="hinge" axis="0 1 0" range="-2 2"/>
          <body name="armA2" pos="0.3 0 0" euler="0.2 0 -0.1">
            <inertial pos="0.1 0.01 0" quat="0.9800666 0 0.1986693 0" mass="0.5"
                      diaginertia="0.002 0.005 0.005"/>
            <joint name="a2" type="slide" axis="1 0 0" range="-0.1 0.2"/>
          </body>
        </body>
        <body name="armB1" pos="-0.08 -0.05 0.12" euler="-0.3 0.15 0">
          <inertial pos="0 0.12 0" mass="0.7" diaginertia="0.006 0.002 0.006"/>
          <joint name="b1" type="hinge" axis="0.5812382 0.5812382 0.5695334"
                 range="-1.8 1.8"/>
          <body name="armB2" pos="0 0.3 0.02" euler="0.4 0 0">
            <inertial pos="0 0 0.08" mass="0.35" diaginertia="0.0015 0.0015 0.0008"/>
            <joint name="b2" type="hinge" axis="1 0 0" range="-2.2 2.2"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _mujoco_compare(xml, gravity, seed=0, n=8, tol=2e-4):
    """Compare crba/bias/rnea against mj_fullM and qfrc_bias at random states."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np
    from kinfast.mjcf.parse import parse_mjcf_string
    from kinfast.urdf.repair import repair

    m = mujoco.MjModel.from_xml_string(xml)
    m.dof_armature[:] = 0.0          # kinfast has no armature concept
    d = mujoco.MjData(m)
    ir, _ = repair(parse_mjcf_string(xml))
    chain = compile_robot(ir, dtype=torch.float64)

    addr = {}
    for j in range(m.njnt):
        addr[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = \
            (m.jnt_qposadr[j], m.jnt_dofadr[j])
    cols = [(k, addr[nm]) for k, nm in enumerate(chain.joint_names)]
    order = [va for _, (_, va) in cols]

    rng = np.random.RandomState(seed)
    lo, hi = chain.lower.numpy(), chain.upper.numpy()
    for _ in range(n):
        q = lo + (hi - lo) * rng.rand(chain.dof)
        qd = rng.randn(chain.dof)
        qdd = rng.randn(chain.dof)
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        for k, (qa, va) in cols:
            d.qpos[qa] = q[k]
            d.qvel[va] = qd[k]
        mujoco.mj_forward(m, d)
        M_mj = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M_mj)
        M_mj = M_mj[np.ix_(order, order)]
        bias_mj = d.qfrc_bias[order]
        tau_mj = M_mj @ qdd + bias_mj

        qt = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
        qdt = torch.tensor(qd, dtype=torch.float64).unsqueeze(0)
        qddt = torch.tensor(qdd, dtype=torch.float64).unsqueeze(0)

        M_kf = R.crba(chain, qt)[0].numpy()
        bias_kf = R.bias(chain, qt, qdt, gravity=gravity)[0].numpy()
        tau_kf = R.rnea(chain, qt, qdt, qddt, gravity=gravity)[0].numpy()

        assert np.abs(M_kf - M_mj).max() < tol * max(1.0, np.abs(M_mj).max())
        assert np.abs(bias_kf - bias_mj).max() < tol * max(1.0, np.abs(bias_mj).max())
        assert np.abs(tau_kf - tau_mj).max() < tol * max(1.0, np.abs(tau_mj).max())


def test_branching_tree_matches_mujoco_tilted_gravity():
    """A branching MJCF model under gravity that is not straight down.

    The vector is passed explicitly rather than read off the model, so this
    covers the gravity argument as well as the tree handling.
    """
    _mujoco_compare(MJ_TREE, gravity=(1.7, -2.4, -9.0), seed=4)


# --------------------------------------------------------------------------
# odds and ends
# --------------------------------------------------------------------------

def test_runs_under_inference_mode():
    """RNEA needs no autograd at all, so unlike dynamics.coriolis it is happy
    inside torch.inference_mode. That is a large part of the point of it."""
    chain = _chain(MIXED)
    q, qd, qdd = _sample(chain, 4, seed=34)
    ref = R.rnea(chain, q, qd, qdd)
    refM = R.crba(chain, q)
    with torch.inference_mode():
        tau = R.rnea(chain, q, qd, qdd)
        M = R.crba(chain, q)
        assert torch.allclose(tau, ref, atol=1e-14)
        assert torch.allclose(M, refM, atol=1e-14)
    with pytest.raises(ValueError):
        with torch.inference_mode():
            D.inverse_dynamics(chain, q, qd, qdd)


def test_fixed_only_chain_has_no_torques():
    """A chain with no movable joint has zero degrees of freedom, and both entry
    points return correctly shaped empty tensors instead of failing."""
    chain = _chain("""
      <robot name="rigid"><link name="base"/>
      <link name="l1"><inertial><origin xyz="0.1 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
      </inertial></link>
      <joint name="f" type="fixed"><parent link="base"/><child link="l1"/>
        <origin xyz="0.2 0 0"/></joint></robot>""")
    assert chain.dof == 0
    empty = torch.zeros(3, 0, dtype=torch.float64)
    assert R.rnea(chain, empty, empty, empty).shape == (3, 0)
    assert R.crba(chain, empty).shape == (3, 0, 0)
