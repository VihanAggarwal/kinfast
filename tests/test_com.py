# tests/test_com.py
"""Center of mass against independent oracles.

- MuJoCo's own subtree_com on the same MJCF arm the dynamics oracle uses
- float64 central differences of com(q) for the COM Jacobian
- a hand-computed gantry (one prismatic + one revolute, two point masses)
  whose COM and its derivative are written out in closed form
- a single massive link, whose whole-body COM is just its transformed link COM
"""
import math

import pytest
import torch

from kinfast import com as C
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.urdf.parse import parse_urdf_string

from tests.test_dynamics import DYN_ARM
from tests.test_dynamics_oracle import ARM

# A prismatic base sliding along x, then a revolute elbow about z. Two point
# masses (diagonal inertia is present only because the parser wants a full
# inertial block; the COM ignores it). Written so the answer is doable by hand:
#   com(d, t) = (d + 0.6 + 0.3 cos t,  0.3 sin t,  0)
GANTRY = """
<robot name="gantry">
  <link name="base"/>
  <link name="slider">
    <inertial><origin xyz="0 0 0"/><mass value="2.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="arm">
    <inertial><origin xyz="0.5 0 0"/><mass value="3.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="slide" type="prismatic"><parent link="base"/><child link="slider"/>
    <origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" velocity="1" effort="50"/></joint>
  <joint name="elbow" type="revolute"><parent link="slider"/><child link="arm"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="1" effort="50"/></joint>
</robot>
"""

# One movable link carrying all the mass, with an off-centre, rotated inertial
# frame so a wrong transform cannot hide.
SINGLE = """
<robot name="single">
  <link name="base"/>
  <link name="body">
    <inertial><origin xyz="0.3 -0.2 0.1" rpy="0.4 0.2 -0.7"/><mass value="4.0"/>
      <inertia ixx="0.05" iyy="0.04" izz="0.03" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j" type="revolute"><parent link="base"/><child link="body"/>
    <origin xyz="0.1 0.2 0.3" rpy="0.2 -0.3 0.1"/><axis xyz="0.3 0.5 0.8"/>
    <limit lower="-3" upper="3" velocity="1" effort="50"/></joint>
</robot>
"""

MASSLESS = """
<robot name="massless">
  <link name="base"/><link name="l1"/>
  <joint name="j" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
</robot>
"""


def _chain(xml, dtype=torch.float64):
    return compile_robot(parse_urdf_string(xml), dtype=dtype)


def _random_q(chain, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo, hi = chain.lower.double(), chain.upper.double()
    u = torch.rand(n, chain.dof, generator=g, dtype=torch.float64)
    return lo + (hi - lo) * u


# ----------------------------------------------------------------- mujoco


def test_com_matches_mujoco_subtree_com():
    """MuJoCo reports subtree_com per body; the root body's subtree is the whole
    robot, so it is exactly the whole-body COM kinfast computes."""
    mujoco = pytest.importorskip("mujoco")
    from kinfast.mjcf.parse import parse_mjcf_string
    from kinfast.urdf.repair import repair

    m = mujoco.MjModel.from_xml_string(ARM)
    d = mujoco.MjData(m)
    ir, _ = repair(parse_mjcf_string(ARM))
    chain = compile_robot(ir, dtype=torch.float64)

    # mujoco total mass is an independent check on ours
    assert abs(float(C.total_mass(chain)) - float(m.body_mass.sum())) < 1e-12

    addr = {}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        addr[name] = m.jnt_qposadr[j]
    cols = [(k, addr[nm]) for k, nm in enumerate(chain.joint_names)]
    root_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l1")

    qs = _random_q(chain, 10, seed=3)
    for row in qs:
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        for k, qa in cols:
            d.qpos[qa] = float(row[k])
        mujoco.mj_forward(m, d)
        expect = d.subtree_com[root_body].copy()
        # the world body's subtree is the same set of bodies here
        assert abs(d.subtree_com[0] - expect).max() < 1e-12

        got = C.com(chain, row.unsqueeze(0))[0].numpy()
        assert abs(got - expect).max() < 1e-9, f"{got} vs {expect}"


def test_com_jacobian_matches_mujoco_subtree_com_derivative():
    """Differentiate MuJoCo's own subtree_com by central differences: an oracle
    for the Jacobian that never touches kinfast kinematics."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np
    from kinfast.mjcf.parse import parse_mjcf_string
    from kinfast.urdf.repair import repair

    m = mujoco.MjModel.from_xml_string(ARM)
    d = mujoco.MjData(m)
    chain = compile_robot(repair(parse_mjcf_string(ARM))[0], dtype=torch.float64)
    addr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)]
            for nm in chain.joint_names]

    def mj_com(qvec):
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        for k, qa in enumerate(addr):
            d.qpos[qa] = qvec[k]
        mujoco.mj_forward(m, d)
        return d.subtree_com[0].copy()

    q = _random_q(chain, 1, seed=11)[0].numpy()
    h = 1e-6
    fd = np.zeros((3, chain.dof))
    for k in range(chain.dof):
        plus, minus = q.copy(), q.copy()
        plus[k] += h
        minus[k] -= h
        fd[:, k] = (mj_com(plus) - mj_com(minus)) / (2 * h)

    J = C.com_jacobian(chain, torch.tensor(q).unsqueeze(0))[0].numpy()
    assert abs(J - fd).max() < 1e-7, f"max diff {abs(J - fd).max():.2e}"


# ------------------------------------------------------- finite differences


def _fd_com_jacobian(chain, q, h=1e-6):
    """Central-difference dc/dq in float64. (B, 3, dof)."""
    cols = []
    for k in range(chain.dof):
        e = torch.zeros_like(q)
        e[:, k] = h
        cols.append((C.com(chain, q + e) - C.com(chain, q - e)) / (2 * h))
    return torch.stack(cols, dim=-1)


@pytest.mark.parametrize("xml", [DYN_ARM, GANTRY, SINGLE])
def test_com_jacobian_matches_finite_differences(xml):
    chain = _chain(xml)
    q = _random_q(chain, 5, seed=7)
    J = C.com_jacobian(chain, q)
    fd = _fd_com_jacobian(chain, q)
    assert J.shape == (5, 3, chain.dof)
    assert (J - fd).abs().max() < 1e-8, f"max diff {(J - fd).abs().max():.2e}"


def test_com_jacobian_matches_finite_differences_on_mjcf_arm():
    """The MJCF arm mixes a slide joint with hinges and rotated inertial frames,
    which the URDF fixtures do not."""
    pytest.importorskip("mujoco")
    from kinfast.mjcf.parse import parse_mjcf_string
    from kinfast.urdf.repair import repair
    chain = compile_robot(repair(parse_mjcf_string(ARM))[0], dtype=torch.float64)
    q = _random_q(chain, 4, seed=13)
    J = C.com_jacobian(chain, q)
    fd = _fd_com_jacobian(chain, q)
    assert (J - fd).abs().max() < 1e-8, f"max diff {(J - fd).abs().max():.2e}"


# ------------------------------------------------------------ closed form


def test_gantry_com_and_jacobian_hand_computed():
    """com(d, t) = (d + 0.6 + 0.3 cos t, 0.3 sin t, 0) for the two point masses:
    2 kg at the slider origin and 3 kg half a metre out along the elbow link."""
    chain = _chain(GANTRY)
    assert chain.joint_names == ["slide", "elbow"]
    assert float(C.total_mass(chain)) == pytest.approx(5.0)
    for dslide, t in [(0.0, 0.0), (0.3, math.pi / 3), (-0.7, -1.1)]:
        q = torch.tensor([[dslide, t]], dtype=torch.float64)
        expect = torch.tensor(
            [[dslide + 0.6 + 0.3 * math.cos(t), 0.3 * math.sin(t), 0.0]],
            dtype=torch.float64)
        assert torch.allclose(C.com(chain, q), expect, atol=1e-12)
        # d/d(slide) = x_hat (all mass slides); d/d(elbow) rotates the 3 kg only
        expect_J = torch.tensor([[[1.0, -0.3 * math.sin(t)],
                                  [0.0, 0.3 * math.cos(t)],
                                  [0.0, 0.0]]], dtype=torch.float64)
        assert torch.allclose(C.com_jacobian(chain, q), expect_J, atol=1e-12)


def test_double_pendulum_com_hand_computed():
    """Two unit links, each with its COM half a metre out along its own x axis,
    rotating about y in the x-z plane. Straight trigonometry."""
    chain = _chain(DYN_ARM)
    q1, q2 = 0.4, -0.9
    q = torch.tensor([[q1, q2]], dtype=torch.float64)
    # link 1 COM: rotate (0.5, 0, 0) about +y by q1 -> (0.5 cos q1, 0, -0.5 sin q1)
    c1 = (0.5 * math.cos(q1), -0.5 * math.sin(q1))
    # link 2 frame origin at (cos q1, -sin q1) in (x, z); COM 0.5 further along
    # the accumulated angle q1 + q2
    a = q1 + q2
    c2 = (math.cos(q1) + 0.5 * math.cos(a), -math.sin(q1) - 0.5 * math.sin(a))
    expect = torch.tensor([[(c1[0] + c2[0]) / 2, 0.0, (c1[1] + c2[1]) / 2]],
                          dtype=torch.float64)
    assert torch.allclose(C.com(chain, q), expect, atol=1e-12)


def test_single_link_com_equals_transformed_link_com():
    """With one massive link the whole-body COM is exactly that link's COM
    pushed through its world transform."""
    chain = _chain(SINGLE)
    body = chain.link_index["body"]
    q = _random_q(chain, 6, seed=5)
    world = forward_kinematics(chain, q)
    local = torch.cat([chain.link_com[body], torch.ones(1, dtype=torch.float64)])
    expect = (world[:, body] @ local)[:, :3]
    assert torch.allclose(C.com(chain, q), expect, atol=1e-14)
    assert float(C.total_mass(chain)) == pytest.approx(4.0)


# --------------------------------------------------------------- behaviour


def test_com_is_batched_and_independent():
    chain = _chain(DYN_ARM)
    q = _random_q(chain, 9, seed=21)
    batch = C.com(chain, q)
    assert batch.shape == (9, 3)
    for i in range(9):
        one = C.com(chain, q[i:i + 1])
        assert torch.allclose(one[0], batch[i], atol=1e-14)
    Jb = C.com_jacobian(chain, q)
    assert Jb.shape == (9, 3, chain.dof)
    for i in range(9):
        assert torch.allclose(C.com_jacobian(chain, q[i:i + 1])[0], Jb[i],
                              atol=1e-14)


def test_working_dtype_follows_q():
    """The chain is compiled float32 here; a float64 q must still produce a
    float64 answer that matches the float64-compiled chain."""
    c32 = _chain(GANTRY, dtype=torch.float32)
    c64 = _chain(GANTRY, dtype=torch.float64)
    q64 = _random_q(c64, 3, seed=2)
    out64 = C.com(c32, q64)
    assert out64.dtype == torch.float64
    assert torch.allclose(out64, C.com(c64, q64), atol=1e-6)
    out32 = C.com(c64, q64.float())
    assert out32.dtype == torch.float32
    assert C.com_jacobian(c32, q64).dtype == torch.float64
    assert C.com_jacobian(c64, q64.float()).dtype == torch.float32
    assert C.total_mass(c32, dtype=torch.float64).dtype == torch.float64


def test_com_is_differentiable():
    """Autograd through com must reproduce the analytic Jacobian: backward of
    v . com(q) is J^T v."""
    chain = _chain(GANTRY)
    q = _random_q(chain, 2, seed=4).requires_grad_(True)
    v = torch.tensor([[0.3, -1.2, 0.7], [-0.5, 0.9, 0.1]], dtype=torch.float64)
    (C.com(chain, q) * v).sum().backward()
    J = C.com_jacobian(chain, q.detach())
    expect = (v.unsqueeze(1) @ J).squeeze(1)
    assert torch.allclose(q.grad, expect, atol=1e-12)


def test_massless_model_raises():
    chain = _chain(MASSLESS)
    q = torch.zeros(1, chain.dof, dtype=torch.float64)
    assert float(C.total_mass(chain)) == 0.0
    for fn in (C.com, C.com_jacobian):
        with pytest.raises(ValueError, match="mass"):
            fn(chain, q)


def test_bad_q_shape_raises():
    chain = _chain(GANTRY)
    for bad in (torch.zeros(2, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
                torch.zeros(1, 2, 2, dtype=torch.float64)):
        with pytest.raises(ValueError, match="shape"):
            C.com(chain, bad)
        with pytest.raises(ValueError, match="shape"):
            C.com_jacobian(chain, bad)


def test_fixed_joint_and_zero_mass_subtree_columns():
    """A joint with nothing but massless links below it moves no mass, so its
    COM Jacobian column is exactly zero; fixed joints get no column at all."""
    xml = """
    <robot name="tail">
      <link name="base"/>
      <link name="l1">
        <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
          <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
      </link>
      <link name="l2"/><link name="tip"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
        <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-3" upper="3" velocity="1" effort="1"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-3" upper="3" velocity="1" effort="1"/></joint>
      <joint name="jf" type="fixed"><parent link="l2"/><child link="tip"/>
        <origin xyz="0.5 0 0"/></joint>
    </robot>
    """
    chain = _chain(xml)
    assert chain.dof == 2
    q = _random_q(chain, 3, seed=9)
    J = C.com_jacobian(chain, q)
    assert torch.all(J[:, :, 1] == 0)
    assert torch.allclose(J, _fd_com_jacobian(chain, q), atol=1e-8)
