# tests/test_mimic.py
"""Joints driven by other joints.

URDF spells this <mimic>, and it is how every parallel gripper is written: one
actuator, two fingers, q_this = multiplier * q_source + offset. The tag used to
be ignored outright, so a Robotiq or Panda hand reported two degrees of freedom
instead of one and random_configs sampled the fingers independently, producing
states the hardware cannot reach. Planning and IK then explored those states
happily, which is the worst kind of bug: no error, just answers that are wrong.

The relation is folded into the joint rather than bolted onto the chain. A
mimic joint points at its driver's q slot and carries a scale and offset, so it
costs no degree of freedom and every consumer that indexes through q_index
keeps working. Ordinary joints sit at scale 1 and offset 0.
"""
import pytest
import torch

import kinfast
from kinfast.dynamics import gravity, mass_matrix
from kinfast.dynamics_rnea import crba, gravity_torque
from kinfast.fk import forward_kinematics
from kinfast.jacobian import jacobian

GRIPPER = """
<robot name="gripper">
  <link name="base"/><link name="left"/><link name="right"/>
  <joint name="finger_left" type="prismatic">
    <parent link="base"/><child link="left"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.04" velocity="1" effort="10"/></joint>
  <joint name="finger_right" type="prismatic">
    <parent link="base"/><child link="right"/><axis xyz="1 0 0"/>
    <mimic joint="finger_left" multiplier="-1.0" offset="0.0"/>
    <limit lower="-0.04" upper="0" velocity="1" effort="10"/></joint>
</robot>"""

INERTIAL = """<inertial><mass value="{m}"/><origin xyz="0.5 0 0"/>
    <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>"""

SERIAL = """
<robot name="serial">
  <link name="a">%s</link>
  <link name="b">%s</link>
  <link name="c">%s</link>
  <joint name="j1" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
  <joint name="j2" type="revolute"><parent link="b"/><child link="c"/>
    <origin xyz="1 0 0"/><axis xyz="0 1 0"/>
    <mimic joint="j1" multiplier="0.5" offset="0.1"/>
    <limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""" % (INERTIAL.format(m=1), INERTIAL.format(m=2), INERTIAL.format(m=3))


@pytest.fixture
def serial():
    """A mimic joint in series with the joint driving it, so the shared q
    column really does take two contributions. The parallel gripper never
    exercises that, because its fingers are siblings."""
    return kinfast.load_string(SERIAL, dtype=torch.float64)


def test_a_mimic_joint_is_not_a_degree_of_freedom():
    r = kinfast.load_string(GRIPPER)
    assert r.dof == 1
    assert r.joint_names == ["finger_left"]
    assert r.chain.has_mimic


def test_an_ordinary_chain_reports_no_mimic():
    plain = SERIAL.replace(
        '<mimic joint="j1" multiplier="0.5" offset="0.1"/>', "")
    r = kinfast.load_string(plain)
    assert r.dof == 2
    assert not r.chain.has_mimic


def test_the_driven_joint_follows():
    """One input moves both fingers, mirrored."""
    r = kinfast.load_string(GRIPPER)
    w = forward_kinematics(r.chain, torch.tensor([[0.03]]))
    assert float(w[0, r.chain.link_index["left"], 0, 3]) == pytest.approx(
        0.03, abs=1e-6)
    assert float(w[0, r.chain.link_index["right"], 0, 3]) == pytest.approx(
        -0.03, abs=1e-6)


def test_random_configs_cannot_produce_an_impossible_state():
    """The point of all this. Sampling used to treat the fingers as
    independent, which put the gripper in states no hardware can reach."""
    r = kinfast.load_string(GRIPPER)
    q = r.random_configs(64)
    assert q.shape == (64, 1)
    w = forward_kinematics(r.chain, q)
    left = w[:, r.chain.link_index["left"], 0, 3]
    right = w[:, r.chain.link_index["right"], 0, 3]
    assert torch.allclose(left, -right, atol=1e-6)


def test_the_driven_joint_limit_restricts_the_driver():
    """The driver may only go as far as the joint it drives allows."""
    tight = GRIPPER.replace(
        '<limit lower="-0.04" upper="0" velocity="1" effort="10"/>',
        '<limit lower="-0.01" upper="0" velocity="1" effort="10"/>')
    r = kinfast.load_string(tight)
    assert float(r.chain.upper[0]) == pytest.approx(0.01, abs=1e-9)


def test_jacobian_matches_finite_differences(serial):
    """The acid test for the scale factor, on the in-series case where the
    shared column takes two contributions."""
    li = serial.chain.link_index["c"]
    q = torch.tensor([[0.37]], dtype=torch.float64)
    eps = 1e-6
    fd = (forward_kinematics(serial.chain, q + eps)[0, li, :3, 3]
          - forward_kinematics(serial.chain, q - eps)[0, li, :3, 3]) / (2 * eps)
    J = jacobian(serial.chain, q, li)
    assert torch.allclose(J[0, :3, 0], fd, atol=1e-7)


def _potential(chain, qv):
    w = forward_kinematics(chain, qv)
    g = torch.tensor(chain.gravity, dtype=torch.float64)
    U = 0.0
    for i in range(chain.n_links):
        m = float(chain.link_mass[i])
        if m == 0:
            continue
        com_w = w[0, i, :3, 3] + w[0, i, :3, :3] @ chain.link_com[i]
        U = U - m * (g @ com_w)
    return U


def test_gravity_torque_matches_the_gradient_of_potential_energy(serial):
    """An oracle that knows nothing about how the dynamics are implemented."""
    q = torch.tensor([[0.37]], dtype=torch.float64)
    eps = 1e-6
    fd = float((_potential(serial.chain, q + eps)
                - _potential(serial.chain, q - eps)) / (2 * eps))
    assert float(gravity(serial.chain, q)[0, 0]) == pytest.approx(fd, abs=1e-6)


def test_the_two_dynamics_routes_agree(serial):
    """RNEA and the Jacobian route are independent implementations, so them
    disagreeing is how the mimic bug showed itself in dynamics."""
    q = torch.tensor([[0.37]], dtype=torch.float64)
    assert float(gravity_torque(serial.chain, q)[0, 0]) == pytest.approx(
        float(gravity(serial.chain, q)[0, 0]), abs=1e-9)
    assert float(crba(serial.chain, q)[0, 0, 0]) == pytest.approx(
        float(mass_matrix(serial.chain, q)[0, 0, 0]), abs=1e-9)


def test_a_chain_of_mimics_is_composed_not_read_once():
    """j3 mimics j2 which mimics j1, so j3 moves at 0.5 * 0.5 of j1."""
    urdf = """
<robot name="chained">
  <link name="a"/><link name="b"/><link name="c"/><link name="d"/>
  <joint name="j1" type="revolute"><parent link="a"/><child link="b"/>
    <axis xyz="0 0 1"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
  <joint name="j2" type="revolute"><parent link="b"/><child link="c"/>
    <axis xyz="0 0 1"/><mimic joint="j1" multiplier="0.5" offset="0"/>
    <limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
  <joint name="j3" type="revolute"><parent link="c"/><child link="d"/>
    <axis xyz="0 0 1"/><mimic joint="j2" multiplier="0.5" offset="0"/>
    <limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>"""
    r = kinfast.load_string(urdf, dtype=torch.float64)
    assert r.dof == 1
    scale = {r.chain.link_names[i]: float(r.chain.joint_scale[i])
             for i in range(r.chain.n_links)}
    assert scale["c"] == pytest.approx(0.5)
    assert scale["d"] == pytest.approx(0.25)


def test_a_mimic_cycle_is_refused():
    urdf = """
<robot name="cycle">
  <link name="a"/><link name="b"/><link name="c"/>
  <joint name="j1" type="revolute"><parent link="a"/><child link="b"/>
    <axis xyz="0 0 1"/><mimic joint="j2" multiplier="1" offset="0"/>
    <limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
  <joint name="j2" type="revolute"><parent link="b"/><child link="c"/>
    <axis xyz="0 0 1"/><mimic joint="j1" multiplier="1" offset="0"/>
    <limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>"""
    with pytest.raises(ValueError, match="itself"):
        kinfast.load_string(urdf)


def test_a_mimic_of_a_joint_that_does_not_exist_is_refused():
    urdf = GRIPPER.replace('joint="finger_left" multiplier="-1.0"',
                           'joint="nope" multiplier="-1.0"')
    with pytest.raises(ValueError, match="does not exist"):
        kinfast.load_string(urdf)
