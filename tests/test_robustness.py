# tests/test_robustness.py
"""Ingestion robustness + regression guards.

Covers joint types and tree shapes beyond the planar-chain fixture, messy input
ordering, and the non-unit-axis bug that FK/Jacobian consistency once tripped on.
"""
import math
import torch
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.jacobian import jacobian


def _fk(urdf, q, link, dtype=torch.float64):
    chain = compile_robot(parse_urdf_string(urdf), dtype=dtype)
    world = forward_kinematics(chain, q.to(dtype))
    return chain, world[:, chain.link_index[link]]


PRISMATIC = """
<robot name="slider">
  <link name="base"/><link name="car"/>
  <joint name="j" type="prismatic"><parent link="base"/><child link="car"/>
    <origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="2" velocity="1" effort="10"/></joint>
</robot>
"""


def test_prismatic_translates_along_axis():
    q = torch.tensor([[0.75]])
    chain, ee = _fk(PRISMATIC, q, "car")
    assert torch.allclose(ee[0, :3, 3], torch.tensor([0.75, 0.0, 0.0], dtype=torch.float64), atol=1e-9)
    # orientation unchanged by a prismatic joint
    assert torch.allclose(ee[0, :3, :3], torch.eye(3, dtype=torch.float64), atol=1e-9)


def test_prismatic_jacobian_is_axis():
    chain = compile_robot(parse_urdf_string(PRISMATIC), dtype=torch.float64)
    J = jacobian(chain, torch.tensor([[0.4]], dtype=torch.float64), chain.link_index["car"])
    # Jv = +x, Jw = 0
    assert torch.allclose(J[0, :3, 0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(J[0, 3:, 0], torch.zeros(3, dtype=torch.float64), atol=1e-9)


BRANCHED = """
<robot name="branch">
  <link name="base"/><link name="left"/><link name="right"/>
  <joint name="jl" type="revolute"><parent link="base"/><child link="left"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="1" effort="10"/></joint>
  <joint name="jr" type="revolute"><parent link="base"/><child link="right"/>
    <origin xyz="-1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="1" effort="10"/></joint>
</robot>
"""


def test_branched_tree_both_children():
    chain = compile_robot(parse_urdf_string(BRANCHED), dtype=torch.float64)
    q = torch.zeros(1, 2, dtype=torch.float64)
    world = forward_kinematics(chain, q)
    left = world[0, chain.link_index["left"], :3, 3]
    right = world[0, chain.link_index["right"], :3, 3]
    assert torch.allclose(left, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(right, torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-9)


# axis declared NON-UNIT ("0 0 3") — repair is bypassed by compiling directly.
NONUNIT = """
<robot name="nonunit">
  <link name="base"/><link name="l1"/><link name="ee"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 3"/>
    <limit lower="-3" upper="3" velocity="1" effort="10"/></joint>
  <joint name="j2" type="fixed"><parent link="l1"/><child link="ee"/>
    <origin xyz="1 0 0"/></joint>
</robot>
"""


def test_nonunit_axis_regression():
    """Compile-time axis normalization keeps FK and Jacobian consistent even when
    the URDF declares a non-unit axis and repair did not run."""
    chain = compile_robot(parse_urdf_string(NONUNIT), dtype=torch.float64)
    li = chain.link_index["ee"]
    # rotating j1 by +90deg must move the ee (at +x, dist 1) to +y, i.e. exactly
    # a unit-axis rotation of pi/2, NOT 3x that.
    q = torch.tensor([[math.pi / 2]], dtype=torch.float64)
    ee = forward_kinematics(chain, q)[:, li]
    assert torch.allclose(ee[0, :3, 3], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-9)
    # Jacobian matches central differences (would fail if axis magnitude leaked).
    J = jacobian(chain, q, li)
    eps = 1e-6
    Tp = forward_kinematics(chain, q + eps)[:, li, :3, 3]
    Tm = forward_kinematics(chain, q - eps)[:, li, :3, 3]
    dv = (Tp - Tm) / (2 * eps)
    assert torch.allclose(J[0, :3, 0], dv[0], atol=1e-6)


# links and joints declared out of order; child before parent, root last.
SCRAMBLED = """
<robot name="scrambled">
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="1" effort="10"/></joint>
  <link name="l2"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="1" effort="10"/></joint>
  <link name="l1"/>
  <link name="base"/>
</robot>
"""


def test_scrambled_declaration_order():
    chain = compile_robot(parse_urdf_string(SCRAMBLED), dtype=torch.float64)
    assert chain.dof == 2
    # zero config: l2 sits at (1,0,0) regardless of declaration order
    ee = forward_kinematics(chain, torch.zeros(1, 2, dtype=torch.float64))[:, chain.link_index["l2"]]
    assert torch.allclose(ee[0, :3, 3], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-9)
