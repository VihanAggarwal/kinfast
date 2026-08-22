# tests/test_xacro.py
"""xacro support via the standalone `xacro` package (no ROS). A xacro file with
properties, math, a macro, and a local include must expand and load to the
same kinematics as the hand-expanded URDF. Skips if xacro is not installed."""
import math

import pytest
import torch

xacro = pytest.importorskip("xacro")

import kinfast

INCLUDED = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="segment" params="name parent length">
    <link name="${name}"/>
    <joint name="${name}_joint" type="revolute">
      <parent link="${parent}"/><child link="${name}"/>
      <origin xyz="${length} 0 0" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-pi/2}" upper="${pi/2}" velocity="2" effort="10"/>
    </joint>
  </xacro:macro>
</robot>
"""

MAIN = """<?xml version="1.0"?>
<robot name="xarm" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="segments.xacro"/>
  <xacro:property name="l1" value="0.5"/>
  <xacro:property name="l2" value="${l1 * 0.8}"/>
  <link name="base"/>
  <xacro:segment name="s1" parent="base" length="0.0"/>
  <xacro:segment name="s2" parent="s1" length="${l1}"/>
  <xacro:segment name="s3" parent="s2" length="${l2}"/>
</robot>
"""

# the same robot, written out by hand
EXPANDED = """
<robot name="xarm">
  <link name="base"/><link name="s1"/><link name="s2"/><link name="s3"/>
  <joint name="s1_joint" type="revolute"><parent link="base"/><child link="s1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.5707963" upper="1.5707963" velocity="2" effort="10"/></joint>
  <joint name="s2_joint" type="revolute"><parent link="s1"/><child link="s2"/>
    <origin xyz="0.5 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.5707963" upper="1.5707963" velocity="2" effort="10"/></joint>
  <joint name="s3_joint" type="revolute"><parent link="s2"/><child link="s3"/>
    <origin xyz="0.4 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.5707963" upper="1.5707963" velocity="2" effort="10"/></joint>
</robot>
"""


def test_xacro_expands_and_matches_hand_urdf(tmp_path):
    (tmp_path / "segments.xacro").write_text(INCLUDED)
    main = tmp_path / "arm.urdf.xacro"
    main.write_text(MAIN)

    robot = kinfast.load(str(main))
    ref = kinfast.load_string(EXPANDED)
    assert robot.dof == ref.dof == 3
    assert robot.joint_names == ref.joint_names
    torch.manual_seed(0)
    q = ref.random_configs(16)
    assert torch.allclose(robot.fk_all(q), ref.fk_all(q), atol=1e-6)
    assert abs(robot.upper[0].item() - math.pi / 2) < 1e-6   # ${pi/2} evaluated


def test_xacro_mappings_override_properties(tmp_path):
    (tmp_path / "segments.xacro").write_text(INCLUDED)
    main = tmp_path / "arm.urdf.xacro"
    main.write_text(MAIN.replace('value="0.5"', 'value="$(arg l1)"')
                        .replace('<xacro:include', '<xacro:arg name="l1" default="0.5"/>\n  <xacro:include'))
    robot = kinfast.load(str(main), mappings={"l1": "1.0"})
    # s2 joint origin is now 1.0 along x, s3 is 0.8
    q = torch.zeros(1, 3)
    s3 = robot.fk_all(q)[0, robot.link_id("s3"), :3, 3]
    assert torch.allclose(s3, torch.tensor([1.8, 0.0, 0.0]), atol=1e-6)
