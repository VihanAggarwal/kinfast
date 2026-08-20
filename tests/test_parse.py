# tests/test_parse.py
from kinfast.urdf.parse import parse_urdf_string

TWO_LINK = """
<robot name="two_link">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9" upper="2.9" velocity="2.6" effort="87"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9" upper="2.9" velocity="2.6" effort="87"/>
  </joint>
</robot>
"""

def test_parse_counts():
    robot = parse_urdf_string(TWO_LINK)
    assert robot.name == "two_link"
    assert set(robot.links) == {"base", "l1", "l2"}
    assert robot.dof() == 2

def test_parse_origin_and_axis():
    robot = parse_urdf_string(TWO_LINK)
    j2 = robot.joints[1]
    assert j2.origin_xyz == (1.0, 0.0, 0.0)
    assert j2.axis == (0.0, 0.0, 1.0)
    assert j2.limit == (-2.9, 2.9)
