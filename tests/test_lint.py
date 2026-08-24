# tests/test_lint.py
"""Structural lint: one crafted URDF per rule, plus clean fixtures.

Every expectation here is hand-computed from the crafted document (which link
is unreachable, what a span of 8 rad means next to 2*pi, what a 1e5 mass ratio
is). Where a rule claims a model is broken, a second assertion shows the
library itself failing on that model, so the finding is tied to real damage
rather than to lint agreeing with itself.
"""
import math
import os
import re

import pytest

from kinfast.ir import Robot, Link, Joint, Inertial, Geometry
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast import lint

from tests.test_parse import TWO_LINK
from tests.test_spatial import SIX_DOF


def codes(report):
    return sorted(f.code for f in report)


# --- clean models -----------------------------------------------------------

def test_clean_fixture_has_no_critical_findings():
    for text in (TWO_LINK, SIX_DOF):
        report = lint.check_urdf_string(text)
        assert report.critical == [], report.to_markdown()
        assert report.ok


def test_clean_fixture_only_complains_about_collision_geometry():
    """TWO_LINK and SIX_DOF declare limits, unit axes, and unique names. The
    one thing they do lack is collision shapes on the moving links."""
    for text, n_moving in ((TWO_LINK, 2), (SIX_DOF, 6)):
        report = lint.check_urdf_string(text)
        assert set(codes(report)) == {"no_collision_geometry"}
        assert len(report.by_code("no_collision_geometry")) == n_moving


def test_fully_annotated_model_is_silent():
    urdf = """
    <robot name="tidy">
      <link name="base">
        <inertial><mass value="1.0"/><inertia ixx="0.1" iyy="0.1" izz="0.1"/></inertial>
        <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
      </link>
      <link name="l1">
        <inertial><mass value="0.5"/><inertia ixx="0.1" iyy="0.1" izz="0.1"/></inertial>
        <collision><geometry><mesh filename="meshes/l1.stl"/></geometry></collision>
      </link>
      <joint name="j1" type="revolute">
        <parent link="base"/><child link="l1"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.5" upper="1.5" velocity="2.0" effort="30"/>
      </joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    assert len(report) == 0
    assert not report          # empty report is falsey
    assert report.to_markdown().strip().endswith("No issues found.")


# --- duplicate names --------------------------------------------------------

DUP_LINK = """
<robot name="dup_link">
  <link name="base"/>
  <link name="l1">
    <inertial><mass value="3.0"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
  </link>
  <link name="l1"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/>
  </joint>
</robot>
"""


def test_duplicate_link_name_from_source():
    report = lint.check_urdf_string(DUP_LINK)
    hits = report.by_code("duplicate_link_name")
    assert [f.where for f in hits] == ["l1"]
    assert hits[0].severity == "critical"
    assert not report.ok

    # Independent witness: the document declares three <link> elements, the
    # parsed model only has two, and the mass on the first l1 is gone.
    ir = parse_urdf_string(DUP_LINK)
    assert len(re.findall(r"<link ", DUP_LINK)) == 3
    assert len(ir.links) == 2
    assert ir.links["l1"].inertial is None


def test_duplicate_link_name_in_hand_built_ir():
    """No source text needed when the dict key and Link.name disagree, which is
    how a programmatically assembled IR hides a collision."""
    ir = Robot("built", {"a": Link("same"), "b": Link("same")}, [])
    report = lint.check(ir)
    assert [f.code for f in report.critical] == ["duplicate_link_name"]
    assert report.critical[0].where == "same"


def test_duplicate_joint_name():
    urdf = """
    <robot name="dup_joint">
      <link name="base"/><link name="l1"/><link name="l2"/>
      <joint name="j" type="revolute">
        <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
      <joint name="j" type="revolute">
        <parent link="l1"/><child link="l2"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    hits = report.by_code("duplicate_joint_name")
    assert len(hits) == 1 and hits[0].where == "j"
    assert hits[0].severity == "critical"

    # Witness: the compiled chain has two degrees of freedom both called "j".
    chain = compile_robot(parse_urdf_string(urdf))
    assert chain.joint_names == ["j", "j"]


# --- dangling references and topology ---------------------------------------

def test_joint_referencing_missing_link():
    urdf = """
    <robot name="ghost">
      <link name="base"/><link name="l1"/>
      <joint name="j1" type="revolute">
        <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
      <joint name="j2" type="revolute">
        <parent link="l1"/><child link="nowhere"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    hits = report.by_code("missing_link_reference")
    assert [f.where for f in hits] == ["j2"]
    assert "nowhere" in hits[0].message

    # Witness: the dangling joint is silently dropped, so the compiled chain
    # has one degree of freedom where the file asks for two. This is exactly
    # the kind of quiet loss lint exists to surface.
    chain = compile_robot(parse_urdf_string(urdf))
    assert chain.dof == 1
    assert chain.joint_names == ["j1"]


def test_missing_parent_element_in_hand_built_ir():
    ir = Robot("nop", {"base": Link("base")},
               [Joint("j1", "revolute", "", "base")])
    report = lint.check(ir)
    assert "missing_link_reference" in codes(report)


def test_self_loop():
    ir = Robot("loop", {"base": Link("base"), "l1": Link("l1")},
               [Joint("j1", "revolute", "l1", "l1")])
    report = lint.check(ir)
    hits = report.by_code("self_loop")
    assert len(hits) == 1 and hits[0].severity == "critical"


def test_multiple_parents():
    urdf = """
    <robot name="diamond">
      <link name="base"/><link name="a"/><link name="tip"/>
      <joint name="j1" type="fixed"><parent link="base"/><child link="a"/></joint>
      <joint name="j2" type="fixed"><parent link="base"/><child link="tip"/></joint>
      <joint name="j3" type="fixed"><parent link="a"/><child link="tip"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    hits = report.by_code("multiple_parents")
    assert [f.where for f in hits] == ["tip"]
    assert "j2" in hits[0].message and "j3" in hits[0].message


def test_cycle_leaves_no_root():
    ir = Robot("cycle", {"a": Link("a"), "b": Link("b")},
               [Joint("j1", "fixed", "a", "b"), Joint("j2", "fixed", "b", "a")])
    report = lint.check(ir)
    assert [f.code for f in report.critical] == ["no_root"]

    # Witness: the IR itself cannot name a base link for this model.
    with pytest.raises(ValueError):
        ir.root_link()


def test_unreachable_link():
    urdf = """
    <robot name="stray">
      <link name="base"/><link name="l1"/><link name="floater"/>
      <joint name="j1" type="revolute">
        <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    hits = report.by_code("unreachable_link")
    assert [f.where for f in hits] == ["floater"]
    assert hits[0].severity == "major"
    assert "base" in hits[0].message

    # Witness: two roots, so the IR refuses to name one.
    with pytest.raises(ValueError):
        parse_urdf_string(urdf).root_link()


def test_disconnected_subtree_reports_only_the_smaller_side():
    """The larger tree is taken as the robot, so a stray pair is named rather
    than the whole arm."""
    ir = Robot("split", {n: Link(n) for n in ("base", "a", "b", "x", "y")},
               [Joint("j1", "fixed", "base", "a"),
                Joint("j2", "fixed", "a", "b"),
                Joint("j3", "fixed", "x", "y")])
    report = lint.check(ir)
    assert sorted(f.where for f in report.by_code("unreachable_link")) == ["x", "y"]


def test_empty_model():
    report = lint.check(Robot("nothing", {}, []))
    assert [f.code for f in report.critical] == ["empty_model"]


# --- joint types and axes ---------------------------------------------------

def test_unsupported_and_missing_joint_type():
    urdf = """
    <robot name="types">
      <link name="base"/><link name="a"/><link name="b"/>
      <joint name="jf" type="floating"><parent link="base"/><child link="a"/></joint>
      <joint name="jn"><parent link="a"/><child link="b"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    assert [f.where for f in report.by_code("unsupported_joint_type")] == ["jf"]
    assert [f.where for f in report.by_code("missing_joint_type")] == ["jn"]
    assert report.by_code("missing_joint_type")[0].severity == "critical"

    # Witness: both compile to fixed joints, so the model has zero DOF.
    assert compile_robot(parse_urdf_string(urdf)).dof == 0


def test_non_unit_axis():
    urdf = """
    <robot name="axes">
      <link name="base"/><link name="l1"/>
      <joint name="j1" type="revolute">
        <parent link="base"/><child link="l1"/><axis xyz="0 0 2"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    hits = report.by_code("non_unit_axis")
    assert [f.where for f in hits] == ["j1"]
    assert hits[0].severity == "major"
    assert "2" in hits[0].message

    # Hand-computed norms either side of the tolerance.
    ir = parse_urdf_string(urdf)
    ir.joints[0].axis = (0.6, 0.8, 0.0)          # norm exactly 1
    assert lint.check(ir).by_code("non_unit_axis") == []
    ir.joints[0].axis = (1.0, 1.0, 1.0)          # norm sqrt(3)
    msg = lint.check(ir).by_code("non_unit_axis")[0].message
    assert f"{math.sqrt(3.0):.6g}" in msg


def test_zero_axis_is_critical():
    ir = parse_urdf_string(TWO_LINK)
    ir.joints[0].axis = (0.0, 0.0, 0.0)
    report = lint.check(ir)
    hits = report.by_code("zero_axis")
    assert [f.where for f in hits] == ["j1"]
    assert hits[0].severity == "critical"


def test_fixed_joints_are_exempt_from_axis_and_limit_rules():
    urdf = """
    <robot name="fixed_only">
      <link name="base"/><link name="tool"/>
      <joint name="mount" type="fixed">
        <parent link="base"/><child link="tool"/><axis xyz="0 0 5"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    assert codes(report) == []


# --- limits -----------------------------------------------------------------

def test_missing_velocity_and_effort_limits():
    urdf = """
    <robot name="nolimits">
      <link name="base"/><link name="l1"/><link name="l2"/>
      <joint name="j1" type="revolute">
        <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1"/></joint>
      <joint name="j2" type="prismatic">
        <parent link="l1"/><child link="l2"/><axis xyz="1 0 0"/>
        <limit lower="0" upper="0.2" velocity="0.5"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    assert [f.where for f in report.by_code("missing_velocity_limit")] == ["j1"]
    assert [f.where for f in report.by_code("missing_effort_limit")] == ["j1", "j2"]
    assert all(f.severity == "minor" for f in report.by_code("missing_effort_limit"))


def test_revolute_range_over_two_pi():
    urdf = """
    <robot name="wide">
      <link name="base"/><link name="a"/><link name="b"/><link name="c"/>
      <joint name="wide" type="revolute">
        <parent link="base"/><child link="a"/><axis xyz="0 0 1"/>
        <limit lower="-4" upper="4" velocity="1" effort="1"/></joint>
      <joint name="narrow" type="revolute">
        <parent link="a"/><child link="b"/><axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" velocity="1" effort="1"/></joint>
      <joint name="spin" type="continuous">
        <parent link="b"/><child link="c"/><axis xyz="0 0 1"/>
        <limit velocity="1" effort="1"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    hits = report.by_code("revolute_range_over_2pi")
    # Hand-computed: 8 rad > 2*pi = 6.2832, 6.28 rad < 2*pi, and a continuous
    # joint is allowed to turn forever.
    assert [f.where for f in hits] == ["wide"]
    assert "8" in hits[0].message
    assert 8.0 > 2 * math.pi > 6.28


def test_prismatic_range_is_not_measured_against_two_pi():
    """A 10 metre rail is not a lint finding; 2*pi is an angle."""
    urdf = """
    <robot name="rail">
      <link name="base"/><link name="cart"/>
      <joint name="slide" type="prismatic">
        <parent link="base"/><child link="cart"/><axis xyz="1 0 0"/>
        <limit lower="0" upper="10" velocity="1" effort="1"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    assert report.by_code("revolute_range_over_2pi") == []


# --- mass -------------------------------------------------------------------

def _two_mass_ir(m_base, m_tip):
    links = {"base": Link("base", Inertial(m_base, (0, 0, 0), (1, 1, 1, 0, 0, 0)),
                          None, Geometry("box", size=(0.1, 0.1, 0.1))),
             "tip": Link("tip", Inertial(m_tip, (0, 0, 0), (1, 1, 1, 0, 0, 0)),
                         None, Geometry("box", size=(0.1, 0.1, 0.1)))}
    return Robot("masses", links,
                 [Joint("j1", "revolute", "base", "tip", limit=(-1.0, 1.0),
                        velocity=1.0, effort=1.0)])


def test_mass_ratio_flags_a_unit_mistake():
    report = lint.check(_two_mass_ir(10.0, 1e-4))   # ratio 1e5
    hits = report.by_code("mass_ratio")
    assert len(hits) == 1
    assert hits[0].where == "base"                  # the heavy end is named
    assert "tip" in hits[0].message
    assert hits[0].severity == "major"


def test_mass_ratio_quiet_within_three_orders():
    assert lint.check(_two_mass_ir(10.0, 1e-2)).by_code("mass_ratio") == []


def test_mass_ratio_ignores_massless_links():
    """Massless links are ordinary in URDF (frames, mounts) and must not be
    read as an infinite ratio."""
    report = lint.check(_two_mass_ir(10.0, 0.0))
    assert report.by_code("mass_ratio") == []


# --- collision geometry -----------------------------------------------------

def test_moving_link_without_collision_geometry():
    urdf = """
    <robot name="halfdressed">
      <link name="base"/>
      <link name="l1">
        <collision><geometry><box size="0.1 0.1 0.4"/></geometry></collision>
      </link>
      <link name="l2"/>
      <link name="tool"/>
      <joint name="j1" type="revolute">
        <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
      <joint name="j2" type="revolute">
        <parent link="l1"/><child link="l2"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
      <joint name="mount" type="fixed">
        <parent link="l2"/><child link="tool"/></joint>
    </robot>
    """
    report = lint.check_urdf_string(urdf)
    # l1 has a box, tool is fixed (never moves on its own), so only l2.
    assert [f.where for f in report.by_code("no_collision_geometry")] == ["l2"]


# --- mesh paths -------------------------------------------------------------

MESHES = r"""
<robot name="meshes">
  <link name="base">
    <visual><geometry><mesh filename="package://my_robot/meshes/base.dae"/></geometry></visual>
    <collision><geometry><mesh filename="meshes/base.stl"/></geometry></collision>
  </link>
  <link name="l1">
    <collision><geometry><mesh filename="/home/alice/meshes/l1.stl"/></geometry></collision>
  </link>
  <link name="l2">
    <collision><geometry><mesh filename="C:\models\l2.stl"/></geometry></collision>
  </link>
  <link name="l3">
    <collision><geometry><mesh filename="file:///tmp/l3.stl"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
</robot>
"""


def test_mesh_path_portability():
    report = lint.check_urdf_string(MESHES)
    assert [f.where for f in report.by_code("package_mesh_path")] == ["base.visual"]
    # The relative collision mesh on base is fine and must not appear.
    assert sorted(f.where for f in report.by_code("absolute_mesh_path")) == [
        "l1.collision", "l2.collision", "l3.collision"]
    assert all(f.severity == "minor"
               for f in report.by_code("absolute_mesh_path") + report.by_code("package_mesh_path"))


def test_gazebo_model_scheme_is_flagged_too():
    ir = Robot("gz", {"base": Link("base", visual=Geometry("mesh", "model://cart/x.dae"))}, [])
    report = lint.check(ir)
    hits = report.by_code("package_mesh_path")
    assert [f.where for f in hits] == ["base.visual"]
    assert "model" in hits[0].message


def test_non_mesh_geometry_has_no_path_findings():
    ir = Robot("boxy", {"base": Link("base", collision=Geometry("box", size=(1, 1, 1)))}, [])
    assert lint.check(ir).by_code("package_mesh_path") == []


# --- report object ----------------------------------------------------------

def test_report_groups_and_counts():
    report = lint.Report("demo")
    report.add("a_critical", "critical", "j1", "broken")
    report.add("b_major", "major", "l2", "suspicious")
    report.add("c_minor", "minor", "j3", "cosmetic")
    assert len(report) == 3
    assert [f.code for f in report.critical] == ["a_critical"]
    assert [f.code for f in report.major] == ["b_major"]
    assert [f.code for f in report.minor] == ["c_minor"]
    assert report.codes == ["a_critical", "b_major", "c_minor"]
    assert not report.ok
    assert report.summary() == "3 findings: 1 critical, 1 major, 1 minor"


def test_report_rejects_unknown_severity():
    with pytest.raises(ValueError):
        lint.Report("demo").add("x", "catastrophic", "j1", "boom")


def test_to_markdown_structure():
    report = lint.check_urdf_string(DUP_LINK)
    md = report.to_markdown()
    lines = md.splitlines()
    assert lines[0] == "# Lint report: dup_link"
    assert "## Critical (1)" in lines
    assert any(line.startswith("- `duplicate_link_name` **l1**:") for line in lines)
    assert "-- " not in md and "\u2014" not in md   # no em-dashes


def test_finding_str_is_readable():
    f = lint.Finding("zero_axis", "critical", "j1", "no direction")
    assert str(f) == "[critical] zero_axis (j1): no direction"


# --- behaviour of check itself ----------------------------------------------

def test_check_is_deterministic_and_pure():
    """Lint reports the model, it does not edit it, and two runs agree."""
    ir = parse_urdf_string(DUP_LINK)
    ir.joints[0].axis = (0.0, 0.0, 3.0)
    ir.joints[0].limit = (-4.0, 4.0)
    before = (dict(ir.links), list(ir.joints), ir.joints[0].axis, ir.joints[0].limit)

    first = lint.check(ir, source=DUP_LINK)
    second = lint.check(ir, source=DUP_LINK)
    assert first.findings == second.findings
    assert ir.joints[0].axis == (0.0, 0.0, 3.0)     # repair would have fixed this
    assert ir.joints[0].limit == (-4.0, 4.0)
    assert (dict(ir.links), list(ir.joints), ir.joints[0].axis, ir.joints[0].limit) == before


def test_unparseable_source_is_ignored():
    ir = parse_urdf_string(TWO_LINK)
    report = lint.check(ir, source="<robot name='x'>  <<<not xml")
    assert report.by_code("duplicate_link_name") == []
    assert report.ok


def test_mjcf_source_is_ignored_for_duplicate_links():
    ir = parse_urdf_string(TWO_LINK)
    report = lint.check(ir, source="<mujoco><worldbody/></mujoco>")
    assert report.by_code("duplicate_link_name") == []


def test_summary_pluralisation():
    report = lint.Report("demo")
    report.add("only", "minor", "j1", "cosmetic")
    assert report.summary() == "1 finding: 1 minor"


PANDA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "examples", "assets", "panda.urdf")


@pytest.mark.skipif(not os.path.exists(PANDA), reason="panda.urdf asset not present")
def test_published_robot_has_no_critical_findings():
    """A real, widely used model must come out clean at the critical level.
    The Panda ships package:// mesh paths, which is a portability nit, not a
    structural defect."""
    report = lint.check_urdf_file(PANDA)
    assert report.critical == [], report.to_markdown()
    assert report.major == [], report.to_markdown()
    assert set(codes(report)) <= {"package_mesh_path"}


def test_check_urdf_file(tmp_path):
    path = tmp_path / "dup.urdf"
    path.write_text(DUP_LINK, encoding="utf-8")
    report = lint.check_urdf_file(str(path))
    assert "duplicate_link_name" in codes(report)
    assert report.robot_name == "dup_link"
