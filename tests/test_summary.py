# tests/test_summary.py
"""Tests for the human-readable robot report.

The oracles here are hand-computed from the URDF text: a planar 2R arm whose
farthest reach is 2 m by the law of cosines, a prismatic slider whose reach is
0.1 + 0.5 m, link masses that add to a round number, and a branched tree whose
reach bound can be walked by eye. Nothing checks the report against another
part of kinfast except where that is the point of the test (the reach bound
must not be smaller than what FK actually produces).
"""
import copy
import math
import pathlib
import re

import pytest
import torch

import kinfast
from kinfast.compile import compile_robot
from kinfast.robot import Robot
from kinfast.summary import Summary, reach_estimate, sampled_reach, summary
from kinfast.urdf.parse import parse_urdf_string

# ---------------------------------------------------------------- fixtures

# Planar 2R with unit links and round masses. The ee origin sits at
# (1,0) + R(q2)(1,0) in the shoulder frame, so its distance from the base is
# 2 |cos(q2 / 2)|: exactly 2 m at q2 = 0 and never more, whatever q1 does.
PLANAR = """
<robot name="planar2r">
  <link name="base">
    <inertial><mass value="1.5"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="l1">
    <inertial><origin xyz="0.5 0 0"/><mass value="2.25"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="l2">
    <inertial><origin xyz="0.5 0 0"/><mass value="0.25"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="ee">
    <inertial><mass value="1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14159265358979" upper="3.14159265358979"
           velocity="2.5" effort="40"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14159265358979" upper="3.14159265358979"
           velocity="1.25" effort="20"/>
  </joint>
  <joint name="wrist_fix" type="fixed">
    <parent link="l2"/><child link="ee"/>
    <origin xyz="1 0 0"/>
  </joint>
</robot>
"""

# No inertial tags at all, and one fixed joint next to one prismatic one.
# The slider's origin sits 0.1 m up and travels 0 .. 0.5 m, so the farthest a
# link origin gets from the base is 0.6 m.
SLIDER = """
<robot name="slider">
  <link name="base"/>
  <link name="carriage"/>
  <link name="tool"/>
  <joint name="rail" type="prismatic">
    <parent link="base"/><child link="carriage"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="0.5" velocity="0.8" effort="120"/>
  </joint>
  <joint name="tool_mount" type="fixed">
    <parent link="carriage"/><child link="tool"/>
    <origin xyz="0 0 0"/>
  </joint>
</robot>
"""

# Nothing moves. dof is 0, but the report still has a joint and two masses.
STATIC = """
<robot name="static_rig">
  <link name="base">
    <inertial><mass value="3"/>
      <inertia ixx="0.2" iyy="0.2" izz="0.2" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="bracket">
    <inertial><mass value="0.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="weld" type="fixed">
    <parent link="base"/><child link="bracket"/>
    <origin xyz="0.3 0 0.4"/>
  </joint>
</robot>
"""

# Two branches off the base. Walking it by hand: `upper` sits 0.5 m up, `hand`
# another 0.2 m out (0.7 total), and `lift` rides a 0.4 m rail off the base.
# The bound is therefore 0.7 m, attained at `hand`.
BRANCHED = """
<robot name="branched">
  <link name="base"/><link name="upper"/><link name="hand"/><link name="lift"/>
  <joint name="j_shoulder" type="revolute">
    <parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" velocity="2" effort="30"/>
  </joint>
  <joint name="j_wrist_fix" type="fixed">
    <parent link="upper"/><child link="hand"/>
    <origin xyz="0.2 0 0"/>
  </joint>
  <joint name="j_lift" type="prismatic">
    <parent link="base"/><child link="lift"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-0.1" upper="0.4" velocity="0.5" effort="60"/>
  </joint>
</robot>
"""

# A continuous joint with an axis that is not unit length, so the repair pass
# has something real to report when it is skipped.
SLOPPY = """
<robot name="sloppy">
  <link name="base"/><link name="spinner"/>
  <joint name="spin" type="continuous">
    <parent link="base"/><child link="spinner"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 0 3"/>
  </joint>
</robot>
"""


def _lines(text):
    return [ln for ln in text.splitlines()]


def _split_row(line):
    """Cells of a markdown pipe row, honouring backslash-escaped pipes."""
    return re.split(r"(?<![\\])\|", line.strip().strip("|"))


def _md_rows(md, header_cell):
    """Body rows of the markdown pipe table whose first header cell matches."""
    rows, taking = [], False
    for ln in md.splitlines():
        if not ln.startswith("|"):
            taking = False
            continue
        cells = [c.strip() for c in _split_row(ln)]
        if cells[0] == header_cell:
            taking = True
            continue
        if taking:
            if set(cells[0]) <= {"-", ":"}:
                continue
            rows.append(cells)
    return rows


# ---------------------------------------------------------------- joint names

def test_every_joint_name_appears_in_text_and_markdown():
    for src in (PLANAR, SLIDER, STATIC, BRANCHED):
        ir = parse_urdf_string(src)
        s = summary(kinfast.load_string(src))
        text, md = s.to_text(), s.to_markdown()
        assert ir.joints, "fixture should declare at least one joint"
        for j in ir.joints:
            assert j.name in text, f"{j.name} missing from text of {ir.name}"
            assert j.name in md, f"{j.name} missing from markdown of {ir.name}"


def test_fixed_joints_are_listed_but_carry_no_q():
    s = summary(kinfast.load_string(SLIDER))
    by_name = {j["name"]: j for j in s.joints}
    assert set(by_name) == {"rail", "tool_mount"}
    assert by_name["rail"]["q"] == 0
    assert by_name["rail"]["type"] == "prismatic"
    assert by_name["tool_mount"]["q"] is None
    assert by_name["tool_mount"]["type"] == "fixed"
    # a joint with no q has no limits to print, and prints a dash for each
    row = [r for r in _md_rows(s.to_markdown(), "joint")
           if r[0] == "tool_mount"][0]
    assert row[2:] == ["-", "-", "-", "-"]


def test_joint_rows_carry_limits_and_velocity_from_the_file():
    s = summary(kinfast.load_string(PLANAR))
    by_name = {j["name"]: j for j in s.joints}
    assert by_name["shoulder"]["velocity"] == pytest.approx(2.5, abs=1e-6)
    assert by_name["elbow"]["velocity"] == pytest.approx(1.25, abs=1e-6)
    for name in ("shoulder", "elbow"):
        assert by_name[name]["lower"] == pytest.approx(-math.pi, abs=1e-6)
        assert by_name[name]["upper"] == pytest.approx(math.pi, abs=1e-6)
    text = s.to_text()
    assert "2.5" in text and "1.25" in text


def test_continuous_type_word_survives_the_compile():
    # the chain codes continuous and revolute identically; the report should
    # still say which one the file asked for
    s = summary(kinfast.load_string(SLOPPY))
    assert [j["type"] for j in s.joints] == ["continuous"]


# ---------------------------------------------------------------- mass

def test_total_mass_equals_the_sum_of_the_link_masses():
    s = summary(kinfast.load_string(PLANAR, dtype=torch.float64))
    # hand-computed from the URDF: 1.5 + 2.25 + 0.25 + 1
    assert s.total_mass == pytest.approx(5.0, abs=1e-12)
    assert s.total_mass == pytest.approx(sum(l["mass"] for l in s.links),
                                         abs=1e-12)
    assert {l["name"]: l["mass"] for l in s.links} == {
        "base": pytest.approx(1.5), "l1": pytest.approx(2.25),
        "l2": pytest.approx(0.25), "ee": pytest.approx(1.0)}


def test_mass_table_last_row_is_the_total_of_the_rows_above_it():
    for src in (PLANAR, STATIC, SLIDER):
        md = summary(kinfast.load_string(src, dtype=torch.float64)).to_markdown()
        rows = _md_rows(md, "link")
        assert rows[-1][0] == "total"
        body = [float(r[1]) for r in rows[:-1]]
        assert float(rows[-1][1]) == pytest.approx(sum(body), abs=1e-9)


def test_robot_with_no_inertials_reports_zero_mass_and_says_so():
    s = summary(kinfast.load_string(SLIDER))
    assert s.total_mass == 0.0
    assert s.no_inertial == 3
    assert all(l["mass"] == 0.0 for l in s.links)
    text = s.to_text()
    assert "3 of 3 links have no inertial" in text
    # the table still lists every link plus the total row
    assert [l["name"] for l in s.links] == ["base", "carriage", "tool"]
    assert "total" in text


def test_partially_missing_inertials_use_singular_grammar():
    s = summary(kinfast.load_string(PLANAR + ""))
    assert s.no_inertial == 0
    s2 = summary(kinfast.load_string("""
<robot name="half">
  <link name="base">
    <inertial><mass value="1"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="tip"/>
  <joint name="j" type="fixed">
    <parent link="base"/><child link="tip"/><origin xyz="0 0 1"/>
  </joint>
</robot>"""))
    assert s2.no_inertial == 1
    assert "1 of 2 links has no inertial" in s2.to_text()


# ---------------------------------------------------------------- reach

def test_reach_bound_matches_a_hand_walked_tree():
    chain = compile_robot(parse_urdf_string(BRANCHED), dtype=torch.float64)
    bound, link = reach_estimate(chain)
    assert bound == pytest.approx(0.7, abs=1e-12)
    assert link == "hand"


def test_reach_bound_counts_prismatic_travel_on_the_longer_branch():
    # push the rail past the arm and the bound should follow it
    src = BRANCHED.replace('lower="-0.1" upper="0.4"',
                           'lower="-0.1" upper="2.0"')
    chain = compile_robot(parse_urdf_string(src), dtype=torch.float64)
    bound, link = reach_estimate(chain)
    assert bound == pytest.approx(2.0, abs=1e-12)
    assert link == "lift"


def test_planar_sampled_reach_matches_the_law_of_cosines():
    # |p_ee| = 2 |cos(q2 / 2)|, maximised at q2 = 0 with value 2.0; the
    # quadratic top means a few thousand uniform samples land very close
    chain = compile_robot(parse_urdf_string(PLANAR), dtype=torch.float64)
    got, link = sampled_reach(chain, n=4096, seed=0)
    assert link == "ee"
    assert got == pytest.approx(2.0, abs=1e-4)
    assert got <= 2.0 + 1e-12


def test_slider_sampled_reach_matches_the_hand_computed_travel():
    chain = compile_robot(parse_urdf_string(SLIDER), dtype=torch.float64)
    got, link = sampled_reach(chain, n=4096, seed=0)
    assert link in ("carriage", "tool")   # the two ride together
    assert got == pytest.approx(0.6, abs=1e-3)   # 0.1 origin + 0.5 of rail
    assert got <= 0.6 + 1e-12


def test_sampled_reach_never_exceeds_the_bound():
    for src in (PLANAR, SLIDER, BRANCHED, STATIC):
        chain = compile_robot(parse_urdf_string(src), dtype=torch.float64)
        bound, _ = reach_estimate(chain)
        got, _ = sampled_reach(chain, n=1024, seed=3)
        assert got <= bound + 1e-9, src


def test_sampled_reach_is_seeded_and_reproducible():
    chain = compile_robot(parse_urdf_string(PLANAR), dtype=torch.float64)
    a = sampled_reach(chain, n=64, seed=7)
    b = sampled_reach(chain, n=64, seed=7)
    c = sampled_reach(chain, n=64, seed=8)
    assert a == b
    # a different seed draws different configurations, so the sampled maximum
    # (which never hits the exact optimum) lands somewhere else
    assert a[0] != c[0]


def test_reach_samples_zero_skips_the_monte_carlo_row():
    s = summary(kinfast.load_string(PLANAR), reach_samples=0)
    assert s.sampled is None
    assert "reach (sampled)" not in s.to_text()
    assert "reach (bound)" in s.to_text()


def test_unsampleable_joint_still_reports_the_bound():
    src = """
<robot name="unbounded">
  <link name="base"/><link name="rail"/>
  <joint name="j" type="prismatic">
    <parent link="base"/><child link="rail"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-inf" upper="inf" velocity="1" effort="5"/>
  </joint>
</robot>"""
    r = Robot.from_ir(parse_urdf_string(src), repair_model=False)
    s = summary(r)
    assert s.sampled is None
    assert math.isinf(s.reach)
    assert "inf" in s.to_text()


# ---------------------------------------------------------------- zero dof

def test_zero_dof_robot():
    s = summary(kinfast.load_string(STATIC, dtype=torch.float64))
    assert s.dof == 0
    assert s.n_links == 2
    assert s.n_movable == 0
    assert [j["name"] for j in s.joints] == ["weld"]
    assert s.total_mass == pytest.approx(3.5, abs=1e-12)
    # the fixed offset is the whole reach: |(0.3, 0, 0.4)| = 0.5
    assert s.reach == pytest.approx(0.5, abs=1e-12)
    assert s.reach_link == "bracket"
    assert s.sampled == pytest.approx(0.5, abs=1e-12)
    text = s.to_text()
    assert "dof" in text and "weld" in text
    assert "0 movable, 1 fixed" in text
    md = s.to_markdown()
    assert "weld" in md and "| total |" in md


def test_single_link_robot_has_no_joint_table():
    s = summary(kinfast.load_string('<robot name="solo"><link name="a"/></robot>'))
    assert s.dof == 0
    assert s.joints == []
    assert s.reach == 0.0
    assert "(no joints)" in s.to_text()
    assert "_no joints_" in s.to_markdown()
    assert "a" in s.to_text()


# ---------------------------------------------------------------- findings

def test_findings_count_is_zero_after_a_normal_load():
    # loading repairs first, so a clean reload finds nothing left to fix
    s = summary(kinfast.load_string(SLOPPY))
    assert s.findings == []
    assert "repair findings  0" in s.to_text()


def test_findings_are_counted_when_repair_was_skipped():
    ir = parse_urdf_string(SLOPPY)
    s = summary(Robot.from_ir(ir, repair_model=False))
    codes = sorted(f.code for f in s.findings)
    assert codes == ["unnormalized_axis"]
    text = s.to_text()
    assert "repair findings  1 (unnormalized_axis)" in text
    assert "unnormalized_axis" in s.to_markdown()


def test_findings_check_does_not_mutate_the_caller_s_model():
    ir = parse_urdf_string(SLOPPY)
    before = copy.deepcopy(ir)
    summary(Robot.from_ir(ir, repair_model=False))
    assert ir.joints[0].axis == before.joints[0].axis
    assert ir.joints[0].limit == before.joints[0].limit


def test_findings_can_be_supplied_by_the_caller():
    from kinfast.urdf.repair import Finding
    given = [Finding("made_up", "somewhere", "hand-supplied"),
             Finding("made_up", "elsewhere", "hand-supplied"),
             Finding("other", "x", "hand-supplied")]
    s = summary(kinfast.load_string(PLANAR), findings=given, reach_samples=0)
    assert len(s.findings) == 3
    text = s.to_text()
    assert "repair findings  3 (made_up x2, other)" in text
    assert "hand-supplied" in text


def test_chain_without_an_ir_says_the_findings_were_not_checked():
    chain = compile_robot(parse_urdf_string(PLANAR), dtype=torch.float64)
    s = summary(chain, reach_samples=0)
    assert s.findings is None
    assert "not checked" in s.to_text()
    # a bare chain has forgotten the fixed joints, but keeps the movable ones
    assert [j["name"] for j in s.joints] == ["shoulder", "elbow"]
    assert s.total_mass == pytest.approx(5.0, abs=1e-12)


def test_parser_notes_are_reported():
    ir = parse_urdf_string(SLIDER)
    ir.parse_notes = ["free joint pinned to the world", "mass left at zero"]
    s = summary(Robot.from_ir(ir), reach_samples=0)
    text = s.to_text()
    assert "parser notes" in text
    for note in ir.parse_notes:
        assert note in text
        assert note in s.to_markdown()


# ---------------------------------------------------------------- rendering

def test_text_table_columns_line_up():
    s = summary(kinfast.load_string(PLANAR))
    lines = _lines(s.to_text())
    i = next(k for k, ln in enumerate(lines) if ln.startswith("joint"))
    header, rule = lines[i], lines[i + 1]
    assert set(rule.replace(" ", "")) == {"-"}

    # column spans read off the dashed rule, then every other line of the table
    # must place its cell inside its own span and nowhere else
    spans, pos = [], 0
    for seg in rule.split("  "):
        spans.append((pos, pos + len(seg)))
        pos += len(seg) + 2

    body = lines[i + 2:i + 2 + len(s.joints)]
    assert len(body) == len(s.joints) == 3
    for ln in [header] + body:
        padded = ln.ljust(spans[-1][1])
        cells = [padded[a:b] for a, b in spans]
        assert "".join(c + "  " for c in cells).rstrip() == ln
        for c in cells:
            assert c == c.strip().rjust(len(c)) or c == c.strip().ljust(len(c))
    assert [c.strip() for c in
            [header.ljust(spans[-1][1])[a:b] for a, b in spans]] == \
        ["joint", "type", "q", "lower", "upper", "velocity"]


def test_markdown_tables_are_well_formed():
    md = summary(kinfast.load_string(PLANAR)).to_markdown()
    assert md.startswith("# planar2r")
    assert "## Joints" in md and "## Links" in md
    pipe_lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert pipe_lines, "expected at least one markdown table"
    for ln in pipe_lines:
        assert ln.endswith("|")
        assert len(ln.split("|")) >= 4        # leading, cells, trailing
    joint_rows = _md_rows(md, "joint")
    assert [r[0] for r in joint_rows] == ["shoulder", "elbow", "wrist_fix"]
    assert all(len(r) == 6 for r in joint_rows)


def test_markdown_escapes_a_pipe_in_a_name():
    src = SLOPPY.replace('name="spin"', 'name="spin|weird"')
    md = summary(kinfast.load_string(src), reach_samples=0).to_markdown()
    row = [ln for ln in md.splitlines() if "weird" in ln][0]
    assert "spin\\|weird" in row
    # the escape kept it one cell: six columns, not seven
    assert len(_split_row(row)) == 6


def test_str_is_the_text_report_and_repr_is_short():
    s = summary(kinfast.load_string(PLANAR), reach_samples=0)
    assert str(s) == s.to_text()
    assert s.to_text().endswith("\n")
    assert s.to_markdown().endswith("\n")
    r = repr(s)
    assert r.startswith("<Summary 'planar2r'") and "\n" not in r


def test_report_is_deterministic():
    a = summary(kinfast.load_string(PLANAR)).to_text()
    b = summary(kinfast.load_string(PLANAR)).to_text()
    assert a == b


def test_unnamed_model_renders_a_placeholder():
    # a bare chain has no name to print; URDF always supplies one
    chain = compile_robot(parse_urdf_string(PLANAR), dtype=torch.float64)
    s = summary(chain, reach_samples=0)
    assert s.name is None
    assert "(unnamed)" in s.to_text()
    assert s.to_markdown().startswith("# robot")


# ---------------------------------------------------------------- dtype

def test_dtype_agnostic():
    f32 = summary(kinfast.load_string(PLANAR, dtype=torch.float32))
    f64 = summary(kinfast.load_string(PLANAR, dtype=torch.float64))
    assert [j["name"] for j in f32.joints] == [j["name"] for j in f64.joints]
    assert f32.total_mass == pytest.approx(f64.total_mass, rel=1e-6)
    assert f32.reach == pytest.approx(f64.reach, rel=1e-6)
    assert f32.sampled == pytest.approx(f64.sampled, rel=1e-5)
    # the numbers come out as plain Python floats either way
    assert isinstance(f32.total_mass, float)
    assert isinstance(f32.reach, float)


def test_summary_leaves_no_autograd_graph_behind():
    # the report is data, not a differentiable quantity: nothing it returns
    # should drag a graph (or a device tensor) along with it
    s = summary(kinfast.load_string(PLANAR))
    for value in [s.total_mass, s.reach, s.sampled] + \
                 [l["mass"] for l in s.links]:
        assert not isinstance(value, torch.Tensor)


def test_summary_rejects_something_that_is_not_a_robot():
    with pytest.raises(TypeError, match="Robot or a CompiledChain"):
        summary(object())
    with pytest.raises(TypeError):
        summary("panda.urdf")


def test_summary_is_a_summary_instance():
    assert isinstance(summary(kinfast.load_string(PLANAR), reach_samples=0),
                      Summary)


# ------------------------------------------------------------- other formats

MJCF_ARM = """
<mujoco model="mj_arm">
  <worldbody>
    <body name="upper" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
      <body name="fore" pos="0.4 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-120 10"/>
        <body name="hand" pos="0.3 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_FLOATER = """
<mujoco model="floater">
  <worldbody>
    <body name="torso" pos="0 0 0.6">
      <freejoint name="root"/>
      <body name="leg" pos="0 0 -0.3">
        <joint name="knee" type="hinge" axis="0 1 0" range="-1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_summary_of_an_mjcf_model():
    s = summary(kinfast.load_string(MJCF_ARM, dtype=torch.float64))
    assert s.name == "mj_arm"
    by_name = {j["name"]: j for j in s.joints}
    assert "shoulder" in by_name and "elbow" in by_name
    # MJCF ranges are degrees by default, so -90 .. 90 must arrive as +-pi/2
    assert by_name["shoulder"]["lower"] == pytest.approx(-math.pi / 2, abs=1e-9)
    assert by_name["shoulder"]["upper"] == pytest.approx(math.pi / 2, abs=1e-9)
    assert by_name["elbow"]["lower"] == pytest.approx(math.radians(-120),
                                                      abs=1e-9)
    # hand-walked bound: 0.5 up, then 0.4 out, then 0.3 out
    assert s.reach == pytest.approx(1.2, abs=1e-9)
    text = s.to_text()
    for name in by_name:
        assert name in text


def test_parser_notes_from_a_real_mjcf_free_joint():
    s = summary(kinfast.load_string(MJCF_FLOATER), reach_samples=0)
    assert s.notes, "a free joint should have produced a parser note"
    text = s.to_text()
    assert "parser notes" in text
    for note in s.notes:
        assert note in text


# --------------------------------------------------------------- real files

_ASSETS = pathlib.Path("C:/Users/vihan/urdf-doctor/examples/assets")
_PANDA = _ASSETS / "panda.urdf"
_GALLERY = _ASSETS / "gallery"


@pytest.mark.skipif(not _PANDA.is_file(), reason="panda.urdf not in this tree")
def test_panda_report_is_complete_and_plausible():
    r = kinfast.load(str(_PANDA))
    s = summary(r)
    for j in r.ir.joints:
        assert j.name in s.to_text()
        assert j.name in s.to_markdown()
    assert s.total_mass == pytest.approx(sum(l["mass"] for l in s.links),
                                         rel=1e-6)
    assert 5.0 < s.total_mass < 40.0        # a Panda weighs roughly 18 kg
    assert 0.5 < s.sampled <= s.reach       # sampling cannot beat the bound
    assert s.reach < 2.0


@pytest.mark.skipif(not _GALLERY.is_dir(), reason="gallery not in this tree")
def test_every_gallery_robot_renders():
    files = sorted(_GALLERY.glob("*.urdf"))
    assert files, "gallery directory is empty"
    for path in files:
        r = kinfast.load(str(path))
        s = summary(r, reach_samples=64)
        text, md = s.to_text(), s.to_markdown()
        for j in r.ir.joints:
            assert j.name in text, f"{j.name} missing from {path.name}"
            assert j.name in md, f"{j.name} missing from {path.name}"
        assert s.total_mass == pytest.approx(sum(l["mass"] for l in s.links),
                                             rel=1e-5, abs=1e-9)
        assert s.dof == r.dof
        assert s.n_links == r.n_links
        if s.sampled is not None:
            assert s.sampled <= s.reach + 1e-6
