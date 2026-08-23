# tests/test_regressions_examples_polish.py
"""Regressions for the examples polish pass.

Covers three fixed bugs:
  1. examples/menagerie.py read parse notes off the Robot wrapper, where they
     never existed (they live on the IR), so the table never showed them.
  2. examples/demo_10k_arms.py always sliced 48 frames, so any batch smaller
     than 48 blew up with an IndexError that main() swallowed into a one-line
     "gif skipped" message and no gif.
  3. Em-dashes in the shipped prose of src/ and examples/.
"""
import importlib.util
import os
import sys

import pytest
import torch

import kinfast
from kinfast.robot import Robot as RobotWrapper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")


def _load_example(name):
    """Import an examples/*.py module by path, without leaving it on sys.path."""
    path = os.path.join(EXAMPLES, name + ".py")
    if not os.path.exists(path):
        pytest.skip(f"{path} is not present")
    spec = importlib.util.spec_from_file_location("_ex_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:            # optional dependency of the example
        del sys.modules[spec.name]
        pytest.skip(f"{name} needs an unavailable dependency: {e}")
    return mod


# an MJCF whose base is a free body: the parser pins it to the world and says so
FREE_BASE = """<mujoco model="freebase">
  <worldbody>
    <body name="base" pos="0 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 0 1" range="-90 90"/>
        <inertial pos="0 0 0.05" mass="1" diaginertia="1 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

PLAIN_URDF = """<robot name="two">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>"""


# ---- bug 1: parse notes never reached the report ----

def test_parse_notes_reach_the_loaded_robot():
    robot = kinfast.load_string(FREE_BASE)
    notes = robot.parse_notes
    assert isinstance(notes, list)
    assert notes, "the free base should have produced a parse note"
    assert notes == list(robot.ir.parse_notes)
    assert any("free" in n for n in notes)


def test_parse_notes_are_an_empty_list_when_nothing_was_reinterpreted():
    robot = kinfast.load_string(PLAIN_URDF)
    assert robot.parse_notes == []
    # a chain-only Robot has no IR to ask and must still answer
    bare = RobotWrapper(robot.chain)
    assert bare.parse_notes == []


def test_parse_notes_are_a_copy_not_the_live_ir_list():
    robot = kinfast.load_string(FREE_BASE)
    robot.parse_notes.append("scribble")
    assert "scribble" not in robot.ir.parse_notes


def test_menagerie_parse_tier_reports_the_notes(tmp_path):
    men = _load_example("menagerie")
    path = tmp_path / "freebase.xml"
    path.write_text(FREE_BASE, encoding="utf-8")
    robot, notes = men.parse_tier(str(path))
    assert robot.dof == 1
    assert notes, "parse_tier dropped the parse notes"
    assert notes == "; ".join(robot.ir.parse_notes)


def test_menagerie_parse_tier_is_quiet_for_a_clean_model(tmp_path):
    men = _load_example("menagerie")
    path = tmp_path / "clean.urdf"
    path.write_text(PLAIN_URDF, encoding="utf-8")
    robot, notes = men.parse_tier(str(path))
    assert robot.dof == 2
    assert notes == ""


# ---- bug 2: the gif was skipped for any batch under 48 ----

def _small_robot():
    return kinfast.load_string(PLAIN_URDF)


def test_render_gif_clamps_frames_to_a_small_batch(tmp_path):
    demo = _load_example("demo_10k_arms")
    robot = _small_robot()
    torch.manual_seed(0)
    q = robot.random_configs(5)
    out = tmp_path / "demo.gif"
    demo.render_gif(robot, q, str(out))          # default k=48, batch is 5
    assert out.exists() and out.stat().st_size > 0
    # independent oracle: pillow counts the frames actually written
    PIL = pytest.importorskip("PIL.Image")
    with PIL.open(str(out)) as im:
        assert im.n_frames == 5


def test_render_gif_still_honours_a_k_smaller_than_the_batch(tmp_path):
    demo = _load_example("demo_10k_arms")
    robot = _small_robot()
    torch.manual_seed(0)
    q = robot.random_configs(12)
    out = tmp_path / "demo_k3.gif"
    demo.render_gif(robot, q, str(out), k=3)
    PIL = pytest.importorskip("PIL.Image")
    with PIL.open(str(out)) as im:
        assert im.n_frames == 3


def test_render_gif_rejects_a_frame_count_below_one(tmp_path):
    demo = _load_example("demo_10k_arms")
    robot = _small_robot()
    torch.manual_seed(0)
    q = robot.random_configs(4)
    with pytest.raises(ValueError, match="at least 1"):
        demo.render_gif(robot, q, str(tmp_path / "nope.gif"), k=0)


def test_render_gif_names_the_reason_for_an_empty_batch(tmp_path):
    demo = _load_example("demo_10k_arms")
    robot = _small_robot()
    q = robot.random_configs(1)[:0]
    with pytest.raises(ValueError, match="empty"):
        demo.render_gif(robot, q, str(tmp_path / "nope.gif"))


# ---- bug 3: em-dashes in shipped prose ----

def _shipped_python_files():
    for base in (os.path.join(ROOT, "src", "kinfast"), EXAMPLES):
        for dirpath, _dirs, names in os.walk(base):
            for n in names:
                if n.endswith(".py"):
                    yield os.path.join(dirpath, n)


EM_DASH = chr(0x2014)   # by codepoint, so this file carries no em-dash itself


def test_no_em_dashes_in_src_or_examples():
    offenders = []
    for path in _shipped_python_files():
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if EM_DASH in line:
                    offenders.append(f"{os.path.relpath(path, ROOT)}:{i}")
    assert offenders == [], "em-dash found in " + ", ".join(offenders)
