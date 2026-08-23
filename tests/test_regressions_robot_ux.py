# tests/test_regressions_robot_ux.py
"""Regressions for the Robot convenience surface (kinfast.robot).

Three real bugs, all of them things a first-time user hits in the first five
minutes:

1. transform_points crashed with a bare torch "expected scalar type Float but
   found Double" when the points came in as float32 and q was float64. Every
   other method already follows the rule that q's dtype is the working dtype;
   this one did not, and the error named neither argument.
2. kinfast.load on a Menagerie scene.xml (a worldbody that holds a light, a
   floor, and a single <include> that pulls in the actual robot) returned a
   silent 0-dof robot with one link. Includes are now spliced the way MuJoCo
   does it, relative to the including file; a model that still has no bodies
   raises and names the include.
3. Missing files, unknown ee_link, unknown link names and unknown joint names
   produced bare KeyError('l3') / "'j9' is not in list" / a raw OSError. They
   now say what was asked for, what exists, and what to do.

Oracles are hand-computed rigid-body transforms, plus MuJoCo itself for the
include-expansion case (never kinfast checking kinfast).
"""
import math

import pytest
import torch

import kinfast

F32, F64 = torch.float32, torch.float64

# one revolute joint about z whose child sits 1 m out along x, plus a fixed
# tool frame 1 m above it: every transform below is trivial to do by hand.
ONE_R = """<?xml version="1.0"?>
<robot name="one_r">
  <link name="base"/>
  <link name="arm"/>
  <link name="tool"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="10" velocity="1"/>
  </joint>
  <joint name="tool_fix" type="fixed">
    <parent link="arm"/><child link="tool"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
  </joint>
</robot>
"""

# an MJCF "robot file": two hinges about z, ranges in degrees (the include must
# carry its own <compiler> and <default> across the splice)
INC_ARM = """<mujoco model="inc_arm">
  <compiler angle="degree"/>
  <default>
    <joint damping="0.1" armature="0.01"/>
  </default>
  <worldbody>
    <body name="upper" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-90 90"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body name="fore" pos="0.3 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" range="-90 90"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# a Menagerie-shaped scene: no bodies of its own, the robot arrives by include
SCENE = """<mujoco model="inc_arm scene">
  <include file="parts/inc_arm.xml"/>
  <statistic center="0.2 0 0.4" extent="0.8"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
    <geom name="floor" size="0 0 0.05" type="plane"/>
  </worldbody>
</mujoco>
"""

# included from parts/inc_arm.xml by a bare relative name: it must resolve
# against parts/, not against the directory of the top-level scene
POST = """<mujoco model="post">
  <worldbody>
    <body name="post" pos="0 1 0">
      <joint name="lift" type="slide" axis="0 0 1" range="0 0.5"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write_scene(tmp_path, nested=False, arm=INC_ARM):
    parts = tmp_path / "parts"
    parts.mkdir()
    body = arm
    if nested:
        (parts / "post.xml").write_text(POST)
        body = arm.replace("<worldbody>",
                           '<include file="post.xml"/>\n  <worldbody>', 1)
    (parts / "inc_arm.xml").write_text(body)
    scene = tmp_path / "scene.xml"
    scene.write_text(SCENE)
    return scene


# ---------------------------------------------------------------- bug 1 ----
@pytest.mark.parametrize("q_dtype,p_dtype", [(F64, F32), (F32, F64),
                                             (F64, F64), (F32, F32)])
def test_transform_points_takes_any_points_dtype(q_dtype, p_dtype):
    """points are cast to q's dtype instead of raising a raw torch error."""
    robot = kinfast.load_string(ONE_R)
    ang = math.pi / 2
    q = torch.tensor([[ang]], dtype=q_dtype)
    pts = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=p_dtype)

    out = robot.transform_points(pts, q, "arm")

    assert out.dtype == q_dtype                      # q decides, as documented
    assert out.shape == (1, 2, 3)
    # hand oracle: world_T_arm = trans(1,0,0) @ Rz(q); p -> (1,0,0) + Rz(q) p
    want = torch.tensor([[[1.0 + 0.5 * math.cos(ang), 0.5 * math.sin(ang), 0.0],
                          [1.0, 0.0, 0.0]]], dtype=q_dtype)
    assert torch.allclose(out, want, atol=1e-5)


def test_transform_points_to_named_link_and_list_input():
    """arm -> tool is a pure -1 m z shift, whatever q is, and a plain list of
    points works now that they go through torch.as_tensor."""
    robot = kinfast.load_string(ONE_R)
    q = torch.tensor([[0.3], [-1.2]], dtype=F64)     # batch of 2
    out = robot.transform_points([[0.5, 0.0, 0.0]], q, "arm", to_link="tool")
    assert out.dtype == F64 and out.shape == (2, 1, 3)
    want = torch.tensor([0.5, 0.0, -1.0], dtype=F64)
    assert torch.allclose(out[0, 0], want, atol=1e-9)
    assert torch.allclose(out[1, 0], want, atol=1e-9)


def test_transform_points_keeps_gradients_through_the_cast():
    robot = kinfast.load_string(ONE_R)
    q = torch.tensor([[0.0]], dtype=F64)
    pts = torch.tensor([[0.5, 0.0, 0.0]], dtype=F32, requires_grad=True)
    robot.transform_points(pts, q, "arm").sum().backward()
    # d(sum of world xyz)/dp for identity rotation is (1,1,1)
    assert torch.allclose(pts.grad, torch.ones(1, 3), atol=1e-6)


def test_transform_points_rejects_bad_shape():
    robot = kinfast.load_string(ONE_R)
    q = torch.zeros(1, 1, dtype=F64)
    with pytest.raises(ValueError, match=r"points must be \(N,3\)"):
        robot.transform_points(torch.zeros(4, 2), q, "arm")


# ---------------------------------------------------------------- bug 2 ----
def test_scene_include_is_expanded_relative_to_the_file(tmp_path):
    """A scene.xml whose only robot is behind an <include> used to load as a
    silent 0-dof robot."""
    scene = _write_scene(tmp_path)
    robot = kinfast.load(str(scene))

    assert robot.dof == 2, "the included bodies never made it into the chain"
    assert {"upper", "fore"} <= set(robot.chain.link_names)
    # the include's own <compiler angle="degree"> survived the splice
    lo = robot.lower[robot.q_index("shoulder")].item()
    assert lo == pytest.approx(-math.pi / 2, abs=1e-6)
    # hand oracle: fore sits at pos (0.3,0,0) in upper, upper at (0,0,0.5)
    q = torch.zeros(1, 2, dtype=F64)
    p0 = robot.fk_all(q)[0, robot.link_id("fore"), :3, 3]
    assert torch.allclose(p0, torch.tensor([0.3, 0.0, 0.5], dtype=F64), atol=1e-6)
    q[0, robot.q_index("shoulder")] = math.pi / 2
    p1 = robot.fk_all(q)[0, robot.link_id("fore"), :3, 3]
    assert torch.allclose(p1, torch.tensor([0.0, 0.3, 0.5], dtype=F64), atol=1e-6)


def test_scene_include_matches_mujoco(tmp_path):
    """Independent oracle: MuJoCo compiles the same scene.xml, includes and
    all, and must agree on where the links are."""
    mujoco = pytest.importorskip("mujoco")
    scene = _write_scene(tmp_path)
    robot = kinfast.load(str(scene))

    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    qvals = {"shoulder": 0.4, "elbow": -0.7}
    q = torch.zeros(1, robot.dof, dtype=F64)
    for name, val in qvals.items():
        q[0, robot.q_index(name)] = val
        d.qpos[m.joint(name).qposadr[0]] = val
    mujoco.mj_forward(m, d)

    frames = robot.fk_all(q)
    for body in ("upper", "fore"):
        got = frames[0, robot.link_id(body), :3, 3]
        want = torch.tensor(d.body(body).xpos.copy(), dtype=F64)
        assert torch.allclose(got, want, atol=1e-9), body


def test_nested_include_resolves_against_its_own_file(tmp_path):
    scene = _write_scene(tmp_path, nested=True)
    robot = kinfast.load(str(scene))
    assert robot.dof == 3
    assert "post" in robot.chain.link_names
    assert "lift" in robot.joint_names


def test_scene_string_without_a_directory_names_the_include():
    with pytest.raises(ValueError) as e:
        kinfast.load_string(SCENE)
    msg = str(e.value)
    assert "no bodies" in msg
    assert "parts/inc_arm.xml" in msg            # names the file it could not read
    assert "kinfast.load" in msg                 # says what to do instead


def test_bodyless_model_says_so():
    text = ('<mujoco model="empty_scene"><worldbody>'
            '<light pos="0 0 1"/><geom name="floor" type="plane" size="1 1 1"/>'
            "</worldbody></mujoco>")
    with pytest.raises(ValueError, match="has no bodies"):
        kinfast.load_string(text)


def test_missing_include_file_is_reported(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco model="s"><include file="ghost.xml"/>'
                     "<worldbody/></mujoco>")
    with pytest.raises(FileNotFoundError) as e:
        kinfast.load(str(scene))
    assert "ghost.xml" in str(e.value)


def test_missing_include_beside_real_bodies_is_only_a_note(tmp_path):
    """Partial checkouts drop the actuator/sensor fragments a model includes.
    Those hold no kinematics, so the model must still load, with a note."""
    p = tmp_path / "inc_arm.xml"
    p.write_text(INC_ARM.replace("<worldbody>",
                                 '<include file="actuators.xml"/>\n  <worldbody>',
                                 1))
    robot = kinfast.load(str(p))
    assert robot.dof == 2
    notes = " ".join(robot.ir.parse_notes)
    assert "actuators.xml" in notes and "ignored" in notes


def test_plain_mjcf_file_still_loads(tmp_path):
    """The include machinery must not disturb an ordinary robot xml."""
    p = tmp_path / "inc_arm.xml"
    p.write_text(INC_ARM)
    robot = kinfast.load(str(p))
    assert robot.dof == 2 and "fore" in robot.chain.link_names


# ---------------------------------------------------------------- bug 3 ----
def test_missing_path_message(tmp_path):
    missing = tmp_path / "not_here.urdf"
    with pytest.raises(FileNotFoundError) as e:
        kinfast.load(str(missing))
    msg = str(e.value)
    assert "not_here.urdf" in msg
    assert "load_string" in msg


def test_directory_path_message(tmp_path):
    with pytest.raises(IsADirectoryError) as e:
        kinfast.load(str(tmp_path))
    assert "directory" in str(e.value)


def test_unknown_ee_link_message():
    with pytest.raises(ValueError) as e:
        kinfast.load_string(ONE_R, ee_link="tooll")
    msg = str(e.value)
    assert "unknown link 'tooll'" in msg
    assert "'tool'" in msg                       # close-match suggestion
    assert "ee_link" in msg


def test_unknown_link_id_message():
    robot = kinfast.load_string(ONE_R)
    with pytest.raises(KeyError) as e:
        robot.link_id("armm")
    msg = str(e.value)
    assert "unknown link" in msg and "armm" in msg
    assert "'arm'" in msg                        # and lists what exists
    assert "'base'" in msg


def test_unknown_q_index_message():
    robot = kinfast.load_string(ONE_R)
    with pytest.raises(ValueError) as e:
        robot.q_index("tool_fix")                # real joint, but fixed
    msg = str(e.value)
    assert "unknown movable joint 'tool_fix'" in msg
    assert "fixed joints" in msg
    assert "'j1'" in msg


def test_fk_with_unknown_link_reports_the_name():
    robot = kinfast.load_string(ONE_R)
    q = torch.zeros(1, 1)
    with pytest.raises(KeyError, match="unknown link"):
        robot.fk(q, link="wrist")
    with pytest.raises(KeyError, match="unknown link"):
        robot.jacobian(q, link="wrist")


def test_known_names_still_work():
    """The new error paths must not shadow the happy path."""
    robot = kinfast.load_string(ONE_R)
    assert robot.link_id("tool") == robot.chain.link_names.index("tool")
    assert robot.q_index("j1") == 0
    assert kinfast.load_string(ONE_R, ee_link="arm").ee_link == "arm"
