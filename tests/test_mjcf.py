# tests/test_mjcf.py
"""MJCF parsing cross-validated against real MuJoCo: each model is loaded into
BOTH mujoco and kinfast, and every body's world transform must agree at random
configurations. This catches exactly the traps the format hides (degrees by
default, intrinsic euler, quat order, joint anchors, defaults classes).

Skips cleanly if mujoco is not installed.
"""
import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

import kinfast
from kinfast.fk import forward_kinematics

# 1. plain nested arm, DEGREES (the MJCF default), ranges in degrees
ARM = """
<mujoco model="arm">
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

# 2. orientations: quat (w x y z) and euler (intrinsic xyz, degrees)
ORIENT = """
<mujoco model="orient">
  <worldbody>
    <body name="a" pos="0.1 0.2 0.3" quat="0.9238795 0 0 0.3826834">
      <joint name="j1" type="hinge" axis="0 0 1" range="-180 180"/>
      <body name="b" pos="0.2 0 0" euler="30 20 10">
        <joint name="j2" type="hinge" axis="1 0 0" range="-180 180"/>
        <body name="c" pos="0 0.15 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# 3. joint anchor: rotation about a point that is NOT the body origin
ANCHOR = """
<mujoco model="anchor">
  <compiler angle="radian"/>
  <worldbody>
    <body name="door" pos="0.5 0 0">
      <joint name="hinge" type="hinge" axis="0 0 1" pos="-0.5 0 0" range="-3 3"/>
      <body name="knob" pos="0.4 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""

# 4. slide joint (range is meters: must NOT be degree-converted)
SLIDE = """
<mujoco model="slide">
  <worldbody>
    <body name="car" pos="0 0 0.1">
      <joint name="rail" type="slide" axis="1 0 0" range="-2 2"/>
      <body name="mast" pos="0 0 0.5">
        <joint name="lift" type="slide" axis="0 0 1" range="0 1"/>
        <body name="cart" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# 5. defaults classes + childclass inheritance
DEFAULTS = """
<mujoco model="defaults">
  <compiler angle="radian"/>
  <default>
    <joint type="hinge" axis="0 1 0" range="-1 1"/>
    <default class="wide">
      <joint range="-3 3"/>
    </default>
  </default>
  <worldbody>
    <body name="a" pos="0 0 0.2" childclass="wide">
      <joint name="j1"/>
      <body name="b" pos="0.3 0 0">
        <joint name="j2" class="wide" axis="0 0 1"/>
        <body name="c" pos="0.2 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# 6. the gauntlet: stacked joints in one body (universal), nonzero anchors,
# and a rotated body, all at once
STACKED = """
<mujoco model="stacked">
  <compiler angle="radian"/>
  <worldbody>
    <body name="gimbal" pos="0.2 0.1 0.4" euler="0.3 0 0.5">
      <joint name="u1" type="hinge" axis="1 0 0" pos="0.05 0 0" range="-2 2"/>
      <joint name="u2" type="hinge" axis="0 1 0" pos="0.05 0 0" range="-2 2"/>
      <body name="tip" pos="0.3 0 0">
        <joint name="wrist" type="hinge" axis="0 0 1" range="-3 3"/>
        <body name="tool" pos="0.1 0.05 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

MODELS = {"arm": ARM, "orient": ORIENT, "anchor": ANCHOR, "slide": SLIDE,
          "defaults": DEFAULTS, "stacked": STACKED}


_INERTIAL = '<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>'


def _with_inertials(xml):
    """MuJoCo refuses massless moving bodies, so give every body an inertial.
    Both parsers receive the identical patched XML."""
    import re
    xml = re.sub(r'<body ([^>]*[^/])>', r'<body \1>' + _INERTIAL, xml)
    xml = re.sub(r'<body ([^>]*)/>', r'<body \1>' + _INERTIAL + '</body>', xml)
    return xml


def _compare(xml, n_cfg=24, atol=1e-5):
    xml = _with_inertials(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    robot = kinfast.load_string(xml)
    chain = robot.chain

    # map kinfast q columns to mujoco qpos addresses BY JOINT NAME
    addr = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        addr[nm] = m.jnt_qposadr[j]
    cols = [(k, addr[nm]) for k, nm in enumerate(chain.joint_names)]

    rng = np.random.RandomState(0)
    lo = chain.lower.numpy()
    hi = chain.upper.numpy()
    for _ in range(n_cfg):
        q = lo + (hi - lo) * rng.rand(chain.dof)
        d.qpos[:] = 0
        for k, a in cols:
            d.qpos[a] = q[k]
        mujoco.mj_forward(m, d)
        world = forward_kinematics(chain, torch.tensor(q, dtype=torch.float32)
                                   .unsqueeze(0))[0]
        for b in range(1, m.nbody):          # skip mujoco's world body
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            li = chain.link_index[nm]
            ours_p = world[li, :3, 3].numpy()
            ours_R = world[li, :3, :3].numpy()
            assert np.abs(ours_p - d.xpos[b]).max() < atol, f"{nm} pos"
            assert np.abs(ours_R - d.xmat[b].reshape(3, 3)).max() < atol, f"{nm} rot"


@pytest.mark.parametrize("name", list(MODELS), ids=list(MODELS))
def test_mjcf_matches_mujoco(name):
    _compare(MODELS[name])


def test_full_stack_on_mjcf_robot():
    """An MJCF-loaded robot gets the whole library: batched IK and the scalar
    compiler, not just FK."""
    robot = kinfast.load_string(_with_inertials(ARM))
    torch.manual_seed(0)
    target = robot.fk(robot.random_configs(64))
    q_sol, _ = robot.ik(target, iters=100, pos_only=True, restarts=8)
    err = (robot.fk(q_sol)[:, :3, 3] - target[:, :3, 3]).norm(dim=-1)
    assert (err < 5e-2).float().mean() > 0.9

    fast = robot.compile()
    q = robot.random_configs(4)
    ref = robot.fk_all(q)
    for k in range(4):
        got = fast.fk(q[k].tolist())
        assert np.abs(got - ref[k].numpy()).max() < 1e-5


def test_limits_converted_correctly():
    robot = kinfast.load_string(ARM)                  # degrees by default
    k = robot.q_index("shoulder")
    assert abs(robot.lower[k].item() + np.pi / 2) < 1e-6
    robot2 = kinfast.load_string(SLIDE)               # slide ranges stay meters
    k2 = robot2.q_index("rail")
    assert robot2.lower[k2].item() == -2.0
