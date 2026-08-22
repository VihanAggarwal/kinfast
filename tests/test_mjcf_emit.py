# tests/test_mjcf_emit.py
"""URDF -> MJCF converter, verified the only way that counts: the emitted
model is loaded by MuJoCo and its forward kinematics compared body-by-body
against kinfast's FK of the source URDF. Skips if mujoco is absent."""
import os

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

import kinfast
from kinfast.fk import forward_kinematics

# primitives + inertials + rotated joint origins + a prismatic joint
URDF = """
<robot name="prim">
  <link name="base">
    <inertial><mass value="2"/><origin xyz="0 0 0.05"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.05"/><geometry><box size="0.2 0.2 0.1"/></geometry></visual>
    <collision><origin xyz="0 0 0.05"/><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="arm">
    <inertial><mass value="1"/><origin xyz="0.15 0 0" rpy="0 0.2 0"/>
      <inertia ixx="0.001" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.15 0 0" rpy="0 1.5707963 0"/>
      <geometry><cylinder radius="0.02" length="0.3"/></geometry></visual>
  </link>
  <link name="tip">
    <inertial><mass value="0.3"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><sphere radius="0.03"/></geometry></collision>
  </link>
  <link name="slider">
    <inertial><mass value="0.2"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.1" rpy="0.3 -0.2 0.7"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
  <joint name="wrist" type="continuous"><parent link="arm"/><child link="tip"/>
    <origin xyz="0.3 0 0"/><axis xyz="1 0 0"/></joint>
  <joint name="rail" type="prismatic"><parent link="tip"/><child link="slider"/>
    <origin xyz="0.05 0 0" rpy="0 0 1.2"/><axis xyz="0 0 1"/>
    <limit lower="-0.1" upper="0.1" velocity="1" effort="10"/></joint>
</robot>
"""


def _roundtrip(robot, xml, n=20):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    chain = robot.chain
    addr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): m.jnt_qposadr[j]
            for j in range(m.njnt)}
    cols = [(k, addr[nm]) for k, nm in enumerate(chain.joint_names)]
    rng = np.random.RandomState(0)
    lo, hi = chain.lower.numpy(), chain.upper.numpy()
    for _ in range(n):
        q = lo + (hi - lo) * rng.rand(chain.dof)
        d.qpos[:] = 0
        for k, a in cols:
            d.qpos[a] = q[k]
        mujoco.mj_forward(m, d)
        world = forward_kinematics(chain, torch.tensor(q, dtype=torch.float32).unsqueeze(0))[0]
        for b in range(1, m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            li = chain.link_index[nm]
            assert np.abs(world[li, :3, 3].numpy() - d.xpos[b]).max() < 1e-5, f"{nm} pos"
            assert np.abs(world[li, :3, :3].numpy() - d.xmat[b].reshape(3, 3)).max() < 1e-5, f"{nm} rot"
    return m


def test_emitted_mjcf_loads_and_matches_fk():
    robot = kinfast.load_string(URDF)
    xml = robot.to_mjcf()
    m = _roundtrip(robot, xml)
    assert m.ngeom == 4                       # 2 boxes, 1 cylinder, 1 sphere
    assert m.njnt == 3
    # continuous joint must be unlimited; revolute limited with our range
    jw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wrist")
    js = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")
    assert m.jnt_limited[jw] == 0 and m.jnt_limited[js] == 1
    assert np.allclose(m.jnt_range[js], [-2, 2])


def test_emitted_inertials_match():
    robot = kinfast.load_string(URDF)
    m = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm")
    assert abs(m.body_mass[b] - 1.0) < 1e-9
    # mujoco stores principal inertia; our tensor (rotated by the inertial rpy)
    # must have the same eigenvalues
    assert np.allclose(sorted(m.body_inertia[b]), sorted([0.001, 0.01, 0.01]), atol=1e-9)


PANDA = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "panda.urdf")


@pytest.mark.skipif(not os.path.exists(PANDA), reason="panda.urdf not present")
def test_real_panda_converts_and_matches():
    robot = kinfast.load(PANDA)
    xml = robot.to_mjcf(geometry=False)       # meshes not on disk; kinematics+inertia only
    _roundtrip(robot, xml, n=12)


def test_module_level_convert(tmp_path):
    p = tmp_path / "prim.urdf"
    p.write_text(URDF)
    out = tmp_path / "prim.xml"
    text = kinfast.to_mjcf(str(p), out=str(out))
    assert out.exists() and text.startswith("<mujoco")
    assert kinfast.load(str(out)).dof == 3    # our own MJCF parser reads it back
