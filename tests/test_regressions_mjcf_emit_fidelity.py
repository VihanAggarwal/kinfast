# tests/test_regressions_mjcf_emit_fidelity.py
"""Two fidelity regressions in the URDF -> MJCF emitter, both checked against
MuJoCo rather than against kinfast itself.

1. A URDF whose root link is named `world` produced an XML MuJoCo refuses to
   load at all ("repeated name 'world' in body"); xarm6.urdf is one of these.
2. A link with a real inertia tensor but zero mass had its tensor thrown away
   and replaced by a 1e-6 placeholder, so the emitted model's mass matrix
   disagreed with kinfast's: measured against mj_fullM before the fix, 36% on
   the fixture below and 8.9% on the panda.
"""
import os

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

import kinfast
from kinfast.fk import forward_kinematics

# Root link called `world`, the usual URDF way of pinning a robot to the
# world frame. Two branches off it, so the emitter cannot get away with
# assuming the root has a single child.
WORLD_ROOT = """
<robot name="world_rooted">
  <link name="world"/>
  <link name="base">
    <inertial><mass value="2"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.2 0.2 0.1"/></geometry></visual>
  </link>
  <link name="arm">
    <inertial><origin xyz="0 0 0.15"/><mass value="1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.002" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="post">
    <inertial><mass value="0.5"/><inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="mount" type="fixed"><parent link="world"/><child link="base"/>
    <origin xyz="0 0 0.05" rpy="0 0 0.4"/></joint>
  <joint name="stand" type="fixed"><parent link="world"/><child link="post"/>
    <origin xyz="0.4 0 0"/></joint>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.1" rpy="0.3 -0.2 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
</robot>
"""

# `l2` carries a real inertia tensor with zero mass: the flange/tool-frame
# pattern that real URDFs (panda, xarm) use.
ZERO_MASS_INERTIA = """
<robot name="zero_mass">
  <link name="base">
    <inertial><mass value="2"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l1">
    <inertial><origin xyz="0 0 0.15"/><mass value="1.5"/>
      <inertia ixx="0.012" iyy="0.011" izz="0.004" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l2">
    <inertial><origin xyz="0.05 0 0.02"/><mass value="0"/>
      <inertia ixx="0.008" iyy="0.006" izz="0.005" ixy="0.0005" ixz="0" iyz="0.0002"/></inertial>
  </link>
  <link name="l3">
    <inertial><origin xyz="0 0 0.1"/><mass value="0.7"/>
      <inertia ixx="0.005" iyy="0.005" izz="0.002" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.3" rpy="0.2 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0.1 0 0.05"/><axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
</robot>
"""


def _dof_columns(m, chain):
    """kinfast joint order -> (qpos address, dof address) in the loaded model."""
    addr = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        addr[nm] = (m.jnt_qposadr[j], m.jnt_dofadr[j])
    return [addr[nm] for nm in chain.joint_names]


def test_world_named_root_loads_in_mujoco():
    robot = kinfast.load_string(WORLD_ROOT)
    xml = robot.to_mjcf()
    m = mujoco.MjModel.from_xml_string(xml)      # used to raise: repeated name
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
             for b in range(m.nbody)}
    # exactly one "world": MuJoCo's own top-level body, which is body 0
    assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, 0) == "world"
    assert names == {"world", "base", "arm", "post"}
    # the root's own geometry is not lost when its body is elided
    assert m.ngeom == 1


def test_world_named_root_keeps_kinematics():
    """FK of the emitted model must still match kinfast's FK of the URDF."""
    robot = kinfast.load_string(WORLD_ROOT)
    m = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    d = mujoco.MjData(m)
    chain = robot.chain
    cols = _dof_columns(m, chain)
    rng = np.random.RandomState(0)
    lo, hi = chain.lower.numpy(), chain.upper.numpy()
    for _ in range(10):
        q = lo + (hi - lo) * rng.rand(chain.dof)
        d.qpos[:] = 0
        for k, (qa, _va) in enumerate(cols):
            d.qpos[qa] = q[k]
        mujoco.mj_forward(m, d)
        world = forward_kinematics(
            chain, torch.tensor(q, dtype=torch.float32).unsqueeze(0))[0]
        for b in range(1, m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            li = chain.link_index[nm]
            assert np.abs(world[li, :3, 3].numpy() - d.xpos[b]).max() < 1e-5, f"{nm} pos"
            assert np.abs(world[li, :3, :3].numpy()
                          - d.xmat[b].reshape(3, 3)).max() < 1e-5, f"{nm} rot"


def test_non_root_link_named_world_is_renamed():
    """`world` is reserved wherever it appears, not only at the root."""
    urdf = WORLD_ROOT.replace('name="world"', 'name="anchor"') \
                     .replace('link="world"', 'link="anchor"') \
                     .replace('name="post"', 'name="world"') \
                     .replace('link="post"', 'link="world"')
    robot = kinfast.load_string(urdf)
    m = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
             for b in range(m.nbody)]
    assert names.count("world") == 1            # only MuJoCo's own body 0
    assert "world_link" in names                # the URDF link, renamed
    assert "anchor" in names


def test_zero_mass_link_keeps_its_inertia():
    robot = kinfast.load_string(ZERO_MASS_INERTIA)
    m = mujoco.MjModel.from_xml_string(robot.to_mjcf(geometry=False))
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l2")
    assert m.body_mass[b] < 1e-6                # placeholder mass only
    # MuJoCo diagonalises; compare eigenvalues of the source tensor
    src = np.array([[0.008, 0.0005, 0.0],
                    [0.0005, 0.006, 0.0002],
                    [0.0, 0.0002, 0.005]])
    assert np.allclose(sorted(m.body_inertia[b]),
                       sorted(np.linalg.eigvalsh(src)), atol=1e-9)


def test_mass_matrix_matches_mujoco_with_zero_mass_link():
    """Independent oracle: mj_fullM on the emitted model vs kinfast's M(q)."""
    robot = kinfast.load_string(ZERO_MASS_INERTIA)
    m = mujoco.MjModel.from_xml_string(robot.to_mjcf(geometry=False))
    d = mujoco.MjData(m)
    chain = robot.chain
    cols = _dof_columns(m, chain)
    order = [va for _qa, va in cols]
    rng = np.random.RandomState(0)
    for _ in range(8):
        q = rng.uniform(-1.5, 1.5, chain.dof)
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        for k, (qa, _va) in enumerate(cols):
            d.qpos[qa] = q[k]
        mujoco.mj_forward(m, d)
        M_mj = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M_mj)             # mujoco >= 3.12 signature
        M_mj = M_mj[np.ix_(order, order)]
        M_kf = robot.mass_matrix(
            torch.tensor(q, dtype=torch.float64).unsqueeze(0))[0].numpy()
        scale = max(1.0, np.abs(M_mj).max())
        assert np.abs(M_kf - M_mj).max() < 1e-6 * scale, \
            f"mass matrix mismatch {np.abs(M_kf - M_mj).max():.2e}"


_ASSETS = os.path.join(os.path.dirname(__file__), "..", "examples", "assets")
XARM6 = os.path.join(_ASSETS, "gallery", "xarm6.urdf")
PANDA = os.path.join(_ASSETS, "panda.urdf")


@pytest.mark.skipif(not os.path.exists(XARM6), reason="xarm6.urdf not present")
def test_xarm6_emits_a_loadable_model():
    """The robot that first hit the reserved-name bug: its root link is world."""
    robot = kinfast.load(XARM6)
    assert robot.ir.root_link() == "world"
    xml = robot.to_mjcf(geometry=False)
    assert '<body name="world"' not in xml
    # Separate defect in xarm6's own data, which kinfast's repair() already
    # reports: link2's principal inertias break the triangle inequality, and
    # MuJoCo refuses that. Let MuJoCo balance it so this test is about the name.
    xml = xml.replace('<compiler angle="radian"',
                      '<compiler angle="radian" balanceinertia="true"')
    m = mujoco.MjModel.from_xml_string(xml)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
             for b in range(m.nbody)]
    assert names.count("world") == 1 and names[0] == "world"


@pytest.mark.skipif(not os.path.exists(PANDA), reason="panda.urdf not present")
def test_panda_mass_matrix_matches_mujoco():
    """Panda has two zero-mass inertial links (panda_link8, panda_grasptarget);
    M used to be off by 8.9% against mj_fullM.

    The Panda hand is also the reason this test is not a plain elementwise
    comparison. Its URDF declares the second finger as a <mimic> of the first,
    so kinfast reduces the model to one gripper coordinate, while MuJoCo keeps
    a joint per finger because it ignores the tag on URDF import. The two are
    not in disagreement: with one actuator driving both fingers, the inertia
    felt at that actuator is the sum of theirs. The relation between the two
    matrices is exactly the coordinate change M_reduced = A^T M_full A, where A
    maps the actuated vector onto every joint, so that is what gets asserted.
    It is a stronger check than the old one, because it pins the reduction as
    well as the inertias."""
    robot = kinfast.load(PANDA)
    m = mujoco.MjModel.from_xml_string(robot.to_mjcf(geometry=False))
    d = mujoco.MjData(m)
    chain = robot.chain

    # A[j, c] is how far joint j moves per unit of actuated coordinate c
    full = chain.movable_joint_names()
    jaddr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): (
        m.jnt_qposadr[j], m.jnt_dofadr[j]) for j in range(m.njnt)}
    A = np.zeros((len(full), chain.dof))
    movable = [i for i in range(chain.n_links) if int(chain.q_index[i]) >= 0]
    for row, i in enumerate(movable):
        A[row, int(chain.q_index[i])] = float(chain.joint_scale[i])
    order = [jaddr[nm][1] for nm in full]

    rng = np.random.RandomState(0)
    lo, hi = chain.lower.double().numpy(), chain.upper.double().numpy()
    for _ in range(6):
        q = lo + (hi - lo) * rng.rand(chain.dof)
        qt = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
        qfull = chain.expand_q(qt)[0].numpy()
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        for nm, val in zip(full, qfull):
            d.qpos[jaddr[nm][0]] = val
        mujoco.mj_forward(m, d)
        M_mj = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M_mj)
        M_mj = M_mj[np.ix_(order, order)]
        M_reduced = A.T @ M_mj @ A
        M_kf = robot.mass_matrix(qt)[0].numpy()
        scale = max(1.0, np.abs(M_reduced).max())
        assert np.abs(M_kf - M_reduced).max() < 1e-6 * scale
