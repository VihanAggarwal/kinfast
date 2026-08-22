# tests/test_dynamics_oracle.py
"""Dynamics cross-validated against a real physics engine. MuJoCo computes the
joint-space mass matrix (mj_fullM) and the bias force c(q,qd)+g(q) (qfrc_bias);
kinfast must agree at random (q, qd), including on a real Menagerie robot.

The hand model deliberately uses an off-center, ROTATED inertial frame, which
the parsers must express in the body frame (I_body = R I R^T) to pass.
"""
import glob
import os

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

import kinfast
from kinfast import dynamics as D

ARM = """
<mujoco model="dynarm">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="l1" pos="0 0 0.3">
      <inertial pos="0.1 0.02 0" quat="0.9659258 0 0 0.2588190" mass="2.5"
                diaginertia="0.04 0.03 0.02"/>
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
      <body name="l2" pos="0.4 0 0" euler="0 0.3 0">
        <inertial pos="0.2 0 0.05" mass="1.2" diaginertia="0.02 0.015 0.008"/>
        <joint name="j2" type="hinge" axis="0 1 0" range="-2 2"/>
        <body name="l3" pos="0.35 0 0">
          <inertial pos="0.1 0 0" quat="0.7071068 0.7071068 0 0" mass="0.6"
                    diaginertia="0.004 0.003 0.001"/>
          <joint name="j3" type="slide" axis="1 0 0" range="-0.2 0.2"/>
          <body name="l4" pos="0.15 0 0">
            <inertial pos="0 0 0" mass="0.3" diaginertia="0.001 0.001 0.001"/>
            <joint name="j4" type="hinge" axis="1 0 0" range="-3 3"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _compare_dynamics(xml_or_path, from_path=False, n=12, rtol=2e-4):
    m = (mujoco.MjModel.from_xml_path(xml_or_path) if from_path
         else mujoco.MjModel.from_xml_string(xml_or_path))
    m.dof_armature[:] = 0.0          # kinfast has no armature concept
    d = mujoco.MjData(m)
    from kinfast.compile import compile_robot
    from kinfast.mjcf.parse import parse_mjcf_string, parse_mjcf_file
    from kinfast.urdf.repair import repair
    ir = parse_mjcf_file(xml_or_path) if from_path else parse_mjcf_string(xml_or_path)
    ir, _ = repair(ir)
    chain = compile_robot(ir, dtype=torch.float64)
    addr = {}
    for j in range(m.njnt):
        addr[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = \
            (m.jnt_qposadr[j], m.jnt_dofadr[j])
    cols = [(k, addr[nm]) for k, nm in enumerate(chain.joint_names)]

    rng = np.random.RandomState(0)
    lo, hi = chain.lower.double().numpy(), chain.upper.double().numpy()
    for _ in range(n):
        q = lo + (hi - lo) * rng.rand(chain.dof)
        qd = rng.randn(chain.dof)
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        for k, (qa, va) in cols:
            d.qpos[qa] = q[k]
            d.qvel[va] = qd[k]
        mujoco.mj_forward(m, d)
        M_mj = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M_mj)     # mujoco >= 3.12 signature
        # reorder mujoco dofs into kinfast column order
        order = [va for _, (_, va) in cols]
        M_mj = M_mj[np.ix_(order, order)]
        bias_mj = d.qfrc_bias[order]

        qt = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
        qdt = torch.tensor(qd, dtype=torch.float64).unsqueeze(0)
        chain64 = chain
        M_kf = D.mass_matrix(chain64, qt)[0].numpy()
        bias_kf = (D.coriolis(chain64, qt, qdt) + D.gravity(chain64, qt))[0].numpy()

        scale = max(1.0, np.abs(M_mj).max())
        assert np.abs(M_kf - M_mj).max() < rtol * scale, \
            f"mass matrix mismatch {np.abs(M_kf - M_mj).max():.2e}"
        bscale = max(1.0, np.abs(bias_mj).max())
        assert np.abs(bias_kf - bias_mj).max() < rtol * bscale, \
            f"bias mismatch {np.abs(bias_kf - bias_mj).max():.2e}"


def test_dynamics_matches_mujoco_hand_arm():
    _compare_dynamics(ARM)


UR5E = os.path.join(os.path.dirname(__file__), "..", "examples", "assets",
                    "menagerie", "universal_robots_ur5e", "ur5e.xml")


@pytest.mark.skipif(not (os.path.exists(UR5E) and
                         glob.glob(os.path.join(os.path.dirname(UR5E), "assets", "*"))),
                    reason="UR5e menagerie dir not fetched")
def test_dynamics_matches_mujoco_ur5e():
    _compare_dynamics(UR5E, from_path=True, n=8)
