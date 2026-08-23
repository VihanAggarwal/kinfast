# tests/test_regressions_mjcf_parse_fidelity.py
"""MJCF fidelity regressions, all checked against MuJoCo as the oracle.

Three gaps the parser used to have:
  1. joint `ref` (the reference configuration) was dropped, so every body pose
     past a joint with ref was wrong, and so were M, g and c.
  2. `<option gravity>` was ignored: dynamics always used (0, 0, -9.81).
  3. bodies without <inertial> silently got zero mass, with no way to tell,
     and `<compiler settotalmass>` was ignored.
"""
import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

import kinfast
from kinfast import dynamics as D
from kinfast.fk import forward_kinematics

_INR = '<inertial pos="0.05 0.01 0.02" mass="1.3" diaginertia="0.01 0.02 0.015"/>'

# hinge ref in RADIANS, slide ref in meters, and a ref that arrives from a
# defaults class instead of the joint element.
REF_RAD = f"""
<mujoco model="ref_rad">
  <compiler angle="radian"/>
  <default>
    <default class="tilted">
      <joint ref="-0.7"/>
    </default>
  </default>
  <worldbody>
    <body name="l1" pos="0 0 0.3" euler="0.2 -0.1 0.4">{_INR}
      <joint name="j1" type="hinge" axis="0 1 0" ref="0.4" range="-2 2"/>
      <body name="l2" pos="0.4 0 0">{_INR}
        <joint name="j2" type="slide" axis="1 0.2 0" ref="0.13" range="-0.3 0.3"/>
        <body name="l3" pos="0.2 0 0.1">{_INR}
          <joint name="j3" class="tilted" type="hinge" axis="0 0 1" range="-2 2"/>
          <body name="l4" pos="0.1 0 0">{_INR}</body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# hinge ref in DEGREES (the MJCF default), plus nonzero joint anchors and two
# joints stacked in one body: ref must survive the synthetic-link machinery.
REF_DEG = f"""
<mujoco model="ref_deg">
  <worldbody>
    <body name="gimbal" pos="0.2 0.1 0.4" euler="20 0 30">{_INR}
      <joint name="u1" type="hinge" axis="1 0 0" pos="0.05 0 0.02" ref="25"
             range="-90 90"/>
      <joint name="u2" type="hinge" axis="0 1 0" pos="0.05 0 0.02" ref="-40"
             range="-90 90"/>
      <body name="tip" pos="0.3 0 0">{_INR}
        <joint name="wrist" type="hinge" axis="0 0 1" pos="0.1 0 0" ref="15"
               range="-120 120"/>
        <body name="tool" pos="0.1 0.05 0">{_INR}</body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _cols(m, chain):
    addr = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        addr[nm] = (m.jnt_qposadr[j], m.jnt_dofadr[j])
    return [(k, addr[nm]) for k, nm in enumerate(chain.joint_names)]


def _sample(chain, rng):
    lo = chain.lower.double().numpy()
    hi = chain.upper.double().numpy()
    return lo + (hi - lo) * rng.rand(chain.dof)


@pytest.mark.parametrize("xml", [REF_RAD, REF_DEG], ids=["radian", "degree"])
def test_joint_ref_fk_matches_mujoco(xml):
    """Every body pose must match MuJoCo when joints carry a nonzero ref."""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    assert np.abs(m.qpos0).max() > 0.1, "fixture must actually set ref"
    robot = kinfast.load_string(xml)
    chain = robot.chain
    cols = _cols(m, chain)
    rng = np.random.RandomState(0)
    for _ in range(16):
        q = _sample(chain, rng)
        d.qpos[:] = 0
        for k, (qa, _) in cols:
            d.qpos[qa] = q[k]
        mujoco.mj_forward(m, d)
        world = forward_kinematics(
            chain, torch.tensor(q, dtype=torch.float64).unsqueeze(0))[0]
        for b in range(1, m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            li = chain.link_index[nm]
            assert np.abs(world[li, :3, 3].numpy() - d.xpos[b]).max() < 1e-5, nm
            assert np.abs(world[li, :3, :3].numpy()
                          - d.xmat[b].reshape(3, 3)).max() < 1e-5, nm


def test_joint_ref_dynamics_matches_mujoco():
    """ref shifts the whole configuration, so M(q) and the bias force move too."""
    from kinfast.compile import compile_robot
    from kinfast.mjcf.parse import parse_mjcf_string
    m = mujoco.MjModel.from_xml_string(REF_RAD)
    m.dof_armature[:] = 0.0
    d = mujoco.MjData(m)
    chain = compile_robot(parse_mjcf_string(REF_RAD), dtype=torch.float64)
    cols = _cols(m, chain)
    rng = np.random.RandomState(1)
    for _ in range(8):
        q = _sample(chain, rng)
        qd = rng.randn(chain.dof)
        d.qpos[:] = 0
        d.qvel[:] = 0
        for k, (qa, va) in cols:
            d.qpos[qa] = q[k]
            d.qvel[va] = qd[k]
        mujoco.mj_forward(m, d)
        M_mj = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M_mj)
        order = [va for _, (_, va) in cols]
        M_mj = M_mj[np.ix_(order, order)]
        qt = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
        qdt = torch.tensor(qd, dtype=torch.float64).unsqueeze(0)
        M_kf = D.mass_matrix(chain, qt)[0].numpy()
        bias_kf = (D.coriolis(chain, qt, qdt) + D.gravity(chain, qt))[0].numpy()
        assert np.abs(M_kf - M_mj).max() < 1e-8 * max(1.0, np.abs(M_mj).max())
        bias_mj = d.qfrc_bias[order]
        assert np.abs(bias_kf - bias_mj).max() < 1e-6 * max(1.0, np.abs(bias_mj).max())


ARM_FMT = """
<mujoco model="grav">
  <compiler angle="radian"/>
  {option}
  <worldbody>
    <body name="l1" pos="0 0 0.3">
      <inertial pos="0.15 0.02 0" mass="2.0" diaginertia="0.02 0.03 0.01"/>
      <joint name="j1" type="hinge" axis="0 1 0" range="-2 2"/>
      <body name="l2" pos="0.4 0 0" euler="0 0.3 0">
        <inertial pos="0.2 0 0.03" mass="1.1" diaginertia="0.01 0.012 0.005"/>
        <joint name="j2" type="hinge" axis="1 0 0" range="-2 2"/>
        <body name="l3" pos="0.3 0 0">
          <inertial pos="0.1 0 0" mass="0.4" diaginertia="0.002 0.002 0.001"/>
          <joint name="j3" type="slide" axis="0 0 1" range="-0.2 0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

GRAVITIES = {
    "weak": '<option gravity="0 0 -1"/>',
    "tilted": '<option gravity="1.5 -2.0 -9.0"/>',
    "zero": '<option gravity="0 0 0"/>',
    "default": "",
}


@pytest.mark.parametrize("case", list(GRAVITIES), ids=list(GRAVITIES))
def test_option_gravity_matches_mujoco(case):
    """Generalized gravity torque must use the model's own gravity vector.
    Oracle: MuJoCo's qfrc_bias evaluated at zero velocity."""
    xml = ARM_FMT.format(option=GRAVITIES[case])
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    from kinfast.compile import compile_robot
    from kinfast.mjcf.parse import parse_mjcf_string
    ir = parse_mjcf_string(xml)
    expect = {"weak": (0.0, 0.0, -1.0), "tilted": (1.5, -2.0, -9.0),
              "zero": (0.0, 0.0, 0.0), "default": (0.0, 0.0, -9.81)}[case]
    assert ir.gravity == expect
    chain = compile_robot(ir, dtype=torch.float64)
    assert chain.gravity == expect
    cols = _cols(m, chain)
    rng = np.random.RandomState(2)
    for _ in range(8):
        q = _sample(chain, rng)
        d.qpos[:] = 0
        d.qvel[:] = 0
        for k, (qa, _) in cols:
            d.qpos[qa] = q[k]
        mujoco.mj_forward(m, d)
        order = [va for _, (_, va) in cols]
        g_mj = d.qfrc_bias[order]
        g_kf = D.gravity(chain, torch.tensor(q, dtype=torch.float64)
                         .unsqueeze(0))[0].numpy()
        assert np.abs(g_kf - g_mj).max() < 1e-8 * max(1.0, np.abs(g_mj).max())
    if case == "zero":
        assert np.abs(g_kf).max() == 0.0


def test_gravity_scalar_argument_is_still_honored():
    """The old scalar meaning of g (0, 0, -g) stays available and wins over the
    model's vector, and a 3-vector may be passed explicitly."""
    xml = ARM_FMT.format(option=GRAVITIES["weak"])
    robot = kinfast.load_string(xml)
    q = torch.zeros(1, robot.dof, dtype=torch.float64)
    weak = robot.gravity(q)                      # model gravity: 1 m/s^2 down
    strong = robot.gravity(q, 9.81)              # scalar, old meaning
    assert torch.allclose(strong, weak * 9.81, atol=1e-10)
    vec = D.gravity(robot.chain, q, (0.0, 0.0, -9.81))
    assert torch.allclose(vec, strong, atol=1e-12)


def test_urdf_keeps_default_gravity():
    """URDF cannot express gravity, so nothing changes for URDF models."""
    urdf = """
    <robot name="two">
      <link name="base"><inertial><mass value="1"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
      </inertial></link>
      <link name="tip"><inertial><origin xyz="0.2 0 0"/><mass value="2"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
      </inertial></link>
      <joint name="j" type="revolute">
        <parent link="base"/><child link="tip"/>
        <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/>
      </joint>
    </robot>
    """
    robot = kinfast.load_string(urdf)
    assert robot.ir.gravity == (0.0, 0.0, -9.81)
    assert robot.chain.gravity == (0.0, 0.0, -9.81)
    q = torch.zeros(1, robot.dof, dtype=torch.float64)
    # tip COM 0.2 m out along +x, mass 2, axis +y: dU/dq = -m*g*0.2
    # (the chain is compiled in float32, hence the 1e-5 tolerance)
    assert abs(robot.gravity(q)[0, 0].item() + 2.0 * 9.81 * 0.2) < 1e-5


GEOM_ONLY = """
<mujoco model="geomonly">
  <compiler angle="radian"/>
  <worldbody>
    <body name="a" pos="0 0 0.2">
      <geom type="box" size="0.1 0.1 0.1"/>
      <joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
      <body name="b" pos="0.3 0 0">
        <geom type="sphere" size="0.05"/>
        <joint name="j2" type="hinge" axis="0 1 0" range="-1 1"/>
      </body>
      <body name="c" pos="0 0.3 0">
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
      </body>
      <body name="d" pos="0 -0.3 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_bodies_left_at_zero_mass_are_reported():
    """kinfast does not derive inertia from geoms; it must say which bodies it
    left massless instead of silently producing a wrong model. Bodies MuJoCo
    also leaves massless are reported separately, since those are not a gap."""
    from kinfast.mjcf.parse import parse_mjcf_string
    ir = parse_mjcf_string(GEOM_ONLY)
    notes = [n for n in ir.parse_notes if "zero mass" in n]
    assert len(notes) == 2
    geom_note = [n for n in notes if "does not derive inertia" in n]
    bare_note = [n for n in notes if "no geoms" in n]
    assert len(geom_note) == 1 and len(bare_note) == 1
    def listed(note):
        return [s.strip() for s in note.split(":", 1)[1].split(",")]
    assert listed(geom_note[0]) == ["a", "b"]   # c has an <inertial>, d no geom
    assert listed(bare_note[0]) == ["d"]
    # MuJoCo really does give the geom bodies mass, which is what the note warns
    # about, and really does leave d massless, which is why d is reported apart.
    m = mujoco.MjModel.from_xml_string(GEOM_ONLY)
    for nm in ("a", "b"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        assert m.body_mass[b] > 0.0
        assert ir.links[nm].inertial is None
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "d")
    assert m.body_mass[b] == 0.0
    assert ir.links["d"].inertial is None
    # and the note is reachable from the public Robot, not just the IR
    assert kinfast.load_string(GEOM_ONLY).parse_notes == ir.parse_notes
    assert kinfast.load_string(TOTALMASS).parse_notes != []


TOTALMASS = """
<mujoco model="totalmass">
  <compiler angle="radian" settotalmass="10"/>
  <worldbody>
    <body name="a" pos="0 0 0.2">
      <inertial pos="0 0 0" mass="2" diaginertia="0.02 0.02 0.01"/>
      <joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
      <body name="b" pos="0.3 0 0">
        <inertial pos="0.1 0 0" mass="3" diaginertia="0.03 0.03 0.02"/>
        <joint name="j2" type="hinge" axis="0 1 0" range="-1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_settotalmass_scales_like_mujoco():
    """With every body carrying an <inertial>, our scaling must reproduce
    MuJoCo's body_mass and the dynamics that follow from it."""
    from kinfast.compile import compile_robot
    from kinfast.mjcf.parse import parse_mjcf_string
    ir = parse_mjcf_string(TOTALMASS)
    m = mujoco.MjModel.from_xml_string(TOTALMASS)
    m.dof_armature[:] = 0.0
    d = mujoco.MjData(m)
    assert abs(sum(L.inertial.mass for L in ir.links.values()
                   if L.inertial) - 10.0) < 1e-12
    for nm in ("a", "b"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        assert abs(ir.links[nm].inertial.mass - m.body_mass[b]) < 1e-9
    assert any("settotalmass" in n for n in ir.parse_notes)

    chain = compile_robot(ir, dtype=torch.float64)
    cols = _cols(m, chain)
    rng = np.random.RandomState(3)
    q = _sample(chain, rng)
    d.qpos[:] = 0
    d.qvel[:] = 0
    for k, (qa, _) in cols:
        d.qpos[qa] = q[k]
    mujoco.mj_forward(m, d)
    order = [va for _, (_, va) in cols]
    M_mj = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M_mj)
    qt = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
    M_kf = D.mass_matrix(chain, qt)[0].numpy()
    assert np.abs(M_kf - M_mj[np.ix_(order, order)]).max() < 1e-9
    g_kf = D.gravity(chain, qt)[0].numpy()
    assert np.abs(g_kf - d.qfrc_bias[order]).max() < 1e-8


def test_settotalmass_without_any_mass_is_noted_not_applied():
    from kinfast.mjcf.parse import parse_mjcf_string
    xml = TOTALMASS.replace(
        '<inertial pos="0 0 0" mass="2" diaginertia="0.02 0.02 0.01"/>', ""
    ).replace(
        '<inertial pos="0.1 0 0" mass="3" diaginertia="0.03 0.03 0.02"/>', "")
    ir = parse_mjcf_string(xml)
    assert any("settotalmass=10 ignored" in n for n in ir.parse_notes)
