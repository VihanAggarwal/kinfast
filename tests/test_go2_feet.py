# tests/test_go2_feet.py
"""Tests for examples/go2_feet.py, the quadruped stance IK example.

The Go2 model itself is not shipped with kinfast (it lives in the gitignored
examples/assets tree), so the oracle-backed tests run on a small quadruped MJCF
written out in full below. It is built to exercise the same features the Go2
uses: a free-jointed trunk, four legs with an abduction joint and two pitch
joints, and foot spheres declared through nested MuJoCo `<default>` classes.

Where an independent check is possible it is used:
  - foot contact points and solved stances are compared against MuJoCo, which
    knows nothing about kinfast,
  - foot Jacobians are compared against float64 central differences,
  - the contact offsets read out of the XML are compared against the literal
    numbers written in the fixture.
The tests that need the real Go2 skip cleanly when the asset is absent.
"""
import importlib.util
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
GO2 = os.path.join(EXAMPLES, "assets", "menagerie", "unitree_go2", "go2.xml")


def _load_example(name="go2_feet"):
    path = os.path.join(EXAMPLES, name + ".py")
    if not os.path.exists(path):
        pytest.skip(f"{path} is not present")
    spec = importlib.util.spec_from_file_location("_ex_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        del sys.modules[spec.name]
        pytest.skip(f"{name} needs an unavailable dependency: {e}")
    return mod


go2_feet = _load_example()


def _go2_path():
    """The real Go2 model, if this checkout has it. examples/assets is
    gitignored, so KINFAST_GO2_XML can point at a copy somewhere else."""
    for p in (os.environ.get("KINFAST_GO2_XML"), GO2):
        if p and os.path.isfile(p):
            return p
    return None


# --------------------------------------------------------------- the fixture

HIP_X, HIP_Y = 0.15, 0.06        # trunk to abduction joint
THIGH_Y = 0.05                   # abduction joint to thigh plane
SEG = 0.18                       # thigh and calf length
FOOT_Z = -0.18                   # contact sphere inside the calf frame
FOOT_R = 0.02


def _leg(label, sx, sy):
    """One leg of the fixture: abduction, thigh, calf, foot sphere.

    The calf also carries a capsule (not a contact sphere) and a sphere higher
    up the shin, so the reader that picks the foot out of the XML has to choose
    the lowest sphere rather than the first geom it meets.
    """
    return f"""
      <body name="{label}_hip" pos="{sx * HIP_X} {sy * HIP_Y} 0">
        <inertial pos="0 0 0" mass="0.4" diaginertia="0.001 0.001 0.001"/>
        <joint name="{label}_abd" class="abd"/>
        <body name="{label}_thigh" pos="0 {sy * THIGH_Y} 0">
          <inertial pos="0 0 -0.09" mass="0.9" diaginertia="0.004 0.004 0.001"/>
          <joint name="{label}_thigh" class="thigh"/>
          <geom type="capsule" size="0.012 0.09" pos="0 0 -0.09"/>
          <body name="{label}_calf" pos="0 0 {-SEG}">
            <inertial pos="0 0 -0.09" mass="0.2" diaginertia="0.002 0.002 0.0001"/>
            <joint name="{label}_knee" class="knee"/>
            <geom type="capsule" size="0.01 0.08" pos="0 0 -0.08"/>
            <geom size="0.015" pos="0 0 -0.05"/>
            <geom name="{label}_foot" class="foot"/>
          </body>
        </body>
      </body>"""


TOY4 = f"""<mujoco model="toy4">
  <compiler angle="radian"/>
  <default>
    <default class="dog">
      <joint axis="0 1 0" damping="1" armature="0.01"/>
      <geom rgba="0.5 0.5 0.5 1"/>
      <default class="abd">
        <joint axis="1 0 0" range="-0.8 0.8"/>
      </default>
      <default class="thigh">
        <joint range="-1.4 3.0"/>
      </default>
      <default class="knee">
        <joint range="-2.4 -0.4"/>
      </default>
      <default class="collision">
        <geom condim="3"/>
        <default class="foot">
          <geom size="{FOOT_R}" pos="0 0 {FOOT_Z}" friction="0.8"/>
        </default>
      </default>
    </default>
  </default>
  <worldbody>
    <body name="trunk" pos="0 0 0.35" childclass="dog">
      <inertial pos="0 0 0" mass="4" diaginertia="0.05 0.1 0.1"/>
      <freejoint/>
      <geom type="box" size="0.18 0.06 0.04"/>
{_leg("FL", 1, 1)}
{_leg("FR", 1, -1)}
{_leg("RL", -1, 1)}
{_leg("RR", -1, -1)}
    </body>
  </worldbody>
</mujoco>"""


@pytest.fixture(scope="module")
def toy():
    import kinfast
    robot = kinfast.load_string(TOY4, dtype=torch.float64)
    return robot


@pytest.fixture(scope="module")
def toy_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("toy4") / "toy4.xml"
    p.write_text(TOY4, encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def toy_feet(toy, toy_path):
    return go2_feet.find_feet(toy, mjcf_path=toy_path)


def _mujoco_foot_points(xml, q, joint_names, foot_geoms, trunk="trunk"):
    """Foot geom positions in the trunk frame, computed by MuJoCo.

    q is (dof,) ordered like kinfast's joint_names. The free joint keeps its
    qpos0 value, which is what kinfast does when it pins a free base.
    """
    mujoco = pytest.importorskip("mujoco")
    import numpy as np
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    for k, name in enumerate(joint_names):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[j]] = float(q[k])
    mujoco.mj_forward(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, trunk)
    R = d.xmat[b].reshape(3, 3)
    t = d.xpos[b]
    out = []
    for g in foot_geoms:
        gi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        out.append(R.T @ (np.asarray(d.geom_xpos[gi]) - t))
    return torch.tensor(np.stack(out), dtype=torch.float64)


def _random_q(robot, n, seed):
    g = torch.Generator().manual_seed(seed)
    lo = robot.chain.lower.to(torch.float64)
    hi = robot.chain.upper.to(torch.float64)
    u = torch.rand(n, robot.dof, dtype=torch.float64, generator=g)
    return lo + (hi - lo) * u


# ------------------------------------------------------------ identification

def test_finds_four_feet_and_labels_them_by_quadrant(toy_feet):
    assert toy_feet.labels == ("FL", "FR", "RL", "RR")
    assert toy_feet.links == ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
    assert toy_feet.base_link == "trunk"
    # hips: +x forward for the F legs, +y left for the L legs
    hips = toy_feet.hips
    assert hips[0].tolist() == pytest.approx([HIP_X, HIP_Y, 0.0])
    assert hips[1].tolist() == pytest.approx([HIP_X, -HIP_Y, 0.0])
    assert hips[2].tolist() == pytest.approx([-HIP_X, HIP_Y, 0.0])
    assert hips[3].tolist() == pytest.approx([-HIP_X, -HIP_Y, 0.0])


def test_labels_do_not_depend_on_the_name_order(toy, toy_path):
    """Feeding the foot links in a scrambled order still yields FL, FR, RL, RR
    in that order, because the labels come from geometry and not from names."""
    feet = go2_feet.find_feet(toy, mjcf_path=toy_path,
                              links=["RR_calf", "FL_calf", "RL_calf", "FR_calf"])
    assert feet.links == ("FL_calf", "FR_calf", "RL_calf", "RR_calf")


def test_contact_spheres_resolve_nested_default_classes(toy_feet):
    """`class="foot"` inherits through collision -> dog, and the lowest sphere
    in the body wins over the decorative one further up the shin."""
    for k in range(4):
        assert toy_feet.offsets[k].tolist() == pytest.approx([0.0, 0.0, FOOT_Z])
        assert float(toy_feet.radii[k]) == pytest.approx(FOOT_R)


def test_without_the_xml_the_contact_point_is_the_link_origin(toy):
    feet = go2_feet.find_feet(toy)
    assert torch.count_nonzero(feet.offsets) == 0
    assert torch.count_nonzero(feet.radii) == 0


def test_non_quadruped_gets_a_clear_error():
    import kinfast
    from tests.test_parse import TWO_LINK
    robot = kinfast.load_string(TWO_LINK)
    with pytest.raises(ValueError, match="expected 4 foot links"):
        go2_feet.find_feet(robot)


# ------------------------------------------------------- foot kinematics

def test_foot_points_match_mujoco(toy, toy_feet):
    """MuJoCo is the oracle: it places the same foot spheres itself."""
    qs = _random_q(toy, 5, seed=0)
    mine = go2_feet.foot_points(toy, qs, toy_feet)
    geoms = [f"{lab}_foot" for lab in ("FL", "FR", "RL", "RR")]
    for b in range(qs.shape[0]):
        ref = _mujoco_foot_points(TOY4, qs[b], toy.joint_names, geoms)
        assert torch.allclose(mine[b], ref, atol=1e-9)


def test_foot_points_ignore_the_free_base_offset(toy, toy_feet):
    """The trunk sits 0.35 m up in the world; a trunk-frame foot must not."""
    q = _random_q(toy, 1, seed=3)
    pts = go2_feet.foot_points(toy, q, toy_feet)
    assert float(pts[0, :, 2].max()) < 0.0
    assert float(pts[0, :, 2].min()) > -2 * SEG - abs(FOOT_Z)


def test_foot_jacobians_match_finite_differences(toy, toy_feet):
    q = _random_q(toy, 3, seed=1)
    J = go2_feet.foot_jacobians(toy, q, toy_feet)
    assert J.shape == (3, 4, 3, toy.dof)
    eps = 1e-6
    for a in range(toy.dof):
        d = torch.zeros_like(q)
        d[:, a] = eps
        plus = go2_feet.foot_points(toy, q + d, toy_feet)
        minus = go2_feet.foot_points(toy, q - d, toy_feet)
        fd = (plus - minus) / (2 * eps)
        assert torch.allclose(J[..., a], fd, atol=1e-7), f"joint {a}"


def test_foot_jacobian_columns_are_zero_for_other_legs(toy, toy_feet):
    """A leg's foot only moves with its own three joints. That is why one
    stacked 12-row solve does not make the legs fight each other."""
    q = _random_q(toy, 1, seed=2)
    J = go2_feet.foot_jacobians(toy, q, toy_feet)[0]
    for k in range(4):
        own = [i for i, n in enumerate(toy.joint_names)
               if n.startswith(toy_feet.labels[k])]
        assert len(own) == 3
        others = [i for i in range(toy.dof) if i not in own]
        assert torch.all(J[k][:, others] == 0)
        assert float(J[k][:, own].abs().max()) > 0


def test_foot_points_are_differentiable(toy, toy_feet):
    q = _random_q(toy, 2, seed=4).requires_grad_(True)
    pts = go2_feet.foot_points(toy, q, toy_feet)
    pts.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


# ------------------------------------------------------------------- targets

def test_stance_targets_stand_under_the_hips(toy, toy_feet):
    heights = [0.20, 0.30]
    tgt = go2_feet.stance_targets(toy, toy_feet, heights)
    assert tgt.shape == (2, 4, 3)
    for b, h in enumerate(heights):
        for k in range(4):
            assert float(tgt[b, k, 2]) == pytest.approx(-(h - FOOT_R))
            assert float(tgt[b, k, 0]) == pytest.approx(toy_feet.hips[k, 0])
            expect_y = float(toy_feet.hips[k, 1]) + math.copysign(
                THIGH_Y, float(toy_feet.hips[k, 1]))
            assert float(tgt[b, k, 1]) == pytest.approx(expect_y)


def test_stance_shifts_widen_the_stance_outward(toy, toy_feet):
    base = go2_feet.stance_targets(toy, toy_feet, [0.25])[0]
    wide = go2_feet.stance_targets(toy, toy_feet, [0.25], dx=0.03, dy=0.02)[0]
    for k in range(4):
        assert abs(float(wide[k, 0])) > abs(float(base[k, 0]))
        assert abs(float(wide[k, 1])) > abs(float(base[k, 1]))
        assert float(wide[k, 2]) == pytest.approx(float(base[k, 2]))


# --------------------------------------------------------------- the solver

def test_solve_stance_hits_every_target(toy, toy_feet):
    heights = [0.18, 0.24, 0.30]
    tgt = go2_feet.stance_targets(toy, toy_feet, heights)
    q, info = go2_feet.solve_stance(toy, toy_feet, tgt, iters=80)
    assert q.shape == (3, toy.dof)
    assert info["solve_rate"] == 1.0
    assert float(info["max_error"].max()) < 1e-6


def test_solved_stance_verified_by_mujoco(toy, toy_feet):
    """The IK is checked outside kinfast: MuJoCo is asked where the feet ended
    up for the joint angles the solver returned."""
    heights = [0.18, 0.26, 0.32]
    tgt = go2_feet.stance_targets(toy, toy_feet, heights)
    q, info = go2_feet.solve_stance(toy, toy_feet, tgt, iters=120)
    geoms = [f"{lab}_foot" for lab in ("FL", "FR", "RL", "RR")]
    for b in range(len(heights)):
        ref = _mujoco_foot_points(TOY4, q[b], toy.joint_names, geoms)
        assert torch.allclose(ref, tgt[b], atol=1e-5)


def test_solution_respects_the_joint_limits(toy, toy_feet):
    tgt = go2_feet.stance_targets(toy, toy_feet, [0.15, 0.35])
    q, _ = go2_feet.solve_stance(toy, toy_feet, tgt, iters=80)
    lo = toy.chain.lower.to(torch.float64)
    hi = toy.chain.upper.to(torch.float64)
    assert torch.all(q >= lo - 1e-12) and torch.all(q <= hi + 1e-12)


def test_unreachable_target_is_reported_not_faked(toy, toy_feet):
    """A foot asked for a metre below the trunk cannot get there. The solver
    must say so in the solve rate rather than return a bogus success."""
    tgt = go2_feet.stance_targets(toy, toy_feet, [0.25])
    tgt[0, 0, 2] = -1.0
    q, info = go2_feet.solve_stance(toy, toy_feet, tgt, iters=80)
    assert bool(info["solved"][0, 0]) is False
    assert info["solve_rate"] == pytest.approx(0.75)
    assert float(info["foot_error"][0, 1:].max()) < 1e-6


def test_working_dtype_follows_the_seed(toy, toy_feet):
    tgt = go2_feet.stance_targets(toy, toy_feet, [0.22, 0.28])
    q0 = go2_feet.rest_posture(toy, torch.float32)
    q, info = go2_feet.solve_stance(toy, toy_feet, tgt, q0=q0, iters=80,
                                    tol=1e-4)
    assert q.dtype == torch.float32
    assert info["foot_error"].dtype == torch.float32
    assert info["solve_rate"] == 1.0


def test_float32_and_float64_agree_to_float32_precision(toy, toy_feet):
    tgt = go2_feet.stance_targets(toy, toy_feet, [0.24])
    q32, _ = go2_feet.solve_stance(toy, toy_feet, tgt,
                                   q0=go2_feet.rest_posture(toy, torch.float32),
                                   iters=80)
    q64, _ = go2_feet.solve_stance(toy, toy_feet, tgt, iters=80)
    assert torch.allclose(q32.double(), q64, atol=1e-4)


def test_solve_is_differentiable_in_the_targets(toy, toy_feet):
    """Gradients flow from the solved joint angles back to the foot targets,
    and match central differences through the whole solve."""
    tgt = go2_feet.stance_targets(toy, toy_feet, [0.25]).requires_grad_(True)
    q, _ = go2_feet.solve_stance(toy, toy_feet, tgt, iters=60)
    loss = q.sum()
    (grad,) = torch.autograd.grad(loss, tgt)
    assert torch.isfinite(grad).all()
    eps = 1e-6
    for k, axis in ((0, 2), (2, 1)):
        d = torch.zeros_like(tgt.detach())
        d[0, k, axis] = eps
        plus, _ = go2_feet.solve_stance(toy, toy_feet, tgt.detach() + d, iters=60)
        minus, _ = go2_feet.solve_stance(toy, toy_feet, tgt.detach() - d, iters=60)
        fd = float((plus.sum() - minus.sum()) / (2 * eps))
        assert fd == pytest.approx(float(grad[0, k, axis]), abs=1e-4)


def test_restarts_are_deterministic_and_never_worse(toy, toy_feet):
    tgt = go2_feet.stance_targets(toy, toy_feet, [0.20, 0.30])
    one, i1 = go2_feet.solve_stance(toy, toy_feet, tgt, iters=60, restarts=1)
    g = torch.Generator().manual_seed(7)
    many, i3 = go2_feet.solve_stance(toy, toy_feet, tgt, iters=60, restarts=4,
                                     generator=g)
    g2 = torch.Generator().manual_seed(7)
    again, _ = go2_feet.solve_stance(toy, toy_feet, tgt, iters=60, restarts=4,
                                     generator=g2)
    assert torch.equal(many, again)
    assert torch.all(i3["max_error"] <= i1["max_error"] + 1e-12)


# ------------------------------------------------------------ report and plot

def test_report_lines_show_every_height(toy, toy_feet):
    heights = [0.20, 0.30]
    tgt = go2_feet.stance_targets(toy, toy_feet, heights)
    _, info = go2_feet.solve_stance(toy, toy_feet, tgt, iters=80)
    lines = go2_feet.format_report(heights, info)
    text = "\n".join(lines)
    assert "0.200 m" in text and "0.300 m" in text
    assert "4/4" in text
    assert "solve rate 100.0%" in text


def test_plot_writes_a_png(toy, toy_feet, tmp_path):
    pytest.importorskip("matplotlib")
    heights = [0.20, 0.30]
    tgt = go2_feet.stance_targets(toy, toy_feet, heights)
    q, _ = go2_feet.solve_stance(toy, toy_feet, tgt, iters=60)
    out = tmp_path / "stance.png"
    go2_feet.plot_stance(toy, toy_feet, q, tgt, heights, str(out))
    assert out.exists() and out.stat().st_size > 5000


# ---------------------------------------------------------------- the script

def test_missing_model_exits_with_advice(capsys):
    code = go2_feet.main(["--mjcf", "nowhere/go2.xml"])
    assert code == 2
    err = capsys.readouterr().err
    assert "nowhere/go2.xml" in err
    assert "menagerie.py --fetch" in err


def test_runs_end_to_end_on_the_fixture(toy_path, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "toy.png"
    code = go2_feet.main(["--mjcf", toy_path, "--heights", "0.20", "0.28",
                          "--iters", "80", "--out", str(out)])
    assert code == 0
    assert out.exists()


# ------------------------------------------------------------- the real robot

def _without_meshes(path):
    """The Go2 XML with its mesh assets removed, so MuJoCo can load it here.

    Only the .obj files are missing from the checkout (they are a separate
    download); every body, joint, inertial and collision geom stays exactly as
    the model wrote it, so the kinematics MuJoCo computes are the real ones.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "mesh" or (child.tag == "geom"
                                       and child.get("mesh") is not None):
                parent.remove(child)
    return ET.tostring(root, encoding="unicode")



@pytest.mark.skipif(_go2_path() is None, reason="go2.xml not downloaded")
def test_go2_feet_and_contact_offsets():
    import kinfast
    path = _go2_path()
    robot = kinfast.load(path, dtype=torch.float64)
    feet = go2_feet.find_feet(robot, mjcf_path=path)
    assert feet.links == ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
    assert feet.base_link == "base"
    for k in range(4):
        assert feet.offsets[k].tolist() == pytest.approx([-0.002, 0.0, -0.213])
        assert float(feet.radii[k]) == pytest.approx(0.022)


@pytest.mark.skipif(_go2_path() is None, reason="go2.xml not downloaded")
def test_go2_stance_solves_and_matches_mujoco():
    import kinfast
    path = _go2_path()
    robot = kinfast.load(path, dtype=torch.float64)
    feet = go2_feet.find_feet(robot, mjcf_path=path)
    heights = [0.18, 0.26, 0.34]
    tgt = go2_feet.stance_targets(robot, feet, heights)
    q, info = go2_feet.solve_stance(robot, feet, tgt, iters=120)
    assert info["solve_rate"] == 1.0

    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(_without_meshes(path))
    import numpy as np
    d = mujoco.MjData(m)
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
             for g in ("FL", "FR", "RL", "RR")]
    for b in range(len(heights)):
        d.qpos[:] = m.qpos0
        for k, name in enumerate(robot.joint_names):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[j]] = float(q[b, k])
        mujoco.mj_forward(m, d)
        R = d.xmat[trunk].reshape(3, 3)
        t = d.xpos[trunk]
        got = torch.tensor(np.stack([R.T @ (d.geom_xpos[g] - t) for g in geoms]),
                           dtype=torch.float64)
        assert torch.allclose(got, tgt[b], atol=1e-5)
