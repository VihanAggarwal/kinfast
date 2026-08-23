# tests/test_analysis.py
"""Analysis tools against textbook oracles: for a planar 2R arm with unit links,
Yoshikawa manipulability is exactly l1*l2*|sin(q2)| = |sin(q2)|."""
import math
import torch
import kinfast
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast import analysis as A
from tests.test_spatial import SIX_DOF

# planar 2R with a real distal link (ee at 1m from the elbow)
PLANAR_2R = """
<robot name="p2r">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="ee"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" velocity="2" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" velocity="2" effort="50"/></joint>
  <joint name="jf" type="fixed"><parent link="l2"/><child link="ee"/>
    <origin xyz="1 0 0"/></joint>
</robot>
"""


def _p2r():
    return compile_robot(parse_urdf_string(PLANAR_2R), dtype=torch.float64)


def test_manipulability_matches_textbook():
    chain = _p2r()
    li = chain.link_index["ee"]
    for q2, expect in [(0.0, 0.0), (math.pi / 2, 1.0), (math.pi / 6, 0.5)]:
        q = torch.tensor([[0.4, q2]], dtype=torch.float64)
        # planar arm: select the in-plane (x, y) task rows
        w = A.manipulability(chain, q, li, rows=(0, 1))
        # tolerance: sqrt(det) amplifies float64 eps (~1e-16) to ~1e-8
        assert abs(w.item() - expect) < 1e-7, f"q2={q2}"


def test_condition_number_blows_up_at_singularity():
    chain = _p2r()
    li = chain.link_index["ee"]
    q_sing = torch.tensor([[0.4, 1e-6]], dtype=torch.float64)   # arm straight
    q_good = torch.tensor([[0.4, math.pi / 2]], dtype=torch.float64)
    assert A.condition_number(chain, q_sing, li).item() > 1e4
    assert A.condition_number(chain, q_good, li).item() < 10.0


# planar 3R with unit links; the third joint makes the 3-row translational
# Jacobian square (3x3) with a zero z row, which is the condition_number trap
PLANAR_3R = """
<robot name="p3r">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="ee"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" velocity="2" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" velocity="2" effort="50"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" velocity="2" effort="50"/></joint>
  <joint name="jf" type="fixed"><parent link="l3"/><child link="ee"/>
    <origin xyz="1 0 0"/></joint>
</robot>
"""


def _planar_3r_cond_oracle(q1, q2, q3):
    """Hand-assembled in-plane Jacobian of a unit-link planar 3R, via numpy."""
    import numpy as np
    s1, c1 = math.sin(q1), math.cos(q1)
    s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
    s123, c123 = math.sin(q1 + q2 + q3), math.cos(q1 + q2 + q3)
    J = np.array([[-s1 - s12 - s123, -s12 - s123, -s123],
                  [c1 + c12 + c123, c12 + c123, c123]], dtype=np.float64)
    s = np.linalg.svd(J, compute_uv=False)
    return s[0] / s[-1]


def test_condition_number_rows_selects_in_plane_rows_for_planar_3r():
    chain = compile_robot(parse_urdf_string(PLANAR_3R), dtype=torch.float64)
    li = chain.link_index["ee"]
    q = torch.tensor([[0.3, 1.2, -0.8]], dtype=torch.float64)
    # the in-plane 2x3 Jacobian is well conditioned here
    k = A.condition_number(chain, q, li, rows=(0, 1))
    assert k.shape == (1,)
    expect = _planar_3r_cond_oracle(0.3, 1.2, -0.8)
    assert expect < 10.0
    assert abs(k.item() - expect) < 1e-9 * max(1.0, expect)
    # without row selection the zero z row makes sigma_min 0 and the number
    # says nothing about the arm; this is the trap the docstring describes
    assert A.condition_number(chain, q, li).item() > 1e6
    # rows=(0, 1) must also flag a true in-plane singularity (arm straight)
    q_sing = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
    assert A.condition_number(chain, q_sing, li, rows=(0, 1)).item() > 1e6


def test_condition_number_rows_matches_default_on_planar_2r():
    # for a 2-dof arm the zero z row is harmless (only 2 singular values), so
    # selecting rows (0, 1) must reproduce the translational default exactly
    chain = _p2r()
    li = chain.link_index["ee"]
    q = torch.tensor([[0.4, math.pi / 2], [-1.0, 0.7]], dtype=torch.float64)
    k_default = A.condition_number(chain, q, li)
    k_rows = A.condition_number(chain, q, li, rows=(0, 1))
    assert torch.allclose(k_default, k_rows, rtol=1e-12, atol=0.0)


def test_joint_limit_margin():
    chain = _p2r()
    mid = torch.zeros(1, 2, dtype=torch.float64)                 # mid-range
    on_limit = torch.tensor([[3.1, 0.0]], dtype=torch.float64)   # j1 on its limit
    assert abs(A.joint_limit_margin(chain, mid).item() - 1.0) < 1e-9
    assert A.joint_limit_margin(chain, on_limit).item() < 1e-9


def test_workspace_reach_bounds():
    chain = compile_robot(parse_urdf_string(SIX_DOF), dtype=torch.float64)
    li = chain.link_index["ee"]
    ws = A.workspace(chain, li, n=20000)
    # total link length is 1.1 m; sampling must respect it and get close to it
    assert ws["max_reach"].item() <= 1.1 + 1e-6
    assert ws["max_reach"].item() > 1.0
    assert ws["points"].shape == (20000, 3)


def test_transform_points_frames():
    robot = kinfast.load_string(PLANAR_2R.replace('name="p2r"', 'name="p2r2"'))
    q = torch.tensor([[math.pi / 2, 0.0]])
    p_l2 = torch.tensor([[1.0, 0.0, 0.0]])                       # the ee point, in l2
    p_world = robot.transform_points(p_l2, q, from_link="l2")
    # arm straight up: l2 origin at (0,1,0), +x of l2 points along +y
    assert torch.allclose(p_world[0, 0], torch.tensor([0.0, 2.0, 0.0]), atol=1e-5)
    # round-trip world -> l2
    back = robot.transform_points(p_world, q, from_link="ee", to_link="ee")
    assert torch.allclose(back, p_world_to_frame_roundtrip(robot, p_world, q), atol=1e-5)


def p_world_to_frame_roundtrip(robot, p_world, q):
    """Helper: identity transform (from==to) must return the input unchanged."""
    return robot.transform_points(p_world[0], q, from_link="ee", to_link="ee")


def test_joint_limit_margin_with_infinite_limits():
    """A joint with no limit is never near one: its margin is 1, not NaN, and
    it must not hide another joint that really is on a limit."""
    chain = _p2r()
    chain.lower = torch.tensor([-math.inf, -3.1], dtype=torch.float64)
    chain.upper = torch.tensor([math.inf, 3.1], dtype=torch.float64)
    q = torch.tensor([[0.0, 0.0], [5.0, 3.1], [-40.0, 1.55]], dtype=torch.float64)
    m = A.joint_limit_margin(chain, q)
    assert torch.isfinite(m).all()
    assert abs(m[0].item() - 1.0) < 1e-12
    assert m[1].item() < 1e-12              # j2 sits on its limit
    assert abs(m[2].item() - 0.5) < 1e-12   # j2 halfway out; j1 unlimited
    # half-infinite: only the finite side can be violated
    chain.lower = torch.tensor([-math.inf, -3.1], dtype=torch.float64)
    chain.upper = torch.tensor([1.0, 3.1], dtype=torch.float64)
    q = torch.tensor([[-100.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    m = A.joint_limit_margin(chain, q)
    assert abs(m[0].item() - 1.0) < 1e-12
    assert m[1].item() == 0.0


def test_workspace_with_infinite_revolute_limits():
    """Unlimited revolute joints are sampled over a full turn. Oracle: a planar
    2R with unit links and free joints reaches radius 2 and fills the disk, so
    points must be finite, max_reach close to 2 and the centroid near 0. An
    inf-poisoned sampler gives NaN for all of these."""
    chain = _p2r()
    chain.lower = torch.tensor([-math.inf, -math.inf], dtype=torch.float64)
    chain.upper = torch.tensor([math.inf, math.inf], dtype=torch.float64)
    li = chain.link_index["ee"]
    ws = A.workspace(chain, li, n=20000, seed=3)
    assert torch.isfinite(ws["points"]).all()
    assert ws["max_reach"].item() <= 2.0 + 1e-9
    assert ws["max_reach"].item() > 1.99
    assert torch.isfinite(ws["centroid"]).all()
    assert ws["centroid"].abs().max().item() < 0.05
    # the sampler respects a finite side when only one bound is infinite
    chain.lower = torch.tensor([-math.inf, 0.0], dtype=torch.float64)
    chain.upper = torch.tensor([math.inf, 0.0], dtype=torch.float64)
    lo, hi = A.sampling_bounds(chain)
    assert lo.tolist() == [-math.pi, 0.0] and hi.tolist() == [math.pi, 0.0]
    chain.lower = torch.tensor([1.0, -3.1], dtype=torch.float64)
    chain.upper = torch.tensor([math.inf, 3.1], dtype=torch.float64)
    lo, hi = A.sampling_bounds(chain)
    assert lo[0].item() == 1.0 and abs(hi[0].item() - (1.0 + 2 * math.pi)) < 1e-12


def test_workspace_infinite_prismatic_limit_raises():
    """A prismatic joint with an infinite limit has no finite range to draw
    from; that must be a clear error, not a NaN point cloud."""
    import pytest
    urdf = """
    <robot name="slide">
      <link name="base"/><link name="l1"/><link name="ee"/>
      <joint name="s" type="prismatic"><parent link="base"/><child link="l1"/>
        <axis xyz="1 0 0"/><limit lower="0" upper="0.5" velocity="1" effort="1"/></joint>
      <joint name="j" type="revolute"><parent link="l1"/><child link="ee"/>
        <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
    </robot>"""
    chain = compile_robot(parse_urdf_string(urdf), dtype=torch.float64)
    chain.upper = torch.tensor([math.inf, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="prismatic"):
        A.workspace(chain, chain.link_index["ee"], n=8)
    with pytest.raises(ValueError, match="finite"):
        kinfast.Robot(chain).random_configs(4)
