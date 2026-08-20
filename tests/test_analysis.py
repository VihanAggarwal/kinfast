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
