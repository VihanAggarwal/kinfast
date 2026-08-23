# tests/test_repair.py
from kinfast.ir import Robot, Link, Joint
from kinfast.urdf.repair import repair

def _robot_with(joint):
    return Robot("r", {"a": Link("a"), "b": Link("b")}, [joint])

def test_swaps_inverted_limits():
    j = Joint("j", "revolute", "a", "b", limit=(2.0, -2.0))
    r, findings = repair(_robot_with(j))
    assert r.joints[0].limit == (-2.0, 2.0)
    assert any(f.code == "inverted_limits" for f in findings)

def test_normalizes_axis():
    j = Joint("j", "revolute", "a", "b", axis=(0.0, 0.0, 5.0))
    r, findings = repair(_robot_with(j))
    assert abs(sum(c * c for c in r.joints[0].axis) - 1.0) < 1e-9

def test_defaults_missing_revolute_limits():
    j = Joint("j", "revolute", "a", "b", limit=(0.0, 0.0))
    r, findings = repair(_robot_with(j))
    lo, hi = r.joints[0].limit
    assert lo < hi
    assert any(f.code == "missing_limits" for f in findings)

def test_continuous_gets_wide_limits():
    j = Joint("j", "continuous", "a", "b", limit=(0.0, 0.0))
    r, findings = repair(_robot_with(j))
    lo, hi = r.joints[0].limit
    assert lo <= -3.14 and hi >= 3.14


def test_inertia_triangle_inequality_flagged():
    from kinfast.ir import Inertial
    link = Link("bad")
    link.inertial = Inertial(1.0, (0, 0, 0), (0.05, 0.03, 0.01, 0, 0, 0))  # 0.03+0.01 < 0.05
    r, findings = repair(Robot("r", {"bad": link}, []))
    assert any(f.code == "inertia_triangle" for f in findings)
    good = Link("good")
    good.inertial = Inertial(1.0, (0, 0, 0), (0.04, 0.03, 0.02, 0, 0, 0))
    r, findings = repair(Robot("r2", {"good": good}, []))
    assert not any(f.code == "inertia_triangle" for f in findings)


def test_rotate_inertia_matches_numpy():
    import math
    import numpy as np
    from kinfast.urdf.parse import rotate_inertia, _rpy_to_mat
    i6 = (0.04, 0.03, 0.02, 0.005, -0.002, 0.001)
    rpy = (0.3, -0.5, 1.1)
    R = np.array(_rpy_to_mat(*rpy))
    ixx, iyy, izz, ixy, ixz, iyz = i6
    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    expect = R @ I @ R.T
    got = rotate_inertia(i6, R.tolist())
    got_m = np.array([[got[0], got[3], got[4]], [got[3], got[1], got[5]], [got[4], got[5], got[2]]])
    assert np.abs(got_m - expect).max() < 1e-12


def test_nonfinite_limits_replaced_by_defaults():
    """<limit lower="-inf" upper="inf"/> is legal-looking URDF but would make
    every lo + (hi - lo) * u sample NaN. Repair treats it as a missing limit."""
    import math
    for jtype, lo_min, hi_min in [("revolute", -math.pi, math.pi),
                                  ("continuous", -2 * math.pi, 2 * math.pi),
                                  ("prismatic", -1.0, 1.0)]:
        j = Joint("j", jtype, "a", "b", limit=(-math.inf, math.inf))
        r, findings = repair(_robot_with(j))
        lo, hi = r.joints[0].limit
        assert math.isfinite(lo) and math.isfinite(hi), jtype
        assert (lo, hi) == (lo_min, hi_min), jtype
        assert any(f.code == "nonfinite_limits" for f in findings), jtype
    # NaN is just as poisonous as inf
    j = Joint("j", "revolute", "a", "b", limit=(math.nan, 1.0))
    r, findings = repair(_robot_with(j))
    assert all(math.isfinite(v) for v in r.joints[0].limit)
    assert any(f.code == "nonfinite_limits" for f in findings)


def test_half_infinite_limit_keeps_finite_side():
    import math
    j = Joint("j", "revolute", "a", "b", limit=(-math.inf, 1.0))
    r, _ = repair(_robot_with(j))
    lo, hi = r.joints[0].limit
    assert hi == 1.0 and abs((hi - lo) - 2 * math.pi) < 1e-12
    j = Joint("j", "prismatic", "a", "b", limit=(0.2, math.inf))
    r, _ = repair(_robot_with(j))
    lo, hi = r.joints[0].limit
    assert lo == 0.2 and abs(hi - 2.2) < 1e-12


def test_loaded_robot_with_inf_limits_has_finite_everything():
    """End to end: the parser accepts lower="-inf"; nothing downstream may
    see a NaN because of it."""
    import math
    import torch
    import kinfast
    from kinfast import analysis as A
    urdf = """
    <robot name="infl">
      <link name="base"/><link name="l1"/><link name="ee"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
        <axis xyz="0 0 1"/><limit lower="-inf" upper="inf" velocity="1" effort="1"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="ee"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
    </robot>"""
    r = kinfast.load_string(urdf)
    assert torch.isfinite(r.lower).all() and torch.isfinite(r.upper).all()
    assert abs(r.lower[0].item() + math.pi) < 1e-6 and abs(r.upper[0].item() - math.pi) < 1e-6
    torch.manual_seed(0)
    q = r.random_configs(16)
    assert torch.isfinite(q).all()
    assert (q >= r.lower - 1e-6).all() and (q <= r.upper + 1e-6).all()
    ws = A.workspace(r.chain, r.link_id("ee"), n=64)
    assert torch.isfinite(ws["points"]).all()
    assert torch.isfinite(ws["max_reach"]) and torch.isfinite(ws["centroid"]).all()
    m = A.joint_limit_margin(r.chain, torch.zeros(1, 2))
    assert abs(m.item() - 1.0) < 1e-6
