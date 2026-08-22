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
