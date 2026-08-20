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
