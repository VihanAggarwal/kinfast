# tests/test_config.py
"""Configuration-space utilities checked against hand-computed values.

The oracles here are arithmetic anyone can redo on paper (2*pi - 6.2 = 0.0832),
forward kinematics of a one-joint arm, central finite differences in float64,
and the balance property of a Sobol net. Nothing checks kinfast against
kinfast.
"""
import math
import torch
import pytest

import kinfast
from kinfast import config as C
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.urdf.parse import parse_urdf_string
from kinfast.urdf.repair import repair

# a continuous joint (URDF gives it no limit; repair opens it to +-2pi),
# a limited revolute, and a prismatic slider
MIXED = """
<robot name="cfg">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/>
  <joint name="jc" type="continuous"><parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/></joint>
  <joint name="jr" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" velocity="2" effort="50"/></joint>
  <joint name="jp" type="prismatic"><parent link="l2"/><child link="l3"/>
    <origin xyz="1 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.5" velocity="1" effort="50"/></joint>
</robot>
"""

# one revolving joint, end effector 1 m out along +x: fk is a pure rotation
SPINNER = """
<robot name="spin">
  <link name="base"/><link name="l1"/><link name="ee"/>
  <joint name="jc" type="continuous"><parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/></joint>
  <joint name="jf" type="fixed"><parent link="l1"/><child link="ee"/>
    <origin xyz="1 0 0"/></joint>
</robot>
"""

# a revolute joint whose stated range is wider than a full turn: physically
# it can reach every angle, so it should wrap like a continuous one
WIDE = """
<robot name="wide">
  <link name="base"/><link name="l1"/>
  <joint name="jw" type="revolute"><parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-4" upper="4" velocity="2" effort="50"/></joint>
</robot>
"""

TAU = 2.0 * math.pi
SHORT = TAU - 6.2          # 0.08318530717958..., the hand-computed answer


def mixed():
    return kinfast.load_string(MIXED).chain


def mixed64():
    """The same chain compiled in float64, for the checks where the float32
    rounding of a limit (about 1e-7 rad) would swamp the thing being tested."""
    ir, _ = repair(parse_urdf_string(MIXED))
    return compile_robot(ir, dtype=torch.float64)


def f64(*a):
    """float64 tensor: f64([1, 2, 3]) or f64(1.0, 2.0, 3.0), both (3,)."""
    return torch.tensor(a[0] if len(a) == 1 else a, dtype=torch.float64)


# ---------------------------------------------------------------- masks ----

def test_continuous_mask_flags_only_the_revolving_joint():
    chain = mixed()
    assert chain.joint_names == ["jc", "jr", "jp"]
    m = C.continuous_mask(chain)
    assert m.tolist() == [True, False, False]


def test_wide_revolute_wraps_and_an_explicit_mask_overrides():
    chain = kinfast.load_string(WIDE).chain
    assert C.continuous_mask(chain).tolist() == [True]
    a, b = f64([3.1]), f64([-3.1])
    assert C.distance(chain, a, b).item() == pytest.approx(SHORT, abs=1e-12)
    # caller knows better: force the joint to be treated as non-wrapping
    forced = C.distance(chain, a, b, continuous=[False])
    assert forced.item() == pytest.approx(6.2, abs=1e-12)


def test_explicit_mask_shape_is_checked():
    chain = mixed()
    q = torch.zeros(1, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="continuous"):
        C.distance(chain, q, q, continuous=[True, False])


# ------------------------------------------------------------- distance ----

def test_wrap_around_distance_is_the_short_way():
    """3.1 and -3.1 rad on a revolving joint are 0.083 apart, not 6.2."""
    chain = mixed()
    a = f64([3.1, 0.0, 0.0])
    b = f64([-3.1, 0.0, 0.0])
    d = C.distance(chain, a, b)
    assert d.item() == pytest.approx(SHORT, abs=1e-12)
    assert d.item() < 0.09
    assert abs(d.item() - 6.2) > 6.0


def test_limited_revolute_takes_the_long_way():
    """The same numbers on a joint that cannot pass +-pi stay 6.2 apart."""
    chain = mixed()
    a = f64([0.0, 3.1, 0.0])
    b = f64([0.0, -3.1, 0.0])
    assert C.distance(chain, a, b).item() == pytest.approx(6.2, abs=1e-12)


def test_prismatic_never_wraps():
    chain = mixed()
    a = f64([0.0, 0.0, 3.1])
    b = f64([0.0, 0.0, -3.1])
    assert C.distance(chain, a, b).item() == pytest.approx(6.2, abs=1e-12)


def test_difference_is_signed_and_lands_on_an_equivalent_config():
    chain = mixed()
    a = f64([3.1, 0.5, 0.1])
    b = f64([-3.1, -0.5, 0.4])
    d = C.difference(chain, a, b)
    assert d[0].item() == pytest.approx(SHORT, abs=1e-12)     # short way, +ve
    assert d[1].item() == pytest.approx(-1.0, abs=1e-12)
    assert d[2].item() == pytest.approx(0.3, abs=1e-12)
    reached = a + d
    turns = (reached - b) / TAU
    assert turns[0].item() == pytest.approx(1.0, abs=1e-12)   # one whole turn
    assert torch.allclose(reached[1:], b[1:], atol=1e-12)


def test_distance_norms_and_weights_match_hand_values():
    chain = mixed()
    a = f64([0.0, 0.0, 0.0])
    b = f64([0.0, 3.0, 0.4])
    assert C.distance(chain, a, b).item() == pytest.approx(
        math.sqrt(9.0 + 0.16), abs=1e-12)
    assert C.distance(chain, a, b, p=1).item() == pytest.approx(3.4, abs=1e-12)
    assert C.distance(chain, a, b, p=float("inf")).item() == pytest.approx(
        3.0, abs=1e-12)
    # a metre of slide counts for as much as 10 radians of joint travel
    w = C.distance(chain, a, b, weights=[1.0, 1.0, 10.0])
    assert w.item() == pytest.approx(math.sqrt(9.0 + 16.0), abs=1e-12)


def test_distance_is_symmetric_and_zero_on_itself():
    chain = mixed()
    g = torch.Generator().manual_seed(7)
    a = torch.rand(5, 3, generator=g, dtype=torch.float64) * 2 - 1
    b = torch.rand(5, 3, generator=g, dtype=torch.float64) * 2 - 1
    assert torch.allclose(C.distance(chain, a, b), C.distance(chain, b, a))
    # exactly zero, not a smoothing epsilon: planners compare against 0
    assert torch.equal(C.distance(chain, a, a),
                       torch.zeros(5, dtype=torch.float64))


def test_distance_gradients_match_central_differences():
    chain = mixed()
    a = f64([3.05, 0.7, 0.2]).requires_grad_(True)
    b = f64([-3.05, -0.2, 0.45])
    C.distance(chain, a, b).backward()
    got = a.grad.clone()
    eps = 1e-6
    for i in range(3):
        pa, pb = a.detach().clone(), a.detach().clone()
        pa[i] += eps
        pb[i] -= eps
        fd = (C.distance(chain, pa, b) - C.distance(chain, pb, b)) / (2 * eps)
        assert got[i].item() == pytest.approx(fd.item(), abs=1e-7), f"dof {i}"


def test_distance_gradient_at_zero_is_finite():
    chain = mixed()
    a = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    d = C.distance(chain, a, torch.zeros(3, dtype=torch.float64))
    assert d.item() == 0.0
    d.backward()
    assert torch.isfinite(a.grad).all()
    assert a.grad.tolist() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------- interpolate ----

def test_interpolate_takes_the_short_way():
    """Halfway from 3.1 to -3.1 is pi (through the wrap), not 0."""
    chain = mixed()
    a = f64([3.1, 0.0, 0.0])
    b = f64([-3.1, 0.0, 0.0])
    mid = C.interpolate(chain, a, b, 0.5)
    assert mid[0].item() == pytest.approx(math.pi, abs=1e-12)
    assert abs(mid[0].item()) > 3.0          # emphatically not the midpoint 0


def test_interpolated_path_is_short_and_monotone():
    chain = mixed()
    a = f64([3.1, 0.0, 0.0])
    b = f64([-3.1, 0.0, 0.0])
    s = torch.linspace(0, 1, 51, dtype=torch.float64)
    path = C.interpolate(chain, a, b, s)
    assert path.shape == (51, 3)
    steps = path[1:, 0] - path[:-1, 0]
    assert bool((steps > 0).all())                       # never turns back
    total = steps.abs().sum().item()
    assert total == pytest.approx(SHORT, abs=1e-12)      # 0.083 rad of travel


def test_interpolate_hits_the_endpoints():
    chain = mixed()
    a = f64([3.1, 0.6, 0.1])
    b = f64([-3.1, -0.6, 0.4])
    assert torch.allclose(C.interpolate(chain, a, b, 0.0), a, atol=1e-15)
    end = C.interpolate(chain, a, b, 1.0)
    # exact on the joints that cannot wrap, one turn away on the one that can
    assert torch.allclose(end[1:], b[1:], atol=1e-12)
    assert (end[0] - b[0]).item() == pytest.approx(TAU, abs=1e-12)
    # folding it back onto [-pi, pi) recovers the number the caller passed
    assert C.wrap_angle(end[0]).item() == pytest.approx(b[0].item(), abs=1e-12)


def test_interpolate_matches_forward_kinematics():
    """Independent oracle: for a one-joint arm with the tool 1 m out, the
    halfway configuration must put the tool at (-1, 0, 0)."""
    chain = kinfast.load_string(SPINNER).chain
    ee = chain.link_index["ee"]
    a = f64([[3.1]])
    b = f64([[-3.1]])
    mid = C.interpolate(chain, a, b, 0.5)
    p = forward_kinematics(chain, mid)[0, ee, :3, 3]
    assert torch.allclose(p, f64(-1.0, 0.0, 0.0), atol=1e-9)
    # the naive straight-line midpoint would have parked it at (1, 0, 0)
    naive = forward_kinematics(chain, 0.5 * (a + b))[0, ee, :3, 3]
    assert torch.allclose(naive, f64(1.0, 0.0, 0.0), atol=1e-9)


def test_interpolate_broadcasts_over_a_batch():
    chain = mixed()
    a = torch.zeros(4, 3, dtype=torch.float64)
    b = torch.full((4, 3), 0.2, dtype=torch.float64)
    s = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)
    out = C.interpolate(chain, a, b, s)
    assert out.shape == (4, 3)
    assert torch.allclose(out[:, 0], torch.tensor([0.0, 0.05, 0.1, 0.2],
                                                  dtype=torch.float64))


def test_interpolate_is_differentiable_end_to_end():
    chain = mixed()
    a = f64([3.1, 0.0, 0.0]).requires_grad_(True)
    b = f64([-3.1, 0.0, 0.0])
    C.interpolate(chain, a, b, 0.25).sum().backward()
    # d/da of (a + 0.25 * wrap(b - a)) is 0.75 on every joint
    assert torch.allclose(a.grad, torch.full((3,), 0.75, dtype=torch.float64))


# --------------------------------------------------------------- limits ----

def test_clamp_wraps_a_revolving_joint_and_clips_the_others():
    chain = mixed64()
    q = f64([10.0, 5.0, -0.7])
    out = C.clamp_to_limits(chain, q)
    # jc: 10 rad is 10 - 2pi = 3.7168 rad, the same physical angle, in range
    assert out[0].item() == pytest.approx(10.0 - TAU, abs=1e-12)
    assert out[1].item() == pytest.approx(3.1, abs=1e-12)    # hard stop
    assert out[2].item() == pytest.approx(0.0, abs=1e-12)    # slider bottom
    assert bool(C.is_within_limits(chain, out, wrap=False))


def test_clamp_leaves_legal_configurations_alone():
    chain = mixed()
    q = f64([[3.0, -2.0, 0.25], [-6.0, 3.09, 0.5]])
    assert torch.allclose(C.clamp_to_limits(chain, q), q, atol=1e-15)


def test_clamp_output_is_always_legal():
    """Whatever you throw at it, the result satisfies the literal bounds."""
    chain = mixed64()
    g = torch.Generator().manual_seed(23)
    q = torch.rand(500, 3, generator=g, dtype=torch.float64) * 60 - 30
    out = C.clamp_to_limits(chain, q)
    assert bool(C.is_within_limits(chain, out, wrap=False).all())


def test_clamp_of_a_revolving_joint_does_not_move_the_robot():
    """Independent oracle: folding by whole turns is a relabelling, so the
    tool pose before and after the clamp has to be identical."""
    chain = kinfast.load_string(SPINNER).chain
    ee = chain.link_index["ee"]
    q = f64([-20.0, -7.0, -1.0, 0.0, 1.0, 7.0, 20.0]).unsqueeze(-1)
    out = C.clamp_to_limits(chain, q)
    assert not torch.allclose(out, q)                  # it really did fold
    assert bool(C.is_within_limits(chain, out, wrap=False).all())
    before = forward_kinematics(chain, q)[:, ee]
    after = forward_kinematics(chain, out)[:, ee]
    assert torch.allclose(before, after, atol=1e-9)


def test_clamp_gradient_is_one_where_nothing_moved():
    chain = mixed()
    q = f64([1.0, 5.0, 0.25]).requires_grad_(True)
    C.clamp_to_limits(chain, q).sum().backward()
    assert q.grad.tolist() == [1.0, 0.0, 1.0]


def test_is_within_limits_wrap_flag():
    chain = mixed()
    q = f64([[0.0, 0.0, 0.25], [100.0, 0.0, 0.25], [0.0, 4.0, 0.25]])
    assert C.is_within_limits(chain, q).tolist() == [True, True, False]
    assert C.is_within_limits(chain, q, wrap=False).tolist() == \
        [True, False, False]
    per = C.is_within_limits(chain, q, wrap=False, per_joint=True)
    assert per.shape == (3, 3)
    assert per[1].tolist() == [False, True, True]


def test_is_within_limits_tolerance():
    chain = mixed64()          # float32 limits round by ~1e-7, too coarse here
    q = f64([0.0, 3.1 + 1e-9, 0.25])
    assert not bool(C.is_within_limits(chain, q, wrap=False))
    assert bool(C.is_within_limits(chain, q, tol=1e-8, wrap=False))


# ------------------------------------------------------------- sampling ----

def test_sample_respects_the_limits():
    chain = mixed()
    q = C.sample(chain, 256, seed=3, dtype=torch.float64)
    assert q.shape == (256, 3)
    lo = chain.lower.to(torch.float64)
    hi = chain.upper.to(torch.float64)
    assert bool((q >= lo).all()) and bool((q <= hi).all())
    assert bool(C.is_within_limits(chain, q, wrap=False).all())
    # and it actually fills the box rather than hugging one corner
    assert q[:, 2].min().item() < 0.02 and q[:, 2].max().item() > 0.48


def test_sample_is_deterministic_for_a_fixed_seed():
    chain = mixed()
    a = C.sample(chain, 64, seed=11, dtype=torch.float64)
    b = C.sample(chain, 64, seed=11, dtype=torch.float64)
    assert torch.equal(a, b)
    c = C.sample(chain, 64, seed=12, dtype=torch.float64)
    assert not torch.equal(a, c)


def test_sample_is_low_discrepancy():
    """A Sobol block of 2^m points is balanced: split any axis into 2^m equal
    bins and each bin holds exactly one point. Independent uniforms do not do
    that (the same test on torch.rand fails), which is the whole reason to
    prefer this sampler for coverage sweeps."""
    chain = mixed()
    n = 64
    q = C.sample(chain, n, seed=5, dtype=torch.float64)
    lo = chain.lower.to(torch.float64)
    hi = chain.upper.to(torch.float64)
    u = (q - lo) / (hi - lo)
    for d in range(chain.dof):
        bins = torch.clamp((u[:, d] * n).long(), max=n - 1)
        assert sorted(bins.tolist()) == list(range(n)), f"dof {d} not balanced"
    g = torch.Generator().manual_seed(5)
    r = torch.rand(n, generator=g, dtype=torch.float64)
    assert sorted(torch.clamp((r * n).long(), max=n - 1).tolist()) != \
        list(range(n))


def test_unscrambled_sample_starts_at_the_lower_corner():
    chain = mixed()
    q = C.sample(chain, 8, scramble=False, dtype=torch.float64)
    assert torch.allclose(q[0], chain.lower.to(torch.float64), atol=1e-15)


def test_sample_dtype_and_device_default_to_the_chain():
    chain = mixed()
    q = C.sample(chain, 8)
    assert q.dtype == chain.lower.dtype == torch.float32
    assert q.device == chain.lower.device
    assert C.sample(chain, 8, dtype=torch.float64).dtype == torch.float64


def test_sample_of_an_unbounded_revolute_joint_covers_one_turn():
    chain = mixed64()
    chain.lower = chain.lower.clone()
    chain.upper = chain.upper.clone()
    chain.lower[0], chain.upper[0] = float("-inf"), float("inf")
    q = C.sample(chain, 128, seed=2)
    assert bool(torch.isfinite(q).all())
    assert q[:, 0].min().item() >= -math.pi - 1e-12
    assert q[:, 0].max().item() <= math.pi + 1e-12
    assert q[:, 0].max().item() - q[:, 0].min().item() > 0.9 * TAU


def test_triangle_inequality_holds_with_wrapping():
    """Wrapping has to leave a metric behind, or planners built on it break."""
    chain = mixed64()
    g = torch.Generator().manual_seed(19)
    a, b, c = (torch.rand(200, 3, generator=g, dtype=torch.float64) * 20 - 10
               for _ in range(3))
    direct = C.distance(chain, a, c)
    detour = C.distance(chain, a, b) + C.distance(chain, b, c)
    assert bool((direct <= detour + 1e-12).all())


def test_sample_of_an_unbounded_prismatic_joint_raises():
    chain = kinfast.load_string(MIXED).chain
    chain.upper = chain.upper.clone()
    chain.upper[2] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        C.sample(chain, 4)


# ----------------------------------------------------- dtype and shapes ----

def test_working_dtype_follows_q():
    chain = mixed()                       # compiled float32
    assert chain.lower.dtype == torch.float32
    a = torch.zeros(2, 3, dtype=torch.float64)
    b = torch.full((2, 3), 0.3, dtype=torch.float64)
    for out in (C.difference(chain, a, b), C.distance(chain, a, b),
                C.interpolate(chain, a, b, 0.5), C.clamp_to_limits(chain, a)):
        assert out.dtype == torch.float64
    a32 = a.to(torch.float32)
    assert C.distance(chain, a32, b.to(torch.float32)).dtype == torch.float32


def test_extra_leading_batch_dimensions_survive():
    chain = mixed()
    a = torch.zeros(2, 5, 3, dtype=torch.float64)
    b = torch.full((2, 5, 3), 0.1, dtype=torch.float64)
    assert C.distance(chain, a, b).shape == (2, 5)
    assert C.difference(chain, a, b).shape == (2, 5, 3)
    assert C.is_within_limits(chain, a).shape == (2, 5)
    # a single configuration broadcasts against a batch
    assert C.distance(chain, a, b[0, 0]).shape == (2, 5)


def test_bad_shapes_and_arguments_raise():
    chain = mixed()
    q = torch.zeros(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"shape"):
        C.distance(chain, torch.zeros(2, 2, dtype=torch.float64), q)
    with pytest.raises(TypeError):
        C.distance(chain, [0.0, 0.0, 0.0], q)
    with pytest.raises(ValueError, match="weights"):
        C.distance(chain, q, q, weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        C.distance(chain, q, q, weights=[1.0, -2.0, 1.0])
    with pytest.raises(ValueError, match="n >= 1"):
        C.sample(chain, 0)
