# tests/test_ik_ccd.py
"""Cyclic coordinate descent position IK, checked against outside oracles.

The per joint closed form is checked against a dense brute force scan over the
angle and against a rotation matrix built with torch.matrix_exp, neither of
which shares any code with kinfast. The solver itself is checked by feeding its
answer back through forward kinematics, by hand computed 2R geometry, and by
central differences for the gradient.
"""
import math

import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.ik_ccd import ik_ccd, alignment_coeffs, best_alignment_angle
from kinfast.urdf.parse import parse_urdf_string

from tests.test_analysis import PLANAR_2R
from tests.test_spatial import SIX_DOF

D = torch.float64

# A prismatic slide along +x with the tool 0.2 m up the z axis, so the end
# effector sits at (q, 0, 0.2) and every answer can be read off by hand.
PRISM_ARM = """
<robot name="prism">
  <link name="base"/><link name="slide"/><link name="ee"/>
  <joint name="jp" type="prismatic"><parent link="base"/><child link="slide"/>
    <origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.5" velocity="1" effort="10"/></joint>
  <joint name="jf" type="fixed"><parent link="slide"/><child link="ee"/>
    <origin xyz="0 0 0.2"/></joint>
</robot>
"""

# Revolute yaw, then a slide along the rotated x, then a revolute pitch: a
# mixed chain with tight and asymmetric limits, for the limit tests.
MIXED_ARM = """
<robot name="mixed">
  <link name="base"/><link name="a"/><link name="b"/><link name="c"/>
  <link name="ee"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-0.7" upper="1.3" velocity="2" effort="50"/></joint>
  <joint name="j2" type="prismatic"><parent link="a"/><child link="b"/>
    <origin xyz="0.2 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.05" upper="0.4" velocity="1" effort="50"/></joint>
  <joint name="j3" type="revolute"><parent link="b"/><child link="c"/>
    <origin xyz="0.3 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.1" upper="0.2" velocity="2" effort="50"/></joint>
  <joint name="jf" type="fixed"><parent link="c"/><child link="ee"/>
    <origin xyz="0.25 0 0"/></joint>
</robot>
"""

# A joint declared with infinite limits, the way a continuous joint should
# reach the compiler. CCD must not sample or clamp against an inf.
FREE_SPIN = """
<robot name="spin">
  <link name="base"/><link name="a"/><link name="ee"/>
  <joint name="j1" type="continuous"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-inf" upper="inf" velocity="2" effort="50"/></joint>
  <joint name="jf" type="fixed"><parent link="a"/><child link="ee"/>
    <origin xyz="0.4 0 0"/></joint>
</robot>
"""


def _chain(text, dtype=D):
    return compile_robot(parse_urdf_string(text), dtype=dtype)


def _ee_pos(chain, q, link):
    return forward_kinematics(chain, q)[:, link, :3, 3]


# ---------------------------------------------------------------- oracles ---

def _rodrigues_matrix_exp(axis, angle):
    """R = exp(theta * [axis]_x), built with torch.matrix_exp.

    An oracle that shares nothing with kinfast: no Rodrigues expansion, no
    quaternions, just the matrix exponential of the skew symmetric generator.
    axis (..., 3) unit, angle (...,) -> (..., 3, 3).
    """
    a = axis * angle.unsqueeze(-1)
    z = torch.zeros_like(a[..., 0])
    skew = torch.stack([
        torch.stack([z, -a[..., 2], a[..., 1]], dim=-1),
        torch.stack([a[..., 2], z, -a[..., 0]], dim=-1),
        torch.stack([-a[..., 1], a[..., 0], z], dim=-1),
    ], dim=-2)
    return torch.matrix_exp(skew)


def _random_unit(n, gen):
    v = torch.randn(n, 3, generator=gen, dtype=D)
    return v / v.norm(dim=-1, keepdim=True)


def test_alignment_coeffs_match_matrix_exp():
    """v . R(a, theta) u must equal A cos(theta) + B sin(theta) + C exactly."""
    gen = torch.Generator().manual_seed(11)
    n = 200
    a = _random_unit(n, gen)
    u = torch.randn(n, 3, generator=gen, dtype=D)
    v = torch.randn(n, 3, generator=gen, dtype=D)
    A, Bc = alignment_coeffs(a, u, v)
    C = (a * u).sum(-1) * (a * v).sum(-1)
    for theta in (-3.0, -1.0, -0.2, 0.0, 0.37, 1.5, 3.14):
        th = torch.full((n,), theta, dtype=D)
        R = _rodrigues_matrix_exp(a, th)
        lhs = (v * (R @ u.unsqueeze(-1)).squeeze(-1)).sum(-1)
        rhs = A * math.cos(theta) + Bc * math.sin(theta) + C
        assert torch.allclose(lhs, rhs, atol=1e-12), theta


def test_best_alignment_angle_beats_dense_scan():
    """The closed form angle must be at least as good as any angle on a fine
    grid, and within a grid step of the grid's own best."""
    gen = torch.Generator().manual_seed(12)
    n = 128
    a = _random_unit(n, gen)
    u = torch.randn(n, 3, generator=gen, dtype=D)
    v = torch.randn(n, 3, generator=gen, dtype=D)
    theta = best_alignment_angle(a, u, v)
    assert torch.all(theta > -math.pi - 1e-12) and torch.all(theta <= math.pi + 1e-12)

    def residual(th):
        R = _rodrigues_matrix_exp(a, th)
        return ((R @ u.unsqueeze(-1)).squeeze(-1) - v).norm(dim=-1)

    best = residual(theta)
    grid = torch.linspace(-math.pi, math.pi, 4001, dtype=D)
    scan = torch.stack([residual(torch.full((n,), float(g), dtype=D))
                        for g in grid], dim=0)
    grid_best = scan.min(dim=0).values
    # closed form is never worse, and the grid never beats it by more than the
    # curvature over half a grid step
    assert torch.all(best <= grid_best + 1e-12)
    assert torch.all(grid_best - best < 1e-5)


def test_degenerate_alignment_returns_zero():
    """u along the axis, or a zero length u or v, cannot be improved by any
    rotation, so the step must be exactly zero rather than a NaN."""
    a = torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=D)
    u = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=D)
    v = torch.tensor([[1.0, 1.0, 0.0], [3.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=D)
    theta = best_alignment_angle(a, u, v)
    assert torch.all(torch.isfinite(theta))
    assert torch.allclose(theta, torch.zeros(3, dtype=D))


# ------------------------------------------------------------- planar 2R ----

def _polar_targets(n, seed, r_lo=0.25, r_hi=1.9):
    """Reachable planar targets: unit links give a reach of 2 m."""
    gen = torch.Generator().manual_seed(seed)
    r = r_lo + (r_hi - r_lo) * torch.rand(n, generator=gen, dtype=D)
    th = -math.pi + 2 * math.pi * torch.rand(n, generator=gen, dtype=D)
    return torch.stack([r * torch.cos(th), r * torch.sin(th),
                        torch.zeros(n, dtype=D)], dim=1)


def test_planar_2r_converges_within_1mm():
    """Every reachable target on the planar 2R is hit to well under 1 mm, from
    a fixed elbow bent seed. No RNG anywhere in this test."""
    chain = _chain(PLANAR_2R)
    link = chain.link_index["ee"]
    tgt = _polar_targets(400, seed=7)
    q0 = torch.tensor([[0.0, 1.0]], dtype=D).expand(400, 2).clone()
    q, info = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=200,
                     tol=1e-12, check_every=5)
    err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
    assert float(err.max()) < 1e-3
    # the reported error is the real one
    assert torch.allclose(info["final_error"], err, atol=1e-9)


def test_planar_2r_converges_from_random_seeds_with_restarts():
    """Same targets, random seeds plus restarts: the documented way to shake
    off CCD's local minima."""
    chain = _chain(PLANAR_2R)
    link = chain.link_index["ee"]
    tgt = _polar_targets(300, seed=21)
    torch.manual_seed(0)
    q, _ = ik_ccd(chain, tgt, link_index=link, sweeps=200, tol=1e-12,
                  restarts=8, check_every=5)
    err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
    assert float(err.max()) < 1e-3


def test_planar_2r_matches_hand_computed_elbow_angle():
    """Two unit links reaching a point at radius sqrt(2) need an elbow angle of
    exactly +-pi/2 (law of cosines), whatever the shoulder does."""
    chain = _chain(PLANAR_2R)
    link = chain.link_index["ee"]
    tgt = torch.tensor([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
                        [0.0, -math.sqrt(2.0), 0.0]], dtype=D)
    q0 = torch.tensor([[0.3, 0.9]], dtype=D).expand(3, 2).clone()
    q, _ = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=300, tol=1e-14,
                  check_every=0)
    err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
    assert float(err.max()) < 1e-9
    elbow = q[:, 1].abs()
    assert torch.allclose(elbow, torch.full((3,), math.pi / 2, dtype=D),
                          atol=1e-7)


def test_unreachable_target_stops_at_full_extension():
    """A target 5 m out from a 2 m arm: the residual must be 5 - 2 = 3 m and
    the arm must point straight at it, not blow up or return NaN."""
    chain = _chain(PLANAR_2R)
    link = chain.link_index["ee"]
    tgt = torch.tensor([[5.0, 0.0, 0.0], [0.0, -5.0, 0.0]], dtype=D)
    q0 = torch.tensor([[0.2, 0.8]], dtype=D).expand(2, 2).clone()
    q, info = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=400,
                     tol=1e-14, check_every=0)
    assert torch.all(torch.isfinite(q))
    err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
    assert torch.allclose(err, torch.full((2,), 3.0, dtype=D), atol=1e-6)
    # straight arm: the elbow is folded out of the way
    assert torch.allclose(q[:, 1], torch.zeros(2, dtype=D), atol=1e-6)


# --------------------------------------------------------------- 6 dof ------

def _six_dof_targets(n, seed):
    chain = _chain(SIX_DOF)
    lo, hi = chain.lower.to(D), chain.upper.to(D)
    gen = torch.Generator().manual_seed(seed)
    qs = lo + (hi - lo) * torch.rand(n, chain.dof, generator=gen, dtype=D)
    link = chain.link_index["ee"]
    return chain, link, _ee_pos(chain, qs, link)


def test_six_dof_reaches_95_percent_within_5mm_in_200_sweeps():
    chain, link, tgt = _six_dof_targets(300, seed=100)
    torch.manual_seed(0)
    q, info = ik_ccd(chain, tgt, link_index=link, sweeps=200, tol=1e-6,
                     restarts=4, check_every=10)
    err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
    hit = float((err < 5e-3).to(D).mean())
    assert hit >= 0.95, f"only {hit:.3f} of targets within 5 mm"
    assert torch.allclose(info["final_error"], err, atol=1e-9)
    assert info["restarts"] == 4


def test_six_dof_single_seed_solve_rate_is_reasonable():
    """A single random seed is the weakest setting CCD has. It still needs to
    land most targets, otherwise something has broken in the sweep."""
    chain, link, tgt = _six_dof_targets(200, seed=100)
    rates = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        q, _ = ik_ccd(chain, tgt, link_index=link, sweeps=200, tol=1e-6,
                      check_every=10)
        err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
        rates.append(float((err < 5e-3).to(D).mean()))
    assert min(rates) > 0.85, rates


def test_restarts_do_not_hurt():
    """More seeds, solved in one batch, must not lower the solve rate."""
    chain, link, tgt = _six_dof_targets(200, seed=5)
    hits = []
    for restarts in (1, 4):
        torch.manual_seed(3)
        q, _ = ik_ccd(chain, tgt, link_index=link, sweeps=150, tol=1e-6,
                      restarts=restarts, check_every=10)
        err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
        hits.append(float((err < 5e-3).to(D).mean()))
    assert hits[1] >= hits[0]


# --------------------------------------------------------------- limits -----

def _assert_within_limits(chain, q):
    lo = chain.lower.to(dtype=q.dtype)
    hi = chain.upper.to(dtype=q.dtype)
    assert torch.all(q >= lo), (q.min(dim=0).values, lo)
    assert torch.all(q <= hi), (q.max(dim=0).values, hi)


def test_limits_never_violated_on_mixed_chain():
    """Reachable targets, wildly unreachable targets, and seeds that start
    outside the limits: the answer is always inside the box."""
    chain = _chain(MIXED_ARM)
    link = chain.link_index["ee"]
    gen = torch.Generator().manual_seed(4)
    far = 4.0 * (torch.rand(150, 3, generator=gen, dtype=D) - 0.5)
    lo, hi = chain.lower.to(D), chain.upper.to(D)
    near_q = lo + (hi - lo) * torch.rand(150, chain.dof, generator=gen, dtype=D)
    near = _ee_pos(chain, near_q, link)
    tgt = torch.cat([far, near], dim=0)

    # seed deliberately far outside every limit, in both directions
    bad = torch.zeros(tgt.shape[0], chain.dof, dtype=D)
    bad[0::2] = hi + 5.0
    bad[1::2] = lo - 5.0
    for step in (1.0, 0.5, 0.25):
        q, _ = ik_ccd(chain, tgt, q0=bad, link_index=link, sweeps=120,
                      tol=1e-12, step=step, check_every=0)
        _assert_within_limits(chain, q)
        assert torch.all(torch.isfinite(q))


def test_limits_never_violated_on_six_dof():
    chain, link, tgt = _six_dof_targets(120, seed=9)
    far = torch.cat([tgt, 6.0 * torch.ones_like(tgt)], dim=0)
    torch.manual_seed(2)
    q, _ = ik_ccd(chain, far, link_index=link, sweeps=100, tol=1e-6,
                  restarts=3, check_every=10)
    _assert_within_limits(chain, q)


def test_joint_at_its_stop_can_still_unwind():
    """A joint pinned on a limit must be able to take the equivalent rotation
    a full turn away when that one is feasible, instead of parking on the stop.
    Here the shoulder starts at its upper stop and the target is behind it."""
    chain = _chain(PLANAR_2R)
    link = chain.link_index["ee"]
    tgt = torch.tensor([[-1.4, -0.6, 0.0]], dtype=D)
    q0 = torch.tensor([[3.1, 0.05]], dtype=D)
    q, _ = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=300, tol=1e-14,
                  check_every=0)
    _assert_within_limits(chain, q)
    err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
    assert float(err.max()) < 1e-6


# ------------------------------------------------------------ prismatic -----

def test_single_prismatic_is_exact_in_one_sweep():
    chain = _chain(PRISM_ARM)
    link = chain.link_index["ee"]
    tgt = torch.tensor([[0.37, 0.0, 0.2], [0.05, 0.0, 0.2]], dtype=D)
    q0 = torch.zeros(2, 1, dtype=D)
    q, _ = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=1, tol=1e-14,
                  check_every=0)
    assert torch.allclose(q[:, 0], torch.tensor([0.37, 0.05], dtype=D),
                          atol=1e-12)


def test_prismatic_target_outside_range_clamps_to_the_stop():
    chain = _chain(PRISM_ARM)
    link = chain.link_index["ee"]
    tgt = torch.tensor([[0.9, 0.0, 0.2], [-0.9, 0.0, 0.2]], dtype=D)
    q0 = torch.zeros(2, 1, dtype=D)
    q, info = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=20, tol=1e-14,
                     check_every=0)
    assert torch.allclose(q[:, 0], torch.tensor([0.5, 0.0], dtype=D), atol=1e-12)
    # the residual is the leftover distance along the slide
    assert torch.allclose(info["final_error"],
                          torch.tensor([0.4, 0.9], dtype=D), atol=1e-9)


# ------------------------------------------------------------ properties ----

def test_error_never_increases_with_more_sweeps():
    """Each joint step is the exact minimizer over its own feasible interval,
    and a zero step is always feasible, so the position error is monotone non
    increasing in the sweep count. This is the sharpest structural check on the
    clamped step."""
    chain = _chain(MIXED_ARM)
    link = chain.link_index["ee"]
    gen = torch.Generator().manual_seed(33)
    tgt = 1.2 * (torch.rand(40, 3, generator=gen, dtype=D) - 0.4)
    lo, hi = chain.lower.to(D), chain.upper.to(D)
    q0 = lo + (hi - lo) * torch.rand(40, chain.dof, generator=gen, dtype=D)
    for step in (1.0, 0.5):
        prev = None
        for sweeps in range(0, 9):
            _, info = ik_ccd(chain, tgt, q0=q0, link_index=link,
                             sweeps=sweeps, tol=1e-14, step=step,
                             check_every=0)
            err = info["final_error"]
            if prev is not None:
                assert torch.all(err <= prev + 1e-12), (step, sweeps)
            prev = err


def test_zero_sweeps_returns_the_clamped_seed():
    chain = _chain(MIXED_ARM)
    link = chain.link_index["ee"]
    lo, hi = chain.lower.to(D), chain.upper.to(D)
    q0 = torch.stack([lo - 1.0, hi + 1.0, 0.5 * (lo + hi)], dim=0)
    tgt = torch.zeros(3, 3, dtype=D)
    q, info = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=0)
    assert torch.allclose(q, torch.clamp(q0, lo, hi))
    err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
    assert torch.allclose(info["final_error"], err, atol=1e-12)


def test_batch_elements_are_independent():
    """Solving five targets together must give exactly what solving them one at
    a time gives, seeds held fixed."""
    chain = _chain(SIX_DOF)
    link = chain.link_index["ee"]
    gen = torch.Generator().manual_seed(44)
    lo, hi = chain.lower.to(D), chain.upper.to(D)
    qs = lo + (hi - lo) * torch.rand(5, chain.dof, generator=gen, dtype=D)
    tgt = _ee_pos(chain, qs, link)
    q0 = lo + (hi - lo) * torch.rand(5, chain.dof, generator=gen, dtype=D)
    batched, _ = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=25,
                        tol=1e-14, check_every=0)
    for i in range(5):
        one, _ = ik_ccd(chain, tgt[i:i + 1], q0=q0[i:i + 1], link_index=link,
                        sweeps=25, tol=1e-14, check_every=0)
        assert torch.allclose(batched[i], one[0], atol=1e-11), i


def test_pose_target_uses_only_its_translation():
    chain = _chain(SIX_DOF)
    link = chain.link_index["ee"]
    gen = torch.Generator().manual_seed(55)
    pts = torch.stack([
        0.6 * (torch.rand(6, generator=gen, dtype=D) - 0.5),
        0.6 * (torch.rand(6, generator=gen, dtype=D) - 0.5),
        0.4 + 0.5 * torch.rand(6, generator=gen, dtype=D)], dim=1)
    pose = torch.eye(4, dtype=D).repeat(6, 1, 1)
    pose[:, :3, 3] = pts
    # a nonsense rotation block must not change the answer
    pose[:, :3, :3] = torch.tensor([[0.0, -1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0]], dtype=D)
    lo, hi = chain.lower.to(D), chain.upper.to(D)
    q0 = lo + (hi - lo) * torch.rand(6, chain.dof, generator=gen, dtype=D)
    a, _ = ik_ccd(chain, pts, q0=q0, link_index=link, sweeps=30, tol=1e-14,
                  check_every=0)
    b, _ = ik_ccd(chain, pose, q0=q0, link_index=link, sweeps=30, tol=1e-14,
                  check_every=0)
    assert torch.equal(a, b)


def test_link_index_forms_agree():
    chain = _chain(SIX_DOF)
    last = chain.n_links - 1
    assert chain.link_index["ee"] == last
    gen = torch.Generator().manual_seed(66)
    tgt = torch.stack([torch.zeros(4, dtype=D), torch.zeros(4, dtype=D),
                       0.5 + 0.4 * torch.rand(4, generator=gen, dtype=D)], dim=1)
    lo, hi = chain.lower.to(D), chain.upper.to(D)
    q0 = lo + (hi - lo) * torch.rand(4, chain.dof, generator=gen, dtype=D)
    out = [ik_ccd(chain, tgt, q0=q0, link_index=li, sweeps=20, tol=1e-14,
                  check_every=0)[0] for li in (None, -1, last)]
    assert torch.equal(out[0], out[1])
    assert torch.equal(out[0], out[2])


def test_intermediate_link_can_be_the_target():
    """Solving for a mid chain link must leave the joints past it untouched,
    because they cannot move it."""
    chain = _chain(SIX_DOF)
    link = chain.link_index["l3"]
    gen = torch.Generator().manual_seed(77)
    lo, hi = chain.lower.to(D), chain.upper.to(D)
    qs = lo + (hi - lo) * torch.rand(30, chain.dof, generator=gen, dtype=D)
    tgt = _ee_pos(chain, qs, link)
    q0 = lo + (hi - lo) * torch.rand(30, chain.dof, generator=gen, dtype=D)
    q, _ = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=200, tol=1e-14,
                  check_every=0)
    err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
    assert float(err.max()) < 1e-3
    # j4, j5, j6 are distal to l3 and must be exactly as seeded
    assert torch.equal(q[:, 3:], q0[:, 3:])


def test_free_spinning_joint_has_infinite_limits_and_still_solves():
    chain = _chain(FREE_SPIN)
    assert float(chain.lower[0]) == float("-inf")
    assert float(chain.upper[0]) == float("inf")
    link = chain.link_index["ee"]
    ang = torch.tensor([0.3, 2.0, -2.9, 3.0], dtype=D)
    tgt = torch.stack([0.4 * torch.cos(ang), 0.4 * torch.sin(ang),
                       torch.zeros(4, dtype=D)], dim=1)
    # random seeding must not draw an inf out of the infinite range
    torch.manual_seed(1)
    q, _ = ik_ccd(chain, tgt, link_index=link, sweeps=50, tol=1e-14,
                  restarts=2, check_every=0)
    assert torch.all(torch.isfinite(q))
    err = (_ee_pos(chain, q, link) - tgt).norm(dim=-1)
    assert float(err.max()) < 1e-9


# ------------------------------------------------------------- dtype/grad ---

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_working_dtype_follows_the_seed(dtype):
    """The chain is compiled in float64 in both cases; the seed decides."""
    chain = _chain(SIX_DOF, dtype=torch.float64)
    link = chain.link_index["ee"]
    tgt = torch.tensor([[0.2, 0.1, 0.7], [-0.3, 0.2, 0.5]], dtype=torch.float64)
    q0 = torch.zeros(2, chain.dof, dtype=dtype)
    q, info = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=50,
                     tol=1e-9, check_every=0)
    assert q.dtype == dtype
    assert info["final_error"].dtype == dtype
    err = (_ee_pos(chain, q, link) - tgt.to(dtype)).norm(dim=-1)
    assert float(err.max()) < 2e-3


def test_float32_and_float64_agree_to_single_precision():
    chain = _chain(SIX_DOF)
    link = chain.link_index["ee"]
    gen = torch.Generator().manual_seed(88)
    lo, hi = chain.lower.to(D), chain.upper.to(D)
    q0 = lo + (hi - lo) * torch.rand(8, chain.dof, generator=gen, dtype=D)
    tgt = _ee_pos(chain, lo + (hi - lo) * torch.rand(
        8, chain.dof, generator=gen, dtype=D), link)
    a, _ = ik_ccd(chain, tgt, q0=q0, link_index=link, sweeps=15, tol=1e-14,
                  check_every=0)
    b, _ = ik_ccd(chain, tgt.float(), q0=q0.float(), link_index=link,
                  sweeps=15, tol=1e-14, check_every=0)
    assert torch.allclose(a.float(), b, atol=2e-4)


def test_gradient_matches_central_differences():
    """The solve is autograd traceable: d(final error)/d(seed) from backward
    must match float64 central differences. Few sweeps, well inside the limits,
    so the run is smooth in the seed."""
    chain = _chain(SIX_DOF)
    link = chain.link_index["ee"]
    tgt = torch.tensor([[0.25, -0.15, 0.75], [-0.1, 0.3, 0.6]], dtype=D)
    base = torch.tensor([[0.2, 0.4, -0.3, 0.1, 0.5, -0.2],
                         [-0.4, 0.3, 0.6, -0.2, -0.5, 0.3]], dtype=D)

    def loss(seed):
        q, _ = ik_ccd(chain, tgt, q0=seed, link_index=link, sweeps=3,
                      tol=1e-14, step=0.6, check_every=0)
        p = forward_kinematics(chain, q)[:, link, :3, 3]
        return ((p - tgt) ** 2).sum()

    x = base.clone().requires_grad_(True)
    loss(x).backward()
    grad = x.grad.clone()
    assert torch.all(torch.isfinite(grad))
    assert float(grad.abs().max()) > 1e-6

    h = 1e-6
    fd = torch.zeros_like(base)
    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            plus = base.clone()
            plus[i, j] += h
            minus = base.clone()
            minus[i, j] -= h
            fd[i, j] = (loss(plus) - loss(minus)) / (2 * h)
    assert torch.allclose(grad, fd, atol=1e-6, rtol=1e-4), (grad, fd)


# ------------------------------------------------------------ validation ----

def test_rejects_bad_inputs():
    chain = _chain(PLANAR_2R)
    tgt = torch.zeros(2, 3, dtype=D)
    with pytest.raises(TypeError):
        ik_ccd(chain, [[0.0, 0.0, 0.0]])
    with pytest.raises(TypeError):
        ik_ccd(chain, tgt, q0=[[0.0, 0.0]])
    with pytest.raises(ValueError):
        ik_ccd(chain, torch.zeros(2, 5, dtype=D))
    with pytest.raises(ValueError):
        ik_ccd(chain, tgt, step=0.0)
    with pytest.raises(ValueError):
        ik_ccd(chain, tgt, q0=torch.zeros(3, 2, dtype=D))
    with pytest.raises(IndexError):
        ik_ccd(chain, tgt, link_index=99)
    with pytest.raises(TypeError):
        # an integer target cannot seed a random start
        ik_ccd(chain, torch.zeros(2, 3, dtype=torch.long))
