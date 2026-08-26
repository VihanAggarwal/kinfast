# tests/test_acm.py
"""Allowed-collision matrix: bucket classification and masked self-distance.

The fixture is the 6-DOF spatial arm with deliberately overlapping spheres, and
it is built so that most verdicts are exact rather than statistical:

  s0  base link, at the link origin, r = 0.10
  s1  l1   link, at the link origin, r = 0.04
  s2  l2   link, at the link origin, r = 0.05
  s3  ee   link, at the link origin, r = 0.72

Joint j1 sits at the base origin and turns about z, so the l1 frame origin is
pinned to the world origin for every configuration: s0 and s1 are coincident
always, at a signed distance of exactly -(0.10 + 0.04) = -0.14. Joint j2 offsets
l2 by (0, 0, 0.3) along the l1 z axis, which j1 rotates about, so the l2 frame
origin sits at world (0, 0, 0.3) for every configuration too: s2 is exactly 0.3
from both s0 and s1, giving constant signed distances of 0.15 and 0.21. The
oversized ee sphere is the only one whose contacts depend on the joints, and its
radius was picked so it is in contact for roughly half the reachable
configurations, which puts all three of its pairs firmly in the "check" bucket.

Oracles used here are hand-computed geometry, float64 central differences, a
held-out sample set the classifier never saw, and `kinfast.collision`, which was
written independently of this module.
"""
import math

import pytest
import torch

from kinfast.acm import (allowed_pairs, mask_to_pairs, pair_distances,
                         pairs_to_mask, self_distance_masked, upper_pairs)
from kinfast.analysis import sampling_bounds
from kinfast.collision import SphereModel, self_distance
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.urdf.parse import parse_urdf_string

from tests.test_spatial import SIX_DOF

# sphere indices, in the order SphereModel walks the dict below
S_BASE, S_L1, S_L2, S_EE = 0, 1, 2, 3
R = {S_BASE: 0.10, S_L1: 0.04, S_L2: 0.05, S_EE: 0.72}

# the exact signed distances of the three rigid pairs, valid at every q
RIGID = {
    (S_BASE, S_L1): 0.0 - R[S_BASE] - R[S_L1],   # coincident       -> -0.14
    (S_BASE, S_L2): 0.3 - R[S_BASE] - R[S_L2],   # 0.3 apart        ->  0.15
    (S_L1, S_L2): 0.3 - R[S_L1] - R[S_L2],       # 0.3 apart        ->  0.21
}


def _chain(dtype=torch.float64):
    return compile_robot(parse_urdf_string(SIX_DOF), dtype=dtype)


def _model(chain):
    li = chain.link_index
    return SphereModel(chain, {
        li["base"]: [(0.0, 0.0, 0.0, R[S_BASE])],
        li["l1"]: [(0.0, 0.0, 0.0, R[S_L1])],
        li["l2"]: [(0.0, 0.0, 0.0, R[S_L2])],
        li["ee"]: [(0.0, 0.0, 0.0, R[S_EE])],
    })


def _pair_set(t):
    """(K, 2) tensor of pairs -> a set of tuples, for order-free comparison."""
    return {(int(a), int(b)) for a, b in t.tolist()}


def _random_q(chain, n, seed, dtype=torch.float64):
    lo, hi = sampling_bounds(chain)
    lo, hi = lo.to(dtype), hi.to(dtype)
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(n, chain.dof, generator=g, dtype=dtype)


# --------------------------------------------------------------------------
# pair bookkeeping helpers
# --------------------------------------------------------------------------

def test_upper_pairs_enumerates_every_unordered_pair_once():
    p = upper_pairs(4)
    assert p.shape == (6, 2) and p.dtype == torch.long
    assert _pair_set(p) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert bool((p[:, 0] < p[:, 1]).all())


def test_upper_pairs_of_a_single_sphere_is_empty():
    assert upper_pairs(1).shape == (0, 2)


def test_pairs_and_mask_round_trip():
    pairs = torch.tensor([[0, 3], [1, 2]])
    m = pairs_to_mask(pairs, 4)
    assert bool((m == m.t()).all()) and not bool(m.diagonal().any())
    assert _pair_set(mask_to_pairs(m)) == {(0, 3), (1, 2)}


def test_mask_is_read_symmetrically():
    """A caller may fill in only one triangle; the pair below the diagonal is
    the same pair as the one above it and must not be counted twice."""
    m = torch.zeros(3, 3, dtype=torch.bool)
    m[2, 0] = True                      # lower triangle only
    assert _pair_set(mask_to_pairs(m)) == {(0, 2)}


def test_mask_helpers_reject_malformed_input():
    with pytest.raises(TypeError):
        mask_to_pairs(torch.zeros(3, 3))            # not bool
    with pytest.raises(ValueError):
        mask_to_pairs(torch.zeros(3, 4, dtype=torch.bool))
    with pytest.raises(ValueError):
        pairs_to_mask(torch.tensor([[0, 5]]), 3)    # index out of range


# --------------------------------------------------------------------------
# pair_distances against hand-computed geometry and an explicit loop
# --------------------------------------------------------------------------

def test_pair_distances_match_hand_computed_rigid_pairs():
    chain = _chain()
    model = _model(chain)
    q = _random_q(chain, 8, seed=3)
    sd = pair_distances(model, q)                    # (B, S, S)
    for (i, j), expected in RIGID.items():
        got = sd[:, i, j]
        assert torch.allclose(got, torch.full_like(got, expected), atol=1e-12)
        assert torch.allclose(sd[:, j, i], got, atol=1e-12)   # symmetric


def test_pair_distances_match_an_explicit_fk_loop():
    """Independent oracle: rebuild the world sphere centers straight from the
    4x4 FK transforms in a plain Python loop and redo the arithmetic."""
    chain = _chain()
    model = _model(chain)
    q = _random_q(chain, 5, seed=4)
    W = forward_kinematics(chain, q)                 # (B, n_links, 4, 4)
    sd = pair_distances(model, q)
    for b in range(q.shape[0]):
        centers = []
        for s in range(model.n):
            T = W[b, int(model.link[s])]
            c = T[:3, :3] @ model.local[s] + T[:3, 3]
            centers.append(c)
        for i in range(model.n):
            for j in range(model.n):
                want = ((centers[i] - centers[j]).norm()
                        - model.radius[i] - model.radius[j])
                assert abs(float(sd[b, i, j] - want)) < 1e-12


def test_pair_distances_pair_form_matches_the_full_matrix():
    chain = _chain()
    model = _model(chain)
    q = _random_q(chain, 6, seed=5)
    full = pair_distances(model, q)
    pairs = upper_pairs(model.n)
    thin = pair_distances(model, q, pairs)
    assert thin.shape == (6, pairs.shape[0])
    assert torch.allclose(thin, full[:, pairs[:, 0], pairs[:, 1]], atol=1e-14)


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def test_buckets_are_hand_verified():
    chain = _chain()
    model = _model(chain)
    res = allowed_pairs(model, 256, seed=0)
    assert _pair_set(res["always"]) == {(S_BASE, S_L1)}
    assert _pair_set(res["never"]) == {(S_BASE, S_L2), (S_L1, S_L2)}
    assert _pair_set(res["check"]) == {(S_BASE, S_EE), (S_L1, S_EE),
                                       (S_L2, S_EE)}


def test_buckets_partition_every_pair_exactly_once():
    chain = _chain()
    model = _model(chain)
    res = allowed_pairs(model, 128, seed=7)
    a, n, c = (_pair_set(res[k]) for k in ("always", "never", "check"))
    assert not (a & n) and not (a & c) and not (n & c)
    assert a | n | c == _pair_set(upper_pairs(model.n))
    assert len(a) + len(n) + len(c) == model.n * (model.n - 1) // 2


def test_reported_extremes_are_exact_for_the_rigid_pairs():
    chain = _chain()
    model = _model(chain)
    res = allowed_pairs(model, 64, seed=1)
    for (i, j), expected in RIGID.items():
        assert abs(float(res["min_distance"][i, j]) - expected) < 1e-12
        assert abs(float(res["max_distance"][i, j]) - expected) < 1e-12
    assert bool((res["min_distance"] <= res["max_distance"]).all())
    assert float(res["min_distance"].diagonal().abs().max()) == 0.0


def test_contact_fraction_is_zero_one_on_the_pruned_buckets():
    chain = _chain()
    model = _model(chain)
    res = allowed_pairs(model, 256, seed=2)
    frac = res["contact_fraction"]
    for i, j in _pair_set(res["always"]):
        assert float(frac[i, j]) == 1.0
    for i, j in _pair_set(res["never"]):
        assert float(frac[i, j]) == 0.0
    for i, j in _pair_set(res["check"]):
        assert 0.0 < float(frac[i, j]) < 1.0
    assert bool((frac == frac.t()).all())


def test_mask_selects_exactly_the_check_pairs():
    chain = _chain()
    model = _model(chain)
    res = allowed_pairs(model, 128, seed=0)
    m = res["mask"]
    assert m.shape == (model.n, model.n) and m.dtype == torch.bool
    assert bool((m == m.t()).all())
    assert _pair_set(mask_to_pairs(m)) == _pair_set(res["check"])
    # the off-diagonal complement is the allowed-collision matrix proper
    skip = ~m & ~torch.eye(model.n, dtype=torch.bool)
    assert (_pair_set(mask_to_pairs(skip))
            == _pair_set(res["always"]) | _pair_set(res["never"]))


def test_link_mask_is_the_link_level_view():
    chain = _chain()
    model = _model(chain)
    res = allowed_pairs(model, 128, seed=0)
    li = chain.link_index
    lm = res["link_mask"]
    assert lm.shape == (chain.n_links, chain.n_links)
    assert bool((lm == lm.t()).all())
    # the three checked sphere pairs all involve ee
    for a in ("base", "l1", "l2"):
        assert bool(lm[li[a], li["ee"]])
    # the rigid pairs were resolved offline, so their links need no runtime check
    assert not bool(lm[li["base"], li["l1"]])
    assert not bool(lm[li["base"], li["l2"]])
    assert not bool(lm[li["l1"], li["l2"]])
    assert int(lm.sum()) == 6           # three link pairs, both orientations


# --------------------------------------------------------------------------
# stability of the verdict
# --------------------------------------------------------------------------

def test_classification_is_stable_across_seeds():
    chain = _chain()
    model = _model(chain)
    ref = {k: _pair_set(allowed_pairs(model, 256, seed=0)[k])
           for k in ("always", "never", "check")}
    for seed in (1, 2, 3, 17, 4242):
        res = allowed_pairs(model, 256, seed=seed)
        for k, want in ref.items():
            assert _pair_set(res[k]) == want, f"{k} moved at seed {seed}"


def test_classification_is_stable_as_the_sample_count_grows():
    chain = _chain()
    model = _model(chain)
    ref = {k: _pair_set(allowed_pairs(model, 64, seed=0)[k])
           for k in ("always", "never", "check")}
    for n in (128, 512, 1024):
        res = allowed_pairs(model, n, seed=0)
        for k, want in ref.items():
            assert _pair_set(res[k]) == want, f"{k} moved at n={n}"


def test_same_seed_reproduces_the_same_configurations():
    chain = _chain()
    model = _model(chain)
    a = allowed_pairs(model, 32, seed=11)
    b = allowed_pairs(model, 32, seed=11)
    assert torch.equal(a["q"], b["q"])
    assert not torch.equal(a["q"], allowed_pairs(model, 32, seed=12)["q"])
    assert a["q"].shape == (32, chain.dof)
    assert a["n_samples"] == 32


def test_chunking_does_not_change_the_answer():
    chain = _chain()
    model = _model(chain)
    ref = allowed_pairs(model, 100, seed=5, chunk=100)
    for chunk in (1, 7, 64, 1000):
        res = allowed_pairs(model, 100, seed=5, chunk=chunk)
        for k in ("always", "never", "check"):
            assert _pair_set(res[k]) == _pair_set(ref[k])
        assert torch.allclose(res["min_distance"], ref["min_distance"], atol=1e-14)
        assert torch.allclose(res["max_distance"], ref["max_distance"], atol=1e-14)
        assert torch.equal(res["contact_fraction"], ref["contact_fraction"])


def test_classification_is_stable_across_dtypes():
    """float32 and float64 chains must agree; the fixture keeps every pair well
    away from the decision boundary, so precision cannot flip a verdict."""
    m64 = _model(_chain(torch.float64))
    m32 = _model(_chain(torch.float32))
    r64 = allowed_pairs(m64, 256, seed=0)
    r32 = allowed_pairs(m32, 256, seed=0)
    assert r32["min_distance"].dtype == torch.float32
    assert r64["min_distance"].dtype == torch.float64
    for k in ("always", "never", "check"):
        assert _pair_set(r32[k]) == _pair_set(r64[k])


def test_never_pairs_stay_clear_on_held_out_samples():
    """The empirical claim, tested on configurations the classifier never saw."""
    chain = _chain()
    model = _model(chain)
    res = allowed_pairs(model, 256, seed=0)
    held_out = _random_q(chain, 4000, seed=99999)
    sd = pair_distances(model, held_out, res["never"])
    assert float(sd.min()) > 0.0
    always = pair_distances(model, held_out, res["always"])
    assert float(always.max()) <= 0.0


# --------------------------------------------------------------------------
# margin and safety
# --------------------------------------------------------------------------

def test_margin_promotes_a_clear_pair_to_always():
    """s1 and s2 sit at a constant signed distance of exactly 0.21, so a margin
    of 0.25 puts them in contact in every sample and nothing less than 0.21
    does."""
    chain = _chain()
    model = _model(chain)
    assert (S_L1, S_L2) in _pair_set(allowed_pairs(model, 64, margin=0.20)["never"])
    res = allowed_pairs(model, 64, margin=0.25)
    assert (S_L1, S_L2) in _pair_set(res["always"])
    assert (S_BASE, S_L2) in _pair_set(res["always"])     # constant 0.15


def test_safety_keeps_a_near_miss_out_of_never():
    """Same constant-0.21 pair: with no safety it is prunable, with 0.25 of
    required clearance it has to be checked instead."""
    chain = _chain()
    model = _model(chain)
    assert (S_L1, S_L2) in _pair_set(allowed_pairs(model, 64, safety=0.0)["never"])
    res = allowed_pairs(model, 64, safety=0.25)
    assert (S_L1, S_L2) in _pair_set(res["check"])
    assert (S_BASE, S_L1) in _pair_set(res["always"])     # safety cannot move this


def test_margin_and_safety_are_echoed_back():
    model = _model(_chain())
    res = allowed_pairs(model, 8, margin=0.01, safety=0.02)
    assert res["margin"] == pytest.approx(0.01)
    assert res["safety"] == pytest.approx(0.02)


# --------------------------------------------------------------------------
# self_distance_masked
# --------------------------------------------------------------------------

def test_masked_distance_is_hand_computed_at_the_zero_configuration():
    """At q = 0 the ee frame is at (0, 0, 1.1), so the three checked pairs are
    1.1 - 0.72 - 0.10 = 0.28 (base), 1.1 - 0.72 - 0.04 = 0.34 (l1) and
    0.8 - 0.72 - 0.05 = 0.03 (l2, whose origin is at z = 0.3). The minimum over
    the checked pairs is therefore 0.03."""
    chain = _chain()
    model = _model(chain)
    res = allowed_pairs(model, 256, seed=0)
    q = torch.zeros(1, chain.dof, dtype=torch.float64)
    d = self_distance_masked(model, q, res["mask"])
    assert torch.allclose(d, torch.tensor([0.03], dtype=torch.float64), atol=1e-12)


def test_masked_distance_ignores_the_permanent_overlap():
    """Without a mask the permanent base/l1 overlap caps the reported distance
    at -0.14, so every single configuration looks like a self-collision and no
    planner using it can ever find a valid motion. That is the whole reason the
    ACM exists: masking the permanent overlap away lets the genuine clearances
    show through."""
    chain = _chain()
    model = _model(chain)
    q = _random_q(chain, 256, seed=8)
    raw = self_distance_masked(model, q, None)
    assert bool((raw <= RIGID[(S_BASE, S_L1)] + 1e-12).all())
    # -0.14 is the answer whenever the ee sphere is not the deeper penetration
    assert bool(((raw - RIGID[(S_BASE, S_L1)]).abs() < 1e-12).any())

    masked = self_distance_masked(model, q, allowed_pairs(model, 256)["mask"])
    assert bool((masked >= raw - 1e-12).all())
    assert bool((masked > 0).any()), "the masked query must find clear poses"
    # and where the arm really is folded onto itself the masked query still
    # reports the collision rather than hiding it
    assert bool((masked < 0).any())


def test_masked_distance_accepts_a_mask_or_a_pair_list():
    chain = _chain()
    model = _model(chain)
    res = allowed_pairs(model, 128, seed=0)
    q = _random_q(chain, 12, seed=9)
    by_mask = self_distance_masked(model, q, res["mask"])
    by_pairs = self_distance_masked(model, q, res["check"])
    assert torch.allclose(by_mask, by_pairs, atol=1e-14)


def test_masked_distance_is_batched_and_per_configuration():
    chain = _chain()
    model = _model(chain)
    mask = allowed_pairs(model, 128, seed=0)["mask"]
    q = _random_q(chain, 9, seed=10)
    batched = self_distance_masked(model, q, mask)
    assert batched.shape == (9,)
    one_at_a_time = torch.stack(
        [self_distance_masked(model, q[i:i + 1], mask)[0] for i in range(9)])
    assert torch.allclose(batched, one_at_a_time, atol=1e-14)


def test_masked_distance_of_an_empty_mask_is_positive_infinity():
    chain = _chain()
    model = _model(chain)
    q = torch.zeros(3, chain.dof, dtype=torch.float64)
    empty = torch.zeros(model.n, model.n, dtype=torch.bool)
    d = self_distance_masked(model, q, empty)
    assert d.shape == (3,) and bool(torch.isinf(d).all()) and bool((d > 0).all())
    assert torch.equal(d, self_distance_masked(model, q, torch.zeros(0, 2,
                                                                    dtype=torch.long)))


def test_masked_distance_matches_the_independent_collision_module():
    """`kinfast.collision.self_distance` masks by structure alone: same link,
    or parent/child links. Build that mask here by hand and the two must agree,
    which pins the arithmetic against an implementation written separately."""
    chain = _chain()
    model = _model(chain)
    S = model.n
    structural = torch.zeros(S, S, dtype=torch.bool)
    for i in range(S):
        for j in range(i + 1, S):
            li, lj = int(model.link[i]), int(model.link[j])
            same = li == lj
            adjacent = (int(chain.parent[li]) == lj
                        or int(chain.parent[lj]) == li)
            structural[i, j] = not (same or adjacent)
    q = _random_q(chain, 20, seed=12)
    assert torch.allclose(self_distance_masked(model, q, structural),
                          self_distance(model, q), atol=1e-14)


def test_masked_distance_gradient_matches_central_differences():
    """float64 central differences on the masked minimum. The minimum is
    piecewise smooth, so the check uses a configuration where the winning pair
    wins by a clear margin and cannot swap under the perturbation."""
    chain = _chain()
    model = _model(chain)
    mask = allowed_pairs(model, 256, seed=0)["mask"]
    q0 = torch.tensor([[0.4, -0.7, 0.9, 0.3, -0.5, 0.2]], dtype=torch.float64)
    pairs = mask_to_pairs(mask)
    sd = pair_distances(model, q0, pairs)[0]
    ordered = sd.sort().values
    assert float(ordered[1] - ordered[0]) > 1e-3, "argmin is not clearly unique"

    q = q0.clone().requires_grad_(True)
    self_distance_masked(model, q, mask).sum().backward()
    analytic = q.grad[0]
    assert torch.isfinite(analytic).all()

    eps = 1e-6
    numeric = torch.zeros_like(analytic)
    for k in range(chain.dof):
        step = torch.zeros_like(q0)
        step[0, k] = eps
        plus = self_distance_masked(model, q0 + step, mask)
        minus = self_distance_masked(model, q0 - step, mask)
        numeric[k] = (plus - minus)[0] / (2 * eps)
    assert torch.allclose(analytic, numeric, atol=1e-7)
    assert float(analytic.abs().max()) > 1e-3     # a real gradient, not zero


def test_masked_distance_follows_the_working_dtype_of_q():
    chain32 = _chain(torch.float32)
    model32 = _model(chain32)
    mask = allowed_pairs(model32, 64, seed=0)["mask"]
    q = torch.zeros(2, chain32.dof, dtype=torch.float32)
    assert self_distance_masked(model32, q, mask).dtype == torch.float32
    # a float64 q against a float32 chain still works in float64, as elsewhere
    q64 = torch.zeros(2, chain32.dof, dtype=torch.float64)
    assert self_distance_masked(model32, q64, mask).dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runs_on_cuda_and_agrees_with_cpu():
    chain = _chain(torch.float32)
    model = _model(chain)
    cpu = allowed_pairs(model, 128, seed=0)
    gpu = allowed_pairs(model, 128, seed=0, device="cuda")
    for k in ("always", "never", "check"):
        assert _pair_set(gpu[k].cpu()) == _pair_set(cpu[k])
    q = _random_q(chain, 8, seed=13, dtype=torch.float32)
    d_gpu = self_distance_masked(model, q.cuda(), gpu["mask"])
    assert d_gpu.device.type == "cuda"
    assert torch.allclose(d_gpu.cpu(), self_distance_masked(model, q, cpu["mask"]),
                          atol=1e-5)


# --------------------------------------------------------------------------
# caller-supplied configurations
# --------------------------------------------------------------------------

def test_supplied_configurations_replace_the_sampling():
    """Classifying over a narrow set of poses gives a narrow verdict: here the
    arm is held near the zero configuration, where the ee sphere never reaches
    the base, so nothing at all needs checking at runtime."""
    chain = _chain()
    model = _model(chain)
    q = torch.zeros(5, chain.dof, dtype=torch.float64)
    q[:, 1] = torch.linspace(-0.1, 0.1, 5, dtype=torch.float64)
    res = allowed_pairs(model, q=q)
    assert res["n_samples"] == 5
    assert torch.equal(res["q"], q)
    assert _pair_set(res["check"]) == set()
    assert _pair_set(res["always"]) == {(S_BASE, S_L1)}
    assert self_distance_masked(model, q, res["mask"]).allclose(
        torch.full((5,), float("inf"), dtype=torch.float64))


def test_supplied_configurations_are_detached():
    chain = _chain()
    model = _model(chain)
    q = torch.zeros(3, chain.dof, dtype=torch.float64, requires_grad=True)
    res = allowed_pairs(model, q=q)
    assert not res["q"].requires_grad


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def test_rejects_bad_arguments():
    chain = _chain()
    model = _model(chain)
    q = torch.zeros(1, chain.dof, dtype=torch.float64)
    with pytest.raises(ValueError, match="positive number of samples"):
        allowed_pairs(model, 0)
    with pytest.raises(ValueError, match="at least one configuration"):
        allowed_pairs(model, q=torch.zeros(0, chain.dof))
    with pytest.raises(ValueError, match=r"shape \(B, 6\)"):
        allowed_pairs(model, q=torch.zeros(4, 3))
    with pytest.raises(TypeError, match="floating point"):
        allowed_pairs(model, q=torch.zeros(2, chain.dof, dtype=torch.long))
    with pytest.raises(ValueError, match="chunk"):
        allowed_pairs(model, 8, chunk=0)
    with pytest.raises(ValueError, match="safety"):
        allowed_pairs(model, 8, safety=-0.1)
    with pytest.raises(ValueError, match="margin must be finite"):
        allowed_pairs(model, 8, margin=float("nan"))
    with pytest.raises(ValueError, match="shape"):
        self_distance_masked(model, q, torch.zeros(3, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="indices must be in"):
        self_distance_masked(model, q, torch.tensor([[0, 99]]))
    with pytest.raises(ValueError, match="paired with itself"):
        self_distance_masked(model, q, torch.tensor([[2, 2]]))


def test_rejects_a_model_with_no_spheres():
    """An empty sphere dict builds a model that cannot even place a center, so
    say so up front instead of failing deep inside the kinematics."""
    model = SphereModel(_chain(), {})
    assert model.n == 0
    with pytest.raises(ValueError, match="no spheres"):
        allowed_pairs(model, 8)


def test_a_single_sphere_model_has_nothing_to_classify():
    chain = _chain()
    li = chain.link_index
    model = SphereModel(chain, {li["ee"]: [(0.0, 0.0, 0.0, 0.1)]})
    res = allowed_pairs(model, 8)
    for k in ("always", "never", "check"):
        assert res[k].shape == (0, 2)
    q = torch.zeros(2, chain.dof, dtype=torch.float64)
    assert bool(torch.isinf(self_distance_masked(model, q, res["mask"])).all())


# --------------------------------------------------------------------------
# a second robot, to be sure nothing is fitted to the fixture
# --------------------------------------------------------------------------

PRISMATIC = """
<robot name="slider">
  <link name="base"/><link name="rail"/><link name="head"/>
  <joint name="j1" type="prismatic"><parent link="base"/><child link="rail"/>
    <origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="1.0" velocity="1" effort="10"/></joint>
  <joint name="j2" type="revolute"><parent link="rail"/><child link="head"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="10"/></joint>
</robot>
"""


def test_prismatic_arm_classification_is_hand_verifiable():
    """The head rides the rail from x = 0 to x = 1. A base sphere of radius 0.2
    at the origin and a head sphere of radius 0.3 are in contact exactly when
    x < 0.5, so the pair must land in "check"; a second head sphere pushed out
    to x = +2 in the head frame can never come back, so that pair is "never"."""
    chain = compile_robot(parse_urdf_string(PRISMATIC), dtype=torch.float64)
    li = chain.link_index
    model = SphereModel(chain, {
        li["base"]: [(0.0, 0.0, 0.0, 0.2)],
        li["head"]: [(0.0, 0.0, 0.0, 0.3), (2.0, 0.0, 0.0, 0.1)],
    })
    res = allowed_pairs(model, 512, seed=0)
    assert (0, 1) in _pair_set(res["check"])
    assert (0, 2) in _pair_set(res["never"])
    # contact fraction of the sliding pair is the length ratio 0.5, up to noise
    assert float(res["contact_fraction"][0, 1]) == pytest.approx(0.5, abs=0.08)
    # exact extremes: x runs over [0, 1], so the signed distance runs over
    # [-0.5, 0.5] and the far sphere never gets closer than 2 - 0.2 - 0.1 = 1.7
    assert float(res["min_distance"][0, 1]) == pytest.approx(-0.5, abs=0.02)
    assert float(res["max_distance"][0, 1]) == pytest.approx(0.5, abs=0.02)
    assert float(res["min_distance"][0, 2]) == pytest.approx(1.7, abs=0.02)


def test_prismatic_masked_distance_at_a_known_pose():
    chain = compile_robot(parse_urdf_string(PRISMATIC), dtype=torch.float64)
    li = chain.link_index
    model = SphereModel(chain, {
        li["base"]: [(0.0, 0.0, 0.0, 0.2)],
        li["head"]: [(0.0, 0.0, 0.0, 0.3), (2.0, 0.0, 0.0, 0.1)],
    })
    mask = allowed_pairs(model, 512, seed=0)["mask"]
    # slide to x = 0.8: base/head signed distance is 0.8 - 0.2 - 0.3 = 0.3
    q = torch.tensor([[0.8, 0.0]], dtype=torch.float64)
    assert torch.allclose(self_distance_masked(model, q, mask),
                          torch.tensor([0.3], dtype=torch.float64), atol=1e-12)
    # slide to x = 0.1: 0.1 - 0.5 = -0.4, a real self-collision
    q = torch.tensor([[0.1, 0.0]], dtype=torch.float64)
    assert torch.allclose(self_distance_masked(model, q, mask),
                          torch.tensor([-0.4], dtype=torch.float64), atol=1e-12)


def test_rotation_does_not_move_a_sphere_on_the_rotation_axis():
    """A sanity anchor with no sampling in it: the head spins about z through
    its own origin, so the base/head distance depends only on the slide."""
    chain = compile_robot(parse_urdf_string(PRISMATIC), dtype=torch.float64)
    li = chain.link_index
    model = SphereModel(chain, {
        li["base"]: [(0.0, 0.0, 0.0, 0.2)],
        li["head"]: [(0.0, 0.0, 0.0, 0.3)],
    })
    q = torch.tensor([[0.6, a] for a in (-1.0, -0.3, 0.0, 0.5, 1.0)],
                     dtype=torch.float64)
    sd = pair_distances(model, q, torch.tensor([[0, 1]]))[:, 0]
    want = torch.full((5,), 0.6 - 0.2 - 0.3, dtype=torch.float64)
    assert torch.allclose(sd, want, atol=1e-12)
    assert math.isclose(float(sd[0]), 0.1, abs_tol=1e-12)
