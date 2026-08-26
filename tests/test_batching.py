# tests/test_batching.py
"""Chunked evaluation: same answers, smaller peak allocation.

The contract worth testing is that chunking is invisible. Where a closed form
exists the chunked result is checked against hand-written arithmetic rather
than against the unchunked library call, and the gradient that flows through
the chunk boundaries is checked against float64 central differences.
"""
import math

import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.urdf.parse import parse_urdf_string
from kinfast.fk import forward_kinematics
from kinfast.batching import map_in_chunks, ik_chunked
from kinfast.ik import ik

from tests.test_spatial import SIX_DOF
from tests.test_analysis import PLANAR_2R

# large enough that the modulo check in ik never fires
NO_EARLY_EXIT = 10 ** 9


def _arm6(dtype=torch.float64):
    return compile_robot(parse_urdf_string(SIX_DOF), dtype=dtype)


def _p2r(dtype=torch.float64):
    return compile_robot(parse_urdf_string(PLANAR_2R), dtype=dtype)


# ---------------------------------------------------------------- map_in_chunks

def test_matches_hand_computed_values():
    """Oracle: sum of squares per row, worked out without touching the library."""
    x = torch.arange(21, dtype=torch.float64).reshape(7, 3)
    out = map_in_chunks(lambda t: (t ** 2).sum(dim=-1), x, chunk=3)
    expect = torch.tensor(
        [sum(v * v for v in range(3 * i, 3 * i + 3)) for i in range(7)],
        dtype=torch.float64)
    assert out.shape == (7,)
    assert torch.equal(out, expect)


def test_identical_to_unchunked_fk():
    """A real batched kinfast call, chunked and not, must agree exactly."""
    chain = _arm6()
    torch.manual_seed(0)
    q = torch.rand(11, chain.dof, dtype=torch.float64) - 0.5
    ref = forward_kinematics(chain, q)
    for chunk in (1, 2, 3, 5, 10, 11):
        got = map_in_chunks(lambda t: forward_kinematics(chain, t), q, chunk)
        assert torch.equal(got, ref), f"chunk={chunk}"


def test_chunk_larger_than_batch_and_none():
    chain = _arm6()
    q = torch.linspace(-1.0, 1.0, 4 * chain.dof, dtype=torch.float64).reshape(4, -1)
    ref = forward_kinematics(chain, q)
    for chunk in (4, 5, 1000, None):
        got = map_in_chunks(lambda t: forward_kinematics(chain, t), q, chunk)
        assert torch.equal(got, ref), f"chunk={chunk}"


def test_batch_not_divisible_by_chunk():
    """10 rows in chunks of 3 means calls of 3, 3, 3, 1 and nothing dropped."""
    seen = []

    def fn(t):
        seen.append(t.shape[0])
        return t * 2.0

    x = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
    out = map_in_chunks(fn, x, chunk=3)
    assert seen == [3, 3, 3, 1]
    assert torch.equal(out, x * 2.0)


def test_single_call_when_chunk_covers_batch():
    seen = []

    def fn(t):
        seen.append(t.shape[0])
        return t

    x = torch.zeros(6, 2)
    map_in_chunks(fn, x, chunk=6)
    map_in_chunks(fn, x, chunk=99)
    assert seen == [6, 6]


def test_several_inputs_and_none_passthrough():
    a = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    b = torch.full((5, 2), 3.0, dtype=torch.float64)

    def fn(x, y, opt):
        assert opt is None          # None entries are handed over unsliced
        return x * y

    out = map_in_chunks(fn, (a, b, None), chunk=2)
    assert torch.equal(out, a * 3.0)


def test_structured_return_values():
    """Tuples and dicts are walked; per-call metadata is not concatenated."""
    x = torch.arange(9, dtype=torch.float64).unsqueeze(-1)

    def fn(t):
        return t + 1.0, {"err": t.squeeze(-1) * 2.0,
                         "iters": 7,
                         "scalar": torch.tensor(1.5, dtype=torch.float64)}

    q, info = map_in_chunks(fn, x, chunk=4)
    assert torch.equal(q, x + 1.0)
    assert torch.equal(info["err"], x.squeeze(-1) * 2.0)
    assert info["iters"] == 7                      # plain value, taken as is
    assert info["scalar"].item() == 1.5            # 0-dim tensor, not batched


def test_nested_lists_are_rebuilt():
    x = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    out = map_in_chunks(lambda t: [t, [t * 2.0]], x, chunk=4)
    assert isinstance(out, list) and isinstance(out[1], list)
    assert torch.equal(out[0], x)
    assert torch.equal(out[1][0], x * 2.0)


def test_empty_batch_keeps_shape():
    chain = _arm6()
    q = torch.zeros(0, chain.dof, dtype=torch.float64)
    out = map_in_chunks(lambda t: forward_kinematics(chain, t), q, chunk=4)
    assert out.shape == (0, chain.n_links, 4, 4)


def test_bad_arguments_raise():
    x = torch.zeros(4, 2)
    with pytest.raises(ValueError):
        map_in_chunks(lambda t: t, x, chunk=0)
    with pytest.raises(ValueError):
        map_in_chunks(lambda t: t, x, chunk=-3)
    with pytest.raises(TypeError):
        map_in_chunks(lambda t: t, x, chunk=2.5)
    with pytest.raises(TypeError):
        map_in_chunks(lambda t: t, x, chunk=True)      # bool is not a size
    with pytest.raises(ValueError):
        map_in_chunks(lambda a, b: a, (x, torch.zeros(5, 2)), chunk=2)
    with pytest.raises(ValueError):
        map_in_chunks(lambda a: a, (None,), chunk=2)
    with pytest.raises(ValueError):
        map_in_chunks(lambda a: a, torch.tensor(1.0), chunk=2)
    with pytest.raises(TypeError):
        map_in_chunks(lambda a: a, ("not a tensor",), chunk=2)


def test_mismatched_chunk_structures_raise():
    x = torch.zeros(4, 1)
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        return t if calls["n"] == 1 else {"a": t}

    with pytest.raises(TypeError):
        map_in_chunks(fn, x, chunk=2)


def test_dtype_and_shape_follow_the_input():
    chain = _arm6(dtype=torch.float64)
    for dtype in (torch.float32, torch.float64):
        q = torch.zeros(5, chain.dof, dtype=dtype)
        out = map_in_chunks(lambda t: forward_kinematics(chain, t), q, chunk=2)
        assert out.dtype == dtype


# --------------------------------------------------------------- autograd paths

def test_no_grad_path_builds_no_graph():
    chain = _arm6()
    q = torch.zeros(6, chain.dof, dtype=torch.float64, requires_grad=True)
    out = map_in_chunks(lambda t: forward_kinematics(chain, t), q, 2, no_grad=True)
    assert not out.requires_grad
    assert out.grad_fn is None
    with pytest.raises(RuntimeError):
        out.sum().backward()
    assert q.grad is None
    # the ambient grad mode is restored afterwards
    assert torch.is_grad_enabled()


def test_grad_path_keeps_the_graph():
    chain = _arm6()
    q = torch.zeros(6, chain.dof, dtype=torch.float64, requires_grad=True)
    out = map_in_chunks(lambda t: forward_kinematics(chain, t), q, 2)
    assert out.requires_grad and out.grad_fn is not None
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_gradient_matches_central_differences():
    """Independent oracle: float64 central differences across chunk boundaries."""
    chain = _p2r()
    ee = chain.link_index["ee"]

    def loss_of(qv):
        p = forward_kinematics(chain, qv)[:, ee, :3, 3]
        # a per-row weighting so every row contributes a distinct gradient
        w = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        return (p * w).sum()

    torch.manual_seed(3)
    q0 = (torch.rand(5, chain.dof, dtype=torch.float64) - 0.5) * 2.0

    q = q0.clone().requires_grad_(True)
    out = map_in_chunks(lambda t: forward_kinematics(chain, t)[:, ee, :3, 3],
                        q, chunk=2)
    w = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    (out * w).sum().backward()
    got = q.grad.clone()

    h = 1e-6
    fd = torch.zeros_like(q0)
    for i in range(q0.shape[0]):
        for j in range(q0.shape[1]):
            qp, qm = q0.clone(), q0.clone()
            qp[i, j] += h
            qm[i, j] -= h
            fd[i, j] = (loss_of(qp) - loss_of(qm)) / (2 * h)
    assert torch.allclose(got, fd, atol=1e-8), (got - fd).abs().max()


def test_no_grad_passthrough_does_not_leak_a_cat_node():
    """Regression: the stitching must happen under no_grad too.

    A function that hands back one of its inputs untouched used to come out of
    the chunked call wearing a CatBackward node, because the concatenation ran
    after the no_grad block had closed. Slicing a leaf under no_grad still
    yields a view that requires grad, so the later cat was recorded.
    """
    x = torch.ones(4, 1, dtype=torch.float64, requires_grad=True)
    out = map_in_chunks(lambda t: t, x, 2, no_grad=True)
    assert out.grad_fn is None
    assert not out.requires_grad
    q, info = map_in_chunks(lambda t: (t, {"e": t * 3.0}), x, 3, no_grad=True)
    assert q.grad_fn is None and info["e"].grad_fn is None


def test_no_grad_inside_an_enabled_context_only_affects_the_call():
    x = torch.ones(4, 1, requires_grad=True)
    out = map_in_chunks(lambda t: t * 2.0, x, 2, no_grad=True)
    assert not out.requires_grad
    again = map_in_chunks(lambda t: t * 2.0, x, 2)
    assert again.requires_grad


# ------------------------------------------------------------------ ik_chunked

def _reachable_targets(chain, n, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    lo = chain.lower.to(dtype)
    hi = chain.upper.to(dtype)
    q = lo + (hi - lo) * torch.rand(n, chain.dof, generator=g, dtype=dtype)
    return forward_kinematics(chain, q)[:, chain.link_index["ee"]].contiguous()


def test_ik_chunked_identical_to_unchunked_with_q0():
    chain = _arm6()
    ee = chain.link_index["ee"]
    target = _reachable_targets(chain, 9)
    torch.manual_seed(1)
    q0 = torch.rand(9, chain.dof, dtype=torch.float64) - 0.5
    kw = dict(link_index=ee, iters=30, pos_only=True, check_every=NO_EARLY_EXIT)

    q_ref, info_ref = ik(chain, target, q0=q0, **kw)
    for chunk in (1, 2, 4, 9, 50):
        q, info = ik_chunked(chain, target, chunk, q0=q0, **kw)
        assert q.shape == q_ref.shape
        assert torch.allclose(q, q_ref, rtol=0, atol=1e-12), f"chunk={chunk}"
        assert torch.allclose(info["final_error"], info_ref["final_error"],
                              rtol=0, atol=1e-12)
        assert info["iters"] == info_ref["iters"]


def test_ik_chunked_matches_unchunked_random_seeds():
    """No q0: the seeds are drawn once for the batch, so chunking is invisible."""
    chain = _arm6()
    ee = chain.link_index["ee"]
    target = _reachable_targets(chain, 7, seed=5)
    kw = dict(link_index=ee, iters=25, pos_only=True, check_every=NO_EARLY_EXIT)

    torch.manual_seed(11)
    q_ref, _ = ik(chain, target, **kw)
    for chunk in (2, 3, 7):
        torch.manual_seed(11)
        q, _ = ik_chunked(chain, target, chunk, **kw)
        assert torch.allclose(q, q_ref, rtol=0, atol=1e-12), f"chunk={chunk}"


def test_ik_chunked_actually_solves():
    """Oracle: push the answer back through FK and measure the real error."""
    chain = _arm6()
    ee = chain.link_index["ee"]
    target = _reachable_targets(chain, 12, seed=2)
    torch.manual_seed(4)
    q, info = ik_chunked(chain, target, chunk=5, link_index=ee, iters=400,
                         pos_only=True, restarts=4, tol=1e-6)
    p = forward_kinematics(chain, q)[:, ee, :3, 3]
    err = (p - target[:, :3, 3]).norm(dim=-1)
    assert err.shape == (12,)
    assert err.median().item() < 1e-3, err
    assert info["final_error"].shape == (12,)


def test_ik_chunked_restarts_shape_and_info():
    chain = _arm6()
    ee = chain.link_index["ee"]
    target = _reachable_targets(chain, 6, seed=8)
    torch.manual_seed(0)
    q, info = ik_chunked(chain, target, chunk=4, link_index=ee, iters=20,
                         pos_only=True, restarts=3)
    assert q.shape == (6, chain.dof)
    assert info["final_error"].shape == (6,)
    assert info["restarts"] == 3


def test_ik_chunked_no_grad():
    chain = _arm6()
    ee = chain.link_index["ee"]
    target = _reachable_targets(chain, 8, seed=3).requires_grad_(True)
    q, info = ik_chunked(chain, target, 3, no_grad=True, link_index=ee,
                         iters=15, pos_only=True)
    assert not q.requires_grad and q.grad_fn is None
    assert target.grad is None


def test_ik_chunked_is_differentiable_in_the_target():
    chain = _arm6()
    ee = chain.link_index["ee"]
    target = _reachable_targets(chain, 6, seed=6).clone().requires_grad_(True)
    torch.manual_seed(2)
    q0 = torch.zeros(6, chain.dof, dtype=torch.float64)
    q, _ = ik_chunked(chain, target, 2, q0=q0, link_index=ee, iters=10,
                      pos_only=True, check_every=NO_EARLY_EXIT)
    assert q.requires_grad and q.grad_fn is not None
    q.sum().backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert target.grad.abs().sum().item() > 0.0


def test_ik_chunked_working_dtype_follows_the_caller():
    """The chain is float64; a float32 target gives a float32 solve."""
    chain = _arm6(dtype=torch.float64)
    ee = chain.link_index["ee"]
    target64 = _reachable_targets(chain, 5, seed=7)
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(0)
        q, info = ik_chunked(chain, target64.to(dtype), 2, link_index=ee,
                             iters=20, pos_only=True)
        assert q.dtype == dtype
        assert info["final_error"].dtype == dtype
        assert chain.dtype == torch.float64      # the chain is left alone


def test_ik_chunked_planar_hits_a_hand_placed_target():
    """Unit-link planar 2R: (1, 1) is reached exactly at q = (0, pi/2)."""
    chain = _p2r()
    ee = chain.link_index["ee"]
    target = torch.eye(4, dtype=torch.float64).repeat(4, 1, 1)
    target[:, 0, 3] = 1.0
    target[:, 1, 3] = 1.0
    q0 = torch.tensor([[0.3, 1.0], [-0.3, 1.2], [0.1, 1.8], [0.0, 0.9]],
                      dtype=torch.float64)
    q, _ = ik_chunked(chain, target, chunk=3, q0=q0, link_index=ee, iters=300,
                      pos_only=True, tol=1e-10, check_every=NO_EARLY_EXIT)
    p = forward_kinematics(chain, q)[:, ee, :3, 3]
    assert torch.allclose(p[:, 0], torch.ones(4, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(p[:, 1], torch.ones(4, dtype=torch.float64), atol=1e-6)
    # elbow-up solution from these seeds: q2 = +pi/2, q1 = 0
    assert torch.allclose(q[:, 1], torch.full((4,), math.pi / 2,
                                              dtype=torch.float64), atol=1e-5)
    assert torch.allclose(q[:, 0], torch.zeros(4, dtype=torch.float64), atol=1e-5)
