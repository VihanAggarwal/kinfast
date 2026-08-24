# src/kinfast/batching.py
"""Run a batched function over a huge batch in bounded-size pieces.

Everything in kinfast is batched over a leading B dimension, and that is what
makes it fast: one FK sweep for ten thousand arms costs a handful of kernel
launches. The catch is memory. A 100k-target IK solve allocates every
intermediate (poses, Jacobians, the damped normal equations) for all 100k rows
at once, and on a small GPU that is the difference between a solve and an
out-of-memory error.

map_in_chunks splits the leading dimension, calls the function once per slice,
and stitches the results back together, so the peak allocation is set by the
chunk size rather than by the batch. The answer is the same either way: the
rows of a batch never interact, so slicing them is mathematically invisible.

Two honest caveats about memory:

  * Chunking bounds the peak *live* allocation only when the graph is not being
    kept. With autograd on, every chunk's activations stay alive until backward
    runs, so the peak is roughly what the unchunked call would have used. Pass
    no_grad=True when you only want the numbers, which is the usual case for
    a big offline solve.
  * The concatenated output still has to fit. Chunking helps with intermediates,
    not with the result itself.

Device and dtype are whatever the caller's tensors carry; nothing here creates
a tensor of its own except the seeds ik_chunked draws for you.
"""
import contextlib

import torch

from kinfast.compile import CompiledChain
from kinfast.ik import ik

__all__ = ["map_in_chunks", "ik_chunked"]


def _grad_mode(no_grad: bool):
    """torch.no_grad() when asked, otherwise leave the ambient mode alone."""
    return torch.no_grad() if no_grad else contextlib.nullcontext()


def _combine(parts):
    """Stitch the per-chunk return values back into one.

    Tensors with a batch dimension are concatenated along dim 0. Anything that
    is not per-row is passed through from the first chunk: 0-dim tensors and
    plain Python values (an iteration count, a flag) describe the call, not the
    rows, so concatenating them would be wrong. Tuples, lists, namedtuples and
    dicts are walked structurally, which is what lets this handle a function
    like ik that returns (q, info_dict).
    """
    first = parts[0]

    if isinstance(first, torch.Tensor):
        if first.dim() == 0:
            return first
        for p in parts[1:]:
            if not isinstance(p, torch.Tensor):
                raise TypeError(
                    "chunks returned mismatched structures: a tensor and a "
                    f"{type(p).__name__}")
        return torch.cat(parts, dim=0)

    if isinstance(first, dict):
        keys = list(first.keys())
        for p in parts[1:]:
            if not isinstance(p, dict) or list(p.keys()) != keys:
                raise TypeError(
                    "chunks returned dicts with different keys: "
                    f"{keys} vs {list(p.keys()) if isinstance(p, dict) else p!r}")
        return {k: _combine([p[k] for p in parts]) for k in keys}

    if isinstance(first, (list, tuple)):
        n = len(first)
        for p in parts[1:]:
            if not isinstance(p, (list, tuple)) or len(p) != n:
                raise TypeError(
                    "chunks returned sequences of different lengths or types")
        merged = [_combine([p[i] for p in parts]) for i in range(n)]
        if isinstance(first, tuple) and hasattr(first, "_fields"):
            return type(first)(*merged)      # namedtuple
        return type(first)(merged)

    return first


def _batch_size(tensors):
    """The shared leading dimension, or a clear error saying which one differs."""
    B = None
    for i, t in enumerate(tensors):
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"map_in_chunks expects tensors (or None), got "
                f"{type(t).__name__} at position {i}")
        if t.dim() == 0:
            raise ValueError(
                f"tensor at position {i} is 0-dim and has no batch dimension")
        if B is None:
            B = t.shape[0]
        elif t.shape[0] != B:
            raise ValueError(
                "all tensors must share the leading batch dimension, got "
                f"{B} and {t.shape[0]} at position {i}")
    if B is None:
        raise ValueError("map_in_chunks needs at least one tensor, got none")
    return B


def map_in_chunks(fn, tensors, chunk, no_grad: bool = False):
    """Apply a batched function to slices of dim 0 and concatenate the results.

    fn is called as fn(*slices), one positional argument per entry of tensors,
    in the order given. tensors may be a single tensor (fn then takes one
    argument) or a sequence of tensors that share their leading dimension. A
    None entry is handed to fn unsliced, which keeps optional arguments easy to
    pass through.

    chunk is the maximum number of rows per call. A chunk at least as large as
    the batch means exactly one call, and a batch that is not a multiple of
    chunk simply ends with a shorter slice. chunk=None also means one call.

    no_grad=True runs the calls inside torch.no_grad(), so no graph is built
    and the returned tensors carry no grad_fn. Leave it False to stay
    differentiable: gradients then flow back through every chunk exactly as
    they would through one unchunked call.

    Returns whatever fn returns, with per-row tensors concatenated. See
    _combine for how nested tuples, dicts and non-batched values are treated.
    """
    if isinstance(tensors, torch.Tensor) or tensors is None:
        tensors = (tensors,)
    else:
        tensors = tuple(tensors)

    B = _batch_size(tensors)

    if chunk is None:
        chunk = B if B > 0 else 1
    if isinstance(chunk, bool) or not isinstance(chunk, int):
        raise TypeError(f"chunk must be a positive integer or None, got {chunk!r}")
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")

    with _grad_mode(no_grad):
        if B == 0:
            # Nothing to concatenate, but the caller still deserves a result
            # with the right structure, dtype and trailing shape. One call on
            # the empty input gives exactly that.
            return fn(*tensors)
        if chunk >= B:
            return fn(*tensors)
        parts = []
        for start in range(0, B, chunk):
            stop = min(start + chunk, B)
            sliced = tuple(None if t is None else t[start:stop] for t in tensors)
            parts.append(fn(*sliced))
    return _combine(parts)


def ik_chunked(chain: CompiledChain, target: torch.Tensor, chunk,
               no_grad: bool = False, **ik_kwargs):
    """Solve IK for a large batch of targets a chunk at a time.

    Same arguments as kinfast.ik.ik plus the chunk size, same return value
    (q, info). The targets do not interact, so this is the identical solve with
    a smaller memory high-water mark. Pass no_grad=True for an offline solve
    where you want the numbers and not the graph.

    On reproducibility: ik draws random seeds when you do not give it q0. With
    restarts<=1 this function draws the whole batch's seeds up front, using the
    same formula ik would, so a chunked solve matches an unchunked one row for
    row under the same torch.manual_seed. With restarts>1 the seeds are drawn
    inside each chunk and the random draws land differently, so the results are
    statistically the same but not bitwise identical. Pass q0 (or accept the
    difference) if that matters.

    One more source of difference: ik's early exit fires when every row in the
    call is inside tol, so a chunk of easy targets can stop sooner than the
    whole batch would have. Both answers are inside tol, they just took a
    different number of iterations. Pass check_every larger than iters to
    disable the early exit and get identical iteration counts.
    """
    q0 = ik_kwargs.pop("q0", None)
    restarts = ik_kwargs.get("restarts", 1)

    if q0 is None and restarts <= 1:
        # Mirror ik's own seed draw for the full batch, then hand each chunk
        # its slice. One draw for the whole batch is what makes the chunked
        # answer identical to the unchunked one.
        device, dtype = target.device, target.dtype
        lo = chain.lower.to(device=device, dtype=dtype)
        hi = chain.upper.to(device=device, dtype=dtype)
        q0 = lo + (hi - lo) * torch.rand(target.shape[0], chain.dof,
                                         dtype=dtype, device=device)

    def _solve(tgt, seed):
        return ik(chain, tgt, q0=seed, **ik_kwargs)

    return map_in_chunks(_solve, (target, q0), chunk, no_grad=no_grad)
