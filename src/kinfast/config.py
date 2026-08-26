# src/kinfast/config.py
"""Configuration-space utilities: metrics, interpolation, limits, sampling.

Planners and learning code spend most of their time asking three questions
about joint vectors: how far apart are two configurations, what lies between
them, and is this one legal. Doing that naively goes wrong on revolving
joints, where 3.1 rad and -3.1 rad are 0.083 rad apart but subtract to 6.2,
so a straight line between them spins the joint almost all the way around
the wrong way.

Everything here follows the rest of the library: a leading batch dimension is
optional and broadcast, the working dtype and device come from the q you pass
in (the chain's constants are cast to match), and the metric functions are
differentiable so you can put them inside a loss.

  continuous_mask   which dofs revolve, inferred from the compiled chain
  wrap_angle        fold an angle into [-pi, pi)
  difference        signed shortest joint-space displacement q_b - q_a
  distance          norm of that displacement (weights and p optional)
  interpolate       point at fraction s along the shortest arc
  clamp_to_limits   nearest legal configuration
  is_within_limits  legality test, wrap aware
  sample            low-discrepancy (Sobol) configurations inside the limits

Note on "continuous": a CompiledChain stores one revolute type code, so a
URDF <joint type="continuous"> and a wide <joint type="revolute"> look the
same by the time they reach here. We call a rotary dof continuous when its
range covers a whole turn or more (or is not finite), which is exactly the
condition under which wrapping is sound: every angle is then reachable, so
folding by a full turn never leaves the legal set. Pass `continuous=` to any
of these functions to override the inference with your own (dof,) bool mask.
"""
import math
import torch

from kinfast.analysis import sampling_bounds

TAU = 2.0 * math.pi

# A range within a hair of a full turn counts as a full turn: limits like
# (-pi, pi) come out of the repair pass as exact doubles, and we do not want
# a rounding crumb to decide whether a joint wraps.
_SPAN_EPS = 1e-9


def _check_q(chain, q, name="q"):
    """Common shape check: (..., dof), any number of leading batch dims."""
    if not torch.is_tensor(q):
        raise TypeError(f"{name} must be a tensor, got {type(q).__name__}")
    if q.ndim == 0 or q.shape[-1] != chain.dof:
        raise ValueError(
            f"{name} must have shape (..., {chain.dof}) for this chain, "
            f"got {tuple(q.shape)}")
    return q


def _limits(chain, like):
    """Joint limits as (lower, upper) on the device and dtype of `like`."""
    lo = chain.lower.to(device=like.device, dtype=like.dtype)
    hi = chain.upper.to(device=like.device, dtype=like.dtype)
    return lo, hi


def continuous_mask(chain, device=None):
    """(dof,) bool: which dofs may be compared and interpolated modulo a turn.

    True for a rotary joint whose limit range spans at least 2*pi, or whose
    limits are not finite. Prismatic joints are never wrapped: a metre is a
    metre and there is no equivalent second position.
    """
    lo, hi = chain.lower, chain.upper
    span = hi - lo
    wide = ~torch.isfinite(span) | (span >= TAU - _SPAN_EPS)
    q_index = chain.q_index.to(lo.device)
    jtype = chain.joint_type.to(lo.device)
    movable = q_index >= 0
    type_per_dof = torch.zeros(chain.dof, dtype=torch.long, device=lo.device)
    type_per_dof[q_index[movable]] = jtype[movable]
    mask = wide & (type_per_dof == 1)
    return mask.to(device) if device is not None else mask


def _mask(chain, continuous, like):
    """Resolve the `continuous=` argument to a (dof,) bool tensor on `like`."""
    if continuous is None:
        return continuous_mask(chain, device=like.device)
    m = torch.as_tensor(continuous, device=like.device)
    if m.dtype != torch.bool:
        m = m.to(torch.bool)
    if m.shape != (chain.dof,):
        raise ValueError(
            f"continuous must be a (dof,) = ({chain.dof},) bool mask, "
            f"got {tuple(m.shape)}")
    return m


def wrap_angle(x):
    """Fold angles into [-pi, pi), keeping the gradient (it is 1 almost
    everywhere: wrapping only ever subtracts a constant number of turns)."""
    return torch.remainder(x + math.pi, TAU) - math.pi


def difference(chain, q_a, q_b, continuous=None):
    """Signed shortest displacement from q_a to q_b, shape (..., dof).

    On a wrapping dof the result is folded into [-pi, pi), so it always
    describes the short way around: 3.1 to -3.1 gives +0.083, not -6.2.
    Adding the result to q_a lands on a configuration equivalent to q_b (equal
    to it exactly on non-wrapping dofs, equal up to whole turns otherwise).
    """
    q_a = _check_q(chain, q_a, "q_a")
    q_b = _check_q(chain, q_b, "q_b")
    d = q_b - q_a
    m = _mask(chain, continuous, d)
    return torch.where(m, wrap_angle(d), d)


def distance(chain, q_a, q_b, weights=None, p: float = 2.0, continuous=None):
    """Configuration-space distance between q_a and q_b, shape (...).

    Wrap aware, so a revolving joint takes the short way round. Default is the
    unweighted 2-norm of the joint displacement. Pass `weights` (a (dof,)
    tensor or sequence) to scale each joint before the norm: radians and
    metres are not comparable, so a chain that mixes revolute and prismatic
    joints usually wants a weight that converts one into the other. `p`
    selects the norm (1, 2, or inf).

    Differentiable in both arguments. At q_a == q_b the 2-norm has a corner;
    the value there is exactly zero and the gradient is reported as zero
    rather than the NaN a plain sqrt would hand back.
    """
    d = difference(chain, q_a, q_b, continuous=continuous)
    if weights is not None:
        w = torch.as_tensor(weights, dtype=d.dtype, device=d.device)
        if w.shape != (chain.dof,):
            raise ValueError(
                f"weights must have shape (dof,) = ({chain.dof},), "
                f"got {tuple(w.shape)}")
        if bool((w < 0).any()):
            raise ValueError("weights must be non-negative")
        d = d * w
    if d.shape[-1] == 0:
        return torch.zeros(d.shape[:-1], dtype=d.dtype, device=d.device)
    if p == 2.0:
        # sqrt of a clamped sum of squares, then a where() to put the exact
        # zero back. The clamp keeps the backward pass off the vertical
        # tangent at 0 (plain sqrt differentiates to inf there, and
        # vector_norm to NaN); the where restores distance(q, q) == 0 exactly
        # and routes the gradient down the constant branch, so it reads 0.
        sq = (d * d).sum(dim=-1)
        r = torch.sqrt(sq.clamp_min(torch.finfo(d.dtype).tiny))
        return torch.where(sq > 0, r, torch.zeros_like(r))
    return torch.linalg.vector_norm(d, ord=p, dim=-1)


def interpolate(chain, q_a, q_b, s, continuous=None):
    """Configuration a fraction s of the way from q_a to q_b along the short arc.

    s = 0 gives q_a and s = 1 gives q_b (up to whole turns on a wrapping dof,
    where the short arc may end on an equivalent angle rather than the literal
    number you passed). s may be a float or a tensor: a tensor is broadcast
    against the batch dimensions, so a shape (S,) s with a single pair of
    endpoints of shape (dof,) sweeps out a (S, dof) path. Values outside
    [0, 1] extrapolate along the same arc.

    The path is not folded back into [-pi, pi): a joint crossing the wrap
    point walks smoothly past +pi instead of jumping to -pi, which is what
    controllers and plots want. Run the result through clamp_to_limits if you
    need it back inside the stated range.
    """
    d = difference(chain, q_a, q_b, continuous=continuous)
    s = torch.as_tensor(s, dtype=d.dtype, device=d.device)
    if s.ndim:
        s = s.unsqueeze(-1)      # line up the fraction with the batch dims
    return q_a + s * d


def clamp_to_limits(chain, q, continuous=None):
    """Nearest legal configuration, shape (..., dof).

    A wrapping dof that has left its range is folded back by the fewest whole
    turns that land it inside, which keeps the joint angle physically
    identical; every other joint is clamped to its bound. A configuration
    already inside the limits comes back untouched, and a joint whose bound is
    not finite is left alone on that side. Differentiable (gradient 1 where
    nothing was clipped, 0 where it was).
    """
    q = _check_q(chain, q)
    lo, hi = _limits(chain, q)
    m = _mask(chain, continuous, q)
    inside = (q >= lo) & (q <= hi)
    span = hi - lo
    # Only fold when a whole turn is guaranteed to land inside the window.
    foldable = m & torch.isfinite(span) & (span >= TAU - _SPAN_EPS) & ~inside
    # Whole turns to subtract: any k in [k_min, k_max] lands inside (the
    # interval is non-empty because the window is at least a turn wide), and
    # the one nearest zero moves the number the least. ceil and floor have
    # zero gradient, so the fold stays a shift by a constant.
    zero = torch.zeros((), dtype=q.dtype, device=q.device)
    lo_f = torch.where(torch.isfinite(lo), lo, zero)
    hi_f = torch.where(torch.isfinite(hi), hi, zero)
    k_min = torch.ceil((q - hi_f) / TAU)
    k_max = torch.floor((q - lo_f) / TAU)
    k = torch.minimum(torch.maximum(zero, k_min), k_max)
    folded = torch.where(foldable, q - k * TAU, q)
    out = torch.where(torch.isfinite(lo), torch.maximum(folded, lo), folded)
    return torch.where(torch.isfinite(hi), torch.minimum(out, hi), out)


def is_within_limits(chain, q, tol: float = 0.0, wrap: bool = True,
                     continuous=None, per_joint: bool = False):
    """Is every joint inside its limits? bool of shape (...) (or (..., dof)).

    `tol` widens each bound, for callers who do not want a solver's last
    1e-9 of overshoot to read as a violation. With `wrap` (the default) a
    revolving dof always passes, because a joint that can turn a whole way
    round has no illegal angle; pass wrap=False for a literal numeric test
    against the stored bounds. A bound that is not finite is not a bound and
    never fails. `per_joint` returns the per-joint answers instead of the
    and-reduction over joints.
    """
    q = _check_q(chain, q)
    lo, hi = _limits(chain, q)
    ok = ((~torch.isfinite(lo)) | (q >= lo - tol)) & \
         ((~torch.isfinite(hi)) | (q <= hi + tol))
    if wrap:
        ok = ok | _mask(chain, continuous, q)
    if per_joint:
        return ok
    if q.shape[-1] == 0:
        return torch.ones(q.shape[:-1], dtype=torch.bool, device=q.device)
    return ok.all(dim=-1)


def sample(chain, n: int, seed: int = 0, scramble: bool = True, dtype=None,
           device=None):
    """n configurations inside the joint limits, shape (n, dof).

    Uses a scrambled Sobol sequence rather than independent uniforms. For the
    same number of samples a low-discrepancy set covers the box far more
    evenly, which is what you want for workspace sweeps, IK seeding, and
    coverage tests: uniform draws clump and leave holes, and those holes are
    where the interesting configurations hide.

    Deterministic: the same seed gives the same points, on any device, because
    the sequence is generated on the CPU and then moved. Joints with infinite
    limits are handled by analysis.sampling_bounds (a revolving joint gets one
    full turn; an unbounded prismatic joint raises, since it has no range to
    sample). scramble=False gives the raw Sobol sequence, whose first point
    sits exactly on the lower corner of the box; the seed does nothing in that
    mode, since there is no scrambling left for it to drive.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"sample needs an integer n >= 1, got {n!r}")
    lo, hi = sampling_bounds(chain)
    dtype = dtype or lo.dtype
    if not dtype.is_floating_point:
        raise ValueError(f"sample needs a floating dtype, got {dtype}")
    device = device if device is not None else lo.device
    if chain.dof == 0:
        return torch.zeros(n, 0, dtype=dtype, device=device)
    engine = torch.quasirandom.SobolEngine(chain.dof, scramble=scramble,
                                           seed=seed)
    u = engine.draw(n, dtype=dtype)
    lo = lo.to(dtype=dtype).cpu()
    hi = hi.to(dtype=dtype).cpu()
    return (lo + (hi - lo) * u).to(device)
