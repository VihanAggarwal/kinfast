# src/kinfast/acm.py
"""Allowed-collision matrix (ACM) for sphere collision models.

A sphere collision model of a real robot has many pairs that are pointless to
check. Two spheres on the same link never move relative to each other, spheres
on a link and its parent usually overlap by construction (that is how you cover
a joint without leaving a gap), and spheres at opposite ends of a long arm may
be unable to touch no matter what the joints do. Checking all of them is both
slow and wrong: the permanent overlaps report a self-collision at every
configuration, so a planner that trusts the raw distance never finds a valid
motion.

The usual fix, in MoveIt and in cuRobo alike, is to sample a lot of
configurations once, offline, and sort every pair into three buckets:

  always in contact   overlapping in every sample. Built that way on purpose,
                      so the pair is *allowed* and must be ignored at runtime.
  never in contact    clear in every sample, with room to spare. Unreachable
                      by the kinematics, so it can be pruned.
  needs checking      in contact for some configurations and not others. This
                      is the only bucket a runtime collision query has to look
                      at, and it is usually a small fraction of the total.

`allowed_pairs` does that classification, and `self_distance_masked` evaluates
the self-collision distance over just the surviving pairs. The classification
is a fixed piece of bookkeeping, so it is computed under no_grad; the masked
distance is the ordinary differentiable sphere distance and can be used inside
collision-aware IK or trajectory optimization exactly like
`kinfast.collision.self_distance`.

Sampling is an empirical procedure, not a proof. A pair labelled "never" is
only known to be clear over the configurations that were drawn, so sample
generously and use `safety` to keep near misses in the checked bucket.
"""
import math

import torch

from kinfast.analysis import sampling_bounds

__all__ = ["pair_distances", "allowed_pairs", "self_distance_masked",
           "mask_to_pairs", "pairs_to_mask", "upper_pairs"]


def upper_pairs(n: int, device=None) -> torch.Tensor:
    """All unordered index pairs (i, j) with i < j, as a (n*(n-1)/2, 2) long
    tensor. Sphere pairs are symmetric, so everything here works on the strict
    upper triangle and never double counts a pair."""
    idx = torch.triu_indices(n, n, offset=1, device=device)
    return idx.t().contiguous()


def pairs_to_mask(pairs: torch.Tensor, n: int) -> torch.Tensor:
    """(K, 2) index pairs -> symmetric (n, n) bool mask with a False diagonal."""
    pairs = torch.as_tensor(pairs, dtype=torch.long)
    mask = torch.zeros(n, n, dtype=torch.bool, device=pairs.device)
    if pairs.numel():
        if pairs.dim() != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must have shape (K, 2), got {tuple(pairs.shape)}")
        i, j = pairs[:, 0], pairs[:, 1]
        if int(i.min()) < 0 or int(i.max()) >= n or int(j.min()) < 0 or int(j.max()) >= n:
            raise ValueError(f"pair indices must be in [0, {n})")
        mask[i, j] = True
        mask[j, i] = True
    mask.fill_diagonal_(False)
    return mask


def mask_to_pairs(mask: torch.Tensor) -> torch.Tensor:
    """Symmetric (n, n) bool mask -> (K, 2) long tensor of pairs with i < j.

    A pair set either side of the diagonal counts once: the mask is read as
    `mask | mask.T` so a caller may fill in only one triangle."""
    mask = torch.as_tensor(mask)
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be a bool tensor, got {mask.dtype}")
    if mask.dim() != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"mask must be square (S, S), got {tuple(mask.shape)}")
    sym = mask | mask.t()
    sym = torch.triu(sym, diagonal=1)
    return sym.nonzero(as_tuple=False)


def _resolve_pairs(model, mask, device):
    """Accept either a (S, S) bool mask or a (K, 2) index tensor and return the
    (K, 2) long index tensor on `device`. None means every distinct pair."""
    S = model.n
    if mask is None:
        return upper_pairs(S, device=device)
    mask = torch.as_tensor(mask)
    if mask.dtype == torch.bool:
        if mask.shape != (S, S):
            raise ValueError(
                f"mask must have shape ({S}, {S}) for a model with {S} spheres, "
                f"got {tuple(mask.shape)}")
        pairs = mask_to_pairs(mask)
    else:
        if mask.dim() != 2 or mask.shape[1] != 2:
            raise ValueError(
                "mask must be a (S, S) bool tensor or a (K, 2) long tensor of "
                f"sphere index pairs, got shape {tuple(mask.shape)} of "
                f"dtype {mask.dtype}")
        pairs = mask.long()
        if pairs.numel():
            if int(pairs.min()) < 0 or int(pairs.max()) >= S:
                raise ValueError(f"sphere indices must be in [0, {S})")
            if bool((pairs[:, 0] == pairs[:, 1]).any()):
                raise ValueError("a sphere cannot be paired with itself")
    return pairs.to(device)


def pair_distances(model, q: torch.Tensor, pairs=None) -> torch.Tensor:
    """Signed sphere-to-sphere distances for a batch of configurations.

    With `pairs=None` this returns the full (B, S, S) matrix; with a (K, 2)
    long tensor of sphere index pairs it returns just those, as (B, K), which
    is what you want once an ACM has thinned the pair set down. The sign
    convention matches `kinfast.collision`: ||c_i - c_j|| - r_i - r_j, negative
    meaning the two spheres interpenetrate.

    The pair form is the cheap one. The full matrix costs B*S^2 distances and
    computes every pair twice (plus the useless diagonal), while a thinned pair
    list is typically a small fraction of S^2 on a real arm, so the runtime
    query should always be given explicit pairs.

    Differentiable in q, and the working dtype and device follow q.
    """
    C = model.centers_world(q)                                   # (B, S, 3)
    r = model.radius.to(device=q.device, dtype=q.dtype)           # (S,)
    if pairs is None:
        d = (C[:, :, None, :] - C[:, None, :, :]).norm(dim=-1)    # (B, S, S)
        return d - r[None, :, None] - r[None, None, :]
    pairs = torch.as_tensor(pairs, dtype=torch.long, device=C.device)
    i, j = pairs[:, 0], pairs[:, 1]
    d = (C[:, i, :] - C[:, j, :]).norm(dim=-1)                    # (B, K)
    return d - r[i][None, :] - r[j][None, :]


def self_distance_masked(model, q: torch.Tensor, mask=None) -> torch.Tensor:
    """Minimum signed self-collision distance over the enabled pairs. -> (B,).

    `mask` is either a (S, S) bool matrix (True where the pair should be
    checked, read symmetrically) or a (K, 2) long tensor of sphere index pairs,
    which is what `allowed_pairs` hands back. None checks every distinct pair.

    Unlike `kinfast.collision.self_distance` this applies no structural rule of
    its own: it does not skip same-link or parent/child pairs, because with an
    ACM those decisions are already baked into the mask and were made from the
    model's real geometry rather than from an adjacency guess.

    Negative means self-collision. If the mask enables no pair at all the
    result is +inf, the identity of a minimum, which keeps a fully allowed
    model from reading as a collision.

    Differentiable in q, batched, and dtype and device follow q.
    """
    pairs = _resolve_pairs(model, mask, q.device)
    if pairs.numel() == 0:
        return torch.full((q.shape[0],), float("inf"),
                          dtype=q.dtype, device=q.device)
    return pair_distances(model, q, pairs).min(dim=1).values


def _sample_configs(chain, n, seed, dtype, device):
    """Uniform configurations inside the joint limits, drawn from a seeded
    CPU generator so a classification is reproducible on any device."""
    lo, hi = sampling_bounds(chain)
    lo = lo.to(device="cpu", dtype=dtype)
    hi = hi.to(device="cpu", dtype=dtype)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    u = torch.rand(n, chain.dof, generator=g, dtype=dtype)
    return (lo + (hi - lo) * u).to(device)


def allowed_pairs(model, n: int = 256, *, seed: int = 0, q=None,
                  margin: float = 0.0, safety: float = 0.0,
                  chunk: int = 64, dtype=None, device=None) -> dict:
    """Classify every sphere pair of `model` from `n` sampled configurations.

    A pair counts as in contact when its signed distance is at most `margin`
    (0 means literal overlap; a positive margin treats near misses as contact,
    which is what you want if the spheres are a loose bound). Over the samples:

      always   in contact every time. Allowed: ignore it at runtime.
      never    in contact no time, and never closer than `margin + safety`.
               Prunable. Raise `safety` to keep near misses out of this bucket.
      check    everything else, the pairs a runtime query must evaluate.

    Pass `q` (B, dof) to classify over configurations you supply, for instance
    a trajectory or a task-specific region, instead of uniform samples over the
    joint limits; `n` and `seed` are then unused. `dtype` and `device` default
    to the chain's, and if `q` is given its dtype and device win, exactly like
    the rest of the library. Every sample has to be visited, and each one costs
    an S by S distance matrix, so the sweep runs `chunk` configurations at a
    time to hold peak memory at O(chunk * S^2) instead of O(B * S^2).

    Returns a dict:

      "always", "never", "check"  (K, 2) long tensors of sphere index pairs
                                  with i < j, one row per pair
      "mask"                      (S, S) symmetric bool, True on the "check"
                                  pairs. Feed it (or "check") straight to
                                  self_distance_masked. Its off-diagonal
                                  complement is the allowed-collision matrix in
                                  the SRDF sense: every pair a runtime query may
                                  skip, whether because it always touches or
                                  because it never can.
      "link_mask"                 (n_links, n_links) symmetric bool, True where
                                  some checked sphere pair spans those two
                                  links, so this is the sphere verdict collapsed
                                  to the link level. Its complement is the pair
                                  list an SRDF would disable.
      "min_distance",             (S, S) symmetric float, the extreme signed
      "max_distance"              distances seen, diagonal zero. Use these to
                                  judge how safe a "never" verdict is.
      "contact_fraction"          (S, S) symmetric float, fraction of samples
                                  in contact. 0 and 1 are the pruned buckets.
      "q"                         the configurations used, so the verdict can
                                  be reproduced or extended
      "n_samples", "margin", "safety"   echoed back for the record

    The classification itself is bookkeeping, so it runs under no_grad.
    """
    chain = model.chain
    S = model.n
    if S == 0:
        raise ValueError("the sphere model carries no spheres, so there is "
                         "nothing to classify")
    margin, safety = float(margin), float(safety)
    if not math.isfinite(margin):
        raise ValueError(f"margin must be finite, got {margin}")
    if not math.isfinite(safety) or safety < 0.0:
        raise ValueError(
            f"safety must be a finite non-negative distance, got {safety}")
    if q is None:
        n = int(n)
        if n <= 0:
            raise ValueError(f"n must be a positive number of samples, got {n}")
        dtype = chain.joint_origin.dtype if dtype is None else dtype
        device = chain.joint_origin.device if device is None else device
        q = _sample_configs(chain, n, seed, dtype, device)
    else:
        q = torch.as_tensor(q)
        if q.dim() != 2 or q.shape[1] != chain.dof:
            raise ValueError(
                f"q must have shape (B, {chain.dof}), got {tuple(q.shape)}")
        if q.shape[0] == 0:
            raise ValueError("q must contain at least one configuration")
        if not q.is_floating_point():
            raise TypeError(
                f"q must be a floating point tensor, got {q.dtype}")
        # the verdict is bookkeeping, so drop any autograd history rather than
        # keeping the caller's graph alive inside the returned dict
        q = q.detach()
        if dtype is not None or device is not None:
            q = q.to(device=device, dtype=dtype)
    if int(chunk) <= 0:
        raise ValueError(f"chunk must be a positive batch size, got {chunk}")

    B = q.shape[0]
    dev, dt = q.device, q.dtype
    inf = torch.full((S, S), float("inf"), dtype=dt, device=dev)
    lo = inf.clone()                       # running min signed distance
    hi = -inf                              # running max signed distance
    hits = torch.zeros(S, S, dtype=torch.long, device=dev)
    with torch.no_grad():
        for start in range(0, B, int(chunk)):
            sd = pair_distances(model, q[start:start + int(chunk)])  # (b, S, S)
            lo = torch.minimum(lo, sd.amin(dim=0))
            hi = torch.maximum(hi, sd.amax(dim=0))
            hits += (sd <= margin).sum(dim=0)

        upper = torch.triu(torch.ones(S, S, dtype=torch.bool, device=dev),
                           diagonal=1)
        always = upper & (hits == B)
        never = upper & (hits == 0) & (lo > margin + safety)
        check = upper & ~always & ~never

        mask = check | check.t()
        # link-level collapse: a link pair matters if any of its sphere pairs does
        link = model.link.to(dev)
        n_links = chain.n_links
        link_mask = torch.zeros(n_links, n_links, dtype=torch.bool, device=dev)
        cp = check.nonzero(as_tuple=False)
        if cp.numel():
            li, lj = link[cp[:, 0]], link[cp[:, 1]]
            link_mask[li, lj] = True
            link_mask[lj, li] = True

        # the per-pair statistics come from a symmetric (S, S) matrix already;
        # the diagonal is a sphere against itself, so report it as zero
        lo.fill_diagonal_(0.0)
        hi.fill_diagonal_(0.0)
        frac = hits.to(dt) / float(B)
        frac.fill_diagonal_(0.0)

    return {
        "always": always.nonzero(as_tuple=False),
        "never": never.nonzero(as_tuple=False),
        "check": check.nonzero(as_tuple=False),
        "mask": mask,
        "link_mask": link_mask,
        "min_distance": lo,
        "max_distance": hi,
        "contact_fraction": frac,
        "q": q,
        "n_samples": B,
        "margin": margin,
        "safety": safety,
    }
