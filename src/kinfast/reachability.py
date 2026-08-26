# src/kinfast/reachability.py
"""Voxelized reachability maps.

A reachability map answers the question a mechanical engineer actually asks
about an arm: "can it get its wrist to this point, and how well does it move
once it is there?" We answer it the way every practical tool does, by Monte
Carlo: draw a large batch of joint configurations, run forward kinematics on
all of them at once, and drop the resulting end-effector positions into a
regular 3D grid. Each voxel then carries three numbers:

  counts          how many sampled configurations landed in that voxel. A voxel
                  with zero hits is (probably) unreachable, one with many hits
                  is reachable from many different arm postures.
  density         counts divided by the number of binned samples, so the whole
                  array sums to 1. It is the sampling measure of the workspace,
                  not a physical density: it is high where many configurations
                  crowd into a small region (near the shoulder, near folded
                  postures) and low near the reach boundary.
  manipulability  the mean Yoshikawa measure over the samples in the voxel. This
                  is the interesting one. Two voxels can both be reachable while
                  one of them can only be reached in a stretched-out, nearly
                  singular posture; the mean manipulability separates them.

Everything is batched, follows the caller's dtype and device (the working dtype
comes from q, exactly as in fk/ik/dynamics), and the manipulability accumulation
is differentiable with respect to the sampled configurations. The binning itself
is a hard assignment, so counts and density are integers by nature and carry no
gradient; that is a property of voxel grids, not an oversight.

Public surface:

  reachability_map(chain, ...) -> ReachabilityMap
  ReachabilityMap.voxel_of(points)      point -> integer grid index
  ReachabilityMap.is_reachable(points)  batched occupancy test
  ReachabilityMap.query(points)         occupancy + the best configuration found
"""
from dataclasses import dataclass
import torch

from kinfast.analysis import _check_rows, sampling_bounds
from kinfast.fk import forward_kinematics, fk_rp
from kinfast.ik import ik as _ik
from kinfast.jacobian import _resolve_link, jacobian_rp


def _as_vec3(value, name, dtype, device):
    """Broadcast a scalar or 3-sequence into a (3,) tensor of the working dtype."""
    t = torch.as_tensor(value, dtype=dtype, device=device)
    if t.ndim == 0:
        t = t.expand(3)
    t = t.reshape(-1)
    if t.numel() != 3:
        raise ValueError(f"{name} must be a scalar or 3 numbers, got {value!r}")
    return t.contiguous()


def _check_voxel(voxel, dtype, device):
    v = _as_vec3(voxel, "voxel", dtype, device)
    if not bool(torch.isfinite(v).all()) or bool((v <= 0).any()):
        raise ValueError(
            f"voxel size must be finite and positive in every axis, got {voxel!r}")
    return v


def _points_and_manipulability(chain, q, link_index, rows, want_w):
    """End-effector positions and (optionally) Yoshikawa manipulability for one
    batch of configurations, sharing a single forward-kinematics pass.

    analysis.manipulability would run its own FK behind the Jacobian, which
    doubles the cost of the sweep that dominates map building. The measure is the
    same one: sqrt(det(J J^T)) over the selected task rows, defaulting to the
    three linear rows.
    """
    rp = fk_rp(chain, q)
    _wR, wp = rp
    pts = wp[link_index]
    if not want_w:
        return pts, None
    J = jacobian_rp(chain, q, link_index, rp=rp)
    J = J[:, rows, :] if rows is not None else J[:, :3, :]
    JJt = J @ J.transpose(-1, -2)
    return pts, torch.sqrt(torch.linalg.det(JJt).clamp_min(0.0))


@dataclass
class ReachabilityMap:
    """A regular 3D grid over end-effector positions plus the samples behind it.

    The grid covers the axis-aligned box [origin, origin + shape * voxel). Index
    (i, j, k) holds the voxel whose lower corner is origin + (i, j, k) * voxel,
    so voxel_of() is a plain floor division and there is nothing clever to get
    wrong when you compare two maps or plot one.

    counts/density/manipulability are all shaped (nx, ny, nz). The samples that
    built the map (q, points, w) are kept by default because the useful queries
    need them: "which configuration put the tool here" cannot be answered by the
    grid alone.
    """
    chain: object
    link_index: int
    origin: torch.Tensor          # (3,) lower corner of the grid
    voxel: torch.Tensor           # (3,) edge lengths of one voxel
    shape: tuple                  # (nx, ny, nz)
    counts: torch.Tensor          # (nx, ny, nz) long
    density: torch.Tensor         # (nx, ny, nz) counts / n_binned
    manipulability: torch.Tensor  # (nx, ny, nz) mean w, 0 in empty voxels
    n_samples: int                # configurations drawn
    n_binned: int                 # configurations that landed inside the grid
    n_outside: int                # configurations outside the grid (explicit bounds)
    q: torch.Tensor = None        # (N, dof) the sampled configurations
    points: torch.Tensor = None   # (N, 3) their end-effector positions
    w: torch.Tensor = None        # (N,) their manipulability, or None
    rows: tuple = None            # Jacobian rows used for w, None = default
    seed: int = None

    # ---- geometry -------------------------------------------------------
    @property
    def n_voxels(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def voxel_volume(self):
        """Volume of a single voxel, as a 0-dim tensor of the working dtype."""
        return self.voxel.prod()

    @property
    def reachable_volume(self):
        """Occupied voxel count times voxel volume: a coarse workspace volume.

        It is an over-estimate near the boundary (a voxel counts in full even if
        one corner is reachable) and an under-estimate wherever the sampling was
        too thin to hit a genuinely reachable voxel. Shrink the voxel and raise
        n to squeeze it from both sides.
        """
        occupied = (self.counts > 0).sum().to(self.voxel.dtype)
        return occupied * self.voxel_volume

    @property
    def bounds(self):
        """(lower, upper) corners of the grid box as two (3,) tensors."""
        extent = self.voxel * torch.tensor(
            self.shape, dtype=self.voxel.dtype, device=self.voxel.device)
        return self.origin, self.origin + extent

    def voxel_centers(self):
        """Center of every voxel: (nx, ny, nz, 3). Handy for plotting or for
        evaluating an analytic workspace on the same grid the map uses."""
        axes = []
        for a in range(3):
            i = torch.arange(self.shape[a], dtype=self.voxel.dtype,
                             device=self.voxel.device)
            axes.append(self.origin[a] + (i + 0.5) * self.voxel[a])
        gx, gy, gz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1)

    def voxel_of(self, points):
        """Grid index of each point: (..., 3) long, plus an in-bounds mask.

        Returns (idx, inside). idx is clamped into range so it is always safe to
        index with; trust `inside` for whether the point was actually on the
        grid. Accepts a single (3,) point or any batch shape (..., 3).
        """
        p = torch.as_tensor(points)
        if p.ndim == 0 or p.shape[-1] != 3:
            raise ValueError(f"points must have a trailing dim of 3, got {tuple(p.shape)}")
        if not p.is_floating_point():
            p = p.to(self.voxel.dtype)
        origin = self.origin.to(device=p.device, dtype=p.dtype)
        voxel = self.voxel.to(device=p.device, dtype=p.dtype)
        raw = torch.floor((p - origin) / voxel).to(torch.long)
        shape = torch.tensor(self.shape, dtype=torch.long, device=p.device)
        inside = ((raw >= 0) & (raw < shape)).all(dim=-1)
        idx = raw.clamp(torch.zeros_like(shape), shape - 1)
        return idx, inside

    def _gather(self, grid, points):
        """Look grid values up at world points; out-of-grid points read as 0."""
        idx, inside = self.voxel_of(points)
        flat = grid.to(idx.device).reshape(-1)
        lin = (idx[..., 0] * self.shape[1] + idx[..., 1]) * self.shape[2] + idx[..., 2]
        vals = flat[lin.reshape(-1)].reshape(lin.shape)
        return torch.where(inside, vals, torch.zeros_like(vals)), inside

    # ---- queries --------------------------------------------------------
    def is_reachable(self, points):
        """Batched occupancy test: (..., 3) points -> (...) bool.

        "Reachable" here means "at least one sampled configuration landed in the
        same voxel". With a finite sample that is a lower bound on the true
        reachable set: a thin sliver of workspace can be missed, and a voxel that
        straddles the boundary reads as reachable even though part of it is not.
        Use enough samples that the voxels you care about are hit many times.
        """
        counts, _ = self._gather(self.counts, torch.as_tensor(points))
        return counts > 0

    def query(self, points, refine=True, iters=100, damping=0.02, step=0.6,
              tol=1e-4):
        """Everything the map knows about a set of target points.

        points: (..., 3), any batch shape. Returns a dict of tensors with that
        batch shape (or (B,) for the per-point scalars, keeping a trailing 3 for
        positions and dof for configurations):

          reachable       bool, voxel occupancy (see is_reachable)
          inside          bool, whether the point is inside the grid box at all
          count           number of samples in the point's voxel
          density         that voxel's share of the samples
          manipulability  that voxel's mean manipulability (0 if empty)
          q               (..., dof) the best configuration found for the point
          point           (..., 3) where that configuration actually puts the link
          error           (...) distance from `point` to the requested point

        The best configuration starts from the nearest stored sample and, when
        refine=True, is polished with position-only damped-least-squares IK. That
        is what makes the answer useful rather than merely voxel-accurate: for a
        reachable point the error drops to solver tolerance, and for an
        unreachable one you get the closest the arm can get, which is exactly the
        number you want when deciding whether to move the base. The refinement is
        kept per point only where it beat the seed, so the reported error is
        never worse than the nearest sample already achieved.

        The refinement is autograd-traceable, so gradients flow from the returned
        configuration back to the queried point.
        """
        if self.q is None or self.points is None:
            raise ValueError(
                "query needs the samples that built the map; rebuild it with "
                "keep_samples=True")
        p = torch.as_tensor(points)
        if p.ndim == 0 or p.shape[-1] != 3:
            raise ValueError(f"points must have a trailing dim of 3, got {tuple(p.shape)}")
        batch_shape = p.shape[:-1]
        flat_p = p.reshape(-1, 3)
        dtype = flat_p.dtype if flat_p.is_floating_point() else self.q.dtype
        flat_p = flat_p.to(dtype)
        device = flat_p.device

        counts, inside = self._gather(self.counts, flat_p)
        density, _ = self._gather(self.density, flat_p)
        manip, _ = self._gather(self.manipulability, flat_p)

        # nearest stored sample, chunked so a big query does not build a
        # (B, N) distance matrix in one go
        samples = self.points.detach().to(device=device, dtype=dtype)
        seeds = []
        for start in range(0, flat_p.shape[0], 256):
            block = flat_p[start:start + 256]
            d = torch.cdist(block, samples)                       # (b, N)
            seeds.append(d.argmin(dim=1))
        seed_idx = torch.cat(seeds) if seeds else torch.zeros(0, dtype=torch.long,
                                                              device=device)
        q_best = self.q.detach().to(device=device, dtype=dtype)[seed_idx]

        if flat_p.shape[0]:
            reached = forward_kinematics(self.chain, q_best)[:, self.link_index, :3, 3]
        else:
            reached = torch.zeros(0, 3, dtype=dtype, device=device)
        err = (reached - flat_p).norm(dim=-1)

        if refine and flat_p.shape[0]:
            target = torch.eye(4, dtype=dtype, device=device)
            target = target.expand(flat_p.shape[0], 4, 4).clone()
            target[:, :3, 3] = flat_p
            q_ref, _info = _ik(self.chain, target, q0=q_best,
                               link_index=self.link_index, iters=iters,
                               damping=damping, step=step, pos_only=True,
                               tol=tol, restarts=1)
            p_ref = forward_kinematics(self.chain, q_ref)[:, self.link_index, :3, 3]
            e_ref = (p_ref - flat_p).norm(dim=-1)
            # Keep the refinement only where it actually helped. Started at a
            # singular posture (a fully stretched arm aimed at an out-of-reach
            # point) damped least squares can stall or wander off, and returning
            # something worse than the sample we already had would be a lie: the
            # guarantee this makes is that query() never reports a larger error
            # than the nearest stored sample.
            take = torch.isfinite(e_ref) & (e_ref <= err)
            q_best = torch.where(take.unsqueeze(-1), q_ref, q_best)
            reached = torch.where(take.unsqueeze(-1), p_ref, reached)
            err = torch.where(take, e_ref, err)

        dof = self.q.shape[1]
        return {
            "reachable": (counts > 0).reshape(batch_shape),
            "inside": inside.reshape(batch_shape),
            "count": counts.reshape(batch_shape),
            "density": density.reshape(batch_shape),
            "manipulability": manip.reshape(batch_shape),
            "q": q_best.reshape(*batch_shape, dof),
            "point": reached.reshape(*batch_shape, 3),
            "error": err.reshape(batch_shape),
        }

    def to(self, device):
        """Move the grid and the stored samples to a device. The chain is left
        alone: fk/jacobian cast its constants per (device, dtype) themselves, so
        a map can be queried on a device the chain was never moved to."""
        for name in ("origin", "voxel", "counts", "density", "manipulability",
                     "q", "points", "w"):
            t = getattr(self, name)
            if isinstance(t, torch.Tensor):
                setattr(self, name, t.to(device))
        return self

    def __repr__(self):
        occ = int((self.counts > 0).sum())
        return (f"ReachabilityMap(shape={self.shape}, occupied={occ}/{self.n_voxels}, "
                f"samples={self.n_samples}, voxel={self.voxel.tolist()})")


def _grid_from_bounds(lo, hi, voxel):
    """Grid origin and integer shape covering [lo, hi] with the given voxel size.

    The upper corner is rounded up to a whole number of voxels so the requested
    box is always fully covered, and every axis gets at least one voxel (a planar
    arm has zero extent in z and still needs a grid).
    """
    extent = (hi - lo).clamp_min(0.0)
    counts = torch.ceil(extent / voxel - 1e-6).to(torch.long).clamp_min(1)
    return lo.clone(), tuple(int(c) for c in counts)


def reachability_map(chain, link_index=-1, n: int = 20000, voxel=0.05,
                     seed: int = 0, bounds=None, q=None, dtype=None,
                     device=None, rows=None, with_manipulability: bool = True,
                     chunk: int = 8192, keep_samples: bool = True):
    """Build a voxelized reachability map for one link of a chain.

    chain        a CompiledChain.
    link_index   which link's origin is the "end effector". Negative indices work
                 the same way they do in forward_kinematics, so -1 is the last
                 link.
    n            how many configurations to sample. Ignored if you pass q.
    voxel        edge length of a voxel, a scalar or per-axis triple.
    seed         seeds a CPU generator so a map is reproducible bit for bit, on
                 any device, exactly as analysis.workspace does.
    bounds       ((x0,y0,z0), (x1,y1,z1)) to pin the grid to a fixed box, e.g. to
                 compare two robots on the same grid or to zoom in on a cell.
                 Default: the bounding box of the sampled points padded by one
                 voxel, so no sample sits on an edge.
    q            (N, dof) configurations to use instead of sampling. This is the
                 hook for a custom distribution (a task-space prior, a recorded
                 trajectory, a grid over the joints) and for differentiability:
                 gradients flow from the per-voxel mean manipulability back to q.
    dtype/device working precision and placement. Default: the chain's.
    rows         Jacobian rows for the manipulability measure, passed straight
                 through to analysis.manipulability. A planar arm needs
                 rows=(0, 1); leave it None for a spatial arm.
    chunk        samples processed per forward-kinematics batch. Bounds peak
                 memory without changing any result.
    keep_samples keep q/points/w on the map. Needed by query(); turn it off if
                 you only want the grid and n is huge.

    Returns a ReachabilityMap.

    Sampling is uniform over the joint box, which is the honest default but not a
    uniform measure over the workspace: the density array will always pile up
    near the shoulder. Read counts as "is this reachable", density as "how
    redundantly", and manipulability as "how well".
    """
    link_index = _resolve_link(chain, link_index)
    checked_rows = _check_rows(rows) if rows is not None else None
    if not (isinstance(chunk, int) and not isinstance(chunk, bool) and chunk >= 1):
        raise ValueError(f"chunk must be an integer >= 1, got {chunk!r}")

    if q is None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"reachability_map needs an integer n >= 1, got {n!r}")
        want_dtype = dtype or chain.lower.dtype
        want_device = device if device is not None else chain.lower.device
        lo, hi = sampling_bounds(chain)
        lo = lo.to(dtype=want_dtype).cpu()
        hi = hi.to(dtype=want_dtype).cpu()
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        u = torch.rand(n, chain.dof, generator=g, dtype=want_dtype)
        q = (lo + (hi - lo) * u).to(want_device)
    else:
        q = torch.as_tensor(q)
        if q.ndim != 2 or q.shape[1] != chain.dof:
            raise ValueError(
                f"q must be (N, dof={chain.dof}), got {tuple(q.shape)}")
        if q.shape[0] < 1:
            raise ValueError("q must hold at least one configuration")
        if dtype is not None:
            q = q.to(dtype)
        if device is not None:
            q = q.to(device)
        n = q.shape[0]
        seed = None

    work_dtype = q.dtype
    work_device = q.device
    vox = _check_voxel(voxel, work_dtype, work_device)

    # ---- pass 1: end-effector positions (and manipulability) ---------------
    pts_chunks, w_chunks = [], []
    for start in range(0, n, chunk):
        qc = q[start:start + chunk]
        pc, wc = _points_and_manipulability(chain, qc, link_index, checked_rows,
                                            with_manipulability)
        pts_chunks.append(pc)
        if with_manipulability:
            w_chunks.append(wc)
    points = torch.cat(pts_chunks, dim=0)                       # (N, 3)
    w = torch.cat(w_chunks, dim=0) if with_manipulability else None

    # ---- grid geometry -----------------------------------------------------
    if bounds is None:
        lo_b = points.detach().min(dim=0).values - vox
        hi_b = points.detach().max(dim=0).values + vox
    else:
        try:
            b0, b1 = bounds
        except (TypeError, ValueError):
            raise ValueError(
                f"bounds must be a (lower, upper) pair of 3-vectors, got {bounds!r}")
        lo_b = _as_vec3(b0, "bounds lower", work_dtype, work_device)
        hi_b = _as_vec3(b1, "bounds upper", work_dtype, work_device)
        if bool((hi_b <= lo_b).any()):
            raise ValueError(
                f"bounds upper must exceed lower on every axis, got {bounds!r}")
    origin, shape = _grid_from_bounds(lo_b, hi_b, vox)
    n_vox = shape[0] * shape[1] * shape[2]

    # ---- bin ---------------------------------------------------------------
    raw = torch.floor((points.detach() - origin) / vox).to(torch.long)
    shape_t = torch.tensor(shape, dtype=torch.long, device=work_device)
    inside = ((raw >= 0) & (raw < shape_t)).all(dim=-1)
    raw = raw.clamp(torch.zeros_like(shape_t), shape_t - 1)
    lin = (raw[:, 0] * shape[1] + raw[:, 1]) * shape[2] + raw[:, 2]
    lin_in = lin[inside]

    counts = torch.bincount(lin_in, minlength=n_vox).reshape(shape)
    n_binned = int(inside.sum())
    n_outside = int(n) - n_binned

    denom = max(n_binned, 1)
    density = counts.to(work_dtype) / denom

    if with_manipulability:
        # index_add (out of place) keeps the graph, so d(mean w)/dq is available
        # whenever q carries gradients. The counts are a hard assignment and
        # deliberately stay outside the graph.
        sums = torch.zeros(n_vox, dtype=work_dtype, device=work_device)
        sums = sums.index_add(0, lin_in, w[inside])
        mean_w = sums / counts.reshape(-1).to(work_dtype).clamp_min(1.0)
        mean_w = mean_w.reshape(shape)
    else:
        mean_w = torch.zeros(shape, dtype=work_dtype, device=work_device)

    return ReachabilityMap(
        chain=chain, link_index=link_index, origin=origin, voxel=vox,
        shape=shape, counts=counts, density=density, manipulability=mean_w,
        n_samples=int(n), n_binned=n_binned, n_outside=n_outside,
        q=q if keep_samples else None,
        points=points if keep_samples else None,
        w=w if keep_samples else None,
        rows=tuple(checked_rows) if checked_rows is not None else None,
        seed=seed,
    )
