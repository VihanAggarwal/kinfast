# tests/test_reachability.py
"""Voxelized reachability against closed-form planar oracles.

A planar 2R arm with links l1 and l2 reaches exactly the annulus
|l1 - l2| <= r <= l1 + l2 (and nothing else), and at radius r its Yoshikawa
manipulability over the (x, y) task rows is

    w(r) = l1*l2*|sin q2| = sqrt(l1^2*l2^2 - ((r^2 - l1^2 - l2^2) / 2)^2)

because cos q2 = (r^2 - l1^2 - l2^2) / (2*l1*l2). That gives us a per-voxel
oracle that owes nothing to the library: occupancy is decided by the annulus,
mean manipulability by the formula above, and the distance from an unreachable
point to the workspace is the plain geometric gap to the nearest ring.

The fixture uses unequal links (l1 = 1.0, l2 = 0.6) so the annulus has a real
hole of radius 0.4 to test against. Coverage is made deterministic by feeding
the map a regular grid over the joints rather than random samples: with 400
steps per joint the samples are at most 0.026 m apart anywhere in the annulus,
comfortably finer than the 0.05 m voxels, so "every interior voxel is hit" is a
statement about geometry and not about luck. The random path is exercised
separately for reproducibility and bookkeeping.
"""
import math
import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.urdf.parse import parse_urdf_string
from kinfast import analysis as A
from kinfast import reachability as R

from tests.test_analysis import PLANAR_2R
from tests.test_spatial import SIX_DOF

L1, L2 = 1.0, 0.6
PI = "3.14159265358979"

ANNULUS_2R = f"""
<robot name="annulus2r">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="ee"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-{PI}" upper="{PI}" velocity="2" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="{L1} 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-{PI}" upper="{PI}" velocity="2" effort="50"/></joint>
  <joint name="jf" type="fixed"><parent link="l2"/><child link="ee"/>
    <origin xyz="{L2} 0 0"/></joint>
</robot>
"""


def _chain(text=ANNULUS_2R, dtype=torch.float64):
    ir = parse_urdf_string(text)
    chain = compile_robot(ir, dtype=dtype)
    return chain, chain.link_index["ee"]


def _joint_grid(k=400, dtype=torch.float64):
    """Regular k-by-k grid over both joints, covering the full turn once."""
    a = torch.linspace(-math.pi, math.pi, k + 1, dtype=dtype)[:-1]
    q1, q2 = torch.meshgrid(a, a, indexing="ij")
    return torch.stack([q1.reshape(-1), q2.reshape(-1)], dim=-1)


def _w_of_r(r, l1=L1, l2=L2):
    """Closed-form manipulability of a planar 2R at reach radius r."""
    u = (r * r - l1 * l1 - l2 * l2) / 2.0
    return math.sqrt(max(l1 * l1 * l2 * l2 - u * u, 0.0))


@pytest.fixture(scope="module")
def annulus_map():
    chain, li = _chain()
    m = R.reachability_map(chain, li, q=_joint_grid(), voxel=0.05, rows=(0, 1))
    return m


def _radii(m):
    """In-plane radius of every voxel center: (nx, ny, nz)."""
    return m.voxel_centers()[..., :2].norm(dim=-1)


def _plane_mask(m):
    """Voxels in the z layer that contains z = 0 (the arm is planar)."""
    idx, inside = m.voxel_of(torch.zeros(3, dtype=m.voxel.dtype))
    assert bool(inside)
    layer = torch.zeros(m.shape[2], dtype=torch.bool)
    layer[int(idx[2])] = True
    return layer.reshape(1, 1, -1).expand(m.shape)


# --------------------------------------------------------------------------
# occupancy: the reachable set is an annulus
# --------------------------------------------------------------------------
def test_annulus_interior_is_fully_occupied(annulus_map):
    """Every voxel whose center sits well inside the annulus was hit."""
    m = annulus_map
    r = _radii(m)
    plane = _plane_mask(m)
    interior = (r > L1 - L2 + 0.05) & (r < L1 + L2 - 0.05) & plane
    assert int(interior.sum()) > 2000, "oracle mask should cover most of the ring"
    missed = interior & (m.counts == 0)
    assert int(missed.sum()) == 0, f"{int(missed.sum())} interior voxels unhit"


def test_outside_the_annulus_is_empty(annulus_map):
    """Voxels beyond the outer radius or inside the hole have no hits at all."""
    m = annulus_map
    r = _radii(m)
    outside = (r > L1 + L2 + 0.1) | (r < L1 - L2 - 0.1)
    assert int(outside.sum()) > 500, "the grid must contain unreachable voxels"
    assert int(m.counts[outside].sum()) == 0
    assert not bool(m.is_reachable(m.voxel_centers()[outside]).any())


def test_arm_is_planar_so_only_one_z_layer_is_occupied(annulus_map):
    m = annulus_map
    plane = _plane_mask(m)
    assert int(m.counts[~plane].sum()) == 0
    assert int(m.counts[plane].sum()) == m.n_binned


def test_reachable_volume_tracks_the_annulus_area(annulus_map):
    """Occupied voxels times voxel volume approximates area * voxel height."""
    m = annulus_map
    area = math.pi * ((L1 + L2) ** 2 - (L1 - L2) ** 2)
    expect = area * float(m.voxel[2])
    got = float(m.reachable_volume)
    assert abs(got - expect) / expect < 0.10, (got, expect)


def test_planar_2r_with_equal_links_has_no_hole():
    """The shared PLANAR_2R fixture has l1 = l2 = 1, so |l1 - l2| = 0 and the
    annulus degenerates to a disc of radius 2 with no hole to speak of."""
    chain, li = _chain(PLANAR_2R)
    m = R.reachability_map(chain, li, q=_joint_grid(300), voxel=0.05, rows=(0, 1))
    r = _radii(m)
    plane = _plane_mask(m)
    interior = (r > 0.2) & (r < 1.9) & plane
    assert int((interior & (m.counts == 0)).sum()) == 0
    assert int(m.counts[r > 2.1].sum()) == 0


# --------------------------------------------------------------------------
# per-voxel manipulability
# --------------------------------------------------------------------------
def test_voxel_manipulability_matches_closed_form(annulus_map):
    """Mean manipulability in a voxel equals w(r) at its center.

    w varies across a voxel, so the tolerance is derived rather than guessed:
    |dw/dr| <= |r| * r_max / w, and the half-diagonal of a 0.05 voxel in the
    plane is 0.036, which bounds the spread well under 0.02 for these radii.
    """
    m = annulus_map
    for radius in (0.6, 0.8, 1.0, 1.2, 1.4):
        idx, inside = m.voxel_of(torch.tensor([radius, 0.0, 0.0],
                                              dtype=torch.float64))
        assert bool(inside)
        key = tuple(idx.tolist())
        center_r = float(m.voxel_centers()[key][:2].norm())
        got = float(m.manipulability[key])
        assert abs(got - _w_of_r(center_r)) < 0.02, (radius, got)


def test_empty_voxels_report_zero_manipulability(annulus_map):
    m = annulus_map
    empty = m.counts == 0
    assert bool(empty.any())
    assert float(m.manipulability[empty].abs().max()) == 0.0


def test_per_sample_manipulability_is_the_textbook_value():
    """The stored per-sample w is l1*l2*|sin q2|, and agrees with the value
    analysis.manipulability computes independently."""
    chain, li = _chain()
    q = torch.tensor([[0.0, 0.0], [0.3, math.pi / 2], [-1.1, math.pi / 6],
                      [2.0, -0.7]], dtype=torch.float64)
    m = R.reachability_map(chain, li, q=q, voxel=0.5, rows=(0, 1))
    expect = L1 * L2 * q[:, 1].sin().abs()
    assert torch.allclose(m.w, expect, atol=1e-12)
    assert torch.allclose(m.w, A.manipulability(chain, q, li, rows=(0, 1)),
                          atol=1e-12)


# --------------------------------------------------------------------------
# binning arithmetic, worked out by hand
# --------------------------------------------------------------------------
def _hand_map(**kw):
    """Four configurations on a 1 m grid, every voxel index hand-computed.

    positions:  q=(0,0)      -> (1.6, 0.0)   -> voxel (3, 2, 1)
                q=(pi/2,0)   -> (0.0, 1.6)   -> voxel (2, 3, 1)
                q=(0,pi/2)   -> (1.0, 0.6)   -> voxel (3, 2, 1)
                q=(pi/2,pi/2)-> (-0.6, 1.0)  -> voxel (1, 3, 1)
    manipulability l1*l2*|sin q2| = 0, 0, 0.6, 0.6
    """
    chain, li = _chain()
    q = torch.tensor([[0.0, 0.0], [math.pi / 2, 0.0],
                      [0.0, math.pi / 2], [math.pi / 2, math.pi / 2]],
                     dtype=torch.float64)
    bounds = ((-2.0, -2.0, -1.0), (2.0, 2.0, 1.0))
    return chain, li, q, R.reachability_map(chain, li, q=q, voxel=1.0,
                                            bounds=bounds, rows=(0, 1), **kw)


def test_hand_computed_counts_density_and_means():
    chain, li, q, m = _hand_map()
    assert m.shape == (4, 4, 2)
    assert torch.allclose(m.origin, torch.tensor([-2.0, -2.0, -1.0],
                                                 dtype=torch.float64))
    pos = forward_kinematics(chain, q)[:, li, :3, 3]
    expect_pos = torch.tensor([[1.6, 0.0, 0.0], [0.0, 1.6, 0.0],
                               [1.0, 0.6, 0.0], [-0.6, 1.0, 0.0]],
                              dtype=torch.float64)
    assert torch.allclose(pos, expect_pos, atol=1e-12)

    assert int(m.counts[3, 2, 1]) == 2
    assert int(m.counts[2, 3, 1]) == 1
    assert int(m.counts[1, 3, 1]) == 1
    assert int(m.counts.sum()) == 4
    assert float(m.density[3, 2, 1]) == 0.5
    assert abs(float(m.density.sum()) - 1.0) < 1e-12
    # (0 + 0.6) / 2 in the shared voxel, 0 and 0.6 in the singletons
    assert abs(float(m.manipulability[3, 2, 1]) - 0.3) < 1e-12
    assert abs(float(m.manipulability[2, 3, 1]) - 0.0) < 1e-12
    assert abs(float(m.manipulability[1, 3, 1]) - 0.6) < 1e-12
    assert m.n_binned == 4 and m.n_outside == 0


def test_samples_outside_explicit_bounds_are_dropped_and_counted():
    chain, li, q, _ = _hand_map()
    tight = ((-0.5, -0.5, -1.0), (0.5, 0.5, 1.0))
    m = R.reachability_map(chain, li, q=q, voxel=0.25, bounds=tight, rows=(0, 1))
    assert m.n_samples == 4 and m.n_binned == 0 and m.n_outside == 4
    assert int(m.counts.sum()) == 0
    assert float(m.density.sum()) == 0.0
    assert not bool(m.is_reachable(torch.zeros(3, dtype=torch.float64)))


def test_grid_geometry_is_a_plain_floor_division():
    _c, _li, _q, m = _hand_map()
    lo, hi = m.bounds
    assert torch.allclose(lo, torch.tensor([-2.0, -2.0, -1.0], dtype=torch.float64))
    assert torch.allclose(hi, torch.tensor([2.0, 2.0, 1.0], dtype=torch.float64))
    centers = m.voxel_centers()
    assert centers.shape == (4, 4, 2, 3)
    assert torch.allclose(centers[0, 0, 0],
                          torch.tensor([-1.5, -1.5, -0.5], dtype=torch.float64))
    idx, inside = m.voxel_of(centers.reshape(-1, 3))
    assert bool(inside.all())
    grid = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4),
                                      torch.arange(2), indexing="ij"), dim=-1)
    assert torch.equal(idx, grid.reshape(-1, 3))
    # a point past the upper corner is clamped for indexing but flagged outside
    idx, inside = m.voxel_of(torch.tensor([9.0, 0.0, 0.0], dtype=torch.float64))
    assert not bool(inside) and int(idx[0]) == 3


# --------------------------------------------------------------------------
# differentiability
# --------------------------------------------------------------------------
def test_manipulability_gradient_matches_finite_differences():
    """d(sum of per-voxel mean manipulability)/dq against float64 central
    differences. The step is small enough that no sample changes voxel, which
    is the only regime where the derivative exists at all: the binning is a
    hard assignment and is deliberately outside the graph."""
    chain, li = _chain()
    bounds = ((-2.0, -2.0, -0.1), (2.0, 2.0, 0.1))
    base = torch.tensor([[0.3, 0.9], [0.31, 0.95], [-0.4, 1.2], [0.7, -0.8]],
                        dtype=torch.float64)

    def scalar(qv):
        m = R.reachability_map(chain, li, q=qv, voxel=0.2, bounds=bounds,
                               rows=(0, 1))
        return m.manipulability.sum()

    q = base.clone().requires_grad_(True)
    scalar(q).backward()
    assert q.grad is not None and bool(torch.isfinite(q.grad).all())

    h = 1e-6
    fd = torch.zeros_like(base)
    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            up, dn = base.clone(), base.clone()
            up[i, j] += h
            dn[i, j] -= h
            fd[i, j] = (float(scalar(up)) - float(scalar(dn))) / (2 * h)
    assert torch.allclose(q.grad, fd, atol=1e-6), (q.grad, fd)


def test_counts_carry_no_gradient(annulus_map):
    """Counts are integers by construction; nothing should pretend otherwise."""
    assert annulus_map.counts.dtype == torch.long
    assert not annulus_map.counts.is_floating_point()


# --------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------
def test_query_hits_reachable_points_and_reports_the_true_gap(annulus_map):
    """Four targets with hand-computable answers:

      (1.0, 0.2, 0)  inside the annulus            -> exact solve, error 0
      (0.1, 0, 0)    inside the hole (r = 0.1)     -> gap 0.4 - 0.1 = 0.3
      (3.0, 0, 0)    past the outer radius         -> gap 3.0 - 1.6 = 1.4
      (0, 0, 0.3)    off the arm's plane           -> gap sqrt(0.3^2 + 0.4^2)
    """
    m = annulus_map
    pts = torch.tensor([[1.0, 0.2, 0.0], [0.1, 0.0, 0.0],
                        [3.0, 0.0, 0.0], [0.0, 0.0, 0.3]], dtype=torch.float64)
    out = m.query(pts)
    assert out["reachable"].tolist() == [True, False, False, False]
    assert out["inside"].tolist() == [True, True, False, False]
    expect_err = torch.tensor([0.0, 0.3, 1.4, 0.5], dtype=torch.float64)
    assert torch.allclose(out["error"], expect_err, atol=2e-3), out["error"]
    # the reported configuration really does put the link where it says
    fk_pos = forward_kinematics(m.chain, out["q"])[:, m.link_index, :3, 3]
    assert torch.allclose(fk_pos, out["point"], atol=1e-12)
    assert torch.allclose((fk_pos - pts).norm(dim=-1), out["error"], atol=1e-12)
    # the reachable target is solved to solver tolerance, not just voxel accuracy
    assert float(out["error"][0]) < 1e-6
    # voxel readout agrees with the standalone occupancy test
    assert torch.equal(out["reachable"], m.is_reachable(pts))
    assert int(out["count"][0]) > 0 and float(out["density"][0]) > 0
    assert float(out["manipulability"][0]) > 0


def test_query_never_returns_worse_than_the_nearest_sample(annulus_map):
    """Refinement is kept only where it helped, so the error is bounded by the
    distance from the target to the closest stored end-effector position."""
    m = annulus_map
    pts = torch.tensor([[1.0, 0.2, 0.0], [0.1, 0.0, 0.0], [3.0, 0.0, 0.0],
                        [-1.55, 0.0, 0.0], [0.0, 0.0, 0.3]], dtype=torch.float64)
    nearest = torch.cdist(pts, m.points).min(dim=1).values
    refined = m.query(pts)["error"]
    raw = m.query(pts, refine=False)["error"]
    assert torch.allclose(raw, nearest, atol=1e-12)
    assert bool((refined <= nearest + 1e-12).all()), (refined, nearest)


def test_query_preserves_batch_shape(annulus_map):
    m = annulus_map
    pts = torch.tensor([[1.0, 0.2, 0.0], [0.1, 0.0, 0.0],
                        [3.0, 0.0, 0.0], [-1.2, 0.5, 0.0]],
                       dtype=torch.float64).reshape(2, 2, 3)
    out = m.query(pts, refine=False)
    for key in ("reachable", "inside", "count", "density", "manipulability",
                "error"):
        assert out[key].shape == (2, 2), key
    assert out["q"].shape == (2, 2, 2)
    assert out["point"].shape == (2, 2, 3)
    # a single point keeps a scalar answer
    single = m.query(torch.tensor([1.0, 0.2, 0.0], dtype=torch.float64),
                     refine=False)
    assert single["error"].shape == ()
    assert single["q"].shape == (2,)


def test_query_gradient_flows_back_to_the_target_point(annulus_map):
    """The refinement is autograd traceable, so a task-space objective can be
    differentiated through the returned configuration."""
    m = annulus_map
    p = torch.tensor([[1.0, 0.2, 0.0]], dtype=torch.float64, requires_grad=True)
    out = m.query(p, refine=True, iters=30)
    out["q"].sum().backward()
    assert p.grad is not None
    assert bool(torch.isfinite(p.grad).all())
    assert float(p.grad.abs().sum()) > 0


def test_query_needs_the_samples(annulus_map):
    chain, li = _chain()
    m = R.reachability_map(chain, li, n=200, seed=0, voxel=0.1, rows=(0, 1),
                           keep_samples=False)
    assert m.q is None and m.points is None and m.w is None
    with pytest.raises(ValueError, match="keep_samples"):
        m.query(torch.zeros(1, 3, dtype=torch.float64))
    # the grid itself still answers occupancy questions
    assert m.is_reachable(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)).numel() == 1


# --------------------------------------------------------------------------
# sampling, dtype, device
# --------------------------------------------------------------------------
def test_random_sampling_is_reproducible_and_seed_dependent():
    chain, li = _chain()
    kw = dict(n=3000, voxel=0.1, rows=(0, 1))
    a = R.reachability_map(chain, li, seed=7, **kw)
    b = R.reachability_map(chain, li, seed=7, **kw)
    c = R.reachability_map(chain, li, seed=8, **kw)
    assert torch.equal(a.counts, b.counts)
    assert torch.allclose(a.manipulability, b.manipulability, atol=0)
    assert not torch.equal(a.counts, c.counts)
    assert a.n_samples == 3000 and a.n_binned == 3000 and a.n_outside == 0
    assert int(a.counts.sum()) == 3000
    assert abs(float(a.density.sum()) - 1.0) < 1e-12
    assert a.seed == 7


def test_random_samples_respect_the_annulus():
    """Sampled configurations only ever land in the reachable ring."""
    chain, li = _chain()
    m = R.reachability_map(chain, li, n=5000, seed=3, voxel=0.1, rows=(0, 1))
    r = m.points[:, :2].norm(dim=-1)
    assert float(r.max()) <= L1 + L2 + 1e-12
    assert float(r.min()) >= L1 - L2 - 1e-12
    assert float(m.points[:, 2].abs().max()) < 1e-12


def test_dtype_follows_the_caller():
    chain, li = _chain(dtype=torch.float32)
    m32 = R.reachability_map(chain, li, n=500, seed=0, voxel=0.1, rows=(0, 1))
    for t in (m32.density, m32.manipulability, m32.origin, m32.voxel,
              m32.points, m32.q, m32.w):
        assert t.dtype == torch.float32
    q64 = torch.rand(64, 2, dtype=torch.float64) - 0.5
    m64 = R.reachability_map(chain, li, q=q64, voxel=0.1, rows=(0, 1))
    assert m64.density.dtype == torch.float64
    assert m64.manipulability.dtype == torch.float64
    # a float32 query against a float64 map works and answers in float32
    out = m64.query(torch.tensor([[1.0, 0.2, 0.0]], dtype=torch.float32))
    assert out["q"].dtype == torch.float32 and out["point"].dtype == torch.float32


def test_chunking_does_not_change_the_result():
    chain, li = _chain()
    q = _joint_grid(60)
    a = R.reachability_map(chain, li, q=q, voxel=0.1, rows=(0, 1), chunk=17)
    b = R.reachability_map(chain, li, q=q, voxel=0.1, rows=(0, 1), chunk=100000)
    assert torch.equal(a.counts, b.counts)
    assert torch.allclose(a.manipulability, b.manipulability, atol=0)


def test_manipulability_can_be_skipped():
    chain, li = _chain()
    m = R.reachability_map(chain, li, n=200, seed=0, voxel=0.1,
                           with_manipulability=False)
    assert m.w is None
    assert float(m.manipulability.abs().max()) == 0.0
    assert int(m.counts.sum()) == 200


def test_negative_link_index_resolves_like_forward_kinematics():
    chain, li = _chain()
    a = R.reachability_map(chain, li, n=300, seed=2, voxel=0.1, rows=(0, 1))
    b = R.reachability_map(chain, -1, n=300, seed=2, voxel=0.1, rows=(0, 1))
    assert b.link_index == li
    assert torch.equal(a.counts, b.counts)


def test_map_moves_to_a_device():
    chain, li = _chain()
    m = R.reachability_map(chain, li, n=200, seed=0, voxel=0.1, rows=(0, 1))
    m.to("cpu")
    assert m.counts.device.type == "cpu"
    assert m.q.device.type == "cpu"


# --------------------------------------------------------------------------
# a spatial arm, to be sure nothing is quietly planar
# --------------------------------------------------------------------------
def test_six_dof_map_stays_inside_the_kinematic_reach():
    chain = compile_robot(parse_urdf_string(SIX_DOF), dtype=torch.float64)
    li = chain.link_index["ee"]
    m = R.reachability_map(chain, li, n=4000, seed=1, voxel=0.1)
    reach = 0.3 + 0.3 + 0.3 + 0.1 + 0.1          # sum of the link offsets
    assert float(m.points.norm(dim=-1).max()) <= reach + 1e-9
    assert m.shape[0] > 1 and m.shape[1] > 1 and m.shape[2] > 1
    assert int(m.counts.sum()) == 4000
    assert float(m.manipulability.max()) > 0
    # a pose the arm demonstrably holds must come back reachable and solvable
    q = torch.tensor([[0.4, -0.6, 0.9, 0.2, 0.5, -0.3]], dtype=torch.float64)
    target = forward_kinematics(chain, q)[:, li, :3, 3]
    out = m.query(target)
    assert bool(out["reachable"][0])
    assert float(out["error"][0]) < 1e-4
    assert float(out["manipulability"][0]) > 0


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------
def test_bad_inputs_raise_readable_errors():
    chain, li = _chain()
    with pytest.raises(ValueError, match="n >= 1"):
        R.reachability_map(chain, li, n=0)
    with pytest.raises(ValueError, match="n >= 1"):
        R.reachability_map(chain, li, n=2.5)
    with pytest.raises(ValueError, match="voxel"):
        R.reachability_map(chain, li, n=10, voxel=0.0)
    with pytest.raises(ValueError, match="voxel"):
        R.reachability_map(chain, li, n=10, voxel=-0.1)
    with pytest.raises(ValueError, match="voxel"):
        R.reachability_map(chain, li, n=10, voxel=float("inf"))
    with pytest.raises(ValueError, match="voxel"):
        R.reachability_map(chain, li, n=10, voxel=(0.1, 0.1))
    with pytest.raises(ValueError, match="chunk"):
        R.reachability_map(chain, li, n=10, chunk=0)
    with pytest.raises(ValueError, match="bounds"):
        R.reachability_map(chain, li, n=10,
                           bounds=((0.0, 0.0, 0.0), (0.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="q must be"):
        R.reachability_map(chain, li, q=torch.zeros(4, 3))
    with pytest.raises(ValueError, match="at least one"):
        R.reachability_map(chain, li, q=torch.zeros(0, 2))
    with pytest.raises(ValueError, match="rows"):
        R.reachability_map(chain, li, n=10, rows=())
    with pytest.raises(ValueError, match="rows"):
        R.reachability_map(chain, li, n=10, rows=(0, 9))
    with pytest.raises(IndexError, match="link_index"):
        R.reachability_map(chain, 99, n=10)
    m = R.reachability_map(chain, li, n=50, seed=0, voxel=0.2, rows=(0, 1))
    with pytest.raises(ValueError, match="trailing dim of 3"):
        m.is_reachable(torch.zeros(4, 2))
    with pytest.raises(ValueError, match="trailing dim of 3"):
        m.query(torch.zeros(4, 2))


def test_repr_names_the_shape_and_occupancy():
    chain, li = _chain()
    m = R.reachability_map(chain, li, n=100, seed=0, voxel=0.2, rows=(0, 1))
    text = repr(m)
    assert "ReachabilityMap" in text and "shape=" in text and "occupied=" in text
