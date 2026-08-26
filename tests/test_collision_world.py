# tests/test_collision_world.py
"""World geometry distances: hand-computed values, a mujoco oracle, the floor
under a 6-DOF arm, gradients, batching and dtype behaviour.

Where a distance can be written down by hand it is written down by hand, and
the shape primitives are additionally cross-checked against mujoco's
mj_geomDistance, which computes the same quantity by a completely different
route (convex collision detection rather than a closed-form signed distance).
"""
import math

import pytest
import torch

from kinfast.collision import SphereModel
from kinfast.collision_world import (
    Box,
    Capsule,
    HalfSpace,
    Sphere,
    as_shapes,
    distance_to_world,
    in_collision,
    shape_distances,
    world_clearance_cost,
    world_distances,
)
from kinfast.compile import compile_robot
from kinfast.urdf.parse import parse_urdf_string
from tests.test_parse import TWO_LINK
from tests.test_spatial import SIX_DOF

F64 = torch.float64


def _two_link(dtype=F64):
    return compile_robot(parse_urdf_string(TWO_LINK), dtype=dtype)


def _six_dof(dtype=F64):
    return compile_robot(parse_urdf_string(SIX_DOF), dtype=dtype)


def _arm_model(chain, radius=0.05):
    """One sphere at the origin of each moving link of SIX_DOF."""
    idx = chain.link_index
    return SphereModel(chain, {idx[n]: [(0.0, 0.0, 0.0, radius)]
                               for n in ("l2", "l3", "l4", "l5", "ee")})


def _pts(*rows):
    return torch.tensor(rows, dtype=F64)


# --------------------------------------------------------- shape primitives

def test_halfspace_hand_computed():
    """A floor at z=0: the distance of a point is its height."""
    floor = HalfSpace.floor(0.0)
    d = floor.signed_distance(_pts([0, 0, 1.0], [3.0, -4.0, 0.25], [0, 0, 0], [1, 1, -0.5]))
    assert torch.allclose(d, torch.tensor([1.0, 0.25, 0.0, -0.5], dtype=F64), atol=1e-12)


def test_halfspace_raised_floor_and_ceiling():
    d = HalfSpace.floor(0.4).signed_distance(_pts([0, 0, 1.0], [0, 0, 0.1]))
    assert torch.allclose(d, torch.tensor([0.6, -0.3], dtype=F64), atol=1e-12)
    d = HalfSpace.ceiling(2.0).signed_distance(_pts([0, 0, 1.0], [0, 0, 2.5]))
    assert torch.allclose(d, torch.tensor([1.0, -0.5], dtype=F64), atol=1e-12)


def test_halfspace_normal_is_normalized_on_use():
    """An unnormalized normal still gives a distance in metres.

    Plane 3x + 4y = 10, i.e. n=(3,4,0), d=-10, ||n||=5. The point (2,0,0) has
    3*2 - 10 = -4, over 5 -> -0.8. The tilted plane is what catches a missing
    normalization; an axis-aligned one would not.
    """
    plane = HalfSpace(normal=(3.0, 4.0, 0.0), offset=-10.0)
    d = plane.signed_distance(_pts([2.0, 0.0, 0.0], [2.0, 1.0, 7.0], [6.0, 0.0, 0.0]))
    assert torch.allclose(d, torch.tensor([-0.8, 0.0, 1.6], dtype=F64), atol=1e-12)


def test_box_hand_computed_outside_face_edge_corner_and_inside():
    """Exact signed distance for the four regions around an axis-aligned box.

    Box centered at the origin with half extents (1, 2, 3):
      face   (3, 0, 0)      -> 3 - 1 = 2 straight out of the +x face
      edge   (4, 6, 0)      -> e = (3, 4, -3) -> sqrt(9 + 16) = 5
      corner (2, 4, 9)      -> e = (1, 2, 6)  -> sqrt(1 + 4 + 36) = sqrt(41)
      inside (0.5, 0, 0)    -> nearest face is +x at distance 0.5 -> -0.5
      center (0, 0, 0)      -> nearest face is +x at distance 1   -> -1
    """
    box = Box(center=(0.0, 0.0, 0.0), half_extents=(1.0, 2.0, 3.0))
    d = box.signed_distance(_pts([3, 0, 0], [4, 6, 0], [2, 4, 9], [0.5, 0, 0], [0, 0, 0]))
    want = torch.tensor([2.0, 5.0, math.sqrt(41.0), -0.5, -1.0], dtype=F64)
    assert torch.allclose(d, want, atol=1e-12)


def test_box_from_bounds_matches_center_form():
    a = Box.from_bounds((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))
    b = Box(center=(1.0, 2.0, 3.0), half_extents=(1.0, 2.0, 3.0))
    p = _pts([5.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert torch.allclose(a.signed_distance(p), b.signed_distance(p), atol=1e-12)
    with pytest.raises(ValueError):
        Box.from_bounds((0.0, 0.0, 0.0), (1.0, -1.0, 1.0))


def test_capsule_hand_computed():
    """Segment (0,0,0)-(0,0,1) with radius 0.1.

      (0.5, 0, 0.5) sits beside the middle    -> 0.5 - 0.1 = 0.4
      (0, 0, 2)     is past the top cap       -> 1.0 - 0.1 = 0.9
      (0.3, 0.4, -1) is past the bottom cap   -> sqrt(0.09+0.16+1) - 0.1
      (0, 0, 0.5)   is on the axis            -> -0.1, the radius itself
    """
    cap = Capsule(a=(0, 0, 0), b=(0, 0, 1), radius=0.1)
    d = cap.signed_distance(_pts([0.5, 0, 0.5], [0, 0, 2.0], [0.3, 0.4, -1.0], [0, 0, 0.5]))
    want = torch.tensor([0.4, 0.9, math.sqrt(1.25) - 0.1, -0.1], dtype=F64)
    assert torch.allclose(d, want, atol=1e-12)


def test_capsule_slanted_segment_hand_computed():
    """A slanted segment, where the projection actually does something.

    Segment (0,0,0)-(2,2,0), radius 0.25. The point (2,0,0) projects onto the
    segment at (1,1,0), so the distance is sqrt(2) - 0.25.
    """
    cap = Capsule(a=(0, 0, 0), b=(2, 2, 0), radius=0.25)
    d = cap.signed_distance(_pts([2.0, 0.0, 0.0]))
    assert torch.allclose(d, torch.tensor([math.sqrt(2.0) - 0.25], dtype=F64), atol=1e-12)


def test_degenerate_capsule_is_a_sphere():
    cap = Capsule(a=(1, 2, 3), b=(1, 2, 3), radius=0.5)
    ball = Sphere(center=(1, 2, 3), radius=0.5)
    p = _pts([1, 2, 4.0], [0, 0, 0], [1, 2, 3])
    assert torch.allclose(cap.signed_distance(p), ball.signed_distance(p), atol=1e-9)


def test_sphere_hand_computed():
    ball = Sphere(center=(0.0, 0.0, 1.0), radius=0.25)
    d = ball.signed_distance(_pts([0, 0, 2.0], [0, 0, 1.0], [3.0, 4.0, 1.0]))
    assert torch.allclose(d, torch.tensor([0.75, -0.25, 4.75], dtype=F64), atol=1e-12)


# ------------------------------------------------------------ mujoco oracle

def test_shapes_match_mujoco_geom_distance():
    """Cross-check every primitive against mujoco's own collision distance.

    mj_geomDistance answers the same question by convex collision detection,
    so agreement to machine precision over a few hundred scattered points
    (including points deep inside the box and along the capsule axis) is real
    evidence the closed forms here are the true signed distances and not just
    self-consistent.
    """
    mujoco = pytest.importorskip("mujoco")
    xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" pos="0 0 0" size="0 0 1"/>
        <geom name="box" type="box" pos="1 2 3" size="0.3 0.4 0.5"/>
        <geom name="cap" type="capsule" fromto="0.2 -0.1 0.4 1.0 0.5 0.9" size="0.12"/>
        <body name="probe" pos="0 0 0">
          <freejoint/>
          <geom name="probe" type="sphere" size="0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(m)
    probe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe")
    shapes = {
        "floor": HalfSpace.floor(0.0),
        "box": Box(center=(1.0, 2.0, 3.0), half_extents=(0.3, 0.4, 0.5)),
        "cap": Capsule(a=(0.2, -0.1, 0.4), b=(1.0, 0.5, 0.9), radius=0.12),
    }

    torch.manual_seed(0)
    scattered = -1.0 + 5.0 * torch.rand(80, 3, dtype=F64)
    inside_box = torch.tensor([1.0, 2.0, 3.0], dtype=F64) + (
        -0.28 + 0.56 * torch.rand(20, 3, dtype=F64))
    t = torch.linspace(0.0, 1.0, 15, dtype=F64)[:, None]
    along_axis = (1 - t) * torch.tensor([0.2, -0.1, 0.4], dtype=F64) + \
        t * torch.tensor([1.0, 0.5, 0.9], dtype=F64)
    points = torch.cat([scattered, inside_box, along_axis])

    for name, shape in shapes.items():
        geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        mine = shape.signed_distance(points) - 0.05   # probe sphere radius
        ref = []
        for p in points:
            data.qpos[:3] = p.numpy()
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(m, data)
            ref.append(mujoco.mj_geomDistance(m, data, geom, probe, 100.0, None))
        ref = torch.tensor(ref, dtype=F64)
        assert torch.allclose(mine, ref, atol=1e-9), f"{name} disagrees with mujoco"


# ------------------------------------------------------- through a robot

def test_distance_to_world_hand_computed_at_zero_config():
    """SIX_DOF at q=0 stacks its link origins on the z axis.

    l2 .. ee sit at z = 0.3, 0.6, 0.9, 1.0, 1.1 with radius 0.05, so a floor at
    z=0 is 0.3 - 0.05 = 0.25 away and a ceiling at z=1.5 is 1.5 - 1.1 - 0.05 =
    0.35 away. The world minimum is the floor.
    """
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(1, 6, dtype=F64)
    floor, ceiling = HalfSpace.floor(0.0), HalfSpace.ceiling(1.5)
    assert torch.allclose(distance_to_world(model, q, floor),
                          torch.tensor([0.25], dtype=F64), atol=1e-12)
    assert torch.allclose(distance_to_world(model, q, ceiling),
                          torch.tensor([0.35], dtype=F64), atol=1e-12)
    assert torch.allclose(distance_to_world(model, q, [floor, ceiling]),
                          torch.tensor([0.25], dtype=F64), atol=1e-12)


def test_distance_to_world_box_and_capsule_hand_computed():
    """The same q=0 pose against a table top and a vertical pipe.

    Box centered (0.5, 0, 0.55) with half extents (0.2, 0.2, 0.05): its -x face
    is at x = 0.3 and it spans z in [0.5, 0.6], so the l4 sphere at (0, 0, 0.9)
    is outside in x and z, e = (0.3, -0.2, 0.3), distance sqrt(0.18) - 0.05.
    The l3 sphere at (0, 0, 0.6) is level with the top of the box, e =
    (0.3, -0.2, -0.05), distance 0.3 - 0.05 = 0.25, which is the nearer one.
    Capsule along x through (0, 0, 1.1) at height 1.1 offset in y by 0.4:
    distance from the ee sphere is 0.4 - 0.1 - 0.05.
    """
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(1, 6, dtype=F64)
    box = Box(center=(0.5, 0.0, 0.55), half_extents=(0.2, 0.2, 0.05))
    assert torch.allclose(distance_to_world(model, q, box),
                          torch.tensor([0.25], dtype=F64), atol=1e-12)
    pipe = Capsule(a=(-1.0, 0.4, 1.1), b=(1.0, 0.4, 1.1), radius=0.1)
    assert torch.allclose(distance_to_world(model, q, pipe),
                          torch.tensor([0.25], dtype=F64), atol=1e-12)


def test_two_link_wall_hand_computed():
    """A planar arm and a wall at x = 2, solid on the far side.

    normal (-1, 0, 0) with offset 2 makes the free side x < 2, so the distance
    of a point is 2 - x. At q = 0 the l2 sphere sits at (1, 0, 0) with radius
    0.3, giving 2 - 1 - 0.3 = 0.7. Folding the arm back (q = [pi, 0]) puts it
    at (-1, 0, 0) for 2 + 1 - 0.3 = 2.7.
    """
    chain = _two_link()
    model = SphereModel(chain, {chain.link_index["l2"]: [(0.0, 0.0, 0.0, 0.3)]})
    wall = HalfSpace(normal=(-1.0, 0.0, 0.0), offset=2.0)
    q = torch.tensor([[0.0, 0.0], [math.pi, 0.0]], dtype=F64)
    d = distance_to_world(model, q, wall)
    assert torch.allclose(d, torch.tensor([0.7, 2.7], dtype=F64), atol=1e-9)


def test_floor_goes_negative_only_when_a_sphere_dips_below():
    """Sweep joint 2 of SIX_DOF over a floor at z=0.

    Only joint 2 moves, about +y, so the link origins downstream of it are at
    z = 0.3 + L cos(q2) with L in {0.3, 0.6, 0.7, 0.8} (l2 itself stays at
    0.3). The lowest sphere center is therefore min(0.3, 0.3 + 0.8 cos q2) and
    the reported distance must be that minus the sphere radius. That closed
    form is written out here rather than read back from fk, and the sign of it
    is exactly "a sphere dips below the floor".
    """
    chain = _six_dof()
    r = 0.05
    model = _arm_model(chain, radius=r)
    q2 = torch.linspace(-2.0, 2.0, 41, dtype=F64)
    q = torch.zeros(q2.shape[0], 6, dtype=F64)
    q[:, 1] = q2

    lowest = torch.minimum(torch.full_like(q2, 0.3), 0.3 + 0.8 * torch.cos(q2))
    want = lowest - r
    got = distance_to_world(model, q, HalfSpace.floor(0.0))
    assert torch.allclose(got, want, atol=1e-12)

    dips = (model.centers_world(q)[:, :, 2] - r).min(dim=1).values < 0
    assert torch.equal(got < 0, dips)
    assert bool(dips.any()) and not bool(dips.all())   # the sweep covers both


def test_in_collision_and_margin():
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(1, 6, dtype=F64)
    floor = HalfSpace.floor(0.0)          # clearance 0.25 at this pose
    assert not bool(in_collision(model, q, floor)[0])
    assert not bool(in_collision(model, q, floor, margin=0.2)[0])
    assert bool(in_collision(model, q, floor, margin=0.3)[0])
    assert bool(in_collision(model, q, HalfSpace.floor(0.5))[0])


def test_world_distances_shape_and_ordering():
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(2, 6, dtype=F64)
    world = [HalfSpace.floor(0.0), Sphere(center=(0.0, 0.0, 2.0), radius=0.5)]
    d = world_distances(model, q, world)
    assert d.shape == (2, model.n, 2)
    # column 0 is the floor: heights 0.3 .. 1.1 minus the sphere radius
    assert torch.allclose(d[0, :, 0],
                          torch.tensor([0.25, 0.55, 0.85, 0.95, 1.05], dtype=F64),
                          atol=1e-9)
    # column 1 is the ball at z=2: 2 - z - 0.5 - 0.05
    assert torch.allclose(d[0, :, 1],
                          torch.tensor([1.15, 0.85, 0.55, 0.45, 0.35], dtype=F64),
                          atol=1e-9)
    assert torch.allclose(d.reshape(2, -1).min(dim=1).values,
                          distance_to_world(model, q, world), atol=1e-12)


def test_min_over_world_is_min_over_shapes():
    chain = _six_dof()
    model = _arm_model(chain)
    torch.manual_seed(11)
    q = -1.0 + 2.0 * torch.rand(6, 6, dtype=F64)
    world = [HalfSpace.floor(-0.2),
             Box(center=(0.6, 0.0, 0.5), half_extents=(0.2, 0.3, 0.4)),
             Capsule(a=(-0.5, 0.5, 0.0), b=(-0.5, 0.5, 1.5), radius=0.08),
             Sphere(center=(0.0, -0.6, 0.8), radius=0.15)]
    each = torch.stack([distance_to_world(model, q, s) for s in world])
    assert torch.allclose(distance_to_world(model, q, world),
                          each.min(dim=0).values, atol=1e-12)


def test_clearance_cost_counts_every_violated_pair():
    """The hinge cost sees all violations, the minimum only sees the worst one."""
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(1, 6, dtype=F64)
    floor = HalfSpace.floor(0.0)
    # heights 0.3, 0.6, 0.9, 1.0, 1.1 minus r=0.05 -> 0.25, 0.55, 0.85, 0.95, 1.05
    margin = 0.6
    want = ((margin - 0.25) ** 2 + (margin - 0.55) ** 2)
    got = world_clearance_cost(model, q, floor, margin=margin)
    assert torch.allclose(got, torch.tensor([want], dtype=F64), atol=1e-12)
    # nothing within the margin: zero cost
    assert torch.allclose(world_clearance_cost(model, q, floor, margin=0.1),
                          torch.zeros(1, dtype=F64), atol=1e-12)


# --------------------------------------------------------------- gradients

def _grad_wrt_q(model, q, world):
    q = q.clone().requires_grad_(True)
    d = distance_to_world(model, q, world)
    return torch.autograd.grad(d.sum(), q)[0]


def test_gradients_wrt_q_are_finite_and_nonzero():
    chain = _six_dof()
    model = _arm_model(chain)
    torch.manual_seed(5)
    q = -1.0 + 2.0 * torch.rand(4, 6, dtype=F64)
    world = [HalfSpace.floor(-0.3),
             Box(center=(0.5, 0.2, 0.4), half_extents=(0.15, 0.15, 0.3)),
             Capsule(a=(-0.4, -0.4, 0.0), b=(0.4, -0.4, 1.2), radius=0.07),
             Sphere(center=(0.0, 0.5, 0.9), radius=0.1)]
    g = _grad_wrt_q(model, q, world)
    assert g.shape == q.shape
    assert torch.isfinite(g).all()
    assert g.abs().sum() > 0


def test_gradient_matches_central_differences():
    """The analytic gradient against a float64 central difference of the same
    scalar. Each obstacle is placed so one sphere-shape pair is clearly the
    nearest, which keeps the minimum differentiable at the test point."""
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.tensor([[0.2, 0.6, -0.4, 0.3, 0.5, -0.2]], dtype=F64)
    for world in (HalfSpace.floor(-0.2),
                  Box(center=(0.9, 0.0, 0.3), half_extents=(0.2, 0.2, 0.2)),
                  Capsule(a=(0.8, -0.5, 0.6), b=(0.8, 0.5, 0.6), radius=0.05),
                  Sphere(center=(0.7, 0.0, 0.9), radius=0.1)):
        g = _grad_wrt_q(model, q, world)
        eps = 1e-6
        for k in range(6):
            dq = torch.zeros_like(q)
            dq[:, k] = eps
            plus = distance_to_world(model, q + dq, world)
            minus = distance_to_world(model, q - dq, world)
            fd = (plus - minus) / (2 * eps)
            assert torch.allclose(g[:, k], fd, atol=1e-6), f"{world} col {k}"


def test_gradients_stay_finite_at_degenerate_contacts():
    """A sphere center exactly on a shape is where a naive norm gives 0/0.

    At q=0 the ee sphere sits at (0, 0, 1.1). Put a capsule axis, a box center,
    a ball center and a plane exactly through that point and the distance is
    still finite and so is its gradient; nothing here may produce a NaN.
    """
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(1, 6, dtype=F64)
    for world in (Capsule(a=(-1.0, 0.0, 1.1), b=(1.0, 0.0, 1.1), radius=0.2),
                  Box(center=(0.0, 0.0, 1.1), half_extents=(0.3, 0.3, 0.3)),
                  Sphere(center=(0.0, 0.0, 1.1), radius=0.2),
                  HalfSpace.floor(1.1),
                  Box(center=(0.0, 0.0, 1.4), half_extents=(0.3, 0.3, 0.3))):
        d = distance_to_world(model, q, world)
        assert torch.isfinite(d).all()
        g = _grad_wrt_q(model, q, world)
        assert torch.isfinite(g).all(), f"non-finite gradient for {world}"


def test_shape_distance_at_zero_offset_is_accurate():
    """The epsilon under the square root must not show up in the value."""
    p = _pts([1.0, 2.0, 3.0])
    assert abs(Sphere(center=(1.0, 2.0, 3.0), radius=0.4).signed_distance(p).item()
               + 0.4) < 1e-14
    cap = Capsule(a=(1.0, 2.0, 3.0), b=(1.0, 2.0, 4.0), radius=0.25)
    assert abs(cap.signed_distance(p).item() + 0.25) < 1e-14


# ------------------------------------------------- batching, dtype, device

def test_per_batch_element_obstacles():
    """A shape parameter with a leading batch dimension moves per configuration."""
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(3, 6, dtype=F64)
    heights = torch.tensor([0.0, 0.1, 0.2], dtype=F64)
    moving_floor = HalfSpace(normal=(0.0, 0.0, 1.0), offset=-heights)
    got = distance_to_world(model, q, moving_floor)
    assert torch.allclose(got, torch.tensor([0.25, 0.15, 0.05], dtype=F64), atol=1e-12)

    centers = torch.tensor([[0.6, 0.0, 0.6], [0.9, 0.0, 0.6], [1.2, 0.0, 0.6]], dtype=F64)
    moving_box = Box(center=centers, half_extents=(0.1, 0.1, 0.1))
    got = distance_to_world(model, q, moving_box)
    one_at_a_time = torch.cat([
        distance_to_world(model, q[i:i + 1], Box(center=centers[i],
                                                 half_extents=(0.1, 0.1, 0.1)))
        for i in range(3)])
    assert torch.allclose(got, one_at_a_time, atol=1e-12)
    assert got[0] < got[1] < got[2]


def test_working_dtype_follows_q():
    """float32 q on a float64 chain gives a float32 answer that agrees."""
    chain = _six_dof(dtype=F64)
    model = _arm_model(chain)
    world = [HalfSpace.floor(0.0),
             Box(center=(0.4, 0.0, 0.5), half_extents=(0.1, 0.2, 0.3)),
             Capsule(a=(0.0, 0.4, 0.0), b=(0.0, 0.4, 1.0), radius=0.05)]
    torch.manual_seed(2)
    q64 = -0.8 + 1.6 * torch.rand(5, 6, dtype=F64)
    q32 = q64.to(torch.float32)
    d64 = distance_to_world(model, q64, world)
    d32 = distance_to_world(model, q32, world)
    assert d64.dtype == torch.float64 and d32.dtype == torch.float32
    assert torch.allclose(d32.to(F64), d64, atol=1e-5)


def test_float32_chain_with_float64_query():
    """The reverse direction: the chain was compiled in float32, q is float64."""
    chain = _six_dof(dtype=torch.float32)
    model = _arm_model(chain)
    q = torch.zeros(1, 6, dtype=F64)
    d = distance_to_world(model, q, HalfSpace.floor(0.0))
    assert d.dtype == torch.float64
    assert torch.allclose(d, torch.tensor([0.25], dtype=F64), atol=1e-6)


def test_tensor_parameters_are_cast_not_required_to_match():
    """Shape parameters given as float32 tensors work against a float64 query."""
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(1, 6, dtype=F64)
    box = Box(center=torch.tensor([0.5, 0.0, 0.55], dtype=torch.float32),
              half_extents=torch.tensor([0.2, 0.2, 0.05], dtype=torch.float32))
    d = distance_to_world(model, q, box)
    assert d.dtype == torch.float64
    assert torch.allclose(d, torch.tensor([0.25], dtype=F64), atol=1e-7)


# ------------------------------------------------------------ world plumbing

def test_single_shape_and_list_agree():
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(1, 6, dtype=F64)
    floor = HalfSpace.floor(0.0)
    assert torch.allclose(distance_to_world(model, q, floor),
                          distance_to_world(model, q, [floor]), atol=1e-12)
    assert as_shapes(floor) == [floor]
    assert len(as_shapes((floor, floor))) == 2


def test_empty_world_is_infinitely_far():
    chain = _six_dof()
    model = _arm_model(chain)
    q = torch.zeros(2, 6, dtype=F64)
    d = distance_to_world(model, q, [])
    assert d.shape == (2,) and torch.isinf(d).all() and (d > 0).all()
    assert world_distances(model, q, []).shape == (2, model.n, 0)
    assert torch.allclose(world_clearance_cost(model, q, []),
                          torch.zeros(2, dtype=F64), atol=1e-12)
    assert not in_collision(model, q, []).any()


def test_shape_distances_accepts_bare_points():
    """The primitives work on any (..., 3) block, not only on fk output."""
    grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 4, dtype=F64),
                                      torch.linspace(-1, 1, 3, dtype=F64),
                                      indexing="ij"), dim=-1)
    pts = torch.cat([grid, torch.full((4, 3, 1), 0.5, dtype=F64)], dim=-1)
    d = shape_distances(pts, [HalfSpace.floor(0.0), Sphere(center=(0, 0, 0), radius=1.0)])
    assert d.shape == (4, 3, 2)
    assert torch.allclose(d[..., 0], torch.full((4, 3), 0.5, dtype=F64), atol=1e-12)


def test_bad_world_entry_is_rejected():
    with pytest.raises(TypeError):
        as_shapes([HalfSpace.floor(0.0), "a wall"])


def test_bad_parameter_shape_is_rejected():
    p = _pts([0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        Box(center=(0.0, 0.0), half_extents=(1.0, 1.0, 1.0)).signed_distance(p)
    with pytest.raises(ValueError):
        Sphere(center=(0.0, 0.0, 0.0),
               radius=torch.zeros(2, 2)).signed_distance(p)
