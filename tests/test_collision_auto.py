# tests/test_collision_auto.py
"""Automatic sphere models from URDF collision primitives.

The two properties that matter are checked against independently written
oracles: points sampled inside each primitive (sampled with plain math here,
not with library code) must land inside some sphere, and every sphere center
must sit inside the primitive it came from. The rotation used to move between
the geometry frame and the link frame is rebuilt by hand in this file so the
test does not lean on kinfast.transforms.
"""
import math
import torch

from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.collision import distance_to_obstacles
from kinfast.collision_auto import (auto_spheres, auto_sphere_model,
                                    spheres_for_geometry, unsupported_links)
from tests.test_parse import TWO_LINK

D = torch.float64

SHAPES = """
<robot name="shapes">
  <link name="base">
    <collision>
      <origin xyz="0.1 -0.2 0.3" rpy="0.3 -0.4 0.5"/>
      <geometry><sphere radius="0.12"/></geometry>
    </collision>
  </link>
  <link name="rod">
    <collision>
      <origin xyz="0 0.05 0.25" rpy="0.7 0.2 -0.4"/>
      <geometry><cylinder radius="0.06" length="0.5"/></geometry>
    </collision>
  </link>
  <link name="pad">
    <collision>
      <origin xyz="-0.3 0.1 0.02" rpy="0.0 0.9 0.2"/>
      <geometry><box size="0.4 0.2 0.1"/></geometry>
    </collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="rod"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="2.0" velocity="1.0" effort="10"/>
  </joint>
  <joint name="j2" type="prismatic">
    <parent link="rod"/><child link="pad"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.2" upper="0.2" velocity="1.0" effort="10"/>
  </joint>
</robot>
"""

MESHY = """
<robot name="meshy">
  <link name="base">
    <collision><geometry><mesh filename="base.stl"/></geometry></collision>
  </link>
  <link name="tip">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><box size="0.2 0.2 0.2"/></geometry>
    </visual>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="tip"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/>
  </joint>
</robot>
"""


def _rpy(rpy):
    """Extrinsic X-Y-Z rotation, written out here as an independent oracle."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=D)
    Ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=D)
    Rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=D)
    return Rz @ Ry @ Rx


def _chain(urdf=SHAPES, dtype=D):
    ir = parse_urdf_string(urdf)
    return ir, compile_robot(ir, dtype=dtype)


def _tensor(sl):
    """[(x,y,z,r)] -> centers (S,3), radii (S,)."""
    t = torch.tensor(sl, dtype=D)
    return t[:, :3], t[:, 3]


def _to_link_frame(points, geom):
    """Geometry-frame points (N,3) -> link frame, via the hand-built rotation."""
    R = _rpy(geom.origin_rpy)
    t = torch.tensor(geom.origin_xyz, dtype=D)
    return points @ R.transpose(0, 1) + t


def _to_geom_frame(points, geom):
    """Inverse of `_to_link_frame`."""
    R = _rpy(geom.origin_rpy)
    t = torch.tensor(geom.origin_xyz, dtype=D)
    return (points - t) @ R


def _sample_sphere(radius, n, g):
    d = torch.randn(n, 3, generator=g, dtype=D)
    d = d / d.norm(dim=-1, keepdim=True)
    u = torch.rand(n, 1, generator=g, dtype=D) ** (1.0 / 3.0)
    interior = d * u * radius
    surface = d * radius                      # worst case: the outer skin
    return torch.cat([interior, surface], dim=0)


def _sample_cylinder(radius, length, n, g):
    theta = torch.rand(n, generator=g, dtype=D) * (2 * math.pi)
    rho = radius * torch.sqrt(torch.rand(n, generator=g, dtype=D))
    z = (torch.rand(n, generator=g, dtype=D) - 0.5) * length
    inside = torch.stack([rho * torch.cos(theta), rho * torch.sin(theta), z], dim=-1)
    # The hard points are the two rim circles, farthest from every axis center.
    rim_t = torch.linspace(0, 2 * math.pi, 32, dtype=D)
    rc, rs = radius * torch.cos(rim_t), radius * torch.sin(rim_t)
    top = torch.stack([rc, rs, torch.full_like(rc, 0.5 * length)], dim=-1)
    bot = torch.stack([rc, rs, torch.full_like(rc, -0.5 * length)], dim=-1)
    return torch.cat([inside, top, bot], dim=0)


def _sample_box(size, n, g):
    s = torch.tensor(size, dtype=D)
    inside = (torch.rand(n, 3, generator=g, dtype=D) - 0.5) * s
    corners = torch.tensor([[sx, sy, sz]
                            for sx in (-0.5, 0.5)
                            for sy in (-0.5, 0.5)
                            for sz in (-0.5, 0.5)], dtype=D) * s
    return torch.cat([inside, corners], dim=0)


def _samples_for(geom, g, n=4000):
    if geom.kind == "sphere":
        return _sample_sphere(geom.size[0], n, g)
    if geom.kind == "cylinder":
        return _sample_cylinder(geom.size[0], geom.size[1], n, g)
    return _sample_box(geom.size, n, g)


# --- coverage -------------------------------------------------------------

def test_sampled_points_are_covered():
    ir, chain = _chain()
    g = torch.Generator().manual_seed(7)
    spheres = auto_spheres(ir, chain, spacing=0.05)
    assert set(spheres) == {chain.link_index[n] for n in ("base", "rod", "pad")}
    for name in ("base", "rod", "pad"):
        geom = ir.links[name].collision
        centers, radii = _tensor(spheres[chain.link_index[name]])
        pts = _to_link_frame(_samples_for(geom, g), geom)
        sd = (pts[:, None, :] - centers[None]).norm(dim=-1) - radii[None]
        worst = sd.min(dim=1).values.max()
        assert worst <= 1e-12, f"{name} leaves points uncovered by {float(worst)}"


def test_coverage_survives_a_capped_axis():
    """A cap makes fewer, fatter spheres; it must never open a hole."""
    ir, chain = _chain()
    g = torch.Generator().manual_seed(11)
    geom = ir.links["pad"].collision
    sl = spheres_for_geometry(geom, spacing=0.001, max_per_axis=3)
    assert len(sl) == 27
    centers, radii = _tensor(sl)
    pts = _to_link_frame(_samples_for(geom, g), geom)
    sd = (pts[:, None, :] - centers[None]).norm(dim=-1) - radii[None]
    assert sd.min(dim=1).values.max() <= 1e-12


def test_finer_spacing_means_more_and_tighter_spheres():
    ir, chain = _chain()
    coarse = auto_spheres(ir, chain, spacing=0.2, max_per_axis=64)
    fine = auto_spheres(ir, chain, spacing=0.02, max_per_axis=64)
    rod = chain.link_index["rod"]
    assert len(fine[rod]) > len(coarse[rod])
    assert max(s[3] for s in fine[rod]) < max(s[3] for s in coarse[rod])
    # the sphere link is exact at any spacing: one sphere, the original radius
    base = chain.link_index["base"]
    assert len(coarse[base]) == len(fine[base]) == 1
    assert abs(coarse[base][0][3] - 0.12) < 1e-15


# --- centers stay inside the primitive ------------------------------------

def test_centers_lie_inside_their_primitive():
    ir, chain = _chain()
    spheres = auto_spheres(ir, chain, spacing=0.03)
    for name in ("base", "rod", "pad"):
        geom = ir.links[name].collision
        centers, _ = _tensor(spheres[chain.link_index[name]])
        local = _to_geom_frame(centers, geom)
        if geom.kind == "sphere":
            assert float(local.norm(dim=-1).max()) <= geom.size[0] + 1e-12
        elif geom.kind == "cylinder":
            radius, length = geom.size
            assert float(local[:, :2].norm(dim=-1).max()) <= radius + 1e-12
            assert float(local[:, 2].abs().max()) <= 0.5 * length + 1e-12
        else:
            half = torch.tensor(geom.size, dtype=D) * 0.5
            assert bool((local.abs() <= half + 1e-12).all())


# --- hand-computed sphere sets --------------------------------------------

def test_cylinder_layout_hand_computed():
    """radius 0.1, length 1.0, spacing 0.5 -> 2 slabs of half-height 0.25.

    Centers land at z = -0.25 and +0.25, radius sqrt(0.1^2 + 0.25^2).
    """
    from kinfast.ir import Geometry
    geom = Geometry("cylinder", None, (1.0, 1.0, 1.0), (0.1, 1.0))
    sl = spheres_for_geometry(geom, spacing=0.5)
    assert len(sl) == 2
    want_r = math.sqrt(0.1 ** 2 + 0.25 ** 2)
    for (x, y, z, r), want_z in zip(sl, (-0.25, 0.25)):
        assert abs(x) < 1e-15 and abs(y) < 1e-15
        assert abs(z - want_z) < 1e-15
        assert abs(r - want_r) < 1e-15


def test_box_layout_hand_computed():
    """0.4 x 0.2 x 0.1 at spacing 0.2 -> 2x1x1 cells of 0.2 x 0.2 x 0.1.

    Half-diagonal is 0.5 * sqrt(0.04 + 0.04 + 0.01) = 0.15.
    """
    from kinfast.ir import Geometry
    geom = Geometry("box", None, (1.0, 1.0, 1.0), (0.4, 0.2, 0.1))
    sl = spheres_for_geometry(geom, spacing=0.2)
    assert len(sl) == 2
    xs = sorted(s[0] for s in sl)
    assert abs(xs[0] + 0.1) < 1e-15 and abs(xs[1] - 0.1) < 1e-15
    for s in sl:
        assert abs(s[1]) < 1e-15 and abs(s[2]) < 1e-15
        assert abs(s[3] - 0.15) < 1e-15


def test_geometry_origin_is_applied():
    """A sphere primitive offset and rotated ends up exactly at R @ c + t."""
    from kinfast.ir import Geometry
    geom = Geometry("sphere", None, (1.0, 1.0, 1.0), (0.05,),
                    (0.1, 0.2, 0.3), (0.4, -0.5, 0.6))
    sl = spheres_for_geometry(geom)
    got = torch.tensor(sl[0][:3], dtype=D)
    want = _rpy(geom.origin_rpy) @ torch.zeros(3, dtype=D) + torch.tensor(
        geom.origin_xyz, dtype=D)
    assert torch.allclose(got, want, atol=1e-15)


def test_padding_only_inflates_radii():
    ir, chain = _chain()
    plain = auto_spheres(ir, chain, spacing=0.05)
    padded = auto_spheres(ir, chain, spacing=0.05, padding=0.01)
    for k in plain:
        assert len(plain[k]) == len(padded[k])
        for a, b in zip(plain[k], padded[k]):
            assert a[:3] == b[:3]
            assert abs((b[3] - a[3]) - 0.01) < 1e-15


# --- degenerate and missing geometry --------------------------------------

def test_robot_without_collision_geometry_gives_empty_model():
    ir, chain = _chain(TWO_LINK)
    assert auto_spheres(ir, chain) == {}
    model = auto_sphere_model(ir, chain)
    assert model.n == 0
    C = model.centers_world(torch.zeros(3, 2, dtype=D))
    assert C.shape == (3, 0, 3)
    assert unsupported_links(ir, chain) == {"base": "none", "l1": "none", "l2": "none"}


def test_meshes_are_skipped_and_reported():
    ir, chain = _chain(MESHY)
    assert auto_spheres(ir, chain) == {}
    assert unsupported_links(ir, chain) == {"base": "mesh", "tip": "none"}


def test_visual_fallback_is_opt_in():
    ir, chain = _chain(MESHY)
    spheres = auto_spheres(ir, chain, spacing=0.2, fallback_to_visual=True)
    assert set(spheres) == {chain.link_index["tip"]}
    # 0.2 cube at spacing 0.2 is a single cell, half-diagonal 0.1 * sqrt(3),
    # centered on the visual origin (0, 0, 0.1).
    (x, y, z, r), = spheres[chain.link_index["tip"]]
    assert abs(x) < 1e-15 and abs(y) < 1e-15 and abs(z - 0.1) < 1e-15
    assert abs(r - 0.1 * math.sqrt(3.0)) < 1e-15
    assert unsupported_links(ir, chain, fallback_to_visual=True) == {"base": "mesh"}


def test_zero_sized_primitives_are_dropped():
    from kinfast.ir import Geometry
    assert spheres_for_geometry(Geometry("sphere", None, (1, 1, 1), (0.0,))) == []
    assert spheres_for_geometry(Geometry("box", None, (1, 1, 1), (0.0, 0.0, 0.0))) == []
    assert spheres_for_geometry(Geometry("cylinder", None, (1, 1, 1), (0.0, 0.0))) == []
    assert spheres_for_geometry(Geometry("mesh", "a.stl")) == []
    assert spheres_for_geometry(None) == []


def test_degenerate_primitive_is_reported():
    urdf = """
    <robot name="flat">
      <link name="base">
        <collision><geometry><box size="0 0 0"/></geometry></collision>
      </link>
    </robot>
    """
    ir, chain = _chain(urdf)
    assert auto_spheres(ir, chain) == {}
    assert unsupported_links(ir, chain) == {"base": "degenerate"}


def test_bad_arguments_raise():
    from kinfast.ir import Geometry
    geom = Geometry("box", None, (1, 1, 1), (0.2, 0.2, 0.2))
    for kw in ({"spacing": 0.0}, {"max_per_axis": 0}, {"padding": -0.1}):
        try:
            spheres_for_geometry(geom, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kw}")


# --- the model behaves like the rest of the library -----------------------

def test_centers_world_matches_hand_computed_pose():
    """At the zero configuration the rod frame sits at (0, 0, 0.1).

    The rod cylinder is split into slabs; the slab centers in the link frame
    are R(rpy) @ (0, 0, z_k) + (0, 0.05, 0.25), and world = link origin + that.
    """
    ir, chain = _chain()
    model = auto_sphere_model(ir, chain, spacing=0.5)   # one sphere per shape
    C = model.centers_world(torch.zeros(1, 2, dtype=D))[0]
    geom = ir.links["rod"].collision
    want = _to_link_frame(torch.zeros(1, 3, dtype=D), geom)[0] + torch.tensor(
        [0.0, 0.0, 0.1], dtype=D)
    rod = chain.link_index["rod"]
    picked = C[model.link == rod]
    assert picked.shape[0] == 1
    assert torch.allclose(picked[0], want, atol=1e-12)


def test_dtype_and_device_follow_q():
    ir, chain32 = _chain(dtype=torch.float32)
    model = auto_sphere_model(ir, chain32, spacing=0.05)
    q64 = torch.zeros(2, 2, dtype=torch.float64)
    assert model.centers_world(q64).dtype == torch.float64
    q32 = torch.zeros(2, 2, dtype=torch.float32)
    assert model.centers_world(q32).dtype == torch.float32


def test_batched_and_differentiable_through_obstacles():
    ir, chain = _chain()
    model = auto_sphere_model(ir, chain, spacing=0.05)
    g = torch.Generator().manual_seed(3)
    q = (torch.rand(4, 2, generator=g, dtype=D) - 0.5).requires_grad_(True)
    obs_c = torch.tensor([[0.4, 0.1, 0.2]], dtype=D)
    obs_r = torch.tensor([0.05], dtype=D)
    d = distance_to_obstacles(model, q, obs_c, obs_r)
    assert d.shape == (4,)
    d.sum().backward()
    assert torch.isfinite(q.grad).all()
    assert float(q.grad.abs().max()) > 0.0


def test_gradient_matches_finite_differences():
    """float64 central differences on the obstacle distance, as an oracle."""
    ir, chain = _chain()
    model = auto_sphere_model(ir, chain, spacing=0.08)
    obs_c = torch.tensor([[0.35, 0.15, 0.25]], dtype=D)
    obs_r = torch.tensor([0.04], dtype=D)
    q0 = torch.tensor([[0.31, 0.07]], dtype=D)

    q = q0.clone().requires_grad_(True)
    distance_to_obstacles(model, q, obs_c, obs_r).sum().backward()
    analytic = q.grad[0].clone()

    eps = 1e-6
    for i in range(2):
        qp, qm = q0.clone(), q0.clone()
        qp[0, i] += eps
        qm[0, i] -= eps
        fd = (distance_to_obstacles(model, qp, obs_c, obs_r)
              - distance_to_obstacles(model, qm, obs_c, obs_r)) / (2 * eps)
        assert abs(float(fd) - float(analytic[i])) < 1e-6


def test_real_urdf_if_present():
    """Bonus end-to-end check on a shipped robot; the assets are gitignored.

    a1 is a quadruped whose collision shapes are boxes and cylinders, so it
    exercises the whole path on a model nobody wrote for this test.
    """
    import os
    path = "C:/Users/vihan/urdf-doctor/examples/assets/gallery/a1.urdf"
    if not os.path.exists(path):
        return
    from kinfast.robot import load
    r = load(path)
    ir, chain = r.ir, r.chain
    spheres = auto_spheres(ir, chain, spacing=0.08)
    assert spheres, "a1 should yield spheres from its primitive collision shapes"
    for name, sl in ((n, spheres[chain.link_index[n]]) for n in ir.links
                     if chain.link_index.get(n) in spheres):
        geom = ir.links[name].collision
        local = _to_geom_frame(_tensor(sl)[0], geom)
        if geom.kind == "sphere":
            assert float(local.norm(dim=-1).max()) <= geom.size[0] + 1e-9
        elif geom.kind == "cylinder":
            assert float(local[:, :2].norm(dim=-1).max()) <= geom.size[0] + 1e-9
            assert float(local[:, 2].abs().max()) <= 0.5 * geom.size[1] + 1e-9
        else:
            half = torch.tensor(geom.size, dtype=D) * 0.5
            assert bool((local.abs() <= half + 1e-9).all())
    model = auto_sphere_model(ir, chain, spacing=0.08)
    C = model.centers_world(torch.zeros(2, chain.dof, dtype=D))
    assert C.shape == (2, model.n, 3)
    assert torch.isfinite(C).all()
