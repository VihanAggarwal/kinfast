# tests/test_inertia.py
"""Mass properties from primitive geometry.

Oracles are deliberately outside the library: closed-form numbers written out by
hand, Gauss-Legendre volume integrals that reach machine precision on these
shapes, and MuJoCo's own geom-to-body inertia compiler where it is installed.
"""
import math

import numpy as np
import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.dynamics import gravity
from kinfast.inertia import (DEFAULT_DENSITY, MassProperties, as_inertial,
                             box_inertia, capsule_inertia, combine_mass_properties,
                             cylinder_inertia, fill_missing_inertials,
                             geometry_mass_properties, link_mass_properties,
                             primitive_mass_properties, rotate_inertia,
                             shift_inertia, sphere_inertia)
from kinfast.ir import Geometry, Link, Robot
from kinfast.transforms import rpy_to_matrix
from kinfast.urdf.parse import parse_urdf_string
from kinfast.urdf.repair import repair

try:
    import mujoco as _mj
except Exception:                                    # pragma: no cover
    _mj = None

needs_mujoco = pytest.mark.skipif(_mj is None, reason="mujoco not installed")

F64 = dict(dtype=torch.float64)


# ----------------------------------------------------------- hand formulas --

def test_box_matches_hand_computed_numbers():
    """A 0.2 x 0.4 x 0.6 box of aluminium, worked out on paper."""
    rho = 2700.0
    x, y, z = 0.2, 0.4, 0.6
    m, c, I = box_inertia((x, y, z), rho, dtype=torch.float64)
    assert float(m) == pytest.approx(rho * x * y * z)
    assert float(m) == pytest.approx(129.6)
    assert torch.allclose(c, torch.zeros(3, **F64))
    assert float(I[0, 0]) == pytest.approx(129.6 * (0.16 + 0.36) / 12)
    assert float(I[1, 1]) == pytest.approx(129.6 * (0.04 + 0.36) / 12)
    assert float(I[2, 2]) == pytest.approx(129.6 * (0.04 + 0.16) / 12)
    off = I - torch.diag(torch.diagonal(I))
    assert torch.allclose(off, torch.zeros(3, 3, **F64))


def test_cube_is_isotropic():
    """Every axis of a cube is a principal axis with the same moment m a^2 / 6."""
    a, rho = 0.3, 500.0
    m, _, I = box_inertia((a, a, a), rho, dtype=torch.float64)
    expect = float(m) * a * a / 6.0
    assert torch.allclose(I, expect * torch.eye(3, **F64))


def test_cylinder_matches_hand_computed_numbers():
    r, l, rho = 0.05, 0.4, 7800.0                    # steel rod
    m, c, I = cylinder_inertia(r, l, rho, dtype=torch.float64)
    assert float(m) == pytest.approx(rho * math.pi * r * r * l)
    assert torch.allclose(c, torch.zeros(3, **F64))
    assert float(I[2, 2]) == pytest.approx(float(m) * r * r / 2)
    assert float(I[0, 0]) == pytest.approx(float(m) * (3 * r * r + l * l) / 12)
    assert float(I[1, 1]) == pytest.approx(float(I[0, 0]))


def test_thin_rod_limit():
    """A long thin cylinder tends to the textbook rod, m l^2 / 12."""
    m, _, I = cylinder_inertia(1e-4, 1.0, 1000.0, dtype=torch.float64)
    assert float(I[0, 0]) == pytest.approx(float(m) / 12.0, rel=1e-6)


def test_sphere_matches_hand_computed_numbers():
    r, rho = 0.11, 1180.0
    m, c, I = sphere_inertia(r, rho, dtype=torch.float64)
    assert float(m) == pytest.approx(rho * 4.0 / 3.0 * math.pi * r ** 3)
    assert torch.allclose(c, torch.zeros(3, **F64))
    assert torch.allclose(I, (0.4 * float(m) * r * r) * torch.eye(3, **F64))


def test_capsule_with_zero_length_is_a_sphere():
    ms, _, Is = sphere_inertia(0.07, 900.0, dtype=torch.float64)
    mc, _, Ic = capsule_inertia(0.07, 0.0, 900.0, dtype=torch.float64)
    assert float(mc) == pytest.approx(float(ms))
    assert torch.allclose(Ic, Is)


def test_capsule_is_cylinder_plus_two_hemispheres():
    """Independent construction: build the capsule out of three parts and
    combine them with the parallel axis theorem, using only the sphere and
    cylinder formulas, then compare against the closed form."""
    r, l, rho = 0.08, 0.5, 1200.0
    cyl = cylinder_inertia(r, l, rho, dtype=torch.float64)
    ball = sphere_inertia(r, rho, dtype=torch.float64)
    # A whole sphere split in half: each half carries half the mass and, by
    # symmetry about the centre, half the inertia about the sphere centre.
    # Referring that to the half's own centroid (3r/8 out) makes it a part we
    # can place at either end of the cylinder.
    half_m = ball.mass / 2
    half_I_at_centre = ball.inertia / 2
    d = 3.0 * r / 8.0
    top_c = torch.tensor([0.0, 0.0, l / 2 + d], **F64)
    bot_c = torch.tensor([0.0, 0.0, -(l / 2 + d)], **F64)
    half_I = shift_inertia(half_I_at_centre, half_m,
                           torch.tensor([0.0, 0.0, d], **F64), to_com=True)
    parts = [cyl,
             MassProperties(half_m, top_c, half_I),
             MassProperties(half_m, bot_c, half_I)]
    built = combine_mass_properties(parts)
    closed = capsule_inertia(r, l, rho, dtype=torch.float64)
    assert float(built.mass) == pytest.approx(float(closed.mass))
    assert torch.allclose(built.com, closed.com, atol=1e-12)
    assert torch.allclose(built.inertia, closed.inertia, atol=1e-12)


# ------------------------------------------------- numerical integration ---
#
# Both oracles below are Gauss-Legendre quadrature rather than Monte Carlo. A
# rule with n nodes integrates polynomials up to degree 2n-1 exactly, and every
# integrand here is a low order polynomial on each smooth piece of the domain,
# so these are not approximations that need a loose tolerance: they land on the
# closed forms to the last few bits of a float64. They are also completely
# independent derivations. The revolution rule stacks discs and knows nothing
# about splitting a capsule into a cylinder and two hemispheres; the box rule
# integrates over the transformed body directly and never touches the parallel
# axis theorem.


def _revolution_mass_properties(radius_of_z, breaks, rho, n=8):
    """Mass and inertia of a solid of revolution about z, by quadrature.

    The body is described by its radius R(z). A disc of thickness dz at height
    z has mass rho pi R^2 dz, polar moment (rho pi R^4 / 2) dz about z, and
    diametral moment (rho pi R^4 / 4) dz about its own diameter, which the
    parallel axis theorem carries to the origin as an extra rho pi R^2 z^2 dz.

    `breaks` are the z values that bound the pieces on which R(z) is smooth,
    integrated separately so a kink (where a cap meets a cylinder) does not
    spoil the exactness. Returns (mass, Ixx = Iyy, Izz) about the origin, which
    is the centre of mass for every shape used here since all of them are
    symmetric about z = 0.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    mass = ixx = izz = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        z = 0.5 * (b - a) * x + 0.5 * (a + b)
        dz = 0.5 * (b - a) * w
        r2 = radius_of_z(z) ** 2
        mass += rho * float(np.sum(dz * math.pi * r2))
        izz += rho * float(np.sum(dz * (math.pi / 2) * r2 * r2))
        ixx += rho * float(np.sum(dz * ((math.pi / 4) * r2 * r2 + math.pi * r2 * z * z)))
    return mass, ixx, izz


def test_capsule_against_a_disc_stacking_integral():
    """The capsule is the one formula with a parallel axis shift baked into its
    derivation, so it gets an oracle that reaches it a different way: integrate
    the stack of discs that make up the shape."""
    r, l, rho = 0.12, 0.5, 800.0
    half = l / 2

    def radius(z):
        cap = np.sqrt(np.clip(r * r - (np.abs(z) - half) ** 2, 0.0, None))
        return np.where(np.abs(z) <= half, r, cap)

    m_q, ixx_q, izz_q = _revolution_mass_properties(
        radius, [-(half + r), -half, half, half + r], rho)
    m, c, I = capsule_inertia(r, l, rho, dtype=torch.float64)
    assert float(m) == pytest.approx(m_q, rel=1e-13)
    assert torch.allclose(c, torch.zeros(3, **F64))
    assert float(I[0, 0]) == pytest.approx(ixx_q, rel=1e-13)
    assert float(I[1, 1]) == pytest.approx(ixx_q, rel=1e-13)
    assert float(I[2, 2]) == pytest.approx(izz_q, rel=1e-13)


@pytest.mark.parametrize("kind,size,radius,breaks", [
    ("cylinder", (0.09, 0.44), lambda z, r=0.09: np.full_like(z, r),
     [-0.22, 0.22]),
    ("sphere", (0.17,), lambda z, r=0.17: np.sqrt(np.clip(r * r - z * z, 0.0, None)),
     [-0.17, 0.0, 0.17]),
])
def test_axisymmetric_primitives_against_the_same_integral(kind, size, radius, breaks):
    """The cylinder and the sphere are solids of revolution too, so the same
    quadrature checks them without repeating their closed forms."""
    rho = 1150.0
    m_q, ixx_q, izz_q = _revolution_mass_properties(radius, breaks, rho)
    m, _, I = primitive_mass_properties(kind, size, rho, dtype=torch.float64)
    assert float(m) == pytest.approx(m_q, rel=1e-13)
    assert float(I[0, 0]) == pytest.approx(ixx_q, rel=1e-13)
    assert float(I[2, 2]) == pytest.approx(izz_q, rel=1e-13)


def _rpy_matrix(rr, pp, yy):
    """URDF rpy is extrinsic xyz, so R = Rz(yaw) Ry(pitch) Rx(roll). Written out
    here in numpy so the test does not lean on the library's own version."""
    cr, sr, cp, sp, cy, sy = (math.cos(rr), math.sin(rr), math.cos(pp),
                              math.sin(pp), math.cos(yy), math.sin(yy))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _box_samples(size, xyz, R, rho, n=3):
    """Quadrature points and their mass weights for a box placed at `xyz` with
    rotation `R`, in the frame `xyz` and `R` are expressed in.

    Every entry of the inertia integrand is quadratic in the local coordinates,
    and an n node Gauss rule is exact to degree 2n-1, so three nodes per axis
    reproduce the integral exactly up to round-off.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    half = np.asarray(size, float) / 2.0
    xyz = np.asarray(xyz, float)
    pts, dm = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                u = np.array([half[0] * x[i], half[1] * x[j], half[2] * x[k]])
                pts.append(xyz + R @ u)
                dm.append(rho * half[0] * w[i] * half[1] * w[j] * half[2] * w[k])
    return np.array(pts), np.array(dm)


def _props_from_samples(pts, dm):
    """Mass, centre of mass and inertia about it, straight from the definition
    of the integrals. No closed form and no parallel axis theorem in sight."""
    mass = float(dm.sum())
    com = (dm[:, None] * pts).sum(axis=0) / mass
    I = np.zeros((3, 3))
    for p, d in zip(pts - com, dm):
        I += d * (float(p @ p) * np.eye(3) - np.outer(p, p))
    return mass, com, I


def _box_quadrature(size, xyz, R, rho, n=3):
    pts, dm = _box_samples(size, xyz, R, rho, n)
    return _props_from_samples(pts, dm)


def test_rotated_and_offset_box_against_a_volume_integral():
    """The whole geometry-to-link-frame path: build the primitive, rotate its
    tensor by the origin rpy, and place its centre of mass at the origin xyz."""
    rho = 640.0
    size = (0.3, 0.15, 0.5)
    rpy = (0.4, -0.7, 1.1)
    xyz = (0.2, -0.1, 0.35)
    g = Geometry("box", None, (1.0, 1.0, 1.0), size, xyz, rpy)
    props = geometry_mass_properties(g, rho, dtype=torch.float64)

    m_q, c_q, I_q = _box_quadrature(size, xyz, _rpy_matrix(*rpy), rho)
    assert float(props.mass) == pytest.approx(rho * math.prod(size), rel=1e-13)
    assert float(props.mass) == pytest.approx(m_q, rel=1e-12)
    assert np.allclose(props.com.numpy(), c_q, atol=1e-14)
    assert np.allclose(props.inertia.numpy(), I_q, rtol=1e-11, atol=1e-14)
    # the rotation genuinely coupled the axes, so the check is not vacuous
    assert abs(I_q[1, 2]) > 1e-3


def test_two_offset_boxes_against_a_volume_integral():
    """Summing two primitives onto one link, against the same integral run over
    both boxes at once. This is the parallel axis path with an oracle that has
    never heard of it."""
    rho = 1450.0
    a = Geometry("box", None, (1.0, 1.0, 1.0), (0.2, 0.1, 0.3),
                 (0.05, 0.0, 0.15), (0.0, 0.0, 0.7))
    b = Geometry("box", None, (1.0, 1.0, 1.0), (0.4, 0.12, 0.12),
                 (-0.1, 0.2, 0.0), (0.3, -0.5, 0.0))
    props = link_mass_properties([a, b], rho, dtype=torch.float64)

    samples = [_box_samples(g.size, g.origin_xyz, _rpy_matrix(*g.origin_rpy), rho)
               for g in (a, b)]
    pts = np.concatenate([s[0] for s in samples])
    dm = np.concatenate([s[1] for s in samples])
    mass, com, I = _props_from_samples(pts, dm)
    assert float(props.mass) == pytest.approx(mass, rel=1e-12)
    assert np.allclose(props.com.numpy(), com, atol=1e-14)
    assert np.allclose(props.inertia.numpy(), I, rtol=1e-11, atol=1e-14)


# ----------------------------------------------- rotation and shift rules --

def test_box_rotated_90_about_z_swaps_ixx_and_iyy():
    size = (0.2, 0.6, 0.1)
    upright = Geometry("box", None, (1.0, 1.0, 1.0), size, (0, 0, 0), (0, 0, 0))
    turned = Geometry("box", None, (1.0, 1.0, 1.0), size, (0, 0, 0),
                      (0.0, 0.0, math.pi / 2))
    a = geometry_mass_properties(upright, 2000.0, dtype=torch.float64)
    b = geometry_mass_properties(turned, 2000.0, dtype=torch.float64)
    assert float(a.mass) == pytest.approx(float(b.mass))
    assert float(b.inertia[0, 0]) == pytest.approx(float(a.inertia[1, 1]))
    assert float(b.inertia[1, 1]) == pytest.approx(float(a.inertia[0, 0]))
    assert float(b.inertia[2, 2]) == pytest.approx(float(a.inertia[2, 2]))
    # still diagonal: a quarter turn about z keeps the box axis aligned
    off = b.inertia - torch.diag(torch.diagonal(b.inertia))
    assert torch.allclose(off, torch.zeros(3, 3, **F64), atol=1e-14)


def test_rotation_preserves_trace_and_eigenvalues():
    """R I R^T is a similarity transform, so the principal moments cannot move.
    Catches a transposed rotation or a one-sided multiply."""
    _, _, I = box_inertia((0.2, 0.5, 0.9), 1500.0, dtype=torch.float64)
    rpy = torch.tensor([0.3, 1.2, -0.8], **F64)
    R = rpy_to_matrix(rpy)
    J = rotate_inertia(I, R)
    assert float(torch.trace(J)) == pytest.approx(float(torch.trace(I)))
    assert torch.allclose(torch.linalg.eigvalsh(J), torch.linalg.eigvalsh(I),
                          atol=1e-12)
    assert torch.allclose(J, J.T, atol=1e-14)
    # off-diagonal terms really did appear, so the test is not vacuous
    assert float((J - torch.diag(torch.diagonal(J))).abs().max()) > 1e-3


def test_parallel_axis_two_halves_rebuild_the_whole_box():
    """Split a box down the middle in x. Each half is a box in its own right,
    so combining them exercises the shift with an oracle (the closed form of
    the undivided box) that never touches shift_inertia."""
    rho = 1234.0
    x, y, z = 0.8, 0.3, 0.2
    whole = box_inertia((x, y, z), rho, dtype=torch.float64)
    half = box_inertia((x / 2, y, z), rho, dtype=torch.float64)
    left = MassProperties(half.mass, torch.tensor([-x / 4, 0.0, 0.0], **F64),
                          half.inertia)
    right = MassProperties(half.mass, torch.tensor([x / 4, 0.0, 0.0], **F64),
                           half.inertia)
    built = combine_mass_properties([left, right])
    assert float(built.mass) == pytest.approx(float(whole.mass))
    assert torch.allclose(built.com, torch.zeros(3, **F64), atol=1e-15)
    assert torch.allclose(built.inertia, whole.inertia, atol=1e-12)


def test_shift_inertia_is_its_own_inverse():
    _, _, I = cylinder_inertia(0.1, 0.4, 900.0, dtype=torch.float64)
    m = torch.tensor(3.5, **F64)
    d = torch.tensor([0.1, -0.2, 0.05], **F64)
    out = shift_inertia(I, m, d)
    back = shift_inertia(out, m, d, to_com=True)
    assert torch.allclose(back, I, atol=1e-14)
    # the shift adds m|d|^2 to the trace of the tensor (2 * m |d|^2 by the
    # theorem, halved because the trace of |d|^2 E - d d^T is 2 |d|^2)
    added = float(torch.trace(out) - torch.trace(I))
    assert added == pytest.approx(2 * float(m) * float((d * d).sum()))


def test_shift_accepts_plain_numbers():
    """The mass and the offset are allowed to be python values, so a caller
    reading them out of a URDF does not have to wrap them first."""
    _, _, I = sphere_inertia(0.1, 1000.0, dtype=torch.float64)
    a = shift_inertia(I, 2.0, (0.3, 0.0, -0.1))
    b = shift_inertia(I, torch.tensor(2.0, **F64), torch.tensor([0.3, 0.0, -0.1], **F64))
    assert torch.allclose(a, b, atol=1e-15)


def test_rotate_inertia_broadcasts_one_rotation_over_a_batch():
    sizes = torch.tensor([[0.2, 0.4, 0.6], [0.1, 0.1, 0.9], [0.5, 0.3, 0.2]], **F64)
    _, _, I = box_inertia(sizes, 1000.0)
    R = rpy_to_matrix(torch.tensor([0.2, -0.9, 0.4], **F64))
    J = rotate_inertia(I, R)
    assert J.shape == (3, 3, 3)
    for k in range(3):
        assert torch.allclose(J[k], R @ I[k] @ R.T, atol=1e-14)


def test_shift_direction_does_not_matter():
    _, _, I = sphere_inertia(0.2, 1000.0, dtype=torch.float64)
    m = torch.tensor(2.0, **F64)
    d = torch.tensor([0.3, 0.1, -0.4], **F64)
    assert torch.allclose(shift_inertia(I, m, d), shift_inertia(I, m, -d))


def test_combined_com_is_the_mass_weighted_mean():
    a = MassProperties(torch.tensor(2.0, **F64), torch.tensor([1.0, 0.0, 0.0], **F64),
                       torch.zeros(3, 3, **F64))
    b = MassProperties(torch.tensor(6.0, **F64), torch.tensor([-1.0, 2.0, 0.0], **F64),
                       torch.zeros(3, 3, **F64))
    out = combine_mass_properties([a, b])
    assert float(out.mass) == pytest.approx(8.0)
    assert torch.allclose(out.com, torch.tensor([-0.5, 1.5, 0.0], **F64))
    # two point masses about their common COM: sum m d^2 on the off axes
    assert float(out.inertia[2, 2]) == pytest.approx(2 * (1.5 ** 2 + 1.5 ** 2)
                                                     + 6 * (0.5 ** 2 + 0.5 ** 2))


def test_combine_with_zero_mass_does_not_produce_nan():
    zero = MassProperties(torch.zeros((), **F64), torch.zeros(3, **F64),
                          torch.zeros(3, 3, **F64))
    out = combine_mass_properties([zero, zero])
    assert float(out.mass) == 0.0
    assert torch.isfinite(out.com).all()
    assert torch.isfinite(out.inertia).all()


# ------------------------------------------------------------- MuJoCo -----

def _mj_body(geom_xml):
    """Compile a one-body model and return (mass, com, inertia in body frame).

    MuJoCo stores the inertia diagonalized, so it is rotated back through
    body_iquat to get the tensor in the body frame that we compare against.
    """
    xml = ("<mujoco><compiler angle='radian'/><worldbody><body>"
           "<freejoint/>" + geom_xml + "</body></worldbody></mujoco>")
    model = _mj.MjModel.from_xml_string(xml)
    R = np.zeros(9)
    _mj.mju_quat2Mat(R, model.body_iquat[1])
    R = R.reshape(3, 3)
    I = R @ np.diag(model.body_inertia[1]) @ R.T
    return float(model.body_mass[1]), model.body_ipos[1].copy(), I


@needs_mujoco
@pytest.mark.parametrize("kind,size,mj", [
    ("box", (0.24, 0.4, 0.62), "type='box' size='0.12 0.2 0.31'"),
    ("cylinder", (0.09, 0.44), "type='cylinder' size='0.09 0.22'"),
    ("sphere", (0.17,), "type='sphere' size='0.17'"),
    ("capsule", (0.1, 0.5), "type='capsule' size='0.1 0.25'"),
])
def test_primitives_match_mujoco(kind, size, mj):
    """MuJoCo runs its own volume integrals to turn a geom plus a density into
    body mass and inertia, which makes it a clean external oracle. Note its
    box size is a half extent and its cylinder/capsule size is a half length."""
    rho = 1150.0
    m_mj, c_mj, I_mj = _mj_body(f"<geom {mj} density='{rho}'/>")
    m, c, I = primitive_mass_properties(kind, size, rho, dtype=torch.float64)
    assert float(m) == pytest.approx(m_mj, rel=1e-9)
    assert np.allclose(c.numpy(), c_mj, atol=1e-12)
    assert np.allclose(I.numpy(), I_mj, rtol=1e-9, atol=1e-12)


@needs_mujoco
def test_composite_link_matches_mujoco():
    """Two offset, rotated primitives on one link, against MuJoCo composing the
    same two geoms into one body. Single-axis rotations are used so the URDF rpy
    and the MJCF axisangle mean the same thing without arguing about Euler
    conventions.

    The tolerance is looser here than in the single-primitive test above for a
    reason that is MuJoCo's, not ours. MuJoCo does not keep the body inertia as
    a tensor: it stores three principal moments plus the quaternion of the frame
    they are principal in, so a composite body goes through a closed-form 3x3
    eigendecomposition. That step is only good to roughly 1e-7 of the tensor
    norm. A single primitive stays axis aligned, the quaternion is the identity,
    and nothing is lost, which is why that test can ask for 1e-9.
    """
    rho = 2700.0
    geoms = [
        Geometry("box", None, (1.0, 1.0, 1.0), (0.2, 0.1, 0.3),
                 (0.05, 0.0, 0.15), (0.0, 0.0, math.pi / 3)),
        Geometry("cylinder", None, (1.0, 1.0, 1.0), (0.04, 0.5),
                 (-0.1, 0.2, 0.0), (math.pi / 2, 0.0, 0.0)),
    ]
    props = link_mass_properties(geoms, rho, dtype=torch.float64)
    xml = (f"<geom type='box' size='0.1 0.05 0.15' pos='0.05 0 0.15' "
           f"axisangle='0 0 1 {math.pi / 3}' density='{rho}'/>"
           f"<geom type='cylinder' size='0.04 0.25' pos='-0.1 0.2 0' "
           f"axisangle='1 0 0 {math.pi / 2}' density='{rho}'/>")
    m_mj, c_mj, I_mj = _mj_body(xml)
    assert float(props.mass) == pytest.approx(m_mj, rel=1e-12)
    assert np.allclose(props.com.numpy(), c_mj, atol=1e-14)
    scale = float(np.abs(I_mj).max())
    assert np.allclose(props.inertia.numpy(), I_mj, rtol=0.0, atol=1e-6 * scale)
    # the trace is an invariant of the diagonalization, so it survives intact
    assert float(props.inertia.trace()) == pytest.approx(float(np.trace(I_mj)),
                                                         rel=1e-12)


@needs_mujoco
def test_filled_urdf_link_matches_mujoco():
    """End to end: parse a URDF, fill the missing inertial, and check the numbers
    a physics engine would have computed from the same geometry."""
    rho = 1800.0
    urdf = """
    <robot name="one">
      <link name="base">
        <collision>
          <origin xyz="0 0 0.2" rpy="0 0 0"/>
          <geometry><cylinder radius="0.06" length="0.4"/></geometry>
        </collision>
      </link>
    </robot>
    """
    ir, findings = fill_missing_inertials(parse_urdf_string(urdf), rho)
    inr = ir.links["base"].inertial
    m_mj, c_mj, I_mj = _mj_body(
        f"<geom type='cylinder' size='0.06 0.2' pos='0 0 0.2' density='{rho}'/>")
    assert inr.mass == pytest.approx(m_mj, rel=1e-9)
    assert np.allclose(np.array(inr.com), c_mj, atol=1e-12)
    ixx, iyy, izz, ixy, ixz, iyz = inr.inertia
    got = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    assert np.allclose(got, I_mj, rtol=1e-9, atol=1e-14)
    assert [f.code for f in findings] == ["filled_inertial"]


# ------------------------------------------------ batching, dtype, device --

def test_batched_sizes_match_a_python_loop():
    torch.manual_seed(0)
    sizes = torch.rand(7, 3, **F64) + 0.05
    m, c, I = box_inertia(sizes, 750.0)
    assert m.shape == (7,) and c.shape == (7, 3) and I.shape == (7, 3, 3)
    for k in range(7):
        mk, _, Ik = box_inertia(tuple(sizes[k].tolist()), 750.0, dtype=torch.float64)
        assert float(m[k]) == pytest.approx(float(mk))
        assert torch.allclose(I[k], Ik, atol=1e-14)


def test_density_broadcasts_over_a_batch():
    rho = torch.tensor([500.0, 1000.0, 2000.0], **F64)
    m, _, I = cylinder_inertia(0.1, 0.3, rho)
    assert m.shape == (3,) and I.shape == (3, 3, 3)
    # mass and inertia are both linear in density
    assert float(m[1]) == pytest.approx(2 * float(m[0]))
    assert torch.allclose(I[2], 4 * I[0], atol=1e-12)


def test_size_and_density_batches_broadcast_together():
    sizes = torch.rand(4, 1, 3, **F64) + 0.1
    rho = torch.tensor([[300.0, 900.0]], **F64)
    m, c, I = box_inertia(sizes, rho)
    assert m.shape == (4, 2) and c.shape == (4, 2, 3) and I.shape == (4, 2, 3, 3)


def test_dtype_follows_the_inputs_and_the_explicit_argument():
    m32, c32, I32 = box_inertia(torch.tensor([0.1, 0.2, 0.3]), 1000.0)
    assert (m32.dtype, c32.dtype, I32.dtype) == (torch.float32,) * 3
    m64 = box_inertia(torch.tensor([0.1, 0.2, 0.3], **F64), 1000.0).mass
    assert m64.dtype == torch.float64
    # a plain python size with an explicit dtype
    assert sphere_inertia(0.1, 1000.0, dtype=torch.float64).mass.dtype == torch.float64
    # mixed precision promotes rather than silently truncating
    mixed = cylinder_inertia(torch.tensor(0.1), torch.tensor(0.4, **F64), 1000.0)
    assert mixed.mass.dtype == torch.float64


def test_results_stay_on_the_input_device():
    size = torch.tensor([0.2, 0.3, 0.4])
    props = box_inertia(size, 1000.0)
    assert props.mass.device == size.device
    assert props.inertia.device == size.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_runs_on_cuda():                              # pragma: no cover
    size = torch.tensor([0.2, 0.3, 0.4], device="cuda")
    props = box_inertia(size, 1000.0)
    assert props.inertia.is_cuda
    assert float(props.mass.cpu()) == pytest.approx(1000.0 * 0.2 * 0.3 * 0.4)


def test_float32_agrees_with_float64():
    a = capsule_inertia(0.05, 0.3, 1200.0, dtype=torch.float32)
    b = capsule_inertia(0.05, 0.3, 1200.0, dtype=torch.float64)
    assert float(a.mass) == pytest.approx(float(b.mass), rel=1e-6)
    assert torch.allclose(a.inertia.double(), b.inertia, rtol=1e-5, atol=1e-10)


# ------------------------------------------------------------ gradients ---

def test_mass_gradient_matches_the_analytic_derivative():
    """d(mass)/d(size) for a box is rho times the product of the other two
    sides, which is worth pinning down: link sizing loops differentiate this."""
    size = torch.tensor([0.2, 0.3, 0.4], **F64, requires_grad=True)
    rho = 1500.0
    box_inertia(size, rho).mass.backward()
    expect = torch.tensor([rho * 0.3 * 0.4, rho * 0.2 * 0.4, rho * 0.2 * 0.3], **F64)
    assert torch.allclose(size.grad, expect, atol=1e-9)


def test_gradcheck_through_the_whole_geometry_path():
    r = torch.tensor(0.09, **F64, requires_grad=True)
    l = torch.tensor(0.35, **F64, requires_grad=True)
    rho = torch.tensor(1100.0, **F64, requires_grad=True)

    def f(r, l, rho):
        props = capsule_inertia(r, l, rho)
        moved = shift_inertia(props.inertia, props.mass,
                              torch.tensor([0.1, -0.2, 0.3], **F64))
        return moved.sum() + props.mass

    assert torch.autograd.gradcheck(f, (r, l, rho), eps=1e-6, atol=1e-6)


def test_gradient_flows_to_a_geometry_density():
    rho = torch.tensor(900.0, **F64, requires_grad=True)
    g = Geometry("box", None, (1.0, 1.0, 1.0), (0.2, 0.2, 0.6),
                 (0.0, 0.0, 0.3), (0.0, 0.5, 0.0))
    props = geometry_mass_properties(g, rho)
    props.inertia.trace().backward()
    # everything is linear in density, so the gradient is the value over rho
    with torch.no_grad():
        base = geometry_mass_properties(g, 900.0, dtype=torch.float64)
    assert float(rho.grad) == pytest.approx(float(base.inertia.trace()) / 900.0,
                                            rel=1e-9)


# ------------------------------------------------------------ link level ---

def test_link_prefers_collision_over_visual():
    link = Link("l")
    link.visual = Geometry("box", None, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    link.collision = Geometry("box", None, (1.0, 1.0, 1.0), (0.1, 0.1, 0.1))
    props = link_mass_properties(link, 1000.0, dtype=torch.float64)
    assert float(props.mass) == pytest.approx(1000.0 * 0.001)


def test_link_falls_back_to_visual_when_asked():
    link = Link("l")
    link.visual = Geometry("sphere", None, (1.0, 1.0, 1.0), (0.2,))
    props = link_mass_properties(link, 1000.0, dtype=torch.float64)
    assert float(props.mass) == pytest.approx(1000.0 * 4 / 3 * math.pi * 0.008)
    with pytest.raises(ValueError, match="no primitive geometry"):
        link_mass_properties(link, 1000.0, fallback_to_visual=False)


def test_link_accepts_a_list_of_geometries():
    geoms = [Geometry("box", None, (1.0, 1.0, 1.0), (0.2, 0.2, 0.2), (0.5, 0, 0)),
             Geometry("box", None, (1.0, 1.0, 1.0), (0.2, 0.2, 0.2), (-0.5, 0, 0))]
    props = link_mass_properties(geoms, 1000.0, dtype=torch.float64)
    assert float(props.mass) == pytest.approx(2 * 1000.0 * 0.008)
    assert torch.allclose(props.com, torch.zeros(3, **F64), atol=1e-15)
    # two 8 kg blocks half a metre out on either side
    m = 1000.0 * 0.008
    single = m * (0.04 + 0.04) / 12
    assert float(props.inertia[2, 2]) == pytest.approx(2 * (single + m * 0.25))


def test_mesh_geometry_raises_unless_skipped():
    link = Link("l")
    link.collision = [Geometry("mesh", "arm.stl", (1.0, 1.0, 1.0), ()),
                      Geometry("box", None, (1.0, 1.0, 1.0), (0.1, 0.1, 0.1))]
    with pytest.raises(ValueError, match="closed-form"):
        link_mass_properties(link, 1000.0)
    props = link_mass_properties(link, 1000.0, skip_unsupported=True,
                                 dtype=torch.float64)
    assert float(props.mass) == pytest.approx(1.0)


def test_unsupported_kind_names_itself():
    with pytest.raises(ValueError, match="mesh"):
        primitive_mass_properties("mesh", (), 1000.0)


def test_bad_sizes_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        box_inertia((-0.1, 0.2, 0.3), 1000.0)
    with pytest.raises(ValueError, match="non-negative"):
        sphere_inertia(-1.0, 1000.0)
    with pytest.raises(ValueError, match="must end in 3"):
        box_inertia((0.1, 0.2), 1000.0)
    with pytest.raises(ValueError, match="must end in 2"):
        primitive_mass_properties("cylinder", (0.1,), 1000.0)


def test_as_inertial_rejects_a_batch():
    props = box_inertia(torch.rand(3, 3, **F64) + 0.1, 1000.0)
    with pytest.raises(ValueError, match="unbatched"):
        as_inertial(props)


def test_as_inertial_uses_the_urdf_tuple_order():
    g = Geometry("box", None, (1.0, 1.0, 1.0), (0.2, 0.4, 0.6), (0, 0, 0),
                 (0.0, 0.0, 0.3))
    props = geometry_mass_properties(g, 1000.0, dtype=torch.float64)
    inr = as_inertial(props)
    ixx, iyy, izz, ixy, ixz, iyz = inr.inertia
    assert ixx == pytest.approx(float(props.inertia[0, 0]))
    assert iyy == pytest.approx(float(props.inertia[1, 1]))
    assert izz == pytest.approx(float(props.inertia[2, 2]))
    assert ixy == pytest.approx(float(props.inertia[0, 1]))
    assert ixz == pytest.approx(float(props.inertia[0, 2]))
    assert iyz == pytest.approx(float(props.inertia[1, 2]))
    assert abs(ixy) > 1e-6                       # the yaw really did couple x and y


# ----------------------------------------------------- fill_missing_inertials --

FILL_URDF = """
<robot name="fill">
  <link name="base">
    <collision><geometry><box size="0.4 0.4 0.1"/></geometry></collision>
  </link>
  <link name="upper">
    <inertial>
      <mass value="2.5"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.01" iyy="0.02" izz="0.03" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision><geometry><cylinder radius="0.05" length="0.3"/></geometry></collision>
  </link>
  <link name="empty"/>
  <link name="meshy">
    <collision><geometry><mesh filename="hand.stl"/></geometry></collision>
  </link>
  <link name="zero_mass">
    <inertial><mass value="0"/>
      <inertia ixx="0" iyy="0" izz="0" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><sphere radius="0.08"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="upper"/><child link="empty"/><origin xyz="0 0 0.3"/>
  </joint>
  <joint name="j3" type="fixed">
    <parent link="empty"/><child link="meshy"/>
  </joint>
  <joint name="j4" type="fixed">
    <parent link="meshy"/><child link="zero_mass"/>
  </joint>
</robot>
"""


def test_fill_missing_inertials_reports_every_link():
    ir, findings = fill_missing_inertials(parse_urdf_string(FILL_URDF), 500.0)
    codes = {f.where: f.code for f in findings}
    assert codes["base"] == "filled_inertial"
    assert codes["empty"] == "no_inertia_geometry"
    assert codes["meshy"] == "no_inertia_geometry"
    assert codes["zero_mass"] == "filled_inertial"
    assert "upper" not in codes                  # already had real numbers
    assert "mesh" in [f.message for f in findings if f.where == "meshy"][0]


def test_fill_keeps_existing_inertials_and_fills_the_rest():
    ir, _ = fill_missing_inertials(parse_urdf_string(FILL_URDF), 500.0)
    assert ir.links["upper"].inertial.mass == 2.5
    assert ir.links["upper"].inertial.inertia[0] == 0.01
    base = ir.links["base"].inertial
    assert base.mass == pytest.approx(500.0 * 0.4 * 0.4 * 0.1)
    assert base.com == (0.0, 0.0, 0.0)
    assert base.inertia[0] == pytest.approx(base.mass * (0.16 + 0.01) / 12)
    zm = ir.links["zero_mass"].inertial
    assert zm.mass == pytest.approx(500.0 * 4 / 3 * math.pi * 0.08 ** 3)
    assert ir.links["empty"].inertial is None


def test_fill_can_leave_zero_mass_links_alone():
    ir, _ = fill_missing_inertials(parse_urdf_string(FILL_URDF), 500.0,
                                   fill_zero_mass=False)
    assert ir.links["zero_mass"].inertial.mass == 0.0


def test_fill_overwrite_replaces_real_inertials():
    ir, _ = fill_missing_inertials(parse_urdf_string(FILL_URDF), 500.0,
                                   overwrite=True)
    upper = ir.links["upper"].inertial
    assert upper.mass == pytest.approx(500.0 * math.pi * 0.05 ** 2 * 0.3)


def test_fill_accepts_a_per_link_density_map():
    ir, findings = fill_missing_inertials(parse_urdf_string(FILL_URDF),
                                          {"base": 2700.0})
    assert ir.links["base"].inertial.mass == pytest.approx(2700.0 * 0.016)
    # a link the map does not name falls back to the default density
    assert ir.links["zero_mass"].inertial.mass == pytest.approx(
        DEFAULT_DENSITY * 4 / 3 * math.pi * 0.08 ** 3)
    msg = [f.message for f in findings if f.where == "base"][0]
    assert "2700" in msg


def test_fill_flags_a_zero_volume_primitive():
    ir = Robot("degenerate", {"only": Link("only")}, [])
    ir.links["only"].collision = Geometry("box", None, (1.0, 1.0, 1.0),
                                          (0.0, 0.2, 0.3))
    ir, findings = fill_missing_inertials(ir, 1000.0)
    assert [f.code for f in findings] == ["degenerate_geometry"]
    assert ir.links["only"].inertial is None


def test_fill_is_idempotent():
    a, _ = fill_missing_inertials(parse_urdf_string(FILL_URDF), 500.0)
    first = a.links["base"].inertial
    b, findings = fill_missing_inertials(a, 500.0)
    assert b.links["base"].inertial == first
    assert "filled_inertial" not in [f.code for f in findings if f.where == "base"]


def test_filled_inertials_are_physically_valid():
    """A tensor computed from real geometry has to be positive definite and has
    to satisfy the triangle inequality on its principal moments, which is what
    MuJoCo checks before it will load a model. The repair pass already knows how
    to test that, so run it over the filled result and expect silence."""
    ir, _ = fill_missing_inertials(parse_urdf_string(FILL_URDF), 500.0,
                                   overwrite=True)
    _, findings = repair(ir)
    assert [f for f in findings if f.code in ("negative_inertia", "inertia_triangle")] == []
    for name in ("base", "upper", "zero_mass"):
        inr = ir.links[name].inertial
        ixx, iyy, izz, ixy, ixz, iyz = inr.inertia
        I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        moments = np.sort(np.linalg.eigvalsh(I))
        assert moments[0] > 0.0
        assert moments[0] + moments[1] >= moments[2] * (1 - 1e-12)


def test_filled_model_compiles_and_gives_the_right_gravity_torque():
    """The payoff: a URDF with no inertial data at all becomes a model whose
    gravity torque matches the hand-computed m g r of a pendulum."""
    rho = 800.0
    urdf = """
    <robot name="pendulum">
      <link name="base"/>
      <link name="arm">
        <collision>
          <origin xyz="0.25 0 0"/>
          <geometry><box size="0.5 0.05 0.05"/></geometry>
        </collision>
      </link>
      <joint name="j" type="revolute">
        <parent link="base"/><child link="arm"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3" upper="3" effort="10" velocity="1"/>
      </joint>
    </robot>
    """
    ir, _ = fill_missing_inertials(parse_urdf_string(urdf), rho)
    m = rho * 0.5 * 0.05 * 0.05
    assert ir.links["arm"].inertial.mass == pytest.approx(m)
    assert ir.links["arm"].inertial.com[0] == pytest.approx(0.25)
    chain = compile_robot(ir, dtype=torch.float64)
    q = torch.tensor([[0.0], [math.pi / 2], [0.3]], **F64)
    tau = gravity(chain, q)
    # rotating about +y by theta puts the COM at x = r cos(theta), z = -r sin(theta);
    # the gravity torque about +y is then -m g r cos(theta)
    expect = -m * 9.81 * 0.25 * torch.cos(q[:, 0])
    assert torch.allclose(tau[:, 0], expect, atol=1e-9)


def test_filled_inertia_survives_the_compile_step():
    ir, _ = fill_missing_inertials(parse_urdf_string(FILL_URDF), 500.0)
    chain = compile_robot(ir, dtype=torch.float64)
    i = chain.link_index["base"]
    inr = ir.links["base"].inertial
    assert float(chain.link_mass[i]) == pytest.approx(inr.mass)
    assert float(chain.link_inertia[i][0, 0]) == pytest.approx(inr.inertia[0])
    assert float(chain.link_inertia[i][1, 1]) == pytest.approx(inr.inertia[1])
