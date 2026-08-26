# src/kinfast/inertia.py
"""Mass properties from primitive geometry.

Half the URDFs in the wild ship links with geometry but no <inertial>, or with
a placeholder mass of zero. Those models do forward kinematics fine and then
fall over the moment you ask for a mass matrix, gravity torque, or a MuJoCo
export. This module closes that gap: given a density, it computes mass, centre
of mass and inertia tensor for the primitives URDF can express (box, cylinder,
sphere, plus the capsule, which URDF itself has no tag for but MJCF and several
URDF dialects do), moves each one into the link frame, and sums them.

Conventions, all matching the IR and compile.py:

- Primitives are centred on their own origin, with the cylinder and capsule
  axis along +z, which is what URDF <cylinder> means.
- Cylinder and capsule `length` is the full length of the cylindrical section,
  so a capsule's total extent along z is length + 2 * radius.
- An inertia tensor returned by these functions is taken about the centre of
  mass it is returned with, expressed in the axes of the frame the centre of
  mass is expressed in. That is exactly what a URDF <inertial> block holds and
  what CompiledChain.link_com / link_inertia expect.
- Densities are kg/m^3. The default of 1000 (water) is the usual placeholder:
  it puts a robot arm in the right order of magnitude without pretending to
  know the material.

Everything is plain torch, so the functions broadcast over a leading batch
dimension, run on whatever device the inputs live on, and carry gradients back
to the sizes and densities you feed them. That makes them usable inside a
design loop that optimizes link dimensions, not just as a one-shot repair.
The working dtype follows the caller's tensors (as the q dtype does elsewhere
in the library), or the explicit `dtype=` argument if you pass one.
"""
import math
from typing import NamedTuple

import torch

from kinfast.ir import Geometry, Inertial, Link
from kinfast.transforms import rpy_to_matrix
from kinfast.urdf.repair import Finding

DEFAULT_DENSITY = 1000.0

# Kinds we can integrate analytically. A mesh needs the actual triangles, which
# the IR does not carry, so it is not in here.
PRIMITIVE_KINDS = ("box", "cylinder", "sphere", "capsule")

__all__ = [
    "MassProperties", "DEFAULT_DENSITY", "PRIMITIVE_KINDS",
    "box_inertia", "cylinder_inertia", "sphere_inertia", "capsule_inertia",
    "primitive_mass_properties", "rotate_inertia", "shift_inertia",
    "combine_mass_properties", "geometry_mass_properties",
    "link_mass_properties", "as_inertial", "fill_missing_inertials",
]


class MassProperties(NamedTuple):
    """Mass, centre of mass, and inertia about that centre of mass.

    Shapes are (...), (..., 3) and (..., 3, 3) over any leading batch dims.
    It is a tuple, so `m, c, I = props` works.
    """
    mass: torch.Tensor
    com: torch.Tensor
    inertia: torch.Tensor


# ---------------------------------------------------------------- plumbing --

def _working(values, dtype, device):
    """Pick the dtype and device the caller implied.

    Explicit arguments win. Otherwise we follow the floating point tensors that
    were passed in (promoting if they disagree) and fall back to torch's
    default dtype when every input is a plain Python number.
    """
    tensors = [v for v in values if isinstance(v, torch.Tensor)]
    if device is None:
        device = tensors[0].device if tensors else torch.device("cpu")
    if dtype is None:
        for t in tensors:
            if t.is_floating_point():
                dtype = t.dtype if dtype is None else torch.promote_types(dtype, t.dtype)
        if dtype is None:
            dtype = torch.get_default_dtype()
    return dtype, torch.device(device)


def _as(x, dtype, device):
    """Number, sequence or tensor -> tensor of the working dtype and device.

    Tensors go through .to(), which keeps them attached to the autograd graph.
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _check_nonneg(t, what):
    """Reject negative extents. Zero is allowed and simply yields zero mass:
    real URDFs do carry degenerate primitives, and the caller decides whether
    that is fatal."""
    if bool((t < 0).any()):
        raise ValueError(f"{what} must be non-negative, got {t.tolist()}")


def _diag(a, b, c):
    """Batched diagonal 3x3 from three scalars-or-batches."""
    return torch.diag_embed(torch.stack(torch.broadcast_tensors(a, b, c), dim=-1))


def _zeros_com(mass):
    """A (..., 3) zero centre of mass matching a (...) mass."""
    return torch.zeros(*mass.shape, 3, dtype=mass.dtype, device=mass.device)


def _size_tensor(size, n, kind, dtype, device):
    s = _as(size, dtype, device)
    if s.ndim == 0 or s.shape[-1] != n:
        raise ValueError(
            f"{kind} size must end in {n} number(s), got shape "
            f"{tuple(s.shape)}")
    return s


# -------------------------------------------------------------- primitives --

def box_inertia(size, density=DEFAULT_DENSITY, *, dtype=None, device=None):
    """Solid box of full extents (x, y, z), centred on its own origin.

    m = rho * x * y * z, and Ixx = m (y^2 + z^2) / 12 by the usual textbook
    integral, cyclically for the other two axes.
    """
    dtype, device = _working((size, density), dtype, device)
    s = _size_tensor(size, 3, "box", dtype, device)
    _check_nonneg(s, "box size")
    rho = _as(density, dtype, device)
    x, y, z = s.unbind(-1)
    m = rho * x * y * z
    k = m / 12.0
    return MassProperties(
        m, _zeros_com(m),
        _diag(k * (y * y + z * z), k * (x * x + z * z), k * (x * x + y * y)))


def cylinder_inertia(radius, length, density=DEFAULT_DENSITY, *,
                     dtype=None, device=None):
    """Solid cylinder about the +z axis, full length `length`, centred on its
    own origin.

    m = rho * pi * r^2 * l, Izz = m r^2 / 2 (the polar moment) and
    Ixx = Iyy = m (3 r^2 + l^2) / 12.
    """
    dtype, device = _working((radius, length, density), dtype, device)
    r = _as(radius, dtype, device)
    l = _as(length, dtype, device)
    _check_nonneg(r, "cylinder radius")
    _check_nonneg(l, "cylinder length")
    rho = _as(density, dtype, device)
    m = rho * math.pi * r * r * l
    trans = m * (3.0 * r * r + l * l) / 12.0
    axial = m * r * r / 2.0
    return MassProperties(m, _zeros_com(m), _diag(trans, trans, axial))


def sphere_inertia(radius, density=DEFAULT_DENSITY, *, dtype=None, device=None):
    """Solid sphere centred on its own origin. m = rho * 4/3 pi r^3 and
    I = 2/5 m r^2 on every axis."""
    dtype, device = _working((radius, density), dtype, device)
    r = _as(radius, dtype, device)
    _check_nonneg(r, "sphere radius")
    rho = _as(density, dtype, device)
    m = rho * (4.0 / 3.0) * math.pi * r * r * r
    i = 0.4 * m * r * r
    return MassProperties(m, _zeros_com(m), _diag(i, i, i))


def capsule_inertia(radius, length, density=DEFAULT_DENSITY, *,
                    dtype=None, device=None):
    """Capsule about the +z axis: a cylinder of full length `length` capped by
    a hemisphere of the same radius at each end, centred on its own origin.

    Cylinder and caps are integrated separately and added. Each cap has moment
    2/5 m_h r^2 about a transverse axis through the sphere centre and its own
    centroid sits 3r/8 further out, so moving it to the capsule centre with the
    parallel axis theorem contributes m_h (l^2/4 + 3 l r / 8) on top. Written
    out for both caps that is m_caps (2 r^2 / 5 + l^2 / 4 + 3 l r / 8), which
    agrees with MuJoCo's own capsule inertia to machine precision.
    """
    dtype, device = _working((radius, length, density), dtype, device)
    r = _as(radius, dtype, device)
    l = _as(length, dtype, device)
    _check_nonneg(r, "capsule radius")
    _check_nonneg(l, "capsule length")
    rho = _as(density, dtype, device)
    m_cyl = rho * math.pi * r * r * l
    m_cap = rho * (4.0 / 3.0) * math.pi * r * r * r     # both hemispheres
    m = m_cyl + m_cap
    axial = m_cyl * r * r / 2.0 + 0.4 * m_cap * r * r
    trans = (m_cyl * (3.0 * r * r + l * l) / 12.0
             + m_cap * (0.4 * r * r + l * l / 4.0 + 3.0 * l * r / 8.0))
    return MassProperties(m, _zeros_com(m), _diag(trans, trans, axial))


_PRIMITIVE_ARITY = {"box": 3, "cylinder": 2, "sphere": 1, "capsule": 2}


def primitive_mass_properties(kind, size, density=DEFAULT_DENSITY, *,
                              dtype=None, device=None):
    """Dispatch on an IR Geometry.kind string with its IR `size` tuple.

    Sizes follow the IR: box (x, y, z) full extents, cylinder and capsule
    (radius, length), sphere (radius,). The result is in the primitive's own
    frame, where the centre of mass is the origin for all four shapes.
    """
    if kind not in _PRIMITIVE_ARITY:
        raise ValueError(
            f"no closed-form inertia for geometry kind {kind!r}; "
            f"supported kinds are {', '.join(PRIMITIVE_KINDS)}")
    dtype, device = _working((size, density), dtype, device)
    s = _size_tensor(size, _PRIMITIVE_ARITY[kind], kind, dtype, device)
    if kind == "box":
        return box_inertia(s, density, dtype=dtype, device=device)
    if kind == "sphere":
        return sphere_inertia(s[..., 0], density, dtype=dtype, device=device)
    fn = cylinder_inertia if kind == "cylinder" else capsule_inertia
    return fn(s[..., 0], s[..., 1], density, dtype=dtype, device=device)


# ------------------------------------------------------- frame arithmetic ---

def rotate_inertia(inertia, R):
    """Re-express an inertia tensor in a rotated frame: I' = R I R^T.

    R maps vectors from the tensor's current frame into the target frame, which
    is the same direction as a URDF <origin rpy> maps the geometry frame into
    the link frame. Broadcasts a plain (3, 3) rotation over a batch of tensors.
    """
    Rt = R.transpose(-1, -2)
    return R @ inertia @ Rt


def shift_inertia(inertia, mass, offset, *, to_com=False):
    """Parallel axis theorem.

    Moves an inertia taken about the centre of mass to a reference point
    `offset` away: I' = I + m (|d|^2 E - d d^T). The correction is quadratic in
    d, so the direction of `offset` does not matter, only the separation.
    Pass to_com=True to run it backwards, taking an inertia measured about some
    point and referring it to the centre of mass `offset` away.

    `mass` and `offset` may be plain numbers or sequences; they are read into
    the dtype and device of `inertia`, which is the frame everything else is in.
    """
    d = _as(offset, inertia.dtype, inertia.device)
    mass = _as(mass, inertia.dtype, inertia.device)
    d2 = (d * d).sum(dim=-1)
    outer = d.unsqueeze(-1) * d.unsqueeze(-2)
    eye = torch.eye(3, dtype=inertia.dtype, device=inertia.device)
    term = mass.unsqueeze(-1).unsqueeze(-1) * (d2.unsqueeze(-1).unsqueeze(-1) * eye - outer)
    return inertia - term if to_com else inertia + term


def combine_mass_properties(parts):
    """Sum a sequence of MassProperties expressed in one common frame.

    Masses add, the centre of mass is the mass-weighted mean, and each part's
    inertia is carried to the combined centre of mass with the parallel axis
    theorem before summing. A total mass of zero gives a centre of mass of zero
    rather than a NaN, since there is nothing to weight by.
    """
    parts = list(parts)
    if not parts:
        raise ValueError("combine_mass_properties needs at least one part")
    if len(parts) == 1:
        return parts[0]
    total = sum((p.mass for p in parts[1:]), parts[0].mass)
    weighted = sum((p.mass.unsqueeze(-1) * p.com for p in parts[1:]),
                   parts[0].mass.unsqueeze(-1) * parts[0].com)
    safe = torch.where(total > 0, total, torch.ones_like(total))
    com = weighted / safe.unsqueeze(-1)
    inertia = None
    for p in parts:
        term = shift_inertia(p.inertia, p.mass, p.com - com)
        inertia = term if inertia is None else inertia + term
    return MassProperties(total, com, inertia)


# ------------------------------------------------------------ IR wrappers ---

def geometry_mass_properties(geom, density=DEFAULT_DENSITY, *,
                             dtype=None, device=None):
    """Mass properties of one IR Geometry, expressed in the link frame.

    The primitive is built in its own frame, its tensor is rotated by the
    geometry <origin rpy>, and its centre of mass is placed at the <origin xyz>.
    Note that Geometry.scale is a mesh-only field in URDF, so it is ignored for
    primitives, exactly as URDF itself does.
    """
    dtype, device = _working(
        (density, geom.size, geom.origin_xyz, geom.origin_rpy), dtype, device)
    props = primitive_mass_properties(geom.kind, geom.size, density,
                                      dtype=dtype, device=device)
    rpy = _as(geom.origin_rpy, dtype, device)
    xyz = _as(geom.origin_xyz, dtype, device)
    R = rpy_to_matrix(rpy)
    inertia = rotate_inertia(props.inertia, R)
    com = xyz + (R @ props.com.unsqueeze(-1)).squeeze(-1)
    return MassProperties(props.mass, com, inertia)


def _link_geometries(link, fallback_to_visual):
    """Every geometry that should contribute mass, as a flat list.

    The IR holds a single collision (and visual) geometry per link, but this
    accepts a list in either slot so a richer parser can drop in later. Visual
    geometry is only used when there is no collision geometry at all: doubling
    a link's mass because it carries both would be worse than guessing.
    """
    def flat(slot):
        if slot is None:
            return []
        if isinstance(slot, Geometry):
            return [slot]
        return [g for g in slot if g is not None]

    geoms = flat(getattr(link, "collision", None))
    if not geoms and fallback_to_visual:
        geoms = flat(getattr(link, "visual", None))
    return geoms


def link_mass_properties(link, density=DEFAULT_DENSITY, *,
                         fallback_to_visual=True, skip_unsupported=False,
                         dtype=None, device=None):
    """Mass properties of a whole link, in the link frame.

    `link` may be an IR Link, a single Geometry, or any iterable of Geometry.
    Collision geometry is preferred over visual geometry (it is what the
    physics engine sees, and it is usually the cruder, more solid shape).

    A mesh has no closed-form inertia, so by default hitting one raises rather
    than quietly under-reporting the link's mass. Pass skip_unsupported=True to
    drop meshes and keep whatever primitives the link does have.
    """
    if isinstance(link, Geometry):
        geoms = [link]
    elif isinstance(link, Link):
        geoms = _link_geometries(link, fallback_to_visual)
    else:
        geoms = [g for g in link if g is not None]
    usable = [g for g in geoms if g.kind in _PRIMITIVE_ARITY]
    if len(usable) != len(geoms) and not skip_unsupported:
        bad = sorted({g.kind for g in geoms if g.kind not in _PRIMITIVE_ARITY})
        raise ValueError(
            f"link geometry of kind(s) {bad} has no closed-form inertia; "
            "pass skip_unsupported=True to ignore it")
    if not usable:
        raise ValueError("no primitive geometry to compute mass properties from")
    # Resolve the working dtype and device once, up front, so two geometries on
    # the same link cannot end up in different precisions and fail to combine.
    dtype, device = _working((density,), dtype, device)
    parts = [geometry_mass_properties(g, density, dtype=dtype, device=device)
             for g in usable]
    return combine_mass_properties(parts)


def as_inertial(props):
    """MassProperties -> an IR Inertial of plain Python floats.

    The tuple order is the URDF one that ir.Inertial and compile.py use:
    (ixx, iyy, izz, ixy, ixz, iyz), read straight off the tensor, so a positive
    ixy here means a positive I[0][1] entry.
    """
    m, c, I = props
    if m.ndim != 0:
        raise ValueError(
            f"as_inertial needs a single unbatched result, got mass of shape "
            f"{tuple(m.shape)}")
    # symmetrize to kill the last bit of round-off from R I R^T
    I = 0.5 * (I + I.transpose(-1, -2))
    return Inertial(
        mass=float(m),
        com=tuple(float(v) for v in c),
        inertia=(float(I[0, 0]), float(I[1, 1]), float(I[2, 2]),
                 float(I[0, 1]), float(I[0, 2]), float(I[1, 2])),
    )


def fill_missing_inertials(ir, density=DEFAULT_DENSITY, *,
                           fallback_to_visual=True, fill_zero_mass=True,
                           overwrite=False):
    """Give every geometry-bearing link that lacks one a plausible <inertial>.

    Returns (ir, findings) like urdf.repair.repair does, mutating the IR in
    place. A link is repaired when it has no inertial at all, or (with
    fill_zero_mass, the default) when its mass is zero, which is the same
    physical hole written a different way and makes the mass matrix singular
    just as surely. Links that already carry real numbers are left alone unless
    you pass overwrite=True.

    `density` is kg/m^3, either one number for the whole robot or a mapping
    from link name to number; links a mapping does not name fall back to
    DEFAULT_DENSITY. Computation runs in float64 regardless of anything else,
    because the result is stored as Python floats and this is a one-off.

    Findings use the same codes/where/message shape as the URDF repair pass:
      filled_inertial     an inertial was computed and written
      no_inertia_geometry nothing to integrate (mesh-only or no geometry)
      degenerate_geometry the primitives are there but enclose zero volume
    """
    findings = []
    is_map = hasattr(density, "get") and not isinstance(density, (int, float))
    for name, link in ir.links.items():
        inr = link.inertial
        has_real = inr is not None and not (fill_zero_mass and inr.mass <= 0.0)
        if has_real and not overwrite:
            continue
        geoms = _link_geometries(link, fallback_to_visual)
        usable = [g for g in geoms if g.kind in _PRIMITIVE_ARITY]
        if not usable:
            findings.append(Finding(
                "no_inertia_geometry", name,
                "no primitive collision geometry to infer an inertial from"
                + (f" (only {sorted({g.kind for g in geoms})})" if geoms else "")))
            continue
        rho = density.get(name, DEFAULT_DENSITY) if is_map else density
        props = link_mass_properties(link, rho, fallback_to_visual=fallback_to_visual,
                                     skip_unsupported=True, dtype=torch.float64)
        if float(props.mass) <= 0.0:
            findings.append(Finding(
                "degenerate_geometry", name,
                "primitive geometry encloses zero volume; left as is"))
            continue
        link.inertial = as_inertial(props)
        kinds = ", ".join(g.kind for g in usable)
        findings.append(Finding(
            "filled_inertial", name,
            f"mass {link.inertial.mass:.6g} kg from {kinds} at "
            f"{float(rho):.6g} kg/m^3"))
    return ir, findings
