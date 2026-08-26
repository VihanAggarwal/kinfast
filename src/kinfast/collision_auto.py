# src/kinfast/collision_auto.py
"""Build a collision SphereModel automatically from the IR's collision shapes.

Sphere models are what make collision checking cheap and differentiable, but
writing one by hand for a real arm means typing dozens of centers and radii and
keeping them in sync with the URDF. Most URDFs already ship a usable collision
description made of primitives, so this module turns those primitives into
spheres directly:

- sphere   -> one sphere, exactly the original shape
- cylinder -> a row of spheres along the local z axis
- box      -> a regular grid of spheres, one per cell

Two properties are guaranteed by construction, and both are what a collision
checker actually needs:

1. Cover. Every point of the primitive lies inside at least one sphere, so the
   sphere set is a conservative (never optimistic) stand-in for the shape. A
   box cell is covered by a sphere at its center with the cell half-diagonal as
   radius; a cylinder slab of half-height h and radius R is covered by a sphere
   on the axis with radius sqrt(R^2 + h^2).
2. Centers stay inside. Every generated center lies within the primitive, so a
   sphere never sticks out on a side where the real link has nothing, beyond
   the padding the covering radius costs.

The result is deliberately conservative: it over-approximates. Shrink `spacing`
to tighten the fit at the price of more spheres.

Meshes are skipped. There is no mesh loader here, and a bounding sphere around
a whole mesh would be so loose it would report collisions that do not exist.
Links whose collision shape is a mesh, or which have no collision shape at all,
simply contribute nothing; use `unsupported_links` to see which ones.

Generation itself is a one-off setup step on Python floats from the IR, so
there is nothing to differentiate here. The model it returns is the usual
batched, differentiable SphereModel: the working dtype and device follow the
`q` you pass to `centers_world`, exactly like the rest of the library.
"""
import math
import torch

from kinfast import transforms as T
from kinfast.collision import SphereModel

SUPPORTED_KINDS = ("sphere", "cylinder", "box")


def _cell_count(extent: float, spacing: float, cap: int) -> int:
    """How many cells to cut an extent into so no cell is longer than `spacing`.

    Always at least one, never more than `cap`. The covering radius is computed
    from the count that comes back, so clamping stays safe: a capped axis gives
    fewer, fatter spheres, never a hole.
    """
    if extent <= 0.0:
        return 1
    return max(1, min(int(math.ceil(extent / spacing - 1e-9)), cap))


def _sphere_spheres(size):
    """URDF <sphere radius>. -> [((x, y, z), r)] in the geometry frame."""
    if len(size) < 1:
        return []
    radius = float(size[0])
    if radius <= 0.0:
        return []
    return [((0.0, 0.0, 0.0), radius)]


def _cylinder_spheres(size, spacing, cap):
    """URDF <cylinder radius length>: z axis, centered on the origin.

    The cylinder is sliced into `n` equal slabs and each slab gets one sphere on
    the axis. A slab has half-height h = L / (2n) and radius R, and its farthest
    corner from the slab center sits at sqrt(R^2 + h^2), which is the radius we
    use. Centers stay on the axis and inside the length, since the outermost one
    is at L/2 - h.
    """
    if len(size) < 2:
        return []
    radius, length = float(size[0]), float(size[1])
    if radius <= 0.0 and length <= 0.0:
        return []
    n = _cell_count(length, spacing, cap)
    h = 0.5 * length / n
    r = math.sqrt(radius * radius + h * h)
    if r <= 0.0:
        return []
    return [((0.0, 0.0, -0.5 * length + (2 * k + 1) * h), r) for k in range(n)]


def _box_spheres(size, spacing, cap):
    """URDF <box size>: full extents, centered on the origin.

    The box is cut into a regular grid of cells and each cell gets a sphere at
    its center with the cell half-diagonal as radius, which is the smallest
    sphere containing that cell. Cell centers are interior points of the box.
    """
    if len(size) < 3:
        return []
    sx, sy, sz = (float(v) for v in size[:3])
    if sx <= 0.0 and sy <= 0.0 and sz <= 0.0:
        return []
    nx = _cell_count(sx, spacing, cap)
    ny = _cell_count(sy, spacing, cap)
    nz = _cell_count(sz, spacing, cap)
    cx, cy, cz = sx / nx, sy / ny, sz / nz
    r = 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
    if r <= 0.0:
        return []
    out = []
    for i in range(nx):
        x = -0.5 * sx + (i + 0.5) * cx
        for j in range(ny):
            y = -0.5 * sy + (j + 0.5) * cy
            for k in range(nz):
                z = -0.5 * sz + (k + 0.5) * cz
                out.append(((x, y, z), r))
    return out


def spheres_for_geometry(geom, spacing: float = 0.05, padding: float = 0.0,
                         max_per_axis: int = 8):
    """Spheres covering one Geometry, expressed in the link frame.

    The primitive is filled in its own frame, then every center is pushed
    through the geometry origin (xyz, rpy) so the caller gets link-frame
    coordinates ready for SphereModel. Returns [(x, y, z, r), ...], empty for a
    mesh or an unusable size.
    """
    if geom is None or geom.kind not in SUPPORTED_KINDS:
        return []
    if spacing <= 0.0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    if max_per_axis < 1:
        raise ValueError(f"max_per_axis must be at least 1, got {max_per_axis}")
    if padding < 0.0:
        raise ValueError(f"padding must not be negative, got {padding}")

    if geom.kind == "sphere":
        local = _sphere_spheres(geom.size)
    elif geom.kind == "cylinder":
        local = _cylinder_spheres(geom.size, spacing, max_per_axis)
    else:
        local = _box_spheres(geom.size, spacing, max_per_axis)
    if not local:
        return []

    # Build in float64 regardless of the chain dtype: this runs once at setup,
    # and the numbers end up as plain Python floats that SphereModel casts to
    # whatever dtype the caller's q asks for later.
    R = T.rpy_to_matrix(torch.tensor(geom.origin_rpy, dtype=torch.float64))
    t = torch.tensor(geom.origin_xyz, dtype=torch.float64)
    centers = torch.tensor([c for c, _ in local], dtype=torch.float64)  # (S,3)
    world = centers @ R.transpose(0, 1) + t
    return [(float(p[0]), float(p[1]), float(p[2]), float(r) + padding)
            for p, (_, r) in zip(world, local)]


def auto_spheres(robot_ir, chain, spacing: float = 0.05, padding: float = 0.0,
                 max_per_axis: int = 8, fallback_to_visual: bool = False):
    """Sphere dict for a whole robot: {link_index: [(x, y, z, r), ...]}.

    Keys are indices into `chain`, so the result drops straight into
    SphereModel or merges with hand-written spheres. Links that produce nothing
    are left out entirely rather than mapped to an empty list.

    `spacing` is the target cell size in metres: no sphere covers more than
    `spacing` of the primitive along any axis, subject to `max_per_axis`, which
    keeps a large box from turning into thousands of spheres. `padding` inflates
    every radius, which is the cheap way to ask for a safety margin.

    `fallback_to_visual` reuses the visual geometry for links that declare no
    collision geometry. Handy for hand-written URDFs that only ever filled in
    <visual>, but visual meshes are usually finer than collision shapes, so it
    is off by default.
    """
    out = {}
    for name, link in robot_ir.links.items():
        idx = chain.link_index.get(name)
        if idx is None:
            continue  # link is not part of this compiled chain
        geom = link.collision
        if (geom is None or geom.kind not in SUPPORTED_KINDS) and fallback_to_visual:
            geom = link.visual
        sl = spheres_for_geometry(geom, spacing=spacing, padding=padding,
                                  max_per_axis=max_per_axis)
        if sl:
            out[idx] = sl
    return out


def _reason(geom):
    """Why a Geometry yields no spheres, or None if it is usable."""
    if geom is None or geom.kind == "none":
        return "none"
    if geom.kind not in SUPPORTED_KINDS:
        return "mesh"
    if not any(float(v) > 0.0 for v in geom.size):
        return "degenerate"
    return None


def unsupported_links(robot_ir, chain, fallback_to_visual: bool = False):
    """Names of chain links that contribute no spheres, and why.

    Returns {link_name: reason} with reason "none" (no collision geometry),
    "mesh" (a mesh, which this module does not approximate) or "degenerate" (a
    primitive with no size at all). The reason always describes the collision
    geometry, since that is what a URDF author would fix, even when
    `fallback_to_visual` rescued the link; rescued links are left out.

    Worth printing once after building a model: a link listed here is invisible
    to collision checking, which is a silent hole unless you know about it.
    """
    out = {}
    for name, link in robot_ir.links.items():
        if name not in chain.link_index:
            continue
        why = _reason(link.collision)
        if why is None:
            continue
        if fallback_to_visual and _reason(link.visual) is None:
            continue
        out[name] = why
    return out


def auto_sphere_model(robot_ir, chain, spacing: float = 0.05,
                      padding: float = 0.0, max_per_axis: int = 8,
                      fallback_to_visual: bool = False) -> SphereModel:
    """Collision SphereModel built from the IR's collision primitives.

    See `auto_spheres` for what the knobs do. A robot with no usable collision
    geometry gives back an empty model (`model.n == 0`) instead of raising, so
    calling code can check once and skip collision terms rather than guard every
    load.
    """
    spheres = auto_spheres(robot_ir, chain, spacing=spacing, padding=padding,
                           max_per_axis=max_per_axis,
                           fallback_to_visual=fallback_to_visual)
    model = SphereModel(chain, spheres)
    if model.n == 0:
        # torch.tensor([]) is (0,), and the batched center math wants (0, 3).
        # Fixing the shape here keeps an empty model usable in the same code
        # paths as a full one.
        model.local = model.local.reshape(0, 3)
    return model
