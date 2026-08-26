# src/kinfast/collision_world.py
"""Differentiable distances from a robot's bounding spheres to world geometry.

collision.py answers one question: how close is the robot to a set of obstacle
spheres. Real cells are not made of spheres. A robot works over a floor,
beside walls, inside a shelf, around railings and cable trays. Packing a floor
with spheres takes hundreds of them and still leaves scalloped gaps between
them, and every one of those spheres costs a row in the distance computation.
The primitives here describe that geometry exactly and cost one row each:

    HalfSpace   everything on one side of a plane: floors, walls, ceilings,
                and conservative "stay in front of this line" keep-out planes
    Box         an axis-aligned box: tables, bins, shelves, the robot's own
                base cabinet
    Capsule     a segment swept by a radius: pipes, railings, cables, a human
                arm, a link of a second robot
    Sphere      the same shape collision.py already handles, kept here so a
                single world list can mix all four

Every shape answers the same question, `signed_distance(points)`: the distance
from a point to the shape's surface, negative inside the solid. Subtract the
robot sphere's radius and you have the signed distance between that sphere and
that shape, which is exactly what a collision cost or a safety margin wants.
Positive is clearance in metres, zero is touching, negative is penetration
depth.

Three properties are load bearing, and they match the rest of the library:

  * batched. Sphere centers come out of fk as (B, S, 3) and every shape is
    evaluated over that whole block at once, so a thousand candidate
    configurations cost one pass.
  * dtype and device agnostic. q fixes the working dtype and device, as in fk
    and dynamics; shape parameters are cast to follow it, so a float64 finite
    difference over a float32 chain works without touching the world.
  * differentiable. The distance is a smooth function of q wherever the
    nearest point is unique, so it drops straight into a gradient based IK or
    trajectory optimizer. Norms are evaluated with a floor under the square
    root so a point sitting exactly on a face, an edge or a segment endpoint
    yields a zero gradient rather than a NaN one.

Shape parameters may also carry a leading batch dimension, so `Box(center=(B,
3), ...)` describes an obstacle that sits somewhere different for each element
of the batch: a conveyor sampled at B times, or a second robot at B poses.

What this module does not do: rotated boxes, meshes, and swept volumes. An
oriented box is a real gap, not an oversight; expressing one needs a rotation
per shape and a decision about where that rotation comes from, and the
axis-aligned case already covers most workcell furniture.
"""
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Union

import torch

from kinfast.collision import SphereModel


def _eps(dtype) -> float:
    """Floor for the sum of squares under a norm, in the working dtype.

    finfo.eps squared means the floor shows up in the distance only at the
    level of finfo.eps itself (1e-16 in float64, 1e-7 in float32), which is
    below the precision of the distance anyway, while making the gradient of
    the norm exactly zero at the origin instead of 0/0.
    """
    return float(torch.finfo(dtype).eps) ** 2


def _norm(v: torch.Tensor) -> torch.Tensor:
    """Euclidean norm over the last axis with a differentiable value at zero."""
    return torch.sqrt((v * v).sum(dim=-1) + _eps(v.dtype))


def _cast(value: Any, like: torch.Tensor) -> torch.Tensor:
    """Put a shape parameter on the points' device in the points' dtype."""
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _vec(value: Any, points: torch.Tensor, name: str) -> torch.Tensor:
    """A 3-vector parameter, shaped to broadcast against points (..., 3).

    A plain (3,) vector applies to the whole batch. A (B, 3) vector describes
    geometry that moves per batch element; it is reshaped to (B, 1, ..., 1, 3)
    so it lines up with the leading batch axis of `points` rather than with the
    sphere axis.
    """
    t = _cast(value, points)
    if t.dim() == 1 and t.shape[0] == 3:
        return t
    if t.dim() == 2 and t.shape[-1] == 3:
        return t.reshape(t.shape[0], *([1] * (points.dim() - 2)), 3)
    raise ValueError(
        f"{name} must have shape (3,) or (B, 3) for a per-batch-element "
        f"obstacle, got {tuple(t.shape)}.")


def _scalar(value: Any, points: torch.Tensor, name: str) -> torch.Tensor:
    """A scalar parameter, shaped to broadcast against points (..., 3).

    Same rule as _vec: a scalar applies everywhere, a (B,) tensor gives one
    value per batch element.
    """
    t = _cast(value, points)
    if t.dim() == 0:
        return t
    if t.dim() == 1:
        return t.reshape(t.shape[0], *([1] * (points.dim() - 2)))
    raise ValueError(
        f"{name} must be a scalar or have shape (B,) for a per-batch-element "
        f"obstacle, got {tuple(t.shape)}.")


@dataclass
class HalfSpace:
    """The solid region on one side of a plane: n . x + d <= 0.

    The signed distance of a point is (n . x + d) / ||n||, positive on the free
    side. The normal is normalized on use, so an unnormalized normal still
    gives a true distance in metres rather than a scaled one; it must be
    nonzero.

    A floor at z = h is HalfSpace.floor(h): normal (0, 0, 1), offset -h, so the
    distance of a point is simply its height above the floor.
    """
    normal: Any
    offset: Any = 0.0

    @classmethod
    def floor(cls, height: float = 0.0) -> "HalfSpace":
        """The ground plane at z = height, with free space above it."""
        return cls(normal=(0.0, 0.0, 1.0), offset=-float(height))

    @classmethod
    def ceiling(cls, height: float) -> "HalfSpace":
        """The solid above z = height, with free space below it."""
        return cls(normal=(0.0, 0.0, -1.0), offset=float(height))

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        """(..., 3) points -> (...) signed distance, negative inside the solid."""
        n = _vec(self.normal, points, "HalfSpace.normal")
        d = _scalar(self.offset, points, "HalfSpace.offset")
        return ((points * n).sum(dim=-1) + d) / _norm(n)


@dataclass
class Box:
    """An axis-aligned solid box, given by its center and half extents.

    The signed distance is exact both outside and inside. With
    e = |x - center| - half_extents, the distance is

        ||max(e, 0)||  +  min(max(e_x, e_y, e_z), 0)

    The first term is the true distance outside the box (it handles the face,
    edge and corner regions in one expression), the second is zero out there
    and equals the negative distance to the nearest face when the point is
    inside, which is what a penetration depth means for a box.
    """
    center: Any
    half_extents: Any

    @classmethod
    def from_bounds(cls, lower: Sequence[float], upper: Sequence[float]) -> "Box":
        """Build from opposite corners instead of center and half extents."""
        lo = torch.as_tensor(lower, dtype=torch.float64)
        hi = torch.as_tensor(upper, dtype=torch.float64)
        if torch.any(hi < lo):
            raise ValueError("Box.from_bounds needs upper >= lower on every axis.")
        return cls(center=(lo + hi) / 2, half_extents=(hi - lo) / 2)

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        """(..., 3) points -> (...) signed distance, negative inside the box."""
        c = _vec(self.center, points, "Box.center")
        h = _vec(self.half_extents, points, "Box.half_extents")
        e = (points - c).abs() - h
        outside = _norm(e.clamp(min=0.0))
        inside = e.max(dim=-1).values.clamp(max=0.0)
        return outside + inside


@dataclass
class Capsule:
    """A segment from a to b, thickened by a radius.

    The distance from a point to the capsule is the distance to the segment
    minus the radius. The segment parameter is clamped to [0, 1], which is what
    turns an infinite line into a segment with round caps, and it is also why a
    capsule is the cheapest useful non-convex-looking obstacle: pipes, railings
    and limbs are all one segment plus one number.

    A degenerate capsule (a == b) is a sphere and is handled without dividing
    by zero.
    """
    a: Any
    b: Any
    radius: Any

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        """(..., 3) points -> (...) signed distance, negative inside the capsule."""
        a = _vec(self.a, points, "Capsule.a")
        b = _vec(self.b, points, "Capsule.b")
        r = _scalar(self.radius, points, "Capsule.radius")
        ab = b - a
        ap = points - a
        denom = (ab * ab).sum(dim=-1).clamp(min=torch.finfo(points.dtype).tiny)
        t = ((ap * ab).sum(dim=-1) / denom).clamp(0.0, 1.0)
        closest = a + t.unsqueeze(-1) * ab
        return _norm(points - closest) - r


@dataclass
class Sphere:
    """A solid ball. The same shape collision.py handles, in world-list form."""
    center: Any
    radius: Any

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        """(..., 3) points -> (...) signed distance, negative inside the ball."""
        c = _vec(self.center, points, "Sphere.center")
        r = _scalar(self.radius, points, "Sphere.radius")
        return _norm(points - c) - r


Shape = Union[HalfSpace, Box, Capsule, Sphere]
World = Union[Shape, Iterable[Shape]]


def as_shapes(world: World) -> List[Shape]:
    """Normalize a world argument to a list of shapes.

    A single shape is a world of one, which keeps the common case
    (`distance_to_world(model, q, HalfSpace.floor())`) free of brackets.
    """
    if hasattr(world, "signed_distance"):
        return [world]  # type: ignore[list-item]
    shapes = list(world)  # type: ignore[arg-type]
    for s in shapes:
        if not hasattr(s, "signed_distance"):
            raise TypeError(
                f"world entries must be collision_world shapes with a "
                f"signed_distance method, got {type(s).__name__}.")
    return shapes


def shape_distances(points: torch.Tensor, world: World) -> torch.Tensor:
    """Signed distance from every point to every shape. (..., 3) -> (..., K).

    K is the number of shapes, in the order they appear in `world`, so the last
    axis can be used to say which piece of the world is the near one.
    """
    shapes = as_shapes(world)
    if not shapes:
        return points.new_zeros((*points.shape[:-1], 0))
    per_shape = [s.signed_distance(points) for s in shapes]
    if len(per_shape) > 1:
        per_shape = list(torch.broadcast_tensors(*per_shape))
    return torch.stack(per_shape, dim=-1)


def world_distances(model: SphereModel, q: torch.Tensor,
                    world: World) -> torch.Tensor:
    """Signed distance from every robot sphere to every world shape.

    -> (B, S, K) with S the spheres of the model and K the shapes of the world,
    both in their own order. Negative means that sphere overlaps that shape.
    Keep this when you want to know which link is the near one or to build a
    per-sphere cost; use distance_to_world when a single clearance number is
    enough.
    """
    centers = model.centers_world(q)                       # (B, S, 3)
    radius = model.radius.to(device=q.device, dtype=q.dtype)
    return shape_distances(centers, world) - radius[None, :, None]


def distance_to_world(model: SphereModel, q: torch.Tensor,
                      world: World) -> torch.Tensor:
    """Minimum signed distance from the robot to the world. -> (B,).

    Negative means the robot is in collision at that configuration, and the
    magnitude is how deep. An empty world returns +inf, the honest answer for
    "how close are you to nothing".

    Differentiable in q: the gradient flows through whichever sphere and shape
    happen to be nearest, which is the standard subgradient a collision cost
    uses.
    """
    d = world_distances(model, q, world)                   # (B, S, K)
    batch = q.shape[0]
    if d.numel() == 0:
        return torch.full((batch,), float("inf"), dtype=q.dtype, device=q.device)
    return d.reshape(batch, -1).min(dim=1).values


def in_collision(model: SphereModel, q: torch.Tensor, world: World,
                 margin: float = 0.0) -> torch.Tensor:
    """Boolean (B,) mask: is the robot within `margin` of the world.

    A positive margin treats near misses as collisions, which is how you use a
    distance field as a safety check rather than as a physics test.
    """
    return distance_to_world(model, q, world) < margin


def world_clearance_cost(model: SphereModel, q: torch.Tensor, world: World,
                         margin: float = 0.05) -> torch.Tensor:
    """Hinge penalty on every sphere-shape pair closer than `margin`. -> (B,).

    sum over pairs of relu(margin - d)^2. Unlike the plain minimum this sees
    every violated pair at once and is smooth where it is nonzero, so it is the
    better term to hand an optimizer; the minimum is the better number to
    report to a human.
    """
    d = world_distances(model, q, world)
    if d.numel() == 0:
        return torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
    return (torch.relu(margin - d) ** 2).reshape(q.shape[0], -1).sum(dim=1)
