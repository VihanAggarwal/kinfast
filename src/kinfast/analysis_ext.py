# src/kinfast/analysis_ext.py
"""Ellipsoid-level dexterity analysis: the shape of what an arm can do.

kinfast.analysis reduces a Jacobian to a single number (Yoshikawa's
manipulability, the condition number). Those numbers say how much motion is
available but not which way it points. This module keeps the directions:

  manipulability_ellipsoid  velocity ellipsoid {J qd : ||qd|| <= 1}: principal
                            axis directions and semi-axis lengths, i.e. the
                            eigendecomposition of J J^T.
  dynamic_manipulability    acceleration ellipsoid {J M^-1 tau : ||tau|| <= 1}:
                            the same picture for J M^-1 M^-T J^T, which weights
                            each direction by how hard the arm has to push.
  singularity_proximity     the smallest singular value of J: the length of the
                            shortest ellipsoid axis, which is also how far the
                            Jacobian is from losing rank. Zero exactly at a
                            singularity.
  ellipsoid                 the shared primitive, for a matrix you already have.

Why singular values rather than a literal eigendecomposition of J J^T: they are
the same thing (J = U S V^T gives J J^T = U S^2 U^T, so the eigenvectors of
J J^T are the columns of U and its eigenvalues are the squared singular
values), but forming J J^T squares the condition number and loses half the
available precision near a singularity, which is exactly where these functions
are read. So the axes come from the SVD of J and the eigenvalues are reported
as lengths**2.

Everything is batched over a leading B dimension, runs on whatever device q
lives on, and works in q's dtype, like the rest of the library. Singular values
are differentiable wherever they are distinct and non-zero; at a repeated or
zero singular value the axis directions are not unique and the gradient of the
decomposition does not exist, which is a property of the ellipsoid and not of
this implementation.
"""
import torch
from kinfast.analysis import _check_rows
from kinfast.dynamics import mass_matrix
from kinfast.jacobian import jacobian


def task_jacobian(chain, q, link_index, translational: bool = True, rows=None):
    """The rows of the geometric Jacobian that make up the task space. (B,m,dof).

    Same selection rules as kinfast.analysis: `rows` picks explicit rows of the
    6-row Jacobian (use it for planar arms, e.g. rows=(0, 1) for an xy-planar
    arm, otherwise the identically zero out-of-plane row flattens every
    ellipsoid); with no selection you get the 3 linear rows
    (translational=True) or all 6.
    """
    J = jacobian(chain, q, link_index)
    if rows is not None:
        return J[:, _check_rows(rows), :]
    if translational:
        return J[:, :3, :]
    return J


def ellipsoid(A: torch.Tensor) -> dict:
    """Principal axes of the ellipsoid {A u : ||u|| <= 1}, for A of shape (B,m,n).

    Returns a dict:
      axes         (B,m,m) orthonormal directions, one per COLUMN: axes[b,:,i]
                   is the direction of axis i. These are the eigenvectors of
                   A A^T. Sign is arbitrary (an axis and its negative describe
                   the same ellipsoid), so compare directions up to sign.
      lengths      (B,m) semi-axis lengths in descending order, the singular
                   values of A. If n < m the ellipsoid is flat and the last
                   m - n lengths are exactly zero, with axes still spanning the
                   full task space.
      eigenvalues  (B,m) = lengths**2, the eigenvalues of A A^T.
      volume       (B,) product of the semi-axis lengths, equal to
                   sqrt(det(A A^T)). The true m-dimensional volume is this
                   times the volume of the unit m-ball, a constant that drops
                   out of any comparison between configurations.

    A flat ellipsoid deliberately reports volume 0 rather than the 1 that a
    determinant of an empty matrix would give, so a chain with no movable
    joints reads as fully singular instead of perfectly dexterous.
    """
    if A.ndim != 3:
        raise ValueError(
            f"ellipsoid expects a batched matrix of shape (B, m, n), got shape "
            f"{tuple(A.shape)}")
    B, m, n = A.shape
    if m == 0:
        raise ValueError(
            "ellipsoid needs at least one task row; a 0-dimensional task space "
            "has no axes to report")
    U, s, _ = torch.linalg.svd(A, full_matrices=True)
    if s.shape[-1] < m:
        pad = torch.zeros(B, m - s.shape[-1], dtype=s.dtype, device=s.device)
        s = torch.cat([s, pad], dim=-1)
    return {
        "axes": U,
        "lengths": s,
        "eigenvalues": s * s,
        "volume": s.prod(dim=-1),
    }


def manipulability_ellipsoid(chain, q, link_index, translational: bool = True,
                             rows=None) -> dict:
    """Velocity manipulability ellipsoid at q. Keys as in `ellipsoid`.

    The set of end-effector velocities reachable with a unit-norm joint
    velocity, {J qd : ||qd|| <= 1}. A long axis is a direction the arm moves
    easily; a short one is a direction it barely moves at all. `volume` is
    exactly Yoshikawa's manipulability sqrt(det(J J^T)), so this is the
    directional version of kinfast.analysis.manipulability.
    """
    J = task_jacobian(chain, q, link_index, translational, rows)
    return ellipsoid(J)


def dynamic_manipulability(chain, q, link_index, translational: bool = True,
                           rows=None) -> dict:
    """Dynamic (acceleration) manipulability ellipsoid at q. Keys as in `ellipsoid`.

    The set of end-effector accelerations reachable with a unit-norm joint
    torque, {J M(q)^-1 tau : ||tau|| <= 1}, whose shape matrix is
    J M^-1 M^-T J^T. Where the velocity ellipsoid asks how the geometry maps
    joint speed to task speed, this one also asks how much of the arm's mass
    each direction has to accelerate: a direction can be kinematically easy and
    dynamically expensive at the same time, which is what decides whether a
    fast move is actually feasible.

    Only the M^-1 tau part is kept, so the ellipsoid is centred on the origin
    rather than on the acceleration the arm already has from gravity and its
    own velocity (Jdot qd and the M^-1(c + g) offset). That is the usual
    convention and it makes the result a function of q alone.

    Raises ValueError if the mass matrix is singular, which happens when some
    joint moves no mass at all: the ellipsoid would then be infinite in some
    direction. The model needs inertials on the links past that joint.
    """
    J = task_jacobian(chain, q, link_index, translational, rows)
    M = mass_matrix(chain, q)
    # A = J M^-1. Solve M^T X = J^T so that X = M^-T J^T = A^T; going through a
    # solve instead of an explicit inverse keeps the numerics honest, and using
    # M^T (rather than assuming symmetry) makes this literally the M^-1 M^-T of
    # the definition.
    msg = ("the mass matrix is singular at this configuration, so the dynamic "
           "manipulability ellipsoid is unbounded. Usually some joint carries "
           "only massless links; give those links <inertial> elements.")
    try:
        X = torch.linalg.solve(M.transpose(-1, -2), J.transpose(-1, -2))
    except RuntimeError as e:                       # LinAlgError subclasses this
        raise ValueError(msg) from e
    # Some backends return inf/NaN for a singular system instead of raising, so
    # the same case has to be caught by hand. Only blame the mass matrix when
    # the inputs themselves were finite; a NaN q must stay a NaN q.
    if not bool(torch.isfinite(X).all()):
        if bool(torch.isfinite(M).all()) and bool(torch.isfinite(J).all()):
            raise ValueError(msg)
    return ellipsoid(X.transpose(-1, -2))


def singularity_proximity(chain, q, link_index, translational: bool = True,
                          rows=None) -> torch.Tensor:
    """Smallest singular value of the task Jacobian. (B,). 0 exactly at a singularity.

    This is the shortest semi-axis of the velocity ellipsoid. When the arm has
    at least as many joints as task rows it is also the spectral-norm distance
    from J to the nearest rank-deficient Jacobian, so it measures how close the
    configuration really is to losing a degree of freedom. Prefer it to
    manipulability as a singularity alarm: the product of all the singular
    values can stay respectable while one of them collapses, whereas this one
    cannot.

    It carries the units of the Jacobian (metres per radian for a revolute
    arm), so compare it against a length scale of the robot rather than against
    an absolute threshold.

    An arm with fewer joints than task rows cannot span the task space at any
    configuration and reports 0 everywhere, as does a chain with no movable
    joints. That is the same trap the row selection guards against elsewhere:
    on a planar arm pass rows=(0, 1) so the identically zero out-of-plane row
    does not swamp the answer.
    """
    J = task_jacobian(chain, q, link_index, translational, rows)
    if J.shape[-1] == 0:
        return torch.zeros(J.shape[0], dtype=J.dtype, device=J.device)
    s = torch.linalg.svdvals(J)
    if s.shape[-1] < J.shape[-2]:       # fewer joints than task rows: flat
        return torch.zeros(J.shape[0], dtype=J.dtype, device=J.device)
    return s[:, -1]
