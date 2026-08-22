# src/kinfast/urdf/repair.py
"""Safe, kinematics-relevant repairs on the Robot IR.

Only fixes that affect FK/IK correctness live here (Phase 1). Inertia/mass and
collision-geometry repairs are Phase 2. Every change is recorded as a Finding.
"""
import math
from dataclasses import dataclass
from kinfast.ir import Robot, MOVABLE

_DEFAULT_LIMIT = (-math.pi, math.pi)
_CONTINUOUS_LIMIT = (-2.0 * math.pi, 2.0 * math.pi)


@dataclass
class Finding:
    code: str
    where: str
    message: str


def repair(robot: Robot):
    findings = []
    for j in robot.joints:
        if j.type in MOVABLE:
            _fix_axis(j, findings)
            _fix_limits(j, findings)
    for link in robot.links.values():
        _check_inertia(link, findings)
    return robot, findings


def _check_inertia(link, findings):
    """A real rigid body's principal inertias satisfy the triangle inequality
    (A + B >= C for every permutation); MuJoCo refuses models that violate it.
    Detect-only: fixing it means changing the physics, which is the user's call."""
    inr = link.inertial
    if inr is None or inr.mass <= 0.0:
        return
    ixx, iyy, izz, ixy, ixz, iyz = inr.inertia
    if ixx == iyy == izz == 0.0:
        return
    # principal moments via eigenvalues of the (symmetric) tensor
    I = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    try:
        import numpy as np
        a, b, c = sorted(np.linalg.eigvalsh(np.array(I)).tolist())
    except Exception:
        return
    if a < -1e-12:
        findings.append(Finding("negative_inertia", link.name,
                                f"principal inertia {a:.3g} < 0 (not physical)"))
    elif a + b < c * (1 - 1e-6):
        findings.append(Finding("inertia_triangle", link.name,
                                f"principal inertias {a:.3g}+{b:.3g} < {c:.3g}: "
                                "violates the triangle inequality (MuJoCo will reject this)"))


def _fix_axis(j, findings):
    norm = math.sqrt(sum(c * c for c in j.axis))
    if norm == 0.0:
        j.axis = (0.0, 0.0, 1.0)
        findings.append(Finding("zero_axis", j.name, "axis was zero; set to +z"))
    elif abs(norm - 1.0) > 1e-6:
        j.axis = tuple(c / norm for c in j.axis)
        findings.append(Finding("unnormalized_axis", j.name, "axis normalized"))


def _fix_limits(j, findings):
    lo, hi = j.limit
    if j.type == "continuous" and lo == 0.0 and hi == 0.0:
        j.limit = _CONTINUOUS_LIMIT
        return
    if lo == 0.0 and hi == 0.0:
        j.limit = _DEFAULT_LIMIT if j.type == "revolute" else (-1.0, 1.0)
        findings.append(Finding("missing_limits", j.name,
                                f"no limits; defaulted to {j.limit}"))
    elif lo > hi:
        j.limit = (hi, lo)
        findings.append(Finding("inverted_limits", j.name,
                                "lower > upper; swapped"))
