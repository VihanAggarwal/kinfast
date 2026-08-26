# src/kinfast/control_task.py
"""Operational-space (Cartesian) impedance control.

A joint-space PD servo makes joints track angles. An operational-space
controller makes the end effector behave like a mass on a spring in Cartesian
space, which is what you actually want when the task is stated in the world
("hold this point", "stay soft along z while pressing") and when the arm has
to stay compliant against contact.

The core law is

    tau = J^T ( Kp (x_des - x) - Kd (J qd) ) + g(q)

read right to left: g(q) makes the arm weightless, J qd is the current end
effector velocity, the bracket is the wrench a virtual spring-damper anchored
at x_des would apply to the end effector, and J^T maps that wrench back into
joint torques. There is no matrix inverse anywhere in that expression, which
is the point: the law is passive, so the closed loop is stable for any
positive definite gains, and it degrades gracefully at a singularity where an
inverse-kinematics servo would blow up. The price is that the closed-loop
task dynamics are shaped by the arm's own configuration-dependent inertia, so
the response is faster in some directions than others.

Pass `inertia_shaping=True` to buy that back. The wrench is then premultiplied
by the operational-space inertia Lambda = (J M^-1 J^T)^-1 and corrected by the
task-space bias mu, which is Khatib's controller; the closed loop becomes the
decoupled linear system

    xdd = Kp (x_des - x) - Kd xd

in every task direction, at the cost of needing the full dynamic model and a
matrix inverse per step.

A redundant arm has joint motions that move nothing in the task, and those
have to be dealt with or the arm drifts and oscillates in its own nullspace.
`null_kd` adds joint damping there, `null_kp` with `q_rest` adds a posture
spring, and both are pushed through a nullspace projector so they cannot
disturb the task.

  task_error           x_des - x for a position or a full pose target
  task_jacobian        the Jacobian rows the task actually uses
  task_inertia         Lambda, the inertia the end effector appears to have
  jdot_qd              the Jdot qd term of the task acceleration
  nullspace_projector  torques that produce no task motion
  opspace_impedance    the control law above, batched
  impedance_controller wraps it into a controller(t, q, qd) for control.simulate

Everything is batched over a leading B dimension, follows q's dtype and
device the way the rest of the library does, and is autograd-traceable, so
Cartesian gains can be learned through a closed-loop rollout.
"""
import torch

from kinfast import dynamics as D
from kinfast import transforms as T
from kinfast.analysis import _check_rows
from kinfast.control import _gain
from kinfast.fk import fk_rp
from kinfast.jacobian import jacobian_rp, _resolve_link

_POSITION_ROWS = (0, 1, 2)
_POSE_ROWS = (0, 1, 2, 3, 4, 5)


def _target_rows(x_des, rows):
    """Work out which of the six task rows this call servos.

    A (B,3) target is a position, so only the three linear rows are on the
    table; a (B,4,4) target is a full pose and any of the six rows are fair
    game. An explicit `rows` narrows that further, which is how you drive a
    planar arm (rows=(0, 2) for an x-z arm) without the identically zero
    out-of-plane row making the projector and Lambda singular.
    """
    pose = x_des.dim() == 3
    default = _POSE_ROWS if pose else _POSITION_ROWS
    if rows is None:
        return list(default), pose
    rows = _check_rows(rows)
    if not pose:
        bad = [r for r in rows if r > 2]
        if bad:
            raise ValueError(
                f"rows {rows} asks for orientation rows {bad}, but the target "
                "is a (B,3) position; pass a (B,4,4) pose target to servo "
                "orientation")
    return rows, pose


def _check_target(x_des, q):
    """Bring the target onto q's dtype/device and reject shapes early."""
    x_des = torch.as_tensor(x_des, dtype=q.dtype, device=q.device)
    if x_des.dim() == 1 and x_des.shape[0] == 3:
        x_des = x_des.unsqueeze(0)
    if x_des.dim() == 2 and x_des.shape[-1] == 4 and x_des.shape[-2] == 4:
        x_des = x_des.unsqueeze(0)
    ok = (x_des.dim() == 2 and x_des.shape[-1] == 3) or \
         (x_des.dim() == 3 and x_des.shape[-2:] == (4, 4))
    if not ok:
        raise ValueError(
            f"x_des must be a (B,3) position or a (B,4,4) pose, got shape "
            f"{tuple(x_des.shape)}")
    if x_des.shape[0] not in (1, q.shape[0]):
        raise ValueError(
            f"x_des batch {x_des.shape[0]} does not match q batch {q.shape[0]}")
    if x_des.shape[0] == 1 and q.shape[0] != 1:
        x_des = x_des.expand(q.shape[0], *x_des.shape[1:])
    return x_des


def _task_gain(k, m, q, name):
    """Normalize a task gain to an (m,m) (or (B,m,m)) matrix.

    A scalar means the same stiffness in every task direction, an (m,)
    vector means one per direction, and a full matrix lets you tilt the
    ellipsoid off the world axes (stiff along an insertion axis, soft
    across it). Keeping one matrix form internally means the control law
    is a single matmul instead of three broadcasting cases.
    """
    K = k if isinstance(k, torch.Tensor) else torch.as_tensor(k)
    K = K.to(dtype=q.dtype, device=q.device)
    if K.dim() == 0:
        return K * torch.eye(m, dtype=q.dtype, device=q.device)
    if K.dim() == 1:
        if K.shape[0] != m:
            raise ValueError(
                f"{name} has {K.shape[0]} entries but the task has {m} rows")
        return torch.diag_embed(K)
    if K.dim() in (2, 3) and K.shape[-1] == m and K.shape[-2] == m:
        return K
    raise ValueError(
        f"{name} must be a scalar, an ({m},) vector, or an ({m},{m}) matrix, "
        f"got shape {tuple(K.shape)}")


def _mv(A, v):
    """Batched matrix times vector, keeping the (B,n) vector layout."""
    return (A @ v.unsqueeze(-1)).squeeze(-1)


def task_error(chain, q, x_des, link_index=-1, rows=None, rp=None):
    """World-frame task error x_des - x at the given link. (B,m).

    For a (B,3) position target this is simply the position difference. For a
    (B,4,4) pose target the last three entries are the rotation vector of
    R_des R^T, the world-frame rotation that takes the current orientation to
    the target one, which is the standard small-error parameterization that
    pairs with the geometric Jacobian's angular rows.
    """
    link = _resolve_link(chain, link_index)
    x_des = _check_target(x_des, q)
    rows, pose = _target_rows(x_des, rows)
    if rp is None:
        rp = fk_rp(chain, q)
    wR, wp = rp
    if pose:
        cur = T.make_transform(wR[link], wp[link])
        err = T.pose_error(cur, x_des)
    else:
        err = x_des - wp[link]
    return err[:, rows]


def task_jacobian(chain, q, link_index=-1, rows=None, rp=None):
    """The rows of the geometric Jacobian this task uses. (B,m,dof).

    Rows 0-2 are linear velocity, rows 3-5 angular, both in world frame.
    """
    link = _resolve_link(chain, link_index)
    rows = list(_POSE_ROWS) if rows is None else _check_rows(rows)
    J = jacobian_rp(chain, q, link, rp=rp)
    return J[:, rows, :]


def task_inertia(chain, q, link_index=-1, rows=None, damping: float = 1e-4,
                 J=None, M=None):
    """Operational-space inertia Lambda = (J M^-1 J^T)^-1. (B,m,m).

    This is the inertia the end effector appears to have when you push on it
    along each task direction: large along a stretched-out arm, small across
    it. Premultiplying a desired task acceleration by Lambda turns it into the
    wrench that actually produces it.

    `damping` regularizes the inverse (lambda^2 I is added before inverting),
    which matters near a singularity where J M^-1 J^T loses rank.
    """
    if J is None:
        J = task_jacobian(chain, q, link_index, rows)
    if M is None:
        M = D.mass_matrix(chain, q)
    m = J.shape[-2]
    eye = torch.eye(m, dtype=q.dtype, device=q.device)
    JMinvJt = J @ torch.linalg.solve(M, J.transpose(-1, -2))
    return torch.linalg.inv(JMinvJt + (damping * damping) * eye)


def nullspace_projector(J, M=None, damping: float = 1e-4):
    """Projector N^T with J N^T = 0 (or J M^-1 N^T = 0). (B,dof,dof).

    Torques sent through this projector are invisible to the task, so a
    posture spring or joint damping can run underneath a Cartesian task
    without fighting it.

    Without M this is the kinematic projector I - J^T (J J^T)^-1 J, which
    kills the task *velocity* a torque would produce if it were read as a
    joint velocity. With M it is the dynamically consistent projector
    I - J^T Lambda J M^-1, the one that leaves the task *acceleration*
    untouched; that is the right choice when the task loop already shapes
    inertia, and it is what `opspace_impedance` uses under inertia_shaping.
    """
    B, m, dof = J.shape
    eye_m = torch.eye(m, dtype=J.dtype, device=J.device)
    eye_n = torch.eye(dof, dtype=J.dtype, device=J.device).expand(B, dof, dof)
    lam2 = damping * damping
    Jt = J.transpose(-1, -2)
    if M is None:
        inner = torch.linalg.solve(J @ Jt + lam2 * eye_m, J)
        return eye_n - Jt @ inner
    Minv_Jt = torch.linalg.solve(M, Jt)                       # (B,dof,m)
    Lam = torch.linalg.inv(J @ Minv_Jt + lam2 * eye_m)        # (B,m,m)
    Minv = torch.linalg.solve(M, eye_n)
    return eye_n - Jt @ Lam @ J @ Minv


def jdot_qd(chain, q, qd, link_index=-1, rows=None):
    """The Jdot qd term of the task acceleration xdd = J qdd + Jdot qd. (B,m).

    With qd held fixed, d/dt (J qd) is the directional derivative of J qd
    along qd, so it falls out of autograd with the same double-backward trick
    the joint-space Coriolis term uses; no hand-derived Jdot is needed. The
    result keeps its graph when q or qd require grad, and it runs under
    torch.no_grad by turning grad mode back on locally (torch.inference_mode
    cannot be undone from the inside, so that case raises, matching
    dynamics.coriolis).
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("jdot_qd needs autograd; call it outside "
                         "torch.inference_mode (torch.no_grad is fine)")
    qd = qd.to(dtype=q.dtype, device=q.device)
    needs_graph = torch.is_grad_enabled() and (q.requires_grad or qd.requires_grad)
    with torch.enable_grad():
        qr = q if q.requires_grad else q.detach().requires_grad_(True)
        J = task_jacobian(chain, qr, link_index, rows)
        xd = _mv(J, qd)                                        # (B,m)
        v = torch.ones_like(xd, requires_grad=True)
        gq, = torch.autograd.grad((xd * v).sum(), qr, create_graph=True,
                                  allow_unused=True)
        if gq is None:                       # J does not depend on q at all
            out = torch.zeros_like(xd)
        else:
            out, = torch.autograd.grad((gq * qd).sum(), v, create_graph=True,
                                       allow_unused=True)
            if out is None:
                out = torch.zeros_like(xd)
    return out if needs_graph else out.detach()


def opspace_impedance(chain, q, qd, x_des, kp, kd, link_index=-1, *,
                      xd_des=None, rows=None, use_gravity: bool = True,
                      inertia_shaping: bool = False, damping: float = 1e-4,
                      null_kd=0.0, null_kp=0.0, q_rest=None, tau_null=None):
    """Cartesian impedance torque tau = J^T (Kp e - Kd xd) + g(q). (B,dof).

    Arguments
      chain       CompiledChain
      q, qd       (B,dof) state; q fixes the working dtype and device
      x_des       (B,3) position target or (B,4,4) pose target. A single
                  (3,) / (4,4) target is broadcast over the batch.
      kp, kd      task stiffness and damping: scalar, (m,) per task row, or
                  an (m,m) matrix, with m the number of task rows
      link_index  the controlled link; negative indices count from the end,
                  so the default -1 is the last link
      xd_des      optional (B,m) feedforward task velocity; damping then acts
                  on (xd - xd_des), which is what you want when tracking a
                  Cartesian trajectory rather than holding a setpoint
      rows        which of the six task rows to servo (see task_error)
      use_gravity add the gravity feedforward g(q). Off if the hardware
                  already compensates gravity, or to study the sag
      inertia_shaping  premultiply by Lambda and add the task-space bias mu,
                  giving decoupled closed-loop task dynamics at the cost of
                  the full dynamic model
      damping     regularization for the Lambda / projector inverses only;
                  the plain law inverts nothing and ignores it
      null_kd     joint-space damping applied in the nullspace
      null_kp, q_rest  posture spring pulling toward q_rest in the nullspace
      tau_null    optional (B,dof) extra torque to run in the nullspace

    The nullspace terms are only computed when at least one of them is
    non-zero, so the default path costs one FK, one Jacobian and one gravity
    evaluation.
    """
    qd = qd.to(dtype=q.dtype, device=q.device)
    x_des = _check_target(x_des, q)
    rows, _ = _target_rows(x_des, rows)
    m = len(rows)

    rp = fk_rp(chain, q)
    J = task_jacobian(chain, q, link_index, rows, rp=rp)
    e = task_error(chain, q, x_des, link_index, rows, rp=rp)
    xd = _mv(J, qd)
    if xd_des is not None:
        xd_des = torch.as_tensor(xd_des, dtype=q.dtype, device=q.device)
        if xd_des.dim() == 1:
            xd_des = xd_des.unsqueeze(0)
        if xd_des.shape[-1] != m:
            raise ValueError(
                f"xd_des has {xd_des.shape[-1]} entries but the task has {m} rows")
        xd = xd - xd_des

    Kp = _task_gain(kp, m, q, "kp")
    Kd = _task_gain(kd, m, q, "kd")
    a_des = _mv(Kp, e) - _mv(Kd, xd)

    M = None
    if inertia_shaping:
        M = D.mass_matrix(chain, q)
        Lam = task_inertia(chain, q, link_index, rows, damping, J=J, M=M)
        # mu is the task-space bias: it cancels exactly the part of the task
        # acceleration that the joint-space Coriolis force and the moving
        # Jacobian contribute, leaving xdd = a_des.
        c = D.coriolis(chain, q, qd)
        mu = _mv(Lam, _mv(J, torch.linalg.solve(M, c.unsqueeze(-1)).squeeze(-1))
                 - jdot_qd(chain, q, qd, link_index, rows))
        wrench = _mv(Lam, a_des) + mu
    else:
        wrench = a_des

    tau = _mv(J.transpose(-1, -2), wrench)

    tau0 = None
    if not (isinstance(null_kd, (int, float)) and null_kd == 0.0):
        tau0 = -_gain(null_kd, q) * qd
    if not (isinstance(null_kp, (int, float)) and null_kp == 0.0):
        if q_rest is None:
            raise ValueError("null_kp needs q_rest, the posture to pull toward")
        q_rest = torch.as_tensor(q_rest, dtype=q.dtype, device=q.device)
        term = _gain(null_kp, q) * (q_rest - q)
        tau0 = term if tau0 is None else tau0 + term
    if tau_null is not None:
        term = torch.as_tensor(tau_null, dtype=q.dtype, device=q.device)
        tau0 = term if tau0 is None else tau0 + term
    if tau0 is not None:
        N = nullspace_projector(J, M=M, damping=damping)
        tau = tau + _mv(N, tau0.expand_as(q) if tau0.dim() == 1 else tau0)

    if use_gravity:
        tau = tau + D.gravity(chain, q)
    return tau


def impedance_controller(chain, x_des, kp, kd, link_index=-1, *,
                         xd_des=None, **kwargs):
    """Wrap opspace_impedance into a controller(t, q, qd) for control.simulate.

    x_des and xd_des may be tensors (a fixed setpoint) or callables of the
    simulation time t returning the target at that instant, which is how you
    feed a Cartesian trajectory to the same law without rewriting it.
    """
    def controller(t, q, qd):
        target = x_des(t) if callable(x_des) else x_des
        vel = xd_des(t) if callable(xd_des) else xd_des
        return opspace_impedance(chain, q, qd, target, kp, kd, link_index,
                                 xd_des=vel, **kwargs)
    return controller
