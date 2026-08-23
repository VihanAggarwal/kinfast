# src/kinfast/control.py
"""Classical model-based controllers built on the dynamics layer, plus a
semi-implicit Euler simulator to run them in closed loop.

  gravity_compensation  hold against gravity with zero stiffness
  pd_gravity            PD servo + gravity feedforward (setpoint regulation)
  computed_torque       feedback linearization: exact model inversion so the
                        closed-loop error dynamics are linear (trajectory tracking)
  simulate              roll out forward dynamics under any controller callable

Everything is batched: B robots can be simulated/controlled simultaneously,
and the whole loop is autograd-traceable, so controller gains are learnable.
"""
import torch
from kinfast import dynamics as D


def _gain(k, q):
    """Bring a scalar or (dof,) gain onto q's dtype and device. A tensor gain
    that requires grad keeps its graph (the cast is differentiable)."""
    if isinstance(k, torch.Tensor):
        return k.to(dtype=q.dtype, device=q.device)
    return torch.as_tensor(k, dtype=q.dtype, device=q.device)


def gravity_compensation(chain, q, qd=None):
    """tau that exactly cancels gravity at q. (B,dof)."""
    return D.gravity(chain, q)


def pd_gravity(chain, q, qd, q_des, kp, kd):
    """PD around q_des with gravity feedforward. kp/kd: scalars or (dof,)."""
    kp, kd = _gain(kp, q), _gain(kd, q)
    return kp * (q_des - q) - kd * qd + D.gravity(chain, q)


def computed_torque(chain, q, qd, q_des, qd_des, qdd_des, kp, kd):
    """Feedback linearization: tau = M(q) v + c(q,qd) + g(q) with
    v = qdd_des + kp e + kd de. Closed-loop error obeys e'' + kd e' + kp e = 0."""
    kp, kd = _gain(kp, q), _gain(kd, q)
    v = (qdd_des + kp * (q_des - q) + kd * (qd_des - qd)).to(q.dtype)
    M = D.mass_matrix(chain, q)
    return (M @ v.unsqueeze(-1)).squeeze(-1) + D.coriolis(chain, q, qd) + D.gravity(chain, q)


def _as_torque(tau, q):
    """Coerce a controller's output to a (B,dof) tensor matching q.

    Accepts a scalar, a (dof,) vector shared by every robot in the batch, a
    (1,dof) row broadcast over the batch, or a full (B,dof) tensor. Anything
    else is rejected here with a message that names the controller, instead
    of broadcasting into forward_dynamics and blowing up a step later inside
    FK with an unrelated shape error.
    """
    if tau is None:
        raise ValueError("controller returned None, expected a torque of shape "
                         f"{tuple(q.shape)}")
    tau = torch.as_tensor(tau, dtype=q.dtype, device=q.device)
    dof = q.shape[-1]
    if tau.dim() == 0:
        return tau.expand_as(q)
    if tau.dim() == 1 and tau.shape[0] == dof:
        return tau.unsqueeze(0).expand_as(q)
    if tau.dim() == 2 and tau.shape[0] == 1 and tau.shape[1] == dof:
        return tau.expand_as(q)
    if tau.shape != q.shape:
        raise ValueError(f"controller returned tau of shape {tuple(tau.shape)}, "
                         f"expected {tuple(q.shape)} (or a scalar, ({dof},), or (1,{dof}))")
    return tau


def simulate(chain, q0, qd0, controller, dt: float, steps: int,
             record_every: int = 1):
    """Roll out the robot under `controller(t, q, qd) -> tau`.

    Semi-implicit Euler at fixed dt for `steps` steps. The controller is
    called with the time t of the state it is given and must return a torque
    of shape (B,dof); a scalar, a (dof,) vector, or a (1,dof) row is also
    accepted and broadcast over the batch.

    Returns (ts, qs, qds) with qs/qds stacked along dim 0: (n_rec, B, dof).
    qs[i] and qds[i] are the state at time ts[i], where ts[i] is the time
    reached after the step that produced that record. The state after the
    first step is recorded at ts[0] = dt, then every `record_every` steps;
    the initial state (q0, qd0, t = 0) is the caller's input and is not
    included. With record_every > steps only the first step is recorded.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if record_every < 1:
        raise ValueError(f"record_every must be >= 1, got {record_every}")
    q = q0.clone()
    qd = qd0.to(device=q0.device, dtype=q0.dtype).clone()
    ts, qs, qds = [], [], []
    for k in range(steps):
        t = k * dt
        tau = _as_torque(controller(t, q, qd), q)
        qdd = D.forward_dynamics(chain, q, qd, tau)
        qd = qd + dt * qdd
        q = q + dt * qd
        if k % record_every == 0:
            ts.append((k + 1) * dt); qs.append(q.clone()); qds.append(qd.clone())
    return (torch.tensor(ts, dtype=q0.dtype, device=q0.device),
            torch.stack(qs), torch.stack(qds))
