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


def gravity_compensation(chain, q, qd=None):
    """tau that exactly cancels gravity at q. (B,dof)."""
    return D.gravity(chain, q)


def pd_gravity(chain, q, qd, q_des, kp, kd):
    """PD around q_des with gravity feedforward. kp/kd: scalars or (dof,)."""
    return kp * (q_des - q) - kd * qd + D.gravity(chain, q)


def computed_torque(chain, q, qd, q_des, qd_des, qdd_des, kp, kd):
    """Feedback linearization: tau = M(q) v + c(q,qd) + g(q) with
    v = qdd_des + kp e + kd de. Closed-loop error obeys e'' + kd e' + kp e = 0."""
    v = qdd_des + kp * (q_des - q) + kd * (qd_des - qd)
    M = D.mass_matrix(chain, q)
    return (M @ v.unsqueeze(-1)).squeeze(-1) + D.coriolis(chain, q, qd) + D.gravity(chain, q)


def simulate(chain, q0, qd0, controller, dt: float, steps: int,
             record_every: int = 1):
    """Roll out the robot under `controller(t, q, qd) -> tau`.

    Semi-implicit Euler at fixed dt. Returns (ts, qs, qds) with qs/qds stacked
    along dim 0: (n_rec, B, dof).
    """
    q, qd = q0.clone(), qd0.clone()
    ts, qs, qds = [], [], []
    for k in range(steps):
        t = k * dt
        tau = controller(t, q, qd)
        qdd = D.forward_dynamics(chain, q, qd, tau)
        qd = qd + dt * qdd
        q = q + dt * qd
        if k % record_every == 0:
            ts.append(t); qs.append(q.clone()); qds.append(qd.clone())
    return (torch.tensor(ts, dtype=q0.dtype, device=q0.device),
            torch.stack(qs), torch.stack(qds))
