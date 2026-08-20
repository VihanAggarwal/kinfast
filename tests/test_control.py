# tests/test_control.py
"""Controllers validated in closed loop: the controller must actually hold or
track when simulated with the same dynamics — trajectory + dynamics + control
+ simulator all composing is the real integration test."""
import torch
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.trajectory import quintic
from kinfast import control as C
from tests.test_dynamics import DYN_ARM


def _chain():
    return compile_robot(parse_urdf_string(DYN_ARM), dtype=torch.float64)


def test_gravity_compensation_holds_still():
    """Under exact gravity compensation, a robot at rest stays at rest."""
    chain = _chain()
    q0 = torch.tensor([[0.8, -0.5]], dtype=torch.float64)
    qd0 = torch.zeros(1, 2, dtype=torch.float64)
    ctrl = lambda t, q, qd: C.gravity_compensation(chain, q)
    ts, qs, qds = C.simulate(chain, q0, qd0, ctrl, dt=1e-3, steps=500)
    assert (qs[-1] - q0).abs().max() < 1e-6
    assert qds[-1].abs().max() < 1e-6


def test_pd_gravity_regulates_setpoint():
    """PD + gravity feedforward drives the arm to a setpoint and stays there."""
    chain = _chain()
    q0 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    q_des = torch.tensor([[1.0, -0.7]], dtype=torch.float64)
    qd0 = torch.zeros(1, 2, dtype=torch.float64)
    ctrl = lambda t, q, qd: C.pd_gravity(chain, q, qd, q_des, kp=100.0, kd=20.0)
    ts, qs, qds = C.simulate(chain, q0, qd0, ctrl, dt=1e-3, steps=3000)
    assert (qs[-1] - q_des).abs().max() < 5e-3
    assert qds[-1].abs().max() < 1e-2


def test_computed_torque_tracks_trajectory():
    """Feedback linearization tracks a quintic swing tightly the whole way."""
    chain = _chain()
    T_end, steps, dt = 1.0, 1000, 1e-3
    q0 = torch.tensor([0.2, -0.3], dtype=torch.float64)
    qf = torch.tensor([1.4, 0.8], dtype=torch.float64)
    tt, qdes, qddes, qdddes = quintic(q0, qf, T=T_end, n=steps)

    def ctrl(t, q, qd):
        k = min(int(round(t / dt)), steps - 1)
        return C.computed_torque(chain, q, qd,
                                 qdes[k:k+1], qddes[k:k+1], qdddes[k:k+1],
                                 kp=400.0, kd=40.0)

    ts, qs, qds = C.simulate(chain, q0.unsqueeze(0), torch.zeros(1, 2, dtype=torch.float64),
                             ctrl, dt=dt, steps=steps)
    track_err = (qs.squeeze(1) - qdes).abs().max()
    assert track_err < 5e-3            # tight tracking over the whole motion
    assert (qs[-1, 0] - qf).abs().max() < 1e-3


def test_simulate_batched():
    """B independent robots roll out simultaneously and match single rollouts."""
    chain = _chain()
    q0 = torch.tensor([[0.3, -0.2], [0.9, 0.4]], dtype=torch.float64)
    qd0 = torch.zeros(2, 2, dtype=torch.float64)
    ctrl = lambda t, q, qd: C.gravity_compensation(chain, q)
    ts, qs, _ = C.simulate(chain, q0, qd0, ctrl, dt=1e-3, steps=100)
    for b in range(2):
        _, qs1, _ = C.simulate(chain, q0[b:b+1], qd0[b:b+1], ctrl, dt=1e-3, steps=100)
        assert torch.allclose(qs[:, b], qs1[:, 0], atol=1e-12)
