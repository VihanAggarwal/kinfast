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
    # qs[i] is the state at ts[i], so compare it with the reference sampled at ts[i]
    idx = (ts / dt).round().long().clamp(max=steps - 1)
    track_err = (qs.squeeze(1) - qdes[idx]).abs().max()
    assert track_err < 1e-3            # tight tracking over the whole motion
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


# Regression fixture: a single vertical prismatic joint carrying a point mass.
# Under zero torque semi-implicit Euler gives, after n steps of size dt,
#   qd_n = -g n dt            q_n = -g dt^2 n (n+1) / 2
# exactly (up to rounding), which is an oracle independent of the library.
SLIDER = """
<robot name="slider">
  <link name="base"/>
  <link name="mass">
    <inertial><origin xyz="0 0 0"/><mass value="2.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="z" type="prismatic"><parent link="base"/><child link="mass"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-10" upper="10" velocity="5" effort="50"/></joint>
</robot>
"""


def _slider():
    return compile_robot(parse_urdf_string(SLIDER), dtype=torch.float64)


def _zero_torque(t, q, qd):
    return torch.zeros_like(q)


def test_simulate_time_labels_match_recorded_state():
    """ts[i] is the time of the state stored in qs[i]/qds[i]: a free-falling
    slider has qd(t) = -g t exactly, and q0 is not part of the output."""
    from kinfast.dynamics import GRAVITY as g
    chain = _slider()
    q0 = torch.zeros(1, 1, dtype=torch.float64)
    qd0 = torch.zeros(1, 1, dtype=torch.float64)
    dt, steps = 0.01, 7
    ts, qs, qds = C.simulate(chain, q0, qd0, _zero_torque, dt=dt, steps=steps)
    assert ts.shape == (steps,) and qs.shape == (steps, 1, 1)
    n = torch.arange(1, steps + 1, dtype=torch.float64)
    assert torch.allclose(ts, n * dt, atol=1e-15)
    assert ts[0].item() == dt                     # first record is after the first step
    assert torch.allclose(qds[:, 0, 0], -g * ts, atol=1e-12)
    assert torch.allclose(qs[:, 0, 0], -g * dt**2 * n * (n + 1) / 2, atol=1e-12)


def test_simulate_controller_sees_time_of_its_state():
    """The controller is called with t = time of the state it is given, so the
    first call sees t = 0 and q0, and later calls see the recorded states."""
    chain = _slider()
    q0 = torch.tensor([[0.25]], dtype=torch.float64)
    qd0 = torch.zeros(1, 1, dtype=torch.float64)
    seen = []

    def ctrl(t, q, qd):
        seen.append((t, q.clone()))
        return torch.zeros_like(q)

    dt = 0.01
    ts, qs, _ = C.simulate(chain, q0, qd0, ctrl, dt=dt, steps=4)
    assert seen[0][0] == 0.0 and torch.equal(seen[0][1], q0)
    for i in range(1, 4):
        assert abs(seen[i][0] - ts[i - 1].item()) < 1e-15   # call i sees the state recorded at ts[i-1]
        assert torch.equal(seen[i][1], qs[i - 1])


def test_simulate_record_every_subsamples_with_true_times():
    """record_every keeps every r-th step and labels it with its true time."""
    chain = _slider()
    q0 = torch.zeros(1, 1, dtype=torch.float64)
    qd0 = torch.zeros(1, 1, dtype=torch.float64)
    dt = 0.01
    ts_all, qs_all, qds_all = C.simulate(chain, q0, qd0, _zero_torque, dt=dt, steps=10)
    ts, qs, qds = C.simulate(chain, q0, qd0, _zero_torque, dt=dt, steps=10, record_every=3)
    assert torch.equal(ts, ts_all[::3]) and torch.equal(qs, qs_all[::3]) and torch.equal(qds, qds_all[::3])
    # record_every larger than steps keeps only the first step
    ts1, qs1, _ = C.simulate(chain, q0, qd0, _zero_torque, dt=dt, steps=3, record_every=50)
    assert ts1.shape == (1,) and torch.equal(qs1[0], qs_all[0])


def test_simulate_rejects_bad_steps_and_record_every():
    """steps < 1 and record_every < 1 fail up front with a clear ValueError."""
    import pytest
    chain = _slider()
    q0 = torch.zeros(1, 1, dtype=torch.float64)
    qd0 = torch.zeros(1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="steps"):
        C.simulate(chain, q0, qd0, _zero_torque, dt=1e-3, steps=0)
    with pytest.raises(ValueError, match="steps"):
        C.simulate(chain, q0, qd0, _zero_torque, dt=1e-3, steps=-5)
    with pytest.raises(ValueError, match="record_every"):
        C.simulate(chain, q0, qd0, _zero_torque, dt=1e-3, steps=10, record_every=0)
    with pytest.raises(ValueError, match="record_every"):
        C.simulate(chain, q0, qd0, _zero_torque, dt=1e-3, steps=10, record_every=-1)


def test_simulate_validates_controller_output_shape():
    """Scalar, (dof,), (1,dof) and (B,dof) torques are accepted and give the same
    rollout; every other shape (and None) fails at the call site with a message
    that names the controller, before anything reaches the dynamics."""
    import pytest
    chain = _chain()
    B = 2
    q0 = torch.tensor([[0.3, -0.2], [0.1, 0.2]], dtype=torch.float64)
    qd0 = torch.zeros(B, 2, dtype=torch.float64)
    full = torch.tensor([[0.5, -0.25], [0.5, -0.25]], dtype=torch.float64)
    ref_ts, ref_qs, ref_qds = C.simulate(chain, q0, qd0, lambda t, q, qd: full, dt=1e-3, steps=3)
    ok = {
        "(dof,)": lambda t, q, qd: full[0],
        "(1,dof)": lambda t, q, qd: full[:1],
        "(B,dof) float32": lambda t, q, qd: full.to(torch.float32),
    }
    for name, ctrl in ok.items():
        ts, qs, qds = C.simulate(chain, q0, qd0, ctrl, dt=1e-3, steps=3)
        assert qs.shape == (3, B, 2) and qs.dtype == torch.float64, name
        assert torch.allclose(qs, ref_qs, atol=1e-7) and torch.allclose(qds, ref_qds, atol=1e-7), name
    # a python float applies the same torque to every joint of every robot
    _, qs_f, _ = C.simulate(chain, q0, qd0, lambda t, q, qd: 0.0, dt=1e-3, steps=3)
    _, qs_z, _ = C.simulate(chain, q0, qd0, _zero_torque, dt=1e-3, steps=3)
    assert torch.equal(qs_f, qs_z)

    bad = {
        "(B,1)": lambda t, q, qd: torch.zeros(B, 1, dtype=torch.float64),
        "(B,dof,1)": lambda t, q, qd: torch.zeros(B, 2, 1, dtype=torch.float64),
        "(B,dof+1)": lambda t, q, qd: torch.zeros(B, 3, dtype=torch.float64),
        "(B+1,dof)": lambda t, q, qd: torch.zeros(B + 1, 2, dtype=torch.float64),
        "(dof+1,)": lambda t, q, qd: torch.zeros(3, dtype=torch.float64),
        "None": lambda t, q, qd: None,
    }
    for name, ctrl in bad.items():
        with pytest.raises(ValueError, match="controller returned"):
            C.simulate(chain, q0, qd0, ctrl, dt=1e-3, steps=2)

    # a per-robot (B,) vector is only ambiguous when B == dof; with B != dof it is
    # rejected instead of broadcasting into the dynamics
    q3 = torch.tensor([[0.3, -0.2], [0.1, 0.2], [0.0, 0.5]], dtype=torch.float64)
    with pytest.raises(ValueError, match="controller returned"):
        C.simulate(chain, q3, torch.zeros(3, 2, dtype=torch.float64),
                   lambda t, q, qd: torch.zeros(3, dtype=torch.float64), dt=1e-3, steps=2)
