# tests/test_regressions_dynamics.py
"""Regression tests for the dynamics layer.

Covers three bugs that used to ship silently:
  1. coriolis() cut the autograd graph (d c/d q was zero, d c/d qd wrong), so
     gradients through inverse/forward dynamics and control.simulate were biased.
  2. dynamics and jacobian crashed with a dtype error when q's dtype differed
     from the chain's compile dtype (fk already cast; Robot compiles float32).
  3. coriolis() crashed under torch.no_grad(), and a model with no inertials
     failed deep inside autograd instead of with a clear message.

Oracles are independent of the library: hand-derived 2R Christoffel terms and
float64 central finite differences.
"""
import pytest
import torch
import kinfast
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast import dynamics as D
from kinfast import control as C
from kinfast.jacobian import jacobian
from tests.test_dynamics import DYN_ARM
from tests.test_parse import TWO_LINK

Q = torch.tensor([[0.7, -0.4]], dtype=torch.float64)
QD = torch.tensor([[1.5, -2.0]], dtype=torch.float64)
TAU = torch.tensor([[0.3, -0.8]], dtype=torch.float64)


def _chain(dtype=torch.float64):
    return compile_robot(parse_urdf_string(DYN_ARM), dtype=dtype)


def _fd_jac(f, x, eps=1e-6):
    """Central-difference Jacobian of f(x).reshape(-1) w.r.t. x.reshape(-1)."""
    y0 = f(x).reshape(-1)
    J = torch.zeros(y0.numel(), x.numel(), dtype=torch.float64)
    for j in range(x.numel()):
        d = torch.zeros_like(x).reshape(-1)
        d[j] = eps
        d = d.reshape(x.shape)
        J[:, j] = ((f(x + d) - f(x - d)) / (2 * eps)).reshape(-1)
    return J


# ---------------------------------------------------------------- values ----

def test_coriolis_matches_hand_derived_2r():
    """Planar 2R with m1=m2=1, l1=1, lc=0.5: c1 = -h (2 qd1 qd2 + qd2^2),
    c2 = h qd1^2 with h = m2 l1 lc2 sin q2 (textbook Christoffel result)."""
    chain = _chain()
    c = D.coriolis(chain, Q, QD)[0]
    h = 1.0 * 1.0 * 0.5 * torch.sin(Q[0, 1])
    qd1, qd2 = QD[0]
    expected = torch.stack([-h * (2 * qd1 * qd2 + qd2 ** 2), h * qd1 ** 2])
    assert torch.allclose(c, expected, atol=1e-12)


# -------------------------------------------------------------- gradients ----

def test_coriolis_gradients_match_finite_differences():
    chain = _chain()
    for name, f, x in [("q", lambda v: D.coriolis(chain, v, QD), Q),
                       ("qd", lambda v: D.coriolis(chain, Q, v), QD)]:
        J_ad = torch.autograd.functional.jacobian(f, x).reshape(2, 2)
        J_fd = _fd_jac(f, x)
        assert torch.allclose(J_ad, J_fd, atol=1e-7), (name, J_ad, J_fd)
    # the q-gradient is genuinely nonzero at nonzero qd (it used to be all zero)
    J_q = torch.autograd.functional.jacobian(lambda v: D.coriolis(chain, v, QD), Q)
    assert J_q.abs().max() > 0.5


def test_forward_and_inverse_dynamics_gradients_match_finite_differences():
    chain = _chain()
    cases = [
        ("fd/q", lambda v: D.forward_dynamics(chain, v, QD, TAU), Q),
        ("fd/qd", lambda v: D.forward_dynamics(chain, Q, v, TAU), QD),
        ("id/q", lambda v: D.inverse_dynamics(chain, v, QD, TAU), Q),
        ("id/qd", lambda v: D.inverse_dynamics(chain, Q, v, TAU), QD),
    ]
    for name, f, x in cases:
        J_ad = torch.autograd.functional.jacobian(f, x).reshape(2, 2)
        J_fd = _fd_jac(f, x)
        assert torch.allclose(J_ad, J_fd, atol=1e-6), (name, J_ad, J_fd)


def test_coriolis_grad_flows_to_both_inputs():
    chain = _chain()
    q = Q.clone().requires_grad_(True)
    qd = QD.clone().requires_grad_(True)
    L = (D.coriolis(chain, q, qd) ** 2).sum()
    gq, gqd = torch.autograd.grad(L, [q, qd])
    assert gq is not None and gq.abs().max() > 0
    assert gqd is not None and gqd.abs().max() > 0


def test_learnable_gain_through_simulate_matches_finite_differences():
    """dL/dkp for a 20-step pd_gravity rollout, autograd vs central FD."""
    chain = _chain()
    q_des = torch.zeros(1, 2, dtype=torch.float64)

    def loss(kp):
        ctrl = lambda t, q, qd: C.pd_gravity(chain, q, qd, q_des, kp, 0.5)
        _, qs, _ = C.simulate(chain, Q, QD, ctrl, dt=5e-3, steps=20)
        return (qs[-1] ** 2).sum()

    kp = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    g_ad = torch.autograd.grad(loss(kp), kp)[0].item()
    eps = 1e-5
    g_fd = ((loss(torch.tensor(3.0 + eps, dtype=torch.float64))
             - loss(torch.tensor(3.0 - eps, dtype=torch.float64))) / (2 * eps)).item()
    assert abs(g_ad - g_fd) < 1e-7 * max(1.0, abs(g_fd)), (g_ad, g_fd)


def test_coriolis_output_detached_when_inputs_do_not_require_grad():
    chain = _chain()
    c = D.coriolis(chain, Q, QD)
    assert not c.requires_grad


# ------------------------------------------------------------------ dtype ----

def test_float32_chain_accepts_float64_q():
    """The Robot default chain is float32; every dynamics entry point must
    follow q's dtype the way fk does, and agree with a float64 compile."""
    c32, c64 = _chain(torch.float32), _chain(torch.float64)
    ref = {
        "M": D.mass_matrix(c64, Q), "g": D.gravity(c64, Q),
        "c": D.coriolis(c64, Q, QD),
        "tau": D.inverse_dynamics(c64, Q, QD, TAU),
        "qdd": D.forward_dynamics(c64, Q, QD, TAU),
        "J": jacobian(c64, Q, c64.n_links - 1),
    }
    got = {
        "M": D.mass_matrix(c32, Q), "g": D.gravity(c32, Q),
        "c": D.coriolis(c32, Q, QD),
        "tau": D.inverse_dynamics(c32, Q, QD, TAU),
        "qdd": D.forward_dynamics(c32, Q, QD, TAU),
        "J": jacobian(c32, Q, c32.n_links - 1),
    }
    for k in ref:
        assert got[k].dtype == torch.float64, k
        # float32 constants (0.5, 1.0, 0.1, 0.02) are exact, so this is tight
        assert torch.allclose(got[k], ref[k], atol=1e-6), k


def test_float64_chain_accepts_float32_q():
    c64 = _chain(torch.float64)
    q32, qd32 = Q.float(), QD.float()
    assert D.mass_matrix(c64, q32).dtype == torch.float32
    assert D.coriolis(c64, q32, qd32).dtype == torch.float32
    assert D.forward_dynamics(c64, q32, qd32, TAU.float()).dtype == torch.float32
    assert jacobian(c64, q32, c64.n_links - 1).dtype == torch.float32


def test_robot_api_default_chain_with_float64_q():
    robot = kinfast.load_string(DYN_ARM)
    assert robot.chain.joint_origin.dtype == torch.float32
    assert robot.fk(Q).dtype == torch.float64
    assert robot.mass_matrix(Q).dtype == torch.float64
    assert robot.gravity(Q).dtype == torch.float64
    assert robot.jacobian(Q).dtype == torch.float64
    assert robot.inverse_dynamics(Q, QD, TAU).dtype == torch.float64
    assert robot.forward_dynamics(Q, QD, TAU).dtype == torch.float64


def test_controllers_accept_mismatched_gain_dtype():
    c32 = _chain(torch.float32)
    q, qd = Q.float(), QD.float()
    kp = torch.tensor([100.0, 80.0], dtype=torch.float64)   # (dof,) float64 gains
    kd = torch.tensor([20.0, 10.0], dtype=torch.float64)
    tau = C.computed_torque(c32, q, qd, torch.zeros_like(q), torch.zeros_like(q),
                            torch.zeros_like(q), kp, kd)
    assert tau.dtype == torch.float32 and tau.shape == (1, 2)
    tau = C.pd_gravity(c32, q, qd, torch.zeros_like(q), kp, kd)
    assert tau.dtype == torch.float32 and tau.shape == (1, 2)


# ---------------------------------------------------------------- no_grad ----

def test_dynamics_work_under_no_grad():
    chain = _chain()
    ref_c = D.coriolis(chain, Q, QD)
    ref_qdd = D.forward_dynamics(chain, Q, QD, TAU)
    with torch.no_grad():
        c = D.coriolis(chain, Q, QD)
        qdd = D.forward_dynamics(chain, Q, QD, TAU)
        tau = C.computed_torque(chain, Q, QD, Q, QD, TAU, 10.0, 1.0)
        ctrl = lambda t, q, qd: C.gravity_compensation(chain, q)
        _, qs, _ = C.simulate(chain, Q, QD, ctrl, dt=1e-3, steps=5)
    assert not c.requires_grad and not qdd.requires_grad and not tau.requires_grad
    assert not qs.requires_grad
    assert torch.allclose(c, ref_c, atol=1e-12)
    assert torch.allclose(qdd, ref_qdd, atol=1e-12)


def test_coriolis_inference_mode_raises_clearly():
    chain = _chain()
    with torch.inference_mode():
        with pytest.raises(ValueError, match="inference_mode"):
            D.coriolis(chain, Q, QD)


def test_massless_model_raises_clearly():
    chain = compile_robot(parse_urdf_string(TWO_LINK), dtype=torch.float64)
    with pytest.raises(ValueError, match="mass or inertia"):
        D.mass_matrix(chain, Q)
    with pytest.raises(ValueError, match="mass or inertia"):
        D.forward_dynamics(chain, Q, QD, TAU)
    # gravity on a massless model is simply zero, not an error
    assert torch.allclose(D.gravity(chain, Q), torch.zeros(1, 2, dtype=torch.float64))
