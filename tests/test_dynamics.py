# tests/test_dynamics.py
"""Dynamics validated by INDEPENDENT oracles, not self-referential round-trips:
- gravity torque vs finite-difference of potential energy U(q)
- energy conservation under free-fall simulation (validates M, c, g together)
plus structural checks (mass matrix symmetric PD) and the inverse/forward round-trip.
"""
import torch
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast import dynamics as D

# Planar double pendulum in the x-z plane (revolute about y, gravity along -z).
# Each link: mass 1, COM at (0.5,0,0), diagonal inertia.
DYN_ARM = """
<robot name="dpend">
  <link name="base"/>
  <link name="l1">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.02" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l2">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.02" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.1" upper="3.1" velocity="5" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.1" upper="3.1" velocity="5" effort="50"/></joint>
</robot>
"""


def _chain():
    return compile_robot(parse_urdf_string(DYN_ARM), dtype=torch.float64)


def _potential(chain, q, g=D.GRAVITY):
    """U(q) = sum_i m_i g z_i(com), computed straight from FK. (B,)."""
    world = forward_kinematics(chain, q)
    U = torch.zeros(q.shape[0], dtype=q.dtype)
    for L in range(chain.n_links):
        m = float(chain.link_mass[L])
        if m == 0.0:
            continue
        com_h = torch.cat([chain.link_com[L], torch.ones(1, dtype=q.dtype)])
        z = (world[:, L] @ com_h)[:, 2]
        U = U + m * g * z
    return U


def test_mass_matrix_symmetric_pd():
    chain = _chain()
    torch.manual_seed(0)
    q = (chain.lower + (chain.upper - chain.lower) * torch.rand(6, 2))
    M = D.mass_matrix(chain, q)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-10)          # symmetric
    eig = torch.linalg.eigvalsh(M)
    assert (eig > 1e-6).all()                                          # positive-definite


def test_gravity_matches_potential_gradient():
    """Independent check: g(q) must equal the numerical gradient of U(q)."""
    chain = _chain()
    torch.manual_seed(1)
    q = (chain.lower + (chain.upper - chain.lower) * torch.rand(4, 2))
    g_analytic = D.gravity(chain, q)
    eps = 1e-6
    g_fd = torch.zeros_like(q)
    for k in range(2):
        dq = torch.zeros_like(q); dq[:, k] = eps
        g_fd[:, k] = (_potential(chain, q + dq) - _potential(chain, q - dq)) / (2 * eps)
    assert torch.allclose(g_analytic, g_fd, atol=1e-6)


def test_inverse_forward_roundtrip():
    chain = _chain()
    torch.manual_seed(2)
    q = chain.lower + (chain.upper - chain.lower) * torch.rand(8, 2)
    qd = torch.randn(8, 2, dtype=torch.float64)
    qdd = torch.randn(8, 2, dtype=torch.float64)
    tau = D.inverse_dynamics(chain, q, qd, qdd)
    qdd_back = D.forward_dynamics(chain, q, qd, tau)
    assert torch.allclose(qdd, qdd_back, atol=1e-8)


def test_coriolis_zero_at_rest():
    chain = _chain()
    q = torch.tensor([[0.3, -0.5]], dtype=torch.float64)
    c = D.coriolis(chain, q, torch.zeros(1, 2, dtype=torch.float64))
    assert torch.allclose(c, torch.zeros(1, 2, dtype=torch.float64), atol=1e-10)


def test_energy_conservation_freefall():
    """Simulate the pendulum swinging under gravity with no applied torque;
    total mechanical energy must stay ~constant. This jointly validates M, c, g."""
    chain = _chain()
    q = torch.tensor([[1.2, -0.6]], dtype=torch.float64)
    qd = torch.zeros(1, 2, dtype=torch.float64)

    def energy(q, qd):
        M = D.mass_matrix(chain, q)
        ke = 0.5 * (qd.unsqueeze(1) @ M @ qd.unsqueeze(-1)).reshape(-1)
        return (ke + _potential(chain, q))[0]

    E0 = energy(q, qd).item()
    dt, tau = 5e-4, torch.zeros(1, 2, dtype=torch.float64)
    for _ in range(400):  # 0.2 s
        qdd = D.forward_dynamics(chain, q, qd, tau)
        qd = qd + dt * qdd            # semi-implicit Euler
        q = q + dt * qd
    E1 = energy(q, qd).item()
    assert abs(E1 - E0) / abs(E0) < 0.02   # <2% drift over the horizon
