# tests/test_gpu.py
"""GPU correctness: every module must give the same answers on CUDA as on CPU.
These run only where a CUDA device exists (skipped otherwise); run them on the
GPU box with `pytest tests/test_gpu.py -v`.
"""
import os

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
pytestmark = cuda

import kinfast
from kinfast import dynamics as D
from kinfast import analysis as A
from kinfast.collision import SphereModel, self_distance, distance_to_obstacles
from kinfast.trajectory import trapezoidal
from tests.test_spatial import SIX_DOF

PANDA = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "panda.urdf")


def _pair(urdf=None):
    """Same robot twice: one on CPU, one on CUDA."""
    if urdf is None and os.path.exists(PANDA):
        cpu = kinfast.load(PANDA)
        gpu = kinfast.load(PANDA).to("cuda")
    else:
        cpu = kinfast.load_string(urdf or SIX_DOF)
        gpu = kinfast.load_string(urdf or SIX_DOF).to("cuda")
    return cpu, gpu


def _q(robot, n, seed=0):
    torch.manual_seed(seed)
    return robot.random_configs(n)


def test_robot_moves_to_cuda():
    _, gpu = _pair()
    assert gpu.lower.device.type == "cuda"
    q = gpu.random_configs(8)
    assert q.device.type == "cuda"
    assert gpu.fk(q).device.type == "cuda"


def test_fk_matches_cpu():
    cpu, gpu = _pair()
    q = _q(cpu, 256)
    ref = cpu.fk_all(q)
    got = gpu.fk_all(q.cuda()).cpu()
    assert torch.allclose(got, ref, atol=1e-5)


def test_jacobian_matches_cpu():
    cpu, gpu = _pair()
    q = _q(cpu, 64)
    ref = cpu.jacobian(q)
    got = gpu.jacobian(q.cuda()).cpu()
    assert torch.allclose(got, ref, atol=1e-5)


def test_ik_solves_on_cuda():
    cpu, gpu = _pair()
    target = gpu.fk(gpu.random_configs(2048))
    q_sol, info = gpu.ik(target, iters=100, pos_only=True, restarts=4)
    assert q_sol.device.type == "cuda"
    err = (gpu.fk(q_sol)[:, :3, 3] - target[:, :3, 3]).norm(dim=-1)
    assert (err < 5e-2).float().mean().item() > 0.95


def test_ik_differentiable_on_cuda():
    _, gpu = _pair(SIX_DOF)
    target = gpu.fk(torch.zeros(3, 6, device="cuda")).clone()
    q0 = torch.full((3, 6), 0.05, device="cuda", requires_grad=True)
    q_sol, _ = gpu.ik(target, q0=q0, iters=5, pos_only=True)
    q_sol.sum().backward()
    assert q0.grad is not None and torch.isfinite(q0.grad).all()


def test_dynamics_match_cpu():
    from tests.test_dynamics import DYN_ARM
    cpu, gpu = _pair(DYN_ARM)
    q = _q(cpu, 16)
    qd = torch.randn(16, cpu.dof)
    qdd = torch.randn(16, cpu.dof)
    assert torch.allclose(gpu.mass_matrix(q.cuda()).cpu(), cpu.mass_matrix(q), atol=1e-5)
    assert torch.allclose(gpu.inverse_dynamics(q.cuda(), qd.cuda(), qdd.cuda()).cpu(),
                          cpu.inverse_dynamics(q, qd, qdd), atol=1e-4)


def test_collision_matches_cpu():
    cpu, gpu = _pair(SIX_DOF)
    spheres = {"l2": [(0, 0, 0, 0.05)], "l3": [(0, 0, 0, 0.05)], "ee": [(0, 0, 0, 0.04)]}
    mc, mg = cpu.sphere_model(spheres), gpu.sphere_model(spheres)
    q = _q(cpu, 32)
    assert torch.allclose(self_distance(mg, q.cuda()).cpu(), self_distance(mc, q), atol=1e-5)
    obs_c = torch.tensor([[0.3, 0.0, 0.6]])
    obs_r = torch.tensor([0.1])
    assert torch.allclose(distance_to_obstacles(mg, q.cuda(), obs_c.cuda(), obs_r.cuda()).cpu(),
                          distance_to_obstacles(mc, q, obs_c, obs_r), atol=1e-5)


def test_trajectory_on_cuda():
    q0 = torch.zeros(3, device="cuda")
    qf = torch.tensor([1.0, -0.5, 0.3], device="cuda")
    vmax = torch.ones(3, device="cuda")
    amax = torch.full((3,), 2.0, device="cuda")
    t, q, qd, qdd, T = trapezoidal(q0, qf, vmax, amax, n=200)
    assert q.device.type == "cuda"
    assert torch.allclose(q[-1], qf, atol=1e-6)


def test_workspace_on_cuda_robot():
    _, gpu = _pair()
    ws = A.workspace(gpu.chain, gpu.link_id(gpu.ee_link), n=5000)
    assert ws["points"].device.type == "cuda"
    assert ws["max_reach"].item() > 0.5


def test_large_batch_fk_does_not_oom():
    _, gpu = _pair()
    q = gpu.random_configs(200_000)
    out = gpu.fk(q)
    torch.cuda.synchronize()
    assert out.shape == (200_000, 4, 4)
