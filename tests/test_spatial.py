# tests/test_spatial.py
"""3D / spatial coverage: a real 6-DOF arm, full 6-row Jacobian, full-pose IK.

The existing suite only exercises a planar 2R arm, which cannot catch bugs in
3D rotation composition or the orientation rows of the Jacobian (Jw). These
tests use double precision so numerical oracles (central differences) are tight.
"""
import math
import torch
import kinfast
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.jacobian import jacobian
from kinfast.ik import ik
from kinfast import transforms as T

# 6-DOF spatial arm, alternating axes, stacked along +z.
# At q=0: ee at (0,0, 0.3+0.3+0.3+0.1+0.1) = (0,0,1.1), orientation = I.
SIX_DOF = """
<robot name="arm6">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/>
  <link name="l4"/><link name="l5"/><link name="ee"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="2" effort="50"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="2" effort="50"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/></joint>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="2" effort="50"/></joint>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="ee"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/></joint>
</robot>
"""


def _chain(dtype=torch.float64):
    return compile_robot(parse_urdf_string(SIX_DOF), dtype=dtype)


def test_fk_zero_config_spatial():
    chain = _chain()
    q = torch.zeros(1, 6, dtype=torch.float64)
    ee = forward_kinematics(chain, q)[:, chain.link_index["ee"]]
    assert torch.allclose(ee[0, :3, 3], torch.tensor([0.0, 0.0, 1.1], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(ee[0, :3, :3], torch.eye(3, dtype=torch.float64), atol=1e-9)


def test_full_jacobian_matches_finite_difference():
    """All 6 rows (Jv AND Jw) vs float64 central differences at random configs."""
    chain = _chain()
    li = chain.link_index["ee"]
    torch.manual_seed(3)
    q = (chain.lower + (chain.upper - chain.lower) * torch.rand(4, 6, dtype=torch.float64))
    J = jacobian(chain, q, li)  # (4,6,6)
    eps = 1e-6
    for k in range(6):
        dq = torch.zeros_like(q); dq[:, k] = eps
        Tp = forward_kinematics(chain, q + dq)[:, li]
        Tm = forward_kinematics(chain, q - dq)[:, li]
        # linear velocity: central diff of position
        dv = (Tp[:, :3, 3] - Tm[:, :3, 3]) / (2 * eps)
        # angular velocity (world frame): R(q+)=exp([w])R(q-)  =>  w = log(Rp Rm^T)
        R_rel = Tp[:, :3, :3] @ Tm[:, :3, :3].transpose(-1, -2)
        dw = T.so3_log(R_rel) / (2 * eps)
        assert torch.allclose(J[:, :3, k], dv, atol=1e-5), f"Jv col {k}"
        assert torch.allclose(J[:, 3:, k], dw, atol=1e-5), f"Jw col {k}"


def test_ik_full_pose_roundtrip_6dof_multiseed():
    """6-DOF arm hits full position+orientation targets with multi-seed restarts."""
    chain = _chain(dtype=torch.float32)
    li = chain.link_index["ee"]
    torch.manual_seed(0)
    q_true = chain.lower + (chain.upper - chain.lower) * torch.rand(128, 6)
    target = forward_kinematics(chain, q_true)[:, li]
    q_sol, info = ik(chain, target, link_index=li, iters=300, damping=0.02,
                     pos_only=False, restarts=16)
    err = T.pose_error(forward_kinematics(chain, q_sol)[:, li], target)
    ok = (err[:, :3].norm(dim=-1) < 1e-2) & (err[:, 3:].norm(dim=-1) < 1e-2)
    assert ok.float().mean() > 0.9


def test_multiseed_beats_single_seed():
    """Restarts strictly improve solve rate on a hard spatial arm."""
    chain = _chain(dtype=torch.float32)
    li = chain.link_index["ee"]
    torch.manual_seed(0)
    q_true = chain.lower + (chain.upper - chain.lower) * torch.rand(96, 6)
    target = forward_kinematics(chain, q_true)[:, li]

    def solved(restarts):
        q, _ = ik(chain, target, link_index=li, iters=250, damping=0.02,
                  pos_only=False, restarts=restarts)
        e = T.pose_error(forward_kinematics(chain, q)[:, li], target)
        return ((e[:, :3].norm(dim=-1) < 1e-2) & (e[:, 3:].norm(dim=-1) < 1e-2)).float().mean()

    single = solved(1)
    multi = solved(16)
    assert multi > single + 0.2


def test_ik_is_differentiable():
    """Gradient flows through the whole IK solve (learnable pipeline)."""
    chain = _chain(dtype=torch.float32)
    li = chain.link_index["ee"]
    target = forward_kinematics(chain, torch.zeros(3, 6))[:, li].clone()
    q0 = torch.full((3, 6), 0.05, requires_grad=True)
    q_sol, _ = ik(chain, target, q0, li, iters=5, pos_only=True)
    q_sol.sum().backward()
    assert q0.grad is not None and torch.isfinite(q0.grad).all()


def test_batch_matches_single():
    """FK over a batch equals FK of each config computed alone."""
    chain = _chain()
    li = chain.link_index["ee"]
    torch.manual_seed(1)
    q = torch.rand(5, 6, dtype=torch.float64) - 0.5
    batched = forward_kinematics(chain, q)[:, li]
    for i in range(5):
        single = forward_kinematics(chain, q[i:i+1])[:, li]
        assert torch.allclose(batched[i], single[0], atol=1e-10)
