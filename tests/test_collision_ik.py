# tests/test_collision_ik.py
"""Collision-aware IK: a deterministic scenario where the plain IK solution is
provably in collision, and the collision-aware solver must reach the same target
with positive clearance by exploiting the 6-DOF arm's redundancy."""
import torch
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.collision import SphereModel, distance_to_obstacles, collision_aware_ik
from tests.test_spatial import SIX_DOF


def _setup():
    chain = compile_robot(parse_urdf_string(SIX_DOF), dtype=torch.float32)
    li = chain.link_index["ee"]
    # spheres along the arm body
    model = SphereModel(chain, {
        chain.link_index["l2"]: [(0.0, 0.0, 0.0, 0.05)],
        chain.link_index["l3"]: [(0.0, 0.0, 0.0, 0.05)],
        chain.link_index["l4"]: [(0.0, 0.0, 0.0, 0.05)],
    })
    # a fixed, deterministic reaching pose
    q_reach = torch.tensor([[0.5, 0.7, -0.4, 0.3, 0.5, 0.0]])
    world = forward_kinematics(chain, q_reach)
    target = world[:, li, :3, 3].clone()
    # obstacle planted exactly at the elbow (l3) of that pose -> guaranteed collision
    obs_c = world[:, chain.link_index["l3"], :3, 3].clone().squeeze(0).unsqueeze(0)
    obs_r = torch.tensor([0.10])
    return chain, li, model, q_reach, target, obs_c, obs_r


def test_baseline_pose_is_in_collision():
    chain, li, model, q_reach, target, obs_c, obs_r = _setup()
    d = distance_to_obstacles(model, q_reach, obs_c, obs_r)
    assert d.item() < 0                       # sphere on l3 sits inside the obstacle


def test_collision_aware_ik_escapes_and_reaches():
    """Even from the gradient-singular start (sphere center == obstacle center),
    the jittered two-stage solver reaches the target collision-free."""
    chain, li, model, q_reach, target, obs_c, obs_r = _setup()
    q_sol, info = collision_aware_ik(model, target, q_reach, li, obs_c, obs_r)
    assert info["pos_error"].item() < 2e-2    # still reaches the target
    assert info["clearance"].item() > 0.0     # ...but now collision-free


def test_collision_aware_ik_batched_best_of():
    """The landscape has local minima (like all IK); the supported usage is a
    batch of seeds with best-of selection — the best seed must fully succeed,
    and every seed must at least end collision-free."""
    chain, li, model, q_reach, target, obs_c, obs_r = _setup()
    B = 8
    torch.manual_seed(0)
    seeds = q_reach.repeat(B, 1) + 0.1 * torch.randn(B, 6)
    q_sol, info = collision_aware_ik(model, target.repeat(B, 1), seeds, li,
                                     obs_c, obs_r)
    ok = (info["pos_error"] < 2e-2) & (info["clearance"] > 0.0)
    assert ok.any()                            # best-of-batch succeeds
    assert (info["clearance"] > 0.0).all()     # every seed escapes the obstacle
    assert ok.float().mean() >= 0.4            # and a decent fraction fully solve
