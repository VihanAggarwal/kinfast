# src/kinfast/jacobian.py
"""Batched geometric Jacobian read from the FK frames.

For each movable joint on the path from the target link to the root:
  revolute:  Jv = axis_world x (p_ee - p_joint),  Jw = axis_world
  prismatic: Jv = axis_world,                     Jw = 0
Rotating about an axis leaves that axis invariant, so the axis expressed in the
joint's world frame is world[i][:3,:3] @ axis (motion-independent).
"""
import torch
from kinfast.fk import forward_kinematics
from kinfast.compile import CompiledChain


def jacobian(chain: CompiledChain, q: torch.Tensor, link_index: int) -> torch.Tensor:
    B = q.shape[0]
    device, dtype = q.device, q.dtype
    world = forward_kinematics(chain, q)              # (B, n, 4, 4)
    p_ee = world[:, link_index, :3, 3]                # (B, 3)
    J = torch.zeros(B, 6, chain.dof, dtype=dtype, device=device)
    axes = chain.joint_axis.to(device)

    i = link_index
    while i >= 0:
        jt = int(chain.joint_type[i])
        if jt != 0:
            col = int(chain.q_index[i])
            Ti = world[:, i]                          # (B, 4, 4)
            axis_world = (Ti[:, :3, :3] @ axes[i]).to(dtype)   # (B, 3)
            p_j = Ti[:, :3, 3]                        # (B, 3)
            if jt == 1:  # revolute
                J[:, :3, col] = torch.cross(axis_world, p_ee - p_j, dim=-1)
                J[:, 3:, col] = axis_world
            else:        # prismatic
                J[:, :3, col] = axis_world
        i = int(chain.parent[i])
    return J
