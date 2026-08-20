# src/kinfast/fk.py
"""Batched forward kinematics: propagate homogeneous transforms down the tree."""
import torch
from kinfast import transforms as T
from kinfast.compile import CompiledChain


def _joint_motion(chain: CompiledChain, i: int, q: torch.Tensor) -> torch.Tensor:
    """(B, 4, 4) local motion of joint i given batch config q (B, dof)."""
    B = q.shape[0]
    device, dtype = q.device, q.dtype
    jt = int(chain.joint_type[i])
    if jt == 0:
        return torch.eye(4, dtype=dtype, device=device).expand(B, 4, 4)
    qi = q[:, int(chain.q_index[i])]
    axis = chain.joint_axis[i].to(device)
    if jt == 1:  # revolute
        R = T.axis_angle_to_matrix(axis.expand(B, 3), qi)
        t = torch.zeros(B, 3, dtype=dtype, device=device)
        return T.make_transform(R, t)
    # prismatic
    R = torch.eye(3, dtype=dtype, device=device).expand(B, 3, 3)
    t = axis.expand(B, 3) * qi.unsqueeze(-1)
    return T.make_transform(R, t)


def forward_kinematics(chain: CompiledChain, q: torch.Tensor) -> torch.Tensor:
    """q (B, dof) -> world transforms (B, n_links, 4, 4)."""
    B = q.shape[0]
    device, dtype = q.device, q.dtype
    origin = chain.joint_origin.to(device)
    world = [None] * chain.n_links
    for i in chain.topo_order:
        local = origin[i].expand(B, 4, 4) @ _joint_motion(chain, i, q)
        p = int(chain.parent[i])
        world[i] = local if p < 0 else world[p] @ local
    return torch.stack(world, dim=1)
