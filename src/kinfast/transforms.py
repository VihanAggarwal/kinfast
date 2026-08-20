# src/kinfast/transforms.py
"""Pure SE(3)/SO(3) tensor math. No robot concepts live here.

Conventions:
- Transforms are (..., 4, 4) homogeneous matrices.
- URDF rpy is extrinsic X-Y-Z (fixed axis): R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""
import torch


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """(..., 3) roll-pitch-yaw -> (..., 3, 3)."""
    r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    row0 = torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=-1)
    row1 = torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=-1)
    row2 = torch.stack([-sp, cp * sr, cp * cr], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues. axis (..., 3) unit, angle (...) -> (..., 3, 3)."""
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1.0 - c
    row0 = torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], dim=-1)
    row1 = torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], dim=-1)
    row2 = torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """R (..., 3, 3), t (..., 3) -> (..., 4, 4)."""
    shape = R.shape[:-2]
    M = torch.zeros(*shape, 4, 4, dtype=R.dtype, device=R.device)
    M[..., :3, :3] = R
    M[..., :3, 3] = t
    M[..., 3, 3] = 1.0
    return M


def invert_transform(M: torch.Tensor) -> torch.Tensor:
    """Inverse of a homogeneous transform (..., 4, 4)."""
    R = M[..., :3, :3]
    t = M[..., :3, 3]
    Rt = R.transpose(-1, -2)
    inv = torch.zeros_like(M)
    inv[..., :3, :3] = Rt
    inv[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    inv[..., 3, 3] = 1.0
    return inv


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation -> (..., 3) axis-angle vector."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    w = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    small = theta < 1e-5
    # For small angles sin(theta)~theta and 1/(2 sin) ~ 0.5; blend safely.
    denom = torch.where(small, torch.ones_like(theta), 2.0 * torch.sin(theta))
    scale = torch.where(small, 0.5 * torch.ones_like(theta), theta / denom)
    return w * scale.unsqueeze(-1)


def pose_error(T_cur: torch.Tensor, T_tgt: torch.Tensor) -> torch.Tensor:
    """World-frame 6D error [dp(3); dw(3)] moving current toward target. (..., 6)."""
    p_err = T_tgt[..., :3, 3] - T_cur[..., :3, 3]
    R_err = T_tgt[..., :3, :3] @ T_cur[..., :3, :3].transpose(-1, -2)
    w_err = so3_log(R_err)
    return torch.cat([p_err, w_err], dim=-1)
