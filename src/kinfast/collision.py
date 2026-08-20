# src/kinfast/collision.py
"""Batched, differentiable sphere-based collision distance.

Each link carries a set of bounding spheres (center in the link frame + radius).
Spheres are cheap, GPU-batch-friendly, and differentiable — the same
approximation cuRobo uses. Signed distance between two spheres is
||c_i - c_j|| - r_i - r_j  (negative = penetrating), which flows gradients for
collision-avoidance IK / trajectory optimization.
"""
import torch
from kinfast.fk import forward_kinematics


class SphereModel:
    """Bounding spheres attached to links.

    spheres: dict {link_index: [(x, y, z, r), ...]} with centers in the link frame.
    """
    def __init__(self, chain, spheres):
        self.chain = chain
        dtype = chain.joint_origin.dtype
        centers, radii, link = [], [], []
        for L, sl in spheres.items():
            for (x, y, z, r) in sl:
                centers.append([x, y, z]); radii.append(r); link.append(int(L))
        self.local = torch.tensor(centers, dtype=dtype)      # (S,3)
        self.radius = torch.tensor(radii, dtype=dtype)       # (S,)
        self.link = torch.tensor(link, dtype=torch.long)     # (S,)
        self.n = len(link)

    def centers_world(self, q):
        """World-frame sphere centers for batch q. -> (B, S, 3)."""
        world = forward_kinematics(self.chain, q)            # (B, n_links, 4, 4)
        ones = torch.ones(self.n, 1, dtype=q.dtype, device=q.device)
        local_h = torch.cat([self.local.to(q.device), ones], dim=-1)  # (S,4)
        Tl = world[:, self.link.to(q.device)]                # (B, S, 4, 4)
        return torch.einsum("bsij,sj->bsi", Tl, local_h)[:, :, :3]

    def _allowed_pairs(self):
        """(S,S) bool mask: True where a sphere pair may collide.

        Excludes a sphere with itself, spheres on the same link, and spheres on
        adjacent (parent/child) links (which are always in contact by design).
        """
        L = self.link
        same = L[:, None] == L[None, :]
        n = self.chain.n_links
        adj = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            p = int(self.chain.parent[i])
            if p >= 0:
                adj[i, p] = True; adj[p, i] = True
        adj_s = adj[L][:, L]                                  # (S,S)
        return ~same & ~adj_s


def self_distance(model: SphereModel, q: torch.Tensor) -> torch.Tensor:
    """Minimum signed self-collision distance over allowed sphere pairs. -> (B,).

    Negative means the robot is in self-collision at that configuration.
    """
    C = model.centers_world(q)                               # (B,S,3)
    r = model.radius.to(q.device)
    dist = (C[:, :, None, :] - C[:, None, :, :]).norm(dim=-1)  # (B,S,S)
    sd = dist - r[None, :, None] - r[None, None, :]
    allowed = model._allowed_pairs().to(q.device)
    sd = torch.where(allowed[None], sd, torch.full_like(sd, float("inf")))
    return sd.reshape(q.shape[0], -1).min(dim=1).values


def distance_to_obstacles(model: SphereModel, q: torch.Tensor,
                          obs_centers: torch.Tensor,
                          obs_radii: torch.Tensor) -> torch.Tensor:
    """Minimum signed distance from any robot sphere to any obstacle sphere. -> (B,).

    obs_centers (M,3), obs_radii (M,). Negative means collision with an obstacle.
    """
    C = model.centers_world(q)                               # (B,S,3)
    r = model.radius.to(q.device)
    oc = obs_centers.to(q.device)                            # (M,3)
    orad = obs_radii.to(q.device)                            # (M,)
    dist = (C[:, :, None, :] - oc[None, None, :, :]).norm(dim=-1)  # (B,S,M)
    sd = dist - r[None, :, None] - orad[None, None, :]
    return sd.reshape(q.shape[0], -1).min(dim=1).values
