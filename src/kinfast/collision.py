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


def collision_aware_ik(model: SphereModel, target_pos: torch.Tensor,
                       q0: torch.Tensor, link_index: int,
                       obs_centers: torch.Tensor, obs_radii: torch.Tensor,
                       iters: int = 300, refine_iters: int = 200,
                       lr: float = 0.05, refine_lr: float = 0.01,
                       margin: float = 0.02, w_col: float = 10.0,
                       jitter: float = 1e-2, seed: int = 0):
    """Batched gradient-based IK that reaches target_pos (B,3) while keeping the
    whole arm at least `margin` clear of the obstacles.

    This is the payoff of everything being differentiable: FK gives the reaching
    gradient, the sphere distance field gives the avoidance gradient, and Adam
    descends both at once, in two stages (coarse then refine).

    `jitter` deterministically perturbs the start (fixed `seed`): a sphere whose
    center coincides exactly with an obstacle center sits at a gradient singularity
    of the distance field (grad ||c_i - c_j|| = 0 at coincidence), and would never
    move without it.

    Like plain IK, this landscape has local minima — run a batch of seeds and take
    the best by `info["pos_error"]`/`info["clearance"]`. Returns (q (B,dof), info).
    """
    chain = model.chain
    lo, hi = chain.lower.to(q0.device), chain.upper.to(q0.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(q0.shape, generator=g, dtype=q0.dtype).to(q0.device)
    q = (q0.clone() + jitter * noise).requires_grad_(True)
    for stage_iters, stage_lr in ((iters, lr), (refine_iters, refine_lr)):
        opt = torch.optim.Adam([q], lr=stage_lr)
        for _ in range(stage_iters):
            opt.zero_grad()
            ee = forward_kinematics(chain, q)[:, link_index, :3, 3]
            pos_loss = ((ee - target_pos) ** 2).sum(dim=-1)
            d = distance_to_obstacles(model, q, obs_centers, obs_radii)
            col_loss = torch.relu(margin - d) ** 2
            (pos_loss + w_col * col_loss).sum().backward()
            opt.step()
            with torch.no_grad():
                q.clamp_(lo, hi)
    q = q.detach()
    ee = forward_kinematics(chain, q)[:, link_index, :3, 3]
    info = {
        "pos_error": (ee - target_pos).norm(dim=-1),
        "clearance": distance_to_obstacles(model, q, obs_centers, obs_radii),
    }
    return q, info
