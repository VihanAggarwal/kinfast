# src/kinfast/fk_subset.py
"""Forward kinematics for a few links instead of the whole robot.

Full FK sweeps every link in the tree. When the caller only wants the pose of
one gripper on a robot that also carries a torso, a head and a second arm, most
of that sweep is wasted work: a link's world pose depends only on its own
ancestors. This module walks the ancestor closure of the requested links and
nothing else, which on wide trees is the difference between touching two
hundred links and touching eight.

The pruning is purely structural, so it is worth computing once and reusing.
LinkSet holds the pruned topological order plus the per-device, per-dtype
constants that the sweep needs; :func:`fk_links` is the one-shot wrapper that
looks up (or builds) a cached LinkSet on the chain and returns the poses.

Everything here follows the rest of the library: q leads with a batch
dimension, the working dtype and device come from q rather than from the
compiled chain, and the sweep is plain differentiable tensor algebra so
gradients flow back to q.
"""
import torch
from kinfast import transforms as T
from kinfast.compile import CompiledChain


def _resolve(chain: CompiledChain, link) -> int:
    """Turn one link reference into a plain non-negative index.

    Accepts a link name, a non-negative index, or a Python-style negative index
    (-1 is the last link, matching forward_kinematics(...)[:, -1]).
    """
    if isinstance(link, str):
        try:
            return chain.link_index[link]
        except KeyError:
            raise KeyError(
                f"unknown link {link!r}; known links: {chain.link_names}") from None
    n = chain.n_links
    idx = int(link)
    if not -n <= idx < n:
        raise IndexError(
            f"link index {link} out of range for {n} links "
            f"(use a link name or chain.link_index[name])")
    return idx % n


def _as_sequence(link_indices):
    """Let callers pass a single link, or any iterable of links."""
    if isinstance(link_indices, (str, int)):
        return [link_indices]
    if isinstance(link_indices, torch.Tensor):
        if link_indices.ndim == 0:
            return [link_indices]
        return list(link_indices)
    return list(link_indices)


class LinkSet:
    """A fixed set of links, with the pruned FK sweep that reaches them.

    Construction resolves the requested links, closes them under the parent
    relation, and keeps the resulting subset in the chain's topological order,
    so a parent is always evaluated before its children. Positions in that
    pruned array (not link indices) drive the sweep, which is why the parent
    pointers are re-expressed as positions.

    The object is immutable and safe to keep around for the life of the chain.
    Per-device and per-dtype constants are built lazily and cached, the same way
    fk.py caches its own, so moving the chain to a GPU or calling with float64
    costs one rebuild and not one per call.
    """

    __slots__ = ("chain", "links", "order", "parent_pos", "out_pos", "_const")

    def __init__(self, chain: CompiledChain, link_indices):
        links = tuple(_resolve(chain, l) for l in _as_sequence(link_indices))
        parent = chain.parent.tolist()

        needed = set()
        stack = list(links)
        while stack:
            i = stack.pop()
            if i in needed:
                continue
            needed.add(i)
            p = parent[i]
            if p >= 0:
                stack.append(p)

        # chain.topo_order lists every link parents-first, so filtering it keeps
        # a valid topological order over the pruned subset.
        order = tuple(i for i in chain.topo_order if i in needed)
        if len(order) != len(needed):
            # topo_order is a walk down from the root, so a link missing from it
            # is not attached to the root and has no world pose to compute.
            stranded = sorted(needed.difference(order))
            raise ValueError(
                "these links are not connected to the root and cannot be "
                "reached by a forward sweep: "
                + ", ".join(chain.link_names[i] for i in stranded))
        pos_of = {link: k for k, link in enumerate(order)}

        self.chain = chain
        self.links = links
        self.order = order
        self.parent_pos = tuple(
            pos_of[parent[i]] if parent[i] >= 0 else -1 for i in order)
        self.out_pos = tuple(pos_of[l] for l in links)
        self._const = {}

    def __len__(self):
        return len(self.links)

    def __repr__(self):
        names = [self.chain.link_names[i] for i in self.links]
        return (f"LinkSet(links={names}, visited={len(self.order)}"
                f"/{self.chain.n_links})")

    @property
    def n_visited(self) -> int:
        """How many links the pruned sweep touches, ancestors included."""
        return len(self.order)

    def _constants(self, device, dtype):
        """Per-(device, dtype) constants for the pruned subset: origin R/p,
        the revolute and prismatic positions with their q columns, and the
        constant prismatic direction origin_R @ axis."""
        key = (str(device), str(dtype))
        hit = self._const.get(key)
        if hit is not None:
            return hit

        chain = self.chain
        sub = torch.tensor(self.order, dtype=torch.long, device=device)
        origin = chain.joint_origin.to(device=device, dtype=dtype)[sub]
        oR = origin[:, :3, :3].contiguous()
        op = origin[:, :3, 3].contiguous()
        axes = chain.joint_axis.to(device=device, dtype=dtype)[sub]
        jt = chain.joint_type.to(device)[sub]
        qi = chain.q_index.to(device)[sub]

        movable = qi >= 0
        rev = torch.nonzero(movable & (jt == 1), as_tuple=False).flatten()
        pris = torch.nonzero(movable & (jt == 2), as_tuple=False).flatten()
        entry = {
            "oR": oR, "op": op,
            "rev": rev, "rev_q": qi[rev], "rev_axes": axes[rev],
            "rev_oR": oR[rev],
            "pris": pris, "pris_q": qi[pris],
            "pris_dir": (oR[pris] @ axes[pris].unsqueeze(-1)).squeeze(-1),
            "pris_op": op[pris],
        }
        self._const[key] = entry
        return entry

    def _check_q(self, q: torch.Tensor):
        if q.ndim != 2:
            raise ValueError(
                f"q must be (B, dof); got shape {tuple(q.shape)}")
        if q.shape[1] != self.chain.dof:
            raise ValueError(
                f"q has {q.shape[1]} columns but the chain has "
                f"{self.chain.dof} degrees of freedom")

    def _sweep(self, q: torch.Tensor):
        """World R/p for every position in the pruned order."""
        self._check_q(q)
        B = q.shape[0]
        m = len(self.order)
        c = self._constants(q.device, q.dtype)

        local_R = c["oR"].unsqueeze(0).expand(B, m, 3, 3)
        local_p = c["op"].unsqueeze(0).expand(B, m, 3)
        if c["rev"].numel():
            vals = q[:, c["rev_q"]]                                  # (B, r)
            R = T.axis_angle_to_matrix(
                c["rev_axes"].unsqueeze(0).expand(B, -1, 3), vals)
            local_R = local_R.clone()
            local_R[:, c["rev"]] = c["rev_oR"].unsqueeze(0) @ R
        if c["pris"].numel():
            vals = q[:, c["pris_q"]]                                 # (B, p)
            local_p = local_p.clone()
            local_p[:, c["pris"]] = (
                c["pris_op"].unsqueeze(0)
                + c["pris_dir"].unsqueeze(0) * vals.unsqueeze(-1))

        wR = [None] * m
        wp = [None] * m
        for pos in range(m):
            pp = self.parent_pos[pos]
            if pp < 0:
                wR[pos], wp[pos] = local_R[:, pos], local_p[:, pos]
            else:
                wR[pos] = wR[pp] @ local_R[:, pos]
                wp[pos] = wp[pp] + (wR[pp] @ local_p[:, pos].unsqueeze(-1)).squeeze(-1)
        return wR, wp

    def fk_rp(self, q: torch.Tensor):
        """World rotations and positions of the requested links, as lists
        (wR, wp) with wR[k] (B, 3, 3) and wp[k] (B, 3), in the order the links
        were requested. The assembly-free path, matching fk.fk_rp."""
        wR, wp = self._sweep(q)
        return [wR[p] for p in self.out_pos], [wp[p] for p in self.out_pos]

    def fk(self, q: torch.Tensor) -> torch.Tensor:
        """q (B, dof) -> world transforms (B, len(self), 4, 4) for the
        requested links, in the order they were requested."""
        wR, wp = self.fk_rp(q)
        B, k = q.shape[0], len(self.out_pos)
        M = torch.zeros(B, k, 4, 4, dtype=q.dtype, device=q.device)
        if k:
            M[:, :, :3, :3] = torch.stack(wR, dim=1)
            M[:, :, :3, 3] = torch.stack(wp, dim=1)
        M[:, :, 3, 3] = 1.0
        return M


def link_set(chain: CompiledChain, link_indices) -> LinkSet:
    """Get the LinkSet for these links, building and caching it on the chain.

    The cache is keyed by the resolved link indices, so asking for the same
    links by name or by index hits the same entry and repeated fk_links calls
    do not redo the closure walk.
    """
    if isinstance(link_indices, LinkSet):
        return link_indices
    key = tuple(_resolve(chain, l) for l in _as_sequence(link_indices))
    store = getattr(chain, "_link_set_cache", None)
    if store is None:
        store = {}
        object.__setattr__(chain, "_link_set_cache", store)
    hit = store.get(key)
    if hit is None:
        hit = LinkSet(chain, key)
        store[key] = hit
    return hit


def fk_links(chain: CompiledChain, q: torch.Tensor, link_indices) -> torch.Tensor:
    """Forward kinematics for selected links only.

    q is (B, dof); link_indices is a link name, a link index (negatives allowed),
    an iterable of either, or a LinkSet. The result is (B, len(link_indices),
    4, 4) and equals forward_kinematics(chain, q)[:, link_indices] to floating
    point, but only the ancestors of the requested links are ever evaluated.
    """
    return link_set(chain, link_indices).fk(q)
