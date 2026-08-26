# src/kinfast/viz.py
"""Matplotlib views of a compiled chain: a 3D stick figure and a gif.

Everything here is a sink at the end of the pipeline, so it works differently
from the rest of the library on purpose:

* matplotlib is imported lazily, inside the functions. Importing kinfast must
  stay cheap and must not drag a plotting stack into a training job, so the
  cost is paid only by the process that actually draws something.
* The backend defaults to Agg (headless, writes to files) unless the caller has
  already set one up. Plots then render the same over ssh, in CI, and in a
  notebook that picked its own backend first.
* Tensors are detached and moved to the CPU before drawing. A picture is not
  part of anyone's gradient, and this keeps the call safe on a q that requires
  grad or lives on a GPU.

Both entry points are batched: q of shape (B, dof) draws B configurations in
one axes (older ones faded, the last one solid), and animate treats the leading
dimension as time. Forward kinematics runs once for the whole batch, which is
what makes animating a few hundred frames cheap.
"""
import os
import sys

import numpy as np
import torch

from kinfast.fk import fk_rp

_DEFAULT_COLOR = "#1f77b4"
_DEFAULT_SPHERE_COLOR = "#d62728"


def _pyplot():
    """Import pyplot lazily, defaulting to the headless Agg backend.

    We only choose a backend when nobody else has: if pyplot is already
    imported, or MPLBACKEND is set, the caller has expressed a preference and
    we leave it alone. Otherwise Agg, so a plot never needs a display.
    """
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _as_chain(obj):
    """Accept a CompiledChain, or anything holding one (Robot, SphereModel)."""
    chain = getattr(obj, "chain", obj)
    if not hasattr(chain, "topo_order"):
        raise TypeError("expected a CompiledChain (or an object with a .chain), "
                        f"got {type(obj).__name__}")
    return chain


def _prepare_q(chain, q):
    """Normalize q to a floating (B, dof) tensor, leaving device and dtype alone.

    A bare (dof,) vector is the common interactive case, so it is promoted to a
    batch of one rather than rejected.
    """
    q = torch.as_tensor(q)
    if not torch.is_floating_point(q):
        q = q.to(torch.get_default_dtype())
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if q.dim() != 2:
        raise ValueError(f"q must be (dof,) or (B, dof), got shape {tuple(q.shape)}")
    if q.shape[1] != chain.dof:
        raise ValueError(f"q has {q.shape[1]} columns but the chain has "
                         f"dof {chain.dof}")
    return q


def _to_numpy(t):
    """Detach to CPU float64 numpy. float64 so half and bfloat16 also convert."""
    return t.detach().to(device="cpu", dtype=torch.float64).numpy()


def _positions(chain, q):
    """World-frame link origins for a batch. -> (B, n_links, 3) numpy."""
    _, wp = fk_rp(chain, q)
    return _to_numpy(torch.stack(wp, dim=1))


def _edges(chain):
    """Parent -> child index pairs, one per non-root link."""
    parent = chain.parent.tolist()
    return [(p, i) for i, p in enumerate(parent) if p >= 0]


def _resolve_spheres(chain, link_spheres, q):
    """Collision spheres as (centers (B, S, 3), radii (S,)) numpy, or None.

    Accepts a built SphereModel or the raw dict it is built from, keyed by link
    name or link index, so a caller can sketch spheres without constructing a
    model first.
    """
    if link_spheres is None:
        return None
    model = link_spheres
    if isinstance(link_spheres, dict):
        from kinfast.collision import SphereModel
        resolved = {}
        for key, spheres in link_spheres.items():
            if isinstance(key, str):
                if key not in chain.link_index:
                    raise KeyError(f"unknown link {key!r}: the chain has "
                                   f"{chain.link_names}")
                idx = chain.link_index[key]
            else:
                idx = int(key)
                if not 0 <= idx < chain.n_links:
                    raise IndexError(f"link index {idx} out of range for a "
                                     f"chain with {chain.n_links} links")
            resolved[idx] = spheres
        model = SphereModel(chain, resolved)
    if not hasattr(model, "centers_world"):
        raise TypeError("link_spheres must be a SphereModel or a "
                        "{link: [(x, y, z, r), ...]} dict")
    if model.n == 0:
        return None
    centers = _to_numpy(model.centers_world(q))
    radii = _to_numpy(model.radius)
    return centers, radii


def _stick_path(P, edges):
    """Link origins (n, 3) + edges -> one (3E, 3) NaN separated polyline.

    Every parent-child segment goes into a single artist, with a NaN row
    breaking the pen between them. One Line3D per configuration keeps a big
    batch (or a long animation) from piling up thousands of artists.
    """
    if not edges:
        return np.zeros((0, 3))
    out = np.full((3 * len(edges), 3), np.nan)
    for k, (p, c) in enumerate(edges):
        out[3 * k] = P[p]
        out[3 * k + 1] = P[c]
    return out


def _sphere_path(center, radius, seg=24):
    """Three orthogonal great circles as one NaN separated polyline.

    A wireframe ball reads as a volume without hiding the linkage behind it,
    and unlike plot_surface it stays cheap when a robot carries fifty spheres.
    """
    t = np.linspace(0.0, 2.0 * np.pi, seg + 1)
    c, s, z = np.cos(t), np.sin(t), np.zeros_like(t)
    gap = np.full((1, 3), np.nan)
    circles = [np.stack([c, s, z], axis=1),
               np.stack([c, z, s], axis=1),
               np.stack([z, c, s], axis=1)]
    pts = np.concatenate([np.concatenate([circle, gap]) for circle in circles])
    return np.asarray(center, dtype=float) + float(radius) * pts


def _bounds(P, spheres, pad=0.1):
    """A cube covering the drawing. -> (2, 3) numpy of [min; max].

    A cube, not a tight box, because matplotlib's 3D axes scale each axis
    independently: without equal spans a 1 m arm looks bent.
    """
    pts = [P.reshape(-1, 3)]
    if spheres is not None:
        centers, radii = spheres
        r = radii.reshape(1, -1, 1)
        pts.append((centers - r).reshape(-1, 3))
        pts.append((centers + r).reshape(-1, 3))
    allpts = np.concatenate(pts, axis=0)
    allpts = allpts[np.isfinite(allpts).all(axis=1)]
    if allpts.size == 0:
        allpts = np.zeros((1, 3))
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * float((hi - lo).max())
    if half <= 0.0:
        half = 0.5
    half *= 1.0 + 2.0 * pad
    return np.stack([center - half, center + half])


def _new_axes(plt, figsize):
    fig = plt.figure(figsize=figsize)
    return fig.add_subplot(projection="3d")


def _check_axes(ax):
    if getattr(ax, "name", None) != "3d":
        raise ValueError("ax must be a 3D axes, e.g. "
                         "fig.add_subplot(projection='3d')")


def _draw_config(ax, P, edges, spheres_b, color, sphere_color, alpha,
                 sphere_alpha, linewidth, markersize, seg):
    """Draw one configuration: the stick figure, its joints, and its spheres."""
    path = _stick_path(P, edges)
    ax.plot(path[:, 0], path[:, 1], path[:, 2],
            color=color, alpha=alpha, linewidth=linewidth, solid_capstyle="round")
    if markersize > 0:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                   color=color, alpha=alpha, s=markersize ** 2, depthshade=False)
    if spheres_b is not None:
        centers, radii = spheres_b
        for center, radius in zip(centers, radii):
            wire = _sphere_path(center, radius, seg=seg)
            ax.plot(wire[:, 0], wire[:, 1], wire[:, 2],
                    color=sphere_color, alpha=sphere_alpha * alpha, linewidth=1.0)


def _finish_axes(ax, limits, title, labels, chain, P_last):
    lo, hi = limits
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except (AttributeError, TypeError):   # very old matplotlib
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    if labels:
        for name, pos in zip(chain.link_names, P_last):
            ax.text(pos[0], pos[1], pos[2], name, fontsize=7)


def plot(chain, q, ax=None, link_spheres=None, *, color=_DEFAULT_COLOR,
         sphere_color=_DEFAULT_SPHERE_COLOR, sphere_alpha=0.6, linewidth=2.0,
         markersize=4.0, sphere_seg=24, limits=None, title=None, labels=False,
         figsize=(6.0, 6.0), elev=None, azim=None):
    """Draw a 3D stick figure of the robot: link origins joined parent to child.

    chain may be a CompiledChain or anything carrying one (a Robot). q is
    (dof,) or (B, dof); a batch is drawn in one axes with the earlier
    configurations faded, which is how a trajectory or an IK restart fan is
    usually inspected. link_spheres draws the collision model on top, given
    either as a SphereModel or as the {link: [(x, y, z, r), ...]} dict it is
    built from (link names allowed).

    Pass ax to draw into an existing 3D axes; otherwise a figure is created.
    elev and azim set the camera. Returns the matplotlib Figure either way, so
    the caller can savefig it.
    """
    plt = _pyplot()
    chain = _as_chain(chain)
    q = _prepare_q(chain, q)
    P = _positions(chain, q)
    spheres = _resolve_spheres(chain, link_spheres, q)
    edges = _edges(chain)

    if ax is None:
        ax = _new_axes(plt, figsize)
    else:
        _check_axes(ax)
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    B = P.shape[0]
    # fade the history so the newest configuration reads as the current one
    alphas = np.linspace(0.25, 1.0, B) if B > 1 else np.ones(1)
    for b in range(B):
        spheres_b = None if spheres is None else (spheres[0][b], spheres[1])
        _draw_config(ax, P[b], edges, spheres_b, color, sphere_color,
                     float(alphas[b]), sphere_alpha, linewidth, markersize,
                     sphere_seg)

    lim = _bounds(P, spheres) if limits is None else np.asarray(limits, dtype=float)
    if lim.shape != (2, 3):
        raise ValueError(f"limits must be (2, 3) [[xmin, ymin, zmin], "
                         f"[xmax, ymax, zmax]], got shape {tuple(lim.shape)}")
    _finish_axes(ax, lim, title, labels, chain, P[-1])
    return ax.figure


def animate(chain, qs, path, *, fps=20, link_spheres=None,
            color=_DEFAULT_COLOR, sphere_color=_DEFAULT_SPHERE_COLOR,
            sphere_alpha=0.6, linewidth=2.0, markersize=4.0, sphere_seg=24,
            title=None, labels=False, figsize=(6.0, 6.0), dpi=100,
            elev=None, azim=None):
    """Write a gif of the robot moving through qs, using the pillow writer.

    qs is (T, dof): the leading dimension is time, matching the output of the
    trajectory helpers. Forward kinematics runs once over the whole stack, and
    the axis limits are computed from every frame so the view does not jump
    while the arm swings. Pillow is used because it ships with matplotlib's
    dependencies, so no ffmpeg install is needed.

    Returns the path that was written.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt = _pyplot()
    chain = _as_chain(chain)
    qs = _prepare_q(chain, qs)
    T = qs.shape[0]
    if T == 0:
        raise ValueError("qs is empty: need at least one configuration to animate")
    P = _positions(chain, qs)
    spheres = _resolve_spheres(chain, link_spheres, qs)
    edges = _edges(chain)
    limits = _bounds(P, spheres)

    ax = _new_axes(plt, figsize)
    fig = ax.figure
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)
    view = (ax.elev, ax.azim)

    def update(t):
        ax.cla()
        ax.view_init(elev=view[0], azim=view[1])
        spheres_t = None if spheres is None else (spheres[0][t], spheres[1])
        _draw_config(ax, P[t], edges, spheres_t, color, sphere_color, 1.0,
                     sphere_alpha, linewidth, markersize, sphere_seg)
        _finish_axes(ax, limits, title, labels, chain, P[t])
        return ax.get_children()

    anim = FuncAnimation(fig, update, frames=T, blit=False)
    try:
        anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(fig)
    return path
