"""kinfast studio: a desktop window for looking at a robot and measuring it.

    python -m kinfast.studio
    python -m kinfast.studio --robot so101
    python -m kinfast.studio --robot panda --bench      measure on startup

The window has two halves. On the left is the robot, drawn from the same
forward kinematics the tests check, with a slider per joint. On the right are
three panels that stay empty until you press "run benchmarks", because every
number in them is measured on this machine when you ask for it, not read from a
table someone typed:

  throughput against batch size
      Forward kinematics and Jacobians timed from batch 1 up to batch 10,000,
      plotted as configurations per second. The curve bends where the work
      stops being dominated by framework overhead and starts being dominated by
      arithmetic, which is the whole argument for batching.

  single query latency
      The same forward kinematics call through the tensor path and through
      `robot.compile()`, which generates straight line code for this specific
      robot. A control loop makes exactly this call, one configuration at a
      time, and the bar chart is on a log scale because the difference does not
      fit on a linear one.

  inverse kinematics against restarts
      Solve rate and wall time for a batch of reachable targets as the number
      of random seeds per target goes up. Damped least squares finds a local
      minimum; more seeds in the same batched call find more of the global one.

Matplotlib is the only extra dependency and it is optional for the library as a
whole, so this module is never imported at package import time.
"""
import argparse
import glob
import os
import time
from pathlib import Path

import torch

import kinfast
from kinfast import analysis

REPO = Path(__file__).resolve().parent.parent.parent

# The palette is a terminal's: the sixteen ANSI colours as Windows Terminal and
# VS Code render them, on the same near black a console uses. A tool that lives
# next to a prompt may as well look like it does.
BG = "#0c0c0c"          # ANSI black, the console background
PANEL = "#101010"       # a hair lighter, so a plot reads as a pane
INK = "#cccccc"         # ANSI white
DIM = "#767676"         # ANSI bright black, for labels and axes
ACCENT = "#16c60c"      # bright green, the colour a terminal says "ok" in
WARM = "#f9f1a5"        # bright yellow, the second series
ALERT = "#e74856"       # bright red, for the thing you are avoiding
COOL = "#61d6d6"        # bright cyan, for paths and extra series
GRID = "#2a2a2a"


def find_robots():
    """Robot files this checkout can offer, by short name."""
    out = {}
    for pat in ("examples/assets/gallery/*.urdf",
                "examples/assets/menagerie/*/*.xml",
                "examples/assets/*.urdf"):
        for p in sorted(glob.glob(str(REPO / pat))):
            if os.path.basename(p) == "scene.xml":
                continue
            out.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return out


def best_of(fn, runs=7, warmup=2):
    """Fastest of several runs, which is the honest number for a benchmark.

    The mean on a laptop measures the operating system as much as the code:
    something else gets a time slice and the sample is ruined. The minimum is
    the run where that happened least.
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# --------------------------------------------------------------- measurements
def bench_throughput(robot, batches=(1, 10, 100, 1000, 10000), report=print):
    """Configurations per second for FK and Jacobians, per batch size."""
    fk, jac = [], []
    for b in batches:
        q = robot.random_configs(b)
        t_fk = best_of(lambda: robot.fk_all(q), runs=5)
        t_j = best_of(lambda: robot.jacobian(q), runs=3)
        fk.append(b / t_fk)
        jac.append(b / t_j)
        report(f"  batch {b:>6,}: fk {b / t_fk:>12,.0f}/s   jacobian {b / t_j:>12,.0f}/s")
    return list(batches), fk, jac


def bench_single_query(robot, report=print):
    """Latency of one forward kinematics call, both paths, in microseconds."""
    q1 = robot.random_configs(1)
    t_torch = best_of(lambda: robot.fk_all(q1), runs=7) * 1e6
    out = {"tensor path": t_torch}
    try:
        fast = robot.compile()
        ql = q1[0].tolist()
        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            # the public call, which assembles 4x4 transforms like the tensor
            # path it is being compared against. fast._raw is quicker but hands
            # back a flat list of floats, so timing it here would be comparing
            # different work.
            fast.fk(ql)
        out["compiled"] = (time.perf_counter() - t0) / n * 1e6
    except Exception as exc:
        report(f"  compiled path unavailable: {type(exc).__name__}: {exc}")
    for k, v in out.items():
        report(f"  {k:<14} {v:>9.1f} us")
    return out


def bench_ik(robot, restarts=(1, 2, 4, 8), targets=256, iters=60, report=print):
    """Solve rate and wall time for a batch of reachable targets."""
    torch.manual_seed(0)
    goal = robot.fk(robot.random_configs(targets))
    rate, secs = [], []
    for r in restarts:
        t0 = time.perf_counter()
        q, _ = robot.ik(goal, iters=iters, pos_only=True, restarts=r)
        dt = time.perf_counter() - t0
        err = (robot.fk(q)[:, :3, 3] - goal[:, :3, 3]).norm(dim=-1)
        pct = float((err < 0.05).float().mean()) * 100
        rate.append(pct)
        secs.append(dt)
        report(f"  restarts {r}: {pct:5.1f}% within 5 cm   {dt * 1e3:8.1f} ms "
               f"({targets * r / dt:,.0f} seed solves/s)")
    return list(restarts), rate, secs


# ------------------------------------------------------------------- drawing
def _style_axes(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left", pad=6)
    ax.tick_params(colors=DIM, labelsize=7.5)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)


class Studio:
    """The window. Holds the robot, the axes, and the measurements."""

    def __init__(self, robot, name):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        # a console font, with a stack that degrades sensibly off Windows
        plt.rcParams["font.family"] = "monospace"
        plt.rcParams["font.monospace"] = [
            "Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Menlo",
            "Liberation Mono", "monospace"]

        self.plt = plt
        self.robot = robot
        self.name = name
        self.q = torch.zeros(1, robot.dof)
        self.bench = {}

        self.fig = plt.figure(figsize=(15.5, 9.0), facecolor=BG)
        self.fig.canvas.manager.set_window_title(f"kinfast studio - {name}")
        # Two rows of two panels rather than four stacked ones. Four in a
        # column left each of them too short to read a curve in.
        gs = self.fig.add_gridspec(
            2, 3, width_ratios=[1.45, 1, 1], height_ratios=[1, 1],
            left=0.035, right=0.975, top=0.905, bottom=0.36,
            wspace=0.26, hspace=0.42)

        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_thr = self.fig.add_subplot(gs[0, 1])
        self.ax_lat = self.fig.add_subplot(gs[0, 2])
        self.ax_ik = self.fig.add_subplot(gs[1, 1])
        self.ax_plan = self.fig.add_subplot(gs[1, 2])
        for ax, t, hint in (
                (self.ax_thr, "throughput against batch size", "run benchmarks"),
                (self.ax_lat, "single query latency", "run benchmarks"),
                (self.ax_ik, "inverse kinematics against restarts", "run benchmarks"),
                (self.ax_plan, "planned motion", "plan around an obstacle")):
            _style_axes(ax, t)
            # an empty panel showing a 0 to 1 axis is just noise; the ticks
            # arrive with the data
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, f"press  {hint}", color=DIM, fontsize=9,
                    ha="center", va="center", transform=ax.transAxes)

        self.fig.text(0.035, 0.955, "kinfast studio", color=INK, fontsize=15,
                      fontweight="bold")
        self.status = self.fig.text(
            0.163, 0.9565,
            f"{name}   {robot.dof} dof   {robot.n_links} links",
            color=DIM, fontsize=9.5)
        # hairlines separating header, plots and controls
        for y in (0.937, 0.335):
            self.fig.add_artist(plt.Line2D(
                [0.035, 0.975], [y, y], color=GRID, linewidth=0.8))
        self.fig.text(0.035, 0.315, "joints", color=DIM, fontsize=8.5,
                      fontweight="bold")

        # a slider per joint, laid out in two columns under the figure
        self.sliders = []
        lo, hi = robot.lower.tolist(), robot.upper.tolist()
        # Spacing is derived from the joint count rather than fixed, because a
        # 16 dof robot used to run its last slider straight into the buttons.
        n_cols = 3 if robot.dof > 12 else 2
        per_col = -(-robot.dof // n_cols)
        top_y, floor_y = 0.288, 0.108
        step = min(0.029, (top_y - floor_y) / max(per_col, 1))
        # the left margin has to clear the joint name, which matplotlib draws
        # outside the slider axes
        x0, col_w = 0.085, 0.895 / n_cols
        for i, name_j in enumerate(robot.joint_names):
            col, row = divmod(i, per_col)
            ax = self.fig.add_axes([x0 + col * col_w, top_y - row * step,
                                    col_w * 0.58, 0.014], facecolor=PANEL)
            short = name_j if len(name_j) <= 14 else name_j[:6] + ".." + name_j[-6:]
            s = Slider(ax, short, lo[i], hi[i], valinit=0.0, color=ACCENT)
            s.label.set_color(DIM)
            s.label.set_fontsize(7.5)
            s.valtext.set_color(INK)
            s.valtext.set_fontsize(7.5)
            s.on_changed(self._on_slider)
            self.sliders.append(s)

        specs = [("home", self._home), ("random pose", self._random),
                 ("solve ik", self._solve), ("plan around", self._plan),
                 ("workspace", self._workspace),
                 ("run benchmarks", self._run_bench), ("save png", self._save)]
        # centred as a group, so the row stays balanced if buttons are added
        bw, gap, by, bh = 0.118, 0.016, 0.032, 0.040
        span = len(specs) * bw + (len(specs) - 1) * gap
        x = (1.0 - span) / 2.0

        def button(x, label, cb, color=PANEL):
            ax = self.fig.add_axes([x, by, bw, bh])
            b = Button(ax, label, color=color, hovercolor="#1f1f1f")
            b.label.set_color(INK)
            b.label.set_fontsize(9)
            b.on_clicked(cb)
            return b

        self.buttons = []
        for label, cb in specs:
            self.buttons.append(button(x, label, cb))
            x += bw + gap
        self._draw_robot()

    # ---- state changes
    def _on_slider(self, _):
        self.q = torch.tensor([[s.val for s in self.sliders]], dtype=torch.float32)
        self._draw_robot()

    def _set_q(self, q):
        self.q = q
        for s, v in zip(self.sliders, q[0].tolist()):
            s.eventson = False
            s.set_val(v)
            s.eventson = True
        self._draw_robot()

    def _home(self, _):
        self._set_q(torch.zeros(1, self.robot.dof))

    def _random(self, _):
        self._set_q(self.robot.random_configs(1))

    def _solve(self, _):
        """Pick a reachable pose, forget it, and ask the library to find it."""
        goal = self.robot.fk(self.robot.random_configs(1))
        t0 = time.perf_counter()
        q, _info = self.robot.ik(goal, iters=100, pos_only=True, restarts=8)
        ms = (time.perf_counter() - t0) * 1e3
        err = float((self.robot.fk(q)[0, :3, 3] - goal[0, :3, 3]).norm()) * 1000
        self._set_q(q)
        self._say(f"ik: {err:.1f} mm in {ms:.0f} ms (8 seeds, one batched call)")

    def _workspace(self, _):
        self._say("sampling the reachable workspace...")
        ws = analysis.workspace(self.robot.chain,
                                self.robot.link_id(self.robot.ee_link), n=6000)
        self.cloud = ws["points"].numpy()
        self._draw_robot()
        self._say(f"workspace: reach {float(ws['min_reach']):.2f} to "
                  f"{float(ws['max_reach']):.2f} m, 6000 samples")

    def _spheres(self):
        """Collision spheres for this robot, however we can get them.

        A model that ships collision primitives gets real ones. A model that
        ships only meshes, or nothing, gets one sphere per link sized to the
        gap between links, which is enough for a planner to have something to
        keep out of an obstacle.
        """
        if getattr(self, "_sphere_cache", None) is not None:
            return self._sphere_cache
        model = None
        try:
            from kinfast.collision_auto import auto_sphere_model
            model = auto_sphere_model(self.robot.ir, self.robot.chain)
            if getattr(model, "n", 0) == 0:
                model = None
        except Exception:
            model = None
        if model is None:
            P = self.robot.fk_all(torch.zeros(1, self.robot.dof))[0, :, :3, 3]
            span = float((P.max(dim=0).values - P.min(dim=0).values).max())
            r = max(span * 0.06, 0.01)
            model = self.robot.sphere_model(
                {i: [(0.0, 0.0, 0.0, r)] for i in range(self.robot.n_links)})
        self._sphere_cache = model
        return model

    def _plan(self, _):
        """Put an obstacle on the straight line and plan around it.

        The obstacle goes where the tool would pass if the arm simply
        interpolated from start to goal, so the direct move is always blocked
        and the planner always has real work to do.
        """
        from kinfast.collision_world import Sphere
        from kinfast.planning import CollisionChecker, rrt_connect

        start = self.q[0].clone()
        goal = self.robot.random_configs(1)[0]
        mid = ((start + goal) / 2).unsqueeze(0)
        hit = self.robot.fk(mid)[0, :3, 3]
        reach = float(self.robot.fk(self.q)[0, :3, 3].norm()) or 1.0
        radius = max(reach * 0.16, 0.04)
        world = [Sphere(center=hit.tolist(), radius=radius)]

        checker = CollisionChecker(self.robot, self._spheres(), world,
                                   self_collision=False)
        if not bool(checker(torch.stack([start, goal])).all()):
            self._say("start or goal is inside the obstacle, press again")
            return
        self._say("planning...")
        plan = rrt_connect(self.robot.chain, start, goal, checker,
                           seed=int(time.time()) % 10000, max_iters=4000)
        self.obstacle = (hit.tolist(), radius)
        print(plan.stats)
        if not plan.solved:
            self.plan = None
            self._draw_robot()
            self._say(str(plan.stats))
            return
        self.plan = plan
        self._draw_plan_panel(plan)
        self._animate(plan)
        self._say(str(plan.stats))

    def _animate(self, plan):
        """Walk the arm along the planned path so the motion is visible."""
        _t, q, _qd, _qdd, _T = plan.to_trajectory(self.robot)
        keep = max(int(q.shape[0] // 60), 1)
        interactive = self.plt.get_backend().lower() not in ("agg", "pdf", "svg", "ps")
        for row in q[::keep]:
            self.q = row.unsqueeze(0)
            self._draw_robot()
            if interactive:
                self.plt.pause(0.001)
        self._set_q(plan.path[-1].unsqueeze(0))

    def _draw_plan_panel(self, plan):
        """Joint angles against time: the plot a motion is actually read in."""
        t, q, _qd, _qdd, T = plan.to_trajectory(self.robot)
        ax = self.ax_plan
        ax.clear()
        _style_axes(ax, f"planned motion, {len(plan)} waypoints, {T:.2f} s")
        colors = [ACCENT, WARM, COOL, ALERT, INK, DIM]
        for j in range(q.shape[1]):
            ax.plot(t, q[:, j], lw=1.5, color=colors[j % len(colors)],
                    label=self.robot.joint_names[j][:10])
        ax.set_xlabel("seconds", color=DIM, fontsize=8)
        ax.set_ylabel("radians", color=DIM, fontsize=8)
        if q.shape[1] <= 6:
            leg = ax.legend(fontsize=6.5, facecolor=PANEL, edgecolor=GRID,
                            labelcolor=INK, ncol=2, loc="upper right")
            leg.get_frame().set_alpha(0.9)
        self.fig.canvas.draw_idle()

    def _say(self, msg):
        self.status.set_text(f"{self.name}   {self.robot.dof} dof   "
                             f"{self.robot.n_links} links   |   {msg}")
        print(msg)
        self.fig.canvas.draw_idle()

    def _save(self, _):
        out = REPO / f"studio_{self.name}.png"
        self.fig.savefig(out, dpi=160, facecolor=BG)
        self._say(f"wrote {out}")

    # ---- benchmarks
    def _run_bench(self, _):
        self._say("measuring, this takes about a minute...")
        # let the label paint before the machine gets busy; only a real window
        # has an event loop to pump, so headless runs skip it
        if self.plt.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
            self.plt.pause(0.01)
        print(f"\nkinfast benchmarks on {self.name} "
              f"({self.robot.dof} dof, {self.robot.n_links} links)")
        print("throughput")
        self.bench["thr"] = bench_throughput(self.robot)
        print("single query")
        self.bench["lat"] = bench_single_query(self.robot)
        print("inverse kinematics")
        self.bench["ik"] = bench_ik(self.robot)
        self._draw_bench()
        self._say("measured on this machine, just now")

    def _draw_bench(self):
        b, fk, jac = self.bench["thr"]
        ax = self.ax_thr
        ax.clear(); _style_axes(ax, "throughput against batch size")
        ax.loglog(b, fk, "o-", color=ACCENT, lw=2, ms=4, label="forward kinematics")
        ax.loglog(b, jac, "s-", color=WARM, lw=2, ms=4, label="jacobian")
        ax.set_xlabel("configurations per call", color=DIM, fontsize=8)
        ax.set_ylabel("configurations / s", color=DIM, fontsize=8)
        leg = ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID,
                        labelcolor=INK)
        leg.get_frame().set_alpha(0.9)

        lat = self.bench["lat"]
        ax = self.ax_lat
        ax.clear(); _style_axes(ax, "single query latency, one configuration")
        names = list(lat)
        vals = [lat[k] for k in names]
        bars = ax.barh(names, vals, color=[WARM, ACCENT][: len(names)], height=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("microseconds per call, lower is better", color=DIM, fontsize=8)
        for rect, v in zip(bars, vals):
            ax.text(v * 1.15, rect.get_y() + rect.get_height() / 2,
                    f"{v:,.1f} us", va="center", color=INK, fontsize=8)
        if len(vals) == 2 and min(vals) > 0:
            ax.set_title(f"single query latency, one configuration "
                         f"({max(vals) / min(vals):.0f}x)",
                         color=INK, fontsize=9.5, loc="left", pad=6)
        ax.set_xlim(min(vals) * 0.4, max(vals) * 4)

        r, rate, secs = self.bench["ik"]
        ax = self.ax_ik
        ax.clear(); _style_axes(ax, "inverse kinematics against restarts")
        ax.plot(r, rate, "o-", color=ACCENT, lw=2, ms=5)
        ax.set_xlabel("random seeds per target", color=DIM, fontsize=8)
        ax.set_ylabel("% within 5 cm", color=ACCENT, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_xticks(r)
        twin = ax.twinx()
        twin.plot(r, [s * 1e3 for s in secs], "s--", color=WARM, lw=1.6, ms=4)
        twin.set_ylabel("ms for 256 targets", color=WARM, fontsize=8)
        twin.tick_params(colors=DIM, labelsize=7.5)
        for s in twin.spines.values():
            s.set_color(GRID)
        self.fig.canvas.draw_idle()

    # ---- robot
    def _draw_robot(self):
        from kinfast.viz import plot as viz_plot   # lazy: matplotlib only here
        ax = self.ax3d
        ax.clear()
        ax.set_facecolor(BG)
        try:
            viz_plot(self.robot.chain, self.q, ax=ax)
            # the viz helper draws the structure; restyle it to match the window
            for line in ax.lines:
                line.set_color(ACCENT)
                line.set_linewidth(2.6)
                line.set_solid_capstyle("round")
            P = self.robot.fk_all(self.q)[0, :, :3, 3].detach().numpy()
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=22, c=INK,
                       depthshade=False, zorder=5)
        except Exception:
            self._fallback_draw(ax)
        obs = getattr(self, "obstacle", None)
        if obs is not None:
            (cx, cy, cz), r = obs
            u = torch.linspace(0, 6.28318, 22)
            v = torch.linspace(0, 3.14159, 11)
            X = cx + r * torch.outer(torch.cos(u), torch.sin(v))
            Y = cy + r * torch.outer(torch.sin(u), torch.sin(v))
            Z = cz + r * torch.outer(torch.ones_like(u), torch.cos(v))
            ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), color=ALERT,
                            alpha=0.20, linewidth=0, shade=False)
        plan = getattr(self, "plan", None)
        if plan is not None and plan.solved:
            tool = self.robot.fk(plan.densify(0.03))[:, :3, 3].detach().numpy()
            ax.plot(tool[:, 0], tool[:, 1], tool[:, 2], color=COOL, lw=1.5)
        cloud = getattr(self, "cloud", None)
        if cloud is not None:
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=0.5,
                       c=ACCENT, alpha=0.16, depthshade=False)
        ee = self.robot.fk(self.q)[0, :3, 3].tolist()
        ax.scatter([ee[0]], [ee[1]], [ee[2]], s=55, c=WARM, depthshade=False)
        self._equalize(ax)
        ax.set_title(f"{self.name}   end effector "
                     f"({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f}) m",
                     color=INK, fontsize=9.5, loc="left", pad=2)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0.05, 0.05, 0.05, 1.0))
            pane.label.set_color(DIM)
        ax.tick_params(colors=DIM, labelsize=7)
        self.fig.canvas.draw_idle()

    def _equalize(self, ax):
        """One metre on x has to look like one metre on z.

        Matplotlib stretches each axis to fill the box by default, which makes
        a tall arm look squat and a long one look stubby. The robot is the
        thing being judged here, so the box is squared off around the widest
        span and every axis gets the same range.
        """
        import numpy as np
        P = self.robot.fk_all(self.q)[0, :, :3, 3].detach().numpy()
        cloud = getattr(self, "cloud", None)
        if cloud is not None:
            P = np.vstack([P, cloud])
        obs = getattr(self, "obstacle", None)
        if obs is not None:                 # keep the obstacle inside the view
            (cx, cy, cz), r = obs
            P = np.vstack([P, [[cx - r, cy - r, cz - r], [cx + r, cy + r, cz + r]]])
        lo, hi = P.min(axis=0), P.max(axis=0)
        mid = (lo + hi) / 2
        span = max(float((hi - lo).max()), 0.2) * 0.62
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-58)

    def _fallback_draw(self, ax):
        """Draw the skeleton directly if the viz helper cannot handle a model."""
        chain = self.robot.chain
        P = self.robot.fk_all(self.q)[0, :, :3, 3].detach().numpy()
        for i in range(chain.n_links):
            p = int(chain.parent[i])
            if p >= 0:
                ax.plot(*zip(P[p], P[i]), color=ACCENT, lw=2)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=14, c=INK, depthshade=False)

    def show(self):
        self.plt.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", default=None, help="short name, see --list")
    ap.add_argument("--list", action="store_true", help="print the robots found")
    ap.add_argument("--bench", action="store_true", help="measure on startup")
    ap.add_argument("--save", default=None, help="write a png and exit")
    args = ap.parse_args(argv)

    cat = find_robots()
    if args.list or not cat:
        if not cat:
            print("no robot files found. Fetch some first:\n"
                  "    python examples/gallery.py --fetch")
            return 1
        print("\n".join(sorted(cat)))
        return 0

    name = args.robot if args.robot in cat else sorted(cat)[0]
    if args.robot and args.robot not in cat:
        print(f"unknown robot {args.robot!r}, using {name}. "
              f"see --list for the rest")
    print(f"loading {name}...")
    robot = kinfast.load(cat[name])

    if args.save:                        # headless: measure, draw, write, leave
        import matplotlib
        matplotlib.use("Agg")
    studio = Studio(robot, name)
    if args.bench or args.save:
        studio._run_bench(None)
    if args.save:
        studio.fig.savefig(args.save, dpi=160, facecolor=BG)
        print(f"wrote {args.save}")
        return 0
    studio.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
