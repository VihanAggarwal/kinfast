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

INK = "#e6ebf2"
DIM = "#8b97a8"
ACCENT = "#55d6c2"
WARM = "#ffb454"
BG = "#0f1219"
PANEL = "#161b24"


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
            fast._raw(ql)
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
        s.set_color("#232a36")
    ax.grid(True, color="#232a36", linewidth=0.6, alpha=0.8)


class Studio:
    """The window. Holds the robot, the axes, and the measurements."""

    def __init__(self, robot, name):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.robot = robot
        self.name = name
        self.q = torch.zeros(1, robot.dof)
        self.bench = {}

        self.fig = plt.figure(figsize=(15, 8.6), facecolor=BG)
        self.fig.canvas.manager.set_window_title(f"kinfast studio - {name}")
        gs = self.fig.add_gridspec(
            3, 2, width_ratios=[1.35, 1], height_ratios=[1, 1, 1],
            left=0.03, right=0.97, top=0.93, bottom=0.30, wspace=0.18, hspace=0.55)

        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_thr = self.fig.add_subplot(gs[0, 1])
        self.ax_lat = self.fig.add_subplot(gs[1, 1])
        self.ax_ik = self.fig.add_subplot(gs[2, 1])
        for ax, t in ((self.ax_thr, "throughput against batch size"),
                      (self.ax_lat, "single query latency"),
                      (self.ax_ik, "inverse kinematics against restarts")):
            _style_axes(ax, t)
            ax.text(0.5, 0.5, "press  run benchmarks", color=DIM, fontsize=9,
                    ha="center", va="center", transform=ax.transAxes)

        self.fig.text(0.03, 0.965, "kinfast studio", color=INK, fontsize=15,
                      fontweight="600")
        self.status = self.fig.text(
            0.155, 0.966,
            f"{name}   {robot.dof} dof   {robot.n_links} links",
            color=DIM, fontsize=9.5)

        # a slider per joint, laid out in two columns under the figure
        self.sliders = []
        lo, hi = robot.lower.tolist(), robot.upper.tolist()
        per_col = (robot.dof + 1) // 2
        for i, name_j in enumerate(robot.joint_names):
            col, row = divmod(i, per_col)
            ax = self.fig.add_axes([0.06 + col * 0.47, 0.235 - row * 0.028,
                                    0.34, 0.016], facecolor=PANEL)
            short = name_j if len(name_j) <= 14 else name_j[:6] + ".." + name_j[-6:]
            s = Slider(ax, short, lo[i], hi[i], valinit=0.0, color=ACCENT)
            s.label.set_color(DIM)
            s.label.set_fontsize(7.5)
            s.valtext.set_color(INK)
            s.valtext.set_fontsize(7.5)
            s.on_changed(self._on_slider)
            self.sliders.append(s)

        def button(x, label, cb, color=PANEL):
            ax = self.fig.add_axes([x, 0.035, 0.135, 0.042])
            b = Button(ax, label, color=color, hovercolor="#232b38")
            b.label.set_color(INK)
            b.label.set_fontsize(9)
            b.on_clicked(cb)
            return b

        self.buttons = [
            button(0.06, "home", self._home),
            button(0.205, "random pose", self._random),
            button(0.35, "solve ik", self._solve),
            button(0.495, "workspace", self._workspace),
            button(0.64, "run benchmarks", self._run_bench),
            button(0.785, "save png", self._save),
        ]
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
        leg = ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor="#232a36",
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
            s.set_color("#232a36")
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
            pane.set_pane_color((0.06, 0.07, 0.10, 1.0))
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
        P = self.robot.fk_all(self.q)[0, :, :3, 3].detach().numpy()
        cloud = getattr(self, "cloud", None)
        if cloud is not None:
            import numpy as np
            P = np.vstack([P, cloud])
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
