# examples/so101_tutorial.py
"""Companion script for docs/SO101_TUTORIAL.md: every number and figure in the
tutorial is produced by running this.

Usage:
  python examples/gallery.py --fetch          # grabs so101.urdf (once)
  python examples/so101_tutorial.py
"""
import time

import torch

import kinfast
from kinfast import analysis

URDF = "examples/assets/gallery/so101.urdf"


def main():
    # 1. load: the official SO-101 URDF, unmodified, no ROS anywhere
    robot = kinfast.load(URDF)
    print(f"SO-101 loaded: {robot.dof} dof, joints {robot.joint_names}")

    # 2. batched IK: solve 1000 gripper targets at once
    torch.manual_seed(0)
    targets = robot.fk(robot.random_configs(1000))
    t0 = time.perf_counter()
    q_sol, info = robot.ik(targets, iters=100, pos_only=True, restarts=8)
    dt = time.perf_counter() - t0
    err = (robot.fk(q_sol)[:, :3, 3] - targets[:, :3, 3]).norm(dim=-1)
    print(f"batched IK: 1000 targets -> {(err < 5e-2).float().mean()*100:.1f}% "
          f"within 5 cm in {dt:.1f} s (CPU)")

    # 3. workspace: where can my arm actually reach?
    ws = analysis.workspace(robot.chain, robot.link_id(robot.ee_link), n=20000)
    print(f"workspace: reach {ws['min_reach']:.3f} to {ws['max_reach']:.3f} m")

    # 4. the compiler: microsecond FK for your control loop
    fast = robot.compile()
    ql = robot.random_configs(1)[0].tolist()
    t0 = time.perf_counter()
    N = 5000
    for _ in range(N):
        fast._raw(ql)
    us = (time.perf_counter() - t0) / N * 1e6
    print(f"compiled single-query FK: {us:.1f} us "
          f"(~{1e6/us:,.0f} Hz if FK were your whole loop)")

    # 5. workspace figure for the tutorial
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = ws["points"].numpy()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        for ax, (a, b, la, lb) in zip(axes, [(0, 1, "x [m]", "y [m]"),
                                             (0, 2, "x [m]", "z [m]")]):
            ax.scatter(pts[:, a], pts[:, b], s=1, alpha=0.15, c="#1f77b4")
            ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_aspect("equal")
        axes[0].set_title("SO-101 reachable workspace (top)")
        axes[1].set_title("side")
        fig.tight_layout()
        fig.savefig("examples/assets/so101_workspace.png", dpi=130)
        print("wrote examples/assets/so101_workspace.png")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
