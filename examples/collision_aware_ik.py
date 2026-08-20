# examples/collision_aware_ik.py
"""kinfast demo: collision-aware IK.

A 6-DOF arm must reach a target, but an obstacle sits exactly where its elbow
wants to be. Plain IK punches through the obstacle; collision-aware IK exploits
the arm's redundancy to bend around it — same target, collision-free.

Usage:  python examples/collision_aware_ik.py [--out collision_ik.png]
Everything runs on CPU in a few seconds; no assets needed (the robot is inline).
"""
import argparse
import torch

import kinfast
from kinfast.fk import forward_kinematics
from kinfast.collision import SphereModel, distance_to_obstacles, collision_aware_ik

ARM = """
<robot name="arm6">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/>
  <link name="l4"/><link name="l5"/><link name="ee"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="2" effort="50"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="2" effort="50"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/></joint>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" velocity="2" effort="50"/></joint>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="ee"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/></joint>
</robot>
"""


def link_points(robot, q):
    """World positions of every link origin, ordered base -> ee, for plotting."""
    world = forward_kinematics(robot.chain, q)
    order = ["base", "l1", "l2", "l3", "l4", "l5", "ee"]
    return torch.stack([world[0, robot.link_id(n), :3, 3] for n in order])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="collision_ik.png")
    args = ap.parse_args()

    robot = kinfast.load_string(ARM)
    li = robot.link_id("ee")
    model = robot.sphere_model({
        "l2": [(0.0, 0.0, 0.0, 0.05)],
        "l3": [(0.0, 0.0, 0.0, 0.05)],
        "l4": [(0.0, 0.0, 0.0, 0.05)],
    })

    # A reaching pose, its ee target, and an obstacle planted at its elbow.
    q_plain = torch.tensor([[0.5, 0.7, -0.4, 0.3, 0.5, 0.0]])
    world = forward_kinematics(robot.chain, q_plain)
    target = world[:, li, :3, 3].clone()
    obs_c = world[:, robot.link_id("l3"), :3, 3].clone()
    obs_r = torch.tensor([0.10])

    d0 = distance_to_obstacles(model, q_plain, obs_c, obs_r).item()
    q_safe, info = collision_aware_ik(model, target, q_plain, li, obs_c, obs_r)
    print(f"plain IK pose:            clearance {d0:+.3f} m  (negative = collision)")
    print(f"collision-aware IK pose:  clearance {info['clearance'].item():+.3f} m, "
          f"target error {info['pos_error'].item()*100:.1f} cm")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
        for ax, q, title, color in (
            (axes[0], q_plain, f"plain IK: collides ({d0:+.2f} m)", "#d62728"),
            (axes[1], q_safe, f"collision-aware: clear "
                              f"({info['clearance'].item():+.2f} m)", "#2ca02c"),
        ):
            pts = link_points(robot, q).detach()
            ax.add_patch(plt.Circle((obs_c[0, 0], obs_c[0, 2]), obs_r[0],
                                    color="#7f7f7f", alpha=0.7, label="obstacle"))
            ax.plot(pts[:, 0], pts[:, 2], "-o", lw=3, ms=5, color=color)
            ax.plot(target[0, 0], target[0, 2], "*", ms=16, color="#ff7f0e",
                    label="target")
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
            ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print("wrote", args.out)
    except Exception as e:  # rendering is optional
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
