# examples/demo_10k_arms.py
"""kinfast demo: 10,000 arms solving IK at once.

Usage:
  python examples/demo_10k_arms.py --urdf examples/assets/panda.urdf --n 10000
Prints throughput; writes demo.gif of a representative subset (needs matplotlib).
"""
import argparse, time, torch
import kinfast


def bench(robot, n, iters):
    q_true = robot.random_configs(n)
    target = robot.fk(q_true)
    torch.cuda.synchronize() if robot.device.type == "cuda" else None
    t0 = time.perf_counter()
    q_sol, info = robot.ik(target, iters=iters, pos_only=True)
    torch.cuda.synchronize() if robot.device.type == "cuda" else None
    dt = time.perf_counter() - t0
    pos_err = (robot.fk(q_sol)[:, :3, 3] - target[:, :3, 3]).norm(dim=-1)
    return dt, pos_err, q_sol


def render_gif(robot, q_sol, path, k=48):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    frames = robot.fk_all(q_sol[:k].cpu())  # (k, n_links, 4, 4)
    xs = frames[:, :, 0, 3].detach().numpy()
    ys = frames[:, :, 1, 3].detach().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal"); ax.axis("off")

    def draw(f):
        ax.clear(); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{q_sol.shape[0]:,} arms solved — showing {f+1}/{k}")
        for j in range(f + 1):
            ax.plot(xs[j], ys[j], "-o", lw=1, ms=2, alpha=0.6)

    anim = FuncAnimation(fig, draw, frames=k, interval=80)
    anim.save(path, writer="pillow")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gif", default="demo.gif")
    args = ap.parse_args()

    robot = kinfast.load(args.urdf).to(args.device)
    print(f"loaded {args.urdf}: dof={robot.dof}, device={args.device}")
    dt, pos_err, q_sol = bench(robot, args.n, args.iters)
    solved = (pos_err < 5e-2).float().mean().item()
    print(f"solved {args.n:,} IK problems in {dt*1000:.1f} ms "
          f"({args.n/dt:,.0f} solves/s), {solved*100:.1f}% within 5cm")
    try:
        render_gif(robot, q_sol, args.gif)
    except Exception as e:
        print(f"(gif skipped: {e})")


if __name__ == "__main__":
    main()
