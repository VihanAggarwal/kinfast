# examples/gpu_benchmark.py
"""GPU benchmark: batched FK, Jacobian, and IK throughput on CUDA, with the
same honest methodology as the CPU benchmark (interleaved, best-of-N, explicit
synchronization so kernel launches are not mistaken for completed work).

Usage:  python examples/gpu_benchmark.py [--urdf examples/assets/panda.urdf]
Writes examples/assets/BENCHMARK_GPU.md. pytorch_kinematics is included for
comparison when installed. Batches grow until the GPU runs out of memory;
the largest batch that fit is reported.
"""
import argparse
import os
import time

import torch

import kinfast


def best_of(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="examples/assets/panda.urdf")
    ap.add_argument("--out", default="examples/assets/BENCHMARK_GPU.md")
    ap.add_argument("--max-batch", type=int, default=1_000_000)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible to torch")

    gpu_name = torch.cuda.get_device_name(0)
    robot = kinfast.load(args.urdf).to("cuda")
    ee = robot.ee_link

    try:
        import pytorch_kinematics as pk
        with open(args.urdf, "rb") as f:
            pkc = pk.build_serial_chain_from_urdf(f.read(), "panda_hand").to(device="cuda")
        pk_names = pkc.get_joint_parameter_names()
        idx = [robot.q_index(n) for n in pk_names]
        have_pk = True
    except Exception:
        have_pk = False

    lines = [
        f"# kinfast on GPU ({gpu_name})",
        "",
        "Best of 10, explicit cuda synchronization around every timed call.",
        "kinfast FK returns all link frames; pytorch_kinematics is its ee-only",
        "path. Reproduce: `python examples/gpu_benchmark.py`.",
        "",
        "| batch | kinfast FK | kinfast Jacobian | pk FK | kinfast FK configs/s |",
        "|---|---|---|---|---|",
    ]
    batch = 1000
    largest = 0
    while batch <= args.max_batch:
        try:
            torch.manual_seed(0)
            q = robot.random_configs(batch)
            t_fk = best_of(lambda: robot.fk_all(q))
            t_j = best_of(lambda: robot.jacobian(q), reps=5)
            if have_pk:
                q7 = q[:, idx]
                t_pk = best_of(lambda: pkc.forward_kinematics(q7, end_only=True))
                pk_s = f"{t_pk*1e3:.2f} ms"
            else:
                pk_s = "n/a"
            lines.append(f"| {batch:,} | {t_fk*1e3:.2f} ms | {t_j*1e3:.2f} ms | {pk_s} | "
                         f"{batch/t_fk:,.0f} |")
            largest = batch
            del q
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            lines.append(f"| {batch:,} | OOM | | | |")
            break
        batch *= 10

    # the headline: batched IK
    lines += ["", "## Batched IK (position, 100 iterations)", "",
              "| targets | restarts | time | seed-solves/s | within 5 cm |", "|---|---|---|---|---|"]
    for n, restarts in ((10_000, 4), (100_000, 2)):
        if n > largest:
            break
        try:
            torch.manual_seed(1)
            targets = robot.fk(robot.random_configs(n))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            q_sol, info = robot.ik(targets, iters=100, pos_only=True, restarts=restarts)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            err = (robot.fk(q_sol)[:, :3, 3] - targets[:, :3, 3]).norm(dim=-1)
            ok = (err < 5e-2).float().mean().item() * 100
            lines.append(f"| {n:,} | {restarts} | {dt:.2f} s | {n*restarts/dt:,.0f} | {ok:.1f}% |")
            del targets, q_sol
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            lines.append(f"| {n:,} | {restarts} | OOM | | |")
            break

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
