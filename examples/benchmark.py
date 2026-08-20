# examples/benchmark.py
"""Honest, reproducible benchmark: kinfast vs pytorch_kinematics on the real
Franka Panda (batched FK and Jacobian, CPU or CUDA).

Methodology, stated plainly:
- Same task per row: end-effector pose (or Jacobian) for a batch of random
  configurations. Median of 7 timed runs after 2 warmups.
- kinfast's FK computes ALL link frames per call (its API contract);
  pytorch_kinematics is called with end_only=True (its fastest path). This is
  the honest comparison of what each library does on the same request.
- pytorch_kinematics is serial-chain-only here; kinfast handles full trees.
- Run this yourself: python examples/benchmark.py

The point of this benchmark is NOT "kinfast is fastest" (cuRobo's CUDA kernels
win on raw kernels). It is that kinfast is competitive while being the easiest
to get a real robot into.
"""
import argparse
import statistics
import time

import torch

import kinfast
import pytorch_kinematics as pk

PANDA = "examples/assets/panda.urdf"


def timed(fn, runs=7, warmup=2, sync=False):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="examples/assets/BENCHMARK.md")
    args = ap.parse_args()
    dev = args.device
    sync = dev.startswith("cuda")

    robot = kinfast.load(PANDA).to(dev)
    with open(PANDA, "rb") as f:
        chain = pk.build_serial_chain_from_urdf(f.read(), "panda_hand").to(device=dev)
    pk_names = chain.get_joint_parameter_names()
    idx = [robot.q_index(n) for n in pk_names]
    ee = robot.link_id("panda_hand")

    lines = [
        "# kinfast vs pytorch_kinematics (real Franka Panda)",
        "",
        f"Device: {dev}. Median of 7 runs, per-batch wall time. Same request per",
        "row; kinfast computes all 13 link frames per FK call, pytorch_kinematics",
        "runs its fastest path (end_only). Reproduce: `python examples/benchmark.py`.",
        "",
        "| batch | kinfast FK | pk FK | kinfast Jacobian | pk Jacobian |",
        "|---|---|---|---|---|",
    ]
    torch.manual_seed(0)
    for B in (1, 100, 1000, 10000):
        lo, hi = robot.lower[idx], robot.upper[idx]
        q7 = (lo + (hi - lo) * torch.rand(B, len(pk_names))).to(dev)
        qfull = torch.zeros(B, robot.dof, device=dev)
        qfull[:, idx] = q7

        t_ours_fk = timed(lambda: robot.fk_all(qfull), sync=sync)
        t_pk_fk = timed(lambda: chain.forward_kinematics(q7, end_only=True), sync=sync)
        t_ours_j = timed(lambda: robot.jacobian(qfull, link="panda_hand"), sync=sync)
        t_pk_j = timed(lambda: chain.jacobian(q7), sync=sync)

        fmt = lambda t: f"{t*1e3:.2f} ms"
        lines.append(f"| {B:,} | {fmt(t_ours_fk)} | {fmt(t_pk_fk)} | "
                     f"{fmt(t_ours_j)} | {fmt(t_pk_j)} |")

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
