# Running the GPU checks

Everything in kinfast is developed on a CPU, so the GPU numbers and the gif in
the README have to come from a machine with a CUDA card. There are two ways to
get them, and both produce the same two artifacts: a benchmark table and a demo
gif.

## The short way: Colab

Open [examples/kinfast_gpu_colab.ipynb](../examples/kinfast_gpu_colab.ipynb) in
Colab, set the runtime to a T4 GPU, and run every cell. It clones the repo,
installs the dependencies without replacing Colab's CUDA build of torch, fetches
the robot files, and runs the three steps below. Roughly ten minutes, most of it
the install.

Two things to bring back:

- the table printed at the end, which is also written to
  `examples/assets/BENCHMARK_GPU.md`
- `demo_gpu.gif`, downloaded from the file browser on the left

## The longer way: your own machine

```bash
git clone https://github.com/VihanAggarwal/kinfast && cd kinfast
python -m venv .venv && .venv/bin/activate      # Scripts\activate on Windows
pip install -e ".[dev]"
pip install pytorch-kinematics mujoco xacro hypothesis
python examples/gallery.py --fetch
```

Then the same three steps, in this order.

### 1. Correctness before speed

```bash
pytest tests/test_gpu.py -v
```

Ten tests that run only when a CUDA device is present. Each one asks whether a
module gives the same answer on the GPU as on the CPU: forward kinematics,
Jacobians, IK (including gradients through the solve), dynamics, collision
distances, trajectories, and the workspace sampler. There is also a 200,000
configuration forward kinematics call to see that a large batch allocates.

If any of these fail, stop here and send me the output. A wrong answer on the
GPU is worth more than a fast one.

### 2. The benchmark

```bash
python examples/gpu_benchmark.py
```

Batched FK, Jacobian and IK throughput, best of ten with explicit
`torch.cuda.synchronize()` around every timed call so that a queued kernel is
never counted as finished work. Batches grow by powers of ten until the card
runs out of memory, and the largest one that fit is reported. Writes
`examples/assets/BENCHMARK_GPU.md`.

`--urdf` picks a different robot, `--max-batch` caps the growth if you would
rather not push the card to an out of memory error.

### 3. The demo gif

```bash
python examples/demo_10k_arms.py --urdf examples/assets/gallery/panda.urdf \
    --n 10000 --restarts 4 --gif demo_gpu.gif
```

Ten thousand IK problems in one batch, with a gif of a representative subset.
`--restarts 4` solves four seeds per target and keeps the best, which is what
takes the solve rate near 100 percent.

## After the run

Paste the benchmark table into the speed section of the README under the CPU
table, keeping the CPU numbers: the interesting story is the two regimes, not
one number. Commit `BENCHMARK_GPU.md` and the gif.

The CPU baseline the GPU numbers are compared against, measured on a laptop:

| what | CPU |
|---|---|
| FK, batch 10,000, all 13 Panda frames | 17.2 ms |
| internal FK hot path, batch 10,000 | 10.2 ms |
| IK, 10,000 targets, 4 restarts | 6.9 s (about 5,800 seed solves per second) |
| compiled single query FK | 4 to 10 us |
