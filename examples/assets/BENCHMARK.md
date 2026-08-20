# kinfast vs pytorch_kinematics (real Franka Panda)

Device: cpu. Median of 7 runs, per-batch wall time. Same request per
row; kinfast computes all 13 link frames per FK call, pytorch_kinematics
runs its fastest path (end_only). Reproduce: `python examples/benchmark.py`.

| batch | kinfast FK | pk FK | kinfast Jacobian | pk Jacobian |
|---|---|---|---|---|
| 1 | 0.53 ms | 0.53 ms | 0.95 ms | 0.71 ms |
| 100 | 0.95 ms | 2.21 ms | 3.81 ms | 1.41 ms |
| 1,000 | 4.34 ms | 3.29 ms | 8.54 ms | 6.77 ms |
| 10,000 | 15.58 ms | 12.86 ms | 30.13 ms | 18.96 ms |

## Single-query latency (the compiler)

`robot.compile()` generates straight-line code specialized to this robot
(tree unrolled, constants folded, axis zeros eliminated). One call, one
configuration, CPU:

| op | kinfast compiled | kinfast torch | pytorch_kinematics |
|---|---|---|---|
| FK (all links) | 9.8 us | 547.5 us | 656.7 us (ee only) |
| Jacobian | 25.0 us | 898.0 us | |

Control-loop math rate (FK+Jacobian per tick): **28,768 Hz** compiled vs 692 Hz torch.
