# kinfast vs pytorch_kinematics (real Franka Panda)

Device: cpu. Median of 7 runs, per-batch wall time. Same request per
row; kinfast computes all 13 link frames per FK call, pytorch_kinematics
runs its fastest path (end_only). Reproduce: `python examples/benchmark.py`.

| batch | kinfast FK | pk FK | kinfast Jacobian | pk Jacobian |
|---|---|---|---|---|
| 1 | 0.50 ms | 0.38 ms | 0.72 ms | 0.47 ms |
| 100 | 1.21 ms | 1.03 ms | 1.65 ms | 1.54 ms |
| 1,000 | 4.52 ms | 3.10 ms | 4.06 ms | 4.43 ms |
| 10,000 | 17.20 ms | 12.22 ms | 19.06 ms | 15.17 ms |

## Single-query latency (the compiler)

`robot.compile()` generates straight-line code specialized to this robot
(tree unrolled, constants folded, axis zeros eliminated). One call, one
configuration, CPU:

| op | kinfast compiled | kinfast torch | pytorch_kinematics |
|---|---|---|---|
| FK (all links) | 10.1 us | 625.8 us | 472.6 us (ee only) |
| Jacobian | 24.2 us | 702.5 us | |

Control-loop math rate (FK+Jacobian per tick): **29,123 Hz** compiled vs 753 Hz torch.
