# kinfast vs pytorch_kinematics (real Franka Panda)

Device: cpu. Median of 7 runs, per-batch wall time. Same request per
row; kinfast computes all 13 link frames per FK call, pytorch_kinematics
runs its fastest path (end_only). Reproduce: `python examples/benchmark.py`.

| batch | kinfast FK | pk FK | kinfast Jacobian | pk Jacobian |
|---|---|---|---|---|
| 1 | 0.43 ms | 0.48 ms | 0.86 ms | 0.54 ms |
| 100 | 1.04 ms | 1.34 ms | 2.15 ms | 1.51 ms |
| 1,000 | 4.67 ms | 4.11 ms | 7.62 ms | 6.90 ms |
| 10,000 | 15.49 ms | 12.90 ms | 28.01 ms | 22.61 ms |
