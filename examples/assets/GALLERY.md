# kinfast real-robot gallery

Every robot below is a real, unmodified URDF from a public repo
(bullet3 pybullet_data; UR5 from pytorch_kinematics tests), loaded and
exercised by `python examples/gallery.py` on this machine (CPU).

| robot | links | dof | FK per config | IK round-trip (<5cm, restarts=8) |
|---|---|---|---|---|
| a1 | 22 | 12 | 6.7 us | 100% |
| cartpole | 3 | 2 | 1.8 us | 100% |
| husky | 11 | 4 | 3.7 us | 100% |
| kuka_iiwa | 8 | 7 | 4.8 us | 100% |
| laikago | 13 | 12 | 5.1 us | 100% |
| minitaur | 27 | 16 | 9.4 us | 100% |
| panda | 13 | 9 | 5.5 us | 100% |
| r2d2 | 16 | 8 | 6.5 us | 100% |
| racecar | 13 | 6 | 5.0 us | 100% |
| ur5 | 9 | 6 | 4.1 us | 100% |
| xarm6 | 8 | 6 | 4.0 us | 100% |

Loaded 11/11 robots.
