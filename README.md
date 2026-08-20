# kinfast

**Load any robot. Run it 10,000x in parallel on GPU, or in microseconds one
query at a time. Differentiable end to end. Five lines, no ROS.**

```python
import kinfast
robot    = kinfast.load("panda.urdf")            # real robots load, unmodified
q        = robot.random_configs(10_000)          # 10k configs in one batch
ee       = robot.fk(q)                            # batched forward kinematics
q_solved, info = robot.ik(ee, restarts=8)         # batched, differentiable IK
fast     = robot.compile()                        # microsecond scalar backend
```

kinfast is a PyTorch robotics library that covers the whole robot program in one
coherent, batched, differentiable stack: **ingestion, kinematics, dynamics,
control, collision, trajectories, and workspace analysis.** Every claim below is
reproduced by a script in this repo.

## Verified against the real world

Two independent codebases must agree. kinfast's forward kinematics matches
[pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) on the
real Franka Panda to float32 machine precision:

> **512 random configurations, real Panda URDF: max position difference 1.3e-7 m,
> max rotation difference 3.0e-7.** (`tests/test_cross_validation.py`)

## Loads real robots, unmodified

Eleven unmodified URDFs straight from public repos, spanning arms, quadrupeds,
mobile bases, and classics. All eleven load; batched IK round-trips at 100% on
every one (`python examples/gallery.py`, CPU, full table in
`examples/assets/GALLERY.md`):

| robot | dof | FK per config | IK round-trip (<5cm) |
|---|---|---|---|
| Franka Panda | 9 | 5.5 us | 100% |
| KUKA iiwa | 7 | 4.8 us | 100% |
| UR5 | 6 | 4.1 us | 100% |
| xArm6 | 6 | 4.0 us | 100% |
| Unitree A1 (quadruped) | 12 | 6.7 us | 100% |
| Laikago (quadruped) | 12 | 5.1 us | 100% |
| Minitaur (quadruped) | 16 | 9.4 us | 100% |
| Husky, Racecar, R2D2, cartpole | ... | ... | 100% |

That includes URDFs other loaders choke on: Husky ships with unexpanded ROS
`$(optenv ...)` substitution args, and kinfast expands them.

## Fast where it counts

Honest benchmark on the real Panda, CPU, median of 7 runs
(`python examples/benchmark.py`, table in `examples/assets/BENCHMARK.md`).
kinfast computes all 13 link frames per FK call; pytorch_kinematics runs its
fastest end-effector-only path:

| batch | kinfast FK | pk FK | kinfast Jacobian | pk Jacobian |
|---|---|---|---|---|
| 1 | 0.43 ms | 0.48 ms | 0.86 ms | 0.54 ms |
| 100 | 1.04 ms | 1.34 ms | 2.15 ms | 1.51 ms |
| 10,000 | 15.49 ms | 12.90 ms | 28.01 ms | 22.61 ms |

We are not claiming fastest kernels on earth (cuRobo's CUDA wins raw FLOPS).
kinfast's bet is speed you can actually reach: competitive throughput on the
robot you loaded in one line.

## The compiler: C-like speed for control loops

Batched math is only half of robotics. The other half is a control loop or a
planner asking for ONE forward kinematics, right now, where framework overhead
is 99% of the cost. `robot.compile()` removes the framework: at load time
kinfast generates straight-line code specialized to your exact robot, with the
kinematic tree unrolled, every constant folded, and every multiply-by-zero from
axis-aligned joints eliminated at generation time. What remains is a few
hundred fused multiply-adds:

| op (real Panda, one query, CPU) | compiled | torch path | speedup |
|---|---|---|---|
| FK, all 13 link frames | 4-10 us | 350-550 us | 50-80x |
| geometric Jacobian | 10-25 us | 740-900 us | 40-70x |

That turns the FK+Jacobian tick of a controller from ~700 Hz (unusable) to
roughly 30,000-70,000 Hz: a 1 kHz real-time loop spends under 2% of its budget
on kinematics. The generated source is inspectable (`fast.source`), and its
correctness is tested against the batched path (itself cross-validated against
an independent library) on all 11 gallery robots.

## The whole robot program

```python
# dynamics: mass matrix, gravity, Coriolis, inverse/forward dynamics
M   = robot.mass_matrix(q)
tau = robot.inverse_dynamics(q, qd, qdd)

# control + simulation: gravity comp, PD, computed-torque, batched rollout
from kinfast import control
ts, qs, qds = control.simulate(robot.chain, q0, qd0, my_controller, dt=1e-3, steps=1000)

# collision: differentiable sphere distance, self and obstacles
model = robot.sphere_model({"l3": [(0, 0, 0, 0.05)]})
from kinfast.collision import collision_aware_ik   # reaches targets around obstacles

# trajectories: quintic and time-optimal synchronized trapezoid under URDF limits
t, q, qd, qdd, T = robot.point_to_point(q_start, q_goal)

# analysis: manipulability, condition number, joint-limit margin, workspace
from kinfast import analysis
w = analysis.manipulability(robot.chain, q, robot.link_id("ee"))

# frames: tf-style point transforms between any two links
p_world = robot.transform_points(points_in_gripper, q, from_link="panda_hand")
```

Collision-aware IK in one picture (`python examples/collision_aware_ik.py`):
plain IK reaches through the obstacle; kinfast bends around it to the same
target using gradients through FK and the distance field.

![collision-aware IK](examples/assets/collision_ik.png)

## Why trust the math

- FK cross-validated against an independent library on a real robot (above).
- Jacobians checked against float64 central differences, all 6 rows.
- Dynamics validated by energy conservation in free fall and by gravity
  matching the numerical gradient of potential energy.
- Controllers validated in closed loop: computed-torque tracks a quintic swing
  to <0.005 rad through the whole motion.
- Manipulability checked against the textbook 2R result w = |sin q2|.
- 73 tests, every module oracle-tested. `pytest tests` runs them all.

## Install

```bash
git clone <this repo> && cd kinfast
pip install -e ".[dev]"
pytest tests                       # 73 passed
python examples/gallery.py         # measure the gallery on your machine
python examples/demo_10k_arms.py --urdf examples/assets/panda.urdf --restarts 4
```

Requires Python 3.10+ and PyTorch. No ROS anywhere.

## Scope, honestly

kinfast is for robot learning, research, and prototyping: batched and
differentiable everything, easy ingestion, one coherent API. It is not a
physics simulator (use MuJoCo or Genesis), not a motion planner (use OMPL), and
not hard-real-time control. MJCF ingestion and GPU-tuned kernels are the next
milestones; the design doc and parking lot in `docs/` lay out the road.
