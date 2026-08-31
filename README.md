# kinfast

[![tests](https://github.com/VihanAggarwal/kinfast/actions/workflows/tests.yml/badge.svg)](https://github.com/VihanAggarwal/kinfast/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VihanAggarwal/kinfast/blob/main/examples/kinfast_quickstart.ipynb)

Batched, differentiable robot kinematics and dynamics in PyTorch. Loads URDF,
xacro, MJCF and SDF without ROS, runs thousands of configurations in one call,
and compiles a robot to straight-line code when you need a single query fast.

## Start here

```bash
pip install -e ".[dev]"
```

Python 3.10+, PyTorch 2.x. Nothing from ROS is in the dependency tree.

```python
import kinfast

robot = kinfast.load("panda.urdf")     # URDF, xacro, MJCF or SDF, auto-detected
print(robot.summary())                 # links, joints, limits, and any repairs

q = robot.random_configs(10_000)       # (10000, dof), inside the joint limits
poses = robot.fk(q)                    # (10000, 4, 4), differentiable
q_sol, info = robot.ik(poses, restarts=8)
```

That is the whole idea. You hand it a batch and it hands one back. If your
robot description does not load, that is a bug and I would like the file.

Three more lines that cover most of what people ask for next:

```python
tau = robot.inverse_dynamics(q, qd, qdd)          # also mass_matrix, gravity
fast = robot.compile()                            # ~15 us FK for a control loop
kinfast.to_mjcf("panda.urdf", out="panda.xml")    # URDF to MuJoCo, verified
```

There is a [quickstart notebook](examples/kinfast_quickstart.ipynb) that runs
in Colab with nothing installed locally, and `python -m kinfast.studio` opens a
desktop window with the arm, a slider per joint, and benchmarks you can run on
your own machine.

## Why it exists

Getting an arbitrary URDF into anything has been the worst part of every
robotics project I have worked on. The fast options are a fight to set up and
the simple ones are narrow. So the loader is the part I care about most, and
the batching is what makes the rest worth using from Python.

## Accuracy

Everything below is checked against something that is not this library.

FK is cross-checked against
[pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) on the
real Franka Panda: over 512 random configurations the largest position
difference is 1.3e-7 m and the largest rotation difference 3.0e-7, which is
float32 rounding. See `tests/test_cross_validation.py`.

MJCF models are compared body by body against MuJoCo's own forward kinematics
at random configurations. The test models are picked to hit the format's
traps: angles are degrees by default, euler is intrinsic xyz where URDF is
extrinsic, quaternions are wxyz, joint `pos` is a rotation anchor with no URDF
equivalent, and `<default>` classes inherit down the body tree.

Dynamics is compared against MuJoCo's `mj_fullM` and `qfrc_bias` at random
states, on a hand-built arm with rotated off-center inertial frames and on the
real UR5e. That test is why both parsers rotate inertia tensors out of their
inertial frames, and why the repair pass flags inertias that violate the
triangle inequality.

The converter is verified in reverse: the emitted MuJoCo model is loaded by
MuJoCo and its FK compared against kinfast's FK of the source URDF.

Also checked against independent oracles rather than against the library
itself: Jacobians against float64 central differences on all six rows, gravity
torque against the numerical gradient of potential energy, energy conservation
in free fall, controllers by whether they track in closed loop, and
manipulability against the textbook 2R result. 992 tests.

At scale, 19 of 20 production models from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) parse,
including the 29-dof Unitree G1, ANYmal C, Spot, the 24-dof Shadow Hand,
ALOHA and Stretch. The one failure is Cassie, whose ball joints raise a clear
error rather than loading wrong.

## Robots that load

Thirteen unmodified URDFs pulled straight from public repos. All load, and
batched IK round-trips at 100% on each (`python examples/gallery.py --fetch`,
full results in `examples/assets/GALLERY.md`):

| robot | dof | FK per config (CPU) | IK round trip, <5cm |
|---|---|---|---|
| Franka Panda | 8 | 5.5 us | 100% |
| KUKA iiwa | 7 | 4.8 us | 100% |
| UR5 | 6 | 4.1 us | 100% |
| xArm6 | 6 | 4.0 us | 100% |
| SO-101 (the LeRobot arm) | 6 | 8.7 us | 100% |
| SO-100 | 6 | 8.2 us | 100% |
| Unitree A1 | 12 | 6.7 us | 100% |
| Laikago | 12 | 5.1 us | 100% |
| Minitaur | 16 | 9.4 us | 100% |
| Husky, Racecar, R2D2, cartpole | | | 100% |

Some are messier than they look. The Husky URDF ships with unexpanded ROS
substitution args like `$(optenv HUSKY_IMU_XYZ 0.19 0 0.149)` sitting in
numeric fields. The Panda declares its second finger as a `<mimic>` of the
first, so it has eight actuated joints and not nine. Bad inertias, inverted
limits, unnormalized axes and missing limit tags are repaired at load, and
`robot.summary()` prints what changed.

If you have an SO-101 there is a walkthrough for that arm in
[docs/SO101_TUTORIAL.md](docs/SO101_TUTORIAL.md).

## Speed

Two regimes with different answers.

Batched (real Panda, CPU, median of 7, `python examples/benchmark.py`):

| batch | kinfast FK | pytorch_kinematics FK | kinfast Jacobian | pk Jacobian |
|---|---|---|---|---|
| 1 | 0.50 ms | 0.38 ms | 0.72 ms | 0.47 ms |
| 100 | 1.21 ms | 1.03 ms | 1.65 ms | 1.54 ms |
| 10,000 | 17.2 ms | 12.2 ms | 19.1 ms | 15.2 ms |

Read that table carefully, because the two columns are not doing the same
work. kinfast returns all 13 link frames as a (B, 13, 4, 4) tensor; pk is
measured on its fastest end-effector-only path, and about a third of the
kinfast time at batch 10k is materializing that output. The internal path that
IK and Jacobians use skips the assembly and runs about 10 ms at batch 10k,
under pk's ee-only time while carrying every link. That is why batched IK is
the strong suit: 10,000 position targets with 4 restarts each solve in about
7 s on a laptop CPU. cuRobo's CUDA kernels beat all of this on raw throughput,
so the claim is competitive, not fastest.

Single query is the more interesting case. A control loop asking for one FK
pays roughly 400 us of framework overhead for about 200 flops of real math, in
any tensor library. `robot.compile()` removes the framework by generating
straight-line code for your specific robot at load time, with the tree
unrolled, constant origins folded in, and every multiply by zero from an
axis-aligned joint deleted during generation. Measured on the Panda, public
API on both sides:

| op, one query, CPU | compiled | torch path |
|---|---|---|
| FK, all 13 frames | 15 us | 680 us |
| geometric Jacobian | 15 us | 1160 us |

That is 46x on FK and 80x on the Jacobian, and it moves a controller's
kinematics tick from about 540 Hz to about 34 kHz, so a 1 kHz loop spends
around 3% of its budget on kinematics.

One caveat worth stating, since it is easy to overstate this number. The
compiled call returns assembled 4x4 transforms, the same thing the tensor path
returns. There is a lower-level entry point that hands back a flat list of
floats and takes about 5 us, but comparing that against an assembled tensor
would be measuring different work, so the table above does not. The generated
source is a plain Python function you can read (`fast.source`), and it is
tested against the batched path on all thirteen gallery robots. Per-robot
codegen is not new, Pinocchio has had CppADCodeGen paths for years. This
version just needs nothing beyond `pip install`.

Numbers above are from a laptop CPU. For CUDA, `python examples/gpu_benchmark.py`
writes `examples/assets/BENCHMARK_GPU.md` using the same methodology, and
`pytest tests/test_gpu.py` checks that every module gives the same answers on
the GPU as on the CPU.

## The rest of it

```python
t, qt, qdt, qddt, T = robot.point_to_point(a, b)   # limit-safe trapezoid
t, q, qd, qdd, info = robot.time_path(path)        # time-optimal along a path

from kinfast import control
ts, qs, qds = control.simulate(robot.chain, q0, qd0, controller, dt=1e-3, steps=1000)

from kinfast import analysis
ws = analysis.workspace(robot.chain, robot.link_id(robot.ee_link))

from kinfast.planning import CollisionChecker, rrt_connect
from kinfast.collision_world import Sphere
checker = CollisionChecker(robot, robot.sphere_model({"panda_hand": [(0, 0, 0, 0.05)]}),
                           world=[Sphere(center=[0.4, 0.0, 0.5], radius=0.15)])
plan = rrt_connect(robot.chain, q_start, q_goal, checker)
t, q, qd, qdd, T = plan.to_trajectory(robot)
```

The planner is RRT-Connect with shortcut smoothing, built the way the rest of
the library is: an edge between two configurations is interpolated into a
tensor and checked in one batched collision call rather than one call per step
along it. A typical SO-101 solve checks about eight thousand configurations in
roughly a hundred and fifty calls.

The collision-aware IK demo is worth a look
(`python examples/collision_aware_ik.py`). Plain IK reaches straight through an
obstacle; the gradient version bends the arm around it to the same target,
because both the FK and the distance field are differentiable.

![collision-aware IK](examples/assets/collision_ik.png)

## Looking at a robot

```bash
python -m kinfast.studio                     # picks a robot it can find
python -m kinfast.studio --robot so101
python -m kinfast.studio --list              # what is available
python -m kinfast.studio --robot panda --save studio.png   # headless
```

A window with the arm on the left and a slider per joint. The benchmark panels
on the right stay empty until you press "run benchmarks", because those
numbers are measured on your machine when you ask, not read from a table.
"solve ik" picks a reachable pose, throws it away and asks the library to find
it again. "plan around" drops an obstacle on the straight line to a random
goal and walks the arm along a path that avoids it. "workspace" samples where
the arm can reach.

Needs matplotlib, which the library itself does not.

## What it is not

Not a physics simulator, so there is no contact. Use MuJoCo or Genesis for
that, and use this underneath for the kinematics and rigid-body dynamics.
Not hard real-time.

MJCF support covers bodies, joints (hinge and slide, including anchors and
stacked joints), inertials and defaults classes. Ball and free joints are not
supported; a free base loads as fixed with a note. Dynamics needs inertial
tags in the model to mean anything.

Joints driven by other joints are handled: `<mimic>` reduces the model to the
coordinates it actually has, chains of mimics compose, and the relation
survives conversion to MJCF as an equality constraint. Note that MuJoCo and
PyBullet both ignore the tag on URDF import, so if you are comparing against
either, use `chain.expand_q` to get one value per joint.
