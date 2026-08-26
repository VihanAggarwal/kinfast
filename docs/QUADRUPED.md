# Standing a quadruped up: four feet, one solve

`examples/go2_feet.py` takes the Unitree Go2 from the MuJoCo Menagerie, works
out where its four feet are, and solves for the joint angles that put those
feet on the ground at a series of body heights. It prints how many of the
targets it reached and saves a picture of the result.

```
python examples/go2_feet.py --mjcf examples/assets/menagerie/unitree_go2/go2.xml
```

The Go2 model is not shipped with kinfast. Fetch it with
`python examples/menagerie.py --fetch`, or point `--mjcf` at your own copy. If
the file is not there the script says so and stops, rather than failing halfway
through with a stack trace.

## Why a quadruped is not four arms

An arm has one end effector, so inverse kinematics has one target and the whole
robot is free to chase it. A quadruped has four feet that share a single
configuration vector, and a stance is a statement about all four at once: the
front left foot goes here, and the rear right foot goes there, and both have to
be true of the same posture.

The honest way to write that is as one task with twelve rows, three position
rows per foot, over the twelve leg joints. That is what the example builds. The
step it takes is the same damped least squares step the rest of kinfast uses,

```
dq = J^T (J J^T + lambda^2 I)^-1 e
```

only with `J` stacked from the four foot Jacobians and `e` stacked from the
four foot errors, and the result clamped back inside the joint limits.

On this robot the legs happen to be kinematically independent: no joint in the
front left leg moves the rear right foot, so nine of the twelve columns of each
foot's Jacobian are exactly zero, and the stacked solve decouples into four
three-by-three problems on its own. There is a test that checks that zero
structure. The point of writing it as one task anyway is that it keeps working
when the legs stop being independent, which is what happens the moment you add
the floating base back, or ask for a trunk pose as well, or put a weight on one
foot and not another. The twelve-row form is the one that generalizes.

## The four steps the script takes

**Load the model.** The Go2's trunk carries a `<freejoint>`. kinfast has no
floating base, so it pins that joint and records the fact in
`robot.parse_notes`, which the script prints. That is the right approximation
for a stance: with the trunk held still, foot positions in the trunk frame are
exactly the standing problem, and the floating base only decides where the
whole animal ends up in the world.

**Find the feet.** The feet are the leaf links of the tree, the ones with no
child link. Each is labelled FL, FR, RL or RR from the sign of its leg's
attachment point in the trunk frame, so the labelling comes from geometry and
not from whatever the model author called things. Four leaves in four
quadrants is the check that this really is a quadruped; anything else gets a
clear error instead of a wrong answer.

**Find the contact point inside each foot link.** This is the part that is easy
to get wrong. On the Go2 the last link in each leg is the calf, whose frame
origin sits at the knee. The actual foot is a sphere of radius 22 mm hanging
213 mm below that origin, declared as a geom in the MJCF. kinfast's IR keeps
kinematics and inertia, not collision geometry, so the script reads those
spheres back out of the XML itself, resolving MuJoCo's nested `<default>`
classes the same way the parser resolves them for joints. Skip that step and
you solve for the knee, and every stance comes out 21 cm too low.

The contact point being offset inside the link is also why the example does not
just call `robot.ik`. Library IK drives a link frame to a pose. Here the thing
being driven is a point rigidly attached to the link but not at its origin, so
each foot Jacobian is the link's geometric Jacobian corrected for that offset:

```
J_point = Jv - skew(R r) Jw
```

with `r` the contact offset in the link frame. The example computes that and
checks it against float64 central differences in the tests.

**Solve and report.** Targets are placed under each hip: x and y from the
neutral pose, where the leg hangs straight down, and z at `-(height - radius)`
so the sphere touches the ground plane instead of sinking into it. Every height
is one row of a batch, and the whole batch is solved together. The script
prints a table of how many feet reached their target at each height, and writes
a two panel figure: a side view of the solved leg postures, and a top view of
the support polygon.

## What the output looks like

```
loaded go2 from examples/assets/menagerie/unitree_go2/go2.xml
  12 dof, 14 links, trunk frame 'base'
  parser note: body base: freejoint treated as fixed base
  FL foot: link 'FL_calf', contact at (-0.0020, +0.0000, -0.2130) m, radius 0.022 m
  ...
  solved 6 stances x 4 feet in one batch: 80 iterations, 1 seed(s), 418.5 ms

  height   feet solved   worst foot error
  --------------------------------------------
  0.180 m   4/4             0.000 mm
  0.220 m   4/4             0.000 mm
  0.260 m   4/4             0.000 mm
  0.300 m   4/4             0.000 mm
  0.340 m   4/4             0.000 mm
  0.380 m   4/4             0.000 mm
  --------------------------------------------
  solve rate 100.0% over 6 heights x 4 feet
```

"0.000 mm" is not rounding down a near miss: in float64 the residual is around
1e-16 m. The Go2's reachable band with this stance runs from 0.12 m to 0.41 m
of body height, set by the 0.213 m thigh and calf and the knee range of -2.72
to -0.84 radians. Ask for something outside it, say `--heights 0.45`, and the
solve rate drops and the worst error column tells you by how much: 66 mm short,
in that case. That is the intended behaviour, an unreachable target is
reported, never faked.

## Useful options

| flag | what it does |
|---|---|
| `--heights 0.2 0.3 0.4` | body heights to solve for, one batch row each |
| `--dx`, `--dy` | shift every foot outward, for a longer or wider stance |
| `--restarts 4` | extra random seeds per stance, solved in the same batch, best kept |
| `--iters` | damped least squares iterations, 80 by default |
| `--float32` | work in float32 instead of float64 |
| `--device cuda` | run the whole batch on a GPU |
| `--out stance.png` | figure path, or `--out ""` to skip plotting |

## Using the pieces yourself

Everything below the CLI is importable and has no opinion about how you got
there:

```python
import kinfast
from go2_feet import find_feet, stance_targets, solve_stance, foot_points

robot = kinfast.load("go2.xml", dtype=torch.float64)
feet = find_feet(robot, mjcf_path="go2.xml")
targets = stance_targets(robot, feet, [0.20, 0.28, 0.34])   # (B, 4, 3)
q, info = solve_stance(robot, feet, targets)                # (B, 12)
print(info["solve_rate"], foot_points(robot, q, feet).shape)
```

Three properties are worth knowing about, because they are what make this
usable inside something larger rather than only as a demo.

**It is batched.** The leading dimension is a stack of independent stances. On
one CPU core, 80 iterations over six heights take about 0.8 s and over 256
heights about 1.2 s: forty times the work for half again the time, because the
per-iteration cost is a handful of batched matrix operations and the Python
loop overhead is paid once for the whole batch rather than once per stance.
Nothing in the module assumes a CPU either, so `--device cuda` moves the same
batch to a GPU.

**It follows your dtype and device.** The working precision is the precision of
the seed you pass, or of the targets when you pass no seed, exactly like the
rest of the library. A robot compiled in float32 will answer a float64 query in
float64; for genuinely float64 answers load it with `dtype=torch.float64` so
the constants themselves carry the digits.

**It is differentiable.** The solve loop is plain tensor math with no in-place
tricks, so gradients flow from the returned joint angles back to the foot
targets and the seed. You can put a stance solve inside a larger objective, ask
for the derivative of the joint angles with respect to where you put the feet,
and get an answer that agrees with central differences through the whole solve.
There is a test that checks exactly that.

## Testing

`tests/test_go2_feet.py` covers all of it. The Go2 asset is gitignored, so the
oracle-backed tests run on a small quadruped MJCF written out in the test file,
built to exercise the same features: a free-jointed trunk, twelve joints, and
foot spheres declared through nested default classes. MuJoCo is the independent
oracle. It is asked where the foot spheres ended up, both for random
configurations (does the forward kinematics agree) and for the joint angles the
solver returned (did the IK actually stand the robot up). The foot Jacobians
are checked against float64 central differences, and the offsets read out of
the XML against the literal numbers in the fixture. The two tests that need the
real Go2 run when the model is present and skip cleanly when it is not.
