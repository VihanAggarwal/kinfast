# Changelog

Notable changes, newest first. Dates are when the work landed on main.

## Unreleased

### Added
- Joints driven by other joints are honoured (`<mimic>`). A parallel gripper
  now reports the one degree of freedom it has rather than two, and sampling,
  IK and planning can no longer explore states the hardware cannot reach. The
  relation is folded into the joint as a scale and offset on the driving
  joint's coordinate, so it costs no degree of freedom and every algorithm that
  indexes joints keeps working. Chains of mimics compose, and cycles or
  references to a joint that does not exist are refused.
- `chain.expand_q` and `chain.movable_joint_names` map the actuated vector onto
  every joint, for talking to tools that ignore `<mimic>` on URDF import, which
  includes MuJoCo and PyBullet.
- `chain.has_mimic`, and `kinfast.ir.geometries` for reading a link's geometry
  slot without caring whether it holds one shape or several.

### Fixed
- Every `<collision>` and `<visual>` element of a link is kept. Only the first
  survived before, so a link written as several boxes, which is how real files
  describe a base plate or a gripper pad, was collision checked as one of them
  and a planner would return paths through the rest.
- The mass matrix, gravity torque, RNEA and the geometric and COM Jacobians
  were wrong on a chain with a driven joint: they wrote into a joint's column
  instead of adding to it and ignored the coupling factor. All now agree with
  the gradient of potential energy and with each other.
- Emitted MJCF carries the coupling as an equality constraint instead of
  silently dropping it, which had produced a file describing a robot with more
  degrees of freedom than the source.
- The parser records what it repaired instead of doing it silently: invented
  limits for a joint that declared none, inverted limits it swapped, zero width
  limits, duplicate link names, and links carrying several shapes.
- A file declaring two joints with the same name no longer collapses them.

- Time optimal timing along a path (`kinfast.topp`, or `robot.time_path`).
  A planner gives a shape with no speed attached; this finds the fastest
  traverse of that shape inside the joint limits, by reducing the per joint
  constraints to bounds on a single path coordinate and running the standard
  forward backward pass. On a dense path it is 20 to 27 percent quicker than
  running a trapezoid between every pair of waypoints, because it does not
  brake to a halt at points that were never corners. On a sparse path there is
  nothing to win and it returns the same answer. Corners are handled honestly:
  either stop at them, or round them by a bounded amount that is reported.
- Path planning (`kinfast.planning`): RRT-Connect with shortcut smoothing,
  working in configuration space and checking a whole edge in one batched
  collision call. `plan.to_trajectory(robot)` times the result against the
  model's own velocity limits.
- Frame velocities (`kinfast.velocity`): `twist`, `acceleration` and
  `link_velocities`. The Jdot qd term comes from autograd of the Jacobian
  rather than a hand derivation, so it cannot drift from the kinematics.
- A desktop window, `python -m kinfast.studio`: the robot with a slider per
  joint, planning around an obstacle, and three benchmark panels that are
  measured on your machine when you press the button rather than quoted.
- O(n) dynamics (`kinfast.dynamics_rnea`): recursive Newton-Euler and the
  composite rigid body mass matrix.
- Operational space control (`kinfast.control_task`).
- Splines and Cartesian straight lines (`kinfast.trajectory_spline`).
- Collision: spheres derived from the model's own primitives
  (`kinfast.collision_auto`), distances to planes, boxes and capsules
  (`kinfast.collision_world`), and an allowed collision matrix (`kinfast.acm`).
- Floating bases (`kinfast.floating`), cyclic coordinate descent IK
  (`kinfast.ik_ccd`), weighted task space IK with a nullspace posture
  (`kinfast.ik_task`).
- Analysis: whole body centre of mass (`kinfast.com`), manipulability
  ellipsoids (`kinfast.analysis_ext`), voxelised reachability
  (`kinfast.reachability`), mass properties from geometry (`kinfast.inertia`),
  configuration space metrics (`kinfast.config`), pruned forward kinematics
  (`kinfast.fk_subset`), chunked batching (`kinfast.batching`).
- Reports: a structural linter (`kinfast.lint`) and a plain text model summary
  (`kinfast.summary`).
- A registry of public robot descriptions with a local cache (`kinfast.zoo`).
- Generated API reference, `docs/API.md`, checked by a test so it cannot go
  stale.
- GPU notebook and instructions: `examples/kinfast_gpu_colab.ipynb` and
  `docs/GPU.md`. The benchmark grows the batch until the card fills and
  reports peak memory.
- Convenience methods on `Robot` for the modules above, so they are reachable
  without importing each one.

### Fixed
- `coriolis` cut the autograd graph, so gradients through `forward_dynamics`
  and `control.simulate` were wrong.
- `simulate` labelled every record with the time of the previous state.
- The Jacobian rejected a float64 q on a float32 chain, and returned zeros for
  a negative link index instead of the link Python would have indexed.
- Infinite joint limits produced NaN in sampling, workspace and limit margin.
- MJCF: joint `ref` was ignored, `<option gravity>` was ignored, and
  `<include>` was dropped, so a Menagerie `scene.xml` silently loaded as a
  robot with no joints.
- The MJCF emitter produced a model MuJoCo would not load when the URDF root
  link was called `world`, and dropped the inertia of zero mass links.
- Both parsers ignored the rotation of an inertial frame, so any model with a
  rotated inertia tensor had the wrong dynamics.
- `load` had no dtype option, so a float64 q on a loaded robot returned float64
  tensors carrying float32 precision.
- The constants cache could serve stale values after a chain was edited.
- The API reference generator produced different text on different Python
  versions, because `Optional[X]` and `Union[A, B]` are rendered differently
  before and after 3.13.

### Changed
- Batched IK is roughly four times faster: rotations and positions are carried
  separately instead of as homogeneous matrices in the solve loop, and forward
  kinematics is evaluated once per iteration rather than twice.
