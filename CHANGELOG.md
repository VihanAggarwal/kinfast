# Changelog

Notable changes, newest first. Dates are when the work landed on main.

## Unreleased

### Added
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
