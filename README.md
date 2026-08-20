# kinfast

**Load any robot, then run it 10,000× in parallel on your GPU — differentiable, in five lines.**

```python
import kinfast
robot    = kinfast.load("panda.urdf")            # just works, auto-repaired, no ROS
q        = robot.random_configs(10_000)          # 10k configs on GPU
ee       = robot.fk(q)                            # batched forward kinematics
q_solved, info = robot.ik(ee, pos_only=True)      # batched, differentiable IK
```

- **Painless ingestion** — throw any URDF at it; kinematics-relevant defects are auto-repaired.
- **Batched + differentiable** — FK, Jacobians, and IK over thousands of configs at once, autograd end-to-end.
- **No ROS required.**

## Install
```bash
pip install -e ".[dev]"
```

## Demo
```bash
python examples/demo_10k_arms.py --urdf examples/assets/panda.urdf --n 10000
```
