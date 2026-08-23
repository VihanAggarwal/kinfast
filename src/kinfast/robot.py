# src/kinfast/robot.py
"""Ergonomic surface: five-line load + fk/jacobian/ik."""
import difflib
import os
import torch
from kinfast.urdf.parse import parse_urdf_string, parse_urdf_file
from kinfast.urdf.repair import repair
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.jacobian import jacobian as _jacobian
from kinfast.ik import ik as _ik
from kinfast.analysis import sampling_bounds
from kinfast import dynamics as _dyn


def _name_hint(name, known, kind):
    """`unknown X 'foo'; did you mean 'bar'? known X are: ...` with the list
    trimmed so a 200-link humanoid does not print a wall of text."""
    known = list(known)
    close = difflib.get_close_matches(str(name), [str(k) for k in known], n=3,
                                      cutoff=0.6)
    msg = f"unknown {kind} {name!r}"
    if close:
        msg += "; did you mean " + " or ".join(repr(c) for c in close) + "?"
    shown = ", ".join(repr(k) for k in known[:12])
    if len(known) > 12:
        shown += f", ... (+{len(known) - 12} more)"
    return msg + f"; this robot's {kind}s are: {shown or '(none)'}"


class Robot:
    """Loaded robot. The dtype of the q you pass decides the working dtype of
    every method (fk, jacobian, ik, dynamics, collision): the compiled chain's
    constants are cast to it, so a robot compiled as float32 takes float64 q
    and returns float64 results. Velocities, accelerations, torques, and IK
    targets are coerced to the dtype of q (or of the target when q is absent).

    That cast changes the container, not the content. A float32 chain fed
    float64 q gives float64 tensors that still carry only ~7 correct digits,
    because the origins, axes and inertias were rounded when the chain was
    compiled. When you need real float64 accuracy, compile at float64:
    `kinfast.load(path, dtype=torch.float64)` or `robot.double()`, which
    rebuilds the constants from the parsed model."""

    def __init__(self, chain, ee_link=None, ir=None):
        self.chain = chain
        self.ir = ir                      # the parsed model, kept for export
        self.device = torch.device("cpu")
        if ee_link is not None and ee_link not in chain.link_index:
            raise ValueError(_name_hint(ee_link, chain.link_names, "link")
                             + " (ee_link must name one of them)")
        self.ee_link = ee_link or chain.link_names[-1]

    # ---- construction ----
    @classmethod
    def from_ir(cls, robot_ir, repair_model=True, ee_link=None,
                dtype=torch.float32):
        """Compile a parsed model. `dtype` is the precision the constants are
        compiled at (default float32); pass torch.float64 for a double chain."""
        if repair_model:
            robot_ir, _ = repair(robot_ir)
        return cls(compile_robot(robot_ir, dtype=dtype), ee_link=ee_link,
                   ir=robot_ir)

    def to(self, device=None, dtype=None):
        """Move to a device and/or recompile at a float precision, in place.

        `device` moves the compiled constants. `dtype` REBUILDS them from the
        parsed model, so switching to float64 recovers the digits a float32
        compile discarded; casting the existing tensors up could not. A
        torch.dtype may be passed positionally, mirroring Tensor.to. Returns
        self, so `robot.to("cuda", torch.float64)` chains."""
        if isinstance(device, torch.dtype):
            device, dtype = None, device
        if dtype is not None:
            if self.ir is None:
                raise ValueError(
                    "cannot change dtype: this Robot was built without its IR, "
                    "so the constants cannot be recompiled at a new precision. "
                    "Load it with kinfast.load(..., dtype=...) instead.")
            self.chain = compile_robot(self.ir, dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
        self.chain.to(self.device)
        return self

    def double(self):
        """Recompile the chain at float64, in place. Returns self."""
        return self.to(dtype=torch.float64)

    def float(self):
        """Recompile the chain at float32, in place. Returns self."""
        return self.to(dtype=torch.float32)

    # ---- properties ----
    @property
    def dtype(self):
        """The float dtype the chain's constants were compiled at."""
        return self.chain.dtype

    @property
    def dof(self):
        return self.chain.dof

    @property
    def n_links(self):
        return self.chain.n_links

    @property
    def parse_notes(self):
        """Notes the parser recorded about anything it had to reinterpret, such
        as an MJCF free joint pinned to the world. The notes live on the IR, so
        this forwards to `self.ir`; empty when there is no IR to ask."""
        return list(getattr(self.ir, "parse_notes", None) or [])

    @property
    def joint_names(self):
        """Movable joint names, ordered by their index in q."""
        return list(self.chain.joint_names)

    def q_index(self, joint_name):
        """Index into q for a named joint."""
        try:
            return self.chain.joint_names.index(joint_name)
        except ValueError:
            raise ValueError(
                _name_hint(joint_name, self.chain.joint_names, "movable joint")
                + " (fixed joints carry no q entry)") from None

    @property
    def lower(self):
        return self.chain.lower

    @property
    def upper(self):
        return self.chain.upper

    def link_id(self, name):
        """Row of `fk_all` for a named link."""
        try:
            return self.chain.link_index[name]
        except KeyError:
            raise KeyError(_name_hint(name, self.chain.link_names,
                                      "link")) from None

    @property
    def parse_notes(self):
        """What the parser approximated or dropped: free joints treated as
        fixed, bodies left at zero mass, settotalmass scaling. Empty list for
        a model that needed none."""
        return list(getattr(self.ir, "parse_notes", None) or [])

    # ---- kinematics ----
    def random_configs(self, n, dtype=None):
        """Uniform random configurations inside the joint limits, (n, dof).
        Revolute joints with infinite limits are sampled over a full turn;
        prismatic joints with infinite limits raise ValueError. dtype
        defaults to the chain's compiled dtype."""
        lo, hi = sampling_bounds(self.chain)
        if dtype is not None:
            lo, hi = lo.to(dtype), hi.to(dtype)
        return lo + (hi - lo) * torch.rand(n, self.dof, dtype=lo.dtype,
                                           device=self.device)

    def fk_all(self, q):
        return forward_kinematics(self.chain, q)

    def fk(self, q, link=None):
        idx = self.link_id(link) if link else self.link_id(self.ee_link)
        return self.fk_all(q)[:, idx]

    def jacobian(self, q, link=None):
        idx = self.link_id(link) if link else self.link_id(self.ee_link)
        return _jacobian(self.chain, q, idx)

    def ik(self, target, q0=None, link=None, **kw):
        idx = self.link_id(link) if link else self.link_id(self.ee_link)
        if q0 is None and kw.get("restarts", 1) <= 1:
            # seed in the target's dtype: the chain may be float32 while the
            # caller works in float64 (or the other way round)
            q0 = self.random_configs(target.shape[0], dtype=target.dtype)
        return _ik(self.chain, target, q0, idx, **kw)

    # ---- dynamics ----
    def mass_matrix(self, q):
        return _dyn.mass_matrix(self.chain, q)

    def gravity(self, q, g=None):
        """Generalized gravity torque. g defaults to the model's own gravity
        vector (MJCF `<option gravity>`, else (0, 0, -9.81))."""
        return _dyn.gravity(self.chain, q, g)

    def inverse_dynamics(self, q, qd, qdd, use_gravity=True):
        return _dyn.inverse_dynamics(self.chain, q, qd, qdd, use_gravity=use_gravity)

    def forward_dynamics(self, q, qd, tau, use_gravity=True):
        return _dyn.forward_dynamics(self.chain, q, qd, tau, use_gravity=use_gravity)

    # ---- export ----
    def to_mjcf(self, geometry=True, meshdir="", meshdir_strip=""):
        """Export this robot as an MJCF string (URDF -> MuJoCo). See
        kinfast.mjcf.emit for what is reproduced and how it is verified."""
        from kinfast.mjcf.emit import emit_mjcf
        if self.ir is None:
            raise ValueError("this Robot was built without its IR; load it with kinfast.load")
        return emit_mjcf(self.ir, geometry=geometry, meshdir=meshdir,
                         meshdir_strip=meshdir_strip)

    # ---- compiler ----
    def compile(self):
        """Generate robot-specific straight-line code for microsecond
        single-query FK / Jacobian / IK (the scalar backend). Returns a
        CompiledRobot; the batched torch path on this Robot is unaffected.

        The generated code always computes in Python floats (float64), but it
        can only be as accurate as the constants baked into it: compile the
        Robot at float64 (`kinfast.load(..., dtype=torch.float64)` or
        `robot.double()`) if you want float64 answers out of it."""
        from kinfast.codegen import CompiledRobot
        return CompiledRobot(self.chain)

    # ---- frames ----
    def transform_points(self, points, q, from_link, to_link="world"):
        """Express points given in `from_link`'s frame in `to_link`'s frame
        (or the world frame). points: (N,3) shared across the batch or (B,N,3),
        as a tensor or anything torch can make one from. Returns (B,N,3).

        Like every other method here, q's dtype is the working dtype: points
        are cast to it (and to q's device) rather than blowing up on a mixed
        float32/float64 matmul."""
        from kinfast import transforms as _T
        world = self.fk_all(q)                                  # (B,n,4,4)
        B = world.shape[0]
        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points)
        # .to is a no-op when it already matches and stays differentiable
        points = points.to(dtype=world.dtype, device=world.device)
        if points.dim() not in (2, 3) or points.shape[-1] != 3:
            raise ValueError("points must be (N,3) or (B,N,3), got shape "
                             f"{tuple(points.shape)}")
        if points.dim() == 2:
            points = points.unsqueeze(0).expand(B, -1, -1)
        ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype,
                          device=points.device)
        ph = torch.cat([points, ones], dim=-1)                  # (B,N,4)
        M = world[:, self.link_id(from_link)]                   # (B,4,4)
        if to_link != "world":
            M = _T.invert_transform(world[:, self.link_id(to_link)]) @ M
        return torch.einsum("bij,bnj->bni", M, ph)[..., :3]

    # ---- trajectory ----
    def point_to_point(self, q0, qf, amax=None, n=100):
        """Time-optimal synchronized trapezoidal move under the URDF's velocity
        limits (joints with no declared limit fall back to 1 rad/s)."""
        from kinfast.trajectory import trapezoidal
        vmax = self.chain.vmax.to(device=q0.device, dtype=q0.dtype).clone()
        vmax[vmax <= 0] = 1.0
        if amax is None:
            amax = torch.full_like(vmax, 4.0)
        else:
            amax = torch.as_tensor(amax, dtype=q0.dtype, device=q0.device)
        return trapezoidal(q0, qf.to(device=q0.device, dtype=q0.dtype),
                           vmax, amax, n=n)

    # ---- collision ----
    def sphere_model(self, spheres):
        """Build a collision SphereModel. spheres: {link_name_or_index: [(x,y,z,r)]}."""
        from kinfast.collision import SphereModel
        resolved = {(self.link_id(k) if isinstance(k, str) else k): v
                    for k, v in spheres.items()}
        return SphereModel(self.chain, resolved)


def _sniff(text):
    """URDF or MJCF? Decide by the XML root element, not the file extension."""
    head = text.lstrip()[:200].lower()
    if "<mujoco" in head:
        return "mjcf"
    return "urdf"


def _is_xacro(path, text):
    return path.lower().endswith(".xacro") or "xmlns:xacro" in text[:2000]


def _expand_xacro(path, mappings=None):
    """Expand a xacro file with the standalone `xacro` package (no ROS
    needed). $(find pkg) lookups need ROS package paths and will fail here;
    pass property overrides via `mappings` like the xacro CLI's name:=value."""
    try:
        import xacro
    except ImportError as e:
        raise ImportError("this robot is a xacro file; install the expander with "
                          "`pip install xacro` (works without ROS)") from e
    doc = xacro.process_file(path, mappings=mappings or {})
    return doc.toprettyxml(indent="  ")


def load(path, ee_link=None, mappings=None, dtype=torch.float32):
    """Load a robot from URDF, xacro, or MJCF (format auto-detected).
    `mappings` are xacro property overrides, e.g. {"prefix": "left_"}.
    `dtype` is the precision the chain's constants are compiled at: float32
    by default, torch.float64 when you need double-precision kinematics.
    MJCF <include file=...> paths resolve against this file's directory."""
    if not os.path.isfile(path):
        if os.path.isdir(path):
            raise IsADirectoryError(
                f"{path!r} is a directory, not a robot file; pass the .urdf, "
                ".xacro, or MJCF .xml file inside it")
        raise FileNotFoundError(
            f"no robot file at {path!r} (looked at {os.path.abspath(path)}); "
            "pass a path to a .urdf, .xacro, or MJCF .xml file, or use "
            "kinfast.load_string for text you already have")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if _is_xacro(path, text):
        text = _expand_xacro(path, mappings)
    return load_string(text, ee_link=ee_link, dtype=dtype,
                       base_dir=os.path.dirname(os.path.abspath(path)))


def load_string(text, ee_link=None, dtype=torch.float32, base_dir=None):
    """Same as `load`, from URDF/MJCF text already in memory. `base_dir` is the
    directory MJCF <include> paths resolve against; kinfast.load fills it in
    from the file's path."""
    if _sniff(text) == "mjcf":
        from kinfast.mjcf.parse import parse_mjcf_string
        return Robot.from_ir(parse_mjcf_string(text, base_dir=base_dir),
                             ee_link=ee_link, dtype=dtype)
    return Robot.from_ir(parse_urdf_string(text), ee_link=ee_link, dtype=dtype)
