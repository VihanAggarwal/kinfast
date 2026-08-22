# src/kinfast/robot.py
"""Ergonomic surface: five-line load + fk/jacobian/ik."""
import torch
from kinfast.urdf.parse import parse_urdf_string, parse_urdf_file
from kinfast.urdf.repair import repair
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.jacobian import jacobian as _jacobian
from kinfast.ik import ik as _ik
from kinfast import dynamics as _dyn


class Robot:
    def __init__(self, chain, ee_link=None):
        self.chain = chain
        self.device = torch.device("cpu")
        self.ee_link = ee_link or chain.link_names[-1]

    # ---- construction ----
    @classmethod
    def from_ir(cls, robot_ir, repair_model=True, ee_link=None):
        if repair_model:
            robot_ir, _ = repair(robot_ir)
        return cls(compile_robot(robot_ir), ee_link=ee_link)

    def to(self, device):
        self.device = torch.device(device)
        self.chain.to(self.device)
        return self

    # ---- properties ----
    @property
    def dof(self):
        return self.chain.dof

    @property
    def n_links(self):
        return self.chain.n_links

    @property
    def joint_names(self):
        """Movable joint names, ordered by their index in q."""
        return list(self.chain.joint_names)

    def q_index(self, joint_name):
        """Index into q for a named joint."""
        return self.chain.joint_names.index(joint_name)

    @property
    def lower(self):
        return self.chain.lower

    @property
    def upper(self):
        return self.chain.upper

    def link_id(self, name):
        return self.chain.link_index[name]

    # ---- kinematics ----
    def random_configs(self, n):
        lo, hi = self.chain.lower, self.chain.upper
        return lo + (hi - lo) * torch.rand(n, self.dof, device=self.device)

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
            q0 = self.random_configs(target.shape[0])
        return _ik(self.chain, target, q0, idx, **kw)

    # ---- dynamics ----
    def mass_matrix(self, q):
        return _dyn.mass_matrix(self.chain, q)

    def gravity(self, q):
        return _dyn.gravity(self.chain, q)

    def inverse_dynamics(self, q, qd, qdd, use_gravity=True):
        return _dyn.inverse_dynamics(self.chain, q, qd, qdd, use_gravity=use_gravity)

    def forward_dynamics(self, q, qd, tau, use_gravity=True):
        return _dyn.forward_dynamics(self.chain, q, qd, tau, use_gravity=use_gravity)

    # ---- compiler ----
    def compile(self):
        """Generate robot-specific straight-line code for microsecond
        single-query FK / Jacobian / IK (the scalar backend). Returns a
        CompiledRobot; the batched torch path on this Robot is unaffected."""
        from kinfast.codegen import CompiledRobot
        return CompiledRobot(self.chain)

    # ---- frames ----
    def transform_points(self, points, q, from_link, to_link="world"):
        """Express points given in `from_link`'s frame in `to_link`'s frame
        (or the world frame). points: (N,3) shared across the batch or (B,N,3).
        Returns (B,N,3)."""
        from kinfast import transforms as _T
        world = self.fk_all(q)                                  # (B,n,4,4)
        B = world.shape[0]
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
        vmax = self.chain.vmax.clone()
        vmax[vmax <= 0] = 1.0
        if amax is None:
            amax = torch.full_like(vmax, 4.0)
        return trapezoidal(q0, qf, vmax, amax, n=n)

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


def load(path, ee_link=None, mappings=None):
    """Load a robot from URDF, xacro, or MJCF (format auto-detected).
    `mappings` are xacro property overrides, e.g. {"prefix": "left_"}."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if _is_xacro(path, text):
        text = _expand_xacro(path, mappings)
    return load_string(text, ee_link=ee_link)


def load_string(text, ee_link=None):
    if _sniff(text) == "mjcf":
        from kinfast.mjcf.parse import parse_mjcf_string
        return Robot.from_ir(parse_mjcf_string(text), ee_link=ee_link)
    return Robot.from_ir(parse_urdf_string(text), ee_link=ee_link)
