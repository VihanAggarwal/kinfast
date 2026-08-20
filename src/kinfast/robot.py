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


def load(path, ee_link=None):
    return Robot.from_ir(parse_urdf_file(path), ee_link=ee_link)


def load_string(text, ee_link=None):
    return Robot.from_ir(parse_urdf_string(text), ee_link=ee_link)
