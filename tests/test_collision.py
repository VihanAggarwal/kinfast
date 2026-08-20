# tests/test_collision.py
"""Sphere-based collision distance: hand-computed distances, adjacency masking,
collision detection, obstacles, and differentiability."""
import math
import torch
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from kinfast.collision import SphereModel, self_distance, distance_to_obstacles
from tests.test_parse import TWO_LINK


def _chain():
    return compile_robot(parse_urdf_string(TWO_LINK), dtype=torch.float64)


def test_centers_world_follow_fk():
    chain = _chain()
    # sphere at l2 origin; at zero config l2 frame sits at (1,0,0)
    model = SphereModel(chain, {chain.link_index["l2"]: [(0.0, 0.0, 0.0, 0.1)]})
    C = model.centers_world(torch.zeros(1, 2, dtype=torch.float64))
    assert torch.allclose(C[0, 0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-9)


def test_self_distance_hand_computed():
    chain = _chain()
    b, l2 = chain.link_index["base"], chain.link_index["l2"]
    # base sphere at origin r=0.2; l2 sphere at l2 origin r=0.3.
    # zero config: l2 at (1,0,0) -> distance = 1 - 0.2 - 0.3 = 0.5.
    model = SphereModel(chain, {b: [(0, 0, 0, 0.2)], l2: [(0, 0, 0, 0.3)]})
    d = self_distance(model, torch.zeros(1, 2, dtype=torch.float64))
    assert torch.allclose(d, torch.tensor([0.5], dtype=torch.float64), atol=1e-9)


def test_adjacency_is_excluded():
    chain = _chain()
    b, l1, l2 = (chain.link_index["base"], chain.link_index["l1"], chain.link_index["l2"])
    # l1 sphere (adjacent to base and l2) deliberately overlaps base, but base-l1
    # and l1-l2 are adjacent pairs and must be ignored; the reported min stays the
    # non-adjacent base<->l2 distance of 0.5.
    model = SphereModel(chain, {
        b: [(0, 0, 0, 0.2)],
        l1: [(0.0, 0, 0, 0.9)],   # would collide with base if not excluded
        l2: [(0, 0, 0, 0.3)],
    })
    d = self_distance(model, torch.zeros(1, 2, dtype=torch.float64))
    assert torch.allclose(d, torch.tensor([0.5], dtype=torch.float64), atol=1e-9)


def test_detects_self_collision():
    chain = _chain()
    b, l2 = chain.link_index["base"], chain.link_index["l2"]
    # big base sphere: 1 - 0.9 - 0.3 = -0.2 -> collision
    model = SphereModel(chain, {b: [(0, 0, 0, 0.9)], l2: [(0, 0, 0, 0.3)]})
    d = self_distance(model, torch.zeros(1, 2, dtype=torch.float64))
    assert d.item() < 0


def test_self_distance_batched_config_dependent():
    chain = _chain()
    b, l2 = chain.link_index["base"], chain.link_index["l2"]
    model = SphereModel(chain, {b: [(0, 0, 0, 0.2)], l2: [(0, 0, 0, 0.2)]})
    # j1=0 -> l2 at (1,0,0); j1=pi -> l2 at (-1,0,0). Both distance 1-0.4=0.6.
    q = torch.tensor([[0.0, 0.0], [math.pi, 0.0]], dtype=torch.float64)
    d = self_distance(model, q)
    assert torch.allclose(d, torch.tensor([0.6, 0.6], dtype=torch.float64), atol=1e-9)


def test_obstacle_distance_and_grad():
    chain = _chain()
    l2 = chain.link_index["l2"]
    model = SphereModel(chain, {l2: [(0.0, 0.0, 0.0, 0.1)]})
    obs_c = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float64)  # near l2 at q=0
    obs_r = torch.tensor([0.2], dtype=torch.float64)
    q = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    d = distance_to_obstacles(model, q, obs_c, obs_r)
    # l2 at (1,0,0), obstacle at (1,0.5,0): dist 0.5 - 0.1 - 0.2 = 0.2
    assert torch.allclose(d, torch.tensor([0.2], dtype=torch.float64), atol=1e-9)
    d.sum().backward()  # differentiable wrt configuration (collision-avoidance IK)
    assert q.grad is not None and torch.isfinite(q.grad).all()
