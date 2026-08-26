# tests/test_analysis_ext.py
"""Manipulability ellipsoids, dynamic manipulability and singularity proximity,
checked against oracles derived outside the library.

The main oracle is a planar 2R arm with unit links, for which everything can be
written down by hand. With the base joint at the origin, the elbow one metre
out along the arm and the tool one metre past the elbow,

    x = cos(q1) + cos(q1 + q2),      y = sin(q1) + sin(q1 + q2)

so the in-plane Jacobian is

    J = [[-sin q1 - sin(q1+q2),  -sin(q1+q2)],
         [ cos q1 + cos(q1+q2),   cos(q1+q2)]]

At q1 = 0, q2 = 90 degrees that is J = [[-1, -1], [1, 0]], hence

    J J^T = [[2, -1], [-1, 1]],   trace 3, determinant 1

whose eigenvalues solve lam^2 - 3 lam + 1 = 0, i.e. lam = (3 +- sqrt 5) / 2.
Those are phi^2 and phi^-2 for the golden ratio phi = (1 + sqrt 5)/2, so the
ellipsoid semi-axis lengths are exactly phi = 1.6180... and 1/phi = 0.6180...,
and their product is 1 = l1 l2 |sin q2|, matching Yoshikawa's measure.
The eigenvector for phi^2 is (phi, -1)/sqrt(1 + phi^2) = (0.85065, -0.52573)
and the one for phi^-2 is (1, phi)/sqrt(1 + phi^2) = (0.52573, 0.85065).

The dynamic ellipsoid oracle adds a hand-derived mass matrix for the same arm
with rod-like links (mass 1, COM at mid-link, izz = 0.10 each):

    M11 = 1.70 + cos q2,   M12 = M21 = 0.35 + 0.5 cos q2,   M22 = 0.35

and then does the linear algebra in numpy rather than in torch.
"""
import math
import numpy as np
import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.urdf.parse import parse_urdf_string
from kinfast.fk import forward_kinematics
from kinfast import analysis as A
from kinfast import analysis_ext as AE
from kinfast import dynamics as D
from tests.test_spatial import SIX_DOF

PHI = (1.0 + math.sqrt(5.0)) / 2.0

# planar 2R, unit links, tool one metre past the elbow (same geometry as the
# arm the module docstring works through), with rod-like inertials so the
# dynamic ellipsoid has something to weigh
PLANAR_2R_INERTIAL = """
<robot name="p2r_inertial">
  <link name="base"/>
  <link name="l1">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.005" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="l2">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.005" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="ee"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" velocity="2" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" velocity="2" effort="50"/></joint>
  <joint name="jf" type="fixed"><parent link="l2"/><child link="ee"/>
    <origin xyz="1 0 0"/></joint>
</robot>
"""

FIXED_ONLY = """
<robot name="ext_fixed_only">
  <link name="base"/><link name="tip"/>
  <joint name="jf" type="fixed"><parent link="base"/><child link="tip"/>
    <origin xyz="0.5 0 0.2"/></joint>
</robot>
"""


def _p2r(dtype=torch.float64):
    return compile_robot(parse_urdf_string(PLANAR_2R_INERTIAL), dtype=dtype)


def _J_oracle(q1, q2):
    """In-plane 2x2 Jacobian of the unit-link planar 2R, written out by hand."""
    s1, c1 = math.sin(q1), math.cos(q1)
    s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
    return np.array([[-s1 - s12, -s12],
                     [c1 + c12, c12]], dtype=np.float64)


def _M_oracle(q2):
    """Mass matrix of the same arm, from the textbook planar 2R formula."""
    c2 = math.cos(q2)
    return np.array([[1.70 + c2, 0.35 + 0.5 * c2],
                     [0.35 + 0.5 * c2, 0.35]], dtype=np.float64)


def _axes_agree(axes, expect, tol=1e-9):
    """Axis directions match up to sign (an axis and its negative are the same
    axis). `axes` has one direction per column, `expect` is a list of vectors."""
    for i, want in enumerate(expect):
        got = axes[:, i].numpy() if torch.is_tensor(axes) else axes[:, i]
        want = np.asarray(want, dtype=np.float64)
        assert abs(abs(float(got @ want)) - 1.0) < tol, f"axis {i}: {got} vs {want}"


# ---------------------------------------------------------------- velocity


def test_ellipsoid_axes_at_elbow_90_match_hand_computation():
    """The whole hand computation in the module docstring, at q1 = 0, q2 = 90 deg."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q = torch.tensor([[0.0, math.pi / 2]], dtype=torch.float64)
    e = AE.manipulability_ellipsoid(chain, q, li, rows=(0, 1))

    assert e["lengths"].shape == (1, 2)
    assert e["axes"].shape == (1, 2, 2)
    # semi-axis lengths phi and 1/phi
    assert abs(e["lengths"][0, 0].item() - PHI) < 1e-12
    assert abs(e["lengths"][0, 1].item() - 1.0 / PHI) < 1e-12
    # eigenvalues of J J^T are (3 +- sqrt 5)/2
    assert abs(e["eigenvalues"][0, 0].item() - (3 + math.sqrt(5)) / 2) < 1e-12
    assert abs(e["eigenvalues"][0, 1].item() - (3 - math.sqrt(5)) / 2) < 1e-12
    # volume = det = l1 l2 |sin q2| = 1
    assert abs(e["volume"][0].item() - 1.0) < 1e-12
    # axis directions
    n = math.sqrt(1.0 + PHI * PHI)
    _axes_agree(e["axes"][0], [(PHI / n, -1.0 / n), (1.0 / n, PHI / n)])
    # and they are a genuine orthonormal frame
    U = e["axes"][0]
    assert torch.allclose(U @ U.T, torch.eye(2, dtype=torch.float64), atol=1e-12)


def test_ellipsoid_rotates_with_the_base_joint():
    """At q2 = 90 degrees the arm is the same shape whatever q1 is, so the
    lengths must not move and the axes must rotate exactly by R(q1). This is a
    property of the planar arm, checked against an explicit rotation matrix."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q1s = [0.0, 0.4, -1.3, 2.9]
    q = torch.tensor([[a, math.pi / 2] for a in q1s], dtype=torch.float64)
    e = AE.manipulability_ellipsoid(chain, q, li, rows=(0, 1))
    n = math.sqrt(1.0 + PHI * PHI)
    base = np.array([[PHI / n, 1.0 / n], [-1.0 / n, PHI / n]])   # columns = axes
    for b, a in enumerate(q1s):
        assert abs(e["lengths"][b, 0].item() - PHI) < 1e-12
        assert abs(e["lengths"][b, 1].item() - 1.0 / PHI) < 1e-12
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        want = R @ base
        _axes_agree(e["axes"][b], [want[:, 0], want[:, 1]], tol=1e-9)


def test_ellipsoid_volume_is_yoshikawa_manipulability():
    """The product of the semi-axis lengths must be |sin q2| for this arm,
    which is also what kinfast.analysis.manipulability reports."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q2s = [0.0, math.pi / 6, 1.0, math.pi / 2, 2.5, -0.9]
    q = torch.tensor([[0.37, a] for a in q2s], dtype=torch.float64)
    e = AE.manipulability_ellipsoid(chain, q, li, rows=(0, 1))
    hand = torch.tensor([abs(math.sin(a)) for a in q2s], dtype=torch.float64)
    assert torch.allclose(e["volume"], hand, atol=1e-8)
    assert torch.allclose(e["volume"], A.manipulability(chain, q, li, rows=(0, 1)),
                          atol=1e-8)


def test_ellipsoid_is_flat_when_the_arm_cannot_span_the_task_space():
    """With the default 3 translational rows the planar arm has a dead
    direction: the ellipsoid must be a flat disc in the xy plane, the extra
    axis must be +-z, and the volume must be 0 (not the 1 that a determinant of
    an empty matrix would give)."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q = torch.tensor([[0.0, math.pi / 2]], dtype=torch.float64)
    e = AE.manipulability_ellipsoid(chain, q, li)             # translational, 3 rows
    assert e["axes"].shape == (1, 3, 3) and e["lengths"].shape == (1, 3)
    assert abs(e["lengths"][0, 0].item() - PHI) < 1e-12
    assert abs(e["lengths"][0, 1].item() - 1.0 / PHI) < 1e-12
    assert e["lengths"][0, 2].item() == 0.0
    assert e["volume"][0].item() == 0.0
    assert abs(abs(e["axes"][0, 2, 2].item()) - 1.0) < 1e-12   # third axis is z
    U = e["axes"][0]
    assert torch.allclose(U @ U.T, torch.eye(3, dtype=torch.float64), atol=1e-12)


def test_ellipsoid_matches_numpy_eigendecomposition_of_a_finite_difference_jacobian():
    """Spatial 6-DOF cross-check where every step of the oracle is independent:
    the Jacobian comes from central differences of FK, and the decomposition
    from numpy's symmetric eigensolver on J J^T."""
    chain = compile_robot(parse_urdf_string(SIX_DOF), dtype=torch.float64)
    li = chain.link_index["ee"]
    torch.manual_seed(11)
    q = (chain.lower + (chain.upper - chain.lower)
         * torch.rand(4, chain.dof, dtype=torch.float64))
    e = AE.manipulability_ellipsoid(chain, q, li)

    h = 1e-6
    for b in range(q.shape[0]):
        J = np.zeros((3, chain.dof))
        for j in range(chain.dof):
            qp, qm = q[b].clone(), q[b].clone()
            qp[j] += h
            qm[j] -= h
            pp = forward_kinematics(chain, qp[None])[0, li, :3, 3]
            pm = forward_kinematics(chain, qm[None])[0, li, :3, 3]
            J[:, j] = ((pp - pm) / (2 * h)).numpy()
        w, V = np.linalg.eigh(J @ J.T)                 # ascending
        w = np.clip(w[::-1], 0.0, None)                # descending
        V = V[:, ::-1]
        assert np.allclose(e["eigenvalues"][b].numpy(), w, atol=1e-7)
        assert np.allclose(e["lengths"][b].numpy(), np.sqrt(w), atol=1e-7)
        assert abs(e["volume"][b].item() - math.sqrt(max(np.prod(w), 0.0))) < 1e-7
        _axes_agree(e["axes"][b].numpy(), [V[:, 0], V[:, 1], V[:, 2]], tol=1e-5)


# ------------------------------------------------------------- singularity


def test_singular_value_vanishes_at_the_straight_configuration():
    """The arm is singular when it is straight out (q2 = 0) or folded back
    (q2 = pi). Hand computation at q2 -> 0: both Jacobian columns collapse onto
    the same unit direction u = (-sin q1, cos q1), with lengths 2 and 1, so
    J -> u [2, 1] and sigma_max -> sqrt(5). Since sigma_min sigma_max =
    |det J| = |sin q2|, the small singular value must fall off as
    |sin q2| / sqrt(5)."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q = torch.tensor([[0.31, a] for a in
                      [math.pi / 2, 0.1, 1e-2, 1e-4, 1e-8, 0.0, math.pi]],
                     dtype=torch.float64)
    s_min = AE.singularity_proximity(chain, q, li, rows=(0, 1))
    assert s_min.shape == (7,)
    # strictly decreasing as the arm straightens out
    assert torch.all(s_min[:5].diff() < 0.0)
    assert s_min[-2].item() < 1e-15                     # exactly straight
    assert s_min[-1].item() < 1e-15                     # exactly folded back
    # small-angle oracle: sigma_min -> |sin q2| / sqrt(5)
    e = AE.manipulability_ellipsoid(chain, q, li, rows=(0, 1))
    for b, a in [(2, 1e-2), (3, 1e-4)]:
        want = abs(math.sin(a)) / math.sqrt(5.0)
        assert abs(s_min[b].item() - want) < 1e-3 * want
        assert abs(e["lengths"][b, 0].item() - math.sqrt(5.0)) < 1e-3
    # away from the singularity it is the shorter ellipsoid axis, 1/phi
    assert abs(s_min[0].item() - 1.0 / PHI) < 1e-12
    assert torch.allclose(s_min, e["lengths"][:, 1], atol=1e-15)


def test_singular_value_matches_numpy_svd_of_the_hand_written_jacobian():
    chain = _p2r()
    li = chain.link_index["ee"]
    torch.manual_seed(5)
    q = (torch.rand(8, 2, dtype=torch.float64) * 6.0 - 3.0)
    got = AE.singularity_proximity(chain, q, li, rows=(0, 1))
    for b in range(q.shape[0]):
        s = np.linalg.svd(_J_oracle(float(q[b, 0]), float(q[b, 1])),
                          compute_uv=False)
        assert abs(got[b].item() - s[-1]) < 1e-12


def test_singular_value_is_zero_when_the_task_space_cannot_be_spanned():
    """A 2-dof arm asked for 3 translational rows, and a chain with no movable
    joints at all: both report 0 rather than an index error or a stray value."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q = torch.tensor([[0.0, math.pi / 2]], dtype=torch.float64)
    assert AE.singularity_proximity(chain, q, li).item() == 0.0

    fixed = compile_robot(parse_urdf_string(FIXED_ONLY), dtype=torch.float64)
    assert fixed.dof == 0
    tip = fixed.link_index["tip"]
    qz = torch.zeros(3, 0, dtype=torch.float64)
    s = AE.singularity_proximity(fixed, qz, tip)
    assert s.shape == (3,) and torch.all(s == 0.0)
    e = AE.manipulability_ellipsoid(fixed, qz, tip)
    assert torch.all(e["lengths"] == 0.0)
    assert torch.all(e["volume"] == 0.0)
    assert torch.allclose(e["axes"], torch.eye(3, dtype=torch.float64).expand(3, 3, 3))


# ----------------------------------------------------------------- dynamic


def test_mass_matrix_matches_the_hand_derived_oracle():
    """Guard on the oracle used by the dynamic ellipsoid test below: if the
    textbook M is wrong, everything downstream of it is meaningless."""
    chain = _p2r()
    q = torch.tensor([[0.3, 0.0], [-1.1, math.pi / 2], [2.0, 2.4]],
                     dtype=torch.float64)
    M = D.mass_matrix(chain, q)
    for b in range(q.shape[0]):
        assert np.allclose(M[b].numpy(), _M_oracle(float(q[b, 1])), atol=1e-12)


def test_dynamic_manipulability_matches_the_numpy_oracle():
    """J M^-1 M^-T J^T assembled from the hand Jacobian and the hand mass
    matrix, decomposed by numpy."""
    chain = _p2r()
    li = chain.link_index["ee"]
    qs = [(0.0, math.pi / 2), (0.4, 1.1), (-1.2, -2.0), (2.2, 0.6)]
    q = torch.tensor(qs, dtype=torch.float64)
    e = AE.dynamic_manipulability(chain, q, li, rows=(0, 1))
    for b, (q1, q2) in enumerate(qs):
        J, M = _J_oracle(q1, q2), _M_oracle(q2)
        Minv = np.linalg.inv(M)
        S = J @ Minv @ Minv.T @ J.T
        w, V = np.linalg.eigh(S)
        w, V = np.clip(w[::-1], 0.0, None), V[:, ::-1]
        assert np.allclose(e["eigenvalues"][b].numpy(), w, rtol=1e-9, atol=1e-12)
        assert np.allclose(e["lengths"][b].numpy(), np.sqrt(w), rtol=1e-9, atol=1e-12)
        assert abs(e["volume"][b].item() - math.sqrt(np.prod(w))) < 1e-9
        _axes_agree(e["axes"][b].numpy(), [V[:, 0], V[:, 1]], tol=1e-7)


def test_dynamic_and_velocity_ellipsoids_differ_in_shape():
    """The point of the dynamic ellipsoid: a direction can be kinematically
    cheap and dynamically expensive. At the elbow-90 pose the two ellipsoids
    must not be scalar multiples of each other."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q = torch.tensor([[0.0, math.pi / 2]], dtype=torch.float64)
    v = AE.manipulability_ellipsoid(chain, q, li, rows=(0, 1))
    d = AE.dynamic_manipulability(chain, q, li, rows=(0, 1))
    ratio_v = (v["lengths"][0, 0] / v["lengths"][0, 1]).item()
    ratio_d = (d["lengths"][0, 0] / d["lengths"][0, 1]).item()
    assert abs(ratio_v - ratio_d) > 0.1
    # sanity: it really is J M^-1, so scaling every inertia by k scales the
    # acceleration ellipsoid by 1/k and leaves the velocity one untouched
    heavy = compile_robot(parse_urdf_string(
        PLANAR_2R_INERTIAL.replace('name="p2r_inertial"', 'name="heavy"')
        .replace('value="1.0"', 'value="4.0"')
        .replace('"0.005"', '"0.020"').replace('"0.10"', '"0.40"')),
        dtype=torch.float64)
    dh = AE.dynamic_manipulability(heavy, q, heavy.link_index["ee"], rows=(0, 1))
    assert torch.allclose(dh["lengths"], d["lengths"] / 4.0, rtol=1e-10)


def test_dynamic_manipulability_reports_a_singular_mass_matrix():
    """A joint that carries no mass makes M singular and the acceleration
    ellipsoid unbounded; that must be a named error, not inf or NaN."""
    urdf = PLANAR_2R_INERTIAL.replace('name="p2r_inertial"', 'name="massless_tip"')
    urdf = urdf.replace(
        """<inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0.005" iyy="0.10" izz="0.10" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="ee"/>""", "</link>\n  <link name=\"ee\"/>")
    chain = compile_robot(parse_urdf_string(urdf), dtype=torch.float64)
    q = torch.tensor([[0.2, 0.9]], dtype=torch.float64)
    M = D.mass_matrix(chain, q)
    assert abs(torch.linalg.det(M).item()) < 1e-12      # really is singular
    with pytest.raises(ValueError, match="mass matrix is singular"):
        AE.dynamic_manipulability(chain, q, chain.link_index["ee"], rows=(0, 1))


# ------------------------------------------------- batching, dtype, grads


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_and_dtype_follows_q(dtype):
    """The working dtype is q's, not the compiled chain's, and the batch
    dimension flows through untouched."""
    chain = _p2r(dtype=torch.float32)              # compiled in single precision
    li = chain.link_index["ee"]
    torch.manual_seed(2)
    q = torch.rand(7, 2, dtype=dtype) * 2.0 - 1.0
    for e in (AE.manipulability_ellipsoid(chain, q, li, rows=(0, 1)),
              AE.dynamic_manipulability(chain, q, li, rows=(0, 1))):
        assert e["axes"].shape == (7, 2, 2)
        assert e["lengths"].shape == (7, 2)
        assert e["eigenvalues"].shape == (7, 2)
        assert e["volume"].shape == (7,)
        for v in e.values():
            assert v.dtype == dtype and v.device == q.device
        assert torch.all(e["lengths"][:, 0] >= e["lengths"][:, 1])
    s = AE.singularity_proximity(chain, q, li, rows=(0, 1))
    assert s.shape == (7,) and s.dtype == dtype

    # a single-row batch and a wide batch agree elementwise
    one = AE.manipulability_ellipsoid(chain, q[3:4], li, rows=(0, 1))
    many = AE.manipulability_ellipsoid(chain, q, li, rows=(0, 1))
    tol = 1e-6 if dtype is torch.float32 else 1e-12
    assert torch.allclose(one["lengths"][0], many["lengths"][3], atol=tol)


def test_ellipsoid_volume_is_differentiable_with_a_closed_form_gradient():
    """d/dq2 of the velocity-ellipsoid volume is d|sin q2|/dq2 = cos q2 for
    0 < q2 < pi, and the volume does not depend on q1 at all. Autograd must
    reproduce both."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q = torch.tensor([[0.4, 0.7], [-1.0, 2.1]], dtype=torch.float64,
                     requires_grad=True)
    vol = AE.manipulability_ellipsoid(chain, q, li, rows=(0, 1))["volume"]
    vol.sum().backward()
    assert torch.allclose(q.grad[:, 0], torch.zeros(2, dtype=torch.float64),
                          atol=1e-9)
    want = torch.tensor([math.cos(0.7), math.cos(2.1)], dtype=torch.float64)
    assert torch.allclose(q.grad[:, 1], want, atol=1e-9)


def test_gradients_flow_through_the_lengths_and_the_dynamic_ellipsoid():
    """Central differences in float64 on the longest semi-axis of both
    ellipsoids: autograd must match to the accuracy of the difference."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q0 = torch.tensor([[0.35, 1.05]], dtype=torch.float64)

    def loss(qq, fn):
        return fn(chain, qq, li, rows=(0, 1))["lengths"][0, 0]

    for fn in (AE.manipulability_ellipsoid, AE.dynamic_manipulability):
        q = q0.clone().requires_grad_(True)
        loss(q, fn).backward()
        h = 1e-6
        for j in range(2):
            qp, qm = q0.clone(), q0.clone()
            qp[0, j] += h
            qm[0, j] -= h
            fd = (loss(qp, fn) - loss(qm, fn)).item() / (2 * h)
            assert abs(q.grad[0, j].item() - fd) < 1e-6, (fn.__name__, j)


def test_singularity_proximity_is_differentiable():
    chain = _p2r()
    li = chain.link_index["ee"]
    q0 = torch.tensor([[0.2, 1.3]], dtype=torch.float64)
    q = q0.clone().requires_grad_(True)
    AE.singularity_proximity(chain, q, li, rows=(0, 1)).sum().backward()
    h = 1e-6
    for j in range(2):
        qp, qm = q0.clone(), q0.clone()
        qp[0, j] += h
        qm[0, j] -= h
        fd = (AE.singularity_proximity(chain, qp, li, rows=(0, 1))
              - AE.singularity_proximity(chain, qm, li, rows=(0, 1))).item() / (2 * h)
        assert abs(q.grad[0, j].item() - fd) < 1e-6


# ------------------------------------------------------- mujoco cross-check


MJ_ARM = """
<mujoco model="ext_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="l1" pos="0 0 0.3">
      <inertial pos="0.1 0.02 0" quat="0.9659258 0 0 0.2588190" mass="2.5"
                diaginertia="0.04 0.03 0.02"/>
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
      <body name="l2" pos="0.4 0 0" euler="0 0.3 0">
        <inertial pos="0.2 0 0.05" mass="1.2" diaginertia="0.02 0.015 0.008"/>
        <joint name="j2" type="hinge" axis="0 1 0" range="-2 2"/>
        <body name="l3" pos="0.35 0 0">
          <inertial pos="0.1 0 0" quat="0.7071068 0.7071068 0 0" mass="0.6"
                    diaginertia="0.004 0.003 0.001"/>
          <joint name="j3" type="slide" axis="1 0 0" range="-0.2 0.2"/>
          <body name="l4" pos="0.15 0 0">
            <inertial pos="0 0 0" mass="0.3" diaginertia="0.001 0.001 0.001"/>
            <joint name="j4" type="hinge" axis="1 0 0" range="-3 3"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_ellipsoids_match_mujoco_on_a_spatial_arm():
    """Fully independent oracle: MuJoCo supplies both the Jacobian (mj_jac) and
    the mass matrix (mj_fullM) for a 4-dof spatial arm with a prismatic joint,
    numpy does the decomposition, and kinfast has to agree on the velocity
    ellipsoid, the dynamic ellipsoid and the smallest singular value."""
    mujoco = pytest.importorskip("mujoco")
    from kinfast.mjcf.parse import parse_mjcf_string

    m = mujoco.MjModel.from_xml_string(MJ_ARM)
    m.dof_armature[:] = 0.0             # kinfast has no armature concept
    d = mujoco.MjData(m)
    chain = compile_robot(parse_mjcf_string(MJ_ARM), dtype=torch.float64)
    li = chain.link_index["l4"]
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l4")

    addr = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        addr[nm] = (m.jnt_qposadr[j], m.jnt_dofadr[j])
    cols = [addr[nm] for nm in chain.joint_names]
    order = [va for _, va in cols]

    rng = np.random.RandomState(3)
    lo = chain.lower.numpy()
    hi = chain.upper.numpy()
    for _ in range(6):
        qn = lo + (hi - lo) * rng.rand(chain.dof)
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0.0
        for k, (qa, _) in enumerate(cols):
            d.qpos[qa] = qn[k]
        mujoco.mj_forward(m, d)

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, jacr, d.xpos[body], body)
        J = jacp[:, order]                      # 3 x dof, kinfast column order
        M_mj = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M_mj)
        M = M_mj[np.ix_(order, order)]

        q = torch.tensor(qn, dtype=torch.float64).unsqueeze(0)
        v = AE.manipulability_ellipsoid(chain, q, li)
        w, V = np.linalg.eigh(J @ J.T)
        w, V = np.clip(w[::-1], 0.0, None), V[:, ::-1]
        assert np.allclose(v["eigenvalues"][0].numpy(), w, rtol=1e-6, atol=1e-10)
        assert np.allclose(v["lengths"][0].numpy(), np.sqrt(w), rtol=1e-6, atol=1e-10)
        assert abs(v["volume"][0].item() - math.sqrt(max(np.prod(w), 0.0))) < 1e-8
        _axes_agree(v["axes"][0].numpy(), [V[:, 0], V[:, 1], V[:, 2]], tol=1e-5)

        s_min = AE.singularity_proximity(chain, q, li)
        assert abs(s_min.item() - float(np.sqrt(w[-1]))) < 1e-8

        a = AE.dynamic_manipulability(chain, q, li)
        Minv = np.linalg.inv(M)
        wd, Vd = np.linalg.eigh(J @ Minv @ Minv.T @ J.T)
        wd, Vd = np.clip(wd[::-1], 0.0, None), Vd[:, ::-1]
        assert np.allclose(a["eigenvalues"][0].numpy(), wd, rtol=1e-5, atol=1e-10)
        assert np.allclose(a["lengths"][0].numpy(), np.sqrt(wd), rtol=1e-5,
                           atol=1e-10)
        _axes_agree(a["axes"][0].numpy(), [Vd[:, 0], Vd[:, 1], Vd[:, 2]], tol=1e-4)


# -------------------------------------------------------------- validation


def test_ellipsoid_primitive_on_matrices_written_by_hand():
    A2 = torch.tensor([[[3.0, 0.0], [0.0, 1.0]],           # axis-aligned
                       [[0.0, 2.0], [0.0, 0.0]]],          # rank 1
                      dtype=torch.float64)
    e = AE.ellipsoid(A2)
    assert torch.allclose(e["lengths"][0], torch.tensor([3.0, 1.0],
                                                        dtype=torch.float64))
    assert abs(e["volume"][0].item() - 3.0) < 1e-12
    _axes_agree(e["axes"][0], [(1.0, 0.0), (0.0, 1.0)])
    assert torch.allclose(e["lengths"][1], torch.tensor([2.0, 0.0],
                                                        dtype=torch.float64))
    assert e["volume"][1].item() == 0.0

    # a 2x3 map: three joints, two task rows, nothing flat
    A3 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]], dtype=torch.float64)
    e3 = AE.ellipsoid(A3)
    assert e3["axes"].shape == (1, 2, 2) and e3["lengths"].shape == (1, 2)
    assert torch.allclose(e3["lengths"][0],
                          torch.tensor([math.sqrt(2.0), 1.0], dtype=torch.float64))


def test_bad_inputs_are_named():
    chain = _p2r()
    li = chain.link_index["ee"]
    q = torch.tensor([[0.4, 0.7]], dtype=torch.float64)
    with pytest.raises(ValueError, match="B, m, n"):
        AE.ellipsoid(torch.eye(3))
    with pytest.raises(ValueError, match="task row"):
        AE.ellipsoid(torch.zeros(2, 0, 3))
    for bad in [(), (0, 7), [0, 1, 0]]:
        with pytest.raises(ValueError, match="rows"):
            AE.manipulability_ellipsoid(chain, q, li, rows=bad)
        with pytest.raises(ValueError, match="rows"):
            AE.dynamic_manipulability(chain, q, li, rows=bad)
        with pytest.raises(ValueError, match="rows"):
            AE.singularity_proximity(chain, q, li, rows=bad)
    with pytest.raises(IndexError):
        AE.manipulability_ellipsoid(chain, q, 99)


def test_row_selection_and_link_index_conventions():
    """rows accepts the same spellings as kinfast.analysis (tensors, negative
    indices, any order), and a negative link index means the last link."""
    chain = _p2r()
    li = chain.link_index["ee"]
    q = torch.tensor([[0.4, math.pi / 2]], dtype=torch.float64)
    ref = AE.manipulability_ellipsoid(chain, q, li, rows=(0, 1))["lengths"]
    for ok in [[1, 0], torch.tensor([0, 1]), (-6, -5)]:
        got = AE.manipulability_ellipsoid(chain, q, li, rows=ok)["lengths"]
        assert torch.allclose(got, ref, atol=1e-12), f"rows={ok!r}"
    assert li == chain.n_links - 1
    neg = AE.manipulability_ellipsoid(chain, q, -1, rows=(0, 1))["lengths"]
    assert torch.allclose(neg, ref, atol=1e-12)
    # all six rows on a planar arm: three rotational rows, only wz alive
    full = AE.manipulability_ellipsoid(chain, q, li, translational=False)
    assert full["lengths"].shape == (1, 6)
    assert full["volume"][0].item() == 0.0
    assert int((full["lengths"][0] > 1e-9).sum()) == 2      # rank 2, dof 2
