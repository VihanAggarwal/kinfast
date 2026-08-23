# tests/test_regressions_dtype_precision.py
"""Regression: float64 in must mean float64 accuracy, not just float64 boxes.

kinfast.load / load_string / Robot.from_ir had no dtype option, so every loaded
robot was a float32 chain. Feeding it float64 q produced float64 tensors that
carried ~7 correct digits: the origins and axes had already been rounded when
the chain was compiled, and no amount of casting afterwards brings the digits
back. The same rounding leaked into the scalar backend, whose generated source
bakes those constants in as literals, so `robot.compile()` returned a float64
code path with float32 answers.

The oracle here is a hand-written float64 forward kinematics built straight
from the URDF numbers with the `math` module and numpy: rpy -> matrix by the
URDF's own Z*Y*X convention, joints by Rodrigues, composed down the chain. It
never touches kinfast, so the agreement it measures is real.
"""
import math

import numpy as np
import pytest
import torch

import kinfast
from kinfast.codegen import CompiledRobot
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics
from kinfast.urdf.parse import parse_urdf_string

# Deliberately awkward numbers: nothing here is exact in binary, so a float32
# compile has to lose digits. The tilted origins mean the rounding also runs
# through rpy_to_matrix, not just the translations.
TILTED = """
<robot name="tilted">
  <link name="base"/>
  <link name="l1"><inertial><mass value="1.3"/>
    <origin xyz="0.11 0 0.07"/>
    <inertia ixx="0.021" iyy="0.033" izz="0.017" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <link name="l2"><inertial><mass value="0.7"/>
    <origin xyz="0.05 0.02 0"/>
    <inertia ixx="0.011" iyy="0.013" izz="0.009" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <link name="l3"><inertial><mass value="0.4"/>
    <origin xyz="0 0 0.03"/>
    <inertia ixx="0.004" iyy="0.004" izz="0.003" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0.1234567890123 0.3 0.7654321" rpy="0.31 -0.17 0.93"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0.4321 0.111 0.222" rpy="-0.4 0.55 0.13"/>
    <axis xyz="0.3 0.5 0.81"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/></joint>
  <joint name="j3" type="prismatic"><parent link="l2"/><child link="l3"/>
    <origin xyz="0.9 0.13 0.37" rpy="0.7 0.2 -0.6"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" velocity="2" effort="50"/></joint>
</robot>
"""

# (origin xyz, origin rpy, axis, is_revolute) mirroring TILTED, by hand.
JOINTS = [
    ((0.1234567890123, 0.3, 0.7654321), (0.31, -0.17, 0.93), (0.0, 0.0, 1.0), True),
    ((0.4321, 0.111, 0.222), (-0.4, 0.55, 0.13), (0.3, 0.5, 0.81), True),
    ((0.9, 0.13, 0.37), (0.7, 0.2, -0.6), (0.0, 1.0, 0.0), False),
]


def _rpy(r, p, y):
    """URDF fixed-axis roll-pitch-yaw: Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _rodrigues(axis, th):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def oracle_ee(q):
    """float64 world pose of link l3, computed only from the URDF numbers."""
    M = np.eye(4)
    for (xyz, rpy_, axis, revolute), qi in zip(JOINTS, q):
        A = np.eye(4)
        A[:3, :3] = _rpy(*rpy_)
        A[:3, 3] = xyz
        B = np.eye(4)
        if revolute:
            B[:3, :3] = _rodrigues(axis, qi)
        else:
            a = np.asarray(axis, dtype=np.float64)
            B[:3, 3] = a / np.linalg.norm(a) * qi
        M = M @ A @ B
    return M


Q = [(0.3, -0.8, 0.21), (-1.1, 0.45, -0.37), (2.0, 1.3, 0.05)]


def test_oracle_matches_zero_config_by_hand():
    """Sanity-check the oracle itself: at q=0 it is just the origin chain."""
    M = oracle_ee([0.0, 0.0, 0.0])
    A = np.eye(4)
    for (xyz, rpy_, _a, _r) in JOINTS:
        S = np.eye(4)
        S[:3, :3] = _rpy(*rpy_)
        S[:3, 3] = xyz
        A = A @ S
    assert np.abs(M - A).max() == 0.0


# ------------------------------------------------------------------ bug 1
def test_load_string_dtype_option_gives_a_float64_chain():
    r32 = kinfast.load_string(TILTED)                       # unchanged default
    r64 = kinfast.load_string(TILTED, dtype=torch.float64)
    assert r32.dtype == torch.float32
    assert r64.dtype == torch.float64
    assert r64.chain.joint_origin.dtype == torch.float64
    assert r64.chain.link_inertia.dtype == torch.float64
    assert r64.chain.lower.dtype == torch.float64
    # integer bookkeeping stays integer
    assert r64.chain.parent.dtype == torch.long
    assert r64.chain.q_index.dtype == torch.long


def test_from_ir_dtype_option():
    ir = parse_urdf_string(TILTED)
    assert kinfast.Robot.from_ir(ir).dtype == torch.float32
    assert kinfast.Robot.from_ir(ir, dtype=torch.float64).dtype == torch.float64


def test_load_file_dtype_option(tmp_path):
    p = tmp_path / "tilted.urdf"
    p.write_text(TILTED, encoding="utf-8")
    assert kinfast.load(str(p)).dtype == torch.float32
    assert kinfast.load(str(p), dtype=torch.float64).dtype == torch.float64


def test_float64_chain_reaches_1e12_fk_while_float32_does_not():
    """The headline bug: float64 q on a float32 chain is float64-shaped only."""
    r32 = kinfast.load_string(TILTED)
    r64 = kinfast.load_string(TILTED, dtype=torch.float64)
    idx = r32.link_id("l3")
    for q in Q:
        ref = oracle_ee(q)
        qt = torch.tensor([q], dtype=torch.float64)
        e32 = np.abs(r32.fk(qt, "l3")[0].numpy() - ref).max()
        e64 = np.abs(r64.fk(qt, "l3")[0].numpy() - ref).max()
        assert r32.fk(qt, "l3").dtype == torch.float64      # float64 box ...
        assert e32 > 1e-9, "float32 chain should NOT be this accurate"
        assert e64 < 1e-12, f"float64 chain missed the oracle by {e64}"
        assert e64 < e32 / 1e4                              # ... but no digits
    assert idx == r64.link_id("l3")


def test_double_recompiles_from_the_ir_and_float_goes_back():
    """double() must rebuild the constants, not cast the rounded ones up."""
    r = kinfast.load_string(TILTED)
    q = torch.tensor([Q[0]], dtype=torch.float64)
    ref = oracle_ee(Q[0])
    before = np.abs(r.fk(q, "l3")[0].numpy() - ref).max()

    same = r.double()
    assert same is r                                        # in place, like torch
    assert r.dtype == torch.float64
    after = np.abs(r.fk(q, "l3")[0].numpy() - ref).max()
    assert before > 1e-9 and after < 1e-12

    r.float()
    assert r.dtype == torch.float32
    assert np.abs(r.fk(q, "l3")[0].numpy() - ref).max() > 1e-9


def test_casting_the_chain_up_cannot_recover_precision():
    """Why to(dtype=) recompiles: CompiledChain.to only changes the boxes."""
    ir = parse_urdf_string(TILTED)
    cast = compile_robot(ir, dtype=torch.float32).to(dtype=torch.float64)
    built = compile_robot(ir, dtype=torch.float64)
    assert cast.dtype == built.dtype == torch.float64
    q = torch.tensor([Q[0]], dtype=torch.float64)
    ref = oracle_ee(Q[0])
    i = built.link_index["l3"]
    e_cast = np.abs(forward_kinematics(cast, q)[0, i].numpy() - ref).max()
    e_built = np.abs(forward_kinematics(built, q)[0, i].numpy() - ref).max()
    assert e_cast > 1e-9 and e_built < 1e-12


def test_chain_to_dtype_clears_the_stale_fk_cache():
    """fk caches folded constants per (device, dtype). Asking for float64 from
    a float32 chain fills an entry that was upcast from rounded numbers; after
    to(dtype=float64) that entry no longer describes the chain, so fk must
    refold instead of serving it. Checked by result, not by cache internals."""
    ir = parse_urdf_string(TILTED)
    chain = compile_robot(ir, dtype=torch.float32)
    q = torch.tensor([Q[0]], dtype=torch.float64)
    stale = forward_kinematics(chain, q)                    # fills the f64 entry
    chain.to(dtype=torch.float64)
    after = forward_kinematics(chain, q)
    i = chain.link_index["l3"]
    # the cast alone cannot recover digits, so the pose is unchanged...
    assert torch.allclose(after[0, i], stale[0, i], atol=1e-6)
    # ...but the entry really was refolded: it now holds float64 constants,
    # and a chain rebuilt at float64 from the same IR agrees far more tightly
    # with the closed-form pose than the stale float32 fold did
    built = compile_robot(ir, dtype=torch.float64)
    ref = oracle_ee(Q[0])
    e_stale = np.abs(stale[0, i].numpy() - ref).max()
    e_built = np.abs(forward_kinematics(built, q)[0, i].numpy() - ref).max()
    assert e_stale > 1e-9 and e_built < 1e-12


def test_torch_dtype_may_be_passed_positionally():
    r = kinfast.load_string(TILTED)
    assert r.to(torch.float64).dtype == torch.float64
    assert r.device == torch.device("cpu")
    assert r.to("cpu").dtype == torch.float64               # device-only keeps it


def test_dtype_change_needs_the_ir():
    chain = compile_robot(parse_urdf_string(TILTED))
    bare = kinfast.Robot(chain)
    with pytest.raises(ValueError, match="without its IR"):
        bare.double()
    bare.to("cpu")                                          # device-only is fine


def test_float64_chain_carries_into_dynamics_and_limits():
    r = kinfast.load_string(TILTED, dtype=torch.float64)
    q = torch.tensor([Q[0]], dtype=torch.float64)
    assert r.mass_matrix(q).dtype == torch.float64
    assert r.random_configs(4).dtype == torch.float64
    assert torch.allclose(r.lower, torch.tensor([-3.0, -3.0, -0.5],
                                                dtype=torch.float64))


# ------------------------------------------------------------------ bug 2
def test_compiled_robot_precision_follows_the_chain():
    """The scalar backend is float64 math fed by the chain's constants."""
    ir = parse_urdf_string(TILTED)
    fast32 = CompiledRobot(compile_robot(ir, dtype=torch.float32))
    fast64 = CompiledRobot(compile_robot(ir, dtype=torch.float64))
    assert fast32.dtype == torch.float32
    assert fast64.dtype == torch.float64
    link = fast64.chain.link_index["l3"]
    for q in Q:
        ref = oracle_ee(q)
        got32 = fast32.fk(list(q), link)
        got64 = fast64.fk(list(q), link)
        assert got32.dtype == np.float64 and got64.dtype == np.float64
        e32 = np.abs(got32 - ref).max()
        e64 = np.abs(got64 - ref).max()
        assert e32 > 1e-9, "float32 literals should NOT be this accurate"
        assert e64 < 1e-12, f"float64 codegen missed the oracle by {e64}"


def test_robot_compile_inherits_the_loaded_dtype():
    fast = kinfast.load_string(TILTED, dtype=torch.float64).compile()
    assert fast.dtype == torch.float64
    link = fast.chain.link_index["l3"]
    for q in Q:
        assert np.abs(fast.fk(list(q), link) - oracle_ee(q)).max() < 1e-12
    slow = kinfast.load_string(TILTED).compile()
    assert slow.dtype == torch.float32
    assert np.abs(slow.fk(list(Q[0]), link) - oracle_ee(Q[0])).max() > 1e-9


def test_float64_codegen_jacobian_matches_float64_finite_differences():
    """Independent oracle: central differences on the float64 oracle FK."""
    fast = kinfast.load_string(TILTED, dtype=torch.float64).compile()
    link = fast.chain.link_index["l3"]
    q = list(Q[1])
    J = fast.jacobian(q, link)
    h = 1e-6
    for k in range(3):
        qp, qm = list(q), list(q)
        qp[k] += h
        qm[k] -= h
        dp = (oracle_ee(qp)[:3, 3] - oracle_ee(qm)[:3, 3]) / (2 * h)
        assert np.abs(J[:3, k] - dp).max() < 1e-8


def test_float64_panda_agrees_with_the_torch_path_at_1e12():
    """A real robot, if the gitignored asset is around: the scalar and batched
    float64 paths must agree far beyond what a float32 chain could."""
    import os
    panda = "C:/Users/vihan/urdf-doctor/examples/assets/panda.urdf"
    if not os.path.exists(panda):
        pytest.skip("panda.urdf not present in this worktree")
    r = kinfast.load(panda, dtype=torch.float64)
    fast = r.compile()
    torch.manual_seed(0)
    qs = r.random_configs(8)
    assert qs.dtype == torch.float64
    ref = r.fk_all(qs)
    link = r.link_id(r.ee_link)
    for k in range(qs.shape[0]):
        got = fast.fk(qs[k].tolist(), link)
        assert np.abs(got - ref[k, link].numpy()).max() < 1e-12
