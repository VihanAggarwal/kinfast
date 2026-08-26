# tests/test_fk_subset.py
"""Tests for pruned forward kinematics (kinfast.fk_subset).

Two kinds of check run here. The cheap one is agreement with the full sweep:
fk_links must reproduce forward_kinematics column for column on the links it
was asked for. The stronger one is an independent oracle: a small numpy
re-implementation of the URDF convention (extrinsic rpy, Rodrigues, homogeneous
composition down the ancestor path) that never calls kinfast for the math, so a
shared bug in the two torch sweeps cannot hide.

The branching fixture is generated from one seeded spec table. The same table
feeds the URDF text and the numpy oracle, so the fixture stays deterministic
and the oracle stays honest about the geometry it is checking.
"""
import math
import random
import time

import numpy as np
import pytest
import torch

from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics, fk_rp
from kinfast.fk_subset import LinkSet, fk_links, link_set
from kinfast.urdf.parse import parse_urdf_string

from tests.test_spatial import SIX_DOF


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

N_BRANCHES = 40
SPINE = 3


def _branching_spec(n_branches=N_BRANCHES, spine=SPINE, seed=1234):
    """Joint table for a wide tree: a short spine, then n_branches arms.

    Each branch is revolute, revolute, prismatic, fixed, so the pruned sweep has
    to handle all three joint codes plus a fixed tip. Origins carry both an
    offset and an rpy rotation so composition order actually matters.
    """
    rng = random.Random(seed)

    def rand_xyz():
        return tuple(round(rng.uniform(-0.4, 0.4), 4) for _ in range(3))

    def rand_rpy():
        return tuple(round(rng.uniform(-1.2, 1.2), 4) for _ in range(3))

    def rand_axis():
        v = np.array([rng.uniform(-1, 1) for _ in range(3)])
        n = np.linalg.norm(v)
        if n < 1e-6:
            v, n = np.array([0.0, 0.0, 1.0]), 1.0
        return tuple(round(float(x), 6) for x in v / n)

    spec = []
    parent = "base"
    for k in range(spine):
        child = f"s{k}"
        spec.append(dict(name=f"js{k}", parent=parent, child=child,
                         type="revolute", xyz=rand_xyz(), rpy=rand_rpy(),
                         axis=rand_axis()))
        parent = child
    tip_parent = parent

    for b in range(n_branches):
        prev = tip_parent
        for d, jtype in enumerate(("revolute", "revolute", "prismatic", "fixed")):
            child = f"b{b}_{d}"
            spec.append(dict(name=f"jb{b}_{d}", parent=prev, child=child,
                             type=jtype, xyz=rand_xyz(), rpy=rand_rpy(),
                             axis=rand_axis()))
            prev = child
    return spec


def _spec_to_urdf(spec, name="wide"):
    links = ["base"] + [j["child"] for j in spec]
    out = [f'<robot name="{name}">']
    out += [f'  <link name="{n}"/>' for n in links]
    for j in spec:
        limit = ""
        if j["type"] == "revolute":
            limit = '<limit lower="-3.0" upper="3.0" velocity="2" effort="50"/>'
        elif j["type"] == "prismatic":
            limit = '<limit lower="-0.3" upper="0.3" velocity="1" effort="50"/>'
        out.append(
            f'  <joint name="{j["name"]}" type="{j["type"]}">'
            f'<parent link="{j["parent"]}"/><child link="{j["child"]}"/>'
            f'<origin xyz="{" ".join(map(str, j["xyz"]))}" '
            f'rpy="{" ".join(map(str, j["rpy"]))}"/>'
            f'<axis xyz="{" ".join(map(str, j["axis"]))}"/>{limit}</joint>')
    out.append("</robot>")
    return "\n".join(out)


BRANCH_SPEC = _branching_spec()
BRANCHING = _spec_to_urdf(BRANCH_SPEC)
BRANCH_TIPS = [f"b{b}_3" for b in range(N_BRANCHES)]


def _six_dof_chain(dtype=torch.float64):
    return compile_robot(parse_urdf_string(SIX_DOF), dtype=dtype)


def _branch_chain(dtype=torch.float64):
    return compile_robot(parse_urdf_string(BRANCHING), dtype=dtype)


def _rand_q(chain, B, seed=0, scale=0.8):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, chain.dof, generator=g, dtype=torch.float64) * 2 - 1) * scale


# --------------------------------------------------------------------------
# independent numpy oracle
# --------------------------------------------------------------------------

def _np_rpy(r, p, y):
    """URDF extrinsic X-Y-Z: R = Rz(y) @ Ry(p) @ Rx(r). Written out here rather
    than imported so the oracle does not inherit a kinfast convention bug."""
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _np_rodrigues(axis, angle):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _np_homog(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _oracle_pose(spec, q_of_joint, link):
    """World pose of `link`, composed straight down the ancestor path in numpy.

    q_of_joint maps a joint name to its scalar value for one batch element.
    """
    by_child = {j["child"]: j for j in spec}
    path = []
    cur = link
    while cur in by_child:
        j = by_child[cur]
        path.append(j)
        cur = j["parent"]
    path.reverse()

    M = np.eye(4)
    for j in path:
        origin = _np_homog(_np_rpy(*j["rpy"]), np.array(j["xyz"], dtype=float))
        if j["type"] == "revolute":
            local = origin @ _np_homog(_np_rodrigues(j["axis"], q_of_joint[j["name"]]),
                                       np.zeros(3))
        elif j["type"] == "prismatic":
            a = np.asarray(j["axis"], dtype=float)
            a = a / np.linalg.norm(a)
            local = origin @ _np_homog(np.eye(3), a * q_of_joint[j["name"]])
        else:
            local = origin
        M = M @ local
    return M


def test_oracle_agrees_with_pruned_fk_on_branching_tree():
    """The load-bearing test: hand-rolled numpy composition vs fk_links."""
    chain = _branch_chain()
    q = _rand_q(chain, 3, seed=7)
    col = {name: i for i, name in enumerate(chain.joint_names)}

    links = ["b0_3", "b7_2", "b19_0", "b39_3", "s1"]
    got = fk_links(chain, q, links).numpy()

    for b in range(q.shape[0]):
        q_of_joint = {name: float(q[b, i]) for name, i in col.items()}
        for k, name in enumerate(links):
            want = _oracle_pose(BRANCH_SPEC, q_of_joint, name)
            assert np.allclose(got[b, k], want, atol=1e-12), (b, name)


def test_oracle_agrees_on_six_dof_zero_and_random():
    """Same oracle against the 6-DOF arm, whose zero pose is known by hand."""
    chain = _six_dof_chain()
    ee = chain.link_index["ee"]

    zero = fk_links(chain, torch.zeros(1, 6, dtype=torch.float64), [ee])
    assert torch.allclose(zero[0, 0, :3, 3],
                          torch.tensor([0.0, 0.0, 1.1], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(zero[0, 0, :3, :3], torch.eye(3, dtype=torch.float64), atol=1e-12)

    spec = [dict(name="j1", parent="base", child="l1", type="revolute",
                 xyz=(0, 0, 0), rpy=(0, 0, 0), axis=(0, 0, 1)),
            dict(name="j2", parent="l1", child="l2", type="revolute",
                 xyz=(0, 0, 0.3), rpy=(0, 0, 0), axis=(0, 1, 0)),
            dict(name="j3", parent="l2", child="l3", type="revolute",
                 xyz=(0, 0, 0.3), rpy=(0, 0, 0), axis=(0, 1, 0)),
            dict(name="j4", parent="l3", child="l4", type="revolute",
                 xyz=(0, 0, 0.3), rpy=(0, 0, 0), axis=(0, 0, 1)),
            dict(name="j5", parent="l4", child="l5", type="revolute",
                 xyz=(0, 0, 0.1), rpy=(0, 0, 0), axis=(0, 1, 0)),
            dict(name="j6", parent="l5", child="ee", type="revolute",
                 xyz=(0, 0, 0.1), rpy=(0, 0, 0), axis=(0, 0, 1))]
    q = _rand_q(chain, 2, seed=11)
    col = {name: i for i, name in enumerate(chain.joint_names)}
    got = fk_links(chain, q, ["l3", "ee"]).numpy()
    for b in range(2):
        q_of_joint = {name: float(q[b, i]) for name, i in col.items()}
        for k, name in enumerate(["l3", "ee"]):
            assert np.allclose(got[b, k], _oracle_pose(spec, q_of_joint, name), atol=1e-12)


# --------------------------------------------------------------------------
# agreement with the full sweep
# --------------------------------------------------------------------------

def test_matches_full_fk_every_link_six_dof():
    chain = _six_dof_chain()
    q = _rand_q(chain, 5, seed=1)
    full = forward_kinematics(chain, q)
    every = list(range(chain.n_links))
    assert torch.allclose(fk_links(chain, q, every), full, atol=1e-13)


def test_matches_full_fk_subsets_six_dof():
    """Names, indices, negative indices, singletons and repeats all line up."""
    chain = _six_dof_chain()
    q = _rand_q(chain, 4, seed=2)
    full = forward_kinematics(chain, q)
    ee = chain.link_index["ee"]

    cases = [
        (["ee"], [ee]),
        ("ee", [ee]),
        (-1, [chain.n_links - 1]),
        ([-1, 0], [chain.n_links - 1, 0]),
        (["l2", "ee", "l2"], [chain.link_index["l2"], ee, chain.link_index["l2"]]),
        ([3], [3]),
        ((2, 4), [2, 4]),
        (torch.tensor([1, 5]), [1, 5]),
    ]
    for request, expect in cases:
        got = fk_links(chain, q, request)
        assert got.shape == (4, len(expect), 4, 4), request
        assert torch.allclose(got, full[:, expect], atol=1e-13), request


def test_matches_full_fk_every_link_branching():
    chain = _branch_chain()
    q = _rand_q(chain, 3, seed=3)
    full = forward_kinematics(chain, q)
    got = fk_links(chain, q, list(range(chain.n_links)))
    assert torch.allclose(got, full, atol=1e-13)


def test_matches_full_fk_all_tips_branching():
    """One call asking for all 40 leaves, and 40 calls asking for one each."""
    chain = _branch_chain()
    q = _rand_q(chain, 2, seed=4)
    full = forward_kinematics(chain, q)
    idx = [chain.link_index[n] for n in BRANCH_TIPS]

    together = fk_links(chain, q, BRANCH_TIPS)
    assert together.shape == (2, N_BRANCHES, 4, 4)
    assert torch.allclose(together, full[:, idx], atol=1e-13)

    for k, name in enumerate(BRANCH_TIPS):
        one = fk_links(chain, q, name)
        assert torch.allclose(one[:, 0], full[:, idx[k]], atol=1e-13), name


def test_fk_rp_matches_full_fk_rp():
    chain = _branch_chain()
    q = _rand_q(chain, 3, seed=5)
    wR, wp = fk_rp(chain, q)
    links = ["b3_1", "b22_3", "base"]
    sub_R, sub_p = link_set(chain, links).fk_rp(q)
    for k, name in enumerate(links):
        i = chain.link_index[name]
        assert torch.allclose(sub_R[k], wR[i], atol=1e-13), name
        assert torch.allclose(sub_p[k], wp[i], atol=1e-13), name


@pytest.mark.parametrize("B", [1, 2, 257])
def test_batch_shapes(B):
    chain = _branch_chain()
    q = _rand_q(chain, B, seed=6)
    got = fk_links(chain, q, ["b1_3", "b2_3"])
    assert got.shape == (B, 2, 4, 4)
    assert torch.allclose(got[:, :, 3, :],
                          torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=q.dtype).expand(B, 2, 4))


def test_empty_request_returns_empty_column():
    chain = _six_dof_chain()
    q = _rand_q(chain, 3, seed=8)
    got = fk_links(chain, q, [])
    assert got.shape == (3, 0, 4, 4)


# --------------------------------------------------------------------------
# pruning behaviour
# --------------------------------------------------------------------------

def test_prunes_to_the_ancestor_path():
    chain = _branch_chain()
    ls = link_set(chain, "b17_3")
    # base + 3 spine + 4 branch links is the whole ancestor closure.
    assert ls.n_visited == 1 + SPINE + 4
    assert ls.n_visited < chain.n_links // 10
    names = {chain.link_names[i] for i in ls.order}
    assert names == {"base", "s0", "s1", "s2",
                     "b17_0", "b17_1", "b17_2", "b17_3"}


def test_pruned_order_is_topological():
    chain = _branch_chain()
    ls = link_set(chain, ["b0_3", "b39_3", "b12_1"])
    seen = set()
    for pos, i in enumerate(ls.order):
        pp = ls.parent_pos[pos]
        if pp >= 0:
            assert ls.order[pp] in seen
        seen.add(i)
    assert len(ls) == 3
    # two disjoint branches plus a partial third: shared spine counted once
    assert ls.n_visited == 1 + SPINE + 4 + 4 + 2


def test_visited_count_grows_sublinearly_with_shared_ancestors():
    chain = _branch_chain()
    one = link_set(chain, ["b0_3"]).n_visited
    two = link_set(chain, ["b0_3", "b1_3"]).n_visited
    assert two == one + 4          # only the second branch is new
    assert two < 2 * one           # the spine is not walked twice


def test_link_set_is_cached_and_reusable():
    chain = _six_dof_chain()
    ee = chain.link_index["ee"]
    a = link_set(chain, "ee")
    b = link_set(chain, [ee])
    assert a is b
    assert link_set(chain, a) is a

    q = _rand_q(chain, 2, seed=9)
    assert torch.allclose(fk_links(chain, q, a), a.fk(q), atol=1e-15)
    assert len(a) == 1
    assert "ee" in repr(a)


def test_fixed_and_prismatic_links_resolve():
    """A fixed tip and a prismatic parent both need to come back correctly."""
    urdf = """
    <robot name="slider">
      <link name="base"/><link name="rail"/><link name="carriage"/><link name="tool"/>
      <joint name="jr" type="revolute"><parent link="base"/><child link="rail"/>
        <origin xyz="0 0 0.5" rpy="0 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-3" upper="3" velocity="1" effort="10"/></joint>
      <joint name="js" type="prismatic"><parent link="rail"/><child link="carriage"/>
        <origin xyz="0.1 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" velocity="1" effort="10"/></joint>
      <joint name="jf" type="fixed"><parent link="carriage"/><child link="tool"/>
        <origin xyz="0 0.2 0" rpy="0 0 0"/></joint>
    </robot>
    """
    chain = compile_robot(parse_urdf_string(urdf), dtype=torch.float64)
    q = torch.tensor([[math.pi / 2, 0.25]], dtype=torch.float64)
    tool = fk_links(chain, q, "tool")[0, 0]
    # rail rotated +90 deg about z at height 0.5; carriage slides +0.35 along the
    # rotated x (world +y); the fixed tool sits +0.2 along the rotated y (world -x).
    assert torch.allclose(tool[:3, 3],
                          torch.tensor([-0.2, 0.35, 0.5], dtype=torch.float64), atol=1e-12)
    full = forward_kinematics(chain, q)
    assert torch.allclose(tool, full[0, chain.link_index["tool"]], atol=1e-13)


def test_root_only_request():
    chain = _branch_chain()
    q = _rand_q(chain, 2, seed=10)
    got = fk_links(chain, q, "base")
    assert torch.allclose(got, torch.eye(4, dtype=torch.float64).expand(2, 1, 4, 4),
                          atol=1e-15)
    assert link_set(chain, "base").n_visited == 1


# --------------------------------------------------------------------------
# dtype, device, gradients
# --------------------------------------------------------------------------

def test_working_dtype_follows_q_not_the_chain():
    """A float32-compiled chain must still give float64 answers for a float64 q."""
    ir = parse_urdf_string(BRANCHING)
    chain32 = compile_robot(ir, dtype=torch.float32)
    chain64 = compile_robot(ir, dtype=torch.float64)
    q = _rand_q(chain64, 3, seed=12)

    out64 = fk_links(chain32, q, ["b5_3", "b30_1"])
    assert out64.dtype == torch.float64
    ref = fk_links(chain64, q, ["b5_3", "b30_1"])
    # only the stored constants were rounded to float32, so agreement is at
    # float32 precision, not float64.
    assert torch.allclose(out64, ref, atol=1e-6)

    out32 = fk_links(chain32, q.float(), ["b5_3", "b30_1"])
    assert out32.dtype == torch.float32
    assert torch.allclose(out32.double(), ref, atol=1e-5)


def test_gradients_match_full_fk_and_central_differences():
    chain = _six_dof_chain()
    q0 = _rand_q(chain, 1, seed=13)
    ee = chain.link_index["ee"]

    qa = q0.clone().requires_grad_(True)
    fk_links(chain, qa, [ee])[:, 0, :3, 3].sum().backward()

    qb = q0.clone().requires_grad_(True)
    forward_kinematics(chain, qb)[:, ee, :3, 3].sum().backward()

    assert torch.allclose(qa.grad, qb.grad, atol=1e-12)

    eps = 1e-6
    fd = torch.zeros_like(q0)
    for j in range(chain.dof):
        qp, qm = q0.clone(), q0.clone()
        qp[0, j] += eps
        qm[0, j] -= eps
        fp = fk_links(chain, qp, [ee])[0, 0, :3, 3].sum()
        fm = fk_links(chain, qm, [ee])[0, 0, :3, 3].sum()
        fd[0, j] = (fp - fm) / (2 * eps)
    assert torch.allclose(qa.grad, fd, atol=1e-7)


def test_gradients_are_zero_for_joints_outside_the_pruned_path():
    """A branch tip must not depend on any other branch's joints."""
    chain = _branch_chain()
    q = _rand_q(chain, 2, seed=14).requires_grad_(True)
    fk_links(chain, q, "b0_3")[:, 0, :3, 3].sum().backward()

    col = {name: i for i, name in enumerate(chain.joint_names)}
    on_path = {col[f"js{k}"] for k in range(SPINE)}
    on_path |= {col[f"jb0_{d}"] for d in range(3)}   # index 3 is fixed, no column
    off_path = [i for i in range(chain.dof) if i not in on_path]

    assert torch.isfinite(q.grad).all()
    assert (q.grad[:, off_path] == 0).all()
    assert q.grad[:, sorted(on_path)].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_device_follows_q():
    chain = _branch_chain(dtype=torch.float32).to("cuda")
    q = _rand_q(chain, 4, seed=15).float().cuda()
    got = fk_links(chain, q, ["b4_3", "b9_2"])
    assert got.device.type == "cuda"
    ref = forward_kinematics(chain, q)[:, [chain.link_index["b4_3"],
                                           chain.link_index["b9_2"]]]
    assert torch.allclose(got, ref, atol=1e-6)


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------

def test_unknown_link_name_raises():
    chain = _six_dof_chain()
    with pytest.raises(KeyError, match="nope"):
        fk_links(chain, torch.zeros(1, 6, dtype=torch.float64), "nope")


def test_out_of_range_index_raises():
    chain = _six_dof_chain()
    q = torch.zeros(1, 6, dtype=torch.float64)
    with pytest.raises(IndexError):
        fk_links(chain, q, chain.n_links)
    with pytest.raises(IndexError):
        fk_links(chain, q, -chain.n_links - 1)


def test_bad_q_shape_raises():
    chain = _six_dof_chain()
    with pytest.raises(ValueError, match=r"\(B, dof\)"):
        fk_links(chain, torch.zeros(6, dtype=torch.float64), "ee")
    with pytest.raises(ValueError, match="degrees of freedom"):
        fk_links(chain, torch.zeros(2, 5, dtype=torch.float64), "ee")


def test_stranded_link_raises_a_readable_error():
    """A link with no path to the root cannot be swept forward."""
    chain = _six_dof_chain()
    intact = list(chain.topo_order)
    orphan = chain.link_index["l3"]
    chain.topo_order = [i for i in intact if i != orphan]
    try:
        with pytest.raises(ValueError, match="not connected to the root"):
            LinkSet(chain, ["ee"])
    finally:
        chain.topo_order = intact
    assert len(LinkSet(chain, ["ee"])) == 1


# --------------------------------------------------------------------------
# performance (the point of the module)
# --------------------------------------------------------------------------

def _median_time(fn, repeats=9, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def test_one_leaf_is_faster_than_the_full_sweep():
    """Pruning a 164-link tree down to one leaf's 8 ancestors should be a large
    win: the sweep is a sequential python loop, so it scales with link count."""
    chain = _branch_chain(dtype=torch.float32)
    q = _rand_q(chain, 64, seed=16).float()
    ls = link_set(chain, "b20_3")
    leaf = chain.link_index["b20_3"]

    ls.fk(q)                       # build the pruned constants before timing
    forward_kinematics(chain, q)

    full = _median_time(lambda: forward_kinematics(chain, q)[:, leaf])
    pruned = _median_time(lambda: ls.fk(q))
    speedup = full / pruned
    print(f"\nfk_subset: {chain.n_links} links, {chain.dof} dof, B=64; "
          f"full {full * 1e3:.3f} ms, one leaf {pruned * 1e3:.3f} ms, "
          f"speedup {speedup:.1f}x (visited {ls.n_visited}/{chain.n_links})")
    assert speedup > 3.0, f"expected a large win, measured {speedup:.2f}x"
