# tests/test_regressions_fk_input_validation.py
"""Regressions for two FK bugs.

1. The per-(device, dtype) constants cache never invalidated, so an in-place
   edit of chain.joint_origin after the first fk call was silently ignored by
   the torch path while the codegen path saw it. The cache is now keyed on a
   fingerprint: the chain's _version counter, which CompiledChain.to() and
   invalidate_cache() bump, plus each source tensor's storage pointer and
   torch in-place version counter, which catch a direct edit with no
   cooperation from the caller.
2. An integer q, or an unbatched (dof,) q, failed deep inside torch with an
   unrelated message ("too many indices for tensor of dimension 1",
   "linalg.vector_norm: Expected a floating point ... Got Long"). Both now
   raise up front, naming the actual shape and the chain's dof.

Oracles are hand-computed: the fixture is a 2R planar arm whose end effector
position at q = 0 is known exactly, so the post-mutation expectation is
arithmetic, not another kinfast call.
"""
import math
import pytest
import torch

from kinfast import load_string
from kinfast.compile import compile_robot
from kinfast.fk import forward_kinematics, fk_rp
from kinfast.urdf.parse import parse_urdf_string

# 2R planar arm: joint1 at the origin, joint2 one metre along +x, ee one more
# metre along +x. At q = 0 the ee sits at (2, 0, 0).
PLANAR = """
<robot name="planar2r">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <link name="ee"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <joint name="tip" type="fixed">
    <parent link="l2"/><child link="ee"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


def _chain():
    return compile_robot(parse_urdf_string(PLANAR), dtype=torch.float64)


def _ee(chain, q):
    return forward_kinematics(chain, q)[:, chain.link_index["ee"], :3, 3]


# ---- bug 1: stale constants cache ----

def test_hand_computed_baseline():
    """Sanity anchor for the mutation tests below: (2, 0, 0) at q = 0."""
    chain = _chain()
    q = torch.zeros(1, 2, dtype=torch.float64)
    assert torch.allclose(_ee(chain, q)[0],
                          torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))


@pytest.mark.parametrize("announce", [False, True])
def test_joint_origin_edit_is_seen(announce):
    """The exact failure: without the fix the second call returned the stale
    (2, 0, 0). announce=False is the case that used to be silent."""
    chain = _chain()
    q = torch.zeros(1, 2, dtype=torch.float64)
    _ee(chain, q)                                   # warms the constants cache

    li = chain.link_index["l2"]
    chain.joint_origin[li, 0, 3] += 5.0             # j2 origin x: 1 -> 6
    if announce:
        chain.invalidate_cache()

    # Hand computed: base -> l1 at 0, l1 -> l2 at x = 6, l2 -> ee at x = 7.
    got = _ee(chain, q)[0]
    assert torch.allclose(got, torch.tensor([7.0, 0.0, 0.0], dtype=torch.float64))


def test_whole_tensor_replacement_is_seen():
    """Rebinding the attribute rather than editing through it: the storage
    pointer in the fingerprint is what catches this one."""
    chain = _chain()
    q = torch.zeros(1, 2, dtype=torch.float64)
    _ee(chain, q)

    origin = chain.joint_origin.clone()
    origin[chain.link_index["l2"], 0, 3] = 6.0
    chain.joint_origin = origin
    assert torch.allclose(_ee(chain, q)[0],
                          torch.tensor([7.0, 0.0, 0.0], dtype=torch.float64))


def test_joint_axis_edit_is_seen_after_invalidate_cache():
    """The axis feeds both the Rodrigues fold and the prismatic direction, so
    it has its own cached copies."""
    chain = _chain()
    q = torch.tensor([[math.pi / 2, 0.0]], dtype=torch.float64)
    # About +z: the arm folds onto +y, ee at (0, 2, 0).
    assert torch.allclose(_ee(chain, q)[0],
                          torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64),
                          atol=1e-12)

    chain.joint_axis[chain.link_index["l1"]] = torch.tensor(
        [0.0, 0.0, -1.0], dtype=torch.float64)
    chain.invalidate_cache()
    # About -z now, so the same +pi/2 folds onto -y.
    assert torch.allclose(_ee(chain, q)[0],
                          torch.tensor([0.0, -2.0, 0.0], dtype=torch.float64),
                          atol=1e-12)


def test_cache_survives_repeat_calls_without_invalidation():
    """Version keying must not turn every call into a cache miss."""
    chain = _chain()
    q = torch.zeros(4, 2, dtype=torch.float64)
    _ee(chain, q)
    first = chain._fk_cache
    _ee(chain, q)
    assert chain._fk_cache is first
    assert len(first[1]) == 1


def test_to_bumps_version_and_drops_the_cache():
    chain = _chain()
    q = torch.zeros(1, 2, dtype=torch.float64)
    _ee(chain, q)
    before = chain._version
    stale = chain._fk_cache
    chain.to("cpu")
    assert chain._version > before
    # Still correct after the reload, and served from a fresh cache entry.
    assert torch.allclose(_ee(chain, q)[0],
                          torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))
    assert chain._fk_cache is not stale
    assert chain._fk_cache[0][0] == chain._version


def test_torch_and_codegen_agree_after_a_mutation():
    """The original symptom: codegen regenerated from the edited chain while
    the torch path kept serving cached constants, so the two disagreed. No
    invalidate_cache() here on purpose, that is the whole point."""
    from kinfast.codegen import CompiledRobot
    chain = _chain()
    q = torch.tensor([[0.3, -0.7]], dtype=torch.float64)
    _ee(chain, q)

    chain.joint_origin[chain.link_index["l2"], 0, 3] += 5.0

    gen = CompiledRobot(chain)                      # scalar path, single q
    li = chain.link_index["ee"]
    gen_p = torch.tensor(gen.fk([0.3, -0.7], link=li)[:3, 3], dtype=torch.float64)
    assert torch.allclose(gen_p, _ee(chain, q)[0], atol=1e-10)

    # And both match the hand-computed 2R forward map with l1 = 6, l2 = 1.
    a, b = 0.3, 0.3 - 0.7
    expect = torch.tensor([6 * math.cos(a) + math.cos(b),
                           6 * math.sin(a) + math.sin(b), 0.0],
                          dtype=torch.float64)
    assert torch.allclose(_ee(chain, q)[0], expect, atol=1e-12)


def test_invalidate_cache_returns_self_and_is_safe_before_any_fk():
    chain = _chain()
    assert chain.invalidate_cache() is chain
    q = torch.zeros(1, 2, dtype=torch.float64)
    assert torch.allclose(_ee(chain, q)[0],
                          torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))


# ---- bug 2: q input validation ----

def test_integer_q_raises_typeerror_naming_the_dtype():
    chain = _chain()
    with pytest.raises(TypeError) as exc:
        forward_kinematics(chain, torch.zeros(1, 2, dtype=torch.long))
    msg = str(exc.value)
    assert "float" in msg
    assert "int64" in msg


def test_unbatched_q_raises_valueerror_naming_shape_and_dof():
    chain = _chain()
    with pytest.raises(ValueError) as exc:
        forward_kinematics(chain, torch.zeros(2, dtype=torch.float64))
    msg = str(exc.value)
    assert "(2,)" in msg
    assert "(B, 2)" in msg
    assert "unsqueeze(0)" in msg


def test_wrong_dof_width_raises_valueerror():
    chain = _chain()
    with pytest.raises(ValueError) as exc:
        forward_kinematics(chain, torch.zeros(3, 5, dtype=torch.float64))
    msg = str(exc.value)
    assert "(3, 5)" in msg
    assert "(B, 2)" in msg


def test_three_dim_q_raises_valueerror():
    chain = _chain()
    with pytest.raises(ValueError):
        forward_kinematics(chain, torch.zeros(1, 1, 2, dtype=torch.float64))


def test_non_tensor_q_raises_typeerror():
    chain = _chain()
    with pytest.raises(TypeError):
        forward_kinematics(chain, [[0.0, 0.0]])


def test_fk_rp_validates_too():
    chain = _chain()
    with pytest.raises(ValueError):
        fk_rp(chain, torch.zeros(2, dtype=torch.float64))
    with pytest.raises(TypeError):
        fk_rp(chain, torch.zeros(1, 2, dtype=torch.int32))


def test_validation_reaches_the_robot_facade():
    r = load_string(PLANAR)
    with pytest.raises(TypeError):
        r.fk_all(torch.zeros(1, r.dof, dtype=torch.long))
    with pytest.raises(ValueError):
        r.fk_all(torch.zeros(r.dof))


def test_valid_q_still_works_in_both_float_dtypes():
    chain = _chain()
    for dtype in (torch.float32, torch.float64):
        q = torch.zeros(3, 2, dtype=dtype)
        out = forward_kinematics(chain, q)
        assert out.shape == (3, chain.n_links, 4, 4)
        assert out.dtype == dtype


def test_zero_dof_chain_accepts_empty_q():
    fixed = """
    <robot name="fixed">
      <link name="base"/><link name="tip"/>
      <joint name="j" type="fixed">
        <parent link="base"/><child link="tip"/>
        <origin xyz="0 0 0.5" rpy="0 0 0"/>
      </joint>
    </robot>
    """
    chain = compile_robot(parse_urdf_string(fixed), dtype=torch.float64)
    out = forward_kinematics(chain, torch.zeros(2, 0, dtype=torch.float64))
    assert torch.allclose(out[:, chain.link_index["tip"], :3, 3],
                          torch.tensor([[0.0, 0.0, 0.5]] * 2, dtype=torch.float64))
