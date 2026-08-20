# tests/test_compile.py
import torch
from kinfast.urdf.parse import parse_urdf_string
from kinfast.compile import compile_robot
from tests.test_parse import TWO_LINK

def test_compiled_shapes():
    chain = compile_robot(parse_urdf_string(TWO_LINK))
    assert chain.n_links == 3
    assert chain.dof == 2
    assert chain.joint_origin.shape == (3, 4, 4)
    assert chain.joint_axis.shape == (3, 3)
    assert chain.parent.shape == (3,)
    # topo order lists every link, parents before children
    seen = set()
    for i in chain.topo_order:
        p = int(chain.parent[i])
        if p >= 0:
            assert p in seen
        seen.add(i)
    assert len(seen) == 3

def test_q_index_only_movable():
    chain = compile_robot(parse_urdf_string(TWO_LINK))
    movable = (chain.q_index >= 0).sum().item()
    assert movable == 2
