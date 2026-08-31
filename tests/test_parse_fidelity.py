# tests/test_parse_fidelity.py
"""What the URDF parser keeps, and what it admits to changing.

Two failures motivated this file, both silent, which is the dangerous kind.
Only the first <collision> element of a link survived parsing, so a base plate
written as three boxes was collision checked as one and a planner would fly a
path straight through the other two. And the parser quietly repaired malformed
joints, inventing limits for a revolute that declared none and swapping
inverted ones, without recording that it had done so, which is the difference
between a helpful loader and one that hides a broken file from you.
"""
import torch

import kinfast
from kinfast.collision_auto import auto_spheres, unsupported_links
from kinfast.compile import compile_robot
from kinfast.ir import Geometry, geometries
from kinfast.urdf.parse import parse_urdf_string


def _link(body):
    return f'<robot name="t"><link name="base">{body}</link></robot>'


def _joint(attrs, extra=""):
    return (f'<robot name="t"><link name="a"/><link name="b"/>'
            f'<joint name="j" {attrs}><parent link="a"/><child link="b"/>'
            f'<axis xyz="0 0 1"/>{extra}</joint></robot>')


BOX = '<geometry><box size="0.4 0.4 0.4"/></geometry>'
THREE_BOXES = _link(
    f"<collision>{BOX}</collision>"
    f'<collision><origin xyz="1 0 0"/>{BOX}</collision>'
    f'<collision><origin xyz="2 0 0"/>{BOX}</collision>')


def test_every_collision_element_is_kept():
    ir = parse_urdf_string(THREE_BOXES)
    assert len(geometries(ir.links["base"].collision)) == 3


def test_every_visual_element_is_kept():
    ir = parse_urdf_string(_link(
        f"<visual>{BOX}</visual>"
        f'<visual><origin xyz="1 0 0"/>{BOX}</visual>'))
    assert len(geometries(ir.links["base"].visual)) == 2


def test_a_single_shape_is_still_stored_unwrapped():
    """The common link keeps exactly the shape it always had, so nothing that
    reads the slot directly has to change."""
    ir = parse_urdf_string(_link(f"<collision>{BOX}</collision>"))
    assert isinstance(ir.links["base"].collision, Geometry)


def test_spheres_cover_the_shapes_that_used_to_be_dropped():
    """The reason the above matters. Before, only the box at the origin was
    covered and an obstacle two metres away sat inside geometry nothing knew
    about."""
    ir = parse_urdf_string(THREE_BOXES)
    chain = compile_robot(ir)
    spheres = auto_spheres(ir, chain, spacing=0.2)
    centres = torch.tensor([s[:3] for s in spheres[chain.link_index["base"]]])
    assert float(centres[:, 0].max()) > 1.8
    assert float(centres[:, 0].min()) < 0.2


def test_a_link_is_only_unsupported_when_nothing_in_it_is_usable():
    """A mesh next to a box is not a hole; the box is still checkable."""
    ir = parse_urdf_string(_link(
        '<collision><geometry><mesh filename="arm.stl"/></geometry></collision>'
        f"<collision>{BOX}</collision>"))
    chain = compile_robot(ir)
    assert unsupported_links(ir, chain) == {}


def test_a_link_of_only_meshes_is_still_reported():
    ir = parse_urdf_string(_link(
        '<collision><geometry><mesh filename="a.stl"/></geometry></collision>'
        '<collision><geometry><mesh filename="b.stl"/></geometry></collision>'))
    chain = compile_robot(ir)
    assert unsupported_links(ir, chain) == {"base": "mesh"}


def test_emitted_mjcf_carries_every_shape():
    from kinfast.mjcf.emit import emit_mjcf
    xml = emit_mjcf(parse_urdf_string(THREE_BOXES))
    assert xml.count("<geom") == 3


def _notes(xml):
    return " | ".join(parse_urdf_string(xml).parse_notes)


def test_multiple_collisions_are_noted():
    assert "3 collision elements" in _notes(THREE_BOXES)


def test_a_missing_limit_is_noted():
    """The parser assumes a range here. That is a reasonable rescue for a file
    that is technically invalid, but the number is invented and the caller has
    to be able to find that out."""
    notes = _notes(_joint('type="revolute"'))
    assert "no <limit>" in notes and "assumed" in notes


def test_inverted_limits_are_noted():
    notes = _notes(_joint('type="revolute"',
                          '<limit lower="1.5" upper="-1.5" velocity="1" effort="1"/>'))
    assert "inverted" in notes


def test_a_zero_width_limit_is_noted():
    notes = _notes(_joint('type="revolute"',
                          '<limit lower="0" upper="0" velocity="1" effort="1"/>'))
    assert "cannot move" in notes


def test_a_duplicate_link_name_is_noted():
    notes = _notes('<robot name="t"><link name="a"/><link name="a"/></robot>')
    assert "more than once" in notes


def test_a_well_formed_file_is_noted_about_nothing():
    """The notes are only worth reading if a clean file produces none."""
    ir = parse_urdf_string(_joint(
        'type="revolute"', '<limit lower="-1" upper="1" velocity="1" effort="1"/>'))
    assert ir.parse_notes == []
