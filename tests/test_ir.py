# tests/test_ir.py
from kinfast.ir import Robot, Link, Joint

def test_build_two_link_ir():
    robot = Robot(
        name="r",
        links={"base": Link("base"), "l1": Link("l1"), "l2": Link("l2")},
        joints=[
            Joint("j1", "revolute", "base", "l1", (0, 0, 0), (0, 0, 0),
                  (0, 0, 1), (-3.0, 3.0), 10.0, 5.0),
            Joint("j2", "revolute", "l1", "l2", (1, 0, 0), (0, 0, 0),
                  (0, 0, 1), (-3.0, 3.0), 10.0, 5.0),
        ],
    )
    assert robot.root_link() == "base"
    assert [j.name for j in robot.movable_joints()] == ["j1", "j2"]
    assert robot.dof() == 2
