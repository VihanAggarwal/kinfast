# tests/test_menagerie.py
"""Menagerie-scale MJCF checks. Runs only when examples/assets/menagerie is
populated (python examples/menagerie.py --fetch); skips cleanly otherwise.

parse tier: every downloaded production model XML must load (except the
documented ball-joint models, which must fail with a clear error).
oracle tier: models with full assets are FK-compared against MuJoCo itself.
"""
import glob
import os

import pytest

DEST = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "menagerie")
XMLS = sorted(p for p in glob.glob(os.path.join(DEST, "*", "*.xml"))
              if os.path.basename(p) != "scene.xml")   # scenes get their own tier
SCENES = sorted(glob.glob(os.path.join(DEST, "*", "scene.xml")))
BALL_JOINT_MODELS = {"cassie"}          # documented non-feature

pytestmark = pytest.mark.skipif(not XMLS, reason="menagerie not fetched")


@pytest.mark.parametrize("path", XMLS,
                         ids=[os.path.basename(p) for p in XMLS])
def test_menagerie_parses(path):
    import kinfast
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in BALL_JOINT_MODELS:
        with pytest.raises(ValueError, match="ball joints"):
            kinfast.load(path)
        return
    robot = kinfast.load(path)
    assert robot.dof > 0
    assert robot.n_links > 1


@pytest.mark.skipif(not SCENES, reason="no menagerie scene.xml fetched")
@pytest.mark.parametrize("path", SCENES,
                         ids=[os.path.basename(os.path.dirname(p))
                              for p in SCENES])
def test_menagerie_scene_matches_its_robot_xml(path):
    """A Menagerie scene.xml owns no bodies: a light, a floor, and an
    <include> that pulls the robot in from the file beside it. Loading one
    used to return a silent 0-dof robot with a single link. It must now build
    exactly the chain the robot's own xml builds."""
    import kinfast
    siblings = [p for p in glob.glob(os.path.join(os.path.dirname(path), "*.xml"))
                if os.path.basename(p) != "scene.xml"]
    if len(siblings) != 1:
        pytest.skip("expected exactly one robot xml beside the scene")
    scene = kinfast.load(path)
    robot = kinfast.load(siblings[0])
    assert robot.dof > 0
    assert scene.dof == robot.dof
    assert scene.n_links == robot.n_links
    assert scene.joint_names == robot.joint_names
    assert scene.chain.link_names == robot.chain.link_names


@pytest.mark.parametrize("model", ["trs_so_arm100", "universal_robots_ur5e"])
def test_menagerie_oracle(model):
    mujoco = pytest.importorskip("mujoco")
    xmls = glob.glob(os.path.join(DEST, model, "*.xml"))
    assets = glob.glob(os.path.join(DEST, model, "assets", "*"))
    if not xmls or not assets:
        pytest.skip(f"{model} full dir not fetched")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from menagerie import oracle_tier, PARSE_MODELS
    worst = oracle_tier(os.path.join(DEST, model), PARSE_MODELS[model])
    assert worst < 1e-5
