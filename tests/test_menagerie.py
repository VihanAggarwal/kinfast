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
              if os.path.basename(p) != "scene.xml")   # scenes are include-wrappers
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
