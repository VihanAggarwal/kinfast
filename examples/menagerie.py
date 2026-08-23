# examples/menagerie.py
"""Sweep the MuJoCo Menagerie: parse real production MJCF models with kinfast
and, where the full model directory (with mesh assets) is downloaded, verify
forward kinematics body-by-body against MuJoCo itself.

Two tiers, honestly labeled in the output:
- parse tier: model XML only. Checks that kinfast loads it and reports dof.
- oracle tier: full directory. MuJoCo loads the real model and every body's
  world transform is compared at random configurations.

Usage:
  python examples/menagerie.py --fetch          # download (XMLs + oracle dirs)
  python examples/menagerie.py                  # run the sweep
"""
import argparse
import glob
import os
import urllib.request

import numpy as np

BASE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"
DEST = "examples/assets/menagerie"

# parse tier: main model XML per directory
PARSE_MODELS = {
    "franka_emika_panda": "panda.xml",
    "franka_fr3": "fr3.xml",
    "universal_robots_ur5e": "ur5e.xml",
    "universal_robots_ur10e": "ur10e.xml",
    "kuka_iiwa_14": "iiwa14.xml",
    "kinova_gen3": "gen3.xml",
    "ufactory_xarm7": "xarm7.xml",
    "rethink_robotics_sawyer": "sawyer.xml",
    "unitree_go2": "go2.xml",
    "unitree_g1": "g1.xml",
    "anybotics_anymal_c": "anymal_c.xml",
    "boston_dynamics_spot": "spot.xml",
    "agility_cassie": "cassie.xml",
    "shadow_hand": "right_hand.xml",
    "robotiq_2f85": "2f85.xml",
    "trs_so_arm100": "so_arm100.xml",
    "aloha": "aloha.xml",
    "hello_robot_stretch": "stretch.xml",
    "google_barkour_vb": "barkour_vb.xml",
    "pal_tiago": "tiago.xml",
}

# oracle tier: full directories (mesh assets included) for mujoco comparison
ORACLE_MODELS = ["trs_so_arm100", "universal_robots_ur5e"]


def _fetch_file(rel, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"{BASE}/{rel}"
    try:
        urllib.request.urlretrieve(url, out)
    except Exception:
        # some Windows setups reject the cert chain in urllib; curl handles it
        import subprocess
        r = subprocess.run(["curl", "-sL", "--max-time", "120", "-o", out, url])
        if r.returncode != 0 or not os.path.exists(out):
            raise


def _fetch_dir(model):
    """Download a full model directory (including assets) via the GitHub API."""
    import json
    import subprocess
    listing = subprocess.run(
        ["gh", "api", f"repos/google-deepmind/mujoco_menagerie/contents/{model}"],
        capture_output=True, text=True)
    for item in json.loads(listing.stdout):
        if item["type"] == "file":
            _fetch_file(f"{model}/{item['name']}", f"{DEST}/{model}/{item['name']}")
        elif item["type"] == "dir":
            sub = subprocess.run(
                ["gh", "api",
                 f"repos/google-deepmind/mujoco_menagerie/contents/{model}/{item['name']}"],
                capture_output=True, text=True)
            for s in json.loads(sub.stdout):
                if s["type"] == "file":
                    _fetch_file(f"{model}/{item['name']}/{s['name']}",
                                f"{DEST}/{model}/{item['name']}/{s['name']}")


def fetch():
    for model, xml in PARSE_MODELS.items():
        try:
            _fetch_file(f"{model}/{xml}", f"{DEST}/{model}/{xml}")
            print(f"fetched {model}/{xml}")
        except Exception as e:
            print(f"FAILED  {model}/{xml}: {e}")
    for model in ORACLE_MODELS:
        print(f"fetching full dir {model} (assets included)...")
        _fetch_dir(model)


def parse_tier(path):
    """Load a model and return it with whatever the parser had to reinterpret.

    The notes are recorded on the IR by the MJCF parser, so read them through
    the loaded robot's parse_notes, which forwards to robot.ir.
    """
    import kinfast
    robot = kinfast.load(path)
    notes = list(robot.parse_notes)
    return robot, "; ".join(notes)


def oracle_tier(model_dir, xml):
    import mujoco
    import torch
    import kinfast
    from kinfast.fk import forward_kinematics

    m = mujoco.MjModel.from_xml_path(os.path.join(model_dir, xml))
    d = mujoco.MjData(m)
    robot = kinfast.load(os.path.join(model_dir, xml))
    chain = robot.chain
    addr = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            addr[nm] = m.jnt_qposadr[j]
    cols = [(k, addr[nm]) for k, nm in enumerate(chain.joint_names) if nm in addr]

    rng = np.random.RandomState(0)
    lo, hi = chain.lower.numpy(), chain.upper.numpy()
    worst = 0.0
    for _ in range(16):
        q = lo + (hi - lo) * rng.rand(chain.dof)
        d.qpos[:] = m.qpos0
        for k, a in cols:
            d.qpos[a] = q[k]
        mujoco.mj_forward(m, d)
        world = forward_kinematics(chain, torch.tensor(q, dtype=torch.float32).unsqueeze(0))[0]
        for b in range(1, m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm not in chain.link_index:
                continue
            li = chain.link_index[nm]
            dp = float(np.abs(world[li, :3, 3].numpy() - d.xpos[b]).max())
            dr = float(np.abs(world[li, :3, :3].numpy() - d.xmat[b].reshape(3, 3)).max())
            worst = max(worst, dp, dr)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default="examples/assets/MENAGERIE.md")
    args = ap.parse_args()
    if args.fetch:
        fetch()

    lines = [
        "# kinfast on the MuJoCo Menagerie",
        "",
        "Real production MJCF models from google-deepmind/mujoco_menagerie.",
        "parse = kinfast loads the model XML. oracle = full model directory",
        "downloaded and FK compared body-by-body against MuJoCo's mj_forward.",
        "Reproduce: `python examples/menagerie.py --fetch` then run again.",
        "",
        "| model | tier | dof | result |",
        "|---|---|---|---|",
    ]
    for model, xml in PARSE_MODELS.items():
        path = f"{DEST}/{model}/{xml}"
        if not os.path.exists(path):
            continue
        try:
            robot, notes = parse_tier(path)
            note = f" ({notes})" if notes else ""
            if model in ORACLE_MODELS and glob.glob(f"{DEST}/{model}/assets/*"):
                worst = oracle_tier(f"{DEST}/{model}", xml)
                lines.append(f"| {model} | oracle | {robot.dof} | "
                             f"max FK diff vs MuJoCo {worst:.1e}{note} |")
            else:
                lines.append(f"| {model} | parse | {robot.dof} | loads{note} |")
        except Exception as e:
            lines.append(f"| {model} | parse | | FAILED: {type(e).__name__}: {e} |")

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
