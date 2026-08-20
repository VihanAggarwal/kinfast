# tests/fixtures/download_panda.py
"""Fetch a Franka Panda URDF for the integration test and demo.

Run:  python tests/fixtures/download_panda.py
Places panda.urdf (+ any meshes it references are optional for kinematics) in
examples/assets/. Kinematics needs only the URDF, not the meshes.

Recommended source: the Franka description package or the MuJoCo Menagerie's
franka_emika_panda (convert MJCF->URDF is NOT needed; use a URDF distribution).
Verify the license before committing the file to your repo.
"""
import os, urllib.request

DEST = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "assets")
URL = os.environ.get("PANDA_URDF_URL", "")  # set to a URDF raw URL you trust

def main():
    os.makedirs(DEST, exist_ok=True)
    if not URL:
        print("Set PANDA_URDF_URL to a trusted raw URDF URL, then rerun.")
        return
    out = os.path.join(DEST, "panda.urdf")
    urllib.request.urlretrieve(URL, out)
    print("wrote", out)

if __name__ == "__main__":
    main()
