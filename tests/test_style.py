# tests/test_style.py
"""House style, enforced so it does not drift back.

Two rules, both about how the source reads rather than how it runs:

no em or en dashes
    They are easy to type by accident and they read as machine written. A
    comma, a colon, or a second sentence says the same thing.

no decorative emoji
    Output that goes in a terminal or a docstring stays plain text.

Only files git tracks are checked, so a stale build directory or a downloaded
robot cannot fail the suite.
"""
import subprocess

import pytest

BANNED = {
    "—": "em dash",
    "–": "en dash",
    "✅": "check mark emoji",
    "\U0001F680": "rocket emoji",
    "✨": "sparkles emoji",
    "\U0001F3AF": "target emoji",
}

CHECKED_SUFFIXES = (".py", ".md", ".toml", ".yml", ".yaml", ".cff", ".txt")


def tracked_files():
    """Paths git knows about. Anything untracked is not ours to police."""
    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available here")
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return [p for p in out.stdout.splitlines()
            if p.endswith(CHECKED_SUFFIXES)]


def test_no_banned_characters():
    offenders = []
    for path in tracked_files():
        if path == "tests/test_style.py":      # this file names them on purpose
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        for ch, name in BANNED.items():
            if ch in text:
                line = next((i for i, l in enumerate(text.splitlines(), 1)
                             if ch in l), 0)
                offenders.append(f"{path}:{line} contains a {name}")
    assert not offenders, "\n".join(offenders)


def test_the_check_actually_looks_at_files():
    """A guard on the guard: if the file list ever comes back empty the test
    above would pass while checking nothing."""
    files = tracked_files()
    assert len(files) > 20
    assert any(f.startswith("src/kinfast/") for f in files)
