# src/kinfast/__init__.py  (replace file contents)
from kinfast.robot import Robot, load, load_string


def to_mjcf(path, out=None, **kw):
    """Convert a URDF/xacro file to MJCF. Returns the MJCF string; writes it to
    `out` if given. Keyword args are passed to Robot.to_mjcf."""
    text = load(path).to_mjcf(**kw)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    return text

__version__ = "0.2.0"
__all__ = ["Robot", "load", "load_string", "to_mjcf", "__version__"]
