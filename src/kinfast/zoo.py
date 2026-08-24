# src/kinfast/zoo.py
"""A registry of public robot descriptions, fetched on demand and cached.

Every entry points at a file in a public repository (bullet3's pybullet_data,
the MuJoCo Menagerie, and a few vendor repos). Nothing is vendored into this
package: the first time you ask for a robot it is downloaded into
``~/.cache/kinfast`` and every later call reads the cached copy. That keeps the
wheel small and keeps the licenses with their owners, at the price of needing
the network once per robot.

Typical use::

    from kinfast import zoo

    zoo.list()                      # names you can ask for
    p = zoo.path("panda")           # local file, downloading if needed
    robot = zoo.load("panda")       # same thing, already loaded

The registry lists what is available upstream, not only what kinfast can
already handle. Every entry was fetched and loaded while this module was
written and 32 of the 33 load; the exception is agility_cassie, whose ball
joints the MJCF parser rejects on purpose. Its Entry description says so.

The download path prefers urllib. On machines where Python cannot validate the
certificate chain (a common Windows setup where the system trust store is not
visible to Python's bundled CA file) urllib raises an SSL error and the code
falls back to the ``curl`` binary, which uses the OS trust store. HTTP errors
such as 404 are not retried with curl, since a missing file is missing either
way.
"""
import difflib
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field

__all__ = [
    "Entry", "REGISTRY", "ZooError", "UnknownRobot", "DownloadError",
    "list", "names", "info", "cache_dir", "path", "load", "clear_cache",
]


class ZooError(RuntimeError):
    """Base class for every error this module raises on its own."""


class UnknownRobot(ZooError, KeyError):
    """Raised for a name that is not in the registry.

    It subclasses KeyError as well so `except KeyError` around a lookup keeps
    working, but str() gives the readable message rather than a quoted repr.
    """

    def __str__(self):
        return self.args[0] if self.args else ""


class DownloadError(ZooError):
    """Raised when a file could not be fetched by urllib or by curl."""


@dataclass(frozen=True)
class Entry:
    """One robot in the registry.

    `url` is the file kinfast actually loads. `extra` names sibling files that
    the main file includes and that therefore have to land in the same cache
    directory (MJCF ``<include>`` targets); they are given as URLs and are
    saved under their own basenames. `format` is "urdf" or "mjcf" and is
    informational: kinfast sniffs the real format when loading.
    """

    name: str
    url: str
    format: str
    source: str
    description: str
    extra: tuple = field(default_factory=tuple)

    @property
    def filename(self):
        """Basename the file is cached under, taken from the URL."""
        return self.url.rsplit("/", 1)[-1]


_BULLET = ("https://raw.githubusercontent.com/bulletphysics/bullet3/master/"
           "examples/pybullet/gym/pybullet_data")
_MENAGERIE = ("https://raw.githubusercontent.com/google-deepmind/"
              "mujoco_menagerie/main")
_SO_ARM = ("https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/"
           "Simulation")
_PK = ("https://raw.githubusercontent.com/UM-ARM-Lab/pytorch_kinematics/"
       "master/tests")

_BULLET_SRC = "bulletphysics/bullet3 (pybullet_data)"
_MENAGERIE_SRC = "google-deepmind/mujoco_menagerie"


def _urdf(name, url, source, description, extra=()):
    return Entry(name, url, "urdf", source, description, tuple(extra))


def _mjcf(name, url, source, description, extra=()):
    return Entry(name, url, "mjcf", source, description, tuple(extra))


def _menagerie(name, xml, description, extra=()):
    return _mjcf(name, f"{_MENAGERIE}/{name}/{xml}", _MENAGERIE_SRC,
                 description,
                 tuple(f"{_MENAGERIE}/{name}/{e}" for e in extra))


_ENTRIES = (
    # ---- URDF, mostly bullet3's pybullet_data ----
    _urdf("panda", f"{_BULLET}/franka_panda/panda.urdf", _BULLET_SRC,
          "Franka Emika Panda, 7-dof arm with a parallel gripper"),
    _urdf("kuka_iiwa", f"{_BULLET}/kuka_iiwa/model.urdf", _BULLET_SRC,
          "KUKA LBR iiwa, 7-dof arm"),
    _urdf("xarm6", f"{_BULLET}/xarm/xarm6_robot.urdf", _BULLET_SRC,
          "UFACTORY xArm6, 6-dof arm"),
    _urdf("laikago", f"{_BULLET}/laikago/laikago.urdf", _BULLET_SRC,
          "Unitree Laikago quadruped"),
    _urdf("a1", f"{_BULLET}/a1/a1.urdf", _BULLET_SRC,
          "Unitree A1 quadruped"),
    _urdf("minitaur", f"{_BULLET}/quadruped/minitaur.urdf", _BULLET_SRC,
          "Ghost Robotics Minitaur, a closed-chain quadruped"),
    _urdf("racecar", f"{_BULLET}/racecar/racecar.urdf", _BULLET_SRC,
          "MIT racecar, a small wheeled platform"),
    _urdf("r2d2", f"{_BULLET}/r2d2.urdf", _BULLET_SRC,
          "The R2D2 toy model that ships with pybullet"),
    _urdf("cartpole", f"{_BULLET}/cartpole.urdf", _BULLET_SRC,
          "Cartpole, one slide joint and one hinge"),
    _urdf("husky", f"{_BULLET}/husky/husky.urdf", _BULLET_SRC,
          "Clearpath Husky, a skid-steer base"),
    _urdf("ur5", f"{_PK}/ur5.urdf", "UM-ARM-Lab/pytorch_kinematics",
          "Universal Robots UR5, 6-dof arm"),
    _urdf("so101", f"{_SO_ARM}/SO101/so101_new_calib.urdf",
          "TheRobotStudio/SO-ARM100",
          "SO-ARM101, the low-cost 5-dof teaching arm"),
    _urdf("so100", f"{_SO_ARM}/SO100/so100.urdf", "TheRobotStudio/SO-ARM100",
          "SO-ARM100, the earlier revision of the same arm"),

    # ---- MJCF, the MuJoCo Menagerie ----
    _menagerie("franka_emika_panda", "panda.xml",
               "Franka Emika Panda as modelled for MuJoCo"),
    _menagerie("franka_fr3", "fr3.xml", "Franka Research 3, 7-dof arm"),
    _menagerie("universal_robots_ur5e", "ur5e.xml",
               "Universal Robots UR5e, 6-dof arm"),
    _menagerie("universal_robots_ur10e", "ur10e.xml",
               "Universal Robots UR10e, 6-dof arm"),
    _menagerie("kuka_iiwa_14", "iiwa14.xml", "KUKA LBR iiwa 14, 7-dof arm"),
    _menagerie("kinova_gen3", "gen3.xml", "Kinova Gen3, 7-dof arm"),
    _menagerie("ufactory_xarm7", "xarm7.xml", "UFACTORY xArm7, 7-dof arm"),
    _menagerie("rethink_robotics_sawyer", "sawyer.xml",
               "Rethink Robotics Sawyer, 7-dof arm"),
    _menagerie("unitree_go2", "go2.xml", "Unitree Go2 quadruped"),
    _menagerie("unitree_g1", "g1.xml", "Unitree G1 humanoid"),
    _menagerie("anybotics_anymal_c", "anymal_c.xml", "ANYbotics ANYmal C"),
    _menagerie("boston_dynamics_spot", "spot.xml", "Boston Dynamics Spot"),
    _menagerie("agility_cassie", "cassie.xml",
               "Agility Robotics Cassie, a closed-chain biped. Listed but not "
               "loadable today: its achilles rods use ball joints, which the "
               "MJCF parser rejects rather than silently approximating"),
    _menagerie("shadow_hand", "right_hand.xml",
               "Shadow Dexterous Hand, right hand"),
    _menagerie("robotiq_2f85", "2f85.xml",
               "Robotiq 2F-85 parallel gripper"),
    _menagerie("trs_so_arm100", "so_arm100.xml",
               "SO-ARM100 as modelled for MuJoCo"),
    _menagerie("aloha", "aloha.xml",
               "ALOHA bimanual station, two ViperX arms on a frame",
               extra=("joint_position_actuators.xml", "keyframe_ctrl.xml")),
    _menagerie("hello_robot_stretch", "stretch.xml",
               "Hello Robot Stretch, a mobile manipulator"),
    _menagerie("google_barkour_vb", "barkour_vb.xml",
               "Google Barkour vb quadruped"),
    _menagerie("pal_tiago", "tiago.xml",
               "PAL Robotics TIAGo, a mobile manipulator"),
)

REGISTRY = {e.name: e for e in _ENTRIES}
if len(REGISTRY) != len(_ENTRIES):  # pragma: no cover - guards a typo at import
    raise RuntimeError("duplicate name in the kinfast robot zoo registry")


# ---- lookup ----
def names():
    """Every registry name, sorted. Same as `list()` with no filters."""
    return sorted(REGISTRY)


def list(format=None, contains=None):
    """Names in the registry, sorted.

    `format` keeps only "urdf" or only "mjcf" entries. `contains` keeps names
    containing that substring, which is the quickest way to find the spelling
    of a robot you half remember.
    """
    out = names()
    if format is not None:
        fmt = format.lower()
        out = [n for n in out if REGISTRY[n].format == fmt]
    if contains is not None:
        needle = contains.lower()
        out = [n for n in out if needle in n.lower()]
    return out


def info(name):
    """The Entry for `name`, with a helpful error for a name that is not here."""
    try:
        return REGISTRY[name]
    except (KeyError, TypeError):
        pass
    known = names()
    close = difflib.get_close_matches(str(name), known, n=3, cutoff=0.5)
    msg = f"unknown robot {name!r}"
    if close:
        msg += "; did you mean " + " or ".join(repr(c) for c in close) + "?"
    msg += f"; the zoo has {len(known)} robots, see kinfast.zoo.list()"
    raise UnknownRobot(msg)


# ---- cache ----
def cache_dir():
    """Directory the zoo downloads into.

    ``KINFAST_CACHE`` wins if it is set, then ``XDG_CACHE_HOME``/kinfast, and
    otherwise ``~/.cache/kinfast``. The directory is not created here; `path`
    creates it when it actually has something to write.
    """
    env = os.environ.get("KINFAST_CACHE")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return os.path.join(os.path.abspath(os.path.expanduser(xdg)), "kinfast")
    return os.path.join(os.path.expanduser("~"), ".cache", "kinfast")


def _cache_root(explicit=None):
    """The cache root a call should use: what the caller passed, else the
    default from `cache_dir`. Exists because the public functions take a
    `cache_dir` argument that shadows the function of the same name."""
    return explicit if explicit is not None else cache_dir()


def _entry_dir(entry, root=None):
    """Where one robot's files live: one directory per registry name, so the
    sibling files an MJCF includes sit next to it and cannot collide with
    another robot's files of the same name."""
    return os.path.join(_cache_root(root), entry.name)


# ---- download ----
def _http_get(url, timeout=60.0):
    """Fetch `url` and return its bytes, using curl if urllib's TLS fails.

    Separated out so tests can replace it, and so the SSL fallback lives in one
    place rather than being repeated per call site.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        # The server answered, it just said no. curl would say no as well.
        raise DownloadError(f"HTTP {e.code} for {url}") from e
    except OSError as e:
        # urllib.error.URLError and ssl.SSLError are both OSError subclasses,
        # so this one clause covers the DNS, socket, and certificate failures.
        first = e
    return _curl_get(url, timeout, first)


def _curl_get(url, timeout, first):
    """Second attempt through the curl binary, which uses the OS trust store."""
    exe = shutil.which("curl")
    if exe is None:
        raise DownloadError(
            f"could not fetch {url}: {first}; curl is not installed either, so "
            "there is no fallback. Download the file by hand and point "
            "kinfast.load at it.") from first
    try:
        r = subprocess.run(
            [exe, "-fsSL", "--max-time", str(int(timeout)), url],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        raise DownloadError(f"could not fetch {url}: {first}; "
                            f"curl also failed to start: {e}") from first
    if r.returncode != 0 or not r.stdout:
        detail = (r.stderr or b"").decode("utf-8", "replace").strip()
        raise DownloadError(
            f"could not fetch {url}: urllib said {first}; "
            f"curl exited {r.returncode} {detail}".rstrip()) from first
    return r.stdout


def _download(url, dest, timeout=60.0):
    """Fetch `url` to `dest`, writing it in one move so an interrupted download
    never leaves a half file that later runs would treat as cached."""
    data = _http_get(url, timeout=timeout)
    if not data:
        raise DownloadError(f"{url} returned an empty file")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = f"{dest}.part{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def path(name, cache_dir=None, force=False, timeout=60.0):
    """Local path to `name`'s description file, downloading it if it is missing.

    The file, and any sibling files it includes, land in
    ``<cache>/<name>/``. Nothing is re-downloaded once it is there; pass
    `force=True` to refresh a robot whose upstream file has changed.
    """
    entry = info(name)
    root = _entry_dir(entry, cache_dir)
    main = os.path.join(root, entry.filename)
    wanted = [(entry.url, main)]
    wanted += [(u, os.path.join(root, u.rsplit("/", 1)[-1])) for u in entry.extra]
    for url, dest in wanted:
        if force or not os.path.exists(dest):
            _download(url, dest, timeout=timeout)
    return main


def load(name, cache_dir=None, force=False, timeout=60.0, **kw):
    """Download `name` if needed and load it as a kinfast Robot.

    Extra keyword arguments go straight to `kinfast.load`, so
    ``zoo.load("panda", dtype=torch.float64)`` gives a double-precision chain
    and ``ee_link=`` picks the end effector.
    """
    # Imported here, not at module scope, because kinfast/__init__ imports the
    # heavy modules and the zoo is meant to be cheap to import and inspect.
    from kinfast.robot import load as _load
    return _load(path(name, cache_dir=cache_dir, force=force, timeout=timeout),
                 **kw)


def clear_cache(name=None, cache_dir=None):
    """Delete cached files, for one robot or for all of them.

    Returns the paths that were removed, so a script can report what it freed.
    Removing nothing is not an error.
    """
    removed = []
    if name is None:
        root = _cache_root(cache_dir)
        for entry in _ENTRIES:
            d = os.path.join(root, entry.name)
            if os.path.isdir(d):
                shutil.rmtree(d)
                removed.append(d)
        return removed
    d = _entry_dir(info(name), cache_dir)
    if os.path.isdir(d):
        shutil.rmtree(d)
        removed.append(d)
    return removed
