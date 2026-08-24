# tests/test_zoo.py
"""Offline tests for the robot zoo.

Nothing here touches the network. Every test that would download either
replaces `zoo._http_get` (the one place bytes come from) or pre-seeds a fake
cache directory, and several tests actively fail if a fetch is attempted. The
single real-download test is skipped unless KINFAST_NET=1 is set.

The URDF used through the fake cache is a planar 2R arm whose tip position has
a closed-form answer, so the load path is checked against hand arithmetic
rather than against kinfast's own output.
"""
import math
import os
import ssl
import subprocess
import urllib.error

import pytest
import torch

from kinfast import zoo


# ---- fixtures -------------------------------------------------------------

# base -> j1 -> l1 -> j2 (1m out) -> l2 -> fixed tip (1m out).
# Tip at (cos q1 + cos(q1+q2), sin q1 + sin(q1+q2), 0).
PLANAR_2R = """
<robot name="planar_2r">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
  <joint name="jtip" type="fixed">
    <parent link="l2"/><child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""

MINI_MJCF = """
<mujoco model="mini">
  <worldbody>
    <body name="upper" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
      <body name="fore" pos="0.4 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-120 10"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class FakeFetcher:
    """Stand-in for zoo._http_get that records calls and never uses a socket."""

    def __init__(self, body=b"<robot name='x'><link name='base'/></robot>"):
        self.body = body
        self.calls = []

    def __call__(self, url, timeout=60.0):
        self.calls.append(url)
        if isinstance(self.body, dict):
            return self.body[url]
        return self.body


def _seed(cache, name, text, filename=None):
    """Write `text` into the fake cache exactly where zoo.path expects it."""
    entry = zoo.info(name)
    d = os.path.join(str(cache), entry.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, filename or entry.filename)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


@pytest.fixture
def no_network(monkeypatch):
    """Make any real fetch attempt an immediate, loud test failure."""
    def boom(*a, **kw):
        raise AssertionError(f"the test tried to reach the network: {a}")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(zoo, "_http_get", boom)


# ---- registry integrity ---------------------------------------------------

def test_registry_keys_match_entry_names():
    for key, entry in zoo.REGISTRY.items():
        assert key == entry.name


def test_registry_names_are_unique_and_plausible():
    names = zoo.names()
    assert len(names) == len(set(names)) == len(zoo.REGISTRY)
    assert len(names) >= 30
    for n in names:
        assert n == n.strip() and " " not in n
        assert n == n.lower()


def test_registry_urls_are_https_and_unique():
    urls = []
    for entry in zoo.REGISTRY.values():
        for u in (entry.url,) + tuple(entry.extra):
            assert u.startswith("https://"), (entry.name, u)
            assert u == u.strip()
            urls.append(u)
    assert len(urls) == len(set(urls))


def test_registry_formats_and_filenames():
    for entry in zoo.REGISTRY.values():
        assert entry.format in ("urdf", "mjcf"), entry.name
        suffix = ".urdf" if entry.format == "urdf" else ".xml"
        assert entry.filename.endswith(suffix), entry.name
        assert entry.filename == os.path.basename(entry.filename)
        assert ".." not in entry.filename


def test_registry_extras_are_safe_sibling_names():
    for entry in zoo.REGISTRY.values():
        for u in entry.extra:
            base = u.rsplit("/", 1)[-1]
            assert base and base == os.path.basename(base)
            assert ".." not in base
            assert base != entry.filename


def test_registry_entries_are_documented():
    for entry in zoo.REGISTRY.values():
        assert entry.source.strip()
        assert len(entry.description.strip()) > 8


def test_registry_covers_both_formats():
    fmts = {e.format for e in zoo.REGISTRY.values()}
    assert fmts == {"urdf", "mjcf"}
    assert len(zoo.list(format="urdf")) >= 10
    assert len(zoo.list(format="mjcf")) >= 15


def test_aloha_carries_its_included_files():
    entry = zoo.info("aloha")
    assert len(entry.extra) == 2
    assert all(u.endswith(".xml") for u in entry.extra)


# ---- listing and lookup ---------------------------------------------------

def test_list_defaults_to_every_name_sorted():
    assert zoo.list() == sorted(zoo.REGISTRY) == zoo.names()


def test_list_filters():
    mjcf = zoo.list(format="mjcf")
    assert set(mjcf) <= set(zoo.names())
    assert all(zoo.REGISTRY[n].format == "mjcf" for n in mjcf)
    assert "franka_emika_panda" in mjcf and "panda" not in mjcf
    ur = zoo.list(contains="ur5")
    assert "ur5" in ur and "universal_robots_ur5e" in ur
    assert "panda" not in ur


def test_list_filters_compose():
    both = zoo.list(format="mjcf", contains="ur5")
    assert both == ["universal_robots_ur5e"]


def test_info_returns_the_entry():
    entry = zoo.info("panda")
    assert entry.name == "panda"
    assert entry.filename == "panda.urdf"


def test_unknown_name_suggests_a_close_one():
    with pytest.raises(zoo.UnknownRobot) as e:
        zoo.info("pnada")
    msg = str(e.value)
    assert "pnada" in msg and "panda" in msg
    # KeyError-compatible so existing `except KeyError` still catches it
    assert isinstance(e.value, KeyError)


def test_unknown_unhashable_name_is_a_clean_error():
    with pytest.raises(zoo.UnknownRobot):
        zoo.info(["panda"])


# ---- cache directory logic ------------------------------------------------

def test_cache_dir_prefers_kinfast_cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KINFAST_CACHE", str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert zoo.cache_dir() == os.path.abspath(str(tmp_path / "explicit"))


def test_cache_dir_uses_xdg_when_set(monkeypatch, tmp_path):
    monkeypatch.delenv("KINFAST_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert zoo.cache_dir() == os.path.join(
        os.path.abspath(str(tmp_path / "xdg")), "kinfast")


def test_cache_dir_defaults_to_dot_cache_kinfast(monkeypatch, tmp_path):
    monkeypatch.delenv("KINFAST_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    got = zoo.cache_dir()
    assert got.endswith(os.path.join(".cache", "kinfast"))
    assert got == os.path.join(str(tmp_path / "home"), ".cache", "kinfast")


def test_cache_dir_expands_a_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("KINFAST_CACHE", os.path.join("~", "robots"))
    assert zoo.cache_dir() == os.path.join(str(tmp_path / "home"), "robots")


def test_cache_dir_is_not_created_by_asking(monkeypatch, tmp_path):
    monkeypatch.setenv("KINFAST_CACHE", str(tmp_path / "never"))
    zoo.cache_dir()
    assert not (tmp_path / "never").exists()


# ---- path(): downloading and caching --------------------------------------

def test_path_downloads_then_caches(monkeypatch, tmp_path):
    fake = FakeFetcher(b"<robot name='panda'/>")
    monkeypatch.setattr(zoo, "_http_get", fake)
    p1 = zoo.path("panda", cache_dir=str(tmp_path))
    assert os.path.isfile(p1)
    assert open(p1, "rb").read() == b"<robot name='panda'/>"
    assert p1 == os.path.join(str(tmp_path), "panda", "panda.urdf")
    assert fake.calls == [zoo.info("panda").url]

    p2 = zoo.path("panda", cache_dir=str(tmp_path))
    assert p2 == p1
    assert len(fake.calls) == 1          # the cache was used, not the network


def test_path_uses_the_default_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("KINFAST_CACHE", str(tmp_path / "root"))
    fake = FakeFetcher()
    monkeypatch.setattr(zoo, "_http_get", fake)
    p = zoo.path("cartpole")
    assert p == os.path.join(str(tmp_path / "root"), "cartpole", "cartpole.urdf")


def test_path_force_refetches(monkeypatch, tmp_path):
    fake = FakeFetcher(b"first")
    monkeypatch.setattr(zoo, "_http_get", fake)
    p = zoo.path("cartpole", cache_dir=str(tmp_path))
    fake.body = b"second"
    p2 = zoo.path("cartpole", cache_dir=str(tmp_path), force=True)
    assert p2 == p
    assert open(p, "rb").read() == b"second"
    assert len(fake.calls) == 2


def test_path_fetches_included_sibling_files(monkeypatch, tmp_path):
    entry = zoo.info("aloha")
    bodies = {u: f"<!-- {u} -->".encode() for u in (entry.url,) + entry.extra}
    fake = FakeFetcher(bodies)
    monkeypatch.setattr(zoo, "_http_get", fake)
    p = zoo.path("aloha", cache_dir=str(tmp_path))
    d = os.path.dirname(p)
    assert sorted(os.listdir(d)) == sorted(
        [entry.filename] + [u.rsplit("/", 1)[-1] for u in entry.extra])
    assert len(fake.calls) == 3
    # a second call re-fetches nothing, extras included
    zoo.path("aloha", cache_dir=str(tmp_path))
    assert len(fake.calls) == 3


def test_path_only_fetches_the_missing_sibling(monkeypatch, tmp_path):
    entry = zoo.info("aloha")
    _seed(tmp_path, "aloha", "<mujoco/>")
    bodies = {u: b"<x/>" for u in entry.extra}
    fake = FakeFetcher(bodies)
    monkeypatch.setattr(zoo, "_http_get", fake)
    zoo.path("aloha", cache_dir=str(tmp_path))
    assert sorted(fake.calls) == sorted(entry.extra)


def test_path_leaves_nothing_behind_when_the_download_fails(monkeypatch,
                                                            tmp_path):
    def fail(url, timeout=60.0):
        raise zoo.DownloadError("no route to host")
    monkeypatch.setattr(zoo, "_http_get", fail)
    with pytest.raises(zoo.DownloadError):
        zoo.path("panda", cache_dir=str(tmp_path))
    d = tmp_path / "panda"
    assert not d.exists() or os.listdir(str(d)) == []


def test_path_rejects_an_empty_body(monkeypatch, tmp_path):
    monkeypatch.setattr(zoo, "_http_get", FakeFetcher(b""))
    with pytest.raises(zoo.DownloadError) as e:
        zoo.path("panda", cache_dir=str(tmp_path))
    assert "empty" in str(e.value)
    assert not (tmp_path / "panda" / "panda.urdf").exists()


def test_path_unknown_name_does_not_download(monkeypatch, tmp_path):
    fake = FakeFetcher()
    monkeypatch.setattr(zoo, "_http_get", fake)
    with pytest.raises(zoo.UnknownRobot):
        zoo.path("nosuchrobot", cache_dir=str(tmp_path))
    assert fake.calls == []


def test_download_writes_atomically(monkeypatch, tmp_path):
    monkeypatch.setattr(zoo, "_http_get", FakeFetcher(b"hello"))
    dest = str(tmp_path / "sub" / "file.urdf")
    zoo._download("https://example.invalid/file.urdf", dest)
    assert open(dest, "rb").read() == b"hello"
    assert [f for f in os.listdir(os.path.dirname(dest)) if ".part" in f] == []


# ---- load() through a fake cache ------------------------------------------

def test_load_reads_the_cache_without_touching_the_network(no_network,
                                                           tmp_path):
    _seed(tmp_path, "panda", PLANAR_2R)
    robot = zoo.load("panda", cache_dir=str(tmp_path))
    assert robot.dof == 2
    assert robot.joint_names == ["j1", "j2"]
    assert robot.ee_link == "tip"

    q = torch.tensor([[0.3, 0.4], [-1.1, 0.25]])
    pos = robot.fk(q)[:, :3, 3]
    for i, (a, b) in enumerate([(0.3, 0.4), (-1.1, 0.25)]):
        want = torch.tensor([math.cos(a) + math.cos(a + b),
                             math.sin(a) + math.sin(a + b), 0.0])
        assert torch.allclose(pos[i], want, atol=1e-6)


def test_load_forwards_dtype_and_ee_link(no_network, tmp_path):
    _seed(tmp_path, "panda", PLANAR_2R)
    robot = zoo.load("panda", cache_dir=str(tmp_path), dtype=torch.float64,
                     ee_link="l2")
    assert robot.ee_link == "l2"
    q = torch.zeros(1, 2, dtype=torch.float64)
    T = robot.fk(q)
    assert T.dtype == torch.float64
    assert torch.allclose(T[0, :3, 3],
                          torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))


def test_load_handles_an_mjcf_entry(no_network, tmp_path):
    _seed(tmp_path, "universal_robots_ur5e", MINI_MJCF)
    robot = zoo.load("universal_robots_ur5e", cache_dir=str(tmp_path))
    assert robot.dof == 2
    assert robot.joint_names == ["shoulder", "elbow"]
    # degrees are the MJCF default, so the hinge range is radians here
    assert float(robot.lower[0]) == pytest.approx(-math.pi / 2, abs=1e-6)


def test_load_downloads_when_the_cache_is_cold(monkeypatch, tmp_path):
    fake = FakeFetcher(PLANAR_2R.encode())
    monkeypatch.setattr(zoo, "_http_get", fake)
    robot = zoo.load("panda", cache_dir=str(tmp_path))
    assert robot.dof == 2
    assert fake.calls == [zoo.info("panda").url]


# ---- the fetch itself: urllib with a curl fallback ------------------------

class _FakeResponse:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_get_uses_urllib_when_it_works(monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"], seen["timeout"] = url, timeout
        return _FakeResponse(b"<robot/>")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        pytest.fail("curl must not run when urllib works"))
    assert zoo._http_get("https://example.invalid/a.urdf", timeout=7) == b"<robot/>"
    assert seen == {"url": "https://example.invalid/a.urdf", "timeout": 7}


def _ssl_failure(*a, **kw):
    raise ssl.SSLCertVerificationError("certificate verify failed")


class _Completed:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_http_get_falls_back_to_curl_on_ssl_failure(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _ssl_failure)
    monkeypatch.setattr(zoo.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _Completed(0, b"<robot name='curled'/>")

    monkeypatch.setattr(subprocess, "run", fake_run)
    got = zoo._http_get("https://example.invalid/a.urdf", timeout=30)
    assert got == b"<robot name='curled'/>"
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0].endswith("curl") and cmd[-1] == "https://example.invalid/a.urdf"
    assert "-fsSL" in cmd and "30" in cmd


def test_http_get_falls_back_on_a_plain_connection_error(monkeypatch):
    def refused(*a, **kw):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr("urllib.request.urlopen", refused)
    monkeypatch.setattr(zoo.shutil, "which", lambda name: "curl")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Completed(0, b"body"))
    assert zoo._http_get("https://example.invalid/a.urdf") == b"body"


def test_http_get_does_not_retry_an_http_error(monkeypatch):
    def not_found(*a, **kw):
        raise urllib.error.HTTPError("https://example.invalid/a.urdf", 404,
                                     "Not Found", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", not_found)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        pytest.fail("curl must not retry a 404"))
    with pytest.raises(zoo.DownloadError) as e:
        zoo._http_get("https://example.invalid/a.urdf")
    assert "404" in str(e.value)


def test_http_get_explains_a_missing_curl(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _ssl_failure)
    monkeypatch.setattr(zoo.shutil, "which", lambda name: None)
    with pytest.raises(zoo.DownloadError) as e:
        zoo._http_get("https://example.invalid/a.urdf")
    assert "curl is not installed" in str(e.value)


def test_http_get_reports_a_failing_curl(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _ssl_failure)
    monkeypatch.setattr(zoo.shutil, "which", lambda name: "curl")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Completed(22, b"", b"HTTP 403"))
    with pytest.raises(zoo.DownloadError) as e:
        zoo._http_get("https://example.invalid/a.urdf")
    msg = str(e.value)
    assert "curl exited 22" in msg and "HTTP 403" in msg


def test_http_get_reports_an_empty_curl_body(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _ssl_failure)
    monkeypatch.setattr(zoo.shutil, "which", lambda name: "curl")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, b""))
    with pytest.raises(zoo.DownloadError):
        zoo._http_get("https://example.invalid/a.urdf")


# ---- cache maintenance ----------------------------------------------------

def test_clear_cache_for_one_robot(no_network, tmp_path):
    _seed(tmp_path, "panda", PLANAR_2R)
    _seed(tmp_path, "cartpole", PLANAR_2R)
    removed = zoo.clear_cache("panda", cache_dir=str(tmp_path))
    assert removed == [os.path.join(str(tmp_path), "panda")]
    assert not (tmp_path / "panda").exists()
    assert (tmp_path / "cartpole").exists()


def test_clear_cache_for_everything(no_network, tmp_path):
    _seed(tmp_path, "panda", PLANAR_2R)
    _seed(tmp_path, "cartpole", PLANAR_2R)
    (tmp_path / "not_a_robot").mkdir()
    removed = zoo.clear_cache(cache_dir=str(tmp_path))
    assert len(removed) == 2
    assert not (tmp_path / "panda").exists()
    assert (tmp_path / "not_a_robot").exists()   # only registry names are touched


def test_clear_cache_on_a_cold_cache_is_quiet(no_network, tmp_path):
    assert zoo.clear_cache(cache_dir=str(tmp_path)) == []
    assert zoo.clear_cache("panda", cache_dir=str(tmp_path)) == []


# ---- the one test that really downloads -----------------------------------

@pytest.mark.skipif(os.environ.get("KINFAST_NET") != "1",
                    reason="set KINFAST_NET=1 to allow a real download")
def test_real_download_and_load(tmp_path):
    p = zoo.path("cartpole", cache_dir=str(tmp_path))
    assert os.path.getsize(p) > 0
    robot = zoo.load("cartpole", cache_dir=str(tmp_path))
    assert robot.dof >= 1
    q = robot.random_configs(4)
    assert robot.fk(q).shape == (4, 4, 4)
