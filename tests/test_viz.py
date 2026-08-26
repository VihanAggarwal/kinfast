# tests/test_viz.py
"""Visualization tests.

The oracles here are hand-computed link positions (a planar 2R at 90 degrees
puts its elbow on the y axis) and independent readers of the written files:
the PNG/GIF magic bytes and Pillow's frame count, rather than kinfast checking
its own drawing.

matplotlib is pinned to Agg for the in-process tests so nothing tries to open a
window; the automatic backend choice and the lazy import are checked in a clean
subprocess instead.
"""
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pytest                            # noqa: E402
import torch                             # noqa: E402

import kinfast                           # noqa: E402
from kinfast.compile import compile_robot        # noqa: E402
from kinfast.urdf.parse import parse_urdf_string  # noqa: E402
from kinfast import viz                   # noqa: E402
from tests.test_parse import TWO_LINK      # noqa: E402

SLIDER = """
<robot name="slider">
  <link name="base"/><link name="cart"/>
  <joint name="s1" type="prismatic">
    <parent link="base"/><child link="cart"/>
    <origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="1" velocity="1" effort="10"/>
  </joint>
</robot>
"""

# A rotated joint frame plus a fixed tip: the fixed link still has to be drawn,
# and the tip position exercises the origin rpy rather than the joint angle alone.
BENT = """
<robot name="bent">
  <link name="base"/><link name="pivot"/><link name="tool"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="pivot"/>
    <origin xyz="0 0 0.5" rpy="1.5707963267948966 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="1" effort="10"/>
  </joint>
  <joint name="tip" type="fixed">
    <parent link="pivot"/><child link="tool"/>
    <origin xyz="0.4 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""

# Nothing moves at all: compile_robot gives this a dof of 0.
STATIC = """
<robot name="static">
  <link name="base"/><link name="post"/>
  <joint name="weld" type="fixed">
    <parent link="base"/><child link="post"/>
    <origin xyz="0 0 0.75" rpy="0 0 0"/>
  </joint>
</robot>
"""


@pytest.fixture
def chain():
    return compile_robot(parse_urdf_string(TWO_LINK), dtype=torch.float64)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _path_points(line):
    """The (N, 3) polyline of a Line3D, NaN separators included."""
    xs, ys, zs = line.get_data_3d()
    return np.stack([np.asarray(xs, dtype=float),
                     np.asarray(ys, dtype=float),
                     np.asarray(zs, dtype=float)], axis=1)


def _finite(points):
    return points[np.isfinite(points).all(axis=1)]


# ---- the picture matches hand-computed kinematics ----

def test_stick_figure_matches_hand_computed_positions(chain):
    """Two unit links, first joint at 90 degrees: base and l1 origins sit at the
    world origin, l2's origin swings to (0, 1, 0). The drawn polyline is
    base->l1 then l1->l2, with a NaN break between the two segments."""
    q = torch.tensor([[math.pi / 2, 0.3]], dtype=torch.float64)
    fig = viz.plot(chain, q)
    ax = fig.axes[0]
    pts = _path_points(ax.lines[0])
    assert pts.shape == (6, 3)             # 2 edges x (start, end, gap)
    assert np.isnan(pts[2]).all() and np.isnan(pts[5]).all()
    expect = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    got = np.stack([pts[0], pts[1], pts[3], pts[4]])
    assert np.allclose(got, expect, atol=1e-12)


def test_prismatic_joint_translates_the_drawing():
    """A slider along x at 0.25 m puts the cart origin at (0.25, 0, 0)."""
    chain = compile_robot(parse_urdf_string(SLIDER), dtype=torch.float64)
    fig = viz.plot(chain, torch.tensor([0.25], dtype=torch.float64))
    pts = _finite(_path_points(fig.axes[0].lines[0]))
    assert np.allclose(pts[0], [0.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(pts[1], [0.25, 0.0, 0.0], atol=1e-12)


def test_rotated_joint_frame_and_fixed_link():
    """The joint origin rotates 90 degrees about x and sits 0.5 up, then a fixed
    joint offsets the tool 0.4 along the pivot's local x.

    By hand: the pivot origin is at (0, 0, 0.5) for every q, and the tool is at
    Rx(90) @ Rz(q) @ (0.4, 0, 0) + (0, 0, 0.5) = (0.4 cos q, 0, 0.5 + 0.4 sin q).
    The fixed link must still be drawn, so there are two segments.
    """
    chain = compile_robot(parse_urdf_string(BENT), dtype=torch.float64)
    assert chain.dof == 1
    for angle in (0.0, 0.3, math.pi / 2, -1.1):
        fig = viz.plot(chain, torch.tensor([[angle]], dtype=torch.float64))
        pts = _path_points(fig.axes[0].lines[0])
        assert pts.shape == (6, 3)          # base->pivot, pivot->tool
        base, pivot = pts[0], pts[1]
        pivot2, tool = pts[3], pts[4]
        assert np.allclose(base, [0.0, 0.0, 0.0], atol=1e-12)
        assert np.allclose(pivot, [0.0, 0.0, 0.5], atol=1e-12)
        assert np.allclose(pivot2, pivot, atol=1e-12)
        assert np.allclose(
            tool,
            [0.4 * math.cos(angle), 0.0, 0.5 + 0.4 * math.sin(angle)],
            atol=1e-12)
        plt.close(fig)


def test_zero_dof_robot_still_draws():
    """A fully welded robot has dof 0. An empty q is a legal batch of one, and
    the weld's 0.75 m offset is the whole picture."""
    chain = compile_robot(parse_urdf_string(STATIC), dtype=torch.float64)
    assert chain.dof == 0
    fig = viz.plot(chain, torch.zeros(0, dtype=torch.float64))
    pts = _finite(_path_points(fig.axes[0].lines[0]))
    assert np.allclose(pts, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]], atol=1e-12)


def test_working_dtype_follows_q_not_the_chain():
    """The chain is compiled float32 but drawn from a float64 q. As everywhere
    else in the library the caller's q sets the working dtype, so the geometry
    is computed in double even though the constants were stored single."""
    chain32 = compile_robot(parse_urdf_string(BENT), dtype=torch.float32)
    fig = viz.plot(chain32, torch.tensor([[math.pi / 2]], dtype=torch.float64))
    pts = _finite(_path_points(fig.axes[0].lines[0]))
    assert np.allclose(pts[-1], [0.0, 0.0, 0.9], atol=1e-6)


def test_one_dimensional_q_is_a_batch_of_one(chain):
    fig = viz.plot(chain, torch.zeros(chain.dof, dtype=torch.float64))
    assert len(fig.axes[0].lines) == 1


def test_batch_draws_every_configuration_with_a_fade(chain):
    q = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=torch.float64)
    fig = viz.plot(chain, q)
    lines = fig.axes[0].lines
    assert len(lines) == 3
    alphas = [line.get_alpha() for line in lines]
    assert alphas == sorted(alphas) and alphas[-1] == pytest.approx(1.0)
    # the third configuration has its elbow at (cos 1, sin 1, 0)
    last = _finite(_path_points(lines[-1]))
    assert np.allclose(last[-1], [math.cos(1.0), math.sin(1.0), 0.0], atol=1e-12)


def test_axes_box_is_a_cube(chain):
    """3D axes scale each axis on its own, so a non-cubic limit box would bend
    the arm. Auto limits must span the same distance on x, y and z."""
    fig = viz.plot(chain, torch.tensor([[0.4, 0.7]], dtype=torch.float64))
    ax = fig.axes[0]
    spans = [hi - lo for lo, hi in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())]
    assert spans[0] == pytest.approx(spans[1]) == pytest.approx(spans[2])
    assert spans[0] > 0.0


def test_camera_can_be_set(chain):
    fig = viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64), elev=90.0, azim=0.0)
    ax = fig.axes[0]
    assert ax.elev == pytest.approx(90.0)
    assert ax.azim == pytest.approx(0.0)


def test_explicit_limits_are_used(chain):
    lim = [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]
    fig = viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64), limits=lim)
    ax = fig.axes[0]
    assert ax.get_xlim() == pytest.approx((-2.0, 2.0))
    assert ax.get_zlim() == pytest.approx((-2.0, 2.0))


# ---- collision spheres ----

def test_spheres_are_drawn_at_the_world_center(chain):
    """A sphere pinned to the l2 origin with r = 0.3, at q1 = 90 degrees, must
    be drawn as points exactly 0.3 away from (0, 1, 0)."""
    q = torch.tensor([[math.pi / 2, 0.0]], dtype=torch.float64)
    spheres = {"l2": [(0.0, 0.0, 0.0, 0.3)]}
    fig = viz.plot(chain, q, link_spheres=spheres)
    lines = fig.axes[0].lines
    assert len(lines) == 2                  # stick figure + one sphere
    wire = _finite(_path_points(lines[1]))
    center = np.array([0.0, 1.0, 0.0])
    radii = np.linalg.norm(wire - center, axis=1)
    assert np.allclose(radii, 0.3, atol=1e-12)
    # three great circles means points off every coordinate plane through center
    assert np.abs(wire - center).max(axis=0).min() == pytest.approx(0.3)


def test_sphere_model_object_is_accepted(chain):
    from kinfast.collision import SphereModel
    model = SphereModel(chain, {chain.link_index["l2"]: [(0.0, 0.0, 0.0, 0.2)],
                                chain.link_index["base"]: [(0.0, 0.0, 0.0, 0.2)]})
    fig = viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64), link_spheres=model)
    assert len(fig.axes[0].lines) == 3      # stick figure + two spheres


def test_sphere_center_offset_in_the_link_frame(chain):
    """A sphere is placed at (0.5, 0, 0) in l1's frame. With j1 at 90 degrees
    that offset points along world +y, so the wireframe is centred on
    (0, 0.5, 0) rather than on the link origin."""
    q = torch.tensor([[math.pi / 2, 0.0]], dtype=torch.float64)
    fig = viz.plot(chain, q, link_spheres={"l1": [(0.5, 0.0, 0.0, 0.1)]})
    wire = _finite(_path_points(fig.axes[0].lines[1]))
    # the wireframe is symmetric about its centre, so the bounding-box midpoint
    # recovers it exactly (unlike the point mean, which the shared endpoints skew)
    midpoint = 0.5 * (wire.min(axis=0) + wire.max(axis=0))
    assert np.allclose(midpoint, [0.0, 0.5, 0.0], atol=1e-12)
    assert np.allclose(np.linalg.norm(wire - [0.0, 0.5, 0.0], axis=1), 0.1,
                       atol=1e-12)


def test_spheres_keyed_by_link_index(chain):
    fig = viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64),
                   link_spheres={chain.link_index["l2"]: [(0, 0, 0, 0.2)]})
    assert len(fig.axes[0].lines) == 2


def test_empty_sphere_dict_draws_nothing_extra(chain):
    fig = viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64), link_spheres={})
    assert len(fig.axes[0].lines) == 1


def test_unknown_sphere_link_raises(chain):
    with pytest.raises(KeyError, match="nope"):
        viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64),
                 link_spheres={"nope": [(0, 0, 0, 0.1)]})


def test_out_of_range_sphere_index_raises(chain):
    with pytest.raises(IndexError, match="out of range"):
        viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64),
                 link_spheres={99: [(0, 0, 0, 0.1)]})


def test_spheres_widen_the_limits(chain):
    q = torch.zeros(1, 2, dtype=torch.float64)
    plain = viz.plot(chain, q).axes[0].get_zlim()
    withs = viz.plot(chain, q, link_spheres={"l2": [(0, 0, 0, 2.0)]}).axes[0].get_zlim()
    assert (withs[1] - withs[0]) > (plain[1] - plain[0])


# ---- figure plumbing ----

def test_plot_returns_a_figure_and_saves_a_png(chain, tmp_path):
    from matplotlib.figure import Figure
    fig = viz.plot(chain, torch.tensor([[0.2, -0.4]], dtype=torch.float64),
                   title="two link", labels=True)
    assert isinstance(fig, Figure)
    out = tmp_path / "arm.png"
    fig.savefig(out)
    assert out.stat().st_size > 0
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    assert fig.axes[0].get_title() == "two link"


def test_plot_into_existing_axes(chain):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot([0.0], [0.0], [0.0])            # a pre-existing artist stays
    out = viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64), ax=ax)
    assert out is fig
    assert len(ax.lines) == 2


def test_robot_wrapper_is_accepted(chain):
    class Holder:
        pass
    holder = Holder()
    holder.chain = chain
    fig = viz.plot(holder, torch.zeros(1, 2, dtype=torch.float64))
    assert len(fig.axes[0].lines) == 1


# ---- dtype, device and autograd behaviour ----

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_agnostic(chain, dtype):
    q = torch.tensor([[math.pi / 2, 0.0]], dtype=dtype)
    fig = viz.plot(chain, q)
    pts = _finite(_path_points(fig.axes[0].lines[0]))
    assert np.allclose(pts[-1], [0.0, 1.0, 0.0], atol=1e-6)


def test_integer_q_is_promoted(chain):
    fig = viz.plot(chain, torch.zeros(2, dtype=torch.long))
    assert len(fig.axes[0].lines) == 1


def test_plain_python_and_numpy_q(chain):
    """A list or a numpy array is as good as a tensor at the plotting layer.
    A bare list becomes the default dtype (float32), hence the loose tolerance."""
    from_list = viz.plot(chain, [math.pi / 2, 0.0])
    from_numpy = viz.plot(chain, np.array([[math.pi / 2, 0.0]]))
    for fig in (from_list, from_numpy):
        pts = _finite(_path_points(fig.axes[0].lines[0]))
        assert np.allclose(pts[-1], [0.0, 1.0, 0.0], atol=1e-6)


def test_q_requiring_grad_is_detached(chain):
    q = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    fig = viz.plot(chain, q, link_spheres={"l2": [(0, 0, 0, 0.1)]})
    assert len(fig.axes[0].lines) == 2
    assert q.grad is None                   # drawing is a sink, not a loss


# ---- input validation ----

def test_wrong_width_raises(chain):
    with pytest.raises(ValueError, match="dof"):
        viz.plot(chain, torch.zeros(1, 5, dtype=torch.float64))


def test_wrong_rank_raises(chain):
    with pytest.raises(ValueError, match="shape"):
        viz.plot(chain, torch.zeros(1, 1, 2, dtype=torch.float64))


def test_two_d_axes_raises(chain):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="3D axes"):
        viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64), ax=ax)


def test_bad_limits_raise(chain):
    with pytest.raises(ValueError, match="limits"):
        viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64), limits=[0.0, 1.0])


def test_bad_spheres_raise(chain):
    with pytest.raises(TypeError, match="SphereModel"):
        viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64), link_spheres=[1, 2])


def test_non_chain_raises():
    with pytest.raises(TypeError, match="CompiledChain"):
        viz.plot(object(), torch.zeros(1, 2))


# ---- animation ----

def test_animate_writes_a_gif_with_one_frame_per_configuration(chain, tmp_path):
    """Pillow reads the file back: a real gif with exactly len(qs) frames.

    The configurations are all different, because Pillow collapses identical
    consecutive frames when it writes the gif."""
    from PIL import Image
    qs = torch.stack([torch.tensor([a, 0.0], dtype=torch.float64)
                      for a in np.linspace(0.0, 1.2, 5)])
    out = tmp_path / "arm.gif"
    got = viz.animate(chain, qs, out, fps=10, figsize=(2.0, 2.0), dpi=50)
    assert got == out
    assert out.stat().st_size > 0
    with open(out, "rb") as fh:
        assert fh.read(3) == b"GIF"
    with Image.open(out) as img:
        assert img.format == "GIF"
        assert img.n_frames == 5
        assert img.size[0] > 0 and img.size[1] > 0


def test_animate_accepts_spheres_and_str_path(chain, tmp_path):
    from PIL import Image
    qs = torch.zeros(2, 2, dtype=torch.float64)
    qs[1, 0] = 0.5
    out = str(tmp_path / "spheres.gif")
    viz.animate(chain, qs, out, fps=5, figsize=(2.0, 2.0), dpi=50,
                link_spheres={"l2": [(0, 0, 0, 0.2)]}, title="t", labels=True)
    with Image.open(out) as img:
        assert img.n_frames == 2


def test_animate_holds_the_camera_fixed(chain, tmp_path, monkeypatch):
    """Every frame redraws the axes from scratch, so the requested camera has to
    be reapplied; otherwise the view snaps back after the first frame."""
    from PIL import Image
    seen = []
    real = viz._finish_axes

    def spy(ax, *args, **kw):
        seen.append((ax.elev, ax.azim))
        return real(ax, *args, **kw)

    monkeypatch.setattr(viz, "_finish_axes", spy)
    # distinct configurations: Pillow collapses identical consecutive frames
    qs = torch.tensor([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]], dtype=torch.float64)
    viz.animate(chain, qs, tmp_path / "cam.gif", figsize=(2.0, 2.0), dpi=50,
                elev=12.0, azim=-70.0)
    # at least one call per frame (FuncAnimation also does an initial draw)
    assert len(seen) >= 3
    for elev, azim in seen:
        assert elev == pytest.approx(12.0) and azim == pytest.approx(-70.0)
    with Image.open(tmp_path / "cam.gif") as img:
        assert img.n_frames == 3


def test_animate_single_frame(chain, tmp_path):
    """One configuration is a legal, if boring, animation. A (dof,) vector is
    promoted the same way plot promotes it."""
    from PIL import Image
    out = tmp_path / "one.gif"
    viz.animate(chain, torch.tensor([0.3, 0.2], dtype=torch.float64), out,
                figsize=(2.0, 2.0), dpi=50)
    with Image.open(out) as img:
        assert img.format == "GIF"
        assert img.n_frames == 1


def test_animate_limits_cover_every_frame(chain, tmp_path, monkeypatch):
    """The view must be sized once from the whole trajectory, not per frame,
    otherwise the box breathes while the arm swings. The elbow reaches y = 1 at
    90 degrees, so the shared limits have to contain that even at frame 0."""
    seen = []
    real = viz._finish_axes

    def spy(ax, limits, *args, **kw):
        seen.append(np.array(limits, dtype=float))
        return real(ax, limits, *args, **kw)

    monkeypatch.setattr(viz, "_finish_axes", spy)
    qs = torch.tensor([[0.0, 0.0], [math.pi / 4, 0.0], [math.pi / 2, 0.0]],
                      dtype=torch.float64)
    viz.animate(chain, qs, tmp_path / "span.gif", figsize=(2.0, 2.0), dpi=50)
    assert seen
    for lim in seen[1:]:
        assert np.array_equal(lim, seen[0])
    lo, hi = seen[0]
    assert lo[0] <= 0.0 and hi[0] >= 1.0      # elbow at (1, 0, 0) when q = 0
    assert lo[1] <= 0.0 and hi[1] >= 1.0      # elbow at (0, 1, 0) at 90 degrees


def test_animate_closes_its_figure(chain, tmp_path):
    before = len(plt.get_fignums())
    viz.animate(chain, torch.zeros(2, 2, dtype=torch.float64),
                tmp_path / "x.gif", figsize=(2.0, 2.0), dpi=50)
    assert len(plt.get_fignums()) == before


def test_animate_rejects_empty(chain, tmp_path):
    with pytest.raises(ValueError, match="empty"):
        viz.animate(chain, torch.zeros(0, 2, dtype=torch.float64), tmp_path / "e.gif")


# ---- lazy import and backend, in a clean interpreter ----

def test_import_kinfast_does_not_import_matplotlib(tmp_path):
    """Importing kinfast, or kinfast.viz, must not pull in matplotlib: the
    import happens inside the drawing functions. Checked in a fresh process
    because this one has already imported matplotlib."""
    src = os.path.dirname(os.path.dirname(os.path.abspath(kinfast.__file__)))
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        "import kinfast\n"
        "import kinfast.viz as viz\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] "
        "in ('matplotlib', 'pylab'))\n"
        "assert not leaked, leaked\n"
        "import torch\n"
        "from kinfast.urdf.parse import parse_urdf_string\n"
        "from kinfast.compile import compile_robot\n"
        f"chain = compile_robot(parse_urdf_string({TWO_LINK!r}), dtype=torch.float64)\n"
        "fig = viz.plot(chain, torch.zeros(1, 2, dtype=torch.float64))\n"
        "import matplotlib\n"
        "assert matplotlib.get_backend().lower() == 'agg', matplotlib.get_backend()\n"
        "fig.savefig(sys.argv[1])\n"
        "print('ok')\n",
        encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + os.path.dirname(tests_dir)
    env.pop("MPLBACKEND", None)
    png = tmp_path / "headless.png"
    run = subprocess.run([sys.executable, str(probe), str(png)],
                         capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "ok" in run.stdout
    assert png.stat().st_size > 0
