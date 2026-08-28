# tests/test_notebook.py
"""Guards on the GPU notebook.

A notebook is the one file nothing else in the suite exercises, so it rots
quietly and then wastes somebody's GPU session. These are the cheap checks:
that it parses, that the code in it is syntactically Python, and that the
setup cell still does the one non-obvious thing it has to do.
"""
import json
import pathlib

import pytest

NOTEBOOK = pathlib.Path(__file__).resolve().parents[1] / "examples" / "kinfast_gpu_colab.ipynb"


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def setup_cell(nb):
    for cell in _code_cells(nb):
        source = "".join(cell["source"])
        if "clone" in source:
            return source
    pytest.fail("no cell in the notebook clones the repo")


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _strip_magics(source):
    """IPython shell escapes and line magics are not Python, so drop them
    before asking the compiler for an opinion on the rest. A shell escape can
    be wrapped over several lines with a trailing backslash, and all of those
    lines have to go, not just the first, or the leftover argument lines read
    as a stray indent."""
    keep = []
    continued = False
    for line in source.splitlines():
        stripped = line.lstrip()
        if continued or stripped.startswith("!") or stripped.startswith("%"):
            starts_group = not continued
            continued = line.rstrip().endswith("\\")
            # a wrapped command keeps its indent on the second line, so only
            # the first line of a group becomes pass and the rest go blank
            keep.append(" " * (len(line) - len(stripped)) + "pass"
                        if starts_group else "")
        else:
            keep.append(line)
    return "\n".join(keep)


def test_notebook_parses(nb):
    assert nb["nbformat"] == 4
    assert _code_cells(nb), "a notebook with no code in it is not doing anything"


def test_no_stored_outputs(nb):
    """Committed outputs make every rerun a diff and can carry a stale
    benchmark table that reads as a current measurement."""
    dirty = [i for i, c in enumerate(nb["cells"]) if c.get("outputs")]
    assert dirty == [], f"cells {dirty} have outputs saved in them"


def test_every_code_cell_compiles(nb):
    for i, cell in enumerate(_code_cells(nb)):
        source = _strip_magics("".join(cell["source"]))
        try:
            compile(source, f"<cell {i}>", "exec")
        except SyntaxError as exc:
            pytest.fail(f"code cell {i} does not compile: {exc}")


def test_setup_cell_puts_the_source_tree_on_the_path(setup_cell):
    """Regression. An editable install writes its path into a .pth file, and
    .pth files are read only at interpreter startup, so `pip install -e .` has
    no effect on a kernel that is already running. Colab keeps /content on
    sys.path, so the bare clone directory then imports as an empty namespace
    package: `import kinfast` succeeds and the module has nothing in it, which
    surfaces as a missing __version__ several lines later. The cell has to add
    the source directory itself."""
    assert "sys.path.insert" in setup_cell, "the editable install alone will not import"
    assert "invalidate_caches" in setup_cell, "the import system caches the failed lookup"


def test_setup_cell_uses_absolute_paths(setup_cell):
    """Regression. The clone guard used to test a relative path, so running the
    cell a second time cloned the repo inside itself and changed into it."""
    assert "/content/kinfast" in setup_cell
    assert 'os.path.isdir("kinfast")' not in setup_cell
