"""Tests for the API reference generator in docs/gen_api.py.

The generator introspects the imported package, so checking it against the same
introspection would prove nothing. Instead the oracle here is the source text:
the test parses every kinfast source file with `ast` and asserts that each
public top-level function and class it finds shows up in the generated
Markdown. Two independent readings of the library have to agree.
"""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN_PATH = REPO / "docs" / "gen_api.py"
SRC = REPO / "src" / "kinfast"


def _load_generator():
    """Import docs/gen_api.py by path. docs/ is not a package, so no import works."""
    spec = importlib.util.spec_from_file_location("kinfast_gen_api", GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_generator()


@pytest.fixture(scope="module")
def doc(gen, tmp_path_factory):
    """Run the generator to a temp path and return what it wrote."""
    out = tmp_path_factory.mktemp("api") / "API.md"
    code = gen.main(["--out", str(out)])
    assert code == 0
    assert out.exists()
    return out.read_text(encoding="utf-8")


def test_lists_the_headline_api(doc):
    for name in ("forward_kinematics", "ik", "Robot", "CompiledRobot"):
        assert name in doc, f"{name} missing from the generated reference"
    assert "### `forward_kinematics(chain" in doc
    assert "### `ik(chain" in doc
    assert "### class `Robot(chain, ee_link=None, ir=None)`" in doc
    assert "### class `CompiledRobot(chain)`" in doc


def test_groups_entries_under_their_defining_module(doc):
    for heading in ("## `kinfast.fk`", "## `kinfast.ik`", "## `kinfast.robot`",
                    "## `kinfast.codegen`", "## `kinfast.mjcf.parse`",
                    "## `kinfast.urdf.parse`"):
        assert heading in doc, f"{heading} missing"
    # forward_kinematics is imported by several modules but defined in one, so
    # it must be documented exactly once.
    assert doc.count("### `forward_kinematics(") == 1
    # Robot is a name in two modules (the IR dataclass and the user-facing
    # loader); both deserve an entry.
    assert doc.count("### class `Robot(") == 2


def test_reexports_point_at_the_real_home(doc):
    assert "| `kinfast.Robot` | `kinfast.robot.Robot` |" in doc
    assert "| `kinfast.load` | `kinfast.robot.load` |" in doc


def test_carries_signatures_and_first_paragraphs(doc):
    assert "`rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor`" in doc
    assert "roll-pitch-yaw" in doc
    # Only the opening paragraph is copied, not the whole docstring. fk.py's
    # module docstring explains the (B,3,3) layout after a blank line.
    assert "Batched forward kinematics: propagate transforms down the tree." in doc
    assert "27 multiplies per compose" not in doc


def test_methods_are_listed_without_self(doc):
    assert "- `Robot.jacobian(q, link=None)`" in doc
    assert "- `Robot.dof` (property)" in doc
    assert "(self," not in doc
    assert "(self)" not in doc
    # from_ir is a classmethod; cls is dropped the same way self is. The rest
    # of the signature is not pinned here, so adding a keyword (dtype, say)
    # does not break the test that only cares about cls.
    assert "- `Robot.from_ir(robot_ir, repair_model=True, ee_link=None" in doc


def test_private_names_stay_out(doc):
    for name in ("_cache", "_resolve_link", "_check_rows", "_Emitter", "__version__"):
        assert f"`{name}" not in doc, f"{name} should not be documented"


def _public_top_level_names():
    """Read every kinfast source file and collect public module-level defs.

    This is the independent oracle: plain text parsing, no import of the
    package and no use of the generator's own logic.
    """
    found = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC.parent).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if any(part.startswith("_") for part in parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
        ]
        if names:
            found[".".join(parts)] = sorted(names)
    return found


def test_every_public_source_definition_is_documented(doc):
    expected = _public_top_level_names()
    assert expected, "the ast scan found nothing, the oracle is broken"
    missing = []
    for module, names in expected.items():
        section = doc.split(f"## `{module}`\n", 1)
        if len(section) != 2:
            missing.append(f"{module} (no section)")
            continue
        body = section[1].split("\n## ", 1)[0]
        for name in names:
            if f"### `{name}(" not in body and f"### class `{name}(" not in body:
                missing.append(f"{module}.{name}")
    assert not missing, f"undocumented public definitions: {missing}"


def test_output_is_deterministic(gen):
    assert gen.render() == gen.render()


def test_prose_avoids_long_dashes(doc):
    """House style is plain prose, so the long dashes stay out of both files.

    The two characters are built with chr() so this test does not itself
    become the one place they appear.
    """
    long_dash, short_dash = chr(0x2014), chr(0x2013)
    source = GEN_PATH.read_text(encoding="utf-8")
    for text, label in ((doc, "generated reference"), (source, "gen_api.py")):
        assert long_dash not in text, f"long dash in {label}"
        assert short_dash not in text, f"short dash in {label}"


def test_check_mode_flags_a_stale_file(gen, tmp_path, capsys):
    stale = tmp_path / "API.md"
    stale.write_text("out of date\n", encoding="utf-8")
    assert gen.main(["--out", str(stale), "--check"]) == 1
    assert gen.main(["--out", str(stale)]) == 0
    assert gen.main(["--out", str(stale), "--check"]) == 0
    capsys.readouterr()


def test_checked_in_api_md_is_current(gen, capsys):
    """docs/API.md is committed, so it has to match the code it describes."""
    assert gen.main(["--check"]) == 0, "run: python docs/gen_api.py"
    capsys.readouterr()


def test_optional_is_normalized_so_the_doc_matches_on_every_python(gen, doc):
    """Python 3.10 renders Optional[str], 3.12 and later render str | None.
    The reference is committed and checked by test_checked_in_api_md_is_current,
    so it has to look the same whichever interpreter generated it. Everything is
    normalized to the modern spelling, with brackets matched so nested
    annotations survive."""
    assert gen._normalize_optional("(a: Optional[str] = None)") == "(a: str | None = None)"
    # nested generics keep their inner brackets
    assert (gen._normalize_optional("(a: Optional[Dict[str, int]] = None)")
            == "(a: Dict[str, int] | None = None)")
    # more than one on a line, and already-modern text is left alone
    assert (gen._normalize_optional("(a: Optional[int] = 1, b: Optional[str] = None)")
            == "(a: int | None = 1, b: str | None = None)")
    assert gen._normalize_optional("(a: str | None = None)") == "(a: str | None = None)"

    # A string a caller really passes is a value, not an annotation. Rewriting
    # it would document the wrong default, so quoted text is left alone even
    # when it spells Optional[...] or hides a bracket.
    assert (gen._normalize_optional("(mode: str = 'Optional[str]', x: Optional[int] = 1)")
            == "(mode: str = 'Optional[str]', x: int | None = 1)")
    assert (gen._normalize_optional('(m: Literal["Optional[int]", "plain"] = "plain")')
            == '(m: Literal["Optional[int]", "plain"] = "plain")')
    assert (gen._normalize_optional("(sep: str = 'a]b', z: Optional[str] = None)")
            == "(sep: str = 'a]b', z: str | None = None)")
    # rewriting twice changes nothing
    once = gen._normalize_optional("(a: Optional[str] = None)")
    assert gen._normalize_optional(once) == once
    # and the generated reference itself carries no old-style annotation
    assert "Optional[" not in doc
