"""Generate docs/API.md from the kinfast package itself.

The reference is produced by importing kinfast and every submodule under it and
then reading the objects, so it can never drift from the code the way a
hand-written list does. Only signatures and the first paragraph of each
docstring are copied out: the point is a map of the library you can scan in one
sitting, not a replacement for reading the source.

Two rules keep the output honest and stable:

Names are filed under the module that defines them, not the module that
re-exports them. `kinfast.Robot` is really `kinfast.robot.Robot`, so it is
documented once, under `kinfast.robot`, and the re-export shows up in the top
level table instead. Without that rule every convenience import would produce a
duplicate entry.

Everything is sorted and the output is pure text, so regenerating without
changing the code produces a byte-identical file. That makes `--check` usable
in a test or a hook: it regenerates in memory and fails if the checked-in
API.md is stale.

Usage:

    python docs/gen_api.py                 write docs/API.md next to this file
    python docs/gen_api.py --out /tmp/x.md write somewhere else
    python docs/gen_api.py --check         exit 1 if the file on disk is stale
"""
import argparse
import dataclasses
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

PACKAGE = "kinfast"


def _prefer_checkout():
    """Document the source next to this script, not some other installed copy.

    A checkout keeps the package in `src/kinfast` beside `docs/`. If that is
    there and nothing has imported kinfast yet, put it first on the path, so
    running the script in a fresh clone or a worktree describes that tree
    instead of whatever version happens to be installed in the environment.
    """
    src = Path(__file__).resolve().parent.parent / "src"
    if PACKAGE in sys.modules:
        return
    if (src / PACKAGE / "__init__.py").exists():
        sys.path.insert(0, str(src))


_prefer_checkout()


def _first_paragraph(obj):
    """Return the first paragraph of an object's docstring as a single line.

    Docstrings here open with a summary that runs for a line or three and is
    then followed by a blank line and the details. That opening block is the
    part worth putting in a reference, so we cut at the first blank line and
    collapse the newlines so the paragraph reflows in Markdown.
    """
    doc = inspect.getdoc(obj)
    if not doc:
        return ""
    lines = []
    for line in doc.splitlines():
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines).strip()


def _is_generated_dataclass_doc(cls):
    """True if a class only has the docstring dataclasses invents for it.

    `@dataclass` fills in `__doc__` with a rendering of the generated
    `__init__` signature when the author wrote none. Repeating that under a
    signature we already print adds nothing, so it is treated as no docstring.
    """
    if not dataclasses.is_dataclass(cls):
        return False
    doc = cls.__doc__ or ""
    return doc.startswith(cls.__name__ + "(")


def _signature(obj, drop_self=False):
    """Render an object's call signature, or `(...)` when it has none.

    Builtins and C extension types often refuse introspection. The reference is
    more useful with a placeholder than with a traceback, so failures fall back
    rather than propagate. `drop_self` removes the bound first parameter of a
    method, which the caller never writes out.
    """
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    if drop_self:
        params = list(sig.parameters.values())
        if params and params[0].name in ("self", "cls"):
            sig = sig.replace(parameters=params[1:])
    return str(sig)


def _class_signature(cls):
    """Constructor signature of a class, without the `-> None` dataclasses add."""
    text = _signature(cls)
    if text.endswith(" -> None"):
        text = text[: -len(" -> None")]
    return text


def _slug(heading):
    """Anchor a Markdown heading gets on GitHub, so the index can link to it."""
    out = []
    for ch in heading.lower():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch in " \t":
            out.append("-")
    return "".join(out)


def _public(name):
    return not name.startswith("_")


def iter_modules():
    """Import the package and every submodule under it, in a stable order.

    The root package comes first because that is where the names most people
    reach for live. Everything else is alphabetical by dotted name so that two
    modules called `parse` (one for URDF, one for MJCF) never collide or swap
    places between runs.
    """
    root = importlib.import_module(PACKAGE)
    modules = [root]
    names = sorted(
        info.name
        for info in pkgutil.walk_packages(root.__path__, prefix=PACKAGE + ".")
        if all(_public(part) for part in info.name.split("."))
    )
    for name in names:
        modules.append(importlib.import_module(name))
    return modules


def collect_members(module):
    """Split a module's public API into (functions, classes).

    A member counts as belonging to this module when its `__module__` says so.
    That drops imported helpers (torch, math, a sibling module's function) and
    leaves exactly the things this file defines.
    """
    functions, classes = [], []
    for name, obj in vars(module).items():
        if not _public(name):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isclass(obj):
            classes.append((name, obj))
        elif inspect.isfunction(obj):
            functions.append((name, obj))
    functions.sort(key=lambda item: item[0])
    classes.sort(key=lambda item: item[0])
    return functions, classes


def collect_methods(cls):
    """Public methods and properties written on the class itself.

    Only `cls.__dict__` is walked, so an inherited method is documented once,
    on the base class that defines it.
    """
    out = []
    for name, raw in sorted(vars(cls).items()):
        if not _public(name):
            continue
        if isinstance(raw, property):
            out.append((name, "", _first_paragraph(raw.fget), True))
        elif isinstance(raw, classmethod):
            fn = raw.__func__
            out.append((name, _signature(fn, drop_self=True), _first_paragraph(fn), False))
        elif isinstance(raw, staticmethod):
            fn = raw.__func__
            out.append((name, _signature(fn), _first_paragraph(fn), False))
        elif inspect.isfunction(raw):
            out.append((name, _signature(raw, drop_self=True), _first_paragraph(raw), False))
    return out


def _top_level_table(root):
    """Rows of (exported name, where it is really defined) for the root package."""
    names = getattr(root, "__all__", None) or list(vars(root))
    rows = []
    for name in sorted(n for n in names if _public(n)):
        obj = getattr(root, name, None)
        home = getattr(obj, "__module__", None)
        if home and home != PACKAGE:
            rows.append((name, f"{home}.{name}"))
        else:
            rows.append((name, f"{PACKAGE}.{name}"))
    return rows


def render():
    """Build the whole reference as one Markdown string."""
    modules = iter_modules()
    root = modules[0]
    version = getattr(root, "__version__", "unknown")

    out = []
    out.append(f"# kinfast API reference (version {version})")
    out.append("")
    out.append(
        "Generated by `docs/gen_api.py` by importing the package, so it tracks "
        "the code rather than a hand-kept list. Each entry shows the signature "
        "and the opening paragraph of the docstring; read the source for the "
        "rest. Regenerate with `python docs/gen_api.py` after changing a public "
        "signature or summary."
    )
    out.append("")
    out.append("## Top level names")
    out.append("")
    out.append(
        "These are importable straight from `kinfast`. The second column says "
        "which module defines each one, which is also where it is documented "
        "below."
    )
    out.append("")
    out.append("| Import as | Defined in |")
    out.append("| --- | --- |")
    for name, home in _top_level_table(root):
        out.append(f"| `kinfast.{name}` | `{home}` |")
    out.append("")

    documented = []
    for module in modules:
        functions, classes = collect_members(module)
        if functions or classes:
            documented.append((module, functions, classes))

    out.append("## Modules")
    out.append("")
    for module, _, _ in documented:
        heading = f"`{module.__name__}`"
        out.append(f"- [{heading}](#{_slug(heading)})")
    out.append("")

    for module, functions, classes in documented:
        out.append(f"## `{module.__name__}`")
        out.append("")
        summary = _first_paragraph(module)
        if summary:
            out.append(summary)
            out.append("")
        for name, cls in classes:
            out.append(f"### class `{name}{_class_signature(cls)}`")
            out.append("")
            if not _is_generated_dataclass_doc(cls):
                doc = _first_paragraph(cls)
                if doc:
                    out.append(doc)
                    out.append("")
            if dataclasses.is_dataclass(cls):
                fields = [f.name for f in dataclasses.fields(cls) if _public(f.name)]
                if fields:
                    out.append("Fields: " + ", ".join(f"`{f}`" for f in fields))
                    out.append("")
            methods = collect_methods(cls)
            if methods:
                out.append("Methods:")
                out.append("")
                for mname, msig, mdoc, is_prop in methods:
                    label = f"`{name}.{mname}`" if is_prop else f"`{name}.{mname}{msig}`"
                    suffix = " (property)" if is_prop else ""
                    out.append(f"- {label}{suffix}")
                    if mdoc:
                        out.append(f"  {mdoc}")
                out.append("")
        for name, func in functions:
            out.append(f"### `{name}{_signature(func)}`")
            out.append("")
            doc = _first_paragraph(func)
            if doc:
                out.append(doc)
                out.append("")

    text = "\n".join(out).rstrip() + "\n"
    return text


def default_out():
    return Path(__file__).resolve().parent / "API.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the kinfast API reference.")
    parser.add_argument("--out", default=None, help="path to write (default docs/API.md)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the file on disk differs from the generated text",
    )
    args = parser.parse_args(argv)

    path = Path(args.out) if args.out else default_out()
    text = render()

    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != text:
            print(f"{path} is out of date, run: python docs/gen_api.py")
            # Say what differs. A stale reference is usually one renamed symbol,
            # and printing the diff turns a red CI run into an actionable one
            # instead of a puzzle about someone else's machine.
            import difflib
            diff = difflib.unified_diff(
                current.splitlines(), text.splitlines(),
                fromfile=f"{path.name} (on disk)", tofile="regenerated",
                lineterm="", n=1)
            for i, line in enumerate(diff):
                if i >= 60:
                    print("... (diff truncated)")
                    break
                print(line)
            return 1
        print(f"{path} is up to date")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" keeps the newlines this renderer emits, so a file written on
    # Windows is byte-identical to one written on Linux and --check cannot
    # fail for a reason as uninteresting as line endings
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print(f"wrote {path} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
