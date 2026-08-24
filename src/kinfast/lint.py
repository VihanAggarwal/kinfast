# src/kinfast/lint.py
"""Structural lint for a robot model.

`kinfast.urdf.repair` fixes the handful of defects that would otherwise break
FK/IK, and records what it changed. This module is the wider, read-only pass:
it looks at the whole model and reports what a human should read before
trusting a simulation, a collision query, or a torque number. Nothing here
mutates the IR, and nothing here needs torch, because every rule is about
structure and metadata rather than a batch of configurations.

Findings carry one of three severities:

critical
    The model is malformed. Downstream code either raises or quietly builds the
    wrong kinematic tree: a link dropped by a duplicate name, a joint pointing
    at a link that does not exist, a cycle, a degenerate axis.
major
    The model loads, but a number computed from it will be wrong or
    misleading: a link forward kinematics never visits, a moving link with no
    collision shape, a mass scale no solver can condition.
minor
    Metadata and portability: missing effort or velocity limits, mesh paths
    that only resolve on the machine where the model was authored.

Duplicate *link* names deserve a note. `Robot.links` is a dict keyed by name,
so a URDF that declares the same link twice has already lost one of them by
the time the IR exists. To catch that, `check` takes the original document
text as an optional `source` argument, and `check_urdf_string` / `check_urdf_file`
wire it up for you. Without a source, the rule still fires for an IR built by
hand or by the MJCF parser, where a dict key and `Link.name` can disagree.
"""
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from kinfast.ir import MOVABLE

SEVERITIES = ("critical", "major", "minor")

# A joint span wider than a full turn is meaningless for a revolute joint: the
# extra travel maps onto configurations the joint already reached.
_TWO_PI = 2.0 * math.pi
# Ratio between the heaviest and lightest positive link mass above which the
# mass matrix is badly conditioned and inverse dynamics loses digits.
_MASS_RATIO = 1e4
# Axis norms this far from 1 are called out; compile_robot renormalizes, so the
# loaded robot and the file on disk describe different things.
_AXIS_TOL = 1e-6

_URI_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class Finding:
    """One lint result. `where` names the link, joint, or robot it belongs to."""
    code: str
    severity: str
    where: str
    message: str

    def __str__(self):
        return f"[{self.severity}] {self.code} ({self.where}): {self.message}"


@dataclass
class Report:
    """The findings from one `check` call, in the order the rules ran.

    `len(report)` is the number of findings, so an empty report is falsey.
    `report.ok` is the weaker question most callers actually want: did anything
    critical turn up.
    """
    robot_name: str = "robot"
    findings: list = field(default_factory=list)

    def add(self, code, severity, where, message):
        if severity not in SEVERITIES:
            raise ValueError(f"unknown severity {severity!r}")
        self.findings.append(Finding(code, severity, where, message))

    @property
    def critical(self):
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def major(self):
        return [f for f in self.findings if f.severity == "major"]

    @property
    def minor(self):
        return [f for f in self.findings if f.severity == "minor"]

    @property
    def ok(self):
        """True when nothing critical was found. Major and minor findings are
        worth reading but do not stop the model from loading."""
        return not self.critical

    @property
    def codes(self):
        """The distinct rule codes that fired, in the order they first fired."""
        out = []
        for f in self.findings:
            if f.code not in out:
                out.append(f.code)
        return out

    def by_code(self, code):
        return [f for f in self.findings if f.code == code]

    def __len__(self):
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def summary(self):
        """One line: how many findings, split by severity."""
        if not self.findings:
            return "no findings"
        parts = [f"{len(getattr(self, s))} {s}" for s in SEVERITIES if getattr(self, s)]
        noun = "finding" if len(self.findings) == 1 else "findings"
        return f"{len(self.findings)} {noun}: " + ", ".join(parts)

    def to_markdown(self):
        """Render the report as markdown, grouped by severity, worst first."""
        lines = [f"# Lint report: {self.robot_name}", ""]
        if not self.findings:
            lines += ["No issues found.", ""]
            return "\n".join(lines)
        lines += [self.summary(), ""]
        for sev in SEVERITIES:
            group = getattr(self, sev)
            if not group:
                continue
            lines += [f"## {sev.capitalize()} ({len(group)})", ""]
            for f in group:
                lines.append(f"- `{f.code}` **{f.where}**: {f.message}")
            lines.append("")
        return "\n".join(lines)


def check(robot, source=None):
    """Run every structural rule over a Robot IR and return a `Report`.

    `robot` is a `kinfast.ir.Robot`. `source` is the optional original document
    text (URDF XML); pass it to catch duplicate link names, which the IR cannot
    represent because its link container is a dict.
    """
    report = Report(getattr(robot, "name", "robot") or "robot")
    links = dict(getattr(robot, "links", {}) or {})
    joints = list(getattr(robot, "joints", []) or [])

    _check_duplicate_names(links, joints, source, report)
    _check_joint_references(links, joints, report)
    _check_topology(links, joints, report)
    _check_joint_types(joints, report)
    _check_axes(joints, report)
    _check_limits(joints, report)
    _check_masses(links, report)
    _check_collision_geometry(links, joints, report)
    _check_mesh_paths(links, report)
    return report


def check_urdf_string(text):
    """Parse URDF text and lint it, with the text kept as the source so that
    duplicate link names are visible."""
    from kinfast.urdf.parse import parse_urdf_string
    return check(parse_urdf_string(text), source=text)


def check_urdf_file(path):
    """Same as `check_urdf_string`, reading from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return check_urdf_string(f.read())


# --- naming -----------------------------------------------------------------

def _link_name(key, link):
    """The name a link answers to. Normally the dict key; an IR assembled by
    hand can disagree, and then the object's own name is what everything
    downstream prints."""
    name = getattr(link, "name", None)
    return name if name else key


def _source_names(source, tag):
    """The `name` attributes of every <tag> in a URDF document, in file order.
    Returns an empty list if the text is not parseable XML or is not URDF."""
    if not source:
        return []
    try:
        root = ET.fromstring(source)
    except ET.ParseError:
        return []
    if root.tag != "robot":
        return []
    return [el.get("name") for el in root.findall(tag) if el.get("name")]


def _check_duplicate_names(links, joints, source, report):
    """Two links with one name means the second definition silently replaced
    the first, taking its geometry and inertia with it. Two joints with one
    name makes `joint_names`, `q_index`, and every named lookup ambiguous."""
    reported = set()

    ir_counts = {}
    for key, link in links.items():
        name = _link_name(key, link)
        ir_counts[name] = ir_counts.get(name, 0) + 1
    for name, count in ir_counts.items():
        if count > 1:
            reported.add(name)
            report.add("duplicate_link_name", "critical", name,
                       f"{count} links share this name; only one of them can be "
                       "addressed by name")

    seen = set()
    for name in _source_names(source, "link"):
        if name in seen and name not in reported:
            reported.add(name)
            report.add("duplicate_link_name", "critical", name,
                       "the source declares this link more than once; the later "
                       "definition replaced the earlier one and its geometry, "
                       "inertia, and mass were dropped")
        seen.add(name)

    jcounts = {}
    for j in joints:
        jcounts[j.name] = jcounts.get(j.name, 0) + 1
    for name, count in jcounts.items():
        if count > 1:
            report.add("duplicate_joint_name", "critical", name,
                       f"{count} joints share this name; lookups by joint name "
                       "and the order of q are ambiguous")


# --- references and topology ------------------------------------------------

def _check_joint_references(links, joints, report):
    """A joint whose parent or child is not a declared link cannot be placed in
    the tree, and a joint whose parent is its own child is not a joint."""
    for j in joints:
        for role in ("parent", "child"):
            name = getattr(j, role, None)
            if not name:
                report.add("missing_link_reference", "critical", j.name,
                           f"joint has no {role} link")
            elif name not in links:
                report.add("missing_link_reference", "critical", j.name,
                           f"{role} link {name!r} is not defined")
        if j.parent and j.parent == j.child:
            report.add("self_loop", "critical", j.name,
                       f"parent and child are both {j.parent!r}")


def _reachable(root, children):
    """Breadth-first set of link names reachable from `root`. Cycle safe."""
    seen = {root}
    frontier = [root]
    while frontier:
        node = frontier.pop(0)
        for kid in children.get(node, ()):
            if kid not in seen:
                seen.add(kid)
                frontier.append(kid)
    return seen


def _check_topology(links, joints, report):
    """A URDF is a tree. Two joints claiming the same child, a link with no
    path from the root, or no root at all each mean the compiled chain is not
    the robot the author drew."""
    if not links:
        report.add("empty_model", "critical", report.robot_name,
                   "the model declares no links")
        return

    parents_of = {}
    children = {}
    for j in joints:
        if not j.child or not j.parent:
            continue
        if j.parent == j.child:
            continue  # already reported; do not let it poison the walk
        parents_of.setdefault(j.child, []).append(j.name)
        children.setdefault(j.parent, []).append(j.child)

    for child, owners in parents_of.items():
        if len(owners) > 1:
            report.add("multiple_parents", "critical", child,
                       "link is the child of " + str(len(owners)) + " joints ("
                       + ", ".join(owners) + "); a URDF must be a tree")

    roots = [n for n in links if n not in parents_of]
    if not roots:
        report.add("no_root", "critical", report.robot_name,
                   "every link has a parent joint, so the model contains a "
                   "cycle and has no base link")
        return

    # With more than one root the model is several disconnected trees. Take the
    # biggest as the real robot so the report names the stragglers rather than
    # the whole arm.
    best_root, best = roots[0], _reachable(roots[0], children)
    for r in roots[1:]:
        got = _reachable(r, children)
        if len(got) > len(best):
            best_root, best = r, got
    for name in links:
        if name not in best:
            report.add("unreachable_link", "major", name,
                       f"not reachable from root link {best_root!r}; forward "
                       "kinematics never visits it")


# --- joints -----------------------------------------------------------------

def _check_joint_types(joints, report):
    """kinfast models revolute, continuous, prismatic, and fixed joints. Any
    other type compiles down to a fixed joint, which is a silent loss of a
    degree of freedom."""
    for j in joints:
        t = getattr(j, "type", None)
        if not t:
            report.add("missing_joint_type", "critical", j.name,
                       "joint has no type")
        elif t not in MOVABLE and t != "fixed":
            report.add("unsupported_joint_type", "major", j.name,
                       f"type {t!r} is not modelled; it compiles to a fixed "
                       "joint and its degrees of freedom disappear")


def _check_axes(joints, report):
    """FK rotates about a unit axis, so compile_robot renormalizes whatever the
    file says. A zero axis has no direction to recover, and a non-unit axis
    means the file and the loaded robot disagree: tools that scale prismatic
    travel by the axis length will build a different machine."""
    for j in joints:
        if j.type not in MOVABLE:
            continue
        axis = tuple(getattr(j, "axis", (0.0, 0.0, 1.0)))
        norm = math.sqrt(sum(float(c) * float(c) for c in axis))
        if not math.isfinite(norm) or norm < 1e-12:
            report.add("zero_axis", "critical", j.name,
                       f"axis {axis} has no direction; the joint is degenerate")
        elif abs(norm - 1.0) > _AXIS_TOL:
            report.add("non_unit_axis", "major", j.name,
                       f"axis {axis} has norm {norm:.6g}; kinfast normalizes it, "
                       "but the file describes a different joint than it loads as")


def _check_limits(joints, report):
    """Velocity and effort limits are optional in URDF and absent in most
    hand-written files, which leaves trajectory scaling and torque saturation
    with nothing to work from. A revolute span wider than a full turn is a
    joint that should have been declared continuous."""
    for j in joints:
        if j.type not in MOVABLE:
            continue
        if not getattr(j, "velocity", 0.0) or j.velocity <= 0.0:
            report.add("missing_velocity_limit", "minor", j.name,
                       "no positive velocity limit; time scaling has no bound "
                       "to respect")
        if not getattr(j, "effort", 0.0) or j.effort <= 0.0:
            report.add("missing_effort_limit", "minor", j.name,
                       "no positive effort limit; torque saturation cannot be "
                       "checked")
        lo, hi = j.limit
        if j.type == "revolute" and math.isfinite(lo) and math.isfinite(hi):
            span = hi - lo
            if span > _TWO_PI + 1e-9:
                report.add("revolute_range_over_2pi", "major", j.name,
                           f"range [{lo:g}, {hi:g}] spans {span:g} rad, more "
                           "than a full turn; declare it continuous instead")


# --- mass, collision, meshes ------------------------------------------------

def _check_masses(links, report):
    """Inverse dynamics and the mass matrix are conditioned by the spread of
    link masses. Past about four orders of magnitude the small links vanish
    into rounding, and the usual cause is a unit mistake in one link."""
    masses = [(name, link.inertial.mass) for name, link in links.items()
              if getattr(link, "inertial", None) is not None
              and link.inertial.mass > 0.0]
    if len(masses) < 2:
        return
    light = min(masses, key=lambda kv: kv[1])
    heavy = max(masses, key=lambda kv: kv[1])
    if heavy[1] > light[1] * _MASS_RATIO:
        ratio = heavy[1] / light[1]
        report.add("mass_ratio", "major", heavy[0],
                   f"mass {heavy[1]:g} kg is {ratio:.3g}x link {light[0]!r} at "
                   f"{light[1]:g} kg; check the units before trusting any "
                   "dynamics from this model")


def _check_collision_geometry(links, joints, report):
    """A link that moves and has no collision shape is invisible to every
    distance query, so self collision checks and obstacle avoidance quietly
    pass through it."""
    for j in joints:
        if j.type not in MOVABLE:
            continue
        link = links.get(j.child)
        if link is None:
            continue  # missing_link_reference already covers this
        if getattr(link, "collision", None) is None:
            report.add("no_collision_geometry", "major", j.child,
                       f"moving link (driven by joint {j.name!r}) has no "
                       "collision geometry; distance queries ignore it")


def _mesh_path_finding(where, path, report):
    m = _URI_SCHEME.match(path)
    if m:
        scheme = m.group(1).lower()
        if scheme == "file":
            report.add("absolute_mesh_path", "minor", where,
                       f"mesh path {path!r} is an absolute file URI and will "
                       "not resolve on another machine")
        else:
            report.add("package_mesh_path", "minor", where,
                       f"mesh path {path!r} uses the {scheme}:// scheme, which "
                       "only resolves inside a ROS or Gazebo workspace")
        return
    if path.startswith("/") or path.startswith("\\") or _WINDOWS_ABS.match(path):
        report.add("absolute_mesh_path", "minor", where,
                   f"mesh path {path!r} is absolute and will not resolve on "
                   "another machine")


def _check_mesh_paths(links, report):
    """Mesh references that only resolve on the author's machine are the most
    common reason a model that works for one person fails for the next."""
    for name, link in links.items():
        for slot in ("visual", "collision"):
            geo = getattr(link, slot, None)
            if geo is None or geo.kind != "mesh" or not geo.mesh_path:
                continue
            _mesh_path_finding(f"{name}.{slot}", geo.mesh_path, report)
