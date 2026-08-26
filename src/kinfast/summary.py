# src/kinfast/summary.py
"""A human-readable report of what a loaded robot actually is.

`summary(robot)` answers the questions you ask in the first minute after
loading a file you did not write: how many joints move, what their limits are,
where the mass sits, roughly how far the arm can reach, and whether the repair
pass had to touch anything. It prints as an aligned plain-text table, and
`.to_markdown()` renders the same content for a README or a bug report.

Every number is read from the *compiled chain*, not from the raw file, so what
the report shows is what `fk`, `jacobian`, and `dynamics` are using. If the
loader repaired a missing joint limit or normalized an axis, the repaired value
is what you see, and the repair findings are counted in the header.

There is nothing batched or differentiable here on purpose: a summary is a
report about the model itself, not about a configuration, so it takes no q and
has no leading batch dimension. It is device and dtype agnostic in the sense
that matters: the constants are pulled to the CPU as plain Python floats, so it
works the same on a float32 CPU chain and a float64 CUDA one.
"""
import copy
import math

import torch

_TYPE_NAME = {0: "fixed", 1: "revolute", 2: "prismatic"}


# ---------------------------------------------------------------- formatting

def _fmt(x, sig=6):
    """A number as short readable text. `None` prints as a dash, so a column
    can carry 'this joint has no limit' without a second column to say so."""
    if x is None:
        return "-"
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    s = f"{x:.{sig}g}"
    return "0" if s in ("-0", "-0.0") else s


def _widths(headers, rows):
    w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            w[i] = max(w[i], len(cell))
    return w


def _text_table(headers, rows, right=()):
    """Column-aligned plain text: header, a dashed rule, then the rows.
    `right` is the set of column indices to right-align (the numbers)."""
    w = _widths(headers, rows)

    def line(cells):
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.rjust(w[i]) if i in right else cell.ljust(w[i]))
        return "  ".join(out).rstrip()

    return "\n".join([line(headers), line(["-" * n for n in w])]
                     + [line(r) for r in rows])


def _md_table(headers, rows, right=()):
    """The same table as a GitHub-flavoured markdown pipe table."""
    def line(cells):
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    rule = ["---:" if i in right else "---" for i in range(len(headers))]
    return "\n".join([line(headers), line(rule)] + [line(r) for r in rows])


def _key_values(pairs, indent="  "):
    """An indented `key   value` block, used for the report header. A table
    would put a dashed rule under the robot's name, which reads as a column
    heading rather than as the thing the report is about."""
    w = max((len(k) for k, _ in pairs), default=0)
    return "\n".join(f"{indent}{k.ljust(w)}  {v}".rstrip() for k, v in pairs)


# ---------------------------------------------------------------- reach

def reach_estimate(chain):
    """Upper bound on how far any link origin can sit from the base origin.

    Walking the tree, a revolute or fixed joint moves its child's origin by
    exactly the joint origin offset (rotating about an axis through that point
    does not move the point), and a prismatic joint adds at most the largest
    magnitude its limit allows. Summing those along a branch and taking the
    triangle inequality gives a bound that no configuration can beat.

    It is a bound, not the reach: an arm that cannot straighten out will never
    touch it. For the usual serial arm whose links can line up it is tight, and
    unlike a Monte-Carlo estimate it is exact, instant, and the same every run.
    Returns (bound, link_name); the link is the one attaining the bound.
    """
    offsets = chain.joint_origin[:, :3, 3].detach().cpu().to(torch.float64)
    step = offsets.norm(dim=-1).tolist()
    jtype = chain.joint_type.detach().cpu().tolist()
    q_index = chain.q_index.detach().cpu().tolist()
    parent = chain.parent.detach().cpu().tolist()
    lower = chain.lower.detach().cpu().to(torch.float64).tolist()
    upper = chain.upper.detach().cpu().to(torch.float64).tolist()

    cum = [0.0] * chain.n_links
    for i in chain.topo_order:
        p = parent[i]
        if p < 0:
            cum[i] = 0.0
            continue
        travel = 0.0
        if jtype[i] == 2 and q_index[i] >= 0:
            k = q_index[i]
            travel = max(abs(lower[k]), abs(upper[k]))
        cum[i] = cum[p] + step[i] + travel
    best = max(range(chain.n_links), key=lambda i: cum[i]) if chain.n_links else 0
    name = chain.link_names[best] if chain.n_links else ""
    return cum[best] if chain.n_links else 0.0, name


def sampled_reach(chain, n=2048, seed=0):
    """Farthest any link origin got from the base over `n` random configs.

    The bound above is exact but loose on an arm whose links cannot line up
    (the Panda's bound is about 1.4 m against a real reach near 1.0 m), so the
    report shows this next to it. It is a lower bound on the true reach and it
    creeps up with n, which is why it is labelled as sampled. Seeded, so two
    runs of the same report print the same number.

    Returns (reach, link_name). Raises ValueError, via `sampling_bounds`, for
    a prismatic joint with no finite limit to sample inside.
    """
    from kinfast.analysis import sampling_bounds
    from kinfast.fk import forward_kinematics
    if chain.dof == 0:
        n = 1                    # nothing to sample; one pose is the whole set
    lo, hi = sampling_bounds(chain)
    lo, hi = lo.detach().cpu(), hi.detach().cpu()
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.rand(n, chain.dof, generator=g, dtype=lo.dtype)
    q = (lo + (hi - lo) * u).to(device=chain.lower.device)
    with torch.no_grad():
        pts = forward_kinematics(chain, q)[:, :, :3, 3]      # (n, links, 3)
        per_link = pts.norm(dim=-1).max(dim=0).values         # (links,)
    best = int(torch.argmax(per_link))
    return float(per_link[best]), chain.link_names[best]


# ---------------------------------------------------------------- the report

class Summary:
    """The contents of the report, as data plus two renderers.

    The fields are public on purpose: a caller who wants the numbers rather
    than the table (a test, a CI gate, a web view) should read `total_mass` or
    `joints` instead of parsing text back out of `to_text()`.
    """

    def __init__(self, name, dof, n_links, joints, links, total_mass,
                 no_inertial, reach, reach_link, findings, notes,
                 sampled=None, sampled_link=None, samples=0):
        self.name = name
        self.dof = dof
        self.n_links = n_links
        self.joints = joints          # list of dicts, one per joint
        self.links = links            # list of dicts, one per link
        self.total_mass = total_mass
        self.no_inertial = no_inertial  # links with no <inertial> at all
        self.reach = reach            # exact upper bound
        self.reach_link = reach_link
        self.sampled = sampled        # Monte-Carlo reach, None if not sampled
        self.sampled_link = sampled_link
        self.samples = samples
        self.findings = findings      # repair.Finding list, or None if unknown
        self.notes = notes            # parser notes from the IR

    # ---- derived text pieces, shared by both renderers ----
    @property
    def n_movable(self):
        return sum(1 for j in self.joints if j["q"] is not None)

    def _findings_text(self):
        if self.findings is None:
            return "not checked (no source model kept)"
        if not self.findings:
            return "0"
        counts = {}
        for f in self.findings:
            counts[f.code] = counts.get(f.code, 0) + 1
        detail = ", ".join(f"{c} x{n}" if n > 1 else c
                           for c, n in sorted(counts.items()))
        return f"{len(self.findings)} ({detail})"

    def _mass_text(self):
        txt = f"{_fmt(self.total_mass)} kg"
        if self.no_inertial:
            n = self.no_inertial
            txt += (f" ({n} of {self.n_links} links "
                    f"{'has' if n == 1 else 'have'} no inertial)")
        return txt

    def _header_rows(self):
        n_fixed = len(self.joints) - self.n_movable
        rows = [
            ("dof", str(self.dof)),
            ("links", str(self.n_links)),
            ("joints", f"{len(self.joints)} ({self.n_movable} movable, "
                       f"{n_fixed} fixed)"),
            ("total mass", self._mass_text()),
        ]
        if self.sampled is not None:
            over = (f" over {self.samples} configs" if self.samples > 1
                    else "")
            rows.append(("reach (sampled)",
                         f"{_fmt(self.sampled)} m at {self.sampled_link}{over}"))
        bound = f"{_fmt(self.reach)} m"
        if self.reach_link:
            bound += f" at {self.reach_link}"
        rows.append(("reach (bound)", bound))
        rows.append(("repair findings", self._findings_text()))
        return rows

    _JOINT_HEADERS = ("joint", "type", "q", "lower", "upper", "velocity")
    _JOINT_RIGHT = frozenset({2, 3, 4, 5})
    _LINK_HEADERS = ("link", "mass")
    _LINK_RIGHT = frozenset({1})

    def _joint_rows(self):
        rows = []
        for j in self.joints:
            rows.append((
                j["name"],
                j["type"],
                "-" if j["q"] is None else str(j["q"]),
                _fmt(j["lower"]),
                _fmt(j["upper"]),
                _fmt(j["velocity"]),
            ))
        return rows

    def _link_rows(self):
        rows = [(l["name"], _fmt(l["mass"])) for l in self.links]
        rows.append(("total", _fmt(self.total_mass)))
        return rows

    # ---- renderers ----
    def to_text(self):
        """The report as an aligned plain-text table."""
        head = (f"robot: {self.name or '(unnamed)'}\n"
                + _key_values(self._header_rows()))
        joints = (_text_table(self._JOINT_HEADERS, self._joint_rows(),
                              right=self._JOINT_RIGHT)
                  if self.joints else "(no joints)")
        parts = [head, "", joints]
        parts += ["", _text_table(self._LINK_HEADERS, self._link_rows(),
                                  right=self._LINK_RIGHT)]
        if self.notes:
            parts += ["", "parser notes"] + [f"  - {n}" for n in self.notes]
        if self.findings:
            parts += ["", "repair findings"]
            parts += [f"  - {f.code} on {f.where}: {f.message}"
                      for f in self.findings]
        return "\n".join(parts) + "\n"

    def to_markdown(self):
        """The same report as markdown, for a README or an issue."""
        parts = [f"# {self.name or 'robot'}", ""]
        parts.append(_md_table(("property", "value"),
                               [(k, v) for k, v in self._header_rows()]))
        parts += ["", "## Joints", ""]
        if self.joints:
            parts.append(_md_table(self._JOINT_HEADERS, self._joint_rows(),
                                   right=self._JOINT_RIGHT))
        else:
            parts.append("_no joints_")
        parts += ["", "## Links", ""]
        parts.append(_md_table(self._LINK_HEADERS, self._link_rows(),
                               right=self._LINK_RIGHT))
        if self.notes:
            parts += ["", "## Parser notes", ""]
            parts += [f"- {n}" for n in self.notes]
        if self.findings:
            parts += ["", "## Repair findings", ""]
            parts += [f"- `{f.code}` on `{f.where}`: {f.message}"
                      for f in self.findings]
        return "\n".join(parts) + "\n"

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return (f"<Summary {self.name!r} dof={self.dof} links={self.n_links} "
                f"mass={_fmt(self.total_mass)}>")


def _chain_of(robot):
    """Accept a kinfast.Robot, a bare CompiledChain, or anything holding one."""
    chain = getattr(robot, "chain", robot)
    if not hasattr(chain, "link_names"):
        raise TypeError(
            "summary() needs a kinfast.Robot or a CompiledChain, got "
            f"{type(robot).__name__}")
    return chain


def _repair_findings(ir):
    """What the repair pass finds in this model.

    Loading already repairs, and `repair` mutates in place, so the count here
    is normally the leftovers that repair only *detects* (a mass whose inertia
    tensor breaks the triangle inequality, say) rather than fixes. A robot
    loaded with `repair_model=False`, or an IR you parsed yourself, reports
    everything. The check runs on a copy so nothing is mutated behind the
    caller's back.
    """
    from kinfast.urdf.repair import repair
    _, findings = repair(copy.deepcopy(ir))
    return findings


def summary(robot, findings=None, reach_samples=2048, seed=0):
    """Build the report for a loaded robot.

    `robot` is a `kinfast.Robot` (or a bare `CompiledChain`, with fewer
    labels: a chain has forgotten the names of its fixed joints and cannot
    tell a continuous joint from a revolute one). `findings` overrides the
    repair findings, for the caller who kept the list `repair()` returned
    before the model was compiled; leave it None to have them recomputed.

    `reach_samples` is how many random configurations the sampled reach is
    measured over (seeded by `seed`, so the report is reproducible); pass 0 to
    skip it and show only the exact upper bound. Sampling is skipped anyway
    when the model has a joint that cannot be sampled, such as a prismatic
    joint with no finite limit.

    Returns a `Summary`. `str()` of it is the plain-text table and
    `.to_markdown()` is the markdown one.
    """
    chain = _chain_of(robot)
    ir = getattr(robot, "ir", None)

    lower = chain.lower.detach().cpu().to(torch.float64).tolist()
    upper = chain.upper.detach().cpu().to(torch.float64).tolist()
    vmax = chain.vmax.detach().cpu().to(torch.float64).tolist()
    mass = chain.link_mass.detach().cpu().to(torch.float64).tolist()
    q_index = chain.q_index.detach().cpu().tolist()
    jtype = chain.joint_type.detach().cpu().tolist()

    # Joint rows. The IR knows the fixed joints and the exact type word
    # ("continuous" compiles to the same code as "revolute"); the chain only
    # knows the movable ones. Prefer the IR when it is there.
    joint_rows = []
    if ir is not None:
        by_child = {j.child: j for j in ir.joints}
        for name in chain.link_names:
            j = by_child.get(name)
            if j is None:
                continue
            i = chain.link_index[name]
            k = int(q_index[i])
            joint_rows.append({
                "name": j.name,
                "type": j.type,
                "q": None if k < 0 else k,
                "lower": None if k < 0 else lower[k],
                "upper": None if k < 0 else upper[k],
                "velocity": None if k < 0 else vmax[k],
            })
        # a joint whose child link was dropped would vanish above; keep it
        seen = {r["name"] for r in joint_rows}
        for j in ir.joints:
            if j.name not in seen:
                joint_rows.append({"name": j.name, "type": j.type, "q": None,
                                   "lower": None, "upper": None,
                                   "velocity": None})
    else:
        for i in range(chain.n_links):
            k = int(q_index[i])
            if k < 0:
                continue
            joint_rows.append({
                "name": chain.joint_names[k],
                "type": _TYPE_NAME.get(int(jtype[i]), "unknown"),
                "q": k,
                "lower": lower[k],
                "upper": upper[k],
                "velocity": vmax[k],
            })
        joint_rows.sort(key=lambda r: r["q"])

    link_rows = [{"name": n, "mass": mass[chain.link_index[n]]}
                 for n in chain.link_names]
    total_mass = math.fsum(r["mass"] for r in link_rows)
    if ir is not None:
        no_inertial = sum(1 for n in chain.link_names
                          if getattr(ir.links.get(n), "inertial", None) is None)
    else:
        no_inertial = 0

    reach, reach_link = reach_estimate(chain)
    sampled = sampled_link = None
    n_samples = 0
    if reach_samples and chain.n_links:
        try:
            sampled, sampled_link = sampled_reach(chain, n=int(reach_samples),
                                                  seed=seed)
            n_samples = 1 if chain.dof == 0 else int(reach_samples)
        except ValueError:
            # an unsampleable joint (a prismatic one with no finite limit);
            # the exact bound is still worth printing, so report only that
            sampled = sampled_link = None

    if findings is None and ir is not None:
        findings = _repair_findings(ir)

    return Summary(
        name=getattr(ir, "name", None) or getattr(robot, "name", None),
        dof=chain.dof,
        n_links=chain.n_links,
        joints=joint_rows,
        links=link_rows,
        total_mass=total_mass,
        no_inertial=no_inertial,
        reach=reach,
        reach_link=reach_link,
        sampled=sampled,
        sampled_link=sampled_link,
        samples=n_samples,
        findings=findings,
        notes=list(getattr(ir, "parse_notes", None) or []),
    )
