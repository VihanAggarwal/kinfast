# src/kinfast/codegen.py
"""The kinfast compiler: robot-specific code generation for microsecond
single-query kinematics.

PyTorch is near-optimal for large batches (matmul-bound) but pays ~0.5 ms of
framework overhead per call, which kills control loops and planners that need
one FK at a time. This module removes Python-the-framework from the hot path:
at load time it generates straight-line Python source SPECIALIZED to one robot,
with the tree unrolled, every constant folded, and every multiply-by-zero from
axis-aligned joints and identity origins eliminated at generation time. The
result is a few hundred fused multiply-adds per call: raw arithmetic, no loops,
no tensors, no dispatch.

    fast = robot.compile()        # generate + exec the specialized source
    T    = fast.fk(q_list)        # single-query FK in microseconds
    J    = fast.jacobian(q_list)  # geometric Jacobian from the same pass
    q    = fast.ik(target_xyz)    # damped-least-squares IK with restarts

This is deliberately the seed of the kinfast language roadmap: the same IR
already feeds the batched torch backend; this is its scalar backend.
"""
import math

import numpy as np


# ---------------------------------------------------------------- expression IR
def _f(x):
    """Fold float-ish values; keep tiny numbers as exact 0 to maximize pruning."""
    if isinstance(x, float) and abs(x) < 1e-12:
        return 0.0
    return x


def _mul(a, b):
    a, b = _f(a), _f(b)
    if a == 0.0 or b == 0.0:
        return 0.0
    if a == 1.0:
        return b
    if b == 1.0:
        return a
    if isinstance(a, float) and isinstance(b, float):
        return a * b
    return f"{_lit(a)}*{_lit(b)}"


def _add(*terms):
    out, const = [], 0.0
    for t in terms:
        t = _f(t)
        if isinstance(t, float):
            const += t
        elif t != 0.0:
            out.append(t)
    if const != 0.0 or not out:
        out.append(const)
    if len(out) == 1:
        return out[0]
    return "(" + "+".join(_lit(t) for t in out) + ")"


def _lit(x):
    return repr(x) if isinstance(x, float) else x


class _Emitter:
    """Accumulates generated statements; assigns temporaries for compound exprs."""

    def __init__(self):
        self.lines = []
        self.n = 0

    def temp(self, expr, prefix="t"):
        expr = _f(expr)
        if isinstance(expr, float):
            return expr                    # constants stay constants
        if isinstance(expr, str) and expr.isidentifier():
            return expr                    # already a bare variable
        name = f"{prefix}{self.n}"
        self.n += 1
        self.lines.append(f"    {name}={expr}")
        return name


def _mat3_mul(E, A, B, prefix):
    """3x3 product of expression matrices, constant-folded, temporied."""
    return [[E.temp(_add(_mul(A[i][0], B[0][j]),
                         _mul(A[i][1], B[1][j]),
                         _mul(A[i][2], B[2][j])), prefix)
             for j in range(3)] for i in range(3)]


def _mat3_vec(A, v):
    return [_add(_mul(A[i][0], v[0]), _mul(A[i][1], v[1]), _mul(A[i][2], v[2]))
            for i in range(3)]


# ------------------------------------------------------------------- generation
def generate_fk_source(chain):
    """Generate specialized FK source for this robot.

    The emitted function writes 12 floats per link (row-major 3x3 rotation then
    position) into a flat list `out` of length n_links*12.
    """
    E = _Emitter()
    n = chain.n_links
    origin = chain.joint_origin.detach().cpu().double().numpy()
    axes = chain.joint_axis.detach().cpu().double().numpy()
    jtype = chain.joint_type.detach().cpu().numpy()
    qidx = chain.q_index.detach().cpu().numpy()
    parent = chain.parent.detach().cpu().numpy()

    R = [None] * n   # per-link world rotation, 3x3 of exprs
    P = [None] * n   # per-link world position, 3 exprs

    for i in chain.topo_order:
        p = int(parent[i])
        O = origin[i]
        OR = [[_f(float(O[a][b])) for b in range(3)] for a in range(3)]
        Op = [_f(float(O[a][3])) for a in range(3)]
        if p < 0:
            AR = [[1.0 if a == b else 0.0 for b in range(3)] for a in range(3)]
            Ap = [0.0, 0.0, 0.0]
        else:
            AR, Ap = R[p], P[p]
        # A = parent @ origin (identity origins fold to aliases automatically)
        is_ident_R = all(OR[a][b] == (1.0 if a == b else 0.0)
                         for a in range(3) for b in range(3))
        BR = AR if is_ident_R else _mat3_mul(E, AR, OR, f"a{i}_")
        Bp = [E.temp(_add(Ap[a], *_mat3_vec([AR[a]], Op) if False else
                          [_mul(AR[a][k], Op[k]) for k in range(3)]), f"o{i}_")
              for a in range(3)] if any(o != 0.0 for o in Op) else Ap

        jt = int(jtype[i])
        if jt == 0:                                  # fixed
            R[i], P[i] = BR, Bp
        elif jt == 1:                                # revolute
            k = int(qidx[i])
            c, s = f"c{k}", f"s{k}"
            x, y, z = (float(axes[i][0]), float(axes[i][1]), float(axes[i][2]))
            # Rodrigues entries as k_c*c + k_s*s + k_1 with folded coefficients
            def rot(a, b):
                kron = 1.0 if a == b else 0.0
                aa = [x, y, z]
                eps = [[0, -aa[2], aa[1]], [aa[2], 0, -aa[0]], [-aa[1], aa[0], 0]]
                k1 = aa[a] * aa[b]                    # from C = 1-c
                kc = kron - k1
                ks = eps[a][b]
                return _add(_mul(_f(kc), c), _mul(_f(float(ks)), s), _f(k1))
            M = [[rot(a, b) for b in range(3)] for a in range(3)]
            R[i] = _mat3_mul(E, BR, M, f"r{i}_")
            P[i] = Bp
        else:                                        # prismatic
            k = int(qidx[i])
            ax = [float(axes[i][0]), float(axes[i][1]), float(axes[i][2])]
            d = _mat3_vec(BR, [_f(a) for a in ax])
            R[i] = BR
            P[i] = [E.temp(_add(Bp[a], _mul(f"q{k}", d[a])), f"p{i}_")
                    for a in range(3)]

    header = ["def _fk(q, out):"]
    for k in range(chain.dof):
        header.append(f"    q{k}=q[{k}]")
    for i in range(n):
        if int(jtype[i]) == 1:
            k = int(qidx[i])
            header.append(f"    c{k}=cos(q{k}); s{k}=sin(q{k})")
    stores = []
    for i in range(n):
        base = i * 12
        for a in range(3):
            for b in range(3):
                stores.append(f"    out[{base + a*3 + b}]={_lit(R[i][a][b])}")
        for a in range(3):
            stores.append(f"    out[{base + 9 + a}]={_lit(P[i][a])}")
    return "\n".join(header + E.lines + stores + ["    return out"])


class CompiledRobot:
    """Scalar backend: microsecond single-query FK / Jacobian / IK for ONE robot.

    Complements the batched torch path; produced by `Robot.compile()`.
    """

    def __init__(self, chain):
        self.chain = chain
        self.n_links = chain.n_links
        self.dof = chain.dof
        self.source = generate_fk_source(chain)
        ns = {"cos": math.cos, "sin": math.sin}
        exec(compile(self.source, "<kinfast-codegen>", "exec"), ns)
        self._fk = ns["_fk"]
        self._out = [0.0] * (chain.n_links * 12)
        self.lower = chain.lower.detach().cpu().double().numpy()
        self.upper = chain.upper.detach().cpu().double().numpy()
        # per-link chain-to-root of (link_index, qcol, jtype, axis) for jacobian
        parent = chain.parent.detach().cpu().numpy()
        jtype = chain.joint_type.detach().cpu().numpy()
        qidx = chain.q_index.detach().cpu().numpy()
        axes = chain.joint_axis.detach().cpu().double().numpy()
        self._paths = []
        for L in range(chain.n_links):
            path, i = [], L
            while i >= 0:
                if jtype[i] != 0:
                    path.append((i, int(qidx[i]), int(jtype[i]),
                                 (float(axes[i][0]), float(axes[i][1]), float(axes[i][2]))))
                i = int(parent[i])
            self._paths.append(path)

    # ---- single-query kinematics ----
    def fk(self, q, link=None):
        """q: sequence of dof floats. Returns (n_links,4,4) array, or (4,4) for
        one link index."""
        o = self._fk(list(q), self._out)
        M = np.empty((self.n_links, 4, 4))
        M[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
        flat = np.asarray(o).reshape(self.n_links, 12)
        M[:, :3, :3] = flat[:, :9].reshape(-1, 3, 3)
        M[:, :3, 3] = flat[:, 9:]
        return M if link is None else M[link]

    def _raw(self, q):
        return self._fk(list(q), self._out)

    def jacobian(self, q, link):
        """Geometric Jacobian (6,dof) at q for a link index. Pure-scalar math."""
        o = self._raw(q)
        J = np.zeros((6, self.dof))
        b = link * 12
        ex, ey, ez = o[b + 9], o[b + 10], o[b + 11]
        for (i, col, jt, (ax, ay, az)) in self._paths[link]:
            c = i * 12
            r = o[c:c + 9]
            wx = r[0] * ax + r[1] * ay + r[2] * az
            wy = r[3] * ax + r[4] * ay + r[5] * az
            wz = r[6] * ax + r[7] * ay + r[8] * az
            if jt == 1:
                dx, dy, dz = ex - o[c + 9], ey - o[c + 10], ez - o[c + 11]
                J[0, col] = wy * dz - wz * dy
                J[1, col] = wz * dx - wx * dz
                J[2, col] = wx * dy - wy * dx
                J[3, col], J[4, col], J[5, col] = wx, wy, wz
            else:
                J[0, col], J[1, col], J[2, col] = wx, wy, wz
        return J

    def ik(self, target_xyz, link, q0=None, iters=100, damping=0.05,
           tol=1e-4, restarts=8, seed=0):
        """Position IK, damped least squares with random restarts.
        Returns (q, err). Single-query, scalar-path throughout."""
        rng = np.random.RandomState(seed)
        tgt = np.asarray(target_xyz, dtype=float)
        best_q, best_e = None, np.inf
        for r in range(restarts):
            if r == 0 and q0 is not None:
                q = np.asarray(q0, dtype=float).copy()
            else:
                q = self.lower + (self.upper - self.lower) * rng.rand(self.dof)
            lam2 = damping * damping
            for _ in range(iters):
                o = self._raw(q)
                b = link * 12
                e = tgt - np.array(o[b + 9:b + 12])
                en = float(np.linalg.norm(e))
                if en < tol:
                    break
                J = self.jacobian(q, link)[:3]
                H = J @ J.T + lam2 * np.eye(3)
                q += J.T @ np.linalg.solve(H, e)
                np.clip(q, self.lower, self.upper, out=q)
            o = self._raw(q)
            en = float(np.linalg.norm(tgt - np.array(o[link*12+9:link*12+12])))
            if en < best_e:
                best_q, best_e = q.copy(), en
            if best_e < tol:
                break
        return best_q, best_e
