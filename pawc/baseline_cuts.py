"""Baseline B — cut enumeration, the O(N^2 log N) reference solver.

Realises the *exact cut envelope* of the draft, Corollary 3.1:

    C_k^o = min_{r = 1..N} C_{k,r}^line,

where ``C_{k,r}^line`` is the cardinality-``k`` partial transport cost on the line
obtained by cutting the circle inside the ``r``-th original support gap ``G_r`` and
unwrapping (draft Prop. 3.1). Each line profile is produced by one call to the
vendored PAWL solver, so ``N`` cuts cost ``O(N^2 log N)``.

Two interchangeable line solvers are provided:

``line_profile_pawl``
    The vendored PAWL (``vendor/partial.py``, Chapel & Tavenard ICLR 2025). Fast,
    float64, and compiled with ``fastmath=True``.
``line_profile_dp``
    A self-contained O(n m K) dynamic program over *non-crossing* matchings. On the
    line the ground cost ``|u - v|`` satisfies the Monge condition, so an optimal
    partial matching may be taken non-crossing and the DP is exact. Being pure
    Python arithmetic, it accepts ``fractions.Fraction`` coordinates and therefore
    supplies the exact-arithmetic cross-check demanded by CLAUDE.md when a float
    comparison sits near tolerance.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from .circle import gap_midpoints, sorted_union, unroll_at
from .solution import PartialCircleSolution

# `vendor` is a top-level directory of the repo, not part of the `pawc` package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = [
    "all_cut_costs_exact",
    "line_profile_pawl",
    "line_profile_dp",
    "partial_ot_circle_cuts",
    "partial_ot_circle_cuts_exact",
    "line_profile_at_cut",
]


def line_profile_pawl(xu, yu, K: int | None = None):
    """Full line partial-OT profile via the vendored PAWL.

    Parameters
    ----------
    xu, yu : (n,) / (m,) float arrays
        Unwrapped source and target coordinates on the line.
    K : int, optional
        ``min(n, m)`` by default.

    Returns
    -------
    costs : (K+1,) float array
        Unweighted cumulative costs; ``costs[k]`` is the optimal cardinality-``k``
        line cost (``costs[0] == 0``).
    order_x, order_y : (K,) int arrays
        Activation order: for every ``k``, ``order_x[:k]`` / ``order_y[:k]`` index
        the active source / target atoms of an optimal ``A_k`` in ``xu`` / ``yu``.
        PAWL's active sets are nested by construction.
    """
    try:
        from vendor.partial import partial_ot_1d  # noqa: PLC0415  (vendored, lazy import)
    except ImportError as exc:  # pragma: no cover - depends on the checkout
        raise ImportError(
            "the PAWL reference implementation is not present. It is not "
            "redistributed with this repository (upstream carries no licence "
            "file); fetch it at its pinned commit with\n"
            "    python vendor/fetch_pawl.py\n"
            "See vendor/PROVENANCE.md. Only the baselines need it -- PAWC itself "
            "does not."
        ) from exc

    xu = np.ascontiguousarray(np.asarray(xu, dtype=float).ravel())
    yu = np.ascontiguousarray(np.asarray(yu, dtype=float).ravel())
    if K is None:
        K = min(xu.size, yu.size)
    if K == 0:
        return np.zeros(1), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    idx_x, idx_y, marginals = partial_ot_1d(xu, yu, max_iter=K, p=1)
    marginals = np.asarray(marginals, dtype=float)
    if marginals.size != K:
        raise RuntimeError(
            f"vendored PAWL returned {marginals.size} marginal costs, expected K={K}"
        )
    costs = np.concatenate([[0.0], np.cumsum(marginals)])
    return costs, np.asarray(idx_x, dtype=np.int64), np.asarray(idx_y, dtype=np.int64)


def line_profile_dp(xu, yu, K: int | None = None, *, return_matchings: bool = False):
    """Exact line partial-OT profile by dynamic programming over non-crossing matchings.

    For the ground cost ``|u - v|`` on the line, crossing pairs can always be
    uncrossed without increasing the cost (Monge condition), so restricting to
    non-crossing matchings is lossless. With
    ``f[i][j][k] = `` optimal cost using the first ``i`` sorted sources, first ``j``
    sorted targets and ``k`` matched pairs,

        f[i][j][k] = min( f[i-1][j][k], f[i][j-1][k],
                          f[i-1][j-1][k-1] + |x_(i) - y_(j)| ).

    Arithmetic is whatever the inputs carry, so ``fractions.Fraction`` coordinates
    give exact results.

    Returns
    -------
    costs : list, length ``K+1``
        Unweighted optimal cost per cardinality.
    matchings : list of list of (int, int), optional
        ``matchings[k]`` holds ``k`` pairs of indices into the *original* ``xu`` /
        ``yu`` sequences. Only returned when ``return_matchings`` is true.
    """
    xu = list(xu)
    yu = list(yu)
    n, m = len(xu), len(yu)
    if K is None:
        K = min(n, m)
    K = min(K, n, m)

    ord_x = sorted(range(n), key=lambda i: xu[i])
    ord_y = sorted(range(m), key=lambda j: yu[j])
    xs = [xu[i] for i in ord_x]
    ys = [yu[j] for j in ord_y]

    zero = (xs[0] - xs[0]) if n else (ys[0] - ys[0]) if m else 0.0
    inf = None  # sentinel for "infeasible"

    def better(a, b):
        if a is inf:
            return b
        if b is inf:
            return a
        return a if a <= b else b

    # f[i][j][k]
    f = [[[inf] * (K + 1) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            f[i][j][0] = zero
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            fi, fim = f[i], f[i - 1]
            d = xs[i - 1] - ys[j - 1]
            d = d if d >= zero else -d
            upper = min(K, i, j)
            for k in range(1, upper + 1):
                best = better(fim[j][k], fi[j - 1][k])
                prev = fim[j - 1][k - 1]
                if prev is not inf:
                    best = better(best, prev + d)
                fi[j][k] = best

    costs = [f[n][m][k] for k in range(K + 1)]
    if not return_matchings:
        return costs

    matchings = []
    for k in range(K + 1):
        pairs = []
        i, j, kk = n, m, k
        while kk > 0:
            # Reproduce the argmin of the recursion, preferring the match branch
            # only when it attains the optimum.
            cur = f[i][j][kk]
            if i > 0 and f[i - 1][j][kk] is not inf and f[i - 1][j][kk] == cur:
                i -= 1
                continue
            if j > 0 and f[i][j - 1][kk] is not inf and f[i][j - 1][kk] == cur:
                j -= 1
                continue
            d = xs[i - 1] - ys[j - 1]
            d = d if d >= zero else -d
            pairs.append((ord_x[i - 1], ord_y[j - 1]))
            i -= 1
            j -= 1
            kk -= 1
        matchings.append(list(reversed(pairs)))
    return costs, matchings


def line_profile_at_cut(x, y, theta, L: float = 1.0, w: float = 1.0, K: int | None = None):
    """Unweighted-times-``w`` line profile obtained by cutting the circle at ``theta``.

    Draft Prop. 3.1. Used both by Baseline B and by the "no cut beats the gap cuts"
    sufficiency check.
    """
    xu = unroll_at(x, theta, L)
    yu = unroll_at(y, theta, L)
    costs, order_x, order_y = line_profile_pawl(xu, yu, K)
    return w * costs, order_x, order_y


def partial_ot_circle_cuts(
    x, y, L: float = 1.0, w: float = 1.0, *, return_all: bool = False
) -> PartialCircleSolution:
    """Baseline B: exact circular profile by enumerating the ``N`` original gap cuts.

    Implements draft Corollary 3.1. One vendored-PAWL call per gap.

    Parameters
    ----------
    x, y : array_like
        Source and target coordinates in ``[0, L)``; supports must be disjoint.
    L, w : float
        Circumference and common atom weight.
    return_all : bool
        Also attach the full ``(N, K+1)`` table of per-cut line costs as
        ``solution.all_cut_costs`` and the cut midpoints as ``solution.cut_thetas``.
        The envelope table is what the cut-sufficiency gate inspects.

    Returns
    -------
    PartialCircleSolution
        ``costs[k] = min_r C_{k,r}^line`` and ``cuts[k]`` the argmin cut location.
        Active sets come from the argmin cut's PAWL run, so they need *not* be
        nested across ``k`` (draft Remark 6.1) — Baseline B is a cost oracle.
    """
    su = sorted_union(x, y, L)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    K = su.K

    thetas = gap_midpoints(su.z, L)
    n_cuts = thetas.size

    if K == 0:
        return PartialCircleSolution(
            x=x, y=y, L=L, w=w,
            costs=np.zeros(1),
            active_x=[np.zeros(0, dtype=np.int64)],
            active_y=[np.zeros(0, dtype=np.int64)],
            cuts=np.array([thetas[0] if n_cuts else 0.0]),
            solver="baseline_cuts",
        )

    table = np.empty((n_cuts, K + 1))
    orders_x = []
    orders_y = []
    for r, theta in enumerate(thetas):
        costs_r, ox, oy = line_profile_at_cut(x, y, theta, L, w, K)
        table[r] = costs_r
        orders_x.append(ox)
        orders_y.append(oy)

    argmin = np.argmin(table, axis=0)
    costs = table[argmin, np.arange(K + 1)]
    costs[0] = 0.0

    active_x = [orders_x[argmin[k]][:k].copy() for k in range(K + 1)]
    active_y = [orders_y[argmin[k]][:k].copy() for k in range(K + 1)]
    cuts = thetas[argmin]

    sol = PartialCircleSolution(
        x=x, y=y, L=L, w=w,
        costs=costs,
        active_x=active_x,
        active_y=active_y,
        cuts=cuts,
        solver="baseline_cuts",
    )
    if return_all:
        sol.all_cut_costs = table
        sol.cut_thetas = thetas
        # Per-cut activation orders, not just the argmin cut's. Needed by any
        # analysis that asks what active set a *given* cut would have produced
        # (experiments/a1_cut_necessity.py, metric 4); the loop above already
        # computes them and previously threw all but one away.
        sol.all_cut_orders_x = orders_x
        sol.all_cut_orders_y = orders_y
    return sol


# --------------------------------------------------------------------------- #
# Exact-arithmetic variant (fractions.Fraction), CLAUDE.md float-comparison rule
# --------------------------------------------------------------------------- #


def partial_ot_circle_cuts_exact(x, y, L=Fraction(1), w=Fraction(1)):
    """Baseline B in exact rational arithmetic.

    ``x``, ``y``, ``L``, ``w`` are ``fractions.Fraction`` (or int) sequences/values.
    Cuts are placed at exact gap midpoints and each line profile is computed by
    :func:`line_profile_dp`, so the returned costs are exact rationals. Used to
    decide instances where the float comparison against Baseline A sits near the
    ``1e-9`` tolerance.

    Returns
    -------
    costs : list of Fraction, length ``K+1``
    cuts : list of Fraction, length ``K+1``
        Argmin cut per ``k``.
    """
    x = [Fraction(v) for v in x]
    y = [Fraction(v) for v in y]
    L = Fraction(L)
    w = Fraction(w)

    z = sorted(x + y)
    if len(set(z)) != len(z):
        raise ValueError("Assumption 2.1 violated: support points must be pairwise distinct")
    K = min(len(x), len(y))

    # Exact gap midpoints: theta_i = z_i + l_i / 2, reduced mod L.
    thetas = []
    for i, zi in enumerate(z):
        nxt = z[i + 1] if i + 1 < len(z) else z[0] + L
        thetas.append((zi + (nxt - zi) / 2) % L)

    best = [None] * (K + 1)
    best_cut = [None] * (K + 1)
    for theta in thetas:
        xu = [(v - theta) % L for v in x]
        yu = [(v - theta) % L for v in y]
        profile = line_profile_dp(xu, yu, K)
        for k in range(K + 1):
            c = w * profile[k]
            if best[k] is None or c < best[k]:
                best[k] = c
                best_cut[k] = theta
    return best, best_cut


def all_cut_costs_exact(x, y, L=Fraction(1), w=Fraction(1), *, return_matchings=False):
    """Full exact ``(N, K+1)`` per-cut cost table; used by the sufficiency gate.

    Rows are indexed by gap: row ``r`` is the profile obtained by cutting inside
    ``G_r = (z_r, z_{r+1})`` clockwise, matching the row order of
    :func:`partial_ot_circle_cuts`'s ``all_cut_costs``.

    With ``return_matchings`` the per-cut optimal matchings are returned as well,
    as ``matchings[r][k]`` — a list of ``k`` index pairs into ``x`` and ``y``.
    """
    x = [Fraction(v) for v in x]
    y = [Fraction(v) for v in y]
    L, w = Fraction(L), Fraction(w)
    z = sorted(x + y)
    K = min(len(x), len(y))
    thetas = []
    for i, zi in enumerate(z):
        nxt = z[i + 1] if i + 1 < len(z) else z[0] + L
        thetas.append((zi + (nxt - zi) / 2) % L)
    table = []
    matchings = []
    for theta in thetas:
        xu = [(v - theta) % L for v in x]
        yu = [(v - theta) % L for v in y]
        if return_matchings:
            costs, match = line_profile_dp(xu, yu, K, return_matchings=True)
            matchings.append(match)
        else:
            costs = line_profile_dp(xu, yu, K)
        table.append([w * c for c in costs])
    if return_matchings:
        return table, thetas, matchings
    return table, thetas
