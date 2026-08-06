"""Baseline A — the exact LP / min-cost-flow oracle (ground truth, small n).

This is the only solver whose correctness we take on faith (plan, Phase 1). For
each cardinality ``k`` it solves the cardinality-constrained partial transport
problem of draft eq. (5) directly,

    min_{gamma >= 0}  sum_ij d_{S^1}(x_i, y_j) gamma_ij
    s.t.  sum_j gamma_ij <= 1,   sum_i gamma_ij <= 1,   sum_ij gamma_ij = k,

with the full circular cost matrix and **no** use of the circle-cut structure. The
feasible set is the cardinality-``k`` bipartite matching polytope ``P_k`` of draft
§2, whose extreme points are integral; a simplex (basic) solution is therefore a
matching, which the implementation asserts.

Cost is reported as ``C_k^o = w * sum_{(i,j) in M} d_{S^1}(x_i, y_j)``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from .circle import geodesic_distance_matrix
from .solution import PartialCircleSolution

__all__ = ["partial_ot_circle_lp", "lp_cost_for_k"]

#: Tolerance at which an LP vertex is accepted as integral.
_INTEGRALITY_ATOL = 1e-7


def _marginal_constraints(n: int, m: int):
    """Sparse ``A_ub`` enforcing ``sum_j gamma_ij <= 1`` and ``sum_i gamma_ij <= 1``.

    Variables are ``gamma`` flattened in C order, so variable ``i * m + j`` is
    ``gamma_ij``.
    """
    nm = n * m
    cols = np.arange(nm)
    rows_src = np.repeat(np.arange(n), m)  # row i owns variables i*m .. i*m+m-1
    rows_tgt = np.tile(np.arange(m), n)  # column j owns variables j, m+j, 2m+j, ...
    a_src = coo_matrix((np.ones(nm), (rows_src, cols)), shape=(n, nm))
    a_tgt = coo_matrix((np.ones(nm), (rows_tgt, cols)), shape=(m, nm))
    return vstack([a_src, a_tgt]).tocsr()


def lp_cost_for_k(D: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    """Solve draft eq. (5) for one ``k`` given the cost matrix ``D``.

    Returns ``(unweighted_cost, gamma)`` where ``gamma`` is the ``(n, m)`` 0/1
    matching matrix.

    Objective scaling
    -----------------
    HiGHS applies its dual-feasibility (reduced-cost) tolerance in *absolute*
    terms, so an instance whose whole cost matrix is O(1e-4) — every support point
    packed into a tiny arc — can terminate at a basis whose reduced costs are
    negative but smaller in magnitude than the tolerance, i.e. at a strictly
    suboptimal vertex. The objective of an LP may be rescaled freely, so ``D`` is
    normalised to unit maximum before the solve and the optimal value is scaled
    back afterwards. See the ``tiny_arc`` regression case in
    ``tests/test_baselines_agree.py``.
    """
    n, m = D.shape
    if k == 0:
        return 0.0, np.zeros((n, m))

    scale = float(np.max(D))
    if not np.isfinite(scale) or scale <= 0.0:
        # Distinct supports (Assumption 2.1) make this unreachable, but a
        # degenerate all-zero cost matrix has every matching optimal.
        gamma = np.zeros((n, m))
        gamma[np.arange(k), np.arange(k)] = 1.0
        return 0.0, gamma
    D_scaled = D / scale

    a_ub = _marginal_constraints(n, m)
    b_ub = np.ones(n + m)
    a_eq = coo_matrix((np.ones(n * m), (np.zeros(n * m, dtype=int), np.arange(n * m))),
                      shape=(1, n * m)).tocsr()
    b_eq = np.array([float(k)])

    res = linprog(
        c=D_scaled.ravel(order="C"),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        # Dual simplex returns a basic solution, hence a vertex of P_k, hence
        # integral. IPM without crossover could return a fractional interior point.
        method="highs-ds",
        # 1e-10 is HiGHS's smallest accepted value for these tolerances.
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if not res.success:
        raise RuntimeError(f"LP oracle failed for k={k}: {res.message}")

    gamma = np.asarray(res.x, dtype=float).reshape(n, m)
    off = np.abs(gamma - np.round(gamma))
    if np.max(off) > _INTEGRALITY_ATOL:
        raise RuntimeError(
            f"LP oracle returned a fractional vertex for k={k} "
            f"(max deviation {np.max(off):.3e}); P_k should be integral (draft §2)."
        )
    gamma = np.round(gamma)
    # Report the value on the *original* cost scale, summed from the matching
    # itself rather than taken from res.fun (which is the scaled objective).
    return float(np.sum(D * gamma)), gamma


def partial_ot_circle_lp(x, y, L: float = 1.0, w: float = 1.0, ks=None) -> PartialCircleSolution:
    """Baseline A: exact partial 1-Wasserstein profile on ``S^1_L`` by LP.

    Parameters
    ----------
    x, y : array_like
        Source and target coordinates in ``[0, L)``; supports must be disjoint.
    L : float
        Circumference.
    w : float
        Common atom weight (draft eq. (2)).
    ks : iterable of int, optional
        Restrict the solve to these cardinalities. Costs at the omitted ``k`` are
        filled with ``nan`` and their active sets left empty. Defaults to all
        ``k = 0, ..., K``.

    Returns
    -------
    PartialCircleSolution
        With explicit plans stored for every solved ``k``.

    Notes
    -----
    Complexity is one LP per ``k``; intended for ``n, m <~ 60`` (plan, Phase 1).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n, m = x.size, y.size
    K = min(n, m)
    wanted = set(range(K + 1)) if ks is None else {int(k) for k in ks}

    D = geodesic_distance_matrix(x, y, L)

    costs = np.full(K + 1, np.nan)
    active_x: list = [np.zeros(0, dtype=np.int64)] * (K + 1)
    active_y: list = [np.zeros(0, dtype=np.int64)] * (K + 1)
    plans: dict = {}

    for k in sorted(wanted):
        raw, gamma = lp_cost_for_k(D, k)
        costs[k] = w * raw
        ix, iy = np.nonzero(gamma)
        active_x[k] = np.asarray(ix, dtype=np.int64)
        active_y[k] = np.asarray(iy, dtype=np.int64)
        plans[k] = w * gamma

    return PartialCircleSolution(
        x=x,
        y=y,
        L=L,
        w=w,
        costs=costs,
        active_x=active_x,
        active_y=active_y,
        cuts=None,
        plans=plans,
        solver="baseline_lp",
    )
