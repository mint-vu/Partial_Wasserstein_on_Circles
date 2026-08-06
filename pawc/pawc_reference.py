"""PAWC — readable reference implementation, draft Algorithm 1.

This module mirrors the proof structure of `docs/pawc_draft.pdf` rather than
chasing speed: the data structures are explicit, every step is annotated with the
result that licenses it, and with ``PAWC_DEBUG=1`` every induction step is checked
against a from-scratch recomputation (CLAUDE.md, "Invariants to assert").

Structure of one run
--------------------
1. **Sorted union support** ``z`` and labels ``sigma`` (draft §3.2, eq. (23)).
2. **Doubled cyclic sequence** and the tables ``R, S, p, Q`` (draft §7.1,
   eqs. (24)-(30)), so that the cost of any balanced doubled interval is
   ``c_R([a, b]) = Q_b - Q_{a-1}`` (Prop. 7.1) and any candidate-cell marginal is
   four table lookups (Cor. 7.1, eq. (32)).
3. **Current cells**: a circular doubly linked list over the inactive atoms. Each
   directed cell ``[u, v]`` (draft eq. (16)) carries a representative **free**
   original gap (Lemma 6.1).
4. **Greedy**: repeatedly activate the endpoints of the current *candidate* cell
   (opposite-type endpoints, Def. 5.1) of minimum marginal ``m(C)`` (Def. 5.2).
   Theorem 6.1 says this increment is exactly ``C_{k+1}^o - C_k^o``, and
   Corollary 6.1 that the resulting active sets are optimal for every ``k``.
5. **Simultaneous cut**: the free gap that no selected cell ever contained is
   optimal for *every* cardinality (Theorem 6.2), and is what plan recovery uses
   (draft §9.1).

Terminology is the draft's throughout: *chain*, *minimal chain*, *cell*,
*candidate cell*, *free gap*, *cut*.
"""

from __future__ import annotations

import heapq
import os

import numpy as np

from .circle import SOURCE, gap_midpoints, sorted_union
from .solution import PartialCircleSolution
from .verify import circular_w1_of_active_set, free_gaps_of_cells, line_w1_sorted

__all__ = ["partial_ot_circle", "DoubledTables", "build_doubled_tables", "debug_enabled"]


def debug_enabled() -> bool:
    """True when ``PAWC_DEBUG`` is set to something other than ``0``/empty."""
    return os.environ.get("PAWC_DEBUG", "") not in ("", "0", "false", "False")


# --------------------------------------------------------------------------- #
# §7.1 — doubled cyclic sequence and constant-time chain machinery
# --------------------------------------------------------------------------- #


class DoubledTables:
    """The tables ``R``, ``S``, ``p``, ``Q`` of draft §7.1 on the doubled sequence.

    Indexing follows the draft exactly: the doubled sequence is indexed ``1..2N``
    and the tables are indexed ``0..2N``, with ``R_0 = S_0 = Q_0 = 0``. Circle atom
    ``i`` (0-based) occupies doubled positions ``i + 1`` and ``i + 1 + N``.

    Attributes
    ----------
    R : (2N+1,) int array
        Prefix differential ranks, eq. (25). ``[a, b]`` is balanced iff
        ``R[b] == R[a-1]`` (eq. (27)).
    S : (2N+1,) float array
        Signed coordinate prefix sums, eq. (26).
    p : (2N+1,) int array
        ``p_t = max{s < t : R_s = R_t}``, eq. (28), or ``-1`` when no such ``s``
        exists. ``[p_t + 1, t]`` is then the minimal balanced interval ending at
        ``t``.
    Q : (2N+1,) float array
        Maximal-chain prefix costs, eq. (30). Already scaled by the atom weight
        ``w``, so ``c_R([a, b]) = Q[b] - Q[a-1]`` carries ``w`` (Prop. 7.1).
    """

    __slots__ = ("R", "S", "p", "Q", "N", "w", "L")

    def __init__(self, R, S, p, Q, N, w, L):
        self.R, self.S, self.p, self.Q = R, S, p, Q
        self.N, self.w, self.L = N, w, L

    # -- draft eq. (27) ----------------------------------------------------- #
    def is_balanced(self, a: int, b: int) -> bool:
        return bool(self.R[b] == self.R[a - 1])

    # -- draft Prop. 7.1, eq. (31) ------------------------------------------ #
    def chain_cost(self, a: int, b: int) -> float:
        """``c_R([a, b])`` for a balanced doubled interval; ``0`` if ``b < a``."""
        if b < a:
            return 0.0
        return float(self.Q[b] - self.Q[a - 1])

    # -- draft Cor. 7.1, eq. (32) ------------------------------------------- #
    def cell_marginal(self, a: int, b: int) -> float:
        """``m([a, b]) = (Q_b - Q_{a-1}) - (Q_{b-1} - Q_a)``, four table lookups.

        The second term is the cost of the interior ``[a+1, b-1]``, read as
        ``Q_{b-1} - Q_{(a+1)-1}``, and is zero when the interior is empty.
        """
        return float((self.Q[b] - self.Q[a - 1]) - (self.Q[b - 1] - self.Q[a]))

    def doubled_interval(self, u: int, v: int) -> tuple[int, int]:
        """Doubled-sequence interval ``[a, b]`` of the clockwise cell ``[u, v]``.

        Draft §7.1: "every clockwise circular cell can be represented by a standard
        interval ``[a, b] ⊂ {1, ..., 2N}`` with ``b - a < N``".
        """
        a = u + 1
        b = v + 1 if v > u else v + 1 + self.N
        return a, b


def build_doubled_tables(z: np.ndarray, label: np.ndarray, L: float, w: float) -> DoubledTables:
    """Compute ``R``, ``S``, ``p``, ``Q`` on the doubled sequence, draft eqs. (24)-(30).

    ``O(N)`` after sorting: ``p_t`` is found by remembering the most recent index
    at which each differential-rank value occurred.
    """
    z = np.asarray(z, dtype=float)
    N = z.size
    # eq. (23): sigma_i = +1 for a source atom, -1 for a target atom.
    sigma = np.where(np.asarray(label) == SOURCE, 1, -1).astype(np.int64)
    # eq. (24): the doubled sequence, 1-based positions 1..2N.
    z_d = np.concatenate([[0.0], z, z + L])
    s_d = np.concatenate([[0], sigma, sigma])

    R = np.cumsum(s_d)  # eq. (25); R[0] = 0 since s_d[0] = 0
    S = np.cumsum(s_d * z_d)  # eq. (26)

    # eq. (28): p_t = max{s < t : R_s = R_t}, or -1.
    p = np.full(2 * N + 1, -1, dtype=np.int64)
    last_seen: dict[int, int] = {}
    for t in range(2 * N + 1):
        rt = int(R[t])
        if rt in last_seen:
            p[t] = last_seen[rt]
        last_seen[rt] = t

    # eq. (30): Q_t = Q_{p_t} + w |S_t - S_{p_t}| when p_t exists, else 0.
    Q = np.zeros(2 * N + 1)
    for t in range(1, 2 * N + 1):
        pt = int(p[t])
        Q[t] = 0.0 if pt < 0 else Q[pt] + w * abs(S[t] - S[pt])

    return DoubledTables(R=R, S=S, p=p, Q=Q, N=N, w=w, L=L)


# --------------------------------------------------------------------------- #
# The algorithm
# --------------------------------------------------------------------------- #


class _State:
    """Current cells as a circular doubly linked list over inactive atoms.

    ``free_rep[u]`` is the representative *free* original gap of the directed cell
    whose left endpoint is the inactive atom ``u`` (draft §6.1 and Algorithm 1
    lines 4, 19). Keying by the left endpoint is enough because the linked list
    determines the right endpoint.
    """

    def __init__(self, N: int):
        self.N = N
        self.succ = np.roll(np.arange(N), -1)
        self.pred = np.roll(np.arange(N), 1)
        self.inactive = np.ones(N, dtype=bool)
        self.n_inactive = N
        # Algorithm 1 line 4: initially each cell is exactly one original gap,
        # namely G_u = (z_u, z_{u+1}), which is free.
        self.free_rep = {u: u for u in range(N)}
        self.used_gaps: set[int] = set()

    def cell_gaps(self, u: int, v: int) -> list[int]:
        """Original gap indices contained in the clockwise cell ``[u, v]``."""
        out = []
        i = u
        while i != v:
            out.append(i)
            i = (i + 1) % self.N
        return out


def partial_ot_circle(
    x,
    y,
    L: float = 1.0,
    w: float = 1.0,
    *,
    debug: bool | None = None,
) -> PartialCircleSolution:
    """Exact partial 1-Wasserstein profile on ``S^1_L`` — reference PAWC.

    Implements draft Algorithm 1. Returns ``C_k^o`` and an optimal, **nested**
    active set for every ``k = 0, ..., K``, together with the simultaneous cut
    ``theta*`` of Theorem 6.2 (recorded identically in every entry of
    ``solution.cuts``).

    Parameters
    ----------
    x, y : array_like
        Source and target coordinates in ``[0, L)``; supports must be pairwise
        distinct (Assumption 2.1).
    L, w : float
        Circumference and common atom weight.
    debug : bool, optional
        Force the invariant checks on or off. Defaults to the ``PAWC_DEBUG``
        environment variable.

    Notes
    -----
    Transparency over speed, as the plan's Phase 2 asks: the free-gap bookkeeping
    and the cell walk are explicit, which makes the worst case ``O(N^2)`` here.
    ``pawc.pawc`` is the ``O(N log N)`` form.
    """
    if debug is None:
        debug = debug_enabled()

    su = sorted_union(x, y, L)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    N, K = su.N, su.K

    tables = build_doubled_tables(su.z, su.label, L, w)
    state = _State(N)

    costs = np.zeros(K + 1)
    activation_rank = np.zeros(N, dtype=np.int64)  # 0 = never activated
    active: set[int] = set()
    theta_star_gap: int | None = None
    selected_cells: list[tuple[int, int]] = []

    def marginal(u: int, v: int) -> float:
        a, b = tables.doubled_interval(u, v)
        return tables.cell_marginal(a, b)

    # Algorithm 1 line 5: every opposite-endpoint initial cell enters the heap.
    heap: list[tuple[float, int, int]] = []
    if K > 0:
        for u in range(N):
            v = int(state.succ[u])
            if su.label[u] != su.label[v]:
                heapq.heappush(heap, (marginal(u, v), u, v))

    if debug:
        _check_state(su, tables, state, active, w, k=0, cost=0.0)

    # Algorithm 1 lines 7-23.
    for k in range(K):
        u = v = -1
        m_uv = 0.0
        # Lazy deletion: an entry is valid only if both endpoints are still
        # inactive, v is still the clockwise successor of u, and the endpoints
        # have opposite types (draft §8).
        while heap:
            m_uv, cu, cv = heapq.heappop(heap)
            if (
                state.inactive[cu]
                and state.inactive[cv]
                and int(state.succ[cu]) == cv
                and su.label[cu] != su.label[cv]
            ):
                u, v = cu, cv
                break
        if u < 0:
            raise RuntimeError(
                f"no valid candidate cell at k={k} but K={K}: Theorem 6.1 guarantees one "
                "exists while k < K"
            )

        if debug:
            _check_marginal_against_scratch(su, tables, u, v, w, L)

        # Algorithm 1 lines 11-12.
        activation_rank[u] = activation_rank[v] = k + 1
        costs[k + 1] = costs[k] + m_uv
        active.add(u)
        active.add(v)
        selected_cells.append((u, v))
        # Draft §6: the gaps of a selected cell become *used*.
        state.used_gaps.update(state.cell_gaps(u, v))

        p_idx = int(state.pred[u])
        q_idx = int(state.succ[v])

        if state.n_inactive == 2:
            # Algorithm 1 lines 14-16: the complementary directed cell [v, u]
            # supplies the simultaneous cut.
            theta_star_gap = state.free_rep[v]
            state.inactive[u] = state.inactive[v] = False
            state.n_inactive = 0
            state.free_rep.pop(u, None)
            state.free_rep.pop(v, None)
            if debug:
                _check_state(su, tables, state, active, w, k=k + 1, cost=costs[k + 1])
            continue

        # Algorithm 1 line 18: remove u, v and relink.
        state.inactive[u] = state.inactive[v] = False
        state.n_inactive -= 2
        state.succ[p_idx] = q_idx
        state.pred[q_idx] = p_idx
        # Algorithm 1 line 19: the merged cell [p, q] inherits a free-gap
        # representative from an unselected neighbour (Lemma 6.1). free_rep[p] is
        # already the representative of [p, u], whose gaps were not in the selected
        # middle cell, so it remains free and is simply kept.
        state.free_rep.pop(u, None)
        state.free_rep.pop(v, None)

        # Algorithm 1 lines 20-22: at most one new candidate per iteration.
        if p_idx != q_idx and su.label[p_idx] != su.label[q_idx]:
            heapq.heappush(heap, (marginal(p_idx, q_idx), p_idx, q_idx))

        if debug:
            _check_state(su, tables, state, active, w, k=k + 1, cost=costs[k + 1])

    # Algorithm 1 lines 24-26.
    if theta_star_gap is None:
        if state.free_rep:
            theta_star_gap = next(iter(state.free_rep.values()))
        else:  # K == 0 with an empty measure
            theta_star_gap = 0
    if theta_star_gap in state.used_gaps:
        raise AssertionError(
            f"simultaneous cut gap {theta_star_gap} was used by a selected cell — "
            "Theorem 6.2 violated"
        )

    theta_star = float(gap_midpoints(su.z, L)[theta_star_gap]) if N else 0.0

    active_x, active_y = _active_sets_by_rank(su, activation_rank, K)

    sol = PartialCircleSolution(
        x=x,
        y=y,
        L=L,
        w=w,
        costs=costs,
        active_x=active_x,
        active_y=active_y,
        cuts=np.full(K + 1, theta_star),
        solver="pawc_reference",
    )
    sol.theta_star = theta_star
    sol.theta_star_gap = int(theta_star_gap)
    sol.activation_rank = activation_rank
    sol.selected_cells = selected_cells
    sol.sorted_union = su
    return sol


def _active_sets_by_rank(su, activation_rank: np.ndarray, K: int):
    """Turn activation ranks into nested per-``k`` index lists in ``x`` / ``y``."""
    active_x = [[] for _ in range(K + 1)]
    active_y = [[] for _ in range(K + 1)]
    order_x: list[int] = []
    order_y: list[int] = []
    for k in range(1, K + 1):
        for i in np.flatnonzero(activation_rank == k):
            i = int(i)
            if su.label[i] == SOURCE:
                order_x.append(su.original_index(i))
            else:
                order_y.append(su.original_index(i))
        active_x[k] = np.array(order_x, dtype=np.int64)
        active_y[k] = np.array(order_y, dtype=np.int64)
    active_x[0] = np.zeros(0, dtype=np.int64)
    active_y[0] = np.zeros(0, dtype=np.int64)
    for k in range(1, K + 1):
        active_x[k] = np.asarray(active_x[k], dtype=np.int64)
        active_y[k] = np.asarray(active_y[k], dtype=np.int64)
    return active_x, active_y


# --------------------------------------------------------------------------- #
# PAWC_DEBUG=1 invariant checks (CLAUDE.md, "Invariants to assert")
# --------------------------------------------------------------------------- #

_DEBUG_ATOL = 1e-9


def _check_marginal_against_scratch(su, tables, u, v, w, L):
    """``m(C) = c_R(C) - c_R(C°)`` recomputed by explicit sorted matching (Def. 5.2).

    Cross-checks the four-lookup formula of Cor. 7.1 against the definition, using
    clockwise arc coordinates inside the cell — *not* the geodesic distance, as the
    draft insists just after eq. (17).
    """
    a, b = tables.doubled_interval(u, v)
    if not tables.is_balanced(a, b):
        raise AssertionError(f"candidate cell [{u},{v}] -> [{a},{b}] is not balanced")
    if b - a >= tables.N:
        raise AssertionError(f"cell [{u},{v}] -> [{a},{b}] violates b - a < N")

    # Clockwise coordinates along the cell, measured from z_u.
    idx = []
    i = u
    while True:
        idx.append(i)
        if i == v:
            break
        i = (i + 1) % su.N
    coord = {}
    acc = 0.0
    for j, i in enumerate(idx):
        if j == 0:
            coord[i] = 0.0
        else:
            prev = idx[j - 1]
            acc += float(np.mod(su.z[i] - su.z[prev], L))
            coord[i] = acc

    closed_src = [coord[i] for i in idx if su.label[i] == SOURCE]
    closed_tgt = [coord[i] for i in idx if su.label[i] != SOURCE]
    interior = idx[1:-1]
    int_src = [coord[i] for i in interior if su.label[i] == SOURCE]
    int_tgt = [coord[i] for i in interior if su.label[i] != SOURCE]

    m_scratch = line_w1_sorted(closed_src, closed_tgt, w) - line_w1_sorted(int_src, int_tgt, w)
    m_table = tables.cell_marginal(a, b)
    if abs(m_scratch - m_table) > _DEBUG_ATOL * max(1.0, abs(m_scratch)):
        raise AssertionError(
            f"Cor. 7.1 marginal {m_table!r} disagrees with the Def. 5.2 recomputation "
            f"{m_scratch!r} for cell [{u},{v}]"
        )


def _check_state(su, tables, state, active, w, k, cost):
    """Invariants after induction step ``k`` (CLAUDE.md, "Invariants to assert")."""
    # -- cardinality ------------------------------------------------------- #
    if len(active) != 2 * k:
        raise AssertionError(f"active set has {len(active)} atoms at k={k}, expected {2 * k}")
    n_src = sum(1 for i in active if su.label[i] == SOURCE)
    if n_src != k:
        raise AssertionError(f"active set has {n_src} sources at k={k}, expected {k}")

    # -- cost consistency: eq. (8), recomputed from the circulation form ---- #
    scratch = circular_w1_of_active_set(su, active, w)
    if abs(scratch - cost) > _DEBUG_ATOL * max(1.0, abs(scratch)):
        raise AssertionError(
            f"incremental cost {cost!r} at k={k} disagrees with the from-scratch "
            f"balanced circular W1 of A_k, {scratch!r} (draft eq. (8))"
        )

    # -- Lemma 5.1: every current cell has a balanced interior -------------- #
    inactive = [i for i in range(state.N) if state.inactive[i]]
    for u in inactive:
        v = int(state.succ[u])
        if v == u:
            continue
        a, b = tables.doubled_interval(u, v)
        interior = list(range(a + 1, b))
        bal = sum(1 if tables.R[t] - tables.R[t - 1] > 0 else -1 for t in interior)
        if bal != 0:
            raise AssertionError(
                f"Lemma 5.1 violated at k={k}: interior of cell [{u},{v}] is unbalanced"
            )

    # -- Lemma 6.1: free-gap invariant -------------------------------------- #
    if state.n_inactive > 0:
        recomputed = free_gaps_of_cells(su, active, state.used_gaps)
        for u in inactive:
            free_here = recomputed.get(u, [])
            if not free_here:
                raise AssertionError(
                    f"Lemma 6.1 violated at k={k}: current cell with left endpoint {u} "
                    "contains no free original gap"
                )
            rep = state.free_rep.get(u)
            if rep is None:
                raise AssertionError(f"cell {u} has no recorded free-gap representative")
            if rep in state.used_gaps:
                raise AssertionError(
                    f"recorded free-gap representative {rep} of cell {u} is used at k={k}"
                )
            if rep not in free_here:
                raise AssertionError(
                    f"recorded representative {rep} of cell {u} is not one of the cell's "
                    f"free gaps {free_here} at k={k}"
                )
