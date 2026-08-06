"""PAWC — numba-JIT variant of draft Algorithm 1.

Why this module exists
----------------------
``CLAUDE.md`` mandates that ``pawc.py`` stay pure numpy + stdlib so that timing
against the vendored PAWL is fair, and forbids applying numba "to one side only".
That rule assumed the vendored PAWL was itself pure numpy. It is not:
``vendor/partial.py`` is ``@njit(cache=True, fastmath=True)`` throughout (see
``vendor/PROVENANCE.md``), so the pure-numpy comparison actually handicaps PAWC on
the constant factor.

This module restores the symmetry the rule was after — *both* sides JIT-compiled —
without touching ``pawc.py``, which remains the pure numpy + stdlib reference for
the conservative comparison. Benchmarks report both.

Fidelity
--------
Compiled with ``fastmath=False`` **on purpose**. ``fastmath=True`` licenses
reassociation of floating-point sums, which would break bit-for-bit agreement with
``pawc_reference`` — the equivalence gate the plan requires in Phase 4.2. The
arithmetic here is therefore the same operations in the same order as
:mod:`pawc.pawc`:

* ``S`` accumulates left to right, exactly as ``np.cumsum`` does (a cumulative sum
  is sequential by construction — it emits every prefix — so there is no pairwise
  summation to match);
* ``Q`` follows the same eq. (30) recursion, for the accuracy reason recorded in
  :func:`pawc.pawc.prefix_tables`;
* the heap holds the same ``(m, u, v)`` tuples, which are totally ordered and
  distinct in ``u``, so the pop sequence is determined by the tuples themselves and
  cannot depend on the heap implementation.

The predecessor table ``p_t`` uses a direct array over the differential-rank range
``[-2N, 2N]`` rather than a hash table, which is the alternative the draft itself
offers at the end of §7.1.
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit

from .circle import SOURCE, gap_midpoints, sorted_union
from .solution import NestedActiveSets, PartialCircleSolution

__all__ = ["partial_ot_circle", "warmup", "pawc_kernel"]


@njit(cache=True, fastmath=False)
def pawc_kernel(z, sigma, L, w, K):  # pragma: no cover - compiled
    """Draft Algorithm 1 end to end, on the sorted union support.

    Parameters
    ----------
    z : (N,) float64
        Cyclically sorted union support, strictly increasing, in ``[0, L)``.
    sigma : (N,) int64
        ``+1`` for a source atom, ``-1`` for a target atom (draft eq. (23)).
    L, w : float64
    K : int64
        ``min(n, m)``.

    Returns
    -------
    costs : (K+1,) float64
    activation_rank : (N,) int64
        ``0`` for never activated, else the ``k`` at which the atom entered.
    theta_star_gap : int64
        Index of the simultaneous-cut gap (draft Thm. 6.2).
    """
    N = z.shape[0]
    T = 2 * N + 1

    # --- §7.1 eqs. (24)-(26): doubled sequence prefix tables ---------------- #
    R = np.empty(T, dtype=np.int64)
    S = np.empty(T, dtype=np.float64)
    R[0] = 0
    S[0] = 0.0
    for t in range(1, T):
        if t <= N:
            zt = z[t - 1]
            st = sigma[t - 1]
        else:
            zt = z[t - 1 - N] + L
            st = sigma[t - 1 - N]
        R[t] = R[t - 1] + st
        S[t] = S[t - 1] + st * zt

    # --- eq. (28): p_t = max{s < t : R_s = R_t}, via a direct rank array ---- #
    offset = 2 * N + 1
    last_seen = np.full(4 * N + 3, -1, dtype=np.int64)
    p = np.empty(T, dtype=np.int64)
    for t in range(T):
        r = R[t] + offset
        p[t] = last_seen[r]
        last_seen[r] = t

    # --- eq. (30): maximal-chain prefix costs ------------------------------- #
    Q = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        pt = p[t]
        if pt >= 0:
            Q[t] = Q[pt] + w * abs(S[t] - S[pt])

    costs = np.zeros(K + 1, dtype=np.float64)
    activation_rank = np.zeros(N, dtype=np.int64)
    if K == 0:
        return costs, activation_rank, np.int64(0)

    # --- Algorithm 1 lines 3-5 --------------------------------------------- #
    succ = np.empty(N, dtype=np.int64)
    pred = np.empty(N, dtype=np.int64)
    for i in range(N):
        succ[i] = i + 1 if i < N - 1 else 0
        pred[i] = i - 1 if i > 0 else N - 1
    inactive = np.ones(N, dtype=np.bool_)
    # line 4: cell [z_u, z_{u+1}] is exactly the original gap G_u, which is free.
    free_rep = np.arange(N).astype(np.int64)

    heap = [(0.0, np.int64(0), np.int64(0)) for _ in range(0)]
    for u in range(N):
        v = succ[u]
        if sigma[u] != sigma[v]:
            # empty interior, so eq. (32) collapses to Q[u+2] - Q[u]
            heap.append((Q[u + 2] - Q[u], np.int64(u), np.int64(v)))
    heapq.heapify(heap)

    n_inactive = N
    theta_star_gap = np.int64(-1)

    # --- Algorithm 1 lines 7-23 -------------------------------------------- #
    for k in range(K):
        u = np.int64(-1)
        v = np.int64(-1)
        m_uv = 0.0
        while len(heap) > 0:
            entry = heapq.heappop(heap)
            cu = entry[1]
            cv = entry[2]
            # §8 lazy deletion; opposite typing was fixed at insertion.
            if inactive[cu] and inactive[cv] and succ[cu] == cv:
                m_uv = entry[0]
                u = cu
                v = cv
                break
        if u < 0:
            raise RuntimeError(
                "no valid candidate cell while k < K; Theorem 6.1 guarantees one exists"
            )

        activation_rank[u] = k + 1
        activation_rank[v] = k + 1
        costs[k + 1] = costs[k] + m_uv

        p_idx = pred[u]
        q_idx = succ[v]

        if n_inactive == 2:
            # lines 14-16: the complementary directed cell [v, u] carries the cut.
            theta_star_gap = free_rep[v]
            inactive[u] = False
            inactive[v] = False
            n_inactive = 0
            continue

        # line 18
        inactive[u] = False
        inactive[v] = False
        n_inactive -= 2
        succ[p_idx] = q_idx
        pred[q_idx] = p_idx
        # line 19: free_rep[p_idx] already represents [p, u] and stays free
        # (Lemma 6.1), so inheritance is a no-op.

        # lines 20-22: at most one new candidate per iteration.
        if p_idx != q_idx and sigma[p_idx] != sigma[q_idx]:
            a = p_idx + 1
            if q_idx > p_idx:
                b = q_idx + 1
            else:
                b = q_idx + 1 + N
            heapq.heappush(
                heap, ((Q[b] - Q[a - 1]) - (Q[b - 1] - Q[a]), p_idx, q_idx)
            )

    # --- Algorithm 1 lines 24-26 ------------------------------------------- #
    if theta_star_gap < 0:
        for i in range(N):
            if inactive[i]:
                theta_star_gap = free_rep[i]
                break
    if theta_star_gap < 0:
        theta_star_gap = np.int64(0)

    return costs, activation_rank, theta_star_gap


def partial_ot_circle(x, y, L: float = 1.0, w: float = 1.0) -> PartialCircleSolution:
    """Exact partial 1-Wasserstein profile on ``S^1_L`` — numba-JIT PAWC.

    Same contract, same results (bit-for-bit) as :func:`pawc.pawc.partial_ot_circle`
    and :func:`pawc.pawc_reference.partial_ot_circle`; see the module docstring.

    The first call in a process pays JIT compilation. ``cache=True`` persists the
    compiled kernel to disk, and :func:`warmup` forces compilation up front so it is
    never charged to a timed measurement.
    """
    su = sorted_union(x, y, L)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    N, K = su.N, su.K

    if N == 0:
        empty = NestedActiveSets(np.zeros(0, dtype=np.int64), 0)
        sol = PartialCircleSolution(
            x=x, y=y, L=L, w=w, costs=np.zeros(1),
            active_x=empty, active_y=empty, cuts=np.zeros(1), solver="pawc_numba",
        )
        sol.theta_star = 0.0
        sol.theta_star_gap = 0
        sol.activation_rank = np.zeros(0, dtype=np.int64)
        sol.sorted_union = su
        return sol

    sigma = np.where(su.label == SOURCE, 1, -1).astype(np.int64)
    costs, activation_rank, theta_star_gap = pawc_kernel(
        np.ascontiguousarray(su.z), sigma, float(L), float(w), np.int64(K)
    )

    theta_star = float(gap_midpoints(su.z, L)[int(theta_star_gap)])
    order_x, order_y = _activation_order(su, activation_rank, K)

    sol = PartialCircleSolution(
        x=x, y=y, L=L, w=w,
        costs=costs,
        active_x=NestedActiveSets(order_x, K),
        active_y=NestedActiveSets(order_y, K),
        cuts=np.full(K + 1, theta_star),
        solver="pawc_numba",
    )
    sol.theta_star = theta_star
    sol.theta_star_gap = int(theta_star_gap)
    sol.activation_rank = activation_rank
    sol.sorted_union = su
    return sol


def _activation_order(su, activation_rank: np.ndarray, K: int):
    """Activation order as indices into the original ``x`` and ``y`` arrays."""
    activated = np.flatnonzero(activation_rank > 0)
    activated = activated[np.argsort(activation_rank[activated], kind="stable")]
    is_src = su.label[activated] == SOURCE
    own = su.index_in_own[activated]
    order_x = su.order_x[own[is_src]]
    order_y = su.order_y[own[~is_src]]
    if order_x.size != K or order_y.size != K:
        raise AssertionError(
            f"activation produced {order_x.size} sources and {order_y.size} targets, "
            f"expected {K} of each"
        )
    return order_x.astype(np.int64), order_y.astype(np.int64)


def warmup() -> None:
    """Force JIT compilation so it is never charged to a timed measurement."""
    partial_ot_circle(np.array([0.1, 0.4, 0.7]), np.array([0.2, 0.5, 0.85]))
