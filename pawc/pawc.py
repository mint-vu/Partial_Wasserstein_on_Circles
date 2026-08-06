"""PAWC — optimised implementation, draft Algorithm 1. O(N log N) time, O(N) memory.

Behaviourally identical to :mod:`pawc.pawc_reference` (CLAUDE.md, "Performance
rules": optimisations must be behaviour-preserving), with the reference's readable
bookkeeping replaced by:

* **Vectorised §7.1 preprocessing.** ``R`` and ``S`` are plain cumulative sums. The
  predecessor table ``p_t = max{s < t : R_s = R_t}`` (eq. (28)) is obtained without a
  Python loop from the observation that ``p`` chains together exactly the indices
  sharing a differential rank: inside the group ``{t : R_t = r}``, sorted
  increasingly, each element's predecessor is the previous element. One ``lexsort``
  therefore yields all of ``p``. The ``Q`` recursion (eq. (30)) is deliberately
  *not* vectorised — see the comment in :func:`prefix_tables` for the accuracy
  reason.
* **Vectorised initial candidate keys.** With ``a = u + 1`` and ``b = u + 2`` the
  interior of an initial cell is empty, so eq. (32) collapses to
  ``m_u = Q[u + 2] - Q[u]`` for every ``u`` at once — including the wrap-around cell
  ``[z_{N-1}, z_0]``, whose doubled interval is ``[N, N + 1]``. The heap is then
  built by a single ``heapify`` (``O(N)``) rather than ``N`` pushes.
* **Lazy deletion** in the heap and an ``O(1)`` circular doubly linked list, exactly
  as draft §8 prescribes; no candidate-list cleaning pass.
* **``O(1)`` free-gap inheritance.** The merged cell keeps its left neighbour's
  representative, which Lemma 6.1's proof shows is still free — so no gap set is
  ever enumerated.
* **Lazy nested active sets.** Only the activation order is stored; ``A_k`` is a
  view of its first ``k`` entries (:class:`~pawc.solution.NestedActiveSets`), which
  keeps memory ``O(N)`` as Thm. 8.1 claims.

Pure numpy + stdlib (``heapq``), per CLAUDE.md's performance rules.
"""

from __future__ import annotations

import heapq

import numpy as np

from .circle import SOURCE, gap_midpoints, sorted_union
from .solution import NestedActiveSets, PartialCircleSolution

__all__ = ["partial_ot_circle", "prefix_tables"]


def prefix_tables(z: np.ndarray, label: np.ndarray, L: float, w: float):
    """Vectorised ``R``, ``S``, ``p``, ``Q`` on the doubled sequence (draft §7.1).

    Returns ``(R, S, p, Q)``, each of length ``2N + 1`` and indexed as in the draft
    (position ``0`` is the empty prefix; circle atom ``i`` sits at doubled positions
    ``i + 1`` and ``i + 1 + N``).

    Bit-identical to :func:`pawc.pawc_reference.build_doubled_tables`, with the
    ``p`` loop replaced by a sort. The solver itself calls :func:`_prefix_tables`,
    which hands back ``Q`` as a Python list to avoid a round trip.
    """
    R, S, p, Q_list = _prefix_tables(z, label, L, w)
    return R, S, p, np.array(Q_list)


def _prefix_tables(z: np.ndarray, label: np.ndarray, L: float, w: float):
    """As :func:`prefix_tables`, but ``Q`` is returned as a Python list.

    The main loop is scalar-access bound, so it wants ``Q`` as a list; materialising
    it as an ndarray and calling ``.tolist()`` afterwards would allocate the same
    two-million-element structure twice at benchmark sizes.
    """
    z = np.asarray(z, dtype=float)
    N = z.size
    sigma = np.where(np.asarray(label) == SOURCE, 1, -1).astype(np.int64)

    # eq. (24): doubled sequence, with a leading 0 so that index t is the draft's t.
    z_d = np.empty(2 * N + 1)
    z_d[0] = 0.0
    z_d[1 : N + 1] = z
    z_d[N + 1 :] = z + L
    s_d = np.empty(2 * N + 1, dtype=np.int64)
    s_d[0] = 0
    s_d[1 : N + 1] = sigma
    s_d[N + 1 :] = sigma

    R = np.cumsum(s_d)  # eq. (25)
    S = np.cumsum(s_d * z_d)  # eq. (26)

    T = 2 * N + 1
    idx = np.arange(T, dtype=np.int64)
    # Group indices by differential rank, increasing index within each group.
    order = np.lexsort((idx, R))
    R_s = R[order]

    starts = np.empty(T, dtype=bool)
    starts[0] = True
    np.not_equal(R_s[1:], R_s[:-1], out=starts[1:])

    # eq. (28): within a group the predecessor is the previous member; the first
    # member of a group has no predecessor.
    p = np.empty(T, dtype=np.int64)
    p_sorted = np.empty(T, dtype=np.int64)
    p_sorted[0] = -1
    p_sorted[1:] = order[:-1]
    p_sorted[starts] = -1
    p[order] = p_sorted

    # eq. (30): Q_t = Q_{p_t} + w |S_t - S_{p_t}|.
    #
    # This *looks* like a segmented cumulative sum along the rank groups and can be
    # vectorised as `cumsum(steps) - cumsum(steps)[group_start]`. That form is
    # rejected on accuracy grounds, not style: it differences two partial sums
    # accumulated across the whole doubled sequence, whose magnitude grows like
    # N*L, to recover an O(1) chain cost. Measured at N = 10^6 it loses 1.6e-9
    # absolute — past the 1e-9 tolerance CLAUDE.md fixes for cost comparisons —
    # whereas the sequential recursion below stays bit-identical to
    # `pawc_reference.build_doubled_tables` and costs ~0.04 s more at that size.
    # The loop is O(N), so Theorem 8.1's complexity is unaffected.
    S_list = S.tolist()
    p_list = p.tolist()
    Q_list = [0.0] * T
    for t in range(1, T):
        pt = p_list[t]
        if pt >= 0:
            Q_list[t] = Q_list[pt] + w * abs(S_list[t] - S_list[pt])

    return R, S, p, Q_list


def partial_ot_circle(x, y, L: float = 1.0, w: float = 1.0) -> PartialCircleSolution:
    """Exact partial 1-Wasserstein profile on ``S^1_L`` — optimised PAWC.

    Draft Algorithm 1 and Theorem 8.1: returns ``C_k^o`` for every
    ``k = 0, ..., K = min(n, m)``, the nested optimal active sets, and the
    simultaneous cut ``theta*`` valid for every cardinality (Thm. 6.2).

    Parameters
    ----------
    x, y : array_like
        Source and target coordinates in ``[0, L)``; supports must be pairwise
        distinct (Assumption 2.1).
    L, w : float
        Circumference and common atom weight.

    Returns
    -------
    PartialCircleSolution
        ``active_x`` / ``active_y`` are lazy nested views, not materialised lists.
    """
    su = sorted_union(x, y, L)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    N, K = su.N, su.K

    costs = np.zeros(K + 1)
    activation_rank = np.zeros(N, dtype=np.int64)

    if K == 0:
        empty = NestedActiveSets(np.zeros(0, dtype=np.int64), 0)
        sol = PartialCircleSolution(
            x=x, y=y, L=L, w=w, costs=costs,
            active_x=empty, active_y=empty,
            cuts=np.zeros(1), solver="pawc",
        )
        sol.theta_star = 0.0
        sol.theta_star_gap = 0
        sol.activation_rank = activation_rank
        sol.sorted_union = su
        return sol

    _R, _S, _p, Ql = _prefix_tables(su.z, su.label, L, w)

    # --- draft Algorithm 1 lines 3-5 ------------------------------------- #
    # Python lists beat numpy here: the main loop is scalar-access bound.
    label = su.label.tolist()
    succ = list(range(1, N)) + [0]
    pred = [N - 1] + list(range(N - 1))
    inactive = bytearray(b"\x01") * N
    # line 4: cell [z_u, z_{u+1}] is exactly the original gap G_u, which is free.
    free_rep = list(range(N))

    # line 5, vectorised: the initial cells have empty interiors, so eq. (32) is
    # m_u = Q[u+2] - Q[u]; index u + 2 reaches N + 1 for the wrap-around cell.
    Q_arr = np.asarray(Ql)
    m_init = Q_arr[2 : N + 2] - Q_arr[0:N]
    del Q_arr
    succ_arr = np.roll(np.arange(N), -1)
    cand = np.flatnonzero(su.label != su.label[succ_arr])
    heap = list(zip(m_init[cand].tolist(), cand.tolist(), succ_arr[cand].tolist()))
    heapq.heapify(heap)

    heappop = heapq.heappop
    heappush = heapq.heappush
    n_inactive = N
    theta_star_gap = -1

    # --- draft Algorithm 1 lines 7-23 ------------------------------------ #
    for k in range(K):
        u = -1
        m_uv = 0.0
        while heap:
            m_uv, cu, cv = heappop(heap)
            # §8: an entry is valid only if both endpoints are still inactive and
            # cv is still the clockwise successor of cu. Opposite typing was
            # checked at insertion and cannot change.
            if inactive[cu] and inactive[cv] and succ[cu] == cv:
                u, v = cu, cv
                break
        if u < 0:
            raise RuntimeError(
                f"no valid candidate cell at k={k} but K={K}: Theorem 6.1 guarantees "
                "one exists while k < K"
            )

        activation_rank[u] = activation_rank[v] = k + 1
        costs[k + 1] = costs[k] + m_uv

        p_idx = pred[u]
        q_idx = succ[v]

        if n_inactive == 2:
            # lines 14-16: the complementary directed cell [v, u] carries the cut.
            theta_star_gap = free_rep[v]
            inactive[u] = inactive[v] = 0
            n_inactive = 0
            continue

        # line 18
        inactive[u] = inactive[v] = 0
        n_inactive -= 2
        succ[p_idx] = q_idx
        pred[q_idx] = p_idx
        # line 19: free_rep[p] is already the representative of [p, u]; Lemma 6.1's
        # proof shows it survives the merge, so inheritance is a no-op.

        # lines 20-22: at most one new candidate per iteration.
        if p_idx != q_idx and label[p_idx] != label[q_idx]:
            a = p_idx + 1
            b = q_idx + 1 if q_idx > p_idx else q_idx + 1 + N
            heappush(heap, ((Ql[b] - Ql[a - 1]) - (Ql[b - 1] - Ql[a]), p_idx, q_idx))

    # --- draft Algorithm 1 lines 24-26 ----------------------------------- #
    if theta_star_gap < 0:
        remaining = next((i for i in range(N) if inactive[i]), -1)
        theta_star_gap = free_rep[remaining] if remaining >= 0 else 0
    theta_star = float(gap_midpoints(su.z, L)[theta_star_gap])

    order_x, order_y = _activation_order(su, activation_rank, K)

    sol = PartialCircleSolution(
        x=x, y=y, L=L, w=w,
        costs=costs,
        active_x=NestedActiveSets(order_x, K),
        active_y=NestedActiveSets(order_y, K),
        cuts=np.full(K + 1, theta_star),
        solver="pawc",
    )
    sol.theta_star = theta_star
    sol.theta_star_gap = int(theta_star_gap)
    sol.activation_rank = activation_rank
    sol.sorted_union = su
    return sol


def _activation_order(su, activation_rank: np.ndarray, K: int):
    """Activation order as indices into the original ``x`` and ``y`` arrays.

    Vectorised: sort the activated ``z``-positions by their activation rank, then
    map each to its index in the unsorted input via the sorted-union bookkeeping.
    """
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
