"""Independent verification utilities: the draft's own definitions, recomputed.

Nothing here is used by the fast path. Every routine deliberately re-derives a
quantity *from its definition in the draft* so that it can contradict the
incremental bookkeeping of a solver, which is the point of the debug-mode
invariants (CLAUDE.md, "Invariants to assert").
"""

from __future__ import annotations

import numpy as np

from .circle import SOURCE, gap_lengths

__all__ = [
    "cumulative_imbalance",
    "circular_w1_of_active_set",
    "line_w1_sorted",
    "free_gaps_of_cells",
]


def cumulative_imbalance(su, active: set) -> np.ndarray:
    """``H_A(G_i)`` for every original support gap, draft §3.2.

    ``H_A`` increases by one at an active source atom, decreases by one at an
    active target atom, and is unchanged at inactive atoms; it is constant on the
    interior of every original gap ``G_i = (z_i, z_{i+1})``. The returned array is
    that constant, up to the additive constant fixed by the (arbitrary) origin —
    which is exactly the freedom the circulation ``a`` of eq. (8) absorbs.
    """
    b = np.zeros(su.N, dtype=np.int64)
    for i in sorted(active):
        b[i] = 1 if su.label[i] == SOURCE else -1
    return np.cumsum(b)


def circular_w1_of_active_set(su, active: set, w: float = 1.0) -> float:
    """Balanced circular ``W_1`` of a balanced active set, straight from eq. (8).

        W_{1,o}(A) = w * min_a  sum_i  l_i * |H_A(G_i) - a|

    The minimiser is a weighted median of ``{H_A(G_i)}`` with weights ``l_i`` and
    may be taken integral, so the minimum is evaluated over the finitely many
    integer levels actually attained. This uses the circulation formulation only —
    no cut, no chain, no cell — so it is genuinely independent of how PAWC or the
    cut baseline arrives at its number.
    """
    if not active:
        return 0.0
    n_src = sum(1 for i in active if su.label[i] == SOURCE)
    if 2 * n_src != len(active):
        raise AssertionError(
            f"active set is not balanced: {n_src} sources of {len(active)} atoms"
        )
    H = cumulative_imbalance(su, active)
    ell = gap_lengths(su.z, su.L)
    levels = np.unique(H)
    costs = [float(np.sum(ell * np.abs(H - a))) for a in levels]
    return w * float(min(costs))


def line_w1_sorted(xs, ys, w: float = 1.0) -> float:
    """``W_1`` on the line between equal-size point sets: sorted matching, times ``w``."""
    xs = np.sort(np.asarray(xs, dtype=float))
    ys = np.sort(np.asarray(ys, dtype=float))
    if xs.size != ys.size:
        raise ValueError("line_w1_sorted requires balanced supports")
    if xs.size == 0:
        return 0.0
    return w * float(np.sum(np.abs(xs - ys)))


def free_gaps_of_cells(su, active: set, used_gaps: set) -> dict:
    """Recompute, from scratch, which original gaps lie in which current cell.

    A *current cell* is the clockwise arc between two consecutive inactive atoms
    (draft eq. (16)); its interior contains only active atoms. Gap ``G_i`` lies in
    the cell whose left endpoint is the last inactive atom at or before ``z_i``.

    Returns
    -------
    dict
        ``left_inactive_endpoint -> sorted list of free gap indices in that cell``,
        where "free" means not in ``used_gaps`` (draft §6: a gap is *used* once it
        has belonged to a selected candidate cell).
    """
    N = su.N
    inactive = [i for i in range(N) if i not in active]
    if not inactive:
        return {}
    out = {u: [] for u in inactive}
    inactive_set = set(inactive)
    # Walk the circle starting at an inactive atom, so the "current cell" is always
    # defined; gap G_i belongs to the cell whose left endpoint is the most recent
    # inactive atom at index <= i (cyclically).
    start = inactive[0]
    current = start
    for step in range(N):
        i = (start + step) % N
        if i in inactive_set:
            current = i
        if i not in used_gaps:
            out[current].append(i)
    return {u: sorted(v) for u, v in out.items()}
