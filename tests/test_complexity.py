"""Phase-4: complexity confirmation.

Wall-clock timings belong in ``benchmarks/bench_timings.py``; they are too
machine-dependent to gate CI on. What *is* gated here is the machine-independent
evidence for draft Theorem 8.1:

* the number of heap operations grows **linearly** in ``N`` — the theorem's
  "``O(N)`` heap entries are ever created, each popped at most once";
* only ``O(N)`` candidate insertions happen in total, i.e. at most one new
  candidate per iteration (Algorithm 1 lines 20-22);
* no candidate-list cleaning pass exists — staleness is resolved lazily (§8);
* memory stays ``O(N)``: the nested active sets are never materialised.

A loose wall-clock check is also included, with generous slack, purely to catch a
regression that reintroduces a quadratic term.

Seeds used, all fixed: 424, 425.
"""

from __future__ import annotations

import heapq

import numpy as np
import pytest

import pawc.pawc as pawc_module
from pawc.pawc import partial_ot_circle
from pawc.solution import NestedActiveSets

SIZES = [2_000, 4_000, 8_000, 16_000]
SEED = 424


def _instance(N: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, N)
    assert len(np.unique(pts)) == N
    n = N // 2
    return pts[:n], pts[n:]


class _HeapCounter:
    """Count heapq calls made by ``pawc.pawc`` during one solve."""

    def __enter__(self):
        self.pops = 0
        self.pushes = 0
        self._pop = heapq.heappop
        self._push = heapq.heappush
        self._heapify = heapq.heapify
        self.heapify_calls = 0

        def pop(h):
            self.pops += 1
            return self._pop(h)

        def push(h, item):
            self.pushes += 1
            return self._push(h, item)

        def heapify(h):
            self.heapify_calls += 1
            return self._heapify(h)

        pawc_module.heapq.heappop = pop
        pawc_module.heapq.heappush = push
        pawc_module.heapq.heapify = heapify
        return self

    def __exit__(self, *exc):
        pawc_module.heapq.heappop = self._pop
        pawc_module.heapq.heappush = self._push
        pawc_module.heapq.heapify = self._heapify
        return False


def test_heap_operations_are_linear_in_N():
    """Thm. 8.1: only O(N) heap entries are ever created and each is popped once."""
    stats = {}
    for N in SIZES:
        x, y = _instance(N)
        with _HeapCounter() as c:
            partial_ot_circle(x, y)
        stats[N] = (c.pops, c.pushes, c.heapify_calls)

    for N, (pops, pushes, heapifies) in stats.items():
        K = N // 2
        # Every entry ever created: N initial (via one heapify) + at most one per
        # iteration. Every entry is popped at most once.
        assert heapifies == 1, "the initial candidate list must be built by one heapify"
        assert pushes <= K, (
            f"N={N}: {pushes} pushes exceed K={K}; Algorithm 1 inserts at most one "
            "new candidate per iteration"
        )
        assert pops <= N + pushes, (
            f"N={N}: {pops} pops exceed the {N + pushes} entries ever created — "
            "an entry was popped more than once"
        )

    # Ratios must stay flat: doubling N must not more than double the op count.
    for a, b in zip(SIZES, SIZES[1:]):
        for i, name in enumerate(("pops", "pushes")):
            ratio = stats[b][i] / max(1, stats[a][i])
            assert ratio < 2.35, (
                f"{name} grew by {ratio:.2f}x when N went from {a} to {b}; "
                "O(N) entries implies a ratio near 2"
            )


def test_pops_per_iteration_is_bounded():
    """Lazy deletion must not degenerate: pops per iteration stays O(1) on average."""
    for N in SIZES:
        x, y = _instance(N)
        with _HeapCounter() as c:
            partial_ot_circle(x, y)
        per_iter = c.pops / (N // 2)
        assert per_iter < 4.0, (
            f"N={N}: {per_iter:.2f} pops per iteration — stale entries are "
            "accumulating faster than O(1) per step"
        )


def test_memory_active_sets_are_lazy():
    """The nested active sets must be a view, not K+1 materialised arrays."""
    x, y = _instance(4_000)
    sol = partial_ot_circle(x, y)
    assert isinstance(sol.active_x, NestedActiveSets)
    assert isinstance(sol.active_y, NestedActiveSets)
    # the whole family is backed by one length-K array per measure
    assert sol.active_x._order.nbytes <= 8 * (sol.K + 1)
    assert sol.active_y._order.nbytes <= 8 * (sol.K + 1)


@pytest.mark.slow
def test_wall_clock_growth_is_not_quadratic():
    """Loose regression guard, with generous slack for a shared CI machine.

    A genuine ``O(N^2)`` term would show a ~4x cost per doubling; ``O(N log N)``
    shows ~2.1x. The bound below (3.0x per doubling, best of three repetitions)
    sits comfortably between the two.
    """
    import time

    sizes = [20_000, 40_000, 80_000]
    times = []
    for N in sizes:
        x, y = _instance(N, seed=425)
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            partial_ot_circle(x, y)
            reps.append(time.perf_counter() - t0)
        times.append(min(reps))

    for (na, ta), (nb, tb) in zip(zip(sizes, times), zip(sizes[1:], times[1:])):
        assert tb / ta < 3.0, (
            f"time grew {tb / ta:.2f}x from N={na} to N={nb} ({ta:.3f}s -> {tb:.3f}s); "
            "that is closer to quadratic than to N log N"
        )
