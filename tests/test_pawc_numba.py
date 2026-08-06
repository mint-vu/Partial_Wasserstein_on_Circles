"""The numba-JIT PAWC variant must be bit-for-bit identical to the pure-numpy one.

``pawc_numba`` exists so that the timing comparison against Baseline B is
JIT-vs-JIT (the vendored PAWL is numba-compiled — see ``vendor/PROVENANCE.md``).
It is an *optimisation*, so CLAUDE.md's performance rule applies in full: it must
be behaviour-preserving. That is gated here exactly as ``pawc.py`` is gated against
``pawc_reference.py`` — raw float64 equality of costs, plus equality of the
activation ranks and the simultaneous cut, so the two make the same selections in
the same order rather than merely selections of equal cost.

Compiled with ``fastmath=False`` deliberately; ``fastmath=True`` would license
reassociation and break exactly this property.

Seeds used, all fixed: 20260804 (the shared battery), 4040, 909.
"""

from __future__ import annotations

import numpy as np
import pytest

from instances import FAMILIES, iter_instances, make_instance
from pawc.baseline_cuts import partial_ot_circle_cuts
from pawc.baseline_lp import partial_ot_circle_lp
from pawc.pawc import partial_ot_circle as pawc_numpy
from pawc.pawc_numba import partial_ot_circle as pawc_jit
from pawc.pawc_reference import partial_ot_circle as pawc_ref

ATOL = 1e-9
N_BATTERY = 504
SEED_BATTERY = 20260804


def test_numba_matches_numpy_bit_for_bit_on_full_battery():
    for _family, x, y in iter_instances(N_BATTERY, SEED_BATTERY):
        a = pawc_numpy(x, y)
        b = pawc_jit(x, y)
        assert np.array_equal(a.costs, b.costs), (
            "numba PAWC is not bit-identical to pure-numpy PAWC\n"
            f"x={x!r}\ny={y!r}\nnumpy={a.costs!r}\nnumba={b.costs!r}\n"
            f"max|diff|={np.max(np.abs(a.costs - b.costs))!r}"
        )
        assert np.array_equal(a.activation_rank, b.activation_rank)
        assert a.theta_star_gap == b.theta_star_gap


def test_numba_matches_reference_bit_for_bit():
    rng = np.random.default_rng(4040)
    for family in FAMILIES:
        for _ in range(10):
            x, y = make_instance(family, rng)
            ref = pawc_ref(x, y)
            jit = pawc_jit(x, y)
            assert np.array_equal(ref.costs, jit.costs), family
            assert np.array_equal(ref.activation_rank, jit.activation_rank), family


def test_numba_matches_across_weights_and_circumferences():
    rng = np.random.default_rng(4040)
    for family in FAMILIES:
        for _ in range(6):
            x, y = make_instance(family, rng)
            for w, L in ((1.0, 1.0), (0.25, 1.0), (1.0, 7.5), (0.5, 2.0)):
                xs, ys = (x * L, y * L) if L != 1.0 else (x, y)
                a = pawc_numpy(xs, ys, L=L, w=w)
                b = pawc_jit(xs, ys, L=L, w=w)
                assert np.array_equal(a.costs, b.costs), f"{family} w={w} L={L}"


def test_numba_matches_lp_oracle():
    """Against the most trusted solver directly, not only against its own family."""
    rng = np.random.default_rng(909)
    for family in FAMILIES:
        for _ in range(8):
            x, y = make_instance(family, rng)
            jit = pawc_jit(x, y)
            lp = partial_ot_circle_lp(x, y)
            assert np.allclose(jit.costs, lp.costs, atol=ATOL), (
                f"numba PAWC vs LP oracle ({family})\nx={x!r}\ny={y!r}\n"
                f"numba={jit.costs!r}\nlp={lp.costs!r}"
            )


def test_numba_satisfies_the_solver_contract():
    rng = np.random.default_rng(909)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        sol = pawc_jit(x, y)
        assert sol.costs.size == sol.K + 1
        for k in range(sol.K + 1):
            sol.check_feasible(k, atol=ATOL)
            assert sol.plan_cost(k) == pytest.approx(sol.costs[k], abs=ATOL)
            assert len(sol.active_x[k]) == k and len(sol.active_y[k]) == k
        for k in range(sol.K):
            assert set(sol.active_x[k]) <= set(sol.active_x[k + 1])
            assert set(sol.active_y[k]) <= set(sol.active_y[k + 1])


def test_numba_large_instance_matches_numpy_and_baseline():
    rng = np.random.default_rng(909)
    for N in (200, 2000):
        pts = rng.uniform(0, 1, N)
        assert len(np.unique(pts)) == N
        x, y = pts[: N // 2], pts[N // 2 :]
        a = pawc_numpy(x, y)
        b = pawc_jit(x, y)
        assert np.array_equal(a.costs, b.costs), f"N={N}"
        base = partial_ot_circle_cuts(x, y)
        assert np.allclose(b.costs, base.costs, atol=ATOL), f"N={N}"


def test_numba_degenerate_sizes():
    rng = np.random.default_rng(909)
    for n in (1, 2, 3):
        for m in (1, 2, 3):
            pts = rng.uniform(0, 1, n + m)
            if len(np.unique(pts)) != n + m:
                continue
            x, y = pts[:n], pts[n:]
            assert np.array_equal(pawc_numpy(x, y).costs, pawc_jit(x, y).costs)


def test_numba_is_actually_compiled_without_fastmath():
    """Guard the fidelity decision: fastmath would break bit-for-bit equality."""
    from pawc.pawc_numba import pawc_kernel

    assert pawc_kernel.targetoptions.get("fastmath") is False, (
        "pawc_numba must be compiled with fastmath=False; reassociation would break "
        "bit-for-bit agreement with pawc_reference"
    )
