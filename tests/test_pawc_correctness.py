"""Phase-2 gate: ``pawc_reference`` matches Baseline B for every k.

Gate (plan, Phase 2): the reference PAWC reproduces Baseline B's cost profile on
the full Phase-1 test battery, plus larger random instances (``N`` up to ~2000,
checked against B only). Every check here runs with ``PAWC_DEBUG`` forced on, so
the draft's invariants (Lemma 5.1, Lemma 6.1, cost consistency via eq. (8)) are
asserted at every induction step as a side effect.

Also gated here: the constructive simultaneous cut, Theorem 6.2 — cutting at the
returned ``theta*`` must realise ``C_k^o`` as a *line* cost for **every** k, which
is the property plan recovery (draft §9.1) depends on.

Seeds used, all fixed:
    battery vs A and B ........ 20260804 (same battery as Phase 1)
    families sweep ............ 2026
    large N ................... 99
    simultaneous cut .......... 606
    degenerate cases .......... 77
"""

from __future__ import annotations

import numpy as np
import pytest

from instances import FAMILIES, iter_instances, make_instance, make_rational_instance
from pawc.baseline_cuts import (
    line_profile_at_cut,
    partial_ot_circle_cuts,
    partial_ot_circle_cuts_exact,
)
from pawc.baseline_lp import partial_ot_circle_lp
from pawc.circle import SOURCE, sorted_union
from pawc.pawc import partial_ot_circle as partial_ot_circle_fast
from pawc.pawc import prefix_tables
from pawc.pawc_reference import build_doubled_tables, partial_ot_circle
from pawc.verify import circular_w1_of_active_set

ATOL = 1e-9

#: Same battery size as the Phase-1 gate.
N_BATTERY = 504
SEED_BATTERY = 20260804


def _fail(x, y, k, c_ref, c_base, what):
    return (
        f"\n{what}\n"
        f"  n, m = {len(x)}, {len(y)}   k = {k}\n"
        f"  pawc_reference = {c_ref!r}\n"
        f"  baseline       = {c_base!r}\n"
        f"  x = {np.array2string(np.asarray(x), precision=17, separator=', ')}\n"
        f"  y = {np.array2string(np.asarray(y), precision=17, separator=', ')}\n"
    )


# --------------------------------------------------------------------------- #
# Gate — reference PAWC vs the baselines
# --------------------------------------------------------------------------- #


def test_gate2_reference_matches_baseline_cuts_on_full_battery():
    """The Phase-1 battery, all 504 instances, all k, with invariants asserted."""
    for _family, x, y in iter_instances(N_BATTERY, SEED_BATTERY):
        ref = partial_ot_circle(x, y, debug=True)
        base = partial_ot_circle_cuts(x, y)
        diff = np.abs(ref.costs - base.costs)
        k = int(np.argmax(diff))
        assert np.all(diff <= ATOL), _fail(
            x, y, k, ref.costs[k], base.costs[k],
            "pawc_reference disagrees with Baseline B (cut enumeration).",
        )


def test_gate2_reference_matches_lp_oracle():
    """Reference PAWC against the most trusted solver directly."""
    rng = np.random.default_rng(2026)
    for family in FAMILIES:
        for _ in range(12):
            x, y = make_instance(family, rng)
            ref = partial_ot_circle(x, y, debug=True)
            lp = partial_ot_circle_lp(x, y)
            diff = np.abs(ref.costs - lp.costs)
            k = int(np.argmax(diff))
            assert np.all(diff <= ATOL), _fail(
                x, y, k, ref.costs[k], lp.costs[k],
                "pawc_reference disagrees with Baseline A (LP oracle).",
            )


def test_gate2_reference_matches_exact_rational_baseline():
    """Exact-arithmetic cross-check, to rule out tolerance masking."""
    rng = np.random.default_rng(424242)
    for _ in range(40):
        xq, yq = make_rational_instance(rng)
        exact, _ = partial_ot_circle_cuts_exact(xq, yq)
        ref = partial_ot_circle(
            np.array([float(v) for v in xq]), np.array([float(v) for v in yq]), debug=True
        )
        for k, ce in enumerate(exact):
            assert abs(float(ce) - ref.costs[k]) <= ATOL


@pytest.mark.parametrize("N", [200, 600, 2000])
def test_gate2_large_instances_against_baseline_cuts(N):
    """Larger instances, checked against Baseline B only (plan, Phase 2 gate)."""
    rng = np.random.default_rng(99)
    n = N // 2
    pts = rng.uniform(0, 1, N)
    assert len(np.unique(pts)) == N
    x, y = pts[:n], pts[n:]
    ref = partial_ot_circle(x, y)
    base = partial_ot_circle_cuts(x, y)
    diff = np.abs(ref.costs - base.costs)
    k = int(np.argmax(diff))
    assert np.all(diff <= ATOL), _fail(
        x, y, k, ref.costs[k], base.costs[k], f"large-N disagreement at N={N}."
    )


# --------------------------------------------------------------------------- #
# Theorem 6.2 — the constructive simultaneous cut
# --------------------------------------------------------------------------- #


def test_theorem_6_2_simultaneous_cut_realises_every_cardinality():
    """One gap theta* with C_{k,theta*}^line == C_k^o for every k."""
    rng = np.random.default_rng(606)
    for family in FAMILIES:
        for _ in range(15):
            x, y = make_instance(family, rng)
            ref = partial_ot_circle(x, y, debug=True)
            line_costs, _, _ = line_profile_at_cut(x, y, ref.theta_star)
            assert np.allclose(line_costs, ref.costs, atol=ATOL), (
                f"Theorem 6.2 violated ({family}): cutting at theta*={ref.theta_star!r} "
                f"gives line costs {line_costs!r} but C^o = {ref.costs!r}\n"
                f"x={x!r}\ny={y!r}"
            )


def test_theorem_6_2_cut_gap_was_never_selected():
    """theta* lies in a gap contained in no selected cell (draft §6.1)."""
    rng = np.random.default_rng(606)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        ref = partial_ot_circle(x, y, debug=True)
        su = ref.sorted_union
        used = set()
        for u, v in ref.selected_cells:
            i = u
            while i != v:
                used.add(i)
                i = (i + 1) % su.N
        assert ref.theta_star_gap not in used


def test_plan_recovery_at_theta_star():
    """Draft §9.1: sorted matching after cutting at theta* attains C_k^o."""
    rng = np.random.default_rng(606)
    for family in FAMILIES:
        for _ in range(8):
            x, y = make_instance(family, rng)
            ref = partial_ot_circle(x, y, debug=True)
            for k in range(ref.K + 1):
                ref.check_feasible(k, atol=ATOL)
                assert ref.plan_cost(k) == pytest.approx(ref.costs[k], abs=ATOL), (
                    f"plan recovered at theta* costs {ref.plan_cost(k)!r} but "
                    f"C_{k}^o = {ref.costs[k]!r} ({family})"
                )


# --------------------------------------------------------------------------- #
# The §7.1 machinery, checked directly
# --------------------------------------------------------------------------- #


def test_doubled_tables_balanced_interval_query():
    """Prop. 7.1: c_R([a, b]) = Q_b - Q_{a-1} for every balanced doubled interval."""
    rng = np.random.default_rng(2026)
    for _ in range(40):
        x, y = make_instance("uniform", rng)
        su = sorted_union(x, y)
        tab = build_doubled_tables(su.z, su.label, su.L, w=1.0)
        z_d = np.concatenate([[0.0], su.z, su.z + su.L])
        lab_d = np.concatenate([[0], su.label, su.label])
        for _ in range(40):
            a = int(rng.integers(1, 2 * su.N))
            b = int(rng.integers(a, min(2 * su.N, a + su.N - 1) + 1))
            if not tab.is_balanced(a, b):
                continue
            src = np.sort([z_d[t] for t in range(a, b + 1) if lab_d[t] == SOURCE])
            tgt = np.sort([z_d[t] for t in range(a, b + 1) if lab_d[t] != SOURCE])
            assert src.size == tgt.size
            expected = float(np.sum(np.abs(src - tgt)))
            assert tab.chain_cost(a, b) == pytest.approx(expected, abs=1e-9)


def test_doubled_tables_balance_criterion():
    """eq. (27): [a, b] is balanced iff R_b == R_{a-1}."""
    rng = np.random.default_rng(2026)
    x, y = make_instance("uniform", rng)
    su = sorted_union(x, y)
    tab = build_doubled_tables(su.z, su.label, su.L, w=1.0)
    lab_d = np.concatenate([[0], su.label, su.label])
    for a in range(1, 2 * su.N + 1):
        for b in range(a, min(2 * su.N, a + su.N - 1) + 1):
            n_src = sum(1 for t in range(a, b + 1) if lab_d[t] == SOURCE)
            n_tgt = (b - a + 1) - n_src
            assert tab.is_balanced(a, b) == (n_src == n_tgt)


# --------------------------------------------------------------------------- #
# Degenerate cases
# --------------------------------------------------------------------------- #


def test_degenerate_small_instances():
    rng = np.random.default_rng(77)
    for n in (1, 2, 3):
        for m in (1, 2, 3):
            for _ in range(15):
                pts = rng.uniform(0, 1, n + m)
                if len(np.unique(pts)) != n + m:
                    continue
                x, y = pts[:n], pts[n:]
                ref = partial_ot_circle(x, y, debug=True)
                lp = partial_ot_circle_lp(x, y)
                assert np.allclose(ref.costs, lp.costs, atol=ATOL), _fail(
                    x, y, 0, ref.costs, lp.costs, f"degenerate n={n}, m={m}"
                )


def test_k_zero_is_free():
    rng = np.random.default_rng(77)
    x, y = make_instance("uniform", rng)
    ref = partial_ot_circle(x, y, debug=True)
    assert ref.costs[0] == 0.0
    assert ref.plan(0).sum() == 0.0


# --------------------------------------------------------------------------- #
# Phase-4 gate — optimised pawc.py reproduces pawc_reference.py
# --------------------------------------------------------------------------- #


def test_gate4_optimised_matches_reference_bit_for_bit():
    """Plan, Phase 4.2: ``pawc.py`` must reproduce ``pawc_reference.py`` bit-for-bit.

    Equality is asserted on the raw float64 arrays, not up to a tolerance, and on
    the activation ranks — so the two solvers make the *same* selections in the
    *same* order, not merely selections of equal cost. That is why the ``Q``
    recursion in ``pawc.prefix_tables`` is left sequential.
    """
    for _family, x, y in iter_instances(N_BATTERY, SEED_BATTERY):
        ref = partial_ot_circle(x, y)
        opt = partial_ot_circle_fast(x, y)
        assert np.array_equal(ref.costs, opt.costs), (
            "optimised PAWC is not bit-identical to the reference\n"
            f"x={x!r}\ny={y!r}\nref={ref.costs!r}\nopt={opt.costs!r}"
        )
        assert np.array_equal(ref.activation_rank, opt.activation_rank)
        assert ref.theta_star_gap == opt.theta_star_gap


def test_gate4_optimised_matches_reference_on_families_and_weights():
    rng = np.random.default_rng(4040)
    for family in FAMILIES:
        for _ in range(10):
            x, y = make_instance(family, rng)
            for w, L in ((1.0, 1.0), (0.25, 1.0), (1.0, 7.5)):
                xs, ys = (x * L, y * L) if L != 1.0 else (x, y)
                ref = partial_ot_circle(xs, ys, L=L, w=w)
                opt = partial_ot_circle_fast(xs, ys, L=L, w=w)
                assert np.array_equal(ref.costs, opt.costs), f"{family} w={w} L={L}"


def test_gate4_optimised_prefix_tables_match_reference():
    """``pawc.prefix_tables`` == ``pawc_reference.build_doubled_tables``."""
    rng = np.random.default_rng(4041)
    for family in FAMILIES:
        for _ in range(8):
            x, y = make_instance(family, rng)
            su = sorted_union(x, y)
            R, S, p, Q = prefix_tables(su.z, su.label, su.L, 0.7)
            tab = build_doubled_tables(su.z, su.label, su.L, 0.7)
            assert np.array_equal(R, tab.R)
            assert np.array_equal(S, tab.S)
            assert np.array_equal(p, tab.p)
            assert np.array_equal(Q, tab.Q), (
                f"Q differs by up to {np.max(np.abs(Q - tab.Q))!r} ({family})"
            )


def test_gate4_optimised_matches_baselines():
    """The optimised solver against the LP oracle directly."""
    rng = np.random.default_rng(4042)
    for family in FAMILIES:
        for _ in range(8):
            x, y = make_instance(family, rng)
            opt = partial_ot_circle_fast(x, y)
            lp = partial_ot_circle_lp(x, y)
            assert np.allclose(opt.costs, lp.costs, atol=ATOL), _fail(
                x, y, 0, opt.costs, lp.costs, f"optimised PAWC vs LP oracle ({family})"
            )


def test_gate4_lazy_active_sets_behave_like_lists():
    """``NestedActiveSets`` satisfies the solver contract without materialising."""
    rng = np.random.default_rng(4043)
    x, y = make_instance("uniform", rng)
    ref = partial_ot_circle(x, y)
    opt = partial_ot_circle_fast(x, y)
    assert len(opt.active_x) == opt.K + 1
    for k in range(opt.K + 1):
        assert list(opt.active_x[k]) == list(ref.active_x[k])
        assert list(opt.active_y[k]) == list(ref.active_y[k])
        opt.check_feasible(k, atol=ATOL)
        assert opt.plan_cost(k) == pytest.approx(opt.costs[k], abs=ATOL)
    assert [len(a) for a in opt.active_x] == list(range(opt.K + 1))
    assert list(opt.active_x[-1]) == list(opt.active_x[opt.K])


def test_cost_of_every_active_set_matches_circulation_formula():
    """A_k's cost recomputed by eq. (8) — no cut, no chain, no cell."""
    rng = np.random.default_rng(77)
    for family in FAMILIES:
        for _ in range(6):
            x, y = make_instance(family, rng)
            ref = partial_ot_circle(x, y, debug=True)
            su = ref.sorted_union
            for k in range(ref.K + 1):
                active = {i for i in range(su.N) if 0 < ref.activation_rank[i] <= k}
                got = circular_w1_of_active_set(su, active, ref.w)
                assert got == pytest.approx(ref.costs[k], abs=ATOL)
