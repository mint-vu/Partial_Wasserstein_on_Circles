"""Phase-1 gate: Baseline B (cut enumeration) == Baseline A (LP oracle).

Gates (plan, Phase 1):

1. **B == A** at ``atol = 1e-9`` on >= 500 random instances with ``n, m in [2, 30]``
   spanning seven instance families, for every ``k = 0, ..., min(n, m)``; plus an
   exact-arithmetic variant on ``fractions.Fraction`` coordinates to rule out
   tolerance masking.
2. **Cut-at-gap sufficiency**: restricting cuts to the ``N`` inter-point gaps loses
   nothing (draft Prop. 3.1 / Cor. 3.1). Checked two ways — B (gap cuts only)
   reproduces A (which never uses the cut structure), and randomly placed cuts
   never beat the gap-cut envelope.
3. **Balanced sanity**: at ``k = n = m`` the cost matches POT's circular ``W_1``.

Seeds used, all fixed:
    gate 1 float battery ........ 20260804
    gate 1 exact battery ........ 424242
    gate 2 random cuts .......... 31337
    gate 3 balanced ............. 987654
    degenerate/structural ....... 11, 12, 13

CLAUDE.md testing rule 1: none of these tolerances or generators may be relaxed to
make a solver pass. Rule 3: a Baseline-B/LP disagreement, or a non-gap cut beating
every gap cut, is a theory-level finding — report the instance, do not code around it.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from instances import FAMILIES, iter_instances, make_instance, make_rational_instance
from pawc.baseline_cuts import (
    all_cut_costs_exact,
    line_profile_at_cut,
    partial_ot_circle_cuts,
    partial_ot_circle_cuts_exact,
)
from pawc.baseline_lp import partial_ot_circle_lp
from pawc.circle import sorted_union

ATOL = 1e-9

#: Total instances in the gate-1 float battery. The plan requires >= 500.
N_GATE1_INSTANCES = 504  # 72 per family x 7 families
SEED_GATE1 = 20260804


def _report(family, x, y, k, ca, cb, what):
    return (
        f"\n{what}\n"
        f"  family = {family}\n"
        f"  n, m   = {len(x)}, {len(y)}\n"
        f"  k      = {k}\n"
        f"  A (LP)          = {ca!r}\n"
        f"  B (cut enum)    = {cb!r}\n"
        f"  |A - B|         = {abs(ca - cb)!r}\n"
        f"  x = {np.array2string(np.asarray(x), precision=17, separator=', ')}\n"
        f"  y = {np.array2string(np.asarray(y), precision=17, separator=', ')}\n"
        "CLAUDE.md testing rule 3: this is a theory-level finding, not a coding bug."
    )


# --------------------------------------------------------------------------- #
# Gate 1 — B == A
# --------------------------------------------------------------------------- #


def test_gate1_baselines_agree_float_battery():
    """B == A on the full randomized battery, all k, atol 1e-9."""
    worst = 0.0
    worst_case = None
    seen_families = set()
    seen_unequal = False

    for family, x, y in iter_instances(N_GATE1_INSTANCES, SEED_GATE1):
        seen_families.add(family)
        seen_unequal |= len(x) != len(y)
        a = partial_ot_circle_lp(x, y)
        b = partial_ot_circle_cuts(x, y)
        assert a.costs.shape == b.costs.shape
        diff = np.abs(a.costs - b.costs)
        k_bad = int(np.argmax(diff))
        if diff[k_bad] > worst:
            worst = float(diff[k_bad])
            worst_case = (family, x, y, k_bad, a.costs[k_bad], b.costs[k_bad])
        assert np.all(diff <= ATOL), _report(
            family, x, y, k_bad, a.costs[k_bad], b.costs[k_bad],
            "Baseline B (cut enumeration) disagrees with Baseline A (LP oracle).",
        )

    assert seen_families == set(FAMILIES), "battery did not cover every instance family"
    assert seen_unequal, "battery contained no instance with n != m"
    # Informational: keep an eye on the achieved margin.
    assert worst <= ATOL, worst_case


def test_gate1_baselines_agree_exact_rational():
    """B == A in exact rational arithmetic — rules out tolerance masking.

    Baseline B is recomputed with ``Fraction`` coordinates and the exact
    non-crossing DP line solver; Baseline A is the float LP. Their difference must
    be zero to floating-point round-off of the *exact* value, which is a strictly
    stronger statement than the float-vs-float comparison above.
    """
    rng = np.random.default_rng(424242)
    for _ in range(60):
        xq, yq = make_rational_instance(rng)
        exact_costs, _ = partial_ot_circle_cuts_exact(xq, yq)
        xf = np.array([float(v) for v in xq])
        yf = np.array([float(v) for v in yq])
        a = partial_ot_circle_lp(xf, yf)
        for k, ce in enumerate(exact_costs):
            assert abs(float(ce) - a.costs[k]) <= ATOL, _report(
                "rational", xq, yq, k, a.costs[k], float(ce),
                "Exact Baseline B disagrees with the LP oracle.",
            )


def test_gate1_exact_and_float_baseline_b_agree():
    """The Fraction/DP and float/PAWL implementations of Baseline B coincide."""
    rng = np.random.default_rng(424242)
    for _ in range(40):
        xq, yq = make_rational_instance(rng)
        exact_costs, _ = partial_ot_circle_cuts_exact(xq, yq)
        xf = np.array([float(v) for v in xq])
        yf = np.array([float(v) for v in yq])
        b = partial_ot_circle_cuts(xf, yf)
        for k, ce in enumerate(exact_costs):
            assert abs(float(ce) - b.costs[k]) <= ATOL


# --------------------------------------------------------------------------- #
# Gate 2 — cut-at-gap sufficiency
# --------------------------------------------------------------------------- #


def test_gate2_random_cuts_never_beat_the_gap_envelope():
    """No cut anywhere on the circle beats the minimum over the N gap cuts.

    Draft Prop. 3.1 says the unwrapped line problem depends on the cut only
    through which gap contains it, so this must hold; a violation would be a
    theory-level finding (CLAUDE.md rule 3).
    """
    rng = np.random.default_rng(31337)
    for family in FAMILIES:
        for _ in range(12):
            x, y = make_instance(family, rng)
            b = partial_ot_circle_cuts(x, y)
            for theta in rng.uniform(0, 1, 25):
                costs_theta, _, _ = line_profile_at_cut(x, y, float(theta))
                assert np.all(costs_theta >= b.costs - ATOL), _report(
                    family, x, y, int(np.argmin(costs_theta - b.costs)),
                    b.costs[int(np.argmin(costs_theta - b.costs))],
                    costs_theta[int(np.argmin(costs_theta - b.costs))],
                    f"A cut at theta={theta!r} beat the gap-cut envelope.",
                )


def test_gate2_cut_within_a_gap_does_not_matter():
    """Draft Prop. 3.1: the line profile depends on the gap, not on theta inside it."""
    rng = np.random.default_rng(31337)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        su = sorted_union(x, y)
        z = su.z
        ell = np.diff(np.append(z, z[0] + 1.0))
        for i in range(su.N):
            c1, _, _ = line_profile_at_cut(x, y, float(np.mod(z[i] + 0.2 * ell[i], 1.0)))
            c2, _, _ = line_profile_at_cut(x, y, float(np.mod(z[i] + 0.8 * ell[i], 1.0)))
            assert np.allclose(c1, c2, atol=ATOL)


def test_gate2_sufficiency_exact_arithmetic():
    """Exact-arithmetic form of gate 2: min over gap cuts equals the LP optimum."""
    rng = np.random.default_rng(31337)
    for _ in range(30):
        xq, yq = make_rational_instance(rng, max_size=6)
        table, _ = all_cut_costs_exact(xq, yq)
        envelope = [min(row[k] for row in table) for k in range(len(table[0]))]
        a = partial_ot_circle_lp(
            np.array([float(v) for v in xq]), np.array([float(v) for v in yq])
        )
        for k, ce in enumerate(envelope):
            assert abs(float(ce) - a.costs[k]) <= ATOL


# --------------------------------------------------------------------------- #
# Gate 3 — balanced sanity check against POT's circular solver
# --------------------------------------------------------------------------- #


def test_gate3_balanced_matches_pot_wasserstein_circle():
    """At k = n = m the partial problem is balanced circular W1."""
    from ot.lp import wasserstein_circle

    rng = np.random.default_rng(987654)
    for _ in range(120):
        n = int(rng.integers(2, 31))
        pts = rng.uniform(0, 1, 2 * n)
        # keep supports disjoint
        while len(np.unique(pts)) != 2 * n:
            pts = rng.uniform(0, 1, 2 * n)
        x, y = pts[:n], pts[n:]

        w = 1.0 / n  # probability measures, so POT's normalisation applies
        b = partial_ot_circle_cuts(x, y, w=w)
        pot = float(
            np.atleast_1d(
                wasserstein_circle(x.reshape(-1, 1), y.reshape(-1, 1), p=1)
            )[0]
        )
        assert b.costs[n] == pytest.approx(pot, abs=1e-8), (
            f"balanced k=n=m disagreement with POT: PAWC-baseline {b.costs[n]!r} "
            f"vs ot.lp.wasserstein_circle {pot!r}\n x={x!r}\n y={y!r}"
        )


def test_gate3_balanced_matches_lp_oracle():
    rng = np.random.default_rng(987654)
    for _ in range(60):
        n = int(rng.integers(2, 21))
        pts = rng.uniform(0, 1, 2 * n)
        x, y = pts[:n], pts[n:]
        a = partial_ot_circle_lp(x, y, ks=[n])
        b = partial_ot_circle_cuts(x, y)
        assert a.costs[n] == pytest.approx(b.costs[n], abs=ATOL)


# --------------------------------------------------------------------------- #
# Structural properties of the baselines themselves
# --------------------------------------------------------------------------- #


def test_baseline_profiles_are_nondecreasing_and_convex():
    """k -> C_k^o is nondecreasing and convex (Figalli-style; draft Prop. 9.1)."""
    rng = np.random.default_rng(11)
    for family in FAMILIES:
        for _ in range(10):
            x, y = make_instance(family, rng)
            b = partial_ot_circle_cuts(x, y)
            d = np.diff(b.costs)
            assert np.all(d >= -ATOL), f"cost profile decreased: {b.costs!r}"
            if d.size >= 2:
                assert np.all(np.diff(d) >= -ATOL), (
                    f"cost profile not convex ({family}): marginals {d!r}\n"
                    f"x={x!r}\ny={y!r}"
                )


def test_baseline_lp_plans_are_feasible_and_match_reported_cost():
    rng = np.random.default_rng(12)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        a = partial_ot_circle_lp(x, y)
        for k in range(a.K + 1):
            a.check_feasible(k, atol=ATOL)
            assert a.plan_cost(k) == pytest.approx(a.costs[k], abs=ATOL)


def test_baseline_cuts_plans_are_feasible_and_match_reported_cost():
    """Plan recovery at the recorded argmin cut reproduces the cost (draft §9.1)."""
    rng = np.random.default_rng(12)
    for family in FAMILIES:
        for _ in range(6):
            x, y = make_instance(family, rng)
            b = partial_ot_circle_cuts(x, y)
            for k in range(b.K + 1):
                b.check_feasible(k, atol=ATOL)
                assert b.plan_cost(k) == pytest.approx(b.costs[k], abs=ATOL), (
                    f"plan recovered at cut {b.cuts[k]!r} costs {b.plan_cost(k)!r} "
                    f"but the solver reported {b.costs[k]!r} (family {family}, k={k})"
                )


def test_rotation_equivariance_of_baselines():
    """Rotating every point leaves the whole profile unchanged."""
    rng = np.random.default_rng(13)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        base = partial_ot_circle_cuts(x, y).costs
        base_lp = partial_ot_circle_lp(x, y).costs
        for off in rng.uniform(0, 1, 4):
            xr, yr = np.mod(x + off, 1.0), np.mod(y + off, 1.0)
            assert np.allclose(partial_ot_circle_cuts(xr, yr).costs, base, atol=ATOL)
            assert np.allclose(partial_ot_circle_lp(xr, yr).costs, base_lp, atol=ATOL)


def test_reflection_symmetry_of_baselines():
    rng = np.random.default_rng(13)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        base = partial_ot_circle_cuts(x, y).costs
        xr, yr = np.mod(-x, 1.0), np.mod(-y, 1.0)
        assert np.allclose(partial_ot_circle_cuts(xr, yr).costs, base, atol=ATOL)


def test_tiny_arc_reduces_to_the_line():
    """All points inside a short arc: one PAWL call on the line is already optimal."""
    from pawc.baseline_cuts import line_profile_pawl

    rng = np.random.default_rng(13)
    for _ in range(20):
        x, y = make_instance("tiny_arc", rng)
        b = partial_ot_circle_cuts(x, y)
        # the arc is far shorter than L/2, so no geodesic wraps
        line_costs, _, _ = line_profile_pawl(x, y)
        assert np.allclose(b.costs, line_costs, atol=ATOL)


def test_degenerate_sizes():
    rng = np.random.default_rng(13)
    # n = 1 or m = 1
    for _ in range(20):
        m = int(rng.integers(1, 12))
        pts = rng.uniform(0, 1, 1 + m)
        x, y = pts[:1], pts[1:]
        a = partial_ot_circle_lp(x, y)
        b = partial_ot_circle_cuts(x, y)
        assert np.allclose(a.costs, b.costs, atol=ATOL)
        assert a.K == 1 and a.costs[0] == 0.0
    # n = m = 1: cost(1) is just the geodesic distance
    from pawc.circle import geodesic_distance

    b = partial_ot_circle_cuts(np.array([0.05]), np.array([0.9]))
    assert b.costs[1] == pytest.approx(float(geodesic_distance(0.05, 0.9)), abs=ATOL)


def test_general_circumference_scales_costs():
    """Costs scale linearly with the circumference."""
    rng = np.random.default_rng(13)
    x, y = make_instance("uniform", rng)
    c1 = partial_ot_circle_cuts(x, y, L=1.0).costs
    L = 7.5
    cL = partial_ot_circle_cuts(x * L, y * L, L=L).costs
    assert np.allclose(cL, L * c1, atol=1e-8)


def test_weight_scales_costs():
    rng = np.random.default_rng(13)
    x, y = make_instance("uniform", rng)
    c1 = partial_ot_circle_cuts(x, y, w=1.0).costs
    cw = partial_ot_circle_cuts(x, y, w=0.125).costs
    assert np.allclose(cw, 0.125 * c1, atol=ATOL)


def test_regression_tiny_arc_lp_objective_scaling():
    """Permanent regression case (CLAUDE.md rule 4), seed 20260804, family tiny_arc.

    All 28 atoms sit inside an arc of width ~4.3e-4, so the entire cost matrix is
    O(1e-4). With an unscaled objective HiGHS stopped at a strictly suboptimal
    vertex for k=2 (5.760569180335295e-06) while the true optimum, confirmed by
    brute force over all C(14,2)^2 matchings, is 5.739313285180181e-06 — which
    Baseline B had found. The LP oracle now normalises the cost matrix to unit
    maximum before solving.
    """
    x = np.array([
        0.8394790180333656, 0.8392629628879154, 0.8393083897871096,
        0.8393919806694121, 0.8396321284142567, 0.8395087567359489,
        0.8392047091640856, 0.8393011550419234, 0.8395910353067663,
        0.8395375034596502, 0.8394921630857745, 0.8395261963198503,
        0.8395835122859083, 0.8392671623468956,
    ])
    y = np.array([
        0.8394110198954192, 0.8394452339307863, 0.8394966873300598,
        0.8394187840331980, 0.8393484919394186, 0.8392683986717906,
        0.8392809304607101, 0.8393371813240691, 0.8394411652610708,
        0.8394026437807697, 0.8395318146450073, 0.8393179956009703,
        0.8395422373577975, 0.8393128927754998,
    ])
    a = partial_ot_circle_lp(x, y)
    b = partial_ot_circle_cuts(x, y)
    assert a.costs[2] == pytest.approx(5.739313285180181e-06, rel=1e-12)
    assert np.max(np.abs(a.costs - b.costs)) <= ATOL


def test_exact_fraction_regression_case():
    """A permanent exact-arithmetic regression instance (CLAUDE.md rule 4)."""
    d = 60
    x = [Fraction(v, d) for v in (1, 7, 29, 41)]
    y = [Fraction(v, d) for v in (3, 19, 31, 53)]
    costs, cuts = partial_ot_circle_cuts_exact(x, y)
    assert costs[0] == Fraction(0)
    # every value is exactly rational and the profile is convex
    diffs = [costs[k + 1] - costs[k] for k in range(len(costs) - 1)]
    assert all(d2 >= 0 for d2 in diffs)
    assert all(diffs[i + 1] >= diffs[i] for i in range(len(diffs) - 1))
    a = partial_ot_circle_lp(
        np.array([float(v) for v in x]), np.array([float(v) for v in y])
    )
    for k, c in enumerate(costs):
        assert abs(float(c) - a.costs[k]) <= ATOL
