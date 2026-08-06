"""Phase-3: invariant and property tests.

These test the *theory* rather than only the cost values, since optima can be
non-unique and cost equality alone can hide a structurally wrong solution
(plan, Phase 3).

Properties covered:

* **Nestedness** — ``A_k ⊂ A_{k+1}``, exactly one new source and one new target
  (draft Thm. 4.1, Cor. 4.1).
* **Cyclic-neighbour addition** — the two newly activated atoms are consecutive in
  the cyclic ordering of the inactive set ``A_k^c`` (draft Thm. 4.2).
* **Free-gap invariant** — the recorded free gaps are exactly the maximal arcs of
  non-active ``z``-points separating matched regions (draft Lemma 6.1).
* **Cost profile shape** — ``k -> C_k^o`` nondecreasing and convex.
* **Plan feasibility** — marginals ``<= w``, total mass ``k w``, and the plan's
  cost under the *circular* metric equals the reported cost.
* **Rotation equivariance** and **reflection symmetry**.
* **Interpolation** at non-integer transported mass (draft Prop. 9.1).
* **Degenerate cases** — ``k = 0``, ``k = 1``, ``n = 1`` or ``m = 1``, and all
  points in a tiny arc (the circle must reduce to the line).

Randomised generation uses ``hypothesis`` with a fixed derandomised profile;
counterexamples found by hypothesis are added below as permanent regression cases
(CLAUDE.md testing rule 4).

Seeds used, all fixed: 5150, 5151, 5152, 5153, 5154, 5155.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from instances import FAMILIES, make_instance
from pawc.baseline_cuts import line_profile_pawl, partial_ot_circle_cuts
from pawc.baseline_lp import partial_ot_circle_lp
from pawc.circle import geodesic_distance
from pawc.pawc_reference import partial_ot_circle
from pawc.verify import circular_w1_of_active_set, free_gaps_of_cells

ATOL = 1e-9

settings.register_profile(
    "pawc",
    max_examples=60,
    deadline=None,
    derandomize=True,  # fixed seeds in CI, per CLAUDE.md testing rule 4
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)
settings.load_profile("pawc")


def _active_z(ref, k: int) -> set:
    """Active set of ``A_k`` as indices into the sorted union support."""
    return {i for i in range(ref.sorted_union.N) if 0 < ref.activation_rank[i] <= k}


# --------------------------------------------------------------------------- #
# Nestedness and cyclic-neighbour addition
# --------------------------------------------------------------------------- #


def test_nestedness_of_active_sets():
    """Draft Thm. 4.1 / Cor. 4.1: A_k ⊂ A_{k+1}, one new source and one new target."""
    rng = np.random.default_rng(5150)
    for family in FAMILIES:
        for _ in range(12):
            x, y = make_instance(family, rng)
            ref = partial_ot_circle(x, y, debug=True)
            su = ref.sorted_union
            for k in range(ref.K):
                a_k, a_k1 = _active_z(ref, k), _active_z(ref, k + 1)
                assert a_k <= a_k1, f"nestedness violated at k={k} ({family})"
                new = a_k1 - a_k
                assert len(new) == 2, f"step {k}->{k+1} activated {len(new)} atoms"
                labels = {int(su.label[i]) for i in new}
                assert len(labels) == 2, (
                    f"step {k}->{k+1} activated two atoms of the same measure "
                    f"(labels {labels}); Thm. 4.1 requires one source and one target"
                )
            # the index lists exposed on the solution are nested too
            for k in range(ref.K):
                assert set(ref.active_x[k]) <= set(ref.active_x[k + 1])
                assert set(ref.active_y[k]) <= set(ref.active_y[k + 1])
                assert len(ref.active_x[k]) == k and len(ref.active_y[k]) == k


def test_cyclic_neighbour_addition():
    """Draft Thm. 4.2: the two new atoms are consecutive in the cyclic order of A_k^c."""
    rng = np.random.default_rng(5151)
    for family in FAMILIES:
        for _ in range(12):
            x, y = make_instance(family, rng)
            ref = partial_ot_circle(x, y, debug=True)
            su = ref.sorted_union
            for k in range(ref.K):
                inactive = [i for i in range(su.N) if i not in _active_z(ref, k)]
                new = sorted(_active_z(ref, k + 1) - _active_z(ref, k))
                pos = {u: j for j, u in enumerate(inactive)}
                r = len(inactive)
                j0, j1 = pos[new[0]], pos[new[1]]
                adjacent = (j1 - j0) % r == 1 or (j0 - j1) % r == 1
                assert adjacent, (
                    f"Thm. 4.2 violated at k={k} ({family}): newly activated atoms "
                    f"{new} are not cyclic neighbours in the inactive set {inactive}"
                )


def test_selected_cells_have_active_interiors():
    """Draft §5: a current cell's open interior contains only active atoms."""
    rng = np.random.default_rng(5151)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        ref = partial_ot_circle(x, y, debug=True)
        su = ref.sorted_union
        for k, (u, v) in enumerate(ref.selected_cells):
            active_before = _active_z(ref, k)
            i = (u + 1) % su.N
            while i != v:
                assert i in active_before, (
                    f"selected cell [{u},{v}] at step {k} has an inactive interior atom {i}"
                )
                i = (i + 1) % su.N


# --------------------------------------------------------------------------- #
# Free-gap invariant (Lemma 6.1)
# --------------------------------------------------------------------------- #


def test_free_gap_invariant_holds_at_every_step():
    """Every current cell contains at least one free original gap, at every k.

    Recomputed from scratch: a gap is *used* once it has belonged to a selected
    candidate cell, and the current cells are the arcs between consecutive
    inactive atoms.
    """
    rng = np.random.default_rng(5152)
    for family in FAMILIES:
        for _ in range(10):
            x, y = make_instance(family, rng)
            ref = partial_ot_circle(x, y, debug=True)
            su = ref.sorted_union
            used: set[int] = set()
            for k in range(ref.K):
                active = _active_z(ref, k)
                if su.N - len(active) == 0:
                    break
                per_cell = free_gaps_of_cells(su, active, used)
                for u, free_here in per_cell.items():
                    assert free_here, (
                        f"Lemma 6.1 violated at k={k} ({family}): the current cell with "
                        f"left endpoint {u} has no free gap; used = {sorted(used)}"
                    )
                u, v = ref.selected_cells[k]
                i = u
                while i != v:
                    used.add(i)
                    i = (i + 1) % su.N


def test_free_gaps_separate_matched_regions():
    """The free gaps are arcs of non-active z-points separating matched regions.

    Concretely: at every step, each current cell owns at least one gap that no
    selected cell has covered, and Lemma 6.2 says cutting the circle at such a gap
    is compatible with the whole sequence of selections made so far — so the line
    cost under that cut must reproduce ``C_k^o`` exactly, not merely bound it.
    """
    from pawc.baseline_cuts import line_profile_at_cut
    from pawc.circle import gap_midpoints

    rng = np.random.default_rng(5152)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        ref = partial_ot_circle(x, y, debug=True)
        su = ref.sorted_union
        mids = gap_midpoints(su.z, su.L)
        used: set[int] = set()
        for k in range(ref.K):
            active = _active_z(ref, k)
            for u, free_here in free_gaps_of_cells(su, active, used).items():
                assert free_here, f"cell {u} has no free gap at k={k} ({family})"
                for g in free_here:
                    line_costs, _, _ = line_profile_at_cut(x, y, float(mids[g]))
                    assert line_costs[k] == pytest.approx(ref.costs[k], abs=ATOL), (
                        f"Lemma 6.2 violated ({family}, k={k}): cutting at free gap {g} "
                        f"of cell {u} gives {line_costs[k]!r} but C_k^o = {ref.costs[k]!r}"
                    )
            u, v = ref.selected_cells[k]
            i = u
            while i != v:
                used.add(i)
                i = (i + 1) % su.N


# --------------------------------------------------------------------------- #
# Cost profile shape and feasibility
# --------------------------------------------------------------------------- #


def test_cost_profile_nondecreasing_and_convex():
    rng = np.random.default_rng(5153)
    for family in FAMILIES:
        for _ in range(15):
            x, y = make_instance(family, rng)
            ref = partial_ot_circle(x, y, debug=True)
            d = np.diff(ref.costs)
            assert np.all(d >= -ATOL), f"cost profile decreased ({family}): {ref.costs!r}"
            if d.size >= 2:
                assert np.all(np.diff(d) >= -ATOL), (
                    f"cost profile not convex ({family}): marginals {d!r}\n"
                    f"x={x!r}\ny={y!r}"
                )


def test_plans_feasible_and_cost_matches_circular_metric():
    """Marginals <= w, total mass k*w, and cost re-evaluated with d_{S^1}."""
    rng = np.random.default_rng(5153)
    for family in FAMILIES:
        for _ in range(8):
            x, y = make_instance(family, rng)
            for w in (1.0, 0.25):
                ref = partial_ot_circle(x, y, w=w, debug=True)
                for k in range(ref.K + 1):
                    ref.check_feasible(k, atol=ATOL)
                    pi = ref.plan(k)
                    assert np.all(np.isin(np.unique(pi), [0.0, w])) or np.allclose(
                        np.unique(pi), [0.0, w], atol=ATOL
                    )
                    assert ref.plan_cost(k) == pytest.approx(ref.costs[k], abs=ATOL)


def test_active_set_cost_equals_circulation_formula():
    """Independent check of every A_k via draft eq. (8) — no cut, cell or chain."""
    rng = np.random.default_rng(5153)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        ref = partial_ot_circle(x, y, debug=True)
        for k in range(ref.K + 1):
            got = circular_w1_of_active_set(ref.sorted_union, _active_z(ref, k), ref.w)
            assert got == pytest.approx(ref.costs[k], abs=ATOL)


def test_interpolation_at_noninteger_mass():
    """Draft Prop. 9.1: PW(s) is the linear interpolation between C_k and C_{k+1}."""
    rng = np.random.default_rng(5153)
    x, y = make_instance("uniform", rng)
    ref = partial_ot_circle(x, y, debug=True)
    for k in range(ref.K):
        for lam in (0.0, 0.25, 0.5, 0.75):
            pi = (1 - lam) * ref.plan(k) + lam * ref.plan(k + 1)
            expected = (1 - lam) * ref.costs[k] + lam * ref.costs[k + 1]
            nz = np.nonzero(pi)
            got = float(
                np.sum(pi[nz] * geodesic_distance(ref.x[nz[0]], ref.y[nz[1]], ref.L))
            )
            assert got == pytest.approx(expected, abs=ATOL)
            assert pi.sum() == pytest.approx((k + lam) * ref.w, abs=ATOL)
            assert np.all(pi.sum(axis=1) <= ref.w + ATOL)
            assert np.all(pi.sum(axis=0) <= ref.w + ATOL)


# --------------------------------------------------------------------------- #
# Symmetries
# --------------------------------------------------------------------------- #


def test_rotation_equivariance():
    """Rotating every point leaves every cost(k) unchanged — the wrap-handling test."""
    rng = np.random.default_rng(5154)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        base = partial_ot_circle(x, y, debug=True).costs
        for off in rng.uniform(0, 1, 6):
            xr, yr = np.mod(x + off, 1.0), np.mod(y + off, 1.0)
            got = partial_ot_circle(xr, yr, debug=True).costs
            assert np.allclose(got, base, atol=ATOL), (
                f"rotation by {off!r} changed the profile ({family}):\n{base!r}\n{got!r}"
            )


def test_reflection_symmetry():
    rng = np.random.default_rng(5154)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        base = partial_ot_circle(x, y, debug=True).costs
        got = partial_ot_circle(np.mod(-x, 1.0), np.mod(-y, 1.0), debug=True).costs
        assert np.allclose(got, base, atol=ATOL)


def test_swapping_source_and_target_leaves_costs_unchanged():
    """d_{S^1} is symmetric, so exchanging mu and nu cannot change the profile."""
    rng = np.random.default_rng(5154)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        a = partial_ot_circle(x, y, debug=True).costs
        b = partial_ot_circle(y, x, debug=True).costs
        assert np.allclose(a, b, atol=ATOL)


# --------------------------------------------------------------------------- #
# Degenerate cases
# --------------------------------------------------------------------------- #


def test_tiny_arc_reduces_to_the_line():
    """All points in a short arc: a single PAWL call on the line is already optimal."""
    rng = np.random.default_rng(5155)
    for _ in range(25):
        x, y = make_instance("tiny_arc", rng)
        ref = partial_ot_circle(x, y, debug=True)
        line_costs, _, _ = line_profile_pawl(x, y)
        assert np.allclose(ref.costs, line_costs, atol=ATOL)


def test_single_atom_measures():
    rng = np.random.default_rng(5155)
    for _ in range(30):
        m = int(rng.integers(1, 15))
        pts = rng.uniform(0, 1, 1 + m)
        if len(np.unique(pts)) != 1 + m:
            continue
        x, y = pts[:1], pts[1:]
        ref = partial_ot_circle(x, y, debug=True)
        assert ref.K == 1
        assert ref.costs[0] == 0.0
        # cost(1) is the smallest geodesic distance from x to any y
        assert ref.costs[1] == pytest.approx(
            float(np.min(geodesic_distance(x[0], y))), abs=ATOL
        )
        assert np.allclose(ref.costs, partial_ot_circle_lp(x, y).costs, atol=ATOL)


def test_k_equals_one_is_the_closest_pair():
    """C_1^o is the minimum geodesic distance over all source-target pairs."""
    rng = np.random.default_rng(5155)
    for family in FAMILIES:
        for _ in range(10):
            x, y = make_instance(family, rng)
            ref = partial_ot_circle(x, y, debug=True)
            D = geodesic_distance(x.reshape(-1, 1), y.reshape(1, -1))
            assert ref.costs[1] == pytest.approx(float(D.min()), abs=ATOL)


# --------------------------------------------------------------------------- #
# Hypothesis-driven search
# --------------------------------------------------------------------------- #


@st.composite
def _circle_instance(draw, max_n=8, max_m=8, denom=1000):
    """Distinct multiples of 1/denom, split into a source and a target set."""
    n = draw(st.integers(min_value=1, max_value=max_n))
    m = draw(st.integers(min_value=1, max_value=max_m))
    vals = draw(
        st.lists(
            st.integers(min_value=0, max_value=denom - 1),
            min_size=n + m,
            max_size=n + m,
            unique=True,
        )
    )
    pts = np.array(vals, dtype=float) / denom
    return pts[:n], pts[n:]


@given(_circle_instance())
def test_hypothesis_reference_matches_lp_oracle(inst):
    x, y = inst
    ref = partial_ot_circle(x, y, debug=True)
    lp = partial_ot_circle_lp(x, y)
    assert np.allclose(ref.costs, lp.costs, atol=ATOL), (
        f"pawc_reference != LP oracle\nx={x.tolist()!r}\ny={y.tolist()!r}\n"
        f"ref={ref.costs!r}\nlp ={lp.costs!r}"
    )


@given(_circle_instance())
def test_hypothesis_reference_matches_cut_enumeration(inst):
    x, y = inst
    ref = partial_ot_circle(x, y, debug=True)
    base = partial_ot_circle_cuts(x, y)
    assert np.allclose(ref.costs, base.costs, atol=ATOL)


@given(_circle_instance(), st.integers(min_value=0, max_value=999))
def test_hypothesis_rotation_equivariance(inst, shift):
    x, y = inst
    off = shift / 1000.0
    a = partial_ot_circle(x, y, debug=True).costs
    b = partial_ot_circle(np.mod(x + off, 1.0), np.mod(y + off, 1.0), debug=True).costs
    assert np.allclose(a, b, atol=ATOL)


@given(_circle_instance())
def test_hypothesis_profile_convex_and_nested(inst):
    x, y = inst
    ref = partial_ot_circle(x, y, debug=True)
    d = np.diff(ref.costs)
    assert np.all(d >= -ATOL)
    if d.size >= 2:
        assert np.all(np.diff(d) >= -ATOL)
    for k in range(ref.K):
        assert set(ref.active_x[k]) <= set(ref.active_x[k + 1])
        assert set(ref.active_y[k]) <= set(ref.active_y[k + 1])
