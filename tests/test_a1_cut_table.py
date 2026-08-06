"""A1 Stage-1 checkpoint: the exposed cut table reduces to PAWC's profile.

The A1 experiment (``experiments/a1_cut_necessity.py``) is entirely a reduction of
the per-cut table ``T[r][k] = C_{k,r}^line`` that Baseline B computes. This module
gates the refactor that exposes it:

* ``min_r T[r][k]`` reproduces PAWC's ``C_k`` on the full 504-instance battery;
* the exact rational table agrees with the float one;
* row ``r`` really is the cut in ``G_r = (z_r, z_{r+1})`` clockwise (an off-by-one
  here would silently shift every cut trace, trap 3 of the plan);
* the per-cut activation orders correspond to the per-cut costs.

Seeds: 20260804 (shared battery), 424242 (rational), 31337 (indexing).
"""

from __future__ import annotations

import numpy as np
import pytest

from instances import FAMILIES, iter_instances, make_instance, make_rational_instance
from pawc.baseline_cuts import (
    all_cut_costs_exact,
    line_profile_at_cut,
    partial_ot_circle_cuts,
)
from pawc.circle import gap_lengths, gap_midpoints, sorted_union
from pawc.pawc import partial_ot_circle

ATOL = 1e-9
N_BATTERY = 504
SEED_BATTERY = 20260804


def test_stage1_table_reduces_to_pawc_profile():
    """min over cuts of the table == PAWC's C_k, on the whole battery."""
    for _family, x, y in iter_instances(N_BATTERY, SEED_BATTERY):
        b = partial_ot_circle_cuts(x, y, return_all=True)
        T = b.all_cut_costs  # (N, K+1)
        ref = partial_ot_circle(x, y)
        assert T.shape == (len(x) + len(y), ref.K + 1)
        assert np.allclose(T.min(axis=0), ref.costs, atol=ATOL), (
            f"table envelope != PAWC profile\nx={x!r}\ny={y!r}\n"
            f"min_r T = {T.min(axis=0)!r}\nC_k     = {ref.costs!r}"
        )


def test_stage1_no_cut_beats_the_envelope():
    """eps_k(r) >= 0 for every cut: Corollary 3.1, asserted on the table itself."""
    for _family, x, y in iter_instances(120, SEED_BATTERY):
        b = partial_ot_circle_cuts(x, y, return_all=True)
        T = b.all_cut_costs
        env = T.min(axis=0)
        assert np.all(T >= env[None, :] - ATOL), "a cut came in below the envelope"


def test_stage1_row_index_is_the_gap_index():
    """Row r is the cut inside G_r = (z_r, z_{r+1}) clockwise (trap 3)."""
    rng = np.random.default_rng(31337)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        su = sorted_union(x, y)
        b = partial_ot_circle_cuts(x, y, return_all=True)
        ell = gap_lengths(su.z, su.L)
        mids = gap_midpoints(su.z, su.L)
        assert np.allclose(b.cut_thetas, mids)
        for r in range(su.N):
            # the recorded cut sits half a gap clockwise of z_r
            assert float(np.mod(mids[r] - su.z[r], su.L)) == pytest.approx(
                ell[r] / 2, abs=1e-12
            )
            # and re-solving at a different point inside the same gap agrees
            theta_alt = float(np.mod(su.z[r] + 0.23 * ell[r], su.L))
            costs_alt, _, _ = line_profile_at_cut(x, y, theta_alt, su.L, 1.0, su.K)
            assert np.allclose(costs_alt, b.all_cut_costs[r], atol=ATOL), (
                f"row {r} does not correspond to gap G_{r} ({family})"
            )


def test_stage1_per_cut_orders_match_per_cut_costs():
    """The retained activation order for cut r reproduces that cut's costs."""
    from pawc.verify import line_w1_sorted

    rng = np.random.default_rng(31337)
    for family in FAMILIES:
        x, y = make_instance(family, rng)
        b = partial_ot_circle_cuts(x, y, return_all=True)
        su = sorted_union(x, y)
        assert len(b.all_cut_orders_x) == su.N
        for r in (0, su.N // 2, su.N - 1):
            ox, oy = b.all_cut_orders_x[r], b.all_cut_orders_y[r]
            theta = float(b.cut_thetas[r])
            xu = np.mod(x - theta, su.L)
            yu = np.mod(y - theta, su.L)
            for k in range(su.K + 1):
                got = line_w1_sorted(xu[ox[:k]], yu[oy[:k]], 1.0)
                assert got == pytest.approx(b.all_cut_costs[r][k], abs=ATOL), (
                    f"cut {r}, k={k}: order-derived cost {got} != table "
                    f"{b.all_cut_costs[r][k]} ({family})"
                )


def test_stage1_exact_table_matches_float_table():
    rng = np.random.default_rng(424242)
    for _ in range(30):
        xq, yq = make_rational_instance(rng)
        tab, _ = all_cut_costs_exact(xq, yq)
        xf = np.array([float(v) for v in xq])
        yf = np.array([float(v) for v in yq])
        b = partial_ot_circle_cuts(xf, yf, return_all=True)
        assert len(tab) == b.all_cut_costs.shape[0]
        for r, row in enumerate(tab):
            for k, c in enumerate(row):
                assert abs(float(c) - b.all_cut_costs[r][k]) <= ATOL


def test_stage1_exact_matchings_reproduce_exact_costs():
    rng = np.random.default_rng(424242)
    for _ in range(12):
        xq, yq = make_rational_instance(rng, max_size=6)
        tab, _, match = all_cut_costs_exact(xq, yq, return_matchings=True)
        thetas = all_cut_costs_exact(xq, yq)[1]
        for r, theta in enumerate(thetas):
            xu = [(v - theta) % 1 for v in xq]
            yu = [(v - theta) % 1 for v in yq]
            for k, pairs in enumerate(match[r]):
                assert len(pairs) == k
                cost = sum(abs(xu[i] - yu[j]) for i, j in pairs)
                assert cost == tab[r][k], (
                    f"cut {r}, k={k}: matching cost {cost} != table {tab[r][k]}"
                )
