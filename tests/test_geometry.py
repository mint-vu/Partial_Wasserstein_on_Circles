"""Phase-0/1 tests for pawc.circle.

Seeds used here: 0, 1, 2, 12345 (recorded per CLAUDE.md testing rule 4).
"""

import numpy as np
import pytest

from pawc.circle import (
    SOURCE,
    TARGET,
    clockwise_distance,
    gap_lengths,
    gap_midpoints,
    geodesic_distance,
    geodesic_distance_matrix,
    reflect,
    rotate,
    sorted_union,
    unroll_at,
    wrap,
)

ATOL = 1e-9


def test_wrap_range():
    rng = np.random.default_rng(0)
    u = rng.uniform(-10, 10, size=1000)
    w = wrap(u)
    assert np.all(w >= 0.0)
    assert np.all(w < 1.0)
    # wrap is idempotent and agrees with u modulo 1
    assert np.allclose(wrap(w), w, atol=ATOL)
    assert np.allclose(np.mod(u - w, 1.0), 0.0, atol=ATOL) or np.allclose(
        np.minimum(np.mod(u - w, 1.0), 1.0 - np.mod(u - w, 1.0)), 0.0, atol=ATOL
    )


def test_wrap_negative_epsilon_does_not_return_L():
    # np.mod(-1e-18, 1.0) rounds to 1.0 in float64; wrap must push it back to 0.
    assert wrap(-1e-18) == 0.0
    assert wrap(-1e-300) == 0.0


def test_geodesic_distance_basic():
    assert geodesic_distance(0.0, 0.5) == pytest.approx(0.5, abs=ATOL)
    assert geodesic_distance(0.0, 0.75) == pytest.approx(0.25, abs=ATOL)
    assert geodesic_distance(0.9, 0.1) == pytest.approx(0.2, abs=ATOL)
    assert geodesic_distance(0.3, 0.3) == pytest.approx(0.0, abs=ATOL)
    # never exceeds L / 2
    rng = np.random.default_rng(1)
    u, v = rng.random(10_000), rng.random(10_000)
    d = geodesic_distance(u, v)
    assert np.all(d <= 0.5 + ATOL)
    assert np.all(d >= 0.0)
    # symmetry
    assert np.allclose(d, geodesic_distance(v, u), atol=ATOL)


def test_geodesic_distance_triangle_inequality():
    rng = np.random.default_rng(2)
    u, v, t = rng.random(5000), rng.random(5000), rng.random(5000)
    assert np.all(
        geodesic_distance(u, v) <= geodesic_distance(u, t) + geodesic_distance(t, v) + ATOL
    )


def test_geodesic_distance_rotation_and_reflection_invariance():
    rng = np.random.default_rng(2)
    u, v = rng.random(1000), rng.random(1000)
    d = geodesic_distance(u, v)
    for off in rng.random(5):
        assert np.allclose(geodesic_distance(rotate(u, off), rotate(v, off)), d, atol=ATOL)
    assert np.allclose(geodesic_distance(reflect(u), reflect(v)), d, atol=ATOL)


def test_geodesic_distance_matrix_shape_and_values():
    x = np.array([0.0, 0.4])
    y = np.array([0.1, 0.5, 0.9])
    D = geodesic_distance_matrix(x, y)
    assert D.shape == (2, 3)
    assert D[0, 0] == pytest.approx(0.1, abs=ATOL)
    assert D[0, 2] == pytest.approx(0.1, abs=ATOL)
    assert D[1, 1] == pytest.approx(0.1, abs=ATOL)


def test_general_circumference():
    L = 7.0
    assert geodesic_distance(0.0, 6.0, L) == pytest.approx(1.0, abs=ATOL)
    assert np.all(wrap(np.linspace(-20, 20, 101), L) < L)


def test_sorted_union_structure():
    x = np.array([0.7, 0.1])
    y = np.array([0.4, 0.95, 0.2])
    su = sorted_union(x, y)
    assert su.n == 2 and su.m == 3 and su.N == 5 and su.K == 2
    assert np.allclose(su.z, [0.1, 0.2, 0.4, 0.7, 0.95], atol=ATOL)
    assert list(su.label) == [SOURCE, TARGET, TARGET, SOURCE, TARGET]
    assert list(su.sigma) == [1, -1, -1, 1, -1]
    # original_index round-trips to the unsorted input arrays
    for i in range(su.N):
        arr = x if su.label[i] == SOURCE else y
        assert arr[su.original_index(i)] == pytest.approx(su.z[i], abs=ATOL)


def test_sorted_union_rejects_coincident_points():
    with pytest.raises(ValueError, match="pairwise distinct"):
        sorted_union([0.1, 0.5], [0.5, 0.9])
    with pytest.raises(ValueError, match="pairwise distinct"):
        sorted_union([0.1, 0.1], [0.5])
    # wrapping-induced coincidence must also be caught
    with pytest.raises(ValueError, match="pairwise distinct"):
        sorted_union([0.25], [1.25])


def test_gap_lengths_sum_to_L():
    rng = np.random.default_rng(12345)
    for _ in range(200):
        z = np.sort(rng.random(rng.integers(2, 30)))
        ell = gap_lengths(z)
        assert ell.size == z.size
        assert np.all(ell > 0)
        assert ell.sum() == pytest.approx(1.0, abs=ATOL)
    # single point: the one gap is the whole circle
    assert gap_lengths(np.array([0.3])) == pytest.approx(np.array([1.0]), abs=ATOL)


def test_gap_midpoints_lie_strictly_inside_their_gap():
    rng = np.random.default_rng(12345)
    for _ in range(200):
        z = np.sort(rng.random(rng.integers(2, 20)))
        mids = gap_midpoints(z)
        ell = gap_lengths(z)
        for i, theta in enumerate(mids):
            # distance clockwise from z_i to theta is half the gap: strictly inside
            assert clockwise_distance(z[i], theta) == pytest.approx(ell[i] / 2, abs=ATOL)
            # no support point coincides with the cut
            assert np.min(np.abs(z - theta)) > 0.0


def test_unroll_preserves_cyclic_order():
    """Draft Prop. 3.1: cutting inside a gap unwraps the circle preserving order."""
    rng = np.random.default_rng(12345)
    for _ in range(200):
        z = np.sort(rng.random(int(rng.integers(3, 20))))
        for theta in gap_midpoints(z):
            u = unroll_at(z, theta)
            order = np.argsort(u)
            # the unrolled order is a cyclic rotation of 0..N-1
            start = int(order[0])
            expected = (start + np.arange(z.size)) % z.size
            assert np.array_equal(order, expected)
            assert np.all(u >= 0.0) and np.all(u < 1.0)


def test_unroll_does_not_depend_on_cut_position_within_a_gap():
    """Draft Prop. 3.1: only the gap matters, not theta inside it (up to a shift)."""
    rng = np.random.default_rng(12345)
    z = np.sort(rng.random(10))
    ell = gap_lengths(z)
    i = 3
    theta_a = wrap(z[i] + 0.25 * ell[i])
    theta_b = wrap(z[i] + 0.75 * ell[i])
    ua, ub = unroll_at(z, theta_a), unroll_at(z, theta_b)
    shift = 0.5 * ell[i]
    assert np.allclose(np.sort(ua) - np.sort(ub), shift, atol=ATOL)


def test_rotate_reflect_are_involutive_or_group_actions():
    rng = np.random.default_rng(0)
    u = rng.random(100)
    assert np.allclose(reflect(reflect(u)), u, atol=ATOL)
    a, b = 0.3, 0.45
    assert np.allclose(rotate(rotate(u, a), b), rotate(u, a + b), atol=ATOL)
