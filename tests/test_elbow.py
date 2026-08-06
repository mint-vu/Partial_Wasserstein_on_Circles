"""Phase-5: the elbow wrapper, for API parity with PAWL's ``partial_ot_1d_elbow``.

Seeds used, all fixed: 808, 809.
"""

from __future__ import annotations

import numpy as np
import pytest

from instances import make_instance
from pawc.elbow import find_elbow, partial_ot_circle_elbow
from pawc.pawc import partial_ot_circle

ATOL = 1e-9


def _instance_with_a_clear_elbow(rng, n_signal=9, n_noise=4, eps=1e-3):
    """``n_signal`` tightly matched pairs plus ``n_noise`` pairs placed far apart.

    The cost profile then rises very slowly for the first ``n_signal`` steps and
    steeply afterwards, so the knee should land at ``n_signal``.
    """
    base = rng.uniform(0.0, 0.45, n_signal)
    x_noise = rng.uniform(0.50, 0.60, n_noise)
    y_noise = rng.uniform(0.85, 0.95, n_noise)
    x = np.concatenate([base, x_noise])
    y = np.concatenate([np.mod(base + eps, 1.0), y_noise])
    assert len(np.unique(np.concatenate([x, y]))) == x.size + y.size
    return x, y


def test_elbow_finds_the_signal_size():
    rng = np.random.default_rng(808)
    for _ in range(10):
        x, y = _instance_with_a_clear_elbow(rng)
        sol, k = partial_ot_circle_elbow(x, y)
        assert k == 9, f"elbow at k={k}, expected 9; costs={np.round(sol.costs, 6)!r}"


def test_elbow_returns_the_full_profile_and_a_usable_plan():
    """PAWC computes every k anyway, so the wrapper must not truncate anything."""
    rng = np.random.default_rng(808)
    x, y = _instance_with_a_clear_elbow(rng)
    sol, k = partial_ot_circle_elbow(x, y)
    assert sol.costs.size == sol.K + 1
    assert 0 <= k <= sol.K
    sol.check_feasible(k, atol=ATOL)
    assert sol.plan_cost(k) == pytest.approx(sol.costs[k], abs=ATOL)
    assert np.allclose(sol.costs, partial_ot_circle(x, y).costs, atol=0.0)


def test_elbow_falls_back_to_K_when_no_knee_exists():
    """A straight cost profile has no knee; PAWL's fallback is K, and so is ours."""
    # Equally spaced pairs at a constant offset: every marginal is identical.
    k = 12
    base = np.arange(k) / k
    x = base
    y = base + 0.5 / k
    sol, ke = partial_ot_circle_elbow(x, y)
    assert ke == sol.K
    assert np.allclose(np.diff(sol.costs), np.diff(sol.costs)[0], atol=ATOL)


def test_find_elbow_handles_degenerate_profiles():
    assert find_elbow(np.array([0.0])) == 0
    assert find_elbow(np.array([0.0, 1.0])) == 1


def test_elbow_is_rotation_equivariant():
    rng = np.random.default_rng(809)
    for _ in range(6):
        x, y = make_instance("clustered", rng)
        _, k0 = partial_ot_circle_elbow(x, y)
        off = float(rng.uniform(0, 1))
        _, k1 = partial_ot_circle_elbow(np.mod(x + off, 1.0), np.mod(y + off, 1.0))
        assert k0 == k1
