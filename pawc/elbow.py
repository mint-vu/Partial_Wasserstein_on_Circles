"""Elbow-method wrapper, for API parity with PAWL's ``partial_ot_1d_elbow``.

Plan, Phase 5 (optional): downstream experiments — gradient flows, registration on
spherical or angular data — want to pick the transported cardinality automatically
rather than fixing ``k``. Since PAWC returns the *whole* profile ``C_0^o, ..., C_K^o``
in one run, the elbow is found by inspecting an array the solver already produced.

The knee locator is the same ``kneed`` package the vendored PAWL uses, so the two
APIs select ``k`` by the same rule.
"""

from __future__ import annotations

import numpy as np

from .pawc import partial_ot_circle
from .solution import PartialCircleSolution

__all__ = ["partial_ot_circle_elbow", "find_elbow"]


def find_elbow(costs) -> int:
    """Index of the elbow of the cumulative cost curve ``k -> C_k^o``.

    The profile is convex and increasing (draft Prop. 9.1), which is exactly the
    ``curve="convex", direction="increasing"`` case of ``kneed``. Returns ``K`` when
    no knee is detected, matching PAWL's fallback.
    """
    from kneed import KneeLocator

    costs = np.asarray(costs, dtype=float)
    K = costs.size - 1
    if K <= 1:
        return K
    knee = KneeLocator(
        x=np.arange(costs.size),
        y=costs,
        S=1.0,
        curve="convex",
        direction="increasing",
    ).knee
    return K if knee is None else int(knee)


def partial_ot_circle_elbow(
    x, y, L: float = 1.0, w: float = 1.0
) -> tuple[PartialCircleSolution, int]:
    """Solve on the circle and pick ``k`` by the elbow of the cost profile.

    Returns
    -------
    solution : PartialCircleSolution
        The full profile, exactly as :func:`pawc.pawc.partial_ot_circle` returns it.
    k_elbow : int
        The selected cardinality. ``solution.plan(k_elbow)`` is the chosen plan and
        ``solution.costs[k_elbow]`` its cost.

    Notes
    -----
    Unlike PAWL's wrapper this does not truncate anything: PAWC computes every
    cardinality in the same ``O(N log N)`` run, so the caller keeps the whole
    profile for free and the elbow is just an index into it.
    """
    sol = partial_ot_circle(x, y, L=L, w=w)
    return sol, find_elbow(sol.costs)
