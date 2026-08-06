"""Shared instance generators for the PAWC test battery.

Every generator is seeded explicitly; the seeds actually used are recorded in the
test modules (CLAUDE.md testing rule 4). All generators guarantee Assumption 2.1:
supports are pairwise distinct and disjoint.

Families (plan, Phase 1 gate 1):

``uniform``          points i.i.d. uniform on the circle
``clustered``        a few tight clusters — stresses long free gaps and chains
``near_antipodal``   source/target pairs at ~L/2, perturbed off the exact tie
``tiny_arc``         everything inside a short arc; the circle should reduce to the line
``unbalanced``       n and m far apart
``perturbed_grid``   near-regular spacing — many near-equal candidate marginals
``interleaved``      strict x,y,x,y,... alternation around the circle
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

FAMILIES = (
    "uniform",
    "clustered",
    "near_antipodal",
    "tiny_arc",
    "unbalanced",
    "perturbed_grid",
    "interleaved",
)

_MIN_SEP = 1e-6  # enforced minimum separation so Assumption 2.1 holds robustly


def _split(points: np.ndarray, n: int, L: float) -> tuple[np.ndarray, np.ndarray]:
    return np.mod(points[:n], L), np.mod(points[n:], L)


def _dedupe(points: np.ndarray, L: float, rng: np.random.Generator) -> np.ndarray:
    """Nudge points apart until all cyclic separations exceed ``_MIN_SEP``."""
    pts = np.mod(np.asarray(points, dtype=float), L)
    for _ in range(64):
        order = np.argsort(pts)
        s = pts[order]
        gaps = np.diff(np.append(s, s[0] + L))
        bad = gaps < _MIN_SEP * L
        if not bad.any():
            return pts
        pts[order[np.flatnonzero(bad)]] += rng.uniform(
            2 * _MIN_SEP * L, 10 * _MIN_SEP * L, size=int(bad.sum())
        )
        pts = np.mod(pts, L)
    raise RuntimeError("could not separate support points; tighten the generator")


def make_instance(family: str, rng: np.random.Generator, L: float = 1.0):
    """Draw one ``(x, y)`` instance from the named family."""
    if family == "uniform":
        n, m = int(rng.integers(2, 31)), int(rng.integers(2, 31))
        pts = rng.uniform(0, L, n + m)

    elif family == "clustered":
        n, m = int(rng.integers(3, 25)), int(rng.integers(3, 25))
        n_clusters = int(rng.integers(2, 5))
        centers = rng.uniform(0, L, n_clusters)
        width = L * rng.uniform(1e-3, 2e-2)
        which = rng.integers(0, n_clusters, n + m)
        pts = centers[which] + rng.normal(0, width, n + m)

    elif family == "near_antipodal":
        # x_i and y_i sit almost L/2 apart: the geodesic is nearly ambiguous but
        # never exactly tied (the draft excludes exact antipodal ties, §11).
        k = int(rng.integers(2, 20))
        base = rng.uniform(0, L, k)
        jitter = rng.uniform(1e-4, 5e-3, k) * rng.choice([-1.0, 1.0], k) * L
        pts = np.concatenate([base, base + L / 2 + jitter])
        n = k

    elif family == "tiny_arc":
        n, m = int(rng.integers(2, 20)), int(rng.integers(2, 20))
        arc = L * rng.uniform(1e-4, 5e-3)
        start = rng.uniform(0, L)
        pts = start + rng.uniform(0, arc, n + m)

    elif family == "unbalanced":
        small = int(rng.integers(1, 5))
        large = int(rng.integers(20, 41))
        n, m = (small, large) if rng.random() < 0.5 else (large, small)
        pts = rng.uniform(0, L, n + m)

    elif family == "perturbed_grid":
        n, m = int(rng.integers(3, 25)), int(rng.integers(3, 25))
        N = n + m
        pts = np.arange(N) * (L / N) + rng.normal(0, L / (20 * N), N)
        pts = pts[rng.permutation(N)]

    elif family == "interleaved":
        k = int(rng.integers(2, 25))
        step = L / (2 * k)
        base = np.arange(2 * k) * step + rng.uniform(0, step * 0.4, 2 * k)
        pts = np.concatenate([base[0::2], base[1::2]])
        n = k

    else:
        raise ValueError(f"unknown family {family!r}")

    if family in ("near_antipodal", "interleaved"):
        pts = _dedupe(pts, L, rng)
        return _split(pts, n, L)

    pts = _dedupe(pts, L, rng)
    return _split(pts, n, L)


def iter_instances(n_instances: int, seed: int, L: float = 1.0):
    """Yield ``(family, x, y)`` cycling through :data:`FAMILIES`."""
    rng = np.random.default_rng(seed)
    for i in range(n_instances):
        family = FAMILIES[i % len(FAMILIES)]
        x, y = make_instance(family, rng, L)
        yield family, x, y


def make_rational_instance(rng: np.random.Generator, denom: int = 2003, max_size: int = 8):
    """Draw an instance with exact ``Fraction`` coordinates on a circle of length 1.

    Coordinates are distinct multiples of ``1/denom``, so gap midpoints and every
    downstream quantity stay exactly rational.
    """
    n = int(rng.integers(2, max_size + 1))
    m = int(rng.integers(2, max_size + 1))
    vals = rng.permutation(denom)[: n + m]
    x = [Fraction(int(v), denom) for v in vals[:n]]
    y = [Fraction(int(v), denom) for v in vals[n:]]
    return x, y
