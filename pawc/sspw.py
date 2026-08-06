"""SSPW — sliced spherical partial Wasserstein, draft §10.

Implements Definition 10.1 / eq. (43) and its Monte-Carlo estimator eq. (46)
exactly as written, with no reordering of the projection or normalisation steps:

    SSPW(s; mu, nu) = int_{V_{d,2}} PW_o(s; P^U_# mu, P^U_# nu) dsigma(U)

    SSPW_hat(s)    = (1/L) sum_l PW_o(s; P^{U_l}_# mu, P^{U_l}_# nu)

Conventions, fixed from their definitions in the draft (§10.1):

* slices ``U`` are ``d x 2`` with orthonormal columns, drawn uniformly on the
  Stiefel manifold ``V_{d,2}`` by QR of a Gaussian matrix;
* the geodesic projection is ``Pi_U(x) = U U^T x / ||U^T x||`` (eq. (38)) and the
  solver consumes its coordinate representation through
  ``theta_U(x) = atan2((U^T x)_2, (U^T x)_1) mod 2 pi``  (eq. (40));
* slices therefore carry circumference ``L = 2 pi``, and no rescaling is applied,
  because ``z -> Uz`` is an isometry.

The last point is the one to watch when comparing against external code: POT
parametrises the circle as ``[0, 1)``, so its costs are this module's divided by
``2 pi``.  See ``experiments/c3_reduction.py``.
"""

from __future__ import annotations

import numpy as np

from .pawc import partial_ot_circle

__all__ = [
    "TWO_PI",
    "sample_slices",
    "project_to_circle",
    "slice_profiles",
    "sspw_profile",
    "sspw_at",
]

TWO_PI = 2.0 * np.pi


def sample_slices(d: int, n_slices: int, rng) -> np.ndarray:
    """``(n_slices, d, 2)`` uniform draws from the Stiefel manifold ``V_{d,2}``.

    QR of a ``d x 2`` standard Gaussian.  The sign convention of ``R``'s diagonal
    is not fixed here: it changes *which* ``U`` is produced but not the law, and
    every comparison in ``experiments/c3_reduction.py`` shares slices explicitly
    rather than relying on two samplers agreeing.
    """
    if d < 2:
        raise ValueError("spherical slicing needs d >= 2")
    G = rng.standard_normal((n_slices, d, 2))
    out = np.empty_like(G)
    for i in range(n_slices):
        q, _ = np.linalg.qr(G[i])
        out[i] = q
    return out


def project_to_circle(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """``theta_U`` of eq. (40) for one slice: ``(n, d) x (d, 2) -> (n,)`` in ``[0, 2pi)``."""
    P = np.asarray(X, dtype=float) @ np.asarray(U, dtype=float)  # (n, 2) = U^T x rows
    return np.mod(np.arctan2(P[:, 1], P[:, 0]), TWO_PI)


def _min_norm(X: np.ndarray, U: np.ndarray) -> float:
    """``min_i ||U^T x_i||`` — the distance to the codimension-two set where
    ``P^U`` is undefined (draft §10.1).  Null under sigma, but numerically the
    circular coordinate is unstable when this is tiny, so callers assert on it."""
    P = np.asarray(X, dtype=float) @ np.asarray(U, dtype=float)
    return float(np.min(np.linalg.norm(P, axis=1)))


def slice_profiles(X, Y, slices, w: float = 1.0, *, min_norm_tol: float = 1e-8,
                   collect_min_norm: bool = False):
    """Per-slice partial profiles: ``(n_slices, K+1)``.

    Row ``l`` is PAWC's profile ``{C_k^o}`` for the circular problem the slice
    induces.  Raises if a slice comes within ``min_norm_tol`` of the codimension-two
    set, rather than silently returning an unstable coordinate.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    slices = np.asarray(slices, dtype=float)
    K = min(len(X), len(Y))
    out = np.empty((len(slices), K + 1))
    mins = []
    for l, U in enumerate(slices):
        mn = min(_min_norm(X, U), _min_norm(Y, U))
        mins.append(mn)
        if mn < min_norm_tol:
            raise ValueError(
                f"slice {l} passes within {mn:.3e} of the set U^T x = 0, where P^U "
                "is undefined; resample this slice"
            )
        tx = project_to_circle(X, U)
        ty = project_to_circle(Y, U)
        out[l] = partial_ot_circle(tx, ty, L=TWO_PI, w=w).costs
    if collect_min_norm:
        return out, np.asarray(mins)
    return out


def sspw_profile(X, Y, slices, w: float = 1.0, **kw) -> np.ndarray:
    """``SSPW_hat`` at every breakpoint ``s = k w``: the slice average, eq. (46)."""
    return slice_profiles(X, Y, slices, w=w, **kw).mean(axis=0)


def sspw_at(profile: np.ndarray, s: float, w: float = 1.0) -> float:
    """Evaluate a breakpoint profile at arbitrary transported mass ``s``.

    Draft Prop. 9.1: the profile is linear between consecutive breakpoints, so
    non-integer ``s`` is a convex combination of its neighbours.  Applying this
    per slice and then averaging, or averaging and then interpolating, give the
    same value because both operations are linear -- ``experiments/c1_concentration.py``
    asserts that rather than relying on it.
    """
    K = len(profile) - 1
    t = s / w
    if not -1e-12 <= t <= K + 1e-12:
        raise ValueError(f"s={s} outside [0, {K * w}]")
    t = min(max(t, 0.0), float(K))
    k = int(np.floor(t))
    if k >= K:
        return float(profile[K])
    lam = t - k
    return float((1 - lam) * profile[k] + lam * profile[k + 1])
