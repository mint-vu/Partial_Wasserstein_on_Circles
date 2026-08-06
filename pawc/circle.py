"""Circle geometry helpers for PAWC.

Conventions (CLAUDE.md; draft §2 "Problem formulation"):

* The circle is ``S^1_L = R / L Z`` with circumference ``L`` (default ``L = 1``);
  points are represented by coordinates in ``[0, L)``.
* The ground cost is the geodesic distance
  ``d_{S^1}(u, v) = min(|u - v|, L - |u - v|)``  (draft eq. (1)).
* Supports are disjoint (``x`` and ``y`` share no location) and all atoms carry the
  same weight ``w``.
* ``z`` is the cyclically sorted union of ``x`` and ``y``; it is computed once and
  reused by every solver.
* The *original support gaps* are the ``N = n + m`` open arcs
  ``G_i = (z_i, z_{i+1})`` with lengths ``l_i = d_cw(z_i, z_{i+1})`` and
  ``sum_i l_i = L``  (draft §3.2).

Terminology follows the draft: *gap*, *cut*, *cell*, *candidate cell*, *free gap*,
*chain*, *minimal chain*.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "SOURCE",
    "TARGET",
    "wrap",
    "geodesic_distance",
    "geodesic_distance_matrix",
    "SortedUnion",
    "sorted_union",
    "gap_lengths",
    "gap_midpoints",
    "unroll_at",
    "rotate",
    "reflect",
]

#: Label of an atom of the source measure ``mu`` in the sorted union support.
#: Chosen to match the sign convention ``sigma_i = +1`` of draft eq. (23) and the
#: ``sorted_distrib_indicator == 0`` convention of ``vendor/partial.py``.
SOURCE = 0
#: Label of an atom of the target measure ``nu`` (draft ``sigma_i = -1``).
TARGET = 1


def wrap(u, L: float = 1.0):
    """Map coordinates into ``[0, L)`` by reduction modulo ``L``.

    ``np.mod`` can return exactly ``L`` for tiny negative inputs after rounding;
    those are pushed back to ``0`` so the postcondition ``0 <= wrap(u) < L`` holds.
    """
    out = np.mod(np.asarray(u, dtype=float), L)
    return np.where(out >= L, 0.0, out)


def geodesic_distance(u, v, L: float = 1.0):
    """Geodesic distance on ``S^1_L``, draft eq. (1).

    ``d(u, v) = min(|u - v|, L - |u - v|)``, elementwise / broadcasting.
    """
    diff = np.abs(np.asarray(u, dtype=float) - np.asarray(v, dtype=float))
    diff = np.mod(diff, L)
    return np.minimum(diff, L - diff)


def geodesic_distance_matrix(x, y, L: float = 1.0) -> np.ndarray:
    """``(n, m)`` matrix of geodesic distances ``d(x_i, y_j)``."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(1, -1)
    return geodesic_distance(x, y, L)


def clockwise_distance(u, v, L: float = 1.0):
    """Clockwise (increasing-coordinate) arc length ``d_cw`` from ``u`` to ``v``.

    Used for gap lengths in draft §3.2; unlike :func:`geodesic_distance` this is
    directed and lies in ``[0, L)``.
    """
    return np.mod(np.asarray(v, dtype=float) - np.asarray(u, dtype=float), L)


class SortedUnion(NamedTuple):
    """The cyclically sorted union support ``z`` and its bookkeeping.

    Attributes
    ----------
    z : (N,) float array
        Sorted union of ``x`` and ``y``, strictly increasing, all in ``[0, L)``.
    label : (N,) int array
        ``SOURCE`` or ``TARGET`` for each ``z_i``. This is the draft's
        ``sigma_i = +1 / -1`` in 0/1 encoding.
    index_in_own : (N,) int array
        Position of ``z_i`` within the *sorted* array of its own measure. So
        ``z[i] == np.sort(x)[index_in_own[i]]`` when ``label[i] == SOURCE``.
    order_x : (n,) int array
        ``np.argsort(x)``; ``x[order_x]`` is sorted.
    order_y : (m,) int array
        ``np.argsort(y)``.
    n, m : int
        Sizes of the source and target supports. ``N = n + m``, ``K = min(n, m)``.
    L : float
        Circumference.
    """

    z: np.ndarray
    label: np.ndarray
    index_in_own: np.ndarray
    order_x: np.ndarray
    order_y: np.ndarray
    n: int
    m: int
    L: float

    @property
    def N(self) -> int:
        return self.n + self.m

    @property
    def K(self) -> int:
        return min(self.n, self.m)

    @property
    def sigma(self) -> np.ndarray:
        """Draft eq. (23): ``+1`` for source atoms, ``-1`` for target atoms."""
        return np.where(self.label == SOURCE, 1, -1)

    def original_index(self, i: int) -> int:
        """Index of ``z_i`` in the *original* (unsorted) ``x`` or ``y`` array."""
        if self.label[i] == SOURCE:
            return int(self.order_x[self.index_in_own[i]])
        return int(self.order_y[self.index_in_own[i]])


def sorted_union(x, y, L: float = 1.0, *, check_disjoint: bool = True) -> SortedUnion:
    """Build the sorted union support ``z = x u y``; draft §3.2.

    Parameters
    ----------
    x, y : array_like
        Source and target coordinates. Wrapped into ``[0, L)`` first.
    L : float
        Circumference.
    check_disjoint : bool
        Enforce Assumption 2.1 (pairwise distinct support locations). The draft's
        constant-time chain machinery and the positivity of every original gap
        both rely on it, so violations are raised rather than silently patched.
    """
    x = wrap(np.asarray(x, dtype=float).ravel(), L)
    y = wrap(np.asarray(y, dtype=float).ravel(), L)
    n, m = x.size, y.size

    order_x = np.argsort(x, kind="stable")
    order_y = np.argsort(y, kind="stable")
    xs, ys = x[order_x], y[order_y]

    z = np.concatenate([xs, ys])
    label = np.concatenate(
        [np.full(n, SOURCE, dtype=np.int64), np.full(m, TARGET, dtype=np.int64)]
    )
    index_in_own = np.concatenate(
        [np.arange(n, dtype=np.int64), np.arange(m, dtype=np.int64)]
    )

    order = np.argsort(z, kind="stable")
    z, label, index_in_own = z[order], label[order], index_in_own[order]

    if check_disjoint and z.size > 1 and np.any(np.diff(z) <= 0.0):
        bad = int(np.argmin(np.diff(z)))
        raise ValueError(
            "Assumption 2.1 violated: support points must be pairwise distinct; "
            f"z[{bad}] == z[{bad + 1}] == {z[bad]!r}"
        )

    return SortedUnion(
        z=z,
        label=label,
        index_in_own=index_in_own,
        order_x=order_x,
        order_y=order_y,
        n=n,
        m=m,
        L=L,
    )


def gap_lengths(z, L: float = 1.0) -> np.ndarray:
    """Lengths ``l_i = d_cw(z_i, z_{i+1})`` of the ``N`` original support gaps.

    Draft §3.2. ``gap_lengths(z)[i]`` is the length of ``G_i = (z_i, z_{i+1})``,
    with ``G_{N-1}`` the wrap-around gap ``(z_{N-1}, z_0 + L)``. They sum to ``L``.
    """
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return np.zeros(0)
    if z.size == 1:
        return np.array([L])
    return np.diff(np.append(z, z[0] + L))


def gap_midpoints(z, L: float = 1.0) -> np.ndarray:
    """A canonical cut location inside each original support gap ``G_i``.

    Proposition 3.1 of the draft: the unwrapped line problem does not depend on
    where inside ``G_i`` the cut is placed, so the midpoint is a valid
    representative. Returned wrapped into ``[0, L)``.
    """
    z = np.asarray(z, dtype=float)
    return wrap(z + 0.5 * gap_lengths(z, L), L)


def unroll_at(values, theta: float, L: float = 1.0) -> np.ndarray:
    """Cut the circle at ``theta`` and unwrap onto the interval ``[0, L)``.

    Draft Proposition 3.1: a point ``u`` maps to ``(u - theta) mod L``. Cutting at
    a point strictly inside an original gap preserves the cyclic order of the
    support and turns each circular arc not containing ``theta`` into an ordinary
    line interval.
    """
    return wrap(np.asarray(values, dtype=float) - float(theta), L)


def rotate(values, offset: float, L: float = 1.0) -> np.ndarray:
    """Rigid rotation of the circle by ``offset`` (used by equivariance tests)."""
    return wrap(np.asarray(values, dtype=float) + float(offset), L)


def reflect(values, L: float = 1.0) -> np.ndarray:
    """Reflection ``u -> -u mod L`` (used by the reflection-symmetry tests)."""
    return wrap(-np.asarray(values, dtype=float), L)
