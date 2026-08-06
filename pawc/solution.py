"""The common solver-contract container shared by every PAWC solver.

CLAUDE.md, "Solver contract": for every ``k`` in ``{0, ..., min(n, m)}`` a solver
returns ``cost(k)``, the active set ``A_k``, and — on demand — the transport plan
``pi^k``. All tests rely on this contract, so it lives in one place.

Plan recovery follows draft §9.1: cut the circle at a gap ``theta`` at which the
circular cost of ``A_k`` is realised as a line cost, keep the atoms activated by
iteration ``k``, and match the retained source and target atoms in increasing
unwrapped order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .circle import geodesic_distance, unroll_at

__all__ = ["PartialCircleSolution", "NestedActiveSets"]


class NestedActiveSets:
    """Lazy view of a nested family ``A_0 ⊂ A_1 ⊂ ... ⊂ A_K`` given its activation order.

    PAWC's active sets are nested (draft Thm. 4.1 / Cor. 4.1), so the whole family
    is determined by a single length-``K`` activation order: ``A_k`` is its first
    ``k`` entries. Materialising all ``K + 1`` index arrays would cost ``O(K^2)``
    time and memory, which at the ``N = 10^6`` benchmark size is prohibitive and
    would swamp the ``O(N)`` memory claim of draft Thm. 8.1. This class keeps the
    ``O(N)`` storage while behaving like the list the solver contract asks for.
    """

    __slots__ = ("_order", "_K")

    def __init__(self, order, K: int):
        self._order = np.asarray(order, dtype=np.int64)
        self._K = int(K)

    def __len__(self) -> int:
        return self._K + 1

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self[i] for i in range(*k.indices(self._K + 1))]
        k = int(k)
        if k < 0:
            k += self._K + 1
        if not 0 <= k <= self._K:
            raise IndexError(f"k must lie in [0, {self._K}], got {k}")
        return self._order[:k]

    def __iter__(self):
        for k in range(self._K + 1):
            yield self[k]

    def __repr__(self) -> str:
        return f"NestedActiveSets(K={self._K}, order={self._order!r})"


@dataclass
class PartialCircleSolution:
    """Complete partial-transport profile of one circular instance.

    Attributes
    ----------
    x, y : (n,) / (m,) float arrays
        The original (unsorted) source and target coordinates in ``[0, L)``.
    L : float
        Circumference.
    w : float
        Common atom weight, draft eq. (2).
    costs : (K+1,) float array
        ``costs[k] == C_k^o``, draft eq. (5). ``costs[0] == 0``.
    active_x, active_y : list of int arrays, length ``K+1``
        ``active_x[k]`` holds indices into ``x`` of the ``k`` active source atoms
        of an optimal ``A_k`` (draft eq. (6)); likewise ``active_y[k]`` into ``y``.
        Both have length ``k``.
    cuts : (K+1,) float array or None
        ``cuts[k]`` is a cut location whose unwrapping realises ``costs[k]`` as a
        line cost (draft Prop. 3.1). PAWC records the *same* simultaneous cut
        ``theta*`` for every ``k`` (draft Thm. 6.2); Baseline B records a possibly
        different argmin cut per ``k`` (Remark 6.1). ``None`` when the solver did
        not produce one, in which case explicit plans must be supplied.
    plans : dict of int -> (n, m) float array, optional
        Explicit plans, when the solver produced them directly (the LP oracle
        does). Entries take values in ``{0, w}``.
    solver : str
        Name of the producing solver, for error messages and benchmark CSVs.
    """

    x: np.ndarray
    y: np.ndarray
    L: float
    w: float
    costs: np.ndarray
    active_x: list
    active_y: list
    cuts: np.ndarray | None = None
    plans: dict = field(default_factory=dict)
    solver: str = "unknown"

    @property
    def n(self) -> int:
        return int(np.size(self.x))

    @property
    def m(self) -> int:
        return int(np.size(self.y))

    @property
    def K(self) -> int:
        return min(self.n, self.m)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        self.costs = np.asarray(self.costs, dtype=float).ravel()
        if self.cuts is not None:
            self.cuts = np.asarray(self.cuts, dtype=float).ravel()
        if self.costs.size != self.K + 1:
            raise ValueError(
                f"{self.solver}: costs has length {self.costs.size}, expected K+1 = {self.K + 1}"
            )
        if len(self.active_x) != self.K + 1 or len(self.active_y) != self.K + 1:
            raise ValueError(f"{self.solver}: active sets must be given for k = 0, ..., {self.K}")

    def plan(self, k: int) -> np.ndarray:
        """Return ``pi^k`` as an ``(n, m)`` array with entries in ``{0, w}``.

        Uses the explicit plan when the solver stored one; otherwise reconstructs
        it by the sorted matching at ``cuts[k]`` (draft §9.1).
        """
        k = int(k)
        if not 0 <= k <= self.K:
            raise ValueError(f"k must lie in [0, {self.K}], got {k}")
        if k in self.plans:
            return self.plans[k]

        pi = np.zeros((self.n, self.m), dtype=float)
        if k == 0:
            return pi
        if self.cuts is None:
            raise ValueError(
                f"{self.solver}: no explicit plan for k={k} and no cut recorded, so pi^k "
                "cannot be reconstructed"
            )

        ix = np.asarray(self.active_x[k], dtype=np.int64)
        iy = np.asarray(self.active_y[k], dtype=np.int64)
        theta = float(self.cuts[k])
        # Draft §9.1: after cutting at theta the active atoms are matched in
        # increasing unwrapped order.
        ix = ix[np.argsort(unroll_at(self.x[ix], theta, self.L), kind="stable")]
        iy = iy[np.argsort(unroll_at(self.y[iy], theta, self.L), kind="stable")]
        pi[ix, iy] = self.w
        return pi

    def plan_cost(self, k: int) -> float:
        """Geodesic cost of ``pi^k`` evaluated directly with ``d_{S^1}``.

        This deliberately re-evaluates the circular metric rather than the
        unwrapped one, so it is an independent check of ``costs[k]``.
        """
        pi = self.plan(k)
        if pi.size == 0:
            return 0.0
        nz = np.nonzero(pi)
        return float(np.sum(pi[nz] * geodesic_distance(self.x[nz[0]], self.y[nz[1]], self.L)))

    def check_feasible(self, k: int, atol: float = 1e-9) -> None:
        """Assert the marginal and total-mass constraints of draft eq. (4)."""
        pi = self.plan(k)
        if pi.size == 0:
            if k != 0:
                raise AssertionError(f"{self.solver}: empty plan for k={k}")
            return
        if np.any(pi < -atol):
            raise AssertionError(f"{self.solver}: negative mass in pi^{k}")
        row = pi.sum(axis=1)
        col = pi.sum(axis=0)
        if np.any(row > self.w + atol):
            raise AssertionError(f"{self.solver}: source marginal exceeds w in pi^{k}")
        if np.any(col > self.w + atol):
            raise AssertionError(f"{self.solver}: target marginal exceeds w in pi^{k}")
        total = float(pi.sum())
        if abs(total - k * self.w) > atol * max(1.0, k * self.w):
            raise AssertionError(
                f"{self.solver}: total mass of pi^{k} is {total}, expected {k * self.w}"
            )

    def active_set_z(self, k: int, su) -> set:
        """Active set as indices into the sorted union support ``su.z``.

        Convenience for the nestedness and free-gap invariant tests, which reason
        in ``z``-index space.
        """
        pos = {}
        for i in range(su.N):
            pos[(int(su.label[i]), su.original_index(i))] = i
        from .circle import SOURCE, TARGET

        out = {pos[(SOURCE, int(j))] for j in self.active_x[k]}
        out |= {pos[(TARGET, int(j))] for j in self.active_y[k]}
        return out
