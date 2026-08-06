"""PAWC — PArtial Wasserstein on the Circle.

Exact partial 1-Wasserstein transport between two uniformly weighted empirical
measures on the circle S^1_L with the geodesic ground cost, for every admissible
transported cardinality k = 0, ..., min(n, m).

Implements the theory of `docs/pawc_draft.pdf` (Kolouri, 2026), extending PAWL
(Chapel & Tavenard, ICLR 2025; `docs/pawl_iclr2025.pdf`) from the line to the
circle.

Modules
-------
circle
    Geometry helpers: wrapping, geodesic distance, sorted union support, gaps.
baseline_lp
    Baseline A — exact LP / min-cost-flow oracle. Ground truth for small n.
baseline_cuts
    Baseline B — cut enumeration over all N original support gaps, each solved by
    the vendored PAWL. Realises the exact cut envelope, draft Corollary 3.1.
    O(N^2 log N).
pawc_reference
    Readable PAWC with the draft's invariant assertions turned on.
pawc
    Optimised PAWC, draft Algorithm 1. O(N log N) time, O(N) memory.

Solver contract
---------------
Every solver returns, for k = 0, ..., K = min(n, m), the cost C_k^o, an optimal
active set A_k, and (on demand) a transport plan pi^k.
"""

__version__ = "0.1.0.dev0"
