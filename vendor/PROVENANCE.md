# Vendored code provenance

## `partial.py` — fetched, not redistributed

- **Source**: https://github.com/rtavenar/partial_ot_1d
- **File**: `partial.py` (repository root)
- **Pinned commit**: `849e82e57b95d9de0db9884ba397d5b4d9d0b0e0`
- **Commit date**: 2025-02-04T16:54:17+01:00
- **Commit subject**: `Merge branch 'main' of https://github.com/rtavenar/partial_ot_1d`
- **SHA-256**: see `checksums.txt`

**The file is not committed to this repository.** As of 2026-08-06 the upstream
project carries no licence file, so its code is "all rights reserved" by default
and copying it into a third-party repository would be a redistribution we have no
permission for. Fetching it yourself from upstream is another matter, so:

```
python vendor/fetch_pawl.py          # download at the pinned commit, verify SHA-256
python vendor/fetch_pawl.py --check  # verify an existing copy
```

`vendor/partial.py` is listed in `.gitignore` so it cannot be committed by accident.
If upstream later adds a permissive licence, vendoring the file directly becomes an
option and this note should be revisited.

Do not float the pin to a branch name: the recorded checksum, and every baseline
number in the paper, refer to this exact tree.

This is the reference implementation of PAWL from

> Laetitia Chapel and Romain Tavenard. *One for all and all for one: Efficient
> computation of partial Wasserstein distances on the line*. ICLR 2025.
> https://openreview.net/forum?id=kzEPsHbJDv

**Do not modify the downloaded file.** PAWC's Baseline B (`pawc/baseline_cuts.py`)
calls `vendor.partial.partial_ot_1d` unchanged, so that the cut-enumeration baseline
is pinned to a known-good line solver.

## Notes

- `partial.py` imports `kneed` at module scope (for `partial_ot_1d_elbow`) and uses
  `numba` (`@njit(cache=True, fastmath=True)`) throughout. Both are therefore
  dependencies of the baselines and the test suite, and are pinned in
  `requirements.txt`.
- **Timing fairness.** PAWL is numba-JIT compiled; `pawc/pawc.py` is deliberately
  pure numpy + stdlib, so `pawc`-vs-`baseline_cuts` is *conservative* on the
  constant factor and `pawc_numba`-vs-`baseline_cuts` is the JIT-to-JIT comparison.
  This affects constants only, not the fitted log-log slopes (~1 for PAWC, ~2 for
  cut enumeration) that the complexity claim rests on. `demos/demo2_scaling.py`
  reports both.
- `fastmath=True` upstream permits reassociation of floating-point sums, so
  exact-arithmetic cross-checks (`fractions.Fraction`) cannot be routed through it.
  The exact tests use the LP oracle and an exact re-implementation of the line
  profile instead (`pawc/baseline_cuts.py::line_profile_dp`).
