"""Demo 2 --- what PAWC costs: O(N log N) on the circle, against the O(N^2 log N) alternative.

Three curves, all computing the *whole* partial profile (every ``k``) on instances
of total support size ``N``:

``pawc``
    This repository, on the circle.  ``O(N log N)`` (Theorem 8.1).  Pure numpy +
    stdlib.
``pawc_numba``
    The same algorithm, numba-JIT compiled.  Same asymptotics, smaller constant.
``pawl``
    The line solver of Chapel & Tavenard (ICLR 2025) on an instance of the same
    size.  ``O(N log N)``.  It solves a *different* problem --- there is no wrap ---
    and is included as the reference point for what a linear-time-up-to-logs partial
    solver costs.  The comparison to make is "PAWC on the circle costs about what
    PAWL costs on the line", not "PAWC beats PAWL".
``baseline_cuts``
    The naive circular solver: cut the circle at each of the ``N`` inter-atom gaps
    and run PAWL on the resulting line problem, then take the lower envelope
    (Corollary 3.1).  Exact, but ``O(N^2 log N)`` --- this is the cost PAWC removes.

Two dashed guide lines, ``N log N`` and ``N^2 log N``, are drawn through the largest
measured point of the relevant curve, so the slopes can be read off directly.

PAWL is numba-JIT compiled upstream, so ``pawc`` (pure numpy) against
``baseline_cuts`` (which calls PAWL) is *conservative* on the constant factor;
``pawc_numba`` is the JIT-to-JIT comparison.  Neither choice affects the fitted
slopes, which is what the complexity claim rests on.  JIT compilation is excluded
from every timing by an explicit warm-up.

Usage
-----
    python demos/demo2_scaling.py                 # ~2 minutes
    python demos/demo2_scaling.py --full          # to N = 10^6, ~30 minutes
    python demos/demo2_scaling.py --quick         # ~15 seconds, smoke test
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pawc.baseline_cuts import partial_ot_circle_cuts  # noqa: E402
from pawc.pawc import partial_ot_circle as pawc_solve  # noqa: E402

SIZES = {
    "quick": dict(fast=[100, 300, 1_000, 3_000], slow=[100, 300, 1_000]),
    "default": dict(fast=[100, 300, 1_000, 3_000, 10_000, 30_000, 100_000],
                    slow=[100, 300, 1_000, 3_000, 10_000]),
    "full": dict(fast=[100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000],
                 slow=[100, 300, 1_000, 3_000, 10_000, 30_000]),
}
SEED = 20260806

STYLE = {
    "pawc":          dict(color="#1f6fb4", marker="o", label=r"PAWC (circle), $O(N\log N)$"),
    "pawc_numba":    dict(color="#2ca25f", marker="s", label=r"PAWC + numba, $O(N\log N)$"),
    "pawl":          dict(color="#8c8c8c", marker="^", label=r"PAWL (line), $O(N\log N)$"),
    "baseline_cuts": dict(color="#d1495b", marker="D",
                          label=r"cut enumeration (circle), $O(N^2\log N)$"),
}


def make_instance(N: int, rng: np.random.Generator):
    """``N`` distinct uniform points on the unit circle, split evenly into x and y."""
    pts = rng.uniform(0, 1, N)
    while np.unique(pts).size != N:
        pts = rng.uniform(0, 1, N)
    return pts[: N // 2], pts[N // 2 :]


def pawl_solve(x, y):
    """PAWL on the *line*, on the same coordinates.  Same N, no wrap-around."""
    from vendor.partial import partial_ot_1d  # noqa: PLC0415

    xs = np.ascontiguousarray(np.sort(x))
    ys = np.ascontiguousarray(np.sort(y))
    return partial_ot_1d(xs, ys, max_iter=min(xs.size, ys.size), p=1)


def time_solver(fn, x, y, reps: int):
    fn(x, y)  # untimed warm-up: JIT, caches, page faults
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(x, y)
        times.append(time.perf_counter() - t0)
    return statistics.fmean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


def fit_slope(sizes, times):
    """Least-squares slope of log(time) on log(N) --- the empirical exponent."""
    s = np.asarray(sizes, float)
    t = np.asarray(times, float)
    ok = (s > 0) & (t > 0)
    return float(np.polyfit(np.log(s[ok]), np.log(t[ok]), 1)[0]) if ok.sum() > 1 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="smoke test, ~15 s")
    g.add_argument("--full", action="store_true", help="to N = 10^6, ~30 min")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-numba", action="store_true", help="skip the JIT curve")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "figures" / "demo2_scaling.png")
    ap.add_argument("--csv", type=Path, default=_REPO_ROOT / "figures" / "demo2_scaling.csv")
    args = ap.parse_args()

    which = "quick" if args.quick else "full" if args.full else "default"
    fast_sizes, slow_sizes = SIZES[which]["fast"], SIZES[which]["slow"]

    solvers = [("pawc", pawc_solve, fast_sizes)]
    if not args.no_numba:
        try:
            from pawc.pawc_numba import partial_ot_circle as numba_solve  # noqa: PLC0415
            from pawc.pawc_numba import warmup as numba_warmup  # noqa: PLC0415

            print("compiling the numba kernel (not timed) ...")
            numba_warmup()
            solvers.append(("pawc_numba", numba_solve, fast_sizes))
        except ImportError:
            print("numba unavailable, skipping the JIT curve")
    try:
        pawl_solve(*make_instance(64, np.random.default_rng(0)))
        solvers.append(("pawl", pawl_solve, fast_sizes))
        solvers.append(("baseline_cuts", partial_ot_circle_cuts, slow_sizes))
    except ImportError:
        print(
            "\nThe PAWL reference implementation is not present, so the PAWL and\n"
            "cut-enumeration curves are skipped -- you will get the PAWC curves only.\n"
            "It is not redistributed with this repository (upstream carries no licence\n"
            "file); fetch it at its pinned commit with\n"
            "    python vendor/fetch_pawl.py\n"
        )

    results: dict[str, tuple[list[int], list[float], list[float]]] = {}
    rows = []
    print(f"\n{'solver':16s} {'N':>9s} {'mean (s)':>11s} {'std (s)':>10s}")
    print("-" * 50)
    for name, fn, sizes in solvers:
        ns, means, stds = [], [], []
        for N in sizes:
            x, y = make_instance(N, np.random.default_rng(SEED + N))
            reps = max(2, args.reps // 2) if N >= 30_000 else args.reps
            mean, std = time_solver(fn, x, y, reps)
            ns.append(N); means.append(mean); stds.append(std)
            rows.append(dict(solver=name, N=N, mean_s=mean, std_s=std, reps=reps))
            print(f"{name:16s} {N:9d} {mean:11.4f} {std:10.4f}")
        results[name] = (ns, means, stds)

    print("\nfitted slope of log(time) on log(N)  --- the empirical exponent:")
    for name, (ns, means, _) in results.items():
        expect = 2.0 if name == "baseline_cuts" else 1.0
        print(f"  {name:16s} {fit_slope(ns, means):5.2f}   (theory: ~{expect:.0f}, "
              f"up to the log)")
    if "pawc" in results and "baseline_cuts" in results:
        pawc_ns, pawc_t, _ = results["pawc"]
        base_ns, base_t, _ = results["baseline_cuts"]
        shared = [n for n in pawc_ns if n in base_ns]
        if shared:
            n = shared[-1]
            speedup = base_t[base_ns.index(n)] / pawc_t[pawc_ns.index(n)]
            print(f"\n  at the largest shared size N = {n}: PAWC is {speedup:.0f}x faster "
                  f"than cut enumeration,")
            print("  and the gap widens by a further factor of ~N at every decade.")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["solver", "N", "mean_s", "std_s", "reps"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.csv}")

    # ------------------------------------------------------------------ plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for name, (ns, means, stds) in results.items():
        st = dict(STYLE[name])
        st["label"] += f"  [fit {fit_slope(ns, means):.2f}]"
        ax.errorbar(ns, means, yerr=stds, lw=1.7, ms=5.5, capsize=2.5, **st)

    # guide lines through the largest measured point of each curve
    grid = np.logspace(np.log10(min(fast_sizes)), np.log10(max(fast_sizes)), 100)
    if "pawc" in results:
        ns, means, _ = results["pawc"]
        c = means[-1] / (ns[-1] * np.log(ns[-1]))
        ax.plot(grid, c * grid * np.log(grid), ls=":", color="#1f6fb4", lw=1.3,
                label=r"$\propto N\log N$")
    if "baseline_cuts" in results:
        ns, means, _ = results["baseline_cuts"]
        c = means[-1] / (ns[-1] ** 2 * np.log(ns[-1]))
        ax.plot(grid, c * grid**2 * np.log(grid), ls=":", color="#d1495b", lw=1.3,
                label=r"$\propto N^2\log N$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("support size $N = n + m$")
    ax.set_ylabel("time for the full profile (s)")
    ax.set_title("Cost of the whole partial profile, every $k$ at once\n"
                 "$[$fit$]$ is the least-squares slope of $\\log$ time on $\\log N$",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.005,
             f"{platform.platform()} | numpy {np.__version__} | seed {SEED} | "
             f"{args.reps} reps, mean $\\pm$ std",
             fontsize=6.5, color="#666666")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
