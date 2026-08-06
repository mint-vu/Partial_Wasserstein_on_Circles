"""Demo 1 --- applying PAWC: the cost profile, the transport plan, and a walkthrough figure.

Runs the solver on one instance and shows the three things you get back:

1. **the profile** ``C_k`` for every transported cardinality ``k = 0, ..., K``,
2. **the plan** ``pi^k``, an ``(n, m)`` matrix with entries in ``{0, w}``,
3. **the active sets** ``A_0 subset A_1 subset ... subset A_K``, which are nested.

It then draws one circle panel per ``k``, so you can watch the matching grow, plus
a panel of the profile itself.  The default instance is the worked example from the
paper's appendix, so the figure reproduces that walkthrough step by step.

Usage
-----
    python demos/demo1_walkthrough.py
    python demos/demo1_walkthrough.py --random --n 6 --seed 3
    python demos/demo1_walkthrough.py --out figures/my_walkthrough.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pawc.circle import geodesic_distance  # noqa: E402
from pawc.elbow import find_elbow  # noqa: E402
from pawc.pawc import partial_ot_circle  # noqa: E402

#: The worked example of the paper's appendix: L = 12, four sources, four targets.
#: Its optimal profile is C = (0, 0.5, 1.5, 2.8, 6.4).
PAPER_L = 12.0
PAPER_X = np.array([0.4, 4.1, 4.5, 8.0])
PAPER_Y = np.array([1.6, 5.0, 5.4, 11.4])

SRC = "#1f6fb4"  # sources
TGT = "#e07b39"  # targets
CUT = "#8c8c8c"


def random_instance(n: int, m: int, L: float, seed: int):
    """``n`` sources and ``m`` targets, uniform on the circle, supports disjoint."""
    rng = np.random.default_rng(seed)
    while True:
        pts = rng.uniform(0.0, L, n + m)
        if np.unique(pts).size == n + m:
            return pts[:n], pts[n:]


# --------------------------------------------------------------------------- #
# text report
# --------------------------------------------------------------------------- #
def report(sol) -> None:
    L, w, K = sol.L, sol.w, sol.K
    print(f"circle of circumference L = {L:g}, atom weight w = {w:g}")
    print(f"n = {sol.n} sources, m = {sol.m} targets, so K = min(n, m) = {K}\n")
    print(f"  sources x = {np.array2string(sol.x, precision=3)}")
    print(f"  targets y = {np.array2string(sol.y, precision=3)}\n")

    # 1. the profile ------------------------------------------------------- #
    print("profile  C_k = optimal cost of transporting exactly k units:")
    print(f"  {'k':>3s}  {'C_k':>10s}  {'C_k - C_{k-1}':>14s}  {'C_k / (k w)':>12s}")
    for k in range(K + 1):
        inc = "" if k == 0 else f"{sol.costs[k] - sol.costs[k - 1]:14.4f}"
        avg = "" if k == 0 else f"{sol.costs[k] / (k * w):12.4f}"
        print(f"  {k:3d}  {sol.costs[k]:10.4f}  {inc:>14s}  {avg:>12s}")
    print("\n  C_k is nondecreasing and convex in k; the per-unit cost C_k/(kw) is the")
    print("  average geodesic distance travelled by the k transported atoms.")
    print(f"  elbow (pawc.elbow.find_elbow): k = {find_elbow(sol.costs)}")

    # a single number, if that is what you came for
    print(f"\n  full transport, k = K:      C_K = {sol.costs[K]:.6f}")
    try:
        from ot.lp import wasserstein_circle  # noqa: PLC0415

        if sol.n == sol.m:
            # POT works on the unit circle with probability weights 1/n, so its
            # value is C_K / (L * n * w).
            unit = np.asarray(
                wasserstein_circle(
                    (sol.x / L).reshape(-1, 1), (sol.y / L).reshape(-1, 1), p=1
                )
            ).ravel()[0]
            ref = float(unit) * L * (sol.n * w)
            print(f"  POT wasserstein_circle:     {ref:.6f}   (independent check)")
    except Exception:  # pragma: no cover - POT is optional for this demo
        pass

    # 2. the plan ---------------------------------------------------------- #
    print(f"\nthe simultaneous cut theta* = {sol.cuts[0]:.4f} is valid for every k at once,")
    print("so one solve gives every plan.  Plans, as matched (source, target) pairs:\n")
    for k in range(1, K + 1):
        pi = sol.plan(k)
        sol.check_feasible(k)
        rows, cols = np.nonzero(pi)
        pairs = ", ".join(
            f"x[{i}]={sol.x[i]:g} -> y[{j}]={sol.y[j]:g} "
            f"(d={geodesic_distance(sol.x[i], sol.y[j], L):.2f})"
            for i, j in zip(rows, cols)
        )
        print(f"  k={k}: {pairs}")
        assert abs(pi.sum() - k * w) < 1e-9
        assert abs(sol.plan_cost(k) - sol.costs[k]) < 1e-9

    print("\n  every plan is feasible (marginals <= w, total mass k*w) and its geodesic")
    print("  cost, re-evaluated from scratch with d_{S^1}, matches C_k.")

    # 3. nestedness -------------------------------------------------------- #
    print("\nactive sets are nested --- an atom, once transported, stays transported:")
    for k in range(K + 1):
        print(f"  A_{k}: sources {list(map(int, sol.active_x[k]))}, "
              f"targets {list(map(int, sol.active_y[k]))}")
    # Nestedness is a statement about the sets, not about who is matched to whom.
    def pairing(k):
        return {int(i): int(j) for i, j in zip(*np.nonzero(sol.plan(k)))}

    repaired = [
        (k, i, pairing(k - 1)[i], pairing(k)[i])
        for k in range(2, K + 1)
        for i in pairing(k - 1)
        if i in pairing(k) and pairing(k)[i] != pairing(k - 1)[i]
    ]
    print("\n  Note what is and is not nested.  The *sets* grow monotonically, which is")
    print("  what lets a single sweep answer every k.  The *pairing* inside them need")
    print("  not: an already-active atom may be handed a different partner, which is")
    print("  free, since re-pairing active atoms moves no new mass.")
    if repaired:
        for k, i, old, new in repaired:
            print(f"    at k={k}: x[{i}] switches partner y[{old}] -> y[{new}]")
    else:
        print("    (no re-pairing happens on this particular instance)")


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def _angles(v, L):
    """Circle coordinate -> plotting angle, 0 at 12 o'clock, increasing clockwise."""
    return np.pi / 2.0 - 2.0 * np.pi * np.asarray(v, dtype=float) / L


def _geodesic_arc(a, b, L, radius, npts=64):
    """Points along the *shorter* arc from ``a`` to ``b`` --- the geodesic itself."""
    delta = (b - a) % L
    signed = delta if delta <= L / 2.0 else delta - L  # shorter way round
    t = np.linspace(0.0, 1.0, npts)
    ang = _angles(a + signed * t, L)
    return radius * np.cos(ang), radius * np.sin(ang)


def draw_panel(ax, sol, k, src_colors):
    L = sol.L
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="black", lw=1.1, zorder=1)

    # the simultaneous cut
    ang = _angles(sol.cuts[0], L)
    ax.plot([0.80 * np.cos(ang), 1.16 * np.cos(ang)],
            [0.80 * np.sin(ang), 1.16 * np.sin(ang)],
            color=CUT, lw=1.2, ls=(0, (3, 2)), zorder=2)

    active_x = set(map(int, sol.active_x[k]))
    active_y = set(map(int, sol.active_y[k]))

    # matched pairs, drawn as the geodesic arc they travel along
    if k > 0:
        pi = sol.plan(k)
        for i, j in zip(*np.nonzero(pi)):
            col = src_colors[int(i)]
            ax.plot(*_geodesic_arc(sol.x[i], sol.y[j], L, 0.87), color=col, lw=2.6,
                    solid_capstyle="round", zorder=3)
            for v, r0 in ((sol.x[i], 0.87), (sol.y[j], 0.87)):
                a = _angles(v, L)
                ax.plot([r0 * np.cos(a), np.cos(a)], [r0 * np.sin(a), np.sin(a)],
                        color=col, lw=0.9, zorder=3)

    # atoms: filled = active (transported), hollow = inactive
    label_atoms = sol.n + sol.m <= 12  # beyond that the labels collide
    for arr, act, col, sym in ((sol.x, active_x, SRC, "x"), (sol.y, active_y, TGT, "y")):
        for idx, v in enumerate(arr):
            a = _angles(v, L)
            on = idx in act
            ax.plot(np.cos(a), np.sin(a), "o", ms=7.5, zorder=4,
                    color=col if on else "white",
                    markeredgecolor=col, markeredgewidth=1.6)
            if label_atoms:
                ax.text(1.17 * np.cos(a), 1.17 * np.sin(a), f"${sym}_{{{idx}}}$",
                        color=col, fontsize=8, ha="center", va="center", zorder=5)

    ax.set_title(f"$k={k}$,  $C_k={sol.costs[k]:g}$", fontsize=11)
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal")
    ax.axis("off")


def make_figure(sol, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    K = sol.K
    # One colour per *source atom*, assigned in activation order.  Sources are the
    # stable thing to key on: the active sets are nested, so a source that is
    # transported at k stays transported at k+1 -- but the partner it is matched to
    # may change (the paper's matching revocation), and colouring by source is what
    # makes that visible.
    cmap = plt.get_cmap("viridis")
    order = [int(i) for i in sol.active_x[K]]
    src_colors = {
        i: cmap(0.12 + 0.76 * t)
        for t, i in zip(np.linspace(0, 1, max(len(order), 1)), order)
    }

    n_panels = K + 2  # one per k, plus the profile
    ncols = int(min(max(np.ceil(np.sqrt(n_panels)), 2), 5))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for k in range(K + 1):
        draw_panel(axes[k], sol, k, src_colors)

    ax = axes[K + 1]
    ks = np.arange(K + 1)
    ax.plot(ks, sol.costs, "o-", color="#333333", lw=1.6, ms=5)
    ax.set_xlabel("transported cardinality $k$")
    ax.set_ylabel("$C_k$")
    ax.set_title("cost profile", fontsize=11)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    for extra in axes[K + 2:]:
        extra.axis("off")

    handles = [
        Line2D([], [], marker="o", ls="", color=SRC, ms=7, label="source, transported"),
        Line2D([], [], marker="o", ls="", color="white", markeredgecolor=SRC,
               markeredgewidth=1.6, ms=7, label="source, not transported"),
        Line2D([], [], marker="o", ls="", color=TGT, ms=7, label="target, transported"),
        Line2D([], [], marker="o", ls="", color="white", markeredgecolor=TGT,
               markeredgewidth=1.6, ms=7, label="target, not transported"),
        Line2D([], [], color=CUT, ls=(0, (3, 2)), lw=1.2,
               label=r"simultaneous cut $\theta^\ast$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(
        "PAWC: the optimal partial matching as the transported cardinality $k$ grows\n"
        "arcs are the geodesics the mass travels along, coloured by source atom; "
        "filled markers are transported",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"\nwrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", action="store_true",
                    help="random instance instead of the paper's worked example")
    ap.add_argument("--n", type=int, default=4, help="sources, with --random")
    ap.add_argument("--m", type=int, default=None, help="targets, with --random (default: --n)")
    ap.add_argument("--L", type=float, default=1.0, help="circumference, with --random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "figures" / "demo1_walkthrough.png")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    if args.random:
        m = args.n if args.m is None else args.m
        x, y = random_instance(args.n, m, args.L, args.seed)
        L = args.L
    else:
        x, y, L = PAPER_X, PAPER_Y, PAPER_L

    # --- this is the whole API -------------------------------------------- #
    sol = partial_ot_circle(x, y, L=L, w=1.0)
    # sol.costs[k]      the optimal cost at cardinality k
    # sol.plan(k)       the (n, m) transport plan
    # sol.active_x[k]   which sources are transported
    # ---------------------------------------------------------------------- #

    report(sol)
    if not args.no_figure:
        make_figure(sol, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
