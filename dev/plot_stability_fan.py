"""
plot_stability_fan.py
=====================
Create fan plots from stability_trajectory.csv produced by benchmark_reconstruction.py.

Each fan plot shows how a parameter evolves across stability iterations:
  - x-axis  : iteration (0 = seed from ACR, 1..N = simulate→reconstruct cycles)
  - y-axis  : parameter value
  - iteration 0 : single seed-value diamond marker
  - iterations 1+ : quantile fan bands coloured by density (outer bands lighter,
                    inner bands darker), with a median line connecting iterations

Usage
-----
  python dev/plot_stability_fan.py \\
      --trajectory output/stability_trajectory.csv \\
      --output     output/stability_fans/ \\
      [--methods   JOINT_ORIGINAL_DIST FLOW_THRESH_DIST] \\
      [--genes     geneA geneB] \\
      [--parameters gain_rate loss_rate gains losses prevalence]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Quantile band definitions
# Each tuple: (lower quantile, upper quantile, alpha fill opacity)
# Ordered outermost→innermost so inner bands paint over outer bands.
# ---------------------------------------------------------------------------
BANDS = [
    (0.00, 1.00, 0.06),
    (0.05, 0.95, 0.10),
    (0.10, 0.90, 0.15),
    (0.20, 0.80, 0.22),
    (0.30, 0.70, 0.30),
]
BAND_COLOR  = "#2471A3"   # steel-blue family
MEDIAN_COLOR = "#1A5276"
SEED_COLOR   = "#922B21"  # dark red for iteration-0 diamond

PARAM_LABELS = {
    "gain_rate":        "Gain rate  (gains / gain_subsize)",
    "loss_rate":        "Loss rate  (losses / loss_subsize)",
    "gains":            "Gains (count)",
    "losses":           "Losses (count)",
    "prevalence":       "Prevalence (fraction of tips with trait = 1)",
    "parsimony":        "Fitch parsimony (min transitions)",
    "mpd":              "MPD (mean pairwise phylogenetic distance)",
    "expected_pi1":     "Expected stationary prevalence  \u03c0\u2081 = q\u2080\u2081/(q\u2080\u2081+q\u2081\u2080)",
    "prevalence_drift": "Prevalence drift  (\u03c0\u2081 \u2212 observed prevalence)",
}

DEFAULT_PARAMS = [
    "gain_rate", "loss_rate", "gains", "losses", "prevalence",
    "parsimony", "mpd", "expected_pi1", "prevalence_drift",
]


# ---------------------------------------------------------------------------
# Core drawing function
# ---------------------------------------------------------------------------

def _draw_fan(ax: plt.Axes, tdf: pd.DataFrame, param: str,
              ylim: tuple | None = None) -> None:
    """
    Draw a fan plot for a single (gene, parameter) on the given axes.

    tdf  : trajectory DataFrame already filtered to one (method, gene).
    ylim : optional (lo, hi) to override automatic axis limits.
    """
    iters_all = sorted(tdf["iteration"].unique())
    iters_sim = [it for it in iters_all if it > 0]

    # ── Iteration-0 seed point ──────────────────────────────────────────────
    seed_rows = tdf[tdf["iteration"] == 0]
    if not seed_rows.empty:
        seed_val = float(seed_rows[param].iloc[0])
        ax.plot(0, seed_val, "D", color=SEED_COLOR, ms=8, zorder=6,
                label="Seed (iter 0)")
    else:
        seed_val = None

    if not iters_sim:
        return

    # ── Quantile bands across simulation iterations ─────────────────────────
    q_data = {}
    for it in iters_sim:
        vals = tdf[tdf["iteration"] == it][param].dropna()
        q_data[it] = vals

    # If a seed exists, anchor all band edges to seed_val at x=0 so the fan
    # visually expands from a single point.
    has_seed = seed_val is not None
    x_bands = ([0] if has_seed else []) + list(iters_sim)

    for q_lo, q_hi, alpha in BANDS:
        lo_vals = [float(q_data[it].quantile(q_lo)) if len(q_data[it]) else np.nan
                   for it in iters_sim]
        hi_vals = [float(q_data[it].quantile(q_hi)) if len(q_data[it]) else np.nan
                   for it in iters_sim]
        if has_seed:
            lo_vals = [seed_val] + lo_vals
            hi_vals = [seed_val] + hi_vals
        ax.fill_between(x_bands, np.array(lo_vals), np.array(hi_vals),  # type: ignore[arg-type]
                        alpha=alpha, color=BAND_COLOR, linewidth=0)

    # ── Median line connecting all iterations ───────────────────────────────
    medians_sim = [float(q_data[it].median()) if len(q_data[it]) else np.nan
                   for it in iters_sim]
    medians_all = ([seed_val] if has_seed else []) + medians_sim
    ax.plot(x_bands, medians_all, "o-", color=MEDIAN_COLOR, lw=1.8, ms=5,
            zorder=5, label="Median")

    ax.set_xticks(iters_all)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel(PARAM_LABELS.get(param, param), fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif param in ("prevalence", "expected_pi1"):
        ax.set_ylim(0.0, 1.0)
    elif param == "prevalence_drift":
        ax.set_ylim(-1.0, 1.0)
        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
    elif param in ("parsimony", "mpd"):
        ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Per-gene page
# ---------------------------------------------------------------------------

def _gene_page(fig: plt.Figure, tdf: pd.DataFrame,
               params: list[str], method: str, gene: str,
               ylims: dict[str, tuple] | None = None) -> None:
    """Populate a figure with one row of fan plots, one panel per parameter."""
    n = len(params)
    axes = fig.subplots(1, n)
    if n == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        if param not in tdf.columns:
            ax.set_visible(False)
            continue
        _draw_fan(ax, tdf, param, ylim=(ylims or {}).get(param))
        ax.set_title(param, fontsize=10, fontweight="bold")

    fig.suptitle(f"{method}  —  {gene}", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fan plots from stability_trajectory.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--trajectory", required=True, metavar="FILE",
                   help="stability_trajectory.csv produced by benchmark_reconstruction.py")
    p.add_argument("--output", required=True, metavar="DIR",
                   help="Output directory; one PDF per method is written here")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of methods to plot (default: all)")
    p.add_argument("--genes", nargs="+", default=None,
                   help="Subset of genes to plot (default: all)")
    p.add_argument("--parameters", nargs="+", default=DEFAULT_PARAMS,
                   help=f"Parameters to plot (default: {' '.join(DEFAULT_PARAMS)})")
    p.add_argument("--fig_width", default=4.0, type=float,
                   help="Width per panel in inches (default: 4.0)")
    p.add_argument("--fig_height", default=4.5, type=float,
                   help="Figure height in inches (default: 4.5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    traj_path = Path(args.trajectory)
    if not traj_path.exists():
        print(f"[ERROR] trajectory file not found: {traj_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(traj_path)
    required = {"method", "gene", "iteration"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] trajectory file missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods or sorted(df["method"].unique())
    genes   = args.genes   or sorted(df["gene"].unique())
    params  = [p for p in args.parameters if p in df.columns]

    if not params:
        print("[ERROR] None of the requested parameters are present in the trajectory file.",
              file=sys.stderr)
        sys.exit(1)

    missing_params = [p for p in args.parameters if p not in df.columns]
    if missing_params:
        print(f"[WARN] Parameters not found in trajectory (skipped): {missing_params}",
              file=sys.stderr)

    n_panels = len(params)
    fig_w = args.fig_width * n_panels
    fig_h = args.fig_height

    total_pages = 0
    for method in methods:
        mdf = df[df["method"] == method]
        if mdf.empty:
            print(f"  [SKIP] {method}: no data in trajectory file.")
            continue

        # Precompute global y-limits for parsimony and mpd so all gene pages
        # share a consistent scale (standardised across genes for this method).
        ylims: dict[str, tuple] = {}
        for param in ("parsimony", "mpd"):
            if param in mdf.columns:
                col_max = mdf[param].replace([np.inf, -np.inf], np.nan).dropna()
                if not col_max.empty:
                    global_max = float(col_max.max())
                    ylims[param] = (0.0, global_max * 1.1)

        pdf_path = out_dir / f"{method}_stability_fan.pdf"
        with PdfPages(pdf_path) as pdf:
            pages = 0
            for gene in genes:
                gdf = mdf[mdf["gene"] == gene]
                if gdf.empty:
                    continue

                fig = plt.figure(figsize=(fig_w, fig_h))
                _gene_page(fig, gdf, params, method, gene, ylims=ylims)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                pages += 1

            if pages == 0:
                print(f"  [SKIP] {method}: no matching genes.")
                pdf_path.unlink(missing_ok=True)
                continue

        print(f"  [{method}] {pages} gene pages → {pdf_path}")
        total_pages += pages

    print(f"\nDone.  {total_pages} total pages written to {out_dir}/")


if __name__ == "__main__":
    main()
