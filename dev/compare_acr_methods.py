#!/usr/bin/env python
"""
compare_acr_methods.py
======================
Compares JOINT vs FLOW ancestral reconstruction methods from pastmlout_marginal.csv.

Analyses
--------
1. Count audit  — gains/losses and gain/loss rates: JOINT vs FLOW (scatter, histograms)
2. Gains vs losses scatter — visual comparison of event distributions per method
3. Dist parameter comparison — dist and loss_dist distributions per method
4. Stability — simulate N trials per trait from its inferred parameters, re-reconstruct
               each, and track the distribution of inferred gains/losses per trial.
               Representative trait subset used (spans gains/losses/root_state diversity).
5. Simulation accuracy — prevalence, parsimony, MPD, clade-JSD calibration.

Dist threshold mapping
  JOINT: dist / loss_dist             (hard JOINT threshold)
  FLOW:  dist_marginal / loss_dist_marginal  (soft MPPA threshold)

Usage
-----
    python dev/compare_acr_methods.py [--n_sims 20] [--max_workers 8] \\
        [--output_dir dev/acr_plots] [--n_accuracy 50] [--n_traits 10]
"""

import argparse
import importlib.util
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, gaussian_kde

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[1]
PANX_DIR     = REPO_ROOT / "tests" / "panx"
MARGINAL_CSV = PANX_DIR / "pastmlout_marginal.csv"
ANN_FILE     = str(PANX_DIR / "ecoli_accessory.csv")
TREE_FILE    = str(PANX_DIR / "ecoli_accessory.nwk")

sys.path.insert(0, str(REPO_ROOT / "dev"))
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
METHOD_COLORS = {"JOINT": "#2166ac", "FLOW": "#d6604d"}
TRAIT_PALETTE = sns.color_palette("tab10", 10)
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title):
    sep = "=" * 70
    print(f"\n{sep}\n  {title}\n{sep}")


def _corr_row(x, y, label):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return {"column": label, "n": n, "pearson_r": np.nan,
                "spearman_r": np.nan, "mean_abs_diff": np.nan}
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    return {"column": label, "n": n,
            "pearson_r": round(pr, 4), "spearman_r": round(sr, 4),
            "mean_abs_diff": round(float(np.mean(np.abs(x - y))), 4)}


def _print_corr_table(rows):
    hdr = f"  {'Column':<25} {'N':>6}  {'Pearson r':>9}  {'Spearman r':>10}  {'MAD':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['column']:<25} {r['n']:>6}  {r['pearson_r']:>9.4f}  "
              f"{r['spearman_r']:>10.4f}  {r['mean_abs_diff']:>10.4f}")


def _load_benchmark_mod():
    spec = importlib.util.spec_from_file_location(
        "benchmark_reconstruction",
        REPO_ROOT / "dev" / "benchmark_reconstruction.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Build canonical method DataFrames
# ---------------------------------------------------------------------------

def build_method_dfs(df_mar):
    """
    Return {'JOINT': df, 'FLOW': df} with canonical column names for sim_bit.

    Columns produced: gene, gains, losses, gain_subsize, loss_subsize,
                      dist, loss_dist, root_state
    JOINT source columns: gains, losses, gain_subsize, loss_subsize, dist, loss_dist
    FLOW  source columns: gains_flow, losses_flow, gain_subsize_marginal,
                          loss_subsize_marginal, dist_marginal, loss_dist_marginal
    """
    joint_cols = ["gene", "gains", "losses", "gain_subsize", "loss_subsize",
                  "dist", "loss_dist", "root_state"]
    df_joint = df_mar[[c for c in joint_cols if c in df_mar.columns]].copy()

    flow_rename = {
        "gains_flow":            "gains",
        "losses_flow":           "losses",
        "gain_subsize_marginal": "gain_subsize",
        "loss_subsize_marginal": "loss_subsize",
        "dist_marginal":         "dist",
        "loss_dist_marginal":    "loss_dist",
    }
    flow_src = list(flow_rename.keys()) + ["gene", "root_state"]
    df_flow = (df_mar[[c for c in flow_src if c in df_mar.columns]]
               .copy()
               .rename(columns=flow_rename))

    # Only drop rows where subsize is invalid (can't compute rates / simulate)
    for name, df in [("JOINT", df_joint), ("FLOW", df_flow)]:
        bad = df["gain_subsize"].isna() | (df["gain_subsize"] <= 0) | \
              df["loss_subsize"].isna() | (df["loss_subsize"] <= 0)
        if bad.sum():
            print(f"  [{name}] dropping {bad.sum()} rows with zero/NaN subsize")

    df_joint = df_joint[
        df_joint["gain_subsize"].notna() & (df_joint["gain_subsize"] > 0) &
        df_joint["loss_subsize"].notna() & (df_joint["loss_subsize"] > 0)
    ].reset_index(drop=True)

    df_flow = df_flow[
        df_flow["gain_subsize"].notna() & (df_flow["gain_subsize"] > 0) &
        df_flow["loss_subsize"].notna() & (df_flow["loss_subsize"] > 0)
    ].reset_index(drop=True)

    return {"JOINT": df_joint, "FLOW": df_flow}


# ---------------------------------------------------------------------------
# Part 1 — JOINT vs FLOW count / rate audit
# ---------------------------------------------------------------------------

def audit_joint_vs_flow(df_mar):
    """Compare gains/losses counts and rates between JOINT and FLOW."""
    _section("Part 1 — JOINT vs FLOW Count & Rate Audit")

    # Need to compute rates using the right subsizes
    df = df_mar.copy()
    df["gain_rate_joint"] = df["gains"]       / df["gain_subsize"]
    df["loss_rate_joint"] = df["losses"]      / df["loss_subsize"]
    df["gain_rate_flow"]  = df["gains_flow"]  / df["gain_subsize_marginal"]
    df["loss_rate_flow"]  = df["losses_flow"] / df["loss_subsize_marginal"]

    rows = []
    for label, ca, cb in [
        ("gains (count)",     "gains",           "gains_flow"),
        ("losses (count)",    "losses",           "losses_flow"),
        ("gain rate",         "gain_rate_joint",  "gain_rate_flow"),
        ("loss rate",         "loss_rate_joint",  "loss_rate_flow"),
    ]:
        if ca not in df.columns or cb not in df.columns:
            continue
        x = df[ca].to_numpy(float)
        y = df[cb].to_numpy(float)
        rows.append(_corr_row(x, y, f"{label}: JOINT vs FLOW"))

    _print_corr_table(rows)

    # Ratio summaries
    for col_a, col_b, label in [
        ("gains",  "gains_flow",  "gains FLOW/JOINT"),
        ("losses", "losses_flow", "losses FLOW/JOINT"),
    ]:
        x = df[col_a].to_numpy(float)
        y = df[col_b].to_numpy(float)
        mask = (x > 0) & np.isfinite(y)
        r = y[mask] / x[mask]
        print(f"\n  Ratio {label}:  "
              f"median={np.median(r):.3f}  mean={r.mean():.3f}  "
              f"p5={np.percentile(r,5):.3f}  p95={np.percentile(r,95):.3f}")

    return df


# ---------------------------------------------------------------------------
# Representative trait selection
# ---------------------------------------------------------------------------

def select_representative_traits(df, n=10):
    """
    Pick n traits spanning the diversity of gains, losses, and root_state.
    Strategy: split by root_state, then sample at even quantiles of gains
    within each group.  Traits with NaN/zero subsize are excluded (can't simulate).
    0-gains/losses traits are retained.
    """
    df = df.dropna(subset=["gains", "losses", "gain_subsize", "loss_subsize"])
    df = df[df["gain_subsize"] > 0].copy()

    if len(df) <= n:
        return df.reset_index(drop=True)

    def _quantile_sample(sub, k):
        if len(sub) == 0:
            return pd.DataFrame()
        if len(sub) <= k:
            return sub
        idx = np.linspace(0, len(sub) - 1, k, dtype=int)
        return sub.iloc[idx]

    rs0 = df[df["root_state"] == 0].sort_values("gains").reset_index(drop=True)
    rs1 = df[df["root_state"] == 1].sort_values("gains").reset_index(drop=True)

    n1 = n // 2
    n0 = n - n1
    selected = pd.concat([_quantile_sample(rs0, n0),
                           _quantile_sample(rs1, n1)]).reset_index(drop=True)
    return selected.head(n)


# ---------------------------------------------------------------------------
# Stability: per-trial inferred gains/losses
# ---------------------------------------------------------------------------

def stability_by_trait(tree_file, dfs, n_sims, max_workers, n_traits=10):
    """
    For each method and each representative trait:
      - simulate n_sims independent tip-state realisations using sim_bit
      - re-reconstruct each with JOINT ACR
      - record (inferred_gains, inferred_losses) per trial

    Returns
    -------
    DataFrame with columns:
      method, gene, trial, true_gains, true_losses,
      inferred_gains, inferred_losses, root_state
    """
    from simphyni.scripts.run_ancestral_reconstruction import (
        label_internal_nodes, compute_branch_upper_bound, reconstruct_trait,
    )
    from simphyni.Simulation.simulation import sim_bit
    from ete3 import Tree

    master_tree = Tree(tree_file, format=1)
    label_internal_nodes(master_tree)
    upper_bound  = compute_branch_upper_bound(master_tree)
    tree_newick  = master_tree.write(format=1)
    leaf_names   = [n.name for n in master_tree.iter_leaves()]
    n_trials     = min(max(n_sims, 1), 64)

    records = []

    for method_name, df in dfs.items():
        print(f"  [{method_name}] selecting representative traits ...", flush=True)
        sample = select_representative_traits(df, n=n_traits)
        if sample.empty:
            print(f"  [WARN] {method_name}: no traits after filtering.", flush=True)
            continue

        trait_params = sample.set_index("gene") if "gene" in sample.columns else sample

        print(f"  [{method_name}] simulating {n_trials} trials × "
              f"{len(trait_params)} traits ...", flush=True)
        lineages = sim_bit(master_tree, trait_params)  # (n_leaves, n_traits) uint64

        futures_meta = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for col_idx, gene in enumerate(trait_params.index):
                true_gains  = float(trait_params.loc[gene, "gains"])
                true_losses = float(trait_params.loc[gene, "losses"])
                rs          = int(trait_params.loc[gene, "root_state"])
                for trial in range(n_trials):
                    tip_states = ((lineages[:, col_idx] >> np.uint64(trial))
                                  & np.uint64(1)).astype(int)
                    # Allow constant traits through — reconstruct_trait returns None for them
                    sim_series = pd.Series(
                        tip_states.astype(str), index=leaf_names, name=gene,
                    )
                    fut = executor.submit(
                        reconstruct_trait,
                        gene, tree_newick, sim_series, upper_bound,
                        "threshold", int(tip_states.sum()),
                    )
                    futures_meta.append(
                        (fut, gene, trial, true_gains, true_losses, rs)
                    )

            for fut, gene, trial, tg, tl, rs in futures_meta:
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  [FAILED] {gene} t{trial}: {exc}", flush=True)
                    continue
                if res is None:
                    # Constant trait — record as 0 gains/losses
                    records.append({
                        "method": method_name, "gene": gene, "trial": trial,
                        "true_gains": tg, "true_losses": tl,
                        "inferred_gains": 0.0, "inferred_losses": 0.0,
                        "root_state": rs,
                    })
                    continue
                records.append({
                    "method": method_name, "gene": gene, "trial": trial,
                    "true_gains": tg, "true_losses": tl,
                    "inferred_gains": float(res["gains"]),
                    "inferred_losses": float(res["losses"]),
                    "root_state": rs,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Simulation accuracy
# ---------------------------------------------------------------------------

def run_sim_accuracy(dfs, tree_file, ann_file, n_sample):
    _section("Part 3 — Simulation Accuracy (JOINT vs FLOW)")
    print(f"  n_sample={n_sample}", flush=True)
    mod = _load_benchmark_mod()
    acc_df = mod.simulation_accuracy_evaluation(
        tree_file=tree_file, dfs=dfs, ann_file=ann_file, n_sample=n_sample,
    )
    mod.print_sim_accuracy_table(acc_df)
    return acc_df


# ---------------------------------------------------------------------------
# Plots — Part 1: JOINT vs FLOW count / rate comparison
# ---------------------------------------------------------------------------

def plot_joint_vs_flow_comparison(df_aug, outdir):
    """
    3-row figure:
      Row 0: scatter JOINT vs FLOW for gains count and losses count
      Row 1: ratio histograms (FLOW / JOINT) for gains and losses counts
      Row 2: scatter of gain rate and loss rate (JOINT vs FLOW)
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle("JOINT vs FLOW: Counts and Rates", fontsize=13, y=1.01)

    def _scatter_identity(ax, x_col, y_col, x_lbl, y_lbl, color):
        x = df_aug[x_col].to_numpy(float)
        y = df_aug[y_col].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], alpha=0.15, s=4, color=color)
        lo = min(x[valid].min(), y[valid].min())
        hi = max(x[valid].max(), y[valid].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.7)
        if valid.sum() > 4:
            pr, _ = pearsonr(x[valid], y[valid])
            ax.text(0.05, 0.92, f"r={pr:.3f}", transform=ax.transAxes,
                    fontsize=9, color="darkred")
        ax.set_xlabel(x_lbl, fontsize=9)
        ax.set_ylabel(y_lbl, fontsize=9)
        ax.grid(True, alpha=0.2)

    def _ratio_hist(ax, num_col, den_col, title, color):
        x = df_aug[den_col].to_numpy(float)
        y = df_aug[num_col].to_numpy(float)
        mask = (x > 0) & np.isfinite(y)
        ratio = np.clip(y[mask] / x[mask], 0, 20)
        ax.hist(ratio, bins=60, alpha=0.7, density=True,
                color=color, edgecolor="none")
        ax.axvline(1.0, color="black", lw=1.0, ls="--", label="ratio=1")
        med = np.median(ratio)
        ax.axvline(med, color=color, lw=1.0, ls=":", label=f"median={med:.2f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("FLOW / JOINT", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    # Row 0 — counts scatter
    _scatter_identity(axes[0, 0], "gains",  "gains_flow",
                      "gains (JOINT)", "gains (FLOW)", METHOD_COLORS["JOINT"])
    axes[0, 0].set_title("Gains: JOINT vs FLOW", fontsize=10)

    _scatter_identity(axes[0, 1], "losses", "losses_flow",
                      "losses (JOINT)", "losses (FLOW)", METHOD_COLORS["FLOW"])
    axes[0, 1].set_title("Losses: JOINT vs FLOW", fontsize=10)

    # Row 1 — ratio histograms
    _ratio_hist(axes[1, 0], "gains_flow",  "gains",  "Gains ratio FLOW/JOINT",
                METHOD_COLORS["JOINT"])
    _ratio_hist(axes[1, 1], "losses_flow", "losses", "Losses ratio FLOW/JOINT",
                METHOD_COLORS["FLOW"])

    # Row 2 — rate scatter
    _scatter_identity(axes[2, 0], "gain_rate_joint", "gain_rate_flow",
                      "gain rate (JOINT)", "gain rate (FLOW)", METHOD_COLORS["JOINT"])
    axes[2, 0].set_title("Gain rate: JOINT vs FLOW", fontsize=10)

    _scatter_identity(axes[2, 1], "loss_rate_joint", "loss_rate_flow",
                      "loss rate (JOINT)", "loss rate (FLOW)", METHOD_COLORS["FLOW"])
    axes[2, 1].set_title("Loss rate: JOINT vs FLOW", fontsize=10)

    plt.tight_layout()
    out = Path(outdir) / "01_joint_vs_flow_counts_rates.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_gains_vs_losses(dfs, outdir):
    """
    Gains vs losses scatter for each method (JOINT and FLOW), side by side.
    Points coloured by root_state.  Density contours overlaid.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Gains vs Losses per Trait by Method", fontsize=12)

    rs_colors = {0: "#4dac26", 1: "#d01c8b"}
    rs_labels = {0: "root_state=0", 1: "root_state=1"}

    for ax, (method, df) in zip(axes, dfs.items()):
        x = df["gains"].to_numpy(float)
        y = df["losses"].to_numpy(float)
        rs = df["root_state"].fillna(0).astype(int).to_numpy()

        for rv in [0, 1]:
            mask = (rs == rv) & np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], alpha=0.18, s=5,
                       color=rs_colors[rv], label=rs_labels[rv])

        # Density contour on all valid points
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 30:
            try:
                xy  = np.vstack([x[valid], y[valid]])
                kde = gaussian_kde(xy, bw_method=0.15)
                xi  = np.linspace(x[valid].min(), np.percentile(x[valid], 98), 60)
                yi  = np.linspace(y[valid].min(), np.percentile(y[valid], 98), 60)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                ax.contour(Xi, Yi, Zi, levels=5, colors="black",
                           linewidths=0.6, alpha=0.4)
            except Exception:
                pass

        ax.set_xlabel("gains", fontsize=10)
        ax.set_ylabel("losses", fontsize=10)
        ax.set_title(f"{method}: gains vs losses  (n={valid.sum():,})", fontsize=11)
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = Path(outdir) / "02_gains_vs_losses_per_method.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plots — dist parameter distributions
# ---------------------------------------------------------------------------

def plot_dist_parameters(dfs, outdir):
    """
    2×2 grid: dist and loss_dist distributions for JOINT and FLOW.
    Overlaid KDE + histogram for each pair.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Distance Threshold Parameters: JOINT vs FLOW", fontsize=12)

    params = [
        ("dist",       "Gain dist threshold",  axes[0, 0]),
        ("loss_dist",  "Loss dist threshold",  axes[0, 1]),
    ]

    for col, title, ax in params:
        for method, df in dfs.items():
            vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            vals = vals[vals > 0]   # 0 means "no threshold" — plot separately
            if len(vals) == 0:
                continue
            color = METHOD_COLORS.get(method, "#888888")
            ax.hist(vals.values, bins=50, alpha=0.45, density=True,
                    color=color, label=method, edgecolor="none")
            # KDE
            try:
                kde = gaussian_kde(vals.values, bw_method=0.2)
                xs = np.linspace(vals.min(), np.percentile(vals, 99), 300)
                ax.plot(xs, kde(xs), color=color, lw=1.5)
            except Exception:
                pass
        ax.set_title(title + " (excl. 0)", fontsize=10)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    # Panel [1,0]: fraction of traits with dist == 0 per method
    ax = axes[1, 0]
    frac_data = {}
    for method, df in dfs.items():
        frac_data[method] = {
            "dist=0 (gains)": (df["dist"].fillna(0) == 0).mean(),
            "loss_dist=0":    (df["loss_dist"].fillna(0) == 0).mean(),
        }
    x = np.arange(2)
    w = 0.35
    methods = list(frac_data.keys())
    for i, method in enumerate(methods):
        vals = list(frac_data[method].values())
        ax.bar(x + i * w, vals, w, label=method,
               color=METHOD_COLORS.get(method, "#888888"), alpha=0.75)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(list(frac_data[methods[0]].keys()), fontsize=9)
    ax.set_ylabel("Fraction of traits", fontsize=9)
    ax.set_title("Fraction of traits with no emergence threshold (dist=0)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Panel [1,1]: scatter dist (JOINT) vs dist (FLOW = dist_marginal)
    ax = axes[1, 1]
    dj = dfs.get("JOINT")
    df = dfs.get("FLOW")
    if dj is not None and df is not None and "gene" in dj.columns and "gene" in df.columns:
        m = pd.merge(dj[["gene", "dist", "loss_dist"]],
                     df[["gene", "dist", "loss_dist"]],
                     on="gene", suffixes=("_joint", "_flow"))
        for col_j, col_f, label, color in [
            ("dist_joint",      "dist_flow",      "gain dist",  "#2166ac"),
            ("loss_dist_joint", "loss_dist_flow",  "loss dist",  "#d6604d"),
        ]:
            x = m[col_j].replace(np.inf, np.nan).dropna()
            y = m[col_f].reindex(x.index).replace(np.inf, np.nan)
            valid = x.notna() & y.notna()
            ax.scatter(x[valid], y[valid], alpha=0.2, s=5, color=color, label=label)
        hi = max(m[["dist_joint","loss_dist_joint","dist_flow","loss_dist_flow"]]
                 .replace(np.inf, np.nan).max())
        ax.plot([0, hi], [0, hi], "k--", lw=0.8)
        ax.set_xlabel("dist (JOINT)", fontsize=9)
        ax.set_ylabel("dist (FLOW / marginal)", fontsize=9)
        ax.set_title("Emergence threshold: JOINT vs FLOW", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    out = Path(outdir) / "03_dist_parameters.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plots — stability: per-trait lines of inferred gains/losses over trials
# ---------------------------------------------------------------------------

def plot_stability_lines(stab_df, outdir):
    """
    2×2 grid: (method) × (gains / losses).
    Each line = one trait; x = trial index, y = inferred count.
    Horizontal dashed line = true value per trait.
    Bold black line = mean across all traits per trial.
    """
    if stab_df.empty:
        print("  [SKIP] No stability data.")
        return

    methods    = sorted(stab_df["method"].unique())
    count_cols = [("inferred_gains", "true_gains", "Inferred gains"),
                  ("inferred_losses", "true_losses", "Inferred losses")]

    nrow, ncol = len(count_cols), len(methods)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 5 * nrow),
                             squeeze=False)
    fig.suptitle(
        "Stability: Inferred Gains/Losses per Trial\n"
        "(each line = one trait; dashed = true value; bold = mean across traits)",
        fontsize=12)

    for ri, (inf_col, true_col, ylbl) in enumerate(count_cols):
        for ci, method in enumerate(methods):
            ax = axes[ri][ci]
            sub = stab_df[stab_df["method"] == method]
            if sub.empty:
                ax.set_visible(False)
                continue

            genes = sorted(sub["gene"].unique())
            all_trials = sorted(sub["trial"].unique())

            for gi, gene in enumerate(genes):
                g = sub[sub["gene"] == gene].sort_values("trial")
                color = TRAIT_PALETTE[gi % len(TRAIT_PALETTE)]
                ax.plot(g["trial"], g[inf_col],
                        alpha=0.6, lw=1.3, marker="o", ms=3,
                        color=color, label=gene, zorder=2)
                true_val = g[true_col].iloc[0]
                ax.axhline(true_val, color=color, lw=0.8, ls="--",
                           alpha=0.5, zorder=1)

            # Bold mean trajectory
            mean_traj = sub.groupby("trial")[inf_col].mean().reindex(all_trials)
            ax.plot(mean_traj.index, mean_traj.values,
                    color="black", lw=2.2, ls="-", label="mean", zorder=4)

            ax.set_title(f"{method} — {ylbl}", fontsize=11)
            ax.set_xlabel("Trial index", fontsize=9)
            if ci == 0:
                ax.set_ylabel(ylbl, fontsize=9)
            ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.5)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = Path(outdir) / "04_stability_per_trial_lines.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_stability_variance(stab_df, outdir):
    """
    Per-trait std (over trials) for inferred gains and losses.
    Bars grouped by method, one bar per trait.  Gives a quick sense of
    which traits are stable and whether JOINT or FLOW varies more.
    """
    if stab_df.empty:
        return

    # Compute per-(method, gene) std of inferred gains and losses
    var_df = (stab_df
              .groupby(["method", "gene"])
              .agg(std_gains=("inferred_gains", "std"),
                   std_losses=("inferred_losses", "std"),
                   mean_gains=("inferred_gains", "mean"),
                   mean_losses=("inferred_losses", "mean"),
                   true_gains=("true_gains", "first"),
                   true_losses=("true_losses", "first"))
              .reset_index())

    methods = sorted(var_df["method"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stability: Std of Inferred Counts Across Trials per Trait", fontsize=12)

    for ax, (count_col, std_col, true_col, title) in zip(axes, [
        ("mean_gains",  "std_gains",  "true_gains",  "Gains"),
        ("mean_losses", "std_losses", "true_losses", "Losses"),
    ]):
        width = 0.35
        for mi, method in enumerate(methods):
            sub = var_df[var_df["method"] == method].sort_values("gene")
            xs  = np.arange(len(sub))
            ax.bar(xs + mi * width, sub[std_col], width,
                   label=method, alpha=0.75,
                   color=METHOD_COLORS.get(method, "#888888"))
            # Overlay true values as scatter
            ax.scatter(xs + mi * width, sub[true_col], marker="_",
                       s=80, color="black", linewidths=1.5, zorder=4, label=None)

        tick_genes = (var_df[var_df["method"] == methods[0]]
                      .sort_values("gene")["gene"].tolist())
        ax.set_xticks(np.arange(len(tick_genes)) + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(tick_genes, rotation=40, ha="right", fontsize=7)
        ax.set_title(f"{title}: std across trials (— = true value)", fontsize=11)
        ax.set_ylabel("Std of inferred count", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = Path(outdir) / "05_stability_variance_per_trait.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_stability_summary(stab_df, outdir):
    """
    Boxplot + jitter: distribution of |inferred - true| per method,
    split by gains and losses.
    """
    if stab_df.empty:
        return

    stab_df = stab_df.copy()
    stab_df["err_gains"]  = (stab_df["inferred_gains"]  - stab_df["true_gains"]).abs()
    stab_df["err_losses"] = (stab_df["inferred_losses"] - stab_df["true_losses"]).abs()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Stability: |Inferred − True| Distribution by Method", fontsize=12)

    for ax, (err_col, title) in zip(axes, [
        ("err_gains",  "Gains"),
        ("err_losses", "Losses"),
    ]):
        methods = sorted(stab_df["method"].unique())
        data = [stab_df[stab_df["method"] == m][err_col].dropna().values
                for m in methods]
        parts = ax.boxplot(data, labels=methods, patch_artist=True,
                           showfliers=False, widths=0.45)
        for patch, m in zip(parts["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS.get(m, "#888888"))
            patch.set_alpha(0.65)
        rng = np.random.default_rng(42)
        for xpos, (m, d) in enumerate(zip(methods, data), 1):
            jitter = rng.uniform(-0.18, 0.18, len(d))
            ax.scatter(xpos + jitter, d, alpha=0.25, s=9,
                       color=METHOD_COLORS.get(m, "#888888"), zorder=3)
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.set_title(f"{title}: |inferred − true|", fontsize=11)
        ax.set_ylabel("|error|", fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = Path(outdir) / "06_stability_summary_boxplot.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plots — simulation accuracy
# ---------------------------------------------------------------------------

def plot_accuracy_obs_vs_sim(acc_df, outdir):
    if acc_df.empty:
        return
    metrics = [
        ("obs_prevalence",  "sim_prevalence",    "Prevalence"),
        ("obs_parsimony",   "sim_parsimony_mean","Parsimony"),
        ("obs_mpd",         "sim_mpd_mean",      "MPD"),
    ]
    valid = [(oc, sc, lb) for oc, sc, lb in metrics
             if oc in acc_df.columns and sc in acc_df.columns]
    methods = sorted(acc_df["method"].unique())

    fig, axes = plt.subplots(len(valid), len(methods),
                              figsize=(5 * len(methods), 4 * len(valid)),
                              squeeze=False)
    fig.suptitle("Simulation Accuracy: Observed vs Simulated", fontsize=13)

    for ri, (oc, sc, lb) in enumerate(valid):
        for ci, method in enumerate(methods):
            ax = axes[ri][ci]
            sub = acc_df[acc_df["method"] == method][[oc, sc]].dropna()
            ax.scatter(sub[oc], sub[sc], alpha=0.45, s=14,
                       color=METHOD_COLORS.get(method, "#888888"))
            lo = min(sub[oc].min(), sub[sc].min())
            hi = max(sub[oc].max(), sub[sc].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
            if len(sub) > 4:
                pr, _ = pearsonr(sub[oc], sub[sc])
                ax.text(0.05, 0.92, f"r={pr:.3f}", transform=ax.transAxes,
                        fontsize=8, color="darkred")
            ax.set_xlabel(f"{lb} (obs)", fontsize=9)
            ax.set_ylabel(f"{lb} (sim)" if ci == 0 else "")
            ax.set_title(f"{method} — {lb}", fontsize=10)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = Path(outdir) / "07_accuracy_obs_vs_sim.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_accuracy_metric_distributions(acc_df, outdir):
    if acc_df.empty:
        return
    metrics = [
        ("prevalence_error",  "Prevalence |error|",       None),
        ("parsimony_ratio",   "Parsimony ratio (sim/obs)", 1.0),
        ("gain_count_ratio",  "Gain count ratio",          1.0),
        ("mpd_ratio",         "MPD ratio (sim/obs)",       1.0),
        ("clade_profile_jsd", "Clade JSD",                 None),
    ]
    valid = [(c, t, v) for c, t, v in metrics if c in acc_df.columns]
    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    fig.suptitle("Simulation Accuracy: Calibration Metric Distributions", fontsize=12)
    methods = sorted(acc_df["method"].unique())

    for ax, (col, title, vline) in zip(axes[0], valid):
        for method in methods:
            vals = (acc_df[acc_df["method"] == method][col]
                    .replace([np.inf, -np.inf], np.nan).dropna().values)
            if not len(vals):
                continue
            ax.hist(vals, bins=30, alpha=0.5, density=True,
                    color=METHOD_COLORS.get(method, "#888888"),
                    label=method, edgecolor="none")
        if vline is not None:
            ax.axvline(vline, color="black", lw=1.0, ls="--", label="ideal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = Path(outdir) / "08_accuracy_metric_histograms.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_accuracy_error_breakdown(acc_df, outdir):
    if acc_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Simulation Accuracy: Error Breakdown", fontsize=12)

    methods = sorted(acc_df["method"].unique())

    # High-error fraction
    ax = axes[0]
    fracs = [acc_df[acc_df["method"] == m]["is_high_error"].mean() for m in methods]
    bars = ax.bar(methods, fracs,
                  color=[METHOD_COLORS.get(m, "#888888") for m in methods],
                  alpha=0.75, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set_ylabel("Fraction flagged high-error", fontsize=10)
    ax.set_title("High-error fraction per method", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, axis="y")

    # dist_frac vs prevalence_error
    ax = axes[1]
    if "dist_frac_tree_depth" in acc_df.columns and "prevalence_error" in acc_df.columns:
        for method in methods:
            sub = acc_df[acc_df["method"] == method][
                ["dist_frac_tree_depth", "prevalence_error"]].dropna()
            ax.scatter(sub["dist_frac_tree_depth"], sub["prevalence_error"],
                       alpha=0.4, s=12,
                       color=METHOD_COLORS.get(method, "#888888"), label=method)
        ax.set_xlabel("dist / tree depth", fontsize=10)
        ax.set_ylabel("Prevalence |error|", fontsize=10)
        ax.set_title("Emergence depth vs prevalence error", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    out = Path(outdir) / "09_accuracy_error_breakdown.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_sims",         type=int, default=20,
                        help="Simulated trials per trait (≤64, default 20)")
    parser.add_argument("--max_workers",    type=int, default=8)
    parser.add_argument("--n_accuracy",     type=int, default=50,
                        help="Traits for sim accuracy (default 50)")
    parser.add_argument("--n_traits",       type=int, default=10,
                        help="Representative traits for stability (default 10)")
    parser.add_argument("--output_dir",     default="dev/acr_plots")
    parser.add_argument("--skip_stability", action="store_true")
    parser.add_argument("--skip_accuracy",  action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    # Fresh start: remove old outputs
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"Removed old output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    print(f"Plots → {out_dir}/\n", flush=True)

    # Load marginal CSV
    print(f"Loading {MARGINAL_CSV} ...", flush=True)
    df_mar = pd.read_csv(MARGINAL_CSV)
    print(f"  {len(df_mar):,} traits, {len(df_mar.columns)} columns", flush=True)

    # Build canonical DataFrames
    _section("Building method DataFrames")
    dfs = build_method_dfs(df_mar)
    for name, df in dfs.items():
        print(f"  {name}: {len(df):,} traits", flush=True)

    # Part 1 — JOINT vs FLOW audit + plots
    df_aug = audit_joint_vs_flow(df_mar)
    plot_joint_vs_flow_comparison(df_aug, out_dir)
    plot_gains_vs_losses(dfs, out_dir)
    plot_dist_parameters(dfs, out_dir)

    # Part 2 — stability
    if not args.skip_stability:
        _section("Part 2 — Stability Evaluation")
        print(f"  n_sims={args.n_sims}, n_traits={args.n_traits}, "
              f"max_workers={args.max_workers}", flush=True)
        stab_df = stability_by_trait(
            TREE_FILE, dfs, args.n_sims, args.max_workers, args.n_traits,
        )

        if not stab_df.empty:
            _section("Stability Summary")
            stab_df = stab_df.copy()
            stab_df["gain_abs_err"] = (stab_df["inferred_gains"] - stab_df["true_gains"]).abs()
            stab_df["loss_abs_err"] = (stab_df["inferred_losses"] - stab_df["true_losses"]).abs()
            summary = (stab_df
                       .groupby("method")
                       .agg(mean_gain_err=("gain_abs_err", "mean"),
                            median_gain_err=("gain_abs_err", "median"),
                            mean_loss_err=("loss_abs_err", "mean"),
                            median_loss_err=("loss_abs_err", "median"))
                       .reset_index())
            print(summary.to_string(index=False, float_format="{:.3f}".format))

            plot_stability_lines(stab_df, out_dir)
            plot_stability_variance(stab_df, out_dir)
            plot_stability_summary(stab_df, out_dir)
        else:
            print("  [WARN] No stability results produced.")

    # Part 3 — simulation accuracy
    if not args.skip_accuracy:
        acc_df = run_sim_accuracy(dfs, TREE_FILE, ANN_FILE, args.n_accuracy)
        if not acc_df.empty:
            plot_accuracy_obs_vs_sim(acc_df, out_dir)
            plot_accuracy_metric_distributions(acc_df, out_dir)
            plot_accuracy_error_breakdown(acc_df, out_dir)

    print(f"\n[Done] All plots saved to {out_dir}/\n")


if __name__ == "__main__":
    main()
