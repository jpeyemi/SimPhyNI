"""
inspect_null_distributions.py
==============================
Visualises null distribution degeneracy for traits with different sparsity
profiles (gains=0, gains=1/low-rate, gains=1/high-rate, gains~7, gains>10).

For each sparse trait paired with a normal reference trait:
  - Runs sim_bit() to build the simulated null
  - Extracts the 4096 log-odds values from compute_bitwise_cooc
  - Plots: null histogram + KDE, observed log-odds, KDE p-value
  - Flags degenerate nulls (variance near zero)

Usage
-----
    python dev/inspect_null_distributions.py [--output_dir dev/null_diag]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[1]
PANX_DIR  = REPO_ROOT / "tests" / "panx"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))

from simphyni.Simulation.simulation import (
    sim_bit, build_sim_params, compute_bitwise_cooc,
)
from ete3 import Tree

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MARGINAL_CSV = PANX_DIR / "pastmlout_marginal.csv"
ANN_FILE     = PANX_DIR / "ecoli_accessory.csv"
TREE_FILE    = PANX_DIR / "ecoli_accessory.nwk"

TRIALS       = 64    # must be 64 for single-chunk co-occurrence
COUNTING     = "JOINT"
SUBSIZE      = "ORIGINAL"

# Representative trait categories to inspect
CATEGORIES = {
    # gains=0: universally-present traits lost from some taxa — NOT truly sparse
    "gains_0_universal":   dict(label="gains=0, count~471\n(universal, only losses)"),
    # gains=1, losses=0: single gain, no loss — pure clade-specific signal
    "gains_1_loss_0":      dict(label="gains=1, losses=0\n(single gain, no loss)"),
    # gains=1, losses=1, tiny loss_subsize: highly inflated loss rate
    "gains_1_inf_loss":    dict(label="gains=1, losses=1\n(inflated loss rate ~100)"),
    # intermediate
    "gains_5_10":          dict(label="gains 5-10\n(intermediate)"),
    # normal reference
    "normal":              dict(label="gains>15, losses>10\n(normal)"),
}

N_PER_CAT   = 3   # traits to sample per category
REFERENCE_N = 1   # index into "normal" to use as the fixed reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_traits(df: pd.DataFrame, ann_genes: set) -> dict[str, list[str]]:
    """Select N_PER_CAT genes per category, restricted to genes present in ann_genes."""
    df = df.copy()
    df = df[df["gene"].isin(ann_genes)]
    df["gain_rate"] = np.where(df["gain_subsize"] > 0,
                               df["gains"] / df["gain_subsize"], np.nan)
    df["loss_rate"] = np.where(df["loss_subsize"] > 0,
                               df["losses"] / df["loss_subsize"], np.nan)
    selected = {}

    def _pick(mask, n=N_PER_CAT):
        return df[mask]["gene"].head(n).tolist()

    # Universal traits (gains=0, very high count — present in almost all tips)
    selected["gains_0_universal"]  = _pick((df["gains"] == 0) & (df["count"] > 400))
    # Single gain, no loss — clade-specific, no loss process
    selected["gains_1_loss_0"]     = _pick((df["gains"] == 1) & (df["losses"] == 0)
                                            & (df["count"] < 100))
    # Single gain AND single loss with tiny loss_subsize → inflated loss rate
    selected["gains_1_inf_loss"]   = _pick((df["gains"] == 1) & (df["losses"] == 1)
                                            & (df["loss_subsize"] < 0.02))
    selected["gains_5_10"]         = _pick((df["gains"] >= 5) & (df["gains"] <= 10))
    selected["normal"]             = _pick((df["gains"] >= 15) & (df["losses"] >= 10))

    return selected


def observed_log_odds(ann: pd.DataFrame, gene_a: str, gene_b: str,
                      tips: list[str]) -> float:
    """
    Compute observed log-odds ratio from the tip binary matrix.
    Both genes must be present in ann columns; tips are the leaf names.
    """
    eps = 1.0
    common = [t for t in tips if t in ann.index]
    a_vec = ann.loc[common, gene_a].values.astype(float)
    b_vec = ann.loc[common, gene_b].values.astype(float)
    a  = np.sum((a_vec == 1) & (b_vec == 1)) + eps
    b  = np.sum((a_vec == 1) & (b_vec == 0)) + eps
    c  = np.sum((a_vec == 0) & (b_vec == 1)) + eps
    d  = np.sum((a_vec == 0) & (b_vec == 0)) + eps
    return float(np.log((a * d) / (b * c)))


def run_simulation(tree, params_df: pd.DataFrame,
                   trait_names: list[str]) -> np.ndarray:
    """Run sim_bit for a subset of traits; return lineages (tips, traits, chunks)."""
    subset = params_df.loc[trait_names]
    lineages = sim_bit(tree, subset, trials=TRIALS)
    return lineages   # shape: (n_nodes, n_traits, 1)


def null_distribution(lineages: np.ndarray,
                      idx_a: int, idx_b: int) -> np.ndarray:
    """
    Extract the 4096 simulated log-odds values for a pair (idx_a, idx_b).
    lineages shape: (n_nodes, n_traits, n_chunks).
    """
    tp = lineages[:, [idx_a], :]   # (n_nodes, 1, 1)
    tq = lineages[:, [idx_b], :]
    cooc = compute_bitwise_cooc(tp, tq, total_trials=TRIALS)  # (1, 4096)
    return cooc[0]                  # (4096,)


def compute_kde_pvalue(null: np.ndarray, obs: float):
    """Return (p_ant, p_syn) via KDE."""
    std = null.std()
    if std < 1e-10:
        return np.nan, np.nan
    kde     = gaussian_kde(null, bw_method="silverman")
    kde_neg = gaussian_kde(-null, bw_method="silverman")
    p_ant   = float(kde.integrate_box_1d(-np.inf, obs))
    p_syn   = float(kde_neg.integrate_box_1d(-np.inf, -obs))
    return p_ant, p_syn


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CAT_COLORS = {
    "gains_0_universal":  "#d73027",
    "gains_1_loss_0":     "#f46d43",
    "gains_1_inf_loss":   "#fdae61",
    "gains_5_10":         "#74add1",
    "normal":             "#4575b4",
}


def plot_null(ax, null: np.ndarray, obs: float, title: str,
              color: str, cat: str, gene: str, ref_gene: str,
              params_row: pd.Series):
    """Plot null histogram + KDE + observed value on ax."""
    std  = null.std()
    mean = null.mean()
    degenerate = std < 1e-6

    if degenerate:
        ax.axvline(mean, color=color, lw=2, label="null (degenerate)")
        ax.text(0.5, 0.6, "DEGENERATE NULL\n(zero variance)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="red",
                bbox=dict(fc="white", ec="red", alpha=0.8))
    else:
        ax.hist(null, bins=60, density=True, alpha=0.4, color=color,
                edgecolor="none")
        xs = np.linspace(null.min() - null.std(), null.max() + null.std(), 400)
        kde = gaussian_kde(null, bw_method="silverman")
        ax.plot(xs, kde(xs), color=color, lw=1.5)

    ax.axvline(obs, color="black", lw=1.5, ls="--", label=f"observed ({obs:.2f})")

    p_ant, p_syn = compute_kde_pvalue(null, obs)
    p_min = min(p_ant, p_syn) if not np.isnan(p_ant) else np.nan

    gains      = int(params_row["gains"])
    losses     = int(params_row["losses"])
    gain_rate  = (params_row["gains"] / params_row["gain_subsize"]
                  if params_row["gain_subsize"] > 0 else np.nan)
    loss_rate  = (params_row["losses"] / params_row["loss_subsize"]
                  if params_row["loss_subsize"] > 0 else np.nan)

    info = (f"gains={gains}, losses={losses}\n"
            f"gain_rate={gain_rate:.2f}, loss_rate={loss_rate:.2f}\n"
            f"null std={std:.4f}  p={p_min:.4f}" if not np.isnan(p_min)
            else f"gains={gains}, losses={losses}\n"
                 f"gain_rate={gain_rate:.2f}, loss_rate={loss_rate:.2f}\n"
                 f"null std={std:.2e}  p=NaN (degenerate)")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, va="top", ha="left",
            fontsize=7, family="monospace",
            bbox=dict(fc="white", alpha=0.7, ec="none"))

    ax.set_title(f"{title}\n{gene} vs {ref_gene}", fontsize=8)
    ax.set_xlabel("log-odds ratio", fontsize=7)
    ax.set_ylabel("density", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right")
    return p_min, degenerate


def summary_table(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)[
        ["category", "gene", "ref_gene", "gains", "losses",
         "gain_rate", "null_std", "null_mean", "null_iqr",
         "observed_logodds", "p_value", "degenerate"]
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df   = pd.read_csv(MARGINAL_CSV)
    ann  = pd.read_csv(ANN_FILE, index_col=0)
    tree = Tree(str(TREE_FILE), format=1)
    for i, node in enumerate(tree.traverse()):
        if not node.name:
            node.name = f"internal_{i}"

    # Build full sim params (JOINT/ORIGINAL)
    params_df = build_sim_params(df, counting=COUNTING, subsize=SUBSIZE)
    params_df = params_df.set_index("gene") if "gene" in params_df.columns else params_df

    tip_names = [leaf.name for leaf in tree.get_leaves()]

    # Select traits (restricted to genes present in annotation CSV)
    ann_genes = set(ann.columns)
    sel = select_traits(df, ann_genes)
    ref_gene = sel["normal"][REFERENCE_N]
    print(f"Reference trait: {ref_gene}")

    # Collect all traits we need to simulate together
    all_traits = []
    for cat, genes in sel.items():
        all_traits.extend(genes)
    all_traits = list(dict.fromkeys(all_traits))   # deduplicate, preserve order
    # Filter to those present in params_df
    all_traits = [g for g in all_traits if g in params_df.index]
    ref_in_list = ref_gene in all_traits

    print(f"Simulating {len(all_traits)} traits ...")
    lineages = run_simulation(tree, params_df, all_traits)
    trait_idx = {g: i for i, g in enumerate(all_traits)}

    ref_idx = trait_idx[ref_gene]

    # Plot grid: rows=categories, cols=N_PER_CAT
    n_rows = len(CATEGORIES)
    n_cols = N_PER_CAT
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(
        f"Null distribution diagnostics: sparse traits vs reference '{ref_gene}'\n"
        f"(JOINT counting, {TRIALS} trials × {TRIALS} rotations = 4096 null values)",
        fontsize=11, y=1.01
    )

    records = []
    for row_i, (cat, meta) in enumerate(CATEGORIES.items()):
        genes = [g for g in sel.get(cat, []) if g in trait_idx][:N_PER_CAT]
        color = CAT_COLORS[cat]
        for col_j in range(N_PER_CAT):
            ax = axes[row_i, col_j]
            if col_j >= len(genes):
                ax.set_visible(False)
                continue
            gene = genes[col_j]
            idx  = trait_idx[gene]

            null = null_distribution(lineages, idx, ref_idx)
            obs  = observed_log_odds(ann, gene, ref_gene, tip_names)

            p_val, degen = plot_null(
                ax, null, obs,
                title=meta["label"],
                color=color, cat=cat, gene=gene, ref_gene=ref_gene,
                params_row=params_df.loc[gene],
            )

            gains     = int(params_df.loc[gene, "gains"])
            losses    = int(params_df.loc[gene, "losses"])
            gs        = params_df.loc[gene, "gain_subsize"]
            ls        = params_df.loc[gene, "loss_subsize"]
            gain_rate = gains / gs if gs > 0 else np.nan
            records.append(dict(
                category=cat, gene=gene, ref_gene=ref_gene,
                gains=gains, losses=losses,
                gain_rate=round(gain_rate, 4) if not np.isnan(gain_rate) else np.nan,
                null_std=round(null.std(), 6),
                null_mean=round(null.mean(), 4),
                null_iqr=round(float(np.percentile(null,75)-np.percentile(null,25)), 6),
                observed_logodds=round(obs, 4),
                p_value=round(p_val, 6) if not np.isnan(p_val) else np.nan,
                degenerate=degen,
            ))

    plt.tight_layout()
    fig_path = output_dir / "null_distributions.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close(fig)

    # Summary table
    tbl = summary_table(records)
    tbl_path = output_dir / "null_diagnostics.csv"
    tbl.to_csv(tbl_path, index=False)
    print(f"Saved: {tbl_path}")
    print()
    print(tbl.to_string(index=False))

    # --- Extra: pairwise null std heatmap across all selected traits ---
    print("\nBuilding pairwise null-std heatmap ...")
    n = len(all_traits)
    std_mat  = np.full((n, n), np.nan)
    mean_mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            null_ij = null_distribution(lineages, i, j)
            s = null_ij.std()
            m = null_ij.mean()
            std_mat[i, j] = s
            std_mat[j, i] = s
            mean_mat[i, j] = m
            mean_mat[j, i] = m

    short_names = [g[:18] for g in all_traits]
    fig2, (ax_std, ax_mean) = plt.subplots(1, 2, figsize=(max(14, n*0.6+2),
                                                           max(10, n*0.5+2)))
    import matplotlib.colors as mcolors

    im1 = ax_std.imshow(std_mat, aspect="auto", cmap="viridis",
                         norm=mcolors.LogNorm(vmin=max(std_mat[std_mat>0].min(), 1e-8),
                                              vmax=std_mat.max()))
    plt.colorbar(im1, ax=ax_std, label="null std (log scale)")
    ax_std.set_xticks(range(n)); ax_std.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax_std.set_yticks(range(n)); ax_std.set_yticklabels(short_names, fontsize=6)
    ax_std.set_title("Null distribution std (low=degenerate)", fontsize=9)

    im2 = ax_mean.imshow(mean_mat, aspect="auto", cmap="RdBu_r",
                          vmin=-abs(mean_mat).max(), vmax=abs(mean_mat).max())
    plt.colorbar(im2, ax=ax_mean, label="null mean log-odds")
    ax_mean.set_xticks(range(n)); ax_mean.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax_mean.set_yticks(range(n)); ax_mean.set_yticklabels(short_names, fontsize=6)
    ax_mean.set_title("Null distribution mean log-odds", fontsize=9)

    # Label trait categories on axes
    cat_labels = []
    for g in all_traits:
        for cat, genes in sel.items():
            if g in genes:
                cat_labels.append(cat)
                break
        else:
            cat_labels.append("?")
    for ax in (ax_std, ax_mean):
        for tick_i, (tick, lab) in enumerate(zip(ax.get_yticklabels(), cat_labels)):
            tick.set_color(CAT_COLORS.get(lab, "black"))

    plt.tight_layout()
    heat_path = output_dir / "null_std_heatmap.pdf"
    fig2.savefig(heat_path, bbox_inches="tight")
    print(f"Saved: {heat_path}")
    plt.close(fig2)

    # --- Extra: null std vs gains scatter ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    gains_arr    = np.array([r["gains"] for r in records], dtype=float)
    null_std_arr = np.array([r["null_std"] for r in records], dtype=float)
    null_iqr_arr = np.array([r["null_iqr"] for r in records], dtype=float)
    cat_arr      = [r["category"] for r in records]
    colors_arr   = [CAT_COLORS[c] for c in cat_arr]

    for ax3, (y_arr, ylabel) in zip(axes3,
            [(null_std_arr, "null std"), (null_iqr_arr, "null IQR")]):
        ax3.scatter(gains_arr, y_arr, c=colors_arr, s=60, zorder=3, edgecolors="white", lw=0.5)
        ax3.axhline(1e-6, color="red", ls="--", lw=1, label="degeneracy threshold (1e-6)")
        ax3.set_xlabel("JOINT gains", fontsize=9)
        ax3.set_ylabel(ylabel, fontsize=9)
        ax3.set_title(f"Null {ylabel} vs gain count", fontsize=10)
        ax3.set_yscale("symlog", linthresh=1e-5)
        from matplotlib.patches import Patch
        legend_els = [Patch(fc=v, label=k) for k, v in CAT_COLORS.items()]
        ax3.legend(handles=legend_els, fontsize=7, loc="upper left")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_path = output_dir / "null_std_vs_gains.pdf"
    fig3.savefig(scatter_path, bbox_inches="tight")
    print(f"Saved: {scatter_path}")
    plt.close(fig3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="dev/null_diag", type=Path)
    args = parser.parse_args()
    main(args.output_dir)
