#!/usr/bin/env python
"""
test_joint_mppa_independence.py
================================
Tests whether running JOINT reconstruction before MPPA on the same tree
object contaminates the MPPA marginal probabilities compared to running
MPPA on a fresh tree.

Hypothesis (from git history review):
    Commits 8a76b69 and 0b8b7d4 changed the tree passed to acr(..., MPPA)
    from a fresh Tree(newick) to tree.copy() and then to the same object
    already annotated by acr(..., JOINT). If PastML's internal state is
    affected by pre-existing node annotations, mp_df will differ and
    downstream FLOW/MARKOV/ENTROPY stats will be corrupted.

Usage:
    python dev/test_joint_mppa_independence.py
    python dev/test_joint_mppa_independence.py --n_genes 20 --verbose
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree
from pastml.acr import acr
from pastml.ml import MPPA, JOINT

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simphyni.scripts.run_ancestral_reconstruction import (
    label_internal_nodes,
    _node_dists_from_root,
    compute_branch_upper_bound,
    count_all_marginal_stats,
)

TREE_FILE = REPO_ROOT / "tests" / "panx" / "ecoli_accessory.nwk"
ANN_FILE  = REPO_ROOT / "tests" / "panx" / "ecoli_accessory.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ann(df: pd.DataFrame, gene: str) -> pd.DataFrame:
    return df[[gene]].copy().astype(str)


def _run_mppa_fresh(tree_newick: str, ann: pd.DataFrame, gene: str) -> pd.DataFrame:
    """MPPA on a completely fresh tree — no prior annotations."""
    tree = Tree(tree_newick, format=1)
    label_internal_nodes(tree)
    results = acr(tree, df=ann, prediction_method=MPPA, model="F81")
    return results[0]["marginal_probabilities"]


def _run_joint_then_mppa_same(tree_newick: str, ann: pd.DataFrame, gene: str) -> pd.DataFrame:
    """JOINT followed by MPPA on the *same* tree object (current behaviour)."""
    tree = Tree(tree_newick, format=1)
    label_internal_nodes(tree)
    acr(tree, df=ann, prediction_method=JOINT, model="F81")   # annotates internal nodes
    results = acr(tree, df=ann, prediction_method=MPPA, model="F81")
    return results[0]["marginal_probabilities"]


def _run_joint_then_mppa_copy(tree_newick: str, ann: pd.DataFrame, gene: str) -> pd.DataFrame:
    """JOINT on original, MPPA on tree.copy() (intermediate behaviour from 8a76b69)."""
    tree = Tree(tree_newick, format=1)
    label_internal_nodes(tree)
    acr(tree, df=ann, prediction_method=JOINT, model="F81")
    tree_copy = tree.copy()
    results = acr(tree_copy, df=ann, prediction_method=MPPA, model="F81")
    return results[0]["marginal_probabilities"]


def _compare_mp_dfs(mp_fresh: pd.DataFrame, mp_other: pd.DataFrame,
                    label: str) -> dict:
    """
    Compare two marginal_probabilities DataFrames.
    Returns a summary dict with max/mean absolute difference in P(state=1).
    """
    mp_fresh = mp_fresh.rename(columns=str)
    mp_other = mp_other.rename(columns=str)

    if "1" not in mp_fresh.columns or "1" not in mp_other.columns:
        return {"label": label, "n_nodes": 0, "max_abs_diff": np.nan,
                "mean_abs_diff": np.nan, "identical": False, "missing_col": True}

    shared = mp_fresh.index.intersection(mp_other.index)
    p1_fresh = mp_fresh.loc[shared, "1"].astype(float)
    p1_other = mp_other.loc[shared, "1"].astype(float)

    diffs = (p1_fresh - p1_other).abs()
    return {
        "label": label,
        "n_nodes": len(shared),
        "max_abs_diff": float(diffs.max()),
        "mean_abs_diff": float(diffs.mean()),
        "identical": bool((diffs < 1e-9).all()),
    }


def _marginal_stats_from_mp(tree_newick: str, ann: pd.DataFrame,
                             gene: str, mp_df: pd.DataFrame) -> dict:
    """Compute count_all_marginal_stats from a given mp_df on a fresh tree."""
    tree = Tree(tree_newick, format=1)
    label_internal_nodes(tree)
    node_dists = _node_dists_from_root(tree)
    upper_bound = compute_branch_upper_bound(tree)
    return count_all_marginal_stats(tree, gene, mp_df, upper_bound,
                                    node_dists=node_dists)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_genes", type=int, default=10,
                        help="Number of genes to test (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-gene diffs")
    args = parser.parse_args()

    print(f"Loading data from {ANN_FILE.name} / {TREE_FILE.name}")
    df = pd.read_csv(ANN_FILE, index_col=0)
    tree_newick = TREE_FILE.read_text().strip()

    # Pick genes that are binary and have both states present
    genes = []
    for col in df.columns:
        vals = df[col].dropna().astype(str).unique()
        if set(vals) <= {"0", "1"} and len(vals) == 2:
            genes.append(col)
        if len(genes) >= args.n_genes:
            break

    if not genes:
        print("ERROR: no binary genes found", file=sys.stderr)
        sys.exit(1)

    print(f"Testing {len(genes)} genes: {genes}\n")

    # --- Per-gene comparison ---
    mp_diff_rows   = []  # mp_df diffs
    stat_diff_rows = []  # inferred parameter diffs

    stat_keys = ["gains_flow", "losses_flow", "gains_markov", "losses_markov",
                 "gains_entropy", "losses_entropy",
                 "gain_subsize_marginal", "loss_subsize_marginal",
                 "dist_marginal", "loss_dist_marginal", "root_prob"]

    for gene in genes:
        ann = _build_ann(df, gene)

        mp_fresh      = _run_mppa_fresh(tree_newick, ann, gene)
        mp_same_tree  = _run_joint_then_mppa_same(tree_newick, ann, gene)
        mp_copy_tree  = _run_joint_then_mppa_copy(tree_newick, ann, gene)

        mp_diff_rows.append(_compare_mp_dfs(mp_fresh, mp_same_tree,
                                            label=f"{gene} | fresh vs same-tree"))
        mp_diff_rows.append(_compare_mp_dfs(mp_fresh, mp_copy_tree,
                                            label=f"{gene} | fresh vs copy-tree"))

        # Inferred parameter comparison
        stats_fresh     = _marginal_stats_from_mp(tree_newick, ann, gene, mp_fresh)
        stats_same_tree = _marginal_stats_from_mp(tree_newick, ann, gene, mp_same_tree)
        stats_copy_tree = _marginal_stats_from_mp(tree_newick, ann, gene, mp_copy_tree)

        for key in stat_keys:
            v_f = stats_fresh.get(key, np.nan)
            v_s = stats_same_tree.get(key, np.nan)
            v_c = stats_copy_tree.get(key, np.nan)
            if not isinstance(v_f, (int, float)):
                continue
            stat_diff_rows.append({
                "gene": gene, "stat": key,
                "fresh": v_f,
                "same_tree": v_s,
                "copy_tree": v_c,
                "diff_same": abs(v_f - v_s) if np.isfinite(v_f) and np.isfinite(v_s) else np.nan,
                "diff_copy": abs(v_f - v_c) if np.isfinite(v_f) and np.isfinite(v_c) else np.nan,
            })

        if args.verbose:
            print(f"  {gene}: mp_diff(same)={mp_diff_rows[-2]['max_abs_diff']:.2e}  "
                  f"mp_diff(copy)={mp_diff_rows[-1]['max_abs_diff']:.2e}")

    # --- Summary: mp_df level ---
    print("=" * 70)
    print("MARGINAL PROBABILITY COMPARISON  (P(state=1) per node)")
    print("=" * 70)
    mp_df_summary = pd.DataFrame(mp_diff_rows)
    same_rows = mp_df_summary[mp_df_summary["label"].str.contains("same-tree")]
    copy_rows = mp_df_summary[mp_df_summary["label"].str.contains("copy-tree")]

    def _summarise(rows, tag):
        print(f"\n  [{tag}]")
        print(f"    Genes with identical mp_df (|diff| < 1e-9): "
              f"{rows['identical'].sum()} / {len(rows)}")
        print(f"    Max absolute diff in P(1):  {rows['max_abs_diff'].max():.4e}")
        print(f"    Mean absolute diff in P(1): {rows['mean_abs_diff'].mean():.4e}")
        if rows["max_abs_diff"].max() > 1e-6:
            print("    *** WARNING: non-trivial differences detected ***")
        else:
            print("    OK: differences negligible (< 1e-6)")

    _summarise(same_rows, "fresh vs JOINT-then-MPPA (same tree object, current code)")
    _summarise(copy_rows, "fresh vs JOINT-then-MPPA (tree.copy(), 8a76b69 approach)")

    # --- Summary: inferred stats level ---
    print("\n" + "=" * 70)
    print("INFERRED PARAMETER COMPARISON  (ACR stats from mp_df)")
    print("=" * 70)
    stat_df = pd.DataFrame(stat_diff_rows)

    for tag, col in [("same-tree vs fresh", "diff_same"),
                     ("copy-tree vs fresh", "diff_copy")]:
        print(f"\n  [{tag}]")
        worst = stat_df.groupby("stat")[col].max().sort_values(ascending=False)
        print(worst.to_string())

    # Overall verdict
    max_mp_diff = same_rows["max_abs_diff"].max()
    print("\n" + "=" * 70)
    if max_mp_diff < 1e-9:
        print("RESULT: PASS — JOINT pre-annotation does NOT affect MPPA marginal probs.")
        print("        The tree-handling change is safe; look elsewhere for the regression.")
    elif max_mp_diff < 1e-4:
        print("RESULT: MARGINAL — small but non-zero differences detected.")
        print(f"        Max |ΔP(1)| = {max_mp_diff:.2e}. May compound across many genes.")
    else:
        print("RESULT: FAIL — JOINT pre-annotation significantly affects MPPA output.")
        print(f"        Max |ΔP(1)| = {max_mp_diff:.2e}. This is likely the regression cause.")
        print("        Fix: use a fresh Tree(newick) for MPPA in reconstruct_trait().")
    print("=" * 70)


if __name__ == "__main__":
    main()
