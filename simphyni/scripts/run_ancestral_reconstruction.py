#!/usr/bin/env python
"""
run_ancestral_reconstruction.py
================================
Replaces the two-step `pastml.py` + `GL_tab.py` pipeline.

Uses the PastML Python API (no subprocess overhead) to run ancestral
state reconstruction in parallel across all input traits, then counts
gains, losses, subsizes, root state, and emergence threshold in-memory.

Outputs a single CSV matching what `GL_tab.py` previously produced,
with optional additional columns for marginal-probability uncertainty.

Uncertainty modes
-----------------
threshold (default)
    Hard emergence threshold: records the tree distance (from root) at
    which the first gain / loss event is inferred under the JOINT ML
    reconstruction.  Replicates existing SimPhyNI behavior.

marginal
    Soft uncertainty columns derived from MPPA marginal probabilities:
      gains_marginal, losses_marginal,
      gain_subsize_marginal, loss_subsize_marginal,
      dist_marginal, loss_dist_marginal, root_prob
    These are appended alongside the standard threshold columns.

both
    Outputs columns for both modes.
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.special import loggamma

warnings.filterwarnings("ignore")

from pastml.acr import acr
from pastml.ml import JOINT, MPPA

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="PastML API-based ancestral reconstruction + gain/loss counting."
)
parser.add_argument("--inputs_file", required=True,
                    help="Reformatted trait CSV (rows=taxa, cols=traits, index col 0)")
parser.add_argument("--tree_file", required=True,
                    help="Reformatted Newick tree file")
parser.add_argument("--output_csv", required=True,
                    help="Output CSV path (pastmlout.csv equivalent)")
parser.add_argument("--outdir", default=None,
                    help="Directory for per-trait auxiliary files (optional; default: dirname of output_csv)")
parser.add_argument("--max_workers", type=int, default=8,
                    help="Thread pool size for parallel reconstruction")
parser.add_argument("--summary_file", default=None,
                    help="Optional text summary of run status")
parser.add_argument("--uncertainty", choices=["threshold", "marginal", "both"],
                    default="threshold",
                    help="Uncertainty mode (default: threshold)")
parser.add_argument(
    "--prefilter",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Pre-filter traits by Fisher exact test before reconstruction",
)
parser.add_argument("-r", "--run_traits", type=int, default=0,
                    help="Number of 'query' traits for one-vs-rest mode (0 = all-vs-all)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prefiltering(obs: pd.DataFrame) -> np.ndarray:
    """Return trait names that are significant in at least one pairwise Fisher test."""
    valid_obs = np.array(obs.columns)

    def fisher_significant_pairs(vars_df, targets_df, valid_vars, valid_targets, pval_threshold=0.05):
        X = vars_df.to_numpy().astype(bool)
        Y = targets_df.to_numpy().astype(bool)
        n = X.shape[0]

        a = X.T @ Y
        sX = X.sum(axis=0)
        sY = Y.sum(axis=0)
        b = sX[:, None] - a
        c = sY[None, :] - a
        d = n - (a + b + c)

        row1 = a + b
        row2 = c + d
        col1 = a + c
        n_all = n

        def logC(n_, k_):
            return loggamma(n_ + 1) - loggamma(k_ + 1) - loggamma(n_ - k_ + 1)

        logp_obs = logC(row1, a) + logC(row2, col1 - a) - logC(n_all, col1)
        p_obs = np.exp(logp_obs)
        p_two = np.minimum(1.0, 2 * p_obs)
        p_two[np.isnan(p_two)] = 1.0

        sig_mask = p_two < pval_threshold
        i_idx, j_idx = np.where(sig_mask)
        sig_pairs = np.column_stack((valid_vars[i_idx], valid_targets[j_idx]))
        return sig_pairs

    if args.run_traits == 0:
        pairs = fisher_significant_pairs(obs, obs, valid_obs, valid_obs)
    else:
        pairs = fisher_significant_pairs(
            obs[valid_obs[:args.run_traits]],
            obs[valid_obs[args.run_traits:]],
            valid_obs[:args.run_traits],
            valid_obs[args.run_traits:],
        )
    return np.unique(pairs.flatten())


def label_internal_nodes(tree: Tree) -> Tree:
    """Assign unique names to any unnamed internal nodes (in-place)."""
    counter = 0
    for node in tree.traverse():
        if not node.is_leaf() and not node.name:
            node.name = f"N{counter}"
            counter += 1
    return tree


def compute_branch_upper_bound(tree: Tree) -> float:
    """
    Compute the upper branch-length bound used for subsize calculation.
    Matches the logic in countgainloss_tab.py (IQR in log10 space).
    """
    bls = np.array([n.dist for n in tree.traverse() if not n.is_root() and n.dist > 0])
    if len(bls) == 0:
        return float("inf")
    log_bl = np.log10(bls)
    q1, q3 = np.percentile(log_bl, [25, 75])
    iqr = q3 - q1
    return 10 ** (q3 + 0.5 * iqr)


def _node_dists_from_root(tree: Tree) -> dict:
    """Return {node_id: cumulative_distance_from_root} for all nodes."""
    nd = {}
    for node in tree.traverse("preorder"):
        if node.is_root():
            nd[id(node)] = 0.0
        else:
            nd[id(node)] = nd[id(node.up)] + node.dist
    return nd


# ---------------------------------------------------------------------------
# JOINT gain/loss counting  (mirrors countgainloss_tab.py logic)
# ---------------------------------------------------------------------------

def count_joint_stats(tree: Tree, gene: str, upper_bound: float) -> dict:
    """
    Count gains, losses, subsizes, root state, and emergence thresholds
    from a JOINT-annotated ETE3 tree.

    The tree must already have been annotated by pastml.acr(..., JOINT).
    Node attribute `getattr(node, gene)` is the PastML prediction set.
    """
    root_node = tree.get_tree_root()
    node_dists = _node_dists_from_root(tree)

    def _state(node):
        """Return int state (0/1) and whether the node is ambiguous."""
        val = getattr(node, gene, None)
        if val is None:
            return None, False
        if isinstance(val, (set, frozenset)):
            ambiguous = len(val) > 1
            try:
                state = int(next(iter(val)))
            except (ValueError, StopIteration):
                return None, ambiguous
            return state, ambiguous
        # string or int
        s = str(val)
        ambiguous = "|" in s
        try:
            state = int(s.split("|")[0])
        except ValueError:
            return None, ambiguous
        return state, ambiguous

    root_state, _ = _state(root_node)
    if root_state is None:
        root_state = 0

    dist = float("inf")        # first gain distance from root
    loss_dist = float("inf")   # first loss distance from root
    gain_subsize = 0.0
    loss_subsize = 0.0
    node_data = {}  # node.name -> (num_gains, num_losses)

    for node in tree.traverse("postorder"):
        s, ambiguous = _state(node)
        nd = node_dists[id(node)]

        if s == 1 and (root_state == 0 or not node.is_root()):
            dist = min(dist, nd)
        if s == 0 and (root_state == 1 or not node.is_root()):
            loss_dist = min(loss_dist, nd)

        if not node.is_root() and node.dist < upper_bound:
            if s == 0:
                gain_subsize += node.dist
            if s == 1:
                loss_subsize += node.dist

        if node.is_leaf():
            node_data[node.name] = (0, 0)
        else:
            num_gains = 0
            num_losses = 0
            child_states = []
            for child in node.children:
                cg, cl = node_data[child.name]
                num_gains += cg
                num_losses += cl
                cs, _ = _state(child)
                child_states.append(cs)

            if ambiguous:
                num_gains += 0.5 if 1 in child_states else 0
                num_losses += 0.5 if 0 in child_states else 0
            else:
                num_gains += 1 if (1 in child_states and s == 0) else 0
                num_losses += 1 if (0 in child_states and s == 1) else 0

            node_data[node.name] = (num_gains, num_losses)

    gains, losses = node_data.get(root_node.name, (0, 0))

    return {
        "gains": gains,
        "losses": losses,
        "dist": dist,
        "loss_dist": loss_dist,
        "gain_subsize": gain_subsize,
        "loss_subsize": loss_subsize,
        "root_state": root_state,
    }


# ---------------------------------------------------------------------------
# MARGINAL (MPPA) gain/loss counting
# ---------------------------------------------------------------------------

def count_marginal_stats(tree: Tree, gene: str, mp_df: pd.DataFrame,
                         upper_bound: float, p_threshold: float = 0.5) -> dict:
    """
    Soft gain/loss statistics derived from MPPA marginal probabilities.

    For each directed edge (parent → child):
      E[gains]  ≈ max(0, P(child=1) - P(parent=1))
      E[losses] ≈ max(0, P(parent=1) - P(child=1))

    gain_subsize_marginal  = Σ P(parent=0) * min(branch_length, upper_bound)
    loss_subsize_marginal  = Σ P(parent=1) * min(branch_length, upper_bound)

    dist_marginal / loss_dist_marginal:
      Smallest root-distance where P(node=1) >= p_threshold (gain),
      or P(node=0) >= p_threshold while root is state-1 (loss).

    root_prob: P(root=1) from marginal probabilities.
    """
    mp_df = mp_df.copy()
    mp_df.columns = [str(c) for c in mp_df.columns]
    node_dists = _node_dists_from_root(tree)

    def _prob1(node_name: str) -> float:
        if node_name not in mp_df.index:
            return np.nan
        row = mp_df.loc[node_name]
        return float(row.get("1", row.get(1, np.nan)))

    root_node = tree.get_tree_root()
    root_prob = _prob1(root_node.name)
    if np.isnan(root_prob):
        root_prob = 0.0

    gains_m = 0.0
    losses_m = 0.0
    gain_subsize_m = 0.0
    loss_subsize_m = 0.0
    dist_m = float("inf")
    loss_dist_m = float("inf")

    for node in tree.traverse():
        if node.is_root():
            continue

        p1_child = _prob1(node.name)
        p1_parent = _prob1(node.up.name)
        if np.isnan(p1_child) or np.isnan(p1_parent):
            continue

        p0_parent = 1.0 - p1_parent
        nd = node_dists[id(node)]
        bl = node.dist

        # Expected transitions on this edge
        gain_edge = max(0.0, p1_child - p1_parent)
        loss_edge = max(0.0, p1_parent - p1_child)
        gains_m += gain_edge
        losses_m += loss_edge

        # Subsize: available branch length weighted by parent probability
        effective_bl = min(bl, upper_bound)
        gain_subsize_m += p0_parent * effective_bl
        loss_subsize_m += p1_parent * effective_bl

        # Soft emergence threshold: first node where P(state=1) >= p_threshold
        if p1_child >= p_threshold and (root_prob < p_threshold or not node.is_root()):
            dist_m = min(dist_m, nd)
        if p1_child < p_threshold and (root_prob >= p_threshold or not node.is_root()):
            loss_dist_m = min(loss_dist_m, nd)

    return {
        "gains_marginal": gains_m,
        "losses_marginal": losses_m,
        "gain_subsize_marginal": gain_subsize_m,
        "loss_subsize_marginal": loss_subsize_m,
        "dist_marginal": dist_m,
        "loss_dist_marginal": loss_dist_m,
        "root_prob": root_prob,
    }


# ---------------------------------------------------------------------------
# Per-trait reconstruction worker
# ---------------------------------------------------------------------------

def reconstruct_trait(
    gene: str,
    tree_newick: str,
    df_col: pd.DataFrame,
    upper_bound: float,
    uncertainty: str,
    gene_count: int,
) -> dict | None:
    """
    Run PastML reconstruction for a single trait and return a stats dict.

    Parameters
    ----------
    gene         : trait name (column in df_col)
    tree_newick  : Newick string (used to build a fresh tree per call)
    df_col       : DataFrame with a single column `gene`, index = tip names
    upper_bound  : branch-length upper bound for subsize calculation
    uncertainty  : 'threshold', 'marginal', or 'both'
    gene_count   : number of tips with trait = 1 (for 'count' column)
    """
    try:
        # Each call gets its own tree copy to ensure thread safety
        tree = Tree(tree_newick, format=1)
        label_internal_nodes(tree)

        ann = df_col[[gene]].copy()
        ann[gene] = ann[gene].astype(str)

        states = sorted(ann[gene].dropna().unique().tolist())
        if len(states) < 2:
            return None  # can't reconstruct a constant trait

        # --- JOINT reconstruction ---
        joint_results = acr(
            tree,
            df=ann,
            prediction_method=JOINT,
            model="F81",
        )

        stats = count_joint_stats(tree, gene, upper_bound)
        stats["gene"] = gene
        stats["count"] = int(gene_count)

        # --- MPPA marginal reconstruction (if requested) ---
        if uncertainty in ("marginal", "both"):
            tree_mppa = Tree(tree_newick, format=1)
            label_internal_nodes(tree_mppa)

            mppa_results = acr(
                tree_mppa,
                df=ann,
                prediction_method=MPPA,
                model="F81",
            )
            mp_df = mppa_results[0]["marginal_probabilities"]
            marginal_stats = count_marginal_stats(tree_mppa, gene, mp_df, upper_bound)
            stats.update(marginal_stats)

        return stats

    except Exception as exc:
        print(f"  [ERROR] {gene}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    inputs_file = args.inputs_file
    tree_file = args.tree_file
    output_csv = Path(args.output_csv)
    outdir = Path(args.outdir) if args.outdir else output_csv.parent
    max_workers = args.max_workers
    uncertainty = args.uncertainty

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    obs = pd.read_csv(inputs_file, index_col=0)

    # ---- Pre-filter traits ----
    if args.prefilter:
        print("Pre-filtering traits by Fisher exact test ...", flush=True)
        sample_ids = prefiltering(obs)
        print(f"  {len(sample_ids)} traits retained (of {len(obs.columns)} input)", flush=True)
    else:
        sample_ids = list(obs.columns)

    # ---- Load tree once; compute shared upper bound ----
    master_tree = Tree(tree_file, format=1)
    label_internal_nodes(master_tree)
    leaves_in_tree = set(master_tree.get_leaf_names())
    obs_filtered = obs.loc[[i for i in obs.index if i in leaves_in_tree]]
    upper_bound = compute_branch_upper_bound(master_tree)
    tree_newick = master_tree.write(format=1)

    # Pre-compute per-gene counts (positive tip counts)
    gene_sums = {g: int(obs_filtered[g].sum()) for g in sample_ids if g in obs_filtered.columns}
    sample_ids = [g for g in sample_ids if g in obs_filtered.columns]

    print(f"Running reconstruction for {len(sample_ids)} traits "
          f"(uncertainty={uncertainty}, workers={max_workers}) ...", flush=True)

    # ---- Parallel reconstruction ----
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                reconstruct_trait,
                gene,
                tree_newick,
                obs_filtered,
                upper_bound,
                uncertainty,
                gene_sums.get(gene, 0),
            ): gene
            for gene in sample_ids
        }
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            gene = futures[future]
            done += 1
            try:
                res = future.result()
                if res is not None:
                    results[gene] = res
                    if done % max(1, total // 10) == 0:
                        print(f"  [{done}/{total}] {gene}", flush=True)
            except Exception as exc:
                print(f"  [FAILED] {gene}: {exc}", file=sys.stderr)

    # ---- Assemble output DataFrame ----
    if not results:
        print("ERROR: No traits were successfully reconstructed.", file=sys.stderr)
        sys.exit(1)

    rows = [results[g] for g in sample_ids if g in results]
    df_out = pd.DataFrame(rows)

    # Canonical column order — core columns first, marginal columns appended
    core_cols = ["gene", "gains", "losses", "count", "dist", "loss_dist",
                 "gain_subsize", "loss_subsize", "root_state"]
    marginal_cols = ["gains_marginal", "losses_marginal",
                     "gain_subsize_marginal", "loss_subsize_marginal",
                     "dist_marginal", "loss_dist_marginal", "root_prob"]

    present_cols = core_cols + [c for c in marginal_cols if c in df_out.columns]
    df_out = df_out[[c for c in present_cols if c in df_out.columns]]
    df_out.to_csv(output_csv, index=False)
    print(f"\n[OK] Results written -> {output_csv}", flush=True)

    # ---- Optional summary file ----
    success = len(results)
    failed = len(sample_ids) - success
    summary_lines = [
        f"Output CSV: {output_csv}",
        f"Total traits attempted : {len(sample_ids)}",
        f"Successful             : {success}",
        f"Failed                 : {failed}",
        f"Uncertainty mode       : {uncertainty}",
        "\nJob complete.",
    ]
    if args.summary_file:
        Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_file).write_text("\n".join(summary_lines) + "\n")
    else:
        print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
