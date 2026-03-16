#!/usr/bin/env python
"""
run_ancestral_reconstruction.py
================================
Replaces the two-step `pastml.py` + `GL_tab.py` pipeline.

Uses the PastML Python API (no subprocess overhead) to run ancestral
state reconstruction in parallel across all input traits, then counts
gains, losses, subsizes, root state, and emergence threshold in-memory.

Outputs a single CSV.  Two uncertainty modes control what is written:

Uncertainty modes
-----------------
marginal (pipeline default — set by Snakefile.py)
    Runs both JOINT and MPPA reconstruction per trait.  Writes the full
    wide-format CSV: all JOINT columns plus the following marginal columns
    derived from MPPA marginal probabilities:

      gains_flow, losses_flow           — probability-flow rates (FLOW counting)
      gains_markov, losses_markov       — Markov transition rates
      gains_entropy, losses_entropy     — entropy-weighted flow rates
      gain_subsize_marginal / _nofilter / _thresh   — P-weighted subsizes
      loss_subsize_marginal / _nofilter / _thresh
      gain_subsize_entropy  / _nofilter / _thresh   — entropy-weighted subsizes
      loss_subsize_entropy  / _nofilter / _thresh
      dist_marginal, loss_dist_marginal — soft emergence thresholds
      root_prob                         — P(root state = 1)

    When runSimPhyNI.py receives this CSV it detects gains_flow and selects
    FLOW counting (the recommended, best-calibrated method).

threshold (legacy)
    Runs JOINT reconstruction only.  Writes the standard JOINT columns
    (gains, losses, gain_subsize, loss_subsize, dist, loss_dist, root_state).
    runSimPhyNI.py detects the absence of gains_flow and falls back to JOINT
    counting.  Use this to reproduce pre-marginal pipeline results or for
    faster runs where marginal calibration is not needed.

both
    Identical to marginal.  Retained for backward compatibility only.

Counting methods (selected downstream in runSimPhyNI.py / simulation.py)
-------------------------------------------------------------------------
The ACR CSV is produced once.  runSimPhyNI.py selects a counting method
automatically based on which columns are present:

  CSV produced with uncertainty=marginal  →  FLOW counting (default)
  CSV produced with uncertainty=threshold →  JOINT counting (fallback)

For development / benchmarking, all counting × subsize × masking combinations
(JOINT/FLOW/MARKOV/ENTROPY × ORIGINAL/NO_FILTER/THRESH × DIST/NONE/PATH) are
explored in dev/benchmark_reconstruction.py using build_sim_params() column
selection — no re-running ACR per method combination.
"""

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.special import loggamma

warnings.filterwarnings("ignore")

from pastml.acr import acr
from pastml.ml import JOINT, MPPA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prefiltering(obs: pd.DataFrame, run_traits: int = 0) -> np.ndarray:
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

    if run_traits == 0:
        pairs = fisher_significant_pairs(obs, obs, valid_obs, valid_obs)
    else:
        pairs = fisher_significant_pairs(
            obs[valid_obs[:run_traits]],
            obs[valid_obs[run_traits:]],
            valid_obs[:run_traits],
            valid_obs[run_traits:],
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
    """Return {node.name: cumulative_distance_from_root} for all nodes."""
    nd = {}
    for node in tree.traverse("preorder"):
        nd[node.name] = 0.0 if node.is_root() else nd[node.up.name] + node.dist
    return nd


# ---------------------------------------------------------------------------
# JOINT gain/loss counting  (mirrors countgainloss_tab.py logic)
# ---------------------------------------------------------------------------

def count_joint_stats(tree: Tree, gene: str, upper_bound: float,
                      node_dists: dict | None = None) -> dict:
    """
    Count gains, losses, subsizes, root state, and emergence thresholds
    from a JOINT-annotated ETE3 tree.

    The tree must already have been annotated by pastml.acr(..., JOINT).
    Node attribute `getattr(node, gene)` is the PastML prediction set.

    node_dists : optional pre-computed {node.name: dist_from_root} dict;
                 computed internally when not provided.
    """
    if node_dists is None:
        node_dists = _node_dists_from_root(tree)
    root_node = tree.get_tree_root()

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
    gain_subsize_nofilter = 0.0
    loss_subsize_nofilter = 0.0
    node_data = {}  # node.name -> (num_gains, num_losses)

    for node in tree.traverse("postorder"):
        s, ambiguous = _state(node)
        nd = node_dists[node.name]

        if s == 1 and (root_state == 0 or not node.is_root()):
            dist = min(dist, nd)
        if s == 0 and (root_state == 1 or not node.is_root()):
            loss_dist = min(loss_dist, nd)

        if not node.is_root():
            if node.dist < upper_bound:
                if s == 0:
                    gain_subsize += node.dist
                if s == 1:
                    loss_subsize += node.dist
            if node.dist > 0:
                if s == 0:
                    gain_subsize_nofilter += node.dist
                if s == 1:
                    loss_subsize_nofilter += node.dist

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

    # Second pass: threshold-consistent subsizes
    gain_subsize_thresh = 0.0
    loss_subsize_thresh = 0.0
    for node in tree.traverse():
        if node.is_root():
            continue
        nd = node_dists[node.name]
        s, _ = _state(node)
        if node.dist < upper_bound:
            if s == 0 and nd >= dist:
                gain_subsize_thresh += node.dist
            if s == 1 and nd >= loss_dist:
                loss_subsize_thresh += node.dist

    return {
        "gains": gains,
        "losses": losses,
        "dist": dist,
        "loss_dist": loss_dist,
        "gain_subsize": gain_subsize,
        "loss_subsize": loss_subsize,
        "gain_subsize_nofilter": gain_subsize_nofilter,
        "loss_subsize_nofilter": loss_subsize_nofilter,
        "gain_subsize_thresh": gain_subsize_thresh,
        "loss_subsize_thresh": loss_subsize_thresh,
        "root_state": root_state,
    }


# ---------------------------------------------------------------------------
# MARGINAL (MPPA) gain/loss counting — all counting/subsize variants
# ---------------------------------------------------------------------------

def _entropy(p1: float) -> float:
    """Shannon entropy (base-2, normalised to [0,1]) of a Bernoulli(p1) variable."""
    p0 = 1.0 - p1
    eps = 1e-12
    return -(p0 + eps) * np.log2(p0 + eps) - (p1 + eps) * np.log2(p1 + eps)


def count_all_marginal_stats(tree: Tree, gene: str, mp_df: pd.DataFrame,
                              upper_bound: float, p_threshold: float = 0.5,
                              node_dists: dict | None = None) -> dict:
    """
    Compute ALL marginal counting and subsize variants in a single pass.

    node_dists : pre-computed {node.name: dist_from_root} dict.
                 NOTE: safe to reuse from the JOINT tree because both trees
                 share identical topology and branch lengths (same newick source,
                 only internal annotations differ after acr()).
    """
    if node_dists is None:
        node_dists = _node_dists_from_root(tree)

    mp_df = mp_df.rename(columns=str)

    p1_map: dict[str, float] = {
        name: float(row.get("1", row.get(1, np.nan)))
        for name, row in mp_df.iterrows()
    }

    root_node = tree.get_tree_root()
    root_prob = p1_map.get(root_node.name, np.nan)
    if np.isnan(root_prob):
        root_prob = 0.0

    gains_flow = losses_flow = 0.0
    gains_markov = losses_markov = 0.0
    gains_entropy = losses_entropy = 0.0
    gain_subsize_marginal = loss_subsize_marginal = 0.0
    gain_subsize_marginal_nofilter = loss_subsize_marginal_nofilter = 0.0
    gain_subsize_entropy = loss_subsize_entropy = 0.0
    gain_subsize_entropy_nofilter = loss_subsize_entropy_nofilter = 0.0
    gain_subsize_marginal_thresh = loss_subsize_marginal_thresh = 0.0
    gain_subsize_entropy_thresh  = loss_subsize_entropy_thresh  = 0.0
    dist_m = float("inf")
    loss_dist_m = float("inf")

    # First pass: accumulate all stats except thresh (dist_m not yet known)
    edge_data: list = []  # (nd, p0_parent, p1_parent, cert, effective_bl) for thresh pass

    for node in tree.traverse():
        if node.is_root():
            continue

        p1_child  = p1_map.get(node.name,    np.nan)
        p1_parent = p1_map.get(node.up.name, np.nan)
        if np.isnan(p1_child) or np.isnan(p1_parent):
            continue

        p0_parent = 1.0 - p1_parent
        nd = node_dists[node.name]
        bl = node.dist
        effective_bl = min(bl, upper_bound)

        # ---- FLOW ----
        gain_flow = max(0.0, p1_child - p1_parent)
        loss_flow = max(0.0, p1_parent - p1_child)
        gains_flow  += gain_flow
        losses_flow += loss_flow

        # ---- MARKOV ----
        gains_markov  += p0_parent * p1_child * bl
        losses_markov += p1_parent * (1.0 - p1_child) * bl

        # ---- ENTROPY ----
        h_parent = _entropy(p1_parent)
        cert = 1.0 - h_parent
        gains_entropy  += gain_flow * cert
        losses_entropy += loss_flow * cert

        # ---- Subsize: marginal ----
        gain_subsize_marginal  += p0_parent * effective_bl
        loss_subsize_marginal  += p1_parent * effective_bl
        gain_subsize_marginal_nofilter += p0_parent * bl
        loss_subsize_marginal_nofilter += p1_parent * bl

        # ---- Subsize: entropy-weighted ----
        gain_subsize_entropy  += p0_parent * cert * effective_bl
        loss_subsize_entropy  += p1_parent * cert * effective_bl
        gain_subsize_entropy_nofilter += p0_parent * cert * bl
        loss_subsize_entropy_nofilter += p1_parent * cert * bl

        # ---- Soft emergence threshold ----
        if p1_child >= p_threshold and (root_prob < p_threshold or not node.is_root()):
            dist_m = min(dist_m, nd)
        if p1_child < p_threshold and (root_prob >= p_threshold or not node.is_root()):
            loss_dist_m = min(loss_dist_m, nd)

        if bl < upper_bound:
            edge_data.append((nd, p0_parent, p1_parent, cert, effective_bl))

    # Second pass: thresh subsizes — dist_m / loss_dist_m now finalised
    for nd, p0_parent, p1_parent, cert, effective_bl in edge_data:
        if nd >= dist_m:
            gain_subsize_marginal_thresh += p0_parent * effective_bl
            gain_subsize_entropy_thresh  += p0_parent * cert * effective_bl
        if nd >= loss_dist_m:
            loss_subsize_marginal_thresh += p1_parent * effective_bl
            loss_subsize_entropy_thresh  += p1_parent * cert * effective_bl

    return {
        "gains_flow":   gains_flow,
        "losses_flow":  losses_flow,
        "gains_markov": gains_markov,
        "losses_markov": losses_markov,
        "gains_entropy": gains_entropy,
        "losses_entropy": losses_entropy,
        "gain_subsize_marginal":          gain_subsize_marginal,
        "loss_subsize_marginal":          loss_subsize_marginal,
        "gain_subsize_marginal_nofilter": gain_subsize_marginal_nofilter,
        "loss_subsize_marginal_nofilter": loss_subsize_marginal_nofilter,
        "gain_subsize_marginal_thresh":   gain_subsize_marginal_thresh,
        "loss_subsize_marginal_thresh":   loss_subsize_marginal_thresh,
        "gain_subsize_entropy":          gain_subsize_entropy,
        "loss_subsize_entropy":          loss_subsize_entropy,
        "gain_subsize_entropy_nofilter": gain_subsize_entropy_nofilter,
        "loss_subsize_entropy_nofilter": loss_subsize_entropy_nofilter,
        "gain_subsize_entropy_thresh":   gain_subsize_entropy_thresh,
        "loss_subsize_entropy_thresh":   loss_subsize_entropy_thresh,
        "dist_marginal":      dist_m,
        "loss_dist_marginal": loss_dist_m,
        "root_prob":          root_prob,
    }


# ---------------------------------------------------------------------------
# Per-trait reconstruction worker
# ---------------------------------------------------------------------------

def reconstruct_trait(
    gene: str,
    tree_newick: str,
    df_col: pd.Series,
    upper_bound: float,
    uncertainty: str,
    gene_count: int,
) -> dict | None:
    """
    Run PastML reconstruction for a single trait and return a stats dict.

    Parameters
    ----------
    gene        : trait name
    tree_newick : Newick string
    df_col      : Series with index = tip names, values = trait states
    upper_bound : branch-length upper bound for subsize calculation
    uncertainty : 'threshold', 'marginal', or 'both'
    gene_count  : number of tips with trait = 1 (for 'count' column)

    When uncertainty in ('marginal', 'both'), the returned dict also contains
    a '_mp_df' key holding the MPPA marginal_probabilities DataFrame.  This
    key must be stripped before writing to CSV.
    """
    try:
        tree = Tree(tree_newick, format=1)
        label_internal_nodes(tree)

        ann = df_col.astype(str).to_frame(name=gene)

        states = sorted(ann[gene].dropna().unique().tolist())
        if len(states) < 2:
            return None  # can't reconstruct a constant trait

        # node_dists computed once on the JOINT tree and reused for MPPA
        # (safe: both trees share identical topology and branch lengths)
        node_dists = _node_dists_from_root(tree)

        # --- JOINT reconstruction ---
        acr(tree, df=ann, prediction_method=JOINT, model="F81")

        stats = count_joint_stats(tree, gene, upper_bound, node_dists=node_dists)
        stats["gene"] = gene
        stats["count"] = int(gene_count)

        # --- MPPA marginal reconstruction (if requested) ---
        if uncertainty in ("marginal", "both"):
            tree_mppa = tree.copy()

            mppa_results = acr(
                tree_mppa,
                df=ann,
                prediction_method=MPPA,
                model="F81",
            )
            mp_df = mppa_results[0]["marginal_probabilities"]
            marginal_stats = count_all_marginal_stats(
                tree_mppa, gene, mp_df, upper_bound,
                node_dists=node_dists,
            )
            stats.update(marginal_stats)
            stats["_mp_df"] = mp_df

        return stats

    except Exception as exc:
        print(f"  [ERROR] {gene}: {exc}", file=sys.stderr)
        return None


def build_path_mask(
    tree: Tree,
    mp_dfs: dict,
    gene_order: list,
    p_threshold: float = 0.5,
):
    """
    Build per-node × per-trait boolean eligibility masks using path-based
    upstream presence/absence tracking (preorder traversal).
    """
    node_list = list(tree.traverse())
    n_nodes   = len(node_list)
    n_traits  = len(gene_order)
    node_idx  = {node: i for i, node in enumerate(node_list)}

    gain_mask = np.zeros((n_nodes, n_traits), dtype=bool)
    loss_mask = np.zeros((n_nodes, n_traits), dtype=bool)

    for t_idx, gene in enumerate(gene_order):
        mp_df = mp_dfs.get(gene)
        if mp_df is None:
            gain_mask[:, t_idx] = True
            loss_mask[:, t_idx] = True
            continue

        mp_df_s = mp_df.rename(columns=str)  

        def _p1(name):
            if name not in mp_df_s.index:
                return np.nan
            row = mp_df_s.loc[name]
            return float(row.get("1", row.get(1, np.nan)))

        upstream_presence: dict = {}
        upstream_absence:  dict = {}

        for node in node_list:
            if node.is_root():
                p1 = _p1(node.name)
                upstream_presence[node] = (not np.isnan(p1)) and (p1 > p_threshold)
                upstream_absence[node]  = (not np.isnan(p1)) and ((1.0 - p1) > p_threshold)
                continue

            parent = node.up
            p1_parent = _p1(parent.name)
            p1_child  = _p1(node.name)

            if np.isnan(p1_parent) or np.isnan(p1_child):
                gain_mask[node_idx[node], t_idx] = True
                loss_mask[node_idx[node], t_idx] = True
                upstream_presence[node] = upstream_presence.get(parent, False)
                upstream_absence[node]  = upstream_absence.get(parent, False)
                continue

            up_pres = upstream_presence.get(parent, False)
            up_abs  = upstream_absence.get(parent, False)

            if (1.0 - p1_parent) > p_threshold and (p1_child > p_threshold or up_pres):
                gain_mask[node_idx[node], t_idx] = True

            if p1_parent > p_threshold and ((1.0 - p1_child) > p_threshold or up_abs):
                loss_mask[node_idx[node], t_idx] = True

            upstream_presence[node] = up_pres or (p1_parent > p_threshold)
            upstream_absence[node]  = up_abs  or ((1.0 - p1_parent) > p_threshold)

    return gain_mask, loss_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PastML API-based ancestral reconstruction + gain/loss counting."
    )
    parser.add_argument("--inputs_file", required=True,
                        help="Reformatted trait CSV (rows=taxa, cols=traits, index col 0)")
    parser.add_argument("--tree_file", required=True,
                        help="Reformatted Newick tree file")
    parser.add_argument("--output_csv", required=True,
                        help="Output CSV path (pastmlout.csv equivalent)")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Process pool size for parallel reconstruction")
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
    inputs_file = args.inputs_file
    tree_file = args.tree_file
    output_csv = Path(args.output_csv)
    max_workers = args.max_workers
    uncertainty = args.uncertainty

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    obs = pd.read_csv(inputs_file, index_col=0)
    obs.index = obs.index.astype(str)

    # ---- Pre-filter traits ----
    if args.prefilter:
        print("Pre-filtering traits by Fisher exact test ...", flush=True)
        sample_ids = prefiltering(obs, run_traits=args.run_traits)
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

    sample_ids = [g for g in sample_ids if g in obs_filtered.columns]

    gene_sums = {g: int(pd.to_numeric(obs_filtered[g], errors="coerce").fillna(0).sum())
                 for g in sample_ids}

    print(f"Running reconstruction for {len(sample_ids)} traits "
          f"(uncertainty={uncertainty}, workers={max_workers}) ...", flush=True)

    # ---- Parallel reconstruction ----
    results = {}
    total = len(sample_ids)
    log_every = max(1, total // 20) 

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                reconstruct_trait,
                gene,
                tree_newick,
                obs_filtered[gene],         
                upper_bound,
                uncertainty,
                gene_sums.get(gene, 0),
            ): gene
            for gene in sample_ids
        }
        done = 0
        for future in as_completed(futures):
            gene = futures[future]
            done += 1
            try:
                res = future.result()
                if res is not None:
                    results[gene] = res
                if done % log_every == 0 or done == total:
                    print(f"  [{done}/{total}] last completed: {gene}", flush=True)
            except Exception as exc:
                print(f"  [FAILED] {gene}: {exc}", file=sys.stderr)

    # ---- Assemble output DataFrame ----
    if not results:
        print("ERROR: No traits were successfully reconstructed.", file=sys.stderr)
        sys.exit(1)

    # Strip in-memory-only keys before CSV serialization
    rows = [{k: v for k, v in results[g].items() if not k.startswith("_")}
            for g in sample_ids if g in results]
    df_out = pd.DataFrame(rows)

    # Canonical column order
    core_cols = [
        "gene", "gains", "losses", "count", "dist", "loss_dist",
        "gain_subsize", "loss_subsize",
        "gain_subsize_nofilter", "loss_subsize_nofilter",
        "gain_subsize_thresh", "loss_subsize_thresh",
        "root_state",
    ]
    marginal_cols = [
        "gains_flow", "losses_flow",
        "gains_markov", "losses_markov",
        "gains_entropy", "losses_entropy",
        "gain_subsize_marginal", "loss_subsize_marginal",
        "gain_subsize_marginal_nofilter", "loss_subsize_marginal_nofilter",
        "gain_subsize_marginal_thresh", "loss_subsize_marginal_thresh",
        "gain_subsize_entropy", "loss_subsize_entropy",
        "gain_subsize_entropy_nofilter", "loss_subsize_entropy_nofilter",
        "gain_subsize_entropy_thresh", "loss_subsize_entropy_thresh",
        "dist_marginal", "loss_dist_marginal", "root_prob",
    ]

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