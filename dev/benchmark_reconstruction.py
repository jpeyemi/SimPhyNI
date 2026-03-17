#!/usr/bin/env python
"""
benchmark_reconstruction.py
============================
Benchmarks all combinations of ACR parameter-estimation methods in SimPhyNI.

Method combinations
-------------------
  Counting methods  (gain/loss numerator):
    JOINT    — discrete state transitions on JOINT reconstruction (Fitch-like)
    FLOW     — Σ max(0, P(child=1) − P(parent=1)) across edges (MPPA)
    MARKOV   — Σ P(parent=0) × P(child=1) × branch_length (MPPA)
    ENTROPY  — flow × (1 − H(parent))  where H = Shannon entropy (MPPA)

  Subsize methods  (rate denominator):
    ORIGINAL  — Σ bl[state=0 branches], IQR-filtered
    NO_FILTER — Σ bl[state=0 branches], unfiltered
    THRESH    — Σ bl[state=0 branches] downstream of first gain (post-emergence)

  Masking methods  (simulation eligibility gate):
    DIST      — global dist threshold from ACR
    NONE      — no threshold; gains/losses allowed anywhere on tree
    PATH      — path-based upstream presence/absence gate (per-clade aware)

4 × 3 × 3 = 36 combinations, plus 1 legacy reference = 37 total methods.
(Masking methods: DIST, NONE, PATH)
All ACR variants share a single JOINT+MPPA reconstruction run per trait.

Evaluation dimensions
---------------------
1. Speed               — wall time per ACR approach
2. Parameter stability — simulate → re-reconstruct; rate correlation/error
3. Parameter agreement — pairwise Pearson/Spearman r across method variants
4. Simulation accuracy — prevalence, parsimony, MPD, clade-JSD calibration
5. Precision/recall    — direction-aware P/R against known interaction pairs
                         (requires --eval_pr and --known_pairs)

Usage
-----
    python benchmark_reconstruction.py \\
        --tree my_tree.nwk \\
        --annotations traits.csv \\
        --output benchmark_results/ \\
        --eval_sim_accuracy --sim_accuracy_n 50 \\
        --eval_pr --known_pairs pairs.csv \\
        --n_stability 10 --max_workers 16
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections import namedtuple
from itertools import product as _product
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Imports from the SimPhyNI source tree
# ---------------------------------------------------------------------------
from simphyni.scripts.run_ancestral_reconstruction import (
    label_internal_nodes,
    compute_branch_upper_bound,
    count_joint_stats,
    prefiltering,
)

from pastml.acr import acr
from pastml.ml import JOINT, MPPA

# ---------------------------------------------------------------------------
# Method specification infrastructure
# ---------------------------------------------------------------------------

MethodSpec = namedtuple('MethodSpec', ['counting', 'subsize', 'masking', 'label', 'needs_mppa'])


def _build_method_specs(counting=None, subsize=None, masking=None):
    """
    Generate all active method combinations as a list of MethodSpec.

    Parameters filter to subsets; pass None to include all options.
    """
    _COUNTING = counting or ['JOINT', 'FLOW', 'MARKOV', 'ENTROPY']
    _SUBSIZE  = subsize  or ['ORIGINAL', 'NO_FILTER', 'THRESH']
    _MASKING  = masking  or ['DIST', 'NONE', 'PATH']
    specs = []
    for c, s, m in _product(_COUNTING, _SUBSIZE, _MASKING):
        needs_mppa = (c != 'JOINT') or (m == 'PATH')
        specs.append(MethodSpec(c, s, m, f"{c}_{s}_{m}", needs_mppa))
    return specs


def build_all_method_dfs(df_params, mp_dfs, tree_file, specs, p_threshold):
    """
    Build a standardised sim_params DataFrame for every MethodSpec.

    Returns
    -------
    dfs        : {label -> DataFrame with cols: gene, gains, losses, gain_subsize,
                   loss_subsize, dist, loss_dist, root_state}
    masks_dict : {label -> (gain_mask, loss_mask), label+'_gene_order' -> list}
                 Only populated for PATH masking specs when mp_dfs is available.
    """
    from simphyni.Simulation.simulation import build_sim_params
    from simphyni.scripts.run_ancestral_reconstruction import build_path_mask

    gene_order = list(df_params['gene']) if 'gene' in df_params.columns else list(df_params.index)

    # Build path masks once (shared across all PATH masking methods)
    path_gain_mask = path_loss_mask = None
    if mp_dfs and any(s.masking == 'PATH' for s in specs):
        master_tree = Tree(tree_file, format=1)
        label_internal_nodes(master_tree)
        path_gain_mask, path_loss_mask = build_path_mask(
            master_tree, mp_dfs, gene_order, p_threshold
        )

    dfs, masks_dict = {}, {}
    for spec in specs:
        try:
            sim_params = build_sim_params(
                df_params, spec.counting, spec.subsize,
                no_threshold=(spec.masking == 'NONE'),
            )
        except KeyError as e:
            print(f"  [SKIP] {spec.label}: missing column {e}", flush=True)
            continue

        dfs[spec.label] = sim_params.reset_index(drop=True)

        if spec.masking == 'PATH':
            if path_gain_mask is not None:
                masks_dict[spec.label] = (path_gain_mask, path_loss_mask)
                masks_dict[spec.label + '_gene_order'] = gene_order
            else:
                print(f"  [NOTE] {spec.label}: PATH masking skipped "
                      f"(mp_dfs unavailable from checkpoint)", flush=True)

    return dfs, masks_dict

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark SimPhyNI ancestral reconstruction method combinations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1] if "Usage" in __doc__ else "",
    )
    p.add_argument("--tree",        required=True, metavar="FILE",
                   help="Newick tree file")
    p.add_argument("--annotations", required=True, metavar="FILE",
                   help="Trait CSV (rows=taxa, cols=traits)")
    p.add_argument("--output",      required=True, metavar="DIR",
                   help="Output directory for results and figures")
    p.add_argument("--traits",      default=None, metavar="N", type=int,
                   help="Subsample N traits for speed (default: all)")
    p.add_argument("--max_workers", default=8, type=int,
                   help="Threads for API-based methods")
    p.add_argument("--n_stability", default=10, type=int,
                   help="Number of simulations for stability evaluation")
    p.add_argument("--eval_pr",     action="store_true",
                   help="Enable precision/recall evaluation (requires --known_pairs)")
    p.add_argument("--known_pairs", default=None, metavar="FILE",
                   help="CSV with columns T1,T2,direction of known interacting trait pairs")
    p.add_argument("--no_legacy",   action="store_true",
                   help="Skip legacy PastML CLI method")
    p.add_argument("--eval_sim_accuracy", action="store_true",
                   help="Run simulation accuracy diagnostic (prevalence, parsimony, MPD, clade JSD)")
    p.add_argument("--sim_accuracy_n", default=50, type=int,
                   help="Number of traits to sample for simulation accuracy (default: 50)")
    p.add_argument("--force_rerun", action="store_true",
                   help="Delete all checkpoints and re-run from scratch")
    # Method combination selectors
    p.add_argument("--counting_methods", nargs="+",
                   default=["JOINT", "FLOW", "MARKOV", "ENTROPY"],
                   help="Counting methods to benchmark (default: all four)")
    p.add_argument("--subsize_methods", nargs="+",
                   default=["ORIGINAL", "NO_FILTER", "THRESH"],
                   help="Subsize methods to benchmark (default: all three)")
    p.add_argument("--masking_methods", nargs="+",
                   default=["DIST", "NONE", "PATH"],
                   help="Simulation masking methods to benchmark (default: all three)")
    p.add_argument("--p_threshold", default=0.5, type=float,
                   help="Probability threshold for PATH masking (default: 0.5)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Method A: Legacy PastML CLI  (mirrors pastml.py logic)
# ---------------------------------------------------------------------------

def run_legacy_cli(tree_file: str, ann_file: str, traits: list[str],
                   outdir: Path, max_workers: int) -> tuple[pd.DataFrame, float]:
    """
    Run the original two-step pipeline (pastml CLI + GL_tab counting).
    Returns (DataFrame matching pastmlout.csv format, wall_time_seconds).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    results = {}

    def _run_one(gene):
        gene_dir = outdir / gene
        gene_dir.mkdir(exist_ok=True)
        out_file = gene_dir / "combined_ancestral_states.tab"
        if out_file.exists() and out_file.stat().st_size > 10:
            return gene, "skipped"
        try:
            log_path = gene_dir / "pastml.log"
            with open(log_path, "w") as log:
                subprocess.run([
                    "pastml",
                    "--tree", tree_file,
                    "--data", ann_file,
                    "--columns", gene,
                    "--id_index", "0",
                    "-n", "outs",
                    "--work_dir", str(gene_dir),
                    "--prediction_method", "JOINT",
                    "-m", "F81",
                    "--html", str(gene_dir / "out.html"),
                    "--data_sep", ",",
                ], stdout=log, stderr=subprocess.STDOUT, check=True)
            return gene, "success"
        except Exception as exc:
            return gene, f"failed: {exc}"

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, g): g for g in traits}
        for f in as_completed(futures):
            g, status = f.result()
            results[g] = status

    # Now count gains/losses using countgainloss_tab logic
    # Import here to avoid polluting top-level namespace
    from simphyni.scripts.countgainloss_tab import countgainloss
    from ete3 import Tree as _Tree

    tree_obj = _Tree(tree_file, format=1)
    leaves = set(tree_obj.get_leaf_names())
    obs = pd.read_csv(ann_file, index_col=0)
    obs.index = obs.index.astype(str)
    obs = obs.loc[[i for i in obs.index if i in leaves]]

    rows = []
    for gene in traits:
        gene_dir = str(outdir / gene)
        try:
            g, l, d, ld, gs, ls, rs = countgainloss(gene_dir, gene)
            rows.append({
                "gene": gene, "gains": g, "losses": l,
                "count": int(obs[gene].sum()) if gene in obs.columns else 0,
                "dist": d, "loss_dist": ld,
                "gain_subsize": gs, "loss_subsize": ls, "root_state": int(rs),
            })
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    return df, elapsed

# ---------------------------------------------------------------------------
# Method B: API threshold  /  Method C: API marginal
# ---------------------------------------------------------------------------

def run_api_method(tree_file: str, ann_file: str, traits: list[str],
                   uncertainty: str, max_workers: int) -> tuple[pd.DataFrame, float]:
    """
    Run run_ancestral_reconstruction.py logic in-process.
    uncertainty: 'threshold' -> Method B, 'marginal' -> Method C (both cols).
    """
    master_tree = Tree(tree_file, format=1)
    label_internal_nodes(master_tree)
    leaves_in_tree = set(master_tree.get_leaf_names())
    upper_bound = compute_branch_upper_bound(master_tree)
    tree_newick = master_tree.write(format=1)

    obs = pd.read_csv(ann_file, index_col=0)
    obs.index = obs.index.astype(str)
    obs = obs.loc[[i for i in obs.index if i in leaves_in_tree]]
    obs = obs[[c for c in traits if c in obs.columns]]

    gene_sums = {g: int(obs[g].sum()) for g in obs.columns}

    from simphyni.scripts.run_ancestral_reconstruction import reconstruct_trait

    t0 = time.perf_counter()
    rows = {}
    mp_dfs = {}   # {gene: mp_df} — populated only for marginal/both modes
    # ProcessPoolExecutor: reconstruct_trait is CPU-bound (PastML); not GIL-friendly.
    # All args (str, str, DataFrame, float, str, int) are picklable.
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                reconstruct_trait,
                gene, tree_newick, obs[gene], upper_bound, uncertainty, gene_sums.get(gene, 0)
            ): gene
            for gene in obs.columns
        }
        for f in as_completed(futures):
            gene = futures[f]
            res = f.result()
            if res is not None:
                if "_mp_df" in res:
                    mp_dfs[gene] = res.pop("_mp_df")   # extract before DataFrame assembly
                rows[gene] = res

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame([rows[g] for g in obs.columns if g in rows])
    return df, elapsed, mp_dfs

# ---------------------------------------------------------------------------
# Parameter agreement: pairwise correlations between methods
# ---------------------------------------------------------------------------

def parameter_agreement(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute Pearson/Spearman r between canonical method representatives.

    Uses only DIST-masking variants to avoid showing the same correlation
    dozens of times for columns that don't vary with subsize or masking:

      count_reps   — one per counting method (e.g. JOINT_ORIGINAL_DIST)
                     compared on COUNT_COLS (gains, losses, dist, loss_dist)

      subsize_reps — one per (counting, subsize) combo (e.g. FLOW_NO_FILTER_DIST)
                     compared on SUBSIZE_COLS (gain_subsize, loss_subsize)

    A_legacy is included in both comparisons.
    """
    COUNT_COLS   = ["gains", "losses", "dist", "loss_dist"]
    SUBSIZE_COLS = ["gain_subsize", "loss_subsize"]

    # Build canonical reps (DIST-masking only, one per counting / counting+subsize)
    count_reps   = {}   # counting_label -> df
    subsize_reps = {}   # "COUNTING_SUBSIZE" -> df

    for label, df in dfs.items():
        if label == "A_legacy":
            count_reps["A_legacy"]   = df
            subsize_reps["A_legacy"] = df
            continue
        parts = label.split("_")
        if len(parts) != 3:
            continue
        counting, subsize, masking = parts
        if masking == "DIST":
            count_reps.setdefault(counting, df)
            subsize_reps[f"{counting}_{subsize}"] = df

    def _corr_pairs(reps_dict, cols, axis_name):
        names  = list(reps_dict.keys())
        result = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                na, nb  = names[i], names[j]
                dfa, dfb = reps_dict[na], reps_dict[nb]
                shared = pd.merge(
                    dfa[["gene"] + [c for c in cols if c in dfa.columns]],
                    dfb[["gene"] + [c for c in cols if c in dfb.columns]],
                    on="gene", suffixes=("_a", "_b"),
                )
                if len(shared) < 5:
                    continue
                for col in cols:
                    ca, cb = f"{col}_a", f"{col}_b"
                    if ca not in shared.columns or cb not in shared.columns:
                        continue
                    x = shared[ca].replace([np.inf, -np.inf], np.nan)
                    y = shared[cb].replace([np.inf, -np.inf], np.nan)
                    valid = x.notna() & y.notna()
                    x, y = x[valid].values, y[valid].values
                    if len(x) < 5:
                        continue
                    pr, _ = pearsonr(x, y)
                    sr, _ = spearmanr(x, y)
                    result.append({
                        "method_A": na, "method_B": nb, "column": col,
                        "axis": axis_name, "n_shared": len(x),
                        "pearson_r": pr, "spearman_r": sr,
                    })
        return result

    records = (
        _corr_pairs(count_reps,   COUNT_COLS,   "counting")
        + _corr_pairs(subsize_reps, SUBSIZE_COLS, "subsize")
    )
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Stability: simulate → re-reconstruct → compare parameters
# ---------------------------------------------------------------------------

def stability_evaluation(tree_file: str, dfs: dict[str, pd.DataFrame],
                          n_sims: int, max_workers: int) -> pd.DataFrame:
    """
    For each method, pick up to 10 traits, simulate n_sims trees from the
    inferred parameters using sim_bit (the same Poisson-process model used
    by the actual SimPhyNI pipeline), re-reconstruct each simulated trial,
    and compare the re-estimated gain/loss rates to the originals.

    n_sims is capped at 64 — sim_bit always generates exactly 64 trials packed
    as bits in uint64. A value of 10 means 10 trials are re-reconstructed.
    """
    from simphyni.Simulation.simulation import sim_bit
    from simphyni.scripts.run_ancestral_reconstruction import reconstruct_trait
    from concurrent.futures import ProcessPoolExecutor

    master_tree = Tree(tree_file, format=1)
    label_internal_nodes(master_tree)
    upper_bound = compute_branch_upper_bound(master_tree)
    tree_newick = master_tree.write(format=1)

    # Leaf order from master_tree (must match sim_bit's leaf extraction)
    leaf_names = [n.name for n in master_tree.iter_leaves()]
    leaf_indices = [i for i, n in enumerate(master_tree.traverse()) if n.is_leaf()]

    records = []
    n_test_traits = 10
    n_trials = min(max(n_sims, 1), 64)   # sim_bit always produces 64 packed trials

    for method_name, df in dfs.items():
        df = df.copy().dropna(subset=["gains", "losses", "gain_subsize", "loss_subsize"])
        df = df[(df["gain_subsize"] > 0) & (df["loss_subsize"] > 0)]
        df["gain_rate"] = df["gains"] / df["gain_subsize"]
        df["loss_rate"] = df["losses"] / df["loss_subsize"]
        df = df[(df["gain_rate"] > 0) & (df["loss_rate"] > 0)]
        if df.empty:
            print(f"  [WARN] {method_name}: no traits with both rates > 0; skipping.", flush=True)
            continue

        sample = df.sample(n=min(n_test_traits, len(df)), random_state=0)
        if "gene" in sample.columns:
            trait_params = sample.set_index("gene")
        else:
            trait_params = sample

        # Simulate all sampled traits together (64 trials each, bit-packed)
        lineages = sim_bit(master_tree, trait_params)   # (n_leaves, n_traits) uint64
        leaf_lineages = lineages                         # already leaf-only from sim_bit

        # Submit all (trait × trial) reconstruction tasks to the process pool
        futures_meta = []   # (future, gene, trial, true_gain_rate, true_loss_rate)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for col_idx, gene in enumerate(trait_params.index):
                true_gain_rate = float(trait_params.loc[gene, "gain_rate"])
                true_loss_rate = float(trait_params.loc[gene, "loss_rate"])
                for trial in range(n_trials):
                    tip_states = ((leaf_lineages[:, col_idx] >> np.uint64(trial))
                                  & np.uint64(1)).astype(int)
                    if tip_states.sum() == 0 or tip_states.sum() == len(tip_states):
                        continue   # constant trait — reconstruction undefined
                    sim_series = pd.Series(
                        tip_states.astype(str),
                        index=leaf_names,
                        name=gene,
                    )
                    fut = executor.submit(
                        reconstruct_trait,
                        gene, tree_newick, sim_series, upper_bound,
                        "threshold", int(tip_states.sum()),
                    )
                    futures_meta.append((fut, gene, trial, true_gain_rate, true_loss_rate))

            # Collect results
            gene_gain_rates: dict = {}
            gene_loss_rates: dict = {}
            gene_true_gain:  dict = {}
            gene_true_loss:  dict = {}
            for fut, gene, trial, tgr, tlr in futures_meta:
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  [FAILED] {gene} trial {trial}: {exc}", file=sys.stderr)
                    continue
                if res is None:
                    continue
                gene_gain_rates.setdefault(gene, [])
                gene_loss_rates.setdefault(gene, [])
                gene_true_gain[gene]  = tgr
                gene_true_loss[gene]  = tlr
                gs = res.get("gain_subsize_thresh") or res.get("gain_subsize", 0)
                ls = res.get("loss_subsize_thresh") or res.get("loss_subsize", 0)
                if gs > 0:
                    gene_gain_rates[gene].append(res["gains"] / gs)
                if ls > 0:
                    gene_loss_rates[gene].append(res["losses"] / ls)

        for gene in gene_true_gain:
            q01 = gene_true_gain[gene]
            q10 = gene_true_loss[gene]
            if gene_gain_rates.get(gene):
                re = gene_gain_rates[gene]
                records.append({
                    "method": method_name, "gene": gene,
                    "parameter": "gain_rate",
                    "true_value": q01,
                    "mean_reestimate": float(np.mean(re)),
                    "std_reestimate":  float(np.std(re)),
                    "relative_error":  float(abs(np.mean(re) - q01) / max(q01, 1e-9)),
                })
            if gene_loss_rates.get(gene):
                re = gene_loss_rates[gene]
                records.append({
                    "method": method_name, "gene": gene,
                    "parameter": "loss_rate",
                    "true_value": q10,
                    "mean_reestimate": float(np.mean(re)),
                    "std_reestimate":  float(np.std(re)),
                    "relative_error":  float(abs(np.mean(re) - q10) / max(q10, 1e-9)),
                })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Simulation accuracy evaluation
# ---------------------------------------------------------------------------

def _fitch_parsimony_packed(tree, leaf_packed: dict) -> np.ndarray:
    """
    Vectorized Fitch parsimony over all 64 bit-packed trials simultaneously.

    Parameters
    ----------
    tree        : ETE3 Tree
    leaf_packed : {leaf_name -> np.uint64} — bit t of packed value = trial-t state

    Returns
    -------
    parsimony_scores : np.ndarray shape (64,) int — min transitions per trial
    """
    ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
    node_set_1 = {}   # bit t = 1 means trial t assigns state-1 to this node
    node_set_0 = {}   # bit t = 1 means trial t assigns state-0 to this node
    parsimony_scores = np.zeros(64, dtype=np.int64)

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            packed = np.uint64(leaf_packed.get(node.name, 0))
            node_set_1[id(node)] = packed
            node_set_0[id(node)] = ~packed & ALL
        else:
            inter_1 = ALL
            inter_0 = ALL
            for child in node.children:
                inter_1 = inter_1 & node_set_1[id(child)]
                inter_0 = inter_0 & node_set_0[id(child)]
            inter_nonempty = inter_1 | inter_0

            # Trials where intersection is empty require a state change (cost)
            cost_bits = (~inter_nonempty) & ALL
            for t in range(64):
                parsimony_scores[t] += int((cost_bits >> np.uint64(t)) & np.uint64(1))

            # Assign union where intersection was empty
            union_1 = np.uint64(0)
            union_0 = np.uint64(0)
            for child in node.children:
                union_1 = union_1 | node_set_1[id(child)]
                union_0 = union_0 | node_set_0[id(child)]
            # inter_nonempty bits → use intersection; else → use union
            node_set_1[id(node)] = (inter_nonempty & inter_1) | ((~inter_nonempty & ALL) & union_1)
            node_set_0[id(node)] = (inter_nonempty & inter_0) | ((~inter_nonempty & ALL) & union_0)

    return parsimony_scores


def _fitch_parsimony_obs(tree, obs_states: dict) -> int:
    """Fitch parsimony on a single observed trait (obs_states: {leaf_name -> 0/1})."""
    node_set = {}
    cost = 0
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            s = int(obs_states.get(node.name, 0))
            node_set[id(node)] = {s}
        else:
            children_sets = [node_set[id(c)] for c in node.children]
            inter = children_sets[0]
            for cs in children_sets[1:]:
                inter = inter & cs
            if inter:
                node_set[id(node)] = inter
            else:
                union = set()
                for cs in children_sets:
                    union |= cs
                node_set[id(node)] = union
                cost += 1
    return cost


def simulation_accuracy_evaluation(
    tree_file: str,
    dfs: dict,
    ann_file: str,
    n_sample: int = 50,
    masks_dict: dict = None,
) -> pd.DataFrame:
    """
    Multi-metric simulation calibration diagnostic.

    For each reconstruction method, simulates a sample of traits with sim_bit
    and compares simulated distributions to observed across five metrics:

      1. Prevalence          — global tip-count calibration
      2. Fitch parsimony     — min transitions; population-structure-invariant
      3. Gain count ratio    — sim expected gains vs observed gains
      4. MPD ratio           — mean pairwise phylogenetic distance; detects clustering
      5. Clade-profile JSD   — Jensen-Shannon divergence per top-level clade;
                               directly tests outbreak-clade population structure

    Returns a DataFrame with one row per (method, gene).
    """
    try:
        from simphyni.Simulation.simulation import sim_bit
    except ImportError as exc:
        print(f"[WARN] sim_bit not importable: {exc}")
        return pd.DataFrame()

    master_tree = Tree(tree_file, format=1)
    label_internal_nodes(master_tree)
    obs = pd.read_csv(ann_file, index_col=0)
    obs.index = obs.index.astype(str)

    # Ordered leaf list (matches sim_bit's leaf extraction order)
    leaf_list = [n.name for n in master_tree.iter_leaves()]
    n_leaves  = len(leaf_list)
    leaf_index = {name: i for i, name in enumerate(leaf_list)}

    # ── Pairwise distance matrix (precompute once) ──────────────────────────
    print("  Precomputing pairwise leaf distances ...", flush=True)
    D = np.zeros((n_leaves, n_leaves), dtype=float)
    leaves_ete = list(master_tree.iter_leaves())
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            d = master_tree.get_distance(leaves_ete[i], leaves_ete[j])
            D[i, j] = D[j, i] = d
    tree_depth = D.max() / 2.0  # rough estimate for dist_frac

    # ── Per-clade leaf membership (precompute once) ──────────────────────────
    root_children = master_tree.get_tree_root().children
    clade_id = np.zeros(n_leaves, dtype=int)
    for c_idx, clade_root in enumerate(root_children):
        for leaf in clade_root.iter_leaves():
            li = leaf_index.get(leaf.name)
            if li is not None:
                clade_id[li] = c_idx
    n_clades = len(root_children)

    records = []

    for method_name, df in dfs.items():
        print(f"  [{method_name}] running simulation accuracy ...", flush=True)
        df_clean = df.dropna(subset=["gains", "losses", "gain_subsize", "loss_subsize"])
        df_clean = df_clean[(df_clean["gain_subsize"] > 0) & (df_clean["loss_subsize"] > 0)]
        if df_clean.empty:
            print(f"    [WARN] no valid traits; skipping.", flush=True)
            continue

        sample = df_clean.sample(n=min(n_sample, len(df_clean)), random_state=0)
        trait_params = sample.set_index("gene") if "gene" in sample.columns else sample

        # Sub-select mask columns for sampled traits (if masks provided for this method)
        gm = lm = None
        if masks_dict and method_name in masks_dict:
            full_gm, full_lm = masks_dict[method_name]
            # full mask columns correspond to gene_order stored alongside masks
            gene_order_key = method_name + "_gene_order"
            full_gene_order = masks_dict.get(gene_order_key, [])
            if full_gene_order:
                col_idx = [full_gene_order.index(g) for g in trait_params.index
                           if g in full_gene_order]
                if col_idx:
                    gm = full_gm[:, col_idx]
                    lm = full_lm[:, col_idx]

        lineages = sim_bit(master_tree, trait_params, gain_mask=gm, loss_mask=lm)  # (n_leaves, n_traits) uint64

        for col_idx, gene in enumerate(trait_params.index):
            packed_col = lineages[:, col_idx]   # (n_leaves,) uint64

            # ── 1. Prevalence ───────────────────────────────────────────────
            trial_prevs = np.array([
                float(((packed_col >> np.uint64(t)) & np.uint64(1)).mean())
                for t in range(64)
            ])
            sim_prev = float(trial_prevs.mean())
            obs_prev = float(obs[gene].mean()) if gene in obs.columns else np.nan

            # ── 2 & 3. Fitch parsimony + gain count ratio ───────────────────
            # Build leaf_packed dict for vectorized Fitch
            leaf_packed = {leaf_list[i]: packed_col[i] for i in range(n_leaves)}
            sim_parsimony = _fitch_parsimony_packed(master_tree, leaf_packed)
            sim_pars_mean = float(sim_parsimony.mean())
            sim_pars_std  = float(sim_parsimony.std())

            # Observed parsimony
            if gene in obs.columns:
                obs_states_dict = {
                    name: int(obs.loc[name, gene])
                    for name in leaf_list if name in obs.index
                }
                obs_pars = _fitch_parsimony_obs(master_tree, obs_states_dict)
            else:
                obs_pars = np.nan

            obs_gains = float(trait_params.loc[gene, "gains"]) if gene in trait_params.index else np.nan
            gain_count_ratio = (sim_pars_mean / max(obs_gains, 1e-9)
                                if not np.isnan(obs_gains) else np.nan)
            parsimony_ratio  = (sim_pars_mean / max(float(obs_pars), 1e-9)
                                if not np.isnan(obs_pars) and obs_pars > 0 else np.nan)

            # ── 4. MPD ratio ─────────────────────────────────────────────────
            # Observed MPD
            if gene in obs.columns:
                obs_pos_idx = np.array([
                    leaf_index[n] for n in leaf_list
                    if n in obs.index and int(obs.loc[n, gene]) == 1
                ])
                obs_mpd = float(D[np.ix_(obs_pos_idx, obs_pos_idx)].mean()) if len(obs_pos_idx) > 1 else 0.0
            else:
                obs_mpd = np.nan

            # Simulated MPD (average over 64 trials; subsample positive set if large)
            trial_mpds = []
            for t in range(64):
                pos = np.where(((packed_col >> np.uint64(t)) & np.uint64(1)).astype(bool))[0]
                if len(pos) < 2:
                    continue
                if len(pos) > 100:
                    pos = np.random.choice(pos, 100, replace=False)
                trial_mpds.append(float(D[np.ix_(pos, pos)].mean()))
            sim_mpd = float(np.mean(trial_mpds)) if trial_mpds else 0.0
            mpd_ratio = (sim_mpd / max(obs_mpd, 1e-9)
                         if not np.isnan(obs_mpd) else np.nan)

            # ── 5. Per-clade prevalence JSD ──────────────────────────────────
            # Observed per-clade prevalence
            if gene in obs.columns:
                obs_clade = np.array([
                    obs.loc[[n for n in leaf_list if n in obs.index and clade_id[leaf_index[n]] == k], gene].mean()
                    for k in range(n_clades)
                ], dtype=float)
                obs_clade = np.nan_to_num(obs_clade)
            else:
                obs_clade = np.zeros(n_clades)

            # Simulated per-clade prevalence (mean across 64 trials)
            sim_clade = np.zeros(n_clades)
            for t in range(64):
                states_t = ((packed_col >> np.uint64(t)) & np.uint64(1)).astype(float)
                for k in range(n_clades):
                    mask = clade_id == k
                    if mask.sum() > 0:
                        sim_clade[k] += states_t[mask].mean()
            sim_clade /= 64.0

            from scipy.spatial.distance import jensenshannon
            clade_jsd = float(
                jensenshannon(obs_clade + 1e-9, sim_clade + 1e-9) ** 2
            )

            # ── dist_frac diagnostic ─────────────────────────────────────────
            dist_val = float(trait_params.loc[gene, "dist"]) if "dist" in trait_params.columns else np.nan
            dist_frac = dist_val / max(tree_depth, 1e-9) if not (np.isnan(dist_val) or np.isinf(dist_val)) else np.nan

            gs       = float(trait_params.loc[gene, "gain_subsize"]) if "gain_subsize" in trait_params.columns else np.nan
            gs_thresh = float(trait_params.loc[gene, "gain_subsize_thresh"]) if "gain_subsize_thresh" in trait_params.columns else np.nan

            # Flag as high-error if ≥2 metrics are simultaneously miscalibrated
            _error_flags = [
                not np.isnan(obs_prev) and abs(sim_prev - obs_prev) > 0.1,
                not np.isnan(parsimony_ratio) and parsimony_ratio > 2.0,
                not np.isnan(mpd_ratio) and (mpd_ratio > 2.0 or mpd_ratio < 0.5),
                clade_jsd > 0.2,
            ]
            is_high_error = sum(_error_flags) >= 2

            records.append({
                "method": method_name,
                "gene": gene,
                "obs_prevalence": obs_prev,
                "sim_prevalence": sim_prev,
                "prevalence_error": abs(sim_prev - obs_prev) if not np.isnan(obs_prev) else np.nan,
                "obs_parsimony": obs_pars,
                "sim_parsimony_mean": sim_pars_mean,
                "sim_parsimony_std": sim_pars_std,
                "parsimony_ratio": parsimony_ratio,
                "obs_gains": obs_gains,
                "gain_count_ratio": gain_count_ratio,
                "obs_mpd": obs_mpd,
                "sim_mpd_mean": sim_mpd,
                "mpd_ratio": mpd_ratio,
                "clade_profile_jsd": clade_jsd,
                "dist_frac_tree_depth": dist_frac,
                "gain_subsize": gs,
                "gain_subsize_thresh": gs_thresh,
                "is_high_error": is_high_error,
            })

    return pd.DataFrame(records)


def print_sim_accuracy_table(df: pd.DataFrame):
    _section("Simulation Accuracy Diagnostic")
    if df.empty:
        print("  No simulation accuracy data.")
        return
    metrics = ["prevalence_error", "parsimony_ratio", "gain_count_ratio",
               "mpd_ratio", "clade_profile_jsd"]
    summary = df.groupby("method")[metrics].agg(["mean", "median"]).round(3)
    print(summary.to_string())
    print()
    flagged = df.groupby("method")["is_high_error"].mean().round(3)
    print("  Fraction of traits flagged as high-error:")
    for method, frac in flagged.items():
        print(f"    {method:<25} {frac:.1%}")


# ---------------------------------------------------------------------------
# Precision / recall (optional, direction-aware)
# ---------------------------------------------------------------------------

def _load_known_pairs(known_pairs_file: str) -> dict[str, set]:
    """
    Load known interacting pairs from a CSV.

    Required columns : T1, T2, direction
      direction must be  1 (positive / synergistic) or -1 (negative / antagonistic).

    Returns a dict with keys:
      'positive'  — set of (T1, T2) tuples with direction == 1
      'negative'  — set of (T1, T2) tuples with direction == -1
      'any'       — union of both (direction-agnostic)

    Each set contains BOTH orderings (T1,T2) and (T2,T1) so lookups are
    symmetric, matching how SimPhyNI de-duplicates pairs.
    """
    known = pd.read_csv(known_pairs_file)
    required = {"T1", "T2", "direction"}
    missing = required - set(known.columns)
    if missing:
        raise ValueError(
            f"known_pairs file must have columns T1, T2, direction.  Missing: {missing}\n"
            f"  direction should be 1 (positive/synergistic) or -1 (negative/antagonistic)."
        )

    known["T1"] = known["T1"].astype(str)
    known["T2"] = known["T2"].astype(str)
    known["direction"] = known["direction"].astype(int)

    def _sym(subset):
        s = set()
        for _, row in subset.iterrows():
            # Force a consistent order (A, B) where A < B
            t1, t2 = sorted([str(row["T1"]), str(row["T2"])])
            s.add((t1, t2, int(row["direction"])))
        return s

    pos = _sym(known[known["direction"] ==  1])
    neg = _sym(known[known["direction"] == -1])
    return {"positive": pos, "negative": neg, "any": pos | neg}


def _pr_stats(pred_directional: set, known_directional: set) -> dict:
    """
    Compute TP/FP/FN/precision/recall/F1 given two sets of (T1, T2, direction) tuples.
    """
    tp = len(pred_directional & known_directional)
    fp = len(pred_directional - known_directional)
    fn = len(known_directional - pred_directional)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "F1": f1}


def precision_recall_evaluation(
    tree_file: str, dfs: dict,
    ann_file: str, known_pairs_file: str,
    masks_dict: dict = None,
) -> pd.DataFrame:
    """
    For each method's reconstruction output, run the SimPhyNI simulation
    test and compute direction-aware precision / recall against known pairs.

    Evaluates three subsets:
      positive  — synergistic (direction == 1)
      negative  — antagonistic (direction == -1)
      any       — direction-agnostic (pair identity only)
    """
    try:
        from simphyni.Simulation.tree_simulator import TreeSimulator
    except ImportError as exc:
        print(f"[WARN] Could not import TreeSimulator: {exc}")
        return pd.DataFrame()

    known_sets = _load_known_pairs(known_pairs_file)
    thresholds = [0.05, 0.01, 0.001]
    records = []

    for method_name, df in dfs.items():
        # Resolve certainty masks for this method (if any)
        gm = lm = gene_order_for_mask = None
        if masks_dict and method_name in masks_dict:
            gm, lm = masks_dict[method_name]
            gene_order_for_mask = masks_dict.get(method_name + "_gene_order", []) or None

        sim = TreeSimulator(tree_file, df, ann_file)
        sim.initialize_simulation_parameters(pre_filter=False, run_traits=1)
        refs = list(range(0,len(sim.obsdf.columns),2))
        pairs = []
        for i in refs:
            pairs.extend([(sim.obsdf.columns[i],sim.obsdf.columns[i+j]) for j in range(1,2)])
        sim.pairs, sim.obspairs = sim._get_pair_data2(sim.obsdf_modified, pairs)
        sim.run_simulation(gain_mask=gm, loss_mask=lm, gene_order=gene_order_for_mask)
        res = sim.get_results()

        for thresh in thresholds:
            sig = res[res["pval_bh"] < thresh].copy()
            sig["T1"] = sig["T1"].astype(str)
            sig["T2"] = sig["T2"].astype(str)
            sig["direction"] = sig["direction"].astype(int)

            # Build directional prediction sets (both orderings, consistent with known_sets)
            pred_dir = set()
            pred_any = set()
            for _, row in sig.iterrows():
                d = int(row["direction"])
                # Canonical order: alphabetic sort
                t1, t2 = sorted([str(row["T1"]), str(row["T2"])])
                pred_dir.add((t1, t2, d))
                pred_any.add((t1, t2, 0))

            # Pair-identity sets for cross-direction exclusion
            known_neg_pairs = {(t1, t2) for t1, t2, _ in known_sets["negative"]}
            known_pos_pairs = {(t1, t2) for t1, t2, _ in known_sets["positive"]}

            # Direction-filtered prediction sets with known-opposite pairs excluded:
            #   "positive" eval: only +1 predictions; exclude pairs known to be -1
            #   "negative" eval: only -1 predictions; exclude pairs known to be +1
            # Rationale: a pair with a known ground-truth direction of -1 is
            # irrelevant to the "positive" P/R question and should not contribute
            # to FP — it conflates direction-switching errors with discovery errors.
            pred_pos = {(t1, t2, d) for t1, t2, d in pred_dir
                        if d == 1 and (t1, t2) not in known_neg_pairs}
            pred_neg = {(t1, t2, d) for t1, t2, d in pred_dir
                        if d == -1 and (t1, t2) not in known_pos_pairs}

            # Direction-agnostic: all predictions vs all known pairs (no exclusion)
            known_any = {(a, b, 0) for a, b, _ in known_sets["any"]}

            for subset_name, pred_set, known_set in [
                ("positive", pred_pos, known_sets["positive"]),
                ("negative", pred_neg, known_sets["negative"]),
                ("any",      pred_any, known_any),
            ]:
                stats = _pr_stats(pred_set, known_set)
                records.append({
                    "method": method_name,
                    "threshold": thresh,
                    "association": subset_name,
                    **stats
                })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _section(title: str):
    sep = "=" * 70
    print(f"\n{sep}\n  {title}\n{sep}")


def print_speed_table(timings: dict[str, float]):
    _section("Speed Comparison (wall time)")
    for method, t in timings.items():
        print(f"  {method:<20} {t:>8.2f} s")


def print_agreement_table(df: pd.DataFrame):
    """
    Print Spearman r as a pivot matrix per column.

    - COUNT_COLS (gains, losses, dist, loss_dist): rows/cols = counting methods
    - SUBSIZE_COLS (gain_subsize, loss_subsize):   rows/cols = counting×subsize combos
    """
    _section("Parameter Agreement (Spearman r — canonical representatives)")
    if df.empty:
        print("  No agreement data.")
        return

    # Determine axis label for each column
    axis_label = {
        "gains": "counting method",
        "losses": "counting method",
        "dist": "counting method",
        "loss_dist": "counting method",
        "gain_subsize": "counting × subsize",
        "loss_subsize": "counting × subsize",
    }

    for col, grp in df.groupby("column"):
        methods = sorted(set(grp["method_A"]) | set(grp["method_B"]))
        idx = {m: i for i, m in enumerate(methods)}
        n   = len(methods)
        mat = np.full((n, n), np.nan)
        np.fill_diagonal(mat, 1.0)
        for _, row in grp.iterrows():
            i, j = idx[row["method_A"]], idx[row["method_B"]]
            mat[i, j] = mat[j, i] = row["spearman_r"]

        ax = grp["axis"].iloc[0] if "axis" in grp.columns else ""
        label = axis_label.get(col, ax)
        print(f"\n  [{col} — {label}]")

        # Short labels (truncate to 12 chars)
        short = [m[:12] for m in methods]
        col_w = max(12, max(len(s) for s in short))
        row_w = max(len(m) for m in methods)

        # Header
        header = " " * (row_w + 2) + "  ".join(f"{s:>{col_w}}" for s in short)
        print(f"  {header}")

        # Rows (upper triangle only; lower = "—")
        for i, m in enumerate(methods):
            cells = []
            for j in range(n):
                if j < i:
                    cells.append(" " * col_w)
                elif j == i:
                    cells.append(f"{'—':>{col_w}}")
                else:
                    v = mat[i, j]
                    cells.append(f"{v:{col_w}.3f}" if not np.isnan(v) else f"{'nan':>{col_w}}")
            print(f"  {m:<{row_w}}  " + "  ".join(cells))


def print_stability_table(df: pd.DataFrame):
    _section("Parameter Stability (self-consistency via re-simulation)")
    if df.empty:
        print("  No stability data.")
        return
    summary = (df.groupby(["method", "parameter"])["relative_error"]
                  .agg(["mean", "median", "std"])
                  .rename(columns={"mean": "mean_rel_err", "median": "median_rel_err",
                                   "std": "std_rel_err"})
                  .reset_index())
    print(summary.to_string(index=False, float_format="{:.4f}".format))

# ---------------------------------------------------------------------------
# Method ranking
# ---------------------------------------------------------------------------

def compute_method_ranking(
    sim_acc_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    pr_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank all methods across simulation accuracy, stability, and P/R.

    Metrics used (lower = better unless noted):
      sim_prev_error   — mean |sim_prevalence − obs_prevalence|
      parsimony_calib  — mean |parsimony_ratio − 1|
      mpd_calib        — mean |mpd_ratio − 1|
      clade_jsd        — mean clade_profile_jsd
      stability_gain   — mean gain_rate relative_error
      stability_loss   — mean loss_rate relative_error
      pr_f1            — F1 @ threshold closest to 0.001, association="any"  (higher = better)

    Each method gets rank 1..N per metric; composite_rank = mean rank across
    all available metrics.
    """
    records: dict[str, dict] = {}

    # Simulation accuracy
    if not sim_acc_df.empty:
        for method, grp in sim_acc_df.groupby("method"):
            records.setdefault(method, {})
            records[method]["sim_prev_error"]  = grp["prevalence_error"].mean()
            records[method]["parsimony_calib"] = (grp["parsimony_ratio"] - 1).abs().mean()
            records[method]["mpd_calib"]       = (grp["mpd_ratio"] - 1).abs().mean()
            records[method]["clade_jsd"]       = grp["clade_profile_jsd"].mean()

    # Stability
    if not stability_df.empty:
        for param, col in [("gain_rate", "stability_gain"), ("loss_rate", "stability_loss")]:
            sub = stability_df[stability_df["parameter"] == param]
            for method, val in sub.groupby("method")["relative_error"].mean().items():
                records.setdefault(method, {})[col] = val

    # Precision / recall — F1 at threshold closest to 0.001, association="any"
    if not pr_df.empty:
        pr_any = pr_df[pr_df["association"] == "any"].copy()
        if not pr_any.empty:
            pr_any["_td"] = (pr_any["threshold"] - 0.001).abs()
            best_thresh = pr_any["_td"].min()
            pr_best = pr_any[pr_any["_td"] == best_thresh]
            for _, row in pr_best.iterrows():
                records.setdefault(row["method"], {})["pr_f1"] = row["F1"]

    if not records:
        return pd.DataFrame()

    raw_df = pd.DataFrame.from_dict(records, orient="index")
    raw_df.index.name = "method"

    # Rank each metric (1 = best)
    lower_better  = ["sim_prev_error", "parsimony_calib", "mpd_calib",
                     "clade_jsd", "stability_gain", "stability_loss"]
    higher_better = ["pr_f1"]

    rank_df = raw_df.copy()
    for col in lower_better:
        if col in rank_df.columns:
            rank_df[f"rank_{col}"] = rank_df[col].rank(ascending=True, na_option="bottom")
    for col in higher_better:
        if col in rank_df.columns:
            rank_df[f"rank_{col}"] = rank_df[col].rank(ascending=False, na_option="bottom")

    rank_cols = [c for c in rank_df.columns if c.startswith("rank_")]
    if rank_cols:
        rank_df["composite_rank"] = rank_df[rank_cols].mean(axis=1)

    return rank_df.sort_values("composite_rank").reset_index()


def print_method_ranking(df: pd.DataFrame):
    _section("Method Ranking Summary")
    if df.empty:
        print("  No ranking data.")
        return

    rank_cols   = [c for c in df.columns if c.startswith("rank_")]
    metric_cols = [c for c in df.columns
                   if c not in {"method", "composite_rank"} and not c.startswith("rank_")]

    # Full ranking table
    print("\n  Full ranking (sorted by composite rank):")
    disp = df[["method", "composite_rank"] + rank_cols].copy()
    print(disp.to_string(index=False, float_format="{:.2f}".format))

    # Metric values table
    if metric_cols:
        print("\n  Metric values (lower = better, except pr_f1):")
        print(df[["method"] + metric_cols].to_string(
            index=False, float_format="{:.4f}".format))

    # Top 5 overall
    print("\n  Top 5 overall:")
    for _, row in df.head(5).iterrows():
        print(f"    #{row['composite_rank']:.1f}  {row['method']}")

    # Best per dimension
    dim_labels = {
        "rank_sim_prev_error":  "Prevalence calibration",
        "rank_parsimony_calib": "Parsimony calibration",
        "rank_mpd_calib":       "MPD calibration",
        "rank_clade_jsd":       "Clade profile accuracy",
        "rank_stability_gain":  "Gain rate stability",
        "rank_stability_loss":  "Loss rate stability",
        "rank_pr_f1":           "Precision/Recall F1",
    }
    available = [(rc, lbl) for rc, lbl in dim_labels.items() if rc in df.columns]
    if available:
        print("\n  Best per evaluation dimension:")
        for rank_col, label in available:
            best_row = df.loc[df[rank_col].idxmin()]
            print(f"    {label:<30}  {best_row['method']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ckpt_exists(path: Path) -> bool:
    """Return True if a checkpoint file exists and is non-empty."""
    return path.exists() and path.stat().st_size > 1


def _checkpoint_has_cols(path: Path, required_cols: list) -> bool:
    """Return True iff checkpoint exists, is non-empty, and contains all required columns."""
    if not _ckpt_exists(path):
        return False
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        return all(c in header for c in required_cols)
    except Exception:
        return False


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build active method specs from CLI selectors ----
    active_specs = _build_method_specs(
        counting=args.counting_methods,
        subsize=args.subsize_methods,
        masking=args.masking_methods,
    )
    print(f"Active method combinations: {len(active_specs)} "
          f"({len(args.counting_methods)}×{len(args.subsize_methods)}×{len(args.masking_methods)})",
          flush=True)

    # ---- Optional: delete all checkpoints for a fresh run ----
    _CKPT_FILES = [
        "params_A_legacy.csv", "params_all_acr.csv",
        "speed_timings.json", "parameter_agreement.csv", "stability.csv",
        "sim_accuracy.csv", "precision_recall.csv", "method_ranking.csv",
    ]
    if args.force_rerun:
        print("[--force_rerun] Deleting all checkpoints ...", flush=True)
        for fname in _CKPT_FILES:
            p = out_dir / fname
            if p.exists():
                p.unlink()
                print(f"  Deleted {p}", flush=True)

    # ---- Load and optionally subsample traits ----
    obs = pd.read_csv(args.annotations, index_col=0)
    traits = list(obs.columns)
    if args.traits:
        traits = traits[:args.traits]
    print(f"Benchmarking {len(traits)} traits on tree {args.tree}", flush=True)

    timings: dict = {}
    mp_dfs_all: dict = {}   # {gene: mp_df} — populated only on fresh ACR run

    timings_file = out_dir / "speed_timings.json"
    if timings_file.exists():
        with open(timings_file) as f:
            timings = json.load(f)

    # ---- Legacy CLI (optional) ----
    df_legacy = None
    ckpt_a = out_dir / "params_A_legacy.csv"
    if not args.no_legacy:
        if _ckpt_exists(ckpt_a):
            print("\n[1/2] Legacy CLI: loading from checkpoint ...", flush=True)
            df_legacy = pd.read_csv(ckpt_a)
            print(f"  Loaded {len(df_legacy)} traits", flush=True)
        else:
            print("\n[1/2] Running legacy PastML CLI ...", flush=True)
            try:
                df_legacy, t_a = run_legacy_cli(
                    args.tree, args.annotations, traits,
                    out_dir / "legacy_cli", args.max_workers,
                )
                timings["A_legacy"] = t_a
                df_legacy.to_csv(ckpt_a, index=False)
                print(f"  Done in {t_a:.1f}s  ({len(df_legacy)} traits)", flush=True)
            except Exception as exc:
                print(f"  [FAILED] {exc}", file=sys.stderr)

    # ---- Combined JOINT+MPPA ACR run → all 36 variant columns ----
    ckpt_acr = out_dir / "params_all_acr.csv"
    _ACR_REQUIRED = ["gains_flow", "gains_markov", "gains_entropy",
                     "gain_subsize_nofilter", "gain_subsize_entropy"]
    acr_valid = _checkpoint_has_cols(ckpt_acr, _ACR_REQUIRED)

    if acr_valid:
        print("\n[2/2] All-ACR params: loading from checkpoint ...", flush=True)
        df_acr = pd.read_csv(ckpt_acr)
        print(f"  Loaded {len(df_acr)} traits ({len(df_acr.columns)} columns)", flush=True)
        print("  [NOTE] mp_dfs unavailable from checkpoint — PATH masking methods will "
              "be skipped.\n         Run with --force_rerun to include PATH masking.",
              flush=True)
    else:
        if _ckpt_exists(ckpt_acr):
            print("\n[2/2] All-ACR checkpoint is stale — re-running ...", flush=True)
            ckpt_acr.unlink()
        else:
            print("\n[2/2] Running combined JOINT+MPPA reconstruction ...", flush=True)
        df_acr, t_acr, mp_dfs_all = run_api_method(
            args.tree, args.annotations, traits,
            uncertainty="both", max_workers=args.max_workers,
        )
        timings["all_acr"] = t_acr
        df_acr.to_csv(ckpt_acr, index=False)
        print(f"  Done in {t_acr:.1f}s  ({len(df_acr)} traits, "
              f"{len(df_acr.columns)} columns)", flush=True)

    # ---- Build per-combination DataFrames ----
    print("\nBuilding method parameter DataFrames ...", flush=True)
    all_dfs, masks_dict = build_all_method_dfs(
        df_acr, mp_dfs_all, args.tree, active_specs, args.p_threshold,
    )

    # Insert legacy as reference (last so it appears at the end of tables)
    if df_legacy is not None:
        all_dfs["A_legacy"] = df_legacy

    n_path_masked = sum(1 for k in masks_dict if not k.endswith("_gene_order"))
    print(f"  {len(all_dfs)} methods ready "
          f"({len(active_specs)} ACR combos"
          f"{', ' + str(n_path_masked) + ' with path masks' if n_path_masked else ''}"
          f"{', A_legacy' if df_legacy is not None else ''})",
          flush=True)

    # Save timings (cumulative across runs)
    with open(timings_file, "w") as f:
        json.dump(timings, f)

    # ---- Speed ----
    print_speed_table(timings)

    # ---- Parameter agreement ----
    # Require new "axis" column so stale checkpoints are transparently regenerated.
    ckpt_agreement = out_dir / "parameter_agreement.csv"
    if _checkpoint_has_cols(ckpt_agreement, ["axis"]):
        print("\n[Checkpoint] Agreement: loading from file ...", flush=True)
        agreement_df = pd.read_csv(ckpt_agreement)
        print_agreement_table(agreement_df)
    else:
        if _ckpt_exists(ckpt_agreement):
            print("\n[Checkpoint] Agreement: stale format — recomputing ...", flush=True)
        agreement_df = parameter_agreement(all_dfs)
        print_agreement_table(agreement_df)
        agreement_df.to_csv(ckpt_agreement, index=False)

    # ---- Stability ----
    ckpt_stability = out_dir / "stability.csv"
    if _ckpt_exists(ckpt_stability):
        print("\n[Checkpoint] Stability: loading from file ...", flush=True)
        stability_df = pd.read_csv(ckpt_stability)
        print_stability_table(stability_df)
    else:
        print("\nRunning stability evaluation ...", flush=True)
        stability_df = stability_evaluation(
            args.tree, all_dfs, args.n_stability, args.max_workers
        )
        print_stability_table(stability_df)
        stability_df.to_csv(ckpt_stability, index=False)

    # ---- Simulation accuracy (optional) ----
    sim_acc_df = pd.DataFrame()
    if args.eval_sim_accuracy:
        ckpt_sim_acc = out_dir / "sim_accuracy.csv"
        if _ckpt_exists(ckpt_sim_acc):
            print("\n[Checkpoint] Simulation accuracy: loading from file ...", flush=True)
            sim_acc_df = pd.read_csv(ckpt_sim_acc)
            print_sim_accuracy_table(sim_acc_df)
        else:
            print("\nRunning simulation accuracy evaluation ...", flush=True)
            sim_acc_df = simulation_accuracy_evaluation(
                args.tree, all_dfs, args.annotations,
                n_sample=args.sim_accuracy_n,
                masks_dict=masks_dict,
            )
            print_sim_accuracy_table(sim_acc_df)
            if not sim_acc_df.empty:
                sim_acc_df.to_csv(ckpt_sim_acc, index=False)

    # ---- Precision / Recall (optional) ----
    pr_df = pd.DataFrame()
    if args.eval_pr:
        if not args.known_pairs:
            print("[WARN] --eval_pr requires --known_pairs; skipping P/R evaluation.")
        else:
            ckpt_pr = out_dir / "precision_recall.csv"
            if _ckpt_exists(ckpt_pr):
                print("\n[Checkpoint] Precision/recall: loading from file ...", flush=True)
                pr_df = pd.read_csv(ckpt_pr)
            else:
                print("\nRunning precision/recall evaluation ...", flush=True)
                pr_df = precision_recall_evaluation(
                    args.tree, all_dfs, args.annotations, args.known_pairs,
                    masks_dict=masks_dict,
                )
                if not pr_df.empty:
                    pr_df.to_csv(ckpt_pr, index=False)
            if not pr_df.empty:
                _section("Precision / Recall (direction-aware)")
                for assoc, grp in pr_df.groupby("association"):
                    print(f"\n  [{assoc}]")
                    print(grp[["method", "threshold", "TP", "FP", "FN",
                               "precision", "recall", "F1"]]
                          .to_string(index=False, float_format="{:.3f}".format))

    # ---- Method ranking ----
    if not stability_df.empty:
        ranking_df = compute_method_ranking(sim_acc_df, stability_df, pr_df)
        print_method_ranking(ranking_df)
        if not ranking_df.empty:
            ranking_df.to_csv(out_dir / "method_ranking.csv", index=False)

    print(f"\n[OK] All results written to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
