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
import gzip
import json
import os
import pickle
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
    _COUNTING = counting or ['JOINT', 'JOINTP', 'FLOW', 'MARKOV', 'ENTROPY']
    _SUBSIZE  = subsize  or ['ORIGINAL', 'NO_FILTER', 'THRESH']
    _MASKING  = masking  or ['DIST', 'NONE', 'PATH']
    specs = []
    for c, s, m in _product(_COUNTING, _SUBSIZE, _MASKING):
        # JOINTP needs MPPA only for PATH masking (to build path mask); otherwise JOINT-only.
        needs_mppa = (c not in ('JOINT', 'JOINTP')) or (m == 'PATH')
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
                masking=spec.masking,
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
                   help="Subsample N traits (randomly, prevalence 5%%–95%%) for speed (default: all eligible)")
    p.add_argument("--max_workers", default=8, type=int,
                   help="Threads for API-based methods")
    p.add_argument("--n_stability", default=10, type=int,
                   help="Number of simulations (trials) per stability cycle")
    p.add_argument("--n_stability_iters", default=1, type=int,
                   help="Simulate→reconstruct cycles for stability trajectory (default: 1)")
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
                   default=["JOINTP", "FLOW"],
                   help="Counting methods to benchmark (default: JOINT and FLOW)")
    p.add_argument("--subsize_methods", nargs="+",
                   default=["ORIGINAL", "NO_FILTER",],
                   help="Subsize methods to benchmark (default: ORIGINAL and NO_FILTER)")
    p.add_argument("--masking_methods", nargs="+",
                   default=["DIST", "NONE"],
                   help="Simulation masking methods to benchmark (default: DIST and NONE)")
    p.add_argument("--p_threshold", default=0.5, type=float,
                   help="Probability threshold for PATH masking (default: 0.5)")
    p.add_argument("--method", default=None, metavar="COUNTING_SUBSIZE_MASKING",
                   help=("Single method shorthand, e.g. FLOW_THRESH_DIST or "
                         "JOINT_ORIGINAL_PATH. Overrides --counting_methods, "
                         "--subsize_methods, and --masking_methods."))
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
# Shared pre-computation: tree setup + sim_bit lineages (one pass per method)
# ---------------------------------------------------------------------------

def precompute_simulations(
    tree_file: str,
    dfs: dict[str, pd.DataFrame],
    n_sample: int,
    masks_dict: dict = None,
    specs: list = None,
) -> dict:
    """
    Load the tree once and run sim_bit once per method over a sampled trait
    pool.  The returned cache is passed to stability_evaluation and
    simulation_accuracy_evaluation so neither function repeats this work.

    Returns
    -------
    A dict with metadata keys ('_tree', '_upper_bound', '_leaf_names') and
    one entry per method::

        cache[method_name] = {
            'trait_params': DataFrame indexed by gene  (≤n_sample rows),
            'lineages':     ndarray (n_leaves, n_traits) uint64,
        }
    """
    from simphyni.Simulation.simulation import sim_bit

    specs_by_label = {s.label: s for s in specs} if specs else {}

    print("  Precomputing tree setup and simulations ...", flush=True)
    master_tree = Tree(tree_file, format=1)
    label_internal_nodes(master_tree)
    upper_bound = compute_branch_upper_bound(master_tree)
    leaf_names  = [n.name for n in master_tree.iter_leaves()]

    cache: dict = {
        '_tree':        master_tree,
        '_upper_bound': upper_bound,
        '_leaf_names':  leaf_names,
    }

    for method_name, df in dfs.items():
        df = df.copy().dropna(subset=["gains", "losses", "gain_subsize", "loss_subsize"])
        df = df[(df["gain_subsize"] > 0) & (df["loss_subsize"] > 0)]
        df["gain_rate"] = df["gains"] / df["gain_subsize"]
        df["loss_rate"] = df["losses"] / df["loss_subsize"]
        # Keep traits with at least one non-zero rate (gain OR loss).
        # Traits where both rates are zero cannot change state and are uninformative.
        df = df[(df["gain_rate"] > 0) | (df["loss_rate"] > 0)]
        if df.empty:
            continue

        sample = df.sample(n=min(n_sample, len(df)), random_state=0)
        trait_params = sample.set_index("gene") if "gene" in sample.columns else sample

        # Resolve PATH masks for this method
        gm = lm = None
        if masks_dict and method_name in masks_dict:
            full_gm, full_lm = masks_dict[method_name]
            full_gene_order = masks_dict.get(method_name + "_gene_order", [])
            if full_gene_order:
                col_idx = [full_gene_order.index(g) for g in trait_params.index
                           if g in full_gene_order]
                if col_idx:
                    gm = full_gm[:, col_idx]
                    lm = full_lm[:, col_idx]

        lineages = sim_bit(master_tree, trait_params, gain_mask=gm, loss_mask=lm)
        cache[method_name] = {'trait_params': trait_params, 'lineages': lineages}

    return cache


# ---------------------------------------------------------------------------
# Stability: simulate → re-reconstruct → compare parameters
# ---------------------------------------------------------------------------

def stability_evaluation(tree_file: str, dfs: dict[str, pd.DataFrame],
                          n_sims: int, max_workers: int,
                          specs: list = None,
                          masks_dict: dict = None,
                          sims_cache: dict = None,
                          n_iters: int = 1,
                          obs: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each method, pick up to 10 traits and run n_iters simulate→reconstruct
    cycles.  At each iteration the full per-trial distribution of parameters is
    recorded.  The next iteration is seeded from the single most representative
    trial (closest to the mean gains/losses across trials for that gene).

    Parameters
    ----------
    n_sims      : trials decoded per sim_bit call (capped at 64).
    n_iters     : number of simulate→reconstruct cycles.
    specs       : list of MethodSpec — controls counting/subsize/uncertainty.
    masks_dict  : PATH gain/loss masks from build_all_method_dfs.
    sims_cache  : pre-computed tree + iteration-1 lineages from
                  precompute_simulations; used only for iteration 1.

    Returns
    -------
    stability_df   : existing schema (method, gene, parameter, true_value,
                     mean_reestimate, std_reestimate, relative_error) derived
                     from iteration 1 vs 0 — unchanged for ranking/display.
    trajectory_df  : per-trial rows (method, gene, iteration, trial, gains,
                     losses, gain_subsize, loss_subsize, gain_rate, loss_rate,
                     prevalence) — iteration 0 is the seed (prevalence = observed
                     tip fraction when obs is provided, else NaN).
    """
    from simphyni.Simulation.simulation import sim_bit, build_sim_params
    from simphyni.scripts.run_ancestral_reconstruction import reconstruct_trait
    from concurrent.futures import ProcessPoolExecutor

    specs_by_label = {s.label: s for s in specs} if specs else {}

    # Ensure obs index is string so leaf-name lookups work (leaf_names are
    # always strings; a numeric obs.index would silently miss every leaf).
    if obs is not None:
        obs = obs.copy()
        obs.index = obs.index.astype(str)

    # Tree setup — reuse cache if available
    if sims_cache and '_tree' in sims_cache:
        master_tree  = sims_cache['_tree']
        upper_bound  = sims_cache['_upper_bound']
        leaf_names   = sims_cache['_leaf_names']
    else:
        master_tree = Tree(tree_file, format=1)
        label_internal_nodes(master_tree)
        upper_bound = compute_branch_upper_bound(master_tree)
        leaf_names  = [n.name for n in master_tree.iter_leaves()]
    tree_newick = master_tree.write(format=1)

    # ── Pairwise leaf distance matrix (used for per-trial MPD) ──────────────
    # Same root-distance formula as simulation_accuracy_evaluation.
    _root_dist_stab: dict = {}
    for _node in master_tree.traverse('preorder'):
        _root_dist_stab[id(_node)] = (
            0.0 if _node.is_root()
            else _root_dist_stab[id(_node.up)] + _node.dist
        )
    _leaves_ete_stab = list(master_tree.iter_leaves())
    _n_leaves_stab   = len(_leaves_ete_stab)
    _D_stab = np.zeros((_n_leaves_stab, _n_leaves_stab), dtype=float)
    for _i in range(_n_leaves_stab):
        for _j in range(_i + 1, _n_leaves_stab):
            _lca = _leaves_ete_stab[_i].get_common_ancestor(_leaves_ete_stab[_j])
            _d   = (_root_dist_stab[id(_leaves_ete_stab[_i])]
                    + _root_dist_stab[id(_leaves_ete_stab[_j])]
                    - 2.0 * _root_dist_stab[id(_lca)])
            _D_stab[_i, _j] = _D_stab[_j, _i] = _d

    records: list = []            # stability_df rows (relative_error)
    trajectory_records: list = [] # trajectory_df rows (per trial)
    n_test_traits = 10
    n_trials = min(max(n_sims, 1), 64)

    for method_name, df in dfs.items():
        spec = specs_by_label.get(method_name)
        counting = spec.counting if spec else 'JOINT'
        subsize  = spec.subsize  if spec else 'ORIGINAL'
        masking  = spec.masking  if spec else 'DIST'
        uncertainty = "threshold" if counting == 'JOINT' else "marginal"

        # ── Resolve initial trait_params (iteration 0) ──────────────────────
        if sims_cache and method_name in sims_cache:
            entry        = sims_cache[method_name]
            trait_params = entry['trait_params'].iloc[:n_test_traits].copy()
        else:
            df = df.copy().dropna(subset=["gains", "losses", "gain_subsize", "loss_subsize"])
            df = df[(df["gain_subsize"] > 0) & (df["loss_subsize"] > 0)]
            df["gain_rate"] = df["gains"] / df["gain_subsize"]
            df["loss_rate"] = df["losses"] / df["loss_subsize"]
            df = df[(df["gain_rate"] > 0) | (df["loss_rate"] > 0)]
            if df.empty:
                print(f"  [WARN] {method_name}: no valid traits; skipping.", flush=True)
                continue
            sample = df.sample(n=min(n_test_traits, len(df)), random_state=0)
            trait_params = sample.set_index("gene") if "gene" in sample.columns else sample

        if trait_params.empty:
            print(f"  [WARN] {method_name}: no valid traits; skipping.", flush=True)
            continue

        # ── Record iteration 0 (seed values — no simulation) ────────────────
        for gene in trait_params.index:
            if obs is not None and gene in obs.columns:
                seed_prev = float(obs[gene].mean())
            else:
                seed_prev = np.nan
            _gr0 = float(trait_params.loc[gene, "gain_rate"])
            _lr0 = float(trait_params.loc[gene, "loss_rate"])
            # Stationary prevalence implied by the ACR-estimated rates.
            # π₁ = q01/(q01+q10); drift = π₁ − observed prevalence.
            _pi1  = _gr0 / (_gr0 + _lr0 + 1e-12)
            _drift = (_pi1 - seed_prev) if not np.isnan(seed_prev) else np.nan
            _gs0 = float(trait_params.loc[gene, "gain_subsize"])
            _ls0 = float(trait_params.loc[gene, "loss_subsize"])
            # Observed parsimony and MPD — baseline anchors for fan plots
            if obs is not None and gene in obs.columns:
                _obs_states = {n: int(obs.loc[n, gene])
                               for n in leaf_names if n in obs.index}
                _seed_pars = _fitch_parsimony_obs(master_tree, _obs_states)
                _obs_pos = np.array([
                    i for i, n in enumerate(leaf_names)
                    if n in obs.index and int(obs.loc[n, gene]) == 1
                ])
                _seed_mpd = (float(_D_stab[np.ix_(_obs_pos, _obs_pos)].mean())
                             if len(_obs_pos) > 1 else np.nan)
            else:
                _seed_pars = np.nan
                _seed_mpd  = np.nan
            trajectory_records.append({
                "method":               method_name,
                "gene":                 gene,
                "iteration":            0,
                "trial":                0,
                "gains":                float(trait_params.loc[gene, "gains"]),
                "losses":               float(trait_params.loc[gene, "losses"]),
                "gain_subsize":         _gs0,
                "loss_subsize":         _ls0,
                "gain_rate":            _gr0,
                "loss_rate":            _lr0,
                "prevalence":           seed_prev,
                "expected_pi1":         _pi1,
                "prevalence_drift":     _drift,
                "parsimony":            _seed_pars,
                "mpd":                  _seed_mpd,
                "eligible_tree_size":   _gs0 + _ls0,
            })

        # Resolve PATH masks for iteration 1 from the original ACR (masks_dict).
        # For iterations 2+, masks are rebuilt from each iteration's reconstruction
        # results (see seed-selection block below).
        iter_gain_mask = iter_loss_mask = None
        if masking == 'PATH' and masks_dict and method_name in masks_dict:
            full_gm, full_lm = masks_dict[method_name]
            full_gene_order = masks_dict.get(method_name + "_gene_order", [])
            if full_gene_order:
                valid_genes = [g for g in trait_params.index if g in full_gene_order]
                col_idx = [full_gene_order.index(g) for g in valid_genes]
                if col_idx:
                    iter_gain_mask = full_gm[:, col_idx]
                    iter_loss_mask = full_lm[:, col_idx]

        # ── Iteration loop (process pool shared across all iterations) ─────────
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            current_params = trait_params.copy()

            for iteration in range(1, n_iters + 1):
                print(f"  [{method_name}] stability iteration {iteration}/{n_iters} ...",
                      flush=True)

                # Simulation — use cache only for iteration 1
                if iteration == 1 and sims_cache and method_name in sims_cache:
                    iter_lineages = sims_cache[method_name]['lineages'][:, :n_test_traits]
                else:
                    iter_lineages = sim_bit(master_tree, current_params,
                                            gain_mask=iter_gain_mask,
                                            loss_mask=iter_loss_mask)

                # ── Pre-compute per-gene packed metrics (parsimony + MPD) ───────
                # Both are computed over all 64 trials at once from the packed
                # lineages before unpacking individual trials below.
                gene_packed_metrics: dict = {}
                for _col_idx, _gene in enumerate(current_params.index):
                    _packed_col = iter_lineages[:, _col_idx]
                    # Fitch parsimony: vectorized over all 64 packed trials
                    _leaf_packed = {leaf_names[_li]: _packed_col[_li]
                                    for _li in range(len(leaf_names))}
                    _pars_scores = _fitch_parsimony_packed(master_tree, _leaf_packed)
                    # MPD: per trial, subsampled to ≤100 positive tips
                    _trial_mpds = []
                    for _t in range(64):
                        _pos = np.where(
                            ((_packed_col >> np.uint64(_t)) & np.uint64(1)).astype(bool)
                        )[0]
                        if len(_pos) < 2:
                            _trial_mpds.append(np.nan)
                        else:
                            if len(_pos) > 100:
                                _pos = np.random.choice(_pos, 100, replace=False)
                            _trial_mpds.append(float(_D_stab[np.ix_(_pos, _pos)].mean()))
                    gene_packed_metrics[_gene] = {
                        'parsimony': _pars_scores,
                        'mpd':       _trial_mpds,
                    }

                # ── Submit reconstructions ───────────────────────────────────
                # futures_meta: (future, gene, trial, tgr, tlr, prevalence)
                futures_meta = []
                gene_trial_data: dict = {}
                gene_true_gain:  dict = {}
                gene_true_loss:  dict = {}
                for col_idx, gene in enumerate(current_params.index):
                    tgr = float(trait_params.loc[gene, "gain_rate"])  # always vs iter-0
                    tlr = float(trait_params.loc[gene, "loss_rate"])
                    for trial in range(n_trials):
                        tip_states = ((iter_lineages[:, col_idx] >> np.uint64(trial))
                                      & np.uint64(1)).astype(int)
                        prevalence = tip_states.sum() / len(tip_states)
                        # Degenerate (all-0 or all-1): record directly, skip ACR.
                        # Impute interpretable rate values so these trials contribute
                        # to the stability metric rather than being silently dropped:
                        #   all-0 → gain failed  → gain_rate = 0.0, loss_rate = nan
                        #   all-1 → loss failed  → gain_rate = nan, loss_rate = 0.0
                        # eligible_tree_size = 0 in both cases (no eligible branches).
                        if tip_states.sum() == 0 or tip_states.sum() == len(tip_states):
                            is_all_zero = tip_states.sum() == 0
                            gene_true_gain[gene] = tgr
                            gene_true_loss[gene]  = tlr
                            gene_trial_data.setdefault(gene, []).append({
                                "trial":               trial,
                                "gains":               np.nan,
                                "losses":              np.nan,
                                "gain_subsize":        np.nan,
                                "loss_subsize":        np.nan,
                                "gain_rate":           0.0 if is_all_zero else np.nan,
                                "loss_rate":           0.0 if not is_all_zero else np.nan,
                                "dist":                np.nan,
                                "loss_dist":           np.nan,
                                "root_state":          np.nan,
                                "prevalence":          prevalence,
                                "expected_pi1":        0.0 if is_all_zero else 1.0,
                                "parsimony":           0,
                                "mpd":                 np.nan,
                                "eligible_tree_size":  0.0,
                                "_degenerate":         True,
                            })
                            continue
                        sim_series = pd.Series(
                            tip_states.astype(str), index=leaf_names, name=gene,
                        )
                        fut = executor.submit(
                            reconstruct_trait,
                            gene, tree_newick, sim_series, upper_bound,
                            uncertainty, int(tip_states.sum()),
                        )
                        futures_meta.append((fut, gene, trial, tgr, tlr, prevalence))

                # ── Collect results ──────────────────────────────────────────
                for fut, gene, trial, tgr, tlr, prev in futures_meta:
                    try:
                        res = fut.result()
                    except Exception as exc:
                        print(f"  [FAILED] {gene} trial {trial}: {exc}", file=sys.stderr)
                        continue
                    if res is None:
                        continue
                    try:
                        res_df = pd.DataFrame([res])
                        tp = build_sim_params(res_df, counting, subsize,
                                              no_threshold=(masking == 'NONE'),
                                              masking=masking)
                        gs = float(tp['gain_subsize'].iloc[0])
                        ls = float(tp['loss_subsize'].iloc[0])
                        gc = float(tp['gains'].iloc[0])
                        lc = float(tp['losses'].iloc[0])
                    except (KeyError, IndexError, ValueError):
                        continue
                    gene_true_gain[gene] = tgr
                    gene_true_loss[gene] = tlr
                    _pm = gene_packed_metrics.get(gene, {})
                    _gr_t = gc / gs if gs > 0 else np.nan
                    _lr_t = lc / ls if ls > 0 else np.nan
                    _pi1_t = (_gr_t / (_gr_t + _lr_t + 1e-12)
                              if not (np.isnan(_gr_t) or np.isnan(_lr_t)) else np.nan)
                    gene_trial_data.setdefault(gene, []).append({
                        "trial":               trial,
                        "gains":               gc,
                        "losses":              lc,
                        "gain_subsize":        gs,
                        "loss_subsize":        ls,
                        "gain_rate":           _gr_t,
                        "loss_rate":           _lr_t,
                        "expected_pi1":        _pi1_t,
                        "dist":                float(tp['dist'].iloc[0]),
                        "loss_dist":           float(tp['loss_dist'].iloc[0]),
                        "root_state":          float(tp['root_state'].iloc[0]),
                        "prevalence":          prev,
                        "parsimony":           int(_pm['parsimony'][trial]) if 'parsimony' in _pm else np.nan,
                        "mpd":                 _pm['mpd'][trial] if 'mpd' in _pm else np.nan,
                        "eligible_tree_size":  gs + ls,
                        "_degenerate":         False,
                        # PATH masks from this trial's MPPA — used to rebuild
                        # iter_gain_mask / iter_loss_mask for the next iteration.
                        "_gain_mask_1d": res.get("_gain_mask_1d"),
                        "_loss_mask_1d": res.get("_loss_mask_1d"),
                    })

                # ── Record trajectory rows + stability_df (iteration 1 only) ─
                # Always emit at least one row per gene per iteration so the
                # trajectory has no gaps (use NaN when all trials failed).
                for gene in current_params.index:
                    trial_list = gene_trial_data.get(gene, [])
                    if trial_list:
                        for td in trial_list:
                            trajectory_records.append({
                                "method":               method_name,
                                "gene":                 gene,
                                "iteration":            iteration,
                                "trial":                td["trial"],
                                "gains":                td["gains"],
                                "losses":               td["losses"],
                                "gain_subsize":         td["gain_subsize"],
                                "loss_subsize":         td["loss_subsize"],
                                "gain_rate":            td["gain_rate"],
                                "loss_rate":            td["loss_rate"],
                                "prevalence":           td["prevalence"],
                                "parsimony":            td.get("parsimony", np.nan),
                                "mpd":                  td.get("mpd", np.nan),
                                "eligible_tree_size":   td.get("eligible_tree_size", np.nan),
                                "expected_pi1":         td.get("expected_pi1", np.nan),
                                "prevalence_drift":     (
                                    td["expected_pi1"] - td["prevalence"]
                                    if not (np.isnan(td.get("expected_pi1", np.nan))
                                            or np.isnan(td["prevalence"]))
                                    else np.nan
                                ),
                            })
                    else:
                        # All trials were dropped — record NaN so the iteration
                        # is present in the trajectory for every gene.
                        trajectory_records.append({
                            "method":               method_name,
                            "gene":                 gene,
                            "iteration":            iteration,
                            "trial":                np.nan,
                            "gains":                np.nan,
                            "losses":               np.nan,
                            "gain_subsize":         np.nan,
                            "loss_subsize":         np.nan,
                            "gain_rate":            np.nan,
                            "loss_rate":            np.nan,
                            "prevalence":           np.nan,
                            "parsimony":            np.nan,
                            "mpd":                  np.nan,
                            "eligible_tree_size":   np.nan,
                            "expected_pi1":         np.nan,
                            "prevalence_drift":     np.nan,
                        })

                    if iteration == 1 and gene in gene_true_gain:
                        q01 = gene_true_gain[gene]
                        q10 = gene_true_loss[gene]
                        # Use ALL trials (including imputed degenerates) so that
                        # methods which drive traits to fixation are penalised.
                        # nanmean/nanstd skip NaN entries (e.g. loss_rate on
                        # all-0 trials) while including the imputed 0.0 values.
                        gr_vals = [td["gain_rate"] for td in trial_list]
                        lr_vals = [td["loss_rate"] for td in trial_list]
                        _gr_mean = float(np.nanmean(gr_vals)) if gr_vals else np.nan
                        _lr_mean = float(np.nanmean(lr_vals)) if lr_vals else np.nan
                        # Fraction of trials that were degenerate (all-0 or all-1)
                        _n_degen = sum(1 for td in trial_list if td.get("_degenerate", False))
                        _degen_frac = _n_degen / len(trial_list) if trial_list else np.nan
                        # Symmetric relative error: 2|est-true|/(|est|+|true|+ε).
                        _sym_rel_err = lambda est, true: (
                            2.0 * abs(est - true) / (abs(est) + abs(true) + 1e-12)
                        )
                        records.append({
                            "method": method_name, "gene": gene,
                            "parameter": "gain_rate",
                            "true_value":        q01,
                            "mean_reestimate":   _gr_mean,
                            "std_reestimate":    float(np.nanstd(gr_vals)),
                            "relative_error":    _sym_rel_err(_gr_mean, q01) if not np.isnan(_gr_mean) else np.nan,
                            "degenerate_fraction": _degen_frac,
                        })
                        records.append({
                            "method": method_name, "gene": gene,
                            "parameter": "loss_rate",
                            "true_value":        q10,
                            "mean_reestimate":   _lr_mean,
                            "std_reestimate":    float(np.nanstd(lr_vals)),
                            "relative_error":    _sym_rel_err(_lr_mean, q10) if not np.isnan(_lr_mean) else np.nan,
                            "degenerate_fraction": _degen_frac,
                        })

                # ── Select representative trial per gene → seed next iteration ─
                # Selection operates in 5-dim normalised rate/parameter space
                # over ALL trials (including imputed degenerates) so that methods
                # which drive traits to fixation are not rewarded with a
                # cherry-picked intermediate-prevalence seed.
                #
                # Dimensions: gain_rate, loss_rate, dist, loss_dist,
                #             eligible_tree_size (= gain_subsize + loss_subsize)
                # Normalisation: each dimension divided by the iter-0 seed value
                # to express deviations as relative fractions.  NaN in either
                # target or trial for a given dimension → that dimension is
                # excluded from the distance for that trial.
                _SEED_DIMS = ["gain_rate", "loss_rate", "dist", "loss_dist",
                              "eligible_tree_size"]
                _EPS_SEED  = 1e-12

                gene_new_masks: dict = {}  # gene → (gain_mask_1d, loss_mask_1d)
                for gene, trial_list in gene_trial_data.items():
                    if not trial_list:
                        continue

                    # Gather all-trials values per dimension (NaN for missing)
                    dim_vals: dict = {d: [] for d in _SEED_DIMS}
                    for td in trial_list:
                        dim_vals["gain_rate"].append(td.get("gain_rate", np.nan))
                        dim_vals["loss_rate"].append(td.get("loss_rate", np.nan))
                        dim_vals["dist"].append(td.get("dist", np.nan))
                        dim_vals["loss_dist"].append(td.get("loss_dist", np.nan))
                        _ets = td.get("eligible_tree_size", np.nan)
                        dim_vals["eligible_tree_size"].append(_ets)

                    # Target = nanmean per dimension across all trials
                    target: dict = {d: float(np.nanmean(dim_vals[d])) for d in _SEED_DIMS}

                    # Seed normalisation factors (iter-0 current_params values)
                    _gs_seed = float(current_params.loc[gene, "gain_subsize"])
                    _ls_seed = float(current_params.loc[gene, "loss_subsize"])
                    seed_norm: dict = {
                        "gain_rate":          float(current_params.loc[gene, "gain_rate"]),
                        "loss_rate":          float(current_params.loc[gene, "loss_rate"]),
                        "dist":               float(current_params.loc[gene, "dist"]),
                        "loss_dist":          float(current_params.loc[gene, "loss_dist"]),
                        "eligible_tree_size": _gs_seed + _ls_seed,
                    }

                    # Compute normalised RMS distance for each trial
                    distances = []
                    for i, td in enumerate(trial_list):
                        sq_terms = []
                        for d in _SEED_DIMS:
                            t_val = target[d]
                            v     = dim_vals[d][i]
                            norm  = seed_norm[d]
                            if np.isnan(t_val) or np.isnan(v):
                                continue   # dimension excluded if either is NaN
                            sq_terms.append(((v - t_val) / (abs(norm) + _EPS_SEED)) ** 2)
                        distances.append(float(np.mean(sq_terms)) if sq_terms else np.inf)

                    best_idx = int(np.argmin(distances))
                    best = trial_list[best_idx]

                    # Only update current_params if the best trial is non-degenerate;
                    # if all trials are degenerate, carry forward the previous seed.
                    if not best.get("_degenerate", True):
                        for col in ["gains", "losses", "gain_subsize", "loss_subsize",
                                    "dist", "loss_dist", "root_state"]:
                            if not np.isnan(best.get(col, np.nan)):
                                current_params.loc[gene, col] = best[col]
                        current_params.loc[gene, "gain_rate"] = best["gain_rate"]
                        current_params.loc[gene, "loss_rate"] = best["loss_rate"]

                    # Collect the best trial's PATH masks for next iteration's sim_bit
                    if best.get("_gain_mask_1d") is not None:
                        gene_new_masks[gene] = (best["_gain_mask_1d"], best["_loss_mask_1d"])

                # Rebuild iter_gain_mask / iter_loss_mask from best-trial masks.
                # This keeps the simulation's eligible branches aligned with the
                # most recent reconstruction for every subsequent iteration.
                if masking == 'PATH' and gene_new_masks:
                    genes_ordered = list(current_params.index)
                    pairs = [gene_new_masks.get(g) for g in genes_ordered]
                    if all(p is not None for p in pairs):
                        iter_gain_mask = np.column_stack([p[0] for p in pairs])
                        iter_loss_mask = np.column_stack([p[1] for p in pairs])

    return pd.DataFrame(records), pd.DataFrame(trajectory_records)

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
    sims_cache: dict = None,
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

    # Tree setup — reuse cache if available
    if sims_cache and '_tree' in sims_cache:
        master_tree = sims_cache['_tree']
        leaf_list   = sims_cache['_leaf_names']
    else:
        master_tree = Tree(tree_file, format=1)
        label_internal_nodes(master_tree)
        leaf_list = [n.name for n in master_tree.iter_leaves()]

    obs = pd.read_csv(ann_file, index_col=0)
    obs.index = obs.index.astype(str)

    n_leaves   = len(leaf_list)
    leaf_index = {name: i for i, name in enumerate(leaf_list)}

    # ── Pairwise distance matrix (precompute once) ──────────────────────────
    print("  Precomputing pairwise leaf distances ...", flush=True)
    leaves_ete = list(master_tree.iter_leaves())
    # Single preorder pass: root-to-node distances
    _root_dist: dict = {}
    for node in master_tree.traverse('preorder'):
        _root_dist[id(node)] = (0.0 if node.is_root()
                                else _root_dist[id(node.up)] + node.dist)
    # LCA formula: d(i,j) = root_dist[i] + root_dist[j] - 2*root_dist[lca(i,j)]
    D = np.zeros((n_leaves, n_leaves), dtype=float)
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            lca = leaves_ete[i].get_common_ancestor(leaves_ete[j])
            d = (_root_dist[id(leaves_ete[i])]
                 + _root_dist[id(leaves_ete[j])]
                 - 2.0 * _root_dist[id(lca)])
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
        # Use pre-computed simulations if available; otherwise run fresh
        if sims_cache and method_name in sims_cache:
            entry        = sims_cache[method_name]
            trait_params = entry['trait_params']
            lineages     = entry['lineages']
        else:
            df_clean = df.dropna(subset=["gains", "losses", "gain_subsize", "loss_subsize"])
            df_clean = df_clean[(df_clean["gain_subsize"] > 0) & (df_clean["loss_subsize"] > 0)]
            if df_clean.empty:
                print(f"    [WARN] no valid traits; skipping.", flush=True)
                continue

            sample = df_clean.sample(n=min(n_sample, len(df_clean)), random_state=0)
            trait_params = sample.set_index("gene") if "gene" in sample.columns else sample

            gm = lm = None
            if masks_dict and method_name in masks_dict:
                full_gm, full_lm = masks_dict[method_name]
                full_gene_order = masks_dict.get(method_name + "_gene_order", [])
                if full_gene_order:
                    col_idx = [full_gene_order.index(g) for g in trait_params.index
                               if g in full_gene_order]
                    if col_idx:
                        gm = full_gm[:, col_idx]
                        lm = full_lm[:, col_idx]

            lineages = sim_bit(master_tree, trait_params, gain_mask=gm, loss_mask=lm)

        if trait_params.empty:
            print(f"    [WARN] no valid traits; skipping.", flush=True)
            continue

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

    Accepts either column naming convention:
      - T1, T2, direction   (legacy SimPhyNI format)
      - trait1, trait2, direction  (benchmark pair_labels format)

    direction must be 1 (positive/synergistic), -1 (negative/antagonistic),
    or 0 (null — explicitly non-interacting).

    Returns a dict with keys:
      'positive'     — set of (t1, t2, dir) with direction == 1
      'negative'     — set of (t1, t2, dir) with direction == -1
      'null'         — set of (t1, t2, dir) with direction == 0
      'any'          — union of positive and negative (direction-agnostic)
      'all_labeled'  — canonical (t1, t2) pairs across all directions;
                       used to restrict predictions to the labeled universe
                       so null pairs predicted as significant count as FP

    Each directional set uses canonical order (alphabetic sort on t1, t2).
    """
    known = pd.read_csv(known_pairs_file)

    # Normalise column names: accept trait1/trait2 or T1/T2
    if "trait1" in known.columns and "trait2" in known.columns:
        known = known.rename(columns={"trait1": "T1", "trait2": "T2"})
    elif "T1" in known.columns and "T2" in known.columns:
        pass  # already correct
    else:
        raise ValueError(
            f"known_pairs file must have columns (T1, T2) or (trait1, trait2) plus direction. "
            f"Found: {list(known.columns)}"
        )

    if "direction" not in known.columns:
        raise ValueError("known_pairs file is missing the 'direction' column.")

    known["T1"] = known["T1"].astype(str)
    known["T2"] = known["T2"].astype(str)
    known["direction"] = known["direction"].astype(int)

    def _sym(subset):
        s = set()
        for _, row in subset.iterrows():
            t1, t2 = sorted([str(row["T1"]), str(row["T2"])])
            s.add((t1, t2, int(row["direction"])))
        return s

    pos  = _sym(known[known["direction"] ==  1])
    neg  = _sym(known[known["direction"] == -1])
    null = _sym(known[known["direction"] ==  0])

    # Canonical identity set across all labeled directions — used to restrict
    # the prediction universe so null pairs predicted as significant are FP
    all_labeled = {(t1, t2) for t1, t2, _ in (pos | neg | null)}

    return {
        "positive":    pos,
        "negative":    neg,
        "null":        null,
        "any":         pos | neg,
        "all_labeled": all_labeled,
    }


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
    on all labeled pairs and compute direction-aware precision / recall.

    Two evaluation modes are recorded per method:

    1. Continuous PR AUC — uses -log10(pval_naive) as a ranked score with
       sklearn.precision_recall_curve.  Scores are zeroed out for predictions
       in the wrong direction so only correctly-directed hits contribute.
       Null pairs (direction == 0) are included in the universe as ground-truth
       negatives; a null pair predicted as significant counts as FP.

    2. Discrete threshold P/R — binary predictions at pval_bh < {0.05, 0.01, 0.001}
       for backward compatibility.

    Evaluates two subsets per mode:
      positive  — synergistic (direction == 1)
      negative  — antagonistic (direction == -1)
    """
    try:
        from simphyni.Simulation.tree_simulator import TreeSimulator
        from sklearn.metrics import precision_recall_curve, auc as sk_auc
    except ImportError as exc:
        print(f"[WARN] Could not import required module: {exc}")
        return pd.DataFrame()

    known_sets = _load_known_pairs(known_pairs_file)
    thresholds = [0.05, 0.01, 0.001]
    eps = np.finfo(float).eps

    # Build explicit pairs list from known_pairs_file so the simulation tests
    # exactly the labeled pairs (pos + neg + null) regardless of column ordering
    # in the annotation file.
    _pl = pd.read_csv(known_pairs_file)
    if "trait1" in _pl.columns:
        _pl = _pl.rename(columns={"trait1": "T1", "trait2": "T2"})

    records = []

    for method_name, df in dfs.items():
        # Resolve certainty masks for this method (if any)
        gm = lm = gene_order_for_mask = None
        if masks_dict and method_name in masks_dict:
            gm, lm = masks_dict[method_name]
            gene_order_for_mask = masks_dict.get(method_name + "_gene_order", []) or None

        sim = TreeSimulator(tree_file, df, ann_file)
        sim.initialize_simulation_parameters(pre_filter=False, run_traits=1)

        # Build pair list from pair_labels — avoids any dependency on column order
        obs_cols = set(sim.obsdf_modified.columns)
        pairs = [
            (str(r["T1"]), str(r["T2"]))
            for _, r in _pl.iterrows()
            if str(r["T1"]) in obs_cols and str(r["T2"]) in obs_cols
        ]
        if not pairs:
            print(f"[WARN] {method_name}: no labeled pairs found in obsdf columns; skipping.")
            continue

        sim.pairs, sim.obspairs = sim._get_pair_data2(sim.obsdf_modified, pairs)
        # total_tests drives the multiple-testing correction padding; sync it
        # to the actual pairs being evaluated (set during initialize_simulation_parameters
        # on a different pairing, so it would otherwise be wrong).
        sim.total_tests = len(sim.pairs)
        sim.run_simulation(gain_mask=gm, loss_mask=lm, gene_order=gene_order_for_mask)
        res = sim.get_results()

        # ── Continuous PR AUC ────────────────────────────────────────────────
        res = res.copy()
        res["T1"] = res["T1"].astype(str)
        res["T2"] = res["T2"].astype(str)
        res["direction"] = res["direction"].astype(int)

        # Canonical pair order consistent with known_sets
        canonical = res.apply(lambda r: tuple(sorted([r["T1"], r["T2"]])), axis=1)
        res["_ct1"] = canonical.str[0]
        res["_ct2"] = canonical.str[1]
        res["_nlp"] = -np.log10(res["pval_naive"].fillna(1.0).clip(lower=eps))

        for target_dir, assoc_name in [(1, "positive"), (-1, "negative")]:
            known_directional = known_sets["positive" if target_dir == 1 else "negative"]
            y_true, y_score = [], []
            for _, row in res.iterrows():
                ct1, ct2 = row["_ct1"], row["_ct2"]
                # Ground truth: 1 if this is a known pair in the target direction
                y_true.append(1 if (ct1, ct2, target_dir) in known_directional else 0)
                # Score: -log10(pval_naive) only for target-direction predictions;
                # zero otherwise so wrong-direction calls don't boost score
                y_score.append(float(row["_nlp"]) if int(row["direction"]) == target_dir else 0.0)

            if sum(y_true) == 0:
                pr_auc_val = float("nan")
            else:
                prec_c, rec_c, _ = precision_recall_curve(y_true, y_score)
                pr_auc_val = float(sk_auc(rec_c, prec_c))

            records.append({
                "method":      method_name,
                "threshold":   "continuous",
                "association": assoc_name,
                "PR_AUC":      pr_auc_val,
                "TP": None, "FP": None, "FN": None,
                "precision": None, "recall": None, "F1": None,
            })

        # ── Discrete threshold P/R (backward compat) ────────────────────────
        known_neg_pairs = {(t1, t2) for t1, t2, _ in known_sets["negative"]}
        known_pos_pairs = {(t1, t2) for t1, t2, _ in known_sets["positive"]}
        known_any       = {(a, b, 0) for a, b, _ in known_sets["any"]}

        for thresh in thresholds:
            sig = res[res["pval_bh"] < thresh].copy()

            pred_dir = set()
            pred_any = set()
            for _, row in sig.iterrows():
                d   = int(row["direction"])
                t1, t2 = row["_ct1"], row["_ct2"]
                pred_dir.add((t1, t2, d))
                pred_any.add((t1, t2, 0))

            pred_pos = {(t1, t2, d) for t1, t2, d in pred_dir
                        if d == 1 and (t1, t2) not in known_neg_pairs}
            pred_neg = {(t1, t2, d) for t1, t2, d in pred_dir
                        if d == -1 and (t1, t2) not in known_pos_pairs}

            for subset_name, pred_set, known_set in [
                ("positive", pred_pos, known_sets["positive"]),
                ("negative", pred_neg, known_sets["negative"]),
                ("any",      pred_any, known_any),
            ]:
                stats = _pr_stats(pred_set, known_set)
                records.append({
                    "method":      method_name,
                    "threshold":   thresh,
                    "association": subset_name,
                    "PR_AUC":      None,
                    **stats,
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
    agg_cols = {"relative_error": ["mean", "median", "std"]}
    if "degenerate_fraction" in df.columns:
        agg_cols["degenerate_fraction"] = ["mean"]
    summary = (df.groupby(["method", "parameter"])
                  .agg(agg_cols)
                  .reset_index())
    summary.columns = [
        "_".join(c).strip("_") if c[1] else c[0]
        for c in summary.columns
    ]
    rename_map = {
        "relative_error_mean":   "mean_rel_err",
        "relative_error_median": "median_rel_err",
        "relative_error_std":    "std_rel_err",
        "degenerate_fraction_mean": "mean_degen_frac",
    }
    summary = summary.rename(columns=rename_map)
    print(summary.to_string(index=False, float_format="{:.4f}".format))

# ---------------------------------------------------------------------------
# Method ranking
# ---------------------------------------------------------------------------

def compute_method_ranking(
    sim_acc_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    pr_df: pd.DataFrame,
    trajectory_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Rank all methods and track diagnostic metrics.

    Composite ranking dimensions (rank_* columns contribute to composite_rank):
      sim_prev_error   — mean |sim_prevalence − obs_prevalence|        (lower = better)
      parsimony_calib  — mean |parsimony_ratio − 1|                    (lower = better)
      mpd_calib        — mean |mpd_ratio − 1|                          (lower = better)
      pr_f1            — F1 @ threshold 0.001, association="any"       (higher = better)
      pr_auc           — trapezoidal PR-AUC across all thresholds,
                         association="any"                             (higher = better)

    Tracked but not ranked (informational only):
      clade_jsd        — mean clade_profile_jsd
      stability_gain   — mean gain_rate relative_error
      stability_loss   — mean loss_rate relative_error
      degen_fraction   — mean fraction of degenerate simulation trials
      stationary_drift — mean |π₁ − observed_prevalence| at iter-0

    composite_rank = mean of the 5 ranking dimension ranks (1 = best).
    """
    # ── Dimensions that enter the composite rank ─────────────────────────────
    _RANK_LOWER  = {"sim_prev_error", "parsimony_calib", "mpd_calib"}
    _RANK_HIGHER = {"pr_f1", "pr_auc"}

    records: dict[str, dict] = {}

    # Simulation accuracy
    if not sim_acc_df.empty:
        for method, grp in sim_acc_df.groupby("method"):
            records.setdefault(method, {})
            records[method]["sim_prev_error"]  = grp["prevalence_error"].mean()
            records[method]["parsimony_calib"] = (grp["parsimony_ratio"] - 1).abs().mean()
            records[method]["mpd_calib"]       = (grp["mpd_ratio"] - 1).abs().mean()
            records[method]["clade_jsd"]       = grp["clade_profile_jsd"].mean()

    # Stability (tracked, not ranked)
    if not stability_df.empty:
        for param, col in [("gain_rate", "stability_gain"), ("loss_rate", "stability_loss")]:
            sub = stability_df[stability_df["parameter"] == param]
            for method, val in sub.groupby("method")["relative_error"].mean().items():
                records.setdefault(method, {})[col] = val
        if "degenerate_fraction" in stability_df.columns:
            for method, val in stability_df.groupby("method")["degenerate_fraction"].mean().items():
                records.setdefault(method, {})["degen_fraction"] = val

    # Stationary prevalence drift (tracked, not ranked)
    if trajectory_df is not None and not trajectory_df.empty:
        if "prevalence_drift" in trajectory_df.columns and "iteration" in trajectory_df.columns:
            iter0 = trajectory_df[trajectory_df["iteration"] == 0].copy()
            if not iter0.empty:
                iter0["_abs_drift"] = iter0["prevalence_drift"].abs()
                for method, val in iter0.groupby("method")["_abs_drift"].mean().items():
                    records.setdefault(method, {})["stationary_drift"] = val

    # Precision / recall — F1 at threshold 0.001 and continuous PR-AUC
    if not pr_df.empty:
        # Discrete rows (threshold is numeric): used for F1 at threshold 0.001
        pr_discrete = pr_df[
            (pr_df["association"] == "any") & (pr_df["threshold"] != "continuous")
        ].copy()
        if not pr_discrete.empty:
            pr_discrete["_td"] = (pr_discrete["threshold"].astype(float) - 0.001).abs()
            best_thresh = pr_discrete["_td"].min()
            for _, row in pr_discrete[pr_discrete["_td"] == best_thresh].iterrows():
                records.setdefault(row["method"], {})["pr_f1"] = row["F1"]

        # Continuous rows: PR-AUC computed via sklearn precision_recall_curve
        # stored per (positive / negative) direction; average across both.
        pr_cont = pr_df[pr_df["threshold"] == "continuous"].copy()
        if not pr_cont.empty:
            for method, grp in pr_cont.groupby("method"):
                auc_vals = grp["PR_AUC"].dropna().values
                if len(auc_vals) > 0:
                    records.setdefault(method, {})["pr_auc"] = float(np.nanmean(auc_vals))

    if not records:
        return pd.DataFrame()

    raw_df = pd.DataFrame.from_dict(records, orient="index")
    raw_df.index.name = "method"

    # Create rank_* columns only for composite-ranking dimensions
    rank_df = raw_df.copy()
    for col in _RANK_LOWER:
        if col in rank_df.columns:
            rank_df[f"rank_{col}"] = rank_df[col].rank(ascending=True, na_option="bottom")
    for col in _RANK_HIGHER:
        if col in rank_df.columns:
            rank_df[f"rank_{col}"] = rank_df[col].rank(ascending=False, na_option="bottom")

    rank_cols = [c for c in rank_df.columns if c.startswith("rank_")]
    rank_df["composite_rank"] = (
        rank_df[rank_cols].mean(axis=1) if rank_cols else np.nan
    )

    return rank_df.sort_values("composite_rank", na_position="last").reset_index()


def print_method_ranking(df: pd.DataFrame):
    _section("Method Ranking Summary")
    if df.empty:
        print("  No ranking data.")
        return

    rank_cols = [c for c in df.columns if c.startswith("rank_")]

    # Composite ranking dimensions (have rank_* columns)
    ranked_metrics = [c.removeprefix("rank_") for c in rank_cols]
    # Diagnostic metrics tracked but not part of composite rank
    _ALL_DIAG = {"clade_jsd", "stability_gain", "stability_loss",
                 "degen_fraction", "stationary_drift"}
    diag_cols = [c for c in _ALL_DIAG if c in df.columns]

    # Full ranking table
    print("\n  Composite ranking (sorted by composite rank):")
    disp = df[["method", "composite_rank"] + rank_cols].copy()
    print(disp.to_string(index=False, float_format="{:.2f}".format))

    # Ranking metric values
    if ranked_metrics:
        avail = [c for c in ranked_metrics if c in df.columns]
        if avail:
            print("\n  Ranking metric values:")
            print(df[["method"] + avail].to_string(
                index=False, float_format="{:.4f}".format))

    # Diagnostic metric values (not ranked)
    if diag_cols:
        print("\n  Diagnostic metrics (tracked, not ranked):")
        print(df[["method"] + diag_cols].to_string(
            index=False, float_format="{:.4f}".format))

    # Top 5 overall
    print("\n  Top 5 overall:")
    for _, row in df.head(5).iterrows():
        print(f"    #{row['composite_rank']:.1f}  {row['method']}")

    # Best per ranking dimension
    dim_labels = {
        "rank_sim_prev_error":  "Prevalence calibration",
        "rank_parsimony_calib": "Parsimony calibration",
        "rank_mpd_calib":       "MPD calibration",
        "rank_pr_f1":           "Precision/Recall F1",
        "rank_pr_auc":          "Precision/Recall AUC",
    }
    available = [(rc, lbl) for rc, lbl in dim_labels.items() if rc in df.columns]
    if available:
        print("\n  Best per ranking dimension:")
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

    # ---- Parse --method shorthand (overrides per-dimension args) ----
    if args.method is not None:
        _raw = args.method.upper()
        _matched = None
        for _c in ('JOINT', 'JOINTP', 'FLOW', 'MARKOV', 'ENTROPY'):
            for _s in ('ORIGINAL', 'NO_FILTER', 'THRESH'):
                for _m in ('DIST', 'NONE', 'PATH'):
                    if _raw == f"{_c}_{_s}_{_m}":
                        _matched = (_c, _s, _m)
        if _matched is None:
            raise SystemExit(
                f"[ERROR] --method {args.method!r} does not match any valid "
                f"COUNTING_SUBSIZE_MASKING combination. "
                f"Example: FLOW_THRESH_DIST or JOINT_ORIGINAL_PATH"
            )
        args.counting_methods = [_matched[0]]
        args.subsize_methods  = [_matched[1]]
        args.masking_methods  = [_matched[2]]
        print(f"[--method] Single combination: {_matched[0]}_{_matched[1]}_{_matched[2]}",
              flush=True)

    # ---- Build active method specs from CLI selectors ----
    active_specs = _build_method_specs(
        counting=args.counting_methods,
        subsize=args.subsize_methods,
        masking=args.masking_methods,
    )
    needs_mppa = any(s.needs_mppa for s in active_specs)
    acr_uncertainty = "marginal" if needs_mppa else "threshold"
    print(f"Active method combinations: {len(active_specs)} "
          f"({len(args.counting_methods)}×{len(args.subsize_methods)}×{len(args.masking_methods)})",
          flush=True)

    # ---- Optional: delete all checkpoints for a fresh run ----
    _CKPT_FILES = [
        "params_A_legacy.csv", "params_all_acr.csv", "params_all_acr.mppdfs.pkl.gz",
        "speed_timings.json", "parameter_agreement.csv", "stability.csv",
        "sim_accuracy.csv", "precision_recall.csv", "method_ranking.csv",
        "selected_traits.txt",
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
    all_traits = list(obs.columns)
    traits_ckpt = out_dir / "selected_traits.txt"

    if traits_ckpt.exists():
        # Reload the exact trait set from a previous run for checkpoint consistency
        traits = [t for t in traits_ckpt.read_text().splitlines() if t]
        print(f"Loaded {len(traits)} traits from checkpoint ({traits_ckpt.name})",
              flush=True)
    else:
        # Filter to traits with prevalence in [5%, 95%]
        n_leaves = obs.shape[0]
        prev = obs.mean(axis=0)
        eligible = [t for t in all_traits if 0.05 <= prev[t] <= 0.95]
        if args.traits and args.traits < len(eligible):
            rng = np.random.default_rng(seed=0)
            traits = list(rng.choice(eligible, size=args.traits, replace=False))
        else:
            traits = eligible
        traits_ckpt.write_text("\n".join(traits))
        print(f"Selected {len(traits)} traits with prevalence in [5%, 95%]"
              f"{f' (sampled from {len(eligible)})' if args.traits else ''}",
              flush=True)
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

    # ---- Combined JOINT+MPPA ACR run → all variant columns ----
    ckpt_acr = out_dir / "params_all_acr.csv"
    if needs_mppa:
        # gains_flow_path is the sentinel for the new dual-call format;
        # its absence means an old checkpoint with corrupt standard columns.
        _ACR_REQUIRED = ["gains_flow", "gains_markov", "gains_entropy",
                         "gain_subsize_nofilter", "gain_subsize_entropy",
                         "gains_flow_path"]
    else:
        _ACR_REQUIRED = ["gains", "losses", "gain_subsize"]
    acr_valid = _checkpoint_has_cols(ckpt_acr, _ACR_REQUIRED)

    if acr_valid:
        print("\n[2/2] All-ACR params: loading from checkpoint ...", flush=True)
        df_acr = pd.read_csv(ckpt_acr)
        print(f"  Loaded {len(df_acr)} traits ({len(df_acr.columns)} columns)", flush=True)
        mp_dfs_pkl = ckpt_acr.with_suffix('.mppdfs.pkl.gz')
        if mp_dfs_pkl.exists():
            with gzip.open(mp_dfs_pkl, 'rb') as _f:
                mp_dfs_all = pickle.load(_f)
            print(f"  Loaded mp_dfs for {len(mp_dfs_all)} traits from checkpoint.",
                  flush=True)
        else:
            mp_dfs_all = None
            print("  [NOTE] mp_dfs checkpoint not found — PATH masking methods will "
                  "be skipped.\n         Run with --force_rerun to regenerate.",
                  flush=True)
    else:
        if _ckpt_exists(ckpt_acr):
            print("\n[2/2] All-ACR checkpoint is stale — re-running ...", flush=True)
            ckpt_acr.unlink()
            _mp_pkl = ckpt_acr.with_suffix('.mppdfs.pkl.gz')
            if _mp_pkl.exists():
                _mp_pkl.unlink()
        else:
            label = "JOINT+MPPA" if needs_mppa else "JOINT-only"
            print(f"\n[2/2] Running {label} reconstruction "
                  f"(uncertainty={acr_uncertainty}) ...", flush=True)
        df_acr, t_acr, mp_dfs_all = run_api_method(
            args.tree, args.annotations, traits,
            uncertainty=acr_uncertainty, max_workers=args.max_workers,
        )
        timings["all_acr"] = t_acr
        df_acr.to_csv(ckpt_acr, index=False)
        if mp_dfs_all:
            _mp_pkl = ckpt_acr.with_suffix('.mppdfs.pkl.gz')
            with gzip.open(_mp_pkl, 'wb') as _f:
                pickle.dump(mp_dfs_all, _f)
            print(f"  Saved mp_dfs for {len(mp_dfs_all)} traits.", flush=True)
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

    # ---- Pre-compute tree setup + sim_bit lineages (shared by stability and sim_accuracy) ----
    _need_fresh_stability = not _ckpt_exists(out_dir / "stability.csv")
    _need_fresh_sim_acc   = args.eval_sim_accuracy and not _ckpt_exists(out_dir / "sim_accuracy.csv")
    sims_cache = None
    if _need_fresh_stability or _need_fresh_sim_acc:
        n_presample = max(10, args.sim_accuracy_n if _need_fresh_sim_acc else 0)
        sims_cache = precompute_simulations(
            args.tree, all_dfs, n_presample,
            masks_dict=masks_dict, specs=active_specs,
        )

    # ---- Stability ----
    ckpt_stability   = out_dir / "stability.csv"
    ckpt_trajectory  = out_dir / "stability_trajectory.csv"
    if _ckpt_exists(ckpt_stability):
        print("\n[Checkpoint] Stability: loading from file ...", flush=True)
        stability_df  = pd.read_csv(ckpt_stability)
        trajectory_df = pd.read_csv(ckpt_trajectory) if ckpt_trajectory.exists() else pd.DataFrame()
        print_stability_table(stability_df)
    else:
        print("\nRunning stability evaluation ...", flush=True)
        stability_df, trajectory_df = stability_evaluation(
            args.tree, all_dfs, args.n_stability, args.max_workers,
            specs=active_specs, masks_dict=masks_dict,
            sims_cache=sims_cache,
            n_iters=args.n_stability_iters,
            obs=obs,
        )
        print_stability_table(stability_df)
        stability_df.to_csv(ckpt_stability, index=False)
        if not trajectory_df.empty:
            trajectory_df.to_csv(ckpt_trajectory, index=False)

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
                sims_cache=sims_cache,
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
        ranking_df = compute_method_ranking(sim_acc_df, stability_df, pr_df,
                                            trajectory_df=trajectory_df)
        print_method_ranking(ranking_df)
        if not ranking_df.empty:
            ranking_df.to_csv(out_dir / "method_ranking.csv", index=False)

    print(f"\n[OK] All results written to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
