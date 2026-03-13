#!/usr/bin/env python
"""
benchmark_reconstruction.py
============================
Compares three ancestral-reconstruction approaches used in SimPhyNI:

  Method A  legacy    PastML CLI via subprocess (original pastml.py logic)
  Method B  api       PastML Python API, emergence-threshold uncertainty
  Method C  marginal  PastML Python API, marginal-probability uncertainty

Evaluation dimensions
---------------------
1. Speed
   Wall time for each method on the same input dataset.

2. Parameter stability (self-consistency)
   Simulate a population of synthetic trees from the inferred parameters,
   re-run reconstruction on each, then measure how close the re-estimated
   parameters are to the originals (correlation, relative error).

3. Parameter agreement
   Pearson / Spearman correlation between the rate estimates produced by
   each pair of methods on the same real data.

4. (Optional) Pipeline precision / recall  (direction-aware)
   Requires --eval_pr flag and a CSV of known interacting trait pairs
   supplied via --known_pairs.  The CSV must have columns:
     T1, T2, direction   where direction is  1 (synergistic / positive)
                                          or -1 (antagonistic / negative)
   Runs the full SimPhyNI simulation + KDE test on each reconstruction
   output and reports precision/recall separately for positive associations,
   negative associations, and direction-agnostic (pair identity only) at
   several p-value thresholds.

Usage
-----
    # Minimal: speed + stability + agreement
    python benchmark_reconstruction.py \\
        --tree my_tree.nwk \\
        --annotations traits.csv \\
        --output benchmark_results/

    # Full including precision/recall
    python benchmark_reconstruction.py \\
        --tree my_tree.nwk \\
        --annotations traits.csv \\
        --output benchmark_results/ \\
        --eval_pr \\
        --known_pairs pairs.csv \\
        --n_stability 20 \\
        --max_workers 16
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.linalg import expm
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Imports from the SimPhyNI source tree
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_SCRIPTS = _HERE.parent / "simphyni" / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_HERE.parent / "simphyni" / "Simulation"))

from run_ancestral_reconstruction import (
    label_internal_nodes,
    compute_branch_upper_bound,
    count_joint_stats,
    count_marginal_stats,
    prefiltering,
)

from pastml.acr import acr
from pastml.ml import JOINT, MPPA

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark three SimPhyNI ancestral reconstruction methods.",
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
                   help="CSV with columns T1,T2 of known interacting trait pairs")
    p.add_argument("--no_legacy",   action="store_true",
                   help="Skip Method A (PastML CLI) — useful if pastml CLI is unavailable")
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
    from countgainloss_tab import countgainloss
    from ete3 import Tree as _Tree

    tree_obj = _Tree(tree_file, format=1)
    leaves = set(tree_obj.get_leaf_names())
    obs = pd.read_csv(ann_file, index_col=0)
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
    obs = obs.loc[[i for i in obs.index if i in leaves_in_tree]]
    obs = obs[[c for c in traits if c in obs.columns]]

    gene_sums = {g: int(obs[g].sum()) for g in obs.columns}

    from run_ancestral_reconstruction import reconstruct_trait

    t0 = time.perf_counter()
    rows = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                reconstruct_trait,
                gene, tree_newick, obs, upper_bound, uncertainty, gene_sums.get(gene, 0)
            ): gene
            for gene in obs.columns
        }
        for f in as_completed(futures):
            gene = futures[f]
            res = f.result()
            if res is not None:
                rows[gene] = res

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame([rows[g] for g in obs.columns if g in rows])
    return df, elapsed

# ---------------------------------------------------------------------------
# Parameter agreement: pairwise correlations between methods
# ---------------------------------------------------------------------------

def parameter_agreement(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each shared column (gains, losses, gain_subsize, loss_subsize,
    dist, loss_dist) compute Pearson and Spearman r between all method pairs.
    """
    rate_cols = ["gains", "losses", "gain_subsize", "loss_subsize", "dist", "loss_dist"]
    method_names = list(dfs.keys())
    records = []

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            ma, mb = method_names[i], method_names[j]
            dfa, dfb = dfs[ma], dfs[mb]

            # Align on gene names
            shared = pd.merge(
                dfa[["gene"] + [c for c in rate_cols if c in dfa.columns]],
                dfb[["gene"] + [c for c in rate_cols if c in dfb.columns]],
                on="gene", suffixes=("_a", "_b"),
            )
            if len(shared) < 5:
                continue

            for col in rate_cols:
                ca, cb = f"{col}_a", f"{col}_b"
                if ca not in shared.columns or cb not in shared.columns:
                    continue
                x = shared[ca].replace([np.inf, -np.inf], np.nan).dropna()
                y = shared[cb].reindex(x.index).replace([np.inf, -np.inf], np.nan)
                valid = x.notna() & y.notna()
                x, y = x[valid].values, y[valid].values
                if len(x) < 5:
                    continue
                pr, _ = pearsonr(x, y)
                sr, _ = spearmanr(x, y)
                records.append({
                    "method_A": ma, "method_B": mb, "column": col,
                    "n_shared": len(x), "pearson_r": pr, "spearman_r": sr,
                })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Stability: simulate → re-reconstruct → compare parameters
# ---------------------------------------------------------------------------

def _simulate_binary_trait(tree: Tree, q01: float, q10: float,
                            root_state: int, seed: int) -> pd.DataFrame:
    """Simulate a binary trait on the tree using a 2-state CTMC."""
    rng = np.random.default_rng(seed)
    Q = np.array([[-q01, q01], [q10, -q10]], dtype=float)
    node_states = {id(tree): root_state}

    for node in tree.traverse("preorder"):
        if node.is_root():
            continue
        ps = node_states[id(node.up)]
        P = np.clip(expm(Q * node.dist), 0.0, 1.0)
        P /= P.sum(axis=1, keepdims=True)
        node_states[id(node)] = int(rng.choice([0, 1], p=P[ps]))

    tip_states = {
        leaf.name: str(node_states[id(leaf)])
        for leaf in tree.get_leaves()
        if id(leaf) in node_states
    }
    return pd.DataFrame({"sim_trait": tip_states})


def stability_evaluation(tree_file: str, dfs: dict[str, pd.DataFrame],
                          n_sims: int, max_workers: int) -> pd.DataFrame:
    """
    For each method, pick up to 10 traits, simulate n_sims trees from
    the inferred parameters, re-reconstruct, and compare the re-estimated
    gain/loss rates to the original.  Returns a DataFrame of relative errors.
    """
    master_tree = Tree(tree_file, format=1)
    label_internal_nodes(master_tree)
    upper_bound = compute_branch_upper_bound(master_tree)
    tree_newick = master_tree.write(format=1)

    from run_ancestral_reconstruction import reconstruct_trait

    records = []
    n_test_traits = 10

    for method_name, df in dfs.items():
        df = df.copy().dropna(subset=["gains", "losses", "gain_subsize", "loss_subsize"])
        df = df[df["gain_subsize"] > 0]
        df["gain_rate"] = df["gains"] / df["gain_subsize"]
        df["loss_rate"] = df["losses"] / df["loss_subsize"]
        sample = df.sample(n=min(n_test_traits, len(df)), random_state=0)

        for _, row in sample.iterrows():
            gene = row["gene"]
            q01 = float(row["gain_rate"])
            q10 = float(row["loss_rate"])
            rs  = int(row.get("root_state", 0))
            if q01 <= 0 or q10 <= 0:
                continue

            re_gain_rates, re_loss_rates = [], []
            for sim_idx in range(n_sims):
                sim_tree = Tree(tree_newick, format=1)
                label_internal_nodes(sim_tree)
                sim_df = _simulate_binary_trait(sim_tree, q01, q10, rs, seed=sim_idx)
                sim_df.columns = [gene]

                res = reconstruct_trait(
                    gene, tree_newick, sim_df, upper_bound,
                    uncertainty="threshold", gene_count=int(sim_df[gene].astype(int).sum()),
                )
                if res is None:
                    continue
                gs = res.get("gain_subsize", 0)
                ls = res.get("loss_subsize", 0)
                if gs > 0:
                    re_gain_rates.append(res["gains"] / gs)
                if ls > 0:
                    re_loss_rates.append(res["losses"] / ls)

            if re_gain_rates:
                records.append({
                    "method": method_name, "gene": gene,
                    "parameter": "gain_rate",
                    "true_value": q01,
                    "mean_reestimate": float(np.mean(re_gain_rates)),
                    "std_reestimate":  float(np.std(re_gain_rates)),
                    "relative_error":  float(abs(np.mean(re_gain_rates) - q01) / max(q01, 1e-9)),
                })
            if re_loss_rates:
                records.append({
                    "method": method_name, "gene": gene,
                    "parameter": "loss_rate",
                    "true_value": q10,
                    "mean_reestimate": float(np.mean(re_loss_rates)),
                    "std_reestimate":  float(np.std(re_loss_rates)),
                    "relative_error":  float(abs(np.mean(re_loss_rates) - q10) / max(q10, 1e-9)),
                })

    return pd.DataFrame(records)

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
        """Return symmetric set: both (a,b,d) and (b,a,d) orderings as (a,b,d) tuples."""
        s = set()
        for _, row in subset.iterrows():
            s.add((row["T1"], row["T2"], int(row["direction"])))
            s.add((row["T2"], row["T1"], int(row["direction"])))
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
    tree_file: str, dfs: dict[str, pd.DataFrame],
    ann_file: str, known_pairs_file: str,
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
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "simphyni" / "Simulation"))
        from tree_simulator import TreeSimulator
    except ImportError as exc:
        print(f"[WARN] Could not import TreeSimulator: {exc}")
        return pd.DataFrame()

    known_sets = _load_known_pairs(known_pairs_file)
    thresholds = [0.05, 0.01, 0.001]
    records = []

    for method_name, df in dfs.items():
        try:
            sim = TreeSimulator(tree_file, df, ann_file)
            sim.initialize_simulation_parameters(pre_filter=False)
            sim.run_simulation()
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
                    pred_dir.add((row["T1"], row["T2"], d))
                    pred_dir.add((row["T2"], row["T1"], d))
                    pred_any.add((row["T1"], row["T2"], 0))
                    pred_any.add((row["T2"], row["T1"], 0))

                # Direction-agnostic: strip direction from known sets for 'any' comparison
                known_any = {(a, b, 0) for a, b, _ in known_sets["any"]}

                for subset_name, pred_set, known_set in [
                    ("positive", pred_dir, known_sets["positive"]),
                    ("negative", pred_dir, known_sets["negative"]),
                    ("any",      pred_any, known_any),
                ]:
                    # Halve counts because each pair is stored in both orderings
                    stats = _pr_stats(pred_set, known_set)
                    stats = {k: v // 2 if k in ("TP", "FP", "FN") else v
                             for k, v in stats.items()}
                    # Recompute derived metrics after halving
                    tp, fp, fn = stats["TP"], stats["FP"], stats["FN"]
                    precision = tp / max(tp + fp, 1)
                    recall    = tp / max(tp + fn, 1)
                    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
                    records.append({
                        "method": method_name, "threshold": thresh,
                        "association": subset_name,
                        "TP": tp, "FP": fp, "FN": fn,
                        "precision": precision, "recall": recall, "F1": f1,
                    })

        except Exception as exc:
            print(f"[WARN] P/R evaluation failed for {method_name}: {exc}")

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
    _section("Parameter Agreement (Pearson / Spearman r)")
    if df.empty:
        print("  No agreement data.")
        return
    for col, grp in df.groupby("column"):
        print(f"\n  [{col}]")
        for _, row in grp.iterrows():
            print(f"    {row['method_A']} vs {row['method_B']:20s}"
                  f"  pearson={row['pearson_r']:.3f}  spearman={row['spearman_r']:.3f}"
                  f"  n={row['n_shared']}")


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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load and optionally subsample traits ----
    obs = pd.read_csv(args.annotations, index_col=0)
    traits = list(obs.columns)
    if args.traits:
        traits = traits[:args.traits]
    print(f"Benchmarking {len(traits)} traits on tree {args.tree}", flush=True)

    dfs: dict[str, pd.DataFrame] = {}
    timings: dict[str, float] = {}

    # ---- Method A: Legacy CLI ----
    if not args.no_legacy:
        print("\n[1/3] Running Method A: Legacy PastML CLI ...", flush=True)
        try:
            df_a, t_a = run_legacy_cli(
                args.tree, args.annotations, traits,
                out_dir / "legacy_cli", args.max_workers,
            )
            dfs["A_legacy"] = df_a
            timings["A_legacy"] = t_a
            print(f"  Done in {t_a:.1f}s  ({len(df_a)} traits)", flush=True)
        except Exception as exc:
            print(f"  [FAILED] {exc}", file=sys.stderr)

    # ---- Method B: API threshold ----
    print("\n[2/3] Running Method B: API (threshold) ...", flush=True)
    df_b, t_b = run_api_method(
        args.tree, args.annotations, traits,
        uncertainty="threshold", max_workers=args.max_workers,
    )
    dfs["B_api_threshold"] = df_b
    timings["B_api_threshold"] = t_b
    print(f"  Done in {t_b:.1f}s  ({len(df_b)} traits)", flush=True)

    # ---- Method C: API marginal ----
    print("\n[3/3] Running Method C: API (marginal) ...", flush=True)
    df_c, t_c = run_api_method(
        args.tree, args.annotations, traits,
        uncertainty="marginal", max_workers=args.max_workers,
    )
    dfs["C_api_marginal"] = df_c
    timings["C_api_marginal"] = t_c
    print(f"  Done in {t_c:.1f}s  ({len(df_c)} traits)", flush=True)

    # ---- Save raw outputs ----
    for name, df in dfs.items():
        df.to_csv(out_dir / f"params_{name}.csv", index=False)

    # ---- Speed ----
    print_speed_table(timings)

    # ---- Agreement ----
    # Align marginal df to use standard column names for comparison
    df_c_threshold_view = df_c.copy()
    if "gains_marginal" in df_c_threshold_view.columns:
        for src, dst in [
            ("gains_marginal", "gains"), ("losses_marginal", "losses"),
            ("gain_subsize_marginal", "gain_subsize"),
            ("loss_subsize_marginal", "loss_subsize"),
            ("dist_marginal", "dist"), ("loss_dist_marginal", "loss_dist"),
        ]:
            df_c_threshold_view[dst] = df_c_threshold_view[src]
    dfs_for_agreement = {k: v for k, v in dfs.items()}
    dfs_for_agreement["C_api_marginal_cols"] = df_c_threshold_view
    agreement_df = parameter_agreement(dfs_for_agreement)
    print_agreement_table(agreement_df)
    agreement_df.to_csv(out_dir / "parameter_agreement.csv", index=False)

    # ---- Stability ----
    print("\nRunning stability evaluation ...", flush=True)
    stability_df = stability_evaluation(
        args.tree, dfs, args.n_stability, args.max_workers
    )
    print_stability_table(stability_df)
    stability_df.to_csv(out_dir / "stability.csv", index=False)

    # ---- Precision / Recall (optional) ----
    if args.eval_pr:
        if not args.known_pairs:
            print("[WARN] --eval_pr requires --known_pairs; skipping P/R evaluation.")
        else:
            print("\nRunning precision/recall evaluation ...", flush=True)
            pr_df = precision_recall_evaluation(
                args.tree, dfs, args.annotations, args.known_pairs
            )
            if not pr_df.empty:
                _section("Precision / Recall (direction-aware)")
                for assoc, grp in pr_df.groupby("association"):
                    print(f"\n  [{assoc}]")
                    print(grp[["method","threshold","TP","FP","FN",
                               "precision","recall","F1"]]
                          .to_string(index=False, float_format="{:.3f}".format))
                pr_df.to_csv(out_dir / "precision_recall.csv", index=False)

    print(f"\n[OK] All results written to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
