"""
benchmark_fast_acr.py
=====================
Benchmark fast_binary_acr.py against PastML on real data.

Compares:
  1. Wall-clock time at N = 50, 200, 500, 1000, 2000 traits
  2. Marginal probabilities — Pearson r and MAE vs PastML per node
  3. JOINT state agreement — fraction of nodes where states match
  4. Downstream stats — gains_flow, gains (JOINT), subsize correlation
  5. Log-likelihood correlation

Usage
-----
    python benchmark_fast_acr.py \\
        --tree  tests/panx/ecoli_accessory.nwk \\
        --traits tests/panx/ecoli_accessory.csv \\
        --n_traits 50 200 500 \\
        --out_dir dev/acr_plots/fast_acr_benchmark \\
        --pastml_workers 8

All paths are relative to the SimPhyNI repo root.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ---- repo path so we can import both dev and simphyni modules ----
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "dev"))
sys.path.insert(0, str(REPO / "simphyni" / "scripts"))

from fast_binary_acr import (
    FastACRResult,
    TreeArrays,
    build_obs_matrix,
    build_tree_arrays,
    fast_acr,
    marginal_probs_df,
)
from run_ancestral_reconstruction import (
    compute_branch_upper_bound,
    count_all_marginal_stats,
    count_joint_stats,
    label_internal_nodes,
)

try:
    from pastml.acr import acr
    from pastml.ml import JOINT, MPPA
    PASTML_AVAILABLE = True
except ImportError:
    PASTML_AVAILABLE = False
    print("[WARN] pastml not importable — PastML timing/accuracy comparisons skipped.")


# ---------------------------------------------------------------------------
# PastML single-trait helper (mirrors run_ancestral_reconstruction.py)
# ---------------------------------------------------------------------------

def pastml_one_trait(tree: Tree, gene: str, df_col: pd.Series):
    """Run PastML JOINT+MPPA for one trait; return (stats_dict, elapsed_s).

    The tree is a fresh copy (not modified in-place at caller).
    """
    t = tree.copy()
    ann = df_col.astype(str).to_frame(name=gene)

    states = sorted(ann[gene].dropna().unique().tolist())
    if len(states) < 2:
        return None, 0.0

    ann_leaves = set(ann.index)
    missing    = {l.name for l in t.get_leaves()} - ann_leaves
    if missing:
        t.prune([l for l in t.get_leaves() if l.name in ann_leaves],
                preserve_branch_length=True)

    for node in t.traverse():
        if not node.is_root() and node.dist <= 0:
            node.dist = 1e-8

    upper_bound = compute_branch_upper_bound(t)
    node_dists  = {node.name: 0.0 if node.is_root() else
                   sum(a.dist for a in node.iter_ancestors()) + node.dist
                   for node in t.traverse()}
    # Recompute properly
    node_dists = {}
    for node in t.traverse("preorder"):
        node_dists[node.name] = (
            0.0 if node.is_root() else node_dists[node.up.name] + node.dist
        )

    t0 = time.perf_counter()
    acr(t, df=ann, prediction_method=JOINT, model="F81")
    joint_stats = count_joint_stats(t, gene, upper_bound, node_dists=node_dists)

    mppa_res = acr(t, df=ann, prediction_method=MPPA, model="F81")
    mp_df    = mppa_res[0]["marginal_probabilities"]
    marg_stats = count_all_marginal_stats(t, gene, mp_df, upper_bound,
                                          node_dists=node_dists)
    elapsed = time.perf_counter() - t0

    stats = {"gene": gene, **joint_stats, **marg_stats}
    stats["_mp_df"] = mp_df
    return stats, elapsed


# ---------------------------------------------------------------------------
# Downstream stats from fast ACR result for one trait
# ---------------------------------------------------------------------------

def fast_stats_one_trait(
    result: FastACRResult,
    t_idx: int,
    ete_tree: Tree,
    upper_bound: float,
):
    """Compute gain/loss stats from fast ACR output for one trait.

    We replicate count_joint_stats and count_all_marginal_stats using the
    fast ACR marginal probabilities.  To call these existing functions we
    need to annotate an ETE3 tree (needed for the JOINT path) and build a
    marginal_probabilities DataFrame (for the MPPA path).
    """
    gene = result.trait_names[t_idx]
    ta   = result.ta

    # --- Build node_dists from the TreeArrays (no ETE3 needed) ---
    node_dists = {ta.node_names[i]: float(ta.nd[i]) for i in range(ta.n_nodes)}

    # --- Annotate a fresh copy for JOINT counting ---
    t = ete_tree.copy()
    js = result.joint_states[t_idx]
    for node in t.traverse():
        idx = ta.name_to_idx.get(node.name)
        if idx is not None:
            state = int(js[idx])
            setattr(node, gene, {state})
        else:
            setattr(node, gene, {0})  # fallback

    joint_stats = count_joint_stats(t, gene, upper_bound, node_dists=node_dists)

    # --- Marginal stats from fast ACR probabilities ---
    mp_df = marginal_probs_df(result, t_idx)
    marg_stats = count_all_marginal_stats(
        t, gene, mp_df, upper_bound, node_dists=node_dists
    )

    return {"gene": gene, **joint_stats, **marg_stats}


# ---------------------------------------------------------------------------
# Timing benchmark
# ---------------------------------------------------------------------------

def timing_benchmark(
    ta: TreeArrays,
    obs: np.ndarray,
    trait_names: list,
    trait_df: pd.DataFrame,
    ete_tree: Tree,
    sizes: list,
    n_pastml_sample: int = 5,
    no_pastml: bool = False,
):
    """For each N in sizes: time fast ACR (empirical+ml) and extrapolate PastML.

    PastML is timed only ONCE on n_pastml_sample traits; the per-trait average
    is then extrapolated to each N.  This avoids running PastML hundreds of
    times, which would dominate the benchmark wall time.
    """
    rows = []

    # Numba warm-up (compile on 1 trait, not counted in timings)
    fast_acr(ta, obs[:1], trait_names[:1], mode="empirical")
    fast_acr(ta, obs[:1], trait_names[:1], mode="ml")
    print("  [warmup done]", flush=True)

    # PastML per-trait cost — sample once from the first n_pastml_sample traits
    pastml_per_trait_s = np.nan
    if not no_pastml and PASTML_AVAILABLE:
        n_sample = min(n_pastml_sample, len(trait_names))
        print(f"  Timing PastML on {n_sample} traits (for extrapolation) ...",
              flush=True)
        t0 = time.perf_counter()
        for gene in trait_names[:n_sample]:
            if gene not in trait_df.columns:
                continue
            pastml_one_trait(ete_tree, gene, trait_df[gene].dropna())
        pastml_per_trait_s = (time.perf_counter() - t0) / n_sample
        print(f"  PastML per-trait: {pastml_per_trait_s*1000:.0f} ms", flush=True)

    for N in sizes:
        N = min(N, len(trait_names))
        obs_N   = obs[:N]
        names_N = trait_names[:N]

        t0 = time.perf_counter()
        fast_acr(ta, obs_N, names_N, mode="empirical")
        t_empirical = time.perf_counter() - t0

        t0 = time.perf_counter()
        fast_acr(ta, obs_N, names_N, mode="ml")
        t_ml = time.perf_counter() - t0

        row = {
            "N": N,
            "fast_empirical_s":          t_empirical,
            "fast_empirical_per_trait_ms": t_empirical / N * 1000,
            "fast_ml_s":                 t_ml,
            "fast_ml_per_trait_ms":      t_ml / N * 1000,
        }

        if not np.isnan(pastml_per_trait_s):
            t_pastml_extrap = pastml_per_trait_s * N
            row["pastml_extrap_s"]       = t_pastml_extrap
            row["pastml_per_trait_ms"]   = pastml_per_trait_s * 1000
            row["speedup_vs_empirical"]  = t_pastml_extrap / max(t_empirical, 1e-9)
            row["speedup_vs_ml"]         = t_pastml_extrap / max(t_ml, 1e-9)

        rows.append(row)
        msg = (f"  N={N:5d}  empirical={t_empirical:.2f}s"
               f"  ml={t_ml:.2f}s")
        if not np.isnan(pastml_per_trait_s):
            msg += f"  pastml_extrap={t_pastml_extrap:.0f}s"
            msg += f"  speedup={row['speedup_vs_empirical']:.0f}×"
        print(msg, flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Accuracy benchmark: compare fast ACR vs PastML on small N
# ---------------------------------------------------------------------------

def accuracy_benchmark(
    ta: TreeArrays,
    trait_df: pd.DataFrame,
    ete_tree: Tree,
    obs: np.ndarray,
    trait_names: list,
    n_compare: int,
    upper_bound: float,
):
    """Compare marginal probs, JOINT states, and downstream stats vs PastML.

    n_compare traits are evaluated.  Returns three DataFrames:
      - node_level  : per-(trait, node) marginal probability comparison
      - trait_level : per-trait aggregate stats
      - stats_df    : per-trait downstream stats (gains_flow, gains, etc.)
    """
    print(f"  Running PastML + fast ACR on {n_compare} traits for accuracy ...")

    # Run fast ACR once for all n_compare traits
    obs_sub   = obs[:n_compare]
    names_sub = trait_names[:n_compare]

    res_emp = fast_acr(ta, obs_sub, names_sub, mode="empirical")
    res_ml  = fast_acr(ta, obs_sub, names_sub, mode="ml")

    node_rows   = []
    trait_rows  = []
    stats_rows  = []

    for t_idx, gene in enumerate(names_sub):
        if gene not in trait_df.columns:
            continue

        # PastML ground truth
        col = trait_df[gene].dropna()
        pt_stats, pt_elapsed = pastml_one_trait(ete_tree, gene, col)
        if pt_stats is None:
            continue

        pt_mp_df = pt_stats["_mp_df"].rename(columns=str)
        if "1" not in pt_mp_df.columns:
            continue
        pastml_p1 = pt_mp_df["1"].astype(float)

        # Fast empirical marginals at shared nodes
        fast_emp_df = marginal_probs_df(res_emp, t_idx)
        fast_ml_df  = marginal_probs_df(res_ml,  t_idx)

        shared = pastml_p1.index.intersection(fast_emp_df.index)
        if len(shared) < 3:
            continue

        pp1  = pastml_p1.loc[shared].values
        ep1  = fast_emp_df["1"].loc[shared].values
        mp1  = fast_ml_df["1"].loc[shared].values

        r_emp, _ = pearsonr(pp1, ep1) if len(pp1) > 1 else (np.nan, None)
        r_ml,  _ = pearsonr(pp1, mp1) if len(pp1) > 1 else (np.nan, None)
        mae_emp  = np.mean(np.abs(pp1 - ep1))
        mae_ml   = np.mean(np.abs(pp1 - mp1))

        # JOINT agreement
        pt_joint = {}
        for node in ete_tree.traverse():
            v = getattr(node, gene, None)
            if v is not None:
                try:
                    s = int(next(iter(v))) if isinstance(v, (set, frozenset)) else int(str(v).split("|")[0])
                    pt_joint[node.name] = s
                except (ValueError, StopIteration):
                    pass

        fast_joint_emp = {ta.node_names[i]: int(res_emp.joint_states[t_idx, i])
                         for i in range(ta.n_nodes)}
        fast_joint_ml  = {ta.node_names[i]: int(res_ml.joint_states[t_idx, i])
                         for i in range(ta.n_nodes)}

        shared_j = set(pt_joint) & set(fast_joint_emp)
        if shared_j:
            agree_emp = sum(pt_joint[n] == fast_joint_emp[n] for n in shared_j) / len(shared_j)
            agree_ml  = sum(pt_joint[n] == fast_joint_ml[n]  for n in shared_j) / len(shared_j)
        else:
            agree_emp = agree_ml = np.nan

        trait_rows.append({
            "gene": gene,
            "n_nodes_compared": len(shared),
            "pearson_r_empirical": r_emp,
            "pearson_r_ml": r_ml,
            "mae_empirical": mae_emp,
            "mae_ml": mae_ml,
            "joint_agree_empirical": agree_emp,
            "joint_agree_ml": agree_ml,
            "lh_empirical": res_emp.log_lh[t_idx],
            "lh_ml": res_ml.log_lh[t_idx],
            "sf_empirical": res_emp.sf[t_idx],
            "sf_ml": res_ml.sf[t_idx],
            "pi1_empirical": res_emp.pi1[t_idx],
            "pi1_ml": res_ml.pi1[t_idx],
        })

        # Downstream stats comparison
        fast_emp_stats = fast_stats_one_trait(res_emp, t_idx, ete_tree, upper_bound)
        fast_ml_stats  = fast_stats_one_trait(res_ml,  t_idx, ete_tree, upper_bound)

        for stat_key in ["gains", "losses", "gains_flow", "losses_flow",
                         "gain_subsize", "gain_subsize_marginal"]:
            if stat_key in pt_stats and stat_key in fast_emp_stats:
                stats_rows.append({
                    "gene": gene,
                    "stat": stat_key,
                    "pastml": pt_stats[stat_key],
                    "fast_empirical": fast_emp_stats.get(stat_key, np.nan),
                    "fast_ml": fast_ml_stats.get(stat_key, np.nan),
                })

    trait_df_out = pd.DataFrame(trait_rows)
    stats_df_out = pd.DataFrame(stats_rows)
    return trait_df_out, stats_df_out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_timing(timing_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(timing_df["N"], timing_df["fast_empirical_s"], "o-", label="fast (empirical π₁)", color="steelblue")
    ax.plot(timing_df["N"], timing_df["fast_ml_s"],        "s-", label="fast (ML π₁+sf)",    color="darkorange")
    if "pastml_extrap_s" in timing_df:
        ax.plot(timing_df["N"], timing_df["pastml_extrap_s"], "^--", label="PastML (extrapolated)",
                color="firebrick")
    ax.set_xlabel("Number of traits")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("ACR wall time vs trait count")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(timing_df["N"], timing_df["fast_empirical_per_trait_ms"],
            "o-", label="fast empirical", color="steelblue")
    ax.plot(timing_df["N"], timing_df["fast_ml_per_trait_ms"],
            "s-", label="fast ML",       color="darkorange")
    if "pastml_per_trait_ms" in timing_df:
        ax.axhline(timing_df["pastml_per_trait_ms"].mean(), color="firebrick",
                   ls="--", label="PastML (per trait)")
    ax.set_xlabel("Number of traits")
    ax.set_ylabel("ms per trait")
    ax.set_title("Per-trait throughput")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "timing.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_accuracy(trait_df: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Pearson r distribution
    ax = axes[0, 0]
    ax.hist(trait_df["pearson_r_empirical"].dropna(), bins=20, alpha=0.6,
            label="empirical π₁", color="steelblue")
    ax.hist(trait_df["pearson_r_ml"].dropna(), bins=20, alpha=0.6,
            label="ML π₁+sf", color="darkorange")
    ax.set_xlabel("Pearson r (marginal probs vs PastML)")
    ax.set_ylabel("Traits")
    ax.set_title("Marginal probability correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MAE distribution
    ax = axes[0, 1]
    ax.hist(trait_df["mae_empirical"].dropna(), bins=20, alpha=0.6,
            label="empirical", color="steelblue")
    ax.hist(trait_df["mae_ml"].dropna(), bins=20, alpha=0.6,
            label="ML", color="darkorange")
    ax.set_xlabel("MAE (marginal probs vs PastML)")
    ax.set_title("Marginal probability MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. JOINT agreement
    ax = axes[0, 2]
    ax.hist(trait_df["joint_agree_empirical"].dropna(), bins=20, alpha=0.6,
            label="empirical", color="steelblue")
    ax.hist(trait_df["joint_agree_ml"].dropna(), bins=20, alpha=0.6,
            label="ML", color="darkorange")
    ax.set_xlabel("Fraction of nodes with JOINT state agreement")
    ax.set_title("JOINT state agreement vs PastML")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4-6. Downstream stats scatter
    for ax_idx, stat_key in enumerate(["gains", "gains_flow", "gain_subsize_marginal"]):
        ax = axes[1, ax_idx]
        sub = stats_df[stats_df["stat"] == stat_key]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        ax.scatter(sub["pastml"], sub["fast_empirical"], alpha=0.5, s=10,
                   label="empirical", color="steelblue")
        ax.scatter(sub["pastml"], sub["fast_ml"], alpha=0.5, s=10,
                   label="ML", color="darkorange")
        lim = [min(sub[["pastml","fast_empirical","fast_ml"]].min()),
               max(sub[["pastml","fast_empirical","fast_ml"]].max())]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"PastML {stat_key}")
        ax.set_ylabel(f"Fast ACR {stat_key}")
        ax.set_title(f"{stat_key}: fast vs PastML")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def print_summary(timing_df: pd.DataFrame, trait_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(timing_df.to_string(index=False))

    if len(trait_df) == 0:
        return
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY (vs PastML)")
    print("=" * 60)
    cols = ["pearson_r_empirical", "pearson_r_ml",
            "mae_empirical", "mae_ml",
            "joint_agree_empirical", "joint_agree_ml"]
    summary = trait_df[cols].describe().loc[["mean", "50%", "min"]].round(4)
    print(summary.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fast_binary_acr vs PastML"
    )
    parser.add_argument("--tree",   default="tests/panx/ecoli_accessory.nwk")
    parser.add_argument("--traits", default="tests/panx/ecoli_accessory.csv")
    parser.add_argument("--n_traits", type=int, nargs="+",
                        default=[50, 200, 500, 1000, 2000],
                        help="Trait counts for timing benchmark")
    parser.add_argument("--n_compare", type=int, default=50,
                        help="Number of traits for PastML accuracy comparison")
    parser.add_argument("--out_dir",
                        default="dev/acr_plots/fast_acr_benchmark")
    parser.add_argument("--n_pastml_sample", type=int, default=5,
                        help="Traits timed against PastML for extrapolation (default 5)")
    parser.add_argument("--no-pastml", action="store_true", default=False,
                        help="Skip PastML timing/accuracy comparison (fast ACR only).")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    tree_path   = repo / args.tree
    traits_path = repo / args.traits
    out_dir     = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print(f"Loading tree: {tree_path}")
    ete_tree = Tree(str(tree_path), format=1)
    label_internal_nodes(ete_tree)
    upper_bound = compute_branch_upper_bound(ete_tree)

    print(f"Loading traits: {traits_path}")
    trait_df = pd.read_csv(traits_path, index_col=0)

    # Binary check
    unique_vals = np.unique(trait_df.values)
    if not np.all(np.isin(unique_vals, [0, 1, np.nan])):
        print(f"[WARN] Trait values include non-binary: {unique_vals[:5]} ...")

    trait_names = list(trait_df.columns)
    n_avail = len(trait_names)
    max_n   = max(args.n_traits)
    if max_n > n_avail:
        print(f"[WARN] Requested N={max_n} > available {n_avail}; capping.")
        args.n_traits = [n for n in args.n_traits if n <= n_avail]
        if not args.n_traits:
            args.n_traits = [n_avail]

    print(f"Tree: {len(ete_tree.get_leaves())} leaves, "
          f"{len(list(ete_tree.traverse()))} nodes")
    print(f"Traits available: {n_avail}  |  "
          f"Testing sizes: {args.n_traits}")

    # ---- Build tree arrays ----
    print("Building tree arrays ...")
    ta = build_tree_arrays(ete_tree)

    # ---- Build obs matrix (for max size needed) ----
    max_n = max(args.n_traits + [args.n_compare])
    max_n = min(max_n, n_avail)
    sub_traits = trait_names[:max_n]

    print(f"Building obs matrix for {max_n} traits ...")
    obs = build_obs_matrix(ta, trait_df[sub_traits])
    trait_names_sub = sub_traits

    # ---- Timing benchmark ----
    print("\nRunning timing benchmark ...")
    timing_df = timing_benchmark(
        ta, obs, trait_names_sub,
        trait_df, ete_tree, args.n_traits,
        no_pastml=args.no_pastml,
    )
    timing_df.to_csv(out_dir / "timing.csv", index=False)

    # ---- Accuracy benchmark ----
    trait_df_acc = pd.DataFrame()
    stats_df_acc = pd.DataFrame()
    if not args.no_pastml and PASTML_AVAILABLE:
        n_cmp = min(args.n_compare, max_n)
        print(f"\nRunning accuracy benchmark ({n_cmp} traits vs PastML) ...")
        trait_df_acc, stats_df_acc = accuracy_benchmark(
            ta, trait_df, ete_tree,
            obs, trait_names_sub,
            n_compare=n_cmp,
            upper_bound=upper_bound,
        )
        trait_df_acc.to_csv(out_dir / "accuracy_trait_level.csv", index=False)
        stats_df_acc.to_csv(out_dir / "accuracy_stats.csv",       index=False)

    # ---- Plots ----
    print("\nGenerating plots ...")
    plot_timing(timing_df, out_dir)
    if len(trait_df_acc) > 0:
        plot_accuracy(trait_df_acc, stats_df_acc, out_dir)

    print_summary(timing_df, trait_df_acc)
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
