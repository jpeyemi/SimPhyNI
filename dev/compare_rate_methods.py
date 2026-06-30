#!/usr/bin/env python
"""
compare_rate_methods.py
=======================
Compare three gain/loss rate estimation methods by asking: how well do
simulations parameterised from each method reproduce the *observed* trait
on the real tree?

Methods compared
----------------
  JOINT  — discrete 0→1/1→0 transitions from MAP joint_states
            gain_rate = gains_joint / gain_subsize_joint
            loss_rate = losses_joint / loss_subsize_joint

  FLOW   — soft sum from MPPA marginal_p1
            gain_rate = Σ max(0, p1_child − p1_parent) / Σ p0_parent·bl
            loss_rate = Σ max(0, p1_parent − p1_child) / Σ p1_parent·bl

  PI     — model-derived rate from ML-optimised F81 parameters (sf, π₁):
            gain_rate = sf / (2·π₀),  loss_rate = sf / (2·π₁)

Calibration metrics (sim vs observed)
--------------------------------------
  prevalence_err  : |sim_mean_prevalence − obs_prevalence|
  parsimony_ratio : sim_parsimony_mean / obs_parsimony
  d_stat_err      : |sim_D_mean − obs_D|   (Fritz & Purvis D statistic)
  mpd_ratio       : sim_MPD_mean / obs_MPD  (mean pairwise phylo distance)
  clade_jsd       : Jensen-Shannon divergence of per-clade prevalence vectors

Usage
-----
    python dev/compare_rate_methods.py [--n_traits 100] [--trials 64] [--seed 0]
"""

import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.spatial.distance import jensenshannon

# ---------------------------------------------------------------------------
# Paths & direct imports (avoid triggering simphyni/__init__ → matplotlib)
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parents[1]
SCRIPTS_DIR   = REPO_ROOT / "simphyni" / "scripts"
BENCH_DIR     = REPO_ROOT / "dev"
# d_statistic.py lives in the sibling benchmarking repo
BENCH_SCRIPTS = Path("/Users/jpeyemi/SimPhyNI-Benchmarking/scripts")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCH_SCRIPTS))

def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_acr_mod  = _load("run_ancestral_reconstruction",
                  SCRIPTS_DIR / "run_ancestral_reconstruction.py")
_facr_mod = _load("fast_binary_acr",
                  SCRIPTS_DIR / "fast_binary_acr.py")
_sim_mod  = _load("simulation",
                  REPO_ROOT / "simphyni" / "Simulation" / "simulation.py")
_d_mod    = _load("d_statistic",
                  BENCH_SCRIPTS / "d_statistic.py")

label_internal_nodes       = _acr_mod.label_internal_nodes
compute_branch_upper_bound = _acr_mod.compute_branch_upper_bound
build_tree_arrays          = _facr_mod.build_tree_arrays
build_obs_matrix           = _facr_mod.build_obs_matrix
fast_acr                   = _facr_mod.fast_acr
sim_bit                    = _sim_mod.sim_bit

precompute_tree_structure  = _d_mod.precompute_tree_structure
simulate_bm_vectors        = _d_mod.simulate_bm_vectors
get_null_distributions     = _d_mod.get_null_distributions
compute_d_statistic        = _d_mod.compute_d_statistic

TREE_FILE = REPO_ROOT / "tests" / "panx" / "ecoli_accessory.nwk"
ANN_FILE  = REPO_ROOT / "tests" / "panx" / "ecoli_accessory.csv"


# ---------------------------------------------------------------------------
# Build trait_params DataFrames for each method
# ---------------------------------------------------------------------------

def build_joint_params(result, ta, upper_bound, genes):
    """Build trait_params from fast_acr joint_states (JOINT method)."""
    rows = []
    for t_idx, gene in enumerate(genes):
        states = result.joint_states[t_idx]
        gains = losses = 0
        g_sub = l_sub = 0.0
        for ni in range(ta.n_nodes):
            pi = ta.parent[ni]
            if pi < 0:
                continue
            ps  = int(states[pi])
            cs  = int(states[ni])
            bl  = ta.bl[ni]
            eff = min(bl, upper_bound) if bl > 0 else 0.0
            if ps == 0:
                g_sub += eff
                if cs == 1:
                    gains += 1
            else:
                l_sub += eff
                if cs == 0:
                    losses += 1
        root_state = int(states[ta.root_idx])
        rows.append(dict(
            gene=gene,
            gains=float(gains),
            losses=float(losses),
            gain_subsize=g_sub,
            loss_subsize=l_sub,
            dist=0.0,
            loss_dist=0.0,
            root_state=root_state,
        ))
    df = pd.DataFrame(rows).set_index("gene")
    df = df[(df["gain_subsize"] > 0) & (df["loss_subsize"] > 0)]
    return df


def build_flow_params(result, ta, upper_bound, genes):
    """Build trait_params from fast_acr marginal_p1 (FLOW method)."""
    rows = []
    for t_idx, gene in enumerate(genes):
        p1   = result.marginal_p1[t_idx]
        gains_f = losses_f = 0.0
        g_sub   = l_sub   = 0.0
        for ni in range(ta.n_nodes):
            pi  = ta.parent[ni]
            if pi < 0:
                continue
            p1c = p1[ni]
            p1p = p1[pi]
            p0p = 1.0 - p1p
            bl  = ta.bl[ni]
            eff = min(bl, upper_bound) if bl > 0 else 0.0
            gains_f  += max(0.0, p1c - p1p)
            losses_f += max(0.0, p1p - p1c)
            g_sub    += p0p * eff
            l_sub    += p1p * eff
        root_state = int(p1[ta.root_idx] >= 0.5)
        rows.append(dict(
            gene=gene,
            gains=gains_f,
            losses=losses_f,
            gain_subsize=g_sub,
            loss_subsize=l_sub,
            dist=0.0,
            loss_dist=0.0,
            root_state=root_state,
        ))
    df = pd.DataFrame(rows).set_index("gene")
    df = df[(df["gain_subsize"] > 0) & (df["loss_subsize"] > 0)]
    return df


def build_pi_params(result, ta, genes):
    """Build trait_params from ML-optimised (sf, π₁) — PI method.

    gain_rate = sf / (2·π₀),  loss_rate = sf / (2·π₁)
    We encode this as gains = gain_rate, gain_subsize = 1.0  (ratio is all
    that sim_bit uses), and set dist=0 (no emergence threshold from this model).
    """
    rows = []
    total_bl = float(ta.bl.sum())
    for t_idx, gene in enumerate(genes):
        sf  = float(result.sf[t_idx])
        pi1 = float(result.pi1[t_idx])
        pi0 = 1.0 - pi1
        if pi0 < 1e-9 or pi1 < 1e-9:
            continue
        gain_rate = sf / (2.0 * pi0)
        loss_rate = sf / (2.0 * pi1)
        # Encode as count / subsize = rate by using total_bl as common denominator
        rows.append(dict(
            gene=gene,
            gains=gain_rate * total_bl,
            losses=loss_rate * total_bl,
            gain_subsize=total_bl,
            loss_subsize=total_bl,
            dist=0.0,
            loss_dist=0.0,
            root_state=int(pi1 >= 0.5),
        ))
    df = pd.DataFrame(rows).set_index("gene")
    df = df[(df["gain_subsize"] > 0) & (df["loss_subsize"] > 0)]
    return df


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def _fitch_parsimony_obs(tree, obs_states):
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


def _fitch_parsimony_packed(tree, leaf_packed):
    """Vectorized Fitch over 64 bit-packed trials."""
    ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
    node_set_1 = {}
    node_set_0 = {}
    scores = np.zeros(64, dtype=np.int64)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            packed = np.uint64(leaf_packed.get(node.name, 0))
            node_set_1[id(node)] = packed
            node_set_0[id(node)] = ~packed & ALL
        else:
            inter_1 = inter_0 = ALL
            for child in node.children:
                inter_1 &= node_set_1[id(child)]
                inter_0 &= node_set_0[id(child)]
            inter_nonempty = inter_1 | inter_0
            cost_bits = (~inter_nonempty) & ALL
            for t in range(64):
                scores[t] += int((cost_bits >> np.uint64(t)) & np.uint64(1))
            union_1 = union_0 = np.uint64(0)
            for child in node.children:
                union_1 |= node_set_1[id(child)]
                union_0 |= node_set_0[id(child)]
            node_set_1[id(node)] = (inter_nonempty & inter_1) | ((~inter_nonempty & ALL) & union_1)
            node_set_0[id(node)] = (inter_nonempty & inter_0) | ((~inter_nonempty & ALL) & union_0)
    return scores


def compute_metrics(
    tree, packed_col, gene, obs_df, leaf_list, leaf_index,
    D_mat, clade_id, n_clades,
    tree_struct, bm_leaf_vals, tree_path,
    trials=64,
):
    """Compute all calibration metrics for one (method, gene) simulation.

    packed_col : (n_leaves,) uint64 — bit t = trial-t tip state for this gene
    """
    n_leaves = len(leaf_list)

    # -- Prevalence --
    obs_states = {
        name: int(obs_df.loc[name, gene])
        for name in leaf_list if name in obs_df.index
    }
    obs_prev = float(np.mean([v for v in obs_states.values()]))
    trial_prevs = np.array([
        float(((packed_col >> np.uint64(t)) & np.uint64(1)).mean())
        for t in range(trials)
    ])
    sim_prev = float(trial_prevs.mean())
    prev_err = abs(sim_prev - obs_prev)

    # -- Parsimony --
    obs_pars = _fitch_parsimony_obs(tree, obs_states)
    leaf_packed = {leaf_list[i]: packed_col[i] for i in range(n_leaves)}
    sim_pars = _fitch_parsimony_packed(tree, leaf_packed)
    sim_pars_mean = float(sim_pars.mean())
    pars_ratio = sim_pars_mean / max(obs_pars, 1e-9) if obs_pars > 0 else float("nan")

    # -- D-statistic --
    # Calibrate once at obs_prev; reuse for all sim trials (good approx when
    # sim prevalences are close to observed, which they should be if the method
    # is well-calibrated).
    obs_tip_states = np.array(
        [float(obs_states.get(name, 0)) for name in tree_struct["leaf_names"]],
        dtype=float,
    )
    rand_mean, bm_mean = get_null_distributions(
        tree_path, tree_struct, bm_leaf_vals, obs_prev,
    )
    obs_d = compute_d_statistic(tree_struct, obs_tip_states, rand_mean, bm_mean)

    sim_d_vals = []
    leaf_order = tree_struct["leaf_names"]
    leaf_packed_idx = np.array([leaf_index[name] for name in leaf_order], dtype=int)
    for t in range(trials):
        tip_t = ((packed_col[leaf_packed_idx] >> np.uint64(t)) & np.uint64(1)).astype(float)
        if tip_t.sum() < 2 or tip_t.sum() >= len(tip_t) - 1:
            continue
        d_t = compute_d_statistic(tree_struct, tip_t, rand_mean, bm_mean)
        if np.isfinite(d_t):
            sim_d_vals.append(d_t)
    sim_d_mean = float(np.mean(sim_d_vals)) if sim_d_vals else float("nan")
    d_err = abs(sim_d_mean - obs_d) if np.isfinite(obs_d) and np.isfinite(sim_d_mean) else float("nan")

    # -- MPD --
    obs_pos = np.array([leaf_index[n] for n in leaf_list
                        if n in obs_states and obs_states[n] == 1])
    obs_mpd = float(D_mat[np.ix_(obs_pos, obs_pos)].mean()) if len(obs_pos) > 1 else 0.0

    trial_mpds = []
    for t in range(trials):
        pos = np.where(((packed_col >> np.uint64(t)) & np.uint64(1)).astype(bool))[0]
        if len(pos) < 2:
            continue
        if len(pos) > 100:
            pos = np.random.choice(pos, 100, replace=False)
        trial_mpds.append(float(D_mat[np.ix_(pos, pos)].mean()))
    sim_mpd = float(np.mean(trial_mpds)) if trial_mpds else 0.0
    mpd_ratio = sim_mpd / max(obs_mpd, 1e-9)

    # -- Clade JSD --
    obs_clade = np.array([
        np.mean([obs_states.get(leaf_list[i], 0)
                 for i in range(n_leaves) if clade_id[i] == k])
        for k in range(n_clades)
    ], dtype=float)
    obs_clade = np.nan_to_num(obs_clade)

    sim_clade = np.zeros(n_clades)
    for t in range(trials):
        states_t = ((packed_col >> np.uint64(t)) & np.uint64(1)).astype(float)
        for k in range(n_clades):
            mask = clade_id == k
            if mask.sum() > 0:
                sim_clade[k] += states_t[mask].mean()
    sim_clade /= trials
    clade_jsd = float(jensenshannon(obs_clade + 1e-9, sim_clade + 1e-9) ** 2)

    return dict(
        obs_prev=obs_prev,
        sim_prev=sim_prev,
        prev_err=prev_err,
        obs_pars=obs_pars,
        sim_pars_mean=sim_pars_mean,
        pars_ratio=pars_ratio,
        obs_d=obs_d,
        sim_d_mean=sim_d_mean,
        d_err=d_err,
        obs_mpd=obs_mpd,
        sim_mpd=sim_mpd,
        mpd_ratio=mpd_ratio,
        clade_jsd=clade_jsd,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(records_df):
    metrics = {
        "prev_err":   "Prevalence |error|   (↓ better)",
        "pars_ratio": "Parsimony ratio      (ideal = 1)",
        "d_err":      "D-stat |error|       (↓ better)",
        "mpd_ratio":  "MPD ratio            (ideal = 1)",
        "clade_jsd":  "Clade JSD            (↓ better)",
    }
    methods = sorted(records_df["method"].unique())
    hdr_w   = max(len(v) for v in metrics.values()) + 2

    print(f"\n{'Metric':<{hdr_w}}", end="")
    for m in methods:
        print(f"  {m:>10}", end="")
    print()
    print("─" * (hdr_w + 14 * len(methods)))

    for col, label in metrics.items():
        print(f"{label:<{hdr_w}}", end="")
        for m in methods:
            vals = records_df[records_df["method"] == m][col].dropna()
            if len(vals) == 0:
                print(f"  {'—':>10}", end="")
            else:
                print(f"  {vals.median():>10.4f}", end="")
        print()
    print()

    # fraction high-error
    print("High-error fraction (≥2 metrics miscalibrated):")
    for m in methods:
        sub = records_df[records_df["method"] == m]
        flags = (
            (sub["prev_err"]   > 0.10).astype(int) +
            (sub["pars_ratio"] > 2.00).astype(int) +
            ((sub["mpd_ratio"] > 2.0) | (sub["mpd_ratio"] < 0.5)).astype(int) +
            (sub["clade_jsd"]  > 0.20).astype(int)
        )
        frac = (flags >= 2).mean()
        print(f"  {m:<10} {frac:.1%}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_traits", type=int, default=100,
                        help="Number of traits to evaluate (default 100)")
    parser.add_argument("--trials", type=int, default=64,
                        help="sim_bit trials per trait (default 64)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--acr_mode", choices=["ml", "empirical"], default="ml")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- Load tree & annotations ----
    print(f"Loading tree and annotations ...", flush=True)
    tree = Tree(str(TREE_FILE), format=1)
    label_internal_nodes(tree)
    upper_bound = compute_branch_upper_bound(tree)
    ta          = build_tree_arrays(tree)
    leaf_list   = [n.name for n in tree.iter_leaves()]
    leaf_index  = {name: i for i, name in enumerate(leaf_list)}
    n_leaves    = len(leaf_list)

    ann = pd.read_csv(ANN_FILE, index_col=0).astype(str)
    ann.index = ann.index.astype(str)

    # Select binary traits (present in both states), sample n_traits
    binary_genes = [
        c for c in ann.columns
        if set(ann[c].dropna().unique()) <= {"0", "1"}
        and len(ann[c].dropna().unique()) == 2
    ]
    n_sel = min(args.n_traits, len(binary_genes))
    genes = list(rng.choice(binary_genes, size=n_sel, replace=False))
    print(f"  {n_leaves} leaves  |  {len(binary_genes)} binary traits  "
          f"→ evaluating {n_sel}", flush=True)

    # ---- Precompute shared tree geometry ----
    print("Precomputing pairwise distances (MPD) ...", flush=True)
    leaves_ete = list(tree.iter_leaves())
    _rd = {}
    for node in tree.traverse("preorder"):
        _rd[id(node)] = 0.0 if node.is_root() else _rd[id(node.up)] + node.dist
    D = np.zeros((n_leaves, n_leaves))
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            lca = leaves_ete[i].get_common_ancestor(leaves_ete[j])
            d = _rd[id(leaves_ete[i])] + _rd[id(leaves_ete[j])] - 2.0 * _rd[id(lca)]
            D[i, j] = D[j, i] = d

    root_children = tree.get_tree_root().children
    n_clades = len(root_children)
    clade_id = np.zeros(n_leaves, dtype=int)
    for k, cr in enumerate(root_children):
        for lf in cr.iter_leaves():
            li = leaf_index.get(lf.name)
            if li is not None:
                clade_id[li] = k

    print("Precomputing D-stat calibration (BM simulation) ...", flush=True)
    tree_path  = str(TREE_FILE)
    tree_struct, bm_leaf_vals = _d_mod.get_or_calibrate(tree_path, tree)

    # ---- Run fast_acr on all selected traits ----
    print(f"\nRunning fast_acr (mode={args.acr_mode}) on {n_sel} traits ...",
          flush=True)
    obs_df = ann[genes].copy()
    obs_df.index = obs_df.index.astype(str)

    tip_df = obs_df.applymap(lambda x: int(x) if x in ("0", "1") else -1)
    obs_matrix = build_obs_matrix(ta, tip_df)
    result = fast_acr(ta, obs_matrix, genes, mode=args.acr_mode)
    print(f"  Done in {result.elapsed_s:.2f}s", flush=True)

    # ---- Build trait_params for each method ----
    print("Building trait_params ...", flush=True)
    df_joint = build_joint_params(result, ta, upper_bound, genes)
    df_flow  = build_flow_params(result, ta, upper_bound, genes)
    df_pi    = build_pi_params(result, ta, genes)

    common_genes = (
        set(df_joint.index) & set(df_flow.index) & set(df_pi.index)
        & set(ann.index.intersection(leaf_list).__class__([]))  # dummy — overridden below
    )
    # Keep genes valid for all three methods
    common_genes = set(df_joint.index) & set(df_flow.index) & set(df_pi.index)
    eval_genes   = sorted(common_genes)
    print(f"  {len(eval_genes)} genes valid for all three methods", flush=True)

    # ---- Simulate and score ----
    records = []
    print(f"\nSimulating ({args.trials} trials) and scoring ...", flush=True)
    for i, gene in enumerate(eval_genes):
        if i % 20 == 0:
            print(f"  {i}/{len(eval_genes)} ...", flush=True)

        obs_states = {
            name: int(obs_df.loc[name, gene])
            for name in leaf_list if name in obs_df.index
        }
        if len(obs_states) < 10:
            continue

        for method_name, df_params in [("JOINT", df_joint),
                                        ("FLOW",  df_flow),
                                        ("PI",    df_pi)]:
            if gene not in df_params.index:
                continue
            params_row = df_params.loc[[gene]]
            # sim_bit expects columns: gains, losses, gain_subsize, loss_subsize,
            # dist, loss_dist, root_state  (all in one DataFrame indexed by gene)
            lineages = sim_bit(tree, params_row, trials=args.trials)
            # lineages shape: (n_nodes, 1, n_chunks) — take leaf rows, squeeze trait dim
            leaf_node_order = list(tree)   # same order as sim's node_map
            leaf_rows = [j for j, node in enumerate(leaf_node_order)
                         if node.is_leaf()]
            leaf_names_sim = [leaf_node_order[j].name for j in leaf_rows]
            packed = lineages[leaf_rows, 0, 0]   # (n_leaves,) uint64

            # Re-order to match leaf_list
            name_to_packed = dict(zip(leaf_names_sim, packed))
            packed_ordered = np.array(
                [name_to_packed.get(name, np.uint64(0)) for name in leaf_list],
                dtype=np.uint64,
            )

            m = compute_metrics(
                tree, packed_ordered,
                gene, obs_df.apply(lambda col: col.map(lambda x: int(x) if x in ("0","1") else 0)),
                leaf_list, leaf_index,
                D, clade_id, n_clades,
                tree_struct, bm_leaf_vals, tree_path,
                trials=args.trials,
            )
            records.append({"method": method_name, "gene": gene, **m})

    df = pd.DataFrame(records)
    print(f"\n{'='*60}")
    print("SIMULATION CALIBRATION vs OBSERVED  (median across genes)")
    print(f"{'='*60}")
    _print_summary(df)

    # Per-quartile breakdown by observed prevalence
    print("Prevalence error breakdown by observed prevalence quartile:")
    qs = df[df["method"] == "JOINT"]["obs_prev"].quantile([0, .25, .5, .75, 1.0]).values
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask_genes = df[(df["method"] == "JOINT") &
                        (df["obs_prev"] >= lo) & (df["obs_prev"] <= hi)]["gene"].values
        sub = df[df["gene"].isin(mask_genes)]
        if len(sub) == 0:
            continue
        print(f"  prev ∈ [{lo:.2f}, {hi:.2f}]  n={len(mask_genes):3d}  ", end="")
        for m in ["JOINT", "FLOW", "PI"]:
            val = sub[sub["method"] == m]["prev_err"].median()
            print(f"  {m}={val:.3f}", end="")
        print()
    print()


if __name__ == "__main__":
    main()
