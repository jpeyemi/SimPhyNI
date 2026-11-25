#!/usr/bin/env python
"""
Accuracy of ancestral reconstruction and of the null simulator against truth.

ACR  : fast_acr on the synthetic null traits (true paths known)
       - internal-node state accuracy (joint) and Brier score (marginal)
       - root state accuracy
       - total changes: JOINT counts and FLOW expected counts vs true changes
       - optional: PastML on a subset (time and agreement)
SIM  : sim_bit_fast at the TRUE rates vs exact F81 paths, root fixed at the true
       root, uncapped and IQR-capped branches: tip prevalence (mean, sd) and
       parsimony changes per trait.

usage: eval_acr_sim.py <syn_dir> [n_sim_traits] [--pastml N]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.stats import spearmanr

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(SRC / "simphyni" / "scripts"))
from fast_binary_acr import build_tree_arrays, build_obs_matrix, fast_acr  # noqa: E402
from run_ancestral_reconstruction import (_vectorized_stats, compute_branch_upper_bound,  # noqa: E402
                                          label_internal_nodes)
from simphyni.Simulation import fastsim  # noqa: E402
from simphyni.Simulation.simulation import fitch_changes_packed, iqr_branch_cap  # noqa: E402

syn = Path(sys.argv[1])
NS = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 300
truth = pd.read_csv(syn / "truth_null.csv")
Z = np.load(syn / "truth_null_states.npz", allow_pickle=True)
S_true, names = Z["states"], list(Z["names"])
X = pd.read_csv(syn / "traits.csv", index_col=0)
tree = Tree(str(syn / "tree.nwk"), 1)
label_internal_nodes(tree)
ta = build_tree_arrays(tree)
traits = truth.trait.tolist()
obs = X[traits]

# ---------------------------------------------------------------- ACR
SIM_ONLY = "--sim-only" in sys.argv
t = time.perf_counter()
if SIM_ONLY:
    traits = traits[:NS]; obs = obs[traits]
om = None if SIM_ONLY else build_obs_matrix(ta, obs) if build_obs_matrix.__code__.co_argcount == 2 else build_obs_matrix(ta, obs, traits)
res = None if SIM_ONLY else fast_acr(ta, om, traits, mode="ml")
if not SIM_ONLY:
    t_acr = time.perf_counter() - t
    pos = {n: i for i, n in enumerate(names)}
    order = np.array([pos[n] for n in ta.node_names])                 # truth rows in ta order
    St = S_true[order, :].T                                           # traits x nodes
    internal = ~ta.is_leaf
    joint_acc = (res.joint_states[:, internal] == St[:, internal]).mean()
    brier = ((res.marginal_p1[:, internal] - St[:, internal]) ** 2).mean()
    root_acc = (res.joint_states[:, ta.root_idx] == truth.true_root.to_numpy()).mean()
    ub = compute_branch_upper_bound(tree)
    stats = pd.DataFrame(_vectorized_stats(ta, res, traits, {g: int(obs[g].sum()) for g in traits}, ub, "all"))
    true_ch = (truth.true_gains + truth.true_losses).to_numpy()
    joint_ch = (stats.gains + stats.losses).to_numpy()
    flow_ch = (stats.gains_flow + stats.losses_flow).to_numpy()
    print(f"ACR fast_acr: {len(traits)} traits in {t_acr:.1f}s ({1e3 * t_acr / len(traits):.2f} ms/trait)")
    print(f"  internal-node joint state accuracy {joint_acc:.4f}; marginal Brier {brier:.4f}; root accuracy {root_acc:.4f}")
    bins = pd.cut(true_ch, [-1, 2, 5, 10, 20, 50, 1e9], labels=["<=2", "3-5", "6-10", "11-20", "21-50", ">50"])
    tab = pd.DataFrame({"true": true_ch, "joint": joint_ch, "flow": flow_ch, "bin": bins}).groupby("bin", observed=True).agg(
        n=("true", "size"), true_mean=("true", "mean"), joint_mean=("joint", "mean"), flow_mean=("flow", "mean"))
    tab["joint/true"] = tab.joint_mean / tab.true_mean; tab["flow/true"] = tab.flow_mean / tab.true_mean
    print("  changes per trait by true count:\n" + tab.round(3).to_string())
    print(f"  Spearman(true, JOINT) {spearmanr(true_ch, joint_ch).statistic:.3f}; Spearman(true, FLOW) {spearmanr(true_ch, flow_ch).statistic:.3f}")

    if "--pastml" in sys.argv:
        k = int(sys.argv[sys.argv.index("--pastml") + 1])
        try:
            import subprocess, tempfile
            tmp = Path(tempfile.mkdtemp())
            obs.iloc[:, :k].to_csv(tmp / "x.csv")
            t = time.perf_counter()
            r = subprocess.run([sys.executable, str(SRC / "simphyni/scripts/run_ancestral_reconstruction.py"),
                                "--inputs_file", str(tmp / "x.csv"), "--tree_file", str(syn / "tree.nwk"),
                                "--output_csv", str(tmp / "p.csv"), "--reconstruction", "all", "--pastml",
                                "--max_workers", "8"], capture_output=True, text=True)
            tp = time.perf_counter() - t
            if r.returncode == 0:
                P = pd.read_csv(tmp / "p.csv").set_index("gene").loc[traits[:k]]
                pc = (P.gains + P.losses).to_numpy(); fc = joint_ch[:k]
                print(f"  PastML: {k} traits in {tp:.1f}s ({1e3 * tp / k:.1f} ms/trait); "
                      f"Spearman(PastML JOINT, fast JOINT) {spearmanr(pc, fc).statistic:.3f}; "
                      f"Spearman(PastML JOINT, true) {spearmanr(pc, true_ch[:k]).statistic:.3f}; "
                      f"fast JOINT vs true {spearmanr(fc, true_ch[:k]).statistic:.3f}")
            else:
                print("  PastML failed:", r.stderr[-400:])
        except Exception as e:  # noqa: BLE001
            print("  PastML skipped:", e)


# ---------------------------------------------------------------- simulator
sub = truth.iloc[:NS]
pi, lam = sub.pi.to_numpy(), sub["lambda"].to_numpy()
N = 1024
rng = np.random.default_rng(3)
import synth_exact  # noqa: E402  (exact F81 path simulator, no SimPhyNI code)
Xe = synth_exact.exact_tips(tree, pi, lam, sub.true_root.to_numpy(), N, rng)   # tips x traits x N
for cap_name, cap in (("uncapped", None), ("IQR-capped", iqr_branch_cap(tree))):
    P = pd.DataFrame({"gains": lam * pi, "losses": lam * (1 - pi), "gain_subsize": 1.0, "loss_subsize": 1.0,
                      "root_state": sub.true_root.to_numpy()}, index=sub.trait)
    t = time.perf_counter()
    L = fastsim.sim_bit_fast(tree, P, N, cap, 5)
    ts = time.perf_counter() - t
    Uf = fastsim.unpack_u8(L, N).astype(np.float32)
    pe, pf = Xe.mean(0), Uf.mean(0)                                          # traits x N prevalence
    fe = fitch_changes_packed(tree, np.packbits(Xe.astype(np.uint8), axis=-1, bitorder="little").view(np.uint64), N)
    ff = fitch_changes_packed(tree, L, N)
    rel = lambda a, b: np.median(np.abs(a - b) / np.maximum(np.abs(b), 1e-9))
    print(f"SIM {cap_name}: {NS} traits x {N} replicates in {ts:.2f}s; median |rel. diff| vs exact: "
          f"prevalence mean {rel(pf.mean(1), pe.mean(1)):.3f}, prevalence sd {rel(pf.std(1), pe.std(1)):.3f}, "
          f"parsimony changes mean {rel(ff.mean(1), fe.mean(1)):.3f}")
    hi = (lam * fastsim.tree_arrays(tree).bl.max()) > 0.5
    if hi.any():
        print(f"    traits with a branch of > 0.5 expected events ({hi.sum()}): prevalence-sd rel. diff "
              f"{rel(pf.std(1)[hi], pe.std(1)[hi]):.3f}, changes rel. diff {rel(ff.mean(1)[hi], fe.mean(1)[hi]):.3f}")
