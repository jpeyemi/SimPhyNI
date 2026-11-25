#!/usr/bin/env python
"""Why are deep-tail null p-values too small?  For true-null pairs with KDE p < thr,
recompute the anchored p by exact Monte-Carlo counts (fresh replicates, large N).
usage: diag_tail.py <syn_dir> <run_dir> <results.parquet> [thr] [max_pairs]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from ete3 import Tree
SRC = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(SRC))
from simphyni.Simulation import fastsim as fs
from simphyni.Simulation.uncertainty import gsc_weights, anchored_mc_pairs
from simphyni.Simulation.simulation import build_sim_params, iqr_branch_cap
syn, run, resf = map(Path, sys.argv[1:4])
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 80
lab = pd.read_csv(syn / "pairs_labeled.csv"); lk = set(zip(lab.T1, lab.T2)) | set(zip(lab.T2, lab.T1))
d = pd.read_parquet(resf)
isnull = [(a, b) not in lk and not (a.startswith("rst") and b.startswith("rst")) for a, b in zip(d.T1, d.T2)]
d = d[np.array(isnull) & (d.pval_naive < thr)].sort_values("pval_naive").head(mx)
t = Tree(str(run / "tree.nwk"), 1); cap = iqr_branch_cap(t); ta = fs.tree_arrays(t, cap)
P = build_sim_params(pd.read_csv(run / "acr.csv"), counting="FLOW", subsize="ORIGINAL").set_index("gene")
X = pd.read_csv(syn / "traits.csv", index_col=0); X.index = X.index.astype(str)
traits = sorted(set(d.T1) | set(d.T2)); ti = {g: i for i, g in enumerate(traits)}
obs = (X.loc[[l.name for l in t], traits].to_numpy() > 0.5).astype(np.float32)
w = gsc_weights(t, cap).astype(np.float32); Pk = P.loc[traits]
rng = np.random.default_rng(7)
def simulate(idx, N):
    return fs.unpack_u8(fs.sim_bit_fast(t, Pk.iloc[idx], N, cap, int(rng.integers(2**31)), ta), N)
pi = np.array([[ti[a], ti[b]] for a, b in zip(d.T1, d.T2)])
mc = anchored_mc_pairs(simulate, pi, obs, 4096, weights=w, stages=(32768, 262144), h=10)
d = d.assign(p_mc=mc["p-value"].to_numpy(), n_null=mc["n_null"].to_numpy())
floor = 2 / (262144 + 1)
print(f"{len(d)} null pairs with KDE p < {thr:g}")
print(f"  median KDE p {d.pval_naive.median():.2e}; median MC p {d.p_mc.median():.2e}; MC floor {floor:.1e}")
print(f"  MC p < 1e-4: {(d.p_mc < 1e-4).mean():.2f};  MC p at floor: {(d.p_mc <= floor * 1.01).mean():.2f}")
tr = pd.read_csv(syn / "truth_null.csv").set_index("trait")
for c in ("T1", "T2"):
    d[c + "_model"] = d[c].map(tr["rate_model"]); d[c + "_changes"] = d[c].map(tr.true_gains + tr.true_losses)
print("  rate models of the pair's traits:", pd.concat([d.T1_model, d.T2_model]).value_counts().to_dict())
print("  true changes (median, min):", int(pd.concat([d.T1_changes, d.T2_changes]).median()), int(pd.concat([d.T1_changes, d.T2_changes]).min()))
print(d[["T1", "T2", "pval_naive", "p_mc", "T1_changes", "T2_changes"]].head(12).to_string(index=False))

# compare tail estimators on the same pairs (512 fresh replicates each)
U = simulate(np.arange(len(traits)), 512)
for tl in ("kde", "gpd"):
    fs.TAIL[0] = tl
    r = fs.anchored_scores(U, obs, w, pi)
    pv = r["p-value"].to_numpy()
    lr = np.log10(np.clip(pv, 1e-300, 1)) - np.log10(d.p_mc.to_numpy())
    print(f"  tail={tl}: median p {np.median(pv):.2e}; median log10(p / p_MC) {np.median(lr):+.2f}; "
          f"share p < p_MC/10: {(lr < -1).mean():.2f}; share p < 1e-6: {(pv < 1e-6).mean():.2f}")
