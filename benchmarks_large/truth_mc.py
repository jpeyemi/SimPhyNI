#!/usr/bin/env python
"""Exact anchored Monte-Carlo p-values with millions of replicates (chunked), to
check tail estimators on planted pairs.
usage: truth_mc.py <syn_dir> <run_dir> <results.parquet> <kind> [n_pairs] [chunks]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from ete3 import Tree
SRC = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(SRC))
from simphyni.Simulation import fastsim as fs
from simphyni.Simulation.uncertainty import gsc_weights
from simphyni.Simulation.simulation import build_sim_params, iqr_branch_cap
syn, run, resf = map(Path, sys.argv[1:4]); kind = sys.argv[4]
npair = int(sys.argv[5]) if len(sys.argv) > 5 else 20
chunks = int(sys.argv[6]) if len(sys.argv) > 6 else 16
CH = 262144
lab = pd.read_csv(syn / "pairs_labeled.csv"); lab = lab[lab.kind == kind].head(npair)
d = pd.read_parquet(resf, columns=["T1", "T2", "pval_naive"])
key = lambda a, b: a + "|" + b if a < b else b + "|" + a
pk = dict(zip([key(a, b) for a, b in zip(d.T1, d.T2)], d.pval_naive))
t = Tree(str(run / "tree.nwk"), 1); cap = iqr_branch_cap(t); ta = fs.tree_arrays(t, cap)
P = build_sim_params(pd.read_csv(run / "acr.csv"), counting="FLOW", subsize="ORIGINAL").set_index("gene")
X = pd.read_csv(syn / "traits.csv", index_col=0); X.index = X.index.astype(str)
w = gsc_weights(t, cap); W = w.sum(); rng = np.random.default_rng(5)
rows = []
for a, b in zip(lab.T1, lab.T2):
    xa = (X.loc[[l.name for l in t], a].to_numpy() > 0.5).astype(float)
    xb = (X.loc[[l.name for l in t], b].to_numpy() > 0.5).astype(float)
    stat = fs._lor((w * xa * xb).sum(), (w * xa).sum(), (w * xb).sum(), W)
    ge = {}; le = {}
    for anchor, other, nm in ((xa, b, "ab"), (xb, a, "ba")):
        g = l = 0
        for c in range(chunks):
            R = fs.unpack_u8(fs.sim_bit_fast(t, P.loc[[other]], CH, None, int(rng.integers(2**31)), ta), CH)[:, 0, :].astype(np.float32)
            co = (w * anchor).astype(np.float32) @ R
            null = fs._lor(co.astype(float), (w * anchor).sum(), (w[:, None] * R).sum(0), W)
            g += int((null >= stat - 1e-9).sum()); l += int((null <= stat + 1e-9).sum())
        ge[nm], le[nm] = g, l
    N = CH * chunks
    p1 = min(1, 2 * (min(ge["ab"], le["ab"]) + 1) / (N + 1)); p2 = min(1, 2 * (min(ge["ba"], le["ba"]) + 1) / (N + 1))
    rows.append(dict(pair=f"{a}|{b}", stat=round(stat, 3), p_mc=max(p1, p2), n_extreme=min(ge["ab"], le["ab"], ge["ba"], le["ba"]),
                     p_pipeline=pk.get(key(a, b))))
out = pd.DataFrame(rows)
print(f"N = {CH * chunks:,} replicates per direction; MC floor {2 / (CH * chunks + 1):.1e}")
print(out.to_string(index=False))
