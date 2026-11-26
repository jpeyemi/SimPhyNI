#!/usr/bin/env python
"""Small focused analyses run properly (composite candidates chosen within each run):
random sets of n confined genes, all-vs-all, on the real data and the worst-case replica.
usage: rnull_small_runs.py <large_eval_dir> [n_sets] [set_size]"""
import contextlib, io, sys
from pathlib import Path
import numpy as np, pandas as pd
SRC = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(SRC))
from simphyni import TreeSimulator
from simphyni.Simulation.simulation import build_sim_params
E = Path(sys.argv[1]); NS = int(sys.argv[2]) if len(sys.argv) > 2 else 100; K = int(sys.argv[3]) if len(sys.argv) > 3 else 30
conf = pd.read_csv(E / "rnull" / "confinement.csv"); genes = conf.gene[conf.confined].astype(str).to_numpy()
data = {}
for ds, run in (("real", "runs3/real"), ("worst-case replica", "runs_rnull/restricted")):
    P = build_sim_params(pd.read_csv(E / run / "acr.csv"), counting="FLOW", subsize="ORIGINAL")
    X = pd.read_csv(E / ("real" if ds == "real" else "rnull/restricted") / "traits.csv", index_col=0)
    X.columns = X.columns.astype(str); X.index = X.index.astype(str)
    data[ds] = (P, X, str(E / run / "tree.nwk"))
rng = np.random.default_rng(11)
cfgs = {"default": {}, "composite": {"composite": True}}
tot = {(ds, c): [0, 0] for ds in data for c in cfgs}
for s in range(NS):
    S = list(rng.choice(genes, K, replace=False))
    for ds, (P, X, tr) in data.items():
        Pk = P[P.gene.astype(str).isin(S)]
        for c, kw in cfgs.items():
            with contextlib.redirect_stdout(io.StringIO()):
                sim = TreeSimulator(tree=tr, pastmlfile=Pk, obsdatafile=X[S].copy())
                sim.initialize_simulation_parameters(pre_filter=False)
                sim.run_simulation(cores=4, seed=s, **kw)
            r = sim.get_results()
            tot[(ds, c)][0] += int((r.pval_bh < 0.05).sum()); tot[(ds, c)][1] += int((r.pval_by < 0.05).sum())
for c in cfgs:
    real, fake = tot[("real", c)], tot[("worst-case replica", c)]
    print(f"{c:10s} sets of {K} confined genes x {NS}: real BH calls/set {real[0] / NS:.2f}, false (worst case) {fake[0] / NS:.2f}, "
          f"est. FDR BH <= {fake[0] / max(real[0], 1):.3f} | BY: real {real[1] / NS:.2f}, false {fake[1] / NS:.2f}, est. FDR <= {fake[1] / max(real[1], 1):.3f}")
