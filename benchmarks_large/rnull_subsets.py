#!/usr/bin/env python
"""Estimated FDR for focused analyses: random gene subsets tested all-vs-all, BH within
each subset, replica (false) calls / real calls.  Per-pair p-values do not depend on the
other pairs tested, so subsets are evaluated from the genome-wide results.
usage: rnull_subsets.py <large_eval_dir> [n_subsets]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
E = Path(sys.argv[1]); NS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
conf = pd.read_csv(E / "rnull" / "confinement.csv").set_index("gene")["confined"]
genes = np.array(conf.index); confined = genes[conf.to_numpy()]
rng = np.random.default_rng(3)

def load(f):
    d = pd.read_parquet(f, columns=["T1", "T2", "pval_naive"])
    return {(a, b) if a < b else (b, a): p for a, b, p in zip(d.T1, d.T2, d.pval_naive)}

rows = []
for cfg in ("default", "soft_auto_composite"):
    real = load(E / "runs3" / "real" / f"results_{cfg}.parquet")
    reps = {r: load(E / "runs_rnull" / r / f"results_{cfg}.parquet") for r in ("restricted", "free")}
    for design, size in (("random genes", 30), ("random genes", 100), ("confined genes only", 30), ("half confined", 30)):
        tot = {"real": 0, "restricted": 0, "free": 0}; pairs = 0
        for _ in range(NS):
            if design == "random genes":
                S = rng.choice(genes, size, replace=False)
            elif design == "confined genes only":
                S = rng.choice(confined, size, replace=False)
            else:
                S = np.concatenate([rng.choice(confined, size // 2, replace=False),
                                    rng.choice(np.setdiff1d(genes, confined), size - size // 2, replace=False)])
            S = sorted(S); keys = [(a, b) for i, a in enumerate(S) for b in S[i + 1:]]
            pairs += len(keys)
            for nm, dct in (("real", real), ("restricted", reps["restricted"]), ("free", reps["free"])):
                p = np.array([dct.get(k, 1.0) for k in keys])
                tot[nm] += int((multipletests(p, method="fdr_bh")[1] < 0.05).sum())
        rows.append({"config": cfg, "design": f"{design} (n={size}, {pairs // NS} pairs)", "real calls / subset": tot["real"] / NS,
                     "false calls / subset (restricted replica)": tot["restricted"] / NS, "false calls / subset (free replica)": tot["free"] / NS,
                     "est. FDR upper (restricted)": tot["restricted"] / max(tot["real"], 1), "est. FDR lower (free)": tot["free"] / max(tot["real"], 1)})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string(index=False))
