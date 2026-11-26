#!/usr/bin/env python
"""Estimated FDR of real-data calls from realistic null replicas.

estimated FDR = (calls on the null replica) / (calls on the real data), for the same
configuration and q threshold, overall and by pair type (both genes confined, one
confined, neither).  The 'restricted' replica treats all observed confinement as
lineage restriction (upper bound); the 'free' replica has none (lower bound).

usage: summarize_rnull.py <large_eval_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

E = Path(sys.argv[1])
conf = pd.read_csv(E / "rnull" / "confinement.csv").set_index("gene")["confined"]


def calls(f, q=0.05, col="pval_bh"):
    d = pd.read_parquet(f, columns=["T1", "T2", col])
    d = d[d[col] < q]
    c1 = d.T1.map(conf).fillna(False).to_numpy(); c2 = d.T2.map(conf).fillna(False).to_numpy()
    kind = np.where(c1 & c2, "both confined", np.where(c1 | c2, "one confined", "neither"))
    return pd.Series(kind).value_counts().reindex(["both confined", "one confined", "neither"]).fillna(0).astype(int)


n = len(conf); nc = int(conf.sum())
tot_pairs = {"both confined": nc * (nc - 1) // 2, "one confined": nc * (n - nc), "neither": (n - nc) * (n - nc - 1) // 2}
print(f"genes {n}, confined {nc} ({nc / n:.1%}); pairs: " + ", ".join(f"{k} {v:,}" for k, v in tot_pairs.items()))
rows = []
for cfg, label in (("default", "default"), ("soft_auto_composite", "lineage control")):
    for col, qn in (("pval_bh", "BH"), ("pval_by", "BY")):
        real = calls(E / "runs3" / "real" / f"results_{cfg}.parquet", col=col)
        for rep in ("restricted", "free"):
            f = E / "runs_rnull" / rep / f"results_{cfg}.parquet"
            if not f.exists():
                continue
            fake = calls(f, col=col)
            r = {"config": label, "q": qn, "replica": rep, "real calls": int(real.sum()), "replica calls (false)": int(fake.sum()),
                 "est. FDR": fake.sum() / max(real.sum(), 1)}
            for k in ("both confined", "one confined", "neither"):
                r[f"{k}: real / false / est. FDR"] = f"{real[k]} / {fake[k]} / {fake[k] / max(real[k], 1):.2f}"
            rows.append(r)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(E / "rnull_summary.csv", index=False)
