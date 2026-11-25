#!/usr/bin/env python
"""
Calibration and power of a SimPhyNI results table on the synthetic large set.

Pair classes
  planted   pos1/pos05/neg1/neg05 (whole tree), rpos/rneg (inside a shared region)
  shared    independent traits restricted to the same region (co-restricted null)
  separate  restricted traits from different regions (null)
  null      every other pair (independent)

usage: eval_stats.py <syn_dir> <results.parquet> [more results ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

syn = Path(sys.argv[1])
lab = pd.read_csv(syn / "pairs_labeled.csv")
key = lambda a, b: a + "|" + b if a < b else b + "|" + a
lk = {key(a, b): (l, k, i) for a, b, l, k, i in lab.itertuples(index=False)}
ALPHAS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
rows = []
for f in sys.argv[2:]:
    d = pd.read_parquet(f, columns=["T1", "T2", "direction", "pval_naive", "pval_by", "pval_bh"])
    k = [key(a, b) for a, b in zip(d.T1.astype(str), d.T2.astype(str))]
    info = [lk.get(x) for x in k]
    kind = np.array([i[1] if i else None for i in info], dtype=object)
    label = np.array([i[0] if i else 0 for i in info])
    r1 = d.T1.str.startswith("rst"); r2 = d.T2.str.startswith("rst")
    g1 = d.T1.str[3:6]; g2 = d.T2.str[3:6]
    sep = (r1 & r2 & (g1 != g2)).to_numpy() & (kind == None)  # noqa: E711
    kind[sep] = "separate"
    kind[kind == None] = "null"                                 # noqa: E711
    p = d.pval_naive.to_numpy(); by = d.pval_by.to_numpy(); dr = d.direction.to_numpy()
    r = {"run": Path(f).parent.name + "/" + Path(f).stem.replace("results_", ""), "pairs": len(d)}
    nul = kind == "null"
    r["null pairs"] = int(nul.sum())
    for a in ALPHAS:
        r[f"null ratio@{a:g}"] = (p[nul] < a).mean() / a
    r["null BY<.05 calls"] = int((by[nul] < 0.05).sum())
    for kk in ("separate", "shared"):
        m = kind == kk
        r[f"{kk} ratio@.01"] = (p[m] < 0.01).mean() / 0.01
        r[f"{kk} BY<.05"] = int((by[m] < 0.05).sum())
    for kk in ("pos1", "pos05", "neg1", "neg05", "rpos", "rneg"):
        m = kind == kk
        sgn = 1 if kk.startswith(("pos", "rpos")) else -1
        r[f"recall {kk}@BY.05"] = ((by[m] < 0.05) & (dr[m] == sgn)).mean()
    s = -np.log10(np.clip(p, 1e-300, 1))
    for sgn, nm in ((1, "+"), (-1, "−")):
        y = np.isin(kind, ["pos1", "pos05"] if sgn == 1 else ["neg1", "neg05"])
        r[f"AP{nm} genome-wide"] = average_precision_score(y, s * np.where(dr == sgn, 1, -1))
    planted = np.isin(kind, ["pos1", "pos05", "neg1", "neg05", "rpos", "rneg"]) & (dr == np.where(label != 0, label, 9))
    for nm, q in (("BH", d.pval_bh.to_numpy()), ("BY", by)):
        calls = q < 0.05
        tp = calls & planted
        r[f"{nm}<.05 calls"] = int(calls.sum()); r[f"{nm}<.05 FDP"] = 1 - tp.sum() / max(calls.sum(), 1)
        r[f"{nm}<.05 recall (planted)"] = tp.sum() / max(np.isin(kind, ["pos1", "pos05", "neg1", "neg05", "rpos", "rneg"]).sum(), 1)
    rows.append(r)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(pd.DataFrame(rows).set_index("run").T.to_string(float_format=lambda x: f"{x:.4g}"))
