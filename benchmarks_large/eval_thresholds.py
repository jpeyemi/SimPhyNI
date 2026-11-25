#!/usr/bin/env python
"""Power at fixed genome-wide thresholds on a common pair sample (planted pairs +
random null pairs), so methods run on all pairs and on the sample are comparable.
usage: eval_thresholds.py <syn_dir> <pairs_sample.csv> <results.parquet> [...]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
syn = Path(sys.argv[1]); S = pd.read_csv(sys.argv[2])
key = lambda a, b: a + "|" + b if a < b else b + "|" + a
sample = {key(a, b) for a, b in zip(S.T1, S.T2)}
lab = pd.read_csv(syn / "pairs_labeled.csv"); lk = {key(a, b): k for a, b, k in zip(lab.T1, lab.T2, lab.kind)}
M_GENOME = 8402950
BONF = 0.05 / M_GENOME
rows = []
for f in sys.argv[3:]:
    d = pd.read_parquet(f, columns=["T1", "T2", "direction", "pval_naive"])
    k = np.array([key(a, b) for a, b in zip(d.T1.astype(str), d.T2.astype(str))])
    d = d[np.isin(k, list(sample))]; k = k[np.isin(k, list(sample))]
    kind = np.array([lk.get(x, "null") for x in k], dtype=object)
    sep = d.T1.str.startswith("rst").to_numpy() & d.T2.str.startswith("rst").to_numpy() & (kind == "null")
    kind[sep] = "separate"
    p = d.pval_naive.to_numpy(); dr = d.direction.to_numpy()
    r = {"run": Path(f).parent.name + "/" + Path(f).stem.replace("results_", ""), "pairs": len(d)}
    nul = kind == "null"; r["nulls"] = int(nul.sum())
    for a in (1e-3, 1e-4, 1e-5, 1e-6):
        r[f"null ratio@{a:g}"] = (p[nul] < a).mean() / a
    r["null p<Bonf"] = int((p[nul] < BONF).sum())
    for kk, sg in (("pos1", 1), ("pos05", 1), ("neg1", -1), ("neg05", -1), ("rpos", 1)):
        m = kind == kk
        r[f"{kk} p<1e-6"] = ((p[m] < 1e-6) & (dr[m] == sg)).mean()
        r[f"{kk} p<Bonf"] = ((p[m] < BONF) & (dr[m] == sg)).mean()
    s = -np.log10(np.clip(p, 1e-300, 1))
    for sg, nm in ((1, "+"), (-1, "−")):
        y = np.isin(kind, ["pos1", "pos05"] if sg == 1 else ["neg1", "neg05"])
        r[f"AP{nm}"] = average_precision_score(y, s * np.where(dr == sg, 1, -1))
    rows.append(r)
pd.set_option("display.width", 250)
print(f"Bonferroni threshold for {M_GENOME:,} pairs: {BONF:.1e}")
print(pd.DataFrame(rows).set_index("run").T.to_string(float_format=lambda x: f"{x:.4g}"))
