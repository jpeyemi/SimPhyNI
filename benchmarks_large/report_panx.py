#!/usr/bin/env python
"""
PanX benchmark report with FDR-based selection and calibration shown together.

For each variant (runs/<tier>/<variant>/<species>/<scenario>_rep*.csv):

  selection   BH (headline), Storey and BY at q < 0.05: recall of planted pairs
              (correct direction) and the observed false discovery proportion (FDP)
  calibration pooled true nulls (synthetic, real-gene twins, late-emerging,
              separately restricted): observed / nominal at 1e-2, 1e-3, 1e-4;
              co-restricted (shared) nulls at 1e-2
  ordering    average precision and recall at precision >= 0.95 when pairs are
              ranked by p, by |effect| and by the shrunken effect (when present)

usage: report_panx.py <runs_tier_dir> <datasets_tier_dir> v1 v2 ... [--md out.md]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from statsmodels.stats.multitest import multipletests

args = sys.argv[1:]
md = None
if "--md" in args:
    k = args.index("--md"); md = args[k + 1]; args = args[:k] + args[k + 2:]
RUNS, DS = Path(args[0]), Path(args[1]); variants = args[2:]
SCEN = ("synthetic", "real_null", "emergent", "restricted")
key = lambda a, b: tuple(sorted((str(a), str(b))))


def storey(p, lam=0.5):
    pi0 = min(1.0, (p > lam).mean() / (1 - lam))
    return np.minimum(multipletests(p, method="fdr_bh")[1] * pi0, 1)


def recall_at_precision(y, score, target=0.95):
    if y.sum() == 0:
        return np.nan
    pr, rc, _ = precision_recall_curve(y, score)
    ok = pr >= target
    return float(rc[ok].max()) if ok.any() else 0.0


rows = []
for v in variants:
    sel = {(k, sc): np.zeros(3) for k in ("BH", "Storey", "BY") for sc in ("synthetic", "emergent", "restricted")}
    nul_p, sh_p = [], []
    order = {"p": [], "effect": [], "shrunk": []}
    rap = {"p": [], "effect": [], "shrunk": []}
    vdir = RUNS / v
    if not vdir.exists():
        continue
    for spd in sorted(vdir.iterdir()):
        if not (spd / "DONE").exists():
            continue
        for scen in SCEN:
            pr = pd.read_csv(DS / spd.name / f"pairs_{scen}.csv")
            lab = {key(a, b): l for a, b, l in zip(pr.T1, pr.T2, pr.label)}
            kind = {key(a, b): k for a, b, k in zip(pr.T1, pr.T2, pr["kind"])} if "kind" in pr else {}
            for f in sorted(spd.glob(f"{scen}_rep*.csv")):
                d = pd.read_csv(f)
                kk = [key(a, b) for a, b in zip(d.T1, d.T2)]
                lbl = np.array([lab.get(x, 0) for x in kk]); knd = np.array([kind.get(x, scen) for x in kk])
                p = d.pval_naive.clip(0, 1).to_numpy()
                if v == "base":
                    p = np.minimum(2 * p, 1)
                dr = d.direction.to_numpy()
                is_null = (lbl == 0) & (knd != "shared")
                nul_p.append(p[is_null]); sh_p.append(p[knd == "shared"])
                if scen in ("synthetic", "emergent", "restricted"):
                    hit = (lbl != 0) & (dr == lbl)
                    for name, q in (("BH", multipletests(p, method="fdr_bh")[1]), ("Storey", storey(p)),
                                    ("BY", multipletests(p, method="fdr_by")[1])):
                        c = q < 0.05
                        sel[(name, scen)] += (c.sum(), (c & hit).sum(), (lbl != 0).sum())
                if scen == "synthetic":
                    y = (lbl != 0).astype(int)
                    signok = np.where(dr == lbl, 1, -1)
                    scores = {"p": -np.log10(np.clip(p, 1e-300, 1)) * np.where(lbl != 0, signok, 1),
                              "effect": d["effect size"].abs().to_numpy() * np.where(lbl != 0, signok, 1)}
                    if "effect_shrunk" in d.columns:
                        scores["shrunk"] = d["effect_shrunk"].abs().to_numpy() * np.where(lbl != 0, signok, 1)
                    for nm, sc in scores.items():
                        order[nm].append(average_precision_score(y, sc))
                        rap[nm].append(recall_at_precision(y, sc))
    if not nul_p:
        continue
    nul = np.concatenate(nul_p); sh = np.concatenate(sh_p)
    r = {"method": v}
    for name in ("BH", "BY"):
        for sc, ab in (("synthetic", "syn"), ("emergent", "emg"), ("restricted", "rst")):
            calls, tp, pos = sel[(name, sc)]
            r[f"{name} {ab} recall"] = tp / max(pos, 1); r[f"{name} {ab} FDP"] = 1 - tp / max(calls, 1)
    for a in (1e-2, 1e-3, 1e-4):
        r[f"null ratio@{a:g}"] = (nul < a).mean() / a
    r["shared ratio@.01"] = (sh < 0.01).mean() / 0.01
    for nm in ("p", "effect", "shrunk"):
        if order[nm]:
            r[f"AP by {nm}"] = np.mean(order[nm]); r[f"R@P.95 by {nm}"] = np.mean(rap[nm])
    rows.append(r)
T = pd.DataFrame(rows).set_index("method")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(T.round(3).to_string())
if md:
    Path(md).write_text(T.round(3).to_markdown())
