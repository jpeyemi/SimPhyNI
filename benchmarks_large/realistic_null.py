#!/usr/bin/env python
"""
Realistic null replicas of the real E. coli gene set, to estimate the false
discovery rate of real-data calls.

For each real gene (5-95 % prevalence):
  - minority state: presence if prevalence <= 0.5, otherwise absence
  - confinement: the smallest cover of its minority-state tips by at most two
    disjoint clades; the gene is "confined" if the cover holds <= HI of the tips
    (HI = 0.35, as in the benchmark's restricted traits)

Replicas (every trait independent, so every pair is a true null):
  restricted/  confined genes -> trait that can only change inside the gene's own
               cover clades (absent at the root, as in the benchmark); other genes
               -> free trait.  Worst case: all confinement is lineage restriction.
  free/        every gene -> free trait (best case: no restriction).
Parameters per trait: the gene's ML (pi, lambda) (synth.fit_mk2_grid); restricted
traits use the gene's prevalence within its cover as pi.  Draws are repeated until
the trait's prevalence is within a factor 1.5 of the gene's (and 5-95 %).

usage: realistic_null.py <panx_benchmark_dir> <real_dataset_dir> <out_dir> [seed]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree
from numba import njit, prange

PANX = Path(sys.argv[1]).resolve(); REAL = Path(sys.argv[2]); OUT = Path(sys.argv[3])
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
sys.path.insert(0, str(PANX))
import synth  # noqa: E402

HI = 0.35
rng = np.random.default_rng(SEED)
tree = Tree(str(REAL / "tree.nwk"), 1)
ft = synth.flatten(tree)
X = pd.read_csv(REAL / "traits.csv", index_col=0)
X.index = X.index.astype(str)
X = X.loc[ft.leaf_names]
G = (X.fillna(0).to_numpy() > 0.5).astype(np.uint8)
n_tips, n_genes = G.shape
n_nodes = len(ft.parent)
under = np.zeros((n_nodes, n_tips), np.uint8)
for r, v in enumerate(ft.leaves):
    u = v
    while u >= 0:
        under[u, r] = 1
        u = ft.parent[u]
size = under.sum(1).astype(np.int64)
depth = np.zeros(n_nodes, np.int64)
for v in ft.preorder[1:]:
    depth[v] = depth[ft.parent[v]] + 1
# ancestor relation via preorder intervals
pre_index = np.arange(n_nodes)
last_desc = np.zeros(n_nodes, np.int64)
for v in ft.postorder:
    kids = ft.children[v]
    last_desc[v] = v if len(kids) == 0 else max(last_desc[c] for c in kids)


@njit(parallel=True, cache=True)
def best_cover(under, size, depth, last_desc, M, hi_tips):
    """Per gene: best total size of a <=2-clade cover of the minority tips, and the
    clades (v1, v2; v2 = -1 for a single clade)."""
    n_nodes, n_tips = under.shape
    T = M.shape[1]
    tot_out = np.full(T, 10**9); v1o = np.full(T, -1); v2o = np.full(T, -1)
    for t in prange(T):
        cnt = np.zeros(n_nodes, np.int64)
        m = 0
        for r in range(n_tips):
            if M[r, t]:
                m += 1
        if m == 0:
            continue
        for v in range(n_nodes):
            c = 0
            for r in range(n_tips):
                if M[r, t] and under[v, r]:
                    c += 1
            cnt[v] = c
        # single clade: deepest node holding all
        best = 10**9; b1 = -1; b2 = -1
        for v in range(n_nodes):
            if cnt[v] == m and size[v] < best:
                best = size[v]; b1 = v; b2 = -1
        # two clades: v holds part, MRCA of the rest outside v's subtree
        for v in range(n_nodes):
            cv = cnt[v]
            if cv == 0 or cv == m or size[v] > hi_tips:
                continue
            rest = m - cv
            bu = -1; bs = 10**9
            for u in range(n_nodes):
                # u must be disjoint from v: neither ancestor nor descendant
                if (u <= v and last_desc[u] >= v) or (u >= v and u <= last_desc[v]):
                    continue
                if cnt[u] == rest and size[u] < bs:
                    bs = size[u]; bu = u
            if bu >= 0 and size[v] + bs < best:
                best = size[v] + bs; b1 = v; b2 = bu
        tot_out[t] = best; v1o[t] = b1; v2o[t] = b2
    return tot_out, v1o, v2o


prev = G.mean(0)
minority = np.where(prev[None, :] <= 0.5, G, 1 - G).astype(np.uint8)
tot, v1, v2 = best_cover(under, size, depth, last_desc, minority, int(HI * n_tips))
confined = tot <= HI * n_tips
genes = list(X.columns)
conf = pd.DataFrame({"gene": genes, "prevalence": prev, "minority_is_presence": prev <= 0.5,
                     "cover_tips": tot, "cover_frac": tot / n_tips, "n_clades": np.where(v2 >= 0, 2, 1),
                     "confined": confined})
OUT.mkdir(parents=True, exist_ok=True)
conf.to_csv(OUT / "confinement.csv", index=False)
print(f"real genes: {n_genes}; confined (<= 2 clades holding <= {HI:.0%} of tips): "
      f"{confined.sum()} ({confined.mean():.1%})", flush=True)

pool = synth.fit_mk2_grid(tree, X)
pi_ml = pool["pi_ml"].to_numpy(); lam_ml = pool["lambda_ml"].to_numpy()


def region_mask(t):
    R = np.zeros(n_nodes, bool)
    for v in (v1[t], v2[t]):
        if v >= 0:
            R[v:last_desc[v] + 1] = True                        # preorder interval = subtree
    return R


def draw(t, restricted, tries=60):
    target = prev[t]
    for _ in range(tries):
        if restricted:
            R = region_mask(t)
            in_tips = R[ft.leaves]
            pin = min(max(minority[in_tips, t].mean(), 0.02), 0.98) if in_tips.any() else 0.5
            x = synth.simulate_in_region(tree, np.array([pin]), np.array([lam_ml[t]]), R[:, None], rng)[:, 0]
            x = x if prev[t] <= 0.5 else 1 - x                  # back from minority coding
        else:
            x = synth.simulate_independent(tree, np.array([pi_ml[t]]), np.array([lam_ml[t]]), rng)[:, 0]
        p = x.mean()
        if 0.05 <= p <= 0.95 and target / 1.5 <= p <= target * 1.5:
            return x
    return x


for name, use_restriction in (("restricted", True), ("free", False)):
    cols = {}
    ok = 0
    for t, g in enumerate(genes):
        x = draw(t, use_restriction and confined[t])
        cols[g] = x
        ok += 0.05 <= x.mean() <= 0.95
    Y = pd.DataFrame(cols, index=ft.leaf_names).astype(np.uint8)
    d = OUT / name
    d.mkdir(exist_ok=True)
    tree.write(outfile=str(d / "tree.nwk"), format=1)
    Y.to_csv(d / "traits.csv")
    print(f"{name}: {Y.shape[1]} traits ({ok} within 5-95 %), prevalence corr with real "
          f"{np.corrcoef(Y.mean(0), prev)[0, 1]:.3f}", flush=True)
json.dump({"HI": HI, "seed": SEED, "n_confined": int(confined.sum())}, open(OUT / "manifest.json", "w"))
