#!/usr/bin/env python
"""
Large-scale evaluation data on the PanX E. coli tree (500 tips).

real/   the 4,501 real accessory genes with 5-95 % prevalence (~10.1 M pairs)
syn/    traits with known truth, generated without SimPhyNI code:
        - null traits: exact F81 paths (gain rate lambda*pi, loss lambda*(1-pi)),
          parameters drawn from the real genes (synth.empirical_param_pool), with
          true change counts and internal-node states recorded
        - interacting pairs: exact 4-state CTMC (synth.simulate_pairs), I = +-0.5, +-1
        - lineage-restricted traits: groups of traits that can only change inside
          the same random clades (synth.simulate_in_region); pairs within a group
          are co-restricted nulls; interacting pairs inside regions
          (synth.simulate_pairs_in_region)
        Every pair not listed in pairs_labeled.csv is an independent null.

usage: make_data.py <panx_benchmark_dir> <out_dir> [n_null] [seed]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree

PANX = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
N_NULL = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 20261125
sys.path.insert(0, str(PANX))
import synth  # noqa: E402

LO, HI = 0.05, 0.95
rng = np.random.default_rng(SEED)
D = PANX / "data" / "Escherichia_coli"
tree = Tree(str(D / "tree.nwk"), 1)
for i, n in enumerate(tree.traverse()):
    if not n.is_leaf() and not n.name:
        n.name = f"N{i}"
leaves = [l.name for l in tree]
G = pd.read_parquet(D / "genes.parquet")
G.index = G.index.astype(str)
G = (G.loc[leaves].fillna(0) > 0).astype(np.uint8)
prev = G.mean(0)
real = G.loc[:, (prev >= LO) & (prev <= HI)]

(OUT / "real").mkdir(exist_ok=True)
tree.write(outfile=str(OUT / "real" / "tree.nwk"), format=1)
real.to_csv(OUT / "real" / "traits.csv")
print(f"real: {real.shape[1]} genes, {real.shape[1] * (real.shape[1] - 1) // 2:,} pairs", flush=True)

# ---------------------------------------------------------------------------
pool = synth.empirical_param_pool(tree, real)
ft = synth.flatten(tree)


def exact_paths(pi, lam):
    """Exact F81 paths: returns tip states, node states and true gain/loss counts."""
    n = len(pi)
    st = np.zeros((len(ft.parent), n), np.uint8)
    st[0] = rng.random(n) < pi
    gains = np.zeros(n, int); losses = np.zeros(n, int)
    for node in ft.preorder[1:]:
        s = st[ft.parent[node]].copy()
        t = np.zeros(n)
        bl = ft.bl[node]
        active = np.ones(n, bool)
        while active.any():
            rate = np.where(s == 0, lam * pi, lam * (1 - pi))
            dt = rng.exponential(1.0 / np.maximum(rate, 1e-300))
            t = t + dt
            flip = active & (t < bl)
            gains += flip & (s == 0); losses += flip & (s == 1)
            s = np.where(flip, 1 - s, s)
            active = flip
        st[node] = s
    return st[ft.leaves], st, gains, losses


def draw_null(n):
    keep_X, keep_S, rows = [], [], []
    while sum(x.shape[1] for x in keep_X) < n:
        P = synth.sample_params(pool, int((n - sum(x.shape[1] for x in keep_X)) * 1.5) + 50, rng)
        X, S, g, l = exact_paths(P["pi"].to_numpy(), P["lambda"].to_numpy())
        ok = synth.prevalence_ok(X, LO, HI)
        keep_X.append(X[:, ok]); keep_S.append(S[:, ok])
        P = P[ok].assign(true_gains=g[ok], true_losses=l[ok], true_root=S[0, ok])
        rows.append(P)
    X = np.concatenate(keep_X, 1)[:, :n]; S = np.concatenate(keep_S, 1)[:, :n]
    P = pd.concat(rows, ignore_index=True).iloc[:n]
    return X, S, P


Xn, Sn, Pn = draw_null(N_NULL)
names = [f"null{i:05d}" for i in range(N_NULL)]
Pn.insert(0, "trait", names)
cols = {nm: Xn[:, i] for i, nm in enumerate(names)}
labeled = []


def draw_pairs(n, inter, prefix, region=None):
    out = []
    while len(out) < n:
        P = synth.sample_params(pool, 2 * (n - len(out)) + 20, rng)
        h = len(P) // 2
        p1, p2 = P["pi"].to_numpy()[:h], P["pi"].to_numpy()[h:2 * h]
        l1, l2 = P["lambda"].to_numpy()[:h], P["lambda"].to_numpy()[h:2 * h]
        I = np.full(h, inter)
        if region is None:
            A, B = synth.simulate_pairs(tree, p1, p2, l1, l2, I, rng)
        else:
            R = synth.random_regions(tree, h, rng)
            A, B = synth.simulate_pairs_in_region(tree, p1, p2, l1, l2, I, R, rng)
        ok = synth.prevalence_ok(A, LO, HI) & synth.prevalence_ok(B, LO, HI) if region is None else \
            synth.prevalence_ok(A, 0.02, HI) & synth.prevalence_ok(B, 0.02, HI)
        for j in np.flatnonzero(ok):
            if len(out) < n:
                out.append((A[:, j], B[:, j]))
    for k, (a, b) in enumerate(out):
        na, nb = f"{prefix}{k:03d}_a", f"{prefix}{k:03d}_b"
        cols[na] = a; cols[nb] = b
        labeled.append((na, nb, int(np.sign(inter)), prefix.rstrip("_"), abs(inter)))


for inter in (1.0, 0.5, -1.0, -0.5):
    tag = {1.0: "pos1_", 0.5: "pos05_", -1.0: "neg1_", -0.5: "neg05_"}[inter]
    draw_pairs(75, inter, tag)

# co-restricted groups: 4 independent traits per region
n_groups = 100
k = 0
while k < n_groups:
    R = synth.random_regions(tree, 1, rng)
    P = synth.sample_params(pool, 4, rng)
    X = synth.simulate_in_region(tree, P["pi"].to_numpy(), P["lambda"].to_numpy(),
                                 np.repeat(R, 4, axis=1), rng)
    if not synth.prevalence_ok(X, 0.02, HI).all():
        continue
    gnames = [f"rst{k:03d}_{j}" for j in range(4)]
    for j, nm in enumerate(gnames):
        cols[nm] = X[:, j]
    for i in range(4):
        for j in range(i + 1, 4):
            labeled.append((gnames[i], gnames[j], 0, "shared", 0.0))
    k += 1
for inter, tag in ((1.0, "rpos_"), (-1.0, "rneg_")):
    draw_pairs(25, inter, tag, region=True)

X = pd.DataFrame(cols, index=leaves).astype(np.uint8)
(OUT / "syn").mkdir(exist_ok=True)
tree.write(outfile=str(OUT / "syn" / "tree.nwk"), format=1)
X.to_csv(OUT / "syn" / "traits.csv")
pd.DataFrame(labeled, columns=["T1", "T2", "label", "kind", "I"]).to_csv(OUT / "syn" / "pairs_labeled.csv", index=False)
Pn.to_csv(OUT / "syn" / "truth_null.csv", index=False)
np.savez_compressed(OUT / "syn" / "truth_null_states.npz", states=Sn, parent=ft.parent, bl=ft.bl,
                    leaves=ft.leaves, names=np.array(ft.names))
json.dump({"n_traits": X.shape[1], "n_pairs": X.shape[1] * (X.shape[1] - 1) // 2, "seed": SEED},
          open(OUT / "syn" / "manifest.json", "w"))
print(f"syn: {X.shape[1]} traits, {X.shape[1] * (X.shape[1] - 1) // 2:,} pairs, "
      f"{len(labeled)} labelled pairs", flush=True)
