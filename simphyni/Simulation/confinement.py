"""
Descriptive lineage confinement of traits and trait pairs (does not change p-values).

A trait is *confined* when the tips carrying its minority state (presence if its
prevalence is <= 0.5, otherwise absence) can be covered by at most two disjoint
clades that together hold no more than `max_frac` of the tips (default 0.35).

A pair is *co_confined* when both traits are confined and their covers share at
least one tip.  For such pairs, a positive association can reflect shared lineage
restriction or co-transfer (e.g. genes on the same island or mobile element)
rather than a functional interaction; see the user guide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange


@njit(parallel=True, cache=True)
def _best_cover(under, size, last_desc, M, hi_tips):
    n_nodes, n_tips = under.shape
    T = M.shape[1]
    tot = np.full(T, 1 << 30); v1o = np.full(T, -1); v2o = np.full(T, -1)
    for t in prange(T):
        m = 0
        for r in range(n_tips):
            if M[r, t]:
                m += 1
        if m == 0 or m > hi_tips:
            continue
        cnt = np.zeros(n_nodes, np.int64)
        for v in range(n_nodes):
            c = 0
            for r in range(n_tips):
                if M[r, t] and under[v, r]:
                    c += 1
            cnt[v] = c
        best = 1 << 30; b1 = -1; b2 = -1
        for v in range(n_nodes):
            if cnt[v] == m and size[v] < best:
                best = size[v]; b1 = v; b2 = -1
        if best > hi_tips:
            for v in range(n_nodes):
                cv = cnt[v]
                if cv == 0 or cv == m or size[v] > hi_tips:
                    continue
                rest = m - cv
                bs = 1 << 30; bu = -1
                for u in range(n_nodes):
                    if cnt[u] != rest or size[u] >= bs:
                        continue
                    if (u <= v and last_desc[u] >= v) or (u >= v and u <= last_desc[v]):
                        continue                                    # nested, not disjoint
                    bs = size[u]; bu = u
                if bu >= 0 and size[v] + bs < best:
                    best = size[v] + bs; b1 = v; b2 = bu
        tot[t] = best; v1o[t] = b1; v2o[t] = b2
    return tot, v1o, v2o


def trait_confinement(tree, obs: pd.DataFrame, max_frac: float = 0.35):
    """Per-trait confinement.

    tree : ete3 Tree;  obs : tips x traits 0/1 DataFrame (index = tip names).
    Returns (table, cover) where table has columns trait, confined, cover_frac,
    cover_clades, and cover is a (n_tips, n_traits) bool array of each confined
    trait's cover tips (tree leaf order; all False for unconfined traits).
    """
    nodes = list(tree.traverse("preorder"))
    idx = {n: i for i, n in enumerate(nodes)}
    leaves = list(tree)
    n_nodes, n_tips = len(nodes), len(leaves)
    parent = np.array([-1 if n.up is None else idx[n.up] for n in nodes])
    under = np.zeros((n_nodes, n_tips), np.uint8)
    for r, l in enumerate(leaves):
        u = idx[l]
        while u >= 0:
            under[u, r] = 1
            u = parent[u]
    size = under.sum(1).astype(np.int64)
    last_desc = np.arange(n_nodes)
    for v in range(n_nodes - 1, 0, -1):                      # reverse preorder: children first
        p = parent[v]
        last_desc[p] = max(last_desc[p], last_desc[v])
    X = obs.reindex([l.name for l in leaves]).fillna(0).to_numpy() > 0.5
    prev = X.mean(0)
    M = np.where(prev[None, :] <= 0.5, X, ~X).astype(np.uint8)
    hi = int(np.floor(max_frac * n_tips))
    tot, v1, v2 = _best_cover(under, size, last_desc, M, hi)
    confined = tot <= hi
    cover = np.zeros((n_tips, X.shape[1]), bool)
    for t in np.flatnonzero(confined):
        for v in (v1[t], v2[t]):
            if v >= 0:
                cover[:, t] |= under[v].astype(bool)
    table = pd.DataFrame({
        "trait": list(obs.columns),
        "confined": confined,
        "cover_frac": np.where(confined, tot / n_tips, np.nan),
        "cover_clades": np.where(confined, np.where(v2 >= 0, 2, 1), 0),
    })
    return table, cover


def pair_co_confined(table: pd.DataFrame, cover: np.ndarray, t1, t2) -> np.ndarray:
    """co_confined flag for pairs (t1[i], t2[i]) of trait names."""
    pos = {t: i for i, t in enumerate(table["trait"])}
    conf = table["confined"].to_numpy()
    i = np.array([pos.get(a, -1) for a in t1]); j = np.array([pos.get(b, -1) for b in t2])
    ok = (i >= 0) & (j >= 0)
    both = np.zeros(len(i), bool)
    both[ok] = conf[i[ok]] & conf[j[ok]]
    out = np.zeros(len(i), bool)
    if both.any():
        ci = np.flatnonzero(conf)
        loc = {t: k for k, t in enumerate(ci)}
        C = cover[:, ci].astype(np.float32)
        overlap = (C.T @ C) > 0
        sel = np.flatnonzero(both)
        out[sel] = overlap[[loc[x] for x in i[sel]], [loc[x] for x in j[sel]]]
    return out
