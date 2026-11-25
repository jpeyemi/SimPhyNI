"""
Fast paths for the anchored candidate (explore/uncertainty).

sim_bit_fast       : Numba simulation of the gain/loss process for scalar rates,
                     no masks and no emergence threshold.  Each 64-trial word
                     draws its Bernoulli(1 - exp(-rate * bl)) event mask by
                     geometric skipping, so a branch costs O(expected events)
                     instead of 64 uniforms.  Same distribution as sim_bit.
anchored_scores    : vectorised anchored scoring.  Every pair gets the observed
                     first trait against the second trait's replicates and the
                     reverse; tails are exact counts, refined with the Gaussian
                     KDE only where the count is small.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.special import ndtr


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _bern_word(p, nbits):
    """uint64 with each of the low nbits set independently with probability p."""
    if p <= 0.0:
        return np.uint64(0)
    if p >= 1.0:
        return np.uint64(0xFFFFFFFFFFFFFFFF) if nbits == 64 else (np.uint64(1) << np.uint64(nbits)) - np.uint64(1)
    w = np.uint64(0)
    if p < 0.3:
        lq = math.log1p(-p)
        k = 0
        while True:
            u = np.random.random()
            k += int(math.log(1.0 - u) / lq)
            if k >= nbits:
                break
            w |= np.uint64(1) << np.uint64(k)
            k += 1
    else:
        for k in range(nbits):
            if np.random.random() < p:
                w |= np.uint64(1) << np.uint64(k)
    return w


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(parallel=True, cache=True)
def _sim_kernel(preorder, parent, bl, leaf_idx, gain_rate, loss_rate, root, n_chunks, last_bits,
                seeds, mult):
    n_nodes = preorder.shape[0]
    n_traits = gain_rate.shape[0]
    out = np.zeros((leaf_idx.shape[0], n_traits, n_chunks), dtype=np.uint64)
    for t in prange(n_traits):
        np.random.seed(seeds[t])
        st = np.zeros((n_nodes, n_chunks), dtype=np.uint64)
        for oi in range(n_nodes):
            v = preorder[oi]
            for c in range(n_chunks):
                nb = 64 if c < n_chunks - 1 else last_bits
                full = np.uint64(0xFFFFFFFFFFFFFFFF) if nb == 64 else (np.uint64(1) << np.uint64(nb)) - np.uint64(1)
                if parent[v] < 0:
                    st[v, c] = full if root[t] else np.uint64(0)
                    continue
                s = st[parent[v], c]
                m = mult[t, v]
                if m <= 0.0:
                    st[v, c] = s
                    continue
                pg = 1.0 - math.exp(-gain_rate[t] * bl[v] * m)
                pl = 1.0 - math.exp(-loss_rate[t] * bl[v] * m)
                g = _bern_word(pg, nb) & ~s
                l = _bern_word(pl, nb) & s
                st[v, c] = ((s | g) & ~l) & full
        for li in range(leaf_idx.shape[0]):
            for c in range(n_chunks):
                out[li, t, c] = st[leaf_idx[li], c]
    return out


class _TreeArr:
    pass


def tree_arrays(tree, branch_cap=None):
    nodes = list(tree.traverse("preorder"))
    idx = {n: i for i, n in enumerate(nodes)}
    ta = _TreeArr()
    ta.preorder = np.arange(len(nodes), dtype=np.int64)
    ta.parent = np.array([-1 if n.up is None else idx[n.up] for n in nodes], dtype=np.int64)
    bl = np.array([0.0 if n.up is None else n.dist for n in nodes])
    ta.bl = bl if branch_cap is None else np.minimum(bl, branch_cap)
    ta.leaf_idx = np.array([idx[l] for l in tree], dtype=np.int64)   # tree leaf order
    ta.nodes = nodes
    return ta


def sim_bit_fast(tree, trait_params, trials, branch_cap=None, seed=None, ta=None, elig=None):
    """(n_tips, n_traits, ceil(trials/64)) uint64, tips in tree leaf order.
    elig : optional (n_traits, n_nodes) bool or float rate multipliers, nodes in
           preorder (0 = no events on that branch)."""
    ta = ta or tree_arrays(tree, branch_cap)
    g = trait_params["gains"].to_numpy(float); l = trait_params["losses"].to_numpy(float)
    gs = trait_params["gain_subsize"].to_numpy(float); ls = trait_params["loss_subsize"].to_numpy(float)
    gr = np.divide(g, gs, out=np.zeros_like(g), where=gs > 0)
    lr = np.divide(l, ls, out=np.zeros_like(l), where=ls > 0)
    root = (trait_params["root_state"].to_numpy() > 0)
    n_chunks = (trials + 63) // 64
    last = trials - 64 * (n_chunks - 1)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=len(g))
    if elig is None:
        elig = np.ones((len(g), len(ta.parent)))
    return _sim_kernel(ta.preorder, ta.parent, ta.bl, ta.leaf_idx, gr, lr, root, n_chunks, last, seeds,
                       np.ascontiguousarray(elig, dtype=np.float64))


def unpack_u8(L, trials):
    n_tips, n_tr, _ = L.shape
    return np.unpackbits(L.view(np.uint8), axis=-1, bitorder="little").reshape(n_tips, n_tr, -1)[:, :, :trials]


# ---------------------------------------------------------------------------
# Vectorised anchored scoring
# ---------------------------------------------------------------------------

def _lor(a, s1, s2, W, eps=1.0):
    return np.log(((a + eps) * (W - s1 - s2 + a + eps)) / ((s1 - a + eps) * (s2 - a + eps)))


def _kde_tails(null, stat):
    """Row-wise Gaussian-KDE tails (Silverman bandwidth). null (E, N), stat (E,)."""
    N = null.shape[1]
    h = (N * 3.0 / 4.0) ** (-0.2) * null.std(axis=1, ddof=1)
    h = np.maximum(h, 1e-6)
    z = (stat[:, None] - null) / h[:, None]
    return ndtr(z).mean(axis=1), ndtr(-z).mean(axis=1)


def _score_cells(co, so_a, sr_b, st, W, kde_below):
    null = _lor(co, so_a[:, None], sr_b, W)
    tol = 1e-9
    le = (null <= st[:, None] + tol).sum(1); ge = (null >= st[:, None] - tol).sum(1)
    N = null.shape[1]
    pe = np.minimum(1.0, 2.0 * (np.minimum(le, ge) + 1.0) / (N + 1.0))
    small = np.minimum(le, ge) < kde_below
    if small.any():
        pl, pr = _kde_tails(null[small], st[small])
        pe[small] = np.minimum(1.0, 2.0 * np.minimum(pl, pr))
    return pe, null.mean(1), null.std(1)


def anchored_entries(U, obs, w, anchors, partners, batch=256, kde_below=20,
                     dense_min=20000, pblock=256, ablock=32):
    """For each (anchor a, partner b): the observed anchor against b's replicates.

    U   : (n_tips, n_traits, N) uint8 replicates
    obs : (n_tips, n_traits) 0/1 observed
    Small requests gather per batch; large ones (all-vs-all screens) use blocked
    float32 matrix products, observed anchors x partner-replicate blocks.
    Returns stat, two-sided p, mean and sd of the null, per entry.
    """
    n, T, N = U.shape
    W = float(w.sum())
    Ow = (obs * w[:, None]).astype(np.float32)                   # n x T
    so = Ow.sum(0)                                              # T
    sr = np.einsum("t,tjk->jk", w.astype(np.float32), U, dtype=np.float32)  # T x N replicate sums
    E = len(anchors)
    stat = _lor(np.einsum("te,te->e", Ow[:, anchors], obs[:, partners].astype(np.float32)),
                so[anchors], (obs * w[:, None])[:, partners].sum(0), W)
    p = np.empty(E); mu = np.empty(E); sd = np.empty(E)
    if E < dense_min:
        for s in range(0, E, batch):
            e = slice(s, min(E, s + batch))
            a, b = anchors[e], partners[e]
            co = np.einsum("te,tek->ek", Ow[:, a], U[:, b, :].astype(np.float32))
            p[e], mu[e], sd[e] = _score_cells(co, so[a], sr[b], stat[e], W, kde_below)
        return stat, p, mu, sd
    pb_of = partners // pblock
    order = np.argsort(pb_of, kind="stable")
    bounds = np.searchsorted(pb_of[order], np.arange(0, T // pblock + 2))
    for k in range(len(bounds) - 1):
        idx = order[bounds[k]:bounds[k + 1]]
        if idx.size == 0:
            continue
        p0 = k * pblock; p1 = min(T, p0 + pblock)
        Uf = U[:, p0:p1, :].reshape(n, -1).astype(np.float32)   # n x (pb*N)
        ab = anchors[idx] // ablock
        o2 = np.argsort(ab, kind="stable"); idx = idx[o2]; ab = ab[o2]
        cuts = np.flatnonzero(np.diff(ab)) + 1
        for grp in np.split(idx, cuts):
            a0 = (anchors[grp[0]] // ablock) * ablock; a1 = min(T, a0 + ablock)
            C = (Ow[:, a0:a1].T @ Uf).reshape(a1 - a0, p1 - p0, N)
            co = C[anchors[grp] - a0, partners[grp] - p0, :]
            p[grp], mu[grp], sd[grp] = _score_cells(co, so[anchors[grp]], sr[partners[grp]],
                                                    stat[grp], W, kde_below)
    return stat, p, mu, sd


def anchored_scores(U, obs, w, pairs_index, combine="max", batch=256, kde_below=20):
    """Anchored p-values for pairs (i, j): max (or mean) over both anchors."""
    i, j = pairs_index[:, 0], pairs_index[:, 1]
    anchors = np.concatenate([i, j]); partners = np.concatenate([j, i])
    stat, p, mu, sd = anchored_entries(U, obs, w, anchors, partners, batch, kde_below)
    P = len(i)
    p1, p2 = p[:P], p[P:]
    pv = np.maximum(p1, p2) if combine == "max" else np.minimum(1.0, (p1 + p2) / 2)
    m = 0.5 * (mu[:P] + mu[P:]); s_ = np.sqrt(0.5 * (sd[:P] ** 2 + sd[P:] ** 2))
    st = stat[:P]
    return pd.DataFrame({"p-value": pv, "direction": np.where(st >= m, 1, -1),
                         "effect size": (st - m) / np.maximum(s_, 1e-9), "obs_stat": st,
                         "p_min": np.nan})


# ---------------------------------------------------------------------------
# Lineage-overlap gate and pair-region null
# ---------------------------------------------------------------------------

def tip_distance_matrix(ta):
    """Patristic distances between tips (capped lengths), tips in tree leaf order."""
    n_nodes = len(ta.parent)
    depth = np.zeros(n_nodes)
    for v in ta.preorder:
        if ta.parent[v] >= 0:
            depth[v] = depth[ta.parent[v]] + ta.bl[v]
    L = ta.leaf_idx
    anc = np.zeros((len(L), n_nodes), dtype=np.float32)
    for r, v in enumerate(L):
        u = v
        while u >= 0:
            anc[r, u] = 1.0
            u = ta.parent[u]
    shared = (anc * ta.bl.astype(np.float32)) @ anc.T
    d = depth[L]
    return d[:, None] + d[None, :] - 2 * shared


def _nearest(D, X):
    """For presence matrix X (n x K), distance from every tip to the nearest present tip."""
    big = np.where(X.T[:, None, :] > 0, D[None, :, :], np.inf)        # K x n x n
    return big.min(axis=2).T                                          # n x K


def nn_similarity(D, Dmean, A, B, dA=None, dB=None):
    """-log of the mean nearest-presence distance between A and B over the mean distance.
    A, B : (n, K) presence matrices (columns paired).  Larger = more co-localised."""
    dB = _nearest(D, B) if dB is None else dB
    dA = _nearest(D, A) if dA is None else dA
    ma = np.where(A > 0, dB, 0.0).sum(0) / np.maximum(A.sum(0), 1)
    mb = np.where(B > 0, dA, 0.0).sum(0) / np.maximum(B.sum(0), 1)
    ma = np.where(np.isfinite(ma), ma, Dmean); mb = np.where(np.isfinite(mb), mb, Dmean)
    return -np.log(0.5 * (ma + mb) / Dmean + 1e-6)


def overlap_gate(D, obs, U, pairs_index, K=64, q=0.95):
    """Flag pairs whose observed lineage overlap (nn_similarity) exceeds the q-quantile
    of the overlap between each observed trait and the partner's null replicates."""
    iu = np.triu_indices(D.shape[0], 1)
    Dmean = D[iu].mean()
    K = min(K, U.shape[2])
    cache_obs, cache_rep = {}, {}
    def dobs(t):
        if t not in cache_obs:
            cache_obs[t] = _nearest(D, obs[:, [t]].astype(float))[:, 0]
        return cache_obs[t]
    def drep(t):
        if t not in cache_rep:
            cache_rep[t] = _nearest(D, U[:, t, :K].astype(float))
        return cache_rep[t]
    nn_obs = np.empty(len(pairs_index)); thr = np.empty(len(pairs_index))
    for k, (i, j) in enumerate(pairs_index):
        a = obs[:, [i]].astype(float); b = obs[:, [j]].astype(float)
        nn_obs[k] = nn_similarity(D, Dmean, a, b, dobs(i)[:, None], dobs(j)[:, None])[0]
        A1 = np.repeat(a, K, 1); B1 = U[:, j, :K].astype(float)
        s1 = nn_similarity(D, Dmean, A1, B1, np.repeat(dobs(i)[:, None], K, 1), drep(j))
        A2 = U[:, i, :K].astype(float); B2 = np.repeat(b, K, 1)
        s2 = nn_similarity(D, Dmean, A2, B2, drep(i), np.repeat(dobs(j)[:, None], K, 1))
        thr[k] = np.quantile(np.concatenate([s1, s2]), q)
    return nn_obs > thr, nn_obs, thr


def region_eligibility(ta, obs, root_states, pairs_index, tips_names=None):
    """Union of the MRCA clades of A's and B's non-root-state tips, per pair.
    Returns (n_pairs, n_nodes) bool over preorder nodes and the region's capped length."""
    n_nodes = len(ta.parent)
    L = ta.leaf_idx
    # descendants tip sets via preorder parents
    n_tips = len(L)
    under = np.zeros((n_nodes, n_tips), dtype=bool)
    for r, v in enumerate(L):
        u = v
        while u >= 0:
            under[u, r] = True
            u = ta.parent[u]
    depth = np.zeros(n_nodes)
    for v in ta.preorder:
        if ta.parent[v] >= 0:
            depth[v] = depth[ta.parent[v]] + 1
    def clade(t):
        m = obs[:, t] != root_states[t]
        if not m.any():
            return np.zeros(n_nodes, bool)
        cand = np.flatnonzero(under[:, m].all(axis=1))
        mrca = cand[np.argmax(depth[cand])]
        return ~(under & ~under[mrca]).any(axis=1)
    cl = {}
    E = np.zeros((len(pairs_index), n_nodes), bool)
    for k, (i, j) in enumerate(pairs_index):
        for t in (i, j):
            if t not in cl:
                cl[t] = clade(t)
        E[k] = cl[i] | cl[j]
    region_bl = (E * ta.bl[None, :]).sum(1)
    return E, region_bl


# ---------------------------------------------------------------------------
# Full candidate pipeline
# ---------------------------------------------------------------------------

def run_anchored(tree, trait_params, pairs_index, obs_traits, trials=512, branch_cap=None,
                 seed=None, weighted=True, combine="max", gate=None, gate_q=0.95, gate_p=None,
                 refine_p=None, refine_stages=(4096, 32768), refine_h=10, info=None,
                 lineage=None, tau=0.1, sup_p=0.05, composite=False, confine_alpha=0.1):
    """Weighted anchored test with optional lineage-overlap gate (pair-region null for
    gated pairs) and Monte-Carlo refinement of pairs with p < refine_p.

    Returns a DataFrame aligned to pairs_index.
    """
    from .uncertainty import gsc_weights
    ta = tree_arrays(tree, branch_cap)
    obs = (np.nan_to_num(np.asarray(obs_traits, float)) > 0.5).astype(np.float32)
    w = gsc_weights(tree, branch_cap).astype(np.float32) if weighted else np.ones(obs.shape[0], np.float32)
    rng = np.random.default_rng(seed)
    mult = None
    if lineage == "soft_auto":
        tau, scores = choose_tau(tree, ta, obs, trait_params, seed=int(rng.integers(2**31)))
        if info is not None:
            info["tau"] = tau; info["tau_scores"] = scores
        lineage = None if tau is None else "soft"
    if lineage == "soft_adaptive":
        conf, pit = confined_traits(tree, ta, obs, trait_params, alpha=confine_alpha,
                                    seed=int(rng.integers(2**31)))
        mult = np.ones((obs.shape[1], len(ta.parent)))
        if conf.any():
            ms = lineage_multipliers(tree, ta, obs, trait_params, "soft", tau=tau)
            mult[conf] = ms[conf]
        if info is not None:
            info["confined"] = conf
        trait_params = scale_subsizes(trait_params, ta, mult)
        lineage = None
    if lineage in ("mrca", "soft", "dist", "adaptive"):
        Up = None
        if lineage == "adaptive":
            Up = sim_bit_fast(tree, trait_params, trials, branch_cap, int(rng.integers(2**31)), ta)
        mult = lineage_multipliers(tree, ta, obs, trait_params, lineage, tau=tau, U_packed=Up,
                                   trials=trials, gene_names=list(trait_params.index))
        trait_params = scale_subsizes(trait_params, ta, mult)
    U = unpack_u8(sim_bit_fast(tree, trait_params, trials, branch_cap, int(rng.integers(2**31)), ta,
                               mult), trials)
    res = anchored_scores(U, obs, w, pairs_index, combine)
    res["gated"] = False
    res["refined"] = False

    if gate == "overlap":
        D = tip_distance_matrix(ta)
        # Only pairs that could be called need the check (gate_p=None: all pairs).
        cand = np.arange(len(pairs_index)) if gate_p is None else \
            np.flatnonzero(res["p-value"].to_numpy() < gate_p)
        flag = np.zeros(len(pairs_index), bool)
        res["overlap"] = np.nan
        if cand.size:
            f_c, nn_obs, thr = overlap_gate(D, obs, U, pairs_index[cand], q=gate_q)
            flag[cand] = f_c
            res.loc[cand, "overlap"] = nn_obs
        if flag.any():
            gp = pairs_index[flag]
            root = trait_params["root_state"].to_numpy().astype(int)
            E, rbl = region_eligibility(ta, obs, root, gp)
            # two copies per gated pair, restricted to the pair's region
            src = gp.reshape(-1)
            P2 = trait_params.iloc[src].copy()
            rb = np.repeat(rbl, 2)
            P2["gain_subsize"] = rb; P2["loss_subsize"] = rb
            el = np.repeat(E, 2, axis=0)
            Ur = unpack_u8(sim_bit_fast(tree, P2, trials, branch_cap, int(rng.integers(2**31)), ta, el), trials)
            # entries: observed A vs region copy of B, observed B vs region copy of A
            n_g = len(gp)
            Ucat = np.concatenate([U[:, :0, :], Ur], axis=1)       # copies only
            obs_cat = obs[:, src]                                  # observed columns aligned to copies
            k = np.arange(n_g)
            pi = np.stack([2 * k, 2 * k + 1], axis=1)
            rres = anchored_scores(Ucat, obs_cat, w, pi, combine)
            idx = np.flatnonzero(flag)
            for c in ("p-value", "direction", "effect size"):
                res.loc[idx, c] = rres[c].to_numpy()
            res.loc[idx, "gated"] = True

    if lineage == "sup" or composite:
        # Composite null: independence under the whole-tree model OR under the
        # pair-region model; report the supremum p.  Computed only where it can
        # matter (p < sup_p >= the significance level), which leaves BH/BY
        # rejections at levels <= sup_p unchanged.
        cand = np.flatnonzero(res["p-value"].to_numpy() < sup_p)
        res["p_region"] = np.nan
        if cand.size:
            gp = pairs_index[cand]
            root = trait_params["root_state"].to_numpy().astype(int)
            E, rbl = region_eligibility(ta, obs, root, gp)
            src = gp.reshape(-1)
            P2 = trait_params.iloc[src].copy()
            rb = np.repeat(rbl, 2)
            P2["gain_subsize"] = rb; P2["loss_subsize"] = rb
            el = np.repeat(E, 2, axis=0)
            Ur = unpack_u8(sim_bit_fast(tree, P2, trials, branch_cap, int(rng.integers(2**31)), ta, el), trials)
            k = np.arange(len(gp))
            rres = anchored_scores(Ur, obs[:, src], w, np.stack([2 * k, 2 * k + 1], axis=1), combine)
            pr = rres["p-value"].to_numpy()
            res.loc[cand, "p_region"] = pr
            res.loc[cand, "p-value"] = np.maximum(res.loc[cand, "p-value"].to_numpy(), pr)
            res.loc[cand, "gated"] = pr > res.loc[cand, "p-value"].to_numpy() - 1e-15

    if refine_p is not None:
        cand = np.flatnonzero((res["p-value"].to_numpy() < refine_p) & ~res["gated"].to_numpy())
        if cand.size:
            from .uncertainty import anchored_mc_pairs
            def _simulate(tidx, N):
                return unpack_u8(sim_bit_fast(tree, trait_params.iloc[tidx], N, branch_cap,
                                              int(rng.integers(2**31)), ta), N)
            mc = anchored_mc_pairs(_simulate, pairs_index[cand], obs, refine_stages[0],
                                   weights=w, stages=tuple(refine_stages[1:]), h=refine_h,
                                   combine=combine)
            res.loc[cand, "p-value"] = mc["p-value"].to_numpy()
            res.loc[cand, "refined"] = True
            res.loc[cand, "n_null"] = mc["n_null"].to_numpy()
    return res


# ---------------------------------------------------------------------------
# Per-trait lineage accessibility (model refinement)
# ---------------------------------------------------------------------------

def _under_and_depth(ta):
    n_nodes = len(ta.parent); L = ta.leaf_idx
    under = np.zeros((n_nodes, len(L)), dtype=bool)
    for r, v in enumerate(L):
        u = v
        while u >= 0:
            under[u, r] = True
            u = ta.parent[u]
    depth = np.zeros(n_nodes); level = np.zeros(n_nodes)
    for v in ta.preorder:
        if ta.parent[v] >= 0:
            depth[v] = depth[ta.parent[v]] + ta.bl[v]; level[v] = level[ta.parent[v]] + 1
    return under, depth, level


def trait_clades(ta, obs, root_states):
    """(n_traits, n_nodes) bool: nodes in the MRCA clade (incl. its stem) of each
    trait's non-root-state tips, and the MRCA node index."""
    under, depth, level = _under_and_depth(ta)
    T = obs.shape[1]
    C = np.ones((T, len(ta.parent)), dtype=bool); M = np.zeros(T, dtype=int)
    for t in range(T):
        m = obs[:, t] != root_states[t]
        if not m.any():
            continue
        cand = np.flatnonzero(under[:, m].all(axis=1))
        mrca = cand[np.argmax(level[cand])]
        M[t] = mrca
        C[t] = ~(under & ~under[mrca]).any(axis=1)
    return C, M


def node_distances(ta):
    n = len(ta.parent)
    anc = np.zeros((n, n), dtype=np.float32)          # anc[v, u] = 1 if u on path root..v (incl v)
    for v in range(n):
        u = v
        while u >= 0:
            anc[v, u] = 1.0
            u = ta.parent[u]
    depth = anc @ ta.bl.astype(np.float32)
    shared = (anc * ta.bl.astype(np.float32)) @ anc.T
    return depth[:, None] + depth[None, :] - 2 * shared


def lineage_multipliers(tree, ta, obs, trait_params, mode, tau=0.1, U_packed=None, trials=None,
                        depth_frac=0.2, alpha=0.05, gene_names=None):
    """Per-trait branch rate multipliers (n_traits, n_nodes in preorder), each from
    the trait's own data only.

    mrca     : 1 inside the trait's MRCA clade, 0 elsewhere
    soft     : exp(-d / (tau * mean tip depth)), d = distance from the MRCA node
               for branches outside the clade (1 inside)
    dist     : 1 where the node is at or below the trait's emergence depth
    adaptive : phylogroups excluded when the trait is absent there and an
               unrestricted simulation (U_packed) almost never leaves them empty
    """
    root = trait_params["root_state"].to_numpy().astype(int)
    T = obs.shape[1]
    if mode == "mrca":
        C, _ = trait_clades(ta, obs, root)
        return C.astype(float)
    if mode == "soft":
        C, M = trait_clades(ta, obs, root)
        D = node_distances(ta)
        _, depth, _ = _under_and_depth(ta)
        scale = tau * depth[ta.leaf_idx].mean()
        mult = np.exp(-D[M] / max(scale, 1e-12))
        return np.where(C, 1.0, mult)
    if mode == "dist":
        # thresholds come from ACR on uncapped lengths: compare uncapped depths
        raw = np.array([0.0 if n.up is None else n.dist for n in ta.nodes])
        depth = np.zeros(len(raw))
        for v in ta.preorder:
            if ta.parent[v] >= 0:
                depth[v] = depth[ta.parent[v]] + raw[v]
        d = np.nan_to_num(trait_params["dist"].to_numpy(float), posinf=0.0)
        m = (depth[None, :] >= d[:, None]).astype(float)
        m[m.sum(1) == 0] = 1.0                      # never leave a trait without branches
        return m
    if mode == "adaptive":
        from .simulation import adaptive_phylogroup_mask
        obsdf = pd.DataFrame(obs, index=[l.name for l in tree], columns=list(gene_names))
        gm, lm, frac, n_excl = adaptive_phylogroup_mask(
            tree, U_packed, trials, obsdf, list(gene_names),
            pd.Series(root, index=list(gene_names)), depth_frac, alpha)
        lvl = list(tree.traverse())
        pos = {n: i for i, n in enumerate(lvl)}
        order = [pos[n] for n in ta.nodes]
        return gm[order, :].T.astype(float)
    raise ValueError(f"unknown lineage mode {mode!r}")


def scale_subsizes(trait_params, ta, mult):
    """Rescale rate denominators by the accessible share of (capped) branch length."""
    P = trait_params.copy()
    tot = ta.bl.sum()
    frac = (mult * ta.bl[None, :]).sum(1) / max(tot, 1e-12)
    frac = np.clip(frac, 1e-6, 1.0)
    P["gain_subsize"] = P["gain_subsize"].to_numpy(float) * frac
    P["loss_subsize"] = P["loss_subsize"].to_numpy(float) * frac
    return P


def clade_spread(under, X):
    """log(MRCA clade size / number present) for each column of X (n_tips x K, 0/1).
    Larger = more scattered across the tree.  Columns with < 2 present get NaN."""
    cnt = under.astype(np.float32) @ X.astype(np.float32)          # nodes x K
    tot = X.sum(0)
    size = under.sum(1).astype(float)
    full = cnt >= tot[None, :] - 0.5
    clade = np.where(full, size[:, None], np.inf).min(0)            # smallest node holding all
    out = np.log(clade / np.maximum(tot, 1))
    out[tot < 2] = np.nan
    return out


def choose_tau(tree, ta, obs, trait_params, grid=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, None),
               K=64, seed=None):
    """Pick the soft-accessibility scale whose simulated single-trait clade spread
    matches the observed one across traits (KS distance of per-trait PIT values to
    uniform).  Uses no pair information.  Returns (tau or None, {tau: ks})."""
    under, _, _ = _under_and_depth(ta)
    root = trait_params["root_state"].to_numpy().astype(int)
    minority = np.where(root[None, :] == 1, 1 - obs, obs)            # non-root-state tips
    s_obs = np.array([clade_spread(under, minority[:, [t]])[0] for t in range(obs.shape[1])])
    rng = np.random.default_rng(seed)
    scores = {}
    for tau in grid:
        if tau is None:
            mult, P = None, trait_params
        else:
            mult = lineage_multipliers(tree, ta, obs, trait_params, "soft", tau=tau)
            P = scale_subsizes(trait_params, ta, mult)
        U = unpack_u8(sim_bit_fast(tree, P, K, None, int(rng.integers(2**31)), ta, mult), K)
        pit = []
        for t in range(obs.shape[1]):
            if not np.isfinite(s_obs[t]):
                continue
            Xm = U[:, t, :] if root[t] == 0 else 1 - U[:, t, :]
            sim = clade_spread(under, Xm)
            sim = sim[np.isfinite(sim)]
            if len(sim) < 8:
                continue
            u = rng.random()
            pit.append(((sim < s_obs[t]).sum() + u * ((sim == s_obs[t]).sum() + 1)) / (len(sim) + 1))
        pit = np.sort(np.array(pit))
        n = len(pit)
        ks = np.max(np.abs(pit - (np.arange(1, n + 1) - 0.5) / n)) if n else np.inf
        scores[tau] = float(ks)
    best = min(scores, key=scores.get)
    return best, scores


def confined_traits(tree, ta, obs, trait_params, K=64, alpha=0.1, seed=None):
    """Per-trait check: is the trait's clade spread lower than its unrestricted null
    allows?  Returns (confined bool per trait, PIT per trait).  Single-trait data only."""
    under, _, _ = _under_and_depth(ta)
    root = trait_params["root_state"].to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    U = unpack_u8(sim_bit_fast(tree, trait_params, K, None, int(rng.integers(2**31)), ta), K)
    T = obs.shape[1]
    pit = np.full(T, np.nan)
    for t in range(T):
        x = obs[:, [t]] if root[t] == 0 else 1 - obs[:, [t]]
        so = clade_spread(under, x)[0]
        if not np.isfinite(so):
            continue
        Xm = U[:, t, :] if root[t] == 0 else 1 - U[:, t, :]
        sim = clade_spread(under, Xm)
        sim = sim[np.isfinite(sim)]
        if len(sim) < 8:
            continue
        pit[t] = ((sim < so).sum() + 0.5 * (sim == so).sum() + 1) / (len(sim) + 1)
    return np.nan_to_num(pit, nan=1.0) < alpha, pit
