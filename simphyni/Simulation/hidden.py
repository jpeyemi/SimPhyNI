"""
Hidden-accessibility null model (lineage='hidden').

Each trait has a hidden accessibility state H on the tree: a two-state CTMC with
switching rate mu and stationary open probability rho (root ~ rho).  The trait can
change only on open branches.  Tips in the trait's non-root state must be open;
other tips may be open or closed.  Given H, tips are modelled as independent
Bernoulli(pi) in the non-root state when open (an emission approximation of the
trait's evolution inside the open region).

Fit   : (mu, rho, pi) per trait by maximum likelihood on a grid (pruning on H),
        each trait from its own data only.
Null  : every replicate draws a fresh H map from its posterior given the trait,
        top-down: P(H_v = 1 | H_parent = h, data below v); the trait is then
        re-simulated with its gain/loss rates on open branches only (a branch
        whose ends disagree is half open).  Bit-parallel: one Bernoulli word for
        H per branch and chunk, gain/loss words masked by it.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .fastsim import _bern_word


@njit(inline="always", cache=True)
def _p01(mu, rho, t):
    """2-state accessibility transition: P(1|0,t), P(1|1,t)."""
    e = math.exp(-mu * t)
    return rho * (1.0 - e), rho + (1.0 - rho) * e


@njit(cache=True)
def _prune(order_rev, parent, bl, tip_row, xcol, mu, rho, pi, L0, L1):
    """Fill normalised below-partials L0/L1 (nodes); return log-likelihood."""
    n = parent.shape[0]
    a0 = np.ones(n); a1 = np.ones(n)
    logs = 0.0
    for k in range(n):
        v = order_rev[k]
        r = tip_row[v]
        if r >= 0:
            x = xcol[r]
            if x == 1:
                l0, l1 = 0.0, pi
            elif x == 0:
                l0, l1 = 1.0, 1.0 - pi
            else:
                l0, l1 = 1.0, 1.0
        else:
            l0, l1 = a0[v], a1[v]
        m = max(l0, l1)
        if m <= 0.0:
            return -1e300
        l0 /= m; l1 /= m
        logs += math.log(m)
        L0[v] = l0; L1[v] = l1
        p = parent[v]
        if p >= 0:
            p01, p11 = _p01(mu, rho, bl[v])
            a0[p] *= (1.0 - p01) * l0 + p01 * l1
            a1[p] *= (1.0 - p11) * l0 + p11 * l1
    root = order_rev[n - 1]
    return logs + math.log((1.0 - rho) * L0[root] + rho * L1[root])


@njit(parallel=True, cache=True)
def _fit_kernel(order_rev, parent, bl, tip_row, X, mus, rhos, pis):
    T = X.shape[1]; n = parent.shape[0]
    best = np.zeros((T, 3)); bestll = np.full(T, -1e300)
    for t in prange(T):
        L0 = np.empty(n); L1 = np.empty(n)
        xcol = X[:, t]
        for a in range(mus.shape[0]):
            for b in range(rhos.shape[0]):
                for c in range(pis.shape[0]):
                    ll = _prune(order_rev, parent, bl, tip_row, xcol, mus[a], rhos[b], pis[c], L0, L1)
                    if ll > bestll[t]:
                        bestll[t] = ll
                        best[t, 0] = mus[a]; best[t, 1] = rhos[b]; best[t, 2] = pis[c]
    return best, bestll


@njit(parallel=True, cache=True)
def _posterior_kernel(order_rev, parent, bl, tip_row, X, best):
    """Top-down sampling probabilities q0/q1 per node and the root posterior."""
    T = X.shape[1]; n = parent.shape[0]
    q0 = np.zeros((T, n)); q1 = np.zeros((T, n)); proot = np.zeros(T)
    for t in prange(T):
        L0 = np.empty(n); L1 = np.empty(n)
        mu, rho, pi = best[t, 0], best[t, 1], best[t, 2]
        _prune(order_rev, parent, bl, tip_row, X[:, t], mu, rho, pi, L0, L1)
        root = order_rev[n - 1]
        num = rho * L1[root]; den = num + (1.0 - rho) * L0[root]
        proot[t] = num / den if den > 0 else rho
        for v in range(n):
            if parent[v] < 0:
                continue
            p01, p11 = _p01(mu, rho, bl[v])
            d0 = (1.0 - p01) * L0[v] + p01 * L1[v]
            d1 = (1.0 - p11) * L0[v] + p11 * L1[v]
            q0[t, v] = p01 * L1[v] / d0 if d0 > 0 else 0.0
            q1[t, v] = p11 * L1[v] / d1 if d1 > 0 else 0.0
    return q0, q1, proot


@njit(parallel=True, cache=True)
def _open_fraction(preorder, parent, bl, q0, q1, proot, K, seeds):
    """Expected share of (capped) branch length that is open, by sampling K maps."""
    T = q0.shape[0]; n = parent.shape[0]
    out = np.zeros(T)
    tot = 0.0
    for v in range(n):
        if parent[v] >= 0:
            tot += bl[v]
    for t in prange(T):
        np.random.seed(seeds[t])
        acc = 0.0
        H = np.zeros(n, np.uint8)
        for k in range(K):
            for oi in range(n):
                v = preorder[oi]
                p = parent[v]
                if p < 0:
                    H[v] = 1 if np.random.random() < proot[t] else 0
                    continue
                pr = q1[t, v] if H[p] == 1 else q0[t, v]
                H[v] = 1 if np.random.random() < pr else 0
                acc += bl[v] * 0.5 * (H[v] + H[p])
        out[t] = acc / (K * tot) if tot > 0 else 1.0
    return out


@njit(parallel=True, cache=True)
def _sim_hidden_kernel(preorder, parent, bl, leaf_idx, gain_rate, loss_rate, root, n_chunks,
                       last_bits, seeds, q0, q1, proot):
    n = preorder.shape[0]
    T = gain_rate.shape[0]
    out = np.zeros((leaf_idx.shape[0], T, n_chunks), dtype=np.uint64)
    for t in prange(T):
        np.random.seed(seeds[t])
        st = np.zeros((n, n_chunks), dtype=np.uint64)
        H = np.zeros((n, n_chunks), dtype=np.uint64)
        mu_ = gain_rate[t] + loss_rate[t]
        for oi in range(n):
            v = preorder[oi]
            for c in range(n_chunks):
                nb = 64 if c < n_chunks - 1 else last_bits
                full = np.uint64(0xFFFFFFFFFFFFFFFF) if nb == 64 else (np.uint64(1) << np.uint64(nb)) - np.uint64(1)
                if parent[v] < 0:
                    st[v, c] = full if root[t] else np.uint64(0)
                    H[v, c] = _bern_word(proot[t], nb)
                    continue
                p = parent[v]
                hp = H[p, c]
                hv = (hp & _bern_word(q1[t, v], nb)) | (~hp & _bern_word(q0[t, v], nb))
                hv &= full
                H[v, c] = hv
                s = st[p, c]
                both = hp & hv
                half = (hp ^ hv) & full
                if mu_ > 0.0 and (both | half) != np.uint64(0):
                    f1 = 1.0 - math.exp(-mu_ * bl[v])
                    fh = 1.0 - math.exp(-mu_ * bl[v] * 0.5)
                    g = (_bern_word(gain_rate[t] / mu_ * f1, nb) & both) | (_bern_word(gain_rate[t] / mu_ * fh, nb) & half)
                    l = (_bern_word(loss_rate[t] / mu_ * f1, nb) & both) | (_bern_word(loss_rate[t] / mu_ * fh, nb) & half)
                    st[v, c] = ((s | (g & ~s)) & ~(l & s)) & full
                else:
                    st[v, c] = s
        for li in range(leaf_idx.shape[0]):
            for c in range(n_chunks):
                out[li, t, c] = st[leaf_idx[li], c]
    return out


class HiddenModel:
    """Fitted hidden-accessibility model for a set of traits (rows aligned to trait_params)."""

    def __init__(self, ta, obs, root_states, n_mu=10, n_rho=10, n_pi=8, seed=None):
        n_nodes = len(ta.parent)
        tip_row = np.full(n_nodes, -1, np.int64)
        tip_row[ta.leaf_idx] = np.arange(len(ta.leaf_idx))
        self.order_rev = ta.preorder[::-1].copy()
        X = (np.asarray(obs) > 0.5).astype(np.int8)
        X = np.where(np.asarray(root_states)[None, :] == 1, 1 - X, X).astype(np.int8)   # non-root state
        L = ta.bl.sum()
        mus = np.geomspace(0.5, 64.0, n_mu) / max(L, 1e-12)            # expected switches over the tree
        rhos = np.linspace(0.05, 0.995, n_rho)
        pis = np.linspace(0.05, 0.95, n_pi)
        self.best, self.loglik = _fit_kernel(self.order_rev, ta.parent, ta.bl, tip_row, X, mus, rhos, pis)
        self.q0, self.q1, self.proot = _posterior_kernel(self.order_rev, ta.parent, ta.bl, tip_row, X, self.best)
        rng = np.random.default_rng(seed)
        self.open_frac = _open_fraction(ta.preorder, ta.parent, ta.bl, self.q0, self.q1, self.proot, 64,
                                        rng.integers(0, 2**31 - 1, size=X.shape[1]))
        self.ta = ta

    def scaled_params(self, trait_params):
        P = trait_params.copy()
        f = np.clip(self.open_frac, 1e-3, 1.0)
        P["gain_subsize"] = P["gain_subsize"].to_numpy(float) * f
        P["loss_subsize"] = P["loss_subsize"].to_numpy(float) * f
        return P

    def simulate(self, trait_params, idx, trials, seed):
        """Packed (n_tips, len(idx), chunks) replicates for traits idx (rows of trait_params)."""
        ta = self.ta
        P = trait_params.iloc[idx]
        g = P["gains"].to_numpy(float); l = P["losses"].to_numpy(float)
        gs = P["gain_subsize"].to_numpy(float); ls = P["loss_subsize"].to_numpy(float)
        gr = np.divide(g, gs, out=np.zeros_like(g), where=gs > 0)
        lr = np.divide(l, ls, out=np.zeros_like(l), where=ls > 0)
        root = P["root_state"].to_numpy() > 0
        n_chunks = (trials + 63) // 64
        last = trials - 64 * (n_chunks - 1)
        seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=len(idx))
        return _sim_hidden_kernel(ta.preorder, ta.parent, ta.bl, ta.leaf_idx, gr, lr, root, n_chunks, last,
                                  seeds, self.q0[idx], self.q1[idx], self.proot[idx])


# ---------------------------------------------------------------------------
# Covarion version (lineage='hidden'): the trait also evolves inside open regions
# ---------------------------------------------------------------------------
#
# Joint states s = 2*H + X' (X' = non-root state indicator):
#   0 = (closed, 0), 1 = (closed, 1), 2 = (open, 0), 3 = (open, 1)
# H switches 0->1 at mu*rho and 1->0 at mu*(1-rho) whatever X'; X' changes only when
# open, 0->1 at lam*pi and 1->0 at lam*(1-pi).  The chain is reversible with
# stationary distribution (H ~ rho) x (X' ~ pi), so P(t) comes from a symmetric
# eigen-decomposition.  lam = lam_ACR / rho (the trait evolves only while open).

@njit(cache=True)
def _cov_eig(mu, rho, lam, pi):
    Q = np.zeros((4, 4))
    Q[0, 2] = mu * rho; Q[1, 3] = mu * rho
    Q[2, 0] = mu * (1 - rho); Q[3, 1] = mu * (1 - rho)
    Q[2, 3] = lam * pi; Q[3, 2] = lam * (1 - pi)
    for i in range(4):
        Q[i, i] = -(Q[i, 0] + Q[i, 1] + Q[i, 2] + Q[i, 3] - Q[i, i])
    w = np.array([(1 - rho) * (1 - pi), (1 - rho) * pi, rho * (1 - pi), rho * pi])
    sq = np.sqrt(w)
    S = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            S[i, j] = sq[i] * Q[i, j] / sq[j]
    S = 0.5 * (S + S.T)
    ev, U = np.linalg.eigh(S)
    return ev, U, sq


@njit(inline="always", cache=True)
def _cov_P(ev, U, sq, t, P):
    for i in range(4):
        for j in range(4):
            acc = 0.0
            for k in range(4):
                acc += U[i, k] * math.exp(ev[k] * t) * U[j, k]
            P[i, j] = max(acc * sq[j] / sq[i], 0.0)


@njit(cache=True)
def _cov_prune(order_rev, parent, bl, tip_row, xcol, mu, rho, lam, pi, L, scale_out):
    n = parent.shape[0]
    ev, U, sq = _cov_eig(mu, rho, lam, pi)
    A = np.ones((n, 4))
    P = np.empty((4, 4))
    logs = 0.0
    for k in range(n):
        v = order_rev[k]
        r = tip_row[v]
        if r >= 0:
            x = xcol[r]
            for s in range(4):
                L[v, s] = 1.0 if (x < 0 or (s & 1) == x) else 0.0
        else:
            for s in range(4):
                L[v, s] = A[v, s]
        m = max(max(L[v, 0], L[v, 1]), max(L[v, 2], L[v, 3]))
        if m <= 0.0:
            return -1e300
        for s in range(4):
            L[v, s] /= m
        logs += math.log(m)
        p = parent[v]
        if p >= 0:
            _cov_P(ev, U, sq, bl[v], P)
            for s in range(4):
                A[p, s] *= P[s, 0] * L[v, 0] + P[s, 1] * L[v, 1] + P[s, 2] * L[v, 2] + P[s, 3] * L[v, 3]
    root = order_rev[n - 1]
    scale_out[0] = logs
    # root: X' = 0 (the trait's root state); H ~ rho
    return logs + math.log((1 - rho) * L[root, 0] + rho * L[root, 2] + 1e-300)


@njit(parallel=True, cache=True)
def _cov_fit_kernel(order_rev, parent, bl, tip_row, X, lam_acr, pis, mus, rhos, mults):
    """Grid ML over accessibility switching (mu), open fraction (rho) and the
    open-region trait rate lam = lam_ACR * mult, chosen independently of rho."""
    T = X.shape[1]; n = parent.shape[0]
    best = np.zeros((T, 4)); bestll = np.full(T, -1e300)
    for t in prange(T):
        L = np.empty((n, 4)); sc = np.zeros(1)
        for c in range(mults.shape[0]):
            lam = lam_acr[t] * mults[c]
            for a in range(mus.shape[0]):
                for b in range(rhos.shape[0]):
                    ll = _cov_prune(order_rev, parent, bl, tip_row, X[:, t], mus[a], rhos[b], lam, pis[t], L, sc)
                    if ll > bestll[t]:
                        bestll[t] = ll
                        best[t, 0] = mus[a]; best[t, 1] = rhos[b]; best[t, 2] = lam; best[t, 3] = pis[t]
    return best, bestll


@njit(parallel=True, cache=True)
def _cov_posterior_kernel(order_rev, parent, bl, tip_row, X, best):
    """Per node and parent joint state: P(child H=1), P(child X'=1 | H=0), P(child X'=1 | H=1);
    and the root posterior P(H=1)."""
    T = X.shape[1]; n = parent.shape[0]
    C = np.zeros((T, n, 4, 3), dtype=np.float32)
    proot = np.zeros(T)
    for t in prange(T):
        L = np.empty((n, 4)); sc = np.zeros(1)
        mu, rho, lam, pi = best[t, 0], best[t, 1], best[t, 2], best[t, 3]
        _cov_prune(order_rev, parent, bl, tip_row, X[:, t], mu, rho, lam, pi, L, sc)
        ev, U, sq = _cov_eig(mu, rho, lam, pi)
        P = np.empty((4, 4))
        root = order_rev[n - 1]
        a = rho * L[root, 2]; b = (1 - rho) * L[root, 0]
        proot[t] = a / (a + b) if a + b > 0 else rho
        for v in range(n):
            if parent[v] < 0:
                continue
            _cov_P(ev, U, sq, bl[v], P)
            for s in range(4):
                w0 = P[s, 0] * L[v, 0]; w1 = P[s, 1] * L[v, 1]
                w2 = P[s, 2] * L[v, 2]; w3 = P[s, 3] * L[v, 3]
                tot = w0 + w1 + w2 + w3
                C[t, v, s, 0] = (w2 + w3) / tot if tot > 0 else 0.0
                C[t, v, s, 1] = w1 / (w0 + w1) if (w0 + w1) > 0 else 0.0
                C[t, v, s, 2] = w3 / (w2 + w3) if (w2 + w3) > 0 else 0.0
    return C, proot


@njit(parallel=True, cache=True)
def _cov_sim_kernel(preorder, parent, bl, leaf_idx, g_open, l_open, root_state, n_chunks, last_bits,
                    seeds, C, proot):
    """Sample the accessibility map from the joint posterior (bit-parallel), then
    re-simulate the trait with its open-region rates on open branches only."""
    n = preorder.shape[0]
    T = g_open.shape[0]
    out = np.zeros((leaf_idx.shape[0], T, n_chunks), dtype=np.uint64)
    for t in prange(T):
        np.random.seed(seeds[t])
        H = np.zeros((n, n_chunks), dtype=np.uint64)
        Xp = np.zeros((n, n_chunks), dtype=np.uint64)       # posterior-sampled X' (only to condition H)
        st = np.zeros((n, n_chunks), dtype=np.uint64)       # fresh X'
        mu_ = g_open[t] + l_open[t]
        for oi in range(n):
            v = preorder[oi]
            for c in range(n_chunks):
                nb = 64 if c < n_chunks - 1 else last_bits
                full = np.uint64(0xFFFFFFFFFFFFFFFF) if nb == 64 else (np.uint64(1) << np.uint64(nb)) - np.uint64(1)
                p = parent[v]
                if p < 0:
                    H[v, c] = _bern_word(proot[t], nb)
                    Xp[v, c] = np.uint64(0); st[v, c] = np.uint64(0)
                    continue
                hp = H[p, c]; xp = Xp[p, c]
                hv = np.uint64(0); xv = np.uint64(0)
                for s in range(4):
                    ms = ((hp if s >= 2 else ~hp) & (xp if (s & 1) == 1 else ~xp)) & full
                    if ms == np.uint64(0):
                        continue
                    hb = _bern_word(C[t, v, s, 0], nb)
                    x0 = _bern_word(C[t, v, s, 1], nb)
                    x1 = _bern_word(C[t, v, s, 2], nb)
                    hv |= ms & hb
                    xv |= ms & ((hb & x1) | (~hb & x0))
                H[v, c] = hv & full; Xp[v, c] = xv & full
                s_ = st[p, c]
                both = hp & H[v, c]
                half = (hp ^ H[v, c]) & full
                if mu_ > 0.0 and (both | half) != np.uint64(0):
                    f1 = 1.0 - math.exp(-mu_ * bl[v])
                    fh = 1.0 - math.exp(-mu_ * bl[v] * 0.5)
                    g = (_bern_word(g_open[t] / mu_ * f1, nb) & both) | (_bern_word(g_open[t] / mu_ * fh, nb) & half)
                    l = (_bern_word(l_open[t] / mu_ * f1, nb) & both) | (_bern_word(l_open[t] / mu_ * fh, nb) & half)
                    st[v, c] = ((s_ | (g & ~s_)) & ~(l & s_)) & full
                else:
                    st[v, c] = s_
        for li in range(leaf_idx.shape[0]):
            for c in range(n_chunks):
                w = st[leaf_idx[li], c]
                out[li, t, c] = (~w if root_state[t] else w) & (np.uint64(0xFFFFFFFFFFFFFFFF) if (c < n_chunks - 1 or last_bits == 64) else ((np.uint64(1) << np.uint64(last_bits)) - np.uint64(1)))
    return out


class CovarionModel:
    """Hidden accessibility with trait evolution inside open regions (lineage='hidden')."""

    def __init__(self, ta, obs, trait_params, n_mu=6, rhos=None, mults=(0.5, 1, 2, 4, 8, 16), seed=None):
        n_nodes = len(ta.parent)
        tip_row = np.full(n_nodes, -1, np.int64)
        tip_row[ta.leaf_idx] = np.arange(len(ta.leaf_idx))
        self.order_rev = ta.preorder[::-1].copy()
        root = trait_params["root_state"].to_numpy().astype(int)
        X = (np.asarray(obs) > 0.5).astype(np.int8)
        X = np.where(root[None, :] == 1, 1 - X, X).astype(np.int8)
        g = trait_params["gains"].to_numpy(float); l = trait_params["losses"].to_numpy(float)
        gs = trait_params["gain_subsize"].to_numpy(float); ls = trait_params["loss_subsize"].to_numpy(float)
        gr = np.divide(g, gs, out=np.zeros_like(g), where=gs > 0)
        lr = np.divide(l, ls, out=np.zeros_like(l), where=ls > 0)
        gp = np.where(root == 1, lr, gr); lp = np.where(root == 1, gr, lr)   # rates of X' (non-root state)
        lam = np.maximum(gp + lp, 1e-9)
        pi = np.clip(np.divide(gp, lam), 1e-4, 1 - 1e-4)
        Ltot = ta.bl.sum()
        mus = np.geomspace(0.5, 16.0, n_mu) / max(Ltot, 1e-12)
        rhos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]) if rhos is None else np.asarray(rhos)
        self.best, self.loglik = _cov_fit_kernel(self.order_rev, ta.parent, ta.bl, tip_row, X, lam, pi, mus, rhos,
                                                 np.asarray(mults, float))
        self.C, self.proot = _cov_posterior_kernel(self.order_rev, ta.parent, ta.bl, tip_row, X, self.best)
        self.root = root
        self.ta = ta
        self.rho = self.best[:, 1]

    def simulate(self, idx, trials, seed):
        ta = self.ta
        idx = np.asarray(idx)
        lam, pi = self.best[idx, 2], self.best[idx, 3]
        n_chunks = (trials + 63) // 64
        last = trials - 64 * (n_chunks - 1)
        seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=len(idx))
        return _cov_sim_kernel(ta.preorder, ta.parent, ta.bl, ta.leaf_idx, lam * pi, lam * (1 - pi),
                               self.root[idx].astype(np.bool_), n_chunks, last, seeds,
                               np.ascontiguousarray(self.C[idx]), self.proot[idx])


# ---------------------------------------------------------------------------
# Bayesian covarion: integrate over the parameter grid instead of a point fit
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _cov_ll_kernel(order_rev, parent, bl, tip_row, X, lam_acr, pis, grid):
    """Log-likelihood of every trait at every grid point (mu, rho, mult)."""
    T = X.shape[1]; n = parent.shape[0]; G = grid.shape[0]
    LL = np.empty((T, G))
    for t in prange(T):
        L = np.empty((n, 4)); sc = np.zeros(1)
        for g in range(G):
            LL[t, g] = _cov_prune(order_rev, parent, bl, tip_row, X[:, t], grid[g, 0], grid[g, 1],
                                  lam_acr[t] * grid[g, 2], pis[t], L, sc)
    return LL


@njit(parallel=True, cache=True)
def _cov_bayes_sim_kernel(preorder, order_rev, parent, bl, tip_row, leaf_idx, X, lam_acr, pis, root_state,
                          grid, cumw, n_chunks, last_bits, seeds):
    """For every 64-replicate chunk: draw a grid point from the trait's posterior,
    compute its pruning partials, sample the accessibility map from the joint
    posterior and re-simulate the trait on the open branches."""
    n = preorder.shape[0]; T = X.shape[1]; G = grid.shape[0]
    out = np.zeros((leaf_idx.shape[0], T, n_chunks), dtype=np.uint64)
    for t in prange(T):
        np.random.seed(seeds[t])
        L = np.empty((n, 4)); sc = np.zeros(1); P = np.empty((4, 4))
        H = np.zeros(n, dtype=np.uint64); Xp = np.zeros(n, dtype=np.uint64); st = np.zeros(n, dtype=np.uint64)
        for c in range(n_chunks):
            nb = 64 if c < n_chunks - 1 else last_bits
            full = np.uint64(0xFFFFFFFFFFFFFFFF) if nb == 64 else (np.uint64(1) << np.uint64(nb)) - np.uint64(1)
            u = np.random.random()
            g = 0
            while g < G - 1 and cumw[t, g] < u:
                g += 1
            mu, rho, lam, pi = grid[g, 0], grid[g, 1], lam_acr[t] * grid[g, 2], pis[t]
            _cov_prune(order_rev, parent, bl, tip_row, X[:, t], mu, rho, lam, pi, L, sc)
            ev, U, sq = _cov_eig(mu, rho, lam, pi)
            gopen = lam * pi; lopen = lam * (1 - pi); mu_ = gopen + lopen
            for oi in range(n):
                v = preorder[oi]
                p = parent[v]
                if p < 0:
                    a = rho * L[v, 2]; b = (1 - rho) * L[v, 0]
                    H[v] = _bern_word(a / (a + b) if a + b > 0 else rho, nb)
                    Xp[v] = np.uint64(0); st[v] = np.uint64(0)
                    continue
                _cov_P(ev, U, sq, bl[v], P)
                hp = H[p]; xp = Xp[p]
                hv = np.uint64(0); xv = np.uint64(0)
                for s in range(4):
                    ms = ((hp if s >= 2 else ~hp) & (xp if (s & 1) == 1 else ~xp)) & full
                    if ms == np.uint64(0):
                        continue
                    w0 = P[s, 0] * L[v, 0]; w1 = P[s, 1] * L[v, 1]; w2 = P[s, 2] * L[v, 2]; w3 = P[s, 3] * L[v, 3]
                    tot = w0 + w1 + w2 + w3
                    hb = _bern_word((w2 + w3) / tot if tot > 0 else 0.0, nb)
                    x0 = _bern_word(w1 / (w0 + w1) if (w0 + w1) > 0 else 0.0, nb)
                    x1 = _bern_word(w3 / (w2 + w3) if (w2 + w3) > 0 else 0.0, nb)
                    hv |= ms & hb
                    xv |= ms & ((hb & x1) | (~hb & x0))
                H[v] = hv & full; Xp[v] = xv & full
                s_ = st[p]
                both = hp & H[v]
                half = (hp ^ H[v]) & full
                if mu_ > 0.0 and (both | half) != np.uint64(0):
                    f1 = 1.0 - math.exp(-mu_ * bl[v]); fh = 1.0 - math.exp(-mu_ * bl[v] * 0.5)
                    gg = (_bern_word(gopen / mu_ * f1, nb) & both) | (_bern_word(gopen / mu_ * fh, nb) & half)
                    ll_ = (_bern_word(lopen / mu_ * f1, nb) & both) | (_bern_word(lopen / mu_ * fh, nb) & half)
                    st[v] = ((s_ | (gg & ~s_)) & ~(ll_ & s_)) & full
                else:
                    st[v] = s_
            for li in range(leaf_idx.shape[0]):
                w = st[leaf_idx[li]]
                out[li, t, c] = ((~w) if root_state[t] else w) & full
    return out


class BayesCovarionModel:
    """Hidden accessibility, integrated over (mu, rho, rate multiplier) by its posterior
    on a grid.  Prior: P(unrestricted, rho = 0.999) = p_unrestricted, the rest spread
    evenly over the restricted rho values; mu and the multiplier uniform."""

    def __init__(self, ta, obs, trait_params, n_mu=6, rhos=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                 mults=(0.5, 1, 2, 4, 8, 16), p_unrestricted=0.5):
        n_nodes = len(ta.parent)
        self.tip_row = np.full(n_nodes, -1, np.int64)
        self.tip_row[ta.leaf_idx] = np.arange(len(ta.leaf_idx))
        self.order_rev = ta.preorder[::-1].copy()
        root = trait_params["root_state"].to_numpy().astype(int)
        X = (np.asarray(obs) > 0.5).astype(np.int8)
        self.X = np.ascontiguousarray(np.where(root[None, :] == 1, 1 - X, X).astype(np.int8))
        g = trait_params["gains"].to_numpy(float); l = trait_params["losses"].to_numpy(float)
        gs = trait_params["gain_subsize"].to_numpy(float); ls = trait_params["loss_subsize"].to_numpy(float)
        gr = np.divide(g, gs, out=np.zeros_like(g), where=gs > 0)
        lr = np.divide(l, ls, out=np.zeros_like(l), where=ls > 0)
        gp = np.where(root == 1, lr, gr); lp = np.where(root == 1, gr, lr)
        self.lam = np.maximum(gp + lp, 1e-9)
        self.pi = np.clip(gp / self.lam, 1e-4, 1 - 1e-4)
        Ltot = ta.bl.sum()
        mus = np.geomspace(0.5, 16.0, n_mu) / max(Ltot, 1e-12)
        rho_all = np.concatenate([np.asarray(rhos, float), [0.999]])
        grid, prior = [], []
        for m in mults:
            for mu in mus:
                for r in rho_all:
                    grid.append((mu, r, m))
                    prior.append(p_unrestricted if r > 0.99 else (1 - p_unrestricted) / len(rhos))
        self.grid = np.array(grid)
        lp_ = np.log(np.array(prior) / (len(mults) * len(mus)))
        LL = _cov_ll_kernel(self.order_rev, ta.parent, ta.bl, self.tip_row, self.X, self.lam, self.pi, self.grid)
        post = LL + lp_[None, :]
        post -= post.max(1, keepdims=True)
        W = np.exp(post); W /= W.sum(1, keepdims=True)
        self.W = W
        self.p_restricted = W[:, self.grid[:, 1] < 0.99].sum(1)
        self.cumw = np.ascontiguousarray(np.cumsum(W, 1))
        self.root = root.astype(np.bool_)
        self.ta = ta

    def simulate(self, idx, trials, seed):
        ta = self.ta; idx = np.asarray(idx)
        n_chunks = (trials + 63) // 64
        last = trials - 64 * (n_chunks - 1)
        seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=len(idx))
        return _cov_bayes_sim_kernel(ta.preorder, self.order_rev, ta.parent, ta.bl, self.tip_row, ta.leaf_idx,
                                     np.ascontiguousarray(self.X[:, idx]), self.lam[idx], self.pi[idx],
                                     self.root[idx], self.grid, self.cumw[idx], n_chunks, last, seeds)
