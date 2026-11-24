"""
Prototype null-model options that carry parameter uncertainty without
oversampling (explore/uncertainty).

rate_model='eb'      : F81 reparameterisation (pi, mu) with an empirical-Bayes
                       Gamma prior on mu fitted across traits; optional
                       per-replicate posterior draws.
root='sample'        : per-replicate root state drawn from root_prob.
simulator='fixed_k'  : condition on the reconstructed number of changes k;
                       k toggles placed on branches in proportion to (capped)
                       length, so the rate drops out.
statistic='weighted' : log odds ratio on GSC phylogenetic tip weights.
pvalue='normal'      : normal tail instead of the KDE tail.
pvalue='analytic'    : no simulation; z-score from the stationary F81 tip
                       covariance, Var(S) = tr(H Cx H . H Cy H).
testable_filter=a    : Tarone-style filtering; pairs whose minimum achievable
                       p (given their margins) exceeds a are left out of the
                       multiple-testing denominator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, ndtr


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def ancestor_matrix(tree, branch_cap=None):
    """Tips (leaf order) x nodes (traverse order) path matrix and capped branch lengths."""
    nodes = list(tree.traverse())
    node_idx = {n: i for i, n in enumerate(nodes)}
    leaves = list(tree)
    A = np.zeros((len(leaves), len(nodes)), dtype=np.uint8)
    for t, leaf in enumerate(leaves):
        n = leaf
        while n.up is not None:
            A[t, node_idx[n]] = 1
            n = n.up
    w = np.array([0.0 if n.up is None else
                  (n.dist if branch_cap is None else min(n.dist, branch_cap)) for n in nodes])
    return A, w


def tip_distances(tree, branch_cap=None):
    A, w = ancestor_matrix(tree, branch_cap)
    Af = A.astype(float)
    depth = Af @ w
    shared = (Af * w) @ Af.T
    return depth[:, None] + depth[None, :] - 2 * shared


def gsc_weights(tree, branch_cap=None):
    """Gerstein-Sonnhammer-Chothia tip weights on capped lengths, scaled to sum to n_tips."""
    leaves = list(tree)
    pos = {l: i for i, l in enumerate(leaves)}
    wts = np.zeros(len(leaves))
    below = {}
    for node in tree.traverse("postorder"):
        bl = 0.0 if node.up is None else (node.dist if branch_cap is None else min(node.dist, branch_cap))
        if node.is_leaf():
            idx = np.array([pos[node]])
            wts[idx] += bl
        else:
            idx = np.concatenate([below.pop(c) for c in node.children])
            cur = wts[idx]
            tot = cur.sum()
            wts[idx] += bl * (cur / tot if tot > 0 else np.full(len(idx), 1.0 / len(idx)))
        below[node] = idx
    if wts.sum() <= 0:
        return np.ones(len(leaves))
    return wts * len(leaves) / wts.sum()


# ---------------------------------------------------------------------------
# Rate model: (pi, mu) with empirical-Bayes mu
# ---------------------------------------------------------------------------

def _fit_gamma_prior(k, E):
    """ML fit of mu ~ Gamma(a, rate b) from k ~ Poisson(mu E) (negative-binomial marginal)."""
    ok = (E > 0) & np.isfinite(k)
    k, E = k[ok], E[ok]
    mu = k / E
    m, v = mu.mean(), mu.var() + 1e-12
    x0 = np.log([max(m * m / v, 0.1), max(m / v, 1e-6)])

    def nll(x):
        a, b = np.exp(x)
        return -np.sum(gammaln(a + k) - gammaln(a) + a * np.log(b / (b + E)) + k * np.log(E / (b + E)))

    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8})
    return tuple(np.exp(r.x))


def eb_rates(trait_params, obs_prevalence, trials, pi_source="acr", draws=True,
             prior_by_prevalence=False, rng=None):
    """Per-trait (or per-trait, per-replicate) gain and loss rates under F81.

    pi       : 'acr' -> r_g / (r_g + r_l) from the point estimates (falls back to
               prevalence when both are 0); 'prevalence' -> smoothed tip prevalence.
    mu       : posterior Gamma(a + k, b + E), k = gains + losses,
               E = pi * gain_subsize + (1 - pi) * loss_subsize (expected F81
               exposure), with (a, b) fitted across traits.  With
               prior_by_prevalence the prior is fitted separately within
               prevalence tertiles.
    Returns (gain_rates, loss_rates) of shape (n_traits, trials) and a dict of
    diagnostics.
    """
    rng = rng if rng is not None else np.random.default_rng()
    g = np.asarray(trait_params["gains"], float)
    l = np.asarray(trait_params["losses"], float)
    T0 = np.asarray(trait_params["gain_subsize"], float)
    T1 = np.asarray(trait_params["loss_subsize"], float)
    p = np.clip(np.asarray(obs_prevalence, float), 0, 1)
    p_s = np.where(np.isfinite(p), p, 0.5)
    n = len(g)
    rg = np.divide(g, T0, out=np.zeros(n), where=T0 > 0)
    rl = np.divide(l, T1, out=np.zeros(n), where=T1 > 0)
    if pi_source == "prevalence":
        pi = np.clip(p_s, 0.01, 0.99)
    else:
        tot = rg + rl
        pi = np.where(tot > 0, np.divide(rg, tot, out=np.zeros(n), where=tot > 0), p_s)
        pi = np.clip(pi, 0.01, 0.99)
    E = pi * T0 + (1 - pi) * T1
    k = g + l

    a = np.empty(n); b = np.empty(n)
    groups = [np.arange(n)]
    if prior_by_prevalence and n >= 30:
        q = np.quantile(p_s, [1 / 3, 2 / 3])
        lab = np.digitize(p_s, q)
        groups = [np.where(lab == j)[0] for j in range(3)]
    for gi in groups:
        if len(gi) == 0:
            continue
        ai, bi = _fit_gamma_prior(k[gi], E[gi])
        a[gi], b[gi] = ai, bi
    shape, rate = a + k, b + np.maximum(E, 0)
    if draws:
        mu = rng.gamma(shape[:, None], 1.0 / rate[:, None], size=(n, trials))
    else:
        mu = np.repeat((shape / rate)[:, None], trials, axis=1)
    gr = pi[:, None] * mu
    lr = (1 - pi)[:, None] * mu
    gr[T0 <= 0] = 0
    lr[T1 <= 0] = 0
    info = {"prior_a": a, "prior_b": b, "pi": pi, "mu_mean": shape / rate, "mu_point": rg + rl}
    return gr, lr, info


# ---------------------------------------------------------------------------
# Fixed-k simulator
# ---------------------------------------------------------------------------

def sim_fixed_k(tree, k, root_states, trials, branch_cap=None, rng=None, root_bits=None):
    """Condition on the number of changes: place k toggles on branches with
    probability proportional to capped branch length (a Poisson process
    conditioned on its count), tip state = root XOR parity of toggles above it.

    k          : (n_traits,) int number of changes
    root_bits  : optional (n_traits, trials) 0/1 per-replicate root states
    Returns (n_tips, n_traits, ceil(trials/64)) uint64 in tree leaf order.
    """
    rng = rng if rng is not None else np.random.default_rng()
    A, w = ancestor_matrix(tree, branch_cap)
    Af = A.astype(np.float32)
    prob = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    n_tips = A.shape[0]
    n_traits = len(k)
    n_chunks = (trials + 63) // 64
    out = np.zeros((n_tips, n_traits, n_chunks), dtype=np.uint64)
    for j in range(n_traits):
        kj = int(k[j])
        if root_bits is not None:
            rb = np.asarray(root_bits[j], dtype=np.uint8)
        else:
            rb = np.full(trials, int(root_states[j] > 0), dtype=np.uint8)
        if kj > 0:
            cnt = rng.multinomial(kj, prob, size=trials)                    # trials x nodes
            par = (cnt & 1).astype(np.float32)
            tips = (Af @ par.T).astype(np.int64) & 1                         # tips x trials
            tips = tips.astype(np.uint8) ^ rb[None, :]
        else:
            tips = np.repeat(rb[None, :], n_tips, axis=0)
        pad = n_chunks * 64 - trials
        if pad:
            tips = np.pad(tips, ((0, 0), (0, pad)))
        out[:, j, :] = np.packbits(tips, axis=-1, bitorder="little").view(np.uint64)
    return out


# ---------------------------------------------------------------------------
# Dense pair scoring (weighted / alternative tails / minimum achievable p)
# ---------------------------------------------------------------------------

def _unpack(lineages, trials):
    n_tips, n_traits, _ = lineages.shape
    bits = np.unpackbits(lineages.view(np.uint8), axis=-1, bitorder="little")
    return bits.reshape(n_tips, n_traits, -1)[:, :, :trials].astype(np.float64)


def _lor(a, s1, s2, W, eps=1.0):
    b = s1 - a
    c = s2 - a
    d = W - s1 - s2 + a
    return np.log(((a + eps) * (d + eps)) / ((b + eps) * (c + eps)))


def _tails(null, x, mode):
    """Left/right tail probabilities of x under the null sample."""
    if mode == "normal":
        mu, sd = null.mean(), max(null.std(ddof=1), 1e-9)
        z = (x - mu) / sd
        return ndtr(z), ndtr(-z)
    # Gaussian KDE, Silverman bandwidth (matches scipy.stats.gaussian_kde)
    nn = len(null)
    h = (nn * 3.0 / 4.0) ** (-1.0 / 5.0) * null.std(ddof=1)
    h = max(h, 1e-6)
    x = np.atleast_1d(x)
    zl = (x[:, None] - null[None, :]) / h
    return ndtr(zl).mean(axis=1), ndtr(-zl).mean(axis=1)


def score_pairs_dense(lineages, pairs_index, obs_traits, trials, weights=None,
                      pvalue="kde", want_pmin=False):
    """Score every pair against all trials x trials pairings of its simulated traits.

    Equivalent to compres' circular-shift scoring (every replicate of trait 1
    against every replicate of trait 2) but written as a matrix product, so tip
    weights and other tails are easy.  obs_traits: (n_tips, n_traits) 0/1 in
    tree leaf order, columns aligned with the simulated traits.
    """
    U = _unpack(lineages, trials)
    n_tips = U.shape[0]
    w = np.ones(n_tips) if weights is None else np.asarray(weights, float)
    W = w.sum()
    obs = np.nan_to_num(np.asarray(obs_traits, float)) > 0.5
    out = {"p-value": [], "direction": [], "effect size": [], "obs_stat": [], "p_min": []}
    Uw = U * w[:, None, None]
    sums = Uw.sum(axis=0)                                                   # traits x trials
    for i, j in pairs_index:
        X, Y = U[:, i, :], U[:, j, :]
        a = X.T @ (w[:, None] * Y)                                          # trials x trials
        null = _lor(a, sums[i][:, None], sums[j][None, :], W).ravel()
        xo, yo = obs[:, i].astype(float), obs[:, j].astype(float)
        stat = _lor((w * xo * yo).sum(), (w * xo).sum(), (w * yo).sum(), W)
        pl, pr = _tails(null, np.array([stat]), pvalue)
        pl, pr = float(pl[0]), float(pr[0])
        out["p-value"].append(min(1.0, 2 * min(pl, pr)))
        out["direction"].append(-1 if pl < pr else 1)
        q75, q25 = np.percentile(null, [75, 25])
        out["effect size"].append((stat - np.median(null)) / max((q75 - q25) * 1.349, 1))
        out["obs_stat"].append(stat)
        if want_pmin:
            if weights is None:
                n1, n2 = xo.sum(), yo.sum()
                a_lo, a_hi = max(0.0, n1 + n2 - n_tips), min(n1, n2)
                ext = np.array([_lor(a_lo, n1, n2, W), _lor(a_hi, n1, n2, W)])
            else:
                # weighted bounds: place co-occurrence on the lightest / heaviest tips
                sw = np.sort(w)
                n1, n2 = int(xo.sum()), int(yo.sum())
                s1, s2 = (w * xo).sum(), (w * yo).sum()
                lo_n, hi_n = max(0, n1 + n2 - n_tips), min(n1, n2)
                ext = np.array([_lor(sw[:lo_n].sum(), s1, s2, W),
                                _lor(min(sw[::-1][:hi_n].sum(), s1, s2), s1, s2, W)])
            el, er = _tails(null, ext, pvalue)
            out["p_min"].append(float(min(1.0, 2 * min(el.min(), er.min()))))
        else:
            out["p_min"].append(np.nan)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Analytic null
# ---------------------------------------------------------------------------

def analytic_pairs(tree, mu, pairs_index, obs_traits, branch_cap=None):
    """Stationary-F81 z-test.  Tip correlation R_ij = exp(-mu d_ij); with the
    sample-centring matrix H, S = x~' y~ has null variance
    s_x^2 s_y^2 sum(H R_x H * H R_y H) (Hadamard sum), s^2 = observed p(1-p)."""
    D = tip_distances(tree, branch_cap)
    n = D.shape[0]
    H = np.eye(n) - 1.0 / n
    obs = np.nan_to_num(np.asarray(obs_traits, float)) > 0.5
    obs = obs.astype(float)
    need = np.unique(np.asarray(pairs_index).ravel())
    Rc = {}
    for t in need:
        R = np.exp(-mu[t] * D)
        Rc[t] = H @ R @ H
    p_ = obs.mean(axis=0)
    xc = obs - p_[None, :]
    out = {"p-value": [], "direction": [], "effect size": [], "obs_stat": [], "p_min": []}
    for i, j in pairs_index:
        S = float(xc[:, i] @ xc[:, j])
        var = p_[i] * (1 - p_[i]) * p_[j] * (1 - p_[j]) * float(np.sum(Rc[i] * Rc[j]))
        z = S / np.sqrt(var) if var > 0 else 0.0
        out["p-value"].append(float(min(1.0, 2 * ndtr(-abs(z)))))
        out["direction"].append(1 if z > 0 else -1)
        out["effect size"].append(z)
        out["obs_stat"].append(S)
        out["p_min"].append(np.nan)
    return pd.DataFrame(out)
