"""
Empirical-Bayes shrinkage of pair effect sizes (adaptive shrinkage, Stephens 2017).

Each pair has an effect estimate b (observed weighted log odds ratio minus the
null mean) and a standard error s (the null standard deviation).  The prior on
the true effect is a scale mixture  g = pi_0 delta_0 + sum_k pi_k N(0, sigma_k^2)
on a fixed grid of sigma_k; the weights are fitted by EM on all pairs (or a
random subset).  With b | beta ~ N(beta, s^2) the posterior is a mixture of
normals in closed form, giving the posterior mean and the local false sign
rate  lfsr = min(P(beta >= 0 | b), P(beta <= 0 | b)).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def _grid(b, s):
    smin = max(np.nanmin(s[s > 0]) / 10.0, 1e-3) if np.any(s > 0) else 1e-3
    smax = 2.0 * np.sqrt(max(np.nanmax(b ** 2 - s ** 2), smin ** 2))
    k = int(np.ceil(np.log(smax / smin) / np.log(np.sqrt(2)))) + 1
    return np.concatenate([[0.0], smin * np.sqrt(2) ** np.arange(k)])


def fit_ash(b, s, max_fit=200_000, n_iter=200, null_weight=10.0, seed=0):
    """EM for the mixture weights.  null_weight is a Dirichlet prior count on the
    point mass (as in ashr's default nullbiased penalty)."""
    b = np.asarray(b, float); s = np.maximum(np.asarray(s, float), 1e-6)
    ok = np.isfinite(b) & np.isfinite(s)
    b, s = b[ok], s[ok]
    if len(b) > max_fit:
        idx = np.random.default_rng(seed).choice(len(b), max_fit, replace=False)
        b, s = b[idx], s[idx]
    sig = _grid(b, s)
    tot = np.sqrt(sig[None, :] ** 2 + s[:, None] ** 2)
    L = np.exp(-0.5 * (b[:, None] / tot) ** 2) / tot                  # n x K (constant dropped)
    pi = np.full(len(sig), 1.0 / len(sig))
    prior = np.ones(len(sig)); prior[0] = null_weight
    for _ in range(n_iter):
        R = L * pi[None, :]
        R /= np.maximum(R.sum(1, keepdims=True), 1e-300)
        new = (R.sum(0) + prior - 1.0)
        new = np.maximum(new, 0) / max(new.sum(), 1e-300)
        if np.max(np.abs(new - pi)) < 1e-8:
            pi = new
            break
        pi = new
    return sig, pi


def posterior(b, s, sig, pi):
    """Posterior mean, posterior sd and local false sign rate for every pair."""
    b = np.asarray(b, float); s = np.maximum(np.asarray(s, float), 1e-6)
    out_mean = np.full(len(b), np.nan); out_sd = np.full(len(b), np.nan); lfsr = np.full(len(b), np.nan)
    ok = np.isfinite(b) & np.isfinite(s)
    for c0 in range(0, len(b), 500_000):
        sl = np.flatnonzero(ok)[c0:c0 + 500_000]
        bb, ss = b[sl][:, None], s[sl][:, None]
        tot2 = sig[None, :] ** 2 + ss ** 2
        lik = np.exp(-0.5 * bb ** 2 / tot2) / np.sqrt(tot2)
        w = lik * pi[None, :]
        w /= np.maximum(w.sum(1, keepdims=True), 1e-300)
        # component posteriors: N(m_k, v_k) with m_k = b sigma_k^2/(sigma_k^2+s^2)
        frac = sig[None, :] ** 2 / tot2
        m = frac * bb
        v = frac * ss ** 2
        mean = (w * m).sum(1)
        second = (w * (v + m ** 2)).sum(1)
        sdv = np.sqrt(np.maximum(second - mean ** 2, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            pneg = np.where(v > 0, ndtr(-m / np.sqrt(np.where(v > 0, v, 1))), 0.0)
        p_lt0 = (w * pneg).sum(1)
        p_zero = w[:, 0]
        p_gt0 = 1.0 - p_lt0 - p_zero
        out_mean[sl], out_sd[sl] = mean, sdv
        lfsr[sl] = np.minimum(p_lt0 + p_zero, p_gt0 + p_zero)
    return out_mean, out_sd, lfsr


def shrink(b, s, **kw):
    sig, pi = fit_ash(b, s, **kw)
    m, sd, lfsr = posterior(b, s, sig, pi)
    return m, sd, lfsr, (sig, pi)
