"""Pairwise-complete NA handling in the anchored test."""
import numpy as np

from simphyni.Simulation.fastsim import anchored_entries


def test_masked_entries_equal_scoring_without_missing_tips():
    rng = np.random.default_rng(1)
    n, T, N = 80, 6, 512
    U = (rng.random((n, T, N)) < 0.3).astype(np.uint8)
    obs = (rng.random((n, T)) < 0.3).astype(np.float32)
    w = rng.random(n).astype(np.float32)
    obs_na = obs.copy()
    obs_na[rng.random((n, T)) < 0.15] = np.nan
    a = np.array([0, 1, 2, 3]); b = np.array([1, 2, 3, 4])
    st, p, mu, sd = anchored_entries(U, obs_na, w, a, b)
    for k in range(len(a)):
        keep = ~np.isnan(obs_na[:, a[k]]) & ~np.isnan(obs_na[:, b[k]])
        st2, p2, mu2, sd2 = anchored_entries(U[keep], obs[keep], w[keep], a[k:k + 1], b[k:k + 1])
        assert abs(st[k] - st2[0]) < 1e-5
        assert abs(p[k] - p2[0]) < 1e-9
        assert abs(mu[k] - mu2[0]) < 1e-5


def test_no_missing_values_unchanged():
    rng = np.random.default_rng(2)
    n, T, N = 60, 5, 512
    U = (rng.random((n, T, N)) < 0.4).astype(np.uint8)
    obs = (rng.random((n, T)) < 0.4).astype(np.float32)
    w = rng.random(n).astype(np.float32)
    a = np.array([0, 1, 2]); b = np.array([3, 4, 0])
    r1 = anchored_entries(U, obs, w, a, b)
    r2 = anchored_entries(U, obs.astype(np.float64), w, a, b)
    for x, y in zip(r1, r2):
        assert np.allclose(x, y)
