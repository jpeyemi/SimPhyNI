"""
Calibration regression tests for the statistics layer.

Tips are i.i.d. here (no phylogeny), so independent simulated traits form an
exact null: p-values must be ~Uniform(0, 1).
"""
import numpy as np
import pandas as pd
import pytest

from simphyni.Simulation.simulation import compres
from simphyni.Simulation.pair_statistics import pair_statistics


def _pack(bits: np.ndarray) -> np.ndarray:
    """bits (n_tips, n_traits, 64) {0,1} -> (n_tips, n_traits, 1) uint64."""
    return np.packbits(bits.astype(np.uint8), axis=-1, bitorder="little").view(np.uint64)


def _null_pvalues(n_pairs=400, n_tips=200, prev=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n_traits = 2 * n_pairs
    sim = _pack(rng.random((n_tips, n_traits, 64)) < prev)
    obs = (rng.random((n_tips, n_traits)) < prev).astype(float)
    pairs = np.arange(n_traits).reshape(n_pairs, 2)
    obsstats = pair_statistics._log_odds_ratio_statistic(obs[:, pairs[:, 0]], obs[:, pairs[:, 1]])
    res = compres(sim, pairs, obsstats, bits=64, cores=1)
    return res["p-value"].to_numpy()


def test_pvalues_are_two_sided_and_calibrated():
    p = _null_pvalues()
    assert p.max() > 0.9, "two-sided p-values must range up to 1"
    assert p.min() >= 0 and p.max() <= 1
    # nominal 0.05: binomial sd ~0.011 at n=400; allow generous MC/KDE slack
    assert np.mean(p < 0.05) < 0.085
    assert np.mean(p < 0.20) < 0.27
