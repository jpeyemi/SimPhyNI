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


@pytest.mark.parametrize("vars_,targets,expected", [
    (list("abcd"), list("abcd"), 6),          # all-vs-all: n(n-1)/2
    (list("ab"), list("cde"), 6),             # one-vs-rest, disjoint
    (list("abc"), list("bcde"), 3 * 4 - 2 - 1),  # overlap {b, c}: {b,c} counted once
])
def test_count_unique_pairs(vars_, targets, expected):
    from simphyni import TreeSimulator
    assert TreeSimulator.count_unique_pairs(vars_, targets) == expected
    brute = {frozenset((a, b)) for a in vars_ for b in targets if a != b}
    assert len(brute) == expected


def test_total_tests_matches_enumerated_pairs_all_vs_all():
    from ete3 import Tree
    from simphyni import TreeSimulator
    rng = np.random.default_rng(0)
    t = Tree()
    t.populate(30, random_branches=True)
    leaves = t.get_leaf_names()
    obs = pd.DataFrame((rng.random((30, 6)) < 0.4).astype(float), index=leaves,
                       columns=[f"g{i}" for i in range(6)])
    params = pd.DataFrame({"gene": obs.columns, "gains": 1.0, "losses": 1.0, "dist": 0.0,
                           "loss_dist": 0.0, "gain_subsize": 1.0, "loss_subsize": 1.0,
                           "root_state": 0})
    sim = TreeSimulator(t.write(format=1), params, obs)
    sim.initialize_simulation_parameters(pre_filter=False)
    assert sim.total_tests == len(sim.pairs) == 15


def test_flag_prevalence_gate_uses_number_of_tips():
    from simphyni.Simulation.simulation import flag_uncalibratable_traits
    n = 200
    params = pd.DataFrame({
        "gene": [f"g{i}" for i in range(n)],
        "gains": 1.0, "losses": 0.0,
        "gain_subsize": np.r_[np.full(n - 1, 1.0), 1e4],   # one extreme asymmetry
        "loss_subsize": 1.0,
        "_raw_gains": 1.0, "_raw_losses": 0.0,
        "_raw_count": np.r_[np.full(n - 1, 45.0), 40.0],   # 40/100 tips = 40 % prevalence
    })
    # With 100 tips the extreme trait is at 40 % prevalence -> eligible -> flagged
    flagged = flag_uncalibratable_traits(params, n_tips=100)
    assert f"g{n-1}" in set(flagged["gene"])
    # The old max-count proxy (40/45 = 89 %) wrongly exempts it
    assert f"g{n-1}" not in set(flag_uncalibratable_traits(params)["gene"])


def test_flagged_pairs_are_kept_and_marked_by_default(monkeypatch):
    from ete3 import Tree
    import simphyni.Simulation.tree_simulator as ts
    from simphyni import TreeSimulator
    rng = np.random.default_rng(1)
    t = Tree()
    t.populate(40, random_branches=True)
    obs = pd.DataFrame((rng.random((40, 4)) < 0.4).astype(float), index=t.get_leaf_names(),
                       columns=list("abcd"))
    params = pd.DataFrame({"gene": list("abcd"), "gains": 2.0, "losses": 2.0, "dist": 0.0,
                           "loss_dist": 0.0, "gain_subsize": 1.0, "loss_subsize": 1.0,
                           "root_state": 0})
    monkeypatch.setattr(ts, "flag_uncalibratable_traits",
                        lambda p, **kw: pd.DataFrame({"gene": ["a"]}))
    sim = TreeSimulator(t.write(format=1), params, obs)
    sim.initialize_simulation_parameters(pre_filter=False)
    sim.run_simulation(cores=1)
    res = sim.get_results()
    assert len(res) == 6
    involves_a = (res["T1"] == "a") | (res["T2"] == "a")
    assert (~res.loc[involves_a, "null_calibrated"]).all()
    assert res.loc[~involves_a, "null_calibrated"].all()


def _small_sim(seed, cores=2):
    from ete3 import Tree
    from simphyni import TreeSimulator
    import random
    random.seed(7)                      # ete3.populate uses the stdlib RNG
    rng = np.random.default_rng(7)
    t = Tree()
    t.populate(50, random_branches=True)
    obs = pd.DataFrame((rng.random((50, 5)) < 0.4).astype(float), index=t.get_leaf_names(),
                       columns=list("abcde"))
    params = pd.DataFrame({"gene": list("abcde"), "gains": 3.0, "losses": 3.0, "dist": 0.0,
                           "loss_dist": 0.0, "gain_subsize": 1.0, "loss_subsize": 1.0,
                           "root_state": 0})
    sim = TreeSimulator(t.write(format=1), params, obs)
    sim.initialize_simulation_parameters(pre_filter=False)
    sim.run_simulation(cores=cores, seed=seed)
    return sim.get_results().sort_values(["T1", "T2"]).reset_index(drop=True)


def test_seed_makes_results_reproducible():
    a, b, c = _small_sim(11), _small_sim(11, cores=1), _small_sim(12)
    pd.testing.assert_frame_equal(a, b)          # independent of worker count
    assert not np.allclose(a["pval_naive"], c["pval_naive"])


def _deep_threshold_case():
    import random
    from ete3 import Tree
    random.seed(3)
    t = Tree()
    t.populate(60, random_branches=True)
    for i, n in enumerate(t.traverse()):
        if not n.is_leaf():
            n.name = f"n{i}"
    depth = max(t.get_distance(l) for l in t)
    tp = pd.DataFrame({"gains": [5.0], "losses": [5.0], "dist": [0.6 * depth],
                       "loss_dist": [0.6 * depth], "gain_subsize": [1.0], "loss_subsize": [1.0],
                       "root_state": [0]}, index=["g"])
    return t, tp


def _prevalence(lineages, trials):
    bits = np.unpackbits(lineages.view(np.uint8), bitorder="little")
    return bits.reshape(lineages.shape[0], -1)[:, :trials].mean()


def test_emergence_modes_order_prevalence():
    from simphyni.Simulation.simulation import sim_bit
    t, tp = _deep_threshold_case()
    prev = {m: _prevalence(sim_bit(t, tp, trials=128, emergence=m, emergence_scale=0.5, seed=1), 128)
            for m in ("dist", "scaled", "soft", "none")}
    assert prev["dist"] < prev["scaled"] < prev["none"]
    assert prev["dist"] < prev["soft"] < prev["none"]
    assert abs(prev["none"] - 0.5) < 0.1          # equal gain/loss rates -> ~50 %
    with pytest.raises(ValueError):
        sim_bit(t, tp, emergence="bogus")


@pytest.mark.parametrize("emergence,kw", [("none", {}), ("soft", {}), ("scaled", {"emergence_scale": 0.5}),
                                          ("mrca", {}), ("mrca", {"mrca_subsize": "original"})])
def test_emergence_modes_run_end_to_end(emergence, kw):
    from ete3 import Tree
    from simphyni import TreeSimulator
    import random
    random.seed(5)
    rng = np.random.default_rng(5)
    t = Tree()
    t.populate(40, random_branches=True)
    obs = pd.DataFrame((rng.random((40, 4)) < 0.4).astype(float), index=t.get_leaf_names(),
                       columns=list("abcd"))
    params = pd.DataFrame({"gene": list("abcd"), "gains": 3.0, "losses": 3.0, "dist": 0.1,
                           "loss_dist": 0.0, "gain_subsize": 1.0, "loss_subsize": 1.0,
                           "root_state": 0})
    sim = TreeSimulator(t.write(format=1), params, obs)
    sim.initialize_simulation_parameters(pre_filter=False)
    sim.run_simulation(cores=1, seed=0, emergence=emergence, **kw)
    res = sim.get_results()
    assert len(res) == 6 and res["pval_naive"].between(0, 1).all()


def _cherry_sim(threshold):
    from simphyni import TreeSimulator
    nwk = "((A:0,B:0)n1:0.1,(C:0.2,D:0.3)n2:0.1,E:0.5)root;"
    obs = pd.DataFrame({"g1": [1, 0, 1, 0, 1], "g2": [0, 1, 1, 0, 0]}, index=list("ABCDE")).astype(float)
    pm = pd.DataFrame({"gene": ["g1", "g2"], "gains": 1.0, "losses": 1.0, "dist": 0.0, "loss_dist": 0.0,
                       "gain_subsize": 1.0, "loss_subsize": 1.0, "root_state": 0})
    s = TreeSimulator(nwk, pm, obs)
    s.initialize_simulation_parameters(pre_filter=False, collapse_threshold=threshold)
    return s


def test_zero_length_cherries_are_not_collapsed_by_default():
    s = _cherry_sim(0.0)
    assert sorted(s.tree.get_leaf_names()) == list("ABCDE")
    assert s.obsdf_modified.loc["A", "g1"] == 1 and s.obsdf_modified.loc["B", "g2"] == 1
    assert list(s.obsdf_modified.index) == s.tree.get_leaf_names()


def test_positive_collapse_threshold_is_deterministic():
    results = set()
    junk = []
    for i in range(5):
        junk.append([object() for _ in range(5000 * i)])   # vary memory layout
        s = _cherry_sim(0.05)
        results.add((tuple(s.tree.get_leaf_names()), s.obsdf_modified.to_csv()))
    assert len(results) == 1
    leaves = next(iter(results))[0]
    assert "n1" in leaves and "A" not in leaves      # A,B merged into n1
