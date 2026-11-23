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


def _small_sim(seed, cores=2, **kw):
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
    sim.run_simulation(cores=cores, seed=seed, **kw)
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


def test_condition_on_prevalence_selects_closest_replicates():
    from simphyni.Simulation.simulation import condition_on_prevalence
    rng = np.random.default_rng(0)
    n_tips, trials, over = 100, 64, 8
    prev = np.linspace(0.0, 0.8, trials * over)                 # replicate j has prevalence ~prev[j]
    bits = (rng.random((n_tips, 1, trials * over)) < prev).astype(np.uint8)
    packed = np.packbits(bits, axis=-1, bitorder="little").view(np.uint64)
    out = condition_on_prevalence(packed, np.array([0.3]), trials, seed=1)
    kept = np.unpackbits(out.view(np.uint8), axis=-1, bitorder="little").reshape(n_tips, 1, -1)[:, :, :trials]
    p = kept.mean(axis=0).ravel()
    assert out.shape == (n_tips, 1, 1)
    assert abs(np.median(p) - 0.3) < 0.03 and p.min() > 0.15 and p.max() < 0.45


def test_conditioning_off_is_unchanged():
    # Explicitly unconditioned run vs. the same run through _small_sim's path
    a = _small_sim(21, condition=None, eligibility=None)
    from ete3 import Tree
    from simphyni import TreeSimulator
    import random
    random.seed(7)
    rng = np.random.default_rng(7)
    t = Tree()
    t.populate(50, random_branches=True)
    obs = pd.DataFrame((rng.random((50, 5)) < 0.4).astype(float), index=t.get_leaf_names(), columns=list("abcde"))
    params = pd.DataFrame({"gene": list("abcde"), "gains": 3.0, "losses": 3.0, "dist": 0.0, "loss_dist": 0.0,
                           "gain_subsize": 1.0, "loss_subsize": 1.0, "root_state": 0})
    sim = TreeSimulator(t.write(format=1), params, obs)
    sim.initialize_simulation_parameters(pre_filter=False)
    sim.run_simulation(cores=2, seed=21, condition=None, eligibility=None)
    b = sim.get_results().sort_values(["T1", "T2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_phylogroup_mask_confines_events_to_occupied_groups():
    from ete3 import Tree
    from simphyni.Simulation.simulation import build_phylogroup_mask, phylogroups
    t = Tree("((A:1,B:1)x:1,((C:1,D:1)y:0.5,E:1.5)z:1,F:0.1)root;", format=1)
    group, n = phylogroups(t, 0.3)
    assert n == 3 and group[t & "x"] == group[t & "A"] and group[t] == -1
    obs = pd.DataFrame({"g": [1, 0, 0, 0, 0, 0], "all": [1, 1, 1, 1, 1, 0], "inv": [1, 1, 1, 1, 1, 0]},
                       index=list("ABCDEF"), dtype=float)
    gm, lm, bl = build_phylogroup_mask(t, obs, ["g", "all", "inv"],
                                       pd.Series({"g": 0, "all": 0, "inv": 1}), 0.3)
    names = [n.name for n in t.traverse()]
    assert sorted(np.array(names)[gm[:, 0]]) == ["A", "B", "x"] and bl[0] == 3.0
    assert gm[:, 1].sum() == len(names) - 2          # 'all' misses group F only -> F and root excluded
    assert sorted(np.array(names)[gm[:, 2]]) == ["F"]  # root state 1: only F differs
    assert (gm == lm).all()


def test_phylogroup_eligibility_runs_end_to_end():
    res_kw = dict(eligibility="phylogroup", phylogroup_depth=0.2, condition="prevalence", oversample=4)
    from ete3 import Tree
    from simphyni import TreeSimulator
    import random
    random.seed(9)
    rng = np.random.default_rng(9)
    t = Tree()
    t.populate(60, random_branches=True)
    obs = pd.DataFrame((rng.random((60, 4)) < 0.3).astype(float), index=t.get_leaf_names(), columns=list("abcd"))
    params = pd.DataFrame({"gene": list("abcd"), "gains": 3.0, "losses": 3.0, "dist": 0.0, "loss_dist": 0.0,
                           "gain_subsize": 1.0, "loss_subsize": 1.0, "root_state": 0})
    sim = TreeSimulator(t.write(format=1), params, obs)
    sim.initialize_simulation_parameters(pre_filter=False)
    sim.run_simulation(cores=1, seed=3, emergence="none", **res_kw)
    res = sim.get_results()
    assert len(res) == 6 and res["pval_naive"].between(0, 1).all()


def test_adaptive_phylogroup_excludes_only_when_absence_is_unexpected():
    from ete3 import Tree
    from simphyni import TreeSimulator
    import random
    random.seed(11)
    rng = np.random.default_rng(11)
    # two deep clades of 60 tips each joined by long stems -> two phylogroups
    a, b = Tree(), Tree()
    a.populate(60, random_branches=True); b.populate(60, random_branches=True)
    for sub, tag in ((a, "a"), (b, "b")):
        for i, l in enumerate(sub.get_leaves()):
            l.name = f"{tag}{i}"
    root = Tree(); root.add_child(a, dist=2.0); root.add_child(b, dist=2.0)
    leaves = root.get_leaf_names()
    in_a = np.array([n.startswith("a") for n in leaves])
    busy = np.where(in_a, rng.random(len(leaves)) < 0.5, 0)          # many changes, only in clade a
    sparse = np.zeros(len(leaves)); sparse[np.where(in_a)[0][:6]] = 1  # a few tips in clade a
    obs = pd.DataFrame({"busy": busy, "sparse": sparse}, index=leaves, dtype=float)
    total = sum(n.dist for n in root.traverse())
    params = pd.DataFrame({"gene": ["busy", "sparse"], "gains": [40.0, 1.0], "losses": [40.0, 0.5],
                           "dist": 0.0, "loss_dist": 0.0, "gain_subsize": total, "loss_subsize": total,
                           "root_state": 0})
    sim = TreeSimulator(root.write(format=1), params, obs)
    sim.initialize_simulation_parameters(pre_filter=False)
    sim.run_simulation(cores=1, seed=5, emergence="none", eligibility="adaptive", phylogroup_depth=0.3)
    ex = sim.excluded_phylogroups
    assert ex["busy"] >= 1        # absence from clade b is implausible under the null -> closed
    assert ex["sparse"] == 0      # absence is expected for a rare trait -> left open


def test_fitch_changes_packed_matches_hand_counts():
    from ete3 import Tree
    from simphyni.Simulation.simulation import fitch_changes_packed, pack_observed
    t = Tree("(((a:1,b:1):1,(c:1,d:1):1):1,((e:1,f:1):1,(g:1,h:1):1):1);", format=1)
    order = [l.name for l in t]
    cols = {"one_clade": "abcd", "two_tips": "ah", "none": "", "alternating": "aceg"}
    obs = np.array([[1.0 if n in tips else 0.0 for tips in cols.values()] for n in order])
    f = fitch_changes_packed(t, pack_observed(obs), 1)[:, 0]
    assert f.tolist() == [1, 2, 0, 4]


def test_footprint_conditioning_matches_changes_better_than_prevalence():
    import random
    from ete3 import Tree
    from simphyni.Simulation.simulation import (sim_bit, condition_on_prevalence, condition_on_footprint,
                                                fitch_changes_packed, pack_observed)
    random.seed(13)
    t = Tree()
    t.populate(120, random_branches=True)
    for i, n in enumerate(t.traverse()):
        if not n.is_leaf():
            n.name = f"n{i}"
    leaves = [l.name for l in t]
    # observed: one clade of ~15 tips (1 change); simulated rates are 10x too high
    clade = next(n for n in t.traverse() if 12 <= len(n) <= 20)
    obs = np.array([[1.0 if l in set(clade.get_leaf_names()) else 0.0] for l in leaves])
    total = sum(n.dist for n in t.traverse())
    tp = pd.DataFrame({"gains": [10.0], "losses": [10.0], "dist": [0.0], "loss_dist": [0.0],
                       "gain_subsize": [total], "loss_subsize": [total], "root_state": [0]}, index=["g"])
    L = sim_bit(t, tp, trials=64 * 16, seed=1, emergence="none")
    f_obs = fitch_changes_packed(t, pack_observed(obs), 1)[0, 0]
    fp = np.median(fitch_changes_packed(t, condition_on_prevalence(L, obs.mean(0), 64, seed=2), 64))
    ff = np.median(fitch_changes_packed(t, condition_on_footprint(t, L, pack_observed(obs), 64, seed=2), 64))
    assert abs(ff - f_obs) < abs(fp - f_obs)


def test_branch_cap_limits_events_on_long_branches():
    from ete3 import Tree
    from simphyni.Simulation.simulation import sim_bit, iqr_branch_cap
    # star tree: 40 short tips and 1 very long tip
    t = Tree("(" + ",".join(f"s{i}:0.01" for i in range(40)) + ",L:100);", format=1)
    cap = iqr_branch_cap(t)
    assert cap < 100
    tp = pd.DataFrame({"gains": [1.0], "losses": [0.0], "dist": [0.0], "loss_dist": [0.0],
                       "gain_subsize": [10.0], "loss_subsize": [10.0], "root_state": [0]}, index=["g"])
    order = [l.name for l in t]
    li = order.index("L")
    def frac_long(cap_):
        lin = sim_bit(t, tp, trials=640, seed=0, emergence="none", branch_cap=cap_)
        bits = np.unpackbits(lin.view(np.uint8), axis=-1, bitorder="little").reshape(len(order), 1, -1)[:, 0, :640]
        return bits[li].mean()
    assert frac_long(None) > 0.99                      # uncapped: the long branch always flips
    assert frac_long(cap) < 0.5                        # capped: consistent with the denominator


def test_single_shared_clade_is_not_significant():
    """Two traits that each arose once, in the same clade, are weak evidence of
    association (paper Fig. 1a): the default conditioned null must not call
    them significant, whereas the legacy unconditioned null did."""
    from simphyni import TreeSimulator
    obs = pd.DataFrame({"x": [1, 1, 0, 0, 0, 0, 0, 0], "y": [1, 1, 0, 0, 0, 0, 0, 0]},
                       index=list("ABCDEFGH"), dtype=float)
    nwk = "((((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1));"
    pml = pd.DataFrame({"gene": ["x", "y"], "gains": 1.0, "losses": 0.0, "dist": 0.0, "loss_dist": 0.0,
                        "root_state": 0, "gain_subsize": 14.0, "loss_subsize": 14.0})
    def p(**kw):
        sim = TreeSimulator(nwk, pml, obs)
        sim.initialize_simulation_parameters(pre_filter=False)
        sim.run_simulation(cores=1, seed=4, **kw)
        return float(sim.get_results()["pval_naive"].iloc[0])
    assert p() > 0.05
    assert p(condition=None, branch_cap=None) < p()
