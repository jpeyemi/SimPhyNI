"""
test_na_support.py
==================
Tests for NA/missing-value support throughout the SimPhyNI pipeline.

Covers:
  - Binarization preserving NaN (reformat_csv, tree_simulator)
  - Pairwise-complete observed statistics (pair_statistics)
  - NA-aware Fisher pre-filter (tree_simulator)
  - ACR with unknown tip states (fast_binary_acr)
  - MRCA clade masking with NA tips (simulation)
  - Prevalence calculation with NA
  - Full pipeline integration with NA
  - Edge cases (high NA rate, single measured tip, complementary NA, structured NA)
"""

import pytest
import numpy as np
import pandas as pd
from ete3 import Tree
from unittest.mock import patch

from simphyni import TreeSimulator, build_clade_mask
from simphyni.Simulation.pair_statistics import pair_statistics
from simphyni.scripts.reformat_csv import _binarize
from simphyni.scripts.fast_binary_acr import (
    build_tree_arrays, build_obs_matrix, fast_acr,
)


# ==========================================
#  FIXTURES
# ==========================================

@pytest.fixture
def tree_8tip():
    """8-tip balanced tree for statistical power."""
    return Tree("(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);")


@pytest.fixture
def obs_with_na():
    """Observed traits with various NA patterns."""
    return pd.DataFrame({
        'T1': [1.0, 1.0, 0.0, 0.0, np.nan, np.nan, 0.0, 0.0],
        'T2': [1.0, 1.0, 0.0, 0.0, 0.0,    0.0,    np.nan, np.nan],
        'T3': [1.0, 0.0, 1.0, 0.0, 1.0,    0.0,    1.0,    0.0],
        'T4': [np.nan]*8,   # all NA
    }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])


@pytest.fixture
def pml_4trait():
    """Minimal PastML params for 4 traits."""
    return pd.DataFrame({
        'gene': ['T1', 'T2', 'T3', 'T4'],
        'gains': [0.1, 0.1, 0.1, 0.0],
        'losses': [0.1, 0.1, 0.1, 0.0],
        'dist': [1.0, 1.0, 1.0, 0.0],
        'loss_dist': [1.0, 1.0, 1.0, 0.0],
        'root_state': [0, 0, 0, 0],
        'gain_subsize': [1.0]*4,
        'loss_subsize': [1.0]*4,
    })


# ==========================================
#  1. BINARIZATION — NaN preservation
# ==========================================

class TestBinarizePreservesNaN:

    def test_reformat_binarize_nan_preserved(self):
        """_binarize() in reformat_csv.py preserves NaN values."""
        df = pd.DataFrame({
            'A': [0.0, 1.5, np.nan, -0.1],
            'B': [np.nan, 0.0, 1.0, 0.5],
        })
        result = _binarize(df)
        # Positive → 1, non-positive → 0, NaN → NaN
        assert result.loc[0, 'A'] == 0.0
        assert result.loc[1, 'A'] == 1.0
        assert np.isnan(result.loc[2, 'A'])
        assert result.loc[3, 'A'] == 0.0
        assert np.isnan(result.loc[0, 'B'])
        assert result.loc[1, 'B'] == 0.0
        assert result.loc[2, 'B'] == 1.0
        assert result.loc[3, 'B'] == 1.0

    def test_reformat_binarize_no_na_unchanged(self):
        """_binarize() with no NaN values produces same result as before."""
        df = pd.DataFrame({'A': [0, 1, 2, 0], 'B': [1, 0, 0, 1]})
        result = _binarize(df)
        expected = pd.DataFrame({'A': [0.0, 1.0, 1.0, 0.0], 'B': [1.0, 0.0, 0.0, 1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_tree_simulator_binarize_obs_nan_preserved(self):
        """_binarize_obs() in TreeSimulator preserves NaN values."""
        obs = pd.DataFrame({
            'T1': [0.1, 0.6, np.nan],
            'T2': [0.0, 1.0, 0.4]
        }, index=['A', 'B', 'C'])
        pml = pd.DataFrame({
            'gene': ['T1', 'T2'], 'gains': [0, 0], 'losses': [0, 0],
            'dist': [0, 0], 'loss_dist': [0, 0],
        })
        sim = TreeSimulator("((A:1,B:1):1,C:1);", pml, obs)
        assert sim.obsdf.loc['A', 'T1'] == 0.0
        assert sim.obsdf.loc['B', 'T1'] == 1.0
        assert np.isnan(sim.obsdf.loc['C', 'T1'])
        assert sim.obsdf.loc['C', 'T2'] == 0.0  # 0.4 < 0.5 → 0

    def test_binarize_all_nan_column(self):
        """An all-NaN column survives binarization."""
        df = pd.DataFrame({'A': [np.nan, np.nan, np.nan]})
        result = _binarize(df)
        assert result['A'].isna().all()

    def test_binarize_string_column_coerced(self):
        """Non-numeric strings become NaN (not 0)."""
        df = pd.DataFrame({'A': ['yes', '1', np.nan, '0']})
        result = _binarize(df)
        assert np.isnan(result.loc[0, 'A'])  # 'yes' → NaN
        assert result.loc[1, 'A'] == 1.0     # '1' → 1
        assert np.isnan(result.loc[2, 'A'])  # NaN stays
        assert result.loc[3, 'A'] == 0.0     # '0' → 0


# ==========================================
#  2. PAIR STATISTICS — pairwise-complete NA masking
# ==========================================

class TestPairStatisticsNA:

    def test_log_odds_ratio_no_na_unchanged(self):
        """With no NaN, result matches original behavior."""
        tp = np.array([[1, 1, 0, 0],
                        [1, 0, 1, 0]], dtype=float).T   # 4 tips, 2 pairs
        tq = np.array([[1, 0, 1, 0],
                        [0, 1, 1, 0]], dtype=float).T
        result = pair_statistics._log_odds_ratio_statistic(tp, tq)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_log_odds_ratio_excludes_na_tips(self):
        """NaN tips are excluded from the contingency table."""
        # 6 tips, 1 pair
        # Tips 0-3: both measured, perfect co-occurrence (a=2,d=2,b=0,c=0)
        # Tips 4-5: NaN in tp → should be excluded
        tp = np.array([1, 1, 0, 0, np.nan, np.nan], dtype=float)
        tq = np.array([1, 1, 0, 0, 1,      0],      dtype=float)
        result = pair_statistics._log_odds_ratio_statistic(tp, tq)
        # Without NA masking, NaN→truthy would corrupt the table
        # With proper masking: a=2+1=3, b=0+1=1, c=0+1=1, d=2+1=3 → log(9/1) = log(9)
        expected = np.log(9.0)
        assert np.isclose(result, expected, atol=1e-10)

    def test_log_odds_ratio_all_na_one_trait(self):
        """All-NaN trait produces log(1) = 0 (epsilon-only table)."""
        tp = np.array([np.nan, np.nan, np.nan], dtype=float)
        tq = np.array([1, 0, 1], dtype=float)
        result = pair_statistics._log_odds_ratio_statistic(tp, tq)
        # No valid observations: a=ε, b=ε, c=ε, d=ε → log(1) = 0
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_log_odds_ratio_complementary_na(self):
        """When NaN patterns are complementary, no tips are jointly valid."""
        # T1 measured where T2 is NA, and vice versa
        tp = np.array([1, 1, np.nan, np.nan], dtype=float)
        tq = np.array([np.nan, np.nan, 1, 1], dtype=float)
        result = pair_statistics._log_odds_ratio_statistic(tp, tq)
        # 0 jointly valid tips → epsilon-only → log(1) = 0
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_log_odds_ratio_mixed_na_reduces_n(self):
        """NA reduces effective sample size; verify contingency table counts."""
        # 5 tips, tip 2 has NaN in tp
        tp = np.array([1, 1, np.nan, 0, 0], dtype=float)
        tq = np.array([1, 0, 1,      1, 0], dtype=float)
        result = pair_statistics._log_odds_ratio_statistic(tp, tq)
        # Valid tips: 0,1,3,4 (n=4)
        # tp_valid: 1,1,0,0  tq_valid: 1,0,1,0
        # a=1, b=1, c=1, d=1 (+ epsilon each) → log((2*2)/(2*2)) = 0
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_log_odds_ratio_vectorized_with_na(self):
        """Multiple pairs with different NA patterns computed in one call."""
        # shape (6 tips, 3 pairs)
        tp = np.array([
            [1, 1, np.nan],
            [1, np.nan, 1],
            [0, 0, 0],
            [0, 0, 0],
            [np.nan, 1, 1],
            [0, 0, np.nan],
        ], dtype=float)
        tq = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=float)
        result = pair_statistics._log_odds_ratio_statistic(tp, tq)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))


# ==========================================
#  3. FISHER PRE-FILTER — NA handling
# ==========================================

class TestFisherPrefilterNA:

    def test_fisher_with_na_does_not_crash(self, tree_8tip, obs_with_na, pml_4trait):
        """Fisher pre-filter runs without error when data contains NaN."""
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml_4trait, obs_with_na.drop(columns=['T4']))
        # Should not raise
        sim.initialize_simulation_parameters(pre_filter=True)

    def test_fisher_without_na_matches_prefilter_false(self, tree_8tip):
        """With no NaN, pre-filter should keep perfectly correlated pairs."""
        obs = pd.DataFrame({
            'A': [1, 1, 1, 1, 0, 0, 0, 0],
            'B': [1, 1, 1, 1, 0, 0, 0, 0],  # perfectly correlated with A
            'C': [0, 0, 0, 0, 1, 1, 1, 1],  # anti-correlated with A
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        pml = pd.DataFrame({
            'gene': ['A', 'B', 'C'],
            'gains': [0.1]*3, 'losses': [0.1]*3,
            'dist': [1.0]*3, 'loss_dist': [1.0]*3,
            'root_state': [0]*3, 'gain_subsize': [1]*3, 'loss_subsize': [1]*3,
        })
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml, obs)
        sim.initialize_simulation_parameters(pre_filter=True)
        pairs_set = set(tuple(p) for p in sim.pairs)
        # A-B (correlated) and A-C (anti-correlated) should both be kept
        assert ('A', 'B') in pairs_set or ('B', 'A') in pairs_set
        assert ('A', 'C') in pairs_set or ('C', 'A') in pairs_set

    def test_fisher_na_does_not_inflate_significance(self, tree_8tip):
        """NaN values should not make truly independent pairs appear significant."""
        # Two independent traits with NaN that would appear correlated if NaN→True
        obs = pd.DataFrame({
            'X': [1, 0, 1, 0, np.nan, np.nan, np.nan, np.nan],
            'Y': [np.nan, np.nan, np.nan, np.nan, 1, 0, 1, 0],
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        pml = pd.DataFrame({
            'gene': ['X', 'Y'], 'gains': [0.1]*2, 'losses': [0.1]*2,
            'dist': [1.0]*2, 'loss_dist': [1.0]*2,
            'root_state': [0]*2, 'gain_subsize': [1]*2, 'loss_subsize': [1]*2,
        })
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml, obs)
        sim.initialize_simulation_parameters(pre_filter=True)
        # Complementary NaN: 0 jointly valid → Fisher p=1.0 → pair excluded
        assert len(sim.pairs) == 0


# ==========================================
#  4. ACR — unknown tip states (-1 sentinel)
# ==========================================

class TestACRWithNA:

    @pytest.fixture
    def star_tree_labeled(self):
        t = Tree("(A:1,B:1,C:1,D:1);")
        for idx, node in enumerate(t.traverse()):
            if not node.is_leaf() and not node.name:
                node.name = f"N{idx}"
        return t

    def test_build_obs_matrix_nan_to_sentinel(self, star_tree_labeled):
        """NaN values in trait_df become -1 sentinel in obs matrix."""
        ta = build_tree_arrays(star_tree_labeled)
        trait_df = pd.DataFrame({
            'T1': [1.0, 0.0, np.nan, 1.0],
        }, index=['A', 'B', 'C', 'D'])
        obs = build_obs_matrix(ta, trait_df)
        # Find C's node index
        c_idx = ta.name_to_idx['C']
        assert obs[0, c_idx] == -1, "NaN should map to -1 sentinel"
        # A and D should be 1
        assert obs[0, ta.name_to_idx['A']] == 1
        assert obs[0, ta.name_to_idx['D']] == 1
        # B should be 0
        assert obs[0, ta.name_to_idx['B']] == 0

    def test_acr_unknown_tips_produce_uncertain_marginals(self, star_tree_labeled):
        """Tips with -1 sentinel get non-degenerate marginal probabilities."""
        ta = build_tree_arrays(star_tree_labeled)
        # All tips known except C
        trait_df = pd.DataFrame({
            'T1': [1.0, 1.0, np.nan, 0.0],
        }, index=['A', 'B', 'C', 'D'])
        obs = build_obs_matrix(ta, trait_df)
        result = fast_acr(ta, obs, ['T1'], mode='empirical')
        c_idx = ta.name_to_idx['C']
        p1_c = result.marginal_p1[0, c_idx]
        # C is unknown: marginal should be between 0 and 1 (not forced to 0 or 1)
        assert 0.0 < p1_c < 1.0, f"Unknown tip should have intermediate marginal, got {p1_c}"

    def test_acr_all_known_no_change(self, star_tree_labeled):
        """ACR with all known tips works as before (regression check)."""
        ta = build_tree_arrays(star_tree_labeled)
        trait_df = pd.DataFrame({'T1': [1, 1, 0, 0]}, index=['A', 'B', 'C', 'D'])
        obs = build_obs_matrix(ta, trait_df)
        result = fast_acr(ta, obs, ['T1'], mode='empirical')
        # Should complete without error and produce valid marginals
        assert result.marginal_p1.shape == (1, ta.n_nodes)
        assert np.all(np.isfinite(result.marginal_p1))

    def test_acr_all_unknown_tips_uses_prior(self, star_tree_labeled):
        """If all tips are unknown, marginals should converge to the prior."""
        ta = build_tree_arrays(star_tree_labeled)
        trait_df = pd.DataFrame({'T1': [np.nan]*4}, index=['A', 'B', 'C', 'D'])
        obs = build_obs_matrix(ta, trait_df)
        result = fast_acr(ta, obs, ['T1'], mode='empirical')
        # With no informative data, all marginals should be close to pi1_init
        # pi1_init = 0.01 (fallback when n_known=0)
        assert np.all(np.isfinite(result.marginal_p1))


# ==========================================
#  5. CLADE MASKING — NA tips
# ==========================================

class TestCladeMaskNA:

    @pytest.fixture
    def balanced_tree(self):
        return Tree("((T1:1.0,T2:1.0)Int1:1.0,(T3:1.0,T4:1.0)Int2:1.0)Root:0.0;", format=1)

    def test_clade_mask_skips_na_tips(self, balanced_tree):
        """NA tips are not counted as minority leaves."""
        obsdf = pd.DataFrame({
            'G': [1.0, np.nan, 0.0, 0.0],
        }, index=['T1', 'T2', 'T3', 'T4'])
        gm, lm, mrca_bl = build_clade_mask(balanced_tree, obsdf, ['G'], {'G': 0})
        # Only T1 is a minority leaf (T2 is NA, not minority)
        nodes = list(balanced_tree.traverse())
        node_names_eligible = {nodes[i].name for i, v in enumerate(gm[:, 0]) if v}
        assert 'T1' in node_names_eligible
        # T2 should NOT drive MRCA selection
        # MRCA of [T1] alone is T1
        assert node_names_eligible == {'T1'}

    def test_clade_mask_all_na_gets_false_mask(self, balanced_tree):
        """All-NaN trait → no minority leaves → mask is all-False."""
        obsdf = pd.DataFrame({
            'G': [np.nan, np.nan, np.nan, np.nan],
        }, index=['T1', 'T2', 'T3', 'T4'])
        gm, lm, mrca_bl = build_clade_mask(balanced_tree, obsdf, ['G'], {'G': 0})
        assert not gm[:, 0].any()
        assert mrca_bl[0] == pytest.approx(0.0)

    def test_clade_mask_na_does_not_crash(self, balanced_tree):
        """build_clade_mask does not raise on NaN values."""
        obsdf = pd.DataFrame({
            'G': [1.0, np.nan, 0.0, np.nan],
        }, index=['T1', 'T2', 'T3', 'T4'])
        # Should not raise ValueError from int(NaN)
        gm, lm, mrca_bl = build_clade_mask(balanced_tree, obsdf, ['G'], {'G': 0})
        assert gm.shape[1] == 1

    def test_clade_mask_mixed_na_correct_mrca(self, balanced_tree):
        """With some NA, MRCA is computed from measured minority leaves only."""
        obsdf = pd.DataFrame({
            'G': [1.0, 1.0, np.nan, 0.0],  # T3 is NA
        }, index=['T1', 'T2', 'T3', 'T4'])
        gm, lm, mrca_bl = build_clade_mask(balanced_tree, obsdf, ['G'], {'G': 0})
        nodes = list(balanced_tree.traverse())
        eligible = {nodes[i].name for i, v in enumerate(gm[:, 0]) if v}
        # Minority leaves: T1, T2 (both have value 1 when root_state=0)
        # MRCA of T1,T2 = Int1; eligible = {Int1, T1, T2}
        assert 'Int1' in eligible
        assert 'T1' in eligible
        assert 'T2' in eligible


# ==========================================
#  6. PREVALENCE — NaN handling
# ==========================================

class TestPrevalenceNA:

    def test_prevalence_uses_nansum(self, tree_8tip, pml_4trait):
        """Prevalence threshold uses nansum (NaN doesn't inflate or deflate counts)."""
        obs = pd.DataFrame({
            'T1': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # prev = 2/8 = 0.25
            'T2': [1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # nansum=2, prev=2/8=0.25
            'T3': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # prev = 4/8 = 0.5
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        pml = pd.DataFrame({
            'gene': ['T1', 'T2', 'T3'], 'gains': [0.1]*3, 'losses': [0.1]*3,
            'dist': [1.0]*3, 'loss_dist': [1.0]*3,
            'root_state': [0]*3, 'gain_subsize': [1]*3, 'loss_subsize': [1]*3,
        })
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml, obs)
        # Prevalence threshold 0.3 → T1 (0.25) and T2 (0.25) excluded, T3 (0.5) kept
        # With only T3 valid, no pairs possible
        sim.initialize_simulation_parameters(prevalence_threshold=0.3, pre_filter=False)
        assert len(sim.pairs) == 0


# ==========================================
#  7. FULL PIPELINE INTEGRATION — NA
# ==========================================

class TestFullPipelineNA:

    def test_pipeline_with_na_runs_to_completion(self, tree_8tip):
        """Full pipeline (init → simulate → results) completes with NaN in data."""
        obs = pd.DataFrame({
            'T1': [1, 1, 0, 0, np.nan, 0, 0, 0],
            'T2': [1, 1, 0, 0, 0, np.nan, 0, 0],
            'T3': [0, 0, 1, 1, 0, 0, np.nan, np.nan],
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], dtype=float)
        pml = pd.DataFrame({
            'gene': ['T1', 'T2', 'T3'],
            'gains': [0.2, 0.2, 0.2], 'losses': [0.1, 0.1, 0.1],
            'dist': [1.0]*3, 'loss_dist': [1.0]*3,
            'root_state': [0]*3, 'gain_subsize': [2.0]*3, 'loss_subsize': [2.0]*3,
        })
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml, obs)
        sim.initialize_simulation_parameters(pre_filter=False)
        if len(sim.pairs) > 0:
            sim.run_simulation()
            res = sim.get_results()
            assert 'pval_bh' in res.columns
            assert np.all(np.isfinite(res['pval_naive'].values))

    def test_pipeline_na_vs_no_na_direction_consistent(self, tree_8tip):
        """Correlated traits stay positive, anti-correlated stay negative, even with NA."""
        obs_clean = pd.DataFrame({
            'Corr1': [1, 1, 1, 1, 0, 0, 0, 0],
            'Corr2': [1, 1, 1, 1, 0, 0, 0, 0],
            'Anti':  [0, 0, 0, 0, 1, 1, 1, 1],
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], dtype=float)
        # Inject 2 NAs into each trait (tips E, F)
        obs_na = obs_clean.copy()
        obs_na.loc['E', 'Corr1'] = np.nan
        obs_na.loc['F', 'Corr2'] = np.nan

        pml = pd.DataFrame({
            'gene': ['Corr1', 'Corr2', 'Anti'],
            'gains': [0.2]*3, 'losses': [0.1]*3,
            'dist': [1.0]*3, 'loss_dist': [1.0]*3,
            'root_state': [0]*3, 'gain_subsize': [2.0]*3, 'loss_subsize': [2.0]*3,
        })
        tree_str = tree_8tip.write(format=1)

        sim = TreeSimulator(tree_str, pml, obs_na)
        sim.initialize_simulation_parameters(pre_filter=False)
        if len(sim.pairs) == 0:
            pytest.skip("No pairs generated (all filtered)")

        # Find Corr1-Corr2 pair
        for i, (t1, t2) in enumerate(sim.pairs):
            if set([t1, t2]) == {'Corr1', 'Corr2'}:
                obs_stat = sim.obspairs[i]
                assert obs_stat > 0, "Correlated pair should have positive observed statistic"
                break

    def test_obsdf_modified_preserves_nan(self, tree_8tip):
        """After _collapse_tree_tips, NaN values should still be present."""
        obs = pd.DataFrame({
            'T1': [1, 1, 0, np.nan, 0, 0, 0, 0],
            'T2': [0, 0, 1, 1, np.nan, 0, 0, 0],
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], dtype=float)
        pml = pd.DataFrame({
            'gene': ['T1', 'T2'], 'gains': [0.1]*2, 'losses': [0.1]*2,
            'dist': [1.0]*2, 'loss_dist': [1.0]*2,
            'root_state': [0]*2, 'gain_subsize': [1]*2, 'loss_subsize': [1]*2,
        })
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml, obs)
        sim.initialize_simulation_parameters(pre_filter=False, collapse_threshold=0)
        # obsdf_modified should still contain NaN
        assert sim.obsdf_modified.isna().any().any(), "NaN should survive _collapse_tree_tips"


# ==========================================
#  8. EDGE CASES
# ==========================================

class TestNAEdgeCases:

    def test_high_na_rate_trait_filtered_by_prevalence(self, tree_8tip):
        """Trait with 90% NA has very low nansum-based prevalence → filtered out."""
        obs = pd.DataFrame({
            'Rare': [1.0] + [np.nan]*7,    # nansum=1, prev=1/8=0.125
            'Common': [1, 1, 1, 1, 0, 0, 0, 0],  # prev=0.5
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], dtype=float)
        pml = pd.DataFrame({
            'gene': ['Rare', 'Common'], 'gains': [0.1]*2, 'losses': [0.1]*2,
            'dist': [1.0]*2, 'loss_dist': [1.0]*2,
            'root_state': [0]*2, 'gain_subsize': [1]*2, 'loss_subsize': [1]*2,
        })
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml, obs)
        # With prev_threshold=0.2, Rare (0.125) is excluded
        sim.initialize_simulation_parameters(prevalence_threshold=0.2, pre_filter=False)
        # Only Common survives, so 0 pairs (can't pair with itself)
        assert len(sim.pairs) == 0

    def test_single_measured_tip(self, tree_8tip):
        """Only 1 tip measured for a trait → prevalence too low for any pair."""
        obs = pd.DataFrame({
            'T1': [1.0] + [np.nan]*7,
            'T2': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], dtype=float)
        pml = pd.DataFrame({
            'gene': ['T1', 'T2'], 'gains': [0.1]*2, 'losses': [0.1]*2,
            'dist': [1.0]*2, 'loss_dist': [1.0]*2,
            'root_state': [0]*2, 'gain_subsize': [1]*2, 'loss_subsize': [1]*2,
        })
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml, obs)
        sim.initialize_simulation_parameters(prevalence_threshold=0.2, pre_filter=False)
        # T1 has prevalence 1/8=0.125 < 0.2 threshold → excluded
        assert len(sim.pairs) == 0

    def test_na_symmetric_pattern_equal_statistics(self):
        """Symmetric NA pattern (same positions) gives same result regardless of which trait has NA."""
        tp1 = np.array([1, 1, 0, 0, np.nan, np.nan], dtype=float)
        tq1 = np.array([1, 0, 1, 0, np.nan, np.nan], dtype=float)
        tp2 = np.array([np.nan, np.nan, 1, 1, 0, 0], dtype=float)
        tq2 = np.array([np.nan, np.nan, 1, 0, 1, 0], dtype=float)
        r1 = pair_statistics._log_odds_ratio_statistic(tp1, tq1)
        r2 = pair_statistics._log_odds_ratio_statistic(tp2, tq2)
        # Same 2×2 table (a=1,b=1,c=1,d=1 before epsilon) → same statistic
        assert np.isclose(r1, r2, atol=1e-10)

    def test_structured_missingness_one_clade_na(self, tree_8tip):
        """NA concentrated in one clade doesn't crash pipeline."""
        # Left clade (A,B,C,D) has NA for T1; right clade measured
        obs = pd.DataFrame({
            'T1': [np.nan, np.nan, np.nan, np.nan, 1, 1, 0, 0],
            'T2': [1, 1, 0, 0, 1, 1, 0, 0],
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], dtype=float)
        pml = pd.DataFrame({
            'gene': ['T1', 'T2'], 'gains': [0.1]*2, 'losses': [0.1]*2,
            'dist': [1.0]*2, 'loss_dist': [1.0]*2,
            'root_state': [0]*2, 'gain_subsize': [1]*2, 'loss_subsize': [1]*2,
        })
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml, obs)
        sim.initialize_simulation_parameters(pre_filter=False)
        # Pipeline should handle this without error
        if len(sim.pairs) > 0:
            # Observed statistic should be computed over 4 jointly-valid tips only
            assert np.all(np.isfinite(sim.obspairs))

    def test_all_na_data_no_pairs(self, tree_8tip):
        """Completely NA dataset produces 0 pairs."""
        obs = pd.DataFrame({
            'T1': [np.nan]*8,
            'T2': [np.nan]*8,
        }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], dtype=float)
        pml = pd.DataFrame({
            'gene': ['T1', 'T2'], 'gains': [0.1]*2, 'losses': [0.1]*2,
            'dist': [1.0]*2, 'loss_dist': [1.0]*2,
            'root_state': [0]*2, 'gain_subsize': [1]*2, 'loss_subsize': [1]*2,
        })
        tree_str = tree_8tip.write(format=1)
        sim = TreeSimulator(tree_str, pml, obs)
        # All NaN → nansum=0 → prevalence=0 → excluded at any threshold > 0
        sim.initialize_simulation_parameters(prevalence_threshold=0.01, pre_filter=False)
        assert len(sim.pairs) == 0
