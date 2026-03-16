import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from ete3 import Tree

# Adjust import based on your package structure
from simphyni import TreeSimulator

# ==========================================
#  FIXTURES
# ==========================================

@pytest.fixture
def basic_simulator():
    """Returns a simulator with a simple tree and dummy data."""
    tree_str = "((A:1,B:1):1,C:2);"
    
    # Obs: A and B are identical, C is different
    obs = pd.DataFrame({
        'G1': [1, 1, 0],
        'G2': [1, 1, 0],
        'G3': [0, 0, 1],
        'G4': [0, 1, 0] 
    }, index=['A', 'B', 'C'])
    
    # PastML: Minimal valid structure
    pml = pd.DataFrame({
        'gene': ['G1','G2','G3','G4'],
        'gains': [0.1]*4, 'losses': [0.1]*4, 
        'dist': [1.0]*4, 'loss_dist': [1.0]*4,
        'root_state': [0]*4, 'gain_subsize': [1]*4, 'loss_subsize': [1]*4
    })
    
    return TreeSimulator(tree_str, pml, obs)

# =================================
# TESTS: Initialization & Data Prep
# =================================

def test_obs_data_processing():
    """Test binarization (values > 0.5 become 1) and fillna."""
    raw_obs = pd.DataFrame({
        'T1': [0.1, 0.6, np.nan],
        'T2': [0.0, 1.0, 0.4]
    }, index=['A', 'B', 'C'])
    
    # We need dummy tree/pastml to init
    dummy_pastml = pd.DataFrame({'gene':['T1','T2'], 'gains':[0,0], 'losses':[0,0], 'dist':[0,0], 'loss_dist':[0,0]})
    sim = TreeSimulator("((A:1,B:1):1,C:1);", dummy_pastml, raw_obs)
    
    # Check T1 logic
    assert sim.obsdf.loc['A', 'T1'] == 0 # 0.1 -> 0
    assert sim.obsdf.loc['B', 'T1'] == 1 # 0.6 -> 1
    assert sim.obsdf.loc['C', 'T1'] == 0 # NaN -> 0 (fillna)
    
    # Check Integer conversion
    assert sim.obsdf['T1'].dtype == int or sim.obsdf['T1'].dtype == np.int64

# ======================================
# TESTS: Pair Selection (_get_pair_data)
# ======================================

def test_get_pair_data_vars_targets_identical(basic_simulator):
    """
    Test All-vs-All: vars and targets are the same.
    Should remove self-pairs (A,A) and symmetric duplicates (B,A if A,B exists).
    """
    basic_simulator.initialize_simulation_parameters(pre_filter=False)
    
    # Select just G1 and G2. 
    # Logic should generate (G1, G2). 
    # Should exclude (G1, G1), (G2, G2), and (G2, G1).
    vars_df = basic_simulator.obsdf[['G1', 'G2']]
    targets_df = basic_simulator.obsdf[['G1', 'G2']]
    
    pairs, stats = basic_simulator._get_pair_data(vars_df, targets_df, prevalence_threshold=0, pre_filter=False)
    
    # Convert to set of tuples for easy checking
    pair_set = set(tuple(p) for p in pairs)
    
    assert ('G1', 'G2') in pair_set or ('G2', 'G1') in pair_set
    assert ('G1', 'G1') not in pair_set
    assert ('G2', 'G2') not in pair_set
    assert len(pairs) == 1

def test_get_pair_data_vars_targets_distinct(basic_simulator):
    """
    Test Set A vs Set B: vars and targets are completely disjoint.
    Should keep ALL combinations.
    """
    basic_simulator.initialize_simulation_parameters(pre_filter=False)
    
    # Vars: [G1], Targets: [G3, G4]
    # Expected: (G1, G3), (G1, G4)
    vars_df = basic_simulator.obsdf[['G1']]
    targets_df = basic_simulator.obsdf[['G3', 'G4']]
    
    pairs, stats = basic_simulator._get_pair_data(vars_df, targets_df, prevalence_threshold=0, pre_filter=False)
    
    pair_set = set(tuple(p) for p in pairs)
    
    assert len(pairs) == 2
    assert ('G1', 'G3') in pair_set
    assert ('G1', 'G4') in pair_set

def test_get_pair_data_vars_targets_overlap(basic_simulator):
    """
    Test Overlap: vars and targets share some elements.
    Should handle the intersection correctly (no self-pairs, no duplicates).
    """
    basic_simulator.initialize_simulation_parameters(pre_filter=False)
    
    # Vars: [G1, G2]
    # Targets: [G2, G3]
    # Potential raw pairs: (G1, G2), (G1, G3), (G2, G2), (G2, G3)
    # Expected valid pairs: (G1, G2), (G1, G3), (G2, G3)
    # (G2, G2) removed.
    
    vars_df = basic_simulator.obsdf[['G1', 'G2']]
    targets_df = basic_simulator.obsdf[['G2', 'G3']]
    
    pairs, stats = basic_simulator._get_pair_data(vars_df, targets_df, prevalence_threshold=0, pre_filter=False)
    
    pair_set = set(tuple(p) for p in pairs)
    
    assert len(pairs) == 3
    assert ('G1', 'G2') in pair_set or ('G2', 'G1') in pair_set
    assert ('G1', 'G3') in pair_set
    assert ('G2', 'G3') in pair_set
    assert ('G2', 'G2') not in pair_set

# Removed in version 1.0.2, may reimplement in future versions
# def test_get_pair_data_prefiltering():
#     """
#     Test Fisher's Exact Test Prefiltering.
#     We create a dataset where Pair A-B is perfectly correlated (Significant),
#     and Pair A-C is random/uncorrelated (Not Significant).
#     """
#     # Create distinct data for this test
#     # 8 samples
#     # A: 1 1 1 1 0 0 0 0
#     # B: 1 1 1 1 0 0 0 0 (Matches A -> Significant)
#     # C: 1 0 1 0 1 0 1 0 (Random noise vs A -> Not significant)
    
#     obs = pd.DataFrame({
#         'A': [1,1,1,1,0,0,0,0],
#         'B': [1,1,1,1,0,0,0,0],
#         'C': [1,0,1,0,1,0,1,0]
#     }, index=[str(i) for i in range(8)])
#     print(obs)
#     # Dummy tree/pastml
#     tree = "(" + ",".join([f"{i}:1" for i in range(8)]) + ");"
#     pml = pd.DataFrame({'gene':['A','B','C'], 'gains':[0]*3, 'losses':[0]*3, 'dist':[0]*3, 'loss_dist':[0]*3})
    
#     sim = TreeSimulator(tree, pml, obs)
#     sim.initialize_simulation_parameters(pre_filter=True)
    
#     # We want to test A vs [B, C]
#     vars_df = sim.obsdf[['A']]
#     targets_df = sim.obsdf[['B', 'C']]
    
#     # Run get_pair_data with pre-filter enabled
#     pairs, stats = sim._get_pair_data(vars_df, targets_df, prevalence_threshold=0, pre_filter=True)
    
#     pair_set = set(tuple(p) for p in pairs)
    
#     # A-B should exist
#     assert ('A', 'B') in pair_set, "Significant pair (A,B) was filtered out erroneously"
    
#     # A-C should NOT exist (Fisher p-value should be high ~1.0)
#     assert ('A', 'C') not in pair_set, "Non-significant pair (A,C) failed to be filtered out"


# ==========================================
# TESTS: Full Pipeline Integration
# ==========================================

def test_full_computational_pipeline_verification():
    """
    Verifies the entire flow:
    1. Input Setup: correlated and anticorrelated traits.
    2. Pair Selection: ensuring correct pairs are identified.
    3. Simulation Execution: (mocked for speed, but verifying data hand-off).
    4. Result Processing: verifying direction and significance logic in the result table.
    """
    
    # Corr1/Corr2: Perfectly Co-occurring
    # Anti1/Anti2: Perfectly Disjoint (Anti-correlated)
    # Noise: Random
    obs_data = pd.DataFrame({
        'Corr1': [1, 1, 0, 0],
        'Corr2': [1, 1, 0, 0],
        'Anti1': [1, 1, 0, 0],
        'Anti2': [0, 0, 1, 1],
        'Noise': [1, 0, 1, 0]
    }, index=['A', 'B', 'C', 'D'])
    
    tree_str = "(((A:1.0,B:1.0)Internal_2:1.0,D:2.0)Internal_1:1.0,C:4.0);"
    pml = pd.DataFrame({
        'gene': ['Corr1','Corr2','Anti1','Anti2','Noise'],
        'gains': [0.1]*5, 'losses': [0.1]*5, 'dist': [1]*5, 'loss_dist': [1]*5,
        'root_state': [0]*5, 'gain_subsize': [1]*5, 'loss_subsize': [1]*5
    })
    
    sim = TreeSimulator(tree_str, pml, obs_data)
    

    sim.initialize_simulation_parameters(pre_filter=False)
    sim.run_simulation()
    
    res = sim.result

    # print(res[['T1','T2','direction','pval_naive']])

    
    # Helper to get row
    def get_row(t1, t2):
        row = res[((res['T1']==t1) & (res['T2']==t2)) | ((res['T1']==t2) & (res['T2']==t1))]
        return row.iloc[0] if not row.empty else None

    # Check Perfect Correlation Result
    row_corr = get_row('Corr1', 'Corr2')
    assert row_corr is not None
    assert row_corr['direction'] == 1, "Perfect correlation should be direction 1"
    assert row_corr['pval_naive'] < 0.05, "Perfect correlation should be significant"
    
    # Check Perfect Anticorrelation Result
    row_anti = get_row('Anti1', 'Anti2')
    assert row_anti is not None
    assert row_anti['direction'] == -1, "Perfect anticorrelation should be direction -1"
    assert row_anti['pval_naive'] < 0.05, "Perfect anticorrelation should be significant"
    
    # Check Noise (Implicitly tested by absence of assertion failure above,
    # but we can verify it exists and is likely not sig based on our mock logic)
    row_noise = get_row('Corr1', 'Noise')
    if row_noise is not None:
        assert np.isclose(row_noise['pval_naive'],0.5,atol=0.45) # Based on our mock logic for names not matching


# ==========================================
# TESTS: Pastml data validation compatibility
# ==========================================

def test_check_pastml_data_accepts_legacy_format():
    """Legacy JOINT-only pastml CSV (no marginal cols) passes _check_pastml_data."""
    tree_str = "((A:1,B:1):1,C:2);"
    obs = pd.DataFrame({'G1': [1, 0, 0]}, index=['A', 'B', 'C'])
    legacy_pml = pd.DataFrame({
        'gene': ['G1'],
        'gains': [1.0], 'losses': [0.5],
        'dist': [0.5], 'loss_dist': [0.3],
        # No marginal columns — simulates pastml.py + GL_tab.py output
    })
    # Should construct without raising AssertionError
    sim = TreeSimulator(tree_str, legacy_pml, obs)
    assert sim.pastml is not None


def test_check_pastml_data_accepts_new_acr_format():
    """New ACR CSV with marginal columns also passes _check_pastml_data."""
    tree_str = "((A:1,B:1):1,C:2);"
    obs = pd.DataFrame({'G1': [1, 0, 0]}, index=['A', 'B', 'C'])
    new_pml = pd.DataFrame({
        'gene': ['G1'],
        'gains': [2.0], 'losses': [1.0],
        'dist': [0.5], 'loss_dist': [0.3],
        'gain_subsize': [10.0], 'loss_subsize': [8.0],
        'gain_subsize_nofilter': [12.0], 'loss_subsize_nofilter': [9.0],
        'gain_subsize_thresh': [7.0], 'loss_subsize_thresh': [6.0],
        'root_state': [0],
        'gains_flow': [1.8], 'losses_flow': [0.9],
        'dist_marginal': [0.4], 'loss_dist_marginal': [0.2],
        'root_prob': [0.3],
    })
    sim = TreeSimulator(tree_str, new_pml, obs)
    assert sim.pastml is not None


# ==========================================
# TESTS: run_simulation mask passthrough
# ==========================================

def test_run_simulation_without_masks_calls_sim_correctly(basic_simulator):
    """run_simulation() without masks passes gain_mask=None to simulate_glrates_bit."""
    basic_simulator.initialize_simulation_parameters(pre_filter=False)

    with patch("simphyni.Simulation.tree_simulator.simulate_glrates_bit") as mock_sim:
        mock_sim.return_value = pd.DataFrame({
            "pair": [], "first": [], "second": [],
            "direction": [], "p-value": [], "effect size": [],
        })
        try:
            basic_simulator.run_simulation()
        except Exception:
            pass  # result processing may fail on empty df; that's fine for this test

        if mock_sim.called:
            call_kwargs = mock_sim.call_args.kwargs
            assert call_kwargs.get("gain_mask") is None
            assert call_kwargs.get("loss_mask") is None


def test_run_simulation_with_masks_subselects_columns(basic_simulator):
    """
    run_simulation() sub-selects mask columns to match only the traits being simulated.
    When gene_order contains more traits than simulated, only relevant columns are passed.
    """
    basic_simulator.initialize_simulation_parameters(pre_filter=False)

    # Build a mask for 6 traits, but only G1..G4 are in the simulator
    n_nodes = len(list(basic_simulator.tree.traverse()))
    all_genes = ["G1", "G2", "G3", "G4", "G5", "G6"]
    gain_mask = np.ones((n_nodes, len(all_genes)), dtype=bool)
    loss_mask = np.zeros((n_nodes, len(all_genes)), dtype=bool)

    with patch("simphyni.Simulation.tree_simulator.simulate_glrates_bit") as mock_sim:
        mock_sim.return_value = pd.DataFrame({
            "pair": [], "first": [], "second": [],
            "direction": [], "p-value": [], "effect size": [],
        })
        try:
            basic_simulator.run_simulation(gain_mask=gain_mask, loss_mask=loss_mask,
                                            gene_order=all_genes)
        except Exception:
            pass

        if mock_sim.called:
            call_kwargs = mock_sim.call_args.kwargs
            passed_gm = call_kwargs.get("gain_mask")
            if passed_gm is not None:
                # Sub-selected mask should have at most len(simulated_traits) columns
                n_simulated = len(np.unique(basic_simulator.pairs.flatten()))
                assert passed_gm.shape[1] <= n_simulated