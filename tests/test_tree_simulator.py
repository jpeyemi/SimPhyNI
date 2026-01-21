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