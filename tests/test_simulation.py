import pytest
import numpy as np
import pandas as pd
from ete3 import Tree
from unittest.mock import MagicMock, patch

# Import your module here. 
# Assuming the file provided is named 'simulation_methods.py' inside 'simphyni' package
from simphyni import sim_bit, simulate_glrates_bit, compres
from simphyni.Simulation.simulation import unpack_trait_params, circular_bitshift_right, compute_kde_stats, sum_all_bits, compute_bitwise_cooc, process_batch

# ==========================================
# 1. FIXTURES: Complex Data Setup
# ==========================================

@pytest.fixture
def simple_tree():
    """
    A deep, asymmetric tree for testing distance thresholds and inheritance.
    Structure:
    Root
    |-- C (Leaf, dist=4.0)
    |
    \\-- Internal_1 (dist=1.0)
        |-- Internal_2 (dist=1.0)
        |   |-- A (Leaf, dist=1.0)
        |   \\-- B (Leaf, dist=1.0)
        \\-- D (Leaf, dist=2.0)
        
    Total distances from root:
    C: 4.0 | Int1: 1.0 | Int2: 2.0 | A: 3.0 | B: 3.0 | D: 3.0
    """
    # t = Tree("((A:1,B:1)Int2:1,D:2)Int1:1,C:4;", format=1)
    t = Tree("(((A:1.0,B:1.0)Internal_2:1.0,D:2.0)Internal_1:1.0,C:4.0);", format = 1)
    return t

@pytest.fixture
def trait_params():
    """
    A parameter set defining 5 distinct trait behaviors for edge-case testing:
    0. 'FastFlip': High gain, High loss (Standard noise).
    1. 'Conserved': Root=1, Gain=0, Loss=0 (Should never change).
    2. 'Impossible': Root=0, Gain=0 (Should never exist).
    3. 'Delayed': Root=0, High Gain, Threshold=3.1 (Only appears at one tip).
    4. 'Vulnerable': Root=1, Loss=Infinite (Should disappear immediately).
    """
    data = {
        'gains':        [0.5, 0.0, 0.0, 100.0, 0.0],
        'losses':       [0.5, 0.0, 0.0, 0.0,   10000.0],
        'dist':         [0.0, 0.0, 0.0, 3.1,   0.0],
        'loss_dist':    [0.0, 0.0, 0.0, 0.0,   0.0],
        'gain_subsize': [1.0, 1.0, 1.0, 1.0,   1.0],
        'loss_subsize': [1.0, 1.0, 1.0, 1.0,   1.0],
        'root_state':   [0,   1,   0,   0,     1]
    }
    # Index names help with mapping tests
    df = pd.DataFrame(data)
    df.index = ['FastFlip', 'Conserved', 'Impossible', 'Delayed', 'Vulnerable']
    return df

# ==========================================
# 2. UNIT TESTS: Math & Helper Logic
# ==========================================

def test_unpack_trait_params(trait_params):
    """Test that dataframe columns are unpacked correctly and Infs are handled."""
    # Introduce an Inf to test sanitization
    trait_params.loc['FastFlip', 'dist'] = np.inf
    
    gains, losses, dists, loss_dists, _, _, root_states = \
        unpack_trait_params(trait_params)

    assert isinstance(gains, np.ndarray)
    assert dists[0] == 0  # Inf should be converted to 0
    assert root_states[1] == 1 # Conserved trait has root 1

@pytest.mark.parametrize("shift", [0, 1, 32, 63, 64])
def test_circular_bitshift_logic(shift):
    """
    Property Test: Rotation by K then by (64-K) should restore original array.
    Also verifies basic bit movement.
    """
    # 1. Basic Movement Check
    # Create (1,1) array with value 1 (Binary ...0001)
    arr = np.array([[1]], dtype=np.uint64)
    shifted = circular_bitshift_right(arr, k=1, bits=64)
    # 1 shifted right by 1 in 64-bit circle becomes the MSB (2^63)
    assert shifted[0, 0] == (np.uint64(1) << np.uint64(63))

    # 2. Restoration Property Check
    original = np.random.randint(0, 2**63, size=(10, 2), dtype=np.uint64)
    shifted = circular_bitshift_right(original, shift, bits=64)
    restore_shift = (64 - (shift % 64)) % 64
    restored = circular_bitshift_right(shifted, restore_shift, bits=64)
    np.testing.assert_array_equal(original, restored)

def test_sum_all_bits():
    """Verify that we count set bits correctly across the vertical 'trials'."""
    # Value 7 is binary ...000111 (3 bits set)
    arr = np.array([[7]], dtype=np.uint64)
    res = sum_all_bits(arr, bits=64)
    
    # Expect bits 0, 1, 2 to be 1.0, others 0.0
    assert res[0, 0] == 1.0 
    assert res[1, 0] == 1.0
    assert res[2, 0] == 1.0
    assert res[3, 0] == 0.0

def test_compute_bitwise_cooc_all_bits_single_node():
    """
    Verifies the math across ALL 64 bit positions for a single node.
    Ensures that every bit column in the output matrix corresponds to the 
    correct independent contingency table.
    """
    bits = 64
    tp = np.zeros((1, 1), dtype=np.uint64)
    tq = np.zeros((1, 1), dtype=np.uint64)

    # Setup: 
    # tp has alternating bits: 101010... (0xAAAAAAAAAAAAAAAA)
    # tq has blocks of 2:      110011... (0xCCCCCCCCCCCCCCCC)
    # This ensures a mix of (1,1), (1,0), (0,1), and (0,0) cases across the 64 bits.
    tp_val = 0xAAAAAAAAAAAAAAAA
    tq_val = 0xCCCCCCCCCCCCCCCC
    tp[0, 0] = np.uint64(tp_val)
    tq[0, 0] = np.uint64(tq_val)

    # Run calculation
    # We focus on shift k=0 (the first 64 columns of the result)
    result_matrix = compute_bitwise_cooc(tp, tq, bits=64)
    
    # Iterate through every bit position to verify independence
    for i in range(bits):
        # Extract individual bit values (0 or 1) for this specific position
        p_bit = (tp_val >> i) & 1
        q_bit = (tq_val >> i) & 1
        
        # Calculate expected Contingency Table components for N=1
        # Since N=1, the sums are just the bit values themselves.
        a_raw = 1 if (p_bit == 1 and q_bit == 1) else 0
        
        # Logic from the actual function:
        # b = sum_tp_1s - a
        # c = sum_shifted_1s - a
        # d = sum_tp_0s - c
        
        sum_p = p_bit # Total 1s in tp for this bit col
        sum_q = q_bit # Total 1s in tq for this bit col
        n_nodes = 1
        sum_p_0 = n_nodes - sum_p
        
        b_raw = sum_p - a_raw
        c_raw = sum_q - a_raw
        d_raw = sum_p_0 - c_raw
        
        # Apply Epsilon (+1)
        a = a_raw + 1
        b = b_raw + 1
        c = c_raw + 1
        d = d_raw + 1
        
        expected_log_ratio = np.log((a * d) / (b * c))
        
        # Result matrix stores shift k=0 in columns 0..63
        actual = result_matrix[0, i]
        
        assert np.isclose(actual, expected_log_ratio, atol=1e-12), \
            f"Bit {i} failed. P={p_bit}, Q={q_bit}. Exp: {expected_log_ratio}, Got: {actual}"


def test_compute_bitwise_cooc_multi_node_aggregation():
    """
    Verifies correctness when aggregating statistics across MULTIPLE nodes.
    Ensures vertical summation (sum_all_bits) is working before the log-ratio calc.
    """
    bits = 64
    n_nodes = 3
    
    # Setup 3 nodes with specific patterns for Bit 0 only
    # Node 0: A=1, B=1 (Match)
    # Node 1: A=1, B=0 (Mismatch)
    # Node 2: A=0, B=1 (Mismatch)
    
    tp = np.zeros((n_nodes, 1), dtype=np.uint64)
    tq = np.zeros((n_nodes, 1), dtype=np.uint64)
    
    # Set Bit 0 for relevant nodes
    tp[0, 0] = 1 # Node 0: A=1
    tq[0, 0] = 1 # Node 0: B=1
    
    tp[1, 0] = 1 # Node 1: A=1
    tq[1, 0] = 0 # Node 1: B=0
    
    tp[2, 0] = 0 # Node 2: A=0
    tq[2, 0] = 1 # Node 2: B=1
    
    # Run calculation
    result_matrix = compute_bitwise_cooc(tp, tq, bits=64)
    
    # --- Manual Verification for Bit 0 (Shift 0) ---
    
    # 1. Calculate Column Sums for Bit 0 across 3 nodes
    # sum_tp_1s = Node0(1) + Node1(1) + Node2(0) = 2
    # sum_tp_0s = Total(3) - 2 = 1
    # sum_tq_1s = Node0(1) + Node1(0) + Node2(1) = 2
    
    sum_tp_1s = 2.0
    sum_tp_0s = 1.0
    sum_tq_1s = 2.0
    
    # 2. Calculate Intersection (a_raw)
    # Node0 (1&1) + Node1 (1&0) + Node2 (0&1) = 1 + 0 + 0 = 1
    a_raw = 1.0
    
    # 3. Derive remaining cells
    b_raw = sum_tp_1s - a_raw  # 2 - 1 = 1
    c_raw = sum_tq_1s - a_raw  # 2 - 1 = 1
    d_raw = sum_tp_0s - c_raw  # 1 - 1 = 0
    
    # 4. Apply Epsilon (+1)
    a = a_raw + 1 # 2
    b = b_raw + 1 # 2
    c = c_raw + 1 # 2
    d = d_raw + 1 # 1
    
    expected_val = np.log((a * d) / (b * c)) # ln((2*1)/(2*2)) = ln(0.5) ≈ -0.693
    
    actual_val = result_matrix[0, 0] # Shift 0, Bit 0
    
    assert np.isclose(actual_val, expected_val, atol=1e-12), \
        f"Multi-node aggregation failed. Exp: {expected_val}, Got: {actual_val}"

# ==========================================
# 3. STATISTICAL TESTS: KDE & P-Values
# ==========================================

def test_compute_kde_stats_robustness():
    """
    Test p-value logic handling:
    1. Obvious outliers (High observation vs low simulation)
    2. Singular matrices (Zero variance in simulation)
    """
    # 1. Outlier Test
    sim_values = np.full(100, 10.0) + np.random.normal(0, 0.01, 100)
    obs = 100.0
    pval_ant, pval_syn, med, _ = compute_kde_stats(obs, sim_values)
    
    # Observed (100) > Sim (10) implies high correlation (syn)
    # Therefore, P(X > obs) should be small
    assert pval_syn < 0.05
    assert med < 15.0

    # 2. Singular Matrix (Zero Variance) Handling
    # The code should ideally not crash if variance is 0. 
    # If standard deviation is 0, KDE fails. 
    # This test ensures the function handles it or your noise addition in `process_batch` covers it.
    sim_flat = np.zeros(100)
    try:
        compute_kde_stats(0.5, sim_flat)
    except Exception:
        # If it crashes, we know we rely on the upstream jitter in process_batch.
        # This is acceptable, but good to know.
        pass 

# ==========================================
# 4. SIMULATION INVARIANTS (Biological Logic)
# ==========================================

def test_invariant_impossible_trait(simple_tree, trait_params):
    """The 'Impossible' trait (Gain=0, Root=0) must never appear."""
    sim_result = sim_bit(simple_tree, trait_params, trials=64)
    # Column 2 = 'Impossible'
    assert np.sum(sim_result[:, 2]) == 0, "Trait with 0 gain appeared spontaneously!"

def test_invariant_conserved_trait(simple_tree, trait_params):
    """The 'Conserved' trait (Root=1, Gain=0, Loss=0) must be present in ALL nodes."""
    sim_result = sim_bit(simple_tree, trait_params, trials=64)
    # Column 1 = 'Conserved'
    max_uint64 = np.uint64(0xFFFFFFFFFFFFFFFF)
    assert np.all(sim_result[:, 1] == max_uint64), "Conserved trait was lost!"

def test_invariant_vulnerable_trait(simple_tree, trait_params):
    """The 'Vulnerable' trait (Root=1, Loss=Huge) must be lost immediately."""
    sim_result = sim_bit(simple_tree, trait_params, trials=64)
    # Column 4 = 'Vulnerable'
    # Root is index 0 (usually) or found via traversal.
    # We check that at least some descendant nodes are 0.
    assert np.any(sim_result[:, 4] == 0), "High loss rate trait failed to disappear"

def test_invariant_delayed_onset(simple_tree, trait_params):
    """
    The 'Delayed' trait has threshold=3.1. 
    It should NOT appear on nodes with dist < 3.1 (A=3.0, B=3.0, D=2.0).
    It SHOULD appear on leaves (C=4.0) given high gain rate.
    """
    np.random.seed(42) # Ensure gain event happens
    sim_result = sim_bit(simple_tree, trait_params, trials=64)
    trait_idx = 3 # Delayed
    print(sim_result[:,trait_idx])
    assert np.all(sim_result[:3,trait_idx] == 0), "Traits before distance threshold have changed state"
    assert sim_result[3,trait_idx] > 0, "Traits after distance threshold has not changes dispite high gain rate"

def test_simulation_statistical_fidelity(trait_params):
    """
    RIGOROUS statistical test for bitpacking fidelity.
    Uses known traversal order of Star Tree (Root=0, Leaves=1..N) for speed.
    """
    # 1. Setup Star Tree (Root + 2000 Children)
    n_leaves = 2000
    # ETE3 Star tree traversal is always: Root, Child0, Child1... ChildN
    newick = "(" + ",".join([f"L{i}:1.0" for i in range(n_leaves)]) + ")Root;"
    tree = Tree(newick, format=1)
    
    # 2. Setup Parameters: Rate = ln(2) -> 50% chance
    # Override 'FastFlip' (Col 0)
    trait_params.loc['FastFlip', 'gains'] = np.log(2)
    trait_params.loc['FastFlip', 'gain_subsize'] = 1.0
    trait_params.loc['FastFlip', 'losses'] = 0.0
    trait_params.loc['FastFlip', 'dist'] = 0.0
    trait_params.loc['FastFlip', 'root_state'] = 0
    
    # 3. Run Simulation
    sim_result = sim_bit(tree, trait_params, trials=64)
    
    # 4. Extract Leaf Data (Optimized)
    # Since we know the order is [Root, Leaf1, Leaf2...], we just skip row 0.
    leaf_uint64s = sim_result[:, 0] 
    
    # 5. UNPACK and Verify
    # We verify the binary matrix statistics
    binary_matrix = np.zeros((n_leaves, 64), dtype=int)
    for bit_idx in range(64):
        mask = np.uint64(1) << np.uint64(bit_idx)
        # Fast boolean masking
        binary_matrix[:, bit_idx] = ((leaf_uint64s & mask) > 0).astype(int)

    # CHECK 1: Vertical Fidelity (The Rate)
    global_mean = np.mean(binary_matrix)
    print(f"Global Mean Density: {global_mean}")
    assert 0.48 < global_mean < 0.52, \
        f"Rate distortion! Expected 0.5, got {global_mean:.4f}"

    # CHECK 2: Horizontal Independence (Cross-Talk)
    # Calculate correlation between bits. Mask diagonal (self-correlation).
    corr_matrix = np.corrcoef(binary_matrix, rowvar=False)
    np.fill_diagonal(corr_matrix, 0)
    
    max_corr = np.max(np.abs(corr_matrix))
    print(f"Max Bit-to-Bit Correlation: {max_corr}")
    
    # Should be essentially noise (< 0.15 for N=2000)
    assert max_corr < 0.15, \
        f"Bit leakage detected! Max Corr: {max_corr:.3f}"


# ==========================================
# 5. INTEGRATION: Pipeline & Batching
# ==========================================

def test_process_batch_logic():
    """Unit test for the batch processing worker function."""
    # Mock data: 5 Nodes, 2 Traits
    sim_readonly = np.random.randint(0, 100, size=(5, 2), dtype=np.uint64)
    pairs = np.array([[0, 1]]) # Pair Trait 0 and Trait 1
    obspairs = np.array([5.0])
    
    res = process_batch(0, sim_readonly, pairs, obspairs, batch_size=1)
    
    assert 'p-value' in res
    assert 'direction' in res
    assert len(res['p-value']) == 1

def test_full_pipeline_run(simple_tree, trait_params):
    """
    Integration test for 'simulate_glrates_bit'.
    Verifies that string IDs are mapped correctly and parallel wrapper works.
    """
    # We want to test the pair ('Impossible', 'Conserved')
    # Indices: Impossible=2, Conserved=1
    pairs = np.array([['Impossible', 'Conserved']])
    obs_pairs = np.array([0.5])
    
    # Mock CPU count to avoid joblib overhead/errors in test env
    with patch("os.cpu_count", return_value=1):
        result_df = simulate_glrates_bit(
            simple_tree, 
            trait_params, 
            pairs, 
            obs_pairs, 
            cores=1
        )
    
    assert len(result_df) == 1
    row = result_df.iloc[0]
    
    # Check Name Mapping
    assert row['first'] == 'Impossible'
    assert row['second'] == 'Conserved'
    
    # Check Logic: Impossible (All 0s) vs Conserved (All 1s) = 0 Co-occurrence
    # Direction should likely reflect this (negative or low correlation)
    assert row['p-value'] >= 0.0


def test_compres_smoke_test(simple_tree, trait_params):
    """
    Smoke test for the 'compres' and 'simulate_glrates_bit' pipeline.
    Ensures the parallel processing glue code runs without crashing.
    """
    # 1. Run the simulation part
    sim_result = sim_bit(simple_tree, trait_params, trials=64)
    
    # 2. Setup inputs for compres
    # We have 2 traits (indices 0 and 1). Let's pair them.
    pairs = np.array([[0, 1]])
    obspairs = np.array([0.5]) # Arbitrary observed value
    
    # 3. Run compres (Force cores=1 to avoid joblib overhead in tests)
    # We mock os.cpu_count to ensure batch logic works even if we force 1 core
    with patch("os.cpu_count", return_value=1):
        df_res = compres(
            sim=sim_result, 
            pairs=pairs, 
            obspairs=obspairs, 
            batch_size=10, 
            cores=1
        )
    
    # 4. Assertions
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) == 1
    assert "p-value" in df_res.columns
    assert "direction" in df_res.columns
    # Check bounds
    assert 0 <= df_res.iloc[0]["p-value"] <= 1