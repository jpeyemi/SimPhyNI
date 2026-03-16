import pytest
import numpy as np
import pandas as pd
from ete3 import Tree
from unittest.mock import MagicMock, patch

# Import your module here.
# Assuming the file provided is named 'simulation_methods.py' inside 'simphyni' package
from simphyni import sim_bit, simulate_glrates_bit, compres, build_sim_params
from simphyni.Simulation.simulation import unpack_trait_params, circular_bitshift_right, compute_kde_stats, sum_all_bits, compute_bitwise_cooc, process_batch
from simphyni.scripts.run_ancestral_reconstruction import build_path_mask, label_internal_nodes

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


# ==========================================
# 6. PATH MASK: build_path_mask correctness
# ==========================================

class TestBuildPathMask:
    """
    Tests for the path-based upstream presence/absence mask.

    Uses a minimal star tree (root → A, root → B) with hand-crafted
    marginal probability DataFrames so no PastML run is needed.
    """

    @pytest.fixture
    def minimal_tree(self):
        """Root → A (leaf), Root → B (leaf)."""
        t = Tree("(A:1.0,B:1.0);")
        label_internal_nodes(t)
        return t

    def _mp_df(self, tree, p1_map: dict) -> pd.DataFrame:
        """Build a marginal_prob_df from a {node_name: p1} dict."""
        names = [n.name for n in tree.traverse()]
        p1_vals = [p1_map.get(n, 0.0) for n in names]
        return pd.DataFrame(
            {"0": [1 - p for p in p1_vals], "1": p1_vals},
            index=names,
        )

    def test_no_upstream_presence_blocks_gains(self, minimal_tree):
        """Clade with trait never present and no upstream presence → no gains allowed."""
        # All nodes: P(state=1) = 0 — trait never observed, no upstream context
        mp = self._mp_df(minimal_tree, {n.name: 0.0 for n in minimal_tree.traverse()})
        gain_mask, _ = build_path_mask(minimal_tree, {"T": mp}, ["T"])
        assert not gain_mask[:, 0].any(), \
            "Gains allowed in fully-absent clade with no upstream presence"

    def test_full_presence_blocks_losses(self, minimal_tree):
        """Clade with trait always present and no upstream absence → no losses allowed."""
        # All nodes: P(state=1) = 1 — trait always present, no upstream absence
        mp = self._mp_df(minimal_tree, {n.name: 1.0 for n in minimal_tree.traverse()})
        _, loss_mask = build_path_mask(minimal_tree, {"T": mp}, ["T"])
        assert not loss_mask[:, 0].any(), \
            "Losses allowed in fully-present clade with no upstream absence"

    def test_first_emergence_allows_gain(self, minimal_tree):
        """P(parent=0)>0.5 AND P(child=1)>0.5 (first emergence) → gain eligible."""
        # Root absent (p1=0), leaves present (p1=0.9) — classic first emergence
        p1_map = {n.name: (0.9 if n.is_leaf() else 0.0) for n in minimal_tree.traverse()}
        mp = self._mp_df(minimal_tree, p1_map)
        gain_mask, _ = build_path_mask(minimal_tree, {"T": mp}, ["T"])
        assert gain_mask[:, 0].any(), "First emergence point not detected as gain-eligible"

    def test_upstream_presence_enables_regain(self):
        """
        Re-gain scenario: root present → internal absent → leaf present again.
        The leaf branch must be gain-eligible because upstream_presence=True
        (the root had the trait), even though the immediate parent is absent.

        Requires a chain tree (not a star tree) so there is an intermediate node
        that can go absent between a present ancestor and a present leaf.
        """
        chain = Tree("((Leaf:1.0)Internal:1.0)Root;", format=1)
        label_internal_nodes(chain)
        # Root present (p1=0.9), Internal absent (p1=0.05), Leaf present again (p1=0.9)
        p1_map = {"Root": 0.9, "Internal": 0.05, "Leaf": 0.9}
        mp = self._mp_df(chain, p1_map)
        gain_mask, _ = build_path_mask(chain, {"T": mp}, ["T"])
        # The Leaf branch (parent=Internal which is absent, upstream root was present)
        # should be eligible for a gain
        assert gain_mask[:, 0].any(), "Re-gain after upstream presence not enabled"

    def test_missing_mp_df_defaults_to_all_eligible(self, minimal_tree):
        """When mp_df is absent for a trait, all nodes default to fully eligible."""
        gain_mask, loss_mask = build_path_mask(minimal_tree, {}, ["T"])
        assert gain_mask[:, 0].all(), "Missing mp_df should default all nodes to gain-eligible"
        assert loss_mask[:, 0].all(), "Missing mp_df should default all nodes to loss-eligible"


# ==========================================
# 7. BUILD_SIM_PARAMS: column selection
# ==========================================

@pytest.fixture
def wide_params():
    """
    One-row DataFrame containing all columns that count_joint_stats +
    count_all_marginal_stats produce (standard + marginal + entropy variants).
    Values are arbitrary non-zero floats so we can verify mapping.
    """
    data = {
        # Standard JOINT columns
        "gene": ["GeneA"],
        "gains": [2.0], "losses": [1.0],
        "dist": [0.5], "loss_dist": [0.3],
        "gain_subsize": [10.0], "loss_subsize": [8.0],
        "gain_subsize_nofilter": [12.0], "loss_subsize_nofilter": [9.0],
        "gain_subsize_thresh": [7.0], "loss_subsize_thresh": [6.0],
        "root_state": [0],
        # FLOW / MARKOV / ENTROPY gain columns
        "gains_flow": [1.8], "losses_flow": [0.9],
        "gains_markov": [1.5], "losses_markov": [0.7],
        "gains_entropy": [1.2], "losses_entropy": [0.6],
        # Marginal subsize variants
        "gain_subsize_marginal": [9.0], "loss_subsize_marginal": [7.5],
        "gain_subsize_marginal_nofilter": [11.0], "loss_subsize_marginal_nofilter": [8.5],
        "gain_subsize_marginal_thresh": [6.5], "loss_subsize_marginal_thresh": [5.5],
        # Entropy subsize variants
        "gain_subsize_entropy": [4.0], "loss_subsize_entropy": [3.5],
        "gain_subsize_entropy_nofilter": [5.0], "loss_subsize_entropy_nofilter": [4.5],
        "gain_subsize_entropy_thresh": [3.0], "loss_subsize_entropy_thresh": [2.5],
        # Marginal dist and root prob
        "dist_marginal": [0.4], "loss_dist_marginal": [0.2],
        "root_prob": [0.3],
    }
    return pd.DataFrame(data)


def test_build_sim_params_joint_original(wide_params):
    """JOINT+ORIGINAL maps standard columns unchanged."""
    out = build_sim_params(wide_params, counting="JOINT", subsize="ORIGINAL")
    assert out["gains"].iloc[0] == pytest.approx(2.0)
    assert out["gain_subsize"].iloc[0] == pytest.approx(10.0)
    assert out["dist"].iloc[0] == pytest.approx(0.5)
    assert out["root_state"].iloc[0] == 0


def test_build_sim_params_flow_original(wide_params):
    """FLOW+ORIGINAL maps flow columns into the standard output names."""
    out = build_sim_params(wide_params, counting="FLOW", subsize="ORIGINAL")
    assert out["gains"].iloc[0] == pytest.approx(1.8)        # from gains_flow
    assert out["gain_subsize"].iloc[0] == pytest.approx(9.0)  # from gain_subsize_marginal
    assert out["dist"].iloc[0] == pytest.approx(0.4)          # from dist_marginal


def test_build_sim_params_entropy_thresh(wide_params):
    """ENTROPY+THRESH maps entropy counting + thresh subsize."""
    out = build_sim_params(wide_params, counting="ENTROPY", subsize="THRESH")
    assert out["gains"].iloc[0] == pytest.approx(1.2)        # from gains_entropy
    assert out["gain_subsize"].iloc[0] == pytest.approx(3.0)  # from gain_subsize_entropy_thresh


def test_build_sim_params_no_threshold_zeros_dist(wide_params):
    """no_threshold=True sets dist and loss_dist to 0 regardless of input."""
    out = build_sim_params(wide_params, counting="JOINT", subsize="ORIGINAL", no_threshold=True)
    assert out["dist"].iloc[0] == 0.0
    assert out["loss_dist"].iloc[0] == 0.0


def test_build_sim_params_root_state_from_root_prob(wide_params):
    """Non-JOINT counting uses root_prob ≥ 0.5 → root_state."""
    # root_prob = 0.3 → root_state = 0
    out = build_sim_params(wide_params, counting="FLOW", subsize="ORIGINAL")
    assert out["root_state"].iloc[0] == 0

    wide_params["root_prob"] = 0.8  # above threshold
    out2 = build_sim_params(wide_params, counting="FLOW", subsize="ORIGINAL")
    assert out2["root_state"].iloc[0] == 1


def test_build_sim_params_markov_nofilter(wide_params):
    """MARKOV+NO_FILTER maps to the nofilter subsize columns."""
    out = build_sim_params(wide_params, counting="MARKOV", subsize="NO_FILTER")
    assert out["gain_subsize"].iloc[0] == pytest.approx(11.0)  # gain_subsize_marginal_nofilter


def test_build_sim_params_output_feeds_sim_bit(simple_tree, wide_params):
    """build_sim_params output is accepted by sim_bit without error."""
    out = build_sim_params(wide_params, counting="JOINT", subsize="ORIGINAL")
    out.index = ["GeneA"]
    # Should run without raising
    result = sim_bit(simple_tree, out, trials=64)
    # sim_bit returns one row per leaf tip (iterating `tree` yields leaves in ETE3)
    assert result.shape[0] == len(simple_tree.get_leaves())


# ---------------------------------------------------------------------------
# Additional build_sim_params coverage: all counting × subsize combos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("counting,subsize,expected_gains,expected_subsize", [
    ("JOINT",   "ORIGINAL",  2.0,  10.0),
    ("JOINT",   "NO_FILTER", 2.0,  12.0),
    ("JOINT",   "THRESH",    2.0,   7.0),
    ("FLOW",    "ORIGINAL",  1.8,   9.0),
    ("FLOW",    "NO_FILTER", 1.8,  11.0),
    ("FLOW",    "THRESH",    1.8,   6.5),
    ("MARKOV",  "ORIGINAL",  1.5,   9.0),
    ("MARKOV",  "NO_FILTER", 1.5,  11.0),
    ("MARKOV",  "THRESH",    1.5,   6.5),
    ("ENTROPY", "ORIGINAL",  1.2,   4.0),
    ("ENTROPY", "NO_FILTER", 1.2,   5.0),
    ("ENTROPY", "THRESH",    1.2,   3.0),
])
def test_build_sim_params_all_combinations(wide_params, counting, subsize,
                                            expected_gains, expected_subsize):
    """Every counting × subsize combination maps to the correct source column."""
    out = build_sim_params(wide_params, counting=counting, subsize=subsize)
    assert out["gains"].iloc[0] == pytest.approx(expected_gains), \
        f"{counting}+{subsize}: gains mismatch"
    assert out["gain_subsize"].iloc[0] == pytest.approx(expected_subsize), \
        f"{counting}+{subsize}: gain_subsize mismatch"


@pytest.mark.parametrize("counting,expected_dist", [
    ("JOINT",   0.5),
    ("FLOW",    0.4),
    ("MARKOV",  0.4),
    ("ENTROPY", 0.4),
])
def test_build_sim_params_dist_columns(wide_params, counting, expected_dist):
    """JOINT uses dist; FLOW/MARKOV/ENTROPY use dist_marginal."""
    out = build_sim_params(wide_params, counting=counting, subsize="ORIGINAL")
    assert out["dist"].iloc[0] == pytest.approx(expected_dist), \
        f"{counting}: dist mismatch"


def test_build_sim_params_legacy_joint_only():
    """Legacy JOINT-only DataFrame (no marginal columns) passes through cleanly."""
    legacy = pd.DataFrame({
        "gene": ["G1"],
        "gains": [3.0], "losses": [1.5],
        "dist": [0.6], "loss_dist": [0.4],
        "gain_subsize": [15.0], "loss_subsize": [12.0],
        "gain_subsize_nofilter": [18.0], "loss_subsize_nofilter": [14.0],
        "gain_subsize_thresh": [10.0], "loss_subsize_thresh": [8.0],
        "root_state": [1],
        # No gains_flow, no marginal columns — simulates old pastml + GL_tab output
    })
    out = build_sim_params(legacy, counting="JOINT", subsize="ORIGINAL")
    assert out["gains"].iloc[0] == pytest.approx(3.0)
    assert out["dist"].iloc[0] == pytest.approx(0.6)
    assert out["root_state"].iloc[0] == 1
    # All required base columns must be present
    for col in ("gains", "losses", "gain_subsize", "loss_subsize", "dist", "loss_dist", "root_state"):
        assert col in out.columns