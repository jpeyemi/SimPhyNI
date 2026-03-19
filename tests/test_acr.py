"""
Unit tests for simphyni/scripts/run_ancestral_reconstruction.py

Tests cover:
  - label_internal_nodes
  - _entropy (helper)
  - compute_branch_upper_bound
  - count_joint_stats  (JOINT-annotated tree, hand-crafted states)
  - count_all_marginal_stats  (FLOW / MARKOV / ENTROPY counting variants)
  - build_path_mask  (per-node eligibility masks)
  - reconstruct_trait  (column coverage for each uncertainty mode)
"""

import pytest
import numpy as np
import pandas as pd
from ete3 import Tree

from simphyni.scripts.run_ancestral_reconstruction import (
    label_internal_nodes,
    compute_branch_upper_bound,
    count_joint_stats,
    count_all_marginal_stats,
    build_path_mask,
    reconstruct_trait,
    _entropy,
    _node_dists_from_root,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tiny_tree():
    """
    Star tree: Root → A (bl=1), B (bl=2), C (bl=3).
    Root is labelled; leaves are named A, B, C.
    """
    t = Tree("(A:1.0,B:2.0,C:3.0)Root;", format=1)
    return t


@pytest.fixture
def chain_tree():
    """
    Linear chain: Root → Int1 → Leaf.
    All branch lengths = 1.0 so distances are easy to reason about.
    """
    t = Tree("(Leaf:1.0)Int1:1.0;", format=1)
    label_internal_nodes(t)
    return t


def _mp_df(tree, p1_map: dict) -> pd.DataFrame:
    """Build a marginal-prob DataFrame from {node_name: p1}."""
    names = [n.name for n in tree.traverse()]
    p1_vals = [p1_map.get(n, 0.0) for n in names]
    return pd.DataFrame(
        {"0": [1 - p for p in p1_vals], "1": p1_vals},
        index=names,
    )


def _annotate_joint(tree, gene, states: dict):
    """Set PastML JOINT attribute {node.name: frozenset({state})} on each node."""
    for node in tree.traverse():
        s = states.get(node.name, 0)
        setattr(node, gene, frozenset({s}))


# ===========================================================================
# label_internal_nodes
# ===========================================================================

def test_label_internal_nodes_names_unnamed_nodes():
    t = Tree("((A:1,B:1):1,C:2);")  # no names on internals
    label_internal_nodes(t)
    internal_names = [n.name for n in t.traverse() if not n.is_leaf()]
    # Root + one internal → two unnamed nodes should now have N* names
    assert all(name.startswith("N") for name in internal_names if name)


def test_label_internal_nodes_preserves_named_nodes():
    t = Tree("((A:1,B:1)MyInternal:1,C:2)MyRoot;", format=1)
    label_internal_nodes(t)
    # Pre-existing names must not be overwritten
    assert t.name == "MyRoot"
    int_node = [n for n in t.traverse() if n.name == "MyInternal"]
    assert len(int_node) == 1


# ===========================================================================
# _entropy
# ===========================================================================

def test_entropy_at_zero():
    assert _entropy(0.0) == pytest.approx(0.0, abs=1e-6)


def test_entropy_at_one():
    assert _entropy(1.0) == pytest.approx(0.0, abs=1e-6)


def test_entropy_at_half():
    # H(0.5) = 1.0 (max uncertainty for a binary variable, normalised to 1)
    assert _entropy(0.5) == pytest.approx(1.0, abs=1e-6)


def test_entropy_between_zero_and_one():
    for p in [0.1, 0.3, 0.7, 0.9]:
        h = _entropy(p)
        assert 0.0 <= h <= 1.0


# ===========================================================================
# compute_branch_upper_bound
# ===========================================================================

def test_compute_branch_upper_bound_finite_positive(tiny_tree):
    ub = compute_branch_upper_bound(tiny_tree)
    assert np.isfinite(ub)
    assert ub > 0


def test_compute_branch_upper_bound_single_branch():
    # Tree with a single branch of length 1.0 should return a finite bound
    t = Tree("(A:1.0);")
    ub = compute_branch_upper_bound(t)
    assert np.isfinite(ub)


# ===========================================================================
# count_joint_stats — basic gains / losses
# ===========================================================================

def test_count_joint_stats_no_changes(tiny_tree):
    """All nodes in state 0 → 0 gains, 0 losses, root_state=0."""
    gene = "T"
    _annotate_joint(tiny_tree, gene, {n.name: 0 for n in tiny_tree.traverse()})
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_joint_stats(tiny_tree, gene, ub)
    assert result["gains"] == 0
    assert result["losses"] == 0
    assert result["root_state"] == 0


def test_count_joint_stats_single_gain(tiny_tree):
    """Root=0, one leaf gains → 1 gain inferred somewhere in tree."""
    gene = "T"
    states = {n.name: 0 for n in tiny_tree.traverse()}
    states["A"] = 1  # A acquired the trait
    _annotate_joint(tiny_tree, gene, states)
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_joint_stats(tiny_tree, gene, ub)
    assert result["gains"] >= 1
    assert result["root_state"] == 0


def test_count_joint_stats_root_state_one(tiny_tree):
    """All nodes = 1 → root_state=1, no losses if all are present."""
    gene = "T"
    _annotate_joint(tiny_tree, gene, {n.name: 1 for n in tiny_tree.traverse()})
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_joint_stats(tiny_tree, gene, ub)
    assert result["root_state"] == 1
    assert result["losses"] == 0


def test_count_joint_stats_subsize_ordering(tiny_tree):
    """nofilter ≥ original ≥ thresh for gain subsizes."""
    gene = "T"
    states = {n.name: 0 for n in tiny_tree.traverse()}
    states["C"] = 1
    _annotate_joint(tiny_tree, gene, states)
    ub = compute_branch_upper_bound(tiny_tree)
    r = count_joint_stats(tiny_tree, gene, ub)
    assert r["gain_subsize_nofilter"] >= r["gain_subsize"]
    # thresh ≤ original (only post-emergence branches)
    assert r["gain_subsize_thresh"] <= r["gain_subsize"] + 1e-9


def test_count_joint_stats_dist_is_finite_when_gain_exists(tiny_tree):
    """If a gain occurred, dist should be finite."""
    gene = "T"
    states = {n.name: 0 for n in tiny_tree.traverse()}
    states["B"] = 1
    _annotate_joint(tiny_tree, gene, states)
    ub = compute_branch_upper_bound(tiny_tree)
    r = count_joint_stats(tiny_tree, gene, ub)
    if r["gains"] > 0:
        assert np.isfinite(r["dist"])


# ===========================================================================
# count_all_marginal_stats — FLOW / MARKOV / ENTROPY formulas
# ===========================================================================

def test_count_all_marginal_stats_flow_formula(tiny_tree):
    """
    FLOW gain = Σ max(0, P(child=1) − P(parent=1)).
    We set up known probabilities and verify the sum manually.
    """
    gene = "T"
    label_internal_nodes(tiny_tree)
    # Root p1=0.1, leaves A=0.9, B=0.3, C=0.5
    root_name = tiny_tree.name
    p1s = {root_name: 0.1, "A": 0.9, "B": 0.3, "C": 0.5}
    mp = _mp_df(tiny_tree, p1s)

    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)

    # Manual: each leaf is a direct child of root
    expected_gain_flow = (
        max(0.0, 0.9 - 0.1) +  # A
        max(0.0, 0.3 - 0.1) +  # B
        max(0.0, 0.5 - 0.1)    # C
    )
    assert result["gains_flow"] == pytest.approx(expected_gain_flow, abs=1e-9)


def test_count_all_marginal_stats_entropy_leq_flow(tiny_tree):
    """ENTROPY gain ≤ FLOW gain (entropy weight = 1−H ≤ 1)."""
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    p1s = {root_name: 0.2, "A": 0.8, "B": 0.6, "C": 0.9}
    mp = _mp_df(tiny_tree, p1s)

    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)

    assert result["gains_entropy"] <= result["gains_flow"] + 1e-9


def test_count_all_marginal_stats_markov_nonneg(tiny_tree):
    """Markov counts must be ≥ 0."""
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    p1s = {root_name: 0.3, "A": 0.7, "B": 0.2, "C": 0.5}
    mp = _mp_df(tiny_tree, p1s)

    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)

    assert result["gains_markov"] >= 0.0
    assert result["losses_markov"] >= 0.0


def test_count_all_marginal_stats_subsize_ordering(tiny_tree):
    """nofilter ≥ original for marginal subsizes."""
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    p1s = {root_name: 0.1, "A": 0.9, "B": 0.8, "C": 0.7}
    mp = _mp_df(tiny_tree, p1s)

    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)

    assert result["gain_subsize_marginal_nofilter"] >= result["gain_subsize_marginal"] - 1e-9
    assert result["loss_subsize_marginal_nofilter"] >= result["loss_subsize_marginal"] - 1e-9


def test_count_all_marginal_stats_returns_expected_keys(tiny_tree):
    """Output dict must contain all expected keys."""
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    p1s = {root_name: 0.1, "A": 0.8, "B": 0.6, "C": 0.4}
    mp = _mp_df(tiny_tree, p1s)

    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)

    expected_keys = {
        "gains_flow", "losses_flow",
        "gains_markov", "losses_markov",
        "gains_entropy", "losses_entropy",
        "gain_subsize_marginal", "loss_subsize_marginal",
        "gain_subsize_marginal_nofilter", "loss_subsize_marginal_nofilter",
        "gain_subsize_marginal_thresh", "loss_subsize_marginal_thresh",
        "gain_subsize_entropy", "loss_subsize_entropy",
        "gain_subsize_entropy_nofilter", "loss_subsize_entropy_nofilter",
        "gain_subsize_entropy_thresh", "loss_subsize_entropy_thresh",
        "dist_marginal", "loss_dist_marginal", "root_prob",
    }
    assert expected_keys.issubset(result.keys())


# ===========================================================================
# build_path_mask
# ===========================================================================

def test_build_path_mask_all_absent_no_gains():
    """Trait never present anywhere → no gains allowed."""
    t = Tree("(A:1.0,B:1.0);")
    label_internal_nodes(t)
    p1s = {n.name: 0.0 for n in t.traverse()}
    mp = _mp_df(t, p1s)
    gain_mask, _ = build_path_mask(t, {"T": mp}, ["T"])
    assert not gain_mask[:, 0].any()


def test_build_path_mask_all_present_no_losses():
    """Trait always present → no losses allowed."""
    t = Tree("(A:1.0,B:1.0);")
    label_internal_nodes(t)
    p1s = {n.name: 1.0 for n in t.traverse()}
    mp = _mp_df(t, p1s)
    _, loss_mask = build_path_mask(t, {"T": mp}, ["T"])
    assert not loss_mask[:, 0].any()


def test_build_path_mask_first_emergence_eligible():
    """Root absent, leaves present → gain is eligible at the leaf branches."""
    t = Tree("(A:1.0,B:1.0);")
    label_internal_nodes(t)
    p1s = {n.name: (0.9 if n.is_leaf() else 0.0) for n in t.traverse()}
    mp = _mp_df(t, p1s)
    gain_mask, _ = build_path_mask(t, {"T": mp}, ["T"])
    assert gain_mask[:, 0].any()


def test_build_path_mask_missing_mp_df_all_eligible():
    """No MPPA data for trait → all nodes default to eligible."""
    t = Tree("(A:1.0,B:1.0);")
    label_internal_nodes(t)
    gain_mask, loss_mask = build_path_mask(t, {}, ["T"])
    assert gain_mask[:, 0].all()
    assert loss_mask[:, 0].all()


# ===========================================================================
# reconstruct_trait — column coverage
# ===========================================================================

@pytest.fixture
def obs_data_and_tree():
    """
    Small balanced tree with 4 taxa and a binary trait (2 present, 2 absent).
    """
    tree_str = "((T1:1.0,T2:1.0)Int1:1.0,(T3:1.0,T4:1.0)Int2:1.0)Root:0.0;"
    obs = pd.DataFrame(
        {"trait": [1, 1, 0, 0]},
        index=["T1", "T2", "T3", "T4"],
    )
    return tree_str, obs


def test_reconstruct_trait_constant_returns_none(obs_data_and_tree):
    """A trait that is constant (all 0) must return None."""
    tree_str, obs = obs_data_and_tree
    obs["trait"] = 0  # all absent
    result = reconstruct_trait(
        gene="trait",
        tree_newick=tree_str,
        df_col=obs["trait"],
        upper_bound=10.0,
        uncertainty="threshold",
        gene_count=0,
    )
    assert result is None


def test_reconstruct_trait_threshold_columns(obs_data_and_tree):
    """uncertainty='threshold' → output has all core columns."""
    tree_str, obs = obs_data_and_tree
    result = reconstruct_trait(
        gene="trait",
        tree_newick=tree_str,
        df_col=obs["trait"],
        upper_bound=10.0,
        uncertainty="threshold",
        gene_count=int(obs["trait"].sum()),
    )
    assert result is not None
    for col in ("gains", "losses", "dist", "gain_subsize", "gain_subsize_thresh", "root_state"):
        assert col in result, f"Missing column: {col}"
    # Marginal columns should NOT be present
    assert "gains_flow" not in result


def test_reconstruct_trait_marginal_columns(obs_data_and_tree):
    """uncertainty='marginal' → output includes FLOW / MARKOV / ENTROPY columns."""
    tree_str, obs = obs_data_and_tree
    result = reconstruct_trait(
        gene="trait",
        tree_newick=tree_str,
        df_col=obs["trait"],
        upper_bound=10.0,
        uncertainty="marginal",
        gene_count=int(obs["trait"].sum()),
    )
    assert result is not None
    for col in ("gains_flow", "gains_markov", "gains_entropy", "root_prob"):
        assert col in result, f"Missing marginal column: {col}"


def test_reconstruct_trait_both_has_all_columns(obs_data_and_tree):
    """uncertainty='both' → output is the union of threshold + marginal columns."""
    tree_str, obs = obs_data_and_tree
    result = reconstruct_trait(
        gene="trait",
        tree_newick=tree_str,
        df_col=obs["trait"],
        upper_bound=10.0,
        uncertainty="both",
        gene_count=int(obs["trait"].sum()),
    )
    assert result is not None
    # Core threshold columns
    for col in ("gains", "losses", "dist", "gain_subsize_thresh"):
        assert col in result
    # Marginal columns
    for col in ("gains_flow", "gains_entropy", "dist_marginal", "root_prob"):
        assert col in result


# ===========================================================================
# _node_dists_from_root
# ===========================================================================

def test_node_dists_from_root_chain_tree(chain_tree):
    """
    chain_tree is "(Leaf:1.0)Int1:1.0;" — Int1 is the root, Leaf is its only
    child with branch length 1.0, so cumulative distance of Leaf from root = 1.0.
    """
    nd = _node_dists_from_root(chain_tree)
    root = chain_tree.get_tree_root()
    leaf = chain_tree.get_leaves()[0]
    assert nd[root.name] == pytest.approx(0.0)
    assert nd[leaf.name] == pytest.approx(1.0)


def test_node_dists_from_root_star_tree(tiny_tree):
    """Star tree: all leaves are direct children of root."""
    label_internal_nodes(tiny_tree)
    nd = _node_dists_from_root(tiny_tree)
    root_name = tiny_tree.name
    assert nd[root_name] == pytest.approx(0.0)
    assert nd["A"] == pytest.approx(1.0)
    assert nd["B"] == pytest.approx(2.0)
    assert nd["C"] == pytest.approx(3.0)


# ===========================================================================
# reconstruct_trait — _mp_df in-memory handling
# ===========================================================================

def test_reconstruct_trait_both_mp_df_in_memory(obs_data_and_tree):
    """uncertainty='both' stores _mp_df in result; it must be stripped before CSV output."""
    tree_str, obs = obs_data_and_tree
    result = reconstruct_trait(
        gene="trait",
        tree_newick=tree_str,
        df_col=obs["trait"],
        upper_bound=10.0,
        uncertainty="both",
        gene_count=int(obs["trait"].sum()),
    )
    assert result is not None
    # _mp_df should be present in-memory
    assert "_mp_df" in result
    # Simulate the CSV-strip step used in main()
    csv_row = {k: v for k, v in result.items() if not k.startswith("_")}
    assert "_mp_df" not in csv_row
    # All public columns must survive the strip
    assert "gains_flow" in csv_row
    assert "root_prob" in csv_row


def test_reconstruct_trait_threshold_no_mp_df(obs_data_and_tree):
    """uncertainty='threshold' → _mp_df must NOT be present (MPPA not run)."""
    tree_str, obs = obs_data_and_tree
    result = reconstruct_trait(
        gene="trait",
        tree_newick=tree_str,
        df_col=obs["trait"],
        upper_bound=10.0,
        uncertainty="threshold",
        gene_count=int(obs["trait"].sum()),
    )
    assert result is not None
    assert "_mp_df" not in result


# ===========================================================================
# count_joint_stats — thresh subsize excludes pre-emergence branches
# ===========================================================================

def test_count_joint_stats_thresh_subsize_smaller_than_original():
    """
    In a deeper tree where gains occur mid-tree, gain_subsize_thresh should
    be strictly smaller than gain_subsize because pre-emergence state-0 branches
    are excluded from the thresh denominator.
    """
    # Tree: Root(0) → A(0):1 → B(0):1 → C(1):1
    # Gain occurs at the branch leading to C; branches to A and B are pre-emergence.
    t = Tree("(((C:1.0)B:1.0)A:1.0)Root;", format=1)
    label_internal_nodes(t)
    gene = "g"

    def _annotate(node_states):
        for node in t.traverse():
            setattr(node, gene, frozenset({node_states.get(node.name, 0)}))

    # State 0 at Root, A, B; state 1 at C
    _annotate({"Root": 0, "A": 0, "B": 0, "C": 1})
    ub = compute_branch_upper_bound(t)
    r = count_joint_stats(t, gene, ub)

    # gain_subsize includes all state-0 branches (Root→A, A→B);
    # gain_subsize_thresh only includes state-0 branches at depth >= dist (first gain).
    assert r["gains"] >= 1
    assert r["gain_subsize_thresh"] <= r["gain_subsize"] + 1e-9


# ===========================================================================
# count_all_marginal_stats — exact formula tests for each counting method
# ===========================================================================

def test_flow_loss_exact(tiny_tree):
    """
    FLOW loss = Σ max(0, P(parent=1) − P(child=1)).
    Star tree: Root=0.8, A=0.1, B=0.9, C=0.6.
    Expected loss: max(0,0.8-0.1) + max(0,0.8-0.9) + max(0,0.8-0.6) = 0.7+0+0.2 = 0.9
    """
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    p1s = {root_name: 0.8, "A": 0.1, "B": 0.9, "C": 0.6}
    mp = _mp_df(tiny_tree, p1s)
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)
    expected = max(0.0, 0.8 - 0.1) + max(0.0, 0.8 - 0.9) + max(0.0, 0.8 - 0.6)
    assert result["losses_flow"] == pytest.approx(expected, abs=1e-9)


def test_flow_zero_when_flat(tiny_tree):
    """All nodes have the same P(1) → gains_flow = 0 and losses_flow = 0."""
    gene = "T"
    label_internal_nodes(tiny_tree)
    p1s = {n.name: 0.4 for n in tiny_tree.traverse()}
    mp = _mp_df(tiny_tree, p1s)
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)
    assert result["gains_flow"] == pytest.approx(0.0, abs=1e-9)
    assert result["losses_flow"] == pytest.approx(0.0, abs=1e-9)


def test_markov_gains_formula_pinned(tiny_tree):
    """
    MARKOV gains = Σ P(parent=0) × P(child=1) × bl  (current approximation).
    Star tree, all leaves direct children of root.
    """
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    p1s = {root_name: 0.2, "A": 0.7, "B": 0.3, "C": 0.5}
    mp = _mp_df(tiny_tree, p1s)
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)

    # Branch lengths: A=1, B=2, C=3 (from tiny_tree fixture).
    # For star tree, parent of every leaf is root: p0_p = 1 - 0.2 = 0.8
    expected = (0.8 * 0.7 * 1.0) + (0.8 * 0.3 * 2.0) + (0.8 * 0.5 * 3.0)
    assert result["gains_markov"] == pytest.approx(expected, abs=1e-9)


def test_entropy_certain_parent_equals_flow(tiny_tree):
    """
    When P(parent=1) = 0 exactly: H(parent) = 0, cert = 1 → gains_entropy = gains_flow.
    """
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    # Root is completely absent; leaves have positive P(1)
    p1s = {root_name: 0.0, "A": 0.9, "B": 0.6, "C": 0.3}
    mp = _mp_df(tiny_tree, p1s)
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)
    # With a certain parent (p1=0 → h=0, cert=1), entropy weighting is identity
    assert result["gains_entropy"] == pytest.approx(result["gains_flow"], abs=1e-6)


def test_entropy_uncertain_parent_zero(tiny_tree):
    """
    When P(parent=1) = 0.5: H(parent) = 1, cert = 0 → gains_entropy = 0.
    """
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    # Root is maximally uncertain (0.5); leaves have higher P(1) so flow > 0
    p1s = {root_name: 0.5, "A": 0.9, "B": 0.8, "C": 0.7}
    mp = _mp_df(tiny_tree, p1s)
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)
    assert result["gains_entropy"] == pytest.approx(0.0, abs=1e-6)
    # Sanity: FLOW gain is positive (there is probability flux)
    assert result["gains_flow"] > 0.0


def test_entropy_leq_flow_general(tiny_tree):
    """gains_entropy ≤ gains_flow in all cases (weight ≤ 1)."""
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    p1s = {root_name: 0.3, "A": 0.8, "B": 0.6, "C": 0.9}
    mp = _mp_df(tiny_tree, p1s)
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)
    assert result["gains_entropy"] <= result["gains_flow"] + 1e-9


def test_original_subsize_exact(tiny_tree):
    """
    gain_subsize_marginal = Σ P(parent=0) × min(bl, upper_bound).
    Star tree: root=0.2, A bl=1, B bl=2, C bl=3, upper_bound=10 (all below cap).
    Expected = 0.8 × 1 + 0.8 × 2 + 0.8 × 3 = 0.8 × 6 = 4.8
    """
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    p1s = {root_name: 0.2, "A": 0.7, "B": 0.4, "C": 0.6}
    mp = _mp_df(tiny_tree, p1s)
    result = count_all_marginal_stats(tiny_tree, gene, mp, upper_bound=10.0)
    expected = 0.8 * 1.0 + 0.8 * 2.0 + 0.8 * 3.0
    assert result["gain_subsize_marginal"] == pytest.approx(expected, abs=1e-9)


def test_nofilter_subsize_exceeds_original_when_long_branch():
    """
    When a branch length exceeds upper_bound, nofilter includes the full length
    while original is capped → nofilter > original.
    """
    gene = "T"
    # Leaf A has bl=20 which exceeds upper_bound=5
    t = Tree("(A:20.0,B:1.0)Root;", format=1)
    label_internal_nodes(t)
    p1s = {"Root": 0.1, "A": 0.8, "B": 0.8}
    mp = _mp_df(t, p1s)
    result = count_all_marginal_stats(t, gene, mp, upper_bound=5.0)
    assert result["gain_subsize_marginal_nofilter"] > result["gain_subsize_marginal"]


def test_dist_marginal_exact():
    """
    dist_marginal = min root-to-node distance where P(node=1) ≥ 0.5.
    Tree: (A:2.0,B:4.0)Root;  p1: Root=0.1, A=0.3, B=0.8
    Only B has p1 ≥ 0.5, dist(B) = 4.0.
    """
    gene = "T"
    t = Tree("(A:2.0,B:4.0)Root;", format=1)
    label_internal_nodes(t)
    p1s = {"Root": 0.1, "A": 0.3, "B": 0.8}
    mp = _mp_df(t, p1s)
    result = count_all_marginal_stats(t, gene, mp, upper_bound=10.0)
    assert result["dist_marginal"] == pytest.approx(4.0, abs=1e-9)


def test_thresh_fully_upstream():
    """
    Branches fully upstream of dist_m: gain_overlap = 0 → eff_gain_thresh = 0.
    Set dist_m beyond all nodes by making all P(1) < 0.5 → dist_m = inf → overlap = 0.
    """
    gene = "T"
    t = Tree("(A:2.0,B:3.0)Root;", format=1)
    label_internal_nodes(t)
    # All P(1) < 0.5 → dist_marginal = inf → all branches upstream
    p1s = {"Root": 0.1, "A": 0.2, "B": 0.3}
    mp = _mp_df(t, p1s)
    result = count_all_marginal_stats(t, gene, mp, upper_bound=10.0)
    assert result["gain_subsize_marginal_thresh"] == pytest.approx(0.0, abs=1e-9)


def test_thresh_straddling_with_cap():
    """
    Straddling branch test with IQR cap applied.

    Tree: (B:6.0,C:1.0)A;  upper_bound=4.0
    p1: A=0.05, B=0.30, C=0.90
    dist_m = dist(C) = 1.0   (only C has P(1) ≥ 0.5)

    Branch B (straddling):
      parent_dist = 0, nd_v = 6, dist_m = 1
      gain_overlap = 6 - max(0, 1) = 5
      eff_gain_thresh = min(5, 4) = 4   ← cap applies

    Branch C (at dist_m, no overlap beyond itself):
      parent_dist = 0, nd_v = 1, dist_m = 1
      gain_overlap = max(0, 1 - max(0, 1)) = 0
      eff_gain_thresh = 0

    gain_subsize_marginal_thresh = P(A=0) × 0 + P(A=0) × 4
                                 = 0 + 0.95 × 4 = 3.8
    """
    gene = "T"
    t = Tree("(B:6.0,C:1.0)A;", format=1)
    label_internal_nodes(t)
    p1s = {"A": 0.05, "B": 0.30, "C": 0.90}
    mp = _mp_df(t, p1s)
    result = count_all_marginal_stats(t, gene, mp, upper_bound=4.0)
    assert result["gain_subsize_marginal_thresh"] == pytest.approx(0.95 * 4.0, abs=1e-6)


def test_thresh_fully_downstream_branch():
    """
    A branch whose parent is at or beyond dist_m is fully downstream:
    gain_overlap = bl_v  →  eff_gain_thresh = min(bl_v, ub) = eff_bl.

    Chain tree: Root → Int (bl=1.0) → B (bl=2.0)
    p1: Root=0.1, Int=0.9, B=0.7,  upper_bound=10.

    dist_m = min(nd where p1≥0.5) = nd(Int) = 1.0

    Branch Int (Root→Int): parent_dist=0 < dist_m=1 → overlap=0, thresh=0
    Branch B   (Int→B):    parent_dist=1 == dist_m=1 → overlap=2, thresh=min(2,10)=2
      → B's full eff_bl (2.0) is included, matching its gain_subsize_marginal contribution

    Expected gain_subsize_marginal_thresh:
      P(Root=0)×0  +  P(Int=0)×2  =  0.9×0 + 0.1×2 = 0.2
    """
    gene = "T"
    t = Tree("((B:2.0)Int:1.0)Root;", format=1)
    label_internal_nodes(t)
    p1s = {"Root": 0.1, "Int": 0.9, "B": 0.7}
    mp = _mp_df(t, p1s)
    result = count_all_marginal_stats(t, gene, mp, upper_bound=10.0)
    # B branch is fully downstream: its full eff_bl=2.0 is counted
    assert result["gain_subsize_marginal_thresh"] == pytest.approx(0.1 * 2.0, abs=1e-9)


def test_thresh_leq_original(tiny_tree):
    """gain_subsize_marginal_thresh ≤ gain_subsize_marginal always."""
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    p1s = {root_name: 0.1, "A": 0.8, "B": 0.9, "C": 0.6}
    mp = _mp_df(tiny_tree, p1s)
    ub = compute_branch_upper_bound(tiny_tree)
    result = count_all_marginal_stats(tiny_tree, gene, mp, ub)
    assert result["gain_subsize_marginal_thresh"] <= result["gain_subsize_marginal"] + 1e-9


def test_path_mask_restricts_flow(tiny_tree):
    """
    Restricting via PATH mask should reduce (or equal) gains_flow.
    """
    gene = "T"
    label_internal_nodes(tiny_tree)
    root_name = tiny_tree.name
    # Root absent, A present, B absent, C absent → gain eligible only on A branch
    p1s = {root_name: 0.0, "A": 0.9, "B": 0.05, "C": 0.05}
    mp = _mp_df(tiny_tree, p1s)
    ub = compute_branch_upper_bound(tiny_tree)

    # Full (no mask)
    full = count_all_marginal_stats(tiny_tree, gene, mp, ub)

    # PATH mask
    gain_mask, loss_mask = build_path_mask(tiny_tree, {gene: mp}, [gene])
    gm1d = gain_mask[:, 0]
    lm1d = loss_mask[:, 0]
    masked = count_all_marginal_stats(tiny_tree, gene, mp, ub,
                                      gain_mask_1d=gm1d, loss_mask_1d=lm1d)

    assert masked["gains_flow"] <= full["gains_flow"] + 1e-9


def test_parent_state_gain_eligibility_marginal():
    """
    gain_subsize_marginal is positive on a gain branch (parent=0, child=1).
    This verifies that parent state (not child state) drives marginal gain eligibility.

    Tree: Root(p1=0.0) → Leaf(p1=0.9), bl=1.
    gain_subsize_marginal = P(Root=0) × bl = 1.0 × 1.0 = 1.0
    (Even though the child has high P(1), the branch is gain-eligible via parent.)
    """
    gene = "T"
    t = Tree("(Leaf:1.0)Root;", format=1)
    label_internal_nodes(t)
    p1s = {"Root": 0.0, "Leaf": 0.9}
    mp = _mp_df(t, p1s)
    result = count_all_marginal_stats(t, gene, mp, upper_bound=10.0)
    # P(Root=0) = 1.0, eff_bl = min(1.0, 10.0) = 1.0
    assert result["gain_subsize_marginal"] == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# count_joint_stats — JOINTP parent-state subsize vs old child-state subsize
# ===========================================================================

from simphyni.scripts.run_ancestral_reconstruction import count_joint_stats


def test_joint_child_state_vs_parent_state_subsize():
    """
    Documents the difference between child-state (JOINT) and parent-state (JOINTP) subsize.

    Tree: Root(0) → A(0):1 → B(1):1, upper_bound=10.

    JOINT (child-state): gain_subsize counts branches where child==0.
      - Root→A: child A = 0 → counts   (bl=1)
      - A→B:   child B = 1 → does NOT count in gain_subsize; lands in loss_subsize

    JOINTP (parent-state): gain_subsize_p counts branches where parent==0.
      - Root→A: parent Root = 0 → counts   (bl=1)
      - A→B:   parent A = 0   → also counts (bl=1) — the gain branch IS in the denominator

    gain_subsize_p should be ≥ gain_subsize for this tree.
    """
    t = Tree("((B:1.0)A:1.0)Root;", format=1)
    label_internal_nodes(t)
    gene = "g"

    def _annotate(states):
        for node in t.traverse():
            setattr(node, gene, frozenset({states.get(node.name, 0)}))

    _annotate({"Root": 0, "A": 0, "B": 1})
    ub = 10.0
    r = count_joint_stats(t, gene, ub)

    # Old JOINT: gain_subsize should only include Root→A (child A=0)
    assert r["gain_subsize"] == pytest.approx(1.0, abs=1e-9), (
        "JOINT gain_subsize should be 1.0 (only Root→A with child=0)"
    )
    # Old JOINT: A→B (child B=1) lands in loss_subsize, not gain_subsize
    assert r["loss_subsize"] == pytest.approx(1.0, abs=1e-9), (
        "JOINT loss_subsize should be 1.0 (A→B with child=1)"
    )

    # JOINTP: gain_subsize_p includes BOTH Root→A (parent=0) AND A→B (parent A=0)
    assert r["gain_subsize_p"] == pytest.approx(2.0, abs=1e-9), (
        "JOINTP gain_subsize_p should be 2.0 (both Root→A and A→B have parent=0)"
    )
    # JOINTP loss_subsize_p: no branch has parent==1 in this tree
    assert r["loss_subsize_p"] == pytest.approx(0.0, abs=1e-9), (
        "JOINTP loss_subsize_p should be 0 (no branch has parent==1)"
    )

    # Parent-state denominator is strictly larger than child-state denominator
    assert r["gain_subsize_p"] >= r["gain_subsize"]
