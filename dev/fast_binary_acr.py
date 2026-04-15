"""
fast_binary_acr.py
==================
Numba-accelerated binary F81 ancestral character reconstruction.

Designed for the case that dominates SimPhyNI workloads: many thousands of
binary (0/1) traits on a single fixed tree.  The key insight is that the
tree topology and branch lengths are identical for every trait, so all
per-tree pre-computations (traversal orders, CSR children arrays, node
depths) are paid once and then reused across all traits in parallel.

Algorithm
---------
F81 model, binary states {0, 1}.

Transition probabilities on a branch of length t:
    mu     = 1 / (2 · π₀ · π₁)          (F81 normalisation)
    e      = exp(−mu · t · sf)            (sf = scaling factor, free parameter)
    α(t)   = π₁ · (1 − e)                P(0 → 1)
    β(t)   = π₀ · (1 − e)                P(1 → 0)
    P[i,j] = [[1−α, α], [β, 1−β]]

Both BU and TD passes are carried out fully in log-space (see log_sum_exp2)
so no rescaling bookkeeping is needed — numerically stable for any tree depth.

Rate optimisation modes
-----------------------
'empirical'   Fix π₁ = mean observed tip frequency; golden-section search
              for sf only (30 iterations → ~1e-6 relative precision).
              Fastest mode; typically within 1–2% of PastML accuracy.

'ml'          Alternating golden-section search over sf and π₁ (3 outer
              cycles × 2 × 30 inner iterations = ~180 BU passes per trait).
              Closely matches PastML's two-phase L-BFGS-B optimisation.

Both modes run with Numba @njit(parallel=True) across traits via prange.

Outputs
-------
fast_acr() returns a FastACRResult with:
  marginal_p1  : float64[n_traits, n_nodes] — P(state=1 | data) at every node
  joint_states : int8[n_traits, n_nodes]    — MAP state from JOINT traceback
  sf           : float64[n_traits]          — optimised scaling factors
  pi1          : float64[n_traits]          — used equilibrium frequencies
  log_lh       : float64[n_traits]          — final log-likelihoods

The marginal_p1 array can be sliced per-trait to build a PastML-compatible
marginal_probabilities DataFrame via marginal_probs_df().
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ete3 import Tree
from numba import njit, prange

# ---------------------------------------------------------------------------
# Tree array representation
# ---------------------------------------------------------------------------

@dataclass
class TreeArrays:
    """Flat NumPy representation of a phylogenetic tree for Numba kernels.

    All array fields are contiguous C-order NumPy arrays.  The children of
    each node are stored in Compressed Sparse Row (CSR) format so the inner
    loop over children is a simple slice of children_list.
    """
    n_nodes: int
    root_idx: int
    postorder: np.ndarray    # int32[n_nodes]  — bottom-up traversal order
    preorder: np.ndarray     # int32[n_nodes]  — top-down traversal order
    parent: np.ndarray       # int32[n_nodes], -1 for root
    children_ptr: np.ndarray # int32[n_nodes+1] CSR row pointers
    children_list: np.ndarray# int32[total_children] CSR column indices
    bl: np.ndarray           # float64[n_nodes] branch lengths (0 for root)
    nd: np.ndarray           # float64[n_nodes] cumulative dist from root
    is_leaf: np.ndarray      # bool[n_nodes]
    node_names: List[str]    # len = n_nodes
    name_to_idx: Dict[str, int]
    leaf_node_idx: np.ndarray# int32[n_leaves] — which node indices are leaves
    avg_bl: float            # mean non-zero branch length (for sf bounds)


def build_tree_arrays(tree: Tree) -> TreeArrays:
    """Convert an ETE3 tree to flat NumPy arrays suitable for Numba kernels.

    Internal nodes must already be labelled (run label_internal_nodes before
    calling this function if your tree has anonymous internal nodes).
    """
    nodes = list(tree.traverse())          # stable order
    n_nodes = len(nodes)
    node_to_i = {node: i for i, node in enumerate(nodes)}

    # parent
    parent = np.full(n_nodes, -1, dtype=np.int32)
    for node in nodes:
        if not node.is_root():
            parent[node_to_i[node]] = node_to_i[node.up]

    # CSR children
    n_ch = np.zeros(n_nodes, dtype=np.int32)
    for node in nodes:
        if not node.is_root():
            n_ch[node_to_i[node.up]] += 1
    children_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    children_ptr[1:] = np.cumsum(n_ch)
    children_list = np.empty(int(children_ptr[-1]), dtype=np.int32)
    fill = children_ptr[:-1].copy()
    for node in nodes:
        if not node.is_root():
            p = node_to_i[node.up]
            children_list[fill[p]] = node_to_i[node]
            fill[p] += 1

    # branch lengths
    bl = np.array([n.dist for n in nodes], dtype=np.float64)

    # node depths from root (preorder)
    nd = np.zeros(n_nodes, dtype=np.float64)
    for node in tree.traverse("preorder"):
        i = node_to_i[node]
        nd[i] = 0.0 if node.is_root() else nd[node_to_i[node.up]] + node.dist

    is_leaf = np.array([n.is_leaf() for n in nodes], dtype=np.bool_)

    postorder = np.array([node_to_i[n] for n in tree.traverse("postorder")],
                         dtype=np.int32)
    preorder  = np.array([node_to_i[n] for n in tree.traverse("preorder")],
                         dtype=np.int32)

    root_idx = node_to_i[tree.get_tree_root()]
    node_names = [n.name for n in nodes]
    name_to_idx = {n.name: node_to_i[n] for n in nodes}
    leaf_node_idx = np.array([i for i, n in enumerate(nodes) if n.is_leaf()],
                              dtype=np.int32)

    bl_nz = bl[bl > 0]
    avg_bl = float(bl_nz.mean()) if len(bl_nz) > 0 else 1.0

    return TreeArrays(
        n_nodes=n_nodes,
        root_idx=root_idx,
        postorder=postorder,
        preorder=preorder,
        parent=parent,
        children_ptr=children_ptr,
        children_list=children_list,
        bl=bl,
        nd=nd,
        is_leaf=is_leaf,
        node_names=node_names,
        name_to_idx=name_to_idx,
        leaf_node_idx=leaf_node_idx,
        avg_bl=avg_bl,
    )


def build_obs_matrix(
    ta: TreeArrays,
    trait_df: pd.DataFrame,
) -> np.ndarray:
    """Build obs[n_traits, n_nodes] int8 matrix.

    trait_df : DataFrame with taxa as index, traits as columns.
               Values should be 0 / 1 (or convertible).  Missing taxa (not
               in the tree) are silently ignored.  Tree leaves not in
               trait_df are encoded as -1 (unknown/ambiguous).

    Returns int8 array: 0 = absent, 1 = present, -1 = unknown.
    """
    n_traits = len(trait_df.columns)
    n_nodes  = ta.n_nodes
    obs = np.full((n_traits, n_nodes), -1, dtype=np.int8)

    for t_idx, col in enumerate(trait_df.columns):
        series = trait_df[col]
        for taxon, val in series.items():
            if taxon in ta.name_to_idx:
                node_i = ta.name_to_idx[taxon]
                if ta.is_leaf[node_i]:
                    try:
                        obs[t_idx, node_i] = np.int8(int(val))
                    except (ValueError, OverflowError):
                        pass  # leave as -1

    return obs


# ---------------------------------------------------------------------------
# Numba utility
# ---------------------------------------------------------------------------

@njit(inline="always")
def _log_sum_exp2(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf and b == -math.inf:
        return -math.inf
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@njit(inline="always")
def _f81_log_probs(bl: float, sf: float, pi1: float):
    """Return (log_alpha, log_beta, log_1m_alpha, log_1m_beta) for F81 binary.

    alpha = P(0→1) = π₁(1−e),  beta = P(1→0) = π₀(1−e),  e = exp(−mu·t·sf)
    Returns log values (−inf for zero probability).
    """
    pi0   = 1.0 - pi1
    denom = 2.0 * pi0 * pi1
    if denom < 1e-15:
        return -math.inf, -math.inf, 0.0, 0.0
    mu    = 1.0 / denom
    t_eff = bl * sf
    e     = math.exp(-mu * t_eff) if mu * t_eff < 700.0 else 0.0
    alpha = pi1 * (1.0 - e)
    beta  = pi0 * (1.0 - e)
    la    = math.log(alpha) if alpha > 1e-300 else -math.inf
    lb    = math.log(beta)  if beta  > 1e-300 else -math.inf
    l1ma  = math.log1p(-alpha) if alpha < 1.0 - 1e-15 else -math.inf
    l1mb  = math.log1p(-beta)  if beta  < 1.0 - 1e-15 else -math.inf
    return la, lb, l1ma, l1mb


# ---------------------------------------------------------------------------
# Bottom-up pass — marginal (sum-product)
# ---------------------------------------------------------------------------

@njit
def _bu_marginal(postorder, children_ptr, children_list, bl, obs_col,
                 sf, pi1, is_leaf, n_nodes):
    """Bottom-up marginal pass (sum-product) in log-space.

    Returns log_bu[n_nodes, 2] where log_bu[i, s] = log P(data below i | state s).
    """
    log_bu = np.empty((n_nodes, 2), dtype=np.float64)
    NEG_INF = -math.inf

    for ki in range(len(postorder)):
        node = postorder[ki]
        if is_leaf[node]:
            s = obs_col[node]
            if s == 0:
                log_bu[node, 0] = 0.0
                log_bu[node, 1] = NEG_INF
            elif s == 1:
                log_bu[node, 0] = NEG_INF
                log_bu[node, 1] = 0.0
            else:                          # unknown / ambiguous
                log_bu[node, 0] = 0.0
                log_bu[node, 1] = 0.0
        else:
            log_bu[node, 0] = 0.0
            log_bu[node, 1] = 0.0
            cs = children_ptr[node]
            ce = children_ptr[node + 1]
            for ci in range(cs, ce):
                child = children_list[ci]
                la, lb, l1ma, l1mb = _f81_log_probs(bl[child], sf, pi1)
                # sum over child states for each parent state
                log_bu[node, 0] += _log_sum_exp2(l1ma + log_bu[child, 0],
                                                  la   + log_bu[child, 1])
                log_bu[node, 1] += _log_sum_exp2(lb   + log_bu[child, 0],
                                                  l1mb + log_bu[child, 1])
    return log_bu


# ---------------------------------------------------------------------------
# Bottom-up pass — JOINT (max-product)
# ---------------------------------------------------------------------------

@njit
def _bu_joint(postorder, children_ptr, children_list, bl, obs_col,
              sf, pi1, is_leaf, n_nodes):
    """Bottom-up JOINT pass (max-product) in log-space.

    Returns log_bu[n_nodes, 2].
    """
    log_bu = np.empty((n_nodes, 2), dtype=np.float64)
    NEG_INF = -math.inf

    for ki in range(len(postorder)):
        node = postorder[ki]
        if is_leaf[node]:
            s = obs_col[node]
            if s == 0:
                log_bu[node, 0] = 0.0
                log_bu[node, 1] = NEG_INF
            elif s == 1:
                log_bu[node, 0] = NEG_INF
                log_bu[node, 1] = 0.0
            else:
                log_bu[node, 0] = 0.0
                log_bu[node, 1] = 0.0
        else:
            log_bu[node, 0] = 0.0
            log_bu[node, 1] = 0.0
            cs = children_ptr[node]
            ce = children_ptr[node + 1]
            for ci in range(cs, ce):
                child = children_list[ci]
                la, lb, l1ma, l1mb = _f81_log_probs(bl[child], sf, pi1)
                v00 = l1ma + log_bu[child, 0]   # P(par=0, ch=0)
                v01 = la   + log_bu[child, 1]   # P(par=0, ch=1)
                v10 = lb   + log_bu[child, 0]   # P(par=1, ch=0)
                v11 = l1mb + log_bu[child, 1]   # P(par=1, ch=1)
                log_bu[node, 0] += max(v00, v01)
                log_bu[node, 1] += max(v10, v11)
    return log_bu


# ---------------------------------------------------------------------------
# Top-down pass — marginal
# ---------------------------------------------------------------------------

@njit
def _td_marginal(preorder, parent, children_ptr, children_list, bl,
                 log_bu, sf, pi0, pi1, n_nodes, root_idx):
    """Top-down marginal pass in log-space.

    log_td[node][j] integrates over all states of the "rest of the tree"
    (everything outside the subtree rooted at node).

        TD[child][j] = Σ_i  (TD[par][i] · BU[par][i] / contrib[i]) · P[i,j]

    where contrib[i] = Σ_j P[i,j] · BU[child][j].

    Returns log_td[n_nodes, 2].
    """
    log_td = np.empty((n_nodes, 2), dtype=np.float64)
    log_td[root_idx, 0] = math.log(pi0) if pi0 > 1e-300 else -math.inf
    log_td[root_idx, 1] = math.log(pi1) if pi1 > 1e-300 else -math.inf

    for ki in range(len(preorder)):
        node = preorder[ki]
        if node == root_idx:
            continue
        p = parent[node]
        la, lb, l1ma, l1mb = _f81_log_probs(bl[node], sf, pi1)

        # log_contrib[i] = log( Σ_j P[i,j] · BU[node][j] )
        lc0 = _log_sum_exp2(l1ma + log_bu[node, 0], la   + log_bu[node, 1])
        lc1 = _log_sum_exp2(lb   + log_bu[node, 0], l1mb + log_bu[node, 1])

        # log p_without[i] = log_td[p][i] + log_bu[p][i] − log_contrib[i]
        # Guard: -inf - (-inf) = nan; treat as -inf (impossible state).
        NEG_INF = -math.inf
        lpw0 = NEG_INF if lc0 == NEG_INF else log_td[p, 0] + log_bu[p, 0] - lc0
        lpw1 = NEG_INF if lc1 == NEG_INF else log_td[p, 1] + log_bu[p, 1] - lc1

        # log_td[node][j] = log_sum_exp_i( lpw[i] + log P[i,j] )
        log_td[node, 0] = _log_sum_exp2(lpw0 + l1ma, lpw1 + lb)
        log_td[node, 1] = _log_sum_exp2(lpw0 + la,   lpw1 + l1mb)

    return log_td


# ---------------------------------------------------------------------------
# Marginal probabilities from BU + TD
# ---------------------------------------------------------------------------

@njit
def _marginal_p1(log_bu, log_td, pi0, pi1, n_nodes):
    """P(state=1 | data) at every node, given BU and TD log-likelihoods.

    marginal[node][s] ∝ BU[node][s] · TD[node][s]

    The π prior is already embedded in TD via the root initialization
    (log_td[root] = [log(π₀), log(π₁)]), so we do NOT add π here again.
    """
    p1 = np.empty(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        lm0 = log_bu[i, 0] + log_td[i, 0]
        lm1 = log_bu[i, 1] + log_td[i, 1]
        lt  = _log_sum_exp2(lm0, lm1)
        p1[i] = math.exp(lm1 - lt) if lt > -math.inf else 0.5
    return p1


# ---------------------------------------------------------------------------
# JOINT traceback
# ---------------------------------------------------------------------------

@njit
def _joint_traceback(preorder, parent, bl, log_bu, sf, pi1, n_nodes, root_idx):
    """Top-down JOINT traceback — assigns MAP state to every node.

    Returns joint_states[n_nodes] as int8 (0 or 1; -1 = unreachable).
    """
    joint = np.full(n_nodes, -1, dtype=np.int8)
    pi0   = 1.0 - pi1
    # Root: argmax( π_s · BU[root][s] )
    lpi0  = math.log(pi0) if pi0 > 1e-300 else -math.inf
    lpi1  = math.log(pi1) if pi1 > 1e-300 else -math.inf
    joint[root_idx] = np.int8(1 if (lpi1 + log_bu[root_idx, 1]) >= (lpi0 + log_bu[root_idx, 0]) else 0)

    for ki in range(len(preorder)):
        node = preorder[ki]
        if node == root_idx:
            continue
        ps = joint[parent[node]]
        if ps < 0:
            continue
        la, lb, l1ma, l1mb = _f81_log_probs(bl[node], sf, pi1)
        if ps == 0:
            v0 = l1ma + log_bu[node, 0]
            v1 = la   + log_bu[node, 1]
        else:
            v0 = lb   + log_bu[node, 0]
            v1 = l1mb + log_bu[node, 1]
        joint[node] = np.int8(1 if v1 >= v0 else 0)

    return joint


# ---------------------------------------------------------------------------
# Log-likelihood helper (used inside rate optimisation)
# ---------------------------------------------------------------------------

@njit(inline="always")
def _log_lh(log_bu_root, pi0, pi1):
    """log P(data) = log Σ_s π_s · BU[root][s]."""
    lpi0 = math.log(pi0) if pi0 > 1e-300 else -math.inf
    lpi1 = math.log(pi1) if pi1 > 1e-300 else -math.inf
    return _log_sum_exp2(lpi0 + log_bu_root[0], lpi1 + log_bu_root[1])


# ---------------------------------------------------------------------------
# Rate optimisation — golden-section search (1-D, Numba-native)
# ---------------------------------------------------------------------------

@njit(inline="always")
def _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
               pi0, pi1, is_leaf, n_nodes, root_idx, log_sf):
    """Evaluate log P(data | sf) for golden-section use."""
    sf_ = math.exp(log_sf)
    lbu = _bu_marginal(postorder, children_ptr, children_list, bl,
                       obs_col, sf_, pi1, is_leaf, n_nodes)
    return _log_lh(lbu[root_idx], pi0, pi1)


@njit(inline="always")
def _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                sf, is_leaf, n_nodes, root_idx, pi1_):
    """Evaluate log P(data | pi1) for golden-section use."""
    pi0_ = 1.0 - pi1_
    lbu  = _bu_marginal(postorder, children_ptr, children_list, bl,
                        obs_col, sf, pi1_, is_leaf, n_nodes)
    return _log_lh(lbu[root_idx], pi0_, pi1_)


@njit
def _golden_section_sf(
    postorder, children_ptr, children_list, bl, obs_col,
    pi0, pi1, is_leaf, n_nodes, root_idx,
    log_sf_lo, log_sf_hi, n_iter=32,
):
    """Maximise log-likelihood over log(sf) using golden-section search.

    Searches in log-space so the interval is uniform in multiplicative scale.
    Returns (sf_opt, log_lh_opt).
    """
    PHI = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618
    a, b = log_sf_lo, log_sf_hi
    c = b - PHI * (b - a)
    d = a + PHI * (b - a)
    fc = _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
                    pi0, pi1, is_leaf, n_nodes, root_idx, c)
    fd = _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
                    pi0, pi1, is_leaf, n_nodes, root_idx, d)

    for _ in range(n_iter):
        if fc >= fd:
            b  = d
            d  = c; fd = fc
            c  = b - PHI * (b - a)
            fc = _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
                            pi0, pi1, is_leaf, n_nodes, root_idx, c)
        else:
            a  = c
            c  = d; fc = fd
            d  = a + PHI * (b - a)
            fd = _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
                            pi0, pi1, is_leaf, n_nodes, root_idx, d)

    sf_opt = math.exp((a + b) / 2.0)
    return sf_opt, (fc + fd) / 2.0


@njit
def _golden_section_pi1(
    postorder, children_ptr, children_list, bl, obs_col,
    sf, is_leaf, n_nodes, root_idx,
    pi1_lo, pi1_hi, n_iter=32,
):
    """Maximise log-likelihood over π₁ with sf fixed.  Returns (pi1_opt, lh_opt)."""
    PHI = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = pi1_lo, pi1_hi
    c = b - PHI * (b - a)
    d = a + PHI * (b - a)
    fc = _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                     sf, is_leaf, n_nodes, root_idx, c)
    fd = _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                     sf, is_leaf, n_nodes, root_idx, d)

    for _ in range(n_iter):
        if fc >= fd:
            b  = d
            d  = c; fd = fc
            c  = b - PHI * (b - a)
            fc = _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                             sf, is_leaf, n_nodes, root_idx, c)
        else:
            a  = c
            c  = d; fc = fd
            d  = a + PHI * (b - a)
            fd = _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                             sf, is_leaf, n_nodes, root_idx, d)

    return (a + b) / 2.0, (fc + fd) / 2.0


# ---------------------------------------------------------------------------
# Parallel batch kernel
# ---------------------------------------------------------------------------

@njit(parallel=True)
def _batch_acr_kernel(
    postorder, preorder, parent,
    children_ptr, children_list,
    bl, obs,                        # obs: int8[n_traits, n_nodes]
    pi1_init,                        # float64[n_traits]  initial π₁ per trait
    is_leaf, n_nodes, root_idx,
    log_sf_lo, log_sf_hi,
    pi1_lo, pi1_hi,
    n_gs_sf, n_gs_pi1,
    ml_mode,                         # bool: optimise π₁ as well?
):
    """Parallel binary F81 ACR for every trait.

    Returns
    -------
    sf_out       : float64[n_traits]
    pi1_out      : float64[n_traits]
    log_lh_out   : float64[n_traits]
    marginal_p1  : float64[n_traits, n_nodes]
    joint_states : int8[n_traits, n_nodes]
    """
    n_traits     = obs.shape[0]
    sf_out       = np.empty(n_traits, dtype=np.float64)
    pi1_out      = np.empty(n_traits, dtype=np.float64)
    log_lh_out   = np.empty(n_traits, dtype=np.float64)
    marginal_p1  = np.empty((n_traits, n_nodes), dtype=np.float64)
    joint_states = np.empty((n_traits, n_nodes), dtype=np.int8)

    for t in prange(n_traits):
        obs_col   = obs[t]
        pi1_emp   = pi1_init[t]

        # --- Rate optimisation ---
        # Always run from empirical π₁ first.
        pi0_emp       = 1.0 - pi1_emp
        sf, log_lh_v  = _golden_section_sf(
            postorder, children_ptr, children_list, bl, obs_col,
            pi0_emp, pi1_emp, is_leaf, n_nodes, root_idx,
            log_sf_lo, log_sf_hi, n_gs_sf,
        )
        pi1 = pi1_emp

        if ml_mode:
            # Alternating coordinate descent from empirical π₁.
            for _cycle in range(2):
                pi1, log_lh_v = _golden_section_pi1(
                    postorder, children_ptr, children_list, bl, obs_col,
                    sf, is_leaf, n_nodes, root_idx,
                    pi1_lo, pi1_hi, n_gs_pi1,
                )
                pi0 = 1.0 - pi1
                sf, log_lh_v = _golden_section_sf(
                    postorder, children_ptr, children_list, bl, obs_col,
                    pi0, pi1, is_leaf, n_nodes, root_idx,
                    log_sf_lo, log_sf_hi, n_gs_sf,
                )

            # --- Multi-start: also try JC (π₁=0.5) and flipped (π₁=1−emp) ---
            # This is critical for high-frequency traits where the empirical
            # start locks onto a saturated (sf→∞, p≈π₁ everywhere) optimum
            # that gives nearly-flat marginals and wrong gains_flow.
            for pi1_start in (0.5, 1.0 - pi1_emp):
                pi1_s = max(pi1_lo, min(pi1_hi, pi1_start))
                sf_s, lh_s = _golden_section_sf(
                    postorder, children_ptr, children_list, bl, obs_col,
                    1.0 - pi1_s, pi1_s, is_leaf, n_nodes, root_idx,
                    log_sf_lo, log_sf_hi, n_gs_sf,
                )
                for _cycle in range(2):
                    pi1_s, lh_s = _golden_section_pi1(
                        postorder, children_ptr, children_list, bl, obs_col,
                        sf_s, is_leaf, n_nodes, root_idx,
                        pi1_lo, pi1_hi, n_gs_pi1,
                    )
                    sf_s, lh_s = _golden_section_sf(
                        postorder, children_ptr, children_list, bl, obs_col,
                        1.0 - pi1_s, pi1_s, is_leaf, n_nodes, root_idx,
                        log_sf_lo, log_sf_hi, n_gs_sf,
                    )
                if lh_s > log_lh_v:
                    sf, pi1, log_lh_v = sf_s, pi1_s, lh_s

        pi0 = 1.0 - pi1
        sf_out[t]     = sf
        pi1_out[t]    = pi1
        log_lh_out[t] = log_lh_v

        # --- Marginal BU + TD + probabilities ---
        log_bu = _bu_marginal(postorder, children_ptr, children_list, bl,
                              obs_col, sf, pi1, is_leaf, n_nodes)
        log_td = _td_marginal(preorder, parent, children_ptr, children_list,
                              bl, log_bu, sf, pi0, pi1, n_nodes, root_idx)
        marginal_p1[t] = _marginal_p1(log_bu, log_td, pi0, pi1, n_nodes)

        # --- JOINT BU + traceback ---
        log_bu_j = _bu_joint(postorder, children_ptr, children_list, bl,
                             obs_col, sf, pi1, is_leaf, n_nodes)
        joint_states[t] = _joint_traceback(preorder, parent, bl, log_bu_j,
                                           sf, pi1, n_nodes, root_idx)

    return sf_out, pi1_out, log_lh_out, marginal_p1, joint_states


# ---------------------------------------------------------------------------
# Python-level result container and wrapper
# ---------------------------------------------------------------------------

@dataclass
class FastACRResult:
    """Results from fast_acr().

    Arrays are indexed [trait_index, node_index] where node_index matches
    the ordering in ta.node_names.
    """
    marginal_p1:  np.ndarray  # float64[n_traits, n_nodes]
    joint_states: np.ndarray  # int8[n_traits, n_nodes]
    sf:           np.ndarray  # float64[n_traits]
    pi1:          np.ndarray  # float64[n_traits]
    log_lh:       np.ndarray  # float64[n_traits]
    trait_names:  List[str]
    ta:           TreeArrays
    elapsed_s:    float


def fast_acr(
    ta: TreeArrays,
    obs: np.ndarray,
    trait_names: List[str],
    mode: str = "empirical",
    n_gs_sf: int = 32,
    n_gs_pi1: int = 32,
    sf_lo_mult: float = 0.0001,
    sf_hi_mult: float = 100.0,
    pi1_lo: float = 1e-4,
    pi1_hi_off: float = 1e-4,
) -> FastACRResult:
    """Run fast binary F81 ACR on all traits in parallel.

    Parameters
    ----------
    ta           : TreeArrays from build_tree_arrays()
    obs          : int8[n_traits, n_nodes] from build_obs_matrix()
    trait_names  : list of trait column names (len = n_traits)
    mode         : 'empirical' or 'ml'
                   'empirical' — fix π₁ = observed tip frequency, optimise sf only
                   'ml'        — alternating golden-section optimisation of sf and π₁
    n_gs_sf      : golden-section iterations for sf  (default 32 → ~1e-9 precision)
    n_gs_pi1     : golden-section iterations for π₁  (default 32)
    sf_lo_mult   : sf lower bound = sf_lo_mult / avg_bl
    sf_hi_mult   : sf upper bound = sf_hi_mult / avg_bl
    pi1_lo       : π₁ lower bound for ml mode
    pi1_hi_off   : π₁ upper bound = 1 − pi1_hi_off  for ml mode

    Returns
    -------
    FastACRResult
    """
    if mode not in ("empirical", "ml"):
        raise ValueError(f"mode must be 'empirical' or 'ml', got {mode!r}")

    n_traits, n_nodes = obs.shape
    assert n_nodes == ta.n_nodes

    # ---- Empirical π₁ per trait (from observed tip values) ----
    # Use the leaf-restricted obs values; unknown (-1) excluded from mean.
    pi1_init = np.empty(n_traits, dtype=np.float64)
    leaf_idx  = ta.leaf_node_idx
    for t in range(n_traits):
        col = obs[t, leaf_idx]
        known = col[col >= 0]
        if len(known) == 0 or known.sum() == 0:
            pi1_init[t] = 0.01
        elif known.sum() == len(known):
            pi1_init[t] = 0.99
        else:
            pi1_init[t] = float(known.sum()) / float(len(known))

    # ---- sf search bounds (in log-space) ----
    log_sf_lo = math.log(sf_lo_mult / ta.avg_bl)
    log_sf_hi = math.log(sf_hi_mult / ta.avg_bl)
    pi1_lo_v  = float(pi1_lo)
    pi1_hi_v  = float(1.0 - pi1_hi_off)

    ml_mode = (mode == "ml")

    # ---- Warm-up Numba JIT on a single trait (avoids timing the compile) ----
    _batch_acr_kernel(
        ta.postorder, ta.preorder, ta.parent,
        ta.children_ptr, ta.children_list,
        ta.bl, obs[:1],
        pi1_init[:1],
        ta.is_leaf, ta.n_nodes, ta.root_idx,
        log_sf_lo, log_sf_hi,
        pi1_lo_v, pi1_hi_v,
        n_gs_sf, n_gs_pi1,
        ml_mode,
    )

    # ---- Full batch run ----
    t0 = time.perf_counter()
    sf_out, pi1_out, log_lh_out, marginal_p1, joint_states = _batch_acr_kernel(
        ta.postorder, ta.preorder, ta.parent,
        ta.children_ptr, ta.children_list,
        ta.bl, obs,
        pi1_init,
        ta.is_leaf, ta.n_nodes, ta.root_idx,
        log_sf_lo, log_sf_hi,
        pi1_lo_v, pi1_hi_v,
        n_gs_sf, n_gs_pi1,
        ml_mode,
    )
    elapsed = time.perf_counter() - t0

    return FastACRResult(
        marginal_p1=marginal_p1,
        joint_states=joint_states,
        sf=sf_out,
        pi1=pi1_out,
        log_lh=log_lh_out,
        trait_names=list(trait_names),
        ta=ta,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Convenience: convert FastACRResult to PastML-compatible DataFrames
# ---------------------------------------------------------------------------

def marginal_probs_df(result: FastACRResult, trait_idx: int) -> pd.DataFrame:
    """Return a marginal_probabilities DataFrame for one trait.

    The returned DataFrame matches the structure of PastML's
    acr_result[0]['marginal_probabilities']:
      - Index  : node names
      - Columns: ['0', '1']  (string column names, as PastML writes)
    """
    p1 = result.marginal_p1[trait_idx]
    p0 = 1.0 - p1
    return pd.DataFrame(
        {"0": p0, "1": p1},
        index=result.ta.node_names,
    )


def joint_states_series(result: FastACRResult, trait_idx: int) -> pd.Series:
    """Return JOINT states as a Series indexed by node name."""
    return pd.Series(
        result.joint_states[trait_idx],
        index=result.ta.node_names,
        name=result.trait_names[trait_idx],
    )
