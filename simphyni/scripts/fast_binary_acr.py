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

Performance notes
-----------------
* Requires Numba >= 0.55 (for numba.get_thread_id() inside prange).
* The kernel allocates O(n_threads × n_nodes) scratch instead of
  O(n_traits × n_nodes) temporaries per golden-section evaluation, so
  heap allocation pressure inside prange is eliminated.
* Thread count defaults to physical cores (not logical/HT) to avoid FPU
  contention.  Override with n_threads kwarg or NUMBA_NUM_THREADS env var.
* build_obs_matrix() uses a vectorised NumPy reindex path instead of a
  pure-Python double loop — O(n_leaves × n_traits) NumPy ops vs. the same
  count of CPython bytecodes.

Outputs
-------
fast_acr() returns a FastACRResult with:
  marginal_p1  : float64[n_traits, n_nodes] — P(state=1 | data) at every node
  joint_states : int8[n_traits, n_nodes]    — MAP state from JOINT traceback
  sf           : float64[n_traits]          — optimised scaling factors
  pi1          : float64[n_traits]          — used equilibrium frequencies
  log_lh       : float64[n_traits]          — final log-likelihoods
  timing       : dict                       — walltime (s) per phase

The marginal_p1 array can be sliced per-trait to build a PastML-compatible
marginal_probabilities DataFrame via marginal_probs_df().
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numba
import numpy as np
import pandas as pd
from ete3 import Tree
from numba import njit, prange


# ---------------------------------------------------------------------------
# Thread configuration
# ---------------------------------------------------------------------------

_threads_configured: bool = False


def _get_physical_cores() -> Optional[int]:
    """Return the number of physical CPU cores, or None if undetectable."""
    try:
        sys_name = platform.system()
        if sys_name == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=2,
            )
            return int(r.stdout.strip())
        if sys_name == "Linux":
            cores: set = set()
            phys_id = core_id = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":")[1].strip()
                    elif line.strip() == "" and phys_id is not None and core_id is not None:
                        cores.add((phys_id, core_id))
                        phys_id = core_id = None
            return len(cores) if cores else None
    except Exception:
        return None


def configure_numba_threads(n_threads: Optional[int] = None) -> int:
    """Set Numba's thread count and return the value used.

    Call before the first fast_acr() to avoid recompiling with a different
    thread count.  If *n_threads* is None and NUMBA_NUM_THREADS is not set,
    attempts to use physical core count (avoids hyperthreading FPU contention).

    Parameters
    ----------
    n_threads : int or None
        Explicit thread count.  None = auto-detect.

    Returns
    -------
    int : thread count actually set.
    """
    global _threads_configured
    if n_threads is not None:
        numba.set_num_threads(int(n_threads))
        _threads_configured = True
        return numba.get_num_threads()
    if "NUMBA_NUM_THREADS" in os.environ:
        _threads_configured = True
        return numba.get_num_threads()
    physical = _get_physical_cores()
    current  = numba.get_num_threads()
    if physical and 0 < physical < current:
        numba.set_num_threads(physical)
    _threads_configured = True
    return numba.get_num_threads()


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
    """Build obs[n_traits, n_nodes] int8 matrix — vectorised NumPy implementation.

    trait_df : DataFrame with taxa as index, traits as columns.
               Values should be 0 / 1 (or numeric-convertible).  Missing taxa
               (not in the tree) are silently ignored.  Tree leaves not in
               trait_df are encoded as -1 (unknown/ambiguous).

    Returns int8 array: 0 = absent, 1 = present, -1 = unknown.

    Implementation
    --------------
    Replaces the original O(n_traits × n_taxa) CPython double-loop with a
    vectorised NumPy path operating on a (n_leaves × n_traits) block:

      1. reindex rows to tree-leaf order  — drops taxa not in tree,
                                            NaN-fills leaves absent from df
      2. coerce to float64 (non-numeric → NaN)
      3. cast NaN → -1, finite → int8
      4. scatter-write into obs via ta.leaf_node_idx (one row-assignment)

    Peak extra RAM: one float64 (n_leaves × n_traits) slice, released on
    return.  The full n_nodes dimension is never materialised in float64.
    Internal nodes stay at their initial -1 sentinel.
    """
    n_traits = len(trait_df.columns)
    obs = np.full((n_traits, ta.n_nodes), -1, dtype=np.int8)

    if n_traits == 0:
        return obs

    # Leaf names in node-index order (only leaves, not internal nodes)
    leaf_names = [ta.node_names[i] for i in ta.leaf_node_idx]

    # Align rows to leaf order.
    #   taxa not in tree  → not in leaf_names → absent from aligned (no NaN row)
    #   leaves not in df  → NaN row in aligned
    aligned = trait_df.reindex(leaf_names)          # (n_leaves, n_traits)

    # Coerce to float64; non-numeric strings / objects become NaN
    vals = aligned.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    # vals shape: (n_leaves, n_traits)

    # Build int8 with NaN → -1 sentinel; clip guards against out-of-range casts
    known    = np.isfinite(vals)
    int_vals = np.where(known, np.clip(vals, -128, 127).astype(np.int8), np.int8(-1))
    # int_vals shape: (n_leaves, n_traits)

    # obs[t, leaf_node_idx[l]] = int_vals[l, t]  — one vectorised assignment
    obs[:, ta.leaf_node_idx] = int_vals.T          # (n_traits, n_leaves)

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
# Bottom-up pass — marginal (sum-product), in-place
# ---------------------------------------------------------------------------

@njit
def _bu_marginal_into(postorder, children_ptr, children_list, bl, obs_col,
                      sf, pi1, is_leaf, n_nodes, log_bu):
    """Fill log_bu[n_nodes, 2] in-place.

    log_bu[i, s] = log P(data below node i | state s at node i).
    Overwrites all entries; caller must treat the buffer as scratch.
    """
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
                log_bu[node, 0] += _log_sum_exp2(l1ma + log_bu[child, 0],
                                                  la   + log_bu[child, 1])
                log_bu[node, 1] += _log_sum_exp2(lb   + log_bu[child, 0],
                                                  l1mb + log_bu[child, 1])


# ---------------------------------------------------------------------------
# Bottom-up pass — JOINT (max-product), in-place
# ---------------------------------------------------------------------------

@njit
def _bu_joint_into(postorder, children_ptr, children_list, bl, obs_col,
                   sf, pi1, is_leaf, n_nodes, log_bu):
    """Fill log_bu[n_nodes, 2] in-place (bottom-up JOINT, max-product, log-space)."""
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


# ---------------------------------------------------------------------------
# Top-down pass — marginal, in-place
# ---------------------------------------------------------------------------

@njit
def _td_marginal_into(preorder, parent, children_ptr, children_list, bl,
                      log_bu, sf, pi0, pi1, n_nodes, root_idx, log_td):
    """Fill log_td[n_nodes, 2] in-place (top-down marginal, log-space).

    log_td[node][j] integrates over all states of the "rest of the tree"
    (everything outside the subtree rooted at node).

        TD[child][j] = Σ_i  (TD[par][i] · BU[par][i] / contrib[i]) · P[i,j]

    where contrib[i] = Σ_j P[i,j] · BU[child][j].
    log_bu is read-only; log_td is write-only.
    """
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
        NEG_INF = -math.inf
        lpw0 = NEG_INF if lc0 == NEG_INF else log_td[p, 0] + log_bu[p, 0] - lc0
        lpw1 = NEG_INF if lc1 == NEG_INF else log_td[p, 1] + log_bu[p, 1] - lc1

        # log_td[node][j] = log_sum_exp_i( lpw[i] + log P[i,j] )
        log_td[node, 0] = _log_sum_exp2(lpw0 + l1ma, lpw1 + lb)
        log_td[node, 1] = _log_sum_exp2(lpw0 + la,   lpw1 + l1mb)


# ---------------------------------------------------------------------------
# Marginal probabilities from BU + TD, in-place
# ---------------------------------------------------------------------------

@njit
def _marginal_p1_into(log_bu, log_td, pi0, pi1, n_nodes, out_p1):
    """Write P(state=1 | data) at every node into out_p1[n_nodes].

    marginal[node][s] ∝ BU[node][s] · TD[node][s]

    The π prior is embedded in TD via the root initialisation, so π is not
    added here again.  out_p1 must be a contiguous float64[n_nodes] buffer.
    """
    for i in range(n_nodes):
        lm0 = log_bu[i, 0] + log_td[i, 0]
        lm1 = log_bu[i, 1] + log_td[i, 1]
        lt  = _log_sum_exp2(lm0, lm1)
        out_p1[i] = math.exp(lm1 - lt) if lt > -math.inf else 0.5


# ---------------------------------------------------------------------------
# JOINT traceback, in-place
# ---------------------------------------------------------------------------

@njit
def _joint_traceback_into(preorder, parent, bl, log_bu, sf, pi1,
                           n_nodes, root_idx, out_joint):
    """Write MAP state (0 or 1; -1 = unreachable) into out_joint[n_nodes] (int8)."""
    pi0  = 1.0 - pi1
    lpi0 = math.log(pi0) if pi0 > 1e-300 else -math.inf
    lpi1 = math.log(pi1) if pi1 > 1e-300 else -math.inf
    out_joint[root_idx] = np.int8(
        1 if (lpi1 + log_bu[root_idx, 1]) >= (lpi0 + log_bu[root_idx, 0]) else 0
    )

    for ki in range(len(preorder)):
        node = preorder[ki]
        if node == root_idx:
            continue
        ps = out_joint[parent[node]]
        if ps < 0:
            continue
        la, lb, l1ma, l1mb = _f81_log_probs(bl[node], sf, pi1)
        if ps == 0:
            v0 = l1ma + log_bu[node, 0]
            v1 = la   + log_bu[node, 1]
        else:
            v0 = lb   + log_bu[node, 0]
            v1 = l1mb + log_bu[node, 1]
        out_joint[node] = np.int8(1 if v1 >= v0 else 0)


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
# Rate optimisation helpers — accept pre-allocated scratch buffer
# ---------------------------------------------------------------------------

@njit(inline="always")
def _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
               pi0, pi1, is_leaf, n_nodes, root_idx, log_sf, log_bu_scratch):
    """Evaluate log P(data | sf); overwrites log_bu_scratch[n_nodes, 2] in-place."""
    sf_ = math.exp(log_sf)
    _bu_marginal_into(postorder, children_ptr, children_list, bl,
                      obs_col, sf_, pi1, is_leaf, n_nodes, log_bu_scratch)
    return _log_lh(log_bu_scratch[root_idx], pi0, pi1)


@njit(inline="always")
def _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                sf, is_leaf, n_nodes, root_idx, pi1_, log_bu_scratch):
    """Evaluate log P(data | π₁); overwrites log_bu_scratch[n_nodes, 2] in-place."""
    pi0_ = 1.0 - pi1_
    _bu_marginal_into(postorder, children_ptr, children_list, bl,
                      obs_col, sf, pi1_, is_leaf, n_nodes, log_bu_scratch)
    return _log_lh(log_bu_scratch[root_idx], pi0_, pi1_)


@njit
def _golden_section_sf(
    postorder, children_ptr, children_list, bl, obs_col,
    pi0, pi1, is_leaf, n_nodes, root_idx,
    log_sf_lo, log_sf_hi, n_iter, log_bu_scratch,
):
    """Maximise log-likelihood over log(sf) using golden-section search.

    log_bu_scratch[n_nodes, 2] is used as working space and overwritten on
    every likelihood evaluation — no heap allocation inside the loop.
    Returns (sf_opt, log_lh_opt).
    """
    PHI = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618
    a, b = log_sf_lo, log_sf_hi
    c = b - PHI * (b - a)
    d = a + PHI * (b - a)
    fc = _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
                    pi0, pi1, is_leaf, n_nodes, root_idx, c, log_bu_scratch)
    fd = _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
                    pi0, pi1, is_leaf, n_nodes, root_idx, d, log_bu_scratch)

    for _ in range(n_iter):
        if fc >= fd:
            b  = d
            d  = c; fd = fc
            c  = b - PHI * (b - a)
            fc = _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
                            pi0, pi1, is_leaf, n_nodes, root_idx, c, log_bu_scratch)
        else:
            a  = c
            c  = d; fc = fd
            d  = a + PHI * (b - a)
            fd = _lh_for_sf(postorder, children_ptr, children_list, bl, obs_col,
                            pi0, pi1, is_leaf, n_nodes, root_idx, d, log_bu_scratch)

    sf_opt = math.exp((a + b) / 2.0)
    return sf_opt, (fc + fd) / 2.0


@njit
def _golden_section_pi1(
    postorder, children_ptr, children_list, bl, obs_col,
    sf, is_leaf, n_nodes, root_idx,
    pi1_lo, pi1_hi, n_iter, log_bu_scratch,
):
    """Maximise log-likelihood over π₁ with sf fixed.

    log_bu_scratch[n_nodes, 2] is used as working space.
    Returns (pi1_opt, lh_opt).
    """
    PHI = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = pi1_lo, pi1_hi
    c = b - PHI * (b - a)
    d = a + PHI * (b - a)
    fc = _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                     sf, is_leaf, n_nodes, root_idx, c, log_bu_scratch)
    fd = _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                     sf, is_leaf, n_nodes, root_idx, d, log_bu_scratch)

    for _ in range(n_iter):
        if fc >= fd:
            b  = d
            d  = c; fd = fc
            c  = b - PHI * (b - a)
            fc = _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                             sf, is_leaf, n_nodes, root_idx, c, log_bu_scratch)
        else:
            a  = c
            c  = d; fc = fd
            d  = a + PHI * (b - a)
            fd = _lh_for_pi1(postorder, children_ptr, children_list, bl, obs_col,
                             sf, is_leaf, n_nodes, root_idx, d, log_bu_scratch)

    return (a + b) / 2.0, (fc + fd) / 2.0


# ---------------------------------------------------------------------------
# Parallel batch kernel — per-thread scratch, no heap allocation in prange
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

    Per-thread scratch arrays (n_threads × n_nodes × 2) are allocated once
    before the prange loop.  Each thread indexes its own slice via
    numba.get_thread_id(), eliminating all heap allocation inside the loop.
    Requires Numba >= 0.55 for get_thread_id().

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
    joint_states = np.full((n_traits, n_nodes), np.int8(-1), dtype=np.int8)

    # Allocate per-thread scratch: O(n_threads × n_nodes) total, not O(n_traits).
    # Each thread's slice is used exclusively by that thread throughout prange.
    n_th         = numba.get_num_threads()
    scratch_bu   = np.empty((n_th, n_nodes, 2), dtype=np.float64)  # marginal BU / rate opt
    scratch_td   = np.empty((n_th, n_nodes, 2), dtype=np.float64)  # TD pass
    scratch_bu_j = np.empty((n_th, n_nodes, 2), dtype=np.float64)  # joint BU

    for t in prange(n_traits):
        tid      = numba.get_thread_id()
        log_bu   = scratch_bu[tid]    # private (n_nodes, 2) view for this thread
        log_td   = scratch_td[tid]
        log_bu_j = scratch_bu_j[tid]

        obs_col  = obs[t]
        pi1_emp  = pi1_init[t]
        pi0_emp  = 1.0 - pi1_emp

        # --- Rate optimisation: always start from empirical π₁ ---
        sf, log_lh_v = _golden_section_sf(
            postorder, children_ptr, children_list, bl, obs_col,
            pi0_emp, pi1_emp, is_leaf, n_nodes, root_idx,
            log_sf_lo, log_sf_hi, n_gs_sf, log_bu,
        )
        pi1 = pi1_emp

        if ml_mode:
            # Alternating coordinate descent from empirical π₁.
            for _cycle in range(2):
                pi1, log_lh_v = _golden_section_pi1(
                    postorder, children_ptr, children_list, bl, obs_col,
                    sf, is_leaf, n_nodes, root_idx,
                    pi1_lo, pi1_hi, n_gs_pi1, log_bu,
                )
                pi0 = 1.0 - pi1
                sf, log_lh_v = _golden_section_sf(
                    postorder, children_ptr, children_list, bl, obs_col,
                    pi0, pi1, is_leaf, n_nodes, root_idx,
                    log_sf_lo, log_sf_hi, n_gs_sf, log_bu,
                )

            # Multi-start: also try JC (π₁=0.5) and flipped (π₁=1−emp).
            # Critical for high-frequency traits where the empirical start
            # locks onto a saturated (sf→∞, p≈π₁ everywhere) optimum.
            for pi1_start in (0.5, 1.0 - pi1_emp):
                pi1_s = max(pi1_lo, min(pi1_hi, pi1_start))
                sf_s, lh_s = _golden_section_sf(
                    postorder, children_ptr, children_list, bl, obs_col,
                    1.0 - pi1_s, pi1_s, is_leaf, n_nodes, root_idx,
                    log_sf_lo, log_sf_hi, n_gs_sf, log_bu,
                )
                for _cycle in range(2):
                    pi1_s, lh_s = _golden_section_pi1(
                        postorder, children_ptr, children_list, bl, obs_col,
                        sf_s, is_leaf, n_nodes, root_idx,
                        pi1_lo, pi1_hi, n_gs_pi1, log_bu,
                    )
                    sf_s, lh_s = _golden_section_sf(
                        postorder, children_ptr, children_list, bl, obs_col,
                        1.0 - pi1_s, pi1_s, is_leaf, n_nodes, root_idx,
                        log_sf_lo, log_sf_hi, n_gs_sf, log_bu,
                    )
                if lh_s > log_lh_v:
                    sf, pi1, log_lh_v = sf_s, pi1_s, lh_s

        pi0 = 1.0 - pi1
        sf_out[t]     = sf
        pi1_out[t]    = pi1
        log_lh_out[t] = log_lh_v

        # --- Final marginal BU + TD + marginal probabilities ---
        # Re-run BU with the converged (sf, pi1) — golden-section left log_bu
        # in an indeterminate state (last evaluation may not be the optimum).
        _bu_marginal_into(postorder, children_ptr, children_list, bl,
                          obs_col, sf, pi1, is_leaf, n_nodes, log_bu)
        _td_marginal_into(preorder, parent, children_ptr, children_list,
                          bl, log_bu, sf, pi0, pi1, n_nodes, root_idx, log_td)
        _marginal_p1_into(log_bu, log_td, pi0, pi1, n_nodes, marginal_p1[t])

        # --- Final JOINT BU + traceback ---
        _bu_joint_into(postorder, children_ptr, children_list, bl,
                       obs_col, sf, pi1, is_leaf, n_nodes, log_bu_j)
        _joint_traceback_into(preorder, parent, bl, log_bu_j,
                              sf, pi1, n_nodes, root_idx, joint_states[t])

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
    timing:       Dict[str, float] = field(default_factory=dict)


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
    n_threads: Optional[int] = None,
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
    n_threads    : Numba thread count (None = auto via configure_numba_threads())
                   Ignored if NUMBA_NUM_THREADS env var is already set.

    Returns
    -------
    FastACRResult
    """
    if mode not in ("empirical", "ml"):
        raise ValueError(f"mode must be 'empirical' or 'ml', got {mode!r}")

    timing: Dict[str, float] = {}

    # ---- Thread configuration (once per process, or whenever n_threads given) ----
    if n_threads is not None or not _threads_configured:
        n_actual = configure_numba_threads(n_threads)
        print(f"  [fast_acr] Numba threads: {n_actual}", flush=True)

    n_traits, n_nodes = obs.shape
    assert n_nodes == ta.n_nodes

    # ---- Empirical π₁ per trait — vectorised NumPy (replaces Python loop) ----
    t0 = time.perf_counter()
    leaf_idx = ta.leaf_node_idx
    leaf_obs = obs[:, leaf_idx]                          # int8[n_traits, n_leaves]
    known    = leaf_obs >= 0                             # bool[n_traits, n_leaves]
    n_known  = known.sum(axis=1)                         # int64[n_traits]
    n_ones   = np.where(known, leaf_obs.astype(np.int32), 0).sum(axis=1)
    safe_cnt = np.maximum(n_known, 1).astype(np.float64)
    pi1_init = np.where(
        n_known == 0,      0.01,
        np.where(
            n_ones == n_known, 0.99,
            n_ones.astype(np.float64) / safe_cnt,
        ),
    ).astype(np.float64)
    timing["pi1_init"] = time.perf_counter() - t0

    # ---- sf search bounds (in log-space) ----
    log_sf_lo = math.log(sf_lo_mult / ta.avg_bl)
    log_sf_hi = math.log(sf_hi_mult / ta.avg_bl)
    pi1_lo_v  = float(pi1_lo)
    pi1_hi_v  = float(1.0 - pi1_hi_off)
    ml_mode   = (mode == "ml")

    # ---- Warm-up: trigger Numba JIT before timing the real run ----
    t0 = time.perf_counter()
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
    timing["warmup"] = time.perf_counter() - t0

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
    timing["kernel"] = time.perf_counter() - t0
    timing["total"]  = sum(timing.values())

    return FastACRResult(
        marginal_p1=marginal_p1,
        joint_states=joint_states,
        sf=sf_out,
        pi1=pi1_out,
        log_lh=log_lh_out,
        trait_names=list(trait_names),
        ta=ta,
        elapsed_s=timing["total"],
        timing=timing,
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
