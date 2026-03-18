"""
simulation.py — SimPhyNI trait co-evolution simulation engine
=============================================================

Pipeline overview
-----------------
1. **Ancestral reconstruction** (``run_ancestral_reconstruction.py``)
   Runs PastML JOINT and/or MPPA reconstruction for each trait, then counts
   gains, losses, subsizes, emergence thresholds, and marginal probabilities.
   Outputs a CSV with both legacy JOINT columns and (optionally) marginal columns.

2. **Parameter normalisation** (``build_sim_params()``)
   Selects the appropriate counting and subsize columns from the ACR CSV and
   renames them to the canonical base names used by ``sim_bit()``.
   Call signature::

       trait_params = build_sim_params(df, counting='FLOW', subsize='ORIGINAL')

3. **Bit-packed simulation** (``sim_bit()``)
   Simulates gain/loss events along every branch of the tree using Poisson
   sampling, packing 64 independent trials into a single uint64 per node × trait.

4. **Result compilation** (``compres()`` via ``simulate_glrates_bit()``)
   Computes KDE-based p-values for each observed pair against its simulated
   null distribution and applies multiple-testing correction.

Counting methods
----------------
The active counting method is determined automatically by ``runSimPhyNI.py``
based on the columns present in the ACR CSV:

  ACR run with ``--uncertainty marginal`` (Snakefile default)
      → writes ``gains_flow`` + all marginal columns
      → ``runSimPhyNI.py`` selects **FLOW** counting

  ACR run with ``--uncertainty threshold`` (legacy)
      → writes only JOINT columns; ``gains_flow`` absent
      → ``runSimPhyNI.py`` falls back to **JOINT** counting

All four methods are available via ``build_sim_params(df, counting, subsize)``:

FLOW (pipeline default)
    Soft probability-flow rates: Σ max(0, P(child=1) − P(parent=1)) over
    all branches.  Active when ACR was run with ``--uncertainty marginal``
    (the Snakefile default).  Best-calibrated per ``dev/benchmark_reconstruction.py``.
    Source columns: ``gains_flow``, ``losses_flow``, marginal subsize and dist variants.

JOINT (legacy fallback)
    Discrete gain/loss events counted from the JOINT ML ancestral state
    reconstruction.  Active when ACR was run with ``--uncertainty threshold``
    or via the legacy pastml CLI pipeline.
    Source columns: ``gains``, ``losses``, ``gain_subsize``, ``loss_subsize``,
    ``dist``, ``loss_dist``.

MARKOV
    Branch-length-weighted Markov transition rates:
    Σ P(parent=0) × P(child=1) × branch_length.
    Uses the same marginal subsize/distance columns as FLOW.

ENTROPY
    FLOW rates weighted by parent certainty (1 − H(parent)).
    Uses entropy-specific subsize columns.

Subsize variants (``subsize`` parameter)
-----------------------------------------
ORIGINAL  — P-weighted subsize with IQR outlier filtering (pipeline default)
NO_FILTER — P-weighted subsize without outlier cap
THRESH    — Only branches downstream of the first emergence event

Development / benchmarking
--------------------------
All 36 counting × subsize × masking combinations are explored in
``dev/benchmark_reconstruction.py`` using ``_build_method_specs()`` +
``build_all_method_dfs()``.  The ACR CSV is produced once with
``uncertainty='marginal'``; all method variants are derived from it by column
selection via ``build_sim_params()`` — no re-running ACR per combination.
"""

from typing import List
import numpy as np
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from typing import List, Tuple, Set, Dict
from numba import njit, prange
import os


### Helper funcs


def unpack_trait_params(tp: pd.DataFrame):
    gains = np.array(tp['gains'])
    losses = np.array(tp['losses'])
    dists = np.array(tp['dist'])
    loss_dists = np.array(tp['loss_dist'])
    gain_subsize = np.array(tp['gain_subsize'])
    loss_subsize = np.array(tp['loss_subsize'])
    root_states = np.array(tp['root_state'])
    dists[dists == np.inf] = 0
    loss_dists[loss_dists == np.inf] = 0
    return gains,losses,dists,loss_dists,gain_subsize,loss_subsize,root_states


# ---------------------------------------------------------------------------
# build_sim_params: standardise columns for any counting/subsize/masking combo
# ---------------------------------------------------------------------------

_GAIN_COL = {
    'JOINT':   'gains',
    'FLOW':    'gains_flow',
    'MARKOV':  'gains_markov',
    'ENTROPY': 'gains_entropy',
}
_LOSS_COL = {
    'JOINT':   'losses',
    'FLOW':    'losses_flow',
    'MARKOV':  'losses_markov',
    'ENTROPY': 'losses_entropy',
}
_GAIN_SUB_COL = {
    ('JOINT',   'ORIGINAL'):  'gain_subsize',
    ('JOINT',   'NO_FILTER'): 'gain_subsize_nofilter',
    ('JOINT',   'THRESH'):    'gain_subsize_thresh',
    ('FLOW',    'ORIGINAL'):  'gain_subsize_marginal',
    ('FLOW',    'NO_FILTER'): 'gain_subsize_marginal_nofilter',
    ('FLOW',    'THRESH'):    'gain_subsize_marginal_thresh',
    ('MARKOV',  'ORIGINAL'):  'gain_subsize_marginal',
    ('MARKOV',  'NO_FILTER'): 'gain_subsize_marginal_nofilter',
    ('MARKOV',  'THRESH'):    'gain_subsize_marginal_thresh',
    ('ENTROPY', 'ORIGINAL'):  'gain_subsize_entropy',
    ('ENTROPY', 'NO_FILTER'): 'gain_subsize_entropy_nofilter',
    ('ENTROPY', 'THRESH'):    'gain_subsize_entropy_thresh',
}
_LOSS_SUB_COL = {k: v.replace('gain', 'loss') for k, v in _GAIN_SUB_COL.items()}

# PATH-masked variants: denominator restricted to PATH-eligible branches.
# Used by build_sim_params when masking='PATH' and counting != 'JOINT'.
# JOINT+PATH falls back to standard JOINT columns (no PATH variant for hard states).
_GAIN_COL_PATH = {
    'JOINT':   'gains',
    'FLOW':    'gains_flow_path',
    'MARKOV':  'gains_markov_path',
    'ENTROPY': 'gains_entropy_path',
}
_LOSS_COL_PATH = {
    'JOINT':   'losses',
    'FLOW':    'losses_flow_path',
    'MARKOV':  'losses_markov_path',
    'ENTROPY': 'losses_entropy_path',
}
_GAIN_SUB_COL_PATH = {
    ('JOINT',   'ORIGINAL'):  'gain_subsize',
    ('JOINT',   'NO_FILTER'): 'gain_subsize_nofilter',
    ('JOINT',   'THRESH'):    'gain_subsize_thresh',
    ('FLOW',    'ORIGINAL'):  'gain_subsize_marginal_path',
    ('FLOW',    'NO_FILTER'): 'gain_subsize_marginal_nofilter_path',
    ('FLOW',    'THRESH'):    'gain_subsize_marginal_thresh_path',
    ('MARKOV',  'ORIGINAL'):  'gain_subsize_marginal_path',
    ('MARKOV',  'NO_FILTER'): 'gain_subsize_marginal_nofilter_path',
    ('MARKOV',  'THRESH'):    'gain_subsize_marginal_thresh_path',
    ('ENTROPY', 'ORIGINAL'):  'gain_subsize_entropy_path',
    ('ENTROPY', 'NO_FILTER'): 'gain_subsize_entropy_nofilter_path',
    ('ENTROPY', 'THRESH'):    'gain_subsize_entropy_thresh_path',
}
_LOSS_SUB_COL_PATH = {k: v.replace('gain', 'loss') for k, v in _GAIN_SUB_COL_PATH.items()}

_DIST_COL = {
    'JOINT':   ('dist',         'loss_dist'),
    'FLOW':    ('dist_marginal', 'loss_dist_marginal'),
    'MARKOV':  ('dist_marginal', 'loss_dist_marginal'),
    'ENTROPY': ('dist_marginal', 'loss_dist_marginal'),
}


def build_sim_params(df: pd.DataFrame, counting: str, subsize: str,
                     no_threshold: bool = False,
                     masking: str = 'DIST') -> pd.DataFrame:
    """
    Build a standardised trait_params DataFrame for sim_bit by selecting the
    appropriate counting and subsize columns for a given method combination.

    Output always uses the base column names (gains, losses, gain_subsize,
    loss_subsize, dist, loss_dist, root_state) so sim_bit's standard unpacker
    is used regardless of which ACR columns were selected.

    Parameters
    ----------
    df            : wide-format params DataFrame produced by
                    ``run_ancestral_reconstruction.py`` (all columns present),
                    or a legacy pastml CSV (JOINT columns only).
    counting      : 'JOINT' | 'FLOW' | 'MARKOV' | 'ENTROPY'
    subsize       : 'ORIGINAL' | 'NO_FILTER' | 'THRESH'
    no_threshold  : if True, set dist = loss_dist = 0.0 (no emergence gate)
    masking       : 'DIST' | 'NONE' | 'PATH' — when 'PATH' and counting != 'JOINT',
                    selects _path-suffixed columns whose denominators are restricted
                    to PATH-eligible branches.  Default 'DIST' for backward compat.

    Column mapping reference
    ------------------------
    counting  subsize     gains col        loss col        gain_subsize col
    --------  ----------  ---------------  --------------  --------------------------
    JOINT     ORIGINAL    gains            losses          gain_subsize
    JOINT     NO_FILTER   gains            losses          gain_subsize_nofilter
    JOINT     THRESH      gains            losses          gain_subsize_thresh
    FLOW      ORIGINAL    gains_flow       losses_flow     gain_subsize_marginal
    FLOW      NO_FILTER   gains_flow       losses_flow     gain_subsize_marginal_nofilter
    FLOW      THRESH      gains_flow       losses_flow     gain_subsize_marginal_thresh
    MARKOV    ORIGINAL    gains_markov     losses_markov   gain_subsize_marginal
    MARKOV    NO_FILTER   gains_markov     losses_markov   gain_subsize_marginal_nofilter
    MARKOV    THRESH      gains_markov     losses_markov   gain_subsize_marginal_thresh
    ENTROPY   ORIGINAL    gains_entropy    losses_entropy  gain_subsize_entropy
    ENTROPY   NO_FILTER   gains_entropy    losses_entropy  gain_subsize_entropy_nofilter
    ENTROPY   THRESH      gains_entropy    losses_entropy  gain_subsize_entropy_thresh

    Distance columns (dist / loss_dist):
      JOINT   → dist, loss_dist
      FLOW / MARKOV / ENTROPY → dist_marginal, loss_dist_marginal

    Root state:
      JOINT   → root_state (int)
      others  → root_prob >= 0.5 thresholded to 0/1 (falls back to root_state
                if root_prob column absent)
    """
    src = df.set_index('gene') if 'gene' in df.columns else df

    out = pd.DataFrame(index=range(len(src)))
    if 'gene' in df.columns:
        out.insert(0, 'gene', df['gene'].values)

    use_path = (masking == 'PATH') and (counting != 'JOINT')
    _gc  = _GAIN_COL_PATH  if use_path else _GAIN_COL
    _lc  = _LOSS_COL_PATH  if use_path else _LOSS_COL
    _gsc = _GAIN_SUB_COL_PATH if use_path else _GAIN_SUB_COL
    _lsc = _LOSS_SUB_COL_PATH if use_path else _LOSS_SUB_COL

    out['gains']        = src[_gc[counting]].values
    out['losses']       = src[_lc[counting]].values
    out['gain_subsize'] = src[_gsc[(counting, subsize)]].values
    out['loss_subsize'] = src[_lsc[(counting, subsize)]].values

    dist_col, loss_dist_col = _DIST_COL[counting]
    if no_threshold:
        out['dist']      = 0.0
        out['loss_dist'] = 0.0
    else:
        out['dist']      = src[dist_col].replace([np.inf], 0.0).values
        out['loss_dist'] = src[loss_dist_col].replace([np.inf], 0.0).values

    # Root state: for marginal counting methods use root_prob thresholded at 0.5
    if counting != 'JOINT' and 'root_prob' in src.columns:
        out['root_state'] = (src['root_prob'].values >= 0.5).astype(int)
    else:
        out['root_state'] = src['root_state'].values.astype(int)

    return out


### Simulation Methods

def simulate_glrates_bit(tree, trait_params, pairs, obspairs, trials = 64, cores = -1,
                         gain_mask=None, loss_mask=None):

    sim = sim_bit(tree=tree, trait_params=trait_params, trials=64,
                  gain_mask=gain_mask, loss_mask=loss_mask)
    mappingr = dict(enumerate(trait_params.index))
    mapping = dict(zip(trait_params.index,range(len(trait_params.index))))
    pairs_index = np.vectorize(lambda key: mapping[key])(pairs)

    res = compres(sim, pairs_index, obspairs, bits = 64)

    res['first'] = res['first'].map(mappingr)
    res['second'] = res['second'].map(mappingr)
    return res


def sim_bit(tree, trait_params, trials=64, gain_mask=None, loss_mask=None):
    """
    Simulate trait evolution on a phylogenetic tree using bit-packed trials.

    Recommended call path::

        trait_params = build_sim_params(acr_df, counting='FLOW', subsize='ORIGINAL')
        lineages = sim_bit(tree, trait_params)

    ``build_sim_params`` normalises any counting/subsize combination into the
    canonical column names (gains, losses, gain_subsize, loss_subsize, dist,
    loss_dist, root_state) expected here.

    Operating modes
    ---------------
    1. **Mask mode** — ``gain_mask`` / ``loss_mask`` provided (n_nodes × n_traits bool)
       Per-node, per-trait eligibility built by ``build_path_mask()`` from
       ``run_ancestral_reconstruction.py``.  Each node is gated individually
       based on its ancestral lineage rather than a single global distance.
       Use this for the ENTROPY counting method or when fine-grained path-based
       masking is desired.

    2. **Distance mode** — no mask provided; dist / loss_dist > 0 in trait_params
       A global cumulative-distance threshold gates eligible branches: a branch
       at total depth ``d`` from the root is eligible for a gain event only if
       ``d >= dist``.  Thresholds are inferred from ACR output and stored in the
       params DataFrame by ``run_ancestral_reconstruction.py``.

    3. **No-threshold mode** — no mask; dist = loss_dist = 0 (via
       ``build_sim_params(..., no_threshold=True)``)
       Every branch is eligible from the root, equivalent to a fully homogeneous
       Poisson process with no emergence constraint.

    Parameters
    ----------
    tree         : ETE3 Tree (internal nodes must be labelled before calling)
    trait_params : DataFrame indexed by gene; canonical column names expected.
                   Prepare with ``build_sim_params(df, counting, subsize, no_threshold)``.
    trials       : ignored (always 64 for bit-packing); kept for API compatibility
    gain_mask    : optional (n_nodes, n_traits) bool array — Mask mode
    loss_mask    : optional (n_nodes, n_traits) bool array — Mask mode

    Returns
    -------
    lineages : np.ndarray (n_tips, n_traits) dtype=uint64
        Each uint64 encodes 64 independent simulated tip states as bit flags.
    """
    gains, losses, dists, loss_dists, gain_subsize, loss_subsize, root_values = \
        unpack_trait_params(trait_params)
    use_mask = (gain_mask is not None) and (loss_mask is not None)

    # Preprocess and setup
    node_map = {node: ind for ind, node in enumerate(tree.traverse())}
    num_traits = len(gains)
    num_nodes = len(node_map)
    bits = 64
    nptype = np.uint64
    sim = np.zeros((num_nodes, num_traits), dtype=nptype)
    trials = bits

    gain_rates = np.zeros_like(gains, dtype=float)
    loss_rates = np.zeros_like(losses, dtype=float)
    valid_gains = gain_subsize > 0
    valid_losses = loss_subsize > 0
    gain_rates[valid_gains] = gains[valid_gains] / gain_subsize[valid_gains]
    loss_rates[valid_losses] = losses[valid_losses] / loss_subsize[valid_losses]

    # Distance calculations
    node_dists = {}
    node_dists[tree] = tree.dist or 0
    for node in tree.traverse():
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist

    print("Simulating Trees...")
    for node in tree.traverse():

        if node.up is None:
            # Root state: hard 0/1 (build_sim_params thresholds root_prob at 0.5)
            root = root_values > 0
            root_mask = np.zeros(num_traits, dtype=bool)
            root_mask[root] = True
            full_mask_value = (1 << trials) - 1
            sim[node_map[node], root_mask] = full_mask_value
            continue

        parent = sim[node_map[node.up], :]
        node_total_dist = node_dists[node]

        if use_mask:
            # Mask mode: per-node, per-trait eligibility from entropy mask
            node_idx = node_map[node]
            applicable_traits_gains  = gain_mask[node_idx, :]
            applicable_traits_losses = loss_mask[node_idx, :]
        else:
            # Distance mode: global threshold gate
            applicable_traits_gains  = node_total_dist >= dists
            applicable_traits_losses = node_total_dist >= loss_dists

        gain_events = np.zeros((num_traits), dtype=nptype)
        loss_events = np.zeros((num_traits), dtype=nptype)
        gain_events[applicable_traits_gains] = np.packbits((np.random.poisson(node.dist * gain_rates[applicable_traits_gains, np.newaxis], (applicable_traits_gains.sum(), trials)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()
        loss_events[applicable_traits_losses] = np.packbits((np.random.poisson(node.dist * loss_rates[applicable_traits_losses, np.newaxis], (applicable_traits_losses.sum(), trials)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()

        gain_events &= ~parent
        loss_events &= parent

        updated_state = np.bitwise_or(parent, gain_events)
        updated_state = np.bitwise_and(updated_state, np.bitwise_not(loss_events))
        sim[node_map[node], :] = updated_state

    print("Completed Tree Simulation Sucessfully")

    lineages = sim[[node_map[node] for node in tree], :]
    return lineages

# Compiling results

@njit
def circular_bitshift_right(arr: np.ndarray, k: int, bits: int = 64) -> np.ndarray:
    k = k % bits
    n_rows, n_cols = arr.shape
    out = np.empty_like(arr)
    mask = 18446744073709551615 #Max integer  # integer mask, not np.uint64 — faster and correct

    for i in prange(n_rows):
        for j in range(n_cols):
            val = arr[i, j]
            right = val >> k
            left = (val << (bits - k)) & mask
            out[i, j] = np.uint64((right | left) & mask)
    
    return out

@njit
def sum_all_bits(arr, bits=64):
    n_nodes, n_traits = arr.shape
    bit_sums = np.zeros((bits, n_traits), dtype=np.float64)
    for j in range(n_traits):
        for i in range(bits):
            s = 0
            for n in range(n_nodes):
                s += (arr[n, j] >> i) & 1
            bit_sums[i, j] = s
    return bit_sums

@njit
def get_bit_sums_and_neg_sums(arr: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate bitwise sums for arr and ~arr (used for co-occurrence derivations)."""
    n_nodes = arr.shape[0]
    sum_arr = sum_all_bits(arr, bits)
    sum_neg_arr = n_nodes - sum_arr
    return sum_arr, sum_neg_arr

@njit
def compute_bitwise_cooc(tp: np.ndarray, tq: np.ndarray, bits: int = 64) -> np.ndarray:
    """
    Numba-optimized bitwise co-occurrence statistics calculation.
    Replicates the NumPy logic: computes a (bits, N_traits) matrix for each shift k, 
    then conceptually stacks them horizontally.
    """
    n_nodes, n_traits = tp.shape
    out_cols = bits * bits 
    cooc_matrix = np.empty((n_traits, out_cols), dtype=np.float64)

    sum_tp_1s, sum_tp_0s = get_bit_sums_and_neg_sums(tp, bits)
    epsilon = 1#e-2

    for k in prange(bits): 
        shifted = circular_bitshift_right(tq, k)
        sum_shifted_1s, sum_shifted_0s = get_bit_sums_and_neg_sums(shifted, bits)

        a = sum_all_bits(tp & shifted, bits)
        b = sum_tp_1s - a + epsilon 
        c = sum_shifted_1s - a
        d = sum_tp_0s - c + epsilon 

        a += epsilon
        c+= epsilon

        log_ratio_matrix = np.log((a * d) / (b * c))
        
        start_col = k * bits
        end_col = (k + 1) * bits
        
        cooc_matrix[:, start_col:end_col] = log_ratio_matrix.T

    return cooc_matrix


def compute_kde_stats(observed_value: float, simulated_values: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute KDE statistics for a single pair."""

    # kde = gaussian_kde(simulated_values, bw_method='silverman')
    # cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
    # cdf_func_syn = lambda x: kde.integrate_box_1d(x,np.inf)
    
    # kde_pval_ant = cdf_func_ant(observed_value)  # P(X ≤ observed)
    # kde_pval_syn = cdf_func_syn(observed_value) # P(X > observed)

    kde = gaussian_kde(simulated_values,bw_method='silverman')
    cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
    kde_syn = gaussian_kde(-1*simulated_values, bw_method='silverman')
    cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf,-x)

    kde_pval_ant = cdf_func_ant(observed_value)
    kde_pval_syn = cdf_func_syn(observed_value)
    
    med = np.median(simulated_values)
    q75, q25 = np.percentile(simulated_values, [75, 25])
    iqr = q75 - q25
    
    return kde_pval_ant, kde_pval_syn, med, max(iqr * 1.349,1)


def process_batch(index: int, sim_readonly: np.ndarray, pairs: np.ndarray, obspairs: np.ndarray, batch_size: int) -> Dict[str, List]:
    """
    Process a single batch of data.
    """
    pair_batch = pairs[index: index + batch_size]
    current_obspairs = obspairs[index: index + len(pair_batch)]
    
    tp = sim_readonly[:, pair_batch[:, 0]]
    tq = sim_readonly[:, pair_batch[:, 1]]


    batch_cooc = compute_bitwise_cooc(tp, tq)
    noised_batch_cooc = batch_cooc + np.random.normal(0, 1e-12, size=batch_cooc.shape)

    results = [
        compute_kde_stats(current_obspairs[i], noised_batch_cooc[i])
        for i in range(len(pair_batch))
    ]

    kde_pvals_ant, kde_pvals_syn, medians, normalization_factors = map(np.array, zip(*results))

    # Vectorized calculation of final results
    min_pvals = np.minimum(kde_pvals_syn, kde_pvals_ant)
    directions = np.where(kde_pvals_ant < kde_pvals_syn, -1, 1)
    effect_sizes = (current_obspairs - medians) / normalization_factors

    batch_res = {
        "pair": [tuple(p) for p in pair_batch],
        "first": pair_batch[:, 0].tolist(),
        "second": pair_batch[:, 1].tolist(),
        "p-value": min_pvals.tolist(),
        "direction": directions.tolist(),
        "effect size": effect_sizes.tolist(),
    }

    return batch_res

def compres(sim: np.ndarray, pairs: np.ndarray, obspairs: np.ndarray, batch_size: int = 1000, bits: int = 64, cores: int = -1) -> pd.DataFrame:
    """
    Compile KDE results asynchronously using parallel batch processing.
    Optimized for time (no nested parallelism) and memory (read-only array setup).
    """
    res: Dict[str, List] = {
        "pair": [], "first": [], "second": [], 
        "direction": [], "p-value": [], "effect size": []
    }

    sim = np.asarray(sim, order="C") 
    sim.setflags(write=False) 

    num_pairs = len(pairs)
    batch_size = min(int(np.ceil(num_pairs/(os.cpu_count() or 1))),batch_size)
    batch_indices = range(0, num_pairs, batch_size)

    print(f"Processing Batches, Total: {num_pairs//batch_size + 1}")

    batch_results = Parallel(n_jobs=cores, verbose=10)(
        delayed(process_batch)(index, sim, pairs, obspairs, batch_size) 
        for index in batch_indices
    )

    print("Aggregating Results...")

    # Merge batch results
    for batch_res in batch_results:
        for key in res.keys():
            res[key].extend(batch_res[key])

    return pd.DataFrame.from_dict(res)