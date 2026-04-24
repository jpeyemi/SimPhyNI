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
import math
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


def _chunk_info(trials: int) -> Tuple[int, int]:
    """Return (n_chunks, last_chunk_bits) for a given trial count.

    n_chunks = ceil(trials / 64); last_chunk_bits is the number of valid bits
    in the final uint64 word (always in [1, 64]).  All preceding chunks use
    all 64 bits.  Space complexity scales with n_chunks, so trials in
    [k*64+1, (k+1)*64] all use the same n_chunks = k+1 words.
    """
    n_chunks = math.ceil(trials / 64)
    last_chunk_bits = trials - 64 * (n_chunks - 1)   # 1 .. 64
    return n_chunks, last_chunk_bits


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
    'JOINTP':  'gains',   # same count as JOINT; only subsize differs
    'FLOW':    'gains_flow',
    'MARKOV':  'gains_markov',
    'ENTROPY': 'gains_entropy',
}
_LOSS_COL = {
    'JOINT':   'losses',
    'JOINTP':  'losses',
    'FLOW':    'losses_flow',
    'MARKOV':  'losses_markov',
    'ENTROPY': 'losses_entropy',
}
_GAIN_SUB_COL = {
    ('JOINT',   'ORIGINAL'):  'gain_subsize',
    ('JOINT',   'NO_FILTER'): 'gain_subsize_nofilter',
    ('JOINT',   'THRESH'):    'gain_subsize_thresh',
    # JOINTP: parent-state subsize (gain eligible when parent==0)
    ('JOINTP',  'ORIGINAL'):  'gain_subsize_p',
    ('JOINTP',  'NO_FILTER'): 'gain_subsize_nofilter_p',
    ('JOINTP',  'THRESH'):    'gain_subsize_thresh_p',
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
# Used by build_sim_params when masking='PATH' and counting not in ('JOINT','JOINTP').
# JOINT/JOINTP+PATH falls back to standard columns (PATH mask applied at sim level).
_GAIN_COL_PATH = {
    'JOINT':   'gains',
    'JOINTP':  'gains',
    'FLOW':    'gains_flow_path',
    'MARKOV':  'gains_markov_path',
    'ENTROPY': 'gains_entropy_path',
}
_LOSS_COL_PATH = {
    'JOINT':   'losses',
    'JOINTP':  'losses',
    'FLOW':    'losses_flow_path',
    'MARKOV':  'losses_markov_path',
    'ENTROPY': 'losses_entropy_path',
}
_GAIN_SUB_COL_PATH = {
    ('JOINT',   'ORIGINAL'):  'gain_subsize',
    ('JOINT',   'NO_FILTER'): 'gain_subsize_nofilter',
    ('JOINT',   'THRESH'):    'gain_subsize_thresh',
    ('JOINTP',  'ORIGINAL'):  'gain_subsize_p',
    ('JOINTP',  'NO_FILTER'): 'gain_subsize_nofilter_p',
    ('JOINTP',  'THRESH'):    'gain_subsize_thresh_p',
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
    'JOINTP':  ('dist',         'loss_dist'),
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

    use_path = (masking == 'PATH') and (counting not in ('JOINT', 'JOINTP'))
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
    if counting not in ('JOINT', 'JOINTP') and 'root_prob' in src.columns:
        out['root_state'] = (src['root_prob'].values >= 0.5).astype(int)
    else:
        out['root_state'] = src['root_state'].values.astype(int)

    # Carry through raw JOINT discrete counts and tip prevalence as auxiliary
    # columns for downstream flagging (prefix _raw_ to distinguish from
    # simulation params).  These are optional: if the source columns are absent
    # (legacy CSV without JOINT counts) the auxiliary columns are omitted and
    # flag_uncalibratable_traits falls back to the canonical FLOW values.
    for raw_col, raw_alias in [('gains', '_raw_gains'), ('losses', '_raw_losses'),
                                ('count',  '_raw_count')]:
        if raw_col in src.columns:
            out[raw_alias] = src[raw_col].values

    return out


# ---------------------------------------------------------------------------
# flag_uncalibratable_traits: data-driven detection of miscalibrated nulls
# ---------------------------------------------------------------------------

def flag_uncalibratable_traits(
    params: pd.DataFrame,
    subsize_ratio_percentile: float = 99.0,
    rate_iqr_multiplier: float = 3.0,
    max_sparse_events: int = 5,
    max_prevalence_fraction: float = 0.80,
) -> pd.DataFrame:
    """
    Identify traits whose Poisson null distribution will be systematically
    miscalibrated due to sparse events combined with highly asymmetric
    eligible regions.

    All thresholds are derived from the input dataset so the method
    generalises to any tree size or branch-length scale.

    Two structural failure modes are detected, both gated on sparse JOINT
    event counts AND intermediate-to-low tip prevalence.  High-prevalence
    traits (present in >``max_prevalence_fraction`` of tips) are excluded
    from flagging even when they show rate asymmetry: for those traits a
    large rate on a small eligible region correctly simulates high prevalence,
    so the null is well-calibrated.

    **Failure mode 1 — Subsize asymmetry with sparse events**
        The eligible region for gains is orders of magnitude larger than for
        losses (or vice versa).  The rate estimated from the small region is
        then applied across the large simulation region, catastrophically
        inflating or deflating the effective rate.
        Example: perR, yfjM — one loss on a 0.009-unit branch → loss_rate 113,
        which fires ~600× on the full 5.5-unit simulation region.
        Detected via the dimensionless ratio
        ``max(gain_subsize, loss_subsize) / min(gain_subsize, loss_subsize)``
        exceeding the ``subsize_ratio_percentile`` th percentile of the dataset,
        gated on JOINT gains + losses ≤ ``max_sparse_events``.

    **Failure mode 2 — Single clade-specific gain with no loss process**
        The trait was acquired once in a specific large clade and never lost.
        The simulation places that gain uniformly across the entire eligible
        region (proportional to branch length), almost always landing on tiny
        terminal branches and under-predicting prevalence by 7–43×.
        Example: espZ — one gain at an internal node with 61 descendants,
        but the MRCA branch is only 0.16 % of eligible branch length.
        Detected when JOINT gains == 1 AND JOINT losses == 0 AND the subsize
        ratio exceeds the dataset median × 3.

    Parameters
    ----------
    params : pd.DataFrame
        Canonical params DataFrame from ``build_sim_params()``.  Must contain
        columns gene, gains, losses, gain_subsize, loss_subsize.  The auxiliary
        columns ``_raw_gains``, ``_raw_losses``, ``_raw_count`` written by
        ``build_sim_params`` from the ACR CSV are used when present (JOINT
        discrete counts and tip prevalence); the function falls back to the
        canonical columns otherwise.
    subsize_ratio_percentile : float
        Dataset percentile cutoff for the subsize asymmetry flag (default 99.0).
        Traits above this percentile AND satisfying the sparse/prevalence gate
        are flagged.
    rate_iqr_multiplier : float
        Reserved for future use; currently unused in the simplified two-flag
        design.  Kept for API stability.
    max_sparse_events : int
        Maximum total JOINT event count (gains + losses) for a trait to be
        considered sparse (default 5).  Only sparse traits are eligible for
        flagging.
    max_prevalence_fraction : float
        Maximum tip prevalence fraction for a trait to be eligible for
        flagging (default 0.80).  High-prevalence traits are excluded because
        their null distributions are well-calibrated even with asymmetric rates.
        Requires ``_raw_count`` in params; if absent, no prevalence gate is
        applied.

    Returns
    -------
    flagged : pd.DataFrame
        Rows for each flagged trait.  Columns: ``gene``, ``gains``,
        ``losses``, ``gain_subsize``, ``loss_subsize``, ``gain_rate``,
        ``loss_rate``, ``subsize_ratio``, ``flag_subsize``,
        ``flag_single_gain``, ``flag_reasons``.
        Empty DataFrame if no traits are flagged.
    """
    eps = 1e-12

    df = params.reset_index() if params.index.name == 'gene' else params.copy()
    if 'gene' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'gene'})

    base_cols = ['gene', 'gains', 'losses', 'gain_subsize', 'loss_subsize']
    aux_cols  = [c for c in ('_raw_gains', '_raw_losses', '_raw_count')
                 if c in df.columns]
    df = df[base_cols + aux_cols].copy()

    # Use raw JOINT discrete counts for stratification/gating when available
    raw_gains  = df['_raw_gains'].round()  if '_raw_gains'  in df.columns else df['gains'].round()
    raw_losses = df['_raw_losses'].round() if '_raw_losses' in df.columns else df['losses'].round()

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    df['gain_rate'] = np.where(df['gain_subsize'] > eps,
                               df['gains'] / df['gain_subsize'], 0.0)
    df['loss_rate'] = np.where(df['loss_subsize'] > eps,
                               df['losses'] / df['loss_subsize'], 0.0)
    df['subsize_ratio'] = (
        np.maximum(df['gain_subsize'], df['loss_subsize']) /
        (np.minimum(df['gain_subsize'], df['loss_subsize']) + eps)
    )

    # ------------------------------------------------------------------ #
    # Sparse-event and prevalence gates (applied to both failure modes)
    # ------------------------------------------------------------------ #
    sparse = (raw_gains + raw_losses) <= max_sparse_events

    if '_raw_count' in df.columns:
        n_tips_est   = df['_raw_count'].max()
        prevalence   = df['_raw_count'] / n_tips_est
        low_prev     = prevalence < max_prevalence_fraction
    else:
        low_prev = pd.Series(True, index=df.index)   # no prevalence info — flag all

    eligible = sparse & low_prev

    # ------------------------------------------------------------------ #
    # Flag 1 — subsize asymmetry among sparse, intermediate-prevalence traits
    # ------------------------------------------------------------------ #
    ratio_threshold = df['subsize_ratio'].quantile(subsize_ratio_percentile / 100.0)
    df['flag_subsize'] = eligible & (df['subsize_ratio'] > ratio_threshold)

    # ------------------------------------------------------------------ #
    # Flag 2 — single gain, no loss, subsize asymmetry above dataset median
    # (clade-specific acquisition: gain placed randomly on eligible tree)
    # ------------------------------------------------------------------ #
    median_ratio = df['subsize_ratio'].median()
    df['flag_single_gain'] = (
        low_prev &
        (raw_gains == 1) &
        (raw_losses == 0) &
        (df['subsize_ratio'] > median_ratio * 3)
    )

    # ------------------------------------------------------------------ #
    # Combine and annotate reasons
    # ------------------------------------------------------------------ #
    flagged_mask = df['flag_subsize'] | df['flag_single_gain']

    def _reasons(row):
        r = []
        if row['flag_subsize']:
            r.append(
                f"subsize_ratio={row['subsize_ratio']:.1f}>{ratio_threshold:.1f} "
                f"(>{subsize_ratio_percentile:.0f}th pctile) with "
                f"JOINT events={int(raw_gains[row.name]+raw_losses[row.name])}"
            )
        if row['flag_single_gain']:
            r.append(
                f"single_clade_gain: JOINT gains=1, losses=0, "
                f"subsize_ratio={row['subsize_ratio']:.1f}>{median_ratio*3:.1f}"
            )
        return '; '.join(r)

    flagged = df[flagged_mask].copy()
    if not flagged.empty:
        flagged['flag_reasons'] = flagged.apply(_reasons, axis=1)
    else:
        flagged['flag_reasons'] = pd.Series(dtype=str)

    return flagged.reset_index(drop=True)


### Simulation Methods

def simulate_glrates_bit(tree, trait_params, pairs, obspairs, trials = 64, cores = -1,
                         gain_mask=None, loss_mask=None, gamma=False):

    sim = sim_bit(tree=tree, trait_params=trait_params, trials=trials,
                  gain_mask=gain_mask, loss_mask=loss_mask, gamma=gamma)
    mappingr = dict(enumerate(trait_params.index))
    mapping = dict(zip(trait_params.index,range(len(trait_params.index))))
    pairs_index = np.vectorize(lambda key: mapping[key])(pairs)

    res = compres(sim, pairs_index, obspairs, bits=trials)

    res['first'] = res['first'].map(mappingr)
    res['second'] = res['second'].map(mappingr)
    return res


def sim_bit(tree, trait_params, trials=64, gain_mask=None, loss_mask=None, gamma=False):
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
    trials       : number of independent bit-packed trials.  Must be >= 1.
                   Space scales with ceil(trials/64): trials in [k*64+1, (k+1)*64]
                   all use n_chunks = k+1 uint64 words per (node, trait) cell.
    gain_mask    : optional (n_nodes, n_traits) bool array — Mask mode
    loss_mask    : optional (n_nodes, n_traits) bool array — Mask mode
    gamma        : if True, sample each trial's rate independently from the Jeffreys
                   posterior Gamma(count + 0.5, 1/subsize) rather than using the
                   point estimate count/subsize.  This widens the null distribution
                   for traits with few events (sparse data) while having negligible
                   effect when counts are large.  Each trial receives its own rate.

    Returns
    -------
    lineages : np.ndarray (n_tips, n_traits, n_chunks) dtype=uint64
        n_chunks = ceil(trials/64).  Each uint64 encodes 64 independent trial
        states as bit flags; the last chunk uses only (trials - 64*(n_chunks-1))
        of its 64 bits (the remaining high bits are always 0).
    """
    gains, losses, dists, loss_dists, gain_subsize, loss_subsize, root_values = \
        unpack_trait_params(trait_params)
    use_mask = (gain_mask is not None) and (loss_mask is not None)

    # Preprocess and setup
    node_map = {node: ind for ind, node in enumerate(tree.traverse())}
    num_traits = len(gains)
    num_nodes = len(node_map)
    bits = 64                                        # bits per uint64 word
    nptype = np.uint64
    n_chunks, last_chunk_bits = _chunk_info(trials)  # ceil(trials/64), valid bits in last chunk

    # Each (node, trait) cell is n_chunks uint64 words; bit c*64+i of the
    # logical trial vector lives at sim[node, trait, c] bit i.
    sim = np.zeros((num_nodes, num_traits, n_chunks), dtype=nptype)

    valid_gains = gain_subsize > 0
    valid_losses = loss_subsize > 0

    # Masks for the last chunk: only last_chunk_bits low bits are valid.
    FULL_MASK = np.uint64(18446744073709551615)       # all 64 bits set
    last_mask = FULL_MASK if last_chunk_bits == 64 \
        else (np.uint64(1) << np.uint64(last_chunk_bits)) - np.uint64(1)

    if gamma:
        # Sample one rate per trait per trial from the Jeffreys posterior:
        #   rate | data ~ Gamma(count + 0.5, scale=1/subsize)
        # Shape: (n_traits, trials) — each trial uses an independent draw.
        gain_rates = np.zeros((num_traits, trials), dtype=float)
        loss_rates = np.zeros((num_traits, trials), dtype=float)
        gain_rates[valid_gains] = np.random.gamma(
            (gains[valid_gains] + 0.5)[:, np.newaxis],
            (1.0 / gain_subsize[valid_gains])[:, np.newaxis],
            size=(valid_gains.sum(), trials),
        )
        loss_rates[valid_losses] = np.random.gamma(
            (losses[valid_losses] + 0.5)[:, np.newaxis],
            (1.0 / loss_subsize[valid_losses])[:, np.newaxis],
            size=(valid_losses.sum(), trials),
        )
    else:
        gain_rates = np.zeros(num_traits, dtype=float)
        loss_rates = np.zeros(num_traits, dtype=float)
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
            for c in range(n_chunks):
                chunk_val = FULL_MASK if c < n_chunks - 1 else last_mask
                sim[node_map[node], root, c] = chunk_val
            continue

        parent = sim[node_map[node.up], :, :]        # (n_traits, n_chunks)
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

        gain_events = np.zeros((num_traits, n_chunks), dtype=nptype)
        loss_events = np.zeros((num_traits, n_chunks), dtype=nptype)
        n_gain = applicable_traits_gains.sum()
        n_loss = applicable_traits_losses.sum()

        # Generate events chunk by chunk.  Each chunk packs 64 trials (last
        # chunk packs only last_chunk_bits trials, zero-padded to 64 for packbits).
        # gamma=True: rates are (n_traits, trials) — slice per chunk.
        # gamma=False: scalar rate broadcasts to all chunk_trials via np.newaxis.
        for c in range(n_chunks):
            chunk_start  = c * bits
            chunk_trials = bits if c < n_chunks - 1 else last_chunk_bits

            if gamma:
                gr = gain_rates[applicable_traits_gains,  chunk_start:chunk_start + chunk_trials]
                lr = loss_rates[applicable_traits_losses, chunk_start:chunk_start + chunk_trials]
            else:
                gr = gain_rates[applicable_traits_gains,  np.newaxis]
                lr = loss_rates[applicable_traits_losses, np.newaxis]

            g_samp = (np.random.poisson(node.dist * gr, (n_gain, chunk_trials)) > 0).astype(np.uint8)
            l_samp = (np.random.poisson(node.dist * lr, (n_loss, chunk_trials)) > 0).astype(np.uint8)

            if chunk_trials < bits:
                # Zero-pad to exactly 64 columns so packbits produces one uint64
                g_samp = np.pad(g_samp, ((0, 0), (0, bits - chunk_trials)), mode='constant')
                l_samp = np.pad(l_samp, ((0, 0), (0, bits - chunk_trials)), mode='constant')

            gain_events[applicable_traits_gains,  c] = (
                np.packbits(g_samp, axis=-1, bitorder='little').view(nptype).flatten())
            loss_events[applicable_traits_losses, c] = (
                np.packbits(l_samp, axis=-1, bitorder='little').view(nptype).flatten())

        gain_events &= ~parent
        loss_events &= parent

        updated_state = np.bitwise_or(parent, gain_events)
        updated_state = np.bitwise_and(updated_state, np.bitwise_not(loss_events))
        sim[node_map[node], :, :] = updated_state

    print("Completed Tree Simulation Sucessfully")

    lineages = sim[[node_map[node] for node in tree], :, :]
    return lineages

# Compiling results

@njit
def multi_word_circular_shift(arr: np.ndarray, k: int, total_trials: int, bits: int = 64) -> np.ndarray:
    """Right circular rotation of a total_trials-bit value by k positions.

    arr shape: (n_rows, n_cols, n_chunks), dtype uint64.
    Each (row, col) cell stores a total_trials-bit logical value spread across
    n_chunks uint64 words.  Word c holds virtual bits [c*bits .. c*bits + n_bits_c - 1]
    in its low-order positions; the last word may have fewer than bits valid bits.

    A rotation by k maps output virtual bit i to input virtual bit (i+k) % total_trials.
    Each output word is assembled from at most two consecutive source words.
    """
    n_rows, n_cols, n_chunks = arr.shape
    out = np.zeros_like(arr)
    k = k % total_trials
    last_chunk_bits = total_trials - bits * (n_chunks - 1)   # 1 .. bits

    for row in prange(n_rows):
        for col in range(n_cols):
            for c in range(n_chunks):
                # Output chunk c covers virtual bits [c*bits .. c*bits + n_bits_c - 1]
                n_bits_c = bits if c < n_chunks - 1 else last_chunk_bits
                out_start = c * bits

                # Source for output bit 0 of this chunk: input virtual bit (out_start+k)%total_trials
                src_start = (out_start + k) % total_trials
                src_chunk = src_start // bits
                src_bit   = src_start % bits

                # How many valid bits remain in the source chunk from src_bit onward
                src_chunk_size = bits if src_chunk < n_chunks - 1 else last_chunk_bits
                src_capacity   = src_chunk_size - src_bit

                if src_capacity >= n_bits_c:
                    # All n_bits_c source bits live in one chunk
                    val = arr[row, col, src_chunk] >> np.uint64(src_bit)
                    if n_bits_c < 64:
                        val &= (np.uint64(1) << np.uint64(n_bits_c)) - np.uint64(1)
                    out[row, col, c] = val
                else:
                    # Source bits split across two chunks (wrapping within total_trials)
                    n_lo = src_capacity
                    n_hi = n_bits_c - n_lo

                    lo = arr[row, col, src_chunk] >> np.uint64(src_bit)
                    if n_lo < 64:
                        lo &= (np.uint64(1) << np.uint64(n_lo)) - np.uint64(1)

                    next_chunk = (src_chunk + 1) % n_chunks
                    hi = arr[row, col, next_chunk]
                    if n_hi < 64:
                        hi &= (np.uint64(1) << np.uint64(n_hi)) - np.uint64(1)

                    out[row, col, c] = lo | (hi << np.uint64(n_lo))

    return out

@njit
def sum_all_bits(arr, total_trials, bits=64):
    """Count set bits for each of the total_trials trial slots across all nodes.

    arr shape: (n_nodes, n_traits, n_chunks), dtype uint64.
    Returns (total_trials, n_traits) float64 — bit_sums[t, j] is the number of
    nodes where trial t is set for trait j.
    """
    n_nodes, n_traits, n_chunks = arr.shape
    last_chunk_bits = total_trials - bits * (n_chunks - 1)
    bit_sums = np.zeros((total_trials, n_traits), dtype=np.float64)
    for j in range(n_traits):
        for c in range(n_chunks):
            n_bits_c = bits if c < n_chunks - 1 else last_chunk_bits
            for i in range(n_bits_c):
                trial_idx = c * bits + i
                s = np.uint64(0)
                for n in range(n_nodes):
                    s += (arr[n, j, c] >> np.uint64(i)) & np.uint64(1)
                bit_sums[trial_idx, j] = np.float64(s)
    return bit_sums

@njit
def get_bit_sums_and_neg_sums(arr: np.ndarray, total_trials: int, bits: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate bitwise sums for arr and ~arr (used for co-occurrence derivations)."""
    n_nodes = arr.shape[0]
    sum_arr = sum_all_bits(arr, total_trials, bits)
    sum_neg_arr = n_nodes - sum_arr
    return sum_arr, sum_neg_arr

@njit
def compute_bitwise_cooc(tp: np.ndarray, tq: np.ndarray, total_trials: int = 64, bits: int = 64) -> np.ndarray:
    """
    Numba-optimized bitwise co-occurrence statistics calculation.

    tp / tq shape: (n_nodes, n_traits, n_chunks), dtype uint64.
    For each rotation k in [0, total_trials) the shifted tq is ANDed with tp to
    build a 2×2 contingency table per trial slot, giving a log-odds ratio matrix
    of shape (total_trials, n_traits).  The total_trials such matrices are
    concatenated horizontally: output shape (n_traits, total_trials * total_trials).
    """
    n_nodes, n_traits, n_chunks = tp.shape
    out_cols = total_trials * total_trials
    cooc_matrix = np.empty((n_traits, out_cols), dtype=np.float64)

    sum_tp_1s, sum_tp_0s = get_bit_sums_and_neg_sums(tp, total_trials, bits)
    epsilon = np.float64(1)

    for k in prange(total_trials):
        shifted = multi_word_circular_shift(tq, k, total_trials, bits)

        # Element-wise AND across all chunks
        and_arr = np.empty_like(tp)
        for n in range(n_nodes):
            for j in range(n_traits):
                for c in range(n_chunks):
                    and_arr[n, j, c] = tp[n, j, c] & shifted[n, j, c]

        sum_shifted_1s, sum_shifted_0s = get_bit_sums_and_neg_sums(shifted, total_trials, bits)

        a = sum_all_bits(and_arr, total_trials, bits)
        b = sum_tp_1s - a + epsilon
        c_cont = sum_shifted_1s - a
        d = sum_tp_0s - c_cont + epsilon

        a += epsilon
        c_cont += epsilon

        log_ratio_matrix = np.log((a * d) / (b * c_cont))

        start_col = k * total_trials
        end_col   = (k + 1) * total_trials

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


def process_batch(index: int, sim_readonly: np.ndarray, pairs: np.ndarray, obspairs: np.ndarray, batch_size: int, total_trials: int = 64) -> Dict[str, List]:
    """
    Process a single batch of data.
    """
    pair_batch = pairs[index: index + batch_size]
    current_obspairs = obspairs[index: index + len(pair_batch)]

    # sim_readonly shape: (n_nodes, n_traits, n_chunks)
    tp = sim_readonly[:, pair_batch[:, 0], :]
    tq = sim_readonly[:, pair_batch[:, 1], :]

    batch_cooc = compute_bitwise_cooc(tp, tq, total_trials)
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

    Parameters
    ----------
    bits : total number of trials (= ``trials`` passed to ``sim_bit``).
           Determines the output width of ``compute_bitwise_cooc``.
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
        delayed(process_batch)(index, sim, pairs, obspairs, batch_size, bits)
        for index in batch_indices
    )

    print("Aggregating Results...")

    # Merge batch results
    for batch_res in batch_results:
        for key in res.keys():
            res[key].extend(batch_res[key])

    return pd.DataFrame.from_dict(res)