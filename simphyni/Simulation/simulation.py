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

from typing import List, Tuple, Set, Dict
import math
import numpy as np
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
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
        out['root_prob'] = src['root_prob'].values
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
# build_clade_mask: MRCA-based eligibility from observed leaf distributions
# ---------------------------------------------------------------------------

def build_clade_mask(
    tree,
    obsdf: pd.DataFrame,
    gene_order: list,
    root_states,
) -> tuple:
    """
    Build (n_nodes, n_traits) boolean gain/loss eligibility masks from observed
    leaf binary data and root states.

    For each trait the "minority" leaves are those whose observed state differs
    from the root state:
      - root_state = 0 → minority leaves carry the trait (value 1)
      - root_state = 1 → minority leaves lack the trait (value 0)

    The MRCA of the minority leaves is found and every node in that subtree is
    marked eligible for both gains and losses.  This gives the simulation enough
    eligible branch length to reproduce realistic prevalence while restricting
    events to the clade where the trait is actually observed.

    Also returns mrca_bl (shape n_traits,): total branch length of each MRCA
    subtree, to be used as gain_subsize / loss_subsize so that the Poisson rate
    (count / subsize) is calibrated to the actual eligible region.

    Parameters
    ----------
    tree       : ETE3 Tree (same object used by sim_bit)
    obsdf      : binary leaf × trait DataFrame (rows = leaf names, cols = genes)
    gene_order : list of gene names; determines mask column order
    root_states: Series or dict mapping gene → 0/1 root state

    Returns
    -------
    gain_mask : (n_nodes, n_traits) bool array
    loss_mask : (n_nodes, n_traits) bool array — identical to gain_mask
    mrca_bl   : (n_traits,) float array — eligible subtree branch length
    """
    node_list = list(tree.traverse())
    n_nodes   = len(node_list)
    n_traits  = len(gene_order)
    node_idx  = {node: i for i, node in enumerate(node_list)}
    all_leaf_names = {n.name for n in tree.get_leaves()}

    gain_mask = np.zeros((n_nodes, n_traits), dtype=bool)
    loss_mask = np.zeros((n_nodes, n_traits), dtype=bool)
    mrca_bl   = np.zeros(n_traits, dtype=float)

    for t_idx, gene in enumerate(gene_order):
        if hasattr(root_states, 'loc'):
            root_state = int(root_states.loc[gene])
        else:
            root_state = int(root_states[gene])
        minority_val = 1 - root_state   # state that differs from root

        if gene not in obsdf.columns:
            # No observed data → whole tree eligible (conservative fallback)
            gain_mask[:, t_idx] = True
            loss_mask[:, t_idx] = True
            mrca_bl[t_idx] = sum(n.dist for n in node_list if not n.is_root())
            continue

        minority_leaves = [
            n for n in tree.get_leaves()
            if n.name in all_leaf_names
            and n.name in obsdf.index
            and int(obsdf.at[n.name, gene]) == minority_val
        ]

        if not minority_leaves:
            # Trait matches root state everywhere → no transitions; masks stay False
            continue

        mrca = (
            tree.get_common_ancestor(minority_leaves)
            if len(minority_leaves) > 1
            else minority_leaves[0]
        )

        subtree_bl = 0.0
        for node in mrca.traverse():
            idx = node_idx[node]
            gain_mask[idx, t_idx] = True
            loss_mask[idx, t_idx] = True
            if not node.is_root():
                subtree_bl += node.dist
        mrca_bl[t_idx] = subtree_bl

    return gain_mask, loss_mask, mrca_bl


# ---------------------------------------------------------------------------
# build_phylogroup_mask: restrict events to phylogroups where a trait occurs
# ---------------------------------------------------------------------------

def phylogroups(tree, depth_frac: float = 0.2):
    """Partition the tree into phylogroups by cutting at a fraction of its depth.

    A phylogroup is the subtree below each branch that crosses depth
    ``depth_frac * max_root_to_tip``.  Tips shallower than the cut form their
    own group.  Returns (node -> group id or -1 for backbone nodes, n_groups).
    """
    depth = {}
    for n in tree.traverse("preorder"):
        depth[n] = 0.0 if n.up is None else depth[n.up] + n.dist
    cut = depth_frac * max(depth[l] for l in tree.get_leaves())
    group = {}
    n_groups = 0
    for n in tree.traverse("preorder"):
        if n.up is None:
            group[n] = -1
        elif group[n.up] >= 0:
            group[n] = group[n.up]
        elif depth[n] >= cut or n.is_leaf():
            group[n] = n_groups
            n_groups += 1
        else:
            group[n] = -1
    return group, n_groups


def build_phylogroup_mask(tree, obsdf: pd.DataFrame, gene_order: list, root_states,
                          depth_frac: float = 0.2):
    """(n_nodes, n_traits) gain/loss eligibility restricted to occupied phylogroups.

    For each trait, a phylogroup is "occupied" if any of its tips carries the
    non-root state.  Events are allowed only on branches inside occupied
    phylogroups (including each group's stem branch).  Backbone branches above
    the cut are allowed only when every phylogroup is occupied, in which case
    the trait is unrestricted.

    This encodes that some traits never enter some lineages (e.g. genes
    confined to one species or phylogroup of a multi-species tree) while
    still letting the null move the trait anywhere within those lineages.

    Returns (gain_mask, loss_mask, allowed_bl) where allowed_bl is the total
    branch length of eligible branches, used as the rate denominator.
    """
    node_list = list(tree.traverse())
    group, n_groups = phylogroups(tree, depth_frac)
    node_group = np.array([group[n] for n in node_list])
    bl = np.array([0.0 if n.up is None else n.dist for n in node_list])
    leaves = [n for n in node_list if n.is_leaf()]
    leaf_group = np.array([group[l] for l in leaves])
    leaf_names = [l.name for l in leaves]
    mask = np.zeros((len(node_list), len(gene_order)), dtype=bool)
    allowed_bl = np.zeros(len(gene_order))
    for t, gene in enumerate(gene_order):
        rs = int(root_states.loc[gene] if hasattr(root_states, "loc") else root_states[gene])
        if gene not in obsdf.columns:
            mask[:, t] = True
        else:
            vals = obsdf[gene].reindex(leaf_names).to_numpy(dtype=float)
            hit = np.isfinite(vals) & (vals.astype(float) != rs)
            occupied = np.unique(leaf_group[hit])
            if len(occupied) >= n_groups:
                mask[:, t] = True
            else:
                mask[:, t] = np.isin(node_group, occupied)
        allowed_bl[t] = bl[mask[:, t]].sum()
    return mask, mask.copy(), allowed_bl


def adaptive_phylogroup_mask(tree, lineages, trials, obsdf: pd.DataFrame, gene_order: list,
                             root_states, depth_frac: float = 0.2, alpha: float = 0.05,
                             block: int = 256):
    """Posterior-predictive phylogroup restriction.

    ``lineages`` is an *unrestricted* null simulation (n_tips, n_traits,
    n_chunks) for ``gene_order``, tips in ``tree`` leaf order.  For each trait
    and each phylogroup in which the trait's non-root state is NOT observed,
    compute the fraction of simulated replicates that also leave that group
    without the non-root state.  If that fraction is below ``alpha`` the
    unrestricted model would almost always have put the trait there, so the
    observed absence is evidence the lineage is closed to it: the group is
    excluded.  Groups whose emptiness is plausible under the null (e.g. for
    sparse traits) stay eligible, so the restriction only bites when the data
    support it.

    Returns (gain_mask, loss_mask, eligible_frac, n_excluded) where
    eligible_frac is the share of total branch length that remains eligible
    (used to rescale the rate denominator) and n_excluded counts excluded
    groups per trait.
    """
    node_list = list(tree.traverse())
    group, n_groups = phylogroups(tree, depth_frac)
    node_group = np.array([group[n] for n in node_list])
    bl = np.array([0.0 if n.up is None else n.dist for n in node_list])
    leaves = list(tree)                                 # same order as sim_bit lineages
    leaf_group = np.array([group[l] for l in leaves])
    leaf_names = [l.name for l in leaves]
    G = np.zeros((len(leaves), n_groups), dtype=np.float32)
    G[np.arange(len(leaves)), leaf_group] = 1.0

    rs = np.array([int(root_states.loc[g] if hasattr(root_states, "loc") else root_states[g])
                   for g in gene_order])
    n_traits = len(gene_order)
    mask = np.ones((len(node_list), n_traits), dtype=bool)
    eligible_frac = np.ones(n_traits)
    n_excluded = np.zeros(n_traits, dtype=int)
    total_bl = bl.sum()
    for s in range(0, n_traits, block):
        e = min(n_traits, s + block)
        bits = np.unpackbits(lineages[:, s:e, :].view(np.uint8), axis=-1, bitorder="little")
        bits = bits.reshape(len(leaves), e - s, -1)[:, :, :trials].astype(np.float32)
        for j in range(e - s):
            t = s + j
            gene = gene_order[t]
            nonroot = bits[:, j, :] if rs[t] == 0 else 1.0 - bits[:, j, :]     # tips x trials
            per_group = G.T @ nonroot                                           # groups x trials
            p_empty = (per_group == 0).mean(axis=1)                             # groups
            if gene in obsdf.columns:
                v = obsdf[gene].reindex(leaf_names).to_numpy(dtype=float)
                hit = np.isfinite(v) & (v != rs[t])
                occupied = np.zeros(n_groups, bool)
                occupied[np.unique(leaf_group[hit])] = True
            else:
                occupied = np.ones(n_groups, bool)
            excluded = (~occupied) & (p_empty < alpha)
            if excluded.any():
                allowed = ~excluded
                mask[:, t] = (node_group >= 0) & allowed[np.maximum(node_group, 0)]
                eligible_frac[t] = bl[mask[:, t]].sum() / total_bl if total_bl > 0 else 1.0
                n_excluded[t] = int(excluded.sum())
    return mask, mask.copy(), eligible_frac, n_excluded


# ---------------------------------------------------------------------------
# flag_uncalibratable_traits: data-driven detection of miscalibrated nulls
# ---------------------------------------------------------------------------

def flag_uncalibratable_traits(
    params: pd.DataFrame,
    subsize_ratio_percentile: float = 99.0,
    rate_iqr_multiplier: float = 3.0,
    max_sparse_events: int = 5,
    max_prevalence_fraction: float = 0.80,
    n_tips: int | None = None,
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
    n_tips : int, optional
        Number of tips, used as the prevalence denominator.  Falls back to
        the largest trait count in ``params`` when not given (which
        overestimates prevalence when no trait is near fixation).

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
        n_tips_est   = n_tips if n_tips else df['_raw_count'].max()
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

def condition_on_prevalence(lineages, target_prev, trials, seed=None, block=256):
    """Keep, per trait, the ``trials`` simulated replicates whose tip prevalence
    is closest to the observed prevalence.

    lineages    : (n_tips, n_traits, n_chunks_big) uint64 from sim_bit run with
                  trials_big = trials * oversample replicates
    target_prev : (n_traits,) observed prevalence (NaN -> no conditioning)
    Returns (n_tips, n_traits, ceil(trials/64)) uint64.

    This conditions each trait's null on its observed marginal, the way
    Fisher's exact test conditions on the table margins.  Without it, traits
    with few events are simulated with the right number of events but the
    wrong footprint (events land on short tip branches), so their simulated
    copies are far rarer and less clade-like than the observed trait and the
    null co-occurrence spread is too narrow.
    """
    n_tips, n_traits, _ = lineages.shape
    n_chunks, _ = _chunk_info(trials)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_tips, n_traits, n_chunks), dtype=np.uint64)
    for s in range(0, n_traits, block):
        e = min(n_traits, s + block)
        bits = np.unpackbits(lineages[:, s:e, :].view(np.uint8), axis=-1, bitorder="little")
        bits = bits.reshape(n_tips, e - s, -1)                          # tips x traits x trials_big
        prev = bits.mean(axis=0)                                         # traits x trials_big
        tgt = np.asarray(target_prev[s:e], dtype=float)[:, None]
        dist = np.abs(prev - np.where(np.isfinite(tgt), tgt, prev))
        dist = dist + rng.random(dist.shape) * 1e-9                     # random tie-break
        idx = np.argsort(dist, axis=1)[:, :trials]                       # traits x trials
        sel = np.take_along_axis(bits, idx[None, :, :], axis=2)          # tips x traits x trials
        pad = n_chunks * 64 - trials
        if pad:
            sel = np.pad(sel, ((0, 0), (0, 0), (0, pad)))
        out[:, s:e, :] = np.packbits(sel, axis=-1, bitorder="little").view(np.uint64)
    return out


def fitch_changes_packed(tree, lineages, n_trials, block=128):
    """Parsimony change count for every (trait, replicate) of a packed simulation.

    lineages : (n_tips, n_traits, n_chunks) uint64, tips in ``tree`` leaf order.
    Returns (n_traits, n_trials) int32.  Bit-parallel Fitch: each node keeps two
    bitsets (state 0 possible / state 1 possible) per (trait, chunk).
    Multifurcations are resolved left to right, so counts are an upper bound
    on the minimum for polytomies; observed traits must be scored with this
    same function for a like-for-like comparison.
    """
    leaves = list(tree)
    leaf_pos = {l: i for i, l in enumerate(leaves)}
    n_tips, n_traits, n_chunks = lineages.shape
    out = np.zeros((n_traits, n_trials), dtype=np.int32)
    full = np.uint64(0xFFFFFFFFFFFFFFFF)
    for s in range(0, n_traits, block):
        e = min(n_traits, s + block)
        L = lineages[:, s:e, :]
        counts = np.zeros((e - s, n_chunks * 64), dtype=np.int32)
        sets = {}
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                b = L[leaf_pos[node]]
                sets[node] = (~b & full, b)
                continue
            kids = node.children
            s0, s1 = sets.pop(kids[0])
            for c in kids[1:]:
                c0, c1 = sets.pop(c)
                i0, i1 = s0 & c0, s1 & c1
                empty = ~(i0 | i1) & full
                if empty.any():
                    counts += np.unpackbits(empty.view(np.uint8), axis=-1,
                                            bitorder="little").reshape(e - s, -1)
                s0 = np.where(empty, s0 | c0, i0)
                s1 = np.where(empty, s1 | c1, i1)
            sets[node] = (s0, s1)
        out[s:e] = counts[:, :n_trials]
    return out


def condition_on_footprint(tree, lineages, obs_lineages, trials, seed=None, block=256):
    """Keep, per trait, the ``trials`` replicates closest to the observed trait in
    BOTH prevalence and parsimony changes.

    Prevalence alone fails when the estimated rates are off: on clonal trees
    (many zero-length branches) FLOW rates overstate change, so replicates
    reach the observed prevalence through many small clumps instead of the
    few large ones observed, and the null is too narrow.  Matching the number
    of changes as well makes the null reproduce the trait's footprint
    regardless of rate error.  Each distance is scaled by its spread across
    the trait's replicates.

    obs_lineages : (n_tips, n_traits, 1) uint64 with the observed traits in bit 0
                   (NA tips should be 0; they are rare and only affect the score).
    """
    n_tips, n_traits, n_chunks_big = lineages.shape
    big = n_chunks_big * 64
    n_chunks, _ = _chunk_info(trials)
    rng = np.random.default_rng(seed)
    f_sim = fitch_changes_packed(tree, lineages, big)
    f_obs = fitch_changes_packed(tree, obs_lineages, 1)[:, 0]
    out = np.zeros((n_tips, n_traits, n_chunks), dtype=np.uint64)
    for s in range(0, n_traits, block):
        e = min(n_traits, s + block)
        bits = np.unpackbits(lineages[:, s:e, :].view(np.uint8), axis=-1, bitorder="little")
        bits = bits.reshape(n_tips, e - s, -1)
        prev = bits.mean(axis=0)
        obs_bits = np.unpackbits(obs_lineages[:, s:e, :].view(np.uint8), axis=-1,
                                 bitorder="little").reshape(n_tips, e - s, -1)[:, :, 0]
        p_obs = obs_bits.mean(axis=0)[:, None]
        lf = np.log1p(f_sim[s:e])
        lf_obs = np.log1p(f_obs[s:e])[:, None]
        sp = prev.std(axis=1, keepdims=True) + 1e-9
        sf = lf.std(axis=1, keepdims=True) + 1e-9
        dist = ((prev - p_obs) / sp) ** 2 + ((lf - lf_obs) / sf) ** 2
        dist = dist + rng.random(dist.shape) * 1e-9
        idx = np.argsort(dist, axis=1)[:, :trials]
        sel = np.take_along_axis(bits, idx[None, :, :], axis=2)
        pad = n_chunks * 64 - trials
        if pad:
            sel = np.pad(sel, ((0, 0), (0, 0), (0, pad)))
        out[:, s:e, :] = np.packbits(sel, axis=-1, bitorder="little").view(np.uint64)
    return out


def pack_observed(obs: np.ndarray) -> np.ndarray:
    """(n_tips, n_traits) 0/1 (NaN -> 0) -> (n_tips, n_traits, 1) uint64 with bit 0 set."""
    o = np.nan_to_num(np.asarray(obs, dtype=float), nan=0.0) > 0.5
    return o.astype(np.uint64)[:, :, None]


def simulate_glrates_bit(tree, trait_params, pairs, obspairs, trials = 64, cores = -1,
                         gain_mask=None, loss_mask=None, gamma=False, seed=None,
                         emergence='dist', emergence_scale=1.0,
                         condition=None, oversample=8, obs_prevalence=None, obs_traits=None,
                         branch_cap=None, rate_model='point', pi_source='acr', rate_draws=True,
                         prior_by_prevalence=False, root='threshold', simulator='poisson',
                         fixed_k_max=None, statistic='logor', pvalue='kde',
                         testable_filter=None, adaptive_stages=(4096, 32768), adaptive_h=10,
                         anchor_combine='max', gate=None, gate_q=0.95, gate_p=None, refine_p=None,
                         lineage=None, tau=0.1, sup_p=0.05, composite=False,
                         confine_alpha=0.1, refine_mode='gpd', refine_n=32768, refine_frac=0.01,
                         xi_min=0.0, refine_tiers=None, refine_max_pairs=None, tail='gpd',
                         p_unrestricted=0.5):
    """Simulate the null for every trait and score all pairs.

    condition : None | 'prevalence' | 'footprint'.  ``trials * oversample``
                replicates are simulated per trait and the ``trials`` closest
                to the observed trait are kept: by prevalence
                (``obs_prevalence``), or by prevalence and parsimony changes
                (``obs_traits``: (n_tips, n_traits) in tree leaf order).

    Prototype options (see uncertainty.py): rate_model, pi_source, rate_draws,
    prior_by_prevalence, root, simulator, fixed_k_max, statistic, pvalue,
    testable_filter.
    """
    from . import uncertainty as unc
    if condition not in (None, 'none', 'prevalence', 'footprint'):
        raise ValueError(f"condition must be None, 'prevalence' or 'footprint', got {condition!r}")
    mappingr = dict(enumerate(trait_params.index))
    mapping = dict(zip(trait_params.index,range(len(trait_params.index))))
    pairs_index = np.vectorize(lambda key: mapping[key])(pairs)
    urng = np.random.default_rng(None if seed is None else seed + 11)

    if pvalue == 'analytic':
        gr, lr, info = unc.eb_rates(trait_params, obs_prevalence, 1, pi_source=pi_source,
                                    draws=False, prior_by_prevalence=prior_by_prevalence, rng=urng)
        mu = info['mu_mean'] if rate_model == 'eb' else info['mu_point']
        res = unc.analytic_pairs(tree, mu, pairs_index, obs_traits, branch_cap=branch_cap)
        res.insert(0, 'second', pairs_index[:, 1]); res.insert(0, 'first', pairs_index[:, 0])
        res['first'] = res['first'].map(mappingr)
        res['second'] = res['second'].map(mappingr)
        return res

    if pvalue == 'anchored_fast':
        from .fastsim import run_anchored
        res = run_anchored(tree, trait_params, pairs_index, obs_traits, trials=trials,
                           branch_cap=branch_cap, seed=seed, weighted=(statistic == 'weighted'),
                           combine=anchor_combine, gate=gate, gate_q=gate_q, gate_p=gate_p,
                           refine_p=refine_p, refine_stages=tuple(adaptive_stages),
                           refine_h=adaptive_h, lineage=lineage, tau=tau, sup_p=sup_p,
                           composite=composite, confine_alpha=confine_alpha,
                           refine_mode=refine_mode, refine_n=refine_n, refine_frac=refine_frac,
                           xi_min=xi_min, refine_tiers=refine_tiers,
                           refine_max_pairs=refine_max_pairs, tail=tail, p_unrestricted=p_unrestricted)
        res.insert(0, 'second', pairs_index[:, 1]); res.insert(0, 'first', pairs_index[:, 0])
        res['first'] = res['first'].map(mappingr)
        res['second'] = res['second'].map(mappingr)
        return res

    if pvalue == 'anchored_mc':
        # No conditioning: anchored counts against fresh, unselected replicates.
        stage_ctr = [0]
        def _simulate(tidx, N):
            stage_ctr[0] += 1
            sub = trait_params.iloc[tidx]
            sd = None if seed is None else seed + 101 * stage_ctr[0]
            gm = None if gain_mask is None else gain_mask[:, tidx]
            lm = None if loss_mask is None else loss_mask[:, tidx]
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                L = sim_bit(tree=tree, trait_params=sub, trials=N, gain_mask=gm, loss_mask=lm,
                            gamma=gamma, seed=sd, emergence=emergence,
                            emergence_scale=emergence_scale, branch_cap=branch_cap)
            n_tips_, n_tr_, _ = L.shape
            bits = np.unpackbits(L.view(np.uint8), axis=-1, bitorder='little')
            return bits.reshape(n_tips_, n_tr_, -1)[:, :, :N]
        w = unc.gsc_weights(tree, branch_cap) if statistic == 'weighted' else None
        res = unc.anchored_mc_pairs(_simulate, pairs_index, obs_traits, trials, weights=w,
                                    stages=tuple(adaptive_stages), h=adaptive_h,
                                    combine=anchor_combine)
        res.insert(0, 'second', pairs_index[:, 1]); res.insert(0, 'first', pairs_index[:, 0])
        res['first'] = res['first'].map(mappingr)
        res['second'] = res['second'].map(mappingr)
        return res

    conditioned = condition in ('prevalence', 'footprint') and oversample > 1
    n_sim = trials * oversample if conditioned else trials

    rates = None
    if rate_model == 'eb':
        gr, lr, _ = unc.eb_rates(trait_params, obs_prevalence, n_sim, pi_source=pi_source,
                                 draws=rate_draws, prior_by_prevalence=prior_by_prevalence,
                                 rng=urng)
        rates = (gr, lr)
    elif rate_model not in (None, 'point'):
        raise ValueError(f"rate_model must be 'point' or 'eb', got {rate_model!r}")

    root_bits = None
    if root == 'sample' and 'root_prob' in trait_params.columns:
        rp = np.nan_to_num(trait_params['root_prob'].to_numpy(dtype=float), nan=0.5)
        root_bits = (urng.random((len(rp), n_sim)) < rp[:, None]).astype(np.uint8)

    sim = sim_bit(tree=tree, trait_params=trait_params, trials=n_sim,
                  gain_mask=gain_mask, loss_mask=loss_mask, gamma=gamma, seed=seed,
                  emergence=emergence, emergence_scale=emergence_scale, branch_cap=branch_cap,
                  rates=rates, root_bits=root_bits)

    if simulator == 'fixed_k':
        if '_raw_gains' in trait_params.columns:
            k = (trait_params['_raw_gains'] + trait_params['_raw_losses']).to_numpy(dtype=float)
        else:
            k = (trait_params['gains'] + trait_params['losses']).to_numpy(dtype=float)
        k = np.rint(np.nan_to_num(k)).astype(int)
        sel = np.ones(len(k), bool) if fixed_k_max is None else (k <= fixed_k_max)
        if sel.any():
            rs = trait_params['root_state'].to_numpy()[sel]
            fk = unc.sim_fixed_k(tree, k[sel], rs, n_sim, branch_cap=branch_cap, rng=urng,
                                 root_bits=None if root_bits is None else root_bits[sel])
            sim[:, sel, :] = fk
    elif simulator not in (None, 'poisson'):
        raise ValueError(f"simulator must be 'poisson' or 'fixed_k', got {simulator!r}")

    if conditioned and condition == 'prevalence':
        if obs_prevalence is None:
            raise ValueError("condition='prevalence' needs obs_prevalence")
        sim = condition_on_prevalence(sim, np.asarray(obs_prevalence, float), trials,
                                      seed=None if seed is None else seed + 2)
    elif conditioned:
        if obs_traits is None:
            raise ValueError("condition='footprint' needs obs_traits")
        sim = condition_on_footprint(tree, sim, pack_observed(obs_traits), trials,
                                     seed=None if seed is None else seed + 2)

    dense = statistic != "logor" or pvalue != "kde" or testable_filter is not None
    if dense:
        w = unc.gsc_weights(tree, branch_cap) if statistic == 'weighted' else None
        res = unc.score_pairs_dense(sim, pairs_index, obs_traits, trials, weights=w,
                                    pvalue=pvalue, want_pmin=testable_filter is not None)
        res.insert(0, 'second', pairs_index[:, 1]); res.insert(0, 'first', pairs_index[:, 0])
    else:
        res = compres(sim, pairs_index, obspairs, bits=trials, cores=cores,
                      seed=None if seed is None else seed + 1)

    res['first'] = res['first'].map(mappingr)
    res['second'] = res['second'].map(mappingr)
    return res


EMERGENCE_MODES = ('dist', 'none', 'scaled', 'soft')


def iqr_branch_cap(tree) -> float:
    """Branch-length cap used for rate denominators by run_ancestral_reconstruction
    (10 ** (Q3 + 0.5 * IQR) of log10 non-zero branch lengths)."""
    bls = np.array([n.dist for n in tree.traverse() if not n.is_root() and n.dist > 0])
    if len(bls) == 0:
        return float('inf')
    q1, q3 = np.percentile(np.log10(bls), [25, 75])
    return float(10 ** (q3 + 0.5 * (q3 - q1)))


def sim_bit(tree, trait_params, trials=64, gain_mask=None, loss_mask=None, gamma=False,
            seed=None, emergence='dist', emergence_scale=1.0, branch_cap=None,
            rates=None, root_bits=None):
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
    seed         : int or None.  With an int, simulation draws come from
                   ``np.random.default_rng(seed)`` and are reproducible.  With None
                   (default) the global ``np.random`` state is used, so callers can
                   still control it with ``np.random.seed``.
    emergence    : how the emergence threshold (``dist`` / ``loss_dist``) gates
                   events in distance mode (ignored in mask mode):
                   'dist'   — events allowed once root distance >= threshold (default)
                   'none'   — no threshold; every branch eligible
                   'scaled' — threshold multiplied by ``emergence_scale``
                   'soft'   — each trial draws its own threshold ~ U(0, threshold),
                              so uncertainty in emergence time enters the null
                   (For an MRCA clade mask use TreeSimulator.run_simulation(
                   emergence='mrca'), which builds gain/loss masks.)
    emergence_scale : factor for emergence='scaled'.
    branch_cap   : None (default) or float.  Event probabilities use
                   min(branch length, branch_cap).  Pass iqr_branch_cap(tree) to
                   make the simulation consistent with the IQR-capped rate
                   denominators (subsize='ORIGINAL'); otherwise rates estimated
                   on capped lengths are applied to full lengths, which on
                   clonal trees (where most length sits above the cap) makes
                   long branches flip in nearly every replicate.

    Returns
    -------
    lineages : np.ndarray (n_tips, n_traits, n_chunks) dtype=uint64
        n_chunks = ceil(trials/64).  Each uint64 encodes 64 independent trial
        states as bit flags; the last chunk uses only (trials - 64*(n_chunks-1))
        of its 64 bits (the remaining high bits are always 0).
    """
    gains, losses, dists, loss_dists, gain_subsize, loss_subsize, root_values = \
        unpack_trait_params(trait_params)
    rng = np.random if seed is None else np.random.default_rng(seed)
    use_mask = (gain_mask is not None) and (loss_mask is not None)
    if emergence not in EMERGENCE_MODES:
        raise ValueError(f"emergence must be one of {EMERGENCE_MODES}, got {emergence!r}")
    if emergence == 'none':
        dists = np.zeros_like(dists); loss_dists = np.zeros_like(loss_dists)
    elif emergence == 'scaled':
        dists = dists * emergence_scale; loss_dists = loss_dists * emergence_scale
    soft = (emergence == 'soft') and not use_mask

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

    if rates is not None:
        # Caller-supplied (n_traits, trials) gain and loss rates (rate_model='eb').
        gain_rates, loss_rates = (np.asarray(r, dtype=float) for r in rates)
        gamma = True
    elif gamma:
        # Sample one rate per trait per trial from the Jeffreys posterior:
        #   rate | data ~ Gamma(count + 0.5, scale=1/subsize)
        # Shape: (n_traits, trials) — each trial uses an independent draw.
        gain_rates = np.zeros((num_traits, trials), dtype=float)
        loss_rates = np.zeros((num_traits, trials), dtype=float)
        gain_rates[valid_gains] = rng.gamma(
            (gains[valid_gains] + 0.5)[:, np.newaxis],
            (1.0 / gain_subsize[valid_gains])[:, np.newaxis],
            size=(valid_gains.sum(), trials),
        )
        loss_rates[valid_losses] = rng.gamma(
            (losses[valid_losses] + 0.5)[:, np.newaxis],
            (1.0 / loss_subsize[valid_losses])[:, np.newaxis],
            size=(valid_losses.sum(), trials),
        )
    else:
        gain_rates = np.zeros(num_traits, dtype=float)
        loss_rates = np.zeros(num_traits, dtype=float)
        gain_rates[valid_gains] = gains[valid_gains] / gain_subsize[valid_gains]
        loss_rates[valid_losses] = losses[valid_losses] / loss_subsize[valid_losses]

    if soft:
        # Per-trial thresholds, shape (n_traits, trials)
        gain_thr = dists[:, np.newaxis] * rng.random((num_traits, trials))
        loss_thr = loss_dists[:, np.newaxis] * rng.random((num_traits, trials))
        all_traits = np.ones(num_traits, dtype=bool)

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
            if root_bits is not None:
                # Per-replicate root states (root='sample'), (n_traits, trials) 0/1
                rb = np.asarray(root_bits, dtype=np.uint8)
                pad = n_chunks * bits - trials
                if pad:
                    rb = np.pad(rb, ((0, 0), (0, pad)))
                sim[node_map[node], :, :] = np.packbits(
                    rb, axis=-1, bitorder='little').view(nptype).reshape(num_traits, n_chunks)
                continue
            root = root_values > 0
            for c in range(n_chunks):
                chunk_val = FULL_MASK if c < n_chunks - 1 else last_mask
                sim[node_map[node], root, c] = chunk_val
            continue

        parent = sim[node_map[node.up], :, :]        # (n_traits, n_chunks)
        node_total_dist = node_dists[node]
        bl_eff = node.dist if branch_cap is None else min(node.dist, branch_cap)

        if use_mask:
            # Mask mode: per-node, per-trait eligibility from entropy mask
            node_idx = node_map[node]
            applicable_traits_gains  = gain_mask[node_idx, :]
            applicable_traits_losses = loss_mask[node_idx, :]
        elif soft:
            # Soft mode: sample everywhere, gate per trial below
            applicable_traits_gains  = all_traits
            applicable_traits_losses = all_traits
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

            g_samp = (rng.poisson(bl_eff * gr, (n_gain, chunk_trials)) > 0).astype(np.uint8)
            l_samp = (rng.poisson(bl_eff * lr, (n_loss, chunk_trials)) > 0).astype(np.uint8)

            if chunk_trials < bits:
                # Zero-pad to exactly 64 columns so packbits produces one uint64
                g_samp = np.pad(g_samp, ((0, 0), (0, bits - chunk_trials)), mode='constant')
                l_samp = np.pad(l_samp, ((0, 0), (0, bits - chunk_trials)), mode='constant')

            if soft:
                sl = slice(chunk_start, chunk_start + chunk_trials)
                g_ok = (node_total_dist >= gain_thr[:, sl]).astype(np.uint8)
                l_ok = (node_total_dist >= loss_thr[:, sl]).astype(np.uint8)
                if chunk_trials < bits:
                    g_ok = np.pad(g_ok, ((0, 0), (0, bits - chunk_trials)), mode='constant')
                    l_ok = np.pad(l_ok, ((0, 0), (0, bits - chunk_trials)), mode='constant')
                g_samp &= g_ok
                l_samp &= l_ok

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


def process_batch(index: int, sim_readonly: np.ndarray, pairs: np.ndarray, obspairs: np.ndarray, batch_size: int, total_trials: int = 64, seed=None) -> Dict[str, List]:
    """
    Process a single batch of data.
    """
    pair_batch = pairs[index: index + batch_size]
    current_obspairs = obspairs[index: index + len(pair_batch)]

    # sim_readonly shape: (n_nodes, n_traits, n_chunks)
    tp = sim_readonly[:, pair_batch[:, 0], :]
    tq = sim_readonly[:, pair_batch[:, 1], :]

    batch_cooc = compute_bitwise_cooc(tp, tq, total_trials)
    # Tiny jitter keeps the KDE non-singular when all simulated scores tie.
    # Seeded per pair (global index), so results do not depend on how pairs
    # are split into batches or on the number of workers.
    if seed is None:
        noise = np.random.default_rng().normal(0, 1e-12, size=batch_cooc.shape)
    else:
        noise = np.stack([np.random.default_rng((seed, index + i)).normal(0, 1e-12, batch_cooc.shape[1])
                          for i in range(batch_cooc.shape[0])])
    noised_batch_cooc = batch_cooc + noise

    results = [
        compute_kde_stats(current_obspairs[i], noised_batch_cooc[i])
        for i in range(len(pair_batch))
    ]

    kde_pvals_ant, kde_pvals_syn, medians, normalization_factors = map(np.array, zip(*results))

    # Vectorized calculation of final results.
    # The direction is chosen after looking at the data, so the p-value is the
    # two-sided  2 * min(left tail, right tail), capped at 1.  Using the bare
    # minimum would reject at twice the nominal rate under the null.
    min_pvals = np.minimum(1.0, 2.0 * np.minimum(kde_pvals_syn, kde_pvals_ant))
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

def compres(sim: np.ndarray, pairs: np.ndarray, obspairs: np.ndarray, batch_size: int = 1000, bits: int = 64, cores: int = -1, seed=None) -> pd.DataFrame:
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
    n_workers = cores if cores and cores > 0 else (os.cpu_count() or 1)
    batch_size = max(1, min(int(np.ceil(num_pairs / n_workers)), batch_size))
    batch_indices = range(0, num_pairs, batch_size)

    print(f"Processing Batches, Total: {num_pairs//batch_size + 1}")

    batch_results = Parallel(n_jobs=cores, verbose=10)(
        delayed(process_batch)(index, sim, pairs, obspairs, batch_size, bits, seed)
        for index in batch_indices
    )

    print("Aggregating Results...")

    # Merge batch results
    for batch_res in batch_results:
        for key in res.keys():
            res[key].extend(batch_res[key])

    return pd.DataFrame.from_dict(res)