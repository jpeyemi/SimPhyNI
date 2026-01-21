from typing import List
import numpy as np
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed, parallel_backend, Memory
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

### Simulation Methods

def simulate_glrates_bit(tree, trait_params, pairs, obspairs, trials = 64, cores = -1):
    
    sim = sim_bit(tree=tree,trait_params=trait_params, trials = 64)
    mappingr = dict(enumerate(trait_params.index))
    mapping = dict(zip(trait_params.index,range(len(trait_params.index))))
    pairs_index = np.vectorize(lambda key: mapping[key])(pairs)

    res = compres(sim, pairs_index, obspairs, bits = 64)

    res['first'] = res['first'].map(mappingr)
    res['second'] = res['second'].map(mappingr)
    return res


def sim_bit(tree, trait_params, trials = 64):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch
    For each trait only simulates on branches beyond a certain distance from the root, 
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction
    """

    gains,losses,dists,loss_dists,gain_subsize,loss_subsize,root_states = unpack_trait_params(trait_params)

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

        if node.up == None:
            root = root_states > 0
            root_mask = np.zeros(num_traits, dtype=bool)
            root_mask[root] = True
            full_mask_value = (1 << trials) - 1
            sim[node_map[node], root_mask] = full_mask_value
            continue
        
        parent = sim[node_map[node.up], :]
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain and loss rates for traits where the node's distance is less than the specified threshold
        applicable_traits_gains = node_total_dist >= dists
        applicable_traits_losses = node_total_dist >= loss_dists 
        gain_events = np.zeros((num_traits), dtype=nptype)
        loss_events = np.zeros((num_traits), dtype=nptype)
        gain_events[applicable_traits_gains] = np.packbits((np.random.poisson(node.dist * gain_rates[applicable_traits_gains, np.newaxis], (applicable_traits_gains.sum(), trials)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()
        loss_events[applicable_traits_losses] = np.packbits((np.random.poisson(node.dist * loss_rates[applicable_traits_losses, np.newaxis], (applicable_traits_losses.sum(), trials)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()

        gain_events &= ~parent
        loss_events &= parent   

        updated_state = np.bitwise_or(parent, gain_events)  # Gain new traits
        updated_state = np.bitwise_and(updated_state, np.bitwise_not(loss_events))  # Remove lost traits
        sim[node_map[node], :] = updated_state # Store updated node state

        # print(f"Node {node.name} Completed")

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