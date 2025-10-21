from typing import List
import numpy as np
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed, parallel_backend, Memory
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from typing import List, Tuple, Set, Dict

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

    res = compile_results_KDE_bit_async(sim, pairs_index, obspairs, bits = 64)

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
    multiplier = 1e12

    # Preprocess and setup
    node_map = {node: ind for ind, node in enumerate(tree.traverse())}
    num_traits = len(gains)
    num_nodes = len(node_map)
    bits = 64
    nptype = np.uint64
    sim = np.zeros((num_nodes, num_traits), dtype=nptype)
    trials = bits

    gain_rates = gains / (gain_subsize * multiplier)
    loss_rates = losses / (loss_subsize * multiplier)
    np.nan_to_num(gain_rates, copy = False)
    np.nan_to_num(loss_rates, copy = False)

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
        node_dist_multiplier = node.dist * multiplier
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain and loss rates for traits where the node's distance is less than the specified threshold
        applicable_traits_gains = node_total_dist >= dists
        applicable_traits_losses = node_total_dist >= loss_dists 
        gain_events = np.zeros((num_traits), dtype=nptype)
        loss_events = np.zeros((num_traits), dtype=nptype)
        gain_events[applicable_traits_gains] = np.packbits((np.random.binomial(node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], (applicable_traits_gains.sum(), trials)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()
        loss_events[applicable_traits_losses] = np.packbits((np.random.binomial(node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], (applicable_traits_losses.sum(), trials)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()
        updated_state = np.bitwise_or(parent, gain_events)  # Gain new traits
        updated_state = np.bitwise_and(updated_state, np.bitwise_not(loss_events))  # Remove lost traits
        sim[node_map[node], :] = updated_state # Store updated node state

        # print(f"Node {node.name} Completed")

    print("Completed Tree Simulation Sucessfully")

    lineages = sim[[node_map[node] for node in tree], :]
    return lineages

# Compiling results

def compile_results_KDE_bit_async(sim: np.ndarray, pairs: np.ndarray, obspairs: np.ndarray, batch_size: int = 1000,bits = 64,nptype = np.uint64, cores = -1) -> pd.DataFrame:
    """
    Compile KDE results asynchronously using parallel batch processing, optimizing `sim` memory handling.

    :param sim: Large NumPy array storing simulation data.
    :param obspairs: Observed pairs statistics.
    :param batch_size: Size of each batch for processing.
    :return: DataFrame with compiled results.
    """
    # Use Joblib Memory to avoid redundant copies
    memory = Memory(location=None, verbose=0)  # No disk caching, just memory optimization
    res: Dict[str, List] = {
        "pair": [], "first": [], "second": [], 
        "direction": [], "p-value": [], "effect size": []
    }

    # Convert sim to read-only memory-mapped array to reduce memory duplication
    sim = np.asarray(sim, order="C")  # Ensure contiguous memory
    sim.setflags(write=False)  # Set as read-only to avoid unintended copies

    def circular_bitshift_right(arr: np.ndarray, k: int) -> np.ndarray:
        """Perform a circular right bit shift on all np.uint64 entries in an array."""
        k %= bits
        return np.bitwise_or(np.right_shift(arr, k), np.left_shift(arr, bits - k))
    
    def sum_all_bits(arr: np.ndarray) -> np.ndarray:
        """Compute the sum of 1s for all 64 bit positions in an array of uint64 values."""
        bit_sums = np.zeros((bits, arr.shape[-1]), dtype=np.float64)
        for i in range(bits):
            bit_sums[i] = np.sum((arr >> i) & 1, axis=0, dtype=nptype)
        return bit_sums

    def compute_bitwise_cooc(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        """Compute bitwise co-occurrence statistics for a batch."""
        cooc_batch = []
        for k in range(bits):
            shifted = circular_bitshift_right(tq, k)
            a = sum_all_bits(tp & shifted) + 1e-2
            b = sum_all_bits(tp & ~shifted) + 1e-2
            c = sum_all_bits(~tp & shifted) + 1e-2
            d = sum_all_bits(~tp & ~shifted) + 1e-2
            cooc_batch.append(np.log((a * d) / (b * c)))
        return np.vstack(cooc_batch).T  # Shape: (batch_size, bits)

    def compute_kde_stats(observed_value: float, simulated_values: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute KDE statistics for a single pair."""
        kde = gaussian_kde(simulated_values, bw_method='silverman')
        cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
        kde_syn = gaussian_kde(-1*simulated_values, bw_method='silverman')
        cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf,-x)
        kde_pval_ant = cdf_func_ant(observed_value)  # P(X ≤ observed)
        kde_pval_syn = cdf_func_syn(observed_value)  # P(X ≥ observed)
            
        med = np.median(simulated_values)
        q75, q25 = np.percentile(simulated_values, [75, 25])
        iqr = q75 - q25
        return kde_pval_ant, kde_pval_syn, med, iqr

    def process_batch(index: int, sim_readonly: np.ndarray) -> Dict[str, List]:
        """Process a single batch of data, ensuring memory-efficient sim access."""
        pair_batch = pairs[index: index + batch_size]
        tp = sim_readonly[:, pair_batch[:, 0]]
        tq = sim_readonly[:, pair_batch[:, 1]]

        # Compute bitwise co-occurrence in batches
        batch_cooc = compute_bitwise_cooc(tp, tq)

        # Add small noise
        noised_batch_cooc = batch_cooc + np.random.normal(0, 1e-12, size=batch_cooc.shape)

        # Compute KDE statistics in parallel
        results = Parallel(n_jobs=-1, verbose=0, batch_size=25)(
            delayed(compute_kde_stats)(obspairs[index + i], noised_batch_cooc[i])
            for i in range(len(pair_batch))
        )

        kde_pvals_ant, kde_pvals_syn, medians, iqrs = map(np.array, zip(*results))

        batch_res = {
            "pair": [tuple(p) for p in pair_batch],
            "first": pair_batch[:, 0].tolist(),
            "second": pair_batch[:, 1].tolist(),
            "p-value": np.minimum(kde_pvals_syn, kde_pvals_ant).tolist(),
            "direction": np.where(kde_pvals_ant < kde_pvals_syn, -1, 1).tolist(),
            "effect size": ((obspairs[index: index + len(pair_batch)]-medians) / np.maximum(iqrs * 1.349, 1)).tolist(),
        }

        return batch_res

    num_pairs = len(pairs)
    batch_indices = range(0, num_pairs, batch_size)

    print(f"Processing Batches, Total: {num_pairs//batch_size + 1}")

    # Run batches in parallel, passing a read-only copy of sim
    batch_results = Parallel(n_jobs=cores, verbose=10)(
        delayed(process_batch)(index, sim) for index in batch_indices
    )

    print("Aggregating Results...")

    # Merge batch results
    for batch_res in batch_results:
        for key in res.keys():
            res[key].extend(batch_res[key])

    return pd.DataFrame.from_dict(res)