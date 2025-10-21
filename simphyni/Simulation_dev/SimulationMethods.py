from typing import List
import numpy as np
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed, parallel_backend, Memory
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from typing import List, Tuple, Set, Dict

### Simulation Methods

def simulate_events(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml
    Sums gains and losses to a total number of events then alocates each event to a 
    branch on the tree with probability propotional to the length of the branch
    For each trait only simulates on branches beyond a certain distance from the root, 
    This thereshold is chosen as the first branch where a trait arises in the ancestral trait reconsrtuction
    """
    from simphyni import TreeSimulator
    assert(type(self) == TreeSimulator)
    # Preprocess and setup
    all_nodes = list(self.tree.traverse()) # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    node_map_r = {ind: node for ind, node in enumerate(all_nodes)}
    total_events = self.gains #+ self.losses
    losses = self.losses

    # bl = sum([i.dist for i in self.tree.traverse()]) #type: ignore
    # div = self.subsize / bl
    # total_events = np.floor(total_events/div)

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse(): # type: ignore
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist
    node_df = pd.DataFrame.from_dict(node_dists, orient = 'index', columns = ['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    def get_nodes(dist):
        nodes = node_df[node_df['total_dist'] >= dist]
        # nodes.loc[:,'used_dist'] = np.minimum(nodes['total_dist'] - dist, nodes['dist'])
        nodes = nodes.assign(used_dist=np.minimum(nodes['total_dist'] - dist, nodes['dist']))
        bl = nodes['used_dist'].sum()
        p = nodes['used_dist'] / bl
        if any(p.isna()):
            return (None, None)
        node_index = [node_map[node] for node in nodes.index]
        return node_index, p

    def sim_events(trait, total_events, ap):
        a, p = ap
        if not a: return (None,trait)
        event_locs = np.apply_along_axis(lambda x: np.random.choice(a, size=x.shape[0], replace=False, p=p),
                                          arr=np.zeros((int(np.ceil(total_events[trait])), self.NUM_TRIALS)), 
                                          axis=0)
        return (event_locs, trait)
    
    if self.parallel:
        # Simulating events
        branch_probabilities = [get_nodes(self.dists[trait]) for trait, events in enumerate(total_events)]
        all_event_locs = Parallel(n_jobs=-1, batch_size=100)(delayed(sim_events)(trait, total_events, branch_probabilities[trait]) for trait in range(num_traits)) # type: ignore
        for event_locs, trait in all_event_locs: # type: ignore
            if event_locs is not None:
                sim[:, trait, :][event_locs.flatten("F"), np.repeat(np.arange(self.NUM_TRIALS), int(np.ceil(total_events[trait])))] = True
    else:
        for trait, events in enumerate(total_events):
            a, p = get_nodes(self.dists[trait])
            if not a:
                print(trait)
                continue
            # event_locs = np.random.choice(a, size=(int(np.ceil(events)), self.NUM_TRIALS), replace=True, p=p)
            # event_locs = np.random.choice(list(node_df.index.map(lambda x: node_map[x])), size=(int(np.ceil(events)), self.NUM_TRIALS), replace=True, p=node_df['dist']/node_df['dist'].sum())
            # event_locs = np.array([np.random.choice(a, size=(int(np.ceil(events))), replace=False, p=p) for _ in range (self.NUM_TRIALS)]).T
            event_locs = np.apply_along_axis(lambda x: np.random.choice(a, size=x.shape[0], replace=False, p=p), arr = np.zeros((int(np.ceil(events)), self.NUM_TRIALS)), axis = 0) # type: ignore
            sim[:,trait,:][event_locs.flatten("F"),np.repeat(np.arange(self.NUM_TRIALS),int(np.ceil(events)))] = True
        

    # Lineage calculations
    for node in all_nodes:
        if node.up == None:
            root = self.root_states > 0
            sim[node_map[node],root,:] = True
            continue
        parent = sim[node_map[node.up],:,:]
        curr = sim[node_map[node],:,:]
        sim[node_map[node],:,:] = np.logical_xor(parent,curr)

    lineages = sim[[node_map[node] for node in self.tree],:,:]
    #gain & not losses
    # plot_histograms(self,sim)

    # Results compilation
    res = compile_results(self,lineages)

    # Trait data calculation
    trait_data = calculate_trait_data(self,lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:,:,i], index = [node.name for node in all_nodes], columns = [self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)] # type: ignore
    
    return res, trait_data, get_simulated_trees(5)

def simulate_glrates(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch
    For each trait only simulates on branches beyond a certain distance from the root, 
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction
    """
    from simphyni import TreeSimulator
    assert(type(self) == TreeSimulator)
    # Preprocess and setup
    all_nodes = list(self.tree.traverse()) # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    gain_rates = self.gains / (self.gain_subsize * self.MULTIPLIER)
    loss_rates = self.losses / (self.loss_subsize * self.MULTIPLIER)
    np.nan_to_num(gain_rates, copy = False)
    np.nan_to_num(loss_rates, copy = False)

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse(): # type: ignore
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist
    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    for node in self.tree.traverse(): # type: ignore
        if node.up == None:
            root = self.root_states > 0
            sim[node_map[node],root,:] = True
            continue
        
        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain and loss rates for traits where the node's distance is less than the specified threshold
        applicable_traits_gains = node_total_dist >= self.dists
        applicable_traits_losses = node_total_dist >= self.loss_dists
        gain_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)

        # Simulate gain and loss events
        gain_events[applicable_traits_gains] = np.random.binomial(node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], (applicable_traits_gains.sum(), self.NUM_TRIALS)) > 0
        loss_events[applicable_traits_losses] = np.random.binomial(node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], (applicable_traits_losses.sum(), self.NUM_TRIALS)) > 0

        gain_mask = np.logical_not(np.logical_xor(parent,gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent,loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node],:,:] = parent.copy()
        sim[node_map[node],:,:][gain_events] = True
        sim[node_map[node],:,:][loss_events] = False

    lineages = sim[[node_map[node] for node in self.tree], :, :]

    # Results compilation
    res = compile_results(self, lineages)

    # Trait data calculation
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:, :, i], index=[node.name for node in all_nodes], columns=[self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)]
    
    return res, trait_data, get_simulated_trees(5)

def simulate_glrates_ctmp_vectorized(self): 
    #Not funtional
    """
    Vectorized CTMP simulation of trait evolution with gain/loss rates on a tree.
    Gains and losses occur as a Poisson process. Multiple events per branch possible.
    """
    from simphyni import TreeSimulator
    assert isinstance(self, TreeSimulator)

    all_nodes = list(self.tree.traverse()) # pyright: ignore[reportArgumentType]
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: idx for idx, node in enumerate(all_nodes)}

    # Normalize rates
    gain_rates = np.nan_to_num(self.gains / self.gain_subsize)
    loss_rates = np.nan_to_num(self.losses / self.loss_subsize)

    # Root initialization
    sim[node_map[self.tree]] = (self.root_states[:, None] > 0)

    # Compute total distance from root for each node
    node_dists = {self.tree: 0}
    for node in self.tree.traverse():  # type: ignore
        if node not in node_dists:
            node_dists[node] = node_dists[node.up] + node.dist # pyright: ignore[reportArgumentType]
    total_dists = np.array([node_dists[node] for node in all_nodes])
    node_dists_arr = np.array([node.dist for node in all_nodes])

    # Vectorized simulation
    for node in self.tree.traverse():  # type: ignore
        if node.is_root():
            continue

        parent_idx = node_map[node.up] # pyright: ignore[reportArgumentType]
        curr_idx = node_map[node]
        dist = node.dist
        curr_total_dist = total_dists[curr_idx]

        # Broadcast parent state
        sim[curr_idx] = sim[parent_idx]

        # Vectorized rate calculation
        gain_mask = curr_total_dist > self.dists
        loss_mask = curr_total_dist > self.loss_dists
        g_rates = np.where(gain_mask, gain_rates, 0.0)
        l_rates = np.where(loss_mask, loss_rates, 0.0)

        # For each trait, get current state and simulate transitions in bulk
        parent_states = sim[parent_idx]  # shape: (num_traits, num_trials)
        flat_states = parent_states.reshape(num_traits * self.NUM_TRIALS)
        g_repeat = np.repeat(g_rates, self.NUM_TRIALS)
        l_repeat = np.repeat(l_rates, self.NUM_TRIALS)

        # Initial time
        t = np.zeros_like(flat_states, dtype=float)
        state = flat_states.copy()

        while np.any(t < dist):
            current_rates = np.where(state, l_repeat, g_repeat)
            current_rates[current_rates == 0] = np.inf  # Avoid zero division
            waits = np.random.exponential(1.0 / current_rates)
            waits[current_rates == 0] = np.inf
            t_next = t + waits
            flips = t_next <= dist
            state[flips] = ~state[flips]
            t = t_next

        # Assign result
        sim[curr_idx] = state.reshape(num_traits, self.NUM_TRIALS)

    lineages = sim[[node_map[node] for node in self.tree], :, :]
    res = compile_results(self, lineages)
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [
            pd.DataFrame(sim[:, :, i], index=[n.name for n in all_nodes], columns=[self.mapping[str(j)] for j in range(num_traits)])
              .loc[self.tree.get_leaf_names()]
            for i in range(num)
        ]

    return res, trait_data, get_simulated_trees(5)

def simulate_glrates_ctmp(self):
    """
    CTMP simulation of trait evolution with gain/loss rates on a tree.
    Gains and losses occur as a Poisson process, governed by rates inferred from pastML.
    Multiple events per branch possible.
    """
    from simphyni import TreeSimulator
    assert isinstance(self, TreeSimulator)

    all_nodes = list(self.tree.traverse()) # pyright: ignore[reportArgumentType]
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: idx for idx, node in enumerate(all_nodes)}

    gain_rates = self.gains / (self.gain_subsize)
    loss_rates = self.losses / (self.loss_subsize)
    gain_rates = np.nan_to_num(gain_rates)
    loss_rates = np.nan_to_num(loss_rates)

    # root states initialization
    sim[node_map[self.tree], :, :] = self.root_states[:, np.newaxis] > 0

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse(): # type: ignore
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist
    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    for node in self.tree.traverse(): # type: ignore
        if node.is_root():
            continue

        parent_idx = node_map[node.up] # type: ignore
        curr_idx = node_map[node]
        dist = node.dist

        # Start from parent state
        sim[curr_idx] = sim[parent_idx]

        # Simulate each trait and trial independently
        for trait in range(num_traits):
            g = gain_rates[trait] if node_dists[node] > self.dists[trait] else 0
            l = loss_rates[trait] if node_dists[node] > self.loss_dists[trait] else 0
            for trial in range(self.NUM_TRIALS):
                state = sim[parent_idx, trait, trial]
                t = 0.0

                while t < dist:
                    rate = g if not state else l
                    if rate == 0:
                        break
                    wait_time = np.random.exponential(1 / rate)
                    if t + wait_time > dist:
                        break
                    t += wait_time
                    state = not state  # flip state

                sim[curr_idx, trait, trial] = state
 
    lineages = sim[[node_map[node] for node in self.tree], :, :]

    res = compile_results(self, lineages)
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:, :, i], index=[n.name for n in all_nodes], columns=[self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)]

    return res, trait_data, get_simulated_trees(5)

def simulate_glrates_nodist(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch.
    
    Optimization: only simulates traits that appear in self.pairs (unique indices).
    Non-simulated traits remain constant (zeros except for root states) to save compute,
    but sim still has a column for every trait so downstream indexing is unchanged.
    """
    from simphyni import TreeSimulator
    assert(type(self) == TreeSimulator)
    # Preprocess and setup
    all_nodes = list(self.tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    node_map_r = {ind: node for ind, node in enumerate(all_nodes)}
    bl = sum(i.dist for i in self.tree.traverse())  # type: ignore

    gain_rates = self.gains / (self.gain_subsize * self.MULTIPLIER)
    loss_rates = self.losses / (self.loss_subsize * self.MULTIPLIER)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    # Determine which traits to simulate (from self.pairs)
    if hasattr(self, "pairs") and len(self.pairs) > 0:
        pairs_arr = np.array(self.pairs, dtype=int)
        traits_to_simulate = np.unique(pairs_arr.flatten())
        simulate_mask = np.zeros(num_traits, dtype=bool)
        simulate_mask[traits_to_simulate] = True
    else:
        simulate_mask = np.zeros(num_traits, dtype=bool)

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse():  # type: ignore
        if node in node_dists:
            continue
        node_dists[node] = node_dists[node.up] + node.dist
    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    for node in self.tree.traverse():  # type: ignore
        if node.up is None:
            root = self.root_states > 0
            sim[node_map[node], root, :] = True
            continue

        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain/loss rates unless trait is in pairs
        applicable_traits = simulate_mask  # only traits in pairs are eligible
        gain_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)

        if applicable_traits.any():
            idxs = np.nonzero(applicable_traits)[0]

            # Gains
            draws = np.random.binomial(
                node_dist_multiplier,
                gain_rates[idxs, np.newaxis],
                (len(idxs), self.NUM_TRIALS),
            ) > 0
            gain_events[idxs] = draws

            # Losses
            draws = np.random.binomial(
                node_dist_multiplier,
                loss_rates[idxs, np.newaxis],
                (len(idxs), self.NUM_TRIALS),
            ) > 0
            loss_events[idxs] = draws

        # Apply gain/loss logic
        gain_mask = np.logical_not(np.logical_xor(parent, gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent, loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node], :, :] = parent.copy()
        sim[node_map[node], :, :][gain_events] = True
        sim[node_map[node], :, :][loss_events] = False

    lineages = sim[[node_map[node] for node in self.tree], :, :]

    # Results compilation
    res = compile_results(self, lineages)

    # Trait data calculation
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [
            pd.DataFrame(
                sim[:, :, i],
                index=[node.name for node in all_nodes],
                columns=[self.mapping[str(j)] for j in range(num_traits)],
            ).loc[self.tree.get_leaf_names()]
            for i in range(num)
        ]

    return res, trait_data, get_simulated_trees(5)

def simulate_distnorm(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch
    For each trait only simulates on branches beyond a certain distance from the root, 
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction
    """
    from simphyni import TreeSimulator
    assert(type(self) == TreeSimulator)
    # Preprocess and setup
    all_nodes = list(self.tree.traverse()) # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    bl = 2*len(self.tree)-1

    gain_rates = self.gains / bl
    loss_rates = self.losses / bl
    np.nan_to_num(gain_rates, copy = False)
    np.nan_to_num(loss_rates, copy = False)

    for node in self.tree.traverse(): # type: ignore
        if node.up == None:
            prev = self.obsdf_modified.mean()
            high_prev = list(prev[prev >= .5].index.astype(int))
            sim[node_map[node],high_prev,:] = True
            continue
        
        parent = sim[node_map[node.up], :, :]

        gain_events = np.random.binomial(1, gain_rates[:, np.newaxis], (len(gain_rates), self.NUM_TRIALS)) > 0
        loss_events = np.random.binomial(1, loss_rates[:, np.newaxis], (len(loss_rates), self.NUM_TRIALS)) > 0

        gain_mask = np.logical_not(np.logical_xor(parent,gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent,loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node],:,:] = parent.copy()
        sim[node_map[node],:,:][gain_events] = True
        sim[node_map[node],:,:][loss_events] = False

    lineages = sim[[node_map[node] for node in self.tree], :, :]

    # Results compilation
    res = compile_results(self, lineages)

    # Trait data calculation
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [pd.DataFrame(sim[:, :, i], index=[node.name for node in all_nodes], columns=[self.mapping[str(j)] for j in range(num_traits)]).loc[self.tree.get_leaf_names()] for i in range(num)]
    
    return res, trait_data, get_simulated_trees(5)

def simulate_glrates_bit(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch
    For each trait only simulates on branches beyond a certain distance from the root, 
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction
    """
    from simphyni import TreeSimulator
    assert(type(self) == TreeSimulator)
    # Preprocess and setup
    all_nodes = list(self.tree.traverse()) # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    bits = 64
    nptype = np.uint64
    sim = np.zeros((num_nodes, num_traits), dtype=nptype)
    self.NUM_TRIALS = bits

    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    gain_rates = self.gains / (self.gain_subsize * self.MULTIPLIER)
    loss_rates = self.losses / (self.loss_subsize * self.MULTIPLIER)
    np.nan_to_num(gain_rates, copy = False)
    np.nan_to_num(loss_rates, copy = False)

    # Determine which traits to actually simulate (unique indices found in self.pairs)
    if hasattr(self, "pairs") and len(self.pairs) > 0:
        pairs_arr = np.array(self.pairs, dtype=int)
        # flatten and take unique; guard if pairs is shape (N,2)
        traits_to_simulate = np.unique(pairs_arr.flatten())
        # boolean mask length num_traits
        simulate_mask = np.zeros(num_traits, dtype=bool)
        simulate_mask[traits_to_simulate] = True
    else:
        # if no pairs present, nothing to simulate (leave sim zeros) but still run through to set root
        simulate_mask = np.zeros(num_traits, dtype=bool)

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse(): # type: ignore
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist

    print("Simulating Trees...")
    for node in self.tree.traverse(): # type: ignore
        if node.up == None:
            root = self.root_states > 0
            sim[node_map[node],root] = 2^self.NUM_TRIALS-1
            continue
        
        parent = sim[node_map[node.up], :]
        node_dist_multiplier = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain and loss rates for traits where the node's distance is less than the specified threshold
        applicable_traits_gains = node_total_dist >= self.dists
        applicable_traits_losses = node_total_dist >= self.loss_dists
        gain_events = np.zeros((num_traits), dtype=nptype)
        loss_events = np.zeros((num_traits), dtype=nptype)
        gain_events[applicable_traits_gains] = np.packbits((np.random.binomial(node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], (applicable_traits_gains.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()
        loss_events[applicable_traits_losses] = np.packbits((np.random.binomial(node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], (applicable_traits_losses.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),axis=-1, bitorder='little').view(nptype).flatten()

        # Handle event cancellation
        # canceled_events = np.logical_and(gain_events, loss_events)
        # gain_events[canceled_events] = False
        # loss_events[canceled_events] = False

        updated_state = np.bitwise_or(parent, gain_events)  # Gain new traits
        updated_state = np.bitwise_and(updated_state, np.bitwise_not(loss_events))  # Remove lost traits
        # Store updated node state
        sim[node_map[node], :] = updated_state

        # print(f"Node {node.name} Completed")

    print("Completed Tree Simulation Sucessfully")
    lineages = sim[[node_map[node] for node in self.tree], :]
    # return lineages
    res = compile_results_KDE_bit_async(self, lineages, bits = bits, nptype = nptype)

    return res

def simulate_glrates_bit_norm(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch
    For each trait only simulates on branches beyond a certain distance from the root, 
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction

    Optimization: only draws random events for traits that appear in self.pairs (unique indices).
    The returned `sim` still has a column for every trait (so downstream code keeps original indexing),
    but traits not in any pair are left as constant (no simulated gains/losses) to save compute.
    """
    from simphyni import TreeSimulator
    assert(type(self) == TreeSimulator)
    # Preprocess and setup
    all_nodes = list(self.tree.traverse()) # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    bits = 64
    nptype = np.uint64
    sim = np.zeros((num_nodes, num_traits), dtype=nptype)
    self.NUM_TRIALS = bits

    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    gain_rates = self.gains / len(all_nodes)
    loss_rates = self.losses / len(all_nodes)
    np.nan_to_num(gain_rates, copy = False)
    np.nan_to_num(loss_rates, copy = False)

    # Determine which traits to actually simulate (unique indices found in self.pairs)
    if hasattr(self, "pairs") and len(self.pairs) > 0:
        pairs_arr = np.array(self.pairs, dtype=int)
        # flatten and take unique; guard if pairs is shape (N,2)
        traits_to_simulate = np.unique(pairs_arr.flatten())
        # boolean mask length num_traits
        simulate_mask = np.zeros(num_traits, dtype=bool)
        simulate_mask[traits_to_simulate] = True
    else:
        # if no pairs present, nothing to simulate (leave sim zeros) but still run through to set root
        simulate_mask = np.zeros(num_traits, dtype=bool)

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse(): # type: ignore
        if node in node_dists: continue
        node_dists[node] = node_dists[node.up] + node.dist

    print("Simulating Trees...")
    for node in self.tree.traverse(): # type: ignore
        if node.up == None:
            # set root states: root states True -> set all bits to 1 for those traits
            root = self.root_states > 0
            # create mask for root traits
            root_mask = np.zeros(num_traits, dtype=bool)
            root_mask[root] = True
            # assign all 1 bits for the NUM_TRIALS positions
            full_mask_value = (1 << self.NUM_TRIALS) - 1  # correct bitmask for NUM_TRIALS bits
            # only set traits that exist (keep zeros for others)
            sim[node_map[node], root_mask] = full_mask_value
            continue
        
        parent = sim[node_map[node.up], :]
        node_total_dist = node_dists[node]  # Total distance from the root to the current node

        # Zero out gain and loss rates for traits where the node's distance is less than the specified threshold
        applicable_traits_gains = (node_total_dist >= self.dists) & simulate_mask
        applicable_traits_losses = (node_total_dist >= self.loss_dists) & simulate_mask

        # Prepare empty arrays for all traits (full-length), filled with zeros by default
        gain_events = np.zeros((num_traits), dtype=nptype)
        loss_events = np.zeros((num_traits), dtype=nptype)

        # For performance: compute events only for traits in the simulate_mask & distance threshold
        # Work on compact arrays of only the applicable indices, then place back into full-length arrays.
        if applicable_traits_gains.any():
            idxs = np.nonzero(applicable_traits_gains)[0]
            # p should be shape (len(idxs),) or broadcastable; create binomial draws shape (len(idxs), NUM_TRIALS)
            # we generate a (n_traits_subset x NUM_TRIALS) matrix of 0/1 trials, then packbits into uint64
            draws = (np.random.binomial(1, gain_rates[idxs, np.newaxis], (len(idxs), self.NUM_TRIALS)) > 0).astype(np.uint8)
            packed = np.packbits(draws, axis=-1, bitorder='little')
            # packbits yields shape (len(idxs), ceil(bits/8)) where ceil(64/8)=8 -> view as uint64
            packed_uint64 = packed.view(nptype).flatten()
            gain_events[idxs] = packed_uint64

        if applicable_traits_losses.any():
            idxs = np.nonzero(applicable_traits_losses)[0]
            draws = (np.random.binomial(1, loss_rates[idxs, np.newaxis], (len(idxs), self.NUM_TRIALS)) > 0).astype(np.uint8)
            packed = np.packbits(draws, axis=-1, bitorder='little')
            packed_uint64 = packed.view(nptype).flatten()
            loss_events[idxs] = packed_uint64

        # Handle event cancellation if desired (commented out in original)
        # canceled_events = np.logical_and(gain_events, loss_events)
        # gain_events[canceled_events] = 0
        # loss_events[canceled_events] = 0

        # Update: gain then loss
        updated_state = np.bitwise_or(parent, gain_events)  # Gain new traits
        updated_state = np.bitwise_and(updated_state, np.bitwise_not(loss_events))  # Remove lost traits
        # Store updated node state
        sim[node_map[node], :] = updated_state

        # print(f"Node {node.name} Completed")

    print("Completed Tree Simulation Sucessfully")
    lineages = sim[[node_map[node] for node in self.tree], :]
    # return lineages
    res = compile_results_KDE_bit_async(self, lineages, bits = bits, nptype = nptype)

    return res


### Compiling simulation results into null models and perfoming p-value calculation

def compile_results(self,sim,obspairs = []):
    """
    compiles simulation results from give filled simulation np array. Handles calling of compilation funtion based on run options
    : param sim: A filled simulation matrix from a simulation method
    : param obspairs: obsered values for each pair considered, defaults to `self.obspairs`
    """
    if self.kde:
        if self.parallel:
            return compile_results_KDE_async(self,sim,obspairs)
        return compile_results_KDE(self,sim,obspairs)
    elif self.parallel:
        return compile_results_async(self,sim,obspairs)
    else:
        return compile_results_sync(self,sim,obspairs)

def compile_results_sync(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials","direction", "p-value_ant", "p-value_syn", "p-value", "significant", "e_occ", "o_occ", "median", "iqr", "effect size"]}
    
    # Gather the pairs from the object
    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)
    # Extracting the relevant slices for pairs
    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    # Initialize arrays to store results
    syn = np.zeros(len(pairs))
    ant = np.zeros(len(pairs))
    means = np.zeros(len(pairs))
    medians = np.zeros(len(pairs))
    iqrs = np.zeros(len(pairs))

    for roll in range(tq.shape[-1]):
        rolled_tq = np.roll(tq, roll, axis=-1)  # Roll along the last axis
        rolled_cooc = self.pair_statistic(tp, rolled_tq)

        syn += np.sum(rolled_cooc >= obspairs[:, np.newaxis], axis=-1)
        ant += np.sum(rolled_cooc <= obspairs[:, np.newaxis], axis=-1)

        # Accumulate statistics for means, medians, and IQRs
        means += np.mean(rolled_cooc, axis=-1)
        medians += np.median(rolled_cooc, axis=-1)
        q75, q25 = np.percentile(rolled_cooc, [75, 25], axis=-1)
        iqrs += (q75 - q25)

    sim_trials = tq.shape[-1] ** 2

    # Finalize the statistics by dividing by the number of rolls
    means /= tq.shape[-1]
    medians /= tq.shape[-1]
    iqrs /= tq.shape[-1]

    pvals_ant = ant / sim_trials
    pvals_syn = syn / sim_trials
    pvals = np.minimum(pvals_syn, pvals_ant)
    directions = np.where(pvals_ant < pvals_syn, -1, 1)
    significants = pvals < 0.05
    effects = (medians - obspairs) / np.maximum(iqrs, 1)

    res['pair'] = [(p, q) for p, q in pairs]
    res['first'] = pairs[:, 0].tolist()
    res['second'] = pairs[:, 1].tolist()
    res['num_pair_trials'] = [sim_trials] * len(pairs)
    res['o_occ'] = obspairs.tolist()
    res['e_occ'] = means.tolist()
    res['p-value_ant'] = pvals_ant.tolist()
    res['p-value_syn'] = pvals_syn.tolist()
    res['p-value'] = pvals.tolist()
    res['direction'] = directions.tolist()
    res['significant'] = significants.tolist()
    res['median'] = medians.tolist()
    res['iqr'] = iqrs.tolist()
    res['effect size'] = effects.tolist()

    return pd.DataFrame.from_dict(res)

def compile_results_async(self, sim, obspairs = []):

    def process_pair(ind, p, q, obs):
        tp, tq = sim[:, p, :], sim[:, q, :]
        ant, syn = 0,0
        means = []
        medians = []
        iqrs = []
        for roll in range(tq.shape[1]):
            cooc = self.pair_statistic(tp,np.roll(tq,roll, axis = 1))
            syn += np.sum(cooc >= obs)
            ant += np.sum(cooc <= obs)
            means.append(np.mean(cooc))
            medians.append(np.median(cooc))
            q75, q25 = np.percentile(cooc, [75, 25])
            iqr = q75 - q25
            iqrs.append(iqr)

        sim_trials = tq.shape[1] ** 2
        return (p, q, sim_trials, obs, syn, ant, means, medians, iqrs)

    obspairs = obspairs or self.obspairs
    pair_stats = Parallel(n_jobs=-1, batch_size=10, return_as='generator',verbose=10)(delayed(process_pair)(ind, p, q, obs) for ind, ((p, q), obs) in enumerate(zip(self.pairs, obspairs))) # type: ignore

    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials","direction", "p-value_ant", "p-value_syn", "p-value", "significant", "e_occ", "o_occ", "median", "iqr", "effect size"]} # type: ignore
   
    for pair in pair_stats: 
        p, q, sim_trials, obs, syn, ant, means, medians, iqrs = pair # type: ignore
        update_result_dict2(res, p, q, sim_trials, obs, syn, ant, means, medians, iqrs) 
        
    return pd.DataFrame.from_dict(res)

def calculate_trait_data(self, sim, num_traits):
    """
    Formats and returns simulation statistics for each trait in the simulation run of `self`
    """
    sums = (sim > 0).sum(axis=0)
    median_vals = np.median(sums, axis=1)
    q75, q25 = np.percentile(sums, [75, 25], axis=1)
    iqr_vals = q75 - q25

    return pd.DataFrame({
        "trait": np.arange(num_traits),
        "mean": sums.mean(axis=1),
        "std": sums.std(axis=1),
        "iqr": iqr_vals,
        "median": median_vals
    })

def update_result_dict2(res: dict[str,list], p, q, sim_trials, obs, syn, ant, means, medians, iqrs):
    """
    updates a result dictionary, only to be used within compile results
    """
    res['pair'].append((p, q))
    res['first'].append(p)
    res['second'].append(q)
    res['num_pair_trials'].append(sim_trials)
    res['o_occ'].append(obs)
    res['e_occ'].append(np.mean(means))
    pval_ant = ant / sim_trials 
    pval_syn = syn / sim_trials 
    res['p-value_ant'].append(pval_ant)
    res['p-value_syn'].append(pval_syn)
    res['p-value'].append(min(pval_syn,pval_ant))
    res['direction'].append(-1 if pval_ant < pval_syn else 1)
    res['significant'].append(res['p-value'][-1] < .05)
    median = np.mean(medians)
    iqr = np.mean(iqrs)
    res['median'].append(median)
    res['iqr'].append(iqr)
    res['effect size'].append((median - obs)/max(iqr,1))

def compile_results_KDE(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction", 
                               "p-value_ant", "p-value_syn", "p-value", "significant", 
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    medians = np.zeros(len(pairs))
    iqrs = np.zeros(len(pairs))
    kde_pvals_ant = np.zeros(len(pairs))
    kde_pvals_syn = np.zeros(len(pairs))

    all_cooc = []

    for roll in range(tq.shape[-1]):
        rolled_tq = np.roll(tq, roll, axis=-1)
        rolled_cooc = self.pair_statistic(tp, rolled_tq)
        all_cooc.append(rolled_cooc)
    
    # Stack all trials into one array for KDE
    all_cooc = np.concatenate(all_cooc, axis=-1)

    for i in range(len(pairs)):
        
        noised = all_cooc[i] + np.random.normal(0, 1e-9, size=len(all_cooc[i]))
        # print(noised)
        kde = gaussian_kde(noised,bw_method='silverman')
        cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
        kde_syn = gaussian_kde(-1*noised, bw_method='silverman')
        cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf,-x)
        # Compute KDE-based p-values
        kde_pvals_ant[i] = cdf_func_ant(obspairs[i])  # P(X ≤ observed)
        kde_pvals_syn[i] = cdf_func_syn(obspairs[i])  # P(X ≥ observed)

        # Compute statistics
        medians[i] = np.median(all_cooc[i])
        q75, q25 = np.percentile(all_cooc[i], [75, 25])
        iqrs[i] = q75 - q25

        # if i % 1 == 0:
        #     disc_pvals_ant = np.sum(all_cooc[i] <= obspairs[i]) / len(all_cooc[i])
        #     disc_pvals_syn = np.sum(all_cooc[i] >= obspairs[i]) / len(all_cooc[i])
        #     plt.rcParams.update({'font.size': 14})
        #     # --------- FIRST PLOT: Histogram with KDE ----------
        #     plt.figure(figsize=(5, 4))

        #     # Plot histogram with KDE
        #     ax = sns.histplot(noised, bins=50, kde=True, stat="density", color="cornflowerblue", alpha=0.5)
        #     sns.kdeplot(noised, color="orange", alpha=0.8)

        #     # plt.legend()
        #     plt.title(f"Null Distribution for Pair {pairs[i]}")
        #     plt.xlabel("Association Score")
        #     plt.ylabel("Probability Density")

        #     plt.show()
        if i % 1 == 0:
            disc_pvals_ant = np.sum(all_cooc[i] <= obspairs[i]) / len(all_cooc[i])
            disc_pvals_syn = np.sum(all_cooc[i] >= obspairs[i]) / len(all_cooc[i])
            plt.rcParams.update({'font.size': 14})
            # --------- FIRST PLOT: Histogram with KDE ----------
            plt.figure(figsize=(4, 3))

            # Plot histogram with KDE
            ax = sns.histplot(noised, bins=50, kde=True, stat="density", color="#9cdeac", alpha=0.5)
            sns.kdeplot(noised, color="orange", alpha=0.8)

            # Add observed value line
            # plt.axvline(obspairs[i], color="black", linestyle="dashed", label=f"Observed: {obspairs[i]:.2f}")

            # plt.legend()
            # plt.title(f"Null Distribution for Pair {pairs[i]}")
            plt.xlabel("Association Score")
            plt.ylabel("Probability Density")
            
            # plt.savefig('fig.svg', format='svg')
            plt.show()

            # --------- SECOND PLOT: Text Summary ----------
            plt.figure(figsize=(4, 2))

            # Create a text box with the relevant statistics
            textstr = (
                f'KDE P(X≤obs): {kde_pvals_ant[i]:.3e}\n'
                f'KDE P(X≥obs): {kde_pvals_syn[i]:.3e}\n'
                f'Disc P(X≤obs): {disc_pvals_ant:.3e}\n'
                f'Disc P(X≥obs): {disc_pvals_syn:.3e}'
            )

            # Add the text box to the plot
            plt.gca().axis('off')  # Turn off axes
            plt.text(0.5, 0.5, textstr,
                    fontsize=12,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            plt.title(f"Statistical Summary for Pair {pairs[i]}")
            plt.show()


    # Compute p-values, directionality, significance, and effect sizes
    pvals = np.minimum(kde_pvals_syn, kde_pvals_ant)
    directions = np.where(kde_pvals_ant < kde_pvals_syn, -1, 1)
    significants = pvals < 0.05
    effects = (medians - obspairs) / np.maximum(iqrs, 1)

    # Store results
    res['pair'] = [(p, q) for p, q in pairs]
    res['first'] = pairs[:, 0].tolist()
    res['second'] = pairs[:, 1].tolist()
    res['num_pair_trials'] = [tq.shape[-1] ** 2] * len(pairs)
    res['o_occ'] = obspairs.tolist()
    res['e_occ'] = medians.tolist()
    res['p-value_ant'] = kde_pvals_ant.tolist()
    res['p-value_syn'] = kde_pvals_syn.tolist()
    res['p-value'] = pvals.tolist()
    res['direction'] = directions.tolist()
    res['significant'] = significants.tolist()
    res['median'] = medians.tolist()
    res['iqr'] = iqrs.tolist()
    res['effect size'] = effects.tolist()

    return pd.DataFrame.from_dict(res)

def compile_results_KDE_async(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction", 
                               "p-value_ant", "p-value_syn", "p-value", "significant", 
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    # Generate rolled co-occurrence matrices in parallel
    def compute_rolled_cooc(roll):
        rolled_tq = np.roll(tq, roll, axis=-1)
        return self.pair_statistic(tp, rolled_tq)

    all_cooc = Parallel(n_jobs=-1,verbose=10, batch_size= 10)(delayed(compute_rolled_cooc)(roll) for roll in range(tq.shape[-1]))
    all_cooc = np.vstack(all_cooc)

    def compute_pair_stats(i):
        noised = all_cooc[i] + np.random.normal(0, 1e-9, size=len(all_cooc[i]))

        kde = gaussian_kde(noised,bw_method='silverman')
        cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
        kde_syn = gaussian_kde(-1*noised, bw_method='silverman')
        cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf,-x)

        kde_pval_ant = cdf_func_ant(obspairs[i])
        kde_pval_syn = cdf_func_syn(obspairs[i])
        med = np.median(all_cooc[i])
        q75, q25 = np.percentile(all_cooc[i], [75, 25])
        iqr = q75 - q25

        # Visualization every 100 pairs (optional)
        if i % 100 == 0:
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(noised, bins=50, kde=False, stat="density", color="blue", alpha=0.5)
            plt.axvline(obspairs[i], color="black", linestyle="dashed", label=f"Observed: {obspairs[i]:.2f}")
            plt.text(obspairs[i], ax.get_ylim()[1], 
                    f'KDE P(X≤obs): {kde_pval_ant:.3e}\n'
                    f'KDE P(X≥obs): {kde_pval_syn:.3e}',
                    fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7))
            plt.legend()
            plt.title(f"Histogram for Pair {i}")
            
            plt.figure(figsize=(6, 4))
            sns.kdeplot(noised, color="red", alpha=0.5)
            plt.axvline(obspairs[i], color="black", linestyle="dashed", label=f"Observed: {obspairs[i]:.2f}")
            plt.legend()
            plt.title(f"KDE for Pair {i}")

        return (i, kde_pval_ant, kde_pval_syn, med, iqr)

    # Run computations in parallel
    results = Parallel(n_jobs=-1, batch_size=25, return_as='generator',verbose=10)(delayed(compute_pair_stats)(i) for i in range(len(pairs)))

    # Convert results into structured output
    for i, kde_pval_ant, kde_pval_syn, med, iqr in results:
        pval = min(kde_pval_syn, kde_pval_ant)
        direction = -1 if kde_pval_ant < kde_pval_syn else 1
        significant = pval < 0.05
        effect_size = (med - obspairs[i]) / max(iqr, 1)

        res["pair"].append(tuple(pairs[i]))
        res["first"].append(pairs[i, 0])
        res["second"].append(pairs[i, 1])
        res["num_pair_trials"].append(tq.shape[-1] ** 2)
        res["o_occ"].append(obspairs[i])
        res["e_occ"].append(med)
        res["p-value_ant"].append(kde_pval_ant)
        res["p-value_syn"].append(kde_pval_syn)
        res["p-value"].append(pval)
        res["direction"].append(direction)
        res["significant"].append(significant)
        res["median"].append(med)
        res["iqr"].append(iqr)
        res["effect size"].append(effect_size)

    return pd.DataFrame.from_dict(res)

def compile_results_KDE_bit(self, sim, obspairs=[], batch_size=1000, bits = 64, nptype = np.uint64):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction", 
                               "p-value_ant", "p-value_syn", "p-value", "significant", 
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    def circular_bitshift_right(arr: np.ndarray, k: int) -> np.ndarray:
        """Perform a circular right bit shift on all np.uint64 entries in an array."""
        k %= bits
        return np.bitwise_or(np.right_shift(arr, k), np.left_shift(arr, bits - k))
    
    def sum_all_bits(arr: np.ndarray) -> np.ndarray:
        """Compute the sum of 1s for all 64 bit positions in an array of uint64 values."""
        bit_sums = np.zeros((bits, arr.shape[-1]), dtype=np.float64)
        for i in range(bits):
            bit_sums[i] = np.sum((arr >> i) & 1, dtype=nptype, axis=0)
        return bit_sums

    def compute_bitwise_cooc(tp, tq):
        """Compute bitwise co-occurrence statistics for a batch."""
        cooc_batch = []
        for k in range(bits):
            shifted = circular_bitshift_right(tq, k)
            a = sum_all_bits(tp & shifted) + 0.01
            b = sum_all_bits(tp & ~shifted) + 0.01
            c = sum_all_bits(~tp & shifted) + 0.01
            d = sum_all_bits(~tp & ~shifted) + 0.01
            cooc_batch.append(np.log((a * d) / (b * c)))
        return np.vstack(cooc_batch)  # Shape: (64, batch_size)

    print("Computing Co-occurrence Scores...")
    all_medians, all_iqrs, all_pvals_ant, all_pvals_syn = [], [], [], []

    for index in range(0, len(pairs), batch_size):
        print(f"Processing Batch {index}-{min(index+batch_size,len(pairs))}")
        pair_batch = pairs[index: index + batch_size]
        tp = sim[:, pair_batch[:, 0]]
        tq = sim[:, pair_batch[:, 1]]

        # Compute bitwise co-occurrence in batches
        batch_cooc = compute_bitwise_cooc(tp,tq).T

        # Add small noise
        noised_batch_cooc = batch_cooc + np.random.normal(0, 1e-9, size=batch_cooc.shape)

        def compute_kde_stats(i):
            kde = gaussian_kde(noised_batch_cooc[i],bw_method='silverman')
            cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
            kde_syn = gaussian_kde(-1*noised_batch_cooc[i], bw_method='silverman')
            cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf,-x)
            kde_pval_ant = cdf_func_ant(obspairs[index + i])  # P(X ≤ observed)
            kde_pval_syn = cdf_func_syn(obspairs[index + i])  # P(X ≥ observed)
            med = np.median(batch_cooc[i])
            q75, q25 = np.percentile(batch_cooc[i], [75, 25])
            iqr = q75 - q25
            return kde_pval_ant, kde_pval_syn, med, iqr

        print("Computing P-Values for Batch...")
        results = Parallel(n_jobs=-1, verbose=10, batch_size=25)(
            delayed(compute_kde_stats)(i) for i in range(len(pair_batch))
        )

        kde_pvals_ant, kde_pvals_syn, medians, iqrs = map(np.array, zip(*results))

        all_medians.extend(medians.tolist())
        all_iqrs.extend(iqrs.tolist())
        all_pvals_ant.extend(kde_pvals_ant.tolist())
        all_pvals_syn.extend(kde_pvals_syn.tolist())

        # Store results efficiently
        res["pair"].extend([tuple(p) for p in pair_batch])
        res["first"].extend(pair_batch[:, 0].tolist())
        res["second"].extend(pair_batch[:, 1].tolist())
        res["num_pair_trials"].extend([sim.shape[1] ** 2] * len(pair_batch))
        res["o_occ"].extend(obspairs[index: index + len(pair_batch)].tolist())
        res["e_occ"].extend(medians.tolist())
        print(f"Completed Batch {index}-{min(index+batch_size,len(pairs))}")

    # Compute p-values, directionality, significance, and effect sizes
    pvals = np.minimum(all_pvals_syn, all_pvals_ant)
    directions = np.where(np.array(all_pvals_ant) < np.array(all_pvals_syn), -1, 1)
    significants = pvals < 0.05
    effects = (obspairs - np.array(all_medians)) / np.maximum(np.array(all_iqrs) * 1.349, 1)

    # Final results storage
    res["p-value_ant"], res["p-value_syn"], res["p-value"] = all_pvals_ant, all_pvals_syn, pvals.tolist()
    res["direction"], res["significant"] = directions.tolist(), significants.tolist()
    res["median"], res["iqr"], res["effect size"] = all_medians, all_iqrs, effects.tolist()

    return pd.DataFrame.from_dict(res)

def compile_results_KDE_bit_async(
    self, 
    sim: np.ndarray, 
    obspairs: List[float] = [], 
    batch_size: int = 1000,
    bits = 64,
    nptype = np.uint64
) -> pd.DataFrame:
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
        "pair": [], "first": [], "second": [], "num_pair_trials": [], 
        "direction": [], "p-value_ant": [], "p-value_syn": [], "p-value": [], 
        "significant": [], "e_occ": [], "o_occ": [], "median": [], "iqr": [], 
        "effect size": []
    }

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

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
            
        # cdf_func = kde.integrate_box_1d
        # kde_pval_ant = cdf_func(-np.inf, observed_value)  # P(X ≤ observed)
        # kde_pval_syn = 1 - kde_pval_ant  # P(X ≥ observed)
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
            "num_pair_trials": [sim_readonly.shape[1] ** 2] * len(pair_batch),
            "o_occ": obspairs[index: index + len(pair_batch)].tolist(),
            "e_occ": medians.tolist(),
            "median": medians.tolist(),
            "iqr": iqrs.tolist(),
            "p-value_ant": kde_pvals_ant.tolist(),
            "p-value_syn": kde_pvals_syn.tolist(),
            "p-value": np.minimum(kde_pvals_syn, kde_pvals_ant).tolist(),
            "direction": np.where(kde_pvals_ant < kde_pvals_syn, -1, 1).tolist(),
            "significant": (np.minimum(kde_pvals_syn, kde_pvals_ant) < 0.05).tolist(),
            "effect size": ((obspairs[index: index + len(pair_batch)]-medians) / np.maximum(iqrs * 1.349, 1)).tolist(),
        }

        return batch_res

    num_pairs = len(pairs)
    batch_indices = range(0, num_pairs, batch_size)

    print(f"Processing Batches, Total: {num_pairs//batch_size + 1}")

    # Run batches in parallel, passing a read-only copy of sim
    batch_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_batch)(index, sim) for index in batch_indices
    )

    print("Aggregating Results...")

    # Merge batch results
    for batch_res in batch_results:
        for key in res.keys():
            res[key].extend(batch_res[key])

    return pd.DataFrame.from_dict(res)





def simulate_glrates_manual(tree, gains, losses, gain_subsize, loss_subsize, MULTIPLIER, NUM_TRIALS, 
                     dists, loss_dists, root_states):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml.
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch.
    For each trait, only simulates on branches beyond a certain distance from the root.
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction.
    """
    from simphyni import TreeSimulator
    import pandas as pd
    import numpy as np

    # Preprocess and setup
    all_nodes = list(tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(gains)
    sim = np.zeros((num_nodes, num_traits, NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    # Compute total branch length and tree range
    bl = sum(i.dist for i in tree.traverse())  # type: ignore
    tree_range = tree.get_farthest_leaf()[1]
    assert isinstance(tree_range, float)

    # Compute gain and loss modifiers
    gain_mod = np.nan_to_num(tree_range / dists, nan=1)
    gain_mod[gain_mod > 1] = 1
    loss_mod = np.nan_to_num(tree_range / loss_dists, nan=1)
    loss_mod[loss_mod > 1] = 1

    # Compute gain and loss rates
    gain_rates = gains / (gain_subsize * MULTIPLIER)
    loss_rates = losses / (loss_subsize * MULTIPLIER)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    # Compute node distances
    node_dists = {tree: tree.dist or 0}
    for node in tree.traverse():  # type: ignore
        if node not in node_dists:
            node_dists[node] = node_dists[node.up] + node.dist

    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    # Simulate traits across the tree
    for node in tree.traverse():  # type: ignore
        if node.up is None:
            root = root_states > 0
            sim[node_map[node], root, :] = True
            continue

        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * MULTIPLIER
        node_total_dist = node_dists[node]

        # Apply distance-based thresholds
        applicable_traits_gains = node_total_dist >= dists
        applicable_traits_losses = node_total_dist >= loss_dists
        gain_events = np.zeros((num_traits, NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, NUM_TRIALS), dtype=bool)

        # Simulate gain and loss events
        gain_events[applicable_traits_gains] = np.random.binomial(
            node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], 
            (applicable_traits_gains.sum(), NUM_TRIALS)
        ) > 0
        loss_events[applicable_traits_losses] = np.random.binomial(
            node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], 
            (applicable_traits_losses.sum(), NUM_TRIALS)
        ) > 0

        # Handle event cancellation
        gain_mask = np.logical_not(np.logical_xor(parent, gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent, loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node], :, :] = parent.copy()
        sim[node_map[node], :, :][gain_events] = True
        sim[node_map[node], :, :][loss_events] = False

    # Extract lineages
    lineages = sim[[node_map[node] for node in tree], :, :]
    
    return lineages

def simulate_glrates_manual_full_tree(tree, gains, losses, gain_subsize, loss_subsize, MULTIPLIER, NUM_TRIALS, 
                     dists, loss_dists, root_states):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml.
    Sums gains and losses to a total number of events then allocates each event to a 
    branch on the tree with probability proportional to the length of the branch.
    For each trait, only simulates on branches beyond a certain distance from the root.
    This threshold is chosen as the first branch where a trait arises in the ancestral trait reconstruction.
    """
    from simphyni import TreeSimulator
    import pandas as pd
    import numpy as np

    # Preprocess and setup
    all_nodes = list(tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(gains)
    sim = np.zeros((num_nodes, num_traits, NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    # Compute total branch length and tree range
    bl = sum(i.dist for i in tree.traverse())  # type: ignore
    tree_range = tree.get_farthest_leaf()[1]
    assert isinstance(tree_range, float)

    # Compute gain and loss modifiers
    gain_mod = np.nan_to_num(tree_range / dists, nan=1)
    gain_mod[gain_mod > 1] = 1
    loss_mod = np.nan_to_num(tree_range / loss_dists, nan=1)
    loss_mod[loss_mod > 1] = 1

    # Compute gain and loss rates
    gain_rates = gains / (gain_subsize * MULTIPLIER)
    loss_rates = losses / (loss_subsize * MULTIPLIER)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    # Compute node distances
    node_dists = {tree: tree.dist or 0}
    for node in tree.traverse():  # type: ignore
        if node not in node_dists:
            node_dists[node] = node_dists[node.up] + node.dist

    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    # Simulate traits across the tree
    for node in tree.traverse():  # type: ignore
        if node.up is None:
            root = root_states > 0
            sim[node_map[node], root, :] = True
            continue

        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * MULTIPLIER
        node_total_dist = node_dists[node]

        # Apply distance-based thresholds
        applicable_traits_gains = node_total_dist >= dists
        applicable_traits_losses = node_total_dist >= loss_dists
        gain_events = np.zeros((num_traits, NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, NUM_TRIALS), dtype=bool)

        # Simulate gain and loss events
        gain_events[applicable_traits_gains] = np.random.binomial(
            node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis], 
            (applicable_traits_gains.sum(), NUM_TRIALS)
        ) > 0
        loss_events[applicable_traits_losses] = np.random.binomial(
            node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis], 
            (applicable_traits_losses.sum(), NUM_TRIALS)
        ) > 0

        # Handle event cancellation
        gain_mask = np.logical_not(np.logical_xor(parent, gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent, loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node], :, :] = parent.copy()
        sim[node_map[node], :, :][gain_events] = True
        sim[node_map[node], :, :][loss_events] = False

    
    return sim


def simulate_branch_ctmp(state_bits, gain_rates, loss_rates, gain_mask, loss_mask, node_dist):
    num_traits, num_trials = state_bits.shape
    bits = num_trials

    # Initial rates
    rates = np.where(state_bits, loss_rates[:, None], gain_rates[:, None])
    rates = rates * (gain_mask | loss_mask)[:, None]
    rates[rates == 0] = np.inf

    wait_times = np.random.exponential(1 / rates)
    flat_waits = wait_times.flatten()
    flat_indices = np.arange(flat_waits.size)

    order = np.argsort(flat_waits)
    sorted_waits = flat_waits[order]
    sorted_indices = flat_indices[order]

    dist_remaining = node_dist
    event_ptr = 0

    while event_ptr < sorted_waits.size and sorted_waits[event_ptr] < dist_remaining:
        wait_time = sorted_waits[event_ptr]
        dist_remaining -= wait_time

        linear_idx = sorted_indices[event_ptr]
        trait_idx = linear_idx // bits
        trial_idx = linear_idx % bits

        # Flip bit
        state_bits[trait_idx, trial_idx] ^= 1

        # Resample
        new_rate = loss_rates[trait_idx] if state_bits[trait_idx, trial_idx] else gain_rates[trait_idx]
        if (gain_mask[trait_idx] or loss_mask[trait_idx]) and new_rate > 0:
            new_wait = np.random.exponential(1 / new_rate)

            # Insert into sorted array
            insert_pos = event_ptr + 1
            while insert_pos < sorted_waits.size and sorted_waits[insert_pos] < wait_time + new_wait:
                insert_pos += 1

            sorted_waits = np.insert(sorted_waits, insert_pos, wait_time + new_wait)
            sorted_indices = np.insert(sorted_indices, insert_pos, linear_idx)

        event_ptr += 1

    return state_bits
