import os
import numpy as np
import pandas as pd
from scipy import stats
from ete3 import Tree
from .SimulationMethods import *
from .Utils import *
from .PairStatistics import *
from itertools import combinations
from scipy.stats import fisher_exact
import random
import statsmodels.stats.multitest as sm
from matplotlib.colors import LinearSegmentedColormap
from typing import Union
from typing import Literal
from typing import List, Tuple, Set, Dict
import plotly.express as px


class TreeSimulator:
    MULTIPLIER = 1e12
    NUM_TRIALS = 64
    TREE_DISTS = {}

    def __init__(self, tree, pastmlfile, obsdatafile):
        self.treefile = tree  # Initialize the ETE3 Tree object
        self.pastmlfile = pastmlfile
        self.obsdatafile = obsdatafile
        self.leaves = []
        self.node_map = {}
        self.simulation_result: pd.DataFrame = pd.DataFrame()
        self.trait_data: pd.DataFrame = pd.DataFrame()
        self.tree = None
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Prepares necessary data for the simulation, including reading pastml and observation files,
        and initializing various simulation parameters.
        Accepts either file paths or pandas DataFrames directly.
        """

        # Handle pastml input
        if isinstance(self.pastmlfile, pd.DataFrame):
            self.pastml = self.pastmlfile.copy()
        else:
            self.pastml = pd.read_csv(self.pastmlfile, index_col=0)

        # Handle observed data input
        if isinstance(self.obsdatafile, pd.DataFrame):
            self.obsdf = self.obsdatafile.copy()
        else:
            self.obsdf = pd.read_csv(self.obsdatafile, index_col=0)

        # Run validation and preprocessing
        self._check_pastml_data()
        self._process_obs_data()


    def _process_obs_data(self):
        """
        Processes the observation data to be used in the simulation.
        """
        self.pastml = self.pastml.set_index('gene').loc[self.obsdf.columns].reset_index(names='gene')
        self.mapping = dict(zip(self.pastml.index.astype(str),self.pastml["gene"]))
        self.mappingr = dict(zip(self.pastml["gene"],self.pastml.index.astype(str)))
        self.obsdf[self.obsdf > 0.5] = 1
        self.obsdf.fillna(0, inplace=True)
        self.obsdf = self.obsdf.astype(int)
        self.obsdf.index = self.obsdf.index.astype(str)
        self.obsdf.rename(columns = self.mappingr, inplace=True)
        

    def _check_pastml_data(self):
        """
        Checks for a well-formed pastml file with all necessary column labels
        [gene, gains, losses, dist]
        """
        assert "gene" in self.pastml, "pastml file should have label `gene`"
        assert "gains" in self.pastml, "pastml file should have label `gains`"
        assert "losses" in self.pastml, "pastml file should have label `losses`"
        assert "dist" in self.pastml, "pastml file should have label `dist`"
        assert "loss_dist" in self.pastml, "pastml file should have label `loss_dist`"

    def initialize_simulation_parameters(self, pair_statistic = None, prevalence_threshold = 0.05, collapse_theshold = 0.001, single_trait = False, vars = None, targets = None, kde = False):
        """
        Initializes simulation parameters from pastml file and sets the pair_statistic method for run
        Must be run before each simulation

        :param pair_statistic: a fucniton that takes two lists of elements and outputs an interpretable score (float)
        """
        if not self.tree:
            self.tree = Tree(self.treefile, 1)
        # self.tree = Tree(self.treefile, 1)
        def check_internal_node_names(tree):
            internal_names = set()
            for node in tree.traverse():
                if not node.is_leaf():
                    if node.name in internal_names:
                        return False
                    internal_names.add(node.name)
            return True
        if not check_internal_node_names(self.tree):
            for idx, node in enumerate(self.tree.iter_descendants("levelorder")):
                if not node.is_leaf():
                    name = f"internal_{idx}"
                    node.name = name
                    # print(name)
        self.pair_statistic = pair_statistic or PairStatistics._vectorized_pair_statistic
        self.gains = np.array(self.pastml['gains'])
        self.losses = np.array(self.pastml['losses'])
        self.dists = np.array(self.pastml['dist'])
        self.loss_dists = np.array(self.pastml['loss_dist'])
        self.gain_subsize = np.array(self.pastml['gain_subsize'])
        self.loss_subsize = np.array(self.pastml['loss_subsize'])
        self.root_states = np.array(self.pastml['root_state'])
        self.dists[self.dists == np.inf] = 0
        self.loss_dists[self.loss_dists == np.inf] = 0
        self.kde = kde

        self.obsdf_modified = self._collapse_tree_tips(collapse_theshold)
        if vars and targets:
            self.set_pairs(vars,targets, by = 'name')
        else:
            self.pairs, self.obspairs = self._get_pair_data(self.obsdf_modified, self.obsdf_modified, prevalence_threshold,single_trait)

    def set_pairs(self, vars, targets, by:Literal['number','name'] = 'name'):
        if by == 'name':
            obsdf = self.get_obs()
            self.pairs, self.obspairs = self._get_pair_data(obsdf[vars].rename(columns = self.mappingr),obsdf[targets].rename(columns = self.mappingr))
        else:
            obsdf = self.obsdf_modified
            self.pairs, self.obspairs = self._get_pair_data(obsdf[[str(i) for i in vars]],obsdf[[str(i) for i in targets]])


    def _collapse_tree_tips(self, threshold):
        """
        Combine leaves i,j of the obeserved trait dataframe, `obsdf`, for all leaves within
        a distacnce of threshold from eachother. Does not mutate original obsdf 

        :param threshold: a fraction of the longest branch length from root to tip
        :returns: new dataframe of combined leaves
        """
        if threshold == 0: 
            treeleaves = set(self.tree.get_leaf_names())
            self.tree.prune([i for i in self.obsdf.index if i in treeleaves], preserve_branch_length=True)
            return self.obsdf.copy()
        
        threshold = self.tree.get_distance(self.tree,self.tree.get_farthest_leaf()[0]) * threshold
        obsdf = self.obsdf.copy()
        self.tree.prune([i for i in self.obsdf.index if i in set(self.tree.get_leaf_names())], preserve_branch_length=True)
        node_queue = set(self.tree.get_leaves())
        to_prune = set()
        while(node_queue):
            current_node: TreeNode = node_queue.pop()
            sibling: TreeNode = current_node.get_sisters()[0]
            if not sibling.is_leaf():
                to_prune.add(current_node)
                continue
            distance = current_node.dist + sibling.dist
            if distance < threshold:
                # Choosing first node seen as rep
                obsdf.loc[current_node.up.name] = obsdf.loc[current_node.name] # pyright: ignore[reportOptionalMemberAccess]
                node_queue.add(current_node.up)
            else:
                to_prune.add(current_node)
                to_prune.add(sibling)
            if sibling in node_queue:
                node_queue.remove(sibling)
        
        # print(obsdf.index)
        self.tree.prune(to_prune, preserve_branch_length= True)
        obsdf[obsdf>1]=1
        return obsdf.loc(axis = 0)[tuple(map(lambda x: x.name, to_prune))]
    
    

    def _get_pair_data(self, vars: pd.DataFrame, targets: pd.DataFrame, prevalence_threshold: float = 0.00, single_trait: bool = False, batch_size = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a list of trait pairs and a NumPy array of their test statistic results 
        for all traits with prevalence above a given threshold.

        :param prevalence_threshold: Minimum prevalence for traits to be included [0,1]
        :param single_trait: True if only pairs including the first trait are wanted
        :returns: NumPy array of pairs, NumPy array of pair statistics
        """
        ### Current Funtionality only allows for same vars/target dataframes
        # Convert to NumPy arrays for fast computation
        vars_np = vars.to_numpy()
        targets_np = targets.to_numpy()
        var_cols = np.array(vars.columns)
        target_cols = np.array(targets.columns)

        valid_vars_mask = (vars_np.sum(axis=0) >= prevalence_threshold * vars_np.shape[0])
        valid_targets_mask = (targets_np.sum(axis=0) >= prevalence_threshold * targets_np.shape[0])

        valid_vars = var_cols[valid_vars_mask]
        valid_targets = target_cols[valid_targets_mask]

        if valid_vars.size == 0 or valid_targets.size == 0:
            return np.array([]), np.array([])

        # Generate all valid trait pairs
        if single_trait:
            pairs = np.array(np.meshgrid(valid_vars, valid_targets)).T.reshape(-1, 2)[:valid_vars.size]
        elif vars.equals(targets):
            pairs = np.array(list(combinations(valid_vars,2)))
        else:
            pairs = np.array(np.meshgrid(valid_vars, valid_targets)).T.reshape(-1, 2)
            pairs_sorted = np.sort(pairs, axis=1)
            pairs = np.unique(pairs_sorted, axis=0)

        all_stats = []
        # Process pairs in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            pair_vars = vars[batch_pairs[:, 0]].to_numpy()
            pair_targets = targets[batch_pairs[:, 1]].to_numpy()
            
            # Compute statistics for the current batch
            batch_stats = self.pair_statistic(pair_vars, pair_targets)
            all_stats.append(batch_stats)

        # Concatenate all batch statistics
        stats = np.concatenate(all_stats, axis=0)
        pairs = pairs.astype(int)

        return pairs, stats
    
    
    def _get_pair_data2(
        self, 
        obsdf: pd.DataFrame, 
        pairs: List[Tuple[str, str]], 
        prevalence_threshold: float = 0.00, 
        batch_size: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a list of trait pairs and a NumPy array of their test statistic results 
        for all traits with prevalence above the given threshold.

        :param obsdf: DataFrame containing the trait data
        :param pairs: List of tuples containing the column names to compute pair statistics
        :param prevalence_threshold: Minimum prevalence for traits to be included [0,1]
        :param batch_size: Number of pairs processed in a batch
        :returns: NumPy array of valid pairs, NumPy array of pair statistics
        """
        # Convert to NumPy arrays for efficient computation
        obs_np = obsdf.to_numpy()
        obs_cols = np.array(obsdf.columns)

        # Map column names to indices for fast access
        col_idx = {col: i for i, col in enumerate(obs_cols)}

        # Convert pairs from column names to indices
        pair_indices = np.array([(col_idx[str(i)], col_idx[str(j)]) for i, j in pairs])

        # Get valid trait masks based on prevalence
        prevalence_counts = (obs_np != 0).sum(axis=0)
        valid_mask = prevalence_counts >= (prevalence_threshold * obs_np.shape[0])

        # Filter valid pairs based on prevalence
        valid_pairs_mask = valid_mask[pair_indices[:, 0]] & valid_mask[pair_indices[:, 1]]
        valid_pairs = pair_indices[valid_pairs_mask]

        if valid_pairs.size == 0:
            return np.array([]), np.array([])

        # Process valid pairs in batches
        all_stats = []
        for i in range(0, len(valid_pairs), batch_size):
            batch_pairs = valid_pairs[i:i + batch_size]
            pair_vars = obs_np[:, batch_pairs[:, 0]]
            pair_targets = obs_np[:, batch_pairs[:, 1]]

            batch_stats = self.pair_statistic(pair_vars,pair_targets)
            all_stats.append(batch_stats)

        # Concatenate all batch statistics
        stats = np.concatenate(all_stats, axis=0)

        return valid_pairs, stats

    def get_obs(self):
        try:
            return self.obsdf_modified.rename(columns=self.mapping)
        except:
            return self.obsdf.rename(columns=self.mapping)

    def plot_tree(self,traits = [], title = "Tree Plot of Observations", show_tree = True):
        """
        Uses tree_plot from Utils to create a heatmap visualization with a dedrogram of the 
        tree associated with `self` and the observations on the tree after initialization
        """
        if not traits: traits = self.obsdf_modified.columns
        tree_plot(self.tree,self.obsdf_modified.rename(columns=self.mapping)[traits], title=title, show_tree=show_tree)

    def plot_simulated_tree(self, traits = [], title = "Tree Plot of Simulations") -> None:
        """
        Plots simulated trees for given traits
        """
        assert self.sim_trees, "simulation needs to have been run"
        if not traits: traits = self.sim_trees[0].columns
        tree_plot(self.tree,self.sim_trees[random.randint(0,len(self.sim_trees)-1)][traits], title = title)

    def run_simulation(self, simulation_function = None, parallel = True, bit = True, norm = True):
        """
        Runs the tree simulation and stores results.
        """
        if not simulation_function: simulation_function = simulate_distnorm 
        self.parallel = parallel
        if not bit:
            self.simulation_result, self.trait_data, self.sim_trees = simulation_function(self)
            self._post_process_simulation_results()
        else:
            # No run diagnostic files when using bit-optimized simulation. (no trait data file, no )
            # Further, not all simulation implementations are suppportet with bit optimization
            self.simulation_result = simulate_glrates_bit(self) if not norm else simulate_glrates_bit_norm(self)
            self.simulation_result['sys1'] = [self.mapping[str(i)] for i in self.simulation_result['first']]
            self.simulation_result['sys2'] = [self.mapping[str(i)] for i in self.simulation_result['second']]


    def _post_process_simulation_results(self):
        """
        Add formatting to `simulation_results` and `trait_data`
        To be run after simulate
        """
        self.simulation_result['sys1'] = [self.mapping[str(i)] for i in self.simulation_result['first']]
        self.simulation_result['sys2'] = [self.mapping[str(i)] for i in self.simulation_result['second']]
        self.trait_data['gene'] = [self.mapping[str(i)] for i in self.trait_data['trait']]
        self.trait_data['obs'] = [self.obsdf_modified[str(i)].replace(0,np.nan).count() for i in self.trait_data['trait']]
        self.trait_data['z_score'] = [None if k == 0 else (i-j)/k for i,j,k in zip(self.trait_data['obs'],self.trait_data['mean'],self.trait_data['std'])]
        self.trait_data['robust_z_score'] = [ None if k == 0 else (i - j) / k for i, j, k in zip(self.trait_data['obs'], self.trait_data['median'], self.trait_data['iqr'])]
    def get_simulation_result(self):
        """
        Returns the simulation result.
        """
        if self.simulation_result.empty:
            raise ValueError("Simulation not yet run.")
        return self.simulation_result

    def get_trait_data(self):
        """
        Returns the trait data resulting from the simulation.
        """
        if self.trait_data.empty:
            raise ValueError("Simulation was either not yet completed or run with bit-optimized computation (no trait data output)")
        return self.trait_data
    
    def set_trials(self, num_trials: int) -> None:
        """
        Set the number of trials for the simulation.
        """
        self.NUM_TRIALS = num_trials
        
    def plot_results(self, correction: Union[bool, str] = False, prevalence_range = [0,1], figure_size = -1) -> tuple[plt.Axes,pd.DataFrame]: # pyright: ignore[reportPrivateImportUsage]
        """
        Plots a heatmap of simulation results within `self`
        """
        assert not self.simulation_result.empty, "Simulation has not been run"
        res = self._filter_res(correction,prevalence_range)
        
        # Create the pivot tables
        simpiv_ant = pd.pivot_table(res, values='p-value_ant', index='sys2', columns='sys1', aggfunc='sum', sort=False)
        simpiv_syn = pd.pivot_table(res, values='p-value_syn', index='sys1', columns='sys2', aggfunc='sum', sort=False)

        all_indices = sorted(set(simpiv_ant.index).union(set(simpiv_syn.index)).union(set(simpiv_ant.columns)).union(set(simpiv_syn.columns)))
        simpiv_ant = simpiv_ant.reindex(index=all_indices, columns=all_indices)
        simpiv_syn = simpiv_syn.reindex(index=all_indices, columns=all_indices)
        simpiv_ant = simpiv_ant.fillna(1)
        simpiv_syn = simpiv_syn.fillna(1)

        combined_matrix = simpiv_ant.copy()

        # Fill the lower triangle with values from simpiv_ant
        for i in range(len(simpiv_ant)):
            for j in range(i + 1, len(simpiv_ant)):
                combined_matrix.iat[j, i] = simpiv_ant.iat[j, i]

        # Fill the upper triangle with values from simpiv_syn
        for i in range(len(simpiv_syn)):
            for j in range(i + 1, len(simpiv_syn)):
                combined_matrix.iat[i, j] = simpiv_syn.iat[i, j]

        for i in range(len(simpiv_syn)):
            combined_matrix.iat[i, i] = 1

        # Plot the heatmap
        fig, ax = plt.subplots()
        if figure_size != -1:
            fig.set_figwidth(figure_size)
            fig.set_figheight(figure_size)
        mask = np.zeros_like(combined_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = False

        with sns.axes_style("white"):
            cmap1 = LinearSegmentedColormap.from_list('light_blues', ['#0a6bf2','#ffffff'])
            cmap2 = LinearSegmentedColormap.from_list('light_reds', ['#ff711f','#ffffff'])
            grey_cmap = sns.color_palette(["grey"]).as_hex()
            sns.heatmap(combined_matrix, mask=mask, cmap=cmap1, cbar=True, cbar_kws={"shrink": 0.5, 'fraction': .05, 'pad': 0, 'label':'p-value (FDR)'}, ax=ax, square=True)
            sns.heatmap(combined_matrix, mask=~mask, cmap=cmap2, cbar=True, cbar_kws={"shrink": 0.5, 'ticks': [], 'fraction': .05}, ax=ax, square=True)
            


            diagonal_mask = np.zeros_like(combined_matrix, dtype=bool)
            np.fill_diagonal(diagonal_mask, True)
            sns.heatmap(combined_matrix, mask=~diagonal_mask, cmap=grey_cmap, cbar=False, ax=ax, square = True) # pyright: ignore[reportArgumentType]
            plt.xlabel("")
            plt.ylabel("")
            mask_1 = combined_matrix < 0.05 
            mask_2 = combined_matrix < 0.01
            mask_3 = combined_matrix < 0.001

            for i in range(combined_matrix.shape[0]):
                for j in range(combined_matrix.shape[1]):
                    annotation = ""
                    if mask_3.iloc[i, j]:
                        annotation = "***"
                    elif mask_2.iloc[i, j]:
                        annotation = "**"
                    elif mask_1.iloc[i, j]:
                        annotation = "*"
                    if annotation:
                        plt.text(j + 0.5, i + 0.5, annotation, ha='center', va='center', fontsize=10, color='black')

        plt.title("Positive (Red) and Negative (Blue) Associations")
        plt.tight_layout()
        plt.savefig('fig.svg', format='svg')
        plt.show()
        return (ax, combined_matrix)
    
    def plot_effect_size(self, correction: Union[bool, str] = False, prevalence_range = [0,1]):
        x = self._filter_res(correction,prevalence_range)
        # x['log'] = -np.log10(x['p-value'])
        # x['pair'] = [i for i in zip(x['sys1'],x['sys2'])]
        # x['effect size'] = np.log2(x['o_occ']/x['e_occ'])

        if np.any(x['p-value'] == 0):
            print(f"{sum(x['p-value'] == 0)} p-values of 0 set to 1/(10 * num_pair_trials)")
            x.loc[x['p-value'] == 0, 'p-value'] = 0.1 / x.loc[x['p-value'] == 0, 'num_pair_trials']
        
        x = x.assign(
            log=lambda x: -np.log10(x['p-value']),
            pair=lambda x: list(zip(x['sys1'], x['sys2'])) # pyright: ignore[reportArgumentType]
        )
        # filtered_x_neg = x[x['effect size'] > 0].nsmallest(6, 'p-value')
        # filtered_x_pos = x[x['effect size'] < 0].nsmallest(6, 'p-value')
        # filtered_x = pd.concat([filtered_x_pos, filtered_x_neg])
    
        # # Create a 'label' column to label only the filtered points
        # x = x.assign(label=lambda x: x['pair'].where(x['pair'].isin(filtered_x['pair']), ''))
        
        fig = px.scatter(x, x = 'effect size', y = 'log', hover_data= ['pair','direction'],
                labels={
                     'log': '-log p-value',
                     'effect size': 'Effect Size'
                 },
                #  text = x['label']
                 )
        fig.update_traces(
            textposition='middle right',  # Offset text to the right of the points
            textfont=dict(
                size=10  # Make the text smaller
            )
        )
        
        fig.update_layout(
            title='P-value vs Effect Size',
            yaxis_title='-log p-value',
            xaxis_title='Effect Size',
        )
        fig.show()


    def fisher(self, traits: list) -> None:
        """
        performs fishers exact test on the given traits in the observations
        must be exactly 2 traits
        """
        assert len(traits) == 2, "must be given two traits"
        table = [[0, 0], [0, 0]]
        d = self.obsdf_modified.rename(columns=self.mapping)
        for a1, a2 in zip(d[traits[0]], d[traits[1]]):
            table[a1][a2] += 1
        odds_ratio, p_value = fisher_exact(table)

        print("Fischer's Exact Results:")
        print(f"Odds Ratio: {odds_ratio}")
        print(f"P-value: {p_value}")
    
    def trait_suite(self, traits = []):
        """
        runs `plot_tree`, `plot_simulated_tree`, and `fischer` for a given set of traits
        must be exactly 2 traits
        """
        if not traits: traits = list(self.obsdf_modified.index)
        self.plot_tree(traits)
        self.plot_simulated_tree(traits)
        self.fisher(traits)

    def _filter_res(self, correction, prevalence_range, alpha = 0.05):
        res = self.simulation_result[self.simulation_result['first'] != self.simulation_result['second']]

        if prevalence_range != [0,1]:
            prev = self.obsdf_modified.sum()/self.obsdf_modified.count()
            to_keep = prev[(prev >= prevalence_range[0]) & (prev <= prevalence_range[1])]
            res = res[res['first'].isin(to_keep.index.astype(int))] 
            res = res[res['second'].isin(to_keep.index.astype(int))]

        if correction:  
            method = correction if type(correction) == str else 'fdr_bh'
            bhc_s = sm.multipletests(res['p-value_syn'], alpha= alpha, method = method)
            bhc_a = sm.multipletests(res['p-value_ant'], alpha= alpha, method = method)
            res.loc[:,'p-value_syn'] = bhc_s[1]
            res.loc[:,'p-value_ant'] = bhc_a[1]
            res.loc[:, 'p-value'] = np.minimum(res.loc[:,'p-value_syn'],res.loc[:,'p-value_ant'])

            # bhc = sm.multipletests(res['p-value'], alpha= alpha, method = method)

            # res.loc[:,'p-value'] = bhc[1]
            # res.loc[res['direction'] == -1, 'p-value_ant'] = res.loc[res['direction'] == -1, 'p-value']
            # res.loc[res['direction'] == 1, 'p-value_syn'] = res.loc[res['direction'] == 1, 'p-value']
            # res.loc[res['direction'] == 1, 'p-value_ant'] = 1# - res.loc[res['direction'] == 1, 'p-value']
            # res.loc[res['direction'] == -1, 'p-value_syn'] = 1# - res.loc[res['direction'] == -1, 'p-value']

        return res

    def get_top_results(self, prevalence_range=[0,1], top=15, direction: Literal[-1,0,1]=0, 
                        by: Literal['p-value','effect size']='effect size', alpha=0.05):
        """
        Gets the most significant results from the simulation run and returns a table with multiple testing corrections (naive, BH, BY, Bonferroni).

        :param correction: multiple test correction method according to stats models multiple test options, True for fdr, and False for no correction
        :param prevalence_range: range of trait prevalence to be considered for correction and outputs
        :param top: number of top results to report, negative values rerurn all but bottom |i|
        :param direciton: filter on direction of association to report, -1 for negative assosiation, 1 for positive association, and 0 for both 
        """

        # get uncorrected results
        res = self._filter_res(False, prevalence_range, alpha=alpha)

        if direction:
            res = res[res['direction'] == direction]

        prev = self.obsdf_modified.mean()
        res['prevalence_sys1'] = res['first'].astype(str).map(prev)
        res['prevalence_sys2'] = res['second'].astype(str).map(prev)
        res['effect size'] = abs(res['effect size'])

        # ---- corrections ----
        pvals = res['p-value'].values

        # Benjamini-Hochberg
        res['pval_bh'] = sm.multipletests(pvals, alpha=alpha, method='fdr_bh')[1]

        # Benjamini-Yekutieli
        res['pval_by'] = sm.multipletests(pvals, alpha=alpha, method='fdr_by')[1]

        # Bonferroni
        res['pval_bonf'] = sm.multipletests(pvals, alpha=alpha, method='bonferroni')[1]

        # Naive (just keep the original p-value)
        res['pval_naive'] = res['p-value']

        # ---- select columns ----
        out = res[['sys1','sys2','direction','effect size',
                'prevalence_sys1','prevalence_sys2',
                'pval_naive','pval_bh','pval_by','pval_bonf']]

        out = out.sort_values(by=by, ascending=(by == 'p-value')).head(top)

        return out
