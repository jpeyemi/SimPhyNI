import numpy as np
import pandas as pd
from ete3 import Tree
from .SimulationMethods import *
from .Utils import *
import statsmodels.stats.multitest as sm
from matplotlib.colors import LinearSegmentedColormap
from typing import Literal
from typing import List, Tuple, Set, Dict
from .PairStatistics import *
import numpy as np
from scipy.special import loggamma

class TreeSimulator:

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
        self.pastml = self.pastml.set_index('gene')
        self.obsdf[self.obsdf > 0.5] = 1
        self.obsdf.fillna(0, inplace=True)
        self.obsdf = self.obsdf.astype(int)
        

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

    def initialize_simulation_parameters(self, prevalence_threshold = 0.05, collapse_theshold = 0.000, single_trait = False, vars = None, targets = None, pre_filter = True):
        """
        Initializes simulation parameters from pastml file and sets the pair_statistic method for run
        Must be run before each simulation

        :param pair_statistic: a fucniton that takes two lists of elements and outputs an interpretable score (float)
        """
        if not self.tree:
            self.tree = Tree(self.treefile, 1)
        assert(type(self.tree) == TreeNode)

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

        self.pair_statistic = PairStatistics._log_odds_ratio_statistic

        self.obsdf_modified = self._collapse_tree_tips(collapse_theshold)
        if not vars: vars = self.obsdf_modified.columns
        if single_trait: vars = [vars[0]]
        if not targets: targets = self.obsdf_modified.columns
        self.set_pairs(vars,targets, prevalence_threshold=prevalence_threshold, pre_filter = pre_filter)

    def set_pairs(self, vars, targets, prevalence_threshold: float = 0.00, batch_size = 1000, pre_filter = True):
        self.pairs, self.obspairs = self._get_pair_data(self.obsdf[vars],self.obsdf[targets], prevalence_threshold=prevalence_threshold,batch_size=batch_size,pre_filter = pre_filter)


    def _collapse_tree_tips(self, threshold):
        """
        Combine leaves i,j of the obeserved trait dataframe, `obsdf`, for all leaves within
        a distacnce of threshold from eachother. Does not mutate original obsdf 

        :param threshold: a fraction of the longest branch length from root to tip
        :returns: new dataframe of combined leaves
        """
        assert(type(self.tree) == TreeNode)
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
                if current_node.up:
                    # Choosing first node seen as rep
                    obsdf.loc[current_node.up.name] = obsdf.loc[current_node.name]
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
    
    def _get_pair_data(self, vars: pd.DataFrame, targets: pd.DataFrame, prevalence_threshold: float = 0.00, batch_size = 1000, pre_filter = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a list of trait pairs and a NumPy array of their test statistic results 
        for all traits with prevalence above a given threshold.

        :param prevalence_threshold: Minimum prevalence for traits to be included [0,1]
        :param single_trait: True if only pairs including the first trait are wanted
        :returns: NumPy array of pairs, NumPy array of pair statistics
        """

        # Filter by prevalence
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


        def fisher_significant_pairs(vars: pd.DataFrame, targets: pd.DataFrame, valid_vars, valid_targets, pval_threshold: float = 0.05):
            X = vars.to_numpy().astype(bool)
            Y = targets.to_numpy().astype(bool)
            n = X.shape[0]

            # Compute all pairwise contingency counts efficiently
            a = X.T @ Y                      # (n_vars x n_targets) both=1
            sX = X.sum(axis=0)               # (n_vars,)
            sY = Y.sum(axis=0)               # (n_targets,)

            b = sX[:, None] - a              # (i=1, j=0)
            c = sY[None, :] - a              # (i=0, j=1)
            d = n - (a + b + c)              # both=0

            # Compute marginals
            row1 = a + b
            row2 = c + d
            col1 = a + c
            n_all = n

            # Log-binomial function
            def logC(n, k):
                return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

            # log probability of observed a under null
            logp_obs = logC(row1, a) + logC(row2, col1 - a) - logC(n_all, col1)
            p_obs = np.exp(logp_obs)

            # approximate two-sided p-value as 2 * min(one-sided, 1)
            p_two = np.minimum(1.0, 2 * p_obs)

            # Mask NaNs and invalids
            p_two[np.isnan(p_two)] = 1.0

            # Apply significance threshold
            sig_mask = p_two < pval_threshold

            # Get indices of significant pairs
            i_idx, j_idx = np.where(sig_mask)

            sig_pairs = np.column_stack((valid_vars[i_idx], valid_targets[j_idx]))
            sig_pvals = p_two[i_idx, j_idx]

            return sig_pairs, sig_pvals


        # Filter by fishers exact test if prefilter enabled
        if pre_filter:
            pairs, pvals = fisher_significant_pairs(vars[valid_vars],targets[valid_targets],valid_vars,valid_targets)
        else:
            X, Y = np.meshgrid(valid_vars, valid_targets, indexing='ij')
            pairs = np.column_stack((X.ravel(), Y.ravel()))

        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        self.total_tests = len(valid_vars) * len(valid_targets) - len(set(valid_vars) & set(valid_targets))
            

        all_stats = []
        # Process pairs in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            pair_vars = vars.loc[:,batch_pairs[:, 0]].to_numpy()
            pair_targets = targets.loc[:,batch_pairs[:, 1]].to_numpy()
            
            # Compute statistics for the current batch
            batch_stats = self.pair_statistic(pair_vars, pair_targets)
            all_stats.append(batch_stats)

        # Concatenate all batch statistics
        stats = np.concatenate(all_stats, axis=0)

        return pairs, stats

    def _get_pair_data2(self, obsdf: pd.DataFrame, pairs: list[tuple[str, str]], prevalence_threshold: float = 0.00, batch_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a list of trait pairs (as column names) and a NumPy array of their test statistic results 
        for all traits with prevalence above the given threshold.

        :param obsdf: DataFrame containing the trait data
        :param pairs: List of tuples containing the column names to compute pair statistics
        :param prevalence_threshold: Minimum prevalence for traits to be included [0,1]
        :param batch_size: Number of pairs processed in a batch
        :returns: NumPy array of valid pairs (column names), NumPy array of pair statistics
        """
        # Convert to NumPy arrays for efficient computation
        obs_np = obsdf.to_numpy()
        obs_cols = np.array(obsdf.columns)

        # Map column names to indices for fast access
        col_idx = {col: i for i, col in enumerate(obs_cols)}

        # Convert pairs from column names to indices
        pair_indices = np.array([(col_idx[i], col_idx[j]) for i, j in pairs])

        # Get valid trait masks based on prevalence
        prevalence_counts = (obs_np != 0).sum(axis=0)
        valid_mask = prevalence_counts >= (prevalence_threshold * obs_np.shape[0])

        # Filter valid pairs based on prevalence
        valid_pairs_mask = valid_mask[pair_indices[:, 0]] & valid_mask[pair_indices[:, 1]]
        valid_pairs_indices = pair_indices[valid_pairs_mask]

        if valid_pairs_indices.size == 0:
            return np.array([]), np.array([])

        # Convert valid pairs back to column names
        valid_pairs = np.array([(obs_cols[i], obs_cols[j]) for i, j in valid_pairs_indices])

        # Process valid pairs in batches
        all_stats = []
        for i in range(0, len(valid_pairs_indices), batch_size):
            batch_indices = valid_pairs_indices[i:i + batch_size]
            pair_vars = obs_np[:, batch_indices[:, 0]]
            pair_targets = obs_np[:, batch_indices[:, 1]]

            batch_stats = self.pair_statistic(pair_vars, pair_targets)
            all_stats.append(batch_stats)

        # Concatenate all batch statistics
        stats = np.concatenate(all_stats, axis=0)

        valid_pairs = np.array([(obs_cols[i], obs_cols[j]) for i, j in valid_pairs_indices])

        return valid_pairs, stats

    def run_simulation(self, cores = -1):
        """
        Runs the tree simulation and stores results.
        """
        self._simulate_and_evaluate(cores)
    
    def _simulate_and_evaluate(self,alpha = 0.05,cores=-1):
        traits_to_simulate = np.unique(self.pairs.flatten())
        traits_to_simulate_pastml = self.pastml.loc[traits_to_simulate]
        self.simulation_result = simulate_glrates_bit(tree = self.tree, trait_params = traits_to_simulate_pastml, pairs = self.pairs, obspairs=self.obspairs, cores = cores)
        self.simulation_result['T1'] = self.simulation_result['first']
        self.simulation_result['T2'] = self.simulation_result['second']
        self._multiple_test_correction(alpha)
    
    def _multiple_test_correction(self,alpha = 0.05):
        # get uncorrected results
        res = self.simulation_result
        prev = self.obsdf.mean()
        res['prevalence_T1'] = res['first'].map(prev)
        res['prevalence_T2'] = res['second'].map(prev)
        res['effect size'] = abs(res['effect size'])

        # ---- corrections ----
        pvals = res['p-value'].values 
        simulated_pairs = len(pvals)
        filtered_pairs = self.total_tests - len(pvals)
        pvals = np.concatenate((np.array(pvals),np.full(filtered_pairs,0.5)))

        # Benjamini-Hochberg
        res['pval_bh'] = sm.multipletests(pvals, alpha=alpha, method='fdr_bh')[1][:simulated_pairs]

        # Benjamini-Yekutieli
        res['pval_by'] = sm.multipletests(pvals, alpha=alpha, method='fdr_by')[1][:simulated_pairs]

        # Bonferroni
        res['pval_bonf'] = sm.multipletests(pvals, alpha=alpha, method='bonferroni')[1][:simulated_pairs]

        # Naive (just keep the original p-value)
        res['pval_naive'] = res.loc[:,'p-value'][:simulated_pairs]

        self.result = res[['T1','T2','direction','effect size',
                'prevalence_T1','prevalence_T2',
                'pval_naive','pval_bh','pval_by','pval_bonf']]
        

    def get_results(self):
        return self.get_top_results(top = len(self.result['T1']))
    
    def get_top_results(self, top = 15, direction: Literal[-1,0,1]=0, by: Literal['p-value','effect size']='effect size'):
        res = self.result
        if direction != 0:
            res = res[res['direction'] == direction]
        return res.sort_values(by=(by if (by != 'p-value') else 'pval_naive'), ascending=(by == 'p-value')).head(top)
    


    def plot_results(self, pval_col, prevalence_range=[0,1], figure_size=-1, output_file="fig.svg"):

        res = self.result

        # Apply prevalence filter
        if prevalence_range is not None:
            lo, hi = prevalence_range
            res = res[
                (res["prevalence_T1"].between(lo, hi)) &
                (res["prevalence_T2"].between(lo, hi))
            ]

        # Separate positive and negative results
        res_syn = res[res['direction'] == 1]
        res_ant = res[res['direction'] == -1]

        # Create pivot tables
        simpiv_ant = pd.pivot_table(res_ant, values=pval_col,
                                    index='T2', columns='T1',
                                    aggfunc='min', sort=False)
        simpiv_syn = pd.pivot_table(res_syn, values=pval_col,
                                    index='T1', columns='T2',
                                    aggfunc='min', sort=False)

        # Ensure same ordering of rows/columns
        all_indices = sorted(set(simpiv_ant.index)
                            .union(set(simpiv_syn.index))
                            .union(set(simpiv_ant.columns))
                            .union(set(simpiv_syn.columns)))
        simpiv_ant = simpiv_ant.reindex(index=all_indices, columns=all_indices)
        simpiv_syn = simpiv_syn.reindex(index=all_indices, columns=all_indices)

        simpiv_ant = simpiv_ant.fillna(.5)
        simpiv_syn = simpiv_syn.fillna(.5)

        combined_matrix = pd.DataFrame(.5, index=all_indices, columns=all_indices)

        # Fill lower triangle with antagonistic p-values
        for i in range(len(all_indices)):
            for j in range(i + 1, len(all_indices)):
                combined_matrix.iat[j, i] = simpiv_ant.iat[j, i]

        # Fill upper triangle with synergistic p-values
        for i in range(len(all_indices)):
            for j in range(i + 1, len(all_indices)):
                combined_matrix.iat[i, j] = simpiv_syn.iat[i, j]

        # Diagonal = 1
        np.fill_diagonal(combined_matrix.values, .5)

        # --- Plot ---
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

            sns.heatmap(combined_matrix, mask=mask, cmap=cmap1, 
                        cbar=True, cbar_kws={"shrink": 0.5, 'fraction': .05, 'pad': 0, 'label':'p-value'}, 
                        ax=ax, square=True, vmin = 0, vmax = 0.5)
            sns.heatmap(combined_matrix, mask=~mask, cmap=cmap2, 
                        cbar=True, cbar_kws={"shrink": 0.5, 'ticks': [], 'fraction': .05}, 
                        ax=ax, square=True, vmin = 0, vmax = 0.5)

            # Grey diagonal
            diagonal_mask = np.zeros_like(combined_matrix, dtype=bool)
            np.fill_diagonal(diagonal_mask, True)
            sns.heatmap(combined_matrix, mask=~diagonal_mask, cmap=grey_cmap, cbar=False, ax=ax, square=True)

            plt.xlabel("")
            plt.ylabel("")

            # Add significance stars
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
                        plt.text(j + 0.5, i + 0.5, annotation, 
                                ha='center', va='center', fontsize=10, color='black')

        plt.title("Positive (Red) and Negative (Blue) Associations")
        plt.tight_layout()
        plt.savefig(output_file, format='svg')
        plt.show()
    
    def plot_effect_size(self, pval_col, prevalence_range = [0,1], output_file = 'fig.svg'):
        
        x = self.result
        if prevalence_range is not None:
            lo, hi = prevalence_range
            x = x[
                (x["prevalence_T1"].between(lo, hi)) &
                (x["prevalence_T2"].between(lo, hi))
            ]

        ef = abs(x['effect size']) * x['direction']
        pv = -np.log(x[pval_col])
        plt.scatter(x = ef, y = pv)
        plt.tight_layout()
        plt.savefig(output_file, format='svg')
        plt.show()


    


