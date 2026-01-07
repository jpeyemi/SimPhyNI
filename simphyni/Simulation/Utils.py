import os
import numpy as np
import pandas as pd
from scipy import stats
from ete3 import Tree, TreeNode
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt
import sys

#%%
def tree_plot(tree: TreeNode,data: pd.DataFrame, title: str = "Tree Plot", show_tree: bool = True):
    """
    creates a heatmap visualization with a dedrogram of the given tree
    """
    if not show_tree:
        plt.figure(figsize = (10,25))
        sns.heatmap(data)
        return
    
    ordered_data = data.loc[[leaf.name for leaf in tree.iter_leaves()]]

    # Generating the distance matrix from the tree
    distance_matrix = np.zeros((len(tree), len(tree)))
    leaf_names = [leaf.name for leaf in tree.iter_leaves()]
    for i, leaf1 in enumerate(tree.iter_leaves()):
        for j, leaf2 in enumerate(tree.iter_leaves()):
            if i < j:
                distance = tree.get_distance(leaf1, leaf2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    # Convert the distance matrix into a format suitable for linkage
    condensed_matrix = squareform(distance_matrix)
    row_linkage = linkage(condensed_matrix, method='average')


    num_categories = len(data.columns)
    fig_width = num_categories + 1 if num_categories < 5 else 6 + num_categories / 5
    fig_height = max(fig_width,len(tree) / 25)
    g = sns.clustermap(ordered_data, row_linkage=row_linkage, col_cluster=False, yticklabels = [], figsize=(fig_width, fig_height), dendrogram_ratio=(2/fig_width if fig_width > 2 else 1/fig_width, .2), cmap = 'mako',cbar_kws=dict(orientation='horizontal'))
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.02]) # type: ignore
    plt.title(title)
# %%
