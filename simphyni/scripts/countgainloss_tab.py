import os
import numpy as np
from numpy import matlib
from scipy import stats
import math
from ete3 import Tree, ProfileFace
import sys
import pandas as pd

def countgainloss(treepath, gene):
    if not os.path.exists(treepath): return (np.float('NaN'),np.float('NaN'),np.float('NaN'),np.float('NaN'))
    tree = treepath + '/' + [i for i in os.listdir(treepath) if '.nwk' in i][-1]
    with open(tree) as f:
        newick_str = f.read()
    t = Tree(newick_str, format=1)

    # t = Tree(tree, format = 1)
    annot = treepath + '/combined_ancestral_states.tab'
    a = pd.read_csv(annot, sep = '\t', index_col = 0)
    dist = float("inf")
    loss_dist = float('inf')
    if '-' in gene: gene = ''.join(gene.split('-'))
    a[gene] = a[gene].astype(int)



    def iter(tree_root):
        nonlocal dist
        nonlocal loss_dist
        stack = [(tree_root, False)]  # stack stores tuples of (node, processed flag)
        node_data = {}  # to store gains and losses for each node

        while stack:
            node, processed = stack.pop()

            if not processed:
                # Push the current node back onto the stack to process after children
                stack.append((node, True))
                # Push the children onto the stack
                for child in node.children:
                    stack.append((child, False))
            else:
                # Process the node
                if node.is_leaf():
                    if a[gene].loc[node.name] == 1:
                        ndist = node.get_distance(t)
                        dist = min(dist, ndist)
                    elif node.up and a[gene].loc[node.up.name] == 1:
                        ndist = node.get_distance(t)
                        loss_dist = min(loss_dist, ndist)

                    node_data[node] = (0, 0)
                else:
                    num_gains = 0
                    num_losses = 0

                    child_syst = []
                    for child in node.children:
                        child_gains, child_losses = node_data[child]
                        num_gains += child_gains
                        num_losses += child_losses
                        child_syst.append(a[gene].loc[child.name])

                    if '|' in getattr(node, gene):
                        num_gains += (0.5 if 1 in child_syst else 0)
                        num_losses += (0.5 if 0 in child_syst else 0)
                    else:
                        num_gains += (1 if 1 in child_syst and a[gene].loc[node.name] == 0 else 0)
                        num_losses += (1 if 0 in child_syst and a[gene].loc[node.name] == 1 else 0)

                    if a[gene].loc[node.name] == 1:
                        ndist = node.get_distance(t)
                        dist = min(dist, ndist)
                    elif node.up and a[gene].loc[node.up.name] == 1:
                        ndist = node.get_distance(t)
                        loss_  = min(loss_dist, ndist)

                    node_data[node] = (num_gains, num_losses)

        gains, losses = node_data[tree_root]
        return gains, losses

    # Execute the iterative function starting from the root
    gains, losses = iter(t.get_tree_root())

    branch_lengths = np.array([n.dist for n in t.traverse() if not n.is_root() and n.dist > 0])
    log_bl = np.log10(branch_lengths)
    # Compute IQR in log₁₀ space
    Q1 = np.percentile(log_bl, 25)
    Q3 = np.percentile(log_bl, 75)
    IQR = Q3 - Q1

    # Define log₁₀-space bounds
    log_lower_bound = Q1 - .5 * IQR
    log_upper_bound = Q3 + .5 * IQR

    # Convert bounds back to real space
    lower_bound = 10 ** log_lower_bound
    upper_bound = 10 ** log_upper_bound

    gain_subsize = 0
    for n in t.traverse():
        if a[gene].loc[n.name] == 0 and n.dist < upper_bound: gain_subsize += n.dist

    loss_subsize = 0
    for n in t.traverse():
        if a[gene].loc[n.name] == 1 and n.dist < upper_bound: loss_subsize += n.dist

    root = a[gene].loc[t.get_tree_root().name]
    # Return the number of gains, losses, and the minimum distance where a gain and loss occur
    return gains, losses, dist, loss_dist, gain_subsize, loss_subsize, root

