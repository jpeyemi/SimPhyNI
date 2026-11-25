"""Exact F81 path simulation (independent of SimPhyNI), replicated, root fixed."""
import numpy as np


def exact_tips(tree, pi, lam, root, n_rep, rng):
    """(n_tips, n_traits, n_rep) uint8 tip states; gain rate lam*pi, loss lam*(1-pi)."""
    nodes = list(tree.traverse("preorder"))
    idx = {n: i for i, n in enumerate(nodes)}
    T = len(pi)
    pi = np.repeat(np.asarray(pi, float)[:, None], n_rep, 1)
    lam = np.repeat(np.asarray(lam, float)[:, None], n_rep, 1)
    st = np.zeros((len(nodes), T, n_rep), np.uint8)
    st[0] = np.repeat(np.asarray(root, np.uint8)[:, None], n_rep, 1)
    for n in nodes[1:]:
        e = np.exp(-lam * n.dist)
        p1 = pi + (st[idx[n.up]] - pi) * e           # exact transition probability
        st[idx[n]] = rng.random(p1.shape) < p1
    return st[[idx[l] for l in tree]]
