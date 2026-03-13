"""
Ancestral State Reconstruction using PastML — Joint Model
==========================================================
Reconstructs ancestral states for binary (and categorical) traits on a
phylogenetic tree using the Joint maximum-likelihood method implemented in
PastML.  Reports:

  • Most-likely ancestral state at every internal node (JOINT method)
  • Marginal-probability certainty scores per node (MPPA)
  • ML-estimated equilibrium frequencies and scaled rate matrix (Q)
  • Full summary table (CSV) of node predictions + certainties

Model background (F81 / CTMC)
------------------------------
PastML implements continuous-time Markov models on a phylogeny.  For the
F81 model with n states, the instantaneous rate matrix Q is:

    Q[i,j] = sf * pi_j         (i != j)
    Q[i,i] = -sf * (1 - pi_i)

where pi_j are the equilibrium (stationary) frequencies and sf is a
branch-length scaling factor (overall rate).  The probability of state j
given starting state i over branch length t is:

    P(j|i,t) = pi_j + (delta_ij - pi_j) * exp(-sf * t)

For binary traits (states 0/1):
    q(0->1) = sf * pi_1   [rate of gaining state 1]
    q(1->0) = sf * pi_0   [rate of losing  state 1]

References
----------
Ishikawa et al. (2019) A fast likelihood method to reconstruct and visualize
ancestral scenarios. Mol. Biol. Evol. 36(9): 2069-2082.
https://doi.org/10.1093/molbev/msz131

Usage
-----
    python ancestral_reconstruction.py                  # demo (simulated data)

    python ancestral_reconstruction.py \
        --tree my_tree.nwk \
        --annotations traits.csv \
        --column trait_name \
        --model F81 \
        --output results.csv
"""

import argparse
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import expm
from ete3 import Tree

warnings.filterwarnings("ignore")

from pastml.acr import acr
from pastml.ml import JOINT, MPPA


# =============================================================================
#  HELPER: Name all internal nodes (required for merging JOINT + MPPA results)
# =============================================================================

def label_internal_nodes(tree):
    """Ensure every internal node has a unique name. Returns tree (in-place)."""
    counter = 0
    for node in tree.traverse():
        if not node.is_leaf() and not node.name:
            node.name = f"N{counter}"
            counter += 1
    return tree


# =============================================================================
#  SIMULATION HELPERS
# =============================================================================

def simulate_tree(n_tips=25, branch_scale=0.15, seed=42):
    """
    Generate a random bifurcating tree with n_tips leaves.
    Branch lengths ~ Exponential(branch_scale).
    """
    rng = np.random.default_rng(seed)

    t = Tree()
    t.populate(n_tips, random_branches=True)

    # Replace ete3's U[0,1] branches with Exp(branch_scale)
    for node in t.traverse():
        node.dist = float(rng.exponential(branch_scale)) + 1e-4  # no zero-len

    for i, leaf in enumerate(t.get_leaves()):
        leaf.name = f"taxon_{i:03d}"

    return label_internal_nodes(t)


def simulate_binary_trait(tree, q01=1.5, q10=1.0, root_state=None, seed=99):
    """
    Simulate a binary trait (states '0'/'1') along tree using a 2-state CTMC
    with rates q01 (0->1) and q10 (1->0).

    Returns a DataFrame indexed by tip name with column 'trait'.
    """
    rng = np.random.default_rng(seed)

    Q = np.array([[-q01, q01],
                  [ q10, -q10]], dtype=float)

    # Equilibrium frequencies
    pi1 = q01 / (q01 + q10)
    if root_state is None:
        root_state = int(rng.random() < pi1)

    node_states = {id(tree): root_state}

    for node in tree.traverse():
        if node.is_root():
            continue
        parent_state = node_states[id(node.up)]
        P = np.clip(expm(Q * node.dist), 0.0, 1.0)
        P /= P.sum(axis=1, keepdims=True)            # re-normalise numerics
        node_states[id(node)] = int(rng.choice([0, 1], p=P[parent_state]))

    tip_states = {
        leaf.name: str(node_states[id(leaf)])
        for leaf in tree.get_leaves()
        if id(leaf) in node_states
    }
    return pd.DataFrame({"trait": tip_states})


# =============================================================================
#  RATE-MATRIX COMPUTATION (from F81 model parameters)
# =============================================================================

def compute_rate_matrix(model, states):
    """
    Reconstruct the instantaneous rate matrix Q from PastML's F81 model.

    F81:  Q[i,j] = sf * pi_j   (i != j)
          Q[i,i] = -sf * (1 - pi_i)
    """
    freqs = model.frequencies          # equilibrium frequencies (pi)
    sf    = model.sf                   # scaling factor
    n     = len(states)

    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = sf * freqs[j]
        Q[i, i] = -sum(Q[i, j] for j in range(n) if j != i)

    return Q


# =============================================================================
#  CERTAINTY TABLE BUILDER
# =============================================================================

def build_certainty_table(tree, marginal_probs_df, joint_states, states, column):
    """
    Merge JOINT states and MPPA marginal probabilities into a per-node table.

    Columns
    -------
    node         : node name
    joint_state  : ML ancestral state (JOINT reconstruction)
    certainty    : marginal probability of the JOINT state (from MPPA)
    prob_<state> : marginal probability for each possible state
    depth        : number of nodes from root (root = 0)
    n_children   : number of direct children
    """
    rows = []
    for node in tree.traverse():
        if node.is_leaf():
            continue

        name    = node.name
        j_state = joint_states.get(name)

        # Pull marginal probabilities row for this node
        if name in marginal_probs_df.index:
            probs = marginal_probs_df.loc[name]
        else:
            probs = pd.Series({s: np.nan for s in states})

        prob_dict = {f"prob_{s}": float(probs.get(s, np.nan)) for s in states}

        # Certainty = marginal probability of the joint-predicted state
        certainty = prob_dict.get(f"prob_{j_state}", np.nan)

        rows.append({
            "node":        name,
            "joint_state": j_state,
            "certainty":   certainty,
            **prob_dict,
            "depth":       int(node.get_distance(tree, topology_only=True)),
            "n_children":  len(node.children),
        })

    return pd.DataFrame(rows).set_index("node")


# =============================================================================
#  PRINT HELPERS
# =============================================================================

def banner(text):
    sep = "=" * 68
    print(f"\n{sep}\n  {text}\n{sep}")


def print_model_params(model, states, log_lh, Q):
    banner("Estimated Model Parameters (F81 CTMC)")
    print(f"  Log-likelihood       : {log_lh:.4f}")
    print(f"  Scaling factor (sf)  : {model.sf:.6f}")
    print(f"  Effective rate (mu)  : {model.get_mu():.6f}  [changes per branch-length unit]")

    print("\n  Equilibrium frequencies (pi):")
    for s, f in zip(states, model.frequencies):
        print(f"    pi({s}) = {f:.4f}")

    print("\n  Rate matrix Q (instantaneous rates):")
    header = "           " + "  ".join(f"{s:>10}" for s in states)
    print(header)
    for i, si in enumerate(states):
        row = "  ".join(f"{Q[i, j]:>10.4f}" for j in range(len(states)))
        print(f"  -> {si:>4}  {row}")

    if len(states) == 2:
        print("\n  Binary-trait interpretation:")
        print(f"    q({states[0]}->{states[1]}) = {Q[0, 1]:.4f}   [rate of gaining '{states[1]}']")
        print(f"    q({states[1]}->{states[0]}) = {Q[1, 0]:.4f}   [rate of losing  '{states[1]}']")
        ratio = Q[0, 1] / Q[1, 0] if Q[1, 0] > 0 else float("inf")
        dom   = states[1] if Q[0, 1] > Q[1, 0] else states[0]
        print(f"    Gain / loss ratio  = {ratio:.4f}   (net bias toward '{dom}')")


def print_certainty_summary(df, n_show=15):
    banner(f"Ancestral State Certainties -- {len(df)} internal nodes")

    # Show most uncertain nodes first (most interesting for reporting)
    display = df[["joint_state", "certainty"]].sort_values("certainty").head(n_show)
    with pd.option_context(
        "display.float_format", "{:.3f}".format,
        "display.max_columns", 10,
        "display.width", 120,
    ):
        print(display.to_string())

    c = df["certainty"].dropna()
    print(f"\n  Summary statistics (n = {len(c)} internal nodes):")
    print(f"    Mean certainty    : {c.mean():.3f}")
    print(f"    Median certainty  : {c.median():.3f}")
    print(f"    Min  certainty    : {c.min():.3f}")
    print(f"    Certainty < 0.70  : {(c < 0.70).sum()} nodes")
    print(f"    Certainty >= 0.90 : {(c >= 0.90).sum()} nodes")
    print(f"    Certainty >= 0.95 : {(c >= 0.95).sum()} nodes")


# =============================================================================
#  CORE PIPELINE
# =============================================================================

def run_reconstruction(tree, annotations, column, model="F81", output_csv=None):
    """
    Full ancestral reconstruction pipeline:

    1. JOINT ML reconstruction  -> best ancestral state per node
    2. MPPA marginal            -> per-node probability distributions
    3. Derive rate matrix Q from optimised model parameters
    4. Build certainty table (JOINT state + marginal probability of that state)
    5. Optionally save CSV

    Parameters
    ----------
    tree        : ete3.Tree with branch lengths; internal nodes should be named
    annotations : DataFrame indexed by tip name; 'column' is the trait column
    column      : trait column name
    model       : substitution model ('F81', 'JC', 'EFT')
    output_csv  : path to save node-certainty CSV (None = skip)

    Returns
    -------
    certainty_df : DataFrame with per-node certainties
    info         : dict with 'log_likelihood', 'Q', 'model', 'states'
    """
    # Coerce trait to string
    ann = annotations[[column]].copy()
    ann[column] = ann[column].astype(str)
    states = sorted(ann[column].dropna().unique().tolist())

    banner(f"Ancestral Reconstruction  |  trait='{column}'  |  model={model}")
    print(f"  Tips         : {len(ann)}")
    print(f"  States       : {states}")
    print(f"  State counts :")
    for s, cnt in ann[column].value_counts().items():
        pct = 100 * cnt / len(ann)
        print(f"    {s:>8}  ->  {cnt:>4} tips  ({pct:.1f}%)")

    if len(states) < 2:
        raise ValueError(
            f"Need >=2 states for reconstruction; got: {states}. "
            "Check your annotations."
        )

    # ------------------------------------------------------------------
    # 1. JOINT reconstruction
    # ------------------------------------------------------------------
    print("\n  [1/2] Running JOINT ML reconstruction ...")
    tree_joint = tree.copy()
    label_internal_nodes(tree_joint)

    joint_results = acr(
        tree_joint,
        df=ann,
        prediction_method=JOINT,
        model=model,
    )
    res_j = joint_results[0]

    # Collect JOINT state per internal node (stored as set -> take first element)
    joint_states = {}
    for node in tree_joint.traverse():
        if not node.is_leaf():
            val = getattr(node, column, None)
            if isinstance(val, (set, frozenset, list)):
                val = next(iter(val)) if val else None
            joint_states[node.name] = str(val) if val is not None else "?"

    # ------------------------------------------------------------------
    # 2. MPPA reconstruction (marginal probabilities)
    # ------------------------------------------------------------------
    print("  [2/2] Running MPPA marginal reconstruction for certainty ...")
    tree_mppa = tree.copy()
    label_internal_nodes(tree_mppa)

    mppa_results = acr(
        tree_mppa,
        df=ann,
        prediction_method=MPPA,
        model=model,
    )
    res_m = mppa_results[0]

    # marginal_probabilities is a DataFrame with state names as columns
    mp_df = res_m["marginal_probabilities"]
    mp_df.columns = [str(c) for c in mp_df.columns]   # ensure string columns

    # ------------------------------------------------------------------
    # 3. Model parameters & rate matrix
    # ------------------------------------------------------------------
    model_obj = res_j["model"]
    log_lh    = res_j["log_likelihood"]
    Q         = compute_rate_matrix(model_obj, states)
    print_model_params(model_obj, states, log_lh, Q)

    # ------------------------------------------------------------------
    # 4. Build certainty table
    # ------------------------------------------------------------------
    cert_df = build_certainty_table(tree_joint, mp_df, joint_states, states, column)
    print_certainty_summary(cert_df)

    # ------------------------------------------------------------------
    # 5. Save CSV
    # ------------------------------------------------------------------
    if output_csv:
        cert_df.reset_index().to_csv(output_csv, index=False)
        print(f"\n  [OK] Results saved -> {output_csv}")

    info = {
        "log_likelihood": log_lh,
        "Q":              Q,
        "model":          model_obj,
        "states":         states,
        "joint_states":   joint_states,
        "marginal_probs": mp_df,
    }
    return cert_df, info


# =============================================================================
#  DEMO
# =============================================================================

def demo():
    print(textwrap.dedent("""
    +------------------------------------------------------------------+
    |  PastML -- Joint Ancestral State Reconstruction  (built-in demo) |
    |  Simulated binary trait:  q(0->1) = 1.5,  q(1->0) = 1.0        |
    |  25-tip random tree,  F81 substitution model                     |
    +------------------------------------------------------------------+
    """))

    tree = simulate_tree(n_tips=25, branch_scale=0.15, seed=7)
    df   = simulate_binary_trait(tree, q01=1.5, q10=1.0, seed=42)

    print("  Simulated tip-state distribution:")
    print(df["trait"].value_counts().to_string(header=False))

    cert_df, info = run_reconstruction(
        tree=tree,
        annotations=df,
        column="trait",
        model="F81",
        output_csv="ancestral_states_demo.csv",
    )

    # ASCII tree with JOINT annotations
    banner("Annotated Tree  (JOINT state | certainty at each internal node)")
    tree_display = tree.copy()
    label_internal_nodes(tree_display)

    tip_states = df["trait"].to_dict()
    js         = info["joint_states"]

    for node in tree_display.traverse("postorder"):
        if node.is_leaf():
            s = tip_states.get(node.name, "?")
            node.name = f"{node.name}[{s}]"
        else:
            s = js.get(node.name, "?")
            c = cert_df.loc[node.name, "certainty"] if node.name in cert_df.index else float("nan")
            node.name = f"[{s}|p={c:.2f}]"

    print(tree_display.get_ascii(show_internal=True))

    banner("Done")
    print("  Output file : ancestral_states_demo.csv")
    print("  Columns     : node | joint_state | certainty | prob_0 | prob_1 | depth | n_children")
    print()


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Ancestral State Reconstruction with PastML (Joint + MPPA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
        # Built-in demo with simulated data:
        python ancestral_reconstruction.py

        # Real binary trait:
        python ancestral_reconstruction.py \\
            --tree mammals.nwk \\
            --annotations traits.csv \\
            --column hibernation \\
            --model F81 \\
            --output results.csv

        # Categorical trait (states auto-detected from CSV):
        python ancestral_reconstruction.py \\
            --tree bacteria.nwk \\
            --annotations metadata.csv \\
            --column lifestyle

        Notes
        -----
        * Annotation CSV: tip names in the first (index) column.
        * Branch lengths required in the Newick file.
        * Supported models: F81 (default), JC, EFT
        """),
    )
    p.add_argument("--tree",        metavar="FILE", help="Newick tree file")
    p.add_argument("--annotations", metavar="FILE", help="CSV tip-annotation file")
    p.add_argument("--column",      metavar="COL",  help="Column in annotations CSV")
    p.add_argument("--model",       default="F81",  choices=["F81", "JC", "EFT"],
                                    help="Substitution model (default: F81)")
    p.add_argument("--output",      metavar="FILE", default=None,
                                    help="Output CSV (default: <column>_ancestral_states.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.tree and not args.annotations:
        demo()
        return

    if not args.tree or not args.annotations or not args.column:
        print("ERROR: --tree, --annotations, and --column are all required.")
        print("       Run without arguments for a self-contained demo.")
        sys.exit(1)

    tree = Tree(args.tree, format=1)
    label_internal_nodes(tree)

    df = pd.read_csv(args.annotations, index_col=0)
    if args.column not in df.columns:
        print(f"ERROR: Column '{args.column}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    out = args.output or f"{args.column}_ancestral_states.csv"
    run_reconstruction(tree=tree, annotations=df, column=args.column,
                       model=args.model, output_csv=out)


if __name__ == "__main__":
    main()