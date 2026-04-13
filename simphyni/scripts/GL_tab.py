import os
import sys
from pathlib import Path
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed
from countgainloss_tab import countgainloss

# Ensure the scripts directory is on sys.path so traits_io can be imported
# both when run as a script (Snakemake) and when imported as a package module.
sys.path.insert(0, str(Path(__file__).parent))
from traits_io import compute_gene_sums, get_trait_metadata

def process_gene(gene, pastml_dir, gene_count):
    gene_dir = os.path.join(pastml_dir, gene)
    if not os.path.exists(gene_dir):
        print(f'{gene} has no reconstruction')
        return None  # Return None if directory does not exist
    gains, losses, dist, loss_dist, gain_subsize, loss_subsize, root = countgainloss(gene_dir, gene)
    return {
        'gene': gene,
        'gains': gains,
        'losses': losses,
        'count': int(gene_count),
        'dist': dist,
        'loss_dist': loss_dist,
        'gain_subsize': gain_subsize,
        'loss_subsize': loss_subsize,
        'root_state': int(root),
    }

if __name__ == "__main__":
    # Parse arguments
    inpdir = sys.argv[-4]
    tree_file = sys.argv[-3]
    pastml_dir = sys.argv[-2]
    outannot = sys.argv[-1]

    # Determine which genes have PastML reconstructions (metadata only, no data load).
    t = Tree(tree_file, 1)
    leaves_in_tree: set[str] = set(t.get_leaf_names())

    index_col, all_traits = get_trait_metadata(inpdir)
    available_genes = [
        g for g in all_traits if os.path.exists(os.path.join(pastml_dir, g))
    ]

    # Stream per-column sums without loading the full matrix.
    gene_sums = compute_gene_sums(inpdir, index_col, leaves_in_tree)

    # Run in parallel using joblib, passing only gene name and precomputed count
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_gene)(gene, pastml_dir, gene_sums[gene]) for gene in available_genes
    )

    # Aggregate results
    to_df = {k: [] for k in [
        'gene', 'gains', 'losses', 'count', 'dist',
        'loss_dist', 'gain_subsize', 'loss_subsize', 'root_state'
    ]}

    for res in results:
        if res:
            for k in to_df:
                to_df[k].append(res[k])

    # Save aggregated results
    df = pd.DataFrame.from_dict(to_df)
    if not df.empty:
        df = df.set_index('gene')
        df = df.loc[available_genes]
        df = df.reset_index()
        df.rename(columns={'index': 'gene'}, inplace=True)

        os.makedirs(os.path.dirname(outannot), exist_ok=True)
        df.to_csv(outannot)
    else:
        print("No valid reconstructions found. Output file will not be created.")
