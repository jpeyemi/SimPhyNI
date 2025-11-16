import os
import sys
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed
from countgainloss_tab import countgainloss

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

    # Load data and filter leaves
    data = pd.read_csv(inpdir, index_col=0)
    t = Tree(tree_file, 1)
    leaves = [i for i in t.get_leaf_names() if i in data.index]
    data = data.loc[leaves]

    # Only process genes that have PastML reconstructions
    available_genes = [g for g in data.columns if os.path.exists(os.path.join(pastml_dir, g))]

    # Precompute the sum for each gene to reduce memory
    gene_sums = {g: data[g].sum() for g in available_genes}

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
