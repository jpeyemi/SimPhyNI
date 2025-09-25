import os
import sys
from countgainloss_tab import countgainloss
import pandas as pd
from ete3 import Tree
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parse arguments
inpdir = sys.argv[-4]
tree = sys.argv[-3]
pastml_dir = sys.argv[-2]
outannot = sys.argv[-1]

# Load data and filter leaves
data = pd.read_csv(inpdir, index_col=0)
t = Tree(tree, 1)
leaves = [i for i in t.get_leaf_names() if i in data.index]
data = data.loc[leaves]

# Prepare dictionary to collect results
to_df = {
    'gene': [], 'gains': [], 'losses': [], 'count': [],
    'dist': [], 'loss_dist': [], 'gain_subsize': [], 'loss_subsize': [], 'root_state': []
}

# Function to process each gene
def process_gene(c):
    dir = os.path.join(pastml_dir, c)
    if not os.path.exists(dir):
        print(f'{c} has no reconstruction')
        return None  # Return None if the directory does not exist
    gains, losses, dist, loss_dist, gain_subsize, loss_subsize, root = countgainloss(dir, c)
    return {
        'gene': c, 'gains': gains, 'losses': losses, 'count': sum(data[c]),
        'dist': dist, 'loss_dist': loss_dist, 'gain_subsize': gain_subsize,
        'loss_subsize': loss_subsize, 'root_state': int(root)
    }


with ThreadPoolExecutor() as executor:
    future_to_gene = {executor.submit(process_gene, c): c for c in data.columns}
    for future in as_completed(future_to_gene):
        c = future_to_gene[future]
        try:
            result = future.result()
            if result:  # Skip None results
                for key in to_df:
                    to_df[key].append(result[key])
        except Exception as e:
            print(f"Error processing gene {c}: {e}")

# Save the aggregated results to a CSV file
df = pd.DataFrame.from_dict(to_df)
df = df.set_index('gene')  # Set 'gene' as the index for alignment
df = df.loc[data.columns]  # Reindex based on data.columns
df = df.reset_index()
df.rename(columns={'index': 'gene'}, inplace=True)
os.makedirs(os.path.dirname(outannot), exist_ok=True)
df.to_csv(outannot)
