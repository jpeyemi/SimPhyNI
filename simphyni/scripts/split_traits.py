#!/usr/bin/env python
"""
split_traits.py
===============
Split a wide trait matrix (rows=taxa, cols=traits) into column-shards of a
fixed size.  Used by the Snakemake sharded ACR pipeline (acr_shard_size > 0).

Input is a Parquet file (written by reformat_csv.py).  Each shard is written
as a CSV so that the downstream run_ancestral_reconstruction.py shard jobs
receive files in the same format they have always expected.

Each shard preserves the original index column so that
run_ancestral_reconstruction.py can load it with pd.read_csv(..., index_col=0).

Memory profile: only one shard's worth of columns is in RAM at a time.
Peak RAM ≈ shard_size × n_rows × bytes_per_value.

Usage
-----
python split_traits.py \\
    --inputs_file reformatted.parquet \\
    --output_dir shards/ \\
    --shard_size 20000
"""

import argparse
import math
import sys
from pathlib import Path

# Ensure the scripts directory is on sys.path so traits_io can be imported
# both when run as a script (Snakemake) and when imported as a package module.
sys.path.insert(0, str(Path(__file__).parent))
from traits_io import get_trait_metadata, load_trait_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_file", required=True,
                        help="Reformatted trait Parquet file (rows=taxa, cols=traits)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write shard CSVs into")
    parser.add_argument("--shard_size", type=int, default=20_000,
                        help="Number of trait columns per shard")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_col, trait_cols = get_trait_metadata(args.inputs_file)
    n_traits = len(trait_cols)
    n_shards = math.ceil(n_traits / args.shard_size)
    digits = len(str(n_shards - 1))

    for i in range(n_shards):
        shard_cols = trait_cols[i * args.shard_size : (i + 1) * args.shard_size]
        shard_df = load_trait_columns(args.inputs_file, shard_cols, index_col)
        shard_path = out_dir / f"shard_{i:0{digits}d}.csv"
        shard_df.to_csv(shard_path)

    print(f"[split_traits] {n_traits} traits -> {n_shards} shards "
          f"of up to {args.shard_size} in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
