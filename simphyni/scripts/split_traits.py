#!/usr/bin/env python
"""
split_traits.py
===============
Split a wide trait CSV (rows=taxa, cols=traits) into column-shards of a fixed
size.  Used by the Snakemake sharded ACR pipeline (acr_shard_size > 0).

Each shard keeps the original index column so that
run_ancestral_reconstruction.py can load it with pd.read_csv(..., index_col=0).

Usage
-----
python split_traits.py \
    --inputs_file reformatted.csv \
    --output_dir shards/ \
    --shard_size 20000
"""

import argparse
import math
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_file", required=True,
                        help="Reformatted trait CSV (rows=taxa, cols=traits)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write shard CSVs into")
    parser.add_argument("--shard_size", type=int, default=20_000,
                        help="Number of trait columns per shard")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.inputs_file, index_col=0)
    n_traits = len(df.columns)
    n_shards = math.ceil(n_traits / args.shard_size)
    digits = len(str(n_shards - 1))

    for i in range(n_shards):
        shard_cols = df.columns[i * args.shard_size : (i + 1) * args.shard_size]
        shard_path = out_dir / f"shard_{i:0{digits}d}.csv"
        df[shard_cols].to_csv(shard_path)

    print(f"[split_traits] {n_traits} traits -> {n_shards} shards "
          f"of up to {args.shard_size} in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
