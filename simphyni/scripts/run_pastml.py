#!/usr/bin/env python

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import pandas as pd
import numpy as np

# Ensure the scripts directory is on sys.path so traits_io can be imported
# both when run as a script (Snakemake) and when imported as a package module.
sys.path.insert(0, str(Path(__file__).parent))
from traits_io import (
    get_trait_metadata,
    load_trait_columns,
    load_trait_columns_filtered,
    prefilter_traits_chunked,
    _is_parquet,
)

parser = argparse.ArgumentParser()
parser.add_argument("--inputs_file", required=True)
parser.add_argument("--tree_file", required=True)
parser.add_argument("--outdir", required=True)
parser.add_argument("--max_workers", type=int, default=8)
parser.add_argument("--summary_file", required=True)
parser.add_argument(
        "--prefilter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable prefiltering (default: enabled)",
    )
parser.add_argument("-r", "--run_traits", type=int, default=0)
args = parser.parse_args()


inputs_file = args.inputs_file
tree_file = args.tree_file
output_dir = Path(args.outdir)

# ------------------------------------------------------------------
# Prefiltering: use chunked Fisher test — never loads full matrix
# ------------------------------------------------------------------
index_col, all_traits = get_trait_metadata(inputs_file)

if args.prefilter:
    print("Pre-filtering traits by Fisher exact test (chunked) ...", flush=True)
    sample_ids = prefilter_traits_chunked(
        inputs_file,
        index_col,
        valid_rows=None,   # no row restriction at this stage
        run_traits=args.run_traits,
    )
    print(f"  {len(sample_ids)} traits retained (of {len(all_traits)} input)", flush=True)
else:
    sample_ids = list(all_traits)

max_workers = args.max_workers
summary_file = Path(args.summary_file)
summary_file.parent.mkdir(parents=True, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def run_pastml(sample_id):
    sample_dir = output_dir / sample_id
    os.makedirs(sample_dir, exist_ok=True)
    output_file = sample_dir / "combined_ancestral_states.tab"
    if output_file.exists() and os.path.getsize(output_file) > 10:
        return sample_id, "Skipped (output exists)"

    # For Parquet input: write a per-sample mini CSV (index + this one column).
    # This avoids loading the full matrix upfront; peak RAM per thread is
    # proportional to n_rows × 1 column only.
    local_tmp = None
    if _is_parquet(inputs_file):
        col_df = load_trait_columns(inputs_file, [sample_id], index_col)
        local_tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        )
        col_df.to_csv(local_tmp.name)
        local_tmp.close()
        pastml_input = local_tmp.name
    else:
        pastml_input = inputs_file

    try:
        with open(sample_dir / "pastml.log", "w") as log:
            subprocess.run([
                "pastml",
                "--tree", str(tree_file),
                "--data", str(pastml_input),
                "--columns", sample_id,
                "--id_index", "0",
                "-n", "outs",
                "--work_dir", str(sample_dir),
                "--prediction_method", "JOINT",
                "-m", "F81",
                "--html", str(sample_dir / "out.html"),
                "--data_sep", ","
            ], stdout=log, stderr=subprocess.STDOUT, check=True)
        return sample_id, "Success"
    except subprocess.CalledProcessError as e:
        return sample_id, f"Failed with error: {e}"
    finally:
        if local_tmp is not None:
            try:
                os.unlink(local_tmp.name)
            except OSError:
                pass


results = {}
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_sample = {executor.submit(run_pastml, sid): sid for sid in sample_ids}
    for future in as_completed(future_to_sample):
        sample_id = future_to_sample[future]
        try:
            sample_id, status = future.result()
            results[sample_id] = status
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [FAILED] {sample_id}: {e}\n{tb}", file=sys.stderr, flush=True)
            results[sample_id] = f"Failed with exception: {e}"

with open(summary_file, "w") as f:
    total_samples = len(sample_ids)
    processed_samples = sum(1 for s in results.values() if s == "Success")
    skipped_samples = sum(1 for s in results.values() if s.startswith("Skipped"))
    failed_samples = total_samples - processed_samples - skipped_samples
    f.write(f"Files written to: {output_dir}\n")
    f.write(f"Total samples: {total_samples}\n")
    f.write(f"Processed successfully: {processed_samples}\n")
    f.write(f"Skipped (output exists): {skipped_samples}\n")
    f.write(f"Failed: {failed_samples}\n\n")
    if failed_samples > 0:
        f.write("Failures:\n")
        for sample_id, status in results.items():
            if status.startswith("Failed"):
                f.write(f"{sample_id}: {status}\n")
    f.write("\nJob is complete.\n")
