#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--inputs_file", required=True)
parser.add_argument("--tree_file", required=True)
parser.add_argument("--outdir", required=True)
parser.add_argument("--max_workers", type=int, default=8)
parser.add_argument("--summary_file", required=True)
args = parser.parse_args()

inputs_file = args.inputs_file
tree_file = args.tree_file
output_dir = Path(args.outdir)
obs = pd.read_csv(inputs_file, index_col = 0)
sample_ids = list(obs.columns)
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
    try:
        with open(sample_dir / "pastml.log", "w") as log:
            subprocess.run([
                "pastml",
                "--tree", str(tree_file),
                "--data", str(inputs_file),
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

results = {}
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_sample = {executor.submit(run_pastml, sid): sid for sid in sample_ids}
    for future in as_completed(future_to_sample):
        sample_id = future_to_sample[future]
        try:
            sample_id, status = future.result()
            results[sample_id] = status
        except Exception as e:
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
