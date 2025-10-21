#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import pandas as pd
import numpy as np
from scipy.special import loggamma

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
parser.add_argument("-r", "--runtype", type=int, choices=[0, 1], default=0,
                    help="1 for single trait mode, 0 for multi-trait [default: 0]")
args = parser.parse_args()

def prefiltering(obs):
    valid_obs= np.array(obs.columns)

    def fisher_significant_pairs(vars: pd.DataFrame, targets: pd.DataFrame, valid_vars, valid_targets, pval_threshold: float = 0.05):
        X = vars.to_numpy().astype(bool)
        Y = targets.to_numpy().astype(bool)
        n = X.shape[0]

        # Compute all pairwise contingency counts efficiently
        a = X.T @ Y                      # (n_vars x n_targets) both=1
        sX = X.sum(axis=0)               # (n_vars,)
        sY = Y.sum(axis=0)               # (n_targets,)

        b = sX[:, None] - a              # (i=1, j=0)
        c = sY[None, :] - a              # (i=0, j=1)
        d = n - (a + b + c)              # both=0

        # Compute marginals
        row1 = a + b
        row2 = c + d
        col1 = a + c
        n_all = n

        # Log-binomial function
        def logC(n, k):
            return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

        # log probability of observed a under null
        logp_obs = logC(row1, a) + logC(row2, col1 - a) - logC(n_all, col1)
        p_obs = np.exp(logp_obs)

        # approximate two-sided p-value as 2 * min(one-sided, 1)
        p_two = np.minimum(1.0, 2 * p_obs)

        # Mask NaNs and invalids
        p_two[np.isnan(p_two)] = 1.0

        # Apply significance threshold
        sig_mask = p_two < pval_threshold

        # Get indices of significant pairs
        i_idx, j_idx = np.where(sig_mask)

        sig_pairs = np.column_stack((valid_vars[i_idx], valid_targets[j_idx]))
        sig_pvals = p_two[i_idx, j_idx]

        return sig_pairs, sig_pvals

    if args.runtype == 0:
        pairs, pvals = fisher_significant_pairs(obs[[valid_obs[0]]],obs,np.array([valid_obs[0]]),valid_obs)
    else:
        pairs, pvals = fisher_significant_pairs(obs,obs,valid_obs,valid_obs)
    filtered_obs = np.unique(pairs.flatten())
    return filtered_obs


inputs_file = args.inputs_file
tree_file = args.tree_file
output_dir = Path(args.outdir)
obs = pd.read_csv(inputs_file, index_col = 0)
if args.prefilter:
    sample_ids = prefiltering(obs)
else:
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
