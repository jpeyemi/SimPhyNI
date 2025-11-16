#!/bin/bash

#Edit for your HPC
#SBATCH -p sched_any
#SBATCH --job-name=simphyni
#SBATCH --output=simphyni_out_%j.txt
#SBATCH --error=simphyni_err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000

# Modify for input paths
simphyni run --samples example_inputs/simphyni_sample_info.csv --outdir simphyni_results --profile cluster_profile -c 32 #in this case -c is the max cpus that can be allocated to a single job
