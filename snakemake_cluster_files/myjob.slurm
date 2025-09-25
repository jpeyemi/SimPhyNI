#!/bin/bash
#SBATCH -p sched_mit_tami
#SBATCH -n 1
#SBATCH --time=48:00:00
#SBATCH -o mainout.txt
#SBATCH -e mainerr.txt
#SBATCH --mem=16000
#SBATCH --mail-user=EMAIL_HERE
#SBATCH --mail-type=FAIL

bash snakemakeslurm.sh

echo Done!!!
