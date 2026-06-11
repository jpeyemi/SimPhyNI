#!/bin/bash

#Edit for your HPC
#SBATCH -p pi_tami
#SBATCH --job-name=simphyni
#SBATCH --output=simphyni_out_%j.txt
#SBATCH --error=simphyni_err_%j.txt
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64000

# Modify for input paths
simphyni run -T /home/iobal/mit_lieberman/projects/Ishaq/SimPhyNI/panx_ecoli/ecoli_accessory_dedup.csv -t /home/iobal/mit_lieberman/projects/Ishaq/SimPhyNI/panx_ecoli/ecoli_accessory_dedup.nwk --sample-name ecoli_accessory_dedup -c 64
