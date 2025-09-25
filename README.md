# SimPhyNI

## Overview

SimPhyNI is a phylogenetically-aware framework for detecting evolutionary associations between binary traits (e.g., gene presence/absence) on microbial phylogenetic trees. This Snakemake pipeline automates simulation, preprocessing, and analysis tasks necessary to run SimPhyNI on genetic and phenotypic datasets like those from *E. coli* or *S. epidermidis* pangenomes.

This pipeline is designed to:

* Prepare and simulate trait evolution using phylogenetic trees
* Detect associations using SimPhyNI methods
* Support high-performance computing (HPC) environments via SLURM
* Provide reproducible, modular workflows

---

## Getting Started

### Installation

Clone the repository and install dependencies using Conda:

```bash
git clone https://github.com/jpeyemi/SimPhyNISnakemake.git
cd SimPhyNISnakemake
conda env create -f envs/simphyni.yaml -n simphyni
conda activate simphyni
```

Ensure Snakemake is installed:

```bash
conda install -c bioconda snakemake
```

### Directory Structure

```
SimPhyNISnakemake/
├── Snakefile.py             # Main Snakemake workflow
├── envs/simphyni.yaml       # Conda environment for reproducibility
├── inputs/                  # Input trees and trait files (CSV, Newick)
├── samples_example.csv      # Sample file mapping names to trees and traits
├── snakemakeslurm.sh        # SLURM job submission wrapper
├── cluster.slurm.json       # SLURM cluster configuration
├── myjob.slurm              # Example SLURM job
```

---

## Usage
clsa
### Step 1: Configure Input

Edit `samples.csv` to specify the input trees and trait matrices:

```csv
sample,tree,traits,Run Type
IBD,inputs/IBD_ecoli.nwk,inputs/IBD_ecoli.csv,1
Sepi,inputs/Sepi_mega.nwk,inputs/Sepi_mega.csv,0
```

### Step 2: Run the Workflow

On a local machine:

```bash
snakemake --cores 4 -s Snakefile.py
```

On an HPC with SLURM:

Edit `cluster.slurm.json` and `myjob.slurm` to use partitions specific to your HPC. Then run:

```bash
sbatch myjob.slurm
```
---

## Configuration

No explicit `config.yaml` file is needed; configuration is driven by sample metadata (`samples_example.csv`) and `inputs/` folder contents. Ensure trait files are in CSV format with rows as samples and columns as traits.

---

## Outputs

Outputs are placed in structured folders in the snakemake directory with final results under `3-Objects/` (or as defined in your Snakefile), including:

* `simphyni_result.csv` contianing all tested trait pairs with their infered interaction direction, p-value, and effect size
* optional pickled SimPhyNI object of the completed analysis, parsable with the attached environment (not recommended for large analyses, > 100,000 comparisons)

---


## Contact

For questions, please open an issue or contact Ishaq Balogun at https://github.com/jpeyemi.
