# SimPhyNI

## Overview

SimPhyNI is a phylogenetically-aware framework for detecting evolutionary associations between binary traits (e.g., gene presence/absence) on microbial phylogenetic trees. This Snakemake pipeline automates simulation, preprocessing, and analysis tasks necessary to run SimPhyNI on genetic and phenotypic datasets like those from *E. coli* or *S. epidermidis* pangenomes.

This pipeline is designed to:

* Prepare and simulate trait evolution using phylogenetic trees
* Detect associations using SimPhyNI methods
* Output statistical results for associations and python objects for downstream analysis

---

## Getting Started

### Installation

Clone the repository and install dependencies using Conda:

```bash
conda install -c bioconda simphyni
```

### Directory Structure

```
SimPhyNI/
├── simphyni/               # Core package
│   ├── Simulation/          # Simulation scripts
│   ├── scripts/             # Workflow scripts
│   └── envs/simphyni.yaml   # Conda environment (used in snakemake)
├── conda-recipe/           # Build recipe
├── samples.csv             # Example input
├── snakemake_cluster_files # Cluster configs for Snakemake
├── SimPhyNI_local_plotting.py
├── simphyni_notebook.ipynb # Example notebook
└── pyproject.toml
```

---

## Usage

### Step 1: Configure Input

Edit `samples.csv` to specify the input trees and trait matrices:

```csv
sample,tree,traits,Run Type
IBD,inputs/IBD_ecoli.nwk,inputs/IBD_ecoli.csv,1
Sepi,inputs/Sepi_mega.nwk,inputs/Sepi_mega.csv,0
```



### Step 2: Run the Workflow

```bash
simphyni run -s samples.csv
```

or

```bash
simphyni run -t trait_file.csv -T tree_file.nwk --runtype 0-All against All ,1-First agaist All (default: 0)
```
---

## Configuration

No explicit `config.yaml` file is needed; configuration is driven by sample metadata (`samples_example.csv`) and `inputs/` folder contents. Ensure trait files are in CSV format with rows as samples and columns as traits.

---

## Outputs

Outputs are placed in structured folders in the working directory or specified output directory in the `3-Objects/` subdirectory, including:

* `simphyni_result.csv` contianing all tested trait pairs with their infered interaction direction, p-value, and effect size
* optional pickled SimPhyNI object of the completed analysis, parsable with the attached environment (not recommended for large analyses, > 100,000 comparisons)

---


## Contact

For questions, please open an issue or contact Ishaq Balogun at https://github.com/jpeyemi.
