# SimPhyNI

## Overview

**SimPhyNI** (Simulation-based Phylogenetic iNteraction Inference) is a phylogenetically-aware framework for detecting evolutionary associations between binary traits (e.g., gene presence/absence, major/minor alleles, binary phenotypes) on microbial phylogenetic trees. This tool leverages phylogenetic information to correct for spurious associations caused by the relatedness of sister taxa. 

This pipeline is designed to:

* Infer evolutionary parameters for traits (gain/loss rates, time to emergence, ancestral states)
* Estimate trait co-occurence null models through independent simulation of traits
* Output statistical results for associations

**Graphical Workflow:**
<img width="2332" height="594" alt="Figure2gh" src="https://github.com/user-attachments/assets/7d2dc282-c988-4d7a-93ae-cd041cf6d931" />

---

## Getting Started

### Installation

First, ensure bioconda and conda-forge are channels are configured:

```bash
conda config --add channels conda-forge
conda config --add channels bioconda
```

Create a new environment:

```bash
conda create -n simphyni
conda activate simphyni
```

then install SimPhyNI from bioconda:

```bash
conda install simphyni
```

test installation:

```bash
simphyni version
```

### Input Specifications

**1. Phylogenetic Tree (`.nwk`)**

* Standard Newick format.
* Must be **rooted** (both outgroup and midpoint are acceptable).
* Tip labels must match the `Sample` column in your traits file.
* Branch lengths are required for accurate rate estimation.

**2. Traits File (`.csv`)**

* **Rows:** Genomes/Samples (matching tree tips).
* **Columns:** Binary traits (0 = Absent, 1 = Present; non numerical values will be st to 1 and blank values will be set to 0).
* **Header:** Required (Trait names).
* **Index:** The first column must contain sample names.

*Example `traits.csv`:*

```csv
Sample,PhenotypeX,GeneA,GeneB
E_coli_1,1,0,1
E_coli_2,1,1,0
E_coli_3,0,0,1

```

#### Starting from FASTA files and looking for gene-gene or gene-phenotype associations?

If you have raw genome assemblies (FASTA) and need to generate the necessary inputs (gene presence/absence and a phylogenetic tree), we provide a dedicated pipeline: **[SimPhyNI-Prelude](https://github.com/jpeyemi/SimPhyNI-Prelude)**.

This Snakemake workflow is configured for HPC and automates the following steps:

* **Annotation** (Prokka)
* **Pangenome Analysis** (Panaroo)
* **Tree Construction** (PopPUNK or RAxML)
* **Formatting** (Preparation for SimPhyNI)
* **SimPhyNI Analysis** (This repository)

Any steps may be bypassed by providing existing data (e.g. Gene annotations, phylogenetic tree)
For those familiar with Snakemake, rules can be edited, added, or removed to suit your needs

---

## Usage

### Run mode (single-run)

```bash
simphyni run \
  --sample-name my_sample \
  --tree path/to/tree.nwk \
  --traits path/to/traits.csv \
  --run-traits 0,1,2 \
  --outdir my_analysis \
  --cores 4 \
  --temp_dir ./tmp \
  --min_prev 0.05 \
  --max_prev 0.95 \
  --plot
```

* `--run-traits` specifies a comma-separated list of column indices (0-indexed) in the traits CSV for “trait against all” comparisons. Use 'ALL' (default) to include all traits.


### Run mode (batch)

Create a `samples.csv` file:

```csv
Sample,Tree,Traits,run_traits,MinPrev,MaxPrev
run1,tree1.nwk,traits1.csv,All,0.05,0.95
run2,tree2.nwk,traits2.csv,"0,1,2",0.05,0.90
```

* `run_traits`, `MinPrev`, and `MaxPrev` are optional columns that will use default values if not provided.

Then execute:

```bash
simphyni run --samples samples.csv --cores 16
```

### Run with HPC

First, download example cluster scripts:
```bash
simphyni download-cluster-scripts
```

Edit cluster config file for your computing cluster then install the approprate snakemake executor from the avalible catalog: https://snakemake.github.io/snakemake-plugin-catalog/index.html (slurm shown below): 
```bash
pip install snakemake-executor-plugin-slurm
```

run simphyni with the --profile flag:
```bash
simphyni run --samples samples.csv --profile cluster_profile
```

For all run options:

```bash
simphyni run --help
```

## Example data

Download and run example inputs using:
```bash
simphyni download-examples
simphyni run --samples example_inputs/simphyni_sample_info.csv --cores 8 --plot
```
---

## Outputs

Outputs for each sample are placed in structured folders in the working directory or specified output directory in subdirectories by sample name, including:

### Main Result Files

**`simphyni_result.csv`**
Contains the statistical results for all tested trait pairs.

| Column | Description |
| --- | --- |
| `T1` / `T2` | Identifiers for the two traits being compared. |
| `direction` | Direction of association: `1` = Positive, `-1` = Negative. |
| `effect size` | Variance adjusted magnitude of the association. |
| `pval_naive` | Raw empirical P-value from the simulation. |
| `pval_bh` | P-value corrected using the Benjamini-Hochberg FDR method (recommended for phenotype-genotype tests). |
| `pval_by` | P-value corrected using the Benjamini-Yekutieli FDR method. (recommended for genotype-genotype tests)|
| `pval_bonf` | P-value corrected using the strict Bonferroni method. |
| `prevalence_T1` / `_T2` | Fraction of samples containing the trait (0.0 to 1.0). |

### Additional Outputs

* **`simphyni_object.pkl`**: Optional file containing the completed analysis object, parsable with an active SimPhyNI environment. Controlled with the `--save-object` flag (not recommended for large analyses > 1,000,000 comparisons).
* **Plots**: Heatmap summaries of tested associations (if `--plot` is enabled).

---

### Directory Structure

```
SimPhyNI/
├── simphyni/               # Core package
│   ├── Simulation/         # Simulation scripts
│   ├── scripts/            # Snakemake scripts
│   ├── Snakefile.py/       # Workflow build file
│   ├── simphyni_cli.py/    # Command line entry points
│   └── envs/simphyni.yaml  # Conda environment (used in snakemake)
├── test/                   # Testing suite
├── conda-recipe/           # Build recipe 
├── cluster_scripts         # Cluster configs for SLURM
├── example_inputs          # Example inputs to run SimPhyNI
└── pyproject.toml
```

---


## Contact

For questions, please open an issue or contact Ishaq Balogun at https://github.com/jpeyemi.

## Citation

If you use SimPhyNI in your research, please cite:


> **High Precision Binary Trait Association on Phylogenetic Trees**
> Ishaq O Balogun, Christopher P Mancuso, Tami D Lieberman
> bioRxiv 2025.12.24.696407; doi: https://doi.org/10.64898/2025.12.24.696407


