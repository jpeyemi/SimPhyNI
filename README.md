# SimPhyNI

## Overview

**SimPhyNI** (Simulation-based Phylogenetic iNteraction Inference) is a phylogenetically-aware framework for detecting evolutionary associations between binary traits (e.g., gene presence/absence, major/minor alleles, binary phenotypes) on microbial phylogenetic trees. This tool leverages phylogenetic information to correct for spurious associations caused by the relatedness of sister taxa.

This pipeline is designed to:

* Infer evolutionary parameters for traits (gain/loss rates, time to emergence, ancestral states)
* Estimate trait co-occurrence null models through independent simulation of traits
* Output statistical results for associations

**Graphical Workflow:**
<img width="2332" height="594" alt="Figure2gh" src="https://github.com/user-attachments/assets/7d2dc282-c988-4d7a-93ae-cd041cf6d931" />

---

## Getting Started

### Installation

First, ensure bioconda and conda-forge channels are configured:

```bash
conda config --add channels conda-forge
conda config --add channels bioconda
```

Create a new environment:

```bash
conda create -n simphyni
conda activate simphyni
```

Then install SimPhyNI from bioconda:

```bash
conda install simphyni
```

Test installation:

```bash
simphyni version
```

### Input Specifications

**1. Phylogenetic Tree (`.nwk`)**

* Standard Newick format.
* Must be **rooted** (both outgroup and midpoint are acceptable).
* Tip labels must match the sample name index in your traits file.
* Branch lengths are required for accurate rate estimation.

**2. Traits File (`.csv` or `.parquet`)**

* **Rows:** Genomes/Samples (matching tree tips).
* **Columns:** Binary traits (0 = Absent, 1 = Present; non-numerical values will be set to 1 and blank values will be set to 0).
* **Header:** Required (Trait names).
* **Index:** The first column must contain sample names matching tree tip labels.

For large datasets (>10,000 traits), Parquet format is strongly recommended. Parquet enables column-projection and row-predicate pushdown, so only the data actually needed is loaded from disk, dramatically reducing peak memory use.

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

Any steps may be bypassed by providing existing data (e.g. gene annotations, phylogenetic tree).

For those familiar with Snakemake, rules can be edited, added, or removed to suit your needs.

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
  --temp-dir ./tmp \
  --min_prev 0.05 \
  --max_prev 0.95 \
  --plot
```

**Key flags:**

* `--run-traits` — Comma-separated list of N values (e.g. `0,1,2`) to designate the first N traits in the file as query traits for a one-vs-rest comparison. Use `ALL` (default) for all-against-all. For example, `--run-traits 0,1,2` tests the first 3 traits against all others.
* `--include-flagged` / `--no-include-flagged` — Some traits are flagged as having a possibly miscalibrated null distribution (sparse events combined with highly asymmetric eligible regions). By default their pairs are tested normally and marked `null_calibrated=False` in the output so you can assess them separately. `--no-include-flagged` drops them from the results instead. Access the list of flagged traits via `Sim.get_flagged_traits()` when using the Python API.
* `--min_prev` / `--max_prev` — Minimum and maximum tip prevalence thresholds for traits to be included (default: 0.05 / 0.95).
* `--plot` — Generate heatmap summaries of results.
* `--save-object` — Save the full analysis object as a `.pkl` file (not recommended for analyses > 1,000,000 trait pairs).
* `--seed` — Make results reproducible (independent of `--cores`).

#### Null model options

The defaults below were chosen by benchmarking calibration and power on 11 PanX species trees (103–500 strains), with synthetic, real-gene, late-emerging and lineage-restricted traits (`benchmarks/panx/` on the `panx-benchmark` branch).

* `--condition prevalence|footprint|none` (default `prevalence`) and `--oversample K` (default 8) — Each trait's null is conditioned on the observed trait. `K × 64` replicates are simulated and the 64 closest to the observed trait are kept, by prevalence or (`footprint`) by prevalence and number of parsimony changes. Without this, traits that changed only once or twice are simulated with their events on short tip branches, so their null copies are scattered fragments rather than the observed clade, and their null co-occurrence is far too narrow. `footprint` has more power and is close to nominal on most trees, but is less robust on very clonal trees.
* `--branch-cap iqr|none` (default `iqr`) — Simulate with the same long-branch cap used for the rate estimate. Rates are estimated on capped branch lengths. Applying them to uncapped lengths makes long deep branches flip in almost every replicate on clonal trees, where most tree length can lie above the cap.
* `--emergence none|dist|scaled|soft|mrca` (default `none`) — The time-to-first-event threshold (`dist`, the previous default) made the null too narrow, even for traits that really do emerge late. `mrca` restricts events to the subtree of the trait's minority-state tips and is conservative but loses most power.
* `--eligibility none|phylogroup|adaptive` (default `none`) — Restrict each trait's null to lineages it can enter. `phylogroup` cuts the tree at `--phylogroup-depth` (default 0.2 of tree depth) and allows events only in phylogroups where the trait is observed. `adaptive` excludes a phylogroup only when the trait is absent there *and* the unrestricted null would almost never leave it empty (`--adapt-alpha`, default 0.05), so sparse traits are not restricted. Useful for trees spanning several species, where some genes can never enter some lineages.


### Run mode (batch)

Create a `samples.csv` file:

```csv
Sample,Tree,Traits,run_traits,MinPrev,MaxPrev
run1,tree1.nwk,traits1.csv,ALL,0.05,0.95
run2,tree2.nwk,traits2.csv,"0,1,2",0.05,0.90
```

* `run_traits`, `MinPrev`, and `MaxPrev` are optional columns that will use default values if not provided.
* `run_traits` must be `ALL` (case-sensitive) or a comma-separated list of N values (e.g. `0,1,2` to query the first 3 traits against all others).

Then execute:

```bash
simphyni run --samples samples.csv --cores 16
```

### Run with HPC

First, download the cluster profile template:

```bash
simphyni download-cluster-profile
```

Edit `cluster_profile/config.yaml` for your computing cluster, then install the appropriate Snakemake executor from the available catalog: https://snakemake.github.io/snakemake-plugin-catalog/index.html (SLURM shown below):

```bash
pip install snakemake-executor-plugin-slurm
```

Run SimPhyNI with the `--profile` flag:

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

**`simphyni_results.csv`**
Contains the statistical results for all tested trait pairs.

| Column | Description |
| --- | --- |
| `T1` / `T2` | Identifiers for the two traits being compared. |
| `direction` | Direction of association: `1` = Positive, `-1` = Negative. |
| `effect size` | Variance-adjusted magnitude of the association. |
| `prevalence_T1` / `prevalence_T2` | Fraction of samples containing each trait (0.0 to 1.0). |
| `pval_naive` | Raw empirical P-value from the simulation. |
| `pval_bh` | Benjamini-Hochberg FDR correction (recommended for phenotype-genotype tests). |
| `pval_by` | Benjamini-Yekutieli FDR correction (recommended for genotype-genotype tests; accounts for correlated hypotheses, which is the typical case for phylogenetically structured data). |
| `pval_bonf` | Bonferroni correction (strictest; use when a small number of specific hypotheses are tested). |
| `null_calibrated` | `True` for standard results. `False` for pairs involving traits whose null distributions were detected as miscalibrated (see `--include-flagged`). |

### Additional Outputs

* **`simphyni_results.csv`**: Main results table (see above).
* **`simphyni_object.pkl`**: Optional file containing the completed analysis object, parsable with an active SimPhyNI environment. Controlled with the `--save-object` flag (not recommended for large analyses > 1,000,000 comparisons).
* **Plots**: Heatmap summaries of tested associations (if `--plot` is enabled).

---

### Directory Structure

```
SimPhyNI/
├── simphyni/               # Core package
│   ├── Simulation/         # Simulation engine
│   ├── scripts/            # Pipeline scripts
│   ├── Snakefile.py        # Workflow definition
│   ├── simphyni_cli.py     # Command-line entry points
│   └── envs/simphyni.yaml  # Conda environment (used by Snakemake)
├── tests/                  # Testing suite
├── conda-recipe/           # Build recipe
├── cluster_profile/        # Cluster config template for HPC
├── example_inputs/         # Example inputs to run SimPhyNI
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
