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

The defaults were chosen by benchmarking calibration, power and speed on 11 PanX species trees
(103–500 strains) with synthetic, real-gene, late-emerging and lineage-restricted traits, and on
*E. coli* at genome scale (4,501 genes, 10.1 M pairs; 8.4 M known-truth pairs). Details:
`candidates/README.md` and `benchmarks_large/REPORT.md`.

**Default null.** No flags are needed.

- **Weighted statistic.** The log odds ratio is computed on phylogenetic tip weights
  (Gerstein–Sonnhammer–Chothia), so a clade inherited from one gain counts once rather than once per
  tip.
- **Anchored null.** Each trait is simulated 512 times from its fitted gain/loss process; the
  per-branch transitions are exact. Each observed trait is then scored against its partner's
  replicates, in both directions, and the larger p is reported. That p is valid if either trait's
  model is right. No replicates are discarded.
- **Calibrated tails.**
  - Below the resolution of the replicates, p-values come from a generalized-Pareto tail (shape ≥ 0),
    so they are never extrapolated to 0.
  - Pairs with p < 0.01 are re-tested with 32,768 fresh replicates.

**Flags.**

- `--refine-p P` (default 0.01, `0` disables), `--refine-max-pairs N` (default 5000),
  `--refine-n N` (default 32768): the refinement stage.
- `--lineage none|soft_auto` and `--composite`: control for genes confined to the same lineages
  (plasmid-, phage- or phylogroup-specific).
  - `soft_auto` lowers each trait's rate with distance from its own clade, on a scale fitted to the
    data.
  - `--composite` reports max(p whole-tree, p pair-region), which is valid if either null holds.
  - On a replica of the real *E. coli* genome, neither lowered the false-discovery rate, and
    `soft_auto` created false positives, so neither is recommended by default. Genome-wide runs made
    no false calls without them. For small analyses built from lineage-confined genes, use
    `pval_by` and read co-confined positive associations as shared lineage or co-transfer.
- `--trials N` (default 512): replicates per trait.
- `--branch-cap iqr|none` (default `iqr`): simulate with the same long-branch cap used for the rate
  estimate.
- `--null legacy`: the previous null. It is unweighted, simulates replicate-vs-replicate pairs with
  prevalence conditioning (`--condition`, `--oversample`) and uses KDE tails. It is about 50× slower
  at genome scale and anticonservative below 1e-5.

**Reading results.** Select pairs with `pval_bh < 0.05`, then order them by `effect_shrunk`.
`pval_by` is the strict alternative.

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
| `direction` | Direction of association: `1` = positive (co-occurrence), `-1` = negative (avoidance). |
| `effect size` | Standardised association: \|observed − null mean\| / null SD of the weighted log odds ratio. |
| `effect_lor`, `effect_se` | Observed weighted log odds ratio minus the null mean, and the null SD. |
| `effect_shrunk` | Empirical-Bayes posterior mean of `effect_lor` (adaptive shrinkage over all pairs); recommended for ordering. |
| `lfsr` | Local false sign rate of the association. |
| `prevalence_T1` / `prevalence_T2` | Fraction of samples containing each trait (0.0 to 1.0). |
| `pval_naive` | Raw p-value (anchored null, generalized-Pareto tail). |
| `pval_bh` | Benjamini-Hochberg FDR correction (recommended). |
| `pval_by` | Benjamini-Yekutieli FDR correction (strict; valid under arbitrary dependence). |
| `pval_bonf` | Bonferroni correction. |
| `refined` | The p-value comes from the 32,768-replicate refinement stage. |
| `gated` | With `--composite`: the pair-region null gave the larger p-value. |
| `null_calibrated` | `False` for pairs involving traits flagged as possibly miscalibrated (see `--include-flagged`). |

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
