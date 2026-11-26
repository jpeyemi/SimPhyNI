# SimPhyNI

## Overview

**SimPhyNI** (Simulation-based Phylogenetic iNteraction Inference) is a phylogenetically-aware framework for detecting evolutionary associations between binary traits (e.g., gene presence/absence, major/minor alleles, binary phenotypes) on microbial phylogenetic trees. This tool leverages phylogenetic information to correct for spurious associations caused by the relatedness of sister taxa.

This pipeline is designed to:

* Infer evolutionary parameters for traits (gain/loss rates, time to emergence, ancestral states)
* Estimate trait co-occurrence null models through independent simulation of traits
* Output statistical results for associations

**Full documentation:** [docs/USER_GUIDE.md](docs/USER_GUIDE.md) covers installation, every option,
every output column, interpretation and workflows.

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
* **Columns:** Binary traits (0 = Absent, 1 = Present; values above 0.5 count as present). Blank / `NA` cells are treated as **missing**: the ancestral reconstruction treats them as unknown, and each pair is tested on the tips where both traits are known.
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
# every pair of traits
simphyni run --traits traits.csv --tree tree.nwk --sample-name my_run --outdir results --cores 8 --plot

# query traits (columns 0 and 3) against all others
simphyni run --traits traits.csv --tree tree.nwk --run-traits 0,3 --outdir results --plot
```

**Key flags** (`simphyni run --help`; all options are described in the [user guide](docs/USER_GUIDE.md#4-command-line-options)):

* `-T/--traits`, `-t/--tree`: trait table (.csv or .parquet; rows = tips) and rooted Newick tree.
* `-r/--run-traits`: `ALL` (default) for every pair, or comma-separated column indices of query traits
  (0 = first trait) to test against all others.
* `--min-prev` / `--max-prev`: keep traits present in this range of tip fractions (default 0.05–0.95).
* `--output-format csv|parquet`: format of the result tables. Parquet is much smaller for > 1 M pairs.
* `--plot`: write summary plots (volcano, heatmap, p-value histogram, top traits).
* `--seed`: reproducible results (independent of `--cores`).
* `--include-flagged` / `--no-include-flagged`: keep (default) or drop pairs with traits whose null may be
  miscalibrated. Kept pairs are marked `null_calibrated=False`.
* `--save-object`: also save the analysis object as `.pkl` (avoid for > 1 M pairs).

#### Null model options

The defaults suit most analyses; no flags are needed. For how SimPhyNI compares with other tools,
see the live benchmark **[mgwas-bench](https://jpeyemi.github.io/mgwas-bench/)**, which re-runs
SimPhyNI and other methods on every release.

**Default null:**

- **Weighted statistic.** The log odds ratio uses phylogenetic tip weights, so a clade inherited from
  one gain counts about once rather than once per tip.
- **Anchored null.** Each trait is simulated from its estimated gain/loss history. Each observed trait
  is scored against its partner's simulated replicates, in both directions, and the larger p-value is
  reported.
- **Calibrated tails.** p-values are exact Monte-Carlo counts with a conservative tail model beyond the
  most extreme replicate. Pairs with small p-values are re-tested with more replicates.

**Options:**

- `--refine-p` (default 0.01; `0` disables), `--refine-max-pairs`, `--refine-n`: the re-test of
  small p-values.
- `--composite`: an optional, more conservative null for pairs of traits confined to the same lineages.
  It is not needed for genome-wide analyses ([guide §7.4](docs/USER_GUIDE.md#74-lineage-confined-traits-and-co_confined)).
- `--trials` (default 512): replicates per trait.
- `--null legacy`: the SimPhyNI ≤ 2.0 null, for reproducing earlier results.

**Reading results.** Select pairs with `pval_bh < 0.05` and order them by `effect_shrunk`. Use `pval_by`
for small analyses built mostly from lineage-confined traits. See the
[user guide](docs/USER_GUIDE.md#7-interpreting-results) for interpretation and pitfalls.

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

Each sample's results are written to `<outdir>/<sample>/` (`.csv` or `.parquet`). Every column is
explained in the [user guide](docs/USER_GUIDE.md#5-outputs).

| file | one row per | contents |
| --- | --- | --- |
| `simphyni_results` | tested pair | compact table for selecting and ranking associations (below) |
| `simphyni_results_long` | tested pair | every statistic: 2×2 tip counts, weighted log odds ratio, null mean and SD, raw and shrunken effects, raw p, BH/BY/Bonferroni, refinement, lineage confinement, estimated changes per trait |
| `simphyni_traits` | trait | prevalence, estimated gains/losses, root state, lineage confinement, number of significant positive and negative partners |

**`simphyni_results` (compact)**, sorted by `pval_bh` and then by effect size:

| Column | Description |
| --- | --- |
| `T1` / `T2` | The two traits. |
| `direction` | `1` = co-occur more than the phylogeny predicts (positive); `-1` = less (negative). |
| `effect_shrunk` | Association strength in log-odds units after empirical-Bayes shrinkage. **Rank by this.** |
| `lfsr` | Local false sign rate: probability that the direction is wrong. |
| `pval_bh` | Benjamini–Hochberg q-value. **Select pairs with `pval_bh < 0.05`.** |
| `pval_by` | Benjamini–Yekutieli q-value: stricter, valid under any dependence. |
| `prevalence_T1` / `prevalence_T2` | Fraction of tips carrying each trait. |
| `n_both` | Tips carrying both traits. |
| `co_confined` | Both traits are confined to the same small part of the tree. A positive association may then reflect shared lineage or co-transfer rather than interaction ([guide §7.4](docs/USER_GUIDE.md#74-lineage-confined-traits-and-co_confined)). |
| `null_calibrated` | `False` for pairs involving traits whose null may be miscalibrated. |

**Plots** (`--plot`):

* `simphyni_volcano.png`: all pairs, shrunken effect vs significance; significant pairs coloured by
  direction, co-confined pairs outlined.
* `simphyni_heatmap.png`: the 40 most-connected traits, clustered; colour = shrunken effect where q < 0.05.
* `simphyni_pvalues.png`: p-value histogram, a quick calibration check.
* `simphyni_top_traits.png`: traits with the most significant partners.
* `volcano_<trait>.png`: one per query trait in `--run-traits` runs.

**Other files:** `simphyni_object.pkl` (with `--save-object`) and intermediate files in `--temp-dir`.

---

## Contact

For questions, please open an issue or contact Ishaq Balogun at https://github.com/jpeyemi.

## Citation

If you use SimPhyNI in your research, please cite:


> **High Precision Binary Trait Association on Phylogenetic Trees**
> Ishaq O Balogun, Christopher P Mancuso, Tami D Lieberman
> bioRxiv 2025.12.24.696407; doi: https://doi.org/10.64898/2025.12.24.696407
