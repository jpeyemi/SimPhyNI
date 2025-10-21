# SimPhyNI

## Overview

**SimPhyNI** (Simulation-based Phylogenetic iNteraction Inference) is a phylogenetically-aware framework for detecting evolutionary associations between binary traits (e.g., gene presence/absence, major/minor alleles, binary phenotypes) on microbial phylogenetic trees. This tool leverages phylogenetic infromation to correct for surious associations caused by the relatedness of sister taxa. 

This pipeline is designed to:

* Infer evolutionary parameters for traits (gain/loss rates, time to emergence, ancestral states)
* Estimate trait co-occurence null models through independent simulation of traits
* Output statistical results for associations 

---

## Getting Started

### Installation

First create a new environment:

```bash
conda create -n simphyni python=3.11
conda activate simphyni
```

then install using using PyPI

```bash
pip install simphyni
```

test installation:

```bash
simphyni version
```

---

## Usage

### Run mode (single-run)

```bash
simphyni run \
  --tree path/to/tree.nwk \
  --traits path/to/traits.csv \
  --runtype 0 \
  --outdir my_analysis \
  --cores 4 \
  --temp_dir ./temp \
  --min_prev 0.05 \
  --max_prev 0.95 \
  --prefilter \
  --plot
```

### Run mode (batch)

Create a `samples.csv` file like:

```csv
Sample,Tree,Traits,RunType,MinPrev,MaxPrev
run1,tree1.nwk,traits1.csv,0,0.05,0.95
run2,tree2.nwk,traits2.csv,1,0.05,0.90
```

Then execute:

```bash
simphyni run --samples samples.csv --cores 8 --temp_dir ./temp
```

For all run options:

```bash
simphyni run --help
```

## Example data

Download and run example inputs using:
```bash
simphyni download-examples
simphyni run --samples example_inputs/simphyni_sample_info.csv --cores 8 --prefilter --plot
```
---

## Outputs

Outputs are placed in structured folders in the working directory or specified output directory in the `3-Objects/` subdirectory, including:

* `simphyni_result.csv` contianing all tested trait pairs with their infered interaction direction, p-value, and effect size
* `simphyni_object.pkl` containinf the completed analysis, parsable with the attached environment (not recommended for large analyses, > 1,000,000 comparisons)
* heatmap summaries of tested associations if --plot is enabled

---

### Directory Structure

```
SimPhyNI/
├── simphyni/               # Core package
│   ├── Simulation/          # Simulation scripts
│   ├── scripts/             # Workflow scripts
│   └── envs/simphyni.yaml   # Conda environment (used in snakemake)
├── conda-recipe/           # Build recipe 
├── snakemake_cluster_files # Cluster configs for Snakemake
└── pyproject.toml
```

---


## Contact

For questions, please open an issue or contact Ishaq Balogun at https://github.com/jpeyemi.
