# SimPhyNI user guide

SimPhyNI tests pairs of binary traits (gene presence/absence, alleles, phenotypes) for
**co-occurrence or avoidance beyond what the phylogeny explains**.

Closely related genomes share traits by descent. A test that ignores the tree (such as Fisher's exact
test) therefore reports many associations that only reflect shared ancestry. SimPhyNI avoids this. It
estimates each trait's history of gains and losses on your tree, simulates the trait many times under
that history, and asks whether the observed pair is unusual compared with independently evolving
traits.

This guide explains how to prepare inputs, run SimPhyNI, read its outputs and report the results. For
how SimPhyNI compares with other tools, see the live benchmark,
**[mgwas-bench](https://jpeyemi.github.io/mgwas-bench/)**
([repository](https://github.com/jpeyemi/mgwas-bench)). It re-runs SimPhyNI and other methods on
every release.

- [1. Installation](#1-installation)
- [2. Preparing inputs](#2-preparing-inputs)
- [3. Running SimPhyNI](#3-running-simphyni)
- [4. Command-line options](#4-command-line-options)
- [5. Outputs](#5-outputs)
- [6. Plots](#6-plots)
- [7. Interpreting results](#7-interpreting-results)
- [8. Recommended workflows](#8-recommended-workflows)
- [9. Python API](#9-python-api)
- [10. How the method works](#10-how-the-method-works)
- [11. Computational resources](#11-computational-resources)
- [12. Advanced options](#12-advanced-options)
- [13. Reporting and citing](#13-reporting-and-citing)
- [14. Troubleshooting and FAQ](#14-troubleshooting-and-faq)

---

## 1. Installation

**conda (recommended):**

```bash
conda config --add channels conda-forge
conda config --add channels bioconda
conda create -n simphyni simphyni
conda activate simphyni
simphyni version
```

**From source** (development version):

```bash
git clone https://github.com/jpeyemi/SimPhyNI.git
cd SimPhyNI
conda env create -f simphyni/envs/simphyni.yaml -n simphyni
conda activate simphyni
pip install -e .
simphyni version
```

SimPhyNI needs Python 3.11–3.12. The conda package installs everything else (numpy, pandas, numba,
scipy, statsmodels, ete3, matplotlib, pyarrow, snakemake).

To check your installation, run the bundled example:

```bash
simphyni download-examples
simphyni run --samples example_inputs/simphyni_sample_info.csv --cores 8 --plot
```

---

## 2. Preparing inputs

SimPhyNI's results are only as good as its two inputs: the tree and the trait table.

### Phylogenetic tree (`.nwk`)

- **Format.** Newick, **rooted**. Outgroup or midpoint rooting are both fine.
- **Branch lengths are required.** They determine where on the tree changes are likely, so use a
  tree with meaningful lengths (e.g. substitutions per site from a core-genome alignment).
- **Tip names.** They must match the row names of the trait table exactly. Tips missing from the
  table and rows missing from the tree are dropped.
- **Clonal clusters.** Very short and zero-length branches are supported.

### Trait table (`.csv` or `.parquet`)

- **Rows:** genomes. The first column holds names matching the tree tips.
- **Columns:** binary traits. The header row holds trait names.
- **Values:** 1 = present, 0 = absent; values above 0.5 count as present.
- **Missing values:** blank or `NA` cells mean *unknown*. The ancestral reconstruction treats them as
  unknown tips, and each pair is tested only on the tips where both traits are known.

```csv
Sample,PhenotypeX,GeneA,GeneB
E_coli_1,1,0,1
E_coli_2,1,1,0
E_coli_3,0,0,1
```

**Good practice:**

- **Filter by prevalence.** Traits outside `--min-prev` / `--max-prev` (default 5–95 % of tips) are
  dropped. Very rare or nearly fixed traits carry little information, and their histories are poorly
  estimated.
- **Collapse redundant traits.** Gene clusters that are identical across all genomes would each be
  tested against every other trait, which adds tests without adding information.
- **Use Parquet for large tables** (more than about 10,000 traits). It loads only the columns and rows
  needed, which cuts memory use substantially.

### Batch file (optional, `--samples`)

```csv
Sample,Tree,Traits,run_traits,MinPrev,MaxPrev
run1,tree1.nwk,traits1.csv,ALL,0.05,0.95
run2,tree2.nwk,traits2.csv,"0,1,2",0.05,0.90
```

`run_traits`, `MinPrev` and `MaxPrev` are optional.

**Starting from genome assemblies?** [SimPhyNI-Prelude](https://github.com/jpeyemi/SimPhyNI-Prelude)
builds the pangenome presence/absence table and the tree (Prokka, Panaroo, PopPUNK/RAxML).

---

## 3. Running SimPhyNI

**All pairs of traits:**

```bash
simphyni run --traits traits.csv --tree tree.nwk --sample-name my_run --outdir results --cores 8 --plot
```

**Query traits against all others** (e.g. a phenotype in column 0 against every gene):

```bash
simphyni run --traits traits.csv --tree tree.nwk --run-traits 0 --outdir results --plot
```

`--run-traits 0,3,7` uses columns 0, 3 and 7 (0 = first trait column) as queries.

**Several datasets:**

```bash
simphyni run --samples samples.csv --cores 16
```

**On an HPC cluster:**

```bash
simphyni download-cluster-profile        # template profile + submit script
simphyni run --samples samples.csv --profile cluster_profile
```

Results go to `<outdir>/<sample>/`. Set `--seed` for reproducible results; results do not depend on
`--cores`.

---

## 4. Command-line options

`simphyni run --help` lists these options.

### Input

| option | default | description |
|---|---|---|
| `-T, --traits` | — | trait table (single run) |
| `-t, --tree` | — | rooted Newick tree (single run) |
| `-S, --samples` | — | batch file (instead of `-T`/`-t`) |
| `-s, --sample-name` | trait file name | name of the run's output folder |
| `-r, --run-traits` | `ALL` | `ALL` = every pair; or comma-separated column indices of query traits |
| `--min-prev` | 0.05 | drop traits present in fewer than this fraction of tips |
| `--max-prev` | 0.95 | drop traits present in more than this fraction of tips |

### Output

| option | default | description |
|---|---|---|
| `-o, --outdir` | `simphyni_outs` | output directory |
| `--output-format` | `csv` | `csv` or `parquet` (much smaller and faster for large runs) |
| `--plot` | off | write the summary plots (section 6) |
| `--save-object` | off | also save the Python analysis object (`.pkl`); avoid for large runs |
| `--temp-dir` | `<outdir>/tmp` | intermediate files (ancestral reconstruction, reformatted inputs) |

### Statistics

| option | default | description |
|---|---|---|
| `--null` | `default` | `default` = the current null model (section 10); `legacy` = the SimPhyNI ≤ 2.0 null |
| `--refine-p` | 0.01 | pairs with p below this are re-tested with `--refine-n` fresh replicates (0 disables) |
| `--refine-max-pairs` | 5000 | at most this many pairs (the smallest p) are re-tested per run |
| `--refine-n` | 32768 | replicates per trait in the re-test |
| `--trials` | 512 | null replicates per trait in the main test |
| `--composite` | off | optional extra null that confines both traits to their clades (section 7.4) |
| `--include-flagged` | on | keep pairs with traits whose null may be miscalibrated (marked `null_calibrated = False`) |
| `--seed` | none | random seed |

The defaults suit most analyses. Change them only for a specific reason, and report any changes.

### Execution

| option | default | description |
|---|---|---|
| `-c, --cores` | all | maximum cores |
| `--profile` | — | Snakemake profile for clusters |
| `--dry-run` | off | list the jobs without running them |
| extra arguments | — | anything after the options is passed to Snakemake |

---

## 5. Outputs

Each run writes three tables to `<outdir>/<sample>/` (`.csv` or `.parquet`).

| file | one row per | use it for |
|---|---|---|
| `simphyni_results` | tested pair | selecting and ranking associations (compact) |
| `simphyni_results_long` | tested pair | every statistic, for checking or further analysis |
| `simphyni_traits` | trait | trait-level summaries and hubs |

Pairs are sorted by `pval_bh`, then by the size of the shrunken effect.

### 5.1 `simphyni_results` (compact)

| column | meaning |
|---|---|
| `T1`, `T2` | the two traits |
| `direction` | `1` = co-occur more than expected (positive), `-1` = less (negative / avoidance) |
| `effect_shrunk` | association strength in log-odds units, after empirical-Bayes shrinkage; use to **rank** |
| `lfsr` | local false sign rate: the probability that the direction of the association is wrong |
| `pval_bh` | Benjamini–Hochberg q-value; use to **select** |
| `pval_by` | Benjamini–Yekutieli q-value; stricter, valid under any dependence between tests |
| `prevalence_T1`, `prevalence_T2` | fraction of tips carrying each trait |
| `n_both` | tips carrying both traits |
| `co_confined` | both traits are confined to the same small part of the tree (section 7.4) |
| `null_calibrated` | `False` if a trait's null may be miscalibrated (section 7.5) |

### 5.2 `simphyni_results_long` (all fields)

| column | meaning |
|---|---|
| `T1`, `T2`, `direction` | as above |
| `n_tips` | tips where both traits are known |
| `n_both`, `n_T1_only`, `n_T2_only`, `n_neither` | the 2×2 table of those tips |
| `prevalence_T1`, `prevalence_T2` | trait prevalences |
| `weighted_lor` | observed log odds ratio computed with phylogenetic tip weights (section 10) |
| `null_mean`, `null_sd` | mean and SD of that statistic under the null |
| `effect_lor` | `weighted_lor − null_mean`: association beyond phylogeny, in log-odds units |
| `effect_z` | `effect_lor / null_sd`, signed (a z-like score) |
| `effect_shrunk` | `effect_lor` shrunk toward 0 by empirical Bayes; noisy estimates shrink most |
| `lfsr` | local false sign rate of the association |
| `pval_naive` | raw two-sided p-value |
| `pval_bh`, `pval_by`, `pval_bonf` | BH and BY q-values, and the Bonferroni-adjusted p-value |
| `refined` | `True` if the p-value comes from the re-test with `--refine-n` replicates |
| `n_null` | null replicates per trait behind the p-value |
| `p_region`, `gated` | with `--composite` only: the p-value under the confined-region null, and whether it was the larger |
| `null_calibrated` | see above |
| `confined_T1`, `confined_T2`, `co_confined` | lineage confinement of each trait and of the pair (section 7.4) |
| `changes_T1`, `changes_T2` | estimated number of gains + losses of each trait on the tree |

### 5.3 `simphyni_traits`

| column | meaning |
|---|---|
| `trait` | trait name |
| `n_tips`, `n_present`, `prevalence` | tips with a known value, tips carrying the trait, and their ratio |
| `est_gains`, `est_losses` | expected numbers of gains and losses from the ancestral reconstruction |
| `root_state` | reconstructed state at the root (0/1) |
| `confined`, `cover_frac`, `cover_clades` | whether the trait's minority-state tips fit in ≤ 2 clades holding ≤ 35 % of tips; the share of tips in that cover; the number of clades |
| `null_calibrated` | see above |
| `n_pairs_tested` | pairs involving the trait |
| `n_significant_pos`, `n_significant_neg` | partners with `pval_bh < 0.05`, by direction |

Traits are sorted by their number of significant partners.

---

## 6. Plots

With `--plot`, each run writes:

| file | content |
|---|---|
| `simphyni_volcano.png` | every pair: shrunken effect (x) against −log10 q (y). Significant pairs are coloured (red = positive, blue = negative), co-confined pairs are outlined, and the bulk of pairs is shown as density. |
| `simphyni_heatmap.png` | the 40 traits with the most significant partners, clustered. Colour = shrunken effect where q < 0.05 (blank otherwise); `*` marks q < 0.05 and `**` q < 0.005. All-pairs runs only. |
| `simphyni_pvalues.png` | histogram of raw p-values (section 7.6) |
| `simphyni_top_traits.png` | traits with the most significant partners (right = positive, left = negative) |
| `volcano_<trait>.png` | one per query trait, in `--run-traits` runs |

---

## 7. Interpreting results

### 7.1 Selecting and ranking

1. **Select** pairs with `pval_bh < 0.05`. Tests that share a trait are positively dependent, the
   setting in which the Benjamini–Hochberg procedure controls the false discovery rate. Use
   `pval_by` when you need a guarantee under any form of dependence, or for small analyses built
   from lineage-confined traits (section 7.4).
2. **Rank** the selected pairs by `|effect_shrunk|`. Shrinkage pulls noisy estimates (e.g. rare traits
   or traits with few changes) toward zero, so the top of the list is dominated by well-supported
   strong associations rather than by chance extremes.
3. **Check the sign.** `lfsr < 0.05` means the direction (positive vs negative) is well supported.

### 7.2 What an association means

- **Positive:** the two traits are found together more often than their histories on the tree
  predict. Common causes are functional dependence, co-transfer (genes moving together on a
  plasmid, island or phage) and shared selection.
- **Negative:** the traits avoid each other, e.g. through functional redundancy, incompatibility or
  exclusion.
- **Association is not causation.** SimPhyNI does not tell which trait came first, or whether one
  causes the other. Treat significant pairs as hypotheses for follow-up. For confirmed pairs, a
  Pagel-type correlated-evolution test can address the order of events.

### 7.3 Power and genome-wide analyses

- **Genome-wide thresholds are strict.** In an all-against-all analysis, the number of tests grows with
  the square of the number of traits. A pangenome of a few thousand genes gives millions of pairs, and
  the corrected significance threshold becomes very strict. Associations that are real but moderate
  can then fail to reach significance even though they rank highly.
- **Use genome-wide runs to rank and to find strong associations.** Order candidates by
  `effect_shrunk` and look for hubs in `simphyni_traits`. Then test specific hypotheses on a focused
  set of traits (`--run-traits`, or a smaller trait table), where far fewer tests are corrected for.
- **Power depends on how often the traits changed.** Traits that arose once, in a single clade, carry
  little information about association: their co-occurrence is explained by one ancestral event.
  Associations are best supported when traits changed many times in different parts of the tree.
- **Power also depends on the tree.** Few genomes, or very clonal trees with little diversity, limit
  what can be detected.

### 7.4 Lineage-confined traits and `co_confined`

- **Confined trait:** the tips carrying its minority state fit into at most two clades holding ≤ 35 %
  of tips, e.g. a phylogroup-specific gene or a gene on a lineage-specific plasmid.
- **Co-confined pair:** both traits are confined to overlapping parts of the tree.

For a co-confined pair, presence/absence data alone cannot separate three explanations for a
positive association:

1. the genes interact;
2. they are transferred together (same island, plasmid or phage);
3. they are independently restricted to the same lineage (e.g. by niche or host range), which is a
   confounder rather than an association.

The flag is descriptive: it does not change p-values. How much it matters depends on the analysis.

- **Genome-wide or broad analyses.** Confined pairs are a small share of all pairs, and the corrected
  threshold is strict. Lineage confinement is rarely the explanation for a significant call.
- **Small analyses built mostly from confined traits** (e.g. a few dozen plasmid-borne genes).
  Thresholds are looser, and a substantial share of positive calls may reflect shared lineage or
  co-transfer. Use `pval_by` for such analyses.

**Recommended practice:**

- Report positive co-confined associations as "shared lineage or co-transfer" unless other evidence
  supports an interaction.
- Use genomic context: gene coordinates, plasmid or island membership, and annotation often show
  whether co-confined genes travel together.
- `--composite` adds a second null in which both traits are confined to their clades, and reports the
  larger p-value. It is more conservative for co-confined pairs but costs power. It is optional and
  not needed for genome-wide analyses.

### 7.5 `null_calibrated = False`

These pairs involve traits with very few estimated changes and highly asymmetric eligible regions,
for which the simulated null may be miscalibrated. They are tested normally and flagged. Treat them
with more caution, or drop them with `--no-include-flagged`.

### 7.6 Checking a run

- **Look at `simphyni_pvalues.png`.** For a well-behaved run the histogram is roughly flat, with a peak
  near 0 when there is signal. The first bin can sit slightly below its neighbours: pairs with small
  p-values are re-tested with fresh replicates, and those that were small by chance move up.
- **Warning signs:** a U shape or a rise toward 1 usually points to an input problem (section 14).
- **Hubs:** traits with very many significant partners (`simphyni_traits`) are worth inspecting. They
  are often mobile elements, lineage markers or data artefacts.
- **Monte-Carlo stability:** re-run with a different `--seed`. Strong associations should reproduce;
  borderline ones may not.

---

## 8. Recommended workflows

**Genome-wide screen, then focused tests:**

```bash
# 1. screen all pairs
simphyni run -T pangenome.parquet -t tree.nwk -s screen -o out --output-format parquet --plot --seed 1

# 2. focused test of candidate traits (put them first in the table) against all others
simphyni run -T candidates_first.parquet -t tree.nwk -s focus -r 0,1,2,3 -o out --plot --seed 2
```

- **Step 1:** rank by `effect_shrunk` and take hubs from `simphyni_traits`.
- **Step 2:** select `pval_bh < 0.05` (use `pval_by` if the set is mostly co-confined traits) and order
  by `effect_shrunk`.
- **Stability:** confirm key results with a different `--seed`.

**Genotype–phenotype association:**

```bash
simphyni run -T genes_plus_phenotype.csv -t tree.nwk -r 0 -o out --plot     # phenotype in column 0
```

Read `volcano_<phenotype>.png`, and the compact table filtered to pairs that contain the phenotype.

---

## 9. Python API

```python
from simphyni import TreeSimulator
from simphyni.Simulation.simulation import build_sim_params
import pandas as pd

params = build_sim_params(pd.read_csv("acr.csv"), counting="FLOW", subsize="ORIGINAL")
sim = TreeSimulator(tree="tree.nwk", pastmlfile=params, obsdatafile="traits.csv")
sim.initialize_simulation_parameters(pre_filter=False)   # run_traits=k: first k traits vs all
sim.run_simulation(cores=8, seed=1)                       # default null
long, compact, traits = sim.build_outputs()               # DataFrames (section 5)
sim.write_outputs("out/")                                 # writes the three tables

from simphyni.plotting import summary_plots
summary_plots(long, traits, "out/")
```

`acr.csv` comes from `simphyni/scripts/run_ancestral_reconstruction.py`; the command-line pipeline runs
it for you.

---

## 10. How the method works

1. **Ancestral reconstruction.** For every trait, a two-state model on the tree gives the expected
   numbers of gains and losses, the root state and per-branch state probabilities. Branch lengths
   are capped at an outlier threshold, so that a few very long branches do not dominate.
2. **Simulation.** Each trait is simulated many times (512 by default) on the tree from its own gain
   and loss rates, with exact per-branch transition probabilities.
3. **Statistic.**
   - The association is the log odds ratio of the 2×2 table, with each tip weighted by its share of
     the tree's branch length (Gerstein–Sonnhammer–Chothia weights).
   - A clade inherited from one gain therefore counts about once rather than once per tip. This
     removes most of the pseudo-replication that inflates tests ignoring the phylogeny.
4. **Anchored null.**
   - The observed trait A is scored against B's simulated replicates, and B against A's.
   - Each comparison only requires the *partner's* history to be modelled correctly. The larger of the
     two p-values is reported, so the test remains valid if either trait's model is right.
5. **Tail probabilities.** p-values are exact counts of how many null values are at least as extreme as
   the observed one. Beyond the most extreme replicate, a generalized Pareto distribution fitted to
   the top of the null gives conservative tail probabilities, so p-values are never reported as zero.
6. **Re-test.** Pairs with small p-values are re-tested with many more fresh replicates, which gives
   accurate p-values at strict thresholds.
7. **Multiple testing and effects.**
   - BH, BY and Bonferroni corrections are applied across all pairs.
   - The effect estimates and their null SDs go into an adaptive-shrinkage model fitted across all
     pairs, giving `effect_shrunk` and `lfsr`.

Missing values are handled pairwise: for each pair, tips missing either trait are left out of the
observed statistic, the null and the 2×2 counts.

For performance comparisons with other tools, and how they change across releases, see
**[mgwas-bench](https://jpeyemi.github.io/mgwas-bench/)**.

---

## 11. Computational resources

- **Scaling.** Run time and memory grow with the number of pairs tested and the number of tips.
  - Ancestral reconstruction and simulation take a small fraction of the time.
  - Scoring pairs and re-testing the smallest p-values take most of it.
- **Typical runs.** A pangenome of a few thousand genes on a few hundred genomes (millions of pairs)
  runs in minutes on a laptop and needs a few GB of memory. Smaller analyses take seconds.
- **For large runs:**
  - use Parquet input and `--output-format parquet` (a long results table with millions of rows is
    several GB as CSV);
  - avoid `--save-object`;
  - use `--run-traits` when only some traits are of interest.
- **Other settings:**
  - `--composite` adds a second null for candidate pairs and can increase run time severalfold on
    large runs.
  - The legacy null (`--null legacy`) is much slower than the default at large scale.

---

## 12. Advanced options

These options are accepted by `simphyni run` but hidden from `--help`. They exist to reproduce earlier
analyses and for method development; most users should not need them.

| option | purpose |
|---|---|
| `--null legacy` with `--condition prevalence\|footprint\|none`, `--oversample K`, `--emergence`, `--eligibility` | the SimPhyNI ≤ 2.0 null (unweighted statistic, KDE tails). Use only to reproduce results from earlier versions; its tail probabilities are less reliable at strict, genome-wide thresholds. |
| `--lineage soft_auto\|soft\|mrca` | per-trait lineage restriction in the null. Not recommended: when confinement reflects real restriction it can create false positives. |
| `--branch-cap none` | simulate on uncapped branch lengths |
| `--combine mean`, `--null anchored`, `--gate-p` | experimental variants of the anchored test |

---

## 13. Reporting and citing

When you publish results from SimPhyNI, report:

- the SimPhyNI version (`simphyni version`);
- the options you changed from the defaults, and the seed;
- the multiple-testing criterion (e.g. BH q < 0.05) and how associations were ranked
  (e.g. `effect_shrunk`);
- for pangenome analyses, how the trait table was built (annotation, clustering and prevalence
  filters) and how the tree was inferred and rooted.

Cite the SimPhyNI paper (see the repository README). For comparisons with other methods, refer to the
current results on [mgwas-bench](https://jpeyemi.github.io/mgwas-bench/), which records the
SimPhyNI version each result was produced with.

---

## 14. Troubleshooting and FAQ

**No significant pairs.**

- If `simphyni_pvalues.png` is flat, there may be no association beyond phylogeny, or the analysis may
  include too many tests for the signal present (section 7.3). Rank by `effect_shrunk`, or test a
  focused set of traits.
- If the histogram is skewed toward 1, many traits may be nearly fixed or extremely rare. Tighten
  `--min-prev` / `--max-prev`.

**Many significant pairs among a few traits.** Look at `simphyni_traits`. Hubs that are `confined` often
reflect a lineage-specific mobile element (section 7.4).

**Tip names do not match.** The tree's tip labels and the trait table's first column must be identical
strings (watch for suffixes, quotes and whitespace).

**Unrooted tree.** Root it first, e.g. by midpoint rooting (ete3, R `phytools::midpoint.root`) or with an
outgroup.

**Memory.** Use Parquet input and output, and `--run-traits` for one-versus-rest designs; avoid
`--save-object` for large runs.

**Reproducibility.** Use `--seed`; results are independent of `--cores`. Different seeds give slightly
different p-values because the null is simulated. Strong associations are stable.
