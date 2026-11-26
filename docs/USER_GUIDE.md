# SimPhyNI user guide

SimPhyNI tests pairs of binary traits (gene presence/absence, alleles, phenotypes) for
**co-occurrence or avoidance beyond what the phylogeny explains**. Related genomes share traits by
descent, so a naive test of co-occurrence (e.g. Fisher's exact test) calls thousands of false
associations. SimPhyNI builds each trait's null distribution by simulating the trait on the tree
from its own estimated gain/loss history, then asks whether the observed pair is unusual.

- [1. Installation](#1-installation)
- [2. Inputs](#2-inputs)
- [3. Running SimPhyNI](#3-running-simphyni)
- [4. Command-line options](#4-command-line-options)
- [5. Outputs](#5-outputs)
- [6. Plots](#6-plots)
- [7. Interpreting results](#7-interpreting-results)
- [8. Recommended workflows](#8-recommended-workflows)
- [9. Python API](#9-python-api)
- [10. How the method works](#10-how-the-method-works)
- [11. Performance and resources](#11-performance-and-resources)
- [12. Advanced options](#12-advanced-options)
- [13. Troubleshooting and FAQ](#13-troubleshooting-and-faq)

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

Requirements: Python 3.11–3.12, numpy, pandas, numba, scipy, statsmodels, ete3, matplotlib, pyarrow and
snakemake (all installed by the conda package).

Try the bundled example:

```bash
simphyni download-examples
simphyni run --samples example_inputs/simphyni_sample_info.csv --cores 8 --plot
```

---

## 2. Inputs

### Phylogenetic tree (`.nwk`)

- Newick format, **rooted** (outgroup or midpoint rooting are both fine).
- **Branch lengths are required.** They set how likely changes are on each branch.
- Tip names must match the row names of the trait table. Tips missing from the table are ignored;
  rows missing from the tree are dropped.
- Zero-length and very short branches (clonal clusters) are supported.

### Trait table (`.csv` or `.parquet`)

- **Rows:** genomes, with the first column holding names that match the tree tips.
- **Columns:** binary traits; the header row holds trait names.
- **Values:** 1 = present, 0 = absent. Values above 0.5 count as present, blanks as absent.

```csv
Sample,PhenotypeX,GeneA,GeneB
E_coli_1,1,0,1
E_coli_2,1,1,0
E_coli_3,0,0,1
```

For more than about 10,000 traits use Parquet. It loads only the columns and rows needed, which cuts
memory use substantially.

Traits outside `--min-prev`/`--max-prev` (default 5–95 % of tips) are dropped. Very rare or near-fixed
traits carry little information and have poorly estimated histories.

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

Results go to `<outdir>/<sample>/`. Use `--seed` for reproducible results. Results do not depend on
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
| `--output-format` | `csv` | `csv` or `parquet` (much smaller and faster for > 1 M pairs) |
| `--plot` | off | write the summary plots (section 6) |
| `--save-object` | off | also save the Python analysis object (`.pkl`); avoid for > 1 M pairs |
| `--temp-dir` | `<outdir>/tmp` | intermediate files (ancestral reconstruction, reformatted inputs) |

### Statistics

| option | default | description |
|---|---|---|
| `--null` | `default` | `default` = phylogenetically weighted anchored null with calibrated tails; `legacy` = the SimPhyNI ≤ 2.0 null |
| `--refine-p` | 0.01 | pairs with p below this are re-tested with `--refine-n` fresh replicates (0 disables) |
| `--refine-max-pairs` | 5000 | at most this many pairs (the smallest p) are re-tested per run |
| `--refine-n` | 32768 | replicates per trait in the re-test |
| `--trials` | 512 | null replicates per trait in the main test |
| `--composite` | off | optional extra null that confines both traits to their clades (section 7.4) |
| `--include-flagged` | on | keep pairs involving traits whose null may be miscalibrated (they are marked `null_calibrated = False`) |
| `--seed` | none | random seed |

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
| `effect_shrunk` | association strength in log-odds units, after empirical-Bayes shrinkage; **use to rank** |
| `lfsr` | local false sign rate: probability that the sign of the association is wrong |
| `pval_bh` | Benjamini–Hochberg q-value; **use to select** (q < 0.05) |
| `pval_by` | Benjamini–Yekutieli q-value; stricter, valid under any dependence |
| `prevalence_T1`, `prevalence_T2` | fraction of tips carrying each trait |
| `n_both` | tips carrying both traits |
| `co_confined` | both traits are confined to the same small part of the tree (section 7.4) |
| `null_calibrated` | `False` if a trait's null may be miscalibrated (very few changes with unusual structure) |

### 5.2 `simphyni_results_long` (all fields)

| column | meaning |
|---|---|
| `T1`, `T2`, `direction` | as above |
| `n_tips` | tips tested |
| `n_both`, `n_T1_only`, `n_T2_only`, `n_neither` | the 2×2 table of tips |
| `prevalence_T1`, `prevalence_T2` | trait prevalences |
| `weighted_lor` | observed log odds ratio computed with phylogenetic tip weights (section 10) |
| `null_mean`, `null_sd` | mean and SD of that statistic under the null |
| `effect_lor` | `weighted_lor − null_mean`: association beyond phylogeny, in log-odds units |
| `effect_z` | `effect_lor / null_sd`, signed (a z-like score) |
| `effect_shrunk` | `effect_lor` shrunk toward 0 by empirical Bayes; noisy estimates shrink most |
| `lfsr` | local false sign rate of the association |
| `pval_naive` | raw p-value (two-sided) |
| `pval_bh`, `pval_by`, `pval_bonf` | BH and BY q-values, and the Bonferroni-adjusted p-value |
| `refined` | `True` if the p-value comes from the re-test with `--refine-n` replicates |
| `n_null` | null replicates per trait behind the p-value |
| `p_region`, `gated` | only with `--composite`: p under the confined-region null, and whether it was the larger |
| `null_calibrated` | see above |
| `confined_T1`, `confined_T2`, `co_confined` | lineage confinement of each trait and of the pair (section 7.4) |
| `changes_T1`, `changes_T2` | estimated number of gains + losses of each trait on the tree |

### 5.3 `simphyni_traits`

| column | meaning |
|---|---|
| `trait` | trait name |
| `n_tips`, `n_present`, `prevalence` | tips tested and tips carrying the trait |
| `est_gains`, `est_losses` | expected numbers of gains and losses from ancestral reconstruction |
| `root_state` | reconstructed state at the root (0/1) |
| `confined`, `cover_frac`, `cover_clades` | whether the trait's minority-state tips fit in ≤ 2 clades holding ≤ 35 % of tips, the share of tips in that cover and the number of clades |
| `null_calibrated` | see above |
| `n_pairs_tested` | pairs involving the trait |
| `n_significant_pos`, `n_significant_neg` | partners with `pval_bh < 0.05`, by direction |

Traits are sorted by number of significant partners.

---

## 6. Plots

With `--plot`, each run writes:

| file | content |
|---|---|
| `simphyni_volcano.png` | every pair: shrunken effect (x) vs −log10 q (y). Significant pairs are coloured (red = positive, blue = negative), co-confined pairs are outlined, the bulk is shown as density. |
| `simphyni_heatmap.png` | the 40 traits with the most significant partners, clustered. Colour = shrunken effect where q < 0.05 (blank otherwise); `*` q < 0.05, `**` q < 0.005. All-pairs runs only. |
| `simphyni_pvalues.png` | histogram of raw p-values. It should be roughly flat, with a peak near 0 when there is signal. A U shape or a rise toward 1 suggests a problem (section 13). The first bin can sit slightly below its neighbours: pairs with p < 0.01 are re-tested with fresh replicates, and null pairs that were small by chance move up. |
| `simphyni_top_traits.png` | traits with the most significant partners (right = positive, left = negative) |
| `volcano_<trait>.png` | per query trait, in `--run-traits` runs |

---

## 7. Interpreting results

### 7.1 Selecting and ranking

1. **Select** pairs with `pval_bh < 0.05`.
   - On benchmarks the observed false discovery proportion was 0–2.7 % at this level, against a
     nominal 5 %.
   - Pairs share traits and are positively dependent, the condition under which BH controls the FDR.
   - Use `pval_by` when you want a guarantee under any dependence, or for small analyses built from
     lineage-confined genes (section 7.4).
2. **Rank** the selected pairs by `|effect_shrunk|`. It ranks true associations better than p-values
   or raw effect sizes, and shrinks noisy estimates (e.g. rare traits) toward zero.
3. **Sign:** `lfsr < 0.05` means the direction (positive vs negative) is well supported.

### 7.2 What an association means

- **Positive:** the two traits are found together more often than their histories on the tree
  predict. Typical causes are functional dependence, co-transfer (genes moving together on a plasmid,
  island or phage) and shared selection.
- **Negative:** the traits avoid each other, e.g. functional redundancy, incompatibility or
  exclusion.
- **Not causal direction:** SimPhyNI does not tell which trait came first. For confirmed pairs, a
  Pagel-type correlated-evolution test can address ordering.

### 7.3 Genome-wide significance and power

With millions of pairs, the BH threshold is very small (around 10⁻⁷–10⁻⁹). On the *E. coli*
benchmark, many planted interactions of moderate strength had true p-values of 10⁻³–10⁻⁵, too weak to
be genome-wide significant under any correct test. Genome-wide runs are therefore best used to
**rank** candidates (by `effect_shrunk`) and to find strong associations. Test specific hypotheses on
focused sets of traits, where the multiple-testing burden is smaller (section 8).

### 7.4 Lineage-confined traits and `co_confined`

**Definitions.**

- A trait is **confined** when the tips carrying its minority state fit into at most two clades
  holding ≤ 35 % of tips, e.g. a phylogroup-specific gene or one on a lineage-specific plasmid.
- A pair is **co_confined** when both traits are confined to overlapping parts of the tree.

**Why it matters.** For co-confined pairs, a positive association has three possible explanations,
which presence/absence data cannot separate:

1. the genes interact;
2. they are transferred together (same island, plasmid or phage);
3. they are independently restricted to the same lineage (e.g. by niche or compatibility), which is
   a confounder rather than an association.

**How much it matters.** On replicas of a real *E. coli* pangenome in which every confined gene was
treated as independently lineage-restricted:

- **Genome-wide and random-subset analyses made no false calls.**
- **Small analyses built mostly from confined genes** (e.g. 30 plasmid-borne genes) had up to about
  half of BH calls false in that worst case, and about 22 % with BY.

**Recommendations.**

- Read positive co-confined associations as "shared lineage or co-transfer" unless other evidence
  (genomic location, function) points to interaction.
- Use `pval_by` for small analyses dominated by confined genes.
- Gene coordinates or plasmid membership help: adjacent genes or genes on the same element point to
  co-transfer.
- `--composite` adds a null in which both traits are confined to their clades and reports the larger
  p. It reduced false positives for traits confined to random clades in benchmarks, but not in the
  realistic replica, and it costs power. It is optional.

The flag is descriptive. It does not change p-values.

### 7.5 `null_calibrated = False`

These pairs involve traits with very few estimated changes and highly asymmetric eligible regions,
for which the simulated null may be miscalibrated. They are tested normally and flagged; treat them
with more caution, or drop them with `--no-include-flagged`.

---

## 8. Recommended workflows

**Genome-wide screen, then focused tests:**

```bash
# 1. screen all pairs
simphyni run -T pangenome.parquet -t tree.nwk -s screen -o out --output-format parquet --plot --seed 1

# 2. focused test of candidate traits (put them first in the table) against all others
simphyni run -T candidates_first.parquet -t tree.nwk -s focus -r 0,1,2,3 -o out --plot --seed 2
```

- In step 1, rank by `effect_shrunk` and take hubs from `simphyni_traits`.
- In step 2, report `pval_bh < 0.05` (`pval_by` if the set is mostly co-confined genes) and order by
  `effect_shrunk`.
- Check stability by re-running with a different `--seed`; strong hits should reproduce.

**Phenotype association (genotype–phenotype):**

```bash
simphyni run -T genes_plus_phenotype.csv -t tree.nwk -r 0 -o out --plot     # phenotype in column 0
```

Read `volcano_<phenotype>.png` and the compact table filtered to pairs containing the phenotype.

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

`acr.csv` comes from `simphyni/scripts/run_ancestral_reconstruction.py` (the pipeline runs it for you).

---

## 10. How the method works

1. **Ancestral reconstruction.** For every trait, a two-state model on the tree gives expected
   numbers of gains and losses (`est_gains`, `est_losses`), the root state and per-branch state
   probabilities. Branch lengths are capped at an outlier threshold so that a few very long branches
   do not dominate.
2. **Simulation.** Each trait is simulated 512 times on the tree from its own gain and loss rates,
   using exact per-branch transition probabilities. Simulation is bit-parallel and takes about a
   second for thousands of traits.
3. **Statistic.**
   - The association is the log odds ratio of the 2×2 table, with every tip weighted by its share of
     the tree's branch length (Gerstein–Sonnhammer–Chothia weights).
   - A clade inherited from one gain therefore counts about once instead of once per tip. This
     removes most of the pseudo-replication that inflates naive tests.
4. **Anchored null.**
   - The observed trait A is scored against B's 512 simulated replicates, and B against A's.
   - Each comparison is an exact conditional test that only needs the *partner's* model to be right.
     The larger of the two p-values is reported, which is valid if either trait's model is correct.
   - No replicates are discarded or selected.
5. **Tails.** p-values come from exact counts of how many null values are at least as extreme. Beyond
   the most extreme of the 512 replicates, a generalized Pareto distribution is fitted to the top of
   the null (with a conservative shape), so p-values are never extrapolated to zero.
6. **Re-test.** Pairs with p < 0.01 (up to 5,000) are re-tested with 32,768 fresh replicates per trait,
   giving accurate p-values down to about 10⁻⁸.
7. **Multiple testing.** BH, BY and Bonferroni are applied across all pairs.
8. **Effect sizes.** `effect_lor` and its null SD go into an adaptive-shrinkage model fitted across all
   pairs, giving `effect_shrunk` and `lfsr`.

**Validation** (`benchmarks_large/REPORT.md`, `candidates/README.md`):

- 11 PanX species trees with synthetic, real-gene, late-emerging and lineage-restricted traits;
- a tree-swap test on 36 trees, including clonal species;
- an *E. coli* genome-scale set of 8.4 M known-truth pairs, with p-values checked against exact
  4-million-replicate Monte Carlo;
- comparison with Fisher, FaST-LMM (pyseer `--lmm`) and treeWAS.

---

## 11. Performance and resources

| dataset | pairs | time (8 cores) | peak memory |
|---|---|---|---|
| 150 traits, 125 tips | 11,000 | ~15 s end to end | < 1 GB |
| 1,000 genes, 250 tips | 0.5 M | ~1 min | ~2 GB |
| 4,501 genes, 500 tips (*E. coli*) | 10.1 M | ~5 min (`--composite`: +5–25 min) | 4.5 GB |

The legacy null takes about 50 minutes on the *E. coli* example. Use `--output-format parquet` for
large runs: a 10 M-row long table is several GB as CSV.

---

## 12. Advanced options

These are accepted by `simphyni run` but hidden from `--help`. They exist for reproducing earlier
analyses and for method development.

| option | purpose |
|---|---|
| `--null legacy` with `--condition prevalence\|footprint\|none`, `--oversample K`, `--emergence`, `--eligibility` | the SimPhyNI ≤ 2.0 null: unweighted statistic, replicate-vs-replicate pairs, KDE tails. Anticonservative below ~10⁻⁵ at genome scale. |
| `--lineage soft_auto\|soft\|mrca` | per-trait lineage restriction in the null. **Not recommended:** `soft_auto` created false positives in realistic replicas. |
| `--branch-cap none` | simulate on uncapped branch lengths |
| `--combine mean`, `--null anchored`, `--gate-p` | experimental variants of the anchored test |

---

## 13. Troubleshooting and FAQ

**No significant pairs.** Check `simphyni_pvalues.png`.

- If it is flat, there may be no association beyond phylogeny, or the run may be too large for the
  signal (section 7.3). Rank by `effect_shrunk`, or test a focused set of traits.
- If it is skewed toward 1, many traits may be nearly fixed or extremely rare. Tighten
  `--min-prev`/`--max-prev`.

**Many significant pairs among a few traits.** Look at `simphyni_traits`. Hubs that are `confined` may
reflect a lineage-specific mobile element (section 7.4).

**Tip names do not match.** The tree's tip labels and the trait table's first column must be
identical strings. Traits are dropped if fewer than two tips remain.

**Unrooted tree.** Root it first (e.g. midpoint rooting in ete3, R `phytools::midpoint.root`, or an
outgroup).

**Memory.** Use Parquet input and output, and `--run-traits` for one-vs-rest designs. Avoid
`--save-object` for large runs.

**Reproducibility.** Use `--seed`. Results are independent of `--cores`. Different seeds give slightly
different p-values (Monte-Carlo noise). Strong associations are stable.
