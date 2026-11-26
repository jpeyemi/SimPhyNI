# Changelog

## 3.0.0rc1 (release candidate for 3.0.0)

Major release: the default null model, the result files and parts of the CLI changed, so results and
downstream scripts written for 2.x need updating. The full documentation is in
[docs/USER_GUIDE.md](docs/USER_GUIDE.md).

### Statistical method (changes results)

- **Weighted statistic.** The association statistic is a log odds ratio with phylogenetic tip weights
  (Gerstein–Sonnhammer–Chothia). A clade inherited from one gain counts about once rather than once
  per tip.
- **Anchored null.** Each observed trait is scored against its partner's simulated replicates in both
  directions, and the larger p is reported. That p is valid if either trait's model is right. No
  replicates are discarded: prevalence/footprint replicate selection is gone.
- **Exact simulator.** Each branch uses the exact two-state transition with any number of changes.
  The previous model allowed at most one change per branch.
- **Calibrated tails.** p-values come from exact Monte-Carlo counts. Beyond the most extreme replicate
  a generalized Pareto tail is used, so p-values are never extrapolated to zero. The Gaussian-KDE
  tail of 2.x was badly anticonservative below ~10⁻⁵ at genome scale.
- **Refinement.** Pairs with p < 0.01 (at most 5,000 per run) are re-tested with 32,768 fresh
  replicates.
- **Effect sizes.** New empirical-Bayes shrunken effect sizes (`effect_shrunk`) and local false sign
  rates (`lfsr`).
- **Benchmark results** (11 PanX species, BH q < 0.05):
  - synthetic recall 0.36 (2.0.0: about 0.24) with 0 false discoveries;
  - at genome scale (*E. coli*, 10.1 M pairs), no false calls among 8.3 M known-null pairs;
  - about 10× faster.

### Outputs (breaking)

- **Three tables** in csv or parquet (`--output-format`):
  - `simphyni_results`: compact columns `T1`, `T2`, `direction`, `effect_shrunk`, `lfsr`, `pval_bh`,
    `pval_by`, `prevalence_T1`, `prevalence_T2`, `n_both`, `co_confined`, `null_calibrated`;
  - `simphyni_results_long`: every field, including the 2×2 tip counts, weighted log odds ratio, null
    mean/SD, `effect_lor`, `effect_z`, `pval_naive` and `pval_bonf`, refinement, confinement and
    estimated changes;
  - `simphyni_traits`: one row per trait.
- **Column moves and renames.** `effect size` (unsigned) is replaced by `effect_z` (signed) in the long
  table. `pval_naive` and `pval_bonf` moved to the long table.
- **Sorting.** Rows are sorted by `pval_bh`, then by |`effect_shrunk`|.
- **New `co_confined` flag:** both traits confined to the same small part of the tree. It is
  descriptive, so p-values are unchanged.
- **New plots** (`--plot`): genome-scale volcano, clustered heatmap of the most-connected traits,
  p-value histogram, top traits, and per-query volcanoes. They replace the per-correction heatmaps.

### Missing values

- Blank/`NA` trait values are kept as missing (from the dev-branch NA support). The anchored test and
  the output tables use pairwise-complete data: tips missing either trait are left out of that pair's
  observed statistic, null and 2×2 counts.

### Command line

- **`--null`.** `--null default|legacy`; `legacy` reproduces the 2.x null.
- **New options:** `--output-format`, `--refine-p`, `--refine-max-pairs`, `--refine-n`, `--trials`,
  `--composite`. `--min-prev`/`--max-prev` now also accept the dashed form; the underscore form still
  works.
- **Hidden options.** Legacy-null and developer options (`--condition`, `--oversample`, `--emergence`,
  `--eligibility`, `--lineage`, …) are still accepted but hidden from `--help`.
- **Version.** `simphyni.__version__` is now defined (the conda recipe test relied on it); the version
  is set in `simphyni/_version.py`.

### Performance

- **Ancestral-reconstruction statistics** are vectorised, and Numba kernels are cached (about 2× faster).
- **Simulation and scoring** use fused Numba kernels and blocked BLAS: about 36,000 pairs/s for the
  screen.

### Migrating from 2.x

- **Selecting pairs:** use `pval_bh < 0.05` in `simphyni_results` (or `pval_by` for small analyses of
  lineage-confined genes).
- **Ranking:** use `effect_shrunk` instead of `effect size`.
- **Raw p-values** (`pval_naive`) are in `simphyni_results_long`.
- **Reproducing 2.x results:** use `--null legacy`.
