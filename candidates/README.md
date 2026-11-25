# SimPhyNI null-model candidates (branch `explore/uncertainty`)

## Current default

`run_simulation()` with no options, or `simphyni run` with no null-model flags, runs:

- **Statistic:** log odds ratio on Gerstein–Sonnhammer–Chothia phylogenetic tip weights. A clade
  inherited from one gain counts once rather than once per tip.
- **Anchored null:** each observed trait is scored against its partner's 512 null replicates, in
  both directions, and the larger p is reported. That p is valid if either trait's model is right.
  No replicates are discarded, so there is no prevalence or footprint selection.
- **Exact simulator:** the two-state transition over each branch, with any number of changes.
- **Calibrated tails:** a generalized-Pareto tail, fitted to the top 10 % of each null with shape
  ≥ 0, replaces KDE extrapolation. Pairs with p < 0.01 (at most 5,000 per run) are re-tested with
  32,768 fresh replicates and a GPD fitted to the top 1 %.
- **Effect sizes:** `effect_lor` (observed − null mean, in log-odds units), `effect_se`,
  `effect_shrunk` (empirical-Bayes posterior mean under an adaptive-shrinkage prior over all pairs)
  and `lfsr` (local false sign rate).

`--null legacy` restores the previous default (unweighted, prevalence-conditioned, KDE tails).

### Recommended reading of results

1. **Select** pairs with `pval_bh < 0.05`. The pair tests are plausibly positively dependent
   (PRDS), and BH controlled the FDR in every benchmark scenario except lineage restriction (below).
   `pval_by` is the strict alternative.
2. **Order** the selected pairs by `effect_shrunk`. On the benchmark it ranks best, ahead of
   |effect| and p.
3. **Lineage-restricted genes** (plasmid, phage, phylogroup-specific): add
   `--lineage soft_auto --composite` and read results as "association beyond shared confinement"
   (see [LINEAGE.md](LINEAGE.md)).

## Benchmark (PanX full tier: 11 species, ≤250 tips, 3 replicates)

Each cell is recall / observed false discovery proportion (FDP) at BH q < 0.05, per scenario.
Generated with `benchmarks_large/report_panx.py`; the full table is in
`benchmarks_large/report_full_tier.md`.

| configuration | synthetic | late-emerging | restricted | null ratio @1e-4 | AP by p / by shrunken effect |
|---|---|---|---|---|---|
| **default (GPD tail + refinement)** | **0.37 / 0** | **0.51 / 0.014** | 0.36 / 0.59 | 3.9* | 0.841 / **0.857** |
| default + `--lineage soft_auto --composite` | 0.33 / 0 | 0.45 / 0.010 | 0.18 / 0.45 | 4.6* | 0.834 / 0.851 |
| anchored, KDE tail (earlier default) | 0.32 / 0 | 0.47 / 0.011 | 0.29 / 0.57 | 1.8 | 0.841 / — |
| legacy (unweighted, prevalence ×8) | 0.24 / 0.004 | 0.38 / 0.009 | 0.11 / 0.54 | 1.9 | 0.806 / — |
| FaST-LMM | 0.27 / 0.45 | 0.36 / 0.32 | 0.51 / 0.65 | 143 | 0.442 / — |
| Fisher (tips) | 0.77 / 0.65 | 0.81 / 0.52 | 0.52 / 0.81 | 1,243 | 0.590 / — |

\* The null ratio at 1e-4 is driven by separately lineage-restricted null traits, which no model fits
(see Limits). Among 8.3 M unrestricted null pairs on *E. coli*, the default made **0** false calls.

## Genome scale (`benchmarks_large/REPORT.md`)

- **Speed:** *E. coli*, 500 tips, 4,501 genes, **10.1 M pairs**: about 5 min end to end with refinement (4.7 GB peak); about 11 min with lineage control.
  The legacy null would need about 50 min.
- **Calibration:** on 8.4 M known-truth pairs, KDE tails were 15× nominal at 1e-6 (51 % of BY calls
  false). The legacy null was 82× nominal at 1e-6. The GPD tail made no false calls, and its
  p-values track exact 4.2 M-replicate Monte-Carlo.
- **Power:** most planted interactions have true p of 1e-2 to 1e-5, so they are not genome-wide
  significant under the correct null. Use genome-wide runs to rank (by `effect_shrunk`), then test
  hypotheses on focused trait sets, where the multiple-testing burden is smaller.

## Workflow: genome-wide to specific

```bash
# 1. genome-wide screen (defaults: weighted anchored null, GPD tails, refinement)
simphyni run --sample-name screen --tree tree.nwk --traits pangenome.csv --outdir screen --cores 16 --seed 1
#    rank by effect_shrunk; candidates = pval_bh < 0.05, or the top of the ranking

# 2. focused test: candidate traits first in the file, one-vs-rest, with lineage control
simphyni run --sample-name focus --tree tree.nwk --traits candidates_first.csv --run-traits 0,...,K \
  --outdir focus --lineage soft_auto --composite --cores 16 --seed 2
#    report pval_bh < 0.05; order by effect_shrunk; lfsr < 0.05 for a confident sign

# 3. pair level: gated (composite) pairs = association beyond shared confinement; re-run with a
#    different --seed to check Monte-Carlo stability
```

## Limits

- **Lineage restriction.** Lineage restriction drives the false discoveries that remain, for every
  method. In the restricted scenario, BH calls are 45–61 % false for SimPhyNI configurations, 54 %
  for legacy, 65 % for FaST-LMM and 81 % for Fisher. The per-trait hidden-accessibility model
  (Bayesian covarion, `lineage='hidden'`) barely changes this. Restriction is not identifiable from
  a single trait's data when the trait changes rarely: the posterior P(restricted) was about 0.33
  for restricted and unrestricted traits alike. Pair-level control (`--composite`) is what reduces
  it.
- **Negative interactions inside restricted clades** remain near chance for every SimPhyNI
  configuration.
