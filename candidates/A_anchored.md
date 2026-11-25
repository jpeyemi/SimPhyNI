# Candidate A — weighted anchored null with lineage-overlap gate (recommended)

```bash
simphyni run --sample-name S --tree tree.nwk --traits traits.csv --outdir out \
  --null anchored --cores 8 --seed 1
```

Python API:

```python
Sim.run_simulation(condition="none", statistic="weighted", pvalue="anchored_fast",
                   trials=512, gate="overlap", gate_p=0.05, seed=1)
```

## What it does

1. **Null replicates.** 512 per trait from the fitted gain/loss process (FLOW rates, IQR-capped
   branches), with no conditioning or replicate selection. Uses the Numba simulator.
2. **Weighted statistic.** Tips get GSC phylogenetic weights (branch length shared among the tips
   below it, scaled to sum to the number of tips). The 2×2 table and log odds ratio use those
   weights, for both the observed pair and every null pairing.
3. **Anchored p-value.** The observed pair's statistic is compared with:
   - observed trait A against the 512 replicates of B, and
   - the 512 replicates of A against observed trait B.

   Each comparison is an exact conditional test that needs only the partner trait's model. The
   reported p-value is the larger of the two, which is valid if either model is right.
   Tails are exact counts. A Gaussian KDE is used only where fewer than 20 null values are as
   extreme.
4. **Lineage-overlap gate.** This applies to pairs with p < 0.05. If the two traits' tips lie closer
   to each other than the anchored null allows (nearest-presence overlap above the null's 95th
   percentile), the pair is re-tested with both traits simulated only inside the union of their
   MRCA clades. That re-test asks whether they co-occur more or less than expected *within* their
   shared lineages. Gated pairs are marked `gated=True`, with their `overlap` score.
5. **Multiple testing.** BH, BY and Bonferroni over all pairs, as before.

## Why

- **No selection.** The prevalence/footprint nulls keep the 64 replicates nearest the observed trait
  out of 512. That is approximate conditioning without a finite-sample guarantee. The anchored
  null discards nothing.
- **Weights.** Tip counts treat one inherited clade as many observations. Weighting removes most of
  that pseudo-replication. It is the main source of the extra recall.
- **The gate** removes most false positives between independent traits that share a restricted
  lineage. Unguarded anchoring is 2–3× worse there than the old default.

## Benchmark (PanX full tier and tree-swap)

| | A | current default |
|---|---|---|
| null ratio @0.01 / @1e-4 | 0.23 / 0.27 | 0.41 / 1.9 |
| co-restricted null ratio @0.01 | 8.6 | 11.8 |
| worst tree / clonal trees, p < 0.01 | 0.010 / 0.0026 | 0.017 / 0.0054 |
| recall at BY 0.01, + / − | 0.23 / 0.16 | 0.15 / 0.09 |
| PR-AUC + / − | 0.86 / 0.77 | 0.84 / 0.77 |
| seconds per species (incl. ACR) | 22 | 35 |

## Workflow: genome-wide to specific

### 1. Genome-wide screen (all-vs-all)

```bash
simphyni run --sample-name screen --tree tree.nwk --traits pangenome.csv \
  --outdir screen --null anchored --min_prev 0.05 --max_prev 0.95 --cores 16 --seed 1
```

- Throughput is about 15–35k pairs/s for the screen. Gating adds a few seconds per 100k pairs.
- Read `pval_by < 0.05` (or 0.01) as the discovery set. `direction` is +1 for co-occurrence and
  −1 for avoidance. `effect size` is a z-like standardised log odds ratio.
- The default `--prefilter` in `runSimPhyNI.py` drops pairs that fail a Fisher pre-test. For a
  fully anchored analysis, turn it off with `--no-prefilter`. The pipeline CLI already runs
  without the pre-filter.

### 2. Focused re-test of hits (exact p-values)

Re-run the traits involved in the discoveries as query traits against everything, with Monte-Carlo
refinement:

```bash
simphyni run --sample-name focus --tree tree.nwk --traits pangenome_hits_first.csv \
  --run-traits 0,1,...,K --outdir focus --null anchored --refine-p 1e-3 \
  --cores 16 --seed 2
```

- Put the K hit traits first in the file; `--run-traits` tests them against all others.
- Pairs with p < 1e-3 are re-tested on fresh replicates: 4,096, then 32,768, then 131,072. A
  pair stops once at least 10 null values are as extreme. The refined p-value (`refined=True`)
  is an exact Monte-Carlo count, not a KDE extrapolation, with a floor of about 1.5e-5. Raise
  `--refine-max` (for example 524,288) if the BY thresholds you need are lower.
- A different `--seed` gives an independent Monte-Carlo replicate. Hits should reproduce.

### 3. Pair-level interpretation

For each confirmed pair:

- **`gated=True`**: the two traits share lineages more than expected, and the reported p-value is
  from the within-region null. Interpret a significant gated pair as "association beyond shared
  ancestry/restriction". A non-significant gated pair may still be a real interaction confined
  to one clade, because the data can't separate the two.
- **Direction and order**: for which trait arose first, or a likelihood-based effect, run a
  Pagel-style correlated-evolution test on the confirmed pairs only. It is too slow genome-wide
  but cheap for tens of pairs.
- **Sensitivity**: re-run the pair set with `--null legacy` (prevalence ×8). Pairs significant
  under both nulls are the most robust. Pairs significant only under A are usually sparse traits
  whose co-occurrence spans several independent lineages, which is what the weighting rewards.

## Output columns added

| column | meaning |
|---|---|
| `gated` | the pair was re-tested with the pair-region null |
| `overlap` | nearest-presence lineage overlap (larger = more co-localised); set for pairs that reached the gate |
| `refined` | the p-value is an exact Monte-Carlo count from `--refine-p` |
| `n_null` | null values behind a refined p-value |
