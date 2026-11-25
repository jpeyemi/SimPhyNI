# Large-scale evaluation: E. coli, ~10 M pairs

**Data.** PanX *E. coli* tree, 500 tips (`make_data.py`).

- **real:** the 4,501 accessory genes with 5–95 % prevalence, all-vs-all = **10,127,250 pairs**.
- **syn:** 4,100 traits with known truth on the same tree, **8,402,950 pairs**. They are generated
  without SimPhyNI code:
  - 3,000 null traits: exact F81 paths, with parameters drawn from the real genes, and true change
    counts and internal states recorded.
  - 300 planted interacting pairs: exact 4-state CTMC, I = ±1 and ±0.5.
  - 100 groups of 4 independent traits confined to the same clade, giving 600 co-restricted null pairs.
  - 50 interacting pairs inside restricted clades.

All runs used 8 cores on a shared laptop, so absolute times are indicative.

## 1. Ancestral state reconstruction (`eval_acr_sim.py`)

| | value |
|---|---|
| fast ACR speed | 4.2 ms / trait (3,000 traits in 12.5 s); real set 17.9 s for 4,501 genes, 0.9 GB peak |
| PastML speed | 593 ms / trait (140× slower) |
| fast vs PastML JOINT change counts | Spearman 1.000 |
| internal-node state accuracy (joint) | 0.966; marginal Brier 0.026 |
| root state accuracy | 0.85 |

The vectorised statistics (identical output to the per-gene loop) and Numba caching cut the
per-run ACR time from 13 s to 6 s at 1,930 traits.

**Change counts by true number of changes:**

| true changes | n | JOINT / true | FLOW / true |
|---|---|---|---|
| ≤2 | 53 | 1.00 | 1.05 |
| 3–5 | 172 | 0.95 | 1.02 |
| 6–10 | 345 | 0.93 | 1.01 |
| 11–20 | 542 | 0.86 | 0.96 |
| 21–50 | 565 | 0.72 | 0.84 |
| >50 | 1,323 | 0.12 | 0.17 |

FLOW counts are accurate up to about 20 changes per trait and saturate beyond that, because
repeated changes on a branch are invisible from the tips. Any tip-based reconstruction has this
limit. The rates for fast-evolving traits are therefore underestimated, so their null copies are
more clumped than the truth, which errs conservative.

## 2. Null simulation

**Accuracy.** `sim_bit_fast` was run at the *true* rates with the true root, against exact F81
paths (1,024 replicates, 300 traits). Values are median |relative difference| of per-trait statistics.

| simulator | mean prevalence | prevalence spread | parsimony changes | fast-evolving traits: changes |
|---|---|---|---|---|
| at most one change per branch (original `sim_bit` model) | 8.4 % | 5.6 % | 6.9 % | 21 % |
| **exact two-state transition per branch (now)** | **1.0 %** | **2.4 %** | **0.7 %** | **0.6 %** |

The original model drew one Bernoulli(1 − e^(−rate·t)) event per branch. It saturates at 1
instead of at the stationary frequency. The kernel now uses P(0→1) = π(1 − e^(−μt)) and
P(1→0) = (1 − π)(1 − e^(−μt)).

**Speed.** 4,501 traits × 512 replicates take 0.8–2.0 s.

## 3. Statistics

### Tail estimation was the problem at genome scale

BY at 8.4 M tests needs p ≈ 10⁻⁸–10⁻¹⁰. With 512 replicates, anything below about 2×10⁻³ is
extrapolated.

On the synthetic set, with the default (weighted + anchored) and the earlier Gaussian-KDE tail:

| threshold | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|
| null ratio (observed ÷ nominal) | 0.26 | 0.22 | 0.70 | **3.0** | **15** |

That gave 150 calls at BY 0.05, of which 69 were true nulls (**51 % false discoveries**).

`diag_tail.py` re-tested the 80 worst null pairs with exact Monte-Carlo counts (262,144
replicates):

- KDE p-values of 10⁻¹⁵–10⁻³⁸ had exact p of 10⁻⁴–10⁻³.
- A Gaussian kernel past the edge of a 512-value sample decays like a normal density with a tiny
  bandwidth.

**Fix: generalized Pareto tail.** It is fitted by probability-weighted moments to the top 10 % of
each null (Knijnenburg et al. 2009), with shape ≥ 0, i.e. exponential or heavier. The constraint is
conservative, and the estimate can never return 0. Letting the shape go negative (a bounded tail)
returns p = 0 beyond the fitted endpoint, which is the KDE failure again.

On the same 80 pairs:

| tail | median p / exact MC p | share > 10× too small |
|---|---|---|
| KDE | 10^−1.5 | 56 % |
| **GPD** | 10^+0.33 (≈2× conservative) | **0 %** |

### Genome-wide comparison on a common pair sample

`eval_thresholds.py` uses all planted pairs plus 625,000 random null pairs, so the old default
could run too. Bonferroni for 8.4 M pairs is 6×10⁻⁹.

| method | null ratio @1e-4 / 1e-5 / 1e-6 | null pairs < Bonferroni | strong + / − pairs with p < Bonferroni | ranking AP + / − | s for 8.4 M pairs |
|---|---|---|---|---|---|
| old default (unweighted, prevalence ×8, KDE over 64×64 pairings) | 1.8 / 12.6 / **82** | **33** | 0.27 / 0.08 | 0.22 / 0.09 | ≈2,900 (extrapolated) |
| anchored, KDE tail | 0.8 / 3.7 / 19 | 3 | 0.37 / 0.19 | 0.38 / 0.31 | 95 |
| **anchored, GPD tail (default)** | 0 / 0 / 0 | **0** | 0.04 / 0.01 | 0.37 / 0.31 | 84 |
| anchored, GPD + refinement (p < 1e-3 → 32,768 replicates) | 0.02 / 0 / 0 | 0 | 0.09 / 0 | 0.37 / 0.30 | 110 |
| soft auto + composite, GPD | 0 / 0 / 0 | 0 | 0.03 / 0 | 0.37 / 0.29 | 109 |

### Is the planted signal detectable at genome-wide thresholds?

`truth_mc.py` ran exact anchored Monte-Carlo with **4.2 million replicates per direction** for 20
planted strong positive pairs (OR = 100):

- 14 have true p between 10⁻² and 10⁻⁵.
- 5 are below the Monte-Carlo floor (4.8×10⁻⁷).

The calibrated pipeline's p-values track these (e.g. 3.9e-5 vs 5.9e-5; 1.1e-3 vs 2.9e-4; 2.7e-6
vs < 4.8e-7), and are conservative where they differ.

So most planted interactions are **not genome-wide significant under the correct null**. The
KDE-based "recall" at Bonferroni came from p ≈ 0 for pairs whose true p was 10⁻³–10⁻⁵. The low
Bonferroni recall of the calibrated configurations is mostly real. Ranking is unaffected
(AP 0.37 / 0.31, against 0.22 / 0.09 for the old default).

More replicates did not change this. A two-tier refinement (p < 1e-2 → 32,768; p < 1e-5 →
262,144) gave the same Bonferroni recall in 16 minutes.

## 4. Lineage control at scale

On the synthetic set, with the GPD tail, at p < 0.01 (BY calls were 0 everywhere):

| | co-restricted ratio @0.01 | separately restricted ratio @0.01 |
|---|---|---|
| default | 11.5 | 1.6 |
| soft auto + composite | 7.2 | 3.4 |

On real *E. coli* (10.1 M pairs), BY < 0.05 hits:

| configuration | hits | genes involved |
|---|---|---|
| default | 22,646 (98 % positive associations) | 2,736 |
| soft auto + composite | 18,970 (92 % of them also default hits) | 2,458 |
| earlier KDE tail | 68,225 | — |

Of the default's GPD hits, 99.7 % were also KDE hits; the KDE added 45,000 more.

Soft auto + composite still reduces same-clade false positives at moderate thresholds. At genome
scale, co-restricted pairs did not reach BY significance under either configuration, because the
deep tail now controls them. Lineage-control performance therefore does **not** fall off with scale
on these data. Separately restricted pairs get worse under soft restriction, about 2× at p < 0.01,
which is worth watching. The hidden-accessibility model is the right next step if real analyses
show co-restriction driving calls, but these data do not show that yet.

## 5. Speed, real data (10,127,250 pairs, 4,501 genes)

| stage | default | soft auto + composite |
|---|---|---|
| ACR (once per tree, cached) | 17.9 s | 17.9 s |
| pair enumeration + observed statistics | 28.8 s | 28.4 s |
| τ selection | — | 27.1 s |
| simulation (4,501 × 512) | 0.9 s | 0.8 s |
| scoring (anchored, GPD tails) | 69.3 s | 70.7 s |
| composite (37,998 candidate pairs) | — | 37.9 s |
| results + write | 4 s | 4 s |
| **total** | **≈2.2 min** (116 s + ACR) | **≈3.3 min** |
| peak memory | 5.9 GB | 6.1 GB |

Scoring went from 184 s to 46 s (KDE) by fusing the null-statistic, tail-count and moment
computations into Numba kernels that read each BLAS block in place, and by using larger blocks.
The GPD tail adds about 23 s.

The old default would need about 50 minutes for the same pairs (626,000 pairs in 218 s) and is
not calibrated at these thresholds.

## Reproduce

```bash
python benchmarks_large/make_data.py <panx-benchmark>/benchmarks/panx <out>
python benchmarks_large/run_pipeline.py <out>/real <out>/runs/real default
python benchmarks_large/run_pipeline.py <out>/syn  <out>/runs/syn  soft_auto_composite
python benchmarks_large/eval_stats.py <out>/syn <out>/runs/syn/results_*.parquet
python benchmarks_large/eval_acr_sim.py <out>/syn 300 --pastml 150
python benchmarks_large/eval_thresholds.py <out>/syn <out>/syn/pairs_sample.csv <results...>
python benchmarks_large/truth_mc.py <out>/syn <out>/runs/syn <results.parquet> pos1 20 16
```
