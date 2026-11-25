# Large-scale evaluation (current defaults)

**Default null.** GSC-weighted anchored null, 512 replicates per trait, exact branch transitions, GPD
tails, and refinement of pairs with p < 0.01 (at most 5,000, 32,768 replicates). Lineage control is
`--lineage soft_auto --composite`.

**Data.** PanX *E. coli* tree, 500 tips (`make_data.py`):

- **real:** 4,501 accessory genes with 5–95 % prevalence, all-vs-all = **10,127,250 pairs**.
- **syn:** 4,100 traits with known truth, **8,402,950 pairs**. They are generated without SimPhyNI code:
  - 3,000 null traits: exact F81 paths with parameters drawn from the real genes; true change
    counts and internal states recorded.
  - 300 planted interacting pairs: exact 4-state CTMC, I = ±1 and ±0.5.
  - 600 co-restricted null pairs.
  - 50 interacting pairs inside restricted clades.

Runs used 8 cores on a shared laptop.

## 1. Ancestral state reconstruction

| | value |
|---|---|
| speed | 4.2 ms / trait; 4,501 genes in 18 s, 0.9 GB peak (PastML: 593 ms / trait) |
| agreement with PastML | JOINT change counts identical (Spearman 1.000) |
| internal-node states | 96.6 % correct (joint); marginal Brier 0.026 |
| root state | 85 % correct |

**FLOW change counts relative to the true number:**

| true changes | ≤2 | 3–5 | 6–10 | 11–20 | 21–50 | >50 |
|---|---|---|---|---|---|---|
| FLOW / true | 1.05 | 1.02 | 1.01 | 0.96 | 0.84 | 0.17 |

Counts are accurate up to about 20 changes per trait and saturate beyond that. Repeated changes on
a branch are invisible from the tips, a limit of any tip-based reconstruction. The underestimated
rates make null copies of fast-evolving traits more clumped than the truth, which errs
conservative.

## 2. Null simulation

**Accuracy.** At the true rates, against exact F81 paths (300 traits × 1,024 replicates), the median
relative difference is:

- 1.0 % in mean prevalence;
- 2.4 % in prevalence spread;
- 0.7 % in parsimony changes, including fast-evolving traits.

Each branch uses the exact two-state transition, P(0→1) = π(1 − e^(−μt)). The earlier model allowed at
most one change per branch and was 7–21 % off.

**Speed.** 4,501 traits × 512 replicates take about 1 s.

## 3. Statistics

### Tail calibration

BY at 8.4 M tests needs p ≈ 10⁻⁸–10⁻¹⁰. With 512 replicates, anything below about 2×10⁻³ is
extrapolated. A Gaussian-KDE tail decays like a normal density with a tiny bandwidth just past the
sample, and returned p of 10⁻¹⁵–10⁻³⁸ for null pairs whose exact Monte-Carlo p was 10⁻⁴–10⁻³
(`diag_tail.py`). The default therefore fits a generalized Pareto distribution to the top of each null,
by probability-weighted moments with shape ≥ 0, so it never returns p = 0.

**Common pair sample** (all planted pairs + 625,000 random null pairs; Bonferroni for 8.4 M = 6×10⁻⁹):

| | null ratio @1e-4 / 1e-5 / 1e-6 | null pairs < Bonferroni | ranking AP + / − |
|---|---|---|---|
| **default** | 0.02 / 0 / 0 | **0** | **0.37 / 0.30** |
| same null with a KDE tail | 0.8 / 3.7 / 19 | 3 | 0.38 / 0.31 |
| legacy null (unweighted, prevalence ×8, KDE) | 1.8 / 12.6 / **82** | **33** | 0.22 / 0.09 |

Over all 8.4 M pairs, the default made no false calls at BH or BY 0.05. With a KDE tail, 66 % of BH
calls were false.

### Detectability at genome scale

`truth_mc.py` ran exact anchored Monte-Carlo with 4.2 M replicates per direction for 20 planted strong
pairs:

- 14 have true p between 10⁻² and 10⁻⁵.
- 5 are below the Monte-Carlo floor (4.8×10⁻⁷).

The default's p-values track these, conservative where they differ. Most planted interactions are
therefore **not genome-wide significant under the correct null**. Genome-wide runs are best used to
rank candidates, followed by focused tests on smaller trait sets.

### FDR-based reporting (PanX full tier: 11 species, ≤250 tips, 3 replicates)

Cells are recall / observed FDP at BH q < 0.05 (`report_panx.py`; full table in
`report_full_tier.md`):

| configuration | synthetic | late-emerging | restricted | BY synthetic recall |
|---|---|---|---|---|
| **default** | **0.37 / 0** | **0.51 / 0.014** | 0.36 / 0.59 | 0.22 |
| default + lineage control | 0.33 / 0 | 0.45 / 0.010 | 0.18 / 0.45 | 0.19 |
| legacy null | 0.24 / 0.004 | 0.38 / 0.009 | 0.11 / 0.54 | 0.15 |
| FaST-LMM | 0.27 / 0.45 | 0.36 / 0.32 | 0.51 / 0.65 | 0.19 |
| Fisher | 0.77 / 0.65 | 0.81 / 0.52 | 0.52 / 0.81 | 0.69 |

- BH controls the FDR everywhere except lineage restriction. That case is covered in
  `candidates/LINEAGE.md`.
- BY costs about 40 % of recall relative to BH.

**Ordering on synthetic pairs:**

| rank by | AP | recall at precision ≥ 0.95 |
|---|---|---|
| p-value | 0.841 | 0.57 |
| \|effect\| | 0.851 | 0.59 |
| **effect_shrunk** | **0.857** | **0.60** |

## 4. Real E. coli (10.1 M pairs): speed and calls

| stage | default | lineage control |
|---|---|---|
| ACR (once per tree) | 18 s | 18 s |
| pairs + observed statistics | 28 s | 28 s |
| τ selection | — | 27 s |
| simulation (4,501 × 512) | 1 s | 1 s |
| scoring (anchored, GPD tails) | 71 s | 71 s |
| refinement (5,000 pairs) | 175 s | 179 s |
| composite (37,645 candidates; 5,000 refined) | — | 331 s |
| **total (excluding ACR)** | **299 s** | **663 s** |
| peak memory | 4.7 GB | 4.4 GB |
| BH / BY < 0.05 calls | 38,174 / 22,629 | 31,618 / 18,851 |

Lineage control removes 8,032 of the default's BH calls (21 %), which shared confinement explains or
which lose significance, and adds 1,476. The legacy null would need about 50 minutes for these pairs
and is anticonservative below 10⁻⁵.

## Reproduce

```bash
python benchmarks_large/make_data.py <panx-benchmark>/benchmarks/panx <out>
python benchmarks_large/run_pipeline.py <out>/real <out>/runs/real default             # or soft_auto_composite
python benchmarks_large/run_pipeline.py <out>/syn  <out>/runs/syn  default
python benchmarks_large/eval_stats.py <out>/syn <out>/runs/syn/results_default.parquet
python benchmarks_large/eval_acr_sim.py <out>/syn 300 --pastml 150
python benchmarks_large/eval_thresholds.py <out>/syn <out>/syn/pairs_sample.csv <results...>
python benchmarks_large/truth_mc.py <out>/syn <out>/runs/syn <results.parquet> pos1 20 16
python benchmarks_large/report_panx.py <runs/full> <datasets/full> <variants...> --md report.md
```
