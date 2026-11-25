# SimPhyNI null-model candidates (branch `explore/uncertainty`)

**Default on this branch** (`run_simulation()` with no options, `simphyni run` with no `--null`):
GSC-weighted statistic + anchored null, 512 replicates per trait, no replicate selection, no lineage
control (`--null anchored-screen`, candidate C below). `--null legacy` restores the earlier default
(unweighted, prevalence-conditioned).

All configurations below:

- compute the log odds ratio on **phylogenetic tip weights** (Gerstein–Sonnhammer–Chothia),
  so a clade inherited from one gain counts once rather than once per tip;
- use an **anchored null**: each observed trait is scored against its partner's
  null replicates, in both directions. No replicates are discarded, so there is no
  prevalence or footprint selection;
- run on the fast path: Numba simulation, vectorised scoring, vectorised ACR statistics.

| configuration | CLI | use it for |
|---|---|---|
| **C — anchored (default)** | *(none)* | benchmarking default; fastest, most recall; not robust to shared lineage restriction |
| **D — anchored + lineage control** | `--lineage soft_auto --composite` | pangenomes with lineage-restricted genes; lowest false-positive rate, formally justified (see [LINEAGE.md](LINEAGE.md)) |
| A — anchored, gated | `--null anchored` | heuristic lineage gate; most recall among lineage-controlled options |
| B — anchored, gated, mean | `--null anchored --combine mean` | more recall, still calibrated |

Any configuration can add `--refine-p P`. Pairs with p < P are then re-tested by exact
Monte-Carlo counts on fresh replicates (up to `--refine-max`, default 131,072).
Use it to confirm hits at genome-wide thresholds.

Details and workflows: [A](A_anchored.md) · [B](B_anchored_mean.md) · [C](C_anchored_screen.md) ·
[lineage control (D)](LINEAGE.md).

## Benchmark summary

**Data.** PanX full tier: 11 species, ≤250 tips, 3 Monte-Carlo replicates (treeWAS: 1 replicate on the 5
smallest species). The datasets are identical for all methods.

**Columns.**

- **Null ratio:** the fraction of pooled true nulls below the threshold, divided by the threshold. A
  calibrated method gives ≈1; below 1 is conservative. Pooled nulls are synthetic, real-gene twins,
  late-emerging and separately restricted traits.
- **Co-restricted ratio:** the same quantity at 0.01 for independent traits confined to the same clade.
- **Recall:** at BY < 0.01, with the correct direction.

| method | null ratio @0.05 | @0.01 | @1e-3 | @1e-4 | BY FPR @0.01 | co-restricted ratio | recall + / − | PR-AUC + / − | late-emerging PR-AUC + | restricted PR-AUC + / − | s per species |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **D anchored + soft auto + composite** | 0.29 | 0.11 | 0.08 | 0.27 | 0 | **7.4** | 0.20 / 0.12 | 0.86 / 0.76 | 0.90 | 0.50 / 0.18 | 27 |
| **A anchored, gated** | 0.42 | 0.23 | 0.16 | 0.27 | 0 | 8.6 | 0.23 / 0.16 | 0.86 / 0.77 | 0.90 | 0.47 / 0.13 | **22** |
| **B anchored, gated, mean** | 0.59 | 0.39 | 0.28 | 0.27 | 0 | 10.7 | **0.25 / 0.18** | **0.87 / 0.78** | 0.91 | 0.47 / 0.13 | 24 |
| C anchored (default) | 0.55 | 0.42 | 0.57 | 3.2 | 3e-4 | 26.7 | 0.28 / 0.17 | 0.87 / 0.77 | 0.91 | 0.55 / 0.14 | 23 |
| current default (prevalence ×8) | 0.57 | 0.41 | 0.43 | 1.9 | 1e-4 | 11.8 | 0.15 / 0.09 | 0.84 / 0.77 | 0.91 | 0.55 / 0.13 | 35 |
| footprint ×8 | 1.22 | 1.06 | 1.16 | 3.5 | 3e-4 | 20.3 | 0.24 / 0.17 | 0.82 / 0.75 | 0.91 | 0.59 / 0.13 | 35 |
| SimPhyNI 2.0.0 | 2.52 | 5.90 | 20.0 | 99.4 | 0.012 | 25.0 | 0.18 / 0.15 | 0.76 / 0.65 | 0.88 | 0.53 / 0.15 | 37 |
| FaST-LMM (pyseer `--lmm` equivalent) | 1.76 | 4.39 | 21.9 | 143 | 0.013 | 43.6 | 0.22 / 0.10 | 0.52 / 0.38 | 0.64 | 0.41 / **0.38** | 2 |
| Fisher (tips) | 6.76 | 24.6 | 170 | 1,243 | 0.14 | 80.2 | 0.75 / 0.54 | 0.65 / 0.53 | 0.77 | 0.39 / 0.13 | 2 |

**The 5 smallest species, with treeWAS:**

| method | null ratio @0.01 | @1e-4 | co-restricted ratio | recall + / − | PR-AUC + / − | s per species |
|---|---|---|---|---|---|---|
| A anchored, gated | 0.28 | 0.59 | 8.4 | 0.13 / 0.11 | 0.82 / 0.71 | 17 |
| B anchored, gated, mean | 0.44 | 0.59 | 10.0 | 0.14 / 0.12 | 0.82 / 0.72 | 17 |
| current default | 0.46 | 2.7 | 9.6 | 0.11 / 0.06 | 0.78 / 0.71 | 27 |
| treeWAS (subsequent score) | 1.48 | 32 | 32.8 | 0.12 / 0.03 | 0.42 / 0.25 | 436 |
| FaST-LMM | 4.03 | 121 | 45.1 | 0.25 / 0.09 | 0.54 / 0.40 | 2 |

**Tree-swap test.** The same 300 null and 60 interacting trait slots are placed on 36 trees, including
clonal *B. pertussis*, *S. aureus* and *S. enterica*. Values are null p < 0.01 rates.

| method | all | traits with ≤2 changes | worst tree | clonal trees | recall at BY 0.01 |
|---|---|---|---|---|---|
| A anchored, gated | 0.0028 | 0.004 | **0.010** | **0.0026** | 0.27 |
| current default | 0.0027 | 0.008 | 0.017 | 0.0054 | 0.18 |
| footprint ×8 | 0.0112 | 0.030 | 0.063 | 0.023 | 0.29 |
| FaST-LMM | 0.046 | 0.051 | 0.093 | 0.061 | 0.23 |
| Fisher | 0.25 | 0.37 | 0.47 | 0.40 | 0.64 |

**Reading the tables.**

- Fisher's and FaST-LMM's recall comes with null rates 4–25× nominal at 0.01, and 100–1,000× at 1e-4.
- treeWAS is near nominal at 0.05, but 32× at 1e-4.
- A and B are the only methods that stay at or below nominal at every threshold down to 1e-4.

**Speed.** *E. coli*, 250 tips, 4 cores, machine shared with other jobs.

| step | before | after |
|---|---|---|
| ACR statistics, 1,930 traits | 13.2 s | 6.0 s (vectorised, Numba cache) |
| simulation + scoring, 500 genes all-vs-all (124,750 pairs) | 55.6 s (default) | 3.8 s (C), 8.7 s (A) |
| simulation + scoring, 1,000 genes all-vs-all (499,500 pairs) | — | 13.9 s (C) |
| full pipeline, *S. pyogenes*, 545,490 pairs, A + refinement | — | 26 s |

## Known limits

- **Interactions inside a restricted clade lose power when gated.** Restricted PR-AUC+ is 0.47 vs 0.55. A gated
  pair is tested against a null confined to the two traits' clades, and that null can't separate
  "co-restricted" from "interacting inside the clade" (see the lineage report on `explore/adaptive-p`).
  The output columns `gated` and `overlap` mark these pairs.
- **Negative interactions inside restricted clades** stay near chance for every SimPhyNI configuration
  (PR-AUC− 0.13). FaST-LMM is better there (0.38).
- **KDE tail p-values are extrapolations.** Below 1/N (N = 512) they can underflow to 0. Use `--refine-p`
  when exact p-values are needed at genome-wide thresholds. Refined p-values have a floor of
  2 / (refine_max + 1) ≈ 1.5e-5.
- **Validity of `--combine mean`.** It is valid at twice the nominal level in theory; on the benchmark it was
  conservative everywhere.
