# Controlling lineage effects

**The problem.** Two independent traits confined to the same clade co-occur at the tips. Every
whole-tree null calls them associated. In the benchmark's restricted scenario these are the "shared"
null pairs. The baseline here is the default null on this branch: GSC-weighted statistic, anchored
p-values, 512 replicates, no replicate selection.

## Approaches tested

All options are on `run_simulation`, and the CLI flags are on `runSimPhyNI.py` / `simphyni run`.

| approach | kind | option |
|---|---|---|
| MRCA restriction | model | `lineage='mrca'`: a trait's events only inside its own MRCA clade |
| distance threshold | model | `lineage='dist'`: no events above the trait's estimated emergence depth |
| simulation threshold | model | `lineage='adaptive'`: exclude phylogroups where the trait is absent *and* unrestricted simulations almost never leave it absent |
| soft accessibility | model | `lineage='soft', tau=τ`: the rate decays as exp(−d / (τ · mean tip depth)) with distance from the trait's MRCA clade |
| soft, per-trait adaptive | model | `lineage='soft_adaptive'`: soft restriction only for traits more confined than their unrestricted null |
| soft, automatic τ | model | `lineage='soft_auto'`: τ chosen by posterior-predictive fit of single-trait clade spread |
| composite null | inference | `composite=True`: p = max(p_whole-tree, p_pair-region) |
| overlap gate | post-hoc | `gate='overlap'` (earlier candidate A) |

Every model option derives each trait's accessibility from **that trait's own data only**. This is
the same kind of conditioning the anchored null already does with the observed anchor trait.
Under the null of independent traits, conditioning on single-trait features leaves the test valid.
What changes is only how faithfully each trait's marginal process is modelled.

## Results

### Quick tier (6 species, ≤150 tips)

Values are null p < 0.01 rates. "Same-clade" is the shared co-restricted null pairs; "separate" is pairs
whose traits are restricted to different clades.

| | real-gene | separate | same-clade | same-clade BY FPR | PR-AUC + / − | recall + / − | restricted PR-AUC + / − |
|---|---|---|---|---|---|---|---|
| no lineage control (default) | 0.002 | 0.023 | 0.197 | 0.028 | 0.795 / 0.686 | 0.19 / 0.08 | 0.50 / 0.13 |
| MRCA | 0.003 | **0.080** | 0.112 | 0.028 | 0.763 / 0.654 | 0.21 / 0.08 | 0.44 / 0.16 |
| distance threshold | 0.002 | 0.023 | 0.173 | 0.035 | 0.792 / 0.665 | 0.19 / 0.11 | 0.49 / 0.13 |
| adaptive phylogroups | 0.003 | 0.027 | 0.197 | 0.027 | 0.791 / 0.685 | 0.22 / 0.11 | 0.51 / 0.13 |
| soft τ = 0.05 / 0.2 / 0.5 / 1.0 | 0.002 | 0.025 / 0.010 / 0.012 / 0.017 | 0.098 / **0.087** / 0.105 / 0.132 | 0.022 / 0.013 / 0.013 / 0.017 | 0.760 / 0.765 / 0.777 / 0.785 (+) | ≈0.18–0.19 / 0.08 | 0.45–0.51 / 0.14–0.16 |
| soft, per-trait adaptive | 0.003 | 0.023 | 0.182 | 0.025 | 0.788 / 0.683 | 0.22 / 0.11 | 0.51 / 0.14 |
| soft, automatic τ | 0.002 | 0.015 | 0.100 | 0.020 | 0.772 / 0.650 | 0.22 / 0.09 | 0.46 / **0.17** |
| composite null | 0.001 | 0.003 | 0.050 | 0.003 | 0.768 / 0.663 | 0.16 / 0.07 | 0.45 / 0.13 |
| soft τ = 0.2 + composite | 0.001 | 0.004 | **0.035** | 0.005 | 0.759 / 0.640 | 0.15 / 0.07 | 0.47 / 0.16 |
| overlap gate | 0.002 | 0.005 | 0.053 | 0.007 | 0.794 / 0.686 | 0.22 / 0.08 | 0.45 / 0.13 |

### Full tier (11 species, ≤250 tips, 3 replicates)

| | null ratio @0.01 | @1e-4 | same-clade ratio @0.01 | recall + / − | PR-AUC + / − | restricted PR-AUC + / − | s per species |
|---|---|---|---|---|---|---|---|
| no lineage control (default) | 0.42 | 3.2 | 26.7 | 0.28 / 0.17 | 0.87 / 0.77 | 0.55 / 0.14 | 18 |
| soft, automatic τ | 0.35 | 2.6 | 15.6 | 0.26 / 0.16 | 0.86 / 0.76 | 0.53 / 0.17 | 26 |
| composite null | 0.13 | **0.13** | 8.2 | 0.21 / 0.12 | 0.86 / 0.77 | 0.47 / 0.13 | 21 |
| **soft auto + composite** | **0.11** | 0.27 | **7.4** | 0.20 / 0.12 | 0.86 / 0.76 | 0.50 / **0.18** | 27 |
| overlap gate | 0.23 | 0.27 | 8.6 | 0.23 / 0.16 | 0.86 / 0.77 | 0.47 / 0.13 | 22 |
| old default (unweighted, prevalence ×8) | 0.41 | 1.9 | 11.8 | 0.15 / 0.09 | 0.84 / 0.77 | 0.55 / 0.13 | 35 |
| FaST-LMM | 4.4 | 143 | 43.6 | 0.22 / 0.10 | 0.52 / 0.38 | 0.41 / 0.38 | 2 |

## What we learned

1. **Hard MRCA restriction over-restricts.** It halves same-clade false positives, but traits restricted to
   *different* clades then look like they avoid each other: false positives rise 4×. It is dominated
   by soft restriction.
2. **The distance threshold and adaptive phylogroups barely help.** The restricted traits change rarely, so
   their absence from most of the tree is not surprising under an unrestricted null. Per-trait tests
   for confinement (adaptive phylogroups, per-trait adaptive soft) have no power for these traits.
   Restriction has to be modelled for every trait.
3. **Soft accessibility is the best model refinement.**
   - At τ ≈ 0.1–0.2 of tip depth it halves same-clade false positives on 150-tip trees. It improves
     separately restricted pairs and is the only change that raises power for negative interactions
     inside restricted clades (PR-AUC− 0.13 → 0.17–0.18).
   - It costs about 0.01–0.03 PR-AUC on unrestricted traits.
   - Choosing τ automatically from single-trait fit matches the best hand-tuned τ, so no benchmark
     tuning is needed.
   - On 250-tip trees, though, model refinement alone removes only about 40% of the excess (26.7 → 15.6).
4. **The composite null provides the guarantee.** Combined with soft restriction, it has the lowest
   same-clade false-positive rate of any method tested (7.4× nominal). Every pooled null ratio is
   ≤ 0.3 down to 1e-4. Positive recall is 0.20 (old default 0.15); the price is negative recall
   (0.16 → 0.12 compared with no lineage control).
5. **The overlap gate has the best recall among controlled options**, but it chooses its null after looking
   at the pair. It is kept only as a heuristic for comparison.

## Statistical justification of the composite null

The null hypothesis is defined in advance as composite:

> H0: the two traits evolve independently, under the whole-tree model M_T **or** under the
> pair-region model M_R (both traits confined to the union of their MRCA clades).

For a composite null, the supremum p-value p = max(p_T, p_R) satisfies P(p ≤ α) ≤ P(p_j ≤ α) ≤ α
under whichever model M_j is true. It is therefore valid whenever either model holds. This is the
standard construction for nuisance parameters (Berger & Boos 1994 give the confidence-set version).

Computing p_R only for pairs with p_T < s, where s is at least the largest level you will test at
(`sup_p`, default 0.05), changes no decision:

- A pair with p_T ≥ s has p ≥ s, so it can't be rejected at any level ≤ s.
- In BH/BY step-up procedures, p-values above every candidate threshold don't change which pairs
  are rejected.

So the shortcut is a computational device, not a data-dependent choice of null. The overlap gate
differs: it decides from the data which null to report.

What the composite null does **not** do is separate "co-restricted and interacting within the
clade" from "co-restricted and independent". A significant pair under the composite null is
associated beyond what shared confinement explains.

## Recommendations

- **Benchmarking default:** no lineage control (`lineage=None`, `composite=False`), as requested. It is
  fast, has the most recall, and is calibrated on single-lineage nulls. Its same-clade
  false-positive rate is the one weakness to track.
- **Pangenome analyses with plasmid-, phage- or phylogroup-restricted genes:**
  `--lineage soft_auto --composite`. Model refinement removes what it can; the composite null
  guarantees the rest.
- **Next model refinement to try:** a hidden-accessibility (covarion-style) model. Each trait evolves
  only on branches open to it, and the null samples open regions from their posterior given that
  trait alone. That would carry uncertainty about restriction into the null instead of fixing a
  decay shape. A genome-wide lineage-flux model (branch turnover multipliers shared by all genes)
  is the complementary refinement for clades that turn genes over faster.

## Update: exact simulator, calibrated tails and the hidden-accessibility model

After the exact branch transitions and GPD tails became the default, every lineage figure above
rose. The old one-change-per-branch simulator had made null traits more clustered, which partly
masked shared restriction.

We then tested a per-trait hidden-accessibility model (`lineage='hidden'`, `simphyni/Simulation/hidden.py`):

- **Model.** A 4-state covarion chain: accessibility × trait, with the trait changing only while
  open.
- **Fit.** The likelihood is computed on a grid of switching rate, open fraction and open-region
  rate, with prior P(unrestricted) = 0.5.
- **Null.** Each 64-replicate chunk draws its parameters from the grid posterior and samples the
  accessibility map from the joint posterior before re-simulating the trait.

Quick tier (6 species), BH q < 0.05:

| configuration | restricted recall / FDP | same-clade ratio @0.01 | synthetic recall / FDP |
|---|---|---|---|
| default | 0.27 / 0.61 | 23.2 | 0.25 / 0.022 |
| hidden | 0.26 / 0.60 | 21.5 | 0.26 / 0.021 |
| hidden + composite | 0.12 / 0.49 | 14.3 | 0.21 / 0.026 |
| soft auto | 0.22 / 0.59 | 14.8 | 0.24 / 0.022 |
| **soft auto + composite** | 0.13 / **0.50** | **9.7** | 0.20 / 0.027 |

The posterior P(restricted) was about 0.33 for restricted and unrestricted traits alike (0.36 vs
0.33). A rarely changing trait confined to one clade is explained equally well by a single
ancestral gain, so restriction is not identifiable from one trait. An honest per-trait model
therefore barely changes the null.

- The point-fit variant (`hidden_ml`) recognised 97 % of unrestricted traits as unrestricted, but
  only 13 % of restricted ones as restricted.
- The emission approximation (`hidden_emission`) over-restricted every clumped trait.

Lineage confounding needs pair-level information. `soft_auto + composite` remains the recommended
control. The next step on the model side would be genome-wide information: shared clade-level
accessibility estimated across many genes, which single-trait models cannot see.
