# Controlling lineage effects

**The problem.** Two independent traits confined to the same clade co-occur at the tips. Any
whole-tree null calls them associated. This affects plasmid-, phage- and phylogroup-specific genes
in pangenomes. It is the main source of false discoveries that remain with the default null.

**Recommended control:** `--lineage soft_auto --composite`. Significant pairs then mean
"associated beyond shared confinement".

## The two components

**1. `--lineage soft_auto` (model refinement).** Each trait's gain/loss rate decays as
exp(−d / (τ · mean tip depth)) with distance d from the trait's own MRCA clade.

- τ is chosen per run by posterior-predictive fit of a single-trait statistic: the clade spread of
  each trait's minority-state tips, compared between simulation and data. No pair information is
  used.
- Each trait's accessibility comes from that trait's own data, the same kind of conditioning the
  anchored null already applies to the observed trait, so the test remains valid under independence.

**2. `--composite` (composite null).** The null hypothesis is defined in advance as composite:
independence under the whole-tree model **or** under the pair-region model, where both traits are
confined to the union of their MRCA clades. The reported p is max(p_tree, p_region).

- For a composite null, the supremum p satisfies P(p ≤ α) ≤ α under whichever model holds, so it is
  valid if either does (Berger & Boos 1994).
- p_region is computed only for pairs BH rejects at level 0.05 on p_tree. BH and BY are monotone
  (larger p-values never add rejections), so decisions at levels ≤ 0.05 are exactly those of the full
  composite test. The shortcut saves computation and involves no data-dependent choice of null.
- Pairs where p_region is the larger value are marked `gated=True`.

## Results with current defaults

The defaults here are the weighted anchored null, exact simulator, GPD tails and refinement.

**PanX full tier (11 species, ≤250 tips, 3 replicates).** Cells are recall / observed FDP at BH
q < 0.05.

| configuration | synthetic | late-emerging | restricted | same-clade null ratio @0.01 |
|---|---|---|---|---|
| default | 0.37 / 0 | 0.51 / 0.014 | 0.36 / 0.59 | 28 |
| `--lineage soft_auto --composite` | 0.33 / 0 | 0.45 / 0.010 | 0.18 / 0.45 | 11 |

**Quick tier (6 species, ≤150 tips).**

| configuration | synthetic | restricted | same-clade null ratio @0.01 |
|---|---|---|---|
| default | 0.25 / 0.022 | 0.27 / 0.61 | 23.2 |
| `--lineage soft_auto` | 0.24 / 0.022 | 0.22 / 0.59 | 14.8 |
| hidden accessibility (`lineage='hidden'`) | 0.26 / 0.021 | 0.26 / 0.60 | 21.5 |
| hidden + composite | 0.21 / 0.026 | 0.12 / 0.49 | 14.3 |
| **soft auto + composite** | 0.20 / 0.027 | 0.13 / **0.50** | **9.7** |

**Real *E. coli* (4,501 genes, 10.1 M pairs).** The default makes 38,174 BH calls in 5 min; with
lineage control there are 31,618 in 11 min. 8,032 of the default's calls (21 %) are explained by
shared confinement or lose significance, and 1,476 are new.

## Approaches that did not help

| approach | outcome |
|---|---|
| hard MRCA restriction (`lineage='mrca'`) | halves same-clade FPs, but separately restricted pairs look like avoidance (4× more FPs) |
| distance (emergence) threshold (`'dist'`) | no effect on same-clade FPs |
| adaptive phylogroups / per-trait confinement tests (`'adaptive'`, `'soft_adaptive'`) | no effect: rarely changing confined traits are not detectably restricted |
| hidden accessibility, Bayesian covarion (`'hidden'`) | almost no effect: posterior P(restricted) ≈ 0.33 for restricted and unrestricted traits alike |
| overlap gate (choose the region null when lineages overlap) | similar FPR to composite, but the null is chosen after looking at the pair; not used |

**Why per-trait models cannot fix this.** A trait that changes rarely and sits in one clade is
explained equally well by one ancestral gain as by restriction. Restriction is not identifiable from
a single trait's data. Controlling co-restriction needs pair-level information (the composite null)
or information shared across genes.

## What the control does not do

- It cannot separate "co-restricted and independent" from "co-restricted and interacting within the
  clade". Interactions confined to a clade lose power (restricted recall 0.36 → 0.18).
- Negative interactions inside restricted clades remain near chance for every SimPhyNI
  configuration.
- About half of the restricted-scenario calls are still false. The scenario is a stress test (see
  `README.md`, Limits).
