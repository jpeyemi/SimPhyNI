# Candidate C — anchored screen, no gate (default on this branch; fastest, exploratory)

```bash
simphyni run ...            # default; same as --null anchored-screen
```

Python API:

```python
Sim.run_simulation(condition="none", statistic="weighted", pvalue="anchored_fast", trials=512)
```

## Difference from A

C is A without the lineage-overlap gate. Every pair keeps its anchored p-value against the
whole-tree null.

## Benchmark (PanX full tier)

| | C | A |
|---|---|---|
| null ratio @0.01 / @1e-4 | 0.42 / 3.2 | 0.23 / 0.27 |
| co-restricted null ratio @0.01 | **26.7** | 8.6 |
| recall at BY 0.01, + / − | 0.28 / 0.17 | 0.23 / 0.16 |
| PR-AUC + / − | 0.87 / 0.77 | 0.86 / 0.77 |
| restricted PR-AUC + | 0.55 | 0.47 |

- C has the most recall of any calibrated configuration. On single-lineage nulls (synthetic,
  real-gene, late-emerging) it is conservative at 0.01.
- It is badly anticonservative for independent traits confined to the same clade (26.7× at 0.01).
  Many accessory genes in real pangenomes are like this: plasmid- or phage-borne,
  phylogroup-specific.

## When to choose C

- **Ranking and hypothesis generation**, where ordering matters and false positives from shared
  restriction are acceptable or handled downstream.
- **Very large screens** where the gate's cost matters. The gate itself is cheap; most of the cost
  is the per-pair pair-region simulation for candidates.
- **As step 1 of a two-step analysis:**

```bash
# 1. screen: every pair, fastest
simphyni run --sample-name screen ... --null anchored-screen --cores 16
# 2. re-test the screen's top pairs (e.g. pval_bh < 0.1) with A, which applies the gate,
#    plus exact refinement
simphyni run --sample-name confirm ... --run-traits 0,...,K --null anchored --refine-p 1e-3
```

Don't report C's BY calls as final discoveries on pangenome data without the step-2 check.
