# Candidate B — anchored, gated, mean combination (more recall)

```bash
simphyni run ... --null anchored --combine mean
```

Python API:

```python
Sim.run_simulation(condition="none", statistic="weighted", pvalue="anchored_fast", trials=512,
                   gate="overlap", gate_p=0.05, anchor_combine="mean")
```

## Difference from A

The two anchored directions are combined as `(p_A + p_B) / 2` instead of `max(p_A, p_B)`:

- p_A is observed A against B's replicates.
- p_B is A's replicates against observed B.

The mean of two valid p-values is valid at twice the nominal level, so B gives up A's
"valid if either model is right" guarantee in exchange for power. When one trait's model is
poor (often a sparse trait whose null is too narrow), max takes the conservative side while
the mean splits the difference.

## Benchmark (PanX full tier)

| | B | A | current default |
|---|---|---|---|
| null ratio @0.05 / @0.01 / @1e-4 | 0.59 / 0.39 / 0.27 | 0.42 / 0.23 / 0.27 | 0.57 / 0.41 / 1.9 |
| co-restricted null ratio @0.01 | 10.7 | 8.6 | 11.8 |
| recall at BY 0.01, + / − | **0.25 / 0.18** | 0.23 / 0.16 | 0.15 / 0.09 |
| PR-AUC + / − | **0.87 / 0.78** | 0.86 / 0.77 | 0.84 / 0.77 |

B is still conservative at every threshold. It beats the current default on every measure
except restricted-scenario PR-AUC+, which it shares with A.

## When to choose B

- **Small or moderately sized trees** (≤ ~150 tips), where the BY threshold leaves few calls and
  power matters most.
- **Analyses with a confirmation step.** Recommended workflow: screen with B, then confirm with A
  plus refinement:

```bash
# 1. sensitive screen
simphyni run --sample-name screen --tree tree.nwk --traits traits.csv --outdir screen \
  --null anchored --combine mean --cores 16 --seed 1
# 2. confirm the screen's hits with the stricter combination and exact p-values
simphyni run --sample-name confirm --tree tree.nwk --traits hits_first.csv --run-traits 0,...,K \
  --outdir confirm --null anchored --refine-p 1e-3 --cores 16 --seed 2
```

Report pairs that pass BY in step 2.
