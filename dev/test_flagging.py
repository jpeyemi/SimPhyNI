#!/usr/bin/env python3
"""
test_flagging.py — end-to-end test of flag_uncalibratable_traits in the pipeline.

Tests two datasets:
  - E. coli pangenome  (tests/panx/)
  - MTB RIF dataset    (MTB_bench)

Run from the SimPhyNI repo root with:
    conda run -n simphyni_dev python dev/test_flagging.py [ecoli|mtb|both]
    (default: both)
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
import numpy as np
from ete3 import Tree
from simphyni import TreeSimulator
from simphyni.Simulation.simulation import build_sim_params, flag_uncalibratable_traits, sim_bit

# ── Dataset paths ─────────────────────────────────────────────────────────────
ECOLI_PASTML = REPO / 'tests/panx/pastmlout_marginal.csv'
ECOLI_OBS    = REPO / 'tests/panx/ecoli_accessory.csv'
ECOLI_TREE   = REPO / 'tests/panx/ecoli_accessory.nwk'

MTB_BASE   = Path('/Users/jpeyemi/Library/CloudStorage/Dropbox/Lieberman Lab'
                  '/Personal lab notebooks/Ishaq/LiebermanLabLocal'
                  '/MTB_bench/2-MTB-Results/RIF/simphyni')
MTB_PASTML = MTB_BASE / 'tmp/RIF/1-PastML-api/pastmlout.csv'
MTB_OBS    = MTB_BASE / 'inputs/RIF.parquet'
MTB_TREE   = MTB_BASE / 'tmp/RIF/0-formatting/RIF.nwk'

run_which = sys.argv[1] if len(sys.argv) > 1 else 'both'


# ── Shared helper ─────────────────────────────────────────────────────────────
def section(title):
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def run_dataset(label, pastml_path, obs_path, tree_path,
                run_traits=3, known_targets=None):
    section(f"{label} — standalone flagging")
    df     = pd.read_csv(pastml_path)
    params = build_sim_params(df, counting='FLOW', subsize='ORIGINAL',
                              no_threshold=False)
    flagged = flag_uncalibratable_traits(params)

    print(f"Total traits : {len(params):,}")
    print(f"Flagged      : {len(flagged)} ({100*len(flagged)/len(params):.1f}%)")
    print(f"  flag_subsize     : {flagged['flag_subsize'].sum()}")
    print(f"  flag_single_gain : {flagged['flag_single_gain'].sum()}")

    if known_targets:
        print()
        print("Known trait status:")
        for gene in known_targets:
            row = flagged[flagged['gene'] == gene]
            if len(row):
                status = f"FLAGGED — {row['flag_reasons'].values[0]}"
            else:
                status = "ok (not flagged)"
            print(f"  {gene:<16}: {status}")

    print()
    print("Top flagged by subsize_ratio:")
    top = flagged.nlargest(5, 'subsize_ratio')[
        ['gene', '_raw_gains', '_raw_losses', '_raw_count',
         'subsize_ratio', 'flag_subsize', 'flag_single_gain']
    ]
    print(top.to_string(index=False))

    section(f"{label} — pipeline include_flagged=False (default)")
    Sim = TreeSimulator(tree=str(tree_path), pastmlfile=params,
                        obsdatafile=str(obs_path))
    Sim.initialize_simulation_parameters(run_traits=run_traits, pre_filter=False)
    Sim.run_simulation(cores=1, include_flagged=False)
    res = Sim.get_results()
    fl  = Sim.get_flagged_traits()
    print(f"Traits flagged by pipeline : {len(fl)}")
    print(f"Results shape              : {res.shape}")
    print(f"null_calibrated column     : {'null_calibrated' in res.columns}")
    print()
    print("Top results:")
    print(res.head(5)[['T1','T2','direction','effect size',
                        'pval_naive','pval_bh']].to_string(index=False))

    section(f"{label} — pipeline include_flagged=True (transparent)")
    Sim2 = TreeSimulator(tree=str(tree_path), pastmlfile=params,
                         obsdatafile=str(obs_path))
    Sim2.initialize_simulation_parameters(run_traits=run_traits, pre_filter=False)
    Sim2.run_simulation(cores=1, include_flagged=True)
    res2 = Sim2.get_results()
    print(f"Results shape              : {res2.shape}")
    if 'null_calibrated' in res2.columns:
        n_uncal = (~res2['null_calibrated']).sum()
        print(f"Pairs marked null_calibrated=False : {n_uncal}")
        if n_uncal:
            print()
            print("Sample of flagged-null pairs:")
            print(res2[~res2['null_calibrated']][
                ['T1','T2','effect size','pval_naive','pval_bh','null_calibrated']
            ].head(8).to_string(index=False))

ECOLI_TARGETS = ['espZ', 'perR', 'yfjM', 'ybcH', 'betI', 'gatB', 'yagU']

if run_which in ('ecoli', 'both'):
    run_dataset(
        label        = 'E. coli pangenome',
        pastml_path  = ECOLI_PASTML,
        obs_path     = ECOLI_OBS,
        tree_path    = ECOLI_TREE,
        run_traits   = 3,
        known_targets= ECOLI_TARGETS,
    )

if run_which in ('mtb', 'both'):
    run_dataset(
        label        = 'MTB RIF',
        pastml_path  = MTB_PASTML,
        obs_path     = MTB_OBS,
        tree_path    = MTB_TREE,
        run_traits   = 3,
        known_targets= ['phenotype_0'],   # RIF resistance phenotype
    )

print("\nDone.")
