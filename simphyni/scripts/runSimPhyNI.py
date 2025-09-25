#!/usr/bin/env python
import pickle
from pathlib import Path
import argparse
from simphyni import TreeSimulator
from simphyni import PairStatistics

# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser(description="Run SimPhyNI KDE-based trait simulation.")
parser.add_argument("-p", "--pastml", required=True, help="Path to PastML output CSV")
parser.add_argument("-s", "--systems", required=True, help="Path to input traits CSV")
parser.add_argument("-t", "--tree", required=True, help="Path to rooted Newick tree")
parser.add_argument("-o", "--outdir", required=True, help="Output path to save the Sim object")
parser.add_argument("-r", "--runtype", type=int, choices=[0, 1], default=0,
                    help="1 for single trait mode, 0 for multi-trait [default: 0]")
args = parser.parse_args()
single_trait = bool(args.runtype)

# ----------------------
# Simulation Setup
# ----------------------
Sim = TreeSimulator(
    tree=args.tree,
    pastmlfile=args.pastml,
    obsdatafile=args.systems
)

Sim.set_trials(64)
print("Initializing SimPhyNI...")

Sim.initialize_simulation_parameters(
    pair_statistic=PairStatistics._log_odds_ratio_statistic,
    collapse_theshold=0.001,
    single_trait=single_trait,
    prevalence_threshold=0.00,
    kde=True
)

# ----------------------
# Run Simulation
# ----------------------
print("Running SimPhyNI analysis...")
Sim.run_simulation(
    parallel=True,
    bit=True,
    norm=True
)

# ----------------------
# Save Outputs
# ----------------------
output_dir = Path(args.outdir)
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'simphyni_object.pkl', 'wb') as f:
    pickle.dump(Sim, f)

Sim.get_top_results(top = 2**63 - 1).to_csv(output_dir / 'simphyni_results.csv')
print("Simulation completed.")
