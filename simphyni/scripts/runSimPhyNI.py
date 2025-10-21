#!/usr/bin/env python
import pickle
from pathlib import Path
import argparse
from simphyni import TreeSimulator

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
parser.add_argument("-c", "--cores", type=int, default=-1, help="number of cores for parallelization")
parser.add_argument("--min_prev", type=float, default=0.05, help="minimum prevalence for trait simuation")
parser.add_argument("--max_prev", type=float, default=0.95, help="maximum prevalence for trait ismulation")
parser.add_argument(
        "--prefilter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable prefiltering (default: enabled)",
    )
parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable plotting of results (default: disabled)",
    )
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

print("Initializing SimPhyNI...")

Sim.initialize_simulation_parameters(
    single_trait=single_trait,
    pre_filter=args.prefilter
)

# ----------------------
# Run Simulation
# ----------------------
print("Running SimPhyNI analysis...")
Sim.run_simulation(cores = args.cores)

# ----------------------
# Save Outputs
# ----------------------
output_dir = Path(args.outdir)
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'simphyni_object.pkl', 'wb') as f:
    pickle.dump(Sim, f)

Sim.get_results().to_csv(output_dir / 'simphyni_results.csv')
print("Simulation completed.")

if args.plot:
    Sim.plot_results(pval_col = 'pval_naive', output_file= str(output_dir / 'heatmap_uncorrected.svg'), figure_size=10)
    Sim.plot_results(pval_col = 'pval_bh', output_file= str(output_dir / 'heatmap_bh.svg'), figure_size=10)
    Sim.plot_results(pval_col = 'pval_by', output_file= str(output_dir / 'heatmap_by.svg'), figure_size=10)
    Sim.plot_results(pval_col = 'pval_bonf', output_file= str(output_dir / 'heatmap_bonf.svg'), figure_size=10)
