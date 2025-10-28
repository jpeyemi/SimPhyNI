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
parser.add_argument("-r", "--run_traits", type=int, default=0,
                    help="First run_traits traits against the rest")
parser.add_argument("-c", "--cores", type=int, default=-1, help="number of cores for parallelization")
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
parser.add_argument("--save-object", action=argparse.BooleanOptionalAction, default=False, help="Saves parsable python object containing the complete analysis of each sample (Default: disabled)")
args = parser.parse_args()

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
    run_traits = args.run_traits,
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

if args.save_object:
    with open(output_dir / 'simphyni_object.pkl', 'wb') as f:
        pickle.dump(Sim, f)

Sim.get_results().to_csv(output_dir / 'simphyni_results.csv')
print("Simulation completed.")

if args.plot:
    Sim.plot_results(pval_col = 'pval_naive', output_file= str(output_dir / 'heatmap_uncorrected.png'), figure_size=10)
    Sim.plot_results(pval_col = 'pval_bh', output_file= str(output_dir / 'heatmap_bh.png'), figure_size=10)
    Sim.plot_results(pval_col = 'pval_by', output_file= str(output_dir / 'heatmap_by.png'), figure_size=10)
    Sim.plot_results(pval_col = 'pval_bonf', output_file= str(output_dir / 'heatmap_bonf.png'), figure_size=10)
