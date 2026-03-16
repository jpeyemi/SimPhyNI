#!/usr/bin/env python
import pickle
from pathlib import Path
import argparse
import pandas as pd
from simphyni import TreeSimulator
from simphyni.Simulation.simulation import build_sim_params


def main():
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
    parser.add_argument("--save-object", action=argparse.BooleanOptionalAction, default=False,
                        help="Saves parsable python object containing the complete analysis of each sample (Default: disabled)")
    args = parser.parse_args()

    # ----------------------
    # Load & normalise ACR parameters
    # ----------------------
    # Read without specifying index_col so 'gene' is always a plain column regardless
    # of whether the CSV came from the API pipeline (index=False) or the legacy
    # pastml.py + GL_tab.py pipeline (written with a numeric index).
    pastml_df = pd.read_csv(args.pastml)

    # If 'gene' ended up as the index name rather than a column (e.g. read with
    # index_col=0 elsewhere), promote it back to a column.
    if 'gene' not in pastml_df.columns and pastml_df.index.name == 'gene':
        pastml_df = pastml_df.reset_index()

    # Counting method selection — intentionally not exposed as a CLI argument.
    # The preferred method is fixed here based on benchmarking results
    # (see dev/benchmark_reconstruction.py); users do not need to tune this.
    #
    # FLOW + ORIGINAL (pipeline default):
    #   Soft probability-flow rates: Σ max(0, P(child=1) − P(parent=1))
    #   with a P(parent=0)-weighted subsize denominator (IQR-filtered) and
    #   dist_marginal as the emergence threshold gate.
    #   Requires 'gains_flow' in the CSV, which is written by
    #   run_ancestral_reconstruction.py when --uncertainty=marginal
    #   (the Snakefile default).
    #
    # JOINT + ORIGINAL (legacy fallback):
    #   Hard discrete counts from the JOINT ML reconstruction.
    #   Used automatically when the CSV lacks 'gains_flow' — i.e. ACR was run
    #   with --uncertainty=threshold or via the legacy pastml CLI pipeline.
    if 'gains_flow' in pastml_df.columns:
        pastml_df = build_sim_params(pastml_df, counting='FLOW', subsize='ORIGINAL')
    else:
        pastml_df = build_sim_params(pastml_df, counting='JOINT', subsize='ORIGINAL')

    # ----------------------
    # Simulation Setup
    # ----------------------
    Sim = TreeSimulator(
        tree=args.tree,
        pastmlfile=pastml_df,
        obsdatafile=args.systems
    )

    print("Initializing SimPhyNI...")

    Sim.initialize_simulation_parameters(
        run_traits=args.run_traits,
        pre_filter=args.prefilter
    )

    # ----------------------
    # Run Simulation
    # ----------------------
    print("Running SimPhyNI analysis...")
    Sim.run_simulation(cores=args.cores)

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
        Sim.plot_results(pval_col='pval_naive', output_file=str(output_dir / 'heatmap_uncorrected.png'), figure_size=10)
        Sim.plot_results(pval_col='pval_bh', output_file=str(output_dir / 'heatmap_bh.png'), figure_size=10)
        Sim.plot_results(pval_col='pval_by', output_file=str(output_dir / 'heatmap_by.png'), figure_size=10)
        Sim.plot_results(pval_col='pval_bonf', output_file=str(output_dir / 'heatmap_bonf.png'), figure_size=10)


if __name__ == '__main__':
    main()
