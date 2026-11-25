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
    parser.add_argument(
        "--include-flagged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Test traits whose null distribution may be miscalibrated (sparse events "
            "with highly asymmetric eligible regions). Their pairs are simulated "
            "normally and marked null_calibrated=False in the output. "
            "--no-include-flagged drops them from the results instead. "
            "(Default: enabled)"
        ),
    )
    parser.add_argument("--emergence", choices=["dist", "none", "scaled", "soft", "mrca"],
                        default="none", help="Emergence-threshold handling (default: none)")
    parser.add_argument("--emergence-scale", type=float, default=1.0,
                        help="Threshold multiplier for --emergence scaled (default: 1.0)")
    parser.add_argument("--condition", choices=["none", "prevalence", "footprint"], default="prevalence",
                        help="Condition each trait's null on the observed trait: prevalence (default) or footprint (prevalence + parsimony changes)")
    parser.add_argument("--oversample", type=int, default=8,
                        help="Replicates simulated per kept replicate with --condition prevalence")
    parser.add_argument("--eligibility", choices=["none", "phylogroup", "adaptive"], default="none",
                        help="Restrict each trait's null to the phylogroups it occupies (default: none)")
    parser.add_argument("--phylogroup-depth", type=float, default=0.2,
                        help="Tree-depth fraction at which phylogroups are cut (default: 0.2)")
    parser.add_argument("--adapt-alpha", type=float, default=0.05,
                        help="Exclusion threshold for --eligibility adaptive (default: 0.05)")
    parser.add_argument("--branch-cap", choices=["none", "iqr"], default="iqr",
                        help="Cap branch lengths in simulation as in the rate denominators (default: iqr)")
    parser.add_argument("--null", choices=["legacy", "anchored", "anchored-screen"], default="anchored-screen",
                        help="Null model. legacy: unweighted, simulated-vs-simulated pairs with --condition. anchored: GSC-weighted statistic, observed trait vs the "
                             "partner's null replicates in both directions (max p), lineage-"
                             "overlap gate on candidate pairs. anchored-screen: anchored without "
                             "the gate (default; fastest, highest recall, not robust to co-restriction).")
    parser.add_argument("--trials", type=int, default=512,
                        help="Null replicates per trait for --null anchored* (default: 512)")
    parser.add_argument("--combine", choices=["max", "mean"], default="max",
                        help="Combine the two anchored directions: max (default; valid if either "
                             "trait's null model is right) or mean (more recall, valid at 2x the level)")
    parser.add_argument("--gate-p", type=float, default=0.05,
                        help="Pairs with p below this get the lineage-overlap check (default: 0.05)")
    parser.add_argument("--refine-p", type=float, default=None,
                        help="Re-test pairs with p below this by exact Monte-Carlo counts on fresh "
                             "replicates (adaptive up to --refine-max replicates). Default: off")
    parser.add_argument("--refine-max", type=int, default=131072,
                        help="Largest refinement stage (default: 131072 replicates)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible simulations (default: unseeded)")
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
    #   run_ancestral_reconstruction.py when --reconstruction=MPPA or all
    #   (the Snakefile default).
    #
    # JOINT + ORIGINAL (legacy fallback):
    #   Hard discrete counts from the JOINT ML reconstruction.
    #   Used automatically when the CSV lacks 'gains_flow' — i.e. ACR was run
    #   with --reconstruction=JOINT or via the legacy pastml CLI pipeline.
    # if 'gains_flow' in pastml_df.columns:
    #     pastml_df = build_sim_params(pastml_df, counting='FLOW', subsize='ORIGINAL')
    # else:
    pastml_df = build_sim_params(pastml_df, counting='FLOW', subsize='ORIGINAL',no_threshold=False)

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
    extra = dict(statistic="logor", pvalue="kde", trials=64)
    if args.null != "legacy":
        stages = [s for s in (4096, 32768, 131072, 524288) if s <= args.refine_max] or [args.refine_max]
        extra = dict(statistic="weighted", pvalue="anchored_fast", trials=args.trials,
                     anchor_combine=args.combine,
                     gate=("overlap" if args.null == "anchored" else None), gate_p=args.gate_p,
                     refine_p=args.refine_p, adaptive_stages=stages)
        args.condition = "none"
    Sim.run_simulation(cores=args.cores, gamma=False, include_flagged=args.include_flagged,
                       use_clade_mask=False, seed=args.seed,
                       emergence=args.emergence, emergence_scale=args.emergence_scale,
                       condition=None if args.condition == 'none' else args.condition,
                       oversample=args.oversample,
                       eligibility=None if args.eligibility == 'none' else args.eligibility,
                       phylogroup_depth=args.phylogroup_depth,
                       adapt_alpha=args.adapt_alpha,
                       branch_cap=None if args.branch_cap == 'none' else args.branch_cap,
                       **extra)

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

    PVAL_COLS = {
        'uncorrected': 'pval_naive',
        'bh':          'pval_bh',
        'by':          'pval_by',
        'bonf':        'pval_bonf',
    }

    if args.plot:
        if args.run_traits > 0 and Sim.run_trait_names:
            # Specific traits mode: volcano per trait+method, rectangular heatmap per method
            for trait in Sim.run_trait_names:
                safe = trait.replace('/', '_').replace(' ', '_')
                for label, col in PVAL_COLS.items():
                    Sim.plot_volcano(
                        trait=trait,
                        pval_col=col,
                        output_file=str(output_dir / f'volcano_{safe}_{label}.png')
                    )
            for label, col in PVAL_COLS.items():
                Sim.plot_heatmap_subset(
                    run_traits=Sim.run_trait_names,
                    pval_col=col,
                    output_file=str(output_dir / f'heatmap_subset_{label}.png')
                )
        else:
            # All-by-all mode: scalable symmetric heatmaps
            for label, col in PVAL_COLS.items():
                Sim.plot_results(
                    pval_col=col,
                    output_file=str(output_dir / f'heatmap_{label}.png'),
                    figure_size=-1
                )


if __name__ == '__main__':
    main()
