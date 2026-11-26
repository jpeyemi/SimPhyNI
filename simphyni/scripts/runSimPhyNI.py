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
    parser.add_argument("--null", choices=["default", "legacy", "anchored", "anchored-screen"], default="default",
                        help="Null model. legacy: unweighted, simulated-vs-simulated pairs with --condition. anchored: GSC-weighted statistic, observed trait vs the "
                             "partner's null replicates in both directions (max p), lineage-"
                             "overlap gate on candidate pairs. anchored-screen: anchored without "
                             "the gate (default; fastest, highest recall, not robust to co-restriction).")
    parser.add_argument("--trials", type=int, default=512,
                        help="Null replicates per trait for --null anchored* (default: 512)")
    parser.add_argument("--combine", choices=["max", "mean"], default="max",
                        help="Combine the two anchored directions: max (default; valid if either "
                             "trait's null model is right) or mean (more recall, valid at 2x the level)")
    parser.add_argument("--lineage", choices=["none", "soft_auto", "soft", "mrca"], default="none",
                        help="Per-trait lineage accessibility in the null simulation (anchored nulls): "
                             "soft_auto = rate decays with distance from the trait's own MRCA clade, scale "
                             "chosen by single-trait posterior-predictive fit (default: none)")
    parser.add_argument("--tau", type=float, default=0.2, help="Decay scale for --lineage soft (fraction of mean tip depth)")
    parser.add_argument("--composite", action=argparse.BooleanOptionalAction, default=False,
                        help="Composite null: report max(p_whole-tree, p_pair-region) (valid if either null holds)")
    parser.add_argument("--gate-p", type=float, default=0.05,
                        help="Pairs with p below this get the lineage-overlap check (default: 0.05)")
    parser.add_argument("--refine-p", type=float, default=0.01,
                        help="Re-test pairs with p below this with --refine-n fresh replicates and a "
                             "GPD tail fitted to the top 1%% (default: 0.01; 0 disables)")
    parser.add_argument("--refine-max-pairs", type=int, default=5000,
                        help="At most this many smallest-p pairs are refined (default: 5000)")
    parser.add_argument("--refine-n", type=int, default=32768,
                        help="Replicates per trait in the refinement stage (default: 32768)")
    parser.add_argument("--output-format", choices=["csv", "parquet"], default="csv",
                        help="File format of the result tables (default: csv)")
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
    if args.null == "default":
        args.null = "anchored-screen"
    if args.null != "legacy":
        extra = dict(statistic="weighted", pvalue="anchored_fast", trials=args.trials,
                     anchor_combine=args.combine,
                     lineage=None if args.lineage == "none" else args.lineage, tau=args.tau,
                     composite=args.composite, sup_p=max(args.gate_p, 0.05),
                     gate=("overlap" if args.null == "anchored" else None), gate_p=args.gate_p,
                     refine_p=args.refine_p or None, refine_max_pairs=args.refine_max_pairs,
                     refine_n=args.refine_n)
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

    paths = Sim.write_outputs(output_dir, fmt=args.output_format)
    print("Simulation completed. Wrote " + ", ".join(str(p) for p in paths.values()))

    if args.plot:
        from simphyni.plotting import summary_plots
        long, _, traits = Sim._outputs
        query = list(Sim.run_trait_names) if args.run_traits > 0 and getattr(Sim, 'run_trait_names', None) else None
        made = summary_plots(long, traits, output_dir, query_traits=query)
        print("Plots: " + ", ".join(str(m) for m in made))


if __name__ == '__main__':
    main()
