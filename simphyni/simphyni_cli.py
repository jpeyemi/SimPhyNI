#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import subprocess

__version__ = "1.0.2"

EXAMPLES_DIR = os.path.join(os.getcwd(), "example_inputs")
GITHUB_EXAMPLES_URL = "https://github.com/jpeyemi/SimPhyNI/raw/master/example_inputs"
EXAMPLE_FILES = [
    "defense_systems_pivot.csv",
    "Sepi_megatree.nwk",
    "simphyni_sample_info.csv"
]

def download_example(name):
    import requests
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    url = f"{GITHUB_EXAMPLES_URL}/{name}"
    local_path = os.path.join(EXAMPLES_DIR, name)
    if not os.path.exists(local_path):
        print(f"Downloading {name} to {local_path}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    else:
        print(f"{name} already exists at {local_path}")
    return local_path

def download_all_examples():
    for example in EXAMPLE_FILES:
        download_example(example)
    print(f"All examples downloaded to {EXAMPLES_DIR}")


CLUST_DIR = os.path.join(os.getcwd(), "cluster_profile")
GITHUB_CLUSTER_URL = "https://github.com/jpeyemi/SimPhyNI/raw/master/cluster_profile"
CLUSTER_FILES = [
    "config.yaml",
    "run_simphyni.sh"
]

def download_cluster_files(name):
    import requests
    os.makedirs(CLUST_DIR, exist_ok=True)
    url = f"{GITHUB_CLUSTER_URL}/{name}"
    local_path = os.path.join(CLUST_DIR, name) if name != 'run_simphyni.sh' else os.path.join(os.getcwd(), name)
    if not os.path.exists(local_path):
        print(f"Downloading {name} to {local_path}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    else:
        print(f"{name} already exists at {local_path}")
    return local_path

def download_all_cluster_files():
    for file in CLUSTER_FILES:
        download_cluster_files(file)
    print(f"All examples downloaded to {CLUST_DIR}")


def run_simphyni(args):
    # Determine samples
    if args.samples:
        samples_file = os.path.abspath(args.samples)
        if not os.path.exists(samples_file):
            sys.exit(f"Samples file not found: {samples_file}")
        samples = pd.read_csv(samples_file)
    elif args.traits and args.tree:    
        sample_name = args.sample_name or os.path.splitext(os.path.basename(args.traits))[0]
        samples = pd.DataFrame([{
            "Sample": sample_name,
            "Traits": os.path.abspath(args.traits),
            "Tree": os.path.abspath(args.tree),
            "run_traits": args.run_traits,
            "MinPrev": args.min_prev,
            "MaxPrev": args.max_prev,
        }])
    else:
        sys.exit("Provide either --samples (batch) or --traits and --tree (single run). See: simphyni run --help")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    samples_file_out = os.path.join(outdir, "simphyni_sample_info.csv")
    samples.to_csv(samples_file_out, index=False)

    if not getattr(args, "temp_dir", None) or args.temp_dir == "tmp":
        args.temp_dir = os.path.join(args.outdir, "tmp")

    # Snakemake config args
    config_args = [
        f"samples={samples_file_out}",
        f"temp_dir={args.temp_dir}",
        f"prefilter=False", #{args.prefilter}", #Top level removal of prefilter funtion due to variable performance and minimal efficiency gains. May reimplement in later version
        f"plot={args.plot}",
        f"directory={outdir}",
        f"save_object={args.save_object}",
        f"include_flagged={args.include_flagged}",
        f"seed={getattr(args, 'seed', None) if getattr(args, 'seed', None) is not None else ''}",
        f"emergence={getattr(args, 'emergence', 'none')}",
        f"condition={getattr(args, 'condition', 'prevalence')}",
        f"oversample={getattr(args, 'oversample', 8)}",
        f"branch_cap={getattr(args, 'branch_cap', 'iqr')}",
        f"eligibility={getattr(args, 'eligibility', 'none')}",
        f"null={getattr(args, 'null', 'default')}",
        f"trials={getattr(args, 'trials', 512)}",
        f"gate_p={getattr(args, 'gate_p', 0.05)}",
        f"combine={getattr(args, 'combine', 'max')}",
        f"lineage={getattr(args, 'lineage', 'none')}",
        f"composite={getattr(args, 'composite', False)}",
        f"refine_p={getattr(args, 'refine_p', 0.01)}",
        f"refine_max_pairs={getattr(args, 'refine_max_pairs', 5000)}",
        f"refine_n={getattr(args, 'refine_n', 32768)}",
        f"output_format={getattr(args, 'output_format', 'csv')}",
    ]

    # profile flag handling
    extra_args = []
    if args.profile:
        extra_args += [
            "--profile", args.profile,
        ]
    
    if args.cores:
        extra_args += ["--cores", str(args.cores)]
    else:
        extra_args += ["--cores", 'all']



    snakefile_path = os.path.join(os.path.dirname(__file__), "Snakefile.py") 
    # Snakemake command
    snakemake_cmd = [
        "snakemake",
        "--snakefile", snakefile_path,
        "--rerun-incomplete",
        "--printshellcmds",
        "--nolock",
        *extra_args,
        *args.snakemake_args,
        "--config",
        *config_args
    ]

    if args.dry_run:
        snakemake_cmd.insert(1, "--dry-run")

    print("Launching SimPhyNI...")
    print("Output directory:", outdir)
    print("Snakemake command:", " ".join(snakemake_cmd), "\n")

    try:
        subprocess.run(snakemake_cmd, check=True)
        print("\nSimPhyNI completed successfully.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"\n SimPhyNI failed with error: {e}")

class _Fmt(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults, but not for options without a meaningful one."""
    def _get_help_string(self, action):
        if action.default in (None, "", False) or action.default is argparse.SUPPRESS:
            return action.help
        return super()._get_help_string(action)


def main():
    parser = argparse.ArgumentParser(prog="simphyni", description="SimPhyNI — Simulation-based Phylogenetic iNteraction Inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Version
    subparsers.add_parser("version", help="Show SimPhyNI version")

    # Download examples
    subparsers.add_parser("download-examples", help="Download example input files from GitHub")

    # Download template cluster scripts
    subparsers.add_parser("download-cluster-profile", help="Download template cluster profile from GitHub")

    # Run
    run_parser = subparsers.add_parser(
        "run", help="Run SimPhyNI on one tree + trait table, or a batch of them",
        description="Test every pair of binary traits (or query traits against all others) for "
                    "co-occurrence or avoidance beyond what the phylogeny explains. "
                    "See docs/USER_GUIDE.md for details.",
        formatter_class=_Fmt)
    S = argparse.SUPPRESS

    inp = run_parser.add_argument_group("input")
    inp.add_argument("-T", "--traits", help="Trait table (.csv or .parquet): rows = tips, columns = binary traits (single run)")
    inp.add_argument("-t", "--tree", help="Rooted Newick tree whose tip names match the trait table's row names (single run)")
    inp.add_argument("-S", "--samples", help="Batch mode: CSV with columns Sample, Tree, Traits [, run_traits, MinPrev, MaxPrev]")
    inp.add_argument("-s", "--sample-name", default="", help="Name for a single run; if omitted, the trait file name")
    inp.add_argument("-r", "--run-traits", default="ALL",
                     help="ALL = every pair; or comma-separated indices of query traits (0 = first column) "
                          "to test against all others")
    inp.add_argument("--min-prev", "--min_prev", dest="min_prev", type=float, default=0.05,
                     help="Drop traits present in fewer than this fraction of tips")
    inp.add_argument("--max-prev", "--max_prev", dest="max_prev", type=float, default=0.95,
                     help="Drop traits present in more than this fraction of tips")

    out = run_parser.add_argument_group("output")
    out.add_argument("-o", "--outdir", default="simphyni_outs", help="Output directory (one subfolder per sample)")
    out.add_argument("--output-format", choices=["csv", "parquet"], default="csv",
                     help="Format of the result tables; parquet is much smaller for > 1 M pairs")
    out.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False,
                     help="Write summary plots (volcano, heatmap, p-value histogram, top traits)")
    out.add_argument("--save-object", action=argparse.BooleanOptionalAction, default=False,
                     help="Also save the analysis object as a .pkl (avoid for > 1 M pairs)")
    out.add_argument("--temp-dir", default="tmp", help="Directory for intermediate files; 'tmp' means <outdir>/tmp")

    st = run_parser.add_argument_group("statistics")
    st.add_argument("--null", choices=["default", "legacy", "anchored", "anchored-screen"], default="default",
                    metavar="{default,legacy}",
                    help="default = phylogenetically weighted anchored null with calibrated tails; "
                         "legacy = the SimPhyNI <= 2.0 null (unweighted, prevalence-conditioned)")
    st.add_argument("--refine-p", type=float, default=0.01,
                    help="Re-test pairs with p below this using --refine-n fresh replicates (0 disables)")
    st.add_argument("--refine-max-pairs", type=int, default=5000, help="At most this many pairs are re-tested")
    st.add_argument("--refine-n", type=int, default=32768, help="Replicates per trait when re-testing")
    st.add_argument("--trials", type=int, default=512, help="Null replicates per trait")
    st.add_argument("--composite", action=argparse.BooleanOptionalAction, default=False,
                    help="Optional: also require significance under a null confining both traits to their "
                         "clades (reports max p). Not needed genome-wide; see the user guide")
    st.add_argument("--include-flagged", action=argparse.BooleanOptionalAction, default=True,
                    help="Keep pairs with traits whose null may be miscalibrated (marked null_calibrated=False)")
    st.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results")
    st.add_argument("--branch-cap", choices=["iqr", "none"], default="iqr", help=S)
    # developer / legacy options: accepted, not shown in --help (see docs/USER_GUIDE.md, "Advanced options")
    st.add_argument("--lineage", choices=["none", "soft_auto", "soft", "mrca"], default="none", help=S)
    st.add_argument("--combine", choices=["max", "mean"], default="max", help=S)
    st.add_argument("--gate-p", type=float, default=0.05, help=S)
    st.add_argument("--emergence", choices=["none", "dist", "scaled", "soft", "mrca"], default="none", help=S)
    st.add_argument("--condition", choices=["prevalence", "footprint", "none"], default="prevalence", help=S)
    st.add_argument("--oversample", type=int, default=8, help=S)
    st.add_argument("--eligibility", choices=["none", "phylogroup", "adaptive"], default="none", help=S)

    ex = run_parser.add_argument_group("execution")
    ex.add_argument("-c", "--cores", type=int, help="Maximum cores; if omitted, all available")
    ex.add_argument("--profile", help="Snakemake profile folder for HPC clusters (see download-cluster-profile)")
    ex.add_argument("--dry-run", action="store_true", help="Show the jobs that would run, without running them")
    run_parser.add_argument("snakemake_args", nargs=argparse.REMAINDER, help="Extra arguments passed to snakemake")

    args = parser.parse_args()

    if args.command == "version":
        print(f"SimPhyNI version {__version__}")
        sys.exit(0)
    elif args.command == "download-examples":
        download_all_examples()
        sys.exit(0)
    elif args.command == "download-cluster-profile":
        download_all_cluster_files()
        sys.exit(0)
    elif args.command == "run":
        run_simphyni(args)

if __name__ == "__main__":
    main()
