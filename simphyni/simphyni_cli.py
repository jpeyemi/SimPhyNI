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
    elif args.traits and args.tree and args.run_traits is not None:    
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
        sys.exit("Must provide either --samples OR -T, -t, and -r for single run.")

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
        f"save_object={args.save_object}"
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
    run_parser = subparsers.add_parser("run", help="Run SimPhyNI workflow")
    run_parser.add_argument("--samples", "-S", help="Path to CSV file with columns: Sample, Traits, Tree, RunType (batch mode)")
    run_parser.add_argument("-T", "--traits", help="Path to a single traits CSV file (single-run mode)")
    run_parser.add_argument("-t", "--tree", help="Path to a single tree file (single-run mode)")
    run_parser.add_argument("-r","--run-traits", default = 'ALL', help="Comma-separated list of column indices (0 is first trait) in traits CSV specifying traits for a traits against all comparison (Default: 'ALL' for all agianst all)")
    run_parser.add_argument("--sample-name", "-s", default = '', help="Sample name (single-run mode)")
    run_parser.add_argument("--min_prev",type=float,default=0.05,help="Minimum prevanece required by a trait to be analyzed (recommended: 0.05)")
    run_parser.add_argument("--max_prev",type=float,default=0.95,help="Maximum prevanece allowed for a trait to be analyzed (recommended: 0.95)")

    run_parser.add_argument("-o","--outdir", default="simphyni_outs", help="Main output directory (Default: simphyni_outs)")
    run_parser.add_argument("--temp-dir", default="tmp", help="Temporary directory for intermediate files (Default: tmp)")
    run_parser.add_argument("-c","--cores", type=int, help="Maximum cores for execution (Default: All when not provided)")
    # run_parser.add_argument("--prefilter", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable prefiltering (Default: enabled)")
    run_parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable plotting (Default: disabled)")
    run_parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without executing")
    run_parser.add_argument("--profile", help="Path to cluster profile folder for HPC usage")
    run_parser.add_argument("--save-object", action=argparse.BooleanOptionalAction, default=False, help="Saves parsable python object containing the complete analysis of each sample (Default: disabled)")
    run_parser.add_argument(
        "snakemake_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed directly to snakemake"
    )

    args = parser.parse_args()

    if args.command == "version":
        print(f"SimPhyNI CLI version {__version__}")
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
