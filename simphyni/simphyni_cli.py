#!/usr/bin/env python
import argparse
import subprocess
import sys
import os
import pandas as pd
import requests

__version__ = "0.1.2"

def main():
    parser = argparse.ArgumentParser(
        prog="simphyni",
        description="Wrapper for running SimPhyNISnakemake workflows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Show version information")

    # run command
    run_parser = subparsers.add_parser("run", help="Run SimPhyNI analysis")

    run_mode = run_parser#.add_mutually_exclusive_group(required=True)
    run_mode.add_argument(
        "-s","--samples",
        type=str,
        help="Path to samples.csv input file"
    )
    run_mode.add_argument(
        "-T", "--tree",
        type=str,
        help="Path to input tree file (.nwk)"
    )
    run_mode.add_argument(
        "-t", "--traits",
        type=str,
        help="Path to input traits file (.csv)"
    )

    run_parser.add_argument(
        "-r", "--runtype",
        choices=['0', '1'],
        default = '0',
        help="Run type: 0 (All Against All) or 1 (First Trait Against All) (default: 0)"
    )

    run_parser.add_argument(
        "--prefilter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable prefiltering (default: enabled)",
    )

    run_parser.add_argument(
        "-o", "--outdir",
        type=str,
        help="Output directory name (single-run mode only, default=simphyni_outs)"
    )

    run_parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable plotting of results (default: disabled)",
    )

    run_parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores to use (default=1)"
    )

    run_parser.add_argument(
        "--temp_dir",
        type=str,
        default='./',
        help="Location to put temporary files (defaults to a subdirectory in the current working directory)"
    )

    run_parser.add_argument(
        "--min_prev",
        type=float,
        default=0.05,
        help="Minimum prevanece required by a trait to be analyzed (recommended: 0.05)"
    )

    run_parser.add_argument(
        "--max_prev",
        type=float,
        default=0.95,
        help="Maximum prevanece allowed for a trait to be analyzed (recommended: 0.95)"
    )

    run_parser.add_argument(
        "snakemake_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed directly to snakemake"
    )

    examples_parser = subparsers.add_parser(
        "download-examples",
        help="Download example input files from GitHub"
    )

    args = parser.parse_args()

    if args.command == "version":
        print(f"SimPhyNI CLI version {__version__}")
        sys.exit(0)
    elif args.command == "download-examples":
        download_all_examples()
        sys.exit(0)

    # Path to Snakefile (assumed in repo root)
    snakefile_path = os.path.join(os.path.dirname(__file__), "Snakefile.py")

    cmd = [
        "snakemake",
        "-s", snakefile_path, 
        "--cores", str(args.cores),
    ]

    if args.samples:
        # Batch mode: pass samples file path
        print(f"Running batch mode with samples file: {args.samples}")
        cmd += ["--config", f"samples={args.samples}"]

    elif args.tree and args.traits and args.runtype:
        # Single-run mode: generate temporary samples.csv
        outdir = args.outdir or "simphyni_outs"
        single_run_file = "simphyni_sample_info.csv"
        print(f"Running single-run mode for tree: {args.tree} and traits: {args.traits}")
        df = pd.DataFrame([{
            "Sample": outdir,
            "Tree": args.tree,
            "Traits": args.traits,
            "RunType": args.runtype,
            "MinPrev": args.min_prev,
            "MaxPrev": args.max_prev,
        }])
        df.to_csv(single_run_file, index=False)

        cmd += ["--config", f"samples={single_run_file}"]

    else:
        sys.exit("Error: Must provide either --samples or --tree/--traits/--runtype")

    cmd += [f"temp_dir={args.temp_dir}", f"prefilter={args.prefilter}", f"plot={args.plot}"]

    if args.snakemake_args:
        cmd += args.snakemake_args


    print("Running workflow with command:")
    print(" ".join(cmd))
    print("This may take a while depending on the dataset size...\n")

    try:
        log_file = './simphyni_log.txt'
        with open(log_file, "w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("\nERROR: Snakemake workflow failed. Check the logs for details.")
        sys.exit(1)

    print("\nWorkflow completed successfully!")
    print(f"Results are in the output directory: 3-Objects")
    print("==========================")

EXAMPLES_DIR = os.path.join(os.getcwd(), "example_inputs")
GITHUB_EXAMPLES_URL = "https://github.com/jpeyemi/SimPhyNI/raw/master/example_inputs"

# List of example files available on GitHub
EXAMPLE_FILES = [
    "defense_systems_pivot.csv",
    "Sepi_megatree.nwk",
    "simphyni_sample_info.csv"
]

def download_example(name):
    """Download an example file from GitHub into the current working directory."""
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
    """Download all example files to the current working directory."""
    for example in EXAMPLE_FILES:
        download_example(example)
    print(f"All examples downloaded to {EXAMPLES_DIR}")

if __name__ == "__main__":
    main()
