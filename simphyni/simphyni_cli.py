#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        prog="simphyni",
        description="Wrapper for running SimPhyNISnakemake workflows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        help="Run type: 0 (AllAgainstAll) or 1 (FirstAgainstAl)"
    )

    run_parser.add_argument(
        "-o", "--outdir",
        type=str,
        help="Output directory name (single-run mode only, default=simphyni_outs)"
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
        default='',
        help="Location to put temporary files (defaults to a subdirectory in the current working directory)"
    )

    run_parser.add_argument(
        "snakemake_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed directly to snakemake"
    )

    args = parser.parse_args()

    # Path to Snakefile (assumed in repo root)
    snakefile_path = os.path.join(os.path.dirname(__file__), "Snakefile.py")

    cmd = [
        "snakemake",
        "-s", snakefile_path, 
        "--cores", str(args.cores)
    ]

    if args.samples:
        # Batch mode: pass samples file path
        cmd += ["--config", f"samples={args.samples}",f"temp_dir={args.temp_dir}"]

    elif args.tree and args.traits and args.runtype:
        # Single-run mode: generate temporary samples.csv
        outdir = args.outdir or "simphyni_outs"
        single_run_file = "single_run_samples.csv"
        df = pd.DataFrame([{
            "Sample": outdir,
            "Tree": args.tree,
            "Traits": args.traits,
            "RunType": args.runtype,
        }])
        df.to_csv(single_run_file, index=False)

        cmd += ["--config", f"samples={single_run_file}", f"temp_dir={args.temp_dir}"]

    else:
        sys.exit("Error: Must provide either --samples or --tree/--traits/--runtype")

    if args.snakemake_args:
        cmd += args.snakemake_args

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
