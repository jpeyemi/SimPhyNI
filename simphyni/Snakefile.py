##########################
# SNAKEFILE FOR SimPhyNI #
##########################

import os
import sys
import pandas as pd
import shutil
from importlib import resources

# Pre-Snakemake

with resources.as_file(resources.files("simphyni") / "scripts") as scripts_dir:
    SCRIPTS_DIRECTORY = str(scripts_dir)

with resources.as_file(resources.files("simphyni") / "envs") as envs_dir:
    ENVIRONMENT_DIRECTORY = str(envs_dir)

samples_file = config.get("samples")
if samples_file is None:
    raise ValueError("Must specify samples=... via --config")

samples = pd.read_csv(samples_file)
required_cols = {"Sample", "Traits", "Tree"}
if not required_cols.issubset(samples.columns):
    raise ValueError(f"Samples file must contain columns: {required_cols}")

outdir = config.get("directory")
base_tmp = config.get('temp_dir', './tmp')
prefilter = 'prefilter' if str(config.get('prefilter')).lower() == 'true' else 'no-prefilter'
plot = 'plot' if str(config.get('plot')).lower() == 'true' else 'no-plot'
save_object = 'save-object' if str(config.get('save_object')).lower() == 'true' else 'no-save-object'

samples["MinPrev"] = samples.get("MinPrev", 0.05)
samples["MaxPrev"] = samples.get("MaxPrev", 0.95)
samples["MinPrev"] = samples["MinPrev"].fillna(0.05)
samples["MaxPrev"] = samples["MaxPrev"].fillna(0.95)

SAMPLE_ls = samples['Sample']
OBS_ls = samples['Traits']
TREE_ls = samples['Tree']


TRAITS_ls_raw = samples.get("run_traits", pd.Series("ALL", index=samples.index))

def parse_trait_cols(val):
    """Convert a comma-separated string to a list of ints; use [] as default."""
    if pd.isna(val) or str(val).strip() == "" or str(val).strip() == "ALL":
        return []
    try:
        return [int(x) for x in str(val).split(",")]
    except ValueError:
        sys.exit(f"trait_cols must be ALL or a comma-separated list of integers, got: {val}")

TRAITS_ls = TRAITS_ls_raw.apply(parse_trait_cols)

run_dict = dict(zip(SAMPLE_ls, TRAITS_ls))
prev_dict = dict(zip(SAMPLE_ls, list(zip(samples['MinPrev'], samples['MaxPrev']))))

def copy_files_to_inputs(file_paths, name):
    input_dir = os.path.join(outdir, "inputs")
    os.makedirs(input_dir, exist_ok=True)
    for file_path, n in zip(file_paths,name):
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)
        destination_path = os.path.join(input_dir, f"{n}{ext}")
        if not os.path.exists(destination_path):
            shutil.copy(file_path, destination_path)
            print(f"Copied {file_path} to {destination_path}")
        else:
            print(f"File {file_name} already exists in {input_dir}, skipping copy.")

copy_files_to_inputs(OBS_ls, SAMPLE_ls)
copy_files_to_inputs(TREE_ls, SAMPLE_ls)

# Snakemake Rules

rule all:
    input:
        expand(f"{outdir}/{{sample}}/simphyni_results.csv", sample=SAMPLE_ls),

rule reformat_csv:
    threads: 1
    input:
        inp=f"{outdir}/inputs/{{sample}}.csv"
    output:
        out= f'{base_tmp}/{{sample}}/0-formatting/{{sample}}.csv'
    params:
        min_prev=lambda w: prev_dict.get(w.sample, (0.05, 0.95))[0],
        max_prev=lambda w: prev_dict.get(w.sample, (0.05, 0.95))[1],
        run_cols=lambda w: ",".join(map(str, run_dict.get(w.sample,[])))
    conda:
        f"{ENVIRONMENT_DIRECTORY}/simphyni.yaml"
    shell:
        "python {SCRIPTS_DIRECTORY}/reformat_csv.py {input.inp} {output.out} {params.min_prev} {params.max_prev} {params.run_cols}"

rule reformat_tree:
    threads: 1
    input:
        inp=f"{outdir}/inputs/{{sample}}.nwk"
    output:
        out=f"{base_tmp}/{{sample}}/0-formatting/{{sample}}.nwk"
    conda:
        f"{ENVIRONMENT_DIRECTORY}/simphyni.yaml"
    shell:
        "python {SCRIPTS_DIRECTORY}/reformat_tree.py {input.inp} {output.out}"

rule pastml:
    threads: 64
    input:
        inputsFile=rules.reformat_csv.output.out,
        tree=rules.reformat_tree.output.out
    output:
        outfile= f"{base_tmp}/{{sample}}/1-PastML/out.txt"
    params:
        outdir=lambda w: os.path.join(base_tmp, w.sample, "1-PastML"),
        max_workers=lambda wildcards, threads: threads,
        runtype=lambda w: len(run_dict.get(w.sample, []))
    conda:
        f"{ENVIRONMENT_DIRECTORY}/simphyni.yaml"
    shell:
        "python {SCRIPTS_DIRECTORY}/pastml.py "
        "--inputs_file {input.inputsFile} "
        "--tree_file {input.tree} "
        "--outdir {params.outdir} "
        "--max_workers {params.max_workers} "
        "--summary_file {output.outfile} "
        "-r {params.runtype} "
        "--{prefilter}"

rule aggregatepastml:
    threads: 64
    input:
        inputsFile=rules.pastml.input.inputsFile,
        tree=rules.pastml.input.tree,
        file=rules.pastml.output.outfile
    output:
        annotation=f"{base_tmp}/{{sample}}/2-Events/pastmlout.csv"
    params:
        pastml_folder = rules.pastml.params.outdir,
    conda:
        f"{ENVIRONMENT_DIRECTORY}/simphyni.yaml"
    shell:
        "python {SCRIPTS_DIRECTORY}/GL_tab.py "
        "{input.inputsFile} {input.tree} {params.pastml_folder} {output.annotation}"

rule SimPhyNI:
    threads: 64
    input:
        pastml=rules.aggregatepastml.output.annotation,
        systems=rules.reformat_csv.output.out,
        tree=rules.reformat_tree.output.out
    output:
        annotation=f"{outdir}/{{sample}}/simphyni_results.csv"
    params:
        outdir=f"{outdir}/{{sample}}/",
        runtype=lambda w: len(run_dict.get(w.sample, [])),
        threads=lambda wildcards, threads: threads

    conda:
        f"{ENVIRONMENT_DIRECTORY}/simphyni.yaml"
    shell:
        "python {SCRIPTS_DIRECTORY}/runSimPhyNI.py "
        "-p {input.pastml} -s {input.systems} -t {input.tree} "
        "-o {params.outdir} -r {params.runtype} "
        "-c {params.threads} "
        "--{prefilter} --{plot} "
        "--{save_object}"
