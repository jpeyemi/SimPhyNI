##########################
# SNAKEFILE FOR SimPhyNI #
##########################

""" PRE-SNAKEMAKE """

# sampleFile = 'samples.csv' #run_type key: 0 - All against All, 1 - First against All
 #EX: '/home/iobal/orcd/c7/scratch/iobal/AssociationSnakemake/' 
# Empty string will put intermediaries in the same file systems as outputs
import sys
from importlib import resources
import os

SCRIPTS_DIRECTORY = resources.files("simphyni") / "scripts"
print(SCRIPTS_DIRECTORY)
sys.path.insert(0, SCRIPTS_DIRECTORY)


import os
import pandas as pd
import shutil

samples_file = config.get("samples", None)
if samples_file is None:
    raise ValueError("Must specify samples=... via --config")

samples = pd.read_csv(samples_file)

required_cols = {"Sample", "Traits", "Tree", "RunType"}
if not required_cols.issubset(samples.columns):
    raise ValueError(f"Samples file must contain columns: {required_cols}")

intermediariesDirectory = config.get('temp_dir','./')

SAMPLE_ls = samples['Sample']
OBS_ls = samples['Traits']
TREE_ls = samples['Tree']
RUNTYPE = samples['RunType']

if "MinPrev" not in samples.columns:
    samples["MinPrev"] = 0.05
if "MaxPrev" not in samples.columns:
    samples["MaxPrev"] = .95

samples["MinPrev"] = samples["MinPrev"].fillna(0.05)
samples["MaxPrev"] = samples["MaxPrev"].fillna(0.95)

MIN_PREVS = samples['MinPrev']
MAX_PREVS = samples['MaxPrev']

current_directory = os.getcwd()


def copy_files_to_inputs(file_paths, name):
    input_dir = "inputs/"
    os.makedirs(input_dir, exist_ok=True)
    for file_path, n in zip(file_paths,name):
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(input_dir, n + '.' + file_name.split('.')[-1])
        if not os.path.exists(destination_path):
            shutil.copy(file_path, destination_path)
            print(f"Copied {file_path} to {destination_path}")
        else:
            print(f"File {file_name} already exists in {input_dir}, skipping copy.")

copy_files_to_inputs(OBS_ls, SAMPLE_ls)
copy_files_to_inputs(TREE_ls, SAMPLE_ls)

run_dict = dict(zip(SAMPLE_ls,RUNTYPE))
prev_dict = dict(zip(SAMPLE_ls,list(zip(MIN_PREVS,MAX_PREVS))))


''' SNAKEMAKE '''

rule all:
    input:
        expand("0-formatting/{sample}/{sample}.csv", zip, sample = SAMPLE_ls),
        expand("0-formatting/{sample}/{sample}.nwk", zip, sample = SAMPLE_ls),
        expand("1-PastML/{sample}/out.txt", sample = SAMPLE_ls),
        expand("2-Events/{sample}/pastmlout.csv", sample = SAMPLE_ls),
        expand("3-Objects/{sample}/simphyni_results.csv", sample = SAMPLE_ls),


rule reformat_csv:
    input:
        inp = 'inputs/{sample}.csv'
    output:
        out = '0-formatting/{sample}/{sample}.csv'
    params:
        min_prev = lambda wildcards: prev_dict.get(wildcards.sample, 0)[0],
        max_prev = lambda wildcards: prev_dict.get(wildcards.sample, 0)[1]
    conda:
        'envs/simphyni.yaml'
    shell:
        'python {SCRIPTS_DIRECTORY}/reformat_csv.py {input.inp} {output.out} {params.min_prev} {params.max_prev}'

rule reformat_tree:
    input:
        inp = 'inputs/{sample}.nwk'
    output:
        out = '0-formatting/{sample}/{sample}.nwk'
    conda:
        'envs/simphyni.yaml'
    shell:
        'python {SCRIPTS_DIRECTORY}/reformat_tree.py {input.inp} {output.out}'


rule pastml:
    input:
        inputsFile = rules.reformat_csv.output.out,
        tree = rules.reformat_tree.output.out
    output:
        outfile = "1-PastML/{sample}/out.txt",
    params:
        outdir = lambda wildcards: os.path.join(intermediariesDirectory,'1-PastML',wildcards.sample),
        max_workers = 64
    conda:
        "envs/simphyni.yaml"
    shell:
        "python {SCRIPTS_DIRECTORY}/pastml.py \
            --inputs_file {input.inputsFile}\
            --tree_file {input.tree}\
            --outdir {params.outdir}\
            --max_workers {params.max_workers}\
            --summary_file {output.outfile}\
        "

rule aggregatepastml:
    input:
        inputsFile = rules.pastml.input.inputsFile,
        tree = rules.pastml.input.tree,
        file = rules.pastml.output.outfile,
    output:
        annotation = "2-Events/{sample}/pastmlout.csv"
    params:
        pastml_folder = rules.pastml.params.outdir,
    conda:
        'envs/simphyni.yaml'
    shell:
        "python {SCRIPTS_DIRECTORY}/GL_tab.py {input.inputsFile} {input.tree} {params.pastml_folder} {output.annotation}"

rule SimPhyNI:
    input:
        pastml = "2-Events/{sample}/pastmlout.csv",
        systems = "0-formatting/{sample}/{sample}.csv",
        tree = "0-formatting/{sample}/{sample}.nwk"
    output:
        annotation = "3-Objects/{sample}/simphyni_results.csv"
    params:
        outdir = "3-Objects/{sample}/",
        runtype = lambda wildcards: run_dict.get(wildcards.sample, 0)
    conda:
        'envs/simphyni.yaml'
    shell:
        "python {SCRIPTS_DIRECTORY}/runSimPhyNI.py \
            -p {input.pastml} \
            -s {input.systems} \
            -t {input.tree} \
            -o {params.outdir} \
            -r {params.runtype} \
        "

