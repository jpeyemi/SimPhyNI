# scripts/reformat_csv.py
import sys
import pandas as pd
import re

def reformat_string_for_filepath(s):
    replacements = {
        ' ': '_', '\\': '', '/': '', ':': '', '*': '',
        '?': '', '"': '', '<': '', '>': '', '|': '', '.':'_', '~':'',
    }
    for key, value in replacements.items():
        s = s.replace(key, value)
    return re.sub(r'[^a-zA-Z0-9_.-]', '', s)

def reformat_columns(input_csv, output_csv, min_prev, max_prev, run_cols = None):
    data = pd.read_csv(input_csv, index_col=0)
    data.columns = [reformat_string_for_filepath(col) for col in data.columns]
    data[data > 0] = 1
    data.fillna(0, inplace=True)
    data = data.astype(int)

    run_columns = pd.Index([]) 
    if run_cols:
        try:
            run_cols_list = [int(x) for x in run_cols.split(",")]
            run_columns = data.columns[run_cols_list]
            other_columns = data.columns.difference(run_columns, sort=False)
            data = data[list(run_columns) + list(other_columns)]
        except ValueError:
            sys.exit(f"trait_cols must be a comma-separated list of integers, got: {run_cols}")

    
    # Compute prevalence (fraction of samples per column)
    prevalence = data.mean(axis=0)

    # Filter columns by prevalence but always keep run_columns
    mask = ((prevalence >= float(min_prev)) & (prevalence <= float(max_prev))) | data.columns.isin(run_columns)
    data = data.loc[:, mask]

    data.to_csv(output_csv)

if __name__ == "__main__":
    reformat_columns(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], run_cols=sys.argv[5] if len(sys.argv) > 5 else None)