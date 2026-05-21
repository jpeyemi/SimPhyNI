# scripts/reformat_csv.py
import sys
import re
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Ensure the scripts directory is on sys.path so traits_io can be imported
# both when run as a script (Snakemake) and when imported as a package module.
sys.path.insert(0, str(Path(__file__).parent))
from traits_io import SCAN_CHUNK, _is_parquet, load_trait_columns


def reformat_string_for_filepath(s):
    replacements = {
        ' ': '_', '\\': '', '/': '', ':': '', '*': '',
        '?': '', '"': '', '<': '', '>': '', '|': '', '.':'_', '~':'',
    }
    for key, value in replacements.items():
        s = s.replace(key, value)
    return re.sub(r'[^a-zA-Z0-9_.-]', '', s)


def reformat_columns(input_file, output_parquet, min_prev, max_prev, run_cols=None):
    """
    Read a traits matrix (CSV or Parquet), binarize, filter by prevalence, and
    write the result as Parquet.

    For Parquet input: uses a two-phase streaming approach so that peak RAM is
    proportional to SCAN_CHUNK * n_rows rather than n_total_cols * n_rows.
      Phase 1 — scan all columns in SCAN_CHUNK batches to identify which pass
                 the [min_prev, max_prev] prevalence filter.
      Phase 2 — load only the surviving columns and process them.

    For CSV input: loads the full file (existing behaviour).

    Output is always written as Parquet regardless of input format.
    """
    min_prev = float(min_prev)
    max_prev = float(max_prev)

    # --- Parse run_cols (protected column indices) -------------------------
    run_cols_indices: list[int] = []
    if run_cols and str(run_cols).strip():
        try:
            run_cols_indices = [int(x) for x in str(run_cols).split(",")]
        except ValueError:
            sys.exit(
                f"trait_cols must be a comma-separated list of integers, got: {run_cols}"
            )

    if _is_parquet(input_file):
        data = _reformat_parquet(
            input_file, min_prev, max_prev, run_cols_indices
        )
    else:
        data = _reformat_csv(
            input_file, min_prev, max_prev, run_cols_indices
        )

    data.to_parquet(output_parquet, index=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [reformat_string_for_filepath(col) for col in df.columns]
    return df


def _binarize(df: pd.DataFrame) -> pd.DataFrame:
    df[df > 0] = 1
    df.fillna(0, inplace=True)
    return df.astype(int)


def _apply_run_cols_order(df: pd.DataFrame, run_columns) -> pd.DataFrame:
    """Move run_columns to the front of the DataFrame."""
    if len(run_columns) == 0:
        return df
    other_columns = df.columns.difference(run_columns, sort=False)
    return df[list(run_columns) + list(other_columns)]


def _reformat_csv(input_file, min_prev, max_prev, run_cols_indices):
    data = pd.read_csv(input_file, index_col=0)
    data = _sanitize_columns(data)
    data = _binarize(data)

    run_columns = pd.Index([])
    if run_cols_indices:
        run_columns = data.columns[run_cols_indices]
        data = _apply_run_cols_order(data, run_columns)

    prevalence = data.mean(axis=0)
    mask = (
        (prevalence >= min_prev) & (prevalence <= max_prev)
    ) | data.columns.isin(run_columns)
    return data.loc[:, mask]


def _reformat_parquet(input_file, min_prev, max_prev, run_cols_indices):
    """
    Two-phase streaming reformat for Parquet input.

    Phase 1: scan all columns in SCAN_CHUNK batches to identify which pass
             the prevalence filter.  Only one batch is in RAM at a time.
    Phase 2: load only the surviving columns in a single read.
    """
    pf = pq.ParquetFile(input_file)
    schema = pf.schema_arrow
    pandas_meta = schema.pandas_metadata
    if pandas_meta and pandas_meta.get("index_columns"):
        idx_cols = [c for c in pandas_meta["index_columns"] if isinstance(c, str)]
        index_col = idx_cols[0] if idx_cols else schema.names[0]
        trait_cols_raw = [n for n in schema.names if n not in set(idx_cols)]
    else:
        index_col = schema.names[0]
        trait_cols_raw = schema.names[1:]
    n_rows = pf.metadata.num_rows

    # Resolve protected columns by position (before sanitization).
    # run_cols_indices refer to positions in the original trait column list.
    protected_raw: set[str] = set()
    if run_cols_indices:
        try:
            protected_raw = {trait_cols_raw[i] for i in run_cols_indices}
        except IndexError as e:
            sys.exit(f"run_cols index out of range: {e}")

    # Phase 1: prevalence scan — one SCAN_CHUNK at a time
    keep_raw: list[str] = []
    for i in range(0, len(trait_cols_raw), SCAN_CHUNK):
        batch_raw = trait_cols_raw[i : i + SCAN_CHUNK]
        table = pf.read(columns=batch_raw)
        for col in batch_raw:
            col_arr = table.column(col)
            col_sum = pc.sum(col_arr.cast("float32")).as_py() or 0.0
            prev = col_sum / n_rows if n_rows else 0.0
            if col in protected_raw or (min_prev <= prev <= max_prev):
                keep_raw.append(col)

    # Phase 2: load only the kept columns
    table2 = pf.read(columns=[index_col] + keep_raw)
    data = table2.to_pandas()
    # pandas metadata auto-restores the index; only call set_index if needed
    if data.index.name != index_col and index_col in data.columns:
        data = data.set_index(index_col)

    data = _sanitize_columns(data)
    data = _binarize(data)

    # Build sanitized run_columns set for ordering + mask protection
    run_columns = pd.Index([])
    if run_cols_indices:
        # Sanitized names of the protected columns
        protected_sanitized = [
            reformat_string_for_filepath(c) for c in protected_raw
        ]
        run_columns = pd.Index(
            [c for c in protected_sanitized if c in data.columns]
        )
        data = _apply_run_cols_order(data, run_columns)

    # Re-apply prevalence filter after binarization (values may change slightly
    # after int casting; also ensures protected columns are always kept).
    prevalence = data.mean(axis=0)
    mask = (
        (prevalence >= min_prev) & (prevalence <= max_prev)
    ) | data.columns.isin(run_columns)
    return data.loc[:, mask]


if __name__ == "__main__":
    reformat_columns(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        run_cols=sys.argv[5] if len(sys.argv) > 5 else None,
    )
