"""
traits_io.py
============
Shared helpers for reading trait matrices from CSV or Parquet files.

All three public functions dispatch on file extension (.csv vs .parquet) so
callers do not need to know which format is on disk.

PyArrow is used for Parquet reads because it supports column-projection
directly at the file-reader level: `pq.read_table(path, columns=[...])` only
decompresses the requested columns, leaving the rest untouched.  This makes
selective column loading memory-efficient even on files with hundreds of
thousands of columns.

Polars was evaluated and rejected: its metadata parser overflows (i32 max)
on Parquet files with ~100k+ columns, which is the scale of this pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Number of trait columns scanned per batch in compute_gene_sums.
# At 5 000 cols × 5 000 rows × 1 byte ≈ 25 MB peak per iteration.
SCAN_CHUNK = 5_000


def _is_parquet(path: str) -> bool:
    return Path(path).suffix.lower() == ".parquet"


def get_trait_metadata(path: str) -> tuple[str, list[str]]:
    """
    Return (index_col_name, [trait_col_names]) with zero data I/O.

    For Parquet: reads only the file footer (schema), which is at most a few
    hundred KB even for 400k-column files.
    For CSV: reads only the header row (nrows=0).
    """
    if _is_parquet(path):
        schema = pq.read_schema(path)  # file footer only — no row data
        pandas_meta = schema.pandas_metadata
        if pandas_meta and pandas_meta.get("index_columns"):
            # Use pandas metadata to identify index columns (written at end of schema)
            idx_cols = [c for c in pandas_meta["index_columns"] if isinstance(c, str)]
            index_col = idx_cols[0] if idx_cols else schema.names[0]
            trait_cols = [n for n in schema.names if n not in set(idx_cols)]
        else:
            index_col = schema.names[0]
            trait_cols = list(schema.names[1:])
        return index_col, trait_cols
    else:
        df = pd.read_csv(path, nrows=0, index_col=0)
        index_name = df.index.name or ""
        return index_name, list(df.columns)


def load_trait_columns(path: str, columns: list[str], index_col: str) -> pd.DataFrame:
    """
    Load a specific subset of trait columns from CSV or Parquet.

    For Parquet: only the requested columns (plus the index column) are
    decompressed; all other columns are skipped at the file-reader level.
    For CSV: uses pandas usecols to skip unneeded columns during parsing.

    Returns a DataFrame indexed by index_col with index dtype str.
    If columns is empty, returns a DataFrame with only the index.

    Note: pandas-written Parquet files store the index in both the column data
    and the pandas metadata.  pq.read_table().to_pandas() auto-restores the
    index, so we only call set_index() when it has not already been applied.
    """
    if _is_parquet(path):
        cols_to_read = [index_col] + list(columns) if columns else [index_col]
        table = pq.read_table(path, columns=cols_to_read)
        df = table.to_pandas()
        # Auto-restored by pandas metadata — only set_index if still a column
        if df.index.name != index_col and index_col in df.columns:
            df = df.set_index(index_col)
    else:
        if columns:
            df = pd.read_csv(path, index_col=0, usecols=[index_col] + list(columns))
        else:
            df = pd.read_csv(path, index_col=0, usecols=[index_col])
    df.index = df.index.astype(str)
    return df


def compute_gene_sums(
    path: str,
    index_col: str,
    valid_index: set[str],
    scan_chunk: int = SCAN_CHUNK,
) -> dict[str, int]:
    """
    Compute the per-column sum (count of 1s) for every trait column, restricted
    to rows whose index value is in valid_index.

    Streams through columns in batches of scan_chunk so that peak RAM is
    proportional to scan_chunk * n_rows, not n_total_cols * n_rows.

    Returns {trait_name: int_sum} for all trait columns.
    """
    _, all_traits = get_trait_metadata(path)
    result: dict[str, int] = {}

    for i in range(0, len(all_traits), scan_chunk):
        batch = all_traits[i : i + scan_chunk]
        df = load_trait_columns(path, batch, index_col)
        df = df.loc[df.index.isin(valid_index)]
        for col in batch:
            if col in df.columns:
                result[col] = int(
                    pd.to_numeric(df[col], errors="coerce").fillna(0).sum()
                )

    return result
