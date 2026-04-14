"""
traits_io.py
============
Shared helpers for reading trait matrices from CSV or Parquet files.

All public functions dispatch on file extension (.csv vs .parquet) so callers
do not need to know which format is on disk.

PyArrow is used for Parquet reads because it supports column-projection directly
at the file-reader level: `pq.read_table(path, columns=[...])` only decompresses
the requested columns, leaving the rest untouched.  Row predicate pushdown
(`filters=[...]`) similarly skips row groups that don't match, enabling efficient
loading of a small subset of samples from a large matrix.

Polars was evaluated and rejected: its metadata parser overflows (i32 max)
on Parquet files with ~100k+ columns, which is the scale of this pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import loggamma

# Number of trait columns scanned per batch in chunked operations.
# At 5 000 cols × 5 000 rows × 1 byte ≈ 25 MB peak per iteration.
SCAN_CHUNK = 5_000


def _is_parquet(path: str) -> bool:
    return Path(path).suffix.lower() == ".parquet"


# ---------------------------------------------------------------------------
# Metadata (zero data I/O)
# ---------------------------------------------------------------------------

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
            # pandas writes the index at the END of schema names; use metadata
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


# ---------------------------------------------------------------------------
# Column loading (with optional row filtering)
# ---------------------------------------------------------------------------

def load_trait_columns(path: str, columns: list[str], index_col: str) -> pd.DataFrame:
    """
    Load a specific subset of trait columns from CSV or Parquet.

    For Parquet: only the requested columns (plus the index column) are
    decompressed; all other columns are skipped at the file-reader level.
    For CSV: uses pandas usecols to skip unneeded columns during parsing.

    Returns a DataFrame indexed by index_col with index dtype str.
    If columns is empty, returns a DataFrame with only the index.
    """
    return load_trait_columns_filtered(path, columns, index_col, row_ids=None)


def load_trait_columns_filtered(
    path: str,
    columns: list[str],
    index_col: str,
    row_ids: set[str] | None = None,
) -> pd.DataFrame:
    """
    Load specific columns with optional row predicate pushdown.

    For Parquet: pyarrow skips row groups that contain no matching rows before
    decompressing any data — peak RAM is O(n_matching_rows × n_cols).
    For CSV: pandas reads all rows then filters.

    row_ids: if provided, only rows whose index value is in this set are returned.

    Returns a DataFrame indexed by index_col with index dtype str.
    """
    if _is_parquet(path):
        cols = [index_col] + list(columns) if columns else [index_col]
        filters = [(index_col, "in", list(row_ids))] if row_ids else None
        table = pq.read_table(path, columns=cols, filters=filters)
        df = table.to_pandas()
        # pandas-written Parquet auto-restores the index from metadata
        if df.index.name != index_col and index_col in df.columns:
            df = df.set_index(index_col)
    else:
        usecols = [index_col] + list(columns) if columns else [index_col]
        df = pd.read_csv(path, index_col=0, usecols=usecols)
        if row_ids is not None:
            df = df[df.index.isin(row_ids)]
    df.index = df.index.astype(str)
    return df


# ---------------------------------------------------------------------------
# Chunked column sums
# ---------------------------------------------------------------------------

def compute_gene_sums(
    path: str,
    index_col: str,
    valid_index: set[str],
    scan_chunk: int = SCAN_CHUNK,
) -> dict[str, int]:
    """
    Compute the per-column sum (count of 1s) for every trait column, restricted
    to rows whose index value is in valid_index.

    Streams through columns in batches of scan_chunk using predicate pushdown so
    that peak RAM is proportional to scan_chunk * len(valid_index), not
    n_total_cols * n_total_rows.

    Returns {trait_name: int_sum} for all trait columns.
    """
    _, all_traits = get_trait_metadata(path)
    result: dict[str, int] = {}

    for i in range(0, len(all_traits), scan_chunk):
        batch = all_traits[i : i + scan_chunk]
        df = load_trait_columns_filtered(path, batch, index_col, row_ids=valid_index)
        for col in batch:
            if col in df.columns:
                result[col] = int(
                    pd.to_numeric(df[col], errors="coerce").fillna(0).sum()
                )

    return result


# ---------------------------------------------------------------------------
# Chunked Fisher-exact prefilter (no full-matrix load)
# ---------------------------------------------------------------------------

def prefilter_traits_chunked(
    path: str,
    index_col: str,
    valid_rows: set[str] | None,
    run_traits: int = 0,
    chunk_size: int = SCAN_CHUNK,
    pval_threshold: float = 0.05,
) -> list[str]:
    """
    Fisher-exact prefilter that never loads the full trait matrix into memory.

    Processes var columns in batches of chunk_size against target column batches
    of the same size.  Peak RAM = O(chunk_size × n_rows) independent of the
    total number of traits.

    run_traits == 0  →  all-by-all test (vars = targets = all traits)
    run_traits  > 0  →  first run_traits cols as vars vs. remaining as targets

    Returns the list of trait names that appear in at least one significant pair.
    """
    _, all_traits = get_trait_metadata(path)

    if run_traits > 0:
        var_traits = all_traits[:run_traits]
        target_traits = all_traits[run_traits:]
    else:
        var_traits = all_traits
        target_traits = all_traits

    def _logC(n: np.ndarray, k: np.ndarray) -> np.ndarray:
        return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

    surviving: set[str] = set()

    for vi in range(0, len(var_traits), chunk_size):
        vc = var_traits[vi : vi + chunk_size]
        vdf = load_trait_columns_filtered(path, vc, index_col, valid_rows)
        X = vdf.to_numpy().astype(bool)
        var_cols = np.array(vdf.columns)
        sX = X.sum(axis=0)
        del vdf

        for ti in range(0, len(target_traits), chunk_size):
            tc = target_traits[ti : ti + chunk_size]
            tdf = load_trait_columns_filtered(path, tc, index_col, valid_rows)
            Y = tdf.to_numpy().astype(bool)
            target_cols = np.array(tdf.columns)
            sY = Y.sum(axis=0)
            del tdf

            n = X.shape[0]
            a = X.T @ Y                         # (n_vc, n_tc)
            b = sX[:, None] - a
            c = sY[None, :] - a
            row1 = (a + b).astype(float)
            col1 = (a + c).astype(float)
            row2 = float(n) - row1
            a_f = a.astype(float)

            logp = _logC(row1, a_f) + _logC(row2, col1 - a_f) - _logC(float(n), col1)
            p_two = np.minimum(1.0, 2.0 * np.exp(logp))
            p_two[~np.isfinite(p_two)] = 1.0

            si, sj = np.where(p_two < pval_threshold)
            if si.size > 0:
                surviving.update(var_cols[si].tolist())
                surviving.update(target_cols[sj].tolist())

    return list(surviving)
