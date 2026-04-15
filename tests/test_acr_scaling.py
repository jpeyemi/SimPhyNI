"""
Tests for scaling improvements in run_ancestral_reconstruction.py:

  1. Worker-process tree cache (_worker_init / _WORKER_TREE)
  2. Chunked parallel processing with streaming CSV output
     - chunk_size < n_traits (multiple chunks)
     - chunk_size > n_traits (single chunk, same as original behaviour)
     - output correctness: same rows regardless of chunk size
  3. split_traits.py column-shard splitting
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ete3 import Tree

import simphyni.scripts.run_ancestral_reconstruction as acr_mod
from simphyni.scripts.run_ancestral_reconstruction import (
    _worker_init,
    label_internal_nodes,
    reconstruct_trait,
    compute_branch_upper_bound,
)

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

# A small 6-leaf tree with varied branch lengths
_NEWICK_6 = "((A:0.5,B:1.0)N1:0.5,(C:1.5,D:0.5)N2:1.0,(E:2.0,F:0.5)N3:0.5)Root;"

# 6 taxa, 8 traits — small enough for fast runs but has multiple chunks at chunk_size=3
_TRAIT_MATRIX = {
    "T1": [1, 1, 0, 0, 0, 0],  # cluster A/B
    "T2": [0, 0, 1, 1, 0, 0],  # cluster C/D
    "T3": [0, 0, 0, 0, 1, 1],  # cluster E/F
    "T4": [1, 0, 1, 0, 1, 0],  # every other
    "T5": [1, 1, 1, 0, 0, 0],  # first half
    "T6": [0, 0, 0, 1, 1, 1],  # second half
    "T7": [1, 0, 0, 0, 0, 1],  # tips A and F
    "T8": [0, 1, 1, 1, 1, 0],  # complement of T7
}
_TAXA = ["A", "B", "C", "D", "E", "F"]


@pytest.fixture(scope="module")
def trait_csv(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "traits.csv"
    df = pd.DataFrame(_TRAIT_MATRIX, index=_TAXA)
    df.index.name = "taxon"
    df.to_csv(p)
    return p


@pytest.fixture(scope="module")
def tree_nwk(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "tree.nwk"
    p.write_text(_NEWICK_6)
    return p


# ---------------------------------------------------------------------------
# Helper: run script via subprocess
# ---------------------------------------------------------------------------

_SCRIPT = str(Path(acr_mod.__file__).resolve())


def _run_acr(trait_csv, tree_nwk, out_csv, extra_args=()):
    """Run ACR via fast ACR engine (default — no --pastml)."""
    cmd = [
        sys.executable, _SCRIPT,
        "--inputs_file", str(trait_csv),
        "--tree_file",   str(tree_nwk),
        "--output_csv",  str(out_csv),
        "--max_workers", "2",
        "--reconstruction", "MPPA",
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def _run_acr_pastml(trait_csv, tree_nwk, out_csv, extra_args=()):
    """Run ACR via PastML engine (--pastml).  Use only 1-trait CSVs for speed."""
    cmd = [
        sys.executable, _SCRIPT,
        "--inputs_file", str(trait_csv),
        "--tree_file",   str(tree_nwk),
        "--output_csv",  str(out_csv),
        "--pastml",
        "--reconstruction", "all",
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


# ===========================================================================
# 1. Worker-process tree cache
# ===========================================================================

class TestWorkerInit:
    def setup_method(self):
        """Reset module globals before each test."""
        acr_mod._WORKER_TREE = None
        acr_mod._WORKER_UPPER_BOUND = None

    def teardown_method(self):
        acr_mod._WORKER_TREE = None
        acr_mod._WORKER_UPPER_BOUND = None

    def test_worker_init_sets_tree(self):
        """_worker_init should populate _WORKER_TREE in the calling process."""
        assert acr_mod._WORKER_TREE is None
        _worker_init(_NEWICK_6, 5.0)
        assert acr_mod._WORKER_TREE is not None
        assert isinstance(acr_mod._WORKER_TREE, Tree)

    def test_worker_init_sets_upper_bound(self):
        _worker_init(_NEWICK_6, 3.14)
        assert acr_mod._WORKER_UPPER_BOUND == pytest.approx(3.14)

    def test_worker_init_labels_internal_nodes(self):
        _worker_init(_NEWICK_6, 5.0)
        internal_names = [
            n.name for n in acr_mod._WORKER_TREE.traverse() if not n.is_leaf()
        ]
        assert all(name for name in internal_names), (
            "All internal nodes should be named after _worker_init"
        )

    def test_reconstruct_trait_works_with_cached_tree(self):
        """reconstruct_trait succeeds when _WORKER_TREE is pre-populated."""
        _worker_init(_NEWICK_6, 5.0)
        df_col = pd.Series(_TRAIT_MATRIX["T1"], index=_TAXA, name="T1")
        result = reconstruct_trait("T1", _NEWICK_6, df_col, 5.0, "MPPA", 2)
        assert result is not None
        assert "gains_flow" in result

    def test_reconstruct_trait_works_without_cached_tree(self):
        """reconstruct_trait should also work with no cache (fallback path)."""
        assert acr_mod._WORKER_TREE is None
        df_col = pd.Series(_TRAIT_MATRIX["T1"], index=_TAXA, name="T1")
        result = reconstruct_trait("T1", _NEWICK_6, df_col, 5.0, "MPPA", 2)
        assert result is not None
        assert "gains_flow" in result

    def test_cached_tree_copy_is_independent(self):
        """Each call to reconstruct_trait must get an independent tree copy."""
        _worker_init(_NEWICK_6, 5.0)
        # Run two traits in sequence; if tree state leaked the second would fail
        df1 = pd.Series(_TRAIT_MATRIX["T1"], index=_TAXA, name="T1")
        df2 = pd.Series(_TRAIT_MATRIX["T2"], index=_TAXA, name="T2")
        r1 = reconstruct_trait("T1", _NEWICK_6, df1, 5.0, "MPPA", 2)
        r2 = reconstruct_trait("T2", _NEWICK_6, df2, 5.0, "MPPA", 2)
        assert r1 is not None and r2 is not None
        # Different traits → different gain counts expected
        # (just verify both returned valid dicts)
        assert r1["gene"] == "T1"
        assert r2["gene"] == "T2"


# ===========================================================================
# 2. Chunked streaming — output correctness
# ===========================================================================

class TestChunkedStreaming:

    def test_output_contains_all_traits(self, trait_csv, tree_nwk, tmp_path):
        """All 8 traits should appear in the output regardless of chunk size."""
        out = tmp_path / "out.csv"
        r = _run_acr(trait_csv, tree_nwk, out, ["--chunk-size", "3"])
        assert r.returncode == 0, f"Script failed:\n{r.stderr}"
        df = pd.read_csv(out)
        assert set(df["gene"]) == set(_TRAIT_MATRIX.keys())

    def test_output_has_expected_row_count(self, trait_csv, tree_nwk, tmp_path):
        out = tmp_path / "out.csv"
        r = _run_acr(trait_csv, tree_nwk, out, ["--chunk-size", "2"])
        assert r.returncode == 0, r.stderr
        df = pd.read_csv(out)
        assert len(df) == len(_TRAIT_MATRIX)

    def test_chunk_size_larger_than_traits_still_works(self, trait_csv, tree_nwk, tmp_path):
        """chunk_size > n_traits should produce a single chunk (original behaviour)."""
        out = tmp_path / "out.csv"
        r = _run_acr(trait_csv, tree_nwk, out, ["--chunk-size", "10000"])
        assert r.returncode == 0, r.stderr
        df = pd.read_csv(out)
        assert len(df) == len(_TRAIT_MATRIX)

    def test_small_chunk_equals_large_chunk_output(self, trait_csv, tree_nwk, tmp_path):
        """Chunked and single-chunk runs should produce identical numeric output."""
        out_small = tmp_path / "small_chunk.csv"
        out_large = tmp_path / "large_chunk.csv"

        r1 = _run_acr(trait_csv, tree_nwk, out_small, ["--chunk-size", "2"])
        r2 = _run_acr(trait_csv, tree_nwk, out_large, ["--chunk-size", "10000"])

        assert r1.returncode == 0, r1.stderr
        assert r2.returncode == 0, r2.stderr

        df1 = pd.read_csv(out_small).set_index("gene").sort_index()
        df2 = pd.read_csv(out_large).set_index("gene").sort_index()

        # Same columns
        assert set(df1.columns) == set(df2.columns), (
            f"Column mismatch:\n  small={sorted(df1.columns)}\n  large={sorted(df2.columns)}"
        )
        # Same numeric values for every trait
        shared_cols = [c for c in df1.columns if pd.api.types.is_numeric_dtype(df1[c])]
        pd.testing.assert_frame_equal(
            df1[shared_cols], df2[shared_cols], check_like=True, atol=1e-9
        )

    def test_no_duplicate_rows_across_chunks(self, trait_csv, tree_nwk, tmp_path):
        out = tmp_path / "out.csv"
        r = _run_acr(trait_csv, tree_nwk, out, ["--chunk-size", "3"])
        assert r.returncode == 0, r.stderr
        df = pd.read_csv(out)
        assert df["gene"].nunique() == len(df), "Duplicate gene rows found"

    def test_csv_header_written_exactly_once(self, trait_csv, tree_nwk, tmp_path):
        """Streaming appends must not repeat the header line."""
        out = tmp_path / "out.csv"
        r = _run_acr(trait_csv, tree_nwk, out, ["--chunk-size", "2"])
        assert r.returncode == 0, r.stderr
        text = out.read_text()
        header_line = text.splitlines()[0]
        # Header should not appear in any subsequent line
        data_lines = text.splitlines()[1:]
        assert not any(line == header_line for line in data_lines), (
            "Header row appears more than once in output CSV"
        )

    def test_mppa_columns_present(self, trait_csv, tree_nwk, tmp_path):
        """MPPA mode must produce expected marginal columns."""
        out = tmp_path / "out.csv"
        r = _run_acr(trait_csv, tree_nwk, out, ["--chunk-size", "3"])
        assert r.returncode == 0, r.stderr
        df = pd.read_csv(out)
        for col in ("gains_flow", "losses_flow", "gain_subsize_marginal", "root_prob"):
            assert col in df.columns, f"Missing expected MPPA column: {col}"


# ===========================================================================
# 3. split_traits.py — shard splitting
# ===========================================================================

_SPLIT_SCRIPT = str(
    Path(acr_mod.__file__).parent / "split_traits.py"
)


class TestSplitTraits:

    def test_correct_number_of_shards(self, trait_csv, tmp_path):
        shard_dir = tmp_path / "shards"
        r = subprocess.run(
            [sys.executable, _SPLIT_SCRIPT,
             "--inputs_file", str(trait_csv),
             "--output_dir",  str(shard_dir),
             "--shard_size",  "3"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        shards = sorted(shard_dir.glob("*.csv"))
        # 8 traits / 3 per shard → 3 shards
        assert len(shards) == 3, f"Expected 3 shards, got {len(shards)}: {shards}"

    def test_shards_cover_all_traits(self, trait_csv, tmp_path):
        shard_dir = tmp_path / "shards"
        subprocess.run(
            [sys.executable, _SPLIT_SCRIPT,
             "--inputs_file", str(trait_csv),
             "--output_dir",  str(shard_dir),
             "--shard_size",  "3"],
            check=True, capture_output=True,
        )
        all_shard_cols = []
        for s in sorted(shard_dir.glob("*.csv")):
            all_shard_cols.extend(pd.read_csv(s, index_col=0).columns.tolist())
        assert sorted(all_shard_cols) == sorted(_TRAIT_MATRIX.keys())

    def test_shards_preserve_taxon_index(self, trait_csv, tmp_path):
        shard_dir = tmp_path / "shards"
        subprocess.run(
            [sys.executable, _SPLIT_SCRIPT,
             "--inputs_file", str(trait_csv),
             "--output_dir",  str(shard_dir),
             "--shard_size",  "4"],
            check=True, capture_output=True,
        )
        for s in shard_dir.glob("*.csv"):
            df = pd.read_csv(s, index_col=0)
            assert sorted(df.index.astype(str)) == sorted(_TAXA), (
                f"Shard {s.name} missing taxa"
            )

    def test_shard_size_larger_than_traits(self, trait_csv, tmp_path):
        """When shard_size ≥ n_traits, exactly one shard is created."""
        shard_dir = tmp_path / "shards"
        subprocess.run(
            [sys.executable, _SPLIT_SCRIPT,
             "--inputs_file", str(trait_csv),
             "--output_dir",  str(shard_dir),
             "--shard_size",  "1000"],
            check=True, capture_output=True,
        )
        shards = list(shard_dir.glob("*.csv"))
        assert len(shards) == 1

    def test_shard_acr_then_merge_roundtrip(self, trait_csv, tree_nwk, tmp_path):
        """Shard → ACR per shard → concat should match single-run output."""
        # Single run
        single_out = tmp_path / "single.csv"
        r = _run_acr(trait_csv, tree_nwk, single_out)
        assert r.returncode == 0, r.stderr

        # Sharded run
        shard_dir = tmp_path / "shards"
        subprocess.run(
            [sys.executable, _SPLIT_SCRIPT,
             "--inputs_file", str(trait_csv),
             "--output_dir",  str(shard_dir),
             "--shard_size",  "3"],
            check=True, capture_output=True,
        )
        shard_outs = []
        for shard_csv in sorted(shard_dir.glob("*.csv")):
            shard_out = tmp_path / f"acr_{shard_csv.stem}.csv"
            r = _run_acr(shard_csv, tree_nwk, shard_out)
            assert r.returncode == 0, f"Shard {shard_csv.name} failed:\n{r.stderr}"
            shard_outs.append(shard_out)

        merged = pd.concat([pd.read_csv(p) for p in shard_outs], ignore_index=True)
        single = pd.read_csv(single_out)

        merged = merged.set_index("gene").sort_index()
        single = single.set_index("gene").sort_index()

        assert set(merged.index) == set(single.index)
        shared_num = [c for c in merged.columns
                      if c in single.columns and pd.api.types.is_numeric_dtype(merged[c])]
        pd.testing.assert_frame_equal(
            merged[shared_num], single[shared_num], check_like=True, atol=1e-9
        )


# ===========================================================================
# 4. --pastml flag — PastML fallback smoke tests
#    Uses a single-trait CSV so these tests complete quickly.
# ===========================================================================

@pytest.fixture(scope="module")
def single_trait_csv(tmp_path_factory) -> Path:
    """CSV with one trait (T1) — keeps PastML tests fast."""
    p = tmp_path_factory.mktemp("pastml_data") / "single_trait.csv"
    df = pd.DataFrame({"T1": _TRAIT_MATRIX["T1"]}, index=_TAXA)
    df.index.name = "taxon"
    df.to_csv(p)
    return p


class TestPastMLFlag:
    """Tests for the --pastml engine fallback (single trait for speed)."""

    def test_pastml_runs_and_produces_output(self, single_trait_csv, tree_nwk, tmp_path):
        """--pastml flag runs successfully and writes a single-row CSV."""
        out = tmp_path / "pastml_out.csv"
        r = _run_acr_pastml(single_trait_csv, tree_nwk, out)
        assert r.returncode == 0, f"--pastml run failed:\n{r.stderr}"
        df = pd.read_csv(out)
        assert len(df) == 1
        assert df["gene"].iloc[0] == "T1"

    def test_pastml_has_core_columns(self, single_trait_csv, tree_nwk, tmp_path):
        """--pastml output includes JOINT + MPPA core columns."""
        out = tmp_path / "pastml_out.csv"
        r = _run_acr_pastml(single_trait_csv, tree_nwk, out)
        assert r.returncode == 0, r.stderr
        df = pd.read_csv(out)
        for col in ("gains", "losses", "root_state",
                    "gains_flow", "losses_flow", "gain_subsize_marginal", "root_prob"):
            assert col in df.columns, f"Missing column '{col}' in --pastml output"

    def test_pastml_has_path_columns(self, single_trait_csv, tree_nwk, tmp_path):
        """--pastml output includes _path suffix columns (PastML-specific)."""
        out = tmp_path / "pastml_out.csv"
        r = _run_acr_pastml(single_trait_csv, tree_nwk, out)
        assert r.returncode == 0, r.stderr
        df = pd.read_csv(out)
        path_cols = [c for c in df.columns if c.endswith("_path")]
        assert len(path_cols) > 0, (
            "Expected _path columns in --pastml output but found none"
        )

    def test_fast_acr_has_no_path_columns(self, single_trait_csv, tree_nwk, tmp_path):
        """Fast ACR output must NOT include any _path suffix columns."""
        out = tmp_path / "fast_out.csv"
        r = _run_acr(single_trait_csv, tree_nwk, out, ["--reconstruction", "all"])
        assert r.returncode == 0, r.stderr
        df = pd.read_csv(out)
        path_cols = [c for c in df.columns if c.endswith("_path")]
        assert len(path_cols) == 0, (
            f"Fast ACR should not produce _path columns; found: {path_cols}"
        )

    def test_short_with_fast_acr_produces_trimmed_output(self, single_trait_csv, tree_nwk, tmp_path):
        """--short without --pastml should succeed and write only core columns."""
        out = tmp_path / "fast_short.csv"
        r = _run_acr(single_trait_csv, tree_nwk, out,
                     extra_args=["--short", "--reconstruction", "all"])
        assert r.returncode == 0, f"--short with fast ACR failed:\n{r.stderr}"
        df = pd.read_csv(out)
        # Short 'all' mode: union of _short_joint + _short_mppa
        for col in ("gains", "losses", "gains_flow", "losses_flow",
                    "gain_subsize_marginal", "root_prob"):
            assert col in df.columns, f"Missing short column '{col}'"
        # Full-width columns should be absent
        assert "gain_subsize_nofilter" not in df.columns
        assert "gains_entropy" not in df.columns

    def test_pastml_and_fast_acr_shared_columns_are_finite_nonneg(
        self, single_trait_csv, tree_nwk, tmp_path
    ):
        """Common numeric columns between fast ACR and PastML must be finite and ≥ 0."""
        out_fast   = tmp_path / "fast.csv"
        out_pastml = tmp_path / "pastml.csv"
        r_f = _run_acr(single_trait_csv, tree_nwk, out_fast, ["--reconstruction", "all"])
        r_p = _run_acr_pastml(single_trait_csv, tree_nwk, out_pastml)
        assert r_f.returncode == 0, r_f.stderr
        assert r_p.returncode == 0, r_p.stderr

        df_fast   = pd.read_csv(out_fast)
        df_pastml = pd.read_csv(out_pastml)

        shared_num = [
            c for c in set(df_fast.columns) & set(df_pastml.columns)
            if pd.api.types.is_numeric_dtype(df_fast[c]) and c != "gene"
        ]
        for col in shared_num:
            assert np.isfinite(float(df_fast[col].iloc[0])),   f"fast ACR   {col} not finite"
            assert np.isfinite(float(df_pastml[col].iloc[0])), f"PastML     {col} not finite"
            assert float(df_fast[col].iloc[0])   >= 0,         f"fast ACR   {col} < 0"
            assert float(df_pastml[col].iloc[0]) >= 0,         f"PastML     {col} < 0"
