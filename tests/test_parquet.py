"""
test_parquet.py
===============
Tests for Parquet input support across the SimPhyNI pipeline.

Covers:
  1. traits_io helpers (get_trait_metadata, load_trait_columns, compute_gene_sums)
  2. reformat_columns() — CSV and Parquet inputs produce identical Parquet outputs
  3. split_traits — Parquet input produces correct CSV shards
  4. run_ancestral_reconstruction — lazy per-chunk loading matches direct load
  5. End-to-end: CSV vs Parquet through reformat + one ACR call

Reuses _NEWICK_6 / _TRAIT_MATRIX / _TAXA from test_acr_scaling.py.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ete3 import Tree

# ---------------------------------------------------------------------------
# Shared synthetic data (mirrors test_acr_scaling.py)
# ---------------------------------------------------------------------------

_NEWICK_6 = "((A:0.5,B:1.0)N1:0.5,(C:1.5,D:0.5)N2:1.0,(E:2.0,F:0.5)N3:0.5)Root;"

_TRAIT_MATRIX = {
    "T1": [1, 1, 0, 0, 0, 0],
    "T2": [0, 0, 1, 1, 0, 0],
    "T3": [0, 0, 0, 0, 1, 1],
    "T4": [1, 0, 1, 0, 1, 0],
    "T5": [1, 1, 1, 0, 0, 0],
    "T6": [0, 0, 0, 1, 1, 1],
    "T7": [1, 0, 0, 0, 0, 1],
    "T8": [0, 1, 1, 1, 1, 0],
}
_TAXA = ["A", "B", "C", "D", "E", "F"]


@pytest.fixture(scope="module")
def trait_df() -> pd.DataFrame:
    df = pd.DataFrame(_TRAIT_MATRIX, index=_TAXA)
    df.index.name = "taxon"
    return df


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory, trait_df) -> Path:
    p = tmp_path_factory.mktemp("data") / "traits.csv"
    trait_df.to_csv(p)
    return p


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory, trait_df) -> Path:
    p = tmp_path_factory.mktemp("data") / "traits.parquet"
    trait_df.to_parquet(p, index=True)
    return p


@pytest.fixture(scope="module")
def tree_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "tree.nwk"
    p.write_text(_NEWICK_6)
    return p


# ---------------------------------------------------------------------------
# 1. traits_io helpers
# ---------------------------------------------------------------------------

import simphyni.scripts.traits_io as tio  # noqa: E402  (also registers scripts/ on sys.path)


class TestTraitsIO:
    def test_get_trait_metadata_csv(self, csv_path):
        index_col, traits = tio.get_trait_metadata(str(csv_path))
        assert index_col == "taxon"
        assert traits == list(_TRAIT_MATRIX.keys())

    def test_get_trait_metadata_parquet(self, parquet_path):
        index_col, traits = tio.get_trait_metadata(str(parquet_path))
        assert index_col == "taxon"
        assert traits == list(_TRAIT_MATRIX.keys())

    def test_get_trait_metadata_matches(self, csv_path, parquet_path):
        csv_meta = tio.get_trait_metadata(str(csv_path))
        pq_meta = tio.get_trait_metadata(str(parquet_path))
        assert csv_meta == pq_meta

    def test_load_trait_columns_subset_csv(self, csv_path):
        df = tio.load_trait_columns(str(csv_path), ["T1", "T3"], "taxon")
        assert list(df.columns) == ["T1", "T3"]
        assert list(df.index) == _TAXA

    def test_load_trait_columns_subset_parquet(self, parquet_path):
        df = tio.load_trait_columns(str(parquet_path), ["T1", "T3"], "taxon")
        assert list(df.columns) == ["T1", "T3"]
        assert list(df.index) == _TAXA

    def test_load_trait_columns_csv_parquet_equal(self, csv_path, parquet_path):
        cols = ["T2", "T5", "T7"]
        df_csv = tio.load_trait_columns(str(csv_path), cols, "taxon")
        df_pq = tio.load_trait_columns(str(parquet_path), cols, "taxon")
        pd.testing.assert_frame_equal(
            df_csv.astype(int), df_pq.astype(int), check_like=True
        )

    def test_load_index_only(self, parquet_path):
        df = tio.load_trait_columns(str(parquet_path), [], "taxon")
        assert df.shape[1] == 0
        assert list(df.index) == _TAXA

    def test_compute_gene_sums_matches_direct(self, csv_path, trait_df):
        valid_index = set(_TAXA)
        sums = tio.compute_gene_sums(str(csv_path), "taxon", valid_index)
        for col in _TRAIT_MATRIX:
            assert sums[col] == int(trait_df[col].sum()), col

    def test_compute_gene_sums_parquet_matches_csv(self, csv_path, parquet_path):
        valid_index = set(_TAXA)
        sums_csv = tio.compute_gene_sums(str(csv_path), "taxon", valid_index)
        sums_pq = tio.compute_gene_sums(str(parquet_path), "taxon", valid_index)
        assert sums_csv == sums_pq

    def test_compute_gene_sums_filters_to_valid_index(self, parquet_path):
        partial_index = {"A", "B"}  # only 2 of 6 taxa
        sums = tio.compute_gene_sums(str(parquet_path), "taxon", partial_index)
        # T1 = [1,1,0,0,0,0] → sum over A,B = 2
        assert sums["T1"] == 2
        # T3 = [0,0,0,0,1,1] → sum over A,B = 0
        assert sums["T3"] == 0


# ---------------------------------------------------------------------------
# 2. reformat_columns — CSV and Parquet produce identical outputs
# ---------------------------------------------------------------------------

import simphyni.scripts.reformat_csv as rc  # noqa: E402


class TestReformatParquet:
    def _run_reformat(self, input_path: Path, out_dir: Path, **kwargs) -> pd.DataFrame:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "reformatted.parquet"
        rc.reformat_columns(
            str(input_path), str(out),
            kwargs.get("min_prev", 0.0),
            kwargs.get("max_prev", 1.0),
            run_cols=kwargs.get("run_cols", None),
        )
        return pd.read_parquet(out)

    def test_csv_and_parquet_input_produce_equal_output(
        self, csv_path, parquet_path, tmp_path
    ):
        df_csv = self._run_reformat(csv_path, tmp_path / "csv")
        df_pq = self._run_reformat(parquet_path, tmp_path / "pq")
        pd.testing.assert_frame_equal(
            df_csv.sort_index(axis=1),
            df_pq.sort_index(axis=1),
        )

    def test_prevalence_filter_csv(self, tmp_path_factory):
        """Traits at 0% and 100% prevalence are filtered out by default window."""
        taxa = ["A", "B", "C", "D"]
        data = pd.DataFrame({
            "all_ones":  [1, 1, 1, 1],   # prev=1.0 → filtered
            "all_zeros": [0, 0, 0, 0],   # prev=0.0 → filtered
            "half":      [1, 1, 0, 0],   # prev=0.5 → kept
        }, index=taxa)
        data.index.name = "taxon"
        d = tmp_path_factory.mktemp("prev")
        csv = d / "in.csv"; data.to_csv(csv)
        out = d / "out.parquet"
        rc.reformat_columns(str(csv), str(out), 0.05, 0.95)
        result = pd.read_parquet(out)
        assert "half" in result.columns
        assert "all_ones" not in result.columns
        assert "all_zeros" not in result.columns

    def test_prevalence_filter_parquet_matches_csv(self, tmp_path_factory):
        """Same prevalence filter results from CSV and Parquet input."""
        taxa = ["A", "B", "C", "D"]
        data = pd.DataFrame({
            "rare":    [1, 0, 0, 0],   # prev=0.25 → filtered at min=0.3
            "common":  [1, 1, 1, 0],   # prev=0.75 → kept
            "extreme": [1, 1, 1, 1],   # prev=1.0  → filtered
        }, index=taxa)
        data.index.name = "taxon"
        d = tmp_path_factory.mktemp("prev2")
        csv = d / "in.csv"; data.to_csv(csv)
        pq_in = d / "in.parquet"; data.to_parquet(pq_in)
        out_csv = d / "out_csv.parquet"
        out_pq = d / "out_pq.parquet"
        rc.reformat_columns(str(csv), str(out_csv), 0.3, 0.95)
        rc.reformat_columns(str(pq_in), str(out_pq), 0.3, 0.95)
        pd.testing.assert_frame_equal(
            pd.read_parquet(out_csv).sort_index(axis=1),
            pd.read_parquet(out_pq).sort_index(axis=1),
        )

    def test_run_cols_always_kept(self, tmp_path_factory):
        """Protected run_cols survive even when outside the prevalence window."""
        taxa = ["A", "B", "C", "D"]
        data = pd.DataFrame({
            "rare_protected": [1, 0, 0, 0],  # prev=0.25, but protected
            "common":         [1, 1, 1, 0],  # prev=0.75, passes filter
        }, index=taxa)
        data.index.name = "taxon"
        d = tmp_path_factory.mktemp("runcols")
        csv = d / "in.csv"; data.to_csv(csv)
        out = d / "out.parquet"
        # run_cols="0" protects the first trait column (index 0 = rare_protected)
        rc.reformat_columns(str(csv), str(out), 0.3, 0.95, run_cols="0")
        result = pd.read_parquet(out)
        assert "rare_protected" in result.columns
        assert "common" in result.columns


# ---------------------------------------------------------------------------
# 3. split_traits — Parquet input produces correct CSV shards
# ---------------------------------------------------------------------------

SCRIPTS_DIR = (
    Path(__file__).parent.parent / "simphyni" / "scripts"
)


class TestSplitTraitsParquet:
    def test_split_parquet_produces_correct_shards(
        self, parquet_path, tmp_path
    ):
        out_dir = tmp_path / "shards"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "split_traits.py"),
                "--inputs_file", str(parquet_path),
                "--output_dir", str(out_dir),
                "--shard_size", "3",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        shards = sorted(out_dir.glob("shard_*.csv"))
        assert len(shards) == 3  # 8 traits / shard_size=3 → ceil = 3

        # All traits appear exactly once across shards
        all_cols = []
        for s in shards:
            df = pd.read_csv(s, index_col=0)
            assert list(df.index) == _TAXA
            all_cols.extend(df.columns.tolist())
        assert sorted(all_cols) == sorted(_TRAIT_MATRIX.keys())

    def test_split_parquet_shards_match_csv_shards(
        self, csv_path, parquet_path, tmp_path
    ):
        out_csv = tmp_path / "shards_csv"
        out_pq = tmp_path / "shards_pq"
        for inp, out in [(csv_path, out_csv), (parquet_path, out_pq)]:
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "split_traits.py"),
                    "--inputs_file", str(inp),
                    "--output_dir", str(out),
                    "--shard_size", "4",
                ],
                capture_output=True, text=True, check=True,
            )
        shards_csv = sorted(out_csv.glob("shard_*.csv"))
        shards_pq = sorted(out_pq.glob("shard_*.csv"))
        assert len(shards_csv) == len(shards_pq)
        for sc, sp in zip(shards_csv, shards_pq):
            df_c = pd.read_csv(sc, index_col=0).sort_index(axis=1)
            df_p = pd.read_csv(sp, index_col=0).sort_index(axis=1)
            pd.testing.assert_frame_equal(df_c.astype(int), df_p.astype(int))


# ---------------------------------------------------------------------------
# 4. ACR lazy load — per-chunk parquet loading matches direct DataFrame path
# ---------------------------------------------------------------------------

import simphyni.scripts.run_ancestral_reconstruction as acr_mod  # noqa: E402
from simphyni.scripts.run_ancestral_reconstruction import (  # noqa: E402
    _worker_init,
    compute_branch_upper_bound,
    label_internal_nodes,
    reconstruct_trait,
)


class TestACRLazyLoad:
    @pytest.fixture(autouse=True)
    def _restore_worker_globals(self):
        """Save and restore acr_mod globals so this class doesn't pollute other tests."""
        orig_tree = acr_mod._WORKER_TREE
        orig_ub = acr_mod._WORKER_UPPER_BOUND
        yield
        acr_mod._WORKER_TREE = orig_tree
        acr_mod._WORKER_UPPER_BOUND = orig_ub

    def _run_single_trait(self, gene: str, df: pd.DataFrame) -> dict:
        tree = Tree(_NEWICK_6, format=1)
        label_internal_nodes(tree)
        upper_bound = compute_branch_upper_bound(tree)
        acr_mod._WORKER_TREE = tree
        acr_mod._WORKER_UPPER_BOUND = upper_bound
        series = df[gene].astype(str)
        result = reconstruct_trait(gene, None, series, upper_bound, "all", int(df[gene].sum()))
        return result

    def test_lazy_load_parquet_matches_direct_df(self, parquet_path, trait_df):
        """
        reconstruct_trait() on a Series loaded from Parquet via load_trait_columns
        should produce the same result as passing the Series from a full in-memory
        DataFrame.
        """
        gene = "T1"
        # Direct path: from full in-memory DataFrame
        result_direct = self._run_single_trait(gene, trait_df)

        # Lazy path: from parquet via traits_io
        df_lazy = tio.load_trait_columns(str(parquet_path), [gene], "taxon")
        result_lazy = self._run_single_trait(gene, df_lazy.astype(int))

        assert result_direct is not None
        assert result_lazy is not None
        for key in ("gains", "losses", "count", "gains_flow", "losses_flow"):
            assert result_direct[key] == pytest.approx(result_lazy[key], rel=1e-6), key


# ---------------------------------------------------------------------------
# 5. End-to-end: ecoli_accessory.csv → parquet → reformat → ACR (one trait)
# ---------------------------------------------------------------------------

PANX_DIR = Path(__file__).parent / "panx"


@pytest.mark.skipif(
    not (PANX_DIR / "ecoli_accessory.csv").exists(),
    reason="panx fixture data not present",
)
class TestEndToEnd:
    @pytest.fixture(autouse=True)
    def _restore_worker_globals(self):
        orig_tree = acr_mod._WORKER_TREE
        orig_ub = acr_mod._WORKER_UPPER_BOUND
        yield
        acr_mod._WORKER_TREE = orig_tree
        acr_mod._WORKER_UPPER_BOUND = orig_ub

    @pytest.fixture(scope="class")
    def ecoli_parquet(self, tmp_path_factory) -> Path:
        src = PANX_DIR / "ecoli_accessory.csv"
        p = tmp_path_factory.mktemp("e2e") / "ecoli_accessory.parquet"
        pd.read_csv(src, index_col=0).to_parquet(p, index=True)
        return p

    def test_reformat_csv_vs_parquet_equal(self, ecoli_parquet, tmp_path):
        src_csv = PANX_DIR / "ecoli_accessory.csv"
        out_csv = tmp_path / "from_csv.parquet"
        out_pq = tmp_path / "from_pq.parquet"
        rc.reformat_columns(str(src_csv), str(out_csv), 0.05, 0.95)
        rc.reformat_columns(str(ecoli_parquet), str(out_pq), 0.05, 0.95)
        df_csv = pd.read_parquet(out_csv).sort_index(axis=1)
        df_pq = pd.read_parquet(out_pq).sort_index(axis=1)
        pd.testing.assert_frame_equal(df_csv, df_pq)

    def test_acr_one_trait_csv_vs_parquet(self, ecoli_parquet, tmp_path):
        """One ACR call from CSV path and parquet path produce identical stats."""
        src_csv = PANX_DIR / "ecoli_accessory.csv"
        out_csv = tmp_path / "from_csv.parquet"
        out_pq = tmp_path / "from_pq.parquet"
        rc.reformat_columns(str(src_csv), str(out_csv), 0.05, 0.95)
        rc.reformat_columns(str(ecoli_parquet), str(out_pq), 0.05, 0.95)

        tree_nwk = PANX_DIR.parent / "tests" / "panx"
        # Find the .nwk file if present; skip otherwise
        nwk_candidates = list(PANX_DIR.glob("*.nwk"))
        if not nwk_candidates:
            pytest.skip("no .nwk tree file found in panx/")
        tree_path = nwk_candidates[0]

        tree = Tree(str(tree_path), format=1)
        label_internal_nodes(tree)
        upper_bound = compute_branch_upper_bound(tree)
        acr_mod._WORKER_TREE = tree
        acr_mod._WORKER_UPPER_BOUND = upper_bound

        # Pick the first available trait; discover index_col from the reformatted file
        index_col, traits = tio.get_trait_metadata(str(out_csv))
        gene = traits[0]

        def run(parquet_file):
            df = tio.load_trait_columns(str(parquet_file), [gene], index_col)
            leaves = set(tree.get_leaf_names())
            df = df.loc[df.index.isin(leaves)]
            series = df[gene].astype(str)
            return reconstruct_trait(
                gene, None, series, upper_bound, "all",
                int(pd.to_numeric(df[gene], errors="coerce").fillna(0).sum()),
            )

        r_csv = run(out_csv)
        r_pq = run(out_pq)
        assert r_csv is not None and r_pq is not None
        for key in ("gains", "losses", "count", "gains_flow", "losses_flow"):
            assert r_csv[key] == pytest.approx(r_pq[key], rel=1e-6), key
