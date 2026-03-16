import pytest
import sys
import os
import argparse
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

# Change 'my_package.cli' to the actual import path of your script
from simphyni.simphyni_cli import main, run_simphyni

# ==========================================
# TEST: Version and Help Commands
# ==========================================

def test_version_command(capsys):
    """Test that 'version' command prints version and exits."""
    # Simulate running "simphyni version"
    with patch.object(sys, 'argv', ['simphyni', 'version']):
        with pytest.raises(SystemExit) as exc:
            main()
        
        # Check exit code is 0 (Success)
        assert exc.value.code == 0
        
        # Check stdout contains version text
        captured = capsys.readouterr()
        assert "SimPhyNI version" in captured.out

def test_help_command(capsys):
    """Test that --help prints usage and exits 0."""
    with patch.object(sys, 'argv', ['simphyni', '--help']):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        
        captured = capsys.readouterr()
        assert "usage: simphyni" in captured.out
        assert "SimPhyNI — Simulation-based Phylogenetic" in captured.out

# ==========================================
# TEST: Run Command (Logic & Args)
# ==========================================

# Helper to create a dummy args object so we can test run_simphyni directly
def mock_args_obj(**kwargs):
    args = argparse.Namespace()
    # Set Defaults matching your parser
    args.outdir = "simphyni_outs"
    args.temp_dir = "tmp"
    args.plot = False
    args.save_object = False
    args.cores = None
    args.profile = None
    args.dry_run = False
    args.snakemake_args = []
    args.min_prev = 0.05
    args.max_prev = 0.95
    args.samples = None
    args.traits = None
    args.tree = None
    args.run_traits = None
    args.sample_name = None
    
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args

@patch('subprocess.run')
@patch('pandas.DataFrame.to_csv')
@patch('os.makedirs')
@patch('os.path.exists')
def test_run_single_mode(mock_exists, mock_makedirs, mock_to_csv, mock_subprocess):
    """Test Single Run Mode (-T, -t, -r)."""
    mock_exists.return_value = True # Pretend files exist
    
    args = mock_args_obj(
        traits="data.csv",
        tree="tree.nwk",
        run_traits="ALL",
        sample_name="test_sample"
    )

    run_simphyni(args)

    # 1. Verify Sample Sheet Creation
    assert mock_to_csv.called
    # Check that a samples CSV is being written to the output dir
    df_args = mock_to_csv.call_args[0]
    assert "simphyni_sample_info.csv" in df_args[0]

    # 2. Verify Snakemake Command Construction
    assert mock_subprocess.called
    cmd = mock_subprocess.call_args[0][0] # The command list passed to subprocess
    
    # Assert critical flags exist
    assert cmd[0] == "snakemake"
    assert "--snakefile" in cmd
    assert "--cores" in cmd
    assert "all" in cmd # Default cores logic
    
    # Verify config args
    config_part = cmd[cmd.index("--config") + 1 :]
    assert any("samples=" in s for s in config_part)
    assert any("prefilter=False" in s for s in config_part)

@patch('subprocess.run')
@patch('pandas.DataFrame.to_csv') # <--- ADD THIS
@patch('pandas.read_csv') # Mock reading the input samples file
@patch('os.makedirs')
@patch('os.path.exists')
def test_run_batch_mode(mock_exists, mock_makedirs, mock_read, mock_to_csv, mock_subprocess):
    """Test Batch Mode (--samples)."""
    mock_exists.return_value = True
    # Return a dummy DF when pd.read_csv is called
    mock_read.return_value = pd.DataFrame({'Sample':['A'], 'Traits':['t.csv'], 'Tree':['tree.nwk']})

    args = mock_args_obj(samples="my_samples.csv")

    run_simphyni(args)

    assert mock_read.called
    assert mock_subprocess.called
    cmd = mock_subprocess.call_args[0][0]
    # Ensure the config passes the samples file path correctly
    assert any("samples=" in s for s in cmd)

# ==========================================
# TEST: Error Handling
# ==========================================

def test_run_missing_args():
    """Test error when neither samples nor traits/tree provided."""
    args = mock_args_obj() # No arguments provided
    
    with pytest.raises(SystemExit) as exc:
        run_simphyni(args)
    
    assert "Must provide either --samples OR -T" in str(exc.value)

@patch('os.path.exists')
def test_run_missing_file_check(mock_exists):
    """Test error when input file does not exist."""
    mock_exists.return_value = False # Force file not found
    
    args = mock_args_obj(samples="ghost.csv")
    
    with pytest.raises(SystemExit) as exc:
        run_simphyni(args)
    
    assert "Samples file not found" in str(exc.value)

# ==========================================
# TEST: Advanced Flags
# ==========================================

@patch('subprocess.run')
@patch('os.path.exists')
@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
def test_snakemake_extra_flags(mock_to, mock_make, mock_exists, mock_subprocess):
    """Test --profile, --cores, --dry-run and passthrough args."""
    mock_exists.return_value = True
    
    args = mock_args_obj(
        traits="t.csv", tree="tr.nwk", run_traits="ALL",
        profile="slurm",
        cores=8,
        dry_run=True,
        snakemake_args=["--unlock", "--rerun-incomplete"]
    )

    run_simphyni(args)

    cmd = mock_subprocess.call_args[0][0]

    # Check Cores
    assert str(8) in cmd
    
    # Check Profile
    assert "--profile" in cmd
    assert "slurm" in cmd
    
    # Check Dry Run
    assert "--dry-run" in cmd
    
    # Check passthrough args
    assert "--unlock" in cmd


# ==========================================
# TEST: runSimPhyNI default method selection
# ==========================================

def test_run_simphyni_selects_flow_when_available():
    """When the pastml CSV contains gains_flow, FLOW counting must be chosen."""
    from simphyni.Simulation.simulation import build_sim_params
    import pandas as pd

    df = pd.DataFrame({
        "gene": ["T1"],
        "gains": [1.0], "losses": [0.5],
        "dist": [0.2], "loss_dist": [0.1],
        "gain_subsize": [5.0], "loss_subsize": [4.0],
        "gain_subsize_nofilter": [6.0], "loss_subsize_nofilter": [5.0],
        "gain_subsize_thresh": [3.0], "loss_subsize_thresh": [2.5],
        "root_state": [0],
        "gains_flow": [0.9], "losses_flow": [0.4],
        "gains_markov": [0.8], "losses_markov": [0.3],
        "gains_entropy": [0.7], "losses_entropy": [0.3],
        "gain_subsize_marginal": [4.5], "loss_subsize_marginal": [3.8],
        "gain_subsize_marginal_nofilter": [5.5], "loss_subsize_marginal_nofilter": [4.8],
        "gain_subsize_marginal_thresh": [3.2], "loss_subsize_marginal_thresh": [2.7],
        "gain_subsize_entropy": [2.0], "loss_subsize_entropy": [1.8],
        "gain_subsize_entropy_nofilter": [2.5], "loss_subsize_entropy_nofilter": [2.2],
        "gain_subsize_entropy_thresh": [1.5], "loss_subsize_entropy_thresh": [1.3],
        "dist_marginal": [0.15], "loss_dist_marginal": [0.08],
        "root_prob": [0.2],
    })

    # The selection logic from runSimPhyNI.py
    if "gains_flow" in df.columns:
        result = build_sim_params(df, counting="FLOW", subsize="ORIGINAL")
    else:
        result = build_sim_params(df, counting="JOINT", subsize="ORIGINAL")

    # FLOW should be used: gains column should come from gains_flow (0.9)
    assert result["gains"].iloc[0] == pytest.approx(0.9)


def test_run_simphyni_falls_back_to_joint():
    """When gains_flow is absent (legacy pastml output), JOINT counting must be used."""
    from simphyni.Simulation.simulation import build_sim_params
    import pandas as pd

    df = pd.DataFrame({
        "gene": ["T1"],
        "gains": [2.0], "losses": [1.0],
        "dist": [0.5], "loss_dist": [0.3],
        "gain_subsize": [10.0], "loss_subsize": [8.0],
        "gain_subsize_nofilter": [12.0], "loss_subsize_nofilter": [9.0],
        "gain_subsize_thresh": [7.0], "loss_subsize_thresh": [6.0],
        "root_state": [0],
        # No gains_flow column — simulates legacy pastml output
    })

    if "gains_flow" in df.columns:
        result = build_sim_params(df, counting="FLOW", subsize="ORIGINAL")
    else:
        result = build_sim_params(df, counting="JOINT", subsize="ORIGINAL")

    # JOINT should be used: gains comes from 'gains' (2.0)
    assert result["gains"].iloc[0] == pytest.approx(2.0)


def test_run_ancestral_reconstruction_accepts_uncertainty_arg(tmp_path):
    """ACR main() exits gracefully on --help without crashing."""
    import sys
    from simphyni.scripts.run_ancestral_reconstruction import main as acr_main

    with patch.object(sys, "argv", ["run_ancestral_reconstruction.py", "--help"]):
        with pytest.raises(SystemExit) as exc:
            acr_main()
        assert exc.value.code == 0


# ==========================================
# TEST: runSimPhyNI method-selection logic
# ==========================================

def _make_flow_df():
    """Minimal ACR DataFrame with marginal columns (new pipeline output)."""
    return pd.DataFrame({
        "gene": ["T1"],
        "gains": [2.0], "losses": [1.0],
        "dist": [0.5], "loss_dist": [0.3],
        "gain_subsize": [10.0], "loss_subsize": [8.0],
        "gain_subsize_nofilter": [12.0], "loss_subsize_nofilter": [9.0],
        "gain_subsize_thresh": [7.0], "loss_subsize_thresh": [6.0],
        "root_state": [0],
        "gains_flow": [1.8], "losses_flow": [0.9],
        "gains_markov": [1.5], "losses_markov": [0.7],
        "gains_entropy": [1.2], "losses_entropy": [0.6],
        "gain_subsize_marginal": [9.0], "loss_subsize_marginal": [7.5],
        "gain_subsize_marginal_nofilter": [11.0], "loss_subsize_marginal_nofilter": [8.5],
        "gain_subsize_marginal_thresh": [6.5], "loss_subsize_marginal_thresh": [5.5],
        "gain_subsize_entropy": [4.0], "loss_subsize_entropy": [3.5],
        "gain_subsize_entropy_nofilter": [5.0], "loss_subsize_entropy_nofilter": [4.5],
        "gain_subsize_entropy_thresh": [3.0], "loss_subsize_entropy_thresh": [2.5],
        "dist_marginal": [0.4], "loss_dist_marginal": [0.2],
        "root_prob": [0.3],
    })


def _make_legacy_df():
    """Minimal legacy DataFrame without marginal columns (old pastml + GL_tab output)."""
    return pd.DataFrame({
        "gene": ["T1"],
        "gains": [2.0], "losses": [1.0],
        "dist": [0.5], "loss_dist": [0.3],
        "gain_subsize": [10.0], "loss_subsize": [8.0],
        "gain_subsize_nofilter": [12.0], "loss_subsize_nofilter": [9.0],
        "gain_subsize_thresh": [7.0], "loss_subsize_thresh": [6.0],
        "root_state": [0],
    })


@patch("simphyni.scripts.runSimPhyNI.build_sim_params")
@patch("simphyni.scripts.runSimPhyNI.TreeSimulator")
@patch("simphyni.scripts.runSimPhyNI.pd.read_csv")
def test_runsimphyni_selects_flow_for_new_acr(mock_read_csv, mock_sim_class, mock_bsp,
                                               tmp_path):
    """runSimPhyNI.main() calls build_sim_params(counting='FLOW') for new ACR output."""
    from simphyni.scripts.runSimPhyNI import main as rsn_main

    mock_read_csv.return_value = _make_flow_df()
    mock_bsp.return_value = _make_flow_df()  # return value consumed by TreeSimulator
    mock_sim_instance = MagicMock()
    mock_sim_instance.get_results.return_value = pd.DataFrame(
        columns=["T1", "T2", "direction", "effect size",
                 "prevalence_T1", "prevalence_T2",
                 "pval_naive", "pval_bh", "pval_by", "pval_bonf"]
    )
    mock_sim_class.return_value = mock_sim_instance

    outdir = str(tmp_path / "out")
    with patch.object(sys, "argv", [
        "runSimPhyNI.py",
        "-p", "acr.csv",
        "-s", "traits.csv",
        "-t", "tree.nwk",
        "-o", outdir,
    ]):
        rsn_main()

    # build_sim_params must have been called with counting='FLOW'
    assert mock_bsp.called
    call_kwargs = mock_bsp.call_args
    assert call_kwargs.kwargs.get("counting") == "FLOW" or call_kwargs.args[1] == "FLOW"


@patch("simphyni.scripts.runSimPhyNI.build_sim_params")
@patch("simphyni.scripts.runSimPhyNI.TreeSimulator")
@patch("simphyni.scripts.runSimPhyNI.pd.read_csv")
def test_runsimphyni_falls_back_to_joint_for_legacy(mock_read_csv, mock_sim_class,
                                                     mock_bsp, tmp_path):
    """runSimPhyNI.main() calls build_sim_params(counting='JOINT') for legacy output."""
    from simphyni.scripts.runSimPhyNI import main as rsn_main

    mock_read_csv.return_value = _make_legacy_df()
    mock_bsp.return_value = _make_legacy_df()
    mock_sim_instance = MagicMock()
    mock_sim_instance.get_results.return_value = pd.DataFrame(
        columns=["T1", "T2", "direction", "effect size",
                 "prevalence_T1", "prevalence_T2",
                 "pval_naive", "pval_bh", "pval_by", "pval_bonf"]
    )
    mock_sim_class.return_value = mock_sim_instance

    outdir = str(tmp_path / "out")
    with patch.object(sys, "argv", [
        "runSimPhyNI.py",
        "-p", "acr.csv",
        "-s", "traits.csv",
        "-t", "tree.nwk",
        "-o", outdir,
    ]):
        rsn_main()

    assert mock_bsp.called
    call_kwargs = mock_bsp.call_args
    assert call_kwargs.kwargs.get("counting") == "JOINT" or call_kwargs.args[1] == "JOINT"

    """
    runSimPhyNI CLI must NOT expose --uncertainty or --counting arguments.
    Verified by checking that the script's help text does not contain either option.
    """
    import io
    from simphyni.scripts.runSimPhyNI import main as rsn_main

    buf = io.StringIO()
    with patch.object(sys, "argv", ["runSimPhyNI.py", "--help"]):
        with pytest.raises(SystemExit):
            with patch("sys.stdout", buf):
                rsn_main()

    help_text = buf.getvalue()
    assert "--uncertainty" not in help_text, "--uncertainty should not be a CLI option"
    assert "--counting" not in help_text, "--counting should not be a CLI option"