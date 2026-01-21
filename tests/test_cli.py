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
        assert "SimPhyNI CLI version" in captured.out

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