#!/usr/bin/env python
"""
End-to-end SimPhyNI run with per-stage timing and peak memory.

Stages: reformat tree -> ACR (subprocess, timed separately) -> parameters ->
TreeSimulator init (all-vs-all pairs + observed statistics) -> null simulation
and scoring (fast path reports its own sub-stages) -> multiple testing -> write.

usage: run_pipeline.py <dataset_dir> <out_dir> <config> [--cores N] [--seed S] [--traits K]
config: default | soft_auto_composite | composite | legacy
"""
import argparse
import contextlib
import io
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("out"); ap.add_argument("config")
ap.add_argument("--cores", type=int, default=8)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--traits", type=int, default=0, help="use only the first K traits (0 = all)")
ap.add_argument("--src", default=str(Path(__file__).resolve().parents[1]))
ap.add_argument("--pairs", default=None, help="CSV with T1,T2: test only these pairs")
a = ap.parse_args()
sys.path.insert(0, a.src)
ds, out = Path(a.dataset), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
T = {}
t0 = time.perf_counter()


def lap(name, t):
    T[name] = round(time.perf_counter() - t, 3)
    print(f"[{name}] {T[name]:.1f}s", flush=True)
    return time.perf_counter()


scripts = Path(a.src) / "simphyni" / "scripts"
t = time.perf_counter()
traits_csv = ds / "traits.csv"
if a.traits:
    X = pd.read_csv(traits_csv, index_col=0)
    X.iloc[:, :a.traits].to_csv(out / "traits_subset.csv")
    traits_csv = out / "traits_subset.csv"
subprocess.run([sys.executable, str(scripts / "reformat_tree.py"), str(ds / "tree.nwk"), str(out / "tree.nwk")],
               check=True, capture_output=True)
t = lap("reformat_tree", t)
acr = out / "acr.csv"
if not acr.exists():
    r = subprocess.run(["/usr/bin/time", "-l", sys.executable, str(scripts / "run_ancestral_reconstruction.py"),
                        "--inputs_file", str(traits_csv), "--tree_file", str(out / "tree.nwk"),
                        "--output_csv", str(acr), "--reconstruction", "all", "--max_workers", str(a.cores)],
                       capture_output=True, text=True, env={"PYTHONPATH": a.src, "PATH": "/usr/bin:/bin"})
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])
    rss = [l for l in r.stderr.splitlines() if "maximum resident set size" in l]
    T["acr_peak_rss_gb"] = round(int(rss[0].split()[0]) / 1e9, 2) if rss else None
    t = lap("acr", t)
else:
    T["acr"] = json.loads((out / "timing.json").read_text()).get("acr") if (out / "timing.json").exists() else None

from simphyni import TreeSimulator  # noqa: E402
from simphyni.Simulation.simulation import build_sim_params  # noqa: E402
from simphyni.Simulation import fastsim  # noqa: E402

t = time.perf_counter()
params = build_sim_params(pd.read_csv(acr), counting="FLOW", subsize="ORIGINAL")
t = lap("params", t)
kw = {
    "default": dict(),
    "soft_auto_composite": dict(lineage="soft_auto", composite=True),
    "composite": dict(composite=True),
    "soft_auto": dict(lineage="soft_auto"),
    "legacy": dict(statistic="logor", pvalue="kde", trials=64, condition="prevalence"),
    "default_refine": dict(refine_p=1e-3),
    "default_r01": dict(refine_p=1e-2, refine_max_pairs=5000),
    "soft_auto_composite_r01": dict(lineage="soft_auto", composite=True, refine_p=1e-2, refine_max_pairs=5000),
    "default_refine2": dict(refine_p=1e-2, refine_tiers=[(1e-2, 32768, 0.01), (1e-5, 262144, 0.001)]),
    "soft_auto_composite_refine2": dict(lineage="soft_auto", composite=True, refine_p=1e-2,
                                        refine_tiers=[(1e-2, 32768, 0.01), (1e-5, 262144, 0.001)]),
    "default_refine_xi01": dict(refine_p=1e-3, xi_min=-0.1),
    "default_refine_xi02": dict(refine_p=1e-3, xi_min=-0.2),
    "default_refine_xi03": dict(refine_p=1e-3, xi_min=-0.3),
    "soft_auto_composite_refine": dict(lineage="soft_auto", composite=True, refine_p=1e-3),
}[a.config]
with contextlib.redirect_stdout(io.StringIO()):
    sim = TreeSimulator(tree=str(out / "tree.nwk"), pastmlfile=params, obsdatafile=str(traits_csv))
    sim.initialize_simulation_parameters(pre_filter=False, run_traits=1 if a.pairs else 0)
    if a.pairs:
        pl = pd.read_csv(a.pairs)
        sim.pairs, sim.obspairs = sim._get_pair_data2(sim.obsdf_modified, list(zip(pl.T1.astype(str), pl.T2.astype(str))))
        sim.total_tests = len(sim.pairs)
T["n_pairs"] = int(len(sim.pairs)); T["n_traits"] = int(len(np.unique(sim.pairs)))
t = lap("init_pairs", t)
with contextlib.redirect_stdout(io.StringIO()):
    sim.run_simulation(cores=a.cores, seed=a.seed, **kw)
t = lap("run_simulation", t)
T["fast_stages"] = dict(fastsim.TIMINGS)
res = sim.get_results()
t = lap("get_results", t)
tag = a.config + ("_pairs" if a.pairs else "")
res.to_parquet(out / f"results_{tag}.parquet", index=False)
t = lap("write", t)
T["total"] = round(time.perf_counter() - t0, 1)
T["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
T["n_by05"] = int((res["pval_by"] < 0.05).sum()); T["n_by01"] = int((res["pval_by"] < 0.01).sum())
(out / f"timing_{tag}.json").write_text(json.dumps(T, indent=2))
print(json.dumps(T, indent=2))
