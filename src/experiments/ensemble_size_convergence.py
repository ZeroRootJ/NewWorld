"""E1 -- ensemble-size convergence for SGS and RBF+bootstrap (user-approved
2026-11-16; first step of the F1-F4 stabilization experiments E1-E5).

Purpose
-------
Measure how UMG, variance_mean, 90% interval coverage and MSE of the two
ENSEMBLE methods (SGS realizations, RBF bootstrap replicates) change with the
ensemble size N, so the production N for E2-E5 can be chosen from data by the
user-approved rule (applied in
results/processed/ensemble_size_convergence/make_convergence_summary.py, not
here). The project default stays N_BOOTSTRAP = N_REALIZATIONS = 10; this
script only passes an explicit ``n_ensemble`` override.

Design (fixed by the task, not chosen here)
-------------------------------------------
* 2 cases, each reusing an EXISTING truth + conditioning-sample set:
    - ``base_case``  : base_case_conditioning defaults (TRUTH_SEED=101,
                       SAMPLE_SEED=20, N_SAMPLES=125 requested / 121 actual,
                       range 300 m, base-case nugget).
    - ``density_1pct_rep0`` : sample_replicate_axis level "1"
                       (n_samples_requested=25), replicate rep0
                       (sample_seed=1001), same truth (TRUTH_SEED=101,
                       range AXIS_HMAJ1/AXIS_HMIN1=300 m).
* Each case x method is run ONCE with N_MAX = 200 ensemble members, for 3
  different seeds (new named seeds below). N_MAX is passed through
  ``_axis_parallel.run_levels_parallel(n_ensemble=N_MAX)``, which injects the
  SAME value into sgs and rbf_bootstrap.
* Nested subsets: for N in N_GRID the FIRST N members of the N_MAX ensemble
  are used (realizations / replicates 0..N-1), so N-subsets of one run are
  nested by construction.
* Evaluation per (case, method, seed, N) follows
  src/experiments/evaluate_sample_replicate_axis.py:evaluate_one_cell exactly:
    - mask      : conditioning_cell_mask of the regenerated conditioning
                  samples (conditioning cells excluded), identical for both
                  methods within a case;
    - SGS       : mean / var(ddof=1) over the first N realizations (the
                  honor-re-enforced realizations saved by sgs.py, i.e. the
                  arrays sgs_mean_map / sgs_var_map are computed from). MSE
                  from that ensemble mean; UMG / coverage90 from N(mean, std);
    - RBF       : bootstrap mean / var(ddof=1) over the first N replicates for
                  UMG / coverage90 / variance_mean; MSE from
                  point_estimate_map (the single full-data fit, as in
                  evaluate_one_cell) -- therefore RBF MSE does not depend on N;
    - UMG       : calc_umg on accuracy_plot_fraction_in's default
                  np.linspace(0, 1, 20) grid;
    - coverage90: accuracy_plot_fraction_in at p = 0.90, obtained in the SAME
                  call via p_intervals = [linspace(0,1,20)..., 0.90] (same
                  construction as results/processed/sample_replicate_axis/
                  make_coverage90.py); the 20-point prefix is asserted
                  bit-identical to a default call;
    - variance_mean : mean of the predictive variance over the mask.
  At N = N_MAX the subset statistics are asserted equal (allclose) to the
  run's own saved *_mean_map.npy / *_var_map.npy.

Seeds (new, recorded in the manifest)
-------------------------------------
SGS_SEEDS = [3101, 3102, 3103] (sgsim ``seed``), BOOTSTRAP_SEEDS = [3201,
3202, 3203] (bootstrap RandomState). Asserted at import time not to collide
with any existing pinned seed constant of the project (listed in
_EXISTING_SEEDS). SGS_JITTER_SEED (70) and CV_SEED (40) are NOT varied -- they
are part of each method's fixed configuration, so across the 3 seeds only
the sgsim random path / bootstrap resampling changes.

Outputs
-------
Method runs go to their usual results/raw/sgs/<ts>/ and
results/raw/rbf_bootstrap/<ts>/ directories (as for every axis experiment).
This script's own run directory results/raw/ensemble_size_convergence/<ts>/
holds:
  nested_metrics.csv : tidy (case, method, seed_index, seed, n_ensemble,
                       metric, value), metrics = mse, umg, variance_mean,
                       coverage90
  timings.csv        : per run: ensemble_seconds (sgsim_seconds for SGS,
                       bootstrap_seconds for RBF), cv_seconds (RBF),
                       total_seconds, n_ensemble, concurrency note
  source_runs.json   : {case: {seed_index: {method: run_dir}}}
  manifest.json      : via src.io.save_result

Timing caveat: all 12 runs execute CONCURRENTLY in one process pool (BLAS
threads capped to 1 per process), so wall times are under that load; the
pool size and os.cpu_count() are recorded in the manifest.

Run with:
.venv/Scripts/python.exe -m src.experiments.ensemble_size_convergence
Re-evaluate existing runs without re-running them:
.venv/Scripts/python.exe -m src.experiments.ensemble_size_convergence --reuse-source-runs <path/to/source_runs.json>
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import (
    DEFAULT_BINS,
    accuracy_plot_fraction_in,
    calc_umg,
    conditioning_cell_mask,
    mse,
)
from src.experiments import rbf_bootstrap, sgs
from src.experiments._axis_parallel import (
    ENSEMBLE_METHODS,
    default_max_workers,
    run_levels_parallel,
)
from src.experiments.base_case import NX, NY, XMN, YMN, XSIZ, YSIZ
from src.experiments.base_case_conditioning import (
    HMAJ1,
    HMIN1,
    NUG,
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_conditioning_data,
)
from src.experiments.evaluate_sample_replicate_axis import safe_sqrt_variance
from src.experiments.gp_mle import GP_RANDOM_STATE
from src.experiments.sample_replicate_axis import (
    AXIS_HMAJ1,
    AXIS_HMIN1,
    N_SAMPLES_REQUESTED_BY_LEVEL,
    REPLICATE_SEED,
    SAMPLE_SEED_REPLICATES,
)
from src.io import make_run_dir, save_result

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXPERIMENT_NAME = "ensemble_size_convergence"
CODE_ENTRYPOINT = "src/experiments/ensemble_size_convergence.py"

N_MAX = 200
N_GRID = (10, 25, 50, 100, 150, 200)
NOMINAL_P_COVERAGE = 0.90
METRICS = ("mse", "umg", "variance_mean", "coverage90")
METHODS = ("sgs", "rbf_bootstrap")

# --- New named seeds (3 per method) ---------------------------------------
SGS_SEEDS = [3101, 3102, 3103]
BOOTSTRAP_SEEDS = [3201, 3202, 3203]
N_SEEDS = 3

# Existing pinned seed constants in the project (grep of src/ and
# results/processed/ on 2026-11-16). The new seeds must not collide.
_EXISTING_SEEDS = (
    {TRUTH_SEED, SAMPLE_SEED, rbf_bootstrap.BOOTSTRAP_SEED, rbf_bootstrap.CV_SEED,
     sgs.SGS_SEED, sgs.SGS_JITTER_SEED, GP_RANDOM_STATE,
     7, 11, 13, 55, 80, 85, 90, 73073, 20261116, 2001, 2002,
     101, 102, 103, 104, 105}
    | set(SAMPLE_SEED_REPLICATES)
    | set(range(1000, 1301))      # NULL_SEED_START=1000, N_NULL_SEEDS=300
    | set(range(9600, 9700))      # GP_DRAW_SEED_BASE=9600 (+ offsets)
)
_new_seeds = SGS_SEEDS + BOOTSTRAP_SEEDS
if len(SGS_SEEDS) != N_SEEDS or len(BOOTSTRAP_SEEDS) != N_SEEDS:
    raise ValueError("SGS_SEEDS / BOOTSTRAP_SEEDS must each have N_SEEDS entries.")
if len(set(_new_seeds)) != len(_new_seeds):
    raise ValueError(f"new seeds are not distinct: {_new_seeds}")
if set(_new_seeds) & _EXISTING_SEEDS:
    raise ValueError(f"new seeds collide with existing seeds: {sorted(set(_new_seeds) & _EXISTING_SEEDS)}")
if set(METHODS) != set(ENSEMBLE_METHODS):
    raise ValueError(f"METHODS {METHODS} != _axis_parallel.ENSEMBLE_METHODS {ENSEMBLE_METHODS}")
if N_GRID[-1] != N_MAX or list(N_GRID) != sorted(N_GRID):
    raise ValueError("N_GRID must be ascending and end at N_MAX.")

# --- The two cases (existing truth + sample sets, reused) -----------------
CASES = {
    "base_case": dict(
        truth_seed=TRUTH_SEED,
        sample_seed=SAMPLE_SEED,
        hmaj1=HMAJ1,
        hmin1=HMIN1,
        n_samples=N_SAMPLES,
        nug=NUG,
    ),
    "density_1pct_rep0": dict(
        truth_seed=TRUTH_SEED,
        sample_seed=REPLICATE_SEED["rep0"],
        hmaj1=AXIS_HMAJ1,
        hmin1=AXIS_HMIN1,
        n_samples=N_SAMPLES_REQUESTED_BY_LEVEL["1"],
        nug=NUG,
    ),
}
CASE_NOTES = {
    "base_case": "base_case_conditioning defaults (results/processed/base_case/source_runs.json conditioning set)",
    "density_1pct_rep0": "sample_replicate_axis level '1' (n_samples_requested=25), replicate rep0 (sample_seed=1001)",
}

SAMPLES_MATCH_ATOL = 1e-10
MAP_MATCH_RTOL = 1e-10
MAP_MATCH_ATOL = 1e-10

METHOD_SEED_KW = {"sgs": "sgs_seed", "rbf_bootstrap": "bootstrap_seed"}
METHOD_SEEDS = {"sgs": SGS_SEEDS, "rbf_bootstrap": BOOTSTRAP_SEEDS}


def _rel(p) -> str:
    return Path(p).resolve().relative_to(_REPO_ROOT).as_posix()


def _level_key(case: str, k: int) -> str:
    return f"{case}__seed{k}"


def run_all(max_workers=None) -> dict:
    """Run 2 cases x 3 seeds x 2 methods = 12 tasks at N_MAX in one pool.
    Returns {case: {seed_index(str): {method: rel_run_dir}}}."""
    level_kwargs, method_kwargs = {}, {}
    for case, kw in CASES.items():
        for k in range(N_SEEDS):
            lk = _level_key(case, k)
            level_kwargs[lk] = dict(kw)
            method_kwargs[lk] = {m: {METHOD_SEED_KW[m]: METHOD_SEEDS[m][k]} for m in METHODS}
    results = run_levels_parallel(
        level_kwargs,
        methods=METHODS,
        max_workers=max_workers,
        n_ensemble=N_MAX,
        method_kwargs=method_kwargs,
    )
    source_runs = {case: {} for case in CASES}
    for case in CASES:
        for k in range(N_SEEDS):
            lk = _level_key(case, k)
            source_runs[case][str(k)] = {m: _rel(results[lk][m]) for m in METHODS}
    return source_runs


def _load_and_verify(case: str, k: int, method: str, rel_run_dir: str, samples_df: pd.DataFrame):
    """Verify the run's manifest (case parameters, seed, N_MAX) and
    samples.csv, return (manifest_params, ensemble[N_MAX,NY,NX], point_map or
    None, saved_mean_map, saved_var_map)."""
    run_dir = _REPO_ROOT / rel_run_dir
    params = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["params"]
    kw = CASES[case]
    for key in ("truth_seed", "sample_seed", "n_samples_requested"):
        expected = kw["n_samples"] if key == "n_samples_requested" else kw[key]
        if params.get(key) != expected:
            raise ValueError(f"{rel_run_dir}: params.{key}={params.get(key)} != expected {expected}")
    for key in ("hmaj1", "hmin1", "nug"):
        if not np.isclose(float(params[key]), float(kw[key])):
            raise ValueError(f"{rel_run_dir}: params.{key}={params[key]} != expected {kw[key]}")
    seed_key = METHOD_SEED_KW[method]
    if params.get(seed_key) != METHOD_SEEDS[method][k]:
        raise ValueError(f"{rel_run_dir}: params.{seed_key}={params.get(seed_key)} != {METHOD_SEEDS[method][k]}")
    n_key = "n_realizations" if method == "sgs" else "n_bootstrap"
    if params.get(n_key) != N_MAX:
        raise ValueError(f"{rel_run_dir}: params.{n_key}={params.get(n_key)} != N_MAX={N_MAX}")

    recorded = pd.read_csv(run_dir / "samples.csv")
    if len(recorded) != len(samples_df) or not np.allclose(
        recorded[["X", "Y", VCOL]].values, samples_df[["X", "Y", VCOL]].values,
        rtol=0.0, atol=SAMPLES_MATCH_ATOL,
    ):
        raise ValueError(f"{rel_run_dir}: samples.csv does not match regenerated conditioning samples ({case}).")

    if method == "sgs":
        ens = np.load(run_dir / "sgs_realizations.npy")
        point = None
        saved_mean = np.load(run_dir / "sgs_mean_map.npy")
        saved_var = np.load(run_dir / "sgs_var_map.npy")
    else:
        ens = np.load(run_dir / "bootstrap_replicate_maps.npy")
        point = np.load(run_dir / "point_estimate_map.npy")
        saved_mean = np.load(run_dir / "bootstrap_mean_map.npy")
        saved_var = np.load(run_dir / "bootstrap_var_map.npy")
    if ens.shape != (N_MAX, NY, NX):
        raise ValueError(f"{rel_run_dir}: ensemble shape {ens.shape} != {(N_MAX, NY, NX)}")
    return params, ens, point, saved_mean, saved_var


def evaluate(source_runs: dict):
    """Nested-subset metrics for every (case, seed, method, N). Returns
    (metrics_df, timings_df, case_info)."""
    p_grid = np.append(np.linspace(0.0, 1.0, DEFAULT_BINS), NOMINAL_P_COVERAGE)
    rows, timing_rows, case_info = [], [], {}
    for case, kw in CASES.items():
        truth, samples_df = get_base_case_conditioning_data(**kw)
        mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
        truth_m = truth[mask]
        case_info[case] = {
            **{k: (float(v) if isinstance(v, float) else v) for k, v in kw.items()},
            "n_samples_actual": int(len(samples_df)),
            "n_cells_evaluated": int(mask.sum()),
            "n_cells_excluded": int((~mask).sum()),
            "note": CASE_NOTES[case],
        }
        print(f"\ncase {case}: n_samples_actual={len(samples_df)}, evaluated cells={int(mask.sum())}")
        for k in range(N_SEEDS):
            for method in METHODS:
                rel = source_runs[case][str(k)][method]
                params, ens, point, saved_mean, saved_var = _load_and_verify(case, k, method, rel, samples_df)
                seed = METHOD_SEEDS[method][k]
                for n in N_GRID:
                    sub = ens[:n]
                    mean_map = sub.mean(axis=0)
                    var_map = sub.var(axis=0, ddof=1)
                    if n == N_MAX and not (
                        np.allclose(mean_map, saved_mean, rtol=MAP_MATCH_RTOL, atol=MAP_MATCH_ATOL)
                        and np.allclose(var_map, saved_var, rtol=MAP_MATCH_RTOL, atol=MAP_MATCH_ATOL)
                    ):
                        raise AssertionError(f"{rel}: N_MAX subset mean/var differ from the run's saved maps.")
                    point_map = mean_map if method == "sgs" else point
                    var_m = var_map[mask]
                    std_m = safe_sqrt_variance(var_m, f"{case} {method} seed{k} N={n}")
                    mean_m = mean_map[mask]
                    p, frac = accuracy_plot_fraction_in(truth_m, mean_m, std_m, p_intervals=p_grid)
                    p_def, frac_def = accuracy_plot_fraction_in(truth_m, mean_m, std_m)
                    if not (np.array_equal(p[:DEFAULT_BINS], p_def) and np.array_equal(frac[:DEFAULT_BINS], frac_def)):
                        raise AssertionError("explicit p_intervals prefix not bit-identical to default call.")
                    if p[-1] != NOMINAL_P_COVERAGE:
                        raise AssertionError("last p_interval is not 0.90.")
                    values = {
                        "mse": mse(truth_m, point_map[mask]),
                        "umg": calc_umg(p_def, frac_def),
                        "variance_mean": float(np.mean(var_m)),
                        "coverage90": float(frac[-1]),
                    }
                    for metric in METRICS:
                        rows.append({
                            "case": case, "method": method, "seed_index": k, "seed": seed,
                            "n_ensemble": n, "metric": metric, "value": values[metric],
                        })
                    print(f"  {method:13s} seed{k} ({seed}) N={n:3d}: "
                          + ", ".join(f"{m}={values[m]:.5f}" for m in METRICS))
                timing_rows.append({
                    "case": case, "method": method, "seed_index": k, "seed": seed,
                    "n_ensemble": N_MAX,
                    "ensemble_seconds": params["sgsim_seconds"] if method == "sgs" else params["bootstrap_seconds"],
                    "ensemble_seconds_definition": (
                        "sgsim_seconds (geostats.sgsim call, all N realizations)" if method == "sgs"
                        else "bootstrap_seconds (N RBF refits + grid predictions)"
                    ),
                    "cv_seconds": params.get("cv_seconds"),
                    "total_seconds": params["total_seconds"],
                    "run_dir": rel,
                })
    metrics_df = pd.DataFrame(rows)
    expected = len(CASES) * N_SEEDS * len(METHODS) * len(N_GRID) * len(METRICS)
    if len(metrics_df) != expected or not np.all(np.isfinite(metrics_df["value"].values)):
        raise ValueError(f"metrics table has {len(metrics_df)} rows (expected {expected}) or non-finite values.")
    return metrics_df, pd.DataFrame(timing_rows), case_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-source-runs", default=None,
                        help="source_runs.json of a previous invocation: skip running, only re-evaluate.")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    max_workers = args.max_workers if args.max_workers is not None else default_max_workers()
    t0 = time.time()
    if args.reuse_source_runs:
        source_runs = json.loads(Path(args.reuse_source_runs).read_text(encoding="utf-8"))
        runs_were_executed_here = False
    else:
        source_runs = run_all(max_workers=max_workers)
        runs_were_executed_here = True
    pool_wall_seconds = time.time() - t0

    metrics_df, timings_df, case_info = evaluate(source_runs)

    run_dir = make_run_dir(EXPERIMENT_NAME)
    output_files = []
    metrics_df.to_csv(run_dir / "nested_metrics.csv", index=False)
    output_files.append("nested_metrics.csv")
    timings_df.to_csv(run_dir / "timings.csv", index=False)
    output_files.append("timings.csv")
    (run_dir / "source_runs.json").write_text(json.dumps(source_runs, indent=2), encoding="utf-8")
    output_files.append("source_runs.json")

    params = {
        "cases": case_info,
        "methods": list(METHODS),
        "n_max": N_MAX,
        "n_grid": list(N_GRID),
        "nested_subset_rule": "first N members (indices 0..N-1) of the single N_MAX-member ensemble of each run",
        "metrics": list(METRICS),
        "coverage_nominal_p": NOMINAL_P_COVERAGE,
        "umg_bins": DEFAULT_BINS,
        "evaluation_definitions": {
            "mask": "conditioning_cell_mask of the regenerated conditioning samples (same as evaluate_one_cell)",
            "sgs": "mean/var(ddof=1) over first N realizations; MSE from that mean; UMG/coverage90 from N(mean, sqrt(var))",
            "rbf_bootstrap": ("mean/var(ddof=1) over first N bootstrap replicates for UMG/coverage90/variance_mean; "
                              "MSE from point_estimate_map (N-independent), as in evaluate_one_cell"),
        },
        "sgs_seeds": SGS_SEEDS,
        "bootstrap_seeds": BOOTSTRAP_SEEDS,
        "fixed_seeds_not_varied": {
            "sgs_jitter_seed": sgs.SGS_JITTER_SEED,
            "rbf_cv_seed": rbf_bootstrap.CV_SEED,
        },
        "project_default_n_ensemble": rbf_bootstrap.N_BOOTSTRAP,
        "runs_were_executed_by_this_invocation": runs_were_executed_here,
        "reused_source_runs_path": args.reuse_source_runs,
        "parallel": {
            "max_workers": max_workers,
            "n_tasks": len(CASES) * N_SEEDS * len(METHODS),
            "os_cpu_count": os.cpu_count(),
            "blas_threads_per_process": 1,
            "note": "all tasks ran concurrently in one process pool; per-run seconds are under that load",
        },
        "pool_wall_seconds": pool_wall_seconds if runs_were_executed_here else None,
    }
    manifest_path = save_result(
        experiment=EXPERIMENT_NAME,
        params=params,
        seed_or_seeds={"sgs_seeds": SGS_SEEDS, "bootstrap_seeds": BOOTSTRAP_SEEDS,
                       "truth_seed": TRUTH_SEED,
                       "sample_seeds": {c: CASES[c]["sample_seed"] for c in CASES}},
        run_dir=run_dir,
        code_entrypoint=CODE_ENTRYPOINT,
        output_files=output_files,
    )
    print(f"\nRun directory: {run_dir}\nManifest: {manifest_path}")
    print(timings_df.drop(columns=["run_dir", "ensemble_seconds_definition"]).to_string(index=False))
    if runs_were_executed_here:
        print(f"pool wall time: {pool_wall_seconds:.1f}s")
    return run_dir


if __name__ == "__main__":
    main()
