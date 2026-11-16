"""F7 -- accuracy / uncertainty-goodness RESPONSE SURFACES over a FIXED
hyperparameter grid (docs/paper_outline.md, conclusion 2: "grid search로도
동시 최적화는 어렵다").

What this produces
------------------
For each of the four methods, MSE, UMG and variance_mean are evaluated at
every point of a 2-D grid of FIXED hyperparameters (no automatic tuning of
any kind), on the base-case ground truth, for several truth realizations.
The MSE-optimal and UMG-optimal grid points can later be located on the same
surface from the tidy table alone (no plotting here, no re-run needed).

NOT computed, by design: any joint accuracy/uncertainty objective (J, a
lambda-weighted combination, a Pareto front, ...). That objective is OUT OF
SCOPE (docs/experiment_context.md section 1, "Explicitly out of scope"). Only
the separate metrics are stored. coverage90 is also not stored (task spec:
mse / umg / variance_mean only); it can be recomputed from the saved maps.

Grid (each method on its own natural parameterization, all sharing ONE range
axis in metres so the four surfaces line up on the x-axis)
-------------------------------------------------------------------------------
range levels RANGE_LEVELS_M = 300 m * 2**(k/2), k = -3..4, rounded to 1 m:
    106, 150, 212, 300, 424, 600, 849, 1200 m   (8 levels; user decision)
    (half-octave steps; the TRUE base-case range 300 m is itself a grid point.)

method         param1 = NATIVE parameter (from range r)   param2
-------------  -----------------------------------------  ----------------------------------------
kriging        variogram_range_m = r                      nugget_fraction_of_sill, NUGGET_LEVELS
               (spherical, isotropic)                     (0, .025, .05, .1, .15, .2, .3, .4, .5) -> 8x9
sgs            SGS_RANGE_LEVELS_M (106,150,300,600,1200)  SGS_NUGGET_LEVELS (0, .05, .1, .2, .4) -> 5x5
               (subset of the kriging grid -- SGS is the expensive method, see runtime below)
rbf_bootstrap  epsilon = sqrt(-ln 0.05) / r  [1/m]        smoothing, RBF_SMOOTHING_LEVELS =
                                                          geomspace(1e-5, 10, 13) (half-decade steps) -> 8x13
gp             length_scale_m = r / sqrt(-2 ln 0.05)      noise_variance_normalized (WhiteKernel, in
                            = r / 2.4477                  normalize_y units), GP_NOISE_LEVELS =
                                                          geomspace(1e-5, 0.5, 11) -> 8x11; optimizer=None

RANGE-AXIS SEMANTICS (for the figure text). The four surfaces share the x-axis
``range_equiv_m`` = r, but each method is run on its NATIVE parameter, which
is what the table stores as param1_name / param1_value; the rule that maps r
to it is stored per row in ``param1_from_range_rule``:
  * kriging / SGS : spherical variogram range = r (cov = 0 beyond r).
  * RBF           : gaussian phi(d) = exp(-(eps d)^2); eps chosen so that phi
                    decays to 0.05 at d = r, i.e. eps = sqrt(-ln 0.05) / r
                    (rbf_bootstrap.py EPSILON_CUTOFF convention).
  * GP            : sklearn RBF k(d) = exp(-d^2 / (2 l^2)); l chosen so that
                    k decays to 0.05 at d = r, i.e. l = r / sqrt(-2 ln 0.05)
                    (project GP practical-range convention, GP_PRACTICAL_RANGE_FACTOR).
  So r is the 0.05-correlation distance for RBF/GP and the spherical range for
  kriging/SGS: equal "range" is an equivalence by convention, not identical
  covariance functions.

The kriging/SGS variogram (range, nugget) is the grid point's value, used AS
GIVEN -- not inferred, and NOT the truth's variogram (except at the grid point
(300 m, 0.05), which equals it). Unit sill on normal-score space, cc1 = 1 -
nug, via base_case_conditioning.build_vario (spherical, isotropic).

GP kernel (same STRUCTURE as gp_mle.build_kernel, every hyperparameter fixed):
    ConstantKernel(c, "fixed") * RBF(length_scale, "fixed") + WhiteKernel(noise, "fixed"),
    GaussianProcessRegressor(optimizer=None, normalize_y=True, alpha=1e-10)
with GP_SIGNAL_MODE = "unit_sill" (user decision): c = 1 - noise, i.e. the
prior sill is 1 in normalize_y units (sample variance of the conditioning
data) and the noise is a FRACTION of it -- the same nug / cc1 split as the
kriging nugget axis; therefore noise < 1 (max level 0.5). The noise enters
through a WhiteKernel (as in gp_mle; user decision), so -- exactly as in
gp_mle -- the predictive variance INCLUDES the noise term. ("unit_signal",
c = 1 with noise on top, is still implemented but not used.)

Ground truths, samples, seeds
-----------------------------
* Truth: base case (range 300 m, nug 0.05, spherical, isotropic, 50x50,
  POR 15 +/- 3), TRUTH_SEEDS = multi_truth_axes.TRUTH_SEEDS (101, 7101..7109)
  -- i.e. exactly the ground truths of multi_truth_axes E2 range=300 m /
  figure F4. Samples: N_SAMPLES = 125 requested (121 actual for truth 101)
  with SAMPLE_SEED = 20, via get_base_case_conditioning_data. The truth and
  the samples depend ONLY on truth_seed: every method and every grid point of
  a truth uses the identical conditioning data (the grid point never touches
  the truth generator).
* N_ENSEMBLE = multi_truth_axes.N_PER_BATCH (50): SGS realizations ==
  bootstrap replicates (project rule), the current production setting.
* SGS_SEED = multi_truth_axes.sgs_batch_seed(1) = 4101, BOOTSTRAP_SEED =
  bootstrap_batch_seed(1) = 4201 (the production batch-1 seeds, REUSED on
  purpose): (a) the same seed at every grid point = common random numbers, so
  differences between grid points are not sampling noise of the ensemble;
  (b) at (300 m, 0.05) the kriging maps and the SGS mean/var maps must be
  bit-identical to the multi_truth_axes E2 range=300 runs of the same truth
  -- checked automatically at the end of the run (crosscheck.json, see below;
  user decision: keep these seeds). SGS_JITTER_SEED (70) and
  KRIGING_VAR_MC_SEED (80) are the methods' fixed values, as everywhere.

Evaluation (identical definitions to evaluate_multi_truth_axes.py, F1-F4)
-------------------------------------------------------------------------
Over the full 50x50 truth grid minus conditioning cells
(src.evaluation.conditioning_cell_mask), in physical porosity units:
  mse           kriging: kriging_mean_map_physical (scalar backtr_value);
                sgs: ensemble mean; rbf: single full-data fit
                (point_estimate_map); gp: posterior mean.
  umg           calc_umg on the default np.linspace(0, 1, 20) grid. kriging:
                kriging_fraction_in (exact per-cell normal-score quantile
                back-transform); others: accuracy_plot_fraction_in under
                N(mean, sqrt(var)), mean = sgs ens. mean / rbf bootstrap mean
                / gp posterior mean.
  variance_mean mean predictive variance over the mask (kriging: Monte-Carlo
                physical-unit map, kriging.py's N_MC / KRIGING_VAR_MC_SEED).

Outputs: results/raw/hyperparam_grid_surface/<timestamp>/
---------------------------------------------------------
  run_config.json      full configuration incl. grids + task list (resume uses it)
  run_registry.jsonl   one line per COMPLETED task (append-only resume ledger)
  failures.jsonl       one line per failed TASK (whole task crashed; retried on
                       --resume) and one line per failed GRID POINT (recorded,
                       not retried -- the point is in the table as failed)
  progress.log         completion log with ETA
  tasks/<task_id>.csv  tidy rows of one task (written by the worker, atomically)
  maps/<task_id>.npz   per OK grid point the maps needed to recompute every
                       metric (SGS realizations / bootstrap replicates are NOT
                       kept: they are regenerable from the recorded seeds)
  metrics.csv          tidy long table, concatenation of all registered tasks:
                       experiment, case, axis, axis_level, truth_seed,
                       sample_seed, method, grid_i, grid_j, range_equiv_m,
                       param1_name, param1_value, param1_from_range_rule,
                       param2_name, param2_value, n_ensemble, point_status
                       ('ok' / 'failed'), failure_reason, sgsim_grid_node_warnings
                       (SGS only: count of sgsim "WARNING: grid node location"
                       lines at that point; empty for other methods), metric,
                       value (NaN for failed points)
  metrics_summary.csv  per (method, grid point, metric): mean / std(ddof=1) / n
                       over the truths with point_status 'ok', plus n_failed and
                       sgsim_warnings_total. Nothing is excluded on the basis of
                       sgsim warnings -- they are only recorded.
  crosscheck.json      written once all tasks are complete: kriging maps
                       (kmap_physical, kmap_ns, vmap_ns, var_physical_mc) and
                       SGS mean/var maps at (300 m, 0.05) vs. the reference
                       multi_truth_axes run (``--crosscheck-reference``,
                       default results/raw/multi_truth_axes/20261116T222609599808Z)
                       for every truth. verdict PASS iff every compared map has
                       max |diff| == 0 (at least one comparison), FAIL otherwise
                       (with the diffs), SKIPPED (with reason) if the reference
                       run / files are missing or nothing could be compared.
  manifest.json        src.io.save_result (rewritten at every session end)

Point-level failures: every grid point is evaluated inside its own
try/except. A point that raises, or whose metrics are non-finite, is
recorded with point_status='failed', failure_reason and NaN values; the
other points of the same row-task are kept. A task whose points are all
recorded (ok or failed) counts as done for --resume.

Task granularity: sgs = one grid point per task (~7-18 min each); kriging /
rbf_bootstrap / gp = one range row (all param2 levels) per task. Tasks are
submitted longest-estimated-first so the long SGS tasks do not form a tail.

Estimated task count / runtime (DEFAULT grid, 10 truths, N=50, 20 workers)
--------------------------------------------------------------------------
Per-grid-point seconds from the running production batch
results/raw/multi_truth_axes/20261116T222609599808Z (20 concurrent workers,
N=50, read-only):
  kriging  kb2d median 28.5 s + MC variance 3.5 s (+ metrics)  -> ~35 s / point
  sgs      measured sgsim seconds by range: 100 m 406, 200 m 749, 300 m 959,
           400 m 1038, 500 m 1070, 600 m 1082; plateau ~1100 s ASSUMED above
           600 m (849 / 1200 m not measured). Linear interpolation between the
           measured ranges, scaled linearly in N.
  rbf      no CV here (the production 13.5 s is CV); bootstrap N=50 ~1 s -> ~1.5 s / point
  gp       no optimizer, no posterior draw (production 15.6 s is the draw) -> ~0.3 s / point
  method   grid  points x truths    tasks   CPU
  kriging  8x9    72 x 10 =  720     80     ~7.0 h
  sgs      5x5    25 x 10 =  250    250     ~57.6 h  (106/150/300/600/1200 m:
                                                     ~427/578/959/1082/1100 s per point)
  rbf      8x13  104 x 10 = 1040     80     ~0.43 h
  gp       8x11   88 x 10 =  880     80     ~0.07 h
  total                     2890    490     ~65 CPU-h
  -> wall time on 20 workers ~3.3 h (~88 % SGS). ``--dry-run`` prints this
     estimate for any configuration without computing anything.

USER DECISIONS applied (coordinator message, 2026-11-16)
-------------------------------------------------------
 1 GP amplitude "unit_sill" (c = 1 - noise); noise geomspace(1e-5, 0.5, 11).
 2 GP noise via WhiteKernel (predictive variance includes noise), as gp_mle.
 3 Production batch-1 seeds (SGS 4101, bootstrap 4201) at every grid point.
 4 Range axis 106..1200 m (8 levels); RBF smoothing geomspace(1e-5, 10, 13);
   kriging nugget levels unchanged -> kriging 8x9.
 5 SGS 5x5 sub-grid (106, 150, 300, 600, 1200 m x 0, .05, .1, .2, .4), 10 truths.
 6 Native parameter + conversion rule stored per row (param1_*), documented above.
 7 Point-level failure recording + per-point sgsim warning count; nothing
   excluded automatically.
 8 Cross-check against multi_truth_axes as the final step of the run
   (crosscheck.json in the run dir; no writes to results/processed/).

Usage (nothing has been run yet)
--------------------------------
  estimate only:  .venv/Scripts/python.exe -m src.experiments.hyperparam_grid_surface --dry-run
  smoke:          .venv/Scripts/python.exe -m src.experiments.hyperparam_grid_surface --smoke --max-workers 4
                  (1 truth, 2x2 grid per method, N=5; is_production_configuration=false)
  production:     .venv/Scripts/python.exe -m src.experiments.hyperparam_grid_surface --max-workers 20
  resume:         .venv/Scripts/python.exe -m src.experiments.hyperparam_grid_surface --resume results/raw/hyperparam_grid_surface/<ts> --max-workers 20
"""

import argparse
import contextlib
import io
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import geostatspy.geostats as geostats

from src.evaluation import (
    accuracy_plot_fraction_in,
    calc_umg,
    conditioning_cell_mask,
    kriging_fraction_in,
    mse,
)
from src.experiments import kriging as _krig
from src.experiments import rbf_bootstrap as _rbf
from src.experiments import sgs as _sgs
from src.experiments._axis_parallel import _BLAS_THREAD_ENV_VARS, default_max_workers
from src.experiments.base_case import NX, NY, XMN, XSIZ, YMN, YSIZ
from src.experiments.base_case_conditioning import (
    HMAJ1,
    HMIN1,
    N_SAMPLES,
    NUG,
    SAMPLE_SEED,
    VCOL,
    build_vario,
    get_base_case_conditioning_data,
)
from src.experiments.evaluate_sample_replicate_axis import (
    CORRELATION_CUTOFF,
    GP_PRACTICAL_RANGE_FACTOR,
    safe_sqrt_variance,
)
from src.experiments.multi_truth_axes import (
    N_PER_BATCH,
    TRUTH_SEEDS as _MTA_TRUTH_SEEDS,
    _append_jsonl,
    _fmt_seconds,
    _now,
    bootstrap_batch_seed,
    config_key as mta_config_key,
    read_jsonl,
    rel_or_abs,
    resolve_run_dir,
    sgs_batch_seed,
)
from src.grid_utils import full_grid_coordinates
from src.io import get_git_commit, get_git_status, make_run_dir, save_result
from src.nscore import nscore as nscore_corrected

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXPERIMENT_NAME = "hyperparam_grid_surface"
CODE_ENTRYPOINT = "src/experiments/hyperparam_grid_surface.py"
FIGURE_ID = "F7"
CASE = "base_case"
AXIS = "hyperparameter_grid"
DEFAULT_CROSSCHECK_REFERENCE = "results/raw/multi_truth_axes/20261116T222609599808Z"
CROSSCHECK_FILE = "crosscheck.json"

METHODS = ("kriging", "sgs", "rbf_bootstrap", "gp")
METRICS = ("mse", "umg", "variance_mean")

# --- Ground truth: the base case, unchanged (imported, not re-declared) ----
TRUTH_HMAJ1 = float(HMAJ1)      # 300 m
TRUTH_HMIN1 = float(HMIN1)      # 300 m (isotropic)
TRUTH_NUG = float(NUG)          # 0.05
TRUTH_N_SAMPLES = int(N_SAMPLES)  # 125 requested
TRUTH_SAMPLE_SEED = int(SAMPLE_SEED)  # 20
TRUTH_SEEDS = tuple(int(s) for s in _MTA_TRUTH_SEEDS)  # 101, 7101..7109

# --- Ensembles (production setting of multi_truth_axes) --------------------
N_ENSEMBLE = int(N_PER_BATCH)   # 50, SGS realizations == bootstrap replicates
SGS_SEED = sgs_batch_seed(1)    # 4101 (reused on purpose, see docstring D3)
BOOTSTRAP_SEED = bootstrap_batch_seed(1)  # 4201

# --- Grid ------------------------------------------------------------------
RANGE_CENTER_M = TRUTH_HMAJ1
RANGE_HALF_OCTAVE_K = tuple(range(-3, 5))
RANGE_LEVELS_M = tuple(float(round(RANGE_CENTER_M * 2.0 ** (k / 2.0))) for k in RANGE_HALF_OCTAVE_K)
# -> (106, 150, 212, 300, 424, 600, 849, 1200)   (75 m dropped, user decision 4)
NUGGET_LEVELS = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
SGS_RANGE_LEVELS_M = (106.0, 150.0, 300.0, 600.0, 1200.0)
SGS_NUGGET_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.4)
RBF_SMOOTHING_LEVELS = tuple(float(v) for v in np.geomspace(1e-5, 10.0, 13))
GP_NOISE_LEVELS = tuple(float(v) for v in np.geomspace(1e-5, 0.5, 11))

# --- RBF / GP fixed-parameter settings --------------------------------------
RBF_EPSILON_CUTOFF = float(_rbf.EPSILON_CUTOFF)   # 0.05, rbf_bootstrap.py convention
RBF_KERNEL = _rbf.RBF_KERNEL                      # 'gaussian'
GP_SIGNAL_MODE = "unit_sill"                       # c = 1 - noise (user decision 1)
GP_SIGNAL_VARIANCE_NORMALIZED = 1.0                # only used in the (unused) "unit_signal" mode
GP_ALPHA_DIAG = 1e-10                              # sklearn default, = what gp_mle uses implicitly
GP_NORMALIZE_Y = True                              # as gp_mle.build_gpr

PARAM_NAMES = {
    "kriging": ("variogram_range_m", "nugget_fraction_of_sill"),
    "sgs": ("variogram_range_m", "nugget_fraction_of_sill"),
    "rbf_bootstrap": ("epsilon", "smoothing"),
    "gp": ("length_scale_m", "noise_variance_normalized"),
}
# One task per grid point ("point") or per range row ("row").
TASK_GRANULARITY = {"kriging": "row", "sgs": "point", "rbf_bootstrap": "row", "gp": "row"}

# --- Runtime model (seconds per grid point; see module docstring) -----------
EST_SECONDS_PER_POINT = {"kriging": 35.0, "rbf_bootstrap": 1.5, "gp": 0.3}
# Measured sgsim seconds per SGS point (production run, 20 workers, N=50);
# the last entry is the ASSUMED plateau above 600 m (not measured).
EST_SGS_SECONDS_BY_RANGE = ((100.0, 406.0), (200.0, 749.0), (300.0, 959.0), (400.0, 1038.0),
                            (500.0, 1070.0), (600.0, 1082.0))
EST_SGS_PLATEAU_SECONDS = 1100.0
EST_SGS_REFERENCE_N = 50
EST_SGS_MIN_SECONDS = 60.0

PARAM1_FROM_RANGE_RULE = {
    "kriging": "spherical variogram range = range_equiv_m",
    "sgs": "spherical variogram range = range_equiv_m",
    "rbf_bootstrap": "epsilon = sqrt(-ln 0.05) / range_equiv_m (gaussian phi=0.05 at range_equiv_m)",
    "gp": "length_scale = range_equiv_m / sqrt(-2 ln 0.05) (sklearn RBF k=0.05 at range_equiv_m)",
}

# --- Smoke configuration -----------------------------------------------------
SMOKE = {
    "truth_seeds": (TRUTH_SEEDS[0],),
    "range_levels_m": (150.0, 300.0),
    "nugget_levels": (0.05, 0.2),
    "sgs_range_levels_m": (150.0, 300.0),
    "sgs_nugget_levels": (0.05, 0.2),
    "rbf_smoothing_levels": (0.1, 1.0),
    "gp_noise_levels": (0.01, 0.1),
    "n_ensemble": 5,
}

# --- Import-time sanity checks ---------------------------------------------
if TRUTH_HMAJ1 not in RANGE_LEVELS_M:
    raise ValueError(f"true range {TRUTH_HMAJ1} must be a grid level: {RANGE_LEVELS_M}")
if TRUTH_NUG not in NUGGET_LEVELS or TRUTH_NUG not in SGS_NUGGET_LEVELS:
    raise ValueError("true nugget must be a kriging and an SGS grid level")
if not set(SGS_RANGE_LEVELS_M) <= set(RANGE_LEVELS_M) or not set(SGS_NUGGET_LEVELS) <= set(NUGGET_LEVELS):
    raise ValueError("SGS sub-grid must be a subset of the kriging grid")
if abs(GP_PRACTICAL_RANGE_FACTOR - math.sqrt(-2.0 * math.log(CORRELATION_CUTOFF))) > 1e-12:
    raise ValueError("GP_PRACTICAL_RANGE_FACTOR no longer the 0.05-cutoff convention")
if RBF_EPSILON_CUTOFF != CORRELATION_CUTOFF:
    raise ValueError("rbf_bootstrap.EPSILON_CUTOFF differs from the evaluation CORRELATION_CUTOFF")
if RBF_KERNEL != "gaussian":
    raise ValueError("epsilon conversion below assumes the gaussian RBF kernel")


# =============================================================================
# Parameter conversions / grid
# =============================================================================
def rbf_epsilon_from_range(range_m: float) -> float:
    """rbf_bootstrap.py: gaussian phi(r) = exp(-(eps r)^2) decays to
    EPSILON_CUTOFF at r = sqrt(-ln cutoff) / eps."""
    return float(np.sqrt(-np.log(RBF_EPSILON_CUTOFF)) / float(range_m))


def gp_length_scale_from_range(range_m: float) -> float:
    """Inverse of the project's GP practical range = sqrt(-2 ln 0.05) * l."""
    return float(range_m) / GP_PRACTICAL_RANGE_FACTOR


def method_axes(method: str, cfg: dict) -> Tuple[List[float], List[float]]:
    """(range levels in m, param2 levels) of ``method``'s grid."""
    g = cfg["grids"][method]
    return list(g["range_levels_m"]), list(g["param2_levels"])


def grid_point(method: str, cfg: dict, i: int, j: int) -> dict:
    ranges, p2 = method_axes(method, cfg)
    r, v2 = float(ranges[i]), float(p2[j])
    n1, n2 = PARAM_NAMES[method]
    if method in ("kriging", "sgs"):
        v1 = r
    elif method == "rbf_bootstrap":
        v1 = rbf_epsilon_from_range(r)
    else:
        v1 = gp_length_scale_from_range(r)
    return {"grid_i": int(i), "grid_j": int(j), "range_equiv_m": r,
            "param1_name": n1, "param1_value": float(v1), "param1_from_range_rule": PARAM1_FROM_RANGE_RULE[method],
            "param2_name": n2, "param2_value": v2}


def build_grids(range_levels, nugget_levels, sgs_range_levels, sgs_nugget_levels,
                rbf_smoothing_levels, gp_noise_levels) -> dict:
    grids = {
        "kriging": {"range_levels_m": [float(v) for v in range_levels],
                    "param2_levels": [float(v) for v in nugget_levels]},
        "sgs": {"range_levels_m": [float(v) for v in sgs_range_levels],
                "param2_levels": [float(v) for v in sgs_nugget_levels]},
        "rbf_bootstrap": {"range_levels_m": [float(v) for v in range_levels],
                          "param2_levels": [float(v) for v in rbf_smoothing_levels]},
        "gp": {"range_levels_m": [float(v) for v in range_levels],
               "param2_levels": [float(v) for v in gp_noise_levels]},
    }
    for m, g in grids.items():
        if any(r <= 0 for r in g["range_levels_m"]):
            raise ValueError(f"{m}: range levels must be > 0")
        if m in ("kriging", "sgs") and any(not 0.0 <= v <= 1.0 for v in g["param2_levels"]):
            raise ValueError(f"{m}: nugget fraction must be in [0, 1]")
        if m == "rbf_bootstrap" and any(v < 0 for v in g["param2_levels"]):
            raise ValueError("smoothing must be >= 0")
        if m == "gp" and any(v <= 0 for v in g["param2_levels"]):
            raise ValueError("GP noise variance must be > 0")
        if m == "gp" and GP_SIGNAL_MODE == "unit_sill" and any(v >= 1.0 for v in g["param2_levels"]):
            raise ValueError("GP_SIGNAL_MODE 'unit_sill' needs noise < 1")
    return grids


def task_id(method: str, truth_seed: int, i: int, j: Optional[int]) -> str:
    tid = f"{method}__t{int(truth_seed)}__i{int(i):02d}"
    return tid if j is None else f"{tid}__j{int(j):02d}"


def est_sgs_seconds(range_m: float, n_ensemble: int) -> float:
    """Linear interpolation in the measured EST_SGS_SECONDS_BY_RANGE table;
    proportional to range below 100 m, assumed plateau above 600 m; linear in N."""
    xs = [r for r, _ in EST_SGS_SECONDS_BY_RANGE]
    ys = [s for _, s in EST_SGS_SECONDS_BY_RANGE]
    r = float(range_m)
    if r < xs[0]:
        base = ys[0] * r / xs[0]
    elif r > xs[-1]:
        base = EST_SGS_PLATEAU_SECONDS
    else:
        base = float(np.interp(r, xs, ys))
    return max(EST_SGS_MIN_SECONDS, base) * float(n_ensemble) / EST_SGS_REFERENCE_N


def build_tasks(cfg: dict) -> List[dict]:
    tasks = []
    for method in cfg["methods"]:
        ranges, p2 = method_axes(method, cfg)
        for t in cfg["truth_seeds"]:
            for i, r in enumerate(ranges):
                if TASK_GRANULARITY[method] == "point":
                    js_list = [[j] for j in range(len(p2))]
                else:
                    js_list = [list(range(len(p2)))]
                for js in js_list:
                    jj = js[0] if TASK_GRANULARITY[method] == "point" else None
                    if method == "sgs":
                        est = est_sgs_seconds(r, cfg["n_ensemble"]) * len(js)
                    else:
                        est = EST_SECONDS_PER_POINT[method] * len(js)
                    tasks.append({"task_id": task_id(method, t, i, jj), "method": method,
                                  "truth_seed": int(t), "i": int(i), "js": js,
                                  "est_seconds": float(est)})
    # longest-estimated first (LPT) so the expensive SGS tasks do not form a tail
    tasks.sort(key=lambda d: -d["est_seconds"])
    return tasks


def runtime_estimate(tasks: List[dict], max_workers: int) -> dict:
    by_method: Dict[str, dict] = {}
    for t in tasks:
        d = by_method.setdefault(t["method"], {"tasks": 0, "points": 0, "cpu_s": 0.0})
        d["tasks"] += 1
        d["points"] += len(t["js"])
        d["cpu_s"] += t["est_seconds"]
    cpu = sum(d["cpu_s"] for d in by_method.values())
    longest = max((t["est_seconds"] for t in tasks), default=0.0)
    return {
        "by_method": by_method,
        "n_tasks": len(tasks),
        "cpu_hours": cpu / 3600.0,
        "max_workers": int(max_workers),
        "wall_hours": max(cpu / max_workers, longest) / 3600.0,
        "basis": "per-point seconds measured in results/raw/multi_truth_axes/20261116T222609599808Z "
                 "(20 concurrent workers, N=50); SGS: measured 100-600 m, plateau "
                 f"{EST_SGS_PLATEAU_SECONDS:.0f} s assumed above 600 m",
    }


# =============================================================================
# Worker side
# =============================================================================
@lru_cache(maxsize=4)
def _truth_context(truth_seed: int):
    """Truth, conditioning samples, mask and normal-score transform for one
    truth -- shared by every method / grid point of that truth (cached per
    worker process). Uses the base-case truth parameters ONLY."""
    truth, samples_df = get_base_case_conditioning_data(
        sample_seed=TRUTH_SAMPLE_SEED, truth_seed=int(truth_seed), hmaj1=TRUTH_HMAJ1,
        hmin1=TRUTH_HMIN1, n_samples=TRUTH_N_SAMPLES, nug=TRUTH_NUG,
    )
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    ns, vr, vrg, corrections = nscore_corrected(samples_df, VCOL)
    samples_ns_df = samples_df.copy()
    samples_ns_df["NPor"] = ns
    return {
        "truth": truth, "samples_df": samples_df, "samples_ns_df": samples_ns_df, "mask": mask,
        "truth_m": truth[mask], "vr": vr, "vrg": vrg, "nscore_corrections": corrections,
        "X": samples_df[["X", "Y"]].values, "d": samples_df[VCOL].values,
        "grid_coords": full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ),
    }


def _gaussian_metrics(ctx, point_map, mean_map, var_map, label) -> dict:
    """Same arithmetic as evaluate_multi_truth_axes._gaussian_metrics (default
    20-bin UMG; coverage90 not stored)."""
    mask = ctx["mask"]
    var_m = var_map[mask]
    std_m = safe_sqrt_variance(var_m, label)
    p, frac = accuracy_plot_fraction_in(ctx["truth_m"], mean_map[mask], std_m)
    return {"mse": mse(ctx["truth_m"], point_map[mask]), "umg": calc_umg(p, frac),
            "variance_mean": float(np.mean(var_m))}


def _eval_kriging(ctx, range_m, nug, cfg) -> Tuple[dict, dict, dict]:
    """kriging.main's computation with a GIVEN (range, nugget) variogram."""
    vario = build_vario(hmaj1=range_m, hmin1=range_m, nug=nug)
    sdf = ctx["samples_ns_df"]
    n_used = int(((sdf["NPor"] >= _krig.KB2D_TMIN) & (sdf["NPor"] <= _krig.KB2D_TMAX)).sum())
    if n_used != len(sdf):
        raise RuntimeError(f"kb2d tmin/tmax would drop {len(sdf) - n_used} conditioning samples")
    kmap_ns, vmap_ns = geostats.kb2d(
        sdf, "X", "Y", "NPor", _krig.KB2D_TMIN, _krig.KB2D_TMAX, NX, XMN, XSIZ, NY, YMN, YSIZ,
        nxdis=_krig.NXDIS, nydis=_krig.NYDIS, ndmin=_krig.NDMIN, ndmax=_krig.NDMAX,
        radius=_krig.RADIUS, ktype=_krig.KTYPE, skmean=_krig.SKMEAN_NS, vario=vario,
    )
    vr, vrg = ctx["vr"], ctx["vrg"]
    bt = dict(zmin=_krig.BACKTR_ZMIN, zmax=_krig.BACKTR_ZMAX, ltail=_krig.LTAIL, ltpar=_krig.LTPAR,
              utail=_krig.UTAIL, utpar=_krig.UTPAR)
    kmap_physical = np.empty((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            kmap_physical[iy, ix] = geostats.backtr_value(kmap_ns[iy, ix], vr, vrg, **bt)
    # Monte-Carlo physical-unit variance, identical to kriging.main
    std_ns_flat = np.sqrt(np.clip(vmap_ns, 0.0, None)).ravel()
    kmap_ns_flat = kmap_ns.ravel()
    mc_rng = np.random.default_rng(_krig.KRIGING_VAR_MC_SEED)
    ns_draws = mc_rng.normal(loc=kmap_ns_flat[:, None], scale=std_ns_flat[:, None],
                             size=(kmap_ns_flat.shape[0], _krig.N_MC))
    physical_draws = _krig.backtr_value_vectorized(
        ns_draws, vr, vrg, _krig.BACKTR_ZMIN, _krig.BACKTR_ZMAX, _krig.LTAIL, _krig.LTPAR,
        _krig.UTAIL, _krig.UTPAR)
    var_phys_mc = physical_draws.var(axis=1, ddof=1).reshape(NY, NX)
    del ns_draws, physical_draws

    mask = ctx["mask"]
    std_ns_m = safe_sqrt_variance(vmap_ns[mask], "kriging vmap_ns")
    p, frac = kriging_fraction_in(ctx["truth_m"], kmap_ns[mask], std_ns_m, vr, vrg,
                                  _krig.BACKTR_ZMIN, _krig.BACKTR_ZMAX, _krig.LTAIL, _krig.LTPAR,
                                  _krig.UTAIL, _krig.UTPAR)
    kvar_m = var_phys_mc[mask]
    safe_sqrt_variance(kvar_m, "kriging variance_mean")
    metrics = {"mse": mse(ctx["truth_m"], kmap_physical[mask]), "umg": calc_umg(p, frac),
               "variance_mean": float(np.mean(kvar_m))}
    maps = {"kmap_physical": kmap_physical, "kmap_ns": kmap_ns, "vmap_ns": vmap_ns,
            "var_physical_mc": var_phys_mc}
    return metrics, maps, {}


def _eval_sgs(ctx, range_m, nug, cfg) -> Tuple[dict, dict, dict]:
    """sgs.main's computation (jitter copy, sgsim kwargs, post-hoc honoring,
    ddof=1 ensemble moments) with a GIVEN (range, nugget) variogram."""
    n_real = int(cfg["n_ensemble"])
    vario = build_vario(hmaj1=range_m, hmin1=range_m, nug=nug)
    samples_df = ctx["samples_df"]
    n = len(samples_df)
    jitter_rng = np.random.RandomState(_sgs.SGS_JITTER_SEED)
    sgsim_input_df = samples_df.copy()
    sgsim_input_df["X"] = sgsim_input_df["X"] + jitter_rng.uniform(
        -_sgs.SGS_JITTER_MAGNITUDE, _sgs.SGS_JITTER_MAGNITUDE, size=n)
    sgsim_input_df["Y"] = sgsim_input_df["Y"] + jitter_rng.uniform(
        -_sgs.SGS_JITTER_MAGNITUDE, _sgs.SGS_JITTER_MAGNITUDE, size=n)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = geostats.sgsim(
            sgsim_input_df, "X", "Y", VCOL, wcol=-1, scol=-1, tmin=-9999, tmax=9999,
            itrans=1, ismooth=0, dftrans=0, tcol=0, twtcol=0,
            zmin=_sgs.SGSIM_ZMIN, zmax=_sgs.SGSIM_ZMAX, ltail=_sgs.SGSIM_LTAIL,
            ltpar=_sgs.SGSIM_LTPAR, utail=_sgs.SGSIM_UTAIL, utpar=_sgs.SGSIM_UTPAR,
            nsim=n_real, nx=NX, xmn=XMN, xsiz=XSIZ, ny=NY, ymn=YMN, ysiz=YSIZ,
            seed=int(cfg["sgs_seed"]), ndmin=_sgs.NDMIN, ndmax=_sgs.NDMAX, nodmax=_sgs.NODMAX,
            mults=_sgs.MULTS, nmult=_sgs.NMULT, noct=_sgs.NOCT, ktype=_sgs.KTYPE,
            colocorr=_sgs.COLOCORR, sec_map=_sgs.SEC_MAP, vario=vario,
        )
    n_warn = sum(1 for line in buf.getvalue().splitlines()
                 if line.strip().startswith(_sgs.SGSIM_NODE_WARNING_PREFIX))
    realizations = np.stack([sim[k] for k in range(n_real)], axis=0)
    row = NY - 1 - np.round((samples_df["Y"].values - YMN) / YSIZ).astype(int)
    col = np.round((samples_df["X"].values - XMN) / XSIZ).astype(int)
    honor_raw_max = float(np.abs(realizations[:, row, col] - samples_df[VCOL].values[None, :]).max())
    if _sgs.APPLY_HONOR_REENFORCEMENT:
        realizations[:, row, col] = samples_df[VCOL].values[None, :]
    mean_map = realizations.mean(axis=0)
    var_map = realizations.var(axis=0, ddof=1)
    metrics = _gaussian_metrics(ctx, mean_map, mean_map, var_map, f"sgs r={range_m} nug={nug}")
    diag = {"sgsim_grid_node_warnings": int(n_warn), "honor_raw_max_abs_error": honor_raw_max}
    return metrics, {"mean": mean_map, "var": var_map}, diag


def _eval_rbf(ctx, range_m, smoothing, cfg) -> Tuple[dict, dict, dict]:
    """rbf_bootstrap.main's point fit + bootstrap loop (incl. its
    smoothing==0 dedup gate) at a GIVEN (epsilon, smoothing); no CV."""
    eps = rbf_epsilon_from_range(range_m)
    X, d, grid = ctx["X"], ctx["d"], ctx["grid_coords"]
    n = len(d)
    point_map = RBFInterpolator(X, d, kernel=RBF_KERNEL, epsilon=eps, smoothing=smoothing)(grid).reshape(NY, NX)
    rng = np.random.RandomState(int(cfg["bootstrap_seed"]))
    n_boot = int(cfg["n_ensemble"])
    reps = np.empty((n_boot, NY, NX))
    n_dedup = 0
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_fit, d_fit = X[idx], d[idx]
        if smoothing == 0.0:
            X_u, u_idx = np.unique(X_fit, axis=0, return_index=True)
            if len(X_u) < len(X_fit):
                n_dedup += 1
                X_fit, d_fit = X_u, d_fit[u_idx]
        reps[b] = RBFInterpolator(X_fit, d_fit, kernel=RBF_KERNEL, epsilon=eps,
                                  smoothing=smoothing)(grid).reshape(NY, NX)
    if not (np.all(np.isfinite(reps)) and np.all(np.isfinite(point_map))):
        raise RuntimeError(f"RBF r={range_m} smoothing={smoothing}: non-finite prediction")
    mean_map = reps.mean(axis=0)
    var_map = reps.var(axis=0, ddof=1)
    metrics = _gaussian_metrics(ctx, point_map, mean_map, var_map, f"rbf r={range_m} sm={smoothing}")
    return metrics, {"point": point_map, "mean": mean_map, "var": var_map}, {"n_bootstrap_deduplicated": n_dedup}


def build_fixed_gpr(length_scale: float, noise: float, signal_mode: str = GP_SIGNAL_MODE,
                    signal_variance: float = GP_SIGNAL_VARIANCE_NORMALIZED) -> GaussianProcessRegressor:
    """gp_mle.build_kernel's structure with every hyperparameter FIXED and the
    optimizer disabled (docstring D1/D2)."""
    if signal_mode == "unit_signal":
        signal = float(signal_variance)
    elif signal_mode == "unit_sill":
        if not 0.0 < float(noise) < 1.0:
            raise ValueError("signal_mode 'unit_sill' needs 0 < noise < 1")
        signal = 1.0 - float(noise)
    else:
        raise ValueError(f"unknown GP signal mode {signal_mode}")
    kernel = (ConstantKernel(signal, constant_value_bounds="fixed")
              * RBF(length_scale=float(length_scale), length_scale_bounds="fixed")
              + WhiteKernel(noise_level=float(noise), noise_level_bounds="fixed"))
    return GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=GP_NORMALIZE_Y,
                                    alpha=GP_ALPHA_DIAG)


def _eval_gp(ctx, range_m, noise, cfg) -> Tuple[dict, dict, dict]:
    ls = gp_length_scale_from_range(range_m)
    # mode / amplitude read from the STORED config (resume-safe), not the module constants
    gpr = build_fixed_gpr(ls, noise, cfg["gp"]["signal_mode"],
                          cfg["gp"]["signal_variance_normalized_unit_signal"])
    gpr.fit(ctx["X"], ctx["d"])
    fitted_ls = float(gpr.kernel_.k1.k2.length_scale)
    fitted_noise = float(gpr.kernel_.k2.noise_level)
    if not (np.isclose(fitted_ls, ls, rtol=1e-12) and np.isclose(fitted_noise, noise, rtol=1e-12)):
        raise AssertionError("GP hyperparameters changed during fit (optimizer should be disabled)")
    mean, std = gpr.predict(ctx["grid_coords"], return_std=True)
    mean_map = mean.reshape(NY, NX)
    var_map = (std ** 2).reshape(NY, NX)
    metrics = _gaussian_metrics(ctx, mean_map, mean_map, var_map, f"gp r={range_m} noise={noise}")
    diag = {"y_train_std": float(gpr._y_train_std),
            "log_marginal_likelihood": float(gpr.log_marginal_likelihood_value_)}
    return metrics, {"mean": mean_map, "var": var_map}, diag


_EVALUATORS = {"kriging": _eval_kriging, "sgs": _eval_sgs, "rbf_bootstrap": _eval_rbf, "gp": _eval_gp}


def _atomic_write_bytes(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def _run_task(task: dict, cfg: dict, exp_dir: str) -> dict:
    """Worker (module-level for Windows spawn pickling). Evaluates every grid
    point of ``task``, writes tasks/<id>.csv + maps/<id>.npz atomically and
    returns a small summary for the registry."""
    t0 = time.time()
    exp_dir = Path(exp_dir)
    method, t_seed, i = task["method"], int(task["truth_seed"]), int(task["i"])
    ctx = _truth_context(t_seed)
    ranges, p2 = method_axes(method, cfg)
    rows, arrays, diags, point_failures = [], {}, {}, []
    for j in task["js"]:
        gp_ = grid_point(method, cfg, i, j)
        # Point-level outcome (user decision 7): a failing point is recorded
        # with NaN metrics + reason; the other points of the task are kept.
        status, reason, maps, diag = "ok", "", {}, {}
        try:
            metrics, maps, diag = _EVALUATORS[method](ctx, float(ranges[i]), float(p2[j]), cfg)
            bad = {k: v for k, v in metrics.items() if not np.isfinite(v)}
            if bad:
                status, reason = "failed", f"non-finite metric(s): {bad!r}"
        except (MemoryError, OSError):
            # transient resource failures -> task-level failure, retried on --resume
            raise
        except Exception as exc:
            status, reason = "failed", repr(exc)
            diag = {"traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}
        if status == "failed":
            metrics = {k: float("nan") for k in METRICS}
            maps = {}
            point_failures.append({"task_id": task["task_id"], "grid_i": i, "grid_j": int(j),
                                   "truth_seed": t_seed, "method": method, "error": reason,
                                   "traceback": diag.get("traceback", "")})
        n_warn = diag.get("sgsim_grid_node_warnings") if method == "sgs" else None
        axis_level = f"{gp_['param1_name']}={gp_['param1_value']:.6g}|{gp_['param2_name']}={gp_['param2_value']:.6g}"
        for metric in METRICS:
            rows.append({
                "experiment": FIGURE_ID, "case": CASE, "axis": AXIS, "axis_level": axis_level,
                "truth_seed": t_seed, "sample_seed": TRUTH_SAMPLE_SEED, "method": method, **gp_,
                "n_ensemble": int(cfg["n_ensemble"]) if method in ("sgs", "rbf_bootstrap") else None,
                "point_status": status, "failure_reason": reason,
                "sgsim_grid_node_warnings": n_warn,
                "metric": metric, "value": float(metrics[metric]),
            })
        for name, arr in maps.items():
            arrays[f"j{int(j):02d}__{name}"] = arr
        diags[f"j{int(j):02d}"] = {"point_status": status, "failure_reason": reason,
                                   **{k: v for k, v in diag.items() if k != "traceback"}}
    arrays["mask"] = ctx["mask"]
    arrays["samples_xyv"] = ctx["samples_df"][["X", "Y", VCOL]].values
    if method == "kriging":
        arrays["nscore_vr"] = ctx["vr"]
        arrays["nscore_vrg"] = ctx["vrg"]

    (exp_dir / "tasks").mkdir(exist_ok=True)
    csv_path = exp_dir / "tasks" / f"{task['task_id']}.csv"
    df = pd.DataFrame(rows)
    _atomic_write_bytes(csv_path, lambda f: f.write(df.to_csv(index=False).encode("utf-8")))
    npz_path = None
    if cfg["save_maps"]:
        (exp_dir / "maps").mkdir(exist_ok=True)
        npz_path = exp_dir / "maps" / f"{task['task_id']}.npz"
        _atomic_write_bytes(npz_path, lambda f: np.savez_compressed(f, **arrays))
    return {
        "task_id": task["task_id"], "method": method, "truth_seed": t_seed, "i": i, "js": list(task["js"]),
        "csv": f"tasks/{csv_path.name}", "npz": None if npz_path is None else f"maps/{npz_path.name}",
        "n_samples_actual": int(len(ctx["samples_df"])), "n_cells_evaluated": int(ctx["mask"].sum()),
        "nscore_n_corrected": int(ctx["nscore_corrections"]["n_corrected"]) if method == "kriging" else None,
        "n_points": len(task["js"]), "n_points_failed": len(point_failures),
        "point_failures": point_failures,
        "diagnostics": diags, "seconds": time.time() - t0, "est_seconds": task["est_seconds"],
    }


# =============================================================================
# Parent side: configuration, pool, bookkeeping
# =============================================================================
def make_config(args) -> dict:
    if args.smoke:
        s = SMOKE
        truth_seeds, n_ens = tuple(s["truth_seeds"]), int(s["n_ensemble"])
        grids = build_grids(s["range_levels_m"], s["nugget_levels"], s["sgs_range_levels_m"],
                            s["sgs_nugget_levels"], s["rbf_smoothing_levels"], s["gp_noise_levels"])
    else:
        truth_seeds, n_ens = TRUTH_SEEDS, N_ENSEMBLE
        grids = build_grids(RANGE_LEVELS_M, NUGGET_LEVELS, SGS_RANGE_LEVELS_M, SGS_NUGGET_LEVELS,
                            RBF_SMOOTHING_LEVELS, GP_NOISE_LEVELS)
    if args.truth_seeds is not None:
        truth_seeds = tuple(int(t) for t in args.truth_seeds)
    if args.n_ensemble is not None:
        n_ens = int(args.n_ensemble)
    if n_ens < 2:
        raise ValueError("n_ensemble must be >= 2 (ddof=1 variance)")
    methods = tuple(m for m in METHODS if m in args.methods)
    is_production = (not args.smoke and truth_seeds == TRUTH_SEEDS and n_ens == N_ENSEMBLE
                     and set(methods) == set(METHODS))
    cfg = {
        "figure": FIGURE_ID,
        "is_production_configuration": bool(is_production),
        "smoke": bool(args.smoke),
        "methods": list(methods),
        "truth_seeds": list(truth_seeds),
        "truth": {"hmaj1": TRUTH_HMAJ1, "hmin1": TRUTH_HMIN1, "nug": TRUTH_NUG, "cc1": 1.0 - TRUTH_NUG,
                  "n_samples_requested": TRUTH_N_SAMPLES, "sample_seed": TRUTH_SAMPLE_SEED,
                  "source": "base_case_conditioning.get_base_case_conditioning_data (base case)"},
        "truth_seed_source": "multi_truth_axes.TRUTH_SEEDS",
        "n_ensemble": n_ens,
        "n_ensemble_source": "multi_truth_axes.N_PER_BATCH (SGS realizations == bootstrap replicates)",
        "sgs_seed": SGS_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "seed_note": "production batch-1 seeds of multi_truth_axes, reused at EVERY grid point "
                     "(common random numbers; exact reproduction of E2 range=300 at (300 m, 0.05))",
        "fixed_seeds": {"sgs_jitter_seed": _sgs.SGS_JITTER_SEED,
                        "kriging_var_mc_seed": _krig.KRIGING_VAR_MC_SEED},
        "grids": grids,
        "param_names": {m: list(v) for m, v in PARAM_NAMES.items()},
        "range_conversions": {
            "rbf_epsilon": f"sqrt(-ln {RBF_EPSILON_CUTOFF}) / range_m (gaussian kernel, rbf_bootstrap.EPSILON_CUTOFF)",
            "gp_length_scale": f"range_m / sqrt(-2 ln {CORRELATION_CUTOFF}) = range_m / {GP_PRACTICAL_RANGE_FACTOR:.6f}",
            "kriging_sgs": "spherical variogram range = range_m, isotropic, unit sill, cc1 = 1 - nug",
        },
        "gp": {"kernel": "ConstantKernel(c, fixed) * RBF(length_scale, fixed) + WhiteKernel(noise, fixed)",
               "signal_mode": GP_SIGNAL_MODE, "signal_variance_normalized_unit_signal": GP_SIGNAL_VARIANCE_NORMALIZED,
               "optimizer": None, "normalize_y": GP_NORMALIZE_Y, "alpha": GP_ALPHA_DIAG},
        "rbf": {"kernel": RBF_KERNEL, "cv": None, "point_estimate": "single full-data fit",
                "dedup_gate": "smoothing == 0.0 (as rbf_bootstrap.py)"},
        "kriging": {"ktype": _krig.KTYPE, "skmean_ns": _krig.SKMEAN_NS, "ndmax": _krig.NDMAX,
                    "radius": _krig.RADIUS, "n_mc": _krig.N_MC,
                    "backtransform": {"zmin": _krig.BACKTR_ZMIN, "zmax": _krig.BACKTR_ZMAX,
                                      "ltail": _krig.LTAIL, "utail": _krig.UTAIL}},
        "sgs": {"ndmax": _sgs.NDMAX, "nodmax": _sgs.NODMAX, "mults": _sgs.MULTS, "nmult": _sgs.NMULT,
                "ktype": _sgs.KTYPE, "jitter_magnitude_m": _sgs.SGS_JITTER_MAGNITUDE,
                "honor_reenforcement": _sgs.APPLY_HONOR_REENFORCEMENT},
        "metrics": list(METRICS),
        "metric_definitions": "as src/experiments/evaluate_multi_truth_axes.py (mask = conditioning_cell_mask; "
                              "UMG on np.linspace(0,1,20)); coverage90 not stored",
        "out_of_scope": "no joint accuracy/uncertainty objective is computed (experiment_context.md section 1)",
        "task_granularity": TASK_GRANULARITY,
        "save_maps": not args.no_maps,
        "raw_root": rel_or_abs(args.raw_root if Path(args.raw_root).is_absolute() else _REPO_ROOT / args.raw_root),
        "param1_from_range_rule": PARAM1_FROM_RANGE_RULE,
        "point_failure_policy": "per grid point: NaN metrics + point_status='failed' + failure_reason; "
                                "other points kept; nothing excluded automatically (sgsim warnings only recorded)",
        "crosscheck_reference": args.crosscheck_reference,
    }
    cfg["tasks"] = build_tasks(cfg)
    cfg["n_tasks_total"] = len(cfg["tasks"])
    cfg["n_grid_points_total"] = sum(len(t["js"]) for t in cfg["tasks"])
    return cfg


def _print_estimate(cfg: dict, max_workers: int) -> dict:
    est = runtime_estimate(cfg["tasks"], max_workers)
    print(f"methods={cfg['methods']} truths={len(cfg['truth_seeds'])} N={cfg['n_ensemble']} "
          f"production={cfg['is_production_configuration']}")
    for m in cfg["methods"]:
        r, p2 = method_axes(m, cfg)
        print(f"  {m:13s} grid {len(r)}x{len(p2)}  ranges={r}  {PARAM_NAMES[m][1]}={[float('%.4g' % v) for v in p2]}")
    for m, d in est["by_method"].items():
        print(f"  {m:13s} tasks={d['tasks']:4d} points={d['points']:5d} CPU~{d['cpu_s'] / 3600:.2f} h")
    print(f"  total tasks={est['n_tasks']}  CPU~{est['cpu_hours']:.1f} h  "
          f"wall on {max_workers} workers ~{est['wall_hours']:.1f} h")
    return est


def _latest_by_task(registry: List[dict]) -> List[dict]:
    """One registry entry per task_id (the last one), so a task re-run after
    its CSV went missing is never concatenated twice."""
    return list({r["task_id"]: r for r in registry}.values())


def _concat_metrics(exp_dir: Path, registry: List[dict]) -> Optional[pd.DataFrame]:
    """Concatenate the registered per-task CSVs into the tidy metrics table
    (no aggregation -- kept separate from _summarize so a summary failure can
    never cost metrics.csv)."""
    latest = _latest_by_task(registry)
    missing = [r["task_id"] for r in latest if not (exp_dir / r["csv"]).exists()]
    if missing:
        print(f"WARNING _concat_metrics: {len(missing)} registered task(s) have no CSV and are NOT in "
              f"metrics.csv (re-run with --resume): {missing[:20]}{' ...' if len(missing) > 20 else ''}",
              flush=True)
    frames = [pd.read_csv(exp_dir / r["csv"]) for r in latest if (exp_dir / r["csv"]).exists()]
    if not frames:
        return None
    metrics = pd.concat(frames, ignore_index=True)
    metrics["n_ensemble"] = metrics["n_ensemble"].astype("Int64")
    return metrics.sort_values(["method", "grid_i", "grid_j", "truth_seed", "metric"],
                               kind="mergesort").reset_index(drop=True)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    grp =["experiment", "case", "axis", "axis_level", "method", "grid_i", "grid_j", "range_equiv_m",
           "param1_name", "param1_value", "param1_from_range_rule", "param2_name", "param2_value",
           "n_ensemble", "metric"]
    ok = metrics[metrics["point_status"] == "ok"]
    summary = (ok.groupby(grp, dropna=False, sort=False)["value"]
               .agg(mean="mean", std=lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, n="count")
               .reset_index())
    extra = (metrics.groupby(grp, dropna=False, sort=False)
             .agg(n_failed=("point_status", lambda s: int((s == "failed").sum())),
                  sgsim_warnings_total=("sgsim_grid_node_warnings", lambda s: float(s.sum(min_count=1))))
             .reset_index())
    # outer merge so a grid point whose truths ALL failed still appears (n=0)
    summary = extra.merge(summary, on=grp, how="left")
    summary["n"] = summary["n"].fillna(0).astype(int)
    return summary


def _write_manifest(exp_dir: Path, cfg: dict, status: str, sessions: list, registry: list,
                    failures: list, output_files: list, estimate: dict,
                    extra: Optional[dict] = None) -> Path:
    params = {k: v for k, v in cfg.items() if k != "tasks"}
    params.update({"status": status, "sessions": sessions, "n_tasks_completed": len(registry),
                   "n_task_failures_logged": len(failures), "runtime_estimate": estimate,
                   "task_list": "run_config.json:tasks"})
    if extra:
        params.update(extra)
    seeds = {"truth_seeds": cfg["truth_seeds"], "sample_seed": TRUTH_SAMPLE_SEED, "sgs_seed": cfg["sgs_seed"],
             "bootstrap_seed": cfg["bootstrap_seed"], **cfg["fixed_seeds"]}
    return save_result(experiment=EXPERIMENT_NAME, params=params, seed_or_seeds=seeds, run_dir=exp_dir,
                       code_entrypoint=CODE_ENTRYPOINT, output_files=output_files)


def run(cfg: dict, exp_dir: Path, max_workers: int) -> int:
    registry_path = exp_dir / "run_registry.jsonl"
    failures_path = exp_dir / "failures.jsonl"
    progress_path = exp_dir / "progress.log"
    (exp_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    registry_path.touch()
    failures_path.touch()
    registry = read_jsonl(registry_path)
    failures = read_jsonl(failures_path)
    done = {r["task_id"] for r in registry if (exp_dir / r["csv"]).exists()}
    todo = [t for t in cfg["tasks"] if t["task_id"] not in done]
    n_total = len(cfg["tasks"])
    estimate = runtime_estimate(todo, max_workers)

    manifest_path = exp_dir / "manifest.json"
    prev = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    sessions = list(prev.get("params", {}).get("sessions", []))
    git_dirty, git_dirty_files = get_git_status()
    session = {"started_utc": _now(), "git_commit": get_git_commit(), "git_dirty": git_dirty,
               "git_dirty_files": git_dirty_files, "max_workers": max_workers,
               "os_cpu_count": os.cpu_count(), "n_tasks_already_done": len(done)}
    sessions.append(session)
    base_outputs = ["run_config.json", "run_registry.jsonl", "failures.jsonl", "progress.log"]
    _write_manifest(exp_dir, cfg, "in_progress", sessions, registry, failures, base_outputs, estimate)

    def _log(msg: str) -> None:
        line = f"{_now()} {msg}"
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    t0 = time.time()
    state = {"done": 0, "failed": 0, "points_failed": 0, "obs_s": 0.0, "est_s_done": 0.0,
             "est_s_left": sum(t["est_seconds"] for t in todo)}
    _log(f"session start: tasks total={n_total}, already done={len(done)}, to run={len(todo)}, "
         f"workers={max_workers}, estimate wall ~{estimate['wall_hours']:.1f} h")

    prev_env = {v: os.environ.get(v) for v in _BLAS_THREAD_ENV_VARS}
    for v in _BLAS_THREAD_ENV_VARS:
        os.environ[v] = "1"   # inherited by spawned workers before numpy import (see _axis_parallel)
    status = "incomplete"
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            cfg_worker = {k: v for k, v in cfg.items() if k != "tasks"}  # do not pickle the task list per task
            futures = {ex.submit(_run_task, t, cfg_worker, str(exp_dir)): t for t in todo}
            try:
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:
                        state["failed"] += 1
                        state["est_s_left"] -= task["est_seconds"]
                        _append_jsonl(failures_path, {
                            "kind": "task", "task_id": task["task_id"], "error": repr(exc),
                            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                            "failed_utc": _now()})
                        _log(f"FAILED {task['task_id']}: {exc!r}")
                        continue
                    res["completed_utc"] = _now()
                    for pf in res.pop("point_failures", []):
                        _append_jsonl(failures_path, {"kind": "grid_point", **pf, "failed_utc": res["completed_utc"]})
                        state["points_failed"] += 1
                        _log(f"POINT FAILED {pf['task_id']} j={pf['grid_j']}: {pf['error']}")
                    _append_jsonl(registry_path, res)
                    state["done"] += 1
                    state["obs_s"] += res["seconds"]
                    state["est_s_done"] += task["est_seconds"]
                    state["est_s_left"] -= task["est_seconds"]
                    ratio = state["obs_s"] / state["est_s_done"] if state["est_s_done"] > 0 else 1.0
                    eta = state["est_s_left"] * ratio / max_workers
                    _log(f"[{len(done) + state['done']}/{n_total}] done {res['task_id']} ({res['seconds']:.0f}s, "
                         f"est {task['est_seconds']:.0f}s) | elapsed {_fmt_seconds(time.time() - t0)} | "
                         f"ETA ~{_fmt_seconds(eta)} (observed/estimated {ratio:.2f}) | failures {state['failed']}")
            except BaseException:
                n_cancel = sum(1 for f in futures if f.cancel())
                _log(f"aborting: cancelled {n_cancel} not-yet-started task(s); running tasks finish first")
                raise
    finally:
        for v, p in prev_env.items():
            if p is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = p
        registry = read_jsonl(registry_path)
        failures = read_jsonl(failures_path)
        # same predicate as `done` above: a task counts only if its CSV exists
        n_done = len({r["task_id"] for r in registry if (exp_dir / r["csv"]).exists()})
        status = "complete" if n_done == n_total else "incomplete"
        outputs = list(base_outputs)
        # metrics.csv first and independently of the summary; neither step may
        # prevent the manifest / crosscheck below (errors -> concat_error.txt).
        concat_errors = []
        metrics = None
        for step in ("metrics.csv", "metrics_summary.csv"):
            try:
                if step == "metrics.csv":
                    metrics = _concat_metrics(exp_dir, registry)
                    if metrics is not None:
                        metrics.to_csv(exp_dir / "metrics.csv", index=False)
                        outputs.append("metrics.csv")
                elif metrics is not None:
                    _summarize(metrics).to_csv(exp_dir / "metrics_summary.csv", index=False)
                    outputs.append("metrics_summary.csv")
            except Exception as exc:
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                concat_errors.append({"step": step, "error": repr(exc), "traceback": tb})
                _log(f"ERROR while writing {step}: {exc!r}\n{tb}")
        if concat_errors:
            try:
                (exp_dir / "concat_error.txt").write_text(
                    "\n\n".join(f"[{_now()}] step={e['step']}\n{e['traceback']}" for e in concat_errors),
                    encoding="utf-8")
                outputs.append("concat_error.txt")
            except Exception as exc:
                _log(f"ERROR writing concat_error.txt: {exc!r}")
        latest = _latest_by_task(registry)
        outputs += sorted(r["csv"] for r in latest) + sorted(r["npz"] for r in latest if r.get("npz"))
        if status == "complete":
            # final step (user decision 8): exact cross-check vs. multi_truth_axes at (300 m, 0.05)
            try:
                cc = crosscheck(exp_dir, cfg)
            except Exception as exc:   # never lose the manifest because of the cross-check
                cc = {"verdict": "SKIPPED", "reason": f"crosscheck raised {exc!r}",
                      "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}
                (exp_dir / CROSSCHECK_FILE).write_text(json.dumps(cc, indent=2), encoding="utf-8")
            outputs.append(CROSSCHECK_FILE)
            _log(f"crosscheck vs multi_truth_axes: verdict={cc['verdict']}"
                 + (f" ({cc.get('reason')})" if cc.get("reason") else ""))
        session.update({"ended_utc": _now(), "wall_seconds": time.time() - t0,
                        "completed_this_session": state["done"], "failed_this_session": state["failed"],
                        "grid_points_failed_this_session": state["points_failed"]})
        mp = _write_manifest(exp_dir, cfg, status, sessions, registry, failures, outputs, estimate,
                             extra={"concat_failed": bool(concat_errors),
                                    "concat_errors": [{"step": e["step"], "error": e["error"]}
                                                      for e in concat_errors],
                                    "concat_error_note": "see concat_error.txt" if concat_errors else ""})
        _log(f"session end: status={status} completed={n_done}/{n_total} task failures this session="
             f"{state['failed']} grid-point failures this session={state['points_failed']} manifest={mp}")
    return 0 if status == "complete" else 1


# =============================================================================
# Cross-check against multi_truth_axes at the true grid point (300 m, 0.05)
# =============================================================================
CROSSCHECK_MAP_PAIRS = {
    "kriging": [("kmap_physical", "kriging_mean_map_physical.npy"), ("kmap_ns", "kmap_ns.npy"),
                ("vmap_ns", "kriging_var_map_ns.npy"), ("var_physical_mc", "kriging_var_map_physical_mc.npy")],
    "sgs": [("mean", "sgs_mean_map.npy"), ("var", "sgs_var_map.npy")],
}


def crosscheck(exp_dir: Path, cfg: dict) -> dict:
    """Final step of a complete run (user decision 8): kriging maps and SGS
    mean/var maps at (TRUTH_HMAJ1, TRUTH_NUG) vs. the multi_truth_axes E2
    range=300 runs of the same truth (cfg['crosscheck_reference']), for every
    truth. Read-only on the reference; writes <exp_dir>/crosscheck.json.

    verdict: PASS iff >= 1 map pair was compared and every compared map has
    max |diff| == 0; FAIL if any compared map differs (diffs listed);
    SKIPPED (with reason) if the reference run / its files are missing or
    nothing could be compared. Individual (method, truth) entries that cannot
    be compared are listed with status 'skipped' and a reason."""
    out_path = exp_dir / CROSSCHECK_FILE
    ref_s = cfg.get("crosscheck_reference")
    out = {"reference_run": ref_s, "point": {"range_m": TRUTH_HMAJ1, "nug": TRUTH_NUG},
           "rule": "PASS iff max |diff| == 0 for every compared map (exact equality)", "results": []}

    def _finish(verdict: str, reason: str = "") -> dict:
        out["verdict"] = verdict
        if reason:
            out["reason"] = reason
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    if not ref_s:
        return _finish("SKIPPED", "no reference run configured")
    mta_dir = resolve_run_dir(ref_s)
    out["reference_run"] = rel_or_abs(mta_dir)
    for f in ("run_config.json", "run_registry.jsonl"):
        if not (mta_dir / f).exists():
            return _finish("SKIPPED", f"reference file missing: {rel_or_abs(mta_dir / f)}")
    mta_cfg = json.loads((mta_dir / "run_config.json").read_text(encoding="utf-8"))
    mta_reg = {(r["config_key"], r["method"]): r["run_dir"] for r in read_jsonl(mta_dir / "run_registry.jsonl")}
    reg = {r["task_id"]: r for r in read_jsonl(exp_dir / "run_registry.jsonl")}
    sgs_comparable = (cfg["n_ensemble"] == mta_cfg.get("n_ensemble_this_batch")
                      and cfg["sgs_seed"] == mta_cfg.get("sgs_seed"))
    out["sgs_comparable"] = sgs_comparable

    def _skip(method, t, reason):
        out["results"].append({"method": method, "truth_seed": t, "status": "skipped", "reason": reason})

    n_compared, any_diff = 0, False
    for method in ("kriging", "sgs"):
        if method not in cfg["methods"]:
            _skip(method, None, "method not in this run")
            continue
        ranges, p2 = method_axes(method, cfg)
        if TRUTH_HMAJ1 not in ranges or TRUTH_NUG not in p2:
            _skip(method, None, "true (range, nugget) is not a grid point of this run")
            continue
        if method == "sgs" and not sgs_comparable:
            _skip(method, None, f"N/seed differ: F7 N={cfg['n_ensemble']} seed={cfg['sgs_seed']}, reference "
                                f"N={mta_cfg.get('n_ensemble_this_batch')} seed={mta_cfg.get('sgs_seed')}")
            continue
        i, j = ranges.index(TRUTH_HMAJ1), p2.index(TRUTH_NUG)
        for t in cfg["truth_seeds"]:
            tid = task_id(method, t, i, j if TASK_GRANULARITY[method] == "point" else None)
            key = mta_config_key(t, TRUTH_SAMPLE_SEED, TRUTH_HMAJ1, TRUTH_HMIN1, TRUTH_NUG, TRUTH_N_SAMPLES)
            if tid not in reg or not reg[tid].get("npz") or not (exp_dir / reg[tid]["npz"]).exists():
                _skip(method, t, f"F7 maps for {tid} not available")
                continue
            if (key, method) not in mta_reg:
                _skip(method, t, f"reference has no {method} run for {key}")
                continue
            theirs_dir = resolve_run_dir(mta_reg[(key, method)])
            ours = np.load(exp_dir / reg[tid]["npz"])
            missing = [f"j{j:02d}__{mine}" for mine, _ in CROSSCHECK_MAP_PAIRS[method]
                       if f"j{j:02d}__{mine}" not in ours.files]
            missing += [rel_or_abs(theirs_dir / theirs) for _, theirs in CROSSCHECK_MAP_PAIRS[method]
                        if not (theirs_dir / theirs).exists()]
            if missing:
                _skip(method, t, f"missing maps/files: {missing} (F7 point failed or reference incomplete)")
                continue
            diffs = {mine: float(np.max(np.abs(ours[f"j{j:02d}__{mine}"] - np.load(theirs_dir / theirs))))
                     for mine, theirs in CROSSCHECK_MAP_PAIRS[method]}
            n_compared += len(diffs)
            any_diff = any_diff or any(v != 0.0 for v in diffs.values())
            out["results"].append({"method": method, "truth_seed": t, "status": "compared",
                                   "reference_run_dir": rel_or_abs(theirs_dir), "max_abs_diff": diffs,
                                   "identical": all(v == 0.0 for v in diffs.values())})
    out["n_maps_compared"] = n_compared
    out["n_entries_skipped"] = sum(1 for r in out["results"] if r["status"] == "skipped")
    if n_compared == 0:
        return _finish("SKIPPED", "nothing could be compared (see results)")
    return _finish("FAIL" if any_diff else "PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description="F7 fixed-hyperparameter MSE/UMG response surfaces")
    ap.add_argument("--smoke", action="store_true", help="1 truth, 2x2 grid per method, N=5")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--truth-seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n-ensemble", type=int, default=None, help=f"default {N_ENSEMBLE} (smoke: {SMOKE['n_ensemble']})")
    ap.add_argument("--no-maps", action="store_true", help="do not write maps/<task>.npz")
    ap.add_argument("--raw-root", default="results/raw")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print grid / task count / runtime estimate only")
    ap.add_argument("--resume", default=None, help="existing hyperparam_grid_surface run folder")
    ap.add_argument("--crosscheck-reference", default=DEFAULT_CROSSCHECK_REFERENCE,
                    help="multi_truth_axes run dir compared against at (300 m, 0.05) as the final step "
                         f"of a complete run (default {DEFAULT_CROSSCHECK_REFERENCE}; ignored on --resume, "
                         "the stored value is used)")
    args = ap.parse_args(argv)

    max_workers = args.max_workers if args.max_workers is not None else default_max_workers()
    if args.resume:
        exp_dir = resolve_run_dir(args.resume)
        cfg = json.loads((exp_dir / "run_config.json").read_text(encoding="utf-8"))
        print(f"resuming {exp_dir} (stored run_config used; grid/method CLI options ignored)")
        _print_estimate(cfg, max_workers)
        if args.dry_run:
            reg = read_jsonl(exp_dir / "run_registry.jsonl")
            done = {r["task_id"] for r in reg if (exp_dir / r["csv"]).exists()}
            remaining = [t for t in cfg["tasks"] if t["task_id"] not in done]
            est = runtime_estimate(remaining, max_workers)
            print(f"dry run (resume): {len(remaining)}/{len(cfg['tasks'])} tasks remaining, "
                  f"CPU~{est['cpu_hours']:.1f} h, wall on {max_workers} workers ~{est['wall_hours']:.1f} h")
            return 0
    else:
        cfg = make_config(args)
        _print_estimate(cfg, max_workers)
        if args.dry_run:
            return 0
        exp_dir = make_run_dir(EXPERIMENT_NAME, args.raw_root)
        print(f"new run folder {exp_dir}")
    return run(cfg, exp_dir, max_workers)


if __name__ == "__main__":
    raise SystemExit(main())
