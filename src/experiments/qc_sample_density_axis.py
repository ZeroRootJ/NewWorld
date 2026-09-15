"""QC checklist for the sample-density axis (3 levels: 5% / 2% / 1% of the
2500-cell grid, n_samples = 125 / 50 / 25).

Collects, per axis level, the specific numbers the orchestrator/reviewer must
see before these results are treated as valid. Nothing here changes any
result -- it reads the pinned runs in
results/processed/sample_density_axis/source_runs.json plus their saved
arrays/manifests, and recomputes a few purely geometric or read-only
diagnostics.

Structure follows src/experiments/qc_range_axis.py (reused, not reinvented),
with three additions specific to THIS axis, where the risk is that small N
breaks something rather than that a long range does:

1. Normal-score transform table size. At n=25 the kriging back-transform
   table has only 25 points, so tail extrapolation matters much more. We
   report the table's vrg range, how many evaluated cells have their point
   estimate / their 95%-interval endpoints / their CRPS-integral EXTREME
   quantiles (tau = 0.5/n_tau and 1 - 0.5/n_tau at the production n_tau=199)
   in the linear-tail-extrapolation region, and how many saturate at
   BACKTR_ZMIN/ZMAX.
2. RBF CV fold size. 5-fold CV on n=25 means 5 points per fold; the per-fold
   train/test sizes are reported alongside the selected hyperparameters so
   the reader can judge whether CV_FOLDS is excessive at this N. Also
   reported: whether CV selected best_smoothing == 0 (which arms the
   duplicate-support-point dedup branch in rbf_bootstrap.py) and how many
   bootstrap replicates were actually deduplicated.
3. SGS near-singular-solve warning count. geostats.sgsim prints
   "WARNING: grid node location ..." to stdout when its local simple-kriging
   solve is near-singular; that is NOT recorded in the run manifest. Because
   sgsim is fully deterministic given (samples, SGS_SEED, jitter seed), this
   script re-executes the identical sgsim call with stdout captured purely to
   COUNT those warnings. It saves nothing and does not touch results/raw/.
   Set RECHECK_SGSIM_WARNINGS = False to skip (it is the slow part).

Run with: .venv/Scripts/python.exe -m src.experiments.qc_sample_density_axis
"""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import geostatspy.geostats as geostats

from src.evaluation import CRPS_DEFAULT_N_TAU, conditioning_cell_mask, crps_tau_grid
from src.experiments.base_case import NX, NY, XMN, YMN, XSIZ, YSIZ
from src.experiments.base_case_conditioning import (
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    build_vario,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.gp_mle import (
    CONSTANT_VALUE_BOUNDS,
    LENGTH_SCALE_BOUNDS,
    NOISE_LEVEL_BOUNDS,
)
from src.experiments.kriging import (
    BACKTR_ZMAX,
    BACKTR_ZMIN,
    LTAIL,
    LTPAR,
    NDMAX as KRIGING_NDMAX,
    UTAIL,
    UTPAR,
    backtr_value_vectorized,
)
from src.experiments.rbf_bootstrap import CV_FOLDS
from src.experiments.sample_density_axis import (
    ALL_AXIS_LEVELS,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    METHODS,
    SAMPLE_COUNTS,
)
from src.experiments import sgs as sgs_module
from src.experiments.sgs import NDMAX as SGS_NDMAX
from src.grid_utils import full_grid_coordinates

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_density_axis"

# A fitted hyperparameter is called "at bound" if it is within this factor of
# either end of its (log-scaled) optimizer bound.
BOUND_PROXIMITY_FACTOR = 1.01

# Absolute tolerance (porosity %) for calling a back-transformed value
# "saturated at" BACKTR_ZMIN/BACKTR_ZMAX.
SATURATION_TOL = 1e-6

QC_INTERVAL_P = 0.95

# Re-run sgsim (stdout captured) purely to count near-singular-solve
# warnings. Slow; nothing is saved.
RECHECK_SGSIM_WARNINGS = True
SGSIM_WARNING_TOKEN = "WARNING: grid node location"


def _manifest(rel_run_dir: str) -> dict:
    return json.loads((_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8"))


def _at_bound(value: float, bounds) -> bool:
    lo, hi = bounds
    return value <= lo * BOUND_PROXIMITY_FACTOR or value >= hi / BOUND_PROXIMITY_FACTOR


def ndmax_capping(samples_df: pd.DataFrame) -> dict:
    """Is each method's fixed search cap actually binding at this sample
    count? Purely geometric: count conditioning samples within one variogram
    range (AXIS_HMAJ1, fixed across this axis) of each grid cell.

    Note the trivial-but-important case this axis creates: when the TOTAL
    sample count is below a method's ndmax, that cap cannot bind anywhere.
    """
    grid = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)  # (n_cells, 2)
    pts = samples_df[["X", "Y"]].values
    d = np.sqrt(
        (grid[:, 0][:, None] - pts[:, 0][None, :]) ** 2
        + (grid[:, 1][:, None] - pts[:, 1][None, :]) ** 2
    )
    n_within = (d <= AXIS_HMAJ1).sum(axis=1)
    n_total = len(samples_df)
    return {
        "kriging_ndmax": KRIGING_NDMAX,
        "sgs_ndmax": SGS_NDMAX,
        "n_samples_within_range_mean": float(n_within.mean()),
        "n_samples_within_range_max": int(n_within.max()),
        "frac_cells_over_kriging_ndmax": float(np.mean(n_within > KRIGING_NDMAX)),
        "frac_cells_over_sgs_ndmax": float(np.mean(n_within > SGS_NDMAX)),
        "kriging_ndmax_can_bind_at_all": bool(n_total > KRIGING_NDMAX),
        "sgs_ndmax_can_bind_at_all": bool(n_total > SGS_NDMAX),
    }


def kriging_tail_usage(rel_run_dir: str, mask: np.ndarray) -> dict:
    """How much of kriging's output relies on the linear tail extrapolation
    of the normal-score back-transform, at the evaluated (non-conditioning)
    cells -- including at the EXTREME quantiles the CRPS integral touches."""
    run_dir = _REPO_ROOT / rel_run_dir
    kmap_ns = np.load(run_dir / "kmap_ns.npy")[mask]
    vmap_ns = np.load(run_dir / "kriging_var_map_ns.npy")[mask]
    tt = pd.read_csv(run_dir / "nscore_transform_table.csv")
    vr, vrg = tt["vr"].values, tt["vrg"].values
    std_ns = np.sqrt(np.clip(vmap_ns, 0.0, None))

    n_cells = kmap_ns.size
    args = (vr, vrg, BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR)

    out = {
        "kriging_n_eval_cells": n_cells,
        "kriging_nscore_table_n_points": int(len(vr)),
        "kriging_nscore_table_vrg_min": float(vrg[0]),
        "kriging_nscore_table_vrg_max": float(vrg[-1]),
        "kriging_nscore_table_vr_min": float(vr[0]),
        "kriging_nscore_table_vr_max": float(vr[-1]),
        "kriging_n_cells_point_estimate_in_tail": int(
            np.sum((kmap_ns <= vrg[0]) | (kmap_ns >= vrg[-1]))
        ),
        "kriging_max_var_ns": float(vmap_ns.max()),
        "kriging_mean_var_ns": float(vmap_ns.mean()),
    }

    # --- 95% interval endpoints ------------------------------------------
    z = norm.ppf(0.5 + 0.5 * QC_INTERVAL_P)
    ns_lo, ns_hi = kmap_ns - z * std_ns, kmap_ns + z * std_ns
    endpoints_in_tail = int(np.sum((ns_lo <= vrg[0]) | (ns_hi >= vrg[-1])))
    phys_lo = backtr_value_vectorized(ns_lo, *args)
    phys_hi = backtr_value_vectorized(ns_hi, *args)
    out.update(
        {
            "kriging_n_cells_p95_endpoint_in_tail": endpoints_in_tail,
            "kriging_frac_cells_p95_endpoint_in_tail": endpoints_in_tail / n_cells,
            "kriging_n_cells_p95_saturated_at_backtr_bound": int(
                np.sum(
                    (phys_lo <= BACKTR_ZMIN + SATURATION_TOL)
                    | (phys_hi >= BACKTR_ZMAX - SATURATION_TOL)
                )
            ),
            "kriging_p95_phys_lo_min": float(phys_lo.min()),
            "kriging_p95_phys_hi_max": float(phys_hi.max()),
        }
    )

    # --- CRPS-integral EXTREME quantiles (tau = 0.5/n_tau, 1 - 0.5/n_tau) --
    taus = crps_tau_grid(CRPS_DEFAULT_N_TAU)
    tau_lo, tau_hi = float(taus[0]), float(taus[-1])
    z_lo, z_hi = norm.ppf(tau_lo), norm.ppf(tau_hi)
    ns_qlo, ns_qhi = kmap_ns + z_lo * std_ns, kmap_ns + z_hi * std_ns
    crps_endpoints_in_tail = int(np.sum((ns_qlo <= vrg[0]) | (ns_qhi >= vrg[-1])))
    q_lo = backtr_value_vectorized(ns_qlo, *args)
    q_hi = backtr_value_vectorized(ns_qhi, *args)
    out.update(
        {
            "crps_n_tau": CRPS_DEFAULT_N_TAU,
            "crps_tau_min": tau_lo,
            "crps_tau_max": tau_hi,
            "kriging_n_cells_crps_extreme_quantile_in_tail": crps_endpoints_in_tail,
            "kriging_frac_cells_crps_extreme_quantile_in_tail": crps_endpoints_in_tail / n_cells,
            "kriging_n_cells_crps_extreme_saturated_at_backtr_bound": int(
                np.sum(
                    (q_lo <= BACKTR_ZMIN + SATURATION_TOL)
                    | (q_hi >= BACKTR_ZMAX - SATURATION_TOL)
                )
            ),
            "kriging_crps_extreme_phys_lo_min": float(q_lo.min()),
            "kriging_crps_extreme_phys_hi_max": float(q_hi.max()),
        }
    )
    return out


def count_sgsim_warnings(samples_df: pd.DataFrame) -> int:
    """Re-execute the identical (deterministic) geostats.sgsim call
    src/experiments/sgs.py makes for this conditioning dataset, with stdout
    captured, purely to COUNT near-singular-solve warnings. Saves nothing."""
    n = len(samples_df)
    jitter_rng = np.random.RandomState(sgs_module.SGS_JITTER_SEED)
    sgsim_input_df = samples_df.copy()
    sgsim_input_df["X"] = sgsim_input_df["X"] + jitter_rng.uniform(
        -sgs_module.SGS_JITTER_MAGNITUDE, sgs_module.SGS_JITTER_MAGNITUDE, size=n
    )
    sgsim_input_df["Y"] = sgsim_input_df["Y"] + jitter_rng.uniform(
        -sgs_module.SGS_JITTER_MAGNITUDE, sgs_module.SGS_JITTER_MAGNITUDE, size=n
    )
    vario = build_vario(hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)

    buf = io.StringIO()
    with redirect_stdout(buf):
        geostats.sgsim(
            sgsim_input_df, "X", "Y", VCOL,
            wcol=-1, scol=-1, tmin=-9999, tmax=9999, itrans=1, ismooth=0,
            dftrans=0, tcol=0, twtcol=0,
            zmin=sgs_module.SGSIM_ZMIN, zmax=sgs_module.SGSIM_ZMAX,
            ltail=sgs_module.SGSIM_LTAIL, ltpar=sgs_module.SGSIM_LTPAR,
            utail=sgs_module.SGSIM_UTAIL, utpar=sgs_module.SGSIM_UTPAR,
            nsim=sgs_module.N_REALIZATIONS,
            nx=NX, xmn=XMN, xsiz=XSIZ, ny=NY, ymn=YMN, ysiz=YSIZ,
            seed=sgs_module.SGS_SEED,
            ndmin=sgs_module.NDMIN, ndmax=sgs_module.NDMAX, nodmax=sgs_module.NODMAX,
            mults=sgs_module.MULTS, nmult=sgs_module.NMULT, noct=sgs_module.NOCT,
            ktype=sgs_module.KTYPE, colocorr=sgs_module.COLOCORR,
            sec_map=sgs_module.SEC_MAP, vario=vario,
        )
    return sum(1 for line in buf.getvalue().splitlines() if SGSIM_WARNING_TOKEN in line)


def qc_one_level(axis_level: str, run_dirs: dict) -> dict:
    n_requested = SAMPLE_COUNTS[axis_level]
    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)
    samples_df = get_conditioning_samples(
        truth, sample_seed=SAMPLE_SEED, n_samples=n_requested
    )
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)

    row = {
        "axis_level": axis_level,
        "sample_fraction_pct": float(axis_level),
        "n_samples_requested": n_requested,
        "n_samples_actual": len(samples_df),
        "n_cells_evaluated": int(mask.sum()),
    }
    row.update(ndmax_capping(samples_df))
    row.update(kriging_tail_usage(run_dirs["kriging"], mask))

    # --- SGS ---------------------------------------------------------------
    sgs_params = _manifest(run_dirs["sgs"])["params"]
    row["sgs_honor_max_abs_error"] = sgs_params["qc_honor"]["max_abs_error"]
    row["sgs_honor_mean_abs_error"] = sgs_params["qc_honor"]["mean_abs_error"]
    row["sgs_honor_raw_max_abs_error"] = sgs_params["qc_honor_raw_sgsim_output"]["max_abs_error"]
    row["sgs_honor_raw_mean_abs_error"] = sgs_params["qc_honor_raw_sgsim_output"][
        "mean_abs_error"
    ]
    means = sgs_params["qc_histogram"]["realization_means"]
    stdevs = sgs_params["qc_histogram"]["realization_stdevs"]
    row["sgs_target_mean"] = sgs_params["qc_histogram"]["target_mean"]
    row["sgs_target_stdev"] = sgs_params["qc_histogram"]["target_stdev"]
    row["sgs_realization_mean_min"] = float(np.min(means))
    row["sgs_realization_mean_max"] = float(np.max(means))
    row["sgs_realization_stdev_min"] = float(np.min(stdevs))
    row["sgs_realization_stdev_max"] = float(np.max(stdevs))
    row["sgs_max_abs_mean_dev_from_target"] = float(
        np.max(np.abs(np.array(means) - sgs_params["qc_histogram"]["target_mean"]))
    )
    row["sgs_max_abs_stdev_dev_from_target"] = float(
        np.max(np.abs(np.array(stdevs) - sgs_params["qc_histogram"]["target_stdev"]))
    )
    row["sgsim_near_singular_warnings"] = (
        count_sgsim_warnings(samples_df) if RECHECK_SGSIM_WARNINGS else np.nan
    )

    # --- RBF+bootstrap -----------------------------------------------------
    rbf_params = _manifest(run_dirs["rbf_bootstrap"])["params"]
    row["rbf_cv_folds"] = rbf_params.get("cv_folds", CV_FOLDS)
    n_act = len(samples_df)
    folds = row["rbf_cv_folds"]
    row["rbf_cv_test_fold_size_min"] = n_act // folds
    row["rbf_cv_test_fold_size_max"] = n_act // folds + (1 if n_act % folds else 0)
    row["rbf_cv_train_size"] = n_act - row["rbf_cv_test_fold_size_max"]
    row["rbf_best_epsilon"] = rbf_params["best_epsilon"]
    row["rbf_best_smoothing"] = rbf_params["best_smoothing"]
    row["rbf_smoothing_is_zero"] = rbf_params["best_smoothing"] == 0.0
    row["rbf_n_bootstrap"] = rbf_params["n_bootstrap"]
    row["rbf_n_bootstrap_deduplicated"] = rbf_params.get("n_bootstrap_deduplicated", 0)

    # --- GP-MLE ------------------------------------------------------------
    gp_params = _manifest(run_dirs["gp_mle"])["params"]
    hp = gp_params["fitted_hyperparameters"]
    row["gp_length_scale_m"] = hp["length_scale_m"]
    row["gp_signal_variance_real"] = hp["signal_variance_real_units"]
    row["gp_noise_variance_real"] = hp["noise_variance_real_units"]
    row["gp_length_scale_at_bound"] = _at_bound(hp["length_scale_m"], LENGTH_SCALE_BOUNDS)
    row["gp_signal_var_at_bound"] = _at_bound(
        hp["signal_variance_normalized"], CONSTANT_VALUE_BOUNDS
    )
    row["gp_noise_var_at_bound"] = _at_bound(hp["noise_variance_normalized"], NOISE_LEVEL_BOUNDS)
    row["gp_log_marginal_likelihood"] = gp_params["log_marginal_likelihood"]
    row["gp_posterior_sample_method"] = gp_params.get("posterior_sample_method", "n/a")

    return row


def main():
    source_runs = json.loads((PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8"))

    rows = []
    for axis_level in ALL_AXIS_LEVELS:
        print(f"QC sample fraction = {axis_level}% (n={SAMPLE_COUNTS[axis_level]}) ...")
        rows.append(qc_one_level(axis_level, source_runs[axis_level]))

    qc_df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "qc_summary.csv"
    qc_df.to_csv(out, index=False)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 120)

    print("\n--- 1. Kriging normal-score transform table + tail usage ---")
    print(
        qc_df[
            [
                "axis_level", "n_samples_actual", "n_cells_evaluated",
                "kriging_nscore_table_n_points",
                "kriging_nscore_table_vrg_min", "kriging_nscore_table_vrg_max",
                "kriging_nscore_table_vr_min", "kriging_nscore_table_vr_max",
                "kriging_n_cells_point_estimate_in_tail",
                "kriging_frac_cells_p95_endpoint_in_tail",
                "kriging_frac_cells_crps_extreme_quantile_in_tail",
                "kriging_n_cells_p95_saturated_at_backtr_bound",
                "kriging_n_cells_crps_extreme_saturated_at_backtr_bound",
                "kriging_mean_var_ns", "kriging_max_var_ns",
            ]
        ].to_string(index=False)
    )
    print("\n--- 2. RBF+bootstrap CV / tuning ---")
    print(
        qc_df[
            [
                "axis_level", "n_samples_actual", "rbf_cv_folds",
                "rbf_cv_train_size", "rbf_cv_test_fold_size_min", "rbf_cv_test_fold_size_max",
                "rbf_best_epsilon", "rbf_best_smoothing", "rbf_smoothing_is_zero",
                "rbf_n_bootstrap", "rbf_n_bootstrap_deduplicated",
            ]
        ].to_string(index=False)
    )
    print("\n--- 3. SGS ---")
    print(
        qc_df[
            [
                "axis_level", "sgs_honor_max_abs_error", "sgs_honor_mean_abs_error",
                "sgs_honor_raw_max_abs_error", "sgs_honor_raw_mean_abs_error",
                "sgsim_near_singular_warnings",
                "sgs_realization_mean_min", "sgs_realization_mean_max",
                "sgs_realization_stdev_min", "sgs_realization_stdev_max",
                "sgs_max_abs_mean_dev_from_target", "sgs_max_abs_stdev_dev_from_target",
            ]
        ].to_string(index=False)
    )
    print("\n--- 4. GP-MLE fitted hyperparameters ---")
    print(
        qc_df[
            [
                "axis_level", "gp_length_scale_m", "gp_signal_variance_real",
                "gp_noise_variance_real", "gp_length_scale_at_bound",
                "gp_signal_var_at_bound", "gp_noise_var_at_bound",
                "gp_log_marginal_likelihood",
            ]
        ].to_string(index=False)
    )
    print("\n--- 5. ndmax binding ---")
    print(
        qc_df[
            [
                "axis_level", "n_samples_actual", "kriging_ndmax", "sgs_ndmax",
                "kriging_ndmax_can_bind_at_all", "sgs_ndmax_can_bind_at_all",
                "n_samples_within_range_mean", "n_samples_within_range_max",
                "frac_cells_over_kriging_ndmax", "frac_cells_over_sgs_ndmax",
            ]
        ].to_string(index=False)
    )
    print(f"\nqc_summary.csv: {out}")
    return qc_df


if __name__ == "__main__":
    main()
