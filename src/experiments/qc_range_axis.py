"""QC checklist for the range-axis experiment (8 levels, 100..800 m).

Collects, per axis level, the specific numbers the orchestrator/reviewer must
see before the range-axis results are treated as valid. Nothing here changes
any result -- it only reads the pinned runs in
results/processed/range_axis/source_runs.json plus their saved arrays, and
computes a few purely geometric quantities from the grid/sample layout.

Checks
------
1. SGS honoring of conditioning data: final (post re-enforcement) and raw
   (as geostats.sgsim returned it) max/mean absolute error.
2. Kriging back-transform tail usage: how many evaluated cells have their
   point estimate, or their 95%-interval endpoints, outside the normal-score
   transform table (i.e. produced by LINEAR tail extrapolation toward
   BACKTR_ZMIN/ZMAX), and how many are numerically saturated AT those bounds.
3. GP-MLE fitted hyperparameters and whether any of them sits at the edge of
   its optimizer bound (which would mean the bound, not the data, chose the
   value).
4. RBF+bootstrap tuned hyperparameters: whether CV selected
   best_smoothing == 0 at this level (the regime where the duplicate-support-
   point degeneracy branch in rbf_bootstrap.py can fire) and how many
   bootstrap replicates were deduplicated as a result.
5. ndmax capping vs. range: the fraction of grid cells that have MORE
   conditioning samples within one variogram range than each method's fixed
   search cap admits (kriging ndmax=50, sgsim ndmax=20). As the range grows
   toward the domain size, a fixed ndmax starts discarding still-correlated
   data; this quantifies how binding that cap is at each axis level. It is a
   purely geometric count of samples within the range radius -- it is NOT a
   claim about how much the estimate actually changes.

Run with: .venv/Scripts/python.exe -m src.experiments.qc_range_axis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.evaluation import conditioning_cell_mask
from src.experiments.base_case import NX, NY, XMN, YMN, XSIZ, YSIZ
from src.experiments.base_case_conditioning import (
    SAMPLE_SEED,
    TRUTH_SEED,
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
from src.experiments.range_axis import ALL_RANGE_VALUES, METHODS
from src.experiments.sgs import NDMAX as SGS_NDMAX
from src.grid_utils import full_grid_coordinates

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"

# A fitted hyperparameter is called "at bound" if it is within this factor of
# either end of its (log-scaled) optimizer bound.
BOUND_PROXIMITY_FACTOR = 1.01

# Absolute tolerance (porosity %) for calling a back-transformed value
# "saturated at" BACKTR_ZMIN/BACKTR_ZMAX.
SATURATION_TOL = 1e-6

QC_INTERVAL_P = 0.95


def _manifest(rel_run_dir: str) -> dict:
    return json.loads((_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8"))


def _at_bound(value: float, bounds) -> bool:
    lo, hi = bounds
    return value <= lo * BOUND_PROXIMITY_FACTOR or value >= hi / BOUND_PROXIMITY_FACTOR


def ndmax_capping(hmaj1: float, samples_df: pd.DataFrame) -> dict:
    """Fraction of grid cells with more than ndmax conditioning samples inside
    the variogram range radius (purely geometric)."""
    grid = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)  # (n_cells, 2)
    pts = samples_df[["X", "Y"]].values
    d = np.sqrt(
        (grid[:, 0][:, None] - pts[:, 0][None, :]) ** 2
        + (grid[:, 1][:, None] - pts[:, 1][None, :]) ** 2
    )
    n_within = (d <= hmaj1).sum(axis=1)
    return {
        "n_samples_within_range_mean": float(n_within.mean()),
        "n_samples_within_range_max": int(n_within.max()),
        "frac_cells_over_kriging_ndmax": float(np.mean(n_within > KRIGING_NDMAX)),
        "frac_cells_over_sgs_ndmax": float(np.mean(n_within > SGS_NDMAX)),
    }


def kriging_tail_usage(rel_run_dir: str, mask: np.ndarray) -> dict:
    """How much of kriging's output relies on the linear tail extrapolation of
    the normal-score back-transform, at the evaluated (non-conditioning) cells."""
    run_dir = _REPO_ROOT / rel_run_dir
    kmap_ns = np.load(run_dir / "kmap_ns.npy")[mask]
    vmap_ns = np.load(run_dir / "kriging_var_map_ns.npy")[mask]
    tt = pd.read_csv(run_dir / "nscore_transform_table.csv")
    vr, vrg = tt["vr"].values, tt["vrg"].values
    std_ns = np.sqrt(np.clip(vmap_ns, 0.0, None))

    n_cells = kmap_ns.size
    point_in_tail = int(np.sum((kmap_ns <= vrg[0]) | (kmap_ns >= vrg[-1])))

    z = norm.ppf(0.5 + 0.5 * QC_INTERVAL_P)
    ns_lo, ns_hi = kmap_ns - z * std_ns, kmap_ns + z * std_ns
    endpoints_in_tail = int(np.sum((ns_lo <= vrg[0]) | (ns_hi >= vrg[-1])))

    args = (vr, vrg, BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR)
    phys_lo = backtr_value_vectorized(ns_lo, *args)
    phys_hi = backtr_value_vectorized(ns_hi, *args)
    saturated = int(
        np.sum(
            (phys_lo <= BACKTR_ZMIN + SATURATION_TOL) | (phys_hi >= BACKTR_ZMAX - SATURATION_TOL)
        )
    )

    return {
        "kriging_n_eval_cells": n_cells,
        "kriging_nscore_table_min": float(vrg[0]),
        "kriging_nscore_table_max": float(vrg[-1]),
        "kriging_n_cells_point_estimate_in_tail": point_in_tail,
        "kriging_n_cells_p95_endpoint_in_tail": endpoints_in_tail,
        "kriging_frac_cells_p95_endpoint_in_tail": endpoints_in_tail / n_cells,
        "kriging_n_cells_p95_saturated_at_backtr_bound": saturated,
        "kriging_p95_phys_lo_min": float(phys_lo.min()),
        "kriging_p95_phys_hi_max": float(phys_hi.max()),
        "kriging_max_var_ns": float(vmap_ns.max()),
        "kriging_mean_var_ns": float(vmap_ns.mean()),
    }


def qc_one_level(hmaj1: float, run_dirs: dict) -> dict:
    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=hmaj1, hmin1=hmaj1)
    samples_df = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED)
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)

    row = {"axis_level": str(int(hmaj1)), "range_m": hmaj1, "n_samples": len(samples_df)}
    row.update(ndmax_capping(hmaj1, samples_df))
    row.update(kriging_tail_usage(run_dirs["kriging"], mask))

    # --- SGS honoring ---------------------------------------------------
    sgs_params = _manifest(run_dirs["sgs"])["params"]
    row["sgs_honor_max_abs_error"] = sgs_params["qc_honor"]["max_abs_error"]
    row["sgs_honor_mean_abs_error"] = sgs_params["qc_honor"]["mean_abs_error"]
    row["sgs_honor_raw_max_abs_error"] = sgs_params["qc_honor_raw_sgsim_output"]["max_abs_error"]
    row["sgs_realization_mean_min"] = float(np.min(sgs_params["qc_histogram"]["realization_means"]))
    row["sgs_realization_mean_max"] = float(np.max(sgs_params["qc_histogram"]["realization_means"]))
    row["sgs_realization_stdev_min"] = float(
        np.min(sgs_params["qc_histogram"]["realization_stdevs"])
    )
    row["sgs_realization_stdev_max"] = float(
        np.max(sgs_params["qc_histogram"]["realization_stdevs"])
    )

    # --- RBF+bootstrap tuning -------------------------------------------
    rbf_params = _manifest(run_dirs["rbf_bootstrap"])["params"]
    row["rbf_best_epsilon"] = rbf_params["best_epsilon"]
    row["rbf_best_smoothing"] = rbf_params["best_smoothing"]
    row["rbf_smoothing_is_zero"] = rbf_params["best_smoothing"] == 0.0
    # n_bootstrap_deduplicated is absent from runs made before that field was
    # added (the base-case rbf run); best_smoothing != 0 there, so the branch
    # could not have fired.
    row["rbf_n_bootstrap_deduplicated"] = rbf_params.get("n_bootstrap_deduplicated", 0)

    # --- GP-MLE fitted hyperparameters ----------------------------------
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
    for hmaj1 in ALL_RANGE_VALUES:
        axis_level = str(int(hmaj1))
        print(f"QC range={hmaj1:g}m ...")
        rows.append(qc_one_level(hmaj1, source_runs[axis_level]))

    qc_df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "qc_summary.csv"
    qc_df.to_csv(out, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)

    print("\n--- ndmax capping (samples within one range radius vs. fixed search caps) ---")
    print(
        qc_df[
            [
                "range_m",
                "n_samples_within_range_mean",
                "n_samples_within_range_max",
                "frac_cells_over_kriging_ndmax",
                "frac_cells_over_sgs_ndmax",
            ]
        ].to_string(index=False)
    )
    print("\n--- SGS honoring ---")
    print(
        qc_df[
            [
                "range_m",
                "sgs_honor_max_abs_error",
                "sgs_honor_mean_abs_error",
                "sgs_honor_raw_max_abs_error",
                "sgs_realization_mean_min",
                "sgs_realization_mean_max",
                "sgs_realization_stdev_min",
                "sgs_realization_stdev_max",
            ]
        ].to_string(index=False)
    )
    print("\n--- Kriging back-transform tail usage ---")
    print(
        qc_df[
            [
                "range_m",
                "kriging_n_eval_cells",
                "kriging_n_cells_point_estimate_in_tail",
                "kriging_n_cells_p95_endpoint_in_tail",
                "kriging_frac_cells_p95_endpoint_in_tail",
                "kriging_n_cells_p95_saturated_at_backtr_bound",
                "kriging_mean_var_ns",
                "kriging_max_var_ns",
            ]
        ].to_string(index=False)
    )
    print("\n--- RBF+bootstrap tuning ---")
    print(
        qc_df[
            [
                "range_m",
                "rbf_best_epsilon",
                "rbf_best_smoothing",
                "rbf_smoothing_is_zero",
                "rbf_n_bootstrap_deduplicated",
            ]
        ].to_string(index=False)
    )
    print("\n--- GP-MLE fitted hyperparameters ---")
    print(
        qc_df[
            [
                "range_m",
                "gp_length_scale_m",
                "gp_signal_variance_real",
                "gp_noise_variance_real",
                "gp_length_scale_at_bound",
                "gp_signal_var_at_bound",
                "gp_noise_var_at_bound",
            ]
        ].to_string(index=False)
    )
    print(f"\nqc_summary.csv: {out}")
    return qc_df


if __name__ == "__main__":
    main()
