"""QC checklist for the nugget-axis experiment (6 levels, normalized nugget
0.0 .. 0.5; docs/experiment_context.md deliverable 3).

Collects, per axis level, the specific numbers the orchestrator/reviewer must
see before the nugget-axis results are treated as valid. Nothing here changes
any result -- it only reads the pinned runs in
results/processed/nugget_axis/source_runs.json plus their saved arrays, and
computes a few purely geometric quantities from the grid/sample layout.

Checks
------
1. Per-level ground-truth nugget bookkeeping: normalized nugget, the derived
   cc1 actually recorded in each kriging/SGS run's INPUT variogram, and the
   physical-unit nugget (nug * POR_STDEV**2) the GP-MLE fit is later compared
   against.
2. GP-MLE fitted hyperparameters -- the CENTRAL diagnostic for this axis
   (Claim 2): fitted WhiteKernel noise variance in real units vs. the truth
   nugget, the fitted length scale (raw and as a 0.05-cutoff practical
   range), and whether any hyperparameter sits at the edge of its optimizer
   bound (which would mean the bound, not the data, chose the value -- most
   likely at nug=0.0, where the true noise is exactly zero and the
   WhiteKernel lower bound is 1e-5 in normalized units).
3. RBF+bootstrap tuned hyperparameters: best_epsilon, best_smoothing
   (RBF's only nugget-absorbing knob), whether CV selected
   best_smoothing == 0 at this level (the regime where the
   duplicate-support-point degeneracy branch in rbf_bootstrap.py can fire),
   and how many bootstrap replicates were deduplicated as a result.
4. SGS honoring of conditioning data: final (post re-enforcement) and raw
   (as geostats.sgsim returned it) max/mean absolute error.

   MEASURED FACT worth reading before interpreting the SGS curve
   (2026-09-22, recorded here at the reviewer's request): the RAW sgsim
   output's honoring error GROWS MONOTONICALLY WITH THE NUGGET --
   sgs_honor_raw_max_abs_error = 1.802 / 4.904 / 6.736 / 8.092 / 9.090 /
   10.142 Porosity % at nug = 0.0 / 0.1 / 0.2 / 0.3 / 0.4 / 0.5. That is the
   expected behaviour of a nugget-bearing SGS (a pure nugget component makes
   a simulated node at a datum location uncorrelated with that datum), not a
   defect. src/experiments/sgs.py re-enforces the conditioning values after
   simulation, so sgs_honor_max_abs_error (the FINAL, saved realizations) is
   exactly 0.0 at every level. Both numbers are tabulated; do not quote the
   final 0.0 without also knowing the raw figure it was produced from.
5. Kriging back-transform tail usage and the kriging variance range, INCLUDING
   the count/most-negative value of floating-point-negative kb2d variances
   (expected at conditioning cells, where the true kriging variance is
   exactly 0 -- most prominently at nug=0.0).

   HISTORY (read this before quoting older numbers): on the FIRST 2026-09-22
   nugget-axis run, nug=0.3 showed 119 evaluated cells with their kriging
   POINT ESTIMATE in the normal-score table's tail, 60 of them clipped
   exactly at BACKTR_ZMIN = 3.0 Porosity %, and 13 such tail cells at
   nug=0.2. That was NOT the +/-4-stdev BACKTR_ZMIN/ZMAX bound biting -- it
   was a consequence of the ``geostats.nscore`` off-by-one (see
   src/nscore.py), which had given the minimum conditioning datum a normal
   score of -52.15 instead of -2.6411 and thereby dragged a whole
   neighbourhood of kriged estimates far below the transform table. After the
   repair and re-run, kriging_n_cells_point_estimate_in_tail = 0 and
   kriging_n_cells_p95_saturated_at_backtr_bound = 0 at EVERY level, and
   kriging MSE is monotone in the nugget (2.926 / 3.702 / 4.531 / 5.408 /
   6.234 / 7.141). The +/-4-stdev bound documented at length in
   src/experiments/kriging.py (history items (a)/(b)/(c) there, a settled
   project decision) is therefore not currently binding anywhere on this
   axis; it is still not to be re-litigated here.
6. ndmax capping: the fraction of grid cells with more conditioning samples
   inside one variogram range than each method's fixed search cap admits
   (kriging ndmax=50, sgsim ndmax=20). On THIS axis the range is held fixed
   at the base-case 300 m and the sample locations are identical across
   levels, so these numbers are constant by construction -- they are reported
   anyway, and their constancy is itself asserted, because a level that
   deviated would mean the axis is not one-factor-at-a-time.

Run with: .venv/Scripts/python.exe -m src.experiments.qc_nugget_axis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.evaluation import conditioning_cell_mask
from src.experiments.base_case import NX, NY, XMN, YMN, XSIZ, YSIZ
from src.experiments.base_case_conditioning import (
    N_SAMPLES,
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
from src.experiments.nugget_axis import (
    ALL_NUGGET_VALUES,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    METHODS,
    level_key,
    truth_nugget_real_units,
)
from src.experiments.sgs import NDMAX as SGS_NDMAX
from src.grid_utils import full_grid_coordinates

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"

# A fitted hyperparameter is called "at bound" if it is within this factor of
# either end of its (log-scaled) optimizer bound. Same convention as
# qc_range_axis.py.
BOUND_PROXIMITY_FACTOR = 1.01

# Absolute tolerance (Porosity %) for calling a back-transformed value
# "saturated at" BACKTR_ZMIN/BACKTR_ZMAX.
SATURATION_TOL = 1e-6

QC_INTERVAL_P = 0.95

# Same 0.05-correlation-cutoff convention used throughout this project for
# converting a sklearn RBF length_scale to a practical range.
CORRELATION_CUTOFF = 0.05
GP_PRACTICAL_RANGE_FACTOR = float(np.sqrt(-2.0 * np.log(CORRELATION_CUTOFF)))  # 2.4477


def _manifest(rel_run_dir: str) -> dict:
    return json.loads((_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8"))


def _at_bound(value: float, bounds) -> bool:
    lo, hi = bounds
    return value <= lo * BOUND_PROXIMITY_FACTOR or value >= hi / BOUND_PROXIMITY_FACTOR


def ndmax_capping(samples_df: pd.DataFrame) -> dict:
    """Fraction of grid cells with more than ndmax conditioning samples inside
    the variogram range radius (purely geometric). The radius is the axis's
    fixed range AXIS_HMAJ1, identical at every nugget level."""
    grid = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)  # (n_cells, 2)
    pts = samples_df[["X", "Y"]].values
    d = np.sqrt(
        (grid[:, 0][:, None] - pts[:, 0][None, :]) ** 2
        + (grid[:, 1][:, None] - pts[:, 1][None, :]) ** 2
    )
    n_within = (d <= AXIS_HMAJ1).sum(axis=1)
    return {
        "n_samples_within_range_mean": float(n_within.mean()),
        "n_samples_within_range_max": int(n_within.max()),
        "frac_cells_over_kriging_ndmax": float(np.mean(n_within > KRIGING_NDMAX)),
        "frac_cells_over_sgs_ndmax": float(np.mean(n_within > SGS_NDMAX)),
    }


def kriging_tail_usage(rel_run_dir: str, mask: np.ndarray) -> dict:
    """How much of kriging's output relies on the linear tail extrapolation of
    the normal-score back-transform, at the evaluated (non-conditioning)
    cells, plus the negative-variance bookkeeping that matters at nug=0."""
    run_dir = _REPO_ROOT / rel_run_dir
    vmap_ns_full = np.load(run_dir / "kriging_var_map_ns.npy")
    kmap_ns = np.load(run_dir / "kmap_ns.npy")[mask]
    vmap_ns = vmap_ns_full[mask]
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
        # Negative-variance bookkeeping (see module docstring check 5). The
        # "all cells" figures include the conditioning cells, which the
        # evaluation excludes -- at nug=0 that is exactly where kb2d's
        # exactly-zero variance shows up as a tiny negative.
        "kriging_min_var_ns_all_cells": float(vmap_ns_full.min()),
        "kriging_n_negative_var_ns_all_cells": int(np.sum(vmap_ns_full < 0)),
        "kriging_min_var_ns_eval_cells": float(vmap_ns.min()),
        "kriging_n_negative_var_ns_eval_cells": int(np.sum(vmap_ns < 0)),
    }


def qc_one_level(nug: float, run_dirs: dict) -> dict:
    axis_level = level_key(nug)
    truth = get_base_case_truth(
        truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1, nug=float(nug)
    )
    samples_df = get_conditioning_samples(
        truth, sample_seed=SAMPLE_SEED, n_samples=N_SAMPLES
    )
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)

    row = {
        "axis_level": axis_level,
        "nug_normalized": float(nug),
        "cc1_normalized": 1.0 - float(nug),
        "truth_nugget_real_units": truth_nugget_real_units(nug),
        "n_samples_requested": N_SAMPLES,
        "n_samples_actual": len(samples_df),
        "truth_mean": float(np.mean(truth)),
        "truth_stdev": float(np.std(truth)),
    }

    # --- The nugget each method was actually GIVEN ------------------------
    k_params = _manifest(run_dirs["kriging"])["params"]
    s_params = _manifest(run_dirs["sgs"])["params"]
    row["kriging_input_nug"] = float(k_params["variogram"]["nug"])
    row["kriging_input_cc1"] = float(k_params["variogram"]["cc1"])
    row["kriging_input_range_m"] = float(k_params["variogram"]["hmaj1"])
    row["sgs_input_nug"] = float(s_params["variogram"]["nug"])
    row["sgs_input_cc1"] = float(s_params["variogram"]["cc1"])
    row["sgs_input_range_m"] = float(s_params["variogram"]["hmaj1"])
    row["n_samples_used_by_kb2d"] = int(k_params["n_samples_used_by_kb2d"])

    row.update(ndmax_capping(samples_df))
    row.update(kriging_tail_usage(run_dirs["kriging"], mask))

    # --- SGS honoring ---------------------------------------------------
    row["sgs_honor_max_abs_error"] = s_params["qc_honor"]["max_abs_error"]
    row["sgs_honor_mean_abs_error"] = s_params["qc_honor"]["mean_abs_error"]
    row["sgs_honor_raw_max_abs_error"] = s_params["qc_honor_raw_sgsim_output"]["max_abs_error"]
    row["sgs_realization_mean_min"] = float(np.min(s_params["qc_histogram"]["realization_means"]))
    row["sgs_realization_mean_max"] = float(np.max(s_params["qc_histogram"]["realization_means"]))
    row["sgs_realization_stdev_min"] = float(
        np.min(s_params["qc_histogram"]["realization_stdevs"])
    )
    row["sgs_realization_stdev_max"] = float(
        np.max(s_params["qc_histogram"]["realization_stdevs"])
    )

    # --- RBF+bootstrap tuning -------------------------------------------
    rbf_params = _manifest(run_dirs["rbf_bootstrap"])["params"]
    row["rbf_best_epsilon"] = rbf_params["best_epsilon"]
    row["rbf_best_smoothing"] = rbf_params["best_smoothing"]
    row["rbf_smoothing_is_zero"] = rbf_params["best_smoothing"] == 0.0
    row["rbf_n_bootstrap_deduplicated"] = rbf_params.get("n_bootstrap_deduplicated", 0)
    # Whether CV pinned epsilon/smoothing at an endpoint of their grids (a
    # saturated grid means the tuned value is a grid artifact, not an optimum).
    eps_grid = np.asarray(rbf_params["epsilon_grid"], dtype=float)
    sm_grid = np.asarray(rbf_params["smoothing_grid"], dtype=float)
    row["rbf_epsilon_at_grid_endpoint"] = bool(
        np.isclose(rbf_params["best_epsilon"], eps_grid.min())
        or np.isclose(rbf_params["best_epsilon"], eps_grid.max())
    )
    row["rbf_smoothing_at_grid_endpoint"] = bool(
        np.isclose(rbf_params["best_smoothing"], sm_grid.min())
        or np.isclose(rbf_params["best_smoothing"], sm_grid.max())
    )

    # --- GP-MLE fitted hyperparameters (the central diagnostic) ----------
    gp_params = _manifest(run_dirs["gp_mle"])["params"]
    hp = gp_params["fitted_hyperparameters"]
    row["gp_length_scale_m"] = hp["length_scale_m"]
    row["gp_practical_range_m"] = GP_PRACTICAL_RANGE_FACTOR * float(hp["length_scale_m"])
    row["gp_signal_variance_real"] = hp["signal_variance_real_units"]
    row["gp_noise_variance_real"] = hp["noise_variance_real_units"]
    row["gp_noise_variance_normalized"] = hp["noise_variance_normalized"]
    truth_nug_real = truth_nugget_real_units(nug)
    # Ratio is undefined at nug=0 (truth nugget is exactly 0) -- NaN, not a
    # divide-by-zero inf, and never silently replaced by a placeholder number.
    row["gp_noise_over_truth_nugget"] = (
        float(hp["noise_variance_real_units"]) / truth_nug_real
        if truth_nug_real > 0
        else float("nan")
    )
    row["gp_noise_minus_truth_nugget"] = (
        float(hp["noise_variance_real_units"]) - truth_nug_real
    )
    row["gp_length_scale_at_bound"] = _at_bound(hp["length_scale_m"], LENGTH_SCALE_BOUNDS)
    row["gp_signal_var_at_bound"] = _at_bound(
        hp["signal_variance_normalized"], CONSTANT_VALUE_BOUNDS
    )
    row["gp_noise_var_at_bound"] = _at_bound(hp["noise_variance_normalized"], NOISE_LEVEL_BOUNDS)
    # Exact boundary stop at the WhiteKernel LOWER bound -- same definition
    # make_length_vs_smoothing_figures.at_optimizer_noise_floor uses, reading
    # the bound from this run's own manifest rather than a module constant.
    noise_floor = float(gp_params["kernel_init"]["noise_level_bounds"][0])
    row["gp_noise_level_lower_bound"] = noise_floor
    row["gp_noise_at_optimizer_floor"] = bool(
        abs(float(hp["noise_variance_normalized"]) - noise_floor) <= 1e-9 * noise_floor
    )
    row["gp_log_marginal_likelihood"] = gp_params["log_marginal_likelihood"]
    row["gp_posterior_sample_method"] = gp_params.get("posterior_sample_method", "n/a")

    # --- Cross-method sample-count consistency ---------------------------
    n_actual = {m: int(_manifest(run_dirs[m])["params"]["n_samples_actual"]) for m in METHODS}
    if len(set(n_actual.values())) != 1 or set(n_actual.values()) != {len(samples_df)}:
        raise ValueError(
            f"nug={axis_level}: n_samples_actual differs across methods or from the "
            f"regenerated samples: {n_actual} vs. {len(samples_df)}."
        )

    return row


def main():
    source_runs = json.loads((PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8"))

    rows = []
    for nug in ALL_NUGGET_VALUES:
        axis_level = level_key(nug)
        print(f"QC nug={axis_level} ...")
        rows.append(qc_one_level(float(nug), source_runs[axis_level]))

    qc_df = pd.DataFrame(rows)

    # --- Assert the one-factor-at-a-time invariants ------------------------
    # The range, sample count and sample geometry must be identical at every
    # level; only the nugget varies. Failing these means the axis is not what
    # it claims to be.
    for col in (
        "kriging_input_range_m",
        "sgs_input_range_m",
        "n_samples_actual",
        "n_samples_within_range_mean",
        "n_samples_within_range_max",
        "frac_cells_over_kriging_ndmax",
        "frac_cells_over_sgs_ndmax",
        "kriging_n_eval_cells",
    ):
        if qc_df[col].nunique() != 1:
            raise ValueError(
                f"{col} varies across nugget levels ({qc_df[col].tolist()}) -- the nugget "
                "axis is supposed to hold everything except the nugget fixed."
            )
    if not np.allclose(qc_df["kriging_input_nug"], qc_df["nug_normalized"]) or not np.allclose(
        qc_df["sgs_input_nug"], qc_df["nug_normalized"]
    ):
        raise ValueError(
            "kriging/SGS input nugget does not equal the level's ground-truth nugget -- "
            "these baselines must be handed the TRUE nugget on this axis."
        )
    print(
        "\ninvariants OK: range, sample count/geometry and evaluated-cell count are "
        "constant across levels; kriging/SGS input nugget == ground-truth nugget."
    )

    out = PROCESSED_DIR / "qc_summary.csv"
    qc_df.to_csv(out, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)

    print("\n--- Ground-truth nugget bookkeeping ---")
    print(
        qc_df[
            [
                "axis_level", "nug_normalized", "cc1_normalized",
                "truth_nugget_real_units", "kriging_input_nug", "sgs_input_nug",
                "n_samples_actual", "n_samples_used_by_kb2d",
                "truth_mean", "truth_stdev",
            ]
        ].to_string(index=False)
    )
    print("\n--- GP-MLE fitted hyperparameters (Claim 2 central diagnostic) ---")
    print(
        qc_df[
            [
                "axis_level", "truth_nugget_real_units", "gp_noise_variance_real",
                "gp_noise_over_truth_nugget", "gp_noise_minus_truth_nugget",
                "gp_signal_variance_real", "gp_length_scale_m", "gp_practical_range_m",
                "gp_noise_at_optimizer_floor", "gp_noise_var_at_bound",
                "gp_length_scale_at_bound", "gp_signal_var_at_bound",
                "gp_log_marginal_likelihood",
            ]
        ].to_string(index=False)
    )
    print("\n--- RBF+bootstrap tuning ---")
    print(
        qc_df[
            [
                "axis_level", "rbf_best_epsilon", "rbf_best_smoothing",
                "rbf_smoothing_is_zero", "rbf_n_bootstrap_deduplicated",
                "rbf_epsilon_at_grid_endpoint", "rbf_smoothing_at_grid_endpoint",
            ]
        ].to_string(index=False)
    )
    print("\n--- SGS honoring ---")
    print(
        qc_df[
            [
                "axis_level", "sgs_honor_max_abs_error", "sgs_honor_mean_abs_error",
                "sgs_honor_raw_max_abs_error", "sgs_realization_mean_min",
                "sgs_realization_mean_max", "sgs_realization_stdev_min",
                "sgs_realization_stdev_max",
            ]
        ].to_string(index=False)
    )
    print("\n--- Kriging variance / back-transform tail usage ---")
    print(
        qc_df[
            [
                "axis_level", "kriging_n_eval_cells", "kriging_mean_var_ns",
                "kriging_max_var_ns", "kriging_min_var_ns_all_cells",
                "kriging_n_negative_var_ns_all_cells", "kriging_min_var_ns_eval_cells",
                "kriging_n_negative_var_ns_eval_cells",
                "kriging_n_cells_point_estimate_in_tail",
                "kriging_n_cells_p95_endpoint_in_tail",
                "kriging_n_cells_p95_saturated_at_backtr_bound",
            ]
        ].to_string(index=False)
    )
    print("\n--- ndmax capping (constant by construction on this axis) ---")
    print(
        qc_df[
            [
                "axis_level", "n_samples_within_range_mean", "n_samples_within_range_max",
                "frac_cells_over_kriging_ndmax", "frac_cells_over_sgs_ndmax",
            ]
        ].to_string(index=False)
    )
    print(f"\nqc_summary.csv: {out}")
    return qc_df


if __name__ == "__main__":
    main()
