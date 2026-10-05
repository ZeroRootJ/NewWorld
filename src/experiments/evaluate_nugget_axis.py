"""Evaluate the nugget-axis 4-method comparison (kriging / SGS / RBF+
bootstrap / GP-MLE) over the 6 axis levels normalized nugget = 0.0, 0.1,
0.2, 0.3, 0.4, 0.5 (docs/experiment_context.md deliverable 3).

METRICS
-------
Accuracy and uncertainty quality are reported SEPARATELY and are never
combined into one number (CLAUDE.md: showing that they can diverge IS the
paper's argument).

  accuracy
    - ``mse``            (Porosity %^2, lower better)
  uncertainty quality
    - ``umg``            (coverage goodness from the accuracy plot,
                          1.0 = ideal)
  predictive-variance magnitude
    - ``variance_mean``  (Porosity %^2, each method's own predictive
                          variance averaged over the evaluated cells).
                          Definition and per-method source array are reused
                          VERBATIM from evaluate_sample_density_axis.py /
                          evaluate_sample_replicate_axis.py -- same
                          VARIANCE_SOURCE_FILES mapping, same
                          ``float(np.mean(var_map[mask]))``, same
                          safe_sqrt_variance guard -- so the number is
                          comparable across axes rather than being a new,
                          subtly-different quantity.

Sharpness metrics (interval_width_mean_nominal, interval_width_p95, crps),
which used to be computed and stored here as "archive" metrics, and their
diagnostics (crps_convergence.csv, the CRPS-vs-analytic and
backtr_value_vectorized-validation columns of evaluation_diagnostics.csv)
were removed on 2026-10-05 per user decision: they are no longer computed or
stored.

NO BASE-CASE REUSE (unlike the range and sample-density axes)
-------------------------------------------------------------
The base case is NUG=0.05, which is not one of the 6 axis levels (it lies
between 0.0 and 0.1), so there is no level whose MSE/UMG can be re-labeled
from results/processed/base_case/. Every number in this table is computed
from this axis's own 24 runs.

EVALUATION CELL SET
-------------------
Identical at every level: the conditioning-sample LOCATIONS do not change
across nugget levels (same SAMPLE_SEED, same grid, same n_samples), only the
sample VALUES do. So ``conditioning_cell_mask`` yields the same mask at every
level, and cross-level comparisons of any metric are over exactly the same
cells -- unlike the sample-density axis, where the evaluated-cell count
differs per level. This is verified (not assumed) in main().

Axis-level source data comes from
results/processed/nugget_axis/source_runs.json (written by
src.experiments.nugget_axis, which verifies each run against its level).

NUMBERS PUBLISHED HERE CHANGED ON 2026-09-22 (kriging only)
------------------------------------------------------------
The kriging arm was re-run after ``src/nscore.py`` repaired an off-by-one in
``geostatspy.geostats.nscore`` (see nugget_axis.py's docstring). Effect on
this table, measured: kriging MSE 2.9158 -> 2.9261 / 3.7032 -> 3.7025 /
4.6933 -> 4.5308 / 9.4425 -> 5.4079 / 6.2343 -> 6.2336 / 7.1443 -> 7.1406
and kriging UMG 0.9426 -> 0.9429 / 0.9466 -> 0.9470 / 0.9210 -> 0.9311 /
0.8361 -> 0.9244 / 0.9351 -> 0.9351 / 0.9467 -> 0.9467, at nug = 0.0 .. 0.5.
sgs / rbf_bootstrap / gp_mle are byte-identical before and after. The
pre-fix "kriging MSE spikes at nug=0.3" and "kriging UMG dips to 0.836"
features were artifacts of that bug and no longer exist; kriging MSE is now
monotone in the nugget.

Run with: .venv/Scripts/python.exe -m src.experiments.evaluate_nugget_axis
"""

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.evaluation import (
    accuracy_plot_fraction_in,
    calc_umg,
    conditioning_cell_mask,
    kriging_fraction_in,
    mse,
)
from src.experiments.base_case import NX, NY, XMN, YMN, XSIZ, YSIZ
from src.experiments.base_case_conditioning import (
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.kriging import (
    BACKTR_ZMAX,
    BACKTR_ZMIN,
    LTAIL,
    LTPAR,
    UTAIL,
    UTPAR,
)
from src.experiments.nugget_axis import (
    ALL_NUGGET_VALUES,
    AXIS,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    CASE,
    METHODS,
    SAMPLES_MATCH_ATOL,
    level_key,
    truth_nugget_real_units,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"
SOURCE_RUNS_PATH = PROCESSED_DIR / "source_runs.json"

ALL_METRICS = ("mse", "umg", "variance_mean")

# Per-method source array for variance_mean. All four are in physical units
# (Porosity %^2), BUT kriging's is a Monte Carlo back-transform approximation
# of its normal-score variance while the other three are native -- the same
# asymmetry evaluate_sample_density_axis.py records, carried over unchanged.
VARIANCE_SOURCE_FILES = {
    "kriging": "kriging_var_map_physical_mc.npy",
    "sgs": "sgs_var_map.npy",
    "rbf_bootstrap": "bootstrap_var_map.npy",
    "gp_mle": "posterior_var_map.npy",
}
VARIANCE_SOURCE_IS_MC_BACKTRANSFORM = {
    "kriging": True,
    "sgs": False,
    "rbf_bootstrap": False,
    "gp_mle": False,
}

# Same tolerance convention as evaluate_base_case.py / evaluate_range_axis.py
# / evaluate_sample_density_axis.py.
VARIANCE_CLIP_TOLERANCE = 1e-6


def safe_sqrt_variance(var_map: np.ndarray, name: str) -> np.ndarray:
    """sqrt of a variance map, defensively clipping small-negative
    floating-point noise to 0 while raising on anything larger (a real bug).
    Same logic as evaluate_base_case.py / evaluate_range_axis.py /
    evaluate_sample_density_axis.py (duplicated rather than imported, matching
    the project's per-axis evaluation-script pattern of keeping each script
    independently runnable).

    RELEVANT TO THIS AXIS SPECIFICALLY: at nug=0.0 the simple-kriging variance
    is exactly 0 at conditioning locations in exact arithmetic, and kb2d
    returns a tiny negative value (~-1e-16..-1e-7) there from floating-point
    roundoff. Those cells are excluded from the evaluation by
    ``conditioning_cell_mask`` anyway, but this guard is what makes the
    remaining path safe (no sqrt of a negative -> no NaN) AND makes any
    genuinely-negative variance a loud failure instead of a silent NaN. The
    per-level count of clipped values is recorded in the diagnostics table.
    """
    min_val = float(np.min(var_map))
    if min_val < -VARIANCE_CLIP_TOLERANCE:
        raise ValueError(
            f"{name}: variance map has a value ({min_val}) more negative "
            f"than the roundoff tolerance ({-VARIANCE_CLIP_TOLERANCE}) -- "
            "this looks like a real bug, not floating-point noise."
        )
    n_clipped = int(np.sum(var_map < 0))
    if n_clipped > 0:
        print(
            f"  note: {name}: clipped {n_clipped} small-negative variance "
            f"value(s) to 0 before sqrt (most negative: {min_val:g})."
        )
    return np.sqrt(np.clip(var_map, 0.0, None))


def _level_data(nug: float):
    """Regenerate this level's truth + conditioning samples and return
    (truth, samples_df, mask). Truth DIFFERS per level (the nugget goes into
    the generating variogram); the sample LOCATIONS do not."""
    truth = get_base_case_truth(
        truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1, nug=float(nug)
    )
    samples_df = get_conditioning_samples(
        truth, sample_seed=SAMPLE_SEED, n_samples=N_SAMPLES
    )
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    return truth, samples_df, mask


def evaluate_one_level(nug: float, run_dirs: dict):
    """Compute all 3 metrics for all 4 methods at one nugget level.

    Returns (metrics, curves, diagnostics, variance_sources).
    """
    axis_level = level_key(nug)
    truth, samples_df, mask = _level_data(nug)
    n_excluded = int((~mask).sum())
    n_evaluated = int(mask.sum())
    print(
        f"  nug={axis_level} (n_actual={len(samples_df)}): conditioning-sample cells "
        f"excluded: {n_excluded} of {mask.size} -> {n_evaluated} evaluated cells"
    )

    # Cross-check against each method's own recorded samples.csv -- the
    # identical-sample-locations-across-methods requirement. (nugget_axis.py
    # already did this at run time; repeated here so this script never trusts
    # a source_runs.json it did not itself validate.)
    for m in METHODS:
        recorded = pd.read_csv(_REPO_ROOT / run_dirs[m] / "samples.csv")
        if len(recorded) != len(samples_df) or not np.allclose(
            recorded[["X", "Y"]].values, samples_df[["X", "Y"]].values,
            rtol=0.0, atol=SAMPLES_MATCH_ATOL,
        ):
            raise ValueError(
                f"{m}'s recorded samples.csv does NOT match the regenerated "
                f"conditioning samples for the nug={axis_level} level -- the "
                "identical-sample-locations assumption is violated for this run."
            )

    truth_masked = truth[mask]
    metrics = {m: {} for m in METHODS}
    curves = {}
    variance_sources = []
    diagnostics = {
        "axis_level": axis_level,
        "truth_nugget_real_units": truth_nugget_real_units(nug),
        "n_samples_actual": len(samples_df),
        "n_cells_total": int(mask.size),
        "n_cells_excluded": n_excluded,
        "n_cells_evaluated": n_evaluated,
        "truth_mean": float(np.mean(truth)),
        "truth_stdev": float(np.std(truth)),
    }

    # ------------------------------------------------------------------
    # Kriging: MSE from kmap_physical; UMG from kmap_ns + vmap_ns via exact
    # quantile back-transform (never a variance back-transform).
    # ------------------------------------------------------------------
    kdir = _REPO_ROOT / run_dirs["kriging"]
    kmap_physical = np.load(kdir / "kriging_mean_map_physical.npy")
    kmap_ns = np.load(kdir / "kmap_ns.npy")
    vmap_ns = np.load(kdir / "kriging_var_map_ns.npy")
    transform_table = pd.read_csv(kdir / "nscore_transform_table.csv")
    vr, vrg = transform_table["vr"].values, transform_table["vrg"].values

    # Negative-variance bookkeeping for the whole kb2d output AND for the
    # evaluated subset -- at nug=0 kb2d returns tiny negatives at conditioning
    # cells (see safe_sqrt_variance's docstring). Recorded so the nug=0 level
    # can be audited without re-loading the arrays.
    diagnostics["kriging_vmap_ns_min_all_cells"] = float(np.min(vmap_ns))
    diagnostics["kriging_vmap_ns_n_negative_all_cells"] = int(np.sum(vmap_ns < 0))
    diagnostics["kriging_vmap_ns_min_evaluated_cells"] = float(np.min(vmap_ns[mask]))
    diagnostics["kriging_vmap_ns_n_negative_evaluated_cells"] = int(
        np.sum(vmap_ns[mask] < 0)
    )

    std_ns_masked = safe_sqrt_variance(vmap_ns[mask], "kriging vmap_ns")

    metrics["kriging"]["mse"] = mse(truth_masked, kmap_physical[mask])
    t0 = time.time()
    kriging_p, kriging_frac_in = kriging_fraction_in(
        truth_masked, kmap_ns[mask], std_ns_masked, vr, vrg,
        BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR,
    )
    metrics["kriging"]["umg"] = calc_umg(kriging_p, kriging_frac_in)
    curves["kriging"] = (kriging_p, kriging_frac_in)
    print(f"  kriging UMG (scalar back-transform path) took {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # The three physical-unit Gaussian-predictive methods. Source arrays per
    # src/evaluation.py's documented convention:
    #   SGS            -> ensemble mean/variance for both MSE and uncertainty
    #   RBF+bootstrap  -> MSE from the single-fit point_estimate_map;
    #                     uncertainty from (bootstrap_mean, bootstrap_var)
    #   GP-MLE         -> posterior mean/variance for both
    # ------------------------------------------------------------------
    sdir = _REPO_ROOT / run_dirs["sgs"]
    rdir = _REPO_ROOT / run_dirs["rbf_bootstrap"]
    gdir = _REPO_ROOT / run_dirs["gp_mle"]

    gaussian_specs = {
        "sgs": (
            np.load(sdir / "sgs_mean_map.npy"),
            np.load(sdir / "sgs_mean_map.npy"),
            np.load(sdir / "sgs_var_map.npy"),
        ),
        "rbf_bootstrap": (
            np.load(rdir / "point_estimate_map.npy"),
            np.load(rdir / "bootstrap_mean_map.npy"),
            np.load(rdir / "bootstrap_var_map.npy"),
        ),
        "gp_mle": (
            np.load(gdir / "posterior_mean_map.npy"),
            np.load(gdir / "posterior_mean_map.npy"),
            np.load(gdir / "posterior_var_map.npy"),
        ),
    }

    for method, (point_map, unc_mean_map, unc_var_map) in gaussian_specs.items():
        std_masked = safe_sqrt_variance(unc_var_map[mask], f"{method} variance map")
        mean_masked = unc_mean_map[mask]

        metrics[method]["mse"] = mse(truth_masked, point_map[mask])
        p, frac_in = accuracy_plot_fraction_in(truth_masked, mean_masked, std_masked)
        metrics[method]["umg"] = calc_umg(p, frac_in)
        curves[method] = (p, frac_in)

    # ------------------------------------------------------------------
    # variance_mean, all 4 methods, over the SAME evaluated cells as every
    # other metric at this level. Definition lifted verbatim from
    # evaluate_sample_density_axis.py / evaluate_sample_replicate_axis.py.
    # ------------------------------------------------------------------
    for method in METHODS:
        fname = VARIANCE_SOURCE_FILES[method]
        var_map = np.load(_REPO_ROOT / run_dirs[method] / fname)
        if var_map.shape != mask.shape:
            raise ValueError(
                f"{method}: {fname} has shape {var_map.shape}, expected {mask.shape}."
            )
        var_masked = var_map[mask]
        # Guard only (raises on a meaningfully-negative variance); the MEAN
        # itself uses the raw values, so a roundoff-level negative is carried
        # rather than hidden.
        safe_sqrt_variance(var_masked, f"{method} variance map (variance_mean)")
        metrics[method]["variance_mean"] = float(np.mean(var_masked))
        variance_sources.append(
            {
                "axis_level": axis_level,
                "method": method,
                "variance_source_file": fname,
                "units": "Porosity %^2",
                "is_monte_carlo_backtransform_approximation":
                    VARIANCE_SOURCE_IS_MC_BACKTRANSFORM[method],
                "n_cells_averaged": n_evaluated,
                "variance_mean": metrics[method]["variance_mean"],
            }
        )

    return metrics, curves, diagnostics, variance_sources


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_RUNS_PATH, "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    expected_levels = {level_key(n) for n in ALL_NUGGET_VALUES}
    if set(source_runs) != expected_levels:
        raise ValueError(
            f"{SOURCE_RUNS_PATH} has axis levels {sorted(source_runs)}; expected "
            f"{sorted(expected_levels)}."
        )

    # The evaluation cell set must be IDENTICAL at every level on this axis
    # (see module docstring) -- verified, not assumed.
    masks = {level_key(n): _level_data(n)[2] for n in ALL_NUGGET_VALUES}
    reference_mask = masks[level_key(ALL_NUGGET_VALUES[0])]
    for lvl, m in masks.items():
        if not np.array_equal(m, reference_mask):
            raise ValueError(
                f"nugget level {lvl} has a different conditioning-cell mask than level "
                f"{level_key(ALL_NUGGET_VALUES[0])} -- on this axis the sample LOCATIONS "
                "are supposed to be identical across levels, so the masks must match."
            )
    print(
        f"verified: all {len(masks)} levels share one identical evaluation cell set "
        f"({int(reference_mask.sum())} cells)."
    )

    rows = []
    curve_rows = []
    diagnostics_rows = []
    variance_source_rows = []

    for nug in ALL_NUGGET_VALUES:
        axis_level = level_key(nug)
        print(
            f"\nEvaluating nug={axis_level} "
            f"({truth_nugget_real_units(nug):g} Porosity %^2)..."
        )
        run_dirs = source_runs[axis_level]
        metrics, curves, diagnostics, variance_sources = evaluate_one_level(nug, run_dirs)
        for method in METHODS:
            for metric_name, value in metrics[method].items():
                rows.append(
                    {
                        "case": CASE,
                        "axis": AXIS,
                        "axis_level": axis_level,
                        "method": method,
                        "metric": metric_name,
                        "value": value,
                    }
                )
            p_intervals, fraction_in = curves[method]
            for p, fr in zip(p_intervals, fraction_in):
                curve_rows.append(
                    {
                        "axis_level": axis_level,
                        "method": method,
                        "p_interval": p,
                        "fraction_in": fr,
                    }
                )
        diagnostics_rows.append(diagnostics)
        variance_source_rows.extend(variance_sources)

    metrics_df = pd.DataFrame(rows)
    if metrics_df["value"].isna().any() or not np.all(np.isfinite(metrics_df["value"].values)):
        bad = metrics_df[~np.isfinite(metrics_df["value"].values)]
        raise ValueError(f"metrics_df contains NaN/inf values:\n{bad.to_string()}")

    expected_n_rows = len(ALL_NUGGET_VALUES) * len(METHODS) * len(ALL_METRICS)
    if len(metrics_df) != expected_n_rows:
        raise ValueError(
            f"metrics_df has {len(metrics_df)} rows; expected {expected_n_rows} "
            f"({len(ALL_NUGGET_VALUES)} levels x {len(METHODS)} methods x "
            f"{len(ALL_METRICS)} metrics)."
        )
    if set(metrics_df["metric"]) != set(ALL_METRICS):
        raise ValueError(
            f"metrics_df metric set {sorted(set(metrics_df['metric']))} != expected "
            f"{sorted(ALL_METRICS)}."
        )

    metrics_df["_axis_level_numeric"] = metrics_df["axis_level"].astype(float)
    metrics_df = (
        metrics_df.sort_values(["_axis_level_numeric", "method", "metric"])
        .drop(columns="_axis_level_numeric")
        .reset_index(drop=True)
    )

    metrics_csv = PROCESSED_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    curves_df = pd.DataFrame(curve_rows)
    curves_csv = PROCESSED_DIR / "accuracy_plot_curves.csv"
    curves_df.to_csv(curves_csv, index=False)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_csv = PROCESSED_DIR / "evaluation_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    variance_sources_df = pd.DataFrame(variance_source_rows)
    variance_sources_df["note"] = (
        "variance_mean is taken over this level's evaluated cells (the conditioning "
        "cells excluded). All four arrays are Porosity %^2, but kriging's physical-unit "
        "variance is a Monte Carlo back-transform of its normal-score variance, whereas "
        "sgs/rbf_bootstrap/gp_mle variances are native physical-unit arrays. On this axis "
        "the evaluated-cell set is identical at every level, so variance_mean is directly "
        "comparable across levels."
    )
    variance_sources_csv = PROCESSED_DIR / "variance_metric_sources.csv"
    variance_sources_df.to_csv(variance_sources_csv, index=False)

    # --- Wide summary table for reporting ---------------------------------
    wide = metrics_df.pivot_table(
        index=["axis_level", "method"], columns="metric", values="value"
    ).reset_index()
    wide["_n"] = wide["axis_level"].astype(float)
    wide = wide.sort_values(["_n", "method"]).drop(columns="_n").reset_index(drop=True)
    wide_csv = PROCESSED_DIR / "metrics_wide.csv"
    wide.to_csv(wide_csv, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)

    print("\nNugget axis metrics (6 levels x 4 methods):")
    print(wide[["axis_level", "method", *ALL_METRICS]].to_string(index=False))
    print("\nEvaluation diagnostics:")
    print(diagnostics_df.to_string(index=False))
    print(f"\nmetrics.csv: {metrics_csv}")
    print(f"metrics_wide.csv: {wide_csv}")
    print(f"accuracy_plot_curves.csv: {curves_csv}")
    print(f"evaluation_diagnostics.csv: {diagnostics_csv}")
    print(f"variance_metric_sources.csv: {variance_sources_csv}")

    return metrics_df, curves_df


if __name__ == "__main__":
    main()
