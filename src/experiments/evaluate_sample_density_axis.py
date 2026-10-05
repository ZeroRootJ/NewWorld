"""Evaluate the sample-density-axis 4-method comparison (kriging / SGS /
RBF+bootstrap / GP-MLE) over the axis levels 5% / 2% / 1% of the 2500-cell
grid (n_samples = 125 / 50 / 25), EXTENDED 2026-09-21 (user request) with two
denser levels 20% / 10% (n_samples_requested = 500 / 250; ``axis_level``
"20"/"10"), i.e. 5 levels total via sample_density_axis.EXTENDED_AXIS_LEVELS.
The added levels are computed exactly like 2%/1% (fresh runs, MSE/UMG computed
here; only the base-case level "5" reuses MSE/UMG from results/processed/
base_case). Existing levels' rows are unchanged; the output tables are sorted
dense -> sparse (20, 10, 5, 2, 1).

Metrics -- reported SEPARATELY (never combined into one number; see
src/evaluation.py's module docstring for the per-method source-array and
predictive-quantile conventions, reused here UNCHANGED):

  accuracy
    - ``mse``                          (porosity %^2, lower better)
  uncertainty quality
    - ``umg``                          (coverage goodness, 1.0 = ideal)
  predictive-variance magnitude
    - ``variance_sum``                 (porosity %^2, sum of the predictive
                                        variance over the evaluated cells)
    - ``variance_mean``                (porosity %^2, the same quantity divided
                                        by that level's evaluated-cell count)

Sharpness metrics (``interval_width_mean_nominal``, ``interval_width_p95``,
``crps``) were parked on 2026-09-15 and then removed entirely on 2026-10-05
per user decision, together with their stored values
(metrics_parked_sharpness.csv) and diagnostics (crps_convergence.csv,
kriging_backtransform_tail_sensitivity.csv): they are no longer computed or
stored.

NOTE on the variance metrics' source arrays (fact, not interpretation):
three of the four methods supply a NATIVE physical-unit (porosity %^2)
predictive variance -- sgs (``sgs_var_map.npy``), rbf_bootstrap
(``bootstrap_var_map.npy``), gp_mle (``posterior_var_map.npy``). Kriging does
NOT: simple kriging runs in normal-score space, and its physical-unit variance
``kriging_var_map_physical_mc.npy`` is a MONTE CARLO APPROXIMATION obtained by
back-transforming samples of the normal-score predictive distribution. So the
kriging column of variance_sum/variance_mean is an approximation with Monte
Carlo error while the other three are exact given their own definitions. This
asymmetry is recorded per level in ``variance_metric_sources.csv``.

IMPORTANT -- the evaluation cell set is NOT the same across axis levels
-----------------------------------------------------------------------
All four methods exclude the conditioning-sample cells before any metric is
computed (``conditioning_cell_mask``), so that no method is scored on cells
where it trivially reproduces the data. That exclusion set is the SAME for
all four methods at a given level (which is what the cross-method comparison
requires) but NECESSARILY DIFFERENT ACROSS LEVELS, because the levels have
different numbers of samples in different places:

    5% -> 121 cells excluded -> 2379 evaluated
    2% ->  50 cells excluded -> 2450 evaluated
    1% ->  25 cells excluded -> 2475 evaluated

So a cross-level comparison of e.g. MSE is NOT a comparison over an
identical cell set. The exact evaluated-cell count per level is written into
metrics.csv's companion file ``evaluation_cell_counts.csv`` (and into the
diagnostics table) so this caveat is never lost downstream. The effect is
expected to be small (the level sets differ by <4% of cells) but it is a
real, non-zero difference and is NOT silently ignored.

Related limitation, inherited from the axis design (see
src/experiments/sample_density_axis.py): the levels' sample LOCATIONS are
drawn independently, not nested, so a level-to-level difference mixes "less
data" with "different data placement".

Axis-level source data
----------------------
Run directories come from
results/processed/sample_density_axis/source_runs.json (written by
src.experiments.sample_density_axis), which pins all 3 levels including the
REUSED 5% level (the base case's own pinned runs).

For the 5% level the already-computed, already-reviewed ``mse``/``umg``
values (and accuracy-plot curve) are read from results/processed/base_case/
rather than recomputed, exactly as the range axis does for its range=300
level. ``variance_sum``/``variance_mean`` ARE computed here for all
levels, since those metrics did not exist when the base case was evaluated.

Run with:
.venv/Scripts/python.exe -m src.experiments.evaluate_sample_density_axis
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
from src.experiments.sample_density_axis import (
    AXIS_HMAJ1,
    AXIS_HMIN1,
    BASE_CASE_AXIS_LEVEL,
    EXTENDED_AXIS_LEVELS,
    METHODS,
    SAMPLE_COUNTS,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_density_axis"
BASE_CASE_PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "base_case"
SOURCE_RUNS_PATH = PROCESSED_DIR / "source_runs.json"

CASE = "sample_density_axis"
AXIS = "sample_fraction_pct"

# Metrics emitted by this script. Used for the row-count self-check in main().
CORE_METRICS = ("mse", "umg", "variance_sum", "variance_mean")

# Per-method source array for variance_sum / variance_mean. All four are in
# physical units (porosity %^2), BUT kriging's is a Monte Carlo back-transform
# approximation while the other three are native (see module docstring).
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

# Same tolerance convention as evaluate_base_case.py / evaluate_range_axis.py.
VARIANCE_CLIP_TOLERANCE = 1e-6

# Metrics whose 5% value is taken from the base case rather than recomputed
# (same convention as evaluate_range_axis.py's range=300 level).
REUSED_FROM_BASE_CASE_METRICS = ("mse", "umg")

# Same float-text round-trip tolerance the axis script uses for samples.csv.
SAMPLES_MATCH_ATOL = 1e-10


def safe_sqrt_variance(var_map: np.ndarray, name: str) -> np.ndarray:
    """sqrt of a variance map, defensively clipping small-negative
    floating-point noise to 0 while raising on anything larger (a real bug).
    Same logic as evaluate_base_case.py / evaluate_range_axis.py (duplicated
    rather than imported, matching the project's per-axis evaluation-script
    pattern of keeping each script independently runnable)."""
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
            "value(s) to 0 before sqrt."
        )
    return np.sqrt(np.clip(var_map, 0.0, None))


def _level_mask(axis_level: str):
    """Regenerate this level's truth + conditioning samples and return
    (truth, samples_df, mask). The truth field is the SAME for all three
    levels (same TRUTH_SEED, same range) -- only the samples differ."""
    n = SAMPLE_COUNTS[axis_level]
    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)
    samples_df = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED, n_samples=n)
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    return truth, samples_df, mask


def evaluate_one_level(axis_level: str, run_dirs: dict, compute_mse_umg: bool = True):
    """Compute all metrics for all 4 methods at this sample-density level.

    ``compute_mse_umg=False`` skips MSE/UMG (and the accuracy-plot curve) for
    the 5% level, whose values are reused from the base case instead.

    Returns (metrics, curves, diagnostics, variance_sources).
    """
    n_requested = SAMPLE_COUNTS[axis_level]
    truth, samples_df, mask = _level_mask(axis_level)
    n_excluded = int((~mask).sum())
    n_evaluated = int(mask.sum())
    print(
        f"  {axis_level}% (n_requested={n_requested}, n_actual={len(samples_df)}): "
        f"conditioning-sample cells excluded: {n_excluded} of {mask.size} "
        f"-> {n_evaluated} evaluated cells"
    )

    # Cross-check against each method's own recorded samples.csv -- the
    # identical-sample-locations-across-methods requirement.
    for m in METHODS:
        recorded = pd.read_csv(_REPO_ROOT / run_dirs[m] / "samples.csv")
        if len(recorded) != len(samples_df) or not np.allclose(
            recorded[["X", "Y"]].values, samples_df[["X", "Y"]].values,
            rtol=0.0, atol=SAMPLES_MATCH_ATOL,
        ):
            raise ValueError(
                f"{m}'s recorded samples.csv does NOT match the regenerated "
                f"conditioning samples for the {axis_level}% level -- the "
                "identical-sample-locations assumption is violated for this run."
            )

    truth_masked = truth[mask]
    metrics = {m: {} for m in METHODS}
    curves = {}
    diagnostics = {
        "n_samples_requested": n_requested,
        "n_samples_actual": len(samples_df),
        "n_cells_total": int(mask.size),
        "n_cells_excluded": n_excluded,
        "n_cells_evaluated": n_evaluated,
    }

    # ------------------------------------------------------------------
    # Kriging: MSE from kmap_physical; UMG from kmap_ns + vmap_ns via exact
    # quantile back-transform.
    # ------------------------------------------------------------------
    kdir = _REPO_ROOT / run_dirs["kriging"]
    kmap_physical = np.load(kdir / "kriging_mean_map_physical.npy")
    kmap_ns = np.load(kdir / "kmap_ns.npy")
    vmap_ns = np.load(kdir / "kriging_var_map_ns.npy")
    transform_table = pd.read_csv(kdir / "nscore_transform_table.csv")
    vr, vrg = transform_table["vr"].values, transform_table["vrg"].values

    diagnostics["kriging_nscore_table_n_points"] = int(len(vr))

    std_ns_masked = safe_sqrt_variance(vmap_ns[mask], "kriging vmap_ns")

    if compute_mse_umg:
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
    # The three physical-unit Gaussian-predictive methods (same source-array
    # convention as every other evaluation script in this project).
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

        if compute_mse_umg:
            metrics[method]["mse"] = mse(truth_masked, point_map[mask])
            p, frac_in = accuracy_plot_fraction_in(truth_masked, mean_masked, std_masked)
            metrics[method]["umg"] = calc_umg(p, frac_in)
            curves[method] = (p, frac_in)

    # ------------------------------------------------------------------
    # Predictive-variance magnitude, all 4 methods.
    #
    # Summed over the SAME evaluated cells as every other metric at this
    # level (this level's conditioning cells excluded), from each method's
    # physical-unit (porosity %^2) variance map. variance_mean is the same
    # quantity divided by n_cells_evaluated and is reported alongside the sum
    # because n_cells_evaluated differs between levels (2379/2450/2475), so a
    # cross-level comparison of the SUM alone is not over an equal number of
    # cells.
    #
    # Source-array asymmetry (fact, no interpretation): kriging's physical-unit
    # variance is a MONTE CARLO back-transform approximation of its
    # normal-score variance; sgs / rbf_bootstrap / gp_mle variances are native
    # physical-unit arrays. Recorded per method in variance_metric_sources.csv.
    # ------------------------------------------------------------------
    variance_sources = []
    for method in METHODS:
        fname = VARIANCE_SOURCE_FILES[method]
        var_map = np.load(_REPO_ROOT / run_dirs[method] / fname)
        if var_map.shape != mask.shape:
            raise ValueError(
                f"{method}: {fname} has shape {var_map.shape}, expected {mask.shape}."
            )
        var_masked = var_map[mask]
        # Same defensive check used before every sqrt in this script: raises on
        # a meaningfully-negative variance. The SUM itself uses the raw values
        # (no clipping), so a roundoff-level negative is carried, not hidden.
        safe_sqrt_variance(var_masked, f"{method} variance map (variance_sum)")
        metrics[method]["variance_sum"] = float(np.sum(var_masked))
        metrics[method]["variance_mean"] = float(np.mean(var_masked))
        variance_sources.append(
            {
                "axis_level": None,  # filled in by the caller
                "method": method,
                "variance_source_file": fname,
                "units": "porosity %^2",
                "is_monte_carlo_backtransform_approximation": VARIANCE_SOURCE_IS_MC_BACKTRANSFORM[
                    method
                ],
                "n_cells_summed": n_evaluated,
                "variance_sum": metrics[method]["variance_sum"],
                "variance_mean": metrics[method]["variance_mean"],
            }
        )

    return metrics, curves, diagnostics, variance_sources


def load_base_case_reused(rows: list, curve_rows: list) -> None:
    """Append the 5% level's MSE/UMG rows + accuracy-plot curve, read from the
    base case's processed outputs rather than recomputed."""
    base_metrics_path = BASE_CASE_PROCESSED_DIR / "metrics.csv"
    base_curves_path = BASE_CASE_PROCESSED_DIR / "accuracy_plot_curves.csv"
    base_df = pd.read_csv(base_metrics_path)
    base_curves = pd.read_csv(base_curves_path)

    if not set(base_df["method"]) >= set(METHODS) or not set(base_df["metric"]) >= set(
        REUSED_FROM_BASE_CASE_METRICS
    ):
        raise ValueError(
            f"{base_metrics_path} is missing an expected method/metric -- "
            "cannot safely relabel it into the sample-density-axis table."
        )

    for _, r in base_df.iterrows():
        if r["method"] not in METHODS or r["metric"] not in REUSED_FROM_BASE_CASE_METRICS:
            continue
        rows.append(
            {
                "case": CASE,
                "axis": AXIS,
                "axis_level": BASE_CASE_AXIS_LEVEL,
                "method": r["method"],
                "metric": r["metric"],
                "value": r["value"],
            }
        )

    for method in METHODS:
        sub = base_curves[base_curves["method"] == method]
        umg_from_curve = calc_umg(sub["p_interval"].values, sub["fraction_in"].values)
        umg_recorded = float(
            base_df[(base_df["method"] == method) & (base_df["metric"] == "umg")]["value"].iloc[0]
        )
        if not np.isclose(umg_from_curve, umg_recorded):
            raise ValueError(
                f"base-case {method}: UMG recomputed from accuracy_plot_curves.csv "
                f"({umg_from_curve}) != metrics.csv value ({umg_recorded}) -- the two "
                "base-case processed files are inconsistent."
            )
        for _, r in sub.iterrows():
            curve_rows.append(
                {
                    "axis_level": BASE_CASE_AXIS_LEVEL,
                    "method": method,
                    "p_interval": r["p_interval"],
                    "fraction_in": r["fraction_in"],
                }
            )
    print(
        f"\n{BASE_CASE_AXIS_LEVEL}% MSE/UMG + accuracy-plot curve sourced from "
        f"{BASE_CASE_PROCESSED_DIR} (re-labeled, not recomputed); its "
        "variance_sum/variance_mean values ARE computed here."
    )


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_RUNS_PATH, "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    if set(source_runs) != set(EXTENDED_AXIS_LEVELS):
        raise ValueError(
            f"{SOURCE_RUNS_PATH} has axis levels {sorted(source_runs)}; expected "
            f"{sorted(EXTENDED_AXIS_LEVELS)}."
        )

    rows = []
    curve_rows = []
    diagnostics_rows = []
    variance_source_rows = []

    for axis_level in EXTENDED_AXIS_LEVELS:
        compute_mse_umg = axis_level != BASE_CASE_AXIS_LEVEL
        print(
            f"\nEvaluating sample fraction = {axis_level}% "
            f"(n_samples={SAMPLE_COUNTS[axis_level]})"
            + ("" if compute_mse_umg else " [MSE/UMG reused from base case]")
            + "..."
        )
        metrics, curves, diagnostics, variance_sources = evaluate_one_level(
            axis_level, source_runs[axis_level], compute_mse_umg=compute_mse_umg
        )
        for vs in variance_sources:
            variance_source_rows.append({**vs, "axis_level": axis_level})
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
            if method in curves:
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
        diagnostics_rows.append({"axis_level": axis_level, **diagnostics})

    load_base_case_reused(rows, curve_rows)

    metrics_df = pd.DataFrame(rows)
    if metrics_df["value"].isna().any() or not np.all(np.isfinite(metrics_df["value"].values)):
        raise ValueError("metrics_df contains NaN/inf values -- see printed metrics above.")

    expected_metrics = CORE_METRICS
    expected_n_rows = len(EXTENDED_AXIS_LEVELS) * len(METHODS) * len(expected_metrics)
    if len(metrics_df) != expected_n_rows:
        raise ValueError(
            f"metrics_df has {len(metrics_df)} rows; expected {expected_n_rows} "
            f"({len(EXTENDED_AXIS_LEVELS)} levels x {len(METHODS)} methods x "
            f"{len(expected_metrics)} metrics: {', '.join(expected_metrics)})."
        )
    if set(metrics_df["metric"]) != set(expected_metrics):
        raise ValueError(
            f"metrics_df holds metrics {sorted(set(metrics_df['metric']))}; expected "
            f"{sorted(expected_metrics)}."
        )

    # Sort by DECREASING sample fraction (20 -> 10 -> 5 -> 2 -> 1), the axis's
    # natural reading order.
    metrics_df["_axis_level_numeric"] = metrics_df["axis_level"].astype(float)
    metrics_df = (
        metrics_df.sort_values(
            ["_axis_level_numeric", "method", "metric"], ascending=[False, True, True]
        )
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

    # --- Evaluated-cell counts, as a standalone file ----------------------
    # The single most important caveat of this axis (see module docstring):
    # the levels are NOT scored on the same cell set.
    cell_counts = diagnostics_df[
        [
            "axis_level",
            "n_samples_requested",
            "n_samples_actual",
            "n_cells_total",
            "n_cells_excluded",
            "n_cells_evaluated",
        ]
    ].copy()
    cell_counts["note"] = (
        "Conditioning-sample cells are excluded from evaluation for all 4 methods "
        "identically WITHIN a level, but the excluded set differs BETWEEN levels, so "
        "cross-level metric comparisons are not over an identical cell set."
    )
    cell_counts_csv = PROCESSED_DIR / "evaluation_cell_counts.csv"
    cell_counts.to_csv(cell_counts_csv, index=False)

    # --- Predictive-variance source table ---------------------------------
    # Records, per level and method, WHICH array variance_sum/variance_mean
    # were computed from and whether that array is a Monte Carlo
    # back-transform approximation (kriging) or native physical units (the
    # other three). Fact only -- no interpretation attached.
    variance_sources_df = pd.DataFrame(variance_source_rows)[
        [
            "axis_level",
            "method",
            "variance_source_file",
            "units",
            "is_monte_carlo_backtransform_approximation",
            "n_cells_summed",
            "variance_sum",
            "variance_mean",
        ]
    ]
    variance_sources_df["note"] = (
        "variance_sum/variance_mean are taken over this level's evaluated cells "
        "(its conditioning cells excluded). All four arrays are porosity %^2, but "
        "kriging's physical-unit variance is a Monte Carlo back-transform of its "
        "normal-score variance, whereas sgs/rbf_bootstrap/gp_mle variances are "
        "native physical-unit arrays. n_cells_summed differs between levels, so "
        "variance_sum is not a sum over an equal number of cells across levels; "
        "variance_mean is reported for that reason."
    )
    variance_sources_csv = PROCESSED_DIR / "variance_metric_sources.csv"
    variance_sources_df.to_csv(variance_sources_csv, index=False)

    # --- Wide summary table for reporting ---------------------------------
    wide = metrics_df.pivot_table(
        index=["axis_level", "method"], columns="metric", values="value"
    ).reset_index()
    wide["_n"] = wide["axis_level"].astype(float)
    wide = (
        wide.sort_values(["_n", "method"], ascending=[False, True])
        .drop(columns="_n")
        .reset_index(drop=True)
    )
    wide_csv = PROCESSED_DIR / "metrics_wide.csv"
    wide.to_csv(wide_csv, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)

    print("\nSample-density axis metrics (wide view; 5 levels x 4 methods):")
    print(wide.to_string(index=False))
    print("\nEvaluated-cell counts per level (levels are NOT scored on the same cells):")
    print(cell_counts.drop(columns="note").to_string(index=False))
    print(
        "\nPredictive-variance metric sources (kriging = Monte Carlo back-transform "
        "approximation; others native physical units):"
    )
    print(variance_sources_df.drop(columns="note").to_string(index=False))
    print("\nEvaluation diagnostics:")
    print(diagnostics_df.to_string(index=False))
    print(f"\nmetrics.csv: {metrics_csv}")
    print(f"metrics_wide.csv: {wide_csv}")
    print(f"accuracy_plot_curves.csv: {curves_csv}")
    print(f"evaluation_cell_counts.csv: {cell_counts_csv}")
    print(f"variance_metric_sources.csv: {variance_sources_csv}")
    print(f"evaluation_diagnostics.csv: {diagnostics_csv}")

    return metrics_df, curves_df


if __name__ == "__main__":
    main()
