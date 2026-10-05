"""Evaluate the range-axis 4-method comparison (kriging / SGS / RBF+
bootstrap / GP-MLE) over the 8 axis levels range = 100, 200, ..., 800 m
(docs/experiment_context.md deliverable 2).

Metrics (reported SEPARATELY -- never combined into one number; see
src/evaluation.py's module docstring for the per-method source-array and
predictive-quantile conventions, reused as-is here):

  accuracy
    - ``mse``                          (porosity %^2, lower better)
  uncertainty quality
    - ``umg``                          (coverage goodness, 1.0 = ideal)

Sharpness metrics (interval width, CRPS) and their diagnostics
(crps_convergence.csv, evaluation_diagnostics.csv) were removed on
2026-10-05 per user decision: they are no longer computed or stored.

Axis-level source data
----------------------
Run directories come from results/processed/range_axis/source_runs.json
(written by src.experiments.range_axis), which pins all 8 levels including
the two REUSED ones: range=300 (the base case's own pinned runs) and
range=800 (the first range-axis execution's runs). The dropped range=50
level is not in that file and therefore cannot leak into this table.

For range=300 the already-computed, already-reviewed ``mse``/``umg`` values
(and accuracy-plot curve) are read from results/processed/base_case/ rather
than recomputed, per task instruction.

Run with: .venv/Scripts/python.exe -m src.experiments.evaluate_range_axis
"""

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from src.experiments.range_axis import ALL_RANGE_VALUES, BASE_CASE_AXIS_LEVEL, METHODS

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"
BASE_CASE_PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "base_case"
SOURCE_RUNS_PATH = PROCESSED_DIR / "source_runs.json"

CASE = "range_axis"
AXIS = "range"

# Same tolerance convention as evaluate_base_case.py.
VARIANCE_CLIP_TOLERANCE = 1e-6

# Metrics whose range=300 value is taken from the base case rather than
# recomputed (task instruction). Everything else is computed here for all 8
# levels, including range=300.
REUSED_FROM_BASE_CASE_METRICS = ("mse", "umg")


def safe_sqrt_variance(var_map: np.ndarray, name: str) -> np.ndarray:
    """sqrt of a variance map, defensively clipping small-negative
    floating-point noise to 0 while raising on anything larger (a real bug).
    Identical logic to evaluate_base_case.py's function of the same name
    (duplicated rather than imported to keep each evaluate_*.py script
    self-contained/independently runnable, matching the project's existing
    per-axis evaluation script pattern).
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
        print(f"  note: {name}: clipped {n_clipped} small-negative variance value(s) to 0 before sqrt.")
    return np.sqrt(np.clip(var_map, 0.0, None))


def evaluate_one_range(hmaj1: float, run_dirs: dict, compute_mse_umg: bool = True):
    """Compute all metrics for all 4 methods at variogram range hmaj1
    (== hmin1, isotropic).

    ``compute_mse_umg=False`` skips MSE/UMG (and the accuracy-plot curve) for
    the range=300 level, whose values are reused from the base case instead;
    only the identical-sample-locations cross-check is run for that level.

    Returns (metrics, curves).
    """
    hmin1 = hmaj1

    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=hmaj1, hmin1=hmin1)
    samples_df = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED)
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    n_excluded = int((~mask).sum())
    print(f"  range={hmaj1:g}m: conditioning-sample cells excluded: {n_excluded} of {mask.size}")

    # Cross-check against each method's own recorded samples.csv, purely as
    # a drift-detection sanity check (same as evaluate_base_case.py).
    for m in METHODS:
        recorded = pd.read_csv(_REPO_ROOT / run_dirs[m] / "samples.csv")
        if len(recorded) != len(samples_df) or not np.allclose(
            recorded[["X", "Y"]].values, samples_df[["X", "Y"]].values
        ):
            raise ValueError(
                f"{m}'s recorded samples.csv does NOT match the regenerated "
                f"conditioning samples for range={hmaj1:g}m -- the "
                "identical-sample-locations assumption is violated for this run."
            )

    truth_masked = truth[mask]
    metrics = {m: {} for m in METHODS}
    curves = {}

    if not compute_mse_umg:
        return metrics, curves

    # ------------------------------------------------------------------
    # Kriging: MSE from kmap_physical; UMG from kmap_ns + vmap_ns via exact
    # quantile back-transform (never a variance back-transform).
    # Back-transform parameters (BACKTR_ZMIN/ZMAX/LTAIL/LTPAR/UTAIL/UTPAR)
    # are the fixed base-case constants, not varied by range
    # (one-factor-at-a-time).
    # ------------------------------------------------------------------
    kdir = _REPO_ROOT / run_dirs["kriging"]
    kmap_physical = np.load(kdir / "kriging_mean_map_physical.npy")
    kmap_ns = np.load(kdir / "kmap_ns.npy")
    vmap_ns = np.load(kdir / "kriging_var_map_ns.npy")
    transform_table = pd.read_csv(kdir / "nscore_transform_table.csv")
    vr, vrg = transform_table["vr"].values, transform_table["vrg"].values

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
            np.load(sdir / "sgs_mean_map.npy"),          # MSE point estimate
            np.load(sdir / "sgs_mean_map.npy"),          # uncertainty-model mean
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

    return metrics, curves


def load_base_case_reused(rows: list, curve_rows: list) -> None:
    """Append the range=300 MSE/UMG rows + accuracy-plot curve, read from the
    base case's processed outputs rather than recomputed (task instruction)."""
    base_metrics_path = BASE_CASE_PROCESSED_DIR / "metrics.csv"
    base_curves_path = BASE_CASE_PROCESSED_DIR / "accuracy_plot_curves.csv"
    base_df = pd.read_csv(base_metrics_path)
    base_curves = pd.read_csv(base_curves_path)

    if not set(base_df["method"]) >= set(METHODS) or not set(base_df["metric"]) >= set(
        REUSED_FROM_BASE_CASE_METRICS
    ):
        raise ValueError(
            f"{base_metrics_path} is missing an expected method/metric -- "
            "cannot safely relabel it into the range-axis table."
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
        # Consistency check: the reused UMG value must be exactly what the
        # reused curve implies -- otherwise the two base-case files disagree
        # and one of them is stale.
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
        f"\nrange=300m MSE/UMG + accuracy-plot curve sourced from {BASE_CASE_PROCESSED_DIR} "
        "(re-labeled, not recomputed)."
    )


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_RUNS_PATH, "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    expected_levels = {str(int(r)) for r in ALL_RANGE_VALUES}
    if set(source_runs) != expected_levels:
        raise ValueError(
            f"{SOURCE_RUNS_PATH} has axis levels {sorted(source_runs)}; expected "
            f"{sorted(expected_levels)}."
        )

    rows = []
    curve_rows = []

    for hmaj1 in ALL_RANGE_VALUES:
        axis_level = str(int(hmaj1))
        compute_mse_umg = axis_level != BASE_CASE_AXIS_LEVEL
        print(
            f"\nEvaluating range={hmaj1:g}m (axis_level={axis_level})"
            + ("" if compute_mse_umg else " [MSE/UMG reused from base case]")
            + "..."
        )
        run_dirs = source_runs[axis_level]
        metrics, curves = evaluate_one_range(
            hmaj1, run_dirs, compute_mse_umg=compute_mse_umg
        )
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

    load_base_case_reused(rows, curve_rows)

    metrics_df = pd.DataFrame(rows)
    if metrics_df["value"].isna().any() or not np.all(np.isfinite(metrics_df["value"].values)):
        raise ValueError("metrics_df contains NaN/inf values -- see printed metrics above.")

    expected_n_rows = len(ALL_RANGE_VALUES) * len(METHODS) * len(REUSED_FROM_BASE_CASE_METRICS)
    if len(metrics_df) != expected_n_rows:
        raise ValueError(
            f"metrics_df has {len(metrics_df)} rows; expected {expected_n_rows} "
            f"({len(ALL_RANGE_VALUES)} levels x {len(METHODS)} methods x "
            f"{len(REUSED_FROM_BASE_CASE_METRICS)} metrics)."
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

    # --- Wide summary table for reporting ---------------------------------
    wide = metrics_df.pivot_table(
        index=["axis_level", "method"], columns="metric", values="value"
    ).reset_index()
    wide["_n"] = wide["axis_level"].astype(float)
    wide = wide.sort_values(["_n", "method"]).drop(columns="_n").reset_index(drop=True)
    wide_csv = PROCESSED_DIR / "metrics_wide.csv"
    wide.to_csv(wide_csv, index=False)

    print("\nRange axis metrics (wide view; 8 levels x 4 methods):")
    print(wide.to_string(index=False))
    print(f"\nmetrics.csv: {metrics_csv}")
    print(f"metrics_wide.csv: {wide_csv}")
    print(f"accuracy_plot_curves.csv: {curves_csv}")

    return metrics_df, curves_df


if __name__ == "__main__":
    main()
