"""Evaluate the sample-seed replicate axis (src/experiments/
sample_replicate_axis.py): 10 fresh sample-location draws (sample_seed in
{1001, ..., 1010}) at the SAME ground truth (TRUTH_SEED=101, range=300 m)
and the SAME requested sample count (n_samples=125, the base case's "5%"
density level), for all 4 methods (kriging / SGS / RBF+bootstrap / GP-MLE).

Purpose: quantify how much each method's accuracy/calibration/predictive-
variance moves from sample PLACEMENT alone, with sample count and ground
truth held fixed -- a different question from the sample-density axis
(src/experiments/evaluate_sample_density_axis.py), which varies n_samples
and reports levels that are NOT scored on the same cell set.

Metrics -- reported SEPARATELY (never combined into one number), reusing
src/evaluation.py's functions exactly as evaluate_sample_density_axis.py
does, with the SAME per-method source-array convention:

  accuracy
    - ``mse``           (porosity %^2, lower better)
  uncertainty quality
    - ``umg``            (coverage goodness, 1.0 = ideal)
  predictive-variance magnitude
    - ``variance_mean``  (porosity %^2, the predictive variance averaged
                          over that replicate's evaluated cells)

``variance_sum`` is deliberately NOT computed here (user decision,
2026-09-18, specific to this experiment): variance_mean already normalizes
for the (small, replicate-to-replicate) differences in evaluated-cell count
the way evaluate_sample_density_axis.py's own comment explains, and the sum
adds no information for a fixed-n comparison across replicates that this
mean does not already carry more comparably.

NOTE on the variance metric's source array (fact, not interpretation, same
as evaluate_sample_density_axis.py): three of the four methods supply a
NATIVE physical-unit (porosity %^2) predictive variance -- sgs
(``sgs_var_map.npy``), rbf_bootstrap (``bootstrap_var_map.npy``), gp_mle
(``posterior_var_map.npy``). Kriging does NOT: simple kriging runs in
normal-score space, and its physical-unit variance
(``kriging_var_map_physical_mc.npy``) is a MONTE CARLO APPROXIMATION
obtained by back-transforming samples of the normal-score predictive
distribution. This asymmetry is recorded in ``variance_metric_sources.csv``.

IMPORTANT -- the evaluation cell set is NOT identical across replicates
------------------------------------------------------------------------
All four methods exclude the conditioning-sample cells before any metric is
computed (``conditioning_cell_mask``), identically for all 4 methods WITHIN
a replicate, but the excluded set can differ slightly BETWEEN replicates
because n_samples_actual (after grid-cell dedup) is not always exactly 125
and lands on different cells for each sample_seed. The exact evaluated-cell
count per replicate is written to ``evaluation_cell_counts.csv``.

Cross-method fairness / source-run provenance
-----------------------------------------------
Run directories come from
results/processed/sample_replicate_axis/source_runs.json (written by
src.experiments.sample_replicate_axis, which already verified every run
against its own replicate's conditioning samples before writing that file).
This script INDEPENDENTLY re-verifies the same thing here (regenerating each
replicate's conditioning samples and cross-checking every method's recorded
samples.csv against them) before trusting any run -- belt-and-suspenders,
same convention evaluate_sample_density_axis.py follows for the runs it
consumes.

Run with:
.venv/Scripts/python.exe -m src.experiments.evaluate_sample_replicate_axis
"""

import json
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
    VCOL,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.kriging import BACKTR_ZMAX, BACKTR_ZMIN, LTAIL, LTPAR, UTAIL, UTPAR
from src.experiments.sample_replicate_axis import (
    AXIS_HMAJ1,
    AXIS_HMIN1,
    METHODS,
    N_SAMPLES_REQUESTED,
    REPLICATE_IDS,
    REPLICATE_SEED,
    TRUTH_SEED,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"
SOURCE_RUNS_PATH = PROCESSED_DIR / "source_runs.json"

CASE = "sample_replicate_axis"
AXIS = "sample_seed_replicate"

# Metrics emitted by THIS experiment only -- variance_sum, interval widths and
# CRPS are out of scope here (see module docstring).
METRICS = ("mse", "umg", "variance_mean")

# Per-method source array for variance_mean. Kriging's is a Monte Carlo
# back-transform approximation; the other three are native physical units
# (same convention as evaluate_sample_density_axis.py).
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

# Same defensive tolerance convention as every other evaluation script.
VARIANCE_CLIP_TOLERANCE = 1e-6
SAMPLES_MATCH_ATOL = 1e-10


def safe_sqrt_variance(var_map: np.ndarray, name: str) -> np.ndarray:
    """sqrt of a variance map, defensively clipping small-negative
    floating-point noise to 0 while raising on anything larger (a real bug).
    Duplicated (not imported) from evaluate_sample_density_axis.py /
    evaluate_base_case.py, matching this project's per-axis-script
    convention of staying independently runnable."""
    min_val = float(np.min(var_map))
    if min_val < -VARIANCE_CLIP_TOLERANCE:
        raise ValueError(
            f"{name}: variance map has a value ({min_val}) more negative than the "
            f"roundoff tolerance ({-VARIANCE_CLIP_TOLERANCE}) -- this looks like a real "
            "bug, not floating-point noise."
        )
    n_clipped = int(np.sum(var_map < 0))
    if n_clipped > 0:
        print(f"  note: {name}: clipped {n_clipped} small-negative variance value(s) to 0.")
    return np.sqrt(np.clip(var_map, 0.0, None))


def _replicate_mask(replicate_id: str):
    """Regenerate this replicate's truth + conditioning samples and return
    (truth, samples_df, mask). The truth field is IDENTICAL across all 10
    replicates (same TRUTH_SEED, same range) -- only the samples differ."""
    seed = REPLICATE_SEED[replicate_id]
    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)
    samples_df = get_conditioning_samples(truth, sample_seed=seed, n_samples=N_SAMPLES_REQUESTED)
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    return truth, samples_df, mask


def evaluate_one_replicate(replicate_id: str, run_dirs: dict):
    """Compute mse / umg / variance_mean for all 4 methods at one replicate.

    Returns (metrics, diagnostics, variance_sources).
    """
    seed = REPLICATE_SEED[replicate_id]
    truth, samples_df, mask = _replicate_mask(replicate_id)
    n_excluded = int((~mask).sum())
    n_evaluated = int(mask.sum())
    print(
        f"  {replicate_id} (sample_seed={seed}, n_actual={len(samples_df)}): "
        f"conditioning-sample cells excluded: {n_excluded} of {mask.size} "
        f"-> {n_evaluated} evaluated cells"
    )

    # Cross-check every method's own recorded samples.csv against the
    # regenerated conditioning samples -- independent re-verification of what
    # sample_replicate_axis.py already checked once at run time.
    for m in METHODS:
        recorded = pd.read_csv(_REPO_ROOT / run_dirs[m] / "samples.csv")
        if len(recorded) != len(samples_df) or not np.allclose(
            recorded[["X", "Y"]].values, samples_df[["X", "Y"]].values,
            rtol=0.0, atol=SAMPLES_MATCH_ATOL,
        ):
            raise ValueError(
                f"{m}'s recorded samples.csv does NOT match the regenerated conditioning "
                f"samples for {replicate_id} (sample_seed={seed}) -- the identical-sample-"
                "locations-across-methods assumption is violated for this run."
            )

    truth_masked = truth[mask]
    metrics = {m: {} for m in METHODS}
    diagnostics = {
        "replicate_id": replicate_id,
        "sample_seed": seed,
        "n_samples_requested": N_SAMPLES_REQUESTED,
        "n_samples_actual": len(samples_df),
        "n_cells_total": int(mask.size),
        "n_cells_excluded": n_excluded,
        "n_cells_evaluated": n_evaluated,
    }

    # ------------------------------------------------------------------
    # Kriging: MSE from kmap_physical; UMG from kmap_ns + vmap_ns via the
    # exact per-cell quantile back-transform (kriging_fraction_in) -- same
    # path evaluate_sample_density_axis.py / evaluate_base_case.py use.
    # ------------------------------------------------------------------
    kdir = _REPO_ROOT / run_dirs["kriging"]
    kmap_physical = np.load(kdir / "kriging_mean_map_physical.npy")
    kmap_ns = np.load(kdir / "kmap_ns.npy")
    vmap_ns = np.load(kdir / "kriging_var_map_ns.npy")
    transform_table = pd.read_csv(kdir / "nscore_transform_table.csv")
    vr, vrg = transform_table["vr"].values, transform_table["vrg"].values
    std_ns_masked = safe_sqrt_variance(vmap_ns[mask], "kriging vmap_ns")

    metrics["kriging"]["mse"] = mse(truth_masked, kmap_physical[mask])
    kriging_p, kriging_frac_in = kriging_fraction_in(
        truth_masked, kmap_ns[mask], std_ns_masked, vr, vrg,
        BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR,
    )
    metrics["kriging"]["umg"] = calc_umg(kriging_p, kriging_frac_in)

    # ------------------------------------------------------------------
    # The three physical-unit Gaussian-predictive methods.
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

    # ------------------------------------------------------------------
    # Predictive-variance magnitude (mean only -- see module docstring).
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
        safe_sqrt_variance(var_masked, f"{method} variance map (variance_mean)")
        metrics[method]["variance_mean"] = float(np.mean(var_masked))
        variance_sources.append(
            {
                "replicate_id": replicate_id,
                "method": method,
                "variance_source_file": fname,
                "units": "porosity %^2",
                "is_monte_carlo_backtransform_approximation": VARIANCE_SOURCE_IS_MC_BACKTRANSFORM[
                    method
                ],
                "n_cells_averaged": n_evaluated,
                "variance_mean": metrics[method]["variance_mean"],
            }
        )

    return metrics, diagnostics, variance_sources


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_RUNS_PATH, "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    if set(source_runs) != set(REPLICATE_IDS):
        raise ValueError(
            f"{SOURCE_RUNS_PATH} has replicate ids {sorted(source_runs)}; expected "
            f"{sorted(REPLICATE_IDS)}."
        )

    rows = []
    diagnostics_rows = []
    variance_source_rows = []

    for replicate_id in REPLICATE_IDS:
        print(f"\nEvaluating {replicate_id} (sample_seed={REPLICATE_SEED[replicate_id]})...")
        metrics, diagnostics, variance_sources = evaluate_one_replicate(
            replicate_id, source_runs[replicate_id]
        )
        variance_source_rows.extend(variance_sources)
        diagnostics_rows.append(diagnostics)
        for method in METHODS:
            for metric_name, value in metrics[method].items():
                rows.append(
                    {
                        "case": CASE,
                        "axis": AXIS,
                        "axis_level": replicate_id,
                        "method": method,
                        "metric": metric_name,
                        "value": value,
                    }
                )

    metrics_df = pd.DataFrame(rows)
    if metrics_df["value"].isna().any() or not np.all(np.isfinite(metrics_df["value"].values)):
        raise ValueError("metrics_df contains NaN/inf values -- see printed metrics above.")

    expected_n_rows = len(REPLICATE_IDS) * len(METHODS) * len(METRICS)
    if len(metrics_df) != expected_n_rows:
        raise ValueError(
            f"metrics_df has {len(metrics_df)} rows; expected {expected_n_rows} "
            f"({len(REPLICATE_IDS)} replicates x {len(METHODS)} methods x "
            f"{len(METRICS)} metrics: {', '.join(METRICS)})."
        )
    if set(metrics_df["metric"]) != set(METRICS):
        raise ValueError(
            f"metrics_df holds metrics {sorted(set(metrics_df['metric']))}; expected "
            f"{sorted(METRICS)}."
        )

    # Reading order: rep0 .. rep9.
    metrics_df["_rep_numeric"] = (
        metrics_df["axis_level"].str.replace("rep", "", regex=False).astype(int)
    )
    metrics_df = (
        metrics_df.sort_values(["_rep_numeric", "method", "metric"])
        .drop(columns="_rep_numeric")
        .reset_index(drop=True)
    )

    metrics_csv = PROCESSED_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_csv = PROCESSED_DIR / "evaluation_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    cell_counts = diagnostics_df[
        [
            "replicate_id", "sample_seed", "n_samples_requested", "n_samples_actual",
            "n_cells_total", "n_cells_excluded", "n_cells_evaluated",
        ]
    ].copy()
    cell_counts["note"] = (
        "Conditioning-sample cells are excluded from evaluation for all 4 methods "
        "identically WITHIN a replicate, but the excluded set can differ slightly BETWEEN "
        "replicates (different sample_seed -> different cells, and n_samples_actual is not "
        "always exactly 125 after grid-cell dedup)."
    )
    cell_counts_csv = PROCESSED_DIR / "evaluation_cell_counts.csv"
    cell_counts.to_csv(cell_counts_csv, index=False)

    variance_sources_df = pd.DataFrame(variance_source_rows)[
        [
            "replicate_id", "method", "variance_source_file", "units",
            "is_monte_carlo_backtransform_approximation", "n_cells_averaged", "variance_mean",
        ]
    ]
    variance_sources_df["note"] = (
        "variance_mean is taken over this replicate's evaluated cells (its conditioning "
        "cells excluded). All four arrays are porosity %^2, but kriging's physical-unit "
        "variance is a Monte Carlo back-transform of its normal-score variance, whereas "
        "sgs/rbf_bootstrap/gp_mle variances are native physical-unit arrays."
    )
    variance_sources_csv = PROCESSED_DIR / "variance_metric_sources.csv"
    variance_sources_df.to_csv(variance_sources_csv, index=False)

    # --- Aggregated summary across the 10 replicates -----------------------
    summary_rows = []
    for method in METHODS:
        for metric_name in METRICS:
            vals = metrics_df.loc[
                (metrics_df["method"] == method) & (metrics_df["metric"] == metric_name),
                "value",
            ].values
            if len(vals) != len(REPLICATE_IDS):
                raise ValueError(
                    f"{method}/{metric_name}: expected {len(REPLICATE_IDS)} replicate "
                    f"values, found {len(vals)}."
                )
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric_name,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n_replicates": len(vals),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = PROCESSED_DIR / "metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)

    print("\nSample-replicate axis metrics (10 replicates x 4 methods x 3 metrics):")
    print(metrics_df.to_string(index=False))
    print("\nAggregated across the 10 replicates (mean/std/min/max per method x metric):")
    print(summary_df.to_string(index=False))
    print("\nEvaluated-cell counts per replicate:")
    print(cell_counts.drop(columns="note").to_string(index=False))
    print("\nPredictive-variance metric sources:")
    print(variance_sources_df.drop(columns="note").to_string(index=False))
    print(f"\nmetrics.csv: {metrics_csv}")
    print(f"metrics_summary.csv: {summary_csv}")
    print(f"evaluation_cell_counts.csv: {cell_counts_csv}")
    print(f"variance_metric_sources.csv: {variance_sources_csv}")
    print(f"evaluation_diagnostics.csv: {diagnostics_csv}")

    return metrics_df, summary_df


if __name__ == "__main__":
    main()
