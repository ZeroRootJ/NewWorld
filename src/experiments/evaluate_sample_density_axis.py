"""Evaluate the sample-density-axis 4-method comparison (kriging / SGS /
RBF+bootstrap / GP-MLE) over the 3 axis levels 5% / 2% / 1% of the 2500-cell
grid (n_samples = 125 / 50 / 25).

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

Parked (NOT emitted while ``EMIT_SHARPNESS_METRICS`` is False; see that flag):
``interval_width_mean_nominal``, ``interval_width_p95``, ``crps``.

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
level. ``variance_sum``/``variance_mean`` (and, when re-enabled,
``interval_width_*``/``crps``) ARE computed here for all three levels, since
those metrics did not exist when the base case was evaluated.

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

import geostatspy.geostats as geostats

from src.evaluation import (
    CRPS_DEFAULT_N_TAU,
    INTERVAL_WIDTH_HEADLINE_P,
    accuracy_plot_fraction_in,
    calc_umg,
    conditioning_cell_mask,
    crps_gaussian_analytic,
    gaussian_crps,
    gaussian_interval_widths,
    kriging_crps,
    kriging_fraction_in,
    kriging_interval_widths,
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
    backtr_value_vectorized,
)
from src.experiments.sample_density_axis import (
    ALL_AXIS_LEVELS,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    BASE_CASE_AXIS_LEVEL,
    METHODS,
    SAMPLE_COUNTS,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_density_axis"
BASE_CASE_PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "base_case"
SOURCE_RUNS_PATH = PROCESSED_DIR / "source_runs.json"

CASE = "sample_density_axis"
AXIS = "sample_fraction_pct"

# ---------------------------------------------------------------------------
# PARKED-METRICS FLAG (user decision, 2026-09-15)
# ---------------------------------------------------------------------------
# The sharpness-family metrics -- interval_width_mean_nominal,
# interval_width_p95 and crps -- were judged not to express what this study
# currently needs to show, so they were taken OUT of the active
# sample-density-axis workflow and moved to the TODO pile. This flag is the
# single switch that implements that decision.
#
#   (a) Decision: user, 2026-09-15. It applies to the SAMPLE-DENSITY AXIS
#       ONLY -- the range axis (src/experiments/evaluate_range_axis.py) still
#       reports these metrics and was deliberately left untouched.
#   (b) Nothing here is known to be wrong. The metric DEFINITIONS and their
#       implementations in src/evaluation.py (gaussian_interval_widths,
#       kriging_interval_widths, gaussian_crps, kriging_crps,
#       crps_gaussian_analytic) are UNCHANGED, still exercised by their tests,
#       and still used by the range axis. The values computed before parking
#       are preserved, not deleted, in
#       results/processed/sample_density_axis/metrics_parked_sharpness.csv
#       (same tidy schema), alongside the diagnostics that support them
#       (crps_convergence.csv, kriging_backtransform_tail_sensitivity.csv).
#   (c) Restoring is a one-line change: set this to True and re-run this
#       script. Every code path below is kept intact behind the flag, so the
#       parked rows reappear in metrics.csv immediately with no other edit.
EMIT_SHARPNESS_METRICS = False

# Metrics always emitted, and the parked ones emitted only when the flag above
# is True. Used for the row-count self-check in main().
CORE_METRICS = ("mse", "umg", "variance_sum", "variance_mean")
SHARPNESS_METRICS = ("interval_width_mean_nominal", "interval_width_p95", "crps")

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

# n_tau values used for the CRPS quadrature convergence check. First entry is
# the production value.
CRPS_CONVERGENCE_N_TAUS = (CRPS_DEFAULT_N_TAU, 99, 499, 999)

BACKTR_VALIDATION_SEED = 86
N_BACKTR_VALIDATION_SAMPLES = 500

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


def validate_backtr_vectorized(kmap_ns: np.ndarray, vr, vrg) -> float:
    """Check backtr_value_vectorized against the scalar geostats.backtr_value
    on values drawn from this run's own kmap_ns. Raises on mismatch."""
    rng = np.random.default_rng(BACKTR_VALIDATION_SEED)
    vals = rng.choice(kmap_ns.ravel(), size=N_BACKTR_VALIDATION_SAMPLES, replace=True)
    reference = np.array(
        [
            geostats.backtr_value(
                v, vr, vrg, zmin=BACKTR_ZMIN, zmax=BACKTR_ZMAX,
                ltail=LTAIL, ltpar=LTPAR, utail=UTAIL, utpar=UTPAR,
            )
            for v in vals
        ]
    )
    vectorized = backtr_value_vectorized(
        vals, vr, vrg, BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR
    )
    max_abs_diff = float(np.max(np.abs(reference - vectorized)))
    if not np.allclose(reference, vectorized, rtol=1e-8, atol=1e-8):
        raise RuntimeError(
            "backtr_value_vectorized disagrees with the scalar reference "
            f"geostats.backtr_value (max abs diff = {max_abs_diff}) -- refusing to "
            "compute kriging interval width / CRPS through an unvalidated "
            "back-transform."
        )
    return max_abs_diff


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
    # Kriging: MSE from kmap_physical; every uncertainty metric from
    # kmap_ns + vmap_ns via exact quantile back-transform.
    # ------------------------------------------------------------------
    kdir = _REPO_ROOT / run_dirs["kriging"]
    kmap_physical = np.load(kdir / "kriging_mean_map_physical.npy")
    kmap_ns = np.load(kdir / "kmap_ns.npy")
    vmap_ns = np.load(kdir / "kriging_var_map_ns.npy")
    transform_table = pd.read_csv(kdir / "nscore_transform_table.csv")
    vr, vrg = transform_table["vr"].values, transform_table["vrg"].values

    # Kept running even while EMIT_SHARPNESS_METRICS is False: it validates the
    # back-transform helper the parked metrics depend on, so re-enabling the
    # flag never re-enables an unvalidated path.
    diagnostics["backtr_vectorized_max_abs_diff_vs_reference"] = validate_backtr_vectorized(
        kmap_ns, vr, vrg
    )
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

    if EMIT_SHARPNESS_METRICS:
        k_backtr_args = (vr, vrg, BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR,
                         backtr_value_vectorized)
        k_widths = kriging_interval_widths(kmap_ns[mask], std_ns_masked, *k_backtr_args)
        metrics["kriging"]["interval_width_mean_nominal"] = float(np.mean(k_widths))
        metrics["kriging"]["interval_width_p95"] = float(
            kriging_interval_widths(
                kmap_ns[mask], std_ns_masked, *k_backtr_args,
                p_levels=np.array([INTERVAL_WIDTH_HEADLINE_P]),
            )[0]
        )
        metrics["kriging"]["crps"] = kriging_crps(
            truth_masked, kmap_ns[mask], std_ns_masked, *k_backtr_args
        )

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

        if EMIT_SHARPNESS_METRICS:
            widths = gaussian_interval_widths(mean_masked, std_masked)
            metrics[method]["interval_width_mean_nominal"] = float(np.mean(widths))
            metrics[method]["interval_width_p95"] = float(
                gaussian_interval_widths(
                    mean_masked, std_masked, p_levels=np.array([INTERVAL_WIDTH_HEADLINE_P])
                )[0]
            )
            metrics[method]["crps"] = gaussian_crps(truth_masked, mean_masked, std_masked)

            exact = crps_gaussian_analytic(truth_masked, mean_masked, std_masked)
            diagnostics[f"crps_vs_analytic_rel_err_{method}"] = abs(
                metrics[method]["crps"] - exact
            ) / abs(exact)

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


def crps_convergence_check(axis_level: str, run_dirs: dict) -> pd.DataFrame:
    """Recompute CRPS at several quadrature resolutions n_tau for all 4
    methods, so the reported production value is demonstrably converged
    rather than assumed to be."""
    truth, _, mask = _level_mask(axis_level)
    truth_masked = truth[mask]

    kdir = _REPO_ROOT / run_dirs["kriging"]
    kmap_ns = np.load(kdir / "kmap_ns.npy")
    vmap_ns = np.load(kdir / "kriging_var_map_ns.npy")
    tt = pd.read_csv(kdir / "nscore_transform_table.csv")
    vr, vrg = tt["vr"].values, tt["vrg"].values
    std_ns_masked = safe_sqrt_variance(vmap_ns[mask], "kriging vmap_ns")
    k_backtr_args = (vr, vrg, BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR,
                     backtr_value_vectorized)

    sdir = _REPO_ROOT / run_dirs["sgs"]
    rdir = _REPO_ROOT / run_dirs["rbf_bootstrap"]
    gdir = _REPO_ROOT / run_dirs["gp_mle"]
    gaussian_specs = {
        "sgs": (np.load(sdir / "sgs_mean_map.npy"), np.load(sdir / "sgs_var_map.npy")),
        "rbf_bootstrap": (
            np.load(rdir / "bootstrap_mean_map.npy"),
            np.load(rdir / "bootstrap_var_map.npy"),
        ),
        "gp_mle": (
            np.load(gdir / "posterior_mean_map.npy"),
            np.load(gdir / "posterior_var_map.npy"),
        ),
    }

    rows = []
    for n_tau in CRPS_CONVERGENCE_N_TAUS:
        rows.append(
            {
                "axis_level": axis_level,
                "method": "kriging",
                "n_tau": n_tau,
                "crps": kriging_crps(
                    truth_masked, kmap_ns[mask], std_ns_masked, *k_backtr_args, n_tau=n_tau
                ),
                "crps_analytic_gaussian": np.nan,  # no closed form after back-transform
            }
        )
        for method, (mean_map, var_map) in gaussian_specs.items():
            std_masked = safe_sqrt_variance(var_map[mask], f"{method} variance map")
            rows.append(
                {
                    "axis_level": axis_level,
                    "method": method,
                    "n_tau": n_tau,
                    "crps": gaussian_crps(truth_masked, mean_map[mask], std_masked, n_tau=n_tau),
                    "crps_analytic_gaussian": crps_gaussian_analytic(
                        truth_masked, mean_map[mask], std_masked
                    ),
                }
            )
    return pd.DataFrame(rows)


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
    also_computed = "variance_sum/variance_mean" + (
        " and interval-width/CRPS" if EMIT_SHARPNESS_METRICS else ""
    )
    print(
        f"\n{BASE_CASE_AXIS_LEVEL}% MSE/UMG + accuracy-plot curve sourced from "
        f"{BASE_CASE_PROCESSED_DIR} (re-labeled, not recomputed); its "
        f"{also_computed} values ARE computed here."
    )


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_RUNS_PATH, "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    if set(source_runs) != set(ALL_AXIS_LEVELS):
        raise ValueError(
            f"{SOURCE_RUNS_PATH} has axis levels {sorted(source_runs)}; expected "
            f"{sorted(ALL_AXIS_LEVELS)}."
        )

    if not EMIT_SHARPNESS_METRICS:
        print(
            "NOTE: EMIT_SHARPNESS_METRICS is False (user decision 2026-09-15) -- "
            f"{', '.join(SHARPNESS_METRICS)} are NOT written to metrics.csv. Their "
            "last computed values are preserved in metrics_parked_sharpness.csv and "
            "their implementations in src/evaluation.py are unchanged."
        )

    rows = []
    curve_rows = []
    diagnostics_rows = []
    variance_source_rows = []

    for axis_level in ALL_AXIS_LEVELS:
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

    expected_metrics = CORE_METRICS + (SHARPNESS_METRICS if EMIT_SHARPNESS_METRICS else ())
    expected_n_rows = len(ALL_AXIS_LEVELS) * len(METHODS) * len(expected_metrics)
    if len(metrics_df) != expected_n_rows:
        raise ValueError(
            f"metrics_df has {len(metrics_df)} rows; expected {expected_n_rows} "
            f"({len(ALL_AXIS_LEVELS)} levels x {len(METHODS)} methods x "
            f"{len(expected_metrics)} metrics: {', '.join(expected_metrics)})."
        )
    if set(metrics_df["metric"]) != set(expected_metrics):
        raise ValueError(
            f"metrics_df holds metrics {sorted(set(metrics_df['metric']))}; expected "
            f"{sorted(expected_metrics)} (EMIT_SHARPNESS_METRICS="
            f"{EMIT_SHARPNESS_METRICS})."
        )

    # Sort by DECREASING sample fraction (5 -> 2 -> 1), the axis's natural
    # reading order.
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

    # --- CRPS quadrature convergence check --------------------------------
    # Only re-run while the parked CRPS metric is enabled. With
    # EMIT_SHARPNESS_METRICS = False the existing crps_convergence.csv (which
    # documents the parked values) is deliberately LEFT IN PLACE UNTOUCHED
    # rather than deleted or overwritten.
    convergence_df = None
    convergence_csv = PROCESSED_DIR / "crps_convergence.csv"
    if EMIT_SHARPNESS_METRICS:
        # Run on the two extreme levels (densest / sparsest).
        conv_frames = [
            crps_convergence_check(lvl, source_runs[lvl])
            for lvl in (ALL_AXIS_LEVELS[0], ALL_AXIS_LEVELS[-1])
        ]
        convergence_df = pd.concat(conv_frames, ignore_index=True)
        convergence_df.to_csv(convergence_csv, index=False)
    else:
        print(
            f"\nCRPS quadrature convergence check skipped (EMIT_SHARPNESS_METRICS=False); "
            f"the existing {convergence_csv.name} is left untouched."
        )

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

    print("\nSample-density axis metrics (wide view; 3 levels x 4 methods):")
    print(wide.to_string(index=False))
    print("\nEvaluated-cell counts per level (levels are NOT scored on the same cells):")
    print(cell_counts.drop(columns="note").to_string(index=False))
    print(
        "\nPredictive-variance metric sources (kriging = Monte Carlo back-transform "
        "approximation; others native physical units):"
    )
    print(variance_sources_df.drop(columns="note").to_string(index=False))
    if convergence_df is not None:
        print(
            "\nCRPS quadrature convergence (n_tau refinement; analytic reference where "
            "it exists):"
        )
        print(convergence_df.to_string(index=False))
    print("\nEvaluation diagnostics:")
    print(diagnostics_df.to_string(index=False))
    print(f"\nmetrics.csv: {metrics_csv}")
    print(f"metrics_wide.csv: {wide_csv}")
    print(f"accuracy_plot_curves.csv: {curves_csv}")
    print(f"evaluation_cell_counts.csv: {cell_counts_csv}")
    print(f"variance_metric_sources.csv: {variance_sources_csv}")
    if convergence_df is not None:
        print(f"crps_convergence.csv: {convergence_csv}")
    print(f"evaluation_diagnostics.csv: {diagnostics_csv}")

    return metrics_df, curves_df, convergence_df


if __name__ == "__main__":
    main()
