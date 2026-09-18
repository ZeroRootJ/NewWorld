"""Evaluate the sample-seed replicate axis (src/experiments/
sample_replicate_axis.py), NOW EXTENDED (2026-09-18) to all 3 sample-density
axis levels: 10 sample-location draws (sample_seed in {1001, ..., 1010},
reused UNCHANGED at every level) x 3 density levels ("5"/"2"/"1" ->
n_samples_requested = 125/50/25) x 4 methods (kriging / SGS / RBF+bootstrap /
GP-MLE) = 120 (level, replicate, method) cells.

Purpose: quantify how much each method's accuracy/calibration/predictive-
variance moves from sample PLACEMENT alone, with sample count and ground
truth held fixed WITHIN a level -- a different question from the
sample-density axis (src/experiments/evaluate_sample_density_axis.py), which
varies n_samples at ONE fixed sample_seed per level and reports levels that
are NOT scored on the same cell set. This module answers that same
placement-only question at all 3 of that axis's density levels, using the
SAME 10 replicate seeds at each.

Metrics -- reported SEPARATELY (never combined into one number), reusing
src/evaluation.py's functions exactly as evaluate_sample_density_axis.py
does, with the SAME per-method source-array convention:

  accuracy
    - ``mse``           (porosity %^2, lower better)
  uncertainty quality
    - ``umg``            (coverage goodness, 1.0 = ideal)
  predictive-variance magnitude
    - ``variance_mean``  (porosity %^2, the predictive variance averaged
                          over that (level, replicate)'s evaluated cells)

``variance_sum`` is deliberately NOT computed here (user decision,
2026-09-18, unchanged by this extension): variance_mean already normalizes
for the (small, replicate-to-replicate AND level-to-level) differences in
evaluated-cell count the way evaluate_sample_density_axis.py's own comment
explains, and the sum adds no information this mean does not already carry
more comparably.

Tidy output: metrics.csv has columns (case, axis, axis_level, replicate,
method, metric, value) -- 3 levels x 10 replicates x 4 methods x 3 metrics
= 360 rows. ``axis_level`` uses the SAME "5"/"2"/"1" labels/semantics as
sample_density_axis.ALL_AXIS_LEVELS (so this table can be joined against
sample_density_axis/metrics.csv's single-realization values), and
``replicate`` is the "rep0".."rep9" identifier (the SAME representation used
throughout sample_replicate_axis.py) -- kept as its OWN column (not folded
into axis_level) so a reader can group by density level and by replicate
independently.

NOTE on the variance metric's source array (fact, not interpretation, same
as evaluate_sample_density_axis.py): three of the four methods supply a
NATIVE physical-unit (porosity %^2) predictive variance -- sgs
(``sgs_var_map.npy``), rbf_bootstrap (``bootstrap_var_map.npy``), gp_mle
(``posterior_var_map.npy``). Kriging does NOT: simple kriging runs in
normal-score space, and its physical-unit variance
(``kriging_var_map_physical_mc.npy``) is a MONTE CARLO APPROXIMATION
obtained by back-transforming samples of the normal-score predictive
distribution. This asymmetry is recorded in ``variance_metric_sources.csv``.

IMPORTANT -- the evaluation cell set is NOT identical across (level,
replicate) cells
------------------------------------------------------------------------
All four methods exclude the conditioning-sample cells before any metric is
computed (``conditioning_cell_mask``), identically for all 4 methods WITHIN
a (level, replicate) cell, but the excluded set differs BETWEEN replicates
(different sample_seed -> different cells) AND BETWEEN levels (different
n_samples_requested). The exact evaluated-cell count per (level, replicate)
is written to ``evaluation_cell_counts.csv``.

Per-method structural "length scale" (NOT part of the main tidy table)
-------------------------------------------------------------------------
GP-MLE's MLE-fitted RBF length_scale (converted to a practical range at the
same 0.05-correlation-cutoff convention) and RBF+bootstrap's CV-selected
epsilon (converted to an equivalent 0.05-cutoff distance) are BOTH
replicate-varying quantities -- unlike kriging/SGS, whose variogram range is
a FIXED INPUT CONSTANT (300 m) at every level and every replicate, not
fitted from the conditioning data at all. Putting a per-replicate "length"
number in the SAME tidy metrics.csv for kriging/SGS would misrepresent a
structural constant as if it varied per replicate (it does not -- there is
nothing to average over 10 replicates), so these two methods are
deliberately given NO length-scale rows anywhere, and this fact is recorded
in ``length_scale_by_replicate.csv``'s own header note rather than
represented by a fabricated/NaN row. GP-MLE and RBF+bootstrap's per-replicate
length values are written to a SEPARATE small table,
``length_scale_by_replicate.csv`` (columns: axis_level, replicate, method,
length_scale_m, length_definition, source), using the EXACT SAME conversion
formulas results/processed/sample_density_axis/
make_length_and_variogram_figures.py already uses (not reimplemented
differently):
  - GP-MLE: practical_range_m = sqrt(-2*ln(0.05)) * length_scale_m (factor
    ~= 2.4477), reading length_scale_m from the run's own manifest
    (params.fitted_hyperparameters.length_scale_m).
  - RBF+bootstrap: r_0.05 = sqrt(-ln(0.05)) / epsilon, reading epsilon from
    the run's own manifest (params.best_epsilon), only valid for
    params.rbf_kernel == "gaussian" (checked, not assumed).
Their (axis_level, method) aggregates (mean/std/min/max across the 10
replicates) are appended as EXTRA rows to metrics_summary.csv (metric names
"length_scale_practical_range_m" for gp_mle, "length_scale_rbf_converted_m"
for rbf_bootstrap) -- on top of, not replacing, the 3-levels x 4-methods x
3-metrics = 36 core summary rows.

Cross-method fairness / source-run provenance
-----------------------------------------------
Run directories come from
results/processed/sample_replicate_axis/source_runs.json (written by
src.experiments.sample_replicate_axis, which already verified EVERY run --
both the reused level-5 runs and the fresh level-2/1 runs -- against its own
(level, replicate)'s conditioning samples before writing that file). This
script INDEPENDENTLY re-verifies the same thing here (regenerating each
(level, replicate)'s conditioning samples and cross-checking every method's
recorded samples.csv against them) before trusting any run -- belt-and-
suspenders, applied identically to all 120 (level, replicate, method) cells,
REUSED and FRESH alike (the project's standing convention: never silently
trust a reused run).

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
from src.experiments.sample_density_axis import ALL_AXIS_LEVELS
from src.experiments.sample_replicate_axis import (
    AXIS_HMAJ1,
    AXIS_HMIN1,
    METHODS,
    N_SAMPLES_REQUESTED_BY_LEVEL,
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

# Same 0.05-correlation-cutoff convention used by
# results/processed/sample_density_axis/make_length_and_variogram_figures.py
# (and src/experiments/diagnose_sample_density_axis.py for GP-MLE) -- reused
# verbatim, not redefined differently, for both GP-MLE and RBF+bootstrap's
# length-scale conversions.
CORRELATION_CUTOFF = 0.05
GP_PRACTICAL_RANGE_FACTOR = float(np.sqrt(-2.0 * np.log(CORRELATION_CUTOFF)))  # ~=2.4477

# Methods with NO per-replicate length-scale concept: kriging/SGS's
# variogram range is a fixed INPUT constant (300 m) at every level and every
# replicate, never fitted from the conditioning data. Deliberately excluded
# from length_scale_by_replicate.csv -- see module docstring.
LENGTH_SCALE_METHODS = ("gp_mle", "rbf_bootstrap")
STRUCTURALLY_FIXED_METHODS = ("kriging", "sgs")


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


def _manifest_params(rel_run_dir: str) -> dict:
    return json.loads(
        (_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8")
    )["params"]


def _level_replicate_mask(axis_level: str, replicate_id: str):
    """Regenerate this (level, replicate)'s truth + conditioning samples and
    return (truth, samples_df, mask). The truth field is IDENTICAL across all
    3 levels x 10 replicates (same TRUTH_SEED, same range) -- only the
    samples differ."""
    seed = REPLICATE_SEED[replicate_id]
    n = N_SAMPLES_REQUESTED_BY_LEVEL[axis_level]
    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)
    samples_df = get_conditioning_samples(truth, sample_seed=seed, n_samples=n)
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    return truth, samples_df, mask


def length_scale_rows_for_cell(axis_level: str, replicate_id: str, run_dirs: dict) -> list:
    """GP-MLE / RBF+bootstrap length-scale rows for one (level, replicate)
    cell, using the exact conversion formulas
    make_length_and_variogram_figures.py already uses (see module
    docstring). Kriging/SGS are NOT represented here (see
    STRUCTURALLY_FIXED_METHODS note)."""
    rows = []

    gp_params = _manifest_params(run_dirs["gp_mle"])
    ell = float(gp_params["fitted_hyperparameters"]["length_scale_m"])
    rows.append(
        {
            "axis_level": axis_level,
            "replicate": replicate_id,
            "method": "gp_mle",
            "length_scale_m": GP_PRACTICAL_RANGE_FACTOR * ell,
            "length_definition": (
                f"MLE-fitted sklearn RBF kernel length_scale converted to a practical range "
                f"at the {CORRELATION_CUTOFF} correlation cutoff "
                f"(factor=sqrt(2*ln(20))~={GP_PRACTICAL_RANGE_FACTOR:.4f})"
            ),
            "source": f"{run_dirs['gp_mle']}/manifest.json:params.fitted_hyperparameters.length_scale_m",
        }
    )

    rbf_params = _manifest_params(run_dirs["rbf_bootstrap"])
    kernel = rbf_params["rbf_kernel"]
    if kernel != "gaussian":
        raise ValueError(
            f"level '{axis_level}' {replicate_id}: rbf_bootstrap manifest records "
            f"rbf_kernel='{kernel}', not 'gaussian' -- the phi(r)=exp(-(epsilon*r)^2) "
            "conversion used here does not apply to a different kernel family."
        )
    epsilon = float(rbf_params["best_epsilon"])
    r_cutoff = float(np.sqrt(-np.log(CORRELATION_CUTOFF)) / epsilon)
    rows.append(
        {
            "axis_level": axis_level,
            "replicate": replicate_id,
            "method": "rbf_bootstrap",
            "length_scale_m": r_cutoff,
            "length_definition": (
                f"CV-tuned gaussian RBF interpolation kernel shape parameter epsilon "
                f"converted to an equivalent {CORRELATION_CUTOFF}-cutoff distance "
                f"(r=sqrt(-ln({CORRELATION_CUTOFF}))/epsilon); NOT a fitted spatial "
                "correlation length -- see module docstring"
            ),
            "source": f"{run_dirs['rbf_bootstrap']}/manifest.json:params.best_epsilon",
        }
    )
    return rows


def evaluate_one_cell(axis_level: str, replicate_id: str, run_dirs: dict):
    """Compute mse / umg / variance_mean for all 4 methods at one (level,
    replicate) cell.

    Returns (metrics, diagnostics, variance_sources).
    """
    seed = REPLICATE_SEED[replicate_id]
    truth, samples_df, mask = _level_replicate_mask(axis_level, replicate_id)
    n_excluded = int((~mask).sum())
    n_evaluated = int(mask.sum())
    print(
        f"  level '{axis_level}' {replicate_id} (sample_seed={seed}, n_actual="
        f"{len(samples_df)}): conditioning-sample cells excluded: {n_excluded} of "
        f"{mask.size} -> {n_evaluated} evaluated cells"
    )

    # Cross-check every method's own recorded samples.csv against the
    # regenerated conditioning samples -- independent re-verification of what
    # sample_replicate_axis.py already checked once at run time (applied here
    # identically to REUSED and FRESH runs).
    for m in METHODS:
        recorded = pd.read_csv(_REPO_ROOT / run_dirs[m] / "samples.csv")
        if len(recorded) != len(samples_df) or not np.allclose(
            recorded[["X", "Y"]].values, samples_df[["X", "Y"]].values,
            rtol=0.0, atol=SAMPLES_MATCH_ATOL,
        ):
            raise ValueError(
                f"{m}'s recorded samples.csv does NOT match the regenerated conditioning "
                f"samples for level '{axis_level}' {replicate_id} (sample_seed={seed}) -- "
                "the identical-sample-locations-across-methods assumption is violated for "
                "this run."
            )

    truth_masked = truth[mask]
    metrics = {m: {} for m in METHODS}
    diagnostics = {
        "axis_level": axis_level,
        "replicate": replicate_id,
        "sample_seed": seed,
        "n_samples_requested": N_SAMPLES_REQUESTED_BY_LEVEL[axis_level],
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
                "axis_level": axis_level,
                "replicate": replicate_id,
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

    if set(source_runs) != set(ALL_AXIS_LEVELS):
        raise ValueError(
            f"{SOURCE_RUNS_PATH} has axis levels {sorted(source_runs)}; expected "
            f"{sorted(ALL_AXIS_LEVELS)}."
        )
    for lvl in ALL_AXIS_LEVELS:
        if set(source_runs[lvl]) != set(REPLICATE_IDS):
            raise ValueError(
                f"{SOURCE_RUNS_PATH} level '{lvl}' has replicate ids "
                f"{sorted(source_runs[lvl])}; expected {sorted(REPLICATE_IDS)}."
            )

    rows = []
    diagnostics_rows = []
    variance_source_rows = []
    length_scale_rows = []

    for axis_level in ALL_AXIS_LEVELS:
        for replicate_id in REPLICATE_IDS:
            print(
                f"\nEvaluating level '{axis_level}' {replicate_id} "
                f"(sample_seed={REPLICATE_SEED[replicate_id]})..."
            )
            run_dirs = source_runs[axis_level][replicate_id]
            metrics, diagnostics, variance_sources = evaluate_one_cell(
                axis_level, replicate_id, run_dirs
            )
            variance_source_rows.extend(variance_sources)
            diagnostics_rows.append(diagnostics)
            length_scale_rows.extend(
                length_scale_rows_for_cell(axis_level, replicate_id, run_dirs)
            )
            for method in METHODS:
                for metric_name, value in metrics[method].items():
                    rows.append(
                        {
                            "case": CASE,
                            "axis": AXIS,
                            "axis_level": axis_level,
                            "replicate": replicate_id,
                            "method": method,
                            "metric": metric_name,
                            "value": value,
                        }
                    )

    metrics_df = pd.DataFrame(rows)
    if metrics_df["value"].isna().any() or not np.all(np.isfinite(metrics_df["value"].values)):
        raise ValueError("metrics_df contains NaN/inf values -- see printed metrics above.")

    expected_n_rows = len(ALL_AXIS_LEVELS) * len(REPLICATE_IDS) * len(METHODS) * len(METRICS)
    if len(metrics_df) != expected_n_rows:
        raise ValueError(
            f"metrics_df has {len(metrics_df)} rows; expected {expected_n_rows} "
            f"({len(ALL_AXIS_LEVELS)} levels x {len(REPLICATE_IDS)} replicates x "
            f"{len(METHODS)} methods x {len(METRICS)} metrics: {', '.join(METRICS)})."
        )
    if set(metrics_df["metric"]) != set(METRICS):
        raise ValueError(
            f"metrics_df holds metrics {sorted(set(metrics_df['metric']))}; expected "
            f"{sorted(METRICS)}."
        )

    # Reading order: level 5 -> 2 -> 1 (dense -> sparse), rep0 .. rep9 within
    # a level.
    metrics_df["_axis_level_numeric"] = metrics_df["axis_level"].astype(float)
    metrics_df["_rep_numeric"] = (
        metrics_df["replicate"].str.replace("rep", "", regex=False).astype(int)
    )
    metrics_df = (
        metrics_df.sort_values(
            ["_axis_level_numeric", "_rep_numeric", "method", "metric"],
            ascending=[False, True, True, True],
        )
        .drop(columns=["_axis_level_numeric", "_rep_numeric"])
        .reset_index(drop=True)
    )

    metrics_csv = PROCESSED_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_csv = PROCESSED_DIR / "evaluation_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    cell_counts = diagnostics_df[
        [
            "axis_level", "replicate", "sample_seed", "n_samples_requested",
            "n_samples_actual", "n_cells_total", "n_cells_excluded", "n_cells_evaluated",
        ]
    ].copy()
    cell_counts["note"] = (
        "Conditioning-sample cells are excluded from evaluation for all 4 methods "
        "identically WITHIN a (level, replicate) cell, but the excluded set differs "
        "BETWEEN replicates (different sample_seed -> different cells) AND BETWEEN levels "
        "(different n_samples_requested)."
    )
    cell_counts_csv = PROCESSED_DIR / "evaluation_cell_counts.csv"
    cell_counts.to_csv(cell_counts_csv, index=False)

    variance_sources_df = pd.DataFrame(variance_source_rows)[
        [
            "axis_level", "replicate", "method", "variance_source_file", "units",
            "is_monte_carlo_backtransform_approximation", "n_cells_averaged", "variance_mean",
        ]
    ]
    variance_sources_df["note"] = (
        "variance_mean is taken over this (level, replicate) cell's evaluated cells (its "
        "conditioning cells excluded). All four arrays are porosity %^2, but kriging's "
        "physical-unit variance is a Monte Carlo back-transform of its normal-score "
        "variance, whereas sgs/rbf_bootstrap/gp_mle variances are native physical-unit "
        "arrays."
    )
    variance_sources_csv = PROCESSED_DIR / "variance_metric_sources.csv"
    variance_sources_df.to_csv(variance_sources_csv, index=False)

    # --- Length-scale-by-replicate (GP-MLE / RBF+bootstrap only) -----------
    length_scale_df = pd.DataFrame(length_scale_rows)[
        ["axis_level", "replicate", "method", "length_scale_m", "length_definition", "source"]
    ]
    expected_length_rows = len(ALL_AXIS_LEVELS) * len(REPLICATE_IDS) * len(LENGTH_SCALE_METHODS)
    if len(length_scale_df) != expected_length_rows:
        raise ValueError(
            f"length_scale_df has {len(length_scale_df)} rows; expected "
            f"{expected_length_rows} ({len(ALL_AXIS_LEVELS)} levels x {len(REPLICATE_IDS)} "
            f"replicates x {len(LENGTH_SCALE_METHODS)} methods: {LENGTH_SCALE_METHODS})."
        )
    length_scale_csv = PROCESSED_DIR / "length_scale_by_replicate.csv"
    with open(length_scale_csv, "w", encoding="utf-8", newline="") as f:
        f.write(
            "# kriging/sgs are intentionally NOT represented in this table: their "
            f"variogram range is a FIXED INPUT CONSTANT ({AXIS_HMAJ1:g} m) at every level "
            "and every replicate, never fitted from the conditioning data, so there is no "
            "per-replicate value to aggregate for them (see "
            "src/experiments/evaluate_sample_replicate_axis.py module docstring).\n"
        )
        length_scale_df.to_csv(f, index=False)

    # --- Aggregated summary across the 10 replicates, per (axis_level, ----
    # method, metric) -- 3 levels x 4 methods x 3 metrics = 36 core rows,
    # PLUS gp_mle/rbf_bootstrap length-scale summary rows (3 levels x 2
    # methods = 6 more).
    summary_rows = []
    for axis_level in ALL_AXIS_LEVELS:
        for method in METHODS:
            for metric_name in METRICS:
                vals = metrics_df.loc[
                    (metrics_df["axis_level"] == axis_level)
                    & (metrics_df["method"] == method)
                    & (metrics_df["metric"] == metric_name),
                    "value",
                ].values
                if len(vals) != len(REPLICATE_IDS):
                    raise ValueError(
                        f"level '{axis_level}' {method}/{metric_name}: expected "
                        f"{len(REPLICATE_IDS)} replicate values, found {len(vals)}."
                    )
                summary_rows.append(
                    {
                        "axis_level": axis_level,
                        "method": method,
                        "metric": metric_name,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "n_replicates": len(vals),
                    }
                )

    length_scale_metric_name = {
        "gp_mle": "length_scale_practical_range_m",
        "rbf_bootstrap": "length_scale_rbf_converted_m",
    }
    for axis_level in ALL_AXIS_LEVELS:
        for method in LENGTH_SCALE_METHODS:
            vals = length_scale_df.loc[
                (length_scale_df["axis_level"] == axis_level)
                & (length_scale_df["method"] == method),
                "length_scale_m",
            ].values
            if len(vals) != len(REPLICATE_IDS):
                raise ValueError(
                    f"level '{axis_level}' {method} length scale: expected "
                    f"{len(REPLICATE_IDS)} replicate values, found {len(vals)}."
                )
            summary_rows.append(
                {
                    "axis_level": axis_level,
                    "method": method,
                    "metric": length_scale_metric_name[method],
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

    print("\nSample-replicate axis metrics (3 levels x 10 replicates x 4 methods x 3 metrics):")
    print(metrics_df.to_string(index=False))
    print(
        "\nAggregated across the 10 replicates (mean/std/min/max per axis_level x method x "
        "metric, plus gp_mle/rbf_bootstrap length-scale rows):"
    )
    print(summary_df.to_string(index=False))
    print("\nEvaluated-cell counts per (level, replicate):")
    print(cell_counts.drop(columns="note").to_string(index=False))
    print("\nPredictive-variance metric sources:")
    print(variance_sources_df.drop(columns="note").to_string(index=False))
    print(f"\nmetrics.csv: {metrics_csv}")
    print(f"metrics_summary.csv: {summary_csv}")
    print(f"length_scale_by_replicate.csv: {length_scale_csv}")
    print(f"evaluation_cell_counts.csv: {cell_counts_csv}")
    print(f"variance_metric_sources.csv: {variance_sources_csv}")
    print(f"evaluation_diagnostics.csv: {diagnostics_csv}")

    return metrics_df, summary_df


if __name__ == "__main__":
    main()
