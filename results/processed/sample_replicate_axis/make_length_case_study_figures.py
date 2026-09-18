"""Case-study figures for the two fitted-length-scale EXTREMES within the
sample-seed replicate axis (src/experiments/sample_replicate_axis.py,
src/experiments/evaluate_sample_replicate_axis.py), at the 1% sample-density
level (axis_level="1", n_samples_requested=25, all 10 replicates share the
SAME ground-truth field: TRUTH_SEED=101, range=300 m).

WHY this figure exists
-----------------------
results/processed/sample_replicate_axis/length_scale_by_replicate.csv shows
that at this sparsest level, two of the four methods -- gp_mle (MLE-fitted
RBF length_scale) and rbf_bootstrap (CV-selected epsilon) -- fit wildly
different effective length scales depending ONLY on where the 25 samples
happened to land (sample_seed), with everything else (ground truth, method
tuning constants) held fixed. This script pulls out, for each of those two
methods, the ONE replicate that fit the SHORTEST length scale and the ONE
that fit the LONGEST, and puts their QC panels side by side so the reader
can see directly what a ~5x (gp_mle) or ~11x (rbf_bootstrap) swing in fitted
length scale actually looks like on the map -- not just as a number in a
table.

The two (min, max) replicate pairs are NOT hardcoded here: they are
recomputed at runtime from length_scale_by_replicate.csv (argmin/argmax of
length_scale_m within axis_level=="1", grouped by method), so this script
stays correct if that table is ever regenerated with different values.
Independently confirmed by the orchestrator's own recomputation before this
task was handed off:
  gp_mle:         MIN = rep0 (144.60 m, sample_seed=1001); MAX = rep8 (739.92 m, sample_seed=1009)
  rbf_bootstrap:  MIN = rep1 (141.42 m, sample_seed=1002); MAX = rep5 (1603.81 m, sample_seed=1006)

Layout
------
ONE figure per method, 2 ROWS (top = MIN-length replicate, bottom =
MAX-length replicate) x 6 COLUMNS:
  1. Ground truth + THAT replicate's own conditioning samples overlaid.
  2. Example realization (gp_mle: posterior_sample_map.npy, one posterior
     draw; rbf_bootstrap: bootstrap_replicate_maps.npy[0], the first of 10
     bootstrap replicates) -- the SAME arrays/roles
     make_sample_density_figures.py's make_predictions_figure_set() already
     uses for this QC role, not a new convention.
  3. Prediction map -- the ENSEMBLE mean (gp_mle: posterior_mean_map.npy;
     rbf_bootstrap: bootstrap_mean_map.npy). NOT point_estimate_map.npy:
     that array is only used elsewhere in this project for the MSE-scoring
     crossplot (make_crossplot_figure()'s CROSSPLOT_POINT_ESTIMATE_FILE),
     which for rbf_bootstrap deliberately uses the single-fit point
     estimate rather than the bootstrap mean. This QC figure follows the
     OTHER (QC-panel) convention instead, matching what
     make_predictions_figure_set() puts in its "ensemble mean" panel -- the
     two conventions are kept deliberately separate in this project and
     that separation is preserved here, not blurred.
  4. Predictive variance map (gp_mle: posterior_var_map.npy; rbf_bootstrap:
     bootstrap_var_map.npy).
  5. UMG plot (nominal probability vs. fraction of truth inside that
     interval, ``src.evaluation.accuracy_plot_fraction_in``, plus the ideal
     y=x line) with the recomputed UMG value in the panel title. This is
     called a "UMG plot" throughout this script, NOT a "calibration curve"
     (matching the display-text-only rename applied the same day to
     results/processed/sample_density_axis/make_sample_density_figures.py's
     make_calibration_grid_figure()).
  6. Accuracy crossplot (truth vs. that method's POINT ESTIMATE for that
     specific replicate) with a 1:1 reference line and the recomputed MSE in
     the panel title. This reuses the exact convention
     make_sample_density_figures.py's make_crossplot_figure() already
     established elsewhere in this project (axis limits from the truth
     field's own min/max with the same padding, MSE recomputed and
     cross-checked against the pinned metrics.csv value before plotting) --
     it is NOT a new convention invented for this script. Per that same
     convention, the point-estimate array is METHOD-SPECIFIC and, for
     rbf_bootstrap, DIFFERENT from column 3's ensemble mean: gp_mle uses
     posterior_mean_map.npy (same array as column 3 -- no distinction for
     this method); rbf_bootstrap uses point_estimate_map.npy (the SINGLE-FIT
     estimate), NOT bootstrap_mean_map.npy (the ensemble mean used in
     column 3). This MSE-scoring-vs-QC-panel split is a real, deliberate,
     already-documented distinction in this project (see
     CROSSPLOT_POINT_ESTIMATE_FILE in make_sample_density_figures.py and the
     note under column 3 above) -- it is preserved here, not blurred.

DECISIVE CORRECTNESS CHECK (same pattern make_crossplot_figure() already
uses for MSE elsewhere in this project): for each of the 4 (method,
replicate) case-study cells, this script (a) regenerates that replicate's
conditioning samples from its own sample_seed and cross-checks them against
the run's own recorded samples.csv (atol=1e-10, the project's standing
float-text round-trip tolerance), (b) recomputes UMG from the run's own
mean/variance maps and requires it to match (np.isclose) the value already
stored in metrics.csv for that exact (axis_level="1", replicate, method,
metric="umg") row, and (c) recomputes MSE from the run's own point-estimate
map (masked to that replicate's evaluated cells) and requires it to match
(np.isclose) the value already stored in metrics.csv for that exact
(axis_level="1", replicate, method, metric="mse") row. Any of these checks
failing raises -- this script never silently plots a number that disagrees
with the pinned evaluation output.

COLOR-SCALE CHOICES (stated here, per this project's standing convention of
documenting every color-scale decision in the figure/caption itself, see
make_sample_density_figures.py's module docstring)
-----------------------------------------------------------------------------
- POROSITY (truth / example realization / prediction mean) is SHARED across
  BOTH rows of one method's figure. This is safe here because both rows
  show the SAME ground-truth field (TRUTH_SEED=101 is common to every
  replicate at every level) and the same physical units -- exactly the
  reasoning make_sample_density_figures.py gives for sharing its porosity
  scale across levels that share one truth field. The scale is anchored on
  the truth field's own min/max (same convention every other QC figure in
  this project uses), which is safely inclusive of the mean/example maps in
  practice (kriging/SGS/RBF/GP predictions do not routinely exceed the
  truth's own range in this project's runs).
- VARIANCE is scaled PER ROW (i.e. the min-length and max-length replicate
  each get their OWN [0, max] scale), NOT shared between the two rows and
  NOT shared with the porosity columns. This is a deliberate departure from
  make_sample_density_figures.py's WITHIN-LEVEL (i.e. across methods, same
  level) variance-sharing rule, generalized to this figure's own comparison
  axis: here the "levels" being compared are the two replicates of ONE
  method, and their predictive-variance magnitudes are the entire point of
  the case study -- a short-length-scale fit and a long-length-scale fit
  are expected to (and, as the case study shows, do) produce variance maps
  that differ by roughly an order of magnitude. Forcing one shared scale
  across both rows would flatten the smaller-variance row to a single
  near-uniform color and defeat the figure's purpose. Each row's variance
  panel states its own numeric scale max so the two rows remain
  numerically comparable even though they are not comparable by color.

Run with:
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_length_case_study_figures
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation import (  # noqa: E402
    accuracy_plot_fraction_in,
    calc_umg,
    conditioning_cell_mask,
    mse,
)
from src.experiments.base_case import (  # noqa: E402
    NX, NY, XMIN, XMAX, YMIN, YMAX, XMN, YMN, XSIZ, YSIZ,
)
from src.experiments.base_case_conditioning import (  # noqa: E402
    VCOL,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.evaluate_sample_replicate_axis import safe_sqrt_variance  # noqa: E402
from src.experiments.sample_density_axis import AXIS_HMAJ1, AXIS_HMIN1  # noqa: E402
from src.experiments.sample_replicate_axis import (  # noqa: E402
    N_SAMPLES_REQUESTED_BY_LEVEL,
    REPLICATE_SEED,
    TRUTH_SEED,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "sample_replicate_axis"

# The 1% density level named in this task -- the sparsest of the 3
# sample-density-axis levels, where sample-placement effects on fitted
# length scale are largest.
AXIS_LEVEL = "1"

FIG_DPI = 300

EXTENT = [XMIN, XMAX, YMIN, YMAX]

# Same float-text round-trip tolerance every other script in this project
# uses for samples.csv comparisons (see e.g. sample_replicate_axis.py's
# SAMPLES_MATCH_ATOL, evaluate_sample_replicate_axis.py's own constant).
SAMPLES_MATCH_ATOL = 1e-10

# Same MSE/UMG cross-check tolerance make_crossplot_figure() uses.
UMG_MATCH_RTOL = 1e-8
UMG_MATCH_ATOL = 1e-8
MSE_MATCH_RTOL = 1e-8
MSE_MATCH_ATOL = 1e-8

METHOD_LABELS = {"gp_mle": "GP-MLE", "rbf_bootstrap": "RBF+bootstrap"}
METHOD_ARRAY_FILES = {
    "gp_mle": {
        "mean": "posterior_mean_map.npy",
        "var": "posterior_var_map.npy",
        "example": "posterior_sample_map.npy",
    },
    "rbf_bootstrap": {
        "mean": "bootstrap_mean_map.npy",
        "var": "bootstrap_var_map.npy",
        "example": None,  # example is replicate_maps[0], loaded specially below
    },
}

# Point-estimate array per method, matching make_sample_density_figures.py's
# CROSSPLOT_POINT_ESTIMATE_FILE / MSE-scoring convention -- NOT the same as
# METHOD_ARRAY_FILES["mean"] for rbf_bootstrap (see module docstring, column 6).
METHOD_POINT_ESTIMATE_FILES = {
    "gp_mle": "posterior_mean_map.npy",
    "rbf_bootstrap": "point_estimate_map.npy",
}


def identify_length_extremes(length_df: pd.DataFrame, method: str) -> dict:
    """Recompute (not hardcode) the MIN- and MAX-length-scale replicates for
    ``method`` at AXIS_LEVEL, from length_scale_by_replicate.csv."""
    sub = length_df[
        (length_df["axis_level"].astype(str) == AXIS_LEVEL) & (length_df["method"] == method)
    ]
    if len(sub) != 10:
        raise ValueError(
            f"expected 10 replicate rows for method={method} at axis_level={AXIS_LEVEL}, "
            f"found {len(sub)}."
        )
    min_row = sub.loc[sub["length_scale_m"].idxmin()]
    max_row = sub.loc[sub["length_scale_m"].idxmax()]
    return {"min": min_row, "max": max_row}


def _scatter_samples(ax, samples_df, color="red"):
    ax.scatter(
        samples_df["X"], samples_df["Y"], s=16, c=color, marker="+",
        label="conditioning samples",
    )


def _panel(ax, arr, title, vmin, vmax, cmap, cbar_label, samples_df, scatter_color="red"):
    """Redefined identically (not imported) from
    make_sample_density_figures.py's own ``_panel`` -- following this
    project's own standing precedent (make_sample_replicate_figures.py's
    module docstring) of redefining small plotting helpers locally rather
    than importing them, so this script does not trigger that other
    module's file-reading import-time side effects."""
    im = ax.imshow(arr, extent=EXTENT, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    _scatter_samples(ax, samples_df, color=scatter_color)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    return im


def load_case_study_row(
    method: str, role: str, row, truth: np.ndarray, source_runs: dict, metrics_df: pd.DataFrame
) -> dict:
    """Load + verify everything one row (one replicate) of one method's
    case-study figure needs. Raises on any samples.csv or UMG mismatch."""
    replicate_id = row["replicate"]
    length_scale_m = float(row["length_scale_m"])
    length_definition = row["length_definition"]
    sample_seed = int(REPLICATE_SEED[replicate_id])
    n_requested = N_SAMPLES_REQUESTED_BY_LEVEL[AXIS_LEVEL]

    run_dir = _REPO_ROOT / source_runs[AXIS_LEVEL][replicate_id][method]

    # --- (a) Cross-check regenerated samples against the run's own samples.csv ---
    samples_df = get_conditioning_samples(truth, sample_seed=sample_seed, n_samples=n_requested)
    recorded_samples = pd.read_csv(run_dir / "samples.csv")
    if len(recorded_samples) != len(samples_df) or not np.allclose(
        recorded_samples[["X", "Y", VCOL]].values, samples_df[["X", "Y", VCOL]].values,
        rtol=0.0, atol=SAMPLES_MATCH_ATOL,
    ):
        raise ValueError(
            f"{method} {replicate_id} (role={role}): regenerated conditioning samples "
            f"(sample_seed={sample_seed}, n={n_requested}) do not match {run_dir}/samples.csv "
            f"-- refusing to plot a mismatched run."
        )

    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    truth_masked = truth[mask]

    mean_map = np.load(run_dir / METHOD_ARRAY_FILES[method]["mean"])
    var_map = np.load(run_dir / METHOD_ARRAY_FILES[method]["var"])
    if method == "gp_mle":
        example_map = np.load(run_dir / METHOD_ARRAY_FILES[method]["example"])
        example_label = f"GP-MLE -- posterior sample (1 draw, {replicate_id})"
        mean_label = "GP-MLE -- posterior mean"
        var_label = "GP-MLE -- posterior variance"
    else:  # rbf_bootstrap
        replicate_maps = np.load(run_dir / "bootstrap_replicate_maps.npy")
        example_map = replicate_maps[0]
        n_bootstrap = replicate_maps.shape[0]
        example_label = f"RBF+bootstrap -- example replicate (#0 of {n_bootstrap}, {replicate_id})"
        mean_label = "RBF+bootstrap -- ensemble mean"
        var_label = "RBF+bootstrap -- ensemble variance"

    # --- (b) Recompute UMG and cross-check against the pinned metrics.csv value ---
    std_masked = safe_sqrt_variance(var_map[mask], f"{method} {replicate_id} variance map")
    mean_masked = mean_map[mask]
    p_intervals, fraction_in = accuracy_plot_fraction_in(truth_masked, mean_masked, std_masked)
    umg_recomputed = calc_umg(p_intervals, fraction_in)

    pinned = metrics_df.loc[
        (metrics_df["axis_level"].astype(str) == AXIS_LEVEL)
        & (metrics_df["replicate"] == replicate_id)
        & (metrics_df["method"] == method)
        & (metrics_df["metric"] == "umg"),
        "value",
    ]
    if len(pinned) != 1:
        raise ValueError(
            f"expected exactly 1 pinned UMG row for {method} {replicate_id} at axis_level="
            f"{AXIS_LEVEL}; found {len(pinned)}."
        )
    pinned_umg = float(pinned.iloc[0])
    if not np.isclose(umg_recomputed, pinned_umg, rtol=UMG_MATCH_RTOL, atol=UMG_MATCH_ATOL):
        raise ValueError(
            f"{method} {replicate_id}: recomputed UMG ({umg_recomputed}) does not match "
            f"metrics.csv ({pinned_umg}) -- mask, source array, or metric definition has "
            "drifted from evaluate_sample_replicate_axis.py; not plotting silently."
        )

    # --- (c) Load the point-estimate map and cross-check recomputed MSE against
    # the pinned metrics.csv value (same convention as make_crossplot_figure()) ---
    point_estimate_map = np.load(run_dir / METHOD_POINT_ESTIMATE_FILES[method])
    point_estimate_masked = point_estimate_map[mask]
    mse_recomputed = mse(truth_masked, point_estimate_masked)

    pinned_mse = metrics_df.loc[
        (metrics_df["axis_level"].astype(str) == AXIS_LEVEL)
        & (metrics_df["replicate"] == replicate_id)
        & (metrics_df["method"] == method)
        & (metrics_df["metric"] == "mse"),
        "value",
    ]
    if len(pinned_mse) != 1:
        raise ValueError(
            f"expected exactly 1 pinned MSE row for {method} {replicate_id} at axis_level="
            f"{AXIS_LEVEL}; found {len(pinned_mse)}."
        )
    pinned_mse_value = float(pinned_mse.iloc[0])
    if not np.isclose(mse_recomputed, pinned_mse_value, rtol=MSE_MATCH_RTOL, atol=MSE_MATCH_ATOL):
        raise ValueError(
            f"{method} {replicate_id}: recomputed MSE ({mse_recomputed}) does not match "
            f"metrics.csv ({pinned_mse_value}) -- mask or point-estimate source array has "
            "drifted from evaluate_sample_replicate_axis.py; not plotting silently."
        )

    return {
        "role": role,
        "replicate_id": replicate_id,
        "sample_seed": sample_seed,
        "length_scale_m": length_scale_m,
        "length_definition": length_definition,
        "samples_df": samples_df,
        "example_map": example_map,
        "example_label": example_label,
        "mean_map": mean_map,
        "mean_label": mean_label,
        "var_map": var_map,
        "var_label": var_label,
        "var_row_max": float(np.max(var_map)),
        "p_intervals": p_intervals,
        "fraction_in": fraction_in,
        "umg_recomputed": umg_recomputed,
        "umg_pinned": pinned_umg,
        "truth_masked": truth_masked,
        "point_estimate_masked": point_estimate_masked,
        "mse_recomputed": mse_recomputed,
        "mse_pinned": pinned_mse_value,
    }


def make_case_study_figure(
    method: str, truth: np.ndarray, source_runs: dict, length_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> Path:
    extremes = identify_length_extremes(length_df, method)
    min_row = load_case_study_row(method, "MIN", extremes["min"], truth, source_runs, metrics_df)
    max_row = load_case_study_row(method, "MAX", extremes["max"], truth, source_runs, metrics_df)

    porosity_vmin, porosity_vmax = float(np.min(truth)), float(np.max(truth))
    crossplot_pad = 0.05 * (porosity_vmax - porosity_vmin)
    crossplot_lims = (porosity_vmin - crossplot_pad, porosity_vmax + crossplot_pad)

    fig, axes = plt.subplots(2, 6, figsize=(30, 10.5))
    for row_idx, row_data in enumerate([min_row, max_row]):
        ax_truth, ax_example, ax_mean, ax_var, ax_umg, ax_crossplot = axes[row_idx]

        _panel(
            ax_truth, truth,
            f"[{row_data['role']}] Truth + samples\n{row_data['replicate_id']} "
            f"(sample_seed={row_data['sample_seed']}, n={len(row_data['samples_df'])}); "
            f"length_scale={row_data['length_scale_m']:.1f} m",
            porosity_vmin, porosity_vmax, "viridis", "Porosity (%)", row_data["samples_df"],
        )
        _panel(
            ax_example, row_data["example_map"], row_data["example_label"],
            porosity_vmin, porosity_vmax, "viridis", "Porosity (%)", row_data["samples_df"],
        )
        _panel(
            ax_mean, row_data["mean_map"], row_data["mean_label"],
            porosity_vmin, porosity_vmax, "viridis", "Porosity (%)", row_data["samples_df"],
        )
        _panel(
            ax_var, row_data["var_map"], row_data["var_label"],
            0.0, row_data["var_row_max"], "magma", "Variance (Porosity %^2)",
            row_data["samples_df"], scatter_color="cyan",
        )
        ax_var.text(
            0.02, -0.16,
            f"Row variance scale: [0, {row_data['var_row_max']:.2f}] %^2 "
            "(PER-ROW ONLY -- NOT shared with the other row; see module docstring).",
            transform=ax_var.transAxes, fontsize=7, color="dimgray", wrap=True,
        )

        ax_umg.plot(
            row_data["p_intervals"], row_data["fraction_in"],
            marker="o", markersize=4, color="tab:red" if method == "gp_mle" else "tab:orange",
            label=f"{METHOD_LABELS[method]} (UMG={row_data['umg_recomputed']:.3f})",
        )
        ax_umg.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", label="ideal (y=x)")
        ax_umg.set_xlim(0.0, 1.0)
        ax_umg.set_ylim(0.0, 1.0)
        ax_umg.set_xlabel("Nominal probability interval")
        ax_umg.set_ylabel("Fraction of truth in interval")
        ax_umg.set_title(f"[{row_data['role']}] UMG plot (UMG={row_data['umg_recomputed']:.3f})")
        ax_umg.legend(loc="upper left", fontsize=8)
        ax_umg.grid(alpha=0.3)

        point_estimate_label = (
            "GP-MLE posterior mean" if method == "gp_mle" else "RBF+bootstrap point estimate"
        )
        ax_crossplot.scatter(
            row_data["truth_masked"], row_data["point_estimate_masked"],
            s=10, alpha=0.5, color="tab:red" if method == "gp_mle" else "tab:orange",
            edgecolors="none",
        )
        ax_crossplot.plot(crossplot_lims, crossplot_lims, color="gray", linestyle="--",
                           label="1:1")
        ax_crossplot.set_xlim(crossplot_lims)
        ax_crossplot.set_ylim(crossplot_lims)
        ax_crossplot.set_xlabel("Truth (Porosity %)")
        ax_crossplot.set_ylabel(f"{point_estimate_label} (Porosity %)")
        ax_crossplot.set_title(
            f"[{row_data['role']}] Accuracy crossplot (MSE={row_data['mse_recomputed']:.3f})"
        )
        ax_crossplot.legend(loc="upper left", fontsize=8)
        ax_crossplot.grid(alpha=0.3)
        ax_crossplot.set_aspect("equal", adjustable="box")

        print(
            f"  {method} {row_data['role']} ({row_data['replicate_id']}, sample_seed="
            f"{row_data['sample_seed']}): length_scale={row_data['length_scale_m']:.4f} m, "
            f"UMG recomputed={row_data['umg_recomputed']:.6f} vs. pinned metrics.csv="
            f"{row_data['umg_pinned']:.6f} (match); MSE recomputed="
            f"{row_data['mse_recomputed']:.6f} vs. pinned metrics.csv="
            f"{row_data['mse_pinned']:.6f} (match)"
        )

    length_definition_note = min_row["length_definition"]
    caption = (
        f"{METHOD_LABELS[method]} at the 1% sample-density level (axis_level='1', "
        f"n_samples_requested=25): the MIN-fitted-length-scale replicate (top row, "
        f"{min_row['replicate_id']}, {min_row['length_scale_m']:.1f} m) vs. the "
        f"MAX-fitted-length-scale replicate (bottom row, {max_row['replicate_id']}, "
        f"{max_row['length_scale_m']:.1f} m) among the SAME 10 sample-location replicates "
        f"(sample_seed 1001-1010), same ground-truth field for every replicate (TRUTH_SEED="
        f"{TRUTH_SEED}, range={AXIS_HMAJ1:g} m). Length scale definition: "
        f"{length_definition_note} Porosity color scale (truth/example/mean columns) is "
        "SHARED across both rows (same truth field, same units); variance color scale is "
        "PER ROW (see the note under each variance panel) -- see module docstring for why. "
        "UMG values plotted/annotated are recomputed here from this run's own predictive "
        "mean/variance maps and are verified (np.isclose) to match "
        "results/processed/sample_replicate_axis/metrics.csv's pinned value for the same "
        "(axis_level, replicate, method, metric='umg') row. The 6th (accuracy crossplot) "
        "panel plots truth vs. that replicate's point estimate -- "
        f"{METHOD_POINT_ESTIMATE_FILES[method]}, which for rbf_bootstrap is the SINGLE-FIT "
        "estimate and deliberately NOT the ensemble-mean array used in column 3 (see module "
        "docstring) -- masked to that replicate's own evaluated cells (its conditioning "
        "samples excluded), with axis limits from the truth field's own [min, max] (5% "
        "padding) and MSE recomputed and verified (np.isclose) to match metrics.csv's pinned "
        "value for the same (axis_level, replicate, method, metric='mse') row."
    )

    plt.suptitle(
        f"{METHOD_LABELS[method]}: fitted-length-scale extremes across 10 sample-location "
        "replicates at the 1% sample-density level",
        fontsize=14,
    )
    fig.text(
        0.5, 0.01, textwrap.fill(caption, 200),
        ha="center", va="top", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.035, bottom=0.14, right=0.99, top=0.92, wspace=0.45, hspace=0.42)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"case_study_{method}_level1_length_extremes.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"{out.name}: {out} ({out.stat().st_size} bytes)")
    return out


def main():
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)
    length_df = pd.read_csv(PROCESSED_DIR / "length_scale_by_replicate.csv", comment="#")
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")

    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)

    saved = []
    for method in ["gp_mle", "rbf_bootstrap"]:
        print(f"\nBuilding case-study figure for {method}...")
        saved.append(make_case_study_figure(method, truth, source_runs, length_df, metrics_df))

    print("\nAll length-extreme case-study figures:")
    for f in saved:
        print(" ", f)
    return saved


if __name__ == "__main__":
    main()
