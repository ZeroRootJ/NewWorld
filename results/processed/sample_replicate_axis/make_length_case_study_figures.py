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
ONE figure per method, 4 ROWS x 4 COLUMNS, organised as TWO BLOCKS of
2 rows each (2026-09-18: the panel set grew from 6 to 8, so instead of one
very wide 2x6 strip each replicate is now FOLDED into its own 2x4 block for
legibility). The two blocks are separated by a wider vertical gap than the
gap between the two rows inside a block, and each block sits on its own
tinted background band with its own bold block title carrying the same
identifying information the old per-row panel titles carried (role
MIN/MAX, replicate id, sample_seed, n samples, fitted length scale):

  BLOCK 1 (figure rows 1-2) = MIN-fitted-length-scale replicate
  BLOCK 2 (figure rows 3-4) = MAX-fitted-length-scale replicate

Inside one block:
  ROW A (map panels):        1. truth + samples   2. example realization
                             3. prediction mean   4. predictive variance
  ROW B (diagnostic panels): 5. UMG plot          6. accuracy crossplot
                             7. variogram reproduction
                             8. distribution reproduction

Panels 1-6 are UNCHANGED in content and in their array-selection rules from
the previous 2x6 version; panels 7-8 are new (see their own sections below).

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
  7. VARIOGRAM REPRODUCTION (new, 2026-09-18). Four curves on ONE set of
     lag bins: (i) the ground-truth field's experimental variogram, (ii)
     the experimental variogram of THAT replicate's example realization /
     draw (exactly the array panel 2 shows -- gp_mle:
     posterior_sample_map.npy, rbf_bootstrap: bootstrap_replicate_maps[0]),
     (iii) the experimental variogram of THAT replicate's prediction mean
     (exactly the array panel 3 shows -- gp_mle: posterior_mean_map.npy,
     rbf_bootstrap: bootstrap_mean_map.npy), and (iv) the truth's
     THEORETICAL spherical model. The estimator is NOT reimplemented here:
     ``compute_experimental_variogram`` and ``spherical_semivariance`` are
     imported verbatim from
     results/processed/sample_density_axis/make_length_and_variogram_figures.py
     (that module has no import-time file-reading side effects, so importing
     it is safe; nothing is copy-pasted). Because that estimator draws its
     (i, j) cell-index pairs from a fixed ``RandomState(VARIOGRAM_PAIR_SEED
     = 90)`` seeded INSIDE the function, every call draws the IDENTICAL pair
     set, so all three empirical curves (and the truth curve, computed once
     and reused for both blocks and both methods) are computed on exactly
     the same pairs and the same lag bins and are therefore directly
     comparable; this fact and the seed are printed on the panel. The
     theoretical curve's nugget / structured sill / range come from
     src.experiments.base_case (NUG, CC1, POR_STDEV, HMAJ1) via the imported
     ``spherical_semivariance`` -- nothing is hardcoded here. The total sill
     (POR_STDEV**2) is drawn as a horizontal reference line.

     *** SCOPE NOTE THAT MUST TRAVEL WITH THIS PANEL ***
     Neither method shown here is designed to reproduce the truth's spatial
     statistics. GP-MLE's posterior draw reproduces the GAUSSIAN RBF kernel
     + nugget that GP-MLE itself fitted by marginal likelihood -- not the
     truth's spherical model. RBF+bootstrap's replicate map is a
     resampled-interpolation result, NOT a conditional simulation, and its
     kernel has no statistical variogram interpretation at all (see
     RBF_CAVEAT in make_length_and_variogram_figures.py). This panel
     therefore only STATES what was compared with what and quantifies the
     observed differences; it deliberately does not say either method
     "fails to reproduce" anything, and it attributes no cause. Likewise, a
     prediction MEAN is an estimator and is expected to be smoother /
     lower-variance than the truth -- that is a generic property of
     estimation, not a defect of a method, and is not described as one
     here. Interpretation is left to the paper.
  8. DISTRIBUTION REPRODUCTION (new, 2026-09-18). Overlaid empirical CDFs
     of: the ground-truth field, the same example realization/draw as
     panel 7 (ii), the same prediction mean as panel 7 (iii), and THAT
     replicate's own conditioning-sample values (samples.csv's VCOL). The
     mean and standard deviation (ddof=1 for all four curves, stated on the
     panel) of each curve are printed inside the panel. EVALUATION-CELL
     DEFINITION: the three map-derived CDFs use the SAME mask every other
     scored panel of this figure uses -- ``conditioning_cell_mask`` (that
     replicate's own conditioning cells EXCLUDED), i.e. the identical
     ``mask`` array feeding the UMG and MSE recomputations above. The
     conditioning-sample curve is, by construction, the values AT those
     excluded cells (all 25 of them); that is the only way the four curves
     differ in support and it is stated in the panel note and the caption.
     NOTE that panel 7's variogram, by contrast, is computed on the FULL
     50x50 grid (the imported estimator draws pairs over every cell and has
     no mask argument) -- this difference between panels 7 and 8 is stated
     on panel 7 and in the caption rather than silently glossed over.

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
  BOTH replicate BLOCKS of one method's figure. This is safe here because
  both blocks show the SAME ground-truth field (TRUTH_SEED=101 is common to every
  replicate at every level) and the same physical units -- exactly the
  reasoning make_sample_density_figures.py gives for sharing its porosity
  scale across levels that share one truth field. The scale is anchored on
  the truth field's own min/max (same convention every other QC figure in
  this project uses), which is safely inclusive of the mean/example maps in
  practice (kriging/SGS/RBF/GP predictions do not routinely exceed the
  truth's own range in this project's runs).
- VARIANCE is scaled PER REPLICATE BLOCK (i.e. the min-length and max-length
  replicate each get their OWN [0, max] scale), NOT shared between the two
  blocks and NOT shared with the porosity panels. This is a deliberate departure from
  make_sample_density_figures.py's WITHIN-LEVEL (i.e. across methods, same
  level) variance-sharing rule, generalized to this figure's own comparison
  axis: here the "levels" being compared are the two replicates of ONE
  method, and their predictive-variance magnitudes are the entire point of
  the case study -- a short-length-scale fit and a long-length-scale fit
  are expected to (and, as the case study shows, do) produce variance maps
  that differ by roughly an order of magnitude. Forcing one shared scale
  across both blocks would flatten the smaller-variance block to a single
  near-uniform color and defeat the figure's purpose. Each block's variance
  panel states its own numeric scale max so the two blocks remain
  numerically comparable even though they are not comparable by color.
- The DIAGNOSTIC panels' AXIS RANGES are SHARED across the two replicate
  blocks, explicitly (set_xlim/set_ylim), because comparing the MIN-length
  block against the MAX-length block IS the purpose of the figure: UMG
  [0,1]x[0,1] (as before), accuracy crossplot on the truth field's own
  [min,max]+5% (as before), and -- new -- the variogram panels share one
  lag range and one semivariance range (computed over BOTH blocks' curves
  plus the theoretical model) and the distribution panels share one
  porosity range (computed over BOTH blocks' truth/example/mean/sample
  values) with the CDF axis fixed at [0,1].

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
    CC1, HMAJ1, NUG, POR_STDEV,
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

# The variogram estimator + theoretical spherical model are REUSED, not
# reimplemented (see module docstring, panel 7). That module does no
# file reading at import time (its I/O all lives inside main()), so this
# import has no side effects.
from results.processed.sample_density_axis.make_length_and_variogram_figures import (  # noqa: E402
    LAG_BIN_WIDTH_M,
    LAG_MAX_M,
    MIN_PAIRS_PER_BIN,
    N_PAIRS_DRAWN,
    VARIOGRAM_PAIR_SEED,
    compute_experimental_variogram,
    spherical_semivariance,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "sample_replicate_axis"

# The 1% density level named in this task -- the sparsest of the 3
# sample-density-axis levels, where sample-placement effects on fitted
# length scale are largest.
AXIS_LEVEL = "1"

FIG_DPI = 300

# 4 rows x 4 cols = TWO stacked 2x4 BLOCKS, one per replicate. Each entry is
# that block's (gridspec top, gridspec bottom) in figure coordinates. The
# GAP BETWEEN BLOCKS (0.585 - 0.450 = 0.135) is deliberately ~1.75x the gap
# between the two rows INSIDE a block (block height 0.355 over 2 rows at
# hspace=0.55 -> inner gap ~0.077), so the two replicate blocks read as
# groups; each block additionally sits on its own tinted, outlined band.
BLOCK_GEOMETRY = ((0.940, 0.585), (0.450, 0.085))
BLOCK_BAND_FACECOLORS = ("#eef4fb", "#fbf3ee")
BLOCK_BAND_PAD_TOP = 0.036
BLOCK_BAND_PAD_BOTTOM = 0.058
BLOCK_GRID_KWARGS = dict(left=0.05, right=0.985, wspace=0.42, hspace=0.55)

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

# --- Panels 7-8 (variogram / distribution reproduction) --------------------
# The imported ``spherical_semivariance`` builds its theoretical curve from
# src.experiments.base_case's HMAJ1 internally, while THIS axis's truth field
# is generated with sample_density_axis's AXIS_HMAJ1/AXIS_HMIN1. They are the
# same constant by design (AXIS_HMAJ1 = HMAJ1), but that is checked at import
# time here rather than assumed -- if the axis ever moves off the base-case
# range, the imported theoretical curve would silently stop describing this
# figure's truth field.
if not (AXIS_HMAJ1 == HMAJ1 and AXIS_HMIN1 == HMAJ1):
    raise ValueError(
        f"this figure's truth field uses hmaj1={AXIS_HMAJ1}, hmin1={AXIS_HMIN1}, but the "
        f"imported spherical_semivariance() is built on base_case HMAJ1={HMAJ1} -- the "
        "theoretical variogram curve would not describe the plotted truth field."
    )

# One color per CURVE ROLE, used identically in panels 7 and 8 so the reader
# reads one legend for both. tab:purple (not tab:red) for the theoretical
# model so it never collides with gp_mle's own tab:red method color.
CURVE_COLORS = {
    "truth": "black",
    "example": "tab:blue",
    "mean": "tab:green",
    "samples": "tab:brown",
    "model": "tab:purple",
}

# Lags at which the panel annotates each empirical curve numerically,
# expressed as fractions of the truth's own range (never hardcoded metres):
# half the range and the range itself.
VARIOGRAM_REFERENCE_LAG_FRACTIONS = (0.5, 1.0)

# ddof used for every mean/std printed in the distribution panel. Stated on
# the panel itself; applied identically to all four curves (including the
# 25-value conditioning-sample curve) so the numbers are comparable.
DIST_STD_DDOF = 1


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


def _empirical_cdf(values: np.ndarray):
    """Plotting-position empirical CDF ((i - 0.5)/n, the standard
    mid-point convention) of ``values``. Returns (sorted values, cumulative
    probabilities)."""
    v = np.sort(np.asarray(values, dtype=float).ravel())
    n = v.size
    if n == 0:
        raise ValueError("_empirical_cdf: empty input.")
    return v, (np.arange(1, n + 1) - 0.5) / n


def _curve_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float).ravel()
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=DIST_STD_DDOF)),
    }


def _variogram_at_reference_lags(vario_df: pd.DataFrame) -> list:
    """For each reference lag (a fraction of the truth's own range HMAJ1),
    return the NEAREST available lag bin and each curve's value there.

    Purely descriptive: the empirical value, the theoretical model value at
    the same bin centre, and their ratio. No causal claim is attached to
    these numbers anywhere in this script.
    """
    out = []
    centers = vario_df["lag_bin_center_m"].values
    for frac in VARIOGRAM_REFERENCE_LAG_FRACTIONS:
        target = frac * HMAJ1
        b = int(np.argmin(np.abs(centers - target)))
        out.append({
            "target_lag_m": float(target),
            "bin_center_m": float(centers[b]),
            "empirical": float(vario_df["semivariance_empirical"].values[b]),
            "theoretical": float(vario_df["semivariance_theoretical"].values[b]),
        })
    return out


def compute_map_variogram(map_2d: np.ndarray, label: str) -> pd.DataFrame:
    """Thin logging wrapper around the IMPORTED
    ``compute_experimental_variogram`` (see module docstring, panel 7).
    The imported function re-seeds its own RandomState(VARIOGRAM_PAIR_SEED)
    on every call, so every map passed through here is evaluated on the
    IDENTICAL (i, j) pair set and lag bins."""
    print(f"  [variogram] {label}")
    return compute_experimental_variogram(map_2d)


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
        # Short forms for the panel-7/8 legends -- SAME arrays as panels 2/3.
        example_short_label = "posterior draw (panel 2)"
        mean_short_label = "posterior mean (panel 3)"
    else:  # rbf_bootstrap
        replicate_maps = np.load(run_dir / "bootstrap_replicate_maps.npy")
        example_map = replicate_maps[0]
        n_bootstrap = replicate_maps.shape[0]
        example_label = f"RBF+bootstrap -- example replicate (#0 of {n_bootstrap}, {replicate_id})"
        mean_label = "RBF+bootstrap -- ensemble mean"
        var_label = "RBF+bootstrap -- ensemble variance"
        example_short_label = "bootstrap replicate #0 (panel 2)"
        mean_short_label = "bootstrap ensemble mean (panel 3)"

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

    # --- (d) Panel 7 inputs: experimental variograms of the SAME arrays -----
    # panels 2 and 3 display. Computed on the FULL grid (the imported
    # estimator has no mask argument) -- deliberately different from the
    # masked support panel 8 uses; both are stated on the figure.
    vario_example = compute_map_variogram(
        example_map, f"{method} {replicate_id} ({role}) example realization/draw"
    )
    vario_mean = compute_map_variogram(
        mean_map, f"{method} {replicate_id} ({role}) prediction mean"
    )

    # --- (e) Panel 8 inputs: values on the SAME evaluated-cell mask used ----
    # for the UMG and MSE checks above (conditioning cells excluded), plus
    # the conditioning-sample values themselves (which live exactly at the
    # excluded cells).
    example_masked = example_map[mask]
    sample_values = samples_df[VCOL].values.astype(float)
    dist_stats = {
        "truth": _curve_stats(truth_masked),
        "example": _curve_stats(example_masked),
        "mean": _curve_stats(mean_masked),
        "samples": _curve_stats(sample_values),
    }

    return {
        "role": role,
        "replicate_id": replicate_id,
        "sample_seed": sample_seed,
        "length_scale_m": length_scale_m,
        "length_definition": length_definition,
        "samples_df": samples_df,
        "example_map": example_map,
        "example_label": example_label,
        "example_short_label": example_short_label,
        "mean_map": mean_map,
        "mean_label": mean_label,
        "mean_short_label": mean_short_label,
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
        "n_evaluated_cells": int(mask.sum()),
        "vario_example": vario_example,
        "vario_mean": vario_mean,
        "example_masked": example_masked,
        "mean_masked": mean_masked,
        "sample_values": sample_values,
        "dist_stats": dist_stats,
    }


def make_case_study_figure(
    method: str, truth: np.ndarray, source_runs: dict, length_df: pd.DataFrame,
    metrics_df: pd.DataFrame, truth_vario: pd.DataFrame,
) -> Path:
    extremes = identify_length_extremes(length_df, method)
    min_row = load_case_study_row(method, "MIN", extremes["min"], truth, source_runs, metrics_df)
    max_row = load_case_study_row(method, "MAX", extremes["max"], truth, source_runs, metrics_df)
    block_rows = [min_row, max_row]

    porosity_vmin, porosity_vmax = float(np.min(truth)), float(np.max(truth))
    crossplot_pad = 0.05 * (porosity_vmax - porosity_vmin)
    crossplot_lims = (porosity_vmin - crossplot_pad, porosity_vmax + crossplot_pad)

    # --- Panel 7 bookkeeping: verify every variogram curve really was -------
    # computed on the SAME pair set / lag bins (the imported estimator
    # re-seeds RandomState(VARIOGRAM_PAIR_SEED) per call, so the bin centres
    # AND the per-bin pair counts must come out bit-identical; if they ever
    # do not, the curves are not comparable and this raises instead of
    # drawing a misleading overlay), then extract the annotated reference-lag
    # numbers.
    for rd in block_rows:
        for vdf, what in [(rd["vario_example"], "example"), (rd["vario_mean"], "prediction mean")]:
            same_bins = np.array_equal(
                vdf["lag_bin_center_m"].values, truth_vario["lag_bin_center_m"].values
            )
            same_counts = np.array_equal(vdf["n_pairs"].values, truth_vario["n_pairs"].values)
            if not (same_bins and same_counts):
                raise ValueError(
                    f"{method} {rd['replicate_id']} ({rd['role']}): the {what} variogram was "
                    "not computed on the same lag bins / (i,j) pair set as the truth "
                    "variogram -- the overlay would not be a like-for-like comparison."
                )
        t_refs = _variogram_at_reference_lags(truth_vario)
        e_refs = _variogram_at_reference_lags(rd["vario_example"])
        m_refs = _variogram_at_reference_lags(rd["vario_mean"])
        rd["vario_reference"] = [
            {
                "frac_of_range": frac,
                "bin_center_m": t["bin_center_m"],
                "theoretical": t["theoretical"],
                "truth": t["empirical"],
                "example": e["empirical"],
                "mean": m["empirical"],
            }
            for frac, t, e, m in zip(VARIOGRAM_REFERENCE_LAG_FRACTIONS, t_refs, e_refs, m_refs)
        ]

    # --- SHARED-ACROSS-BOTH-BLOCKS axis ranges for the two new panels ------
    # (explicit set_xlim/set_ylim below; comparing the MIN block against the
    # MAX block is the whole point of the figure, so these must not float.)
    h_smooth = np.linspace(0.0, LAG_MAX_M, 300)
    model_smooth = spherical_semivariance(h_smooth)
    vario_stack = [truth_vario["semivariance_empirical"].values, model_smooth]
    for rd in block_rows:
        vario_stack.append(rd["vario_example"]["semivariance_empirical"].values)
        vario_stack.append(rd["vario_mean"]["semivariance_empirical"].values)
    vario_xlim = (0.0, LAG_MAX_M)
    vario_ylim = (0.0, float(np.max(np.concatenate(vario_stack))) * 1.10)

    dist_all = np.concatenate([
        np.asarray(a, dtype=float).ravel()
        for rd in block_rows
        for a in (rd["truth_masked"], rd["example_masked"], rd["mean_masked"], rd["sample_values"])
    ])
    dist_pad = 0.03 * (float(dist_all.max()) - float(dist_all.min()))
    dist_xlim = (float(dist_all.min()) - dist_pad, float(dist_all.max()) + dist_pad)

    fig = plt.figure(figsize=(21.0, 22.5))
    for block_idx, row_data in enumerate(block_rows):
        gs_top, gs_bottom = BLOCK_GEOMETRY[block_idx]

        # Tinted, outlined background band so the 2 rows of one replicate
        # read as ONE block (zorder=-1 -> drawn under the axes).
        fig.patches.append(plt.Rectangle(
            (0.008, gs_bottom - BLOCK_BAND_PAD_BOTTOM), 0.984,
            (gs_top + BLOCK_BAND_PAD_TOP) - (gs_bottom - BLOCK_BAND_PAD_BOTTOM),
            transform=fig.transFigure, facecolor=BLOCK_BAND_FACECOLORS[block_idx],
            edgecolor="0.55", linewidth=1.2, zorder=-1,
        ))
        gs = fig.add_gridspec(2, 4, top=gs_top, bottom=gs_bottom, **BLOCK_GRID_KWARGS)
        ax_truth = fig.add_subplot(gs[0, 0])
        ax_example = fig.add_subplot(gs[0, 1])
        ax_mean = fig.add_subplot(gs[0, 2])
        ax_var = fig.add_subplot(gs[0, 3])
        ax_umg = fig.add_subplot(gs[1, 0])
        ax_crossplot = fig.add_subplot(gs[1, 1])
        ax_vario = fig.add_subplot(gs[1, 2])
        ax_dist = fig.add_subplot(gs[1, 3])

        fig.text(
            0.5, gs_top + 0.013,
            f"[{row_data['role']}-length replicate] {row_data['replicate_id']} "
            f"(sample_seed={row_data['sample_seed']}, n={len(row_data['samples_df'])} "
            f"conditioning samples) -- fitted length scale = "
            f"{row_data['length_scale_m']:.1f} m  |  UMG={row_data['umg_recomputed']:.4f}, "
            f"MSE={row_data['mse_recomputed']:.4f}  |  top sub-row: maps, "
            "bottom sub-row: diagnostics",
            ha="center", va="bottom", fontsize=13, fontweight="bold", color="black",
        )

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
            f"Block variance scale: [0, {row_data['var_row_max']:.2f}] %^2 "
            "(THIS REPLICATE BLOCK ONLY -- NOT shared with the other block; see module "
            "docstring).",
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

        # ---- 7. Variogram reproduction --------------------------------
        vario_curves = [
            ("truth field", truth_vario, CURVE_COLORS["truth"], "o", "-"),
            (row_data["example_short_label"], row_data["vario_example"],
             CURVE_COLORS["example"], "s", "-"),
            (row_data["mean_short_label"], row_data["vario_mean"],
             CURVE_COLORS["mean"], "^", "-"),
        ]
        for curve_label, vdf, color, marker, linestyle in vario_curves:
            well = vdf[vdf["n_pairs"] >= MIN_PAIRS_PER_BIN]
            sparse = vdf[vdf["n_pairs"] < MIN_PAIRS_PER_BIN]
            ax_vario.plot(
                well["lag_bin_center_m"], well["semivariance_empirical"],
                color=color, marker=marker, linestyle=linestyle, markersize=3.5,
                linewidth=1.4, label=curve_label, alpha=0.9,
            )
            if len(sparse) > 0:
                ax_vario.scatter(
                    sparse["lag_bin_center_m"], sparse["semivariance_empirical"],
                    color=color, marker="x", s=22, alpha=0.35,
                    label=f"{curve_label}, n_pairs<{MIN_PAIRS_PER_BIN}",
                )
        ax_vario.plot(
            h_smooth, model_smooth, color=CURVE_COLORS["model"], linestyle="--",
            linewidth=2, label="truth theoretical spherical model",
        )
        ax_vario.axhline(
            POR_STDEV ** 2, color="gray", linestyle=":", linewidth=1.2,
            label=f"truth total sill ({POR_STDEV ** 2:g})",
        )
        ax_vario.set_xlim(vario_xlim)
        ax_vario.set_ylim(vario_ylim)
        ax_vario.set_xlabel("Lag h (m)")
        ax_vario.set_ylabel("Semivariance (Porosity %$^2$)")
        ax_vario.set_title(
            f"[{row_data['role']}] Variogram of each map (full 50x50 grid)\n"
            f"all curves: SAME {N_PAIRS_DRAWN:,} (i,j) pairs, VARIOGRAM_PAIR_SEED="
            f"{VARIOGRAM_PAIR_SEED}, {LAG_BIN_WIDTH_M:g} m bins",
            fontsize=8.5,
        )
        ax_vario.legend(loc="lower right", fontsize=6.8)
        ax_vario.grid(alpha=0.3)

        ref_lines = []
        for k, ref in enumerate(row_data["vario_reference"]):
            ref_lines.append(
                f"h~{ref['bin_center_m']:.0f} m bin ({ref['frac_of_range']:g}x range "
                f"{HMAJ1:g} m): model {ref['theoretical']:.2f} | truth "
                f"{ref['truth']:.2f} ({ref['truth'] / ref['theoretical']:.2f}x model) | "
                f"example {ref['example']:.2f} ({ref['example'] / ref['theoretical']:.2f}x) "
                f"| pred mean {ref['mean']:.2f} ({ref['mean'] / ref['theoretical']:.2f}x)"
            )
        ax_vario.text(
            0.02, 0.985, "\n".join(textwrap.fill(line, 58) for line in ref_lines),
            transform=ax_vario.transAxes, fontsize=6.2, va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, linewidth=0.4),
        )

        # ---- 8. Distribution reproduction -----------------------------
        dist_curves = [
            ("truth field", row_data["truth_masked"], CURVE_COLORS["truth"], "-"),
            (row_data["example_short_label"], row_data["example_masked"],
             CURVE_COLORS["example"], "-"),
            (row_data["mean_short_label"], row_data["mean_masked"],
             CURVE_COLORS["mean"], "-"),
            ("conditioning samples", row_data["sample_values"], CURVE_COLORS["samples"], "--"),
        ]
        for curve_label, values, color, linestyle in dist_curves:
            xs, ps = _empirical_cdf(values)
            ax_dist.plot(
                xs, ps, color=color, linestyle=linestyle, linewidth=1.6,
                label=curve_label, alpha=0.9,
            )
        ax_dist.set_xlim(dist_xlim)
        ax_dist.set_ylim(0.0, 1.0)
        ax_dist.set_xlabel("Porosity (%)")
        ax_dist.set_ylabel("Cumulative probability")
        ax_dist.set_title(
            f"[{row_data['role']}] Distribution of each map (CDF)\n"
            f"maps on the {row_data['n_evaluated_cells']} evaluated cells; samples = the "
            f"{len(row_data['sample_values'])} excluded conditioning cells",
            fontsize=8.5,
        )
        ax_dist.legend(loc="lower right", fontsize=6.8)
        ax_dist.grid(alpha=0.3)

        stats_lines = [f"mean / std (ddof={DIST_STD_DDOF})"]
        for stats_key, stats_label in [
            ("truth", "truth"), ("example", "example"),
            ("mean", "pred mean"), ("samples", "samples"),
        ]:
            s = row_data["dist_stats"][stats_key]
            stats_lines.append(
                f"{stats_label:<10s}{s['mean']:6.2f} / {s['std']:5.2f}  (n={s['n']})"
            )
        ax_dist.text(
            0.02, 0.985, "\n".join(stats_lines), transform=ax_dist.transAxes,
            fontsize=6.8, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, linewidth=0.4),
        )

        print(
            f"  {method} {row_data['role']} ({row_data['replicate_id']}, sample_seed="
            f"{row_data['sample_seed']}): length_scale={row_data['length_scale_m']:.4f} m, "
            f"UMG recomputed={row_data['umg_recomputed']:.6f} vs. pinned metrics.csv="
            f"{row_data['umg_pinned']:.6f} (match); MSE recomputed="
            f"{row_data['mse_recomputed']:.6f} vs. pinned metrics.csv="
            f"{row_data['mse_pinned']:.6f} (match)"
        )
        print(
            f"    distribution ({row_data['n_evaluated_cells']} evaluated cells; samples = "
            f"{len(row_data['sample_values'])} excluded conditioning cells), mean/std "
            f"(ddof={DIST_STD_DDOF}): "
            + ", ".join(
                f"{k}={row_data['dist_stats'][k]['mean']:.3f}/"
                f"{row_data['dist_stats'][k]['std']:.3f}"
                for k in ("truth", "example", "mean", "samples")
            )
        )
        for ref in row_data["vario_reference"]:
            print(
                f"    variogram @ h~{ref['bin_center_m']:.1f} m bin "
                f"({ref['frac_of_range']:g}x range {HMAJ1:g} m): model={ref['theoretical']:.3f}, "
                f"truth={ref['truth']:.3f} ({100 * ref['truth'] / ref['theoretical']:.1f}% of "
                f"model), example={ref['example']:.3f} "
                f"({100 * ref['example'] / ref['theoretical']:.1f}%), "
                f"pred_mean={ref['mean']:.3f} "
                f"({100 * ref['mean'] / ref['theoretical']:.1f}%)"
            )

    length_definition_note = min_row["length_definition"]
    caption = (
        f"{METHOD_LABELS[method]} at the 1% sample-density level (axis_level='1', "
        f"n_samples_requested=25): the MIN-fitted-length-scale replicate (UPPER BLOCK, "
        f"{min_row['replicate_id']}, {min_row['length_scale_m']:.1f} m) vs. the "
        f"MAX-fitted-length-scale replicate (LOWER BLOCK, {max_row['replicate_id']}, "
        f"{max_row['length_scale_m']:.1f} m) among the SAME 10 sample-location replicates "
        f"(sample_seed 1001-1010), same ground-truth field for every replicate (TRUTH_SEED="
        f"{TRUTH_SEED}, range={AXIS_HMAJ1:g} m). Each replicate occupies ONE 2x4 BLOCK on its "
        "own tinted band: top sub-row = maps (truth+samples / example realization / prediction "
        "mean / predictive variance), bottom sub-row = diagnostics (UMG / accuracy crossplot / "
        f"variogram / distribution). Length scale definition: {length_definition_note} "
        "Porosity color scale (truth/example/mean panels) is SHARED across both blocks (same "
        "truth field, same units); variance color scale is PER BLOCK (see the note under each "
        "variance panel) -- see module docstring for why. The two new diagnostic panels' axis "
        "ranges are SHARED across both blocks (explicit set_xlim/set_ylim over both blocks' "
        "curves) so the blocks can be compared directly. UMG values plotted/annotated are "
        "recomputed here from this run's own predictive mean/variance maps and are verified "
        "(np.isclose) to match results/processed/sample_replicate_axis/metrics.csv's pinned "
        "value for the same (axis_level, replicate, method, metric='umg') row. The accuracy "
        "crossplot panel plots truth vs. that replicate's point estimate -- "
        f"{METHOD_POINT_ESTIMATE_FILES[method]}, which for rbf_bootstrap is the SINGLE-FIT "
        "estimate and deliberately NOT the ensemble-mean array shown in the prediction-mean "
        "panel (see module docstring) -- masked to that replicate's own evaluated cells (its "
        "conditioning samples excluded), with axis limits from the truth field's own [min, max] "
        "(5% padding) and MSE recomputed and verified (np.isclose) to match metrics.csv's "
        "pinned value for the same (axis_level, replicate, method, metric='mse') row. "
        "VARIOGRAM panel: the experimental variogram of the truth field, of the SAME example "
        "realization/draw shown in the example panel, and of the SAME prediction mean shown in "
        "the prediction-mean panel, plus the truth's theoretical spherical model (nugget="
        f"{NUG * POR_STDEV ** 2:g}, structured sill={CC1 * POR_STDEV ** 2:g}, range={HMAJ1:g} m, "
        "imported from src.experiments.base_case, not fitted and not hardcoded) and the truth "
        f"total sill ({POR_STDEV ** 2:g}). All four variogram curves use the IDENTICAL "
        f"{N_PAIRS_DRAWN:,} randomly drawn (i,j) cell pairs (VARIOGRAM_PAIR_SEED="
        f"{VARIOGRAM_PAIR_SEED}, re-seeded inside the shared estimator on every call, verified "
        f"here by comparing bin centres and per-bin pair counts) and the same {LAG_BIN_WIDTH_M:g} m "
        f"lag bins over 0-{LAG_MAX_M:g} m, computed on the FULL {NX}x{NY} grid (the shared "
        "estimator, reused from results/processed/sample_density_axis/"
        "make_length_and_variogram_figures.py, takes no mask) -- i.e. on a DIFFERENT cell set "
        "from the masked, conditioning-cell-excluded support used by the UMG/MSE/distribution "
        "panels. DISTRIBUTION panel: empirical CDFs (plotting position (i-0.5)/n) of the truth "
        "field, the same example realization/draw, and the same prediction mean, all on that "
        "replicate's OWN evaluated cells (identical mask to the UMG/MSE computations, its "
        "conditioning cells excluded), plus that replicate's conditioning-sample values, which "
        "by construction sit at exactly those excluded cells; the mean and standard deviation "
        f"(ddof={DIST_STD_DDOF}) of all four are printed in the panel. NOTE ON READING THE "
        "VARIOGRAM/DISTRIBUTION PANELS: neither method plotted here is designed to reproduce "
        "the truth's spatial statistics -- GP-MLE's posterior draw reproduces GP-MLE's OWN "
        "MLE-fitted gaussian RBF kernel plus nugget, and RBF+bootstrap's replicate map is a "
        "resampled-interpolation result, not a conditional simulation -- and a prediction mean "
        "is an estimator, whose smoothing (and hence lower variance than the truth) is a "
        "generic property of estimation. These panels state what was compared with what and "
        "report the numeric differences; they make no claim about cause or about any method "
        "failing."
    )

    plt.suptitle(
        f"{METHOD_LABELS[method]}: fitted-length-scale extremes across 10 sample-location "
        "replicates at the 1% sample-density level",
        fontsize=16, y=0.985,
    )
    fig.text(
        0.5, 0.012, textwrap.fill(caption, 210),
        ha="center", va="top", fontsize=8, color="dimgray",
    )

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

    # The ground-truth field is the SAME for every replicate and both methods
    # on this axis (TRUTH_SEED, AXIS_HMAJ1/AXIS_HMIN1), so its experimental
    # variogram is computed ONCE here and reused in all 4 variogram panels.
    # The imported estimator re-seeds RandomState(VARIOGRAM_PAIR_SEED) per
    # call, so this curve is on the same pairs/bins as every map curve.
    print(
        f"\nGround-truth experimental variogram (computed once; shared by both methods and "
        f"both blocks; truth_seed={TRUTH_SEED}, hmaj1={AXIS_HMAJ1:g}, hmin1={AXIS_HMIN1:g}); "
        f"truth field mean={float(np.mean(truth)):.4f}, variance (ddof=0)="
        f"{float(np.var(truth)):.4f} (theoretical total sill={POR_STDEV ** 2:g})"
    )
    truth_vario = compute_map_variogram(truth, "ground-truth field")

    saved = []
    for method in ["gp_mle", "rbf_bootstrap"]:
        print(f"\nBuilding case-study figure for {method}...")
        saved.append(
            make_case_study_figure(
                method, truth, source_runs, length_df, metrics_df, truth_vario
            )
        )

    print("\nAll length-extreme case-study figures:")
    for f in saved:
        print(" ", f)
    return saved


if __name__ == "__main__":
    main()
