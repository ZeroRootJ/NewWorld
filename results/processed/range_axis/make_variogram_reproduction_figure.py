"""Variogram reproduction of the REALIZATIONS of SGS and GP-MLE across the
RANGE axis (docs/experiment_context.md deliverable 2): 8 ground-truth
variogram ranges, 100-800 m in 100 m steps (src.experiments.range_axis
ALL_RANGE_VALUES), isotropic, one ground-truth realization per level
(TRUTH_SEED=101), 10 method realizations per level.

This is the RANGE-axis port of
results/processed/sample_replicate_axis/make_variogram_reproduction_figure.py
(sample-density / sample-replicate axes, 2026-09-21). Everything that is not
range-axis-specific is IMPORTED from that script or from the estimator module
it in turn imports, not re-implemented: the experimental-variogram estimator
and the theoretical spherical model
(results/processed/sample_density_axis/make_length_and_variogram_figures.py:
``compute_experimental_variogram`` / ``spherical_semivariance``), the GP
posterior reconstruction + draw + validation
(``rebuild_fitted_gpr`` / ``draw_gp_posterior`` / ``validate_gp_reconstruction``),
the lag-bin reporting helper ``semivariance_at``, and the alpha / linewidth /
colour / y-limit / footnote constants.

WHY ONLY SGS AND GP-MLE (stated on the figures too, not only here)
------------------------------------------------------------------
This artifact compares VARIOGRAM REPRODUCTION, which is a property of a set
of realizations drawn from a method's own uncertainty model.
  * simple kriging is excluded because it produces ONE smooth estimate and no
    realizations at all -- it has no variogram-reproduction behaviour to show
    (its estimate is deliberately smoother than the truth by construction).
  * RBF+bootstrap is excluded because its 10 stored maps are RESAMPLED
    INTERPOLATIONS (the RBF refit on a bootstrap resample of the conditioning
    samples), NOT conditional simulations of a spatial random function. Their
    spread describes sample-set variability, so putting them beside SGS/GP
    realizations on a "does the realization reproduce the truth's variogram?"
    axis would compare two different objects. (The replicate-axis version of
    this artifact does show RBF+bootstrap, explicitly labelled as resampled
    interpolations; here the question is specifically about the two methods
    that DO define a full spatial distribution.)

WHAT "REALIZATION" MEANS PER METHOD
-----------------------------------
  SGS     the 10 conditional simulations in sgs_realizations.npy
          (shape (10, 50, 50)) of the run pinned for that level in
          results/processed/range_axis/source_runs.json.
  GP-MLE  10 joint draws from the fitted GP posterior over the full 50x50
          grid. Each run stores only ONE draw (posterior_sample_map.npy), so
          the 10 draws per level are produced here on the fly (nothing is
          written under results/raw, nothing cached) by the IDENTICAL
          procedure the replicate-axis script uses: rebuild the FITTED GP from
          the run's manifest (fitted hyperparameters, normalize_y) and its
          samples.csv WITHOUT re-optimising (fixed kernel, optimizer=None),
          then gpr.sample_y(grid_coords, n_samples=10) with the same manual
          Cholesky fallback. sklearn's predict(return_cov=True), which
          sample_y uses, includes the fitted WhiteKernel noise on the
          covariance diagonal, so each draw (like the stored one) carries that
          white-noise component -- this is a fact about the method, not a
          choice made here.

GP RECONSTRUCTION VALIDATION (every level; raises on failure)
--------------------------------------------------------------
``validate_gp_reconstruction`` is imported and run UNCHANGED, i.e. exactly the
checks the replicate-axis artifact performs:
  (i)   reconstructed posterior mean / variance maps vs. the run's stored
        posterior_mean_map.npy / posterior_var_map.npy (np.allclose,
        rtol=atol=1e-6);
  (ii)  the CHI-SQUARE CONSISTENCY TEST: the stored draw x must be a plausible
        draw from the reconstructed posterior N(m, C),
        q = (x-m)^T C^-1 (x-m) ~ chi2(2500), |q-2500|/sqrt(2*2500) <= 4. Exact
        equality of the seed-55 draw with the stored draw is NOT required and
        generally does not hold: the posterior covariance has a
        near-degenerate white-noise eigenvalue cluster, so the SVD factor
        sample_y uses is an arbitrary rotation within that subspace (see the
        replicate-axis docstring for the empirical demonstration). max|diff|
        and correlation of the seed-55 draw vs. the stored one, and the
        eigenvalue-cluster size, are printed and recorded per level;
  (iii) determinism: the seed-55 draw is made twice for the first level and
        required to be bit-identical.

RANGE-AXIS SPECIFICS (the four things that differ from the replicate axis)
--------------------------------------------------------------------------
1. THE THEORETICAL MODEL IS PER LEVEL. ``spherical_semivariance`` is called
   with ``range_m=<that level's range>`` (a defaulted keyword added to the
   shared function on 2026-09-22; with no argument it is bit-identical to
   before, so the density/replicate artifacts are unaffected). Only the range
   moves: nugget = NUG*POR_STDEV**2 = 0.45 and structured sill =
   CC1*POR_STDEV**2 = 8.55 (total sill 9.0) are base-case constants at every
   level by this axis's one-factor-at-a-time design, and that invariance is
   asserted at run time.
2. THE TRUTH FIELD IS PER LEVEL. Each level's truth is regenerated with
   get_base_case_truth(truth_seed=101, hmaj1=hmin1=<level range>) and its own
   experimental variogram is the solid black line in that level's panels. The
   regenerated truth is also used to VERIFY (not assume) that both pinned runs
   of that level really belong to it: their samples.csv must match the
   conditioning samples redrawn from that truth at SAMPLE_SEED=20 to
   SAMPLES_MATCH_ATOL, and their manifest hmaj1/hmin1 must equal the level.
3. LAG RANGE = 0-1000 m (LAG_MAX_M_RANGE_AXIS), wider than the 0-750 m used on
   the density / replicate axes, and the SAME for all 8 levels so the panels
   are directly comparable. Why 1000 m: (a) the largest ground-truth range on
   this axis is 800 m, and a spherical model only reaches its sill AT the
   range, so the window must extend past 800 m for the 800 m level's plateau
   to be visible at all -- 750 m would have cut the top level off before its
   sill; (b) 1000 m is the domain side length (50 cells x 20 m), beyond which
   only near-diagonal corner pairs remain and the pair count collapses (the
   975-1000 m bin holds 1709 of the 199,905 drawn pairs, the 1175-1200 m bin
   only 141), so lags past the side length are dominated by a shrinking,
   geometrically special subset of the field. The domain diagonal is 1414 m
   but is not a usable window for that reason. The bin width is UNCHANGED at
   LAG_BIN_WIDTH_M = 25 m (40 bins), so a curve here is directly comparable to
   the same curve in the other axes' figures, and the MIN_PAIRS_PER_BIN = 30
   guard is kept as on the other axes (measured minimum over the 40 bins: 345
   pairs, in the 0-25 m bin; no bin is flagged).
4. THE REPORTED LAG IS PER LEVEL. The density/replicate artifacts report the
   287.5 m bin, i.e. the last 25 m bin BELOW the (then fixed) 300 m range. The
   range-axis correspondent is the last bin below THAT LEVEL's own range:
   REPORT_LAG(R) = R - LAG_BIN_WIDTH_M/2 = 87.5, 187.5, ..., 787.5 m, each
   compared against that level's own theoretical model value at the same lag.

Y-AXIS
------
Fixed at 0 to Y_MAX_SILL_FACTOR x total sill = 1.3 x 9 = 11.7 in every panel,
the same rule and the same numbers as every other variogram-reproduction
figure in this project (the total sill is 9.0 at every level of this axis, so
the rule transfers unchanged). Curves exceeding it are cut off by the axis --
the data are neither clipped nor altered. The share of plotted curve points
above the limit and the number of realizations with at least one bin above it
are printed, recorded per (level, method) in the seeds JSON, and stated in the
figure footnotes whenever they are non-zero.

SEEDS (all deterministic; no other randomness)
----------------------------------------------
  TRUTH_SEED          = 101  (one ground-truth realization per level)
  SAMPLE_SEED         = 20   (conditioning sample locations, identical across
                              levels and across the two methods)
  VARIOGRAM_PAIR_SEED = 90   (200,000 random (i,j) cell pairs, shared
                              estimator; every curve on this axis -- truth
                              included -- is evaluated on the IDENTICAL pair
                              set and the same 25 m bins)
  GP_SAMPLE_SEED      = 55   (validation draw only; the stored runs' own seed)
  GP_DRAW_SEED_BY_LEVEL: a NEW block reserved for the range axis,
                              GP_DRAW_SEED_BASE + level index = 9600..9607 for
                              100..800 m. The replicate axis already uses
                              9000+k / 9100+k / 9200+k / 9300+k / 9400+k for
                              its 1/2/5/10/20% levels, so 96xx cannot collide
                              with any of them. One sample_y(n_samples=10) call
                              per level.

OUTPUTS
-------
  results/figures/range_axis/variogram_reproduction_by_range.png
      8 rows (range levels, 100 m at the top) x 2 columns (SGS | GP-MLE).
      Each panel: that level's 10 realizations at low alpha in the method
      colour, that level's TRUTH experimental variogram as a solid black line,
      that level's THEORETICAL spherical model dashed, the total sill dotted,
      and a vertical marker at that level's own ground-truth range. No mean /
      ensemble-mean variogram is drawn anywhere (project convention).
  results/figures/range_axis/variogram_reproduction_pooled_by_range.png
      1 row x 2 panels (SGS | GP-MLE); each pools all 8 levels x 10 = 80
      curves, coloured by ground-truth range (viridis, the same level colormap
      make_gp_realization_figures.py uses), with each level's truth curve and
      theoretical model drawn in the matching colour.
  results/processed/range_axis/variogram_reproduction_summary.csv
      tidy long, ONE ROW PER (level, method, realization):
      axis_level, truth_range_m, method, realization_index, n_realizations,
      lag_m, model_semivariance_at_lag, truth_semivariance_at_lag,
      value_semivariance, value_pct_of_model, value_pct_of_truth. The truth is
      one row per level with method='truth', an empty realization_index and
      value_pct_of_truth = 100 by definition.
      value_pct_of_model is the requested quantity (the direct range-axis
      correspondent of the density axis's 287.5 m % -of-model column);
      value_pct_of_truth is carried alongside it because on this axis the
      TRUTH itself departs strongly from its own theoretical model at the long
      ranges (90% of model at 300 m up to 130% at 700 m -- one realization of
      a long-range field carries a large-scale trend), so "% of model" mixes
      that departure into every method number and "% of truth" isolates
      "did the realization reproduce THIS field". Neither is dropped; which
      one to read is left to the reader.
  results/processed/range_axis/variogram_reproduction_seeds.json
      seeds, source runs, per-level GP validation, lag-window choice, y-limit
      rule and exceedance statistics, timings.

Run with (~a few minutes; dominated by the 8 GP posterior draws and the 8
eigen-decompositions in the validation):
.venv/Scripts/python.exe -m results.processed.range_axis.make_variogram_reproduction_figure
Optional: --levels 100 800   (only those levels are computed; the CSV/JSON are
rewritten for the levels run, other levels' rows and blocks are kept)
"""

import contextlib
import io
import json
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.base_case import (  # noqa: E402
    CC1, NUG, NX, NY, POR_STDEV, XMN, XSIZ, YMN, YSIZ,
)
from src.experiments.base_case_conditioning import (  # noqa: E402
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.gp_mle import GP_SAMPLE_SEED  # noqa: E402
from src.experiments.range_axis import ALL_RANGE_VALUES  # noqa: E402
from src.grid_utils import full_grid_coordinates  # noqa: E402

# --- the shared variogram estimator + theoretical model (single source) -----
from results.processed.sample_density_axis.make_length_and_variogram_figures import (  # noqa: E402
    LAG_BIN_WIDTH_M,
    MIN_PAIRS_PER_BIN,
    N_PAIRS_DRAWN,
    VARIOGRAM_PAIR_SEED,
    compute_experimental_variogram,
    spherical_semivariance,
)
# --- the replicate-axis artifact: GP reconstruction + every styling constant -
from results.processed.sample_replicate_axis.make_length_case_study_figures import (  # noqa: E402
    CURVE_COLORS,
    FIG_DPI,
    SAMPLES_MATCH_ATOL,
)
from results.processed.sample_replicate_axis.make_variogram_reproduction_figure import (  # noqa: E402
    ALPHA,
    EIG_CLUSTER_REL_TOL,
    FOOTNOTE_FONTSIZE,
    FOOTNOTE_LINE_HEIGHT_IN,
    GP_CHI2_MAX_ABS_Z,
    GP_MAP_ATOL,
    GP_MAP_RTOL,
    LABEL_FONTSIZE,
    LEGEND_SWATCH_ALPHA,
    METHOD_COLORS,
    METHOD_LABELS,
    N_GP_DRAWS,
    N_REALIZATIONS_EXPECTED,
    PHYS_RANGE,
    POOLED_ALPHA,
    POOLED_LINEWIDTH,
    REALIZATION_LINEWIDTH,
    TICK_FONTSIZE,
    TRUTH_LINEWIDTH,
    TRUTH_MARKERSIZE,
    YLIM,
    Y_MAX,
    Y_MAX_SILL_FACTOR,
    draw_gp_posterior,
    rebuild_fitted_gpr,
    semivariance_at,
    validate_gp_reconstruction,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "range_axis"
FIGURE_A_NAME = "variogram_reproduction_by_range.png"
FIGURE_B_NAME = "variogram_reproduction_pooled_by_range.png"
SUMMARY_CSV_NAME = "variogram_reproduction_summary.csv"
SEEDS_JSON_NAME = "variogram_reproduction_seeds.json"

# --- axis definition --------------------------------------------------------
LEVELS = tuple(str(int(r)) for r in ALL_RANGE_VALUES)      # "100" .. "800"
LEVEL_RANGE_M = {lv: float(lv) for lv in LEVELS}
METHODS = ("sgs", "gp_mle")     # figure columns, left to right (see docstring)

# --- range-axis lag window (see docstring point 3) --------------------------
LAG_MAX_M_RANGE_AXIS = 1000.0

# --- range-axis GP draw seed block (see docstring SEEDS) --------------------
GP_DRAW_SEED_BASE = 9600
GP_DRAW_SEED_BY_LEVEL = {lv: GP_DRAW_SEED_BASE + i for i, lv in enumerate(LEVELS)}

# --- reported lag: the last 25 m bin BELOW that level's own range -----------
REPORT_LAG_BY_LEVEL = {lv: LEVEL_RANGE_M[lv] - 0.5 * LAG_BIN_WIDTH_M for lv in LEVELS}

TOTAL_SILL = POR_STDEV ** 2                 # 9.0, fixed on this axis
TRUTH_NUGGET_REAL = NUG * POR_STDEV ** 2    # 0.45, fixed on this axis
TRUTH_STRUCT_SILL_REAL = CC1 * POR_STDEV ** 2  # 8.55, fixed on this axis

LEVEL_CMAP = plt.get_cmap("viridis")        # same level colormap as make_gp_realization_figures
# The top of viridis is a very light yellow that is hard to read as a thin line
# on white, so the 8 levels are mapped into [0, 0.88] of the colormap instead of
# its full span; the ORDER and the perceptual spacing are unchanged, only the
# brightest end is trimmed. The truth curves additionally carry a dark outline.
LEVEL_CMAP_MAX = 0.88
# Alpha of the pooled panels' realization lines. The imported POOLED_ALPHA
# (=0.12) was chosen on the replicate axis for 100 curves drawn in ONE colour
# per panel; here the 80 curves are split into 8 COLOUR GROUPS of 10 (one per
# range level), i.e. the same per-colour density as figure A's panels
# (ALPHA=0.25), and at 0.12 the level grouping -- which is the whole point of
# the pooled figure -- is barely visible under the reference curves. The value
# is therefore set here and stated on the figure; the imported POOLED_ALPHA is
# left untouched for the axes that use it.
POOLED_ALPHA_RANGE_AXIS = 0.22
TRUTH_STROKE_COLOR = "0.15"
RANGE_MARKER_COLOR = "0.35"
LEGEND_REALIZATION_SWATCH_COLOR = "0.35"

# --- figure geometry (this figure is 8x2, not the replicate axis's 10x3) ----
FIG_A_PANEL_W_IN = 5.6
FIG_A_PANEL_H_IN = 2.85
FIG_A_TITLE_IN = 0.78        # two-line suptitle
# The 4 curve-role legend entries are set in TWO rows (ncol=2) rather than one:
# in one row the legend is wider than the figure itself, and bbox_inches="tight"
# then expands the saved image around it, leaving the panels stranded in the
# middle ~70% of the width.
FIG_A_LEGEND_NCOL = 2
FIG_A_LEGEND_IN = 0.62
FIG_B_PANEL_W_IN = 7.2
FIG_B_PANEL_H_IN = 5.4
FIG_B_TITLE_IN = 0.62
FIG_B_LEGEND_IN = 0.55       # two legend rows (8 level swatches + curve roles)
FOOTNOTE_PAD_IN = 0.30


# ---------------------------------------------------------------------------
# Variogram helpers (thin wrappers around the SHARED estimator: the only
# difference from the replicate-axis wrappers is the wider lag window and the
# per-level theoretical range, both passed as arguments).
# ---------------------------------------------------------------------------
def quiet_variogram(map_2d: np.ndarray, range_m: float) -> pd.DataFrame:
    """The shared estimator over 0-LAG_MAX_M_RANGE_AXIS with this level's
    theoretical range, with its per-call progress printing suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_experimental_variogram(
            map_2d, lag_max_m=LAG_MAX_M_RANGE_AXIS, range_m=range_m
        )


def realization_variograms(maps: np.ndarray, truth_vario: pd.DataFrame, range_m: float,
                           what: str):
    """One experimental variogram per realization, each required to land on
    the truth's lag bins and the truth's pair counts (same pair set)."""
    out = []
    for k in range(maps.shape[0]):
        v = quiet_variogram(maps[k], range_m)
        if not (
            np.array_equal(v["lag_bin_center_m"].values, truth_vario["lag_bin_center_m"].values)
            and np.array_equal(v["n_pairs"].values, truth_vario["n_pairs"].values)
        ):
            raise ValueError(f"{what} #{k}: variogram not on the truth's lag bins / pair set.")
        out.append(v)
    return out


def model_at(h, range_m: float):
    """That level's theoretical spherical semivariance (shared function)."""
    return spherical_semivariance(h, range_m=range_m)


# ---------------------------------------------------------------------------
# Data loading (one axis level)
# ---------------------------------------------------------------------------
def load_level(level: str, source_runs: dict, grid_coords, check_determinism: bool):
    """Regenerate this level's truth, verify both pinned runs belong to it,
    load the 10 SGS realizations and draw the 10 GP posterior realizations."""
    range_m = LEVEL_RANGE_M[level]
    runs = {m: _REPO_ROOT / source_runs[level][m] for m in METHODS}

    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=range_m, hmin1=range_m)
    ref = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED, n_samples=N_SAMPLES)

    # Verify (not assume) that each run belongs to THIS range level: the
    # conditioning sample VALUES are read off the truth field, so they are
    # range-specific and an exact match pins the run to this ground truth.
    for m, rd in runs.items():
        params = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))["params"]
        if params.get("truth_seed") != TRUTH_SEED or params.get("sample_seed") != SAMPLE_SEED:
            raise ValueError(
                f"range {level} m {m}: manifest truth_seed/sample_seed "
                f"({params.get('truth_seed')}/{params.get('sample_seed')}) != "
                f"({TRUTH_SEED}/{SAMPLE_SEED})."
            )
        # Recorded range, where the manifest has it. The BASE-CASE runs reused
        # as the 300 m level predate the top-level hmaj1 field and carry no
        # variogram dict either (gp_mle has no variogram of its own), so this
        # check is best-effort -- exactly as in range_axis.verify_reused_run;
        # the samples.csv comparison below is the decisive one.
        recorded = params.get("hmaj1")
        if recorded is None:
            recorded = params.get("variogram", {}).get("hmaj1")
        if recorded is not None and not np.isclose(float(recorded), range_m):
            raise ValueError(
                f"range {level} m {m}: manifest records hmaj1={recorded}, expected {range_m:g}."
            )
        if recorded is None:
            print(f"  NOTE: {m} run for range={range_m:g} m records no hmaj1 in its manifest "
                  "(pre-dates that field); identity rests on the samples.csv match below.")
        rec = pd.read_csv(rd / "samples.csv")
        if len(rec) != len(ref) or not np.allclose(
            rec[["X", "Y", VCOL]].values, ref[["X", "Y", VCOL]].values,
            rtol=0.0, atol=SAMPLES_MATCH_ATOL,
        ):
            raise ValueError(
                f"range {level} m {m}: samples.csv does not match the conditioning samples "
                f"regenerated at hmaj1=hmin1={range_m:g}, sample_seed={SAMPLE_SEED}."
            )
    print(f"  verified both pinned runs belong to range={range_m:g} m "
          f"(n_samples_actual={len(ref)})")

    sgs = np.load(runs["sgs"] / "sgs_realizations.npy")
    if sgs.shape != (N_REALIZATIONS_EXPECTED, NY, NX):
        raise ValueError(
            f"range {level} m: sgs_realizations.npy has shape {sgs.shape}, expected "
            f"({N_REALIZATIONS_EXPECTED}, {NY}, {NX}) (full grid)."
        )
    sgs_oor = []
    for k in range(sgs.shape[0]):
        lo, hi = float(np.min(sgs[k])), float(np.max(sgs[k]))
        if lo < PHYS_RANGE[0] or hi > PHYS_RANGE[1]:
            sgs_oor.append({"realization_index": k, "min": lo, "max": hi})
            print(f"  NOTE: SGS realization {k} of range={level} m has values outside the "
                  f"physical range {PHYS_RANGE}: min={lo:.4f}, max={hi:.4f} (kept as is)")

    gpr, _ = rebuild_fitted_gpr(runs["gp_mle"])
    validation = validate_gp_reconstruction(
        gpr, runs["gp_mle"], grid_coords, f"range{level}", check_determinism
    )
    seed = GP_DRAW_SEED_BY_LEVEL[level]
    t0 = time.time()
    gp_draws, method = draw_gp_posterior(gpr, grid_coords, N_GP_DRAWS, seed)
    seconds = time.time() - t0
    print(f"  [GP range{level}] drew {N_GP_DRAWS} posterior draws, seed={seed}, "
          f"method={method}, {seconds:.1f} s")

    return {
        "range_m": range_m,
        "truth": truth,
        "n_actual": len(ref),
        "maps": {"sgs": sgs, "gp_mle": gp_draws},
        "runs": {m: source_runs[level][m] for m in METHODS},
        "sgs_out_of_range": sgs_oor,
        "gp_validation": validation,
        "gp_draw_seed": seed,
        "gp_draw_method": method,
        "gp_draw_seconds": seconds,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_curves(ax, curves, color, alpha, linewidth):
    for v in curves:
        well = v[v["n_pairs"] >= MIN_PAIRS_PER_BIN]
        ax.plot(well["lag_bin_center_m"], well["semivariance_empirical"], color=color,
                linewidth=linewidth, alpha=alpha, zorder=2)


def _plot_truth(ax, truth_vario, color=None, linewidth=None, alpha=1.0, markers=True,
                stroke=False):
    color = CURVE_COLORS["truth"] if color is None else color
    linewidth = TRUTH_LINEWIDTH if linewidth is None else linewidth
    well = truth_vario[truth_vario["n_pairs"] >= MIN_PAIRS_PER_BIN]
    sparse = truth_vario[truth_vario["n_pairs"] < MIN_PAIRS_PER_BIN]
    effects = (
        [path_effects.Stroke(linewidth=linewidth + 1.3, foreground=TRUTH_STROKE_COLOR),
         path_effects.Normal()]
        if stroke else None
    )
    ax.plot(
        well["lag_bin_center_m"], well["semivariance_empirical"], color=color,
        marker="o" if markers else None, markersize=TRUTH_MARKERSIZE, linestyle="-",
        linewidth=linewidth, alpha=alpha, zorder=5, path_effects=effects,
    )
    if len(sparse) > 0:
        ax.scatter(sparse["lag_bin_center_m"], sparse["semivariance_empirical"], color=color,
                   marker="x", s=22, alpha=0.5 * alpha, zorder=5)


def _style_axes(ax):
    ax.set_xlim(0.0, LAG_MAX_M_RANGE_AXIS)
    ax.set_ylim(YLIM)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def _footnote_geometry(text, width_chars):
    wrapped = textwrap.fill(text, width_chars)
    n_lines = wrapped.count("\n") + 1
    return wrapped, n_lines * FOOTNOTE_LINE_HEIGHT_IN + FOOTNOTE_PAD_IN


def _exclusion_sentence():
    return (
        "ONLY SGS AND GP-MLE ARE SHOWN. Simple kriging produces one smooth estimate and no "
        "realizations, so it has no variogram-reproduction behaviour to display. RBF+bootstrap's "
        "10 maps are RESAMPLED INTERPOLATIONS (the RBF refit on a bootstrap resample of the "
        "conditioning samples), not conditional simulations of a spatial random function, so they "
        "answer a different question (sample-set variability) and are not comparable on this axis."
    )


def _common_footnote_text(levels, data, exceed, exceed_truth):
    lags = (f"{min(REPORT_LAG_BY_LEVEL[lv] for lv in levels):g}-"
            f"{max(REPORT_LAG_BY_LEVEL[lv] for lv in levels):g}")
    n_actual = sorted({data[lv]["n_actual"] for lv in levels})
    oor = [(lv, e) for lv in levels for e in data[lv]["sgs_out_of_range"]]
    ylim_sentence = (
        f"Y-axis fixed at 0-{Y_MAX:g} ({Y_MAX_SILL_FACTOR:g} x total sill {TOTAL_SILL:g}, the "
        "same rule as every other variogram-reproduction figure in this project); curves "
        "exceeding it are cut off at the axis, the data are not clipped."
    )
    above = [
        f"{m}: {exceed[m]['pct_points_above_ylim']:.2f}% of points, "
        f"{exceed[m]['n_realizations_with_any_bin_above_ylim']}/{exceed[m]['n_realizations']} "
        "realizations"
        for m in METHODS if exceed[m]["n_points_above_ylim"] > 0
    ]
    ylim_sentence += (
        " Realization curves reaching above the limit -- " + "; ".join(above) + "."
        if above else
        f" No realization curve of either method reaches above {Y_MAX:g} at any level (0 of "
        f"{sum(exceed[m]['n_curve_points'] for m in METHODS)} plotted curve points)."
    )
    if exceed_truth["n_points_above_ylim"] > 0:
        ylim_sentence += (
            f" The TRUTH curve itself is cut off at the "
            f"{', '.join(exceed_truth['levels_with_any_bin_above_ylim'])} m level(s) "
            f"({exceed_truth['n_points_above_ylim']} of {exceed_truth['n_curve_points']} "
            f"plotted truth points = {exceed_truth['pct_points_above_ylim']:.2f}% above the "
            f"limit): a single realization of a long-range field carries a large-scale trend, "
            f"so its experimental variogram overshoots the model sill at lags approaching the "
            f"domain size. The limit is kept fixed anyway, per project convention, so that all "
            f"panels of all axes share one y-axis."
        )
    else:
        ylim_sentence += " No truth curve reaches above the limit at any level."
    if oor:
        ylim_sentence += (
            f" NOTE: {len(oor)} SGS realization(s) ("
            + ", ".join(f"range {lv} m #{e['realization_index']}" for lv, e in oor)
            + f") contain values outside the physical back-transform range "
            f"{PHYS_RANGE[0]:g}-{PHYS_RANGE[1]:g}; nothing was altered."
        )
    return (
        f"RANGE AXIS: the ground-truth variogram range (isotropic, hmaj1=hmin1) is the ONLY "
        f"factor varied -- {', '.join(levels)} m. Each level has its OWN ground-truth field "
        f"(TRUTH_SEED={TRUTH_SEED} at that range) and therefore its own truth curve and its own "
        f"theoretical model; nugget ({TRUTH_NUGGET_REAL:g}), structured sill "
        f"({TRUTH_STRUCT_SILL_REAL:g}), total sill ({TOTAL_SILL:g}), grid, SAMPLE_SEED="
        f"{SAMPLE_SEED} and n_samples ({N_SAMPLES} requested -> "
        f"{'/'.join(str(n) for n in n_actual)} actual) are identical at every level, verified "
        f"against each run's own samples.csv. Every curve is an experimental variogram of a FULL "
        f"{NX}x{NY} grid map from the project's shared estimator: {N_PAIRS_DRAWN:,} random cell "
        f"pairs (VARIOGRAM_PAIR_SEED={VARIOGRAM_PAIR_SEED}, the identical pair set for every "
        f"curve on every level), {LAG_BIN_WIDTH_M:g} m lag bins over 0-{LAG_MAX_M_RANGE_AXIS:g} m "
        f"-- wider than the 0-750 m used on the density/replicate axes because the 800 m level's "
        f"sill lies beyond 750 m; 1000 m is the domain side length, past which only near-diagonal "
        f"pairs remain (min pairs per bin over the 40 bins: 345, guard MIN_PAIRS_PER_BIN="
        f"{MIN_PAIRS_PER_BIN}). Solid black = that level's truth field; thin coloured lines = one "
        f"variogram per realization (SGS = the 10 stored conditional simulations; GP-MLE = 10 "
        f"joint draws from the fitted GP posterior over the full grid, sklearn sample_y, which "
        f"includes the fitted white-noise term, re-drawn here from each run's fitted "
        f"hyperparameters without re-optimising and validated against each run's stored posterior "
        f"mean/variance plus a chi-square consistency test on its stored draw; draw seeds "
        f"{GP_DRAW_SEED_BASE}-{GP_DRAW_SEED_BASE + len(LEVELS) - 1}). NO mean variogram is drawn. "
        f"Dashed = that level's theoretical spherical model, dotted = total sill, vertical dash-"
        f"dot = that level's own ground-truth range; all three are reference lines, not estimates. "
        f"The summary CSV reports each curve at the last lag bin below its own level's range "
        f"({lags} m). {_exclusion_sentence()} Legend swatches are drawn more opaque than the "
        f"plotted lines for legibility. {ylim_sentence}"
    )


def make_figure_a(levels, data, varios, truth_varios, exceed, exceed_truth):
    n_rows = len(levels)
    footnote, foot_in = _footnote_geometry(
        _common_footnote_text(levels, data, exceed, exceed_truth), 165
    )
    panels_h = n_rows * FIG_A_PANEL_H_IN
    height = panels_h + FIG_A_TITLE_IN + FIG_A_LEGEND_IN + foot_in
    width = len(METHODS) * FIG_A_PANEL_W_IN

    fig, axes = plt.subplots(n_rows, len(METHODS), figsize=(width, height),
                             sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(n_rows, len(METHODS))
    h_smooth = np.linspace(0.0, LAG_MAX_M_RANGE_AXIS, 400)
    for i, level in enumerate(levels):
        range_m = LEVEL_RANGE_M[level]
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            curves = varios[(level, method)]
            _plot_curves(ax, curves, METHOD_COLORS[method], ALPHA, REALIZATION_LINEWIDTH)
            _plot_truth(ax, truth_varios[level])
            ax.plot(h_smooth, model_at(h_smooth, range_m), color=CURVE_COLORS["model"],
                    linestyle="--", linewidth=1.8, zorder=3)
            ax.axhline(TOTAL_SILL, color="gray", linestyle=":", linewidth=1.2, zorder=1)
            ax.axvline(range_m, color=RANGE_MARKER_COLOR, linestyle="-.", linewidth=1.1,
                       alpha=0.8, zorder=1)
            _style_axes(ax)
            ax.text(0.025, 0.96, f"{len(curves)} realizations", transform=ax.transAxes,
                    fontsize=8.5, va="top", ha="left", color="dimgray")
            if i == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=13, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel("Lag h (m)", fontsize=LABEL_FONTSIZE)
            if j == 0:
                ax.set_ylabel(
                    f"GT range = {range_m:g} m\nSemivariance (Porosity %$^2$)",
                    fontsize=LABEL_FONTSIZE - 1,
                )

    fig.suptitle(
        "Range axis: variogram of the individual realizations vs. the truth,\n"
        f"by ground-truth variogram range ({levels[0]}-{levels[-1]} m, "
        f"{N_REALIZATIONS_EXPECTED} realizations per level and method)",
        fontsize=13.5, y=1.0 - 0.10 / height,
    )
    fig.tight_layout(rect=(0.0, foot_in / height, 1.0,
                           1.0 - (FIG_A_TITLE_IN + FIG_A_LEGEND_IN) / height))
    fig.legend(
        handles=_role_legend_handles(N_REALIZATIONS_EXPECTED, ALPHA, REALIZATION_LINEWIDTH),
        loc="upper center", bbox_to_anchor=(0.5, 1.0 - FIG_A_TITLE_IN / height),
        ncol=FIG_A_LEGEND_NCOL, fontsize=10, framealpha=0.95,
    )
    fig.text(0.5, 0.004, footnote, ha="center", va="bottom",
             fontsize=FOOTNOTE_FONTSIZE, color="dimgray")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / FIGURE_A_NAME
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _role_legend_handles(n_curves, alpha, linewidth):
    return [
        Line2D([], [], color=CURVE_COLORS["truth"], marker="o", markersize=TRUTH_MARKERSIZE,
               linewidth=TRUTH_LINEWIDTH, label="truth of THAT level (experimental)"),
        Line2D([], [], color=LEGEND_REALIZATION_SWATCH_COLOR, linewidth=linewidth + 0.8,
               alpha=LEGEND_SWATCH_ALPHA,
               label=f"realizations (n={n_curves} per panel, alpha={alpha:g}; colour = method)"),
        Line2D([], [], color=CURVE_COLORS["model"], linestyle="--", linewidth=1.8,
               label="theoretical spherical model of THAT level"),
        Line2D([], [], color=RANGE_MARKER_COLOR, linestyle="-.", linewidth=1.1,
               label="that level's ground-truth range"),
    ]


def make_figure_b(levels, data, varios, truth_varios, exceed, exceed_truth):
    n_per_panel = len(levels) * N_REALIZATIONS_EXPECTED
    footnote_text = (
        f"Pooled over all {len(levels)} ground-truth range levels x "
        f"{N_REALIZATIONS_EXPECTED} realizations = {n_per_panel} curves per panel, drawn at "
        f"alpha={POOLED_ALPHA_RANGE_AXIS:g} and coloured by that curve's own ground-truth range; each "
        "level's truth curve (thick solid, same colour) and theoretical spherical model (thin "
        "dashed, same colour) are drawn in the matching colour, and the vertical dash-dot lines "
        "mark the 8 ground-truth ranges. "
        + _common_footnote_text(levels, data, exceed, exceed_truth)
    )
    footnote, foot_in = _footnote_geometry(footnote_text, 175)
    height = FIG_B_PANEL_H_IN + FIG_B_TITLE_IN + FIG_B_LEGEND_IN + foot_in
    width = len(METHODS) * FIG_B_PANEL_W_IN

    colors = {lv: LEVEL_CMAP(_level_norm(LEVEL_RANGE_M[lv])) for lv in levels}
    fig, axes = plt.subplots(1, len(METHODS), figsize=(width, height), sharex=True, sharey=True)
    h_smooth = np.linspace(0.0, LAG_MAX_M_RANGE_AXIS, 400)
    for j, method in enumerate(METHODS):
        ax = axes[j]
        for level in levels:
            range_m = LEVEL_RANGE_M[level]
            ax.axvline(range_m, color=colors[level], linestyle="-.", linewidth=0.9,
                       alpha=0.45, zorder=1)
            _plot_curves(ax, varios[(level, method)], colors[level], POOLED_ALPHA_RANGE_AXIS,
                         POOLED_LINEWIDTH)
        for level in levels:
            range_m = LEVEL_RANGE_M[level]
            ax.plot(h_smooth, model_at(h_smooth, range_m), color=colors[level],
                    linestyle="--", linewidth=1.0, alpha=0.6, zorder=4)
            _plot_truth(ax, truth_varios[level], color=colors[level], linewidth=1.8,
                        markers=False, stroke=True)
        ax.axhline(TOTAL_SILL, color="gray", linestyle=":", linewidth=1.2, zorder=1)
        _style_axes(ax)
        ax.set_title(
            f"{METHOD_LABELS[method]}  ({n_per_panel} realizations = {len(levels)} range "
            f"levels x {N_REALIZATIONS_EXPECTED})", fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Lag h (m)", fontsize=LABEL_FONTSIZE)
        if j == 0:
            ax.set_ylabel("Semivariance (Porosity %$^2$)", fontsize=LABEL_FONTSIZE)

    fig.suptitle(
        "Range axis: variogram of the individual realizations, pooled over all "
        f"{len(levels)} ground-truth range levels (colour = ground-truth range)",
        fontsize=14, y=1.0 - 0.14 / height,
    )
    fig.tight_layout(rect=(0.0, foot_in / height, 1.0,
                           1.0 - (FIG_B_TITLE_IN + FIG_B_LEGEND_IN) / height))
    level_handles = [
        Line2D([], [], color=colors[lv], linewidth=2.6, label=f"GT range {lv} m")
        for lv in levels
    ]
    role_handles = [
        Line2D([], [], color="0.35", linewidth=2.2, label="that level's truth (thick solid)"),
        Line2D([], [], color="0.35", linestyle="--", linewidth=1.1,
               label="that level's theoretical model (thin dashed)"),
        Line2D([], [], color="0.35", linewidth=POOLED_LINEWIDTH + 0.8,
               alpha=LEGEND_SWATCH_ALPHA,
               label=f"realizations ({n_per_panel} per panel, alpha={POOLED_ALPHA_RANGE_AXIS:g})"),
        Line2D([], [], color="0.35", linestyle="-.", linewidth=0.9,
               label="the 8 ground-truth ranges"),
    ]
    fig.legend(handles=level_handles + role_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0 - FIG_B_TITLE_IN / height), ncol=6,
               fontsize=9, framealpha=0.95)
    fig.text(0.5, 0.004, footnote, ha="center", va="bottom",
             fontsize=FOOTNOTE_FONTSIZE, color="dimgray")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / FIGURE_B_NAME
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _level_norm(range_m: float) -> float:
    lo, hi = min(ALL_RANGE_VALUES), max(ALL_RANGE_VALUES)
    return LEVEL_CMAP_MAX * (range_m - lo) / (hi - lo)


# ---------------------------------------------------------------------------
def main(levels=LEVELS):
    t_start = time.time()
    levels = tuple(levels)
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)
    missing = [lv for lv in levels if lv not in source_runs]
    if missing:
        raise ValueError(f"source_runs.json has no entry for level(s) {missing}.")

    # The range axis holds the nugget / sill split fixed -- assert, do not assume.
    if abs((TRUTH_NUGGET_REAL + TRUTH_STRUCT_SILL_REAL) - TOTAL_SILL) > 1e-12:
        raise ValueError(
            f"nugget {TRUTH_NUGGET_REAL} + structured sill {TRUTH_STRUCT_SILL_REAL} != total "
            f"sill {TOTAL_SILL}; the fixed y-axis rule (1.3 x total sill) assumes they do."
        )

    grid_coords = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)
    print(
        f"levels={list(levels)} m, TRUTH_SEED={TRUTH_SEED}, SAMPLE_SEED={SAMPLE_SEED}, "
        f"VARIOGRAM_PAIR_SEED={VARIOGRAM_PAIR_SEED}, GP_SAMPLE_SEED={GP_SAMPLE_SEED}, "
        f"GP_DRAW_SEED_BY_LEVEL={GP_DRAW_SEED_BY_LEVEL}, lag window 0-"
        f"{LAG_MAX_M_RANGE_AXIS:g} m in {LAG_BIN_WIDTH_M:g} m bins"
    )

    data, varios, truth_varios = {}, {}, {}
    for i, level in enumerate(levels):
        print(f"\n======== range = {level} m ========")
        t_lv = time.time()
        data[level] = load_level(level, source_runs, grid_coords, check_determinism=(i == 0))
        range_m = LEVEL_RANGE_M[level]
        truth_varios[level] = quiet_variogram(data[level]["truth"], range_m)
        sparse_bins = int((truth_varios[level]["n_pairs"] < MIN_PAIRS_PER_BIN).sum())
        print(f"  truth variogram: {len(truth_varios[level])} bins, min pairs "
              f"{int(truth_varios[level]['n_pairs'].min())}, "
              f"{sparse_bins} bin(s) below MIN_PAIRS_PER_BIN={MIN_PAIRS_PER_BIN}")
        for method in METHODS:
            varios[(level, method)] = realization_variograms(
                data[level]["maps"][method], truth_varios[level], range_m,
                f"range{level} {method}",
            )
        del data[level]["maps"]      # free memory; only variograms are needed onwards
        del data[level]["truth"]
        data[level]["level_seconds"] = round(time.time() - t_lv, 1)

    # --- y-limit exceedance (fixed YLIM; nothing is clipped) -----------------
    exceed = {}
    for method in METHODS:
        n_pts = n_above = n_real = n_real_above = 0
        for level in levels:
            for v in varios[(level, method)]:
                y = v.loc[v["n_pairs"] >= MIN_PAIRS_PER_BIN, "semivariance_empirical"].values
                n_pts += y.size
                n_above += int(np.sum(y > YLIM[1]))
                n_real += 1
                n_real_above += int(np.any(y > YLIM[1]))
        exceed[method] = {
            "n_curve_points": n_pts, "n_points_above_ylim": n_above,
            "pct_points_above_ylim": 100.0 * n_above / n_pts,
            "n_realizations": n_real, "n_realizations_with_any_bin_above_ylim": n_real_above,
        }
        print(f"\n{method:<8}: {exceed[method]['pct_points_above_ylim']:.2f}% of {n_pts} plotted "
              f"curve points above y={YLIM[1]:g}; {n_real_above}/{n_real} realizations have "
              ">=1 bin above")
    # The truth curves are subject to the same fixed y-limit and are reported
    # separately: a truth curve being cut off is a different (and more
    # notable) fact than a realization curve being cut off.
    t_pts = t_above = 0
    t_levels_above = []
    for level in levels:
        y = truth_varios[level].loc[
            truth_varios[level]["n_pairs"] >= MIN_PAIRS_PER_BIN, "semivariance_empirical"
        ].values
        t_pts += y.size
        n_a = int(np.sum(y > YLIM[1]))
        t_above += n_a
        if n_a > 0:
            t_levels_above.append(level)
    exceed_truth = {
        "n_curve_points": t_pts, "n_points_above_ylim": t_above,
        "pct_points_above_ylim": 100.0 * t_above / t_pts,
        "n_levels": len(levels),
        "levels_with_any_bin_above_ylim": t_levels_above,
    }
    print(f"\ntruth   : {exceed_truth['pct_points_above_ylim']:.2f}% of {t_pts} plotted truth "
          f"points above y={YLIM[1]:g}; cut off at level(s) "
          f"{t_levels_above if t_levels_above else 'none'}")

    max_plotted = max(
        max(float(np.max(truth_varios[lv]["semivariance_empirical"].values)) for lv in levels),
        max(float(np.max(v["semivariance_empirical"].values))
            for vs in varios.values() for v in vs),
        TOTAL_SILL,
    )
    print(f"max plotted semivariance over everything drawn: {max_plotted:.4f} "
          f"(y-axis fixed at {YLIM})")

    out_a = make_figure_a(levels, data, varios, truth_varios, exceed, exceed_truth)
    print(f"\nfigure A: {out_a} ({out_a.stat().st_size} bytes)")
    out_b = make_figure_b(levels, data, varios, truth_varios, exceed, exceed_truth)
    print(f"figure B: {out_b} ({out_b.stat().st_size} bytes)")

    # --- summary CSV: one row per (level, method, realization) ---------------
    rows = []
    print(f"\n{'range':>6} {'lag':>7} {'model':>7} {'method':>7} {'min%':>7} {'median%':>8} "
          f"{'max%':>7}")
    per_level_report = {}
    for level in levels:
        range_m = LEVEL_RANGE_M[level]
        lag = REPORT_LAG_BY_LEVEL[level]
        model_v = float(model_at(lag, range_m))
        truth_v = semivariance_at(truth_varios[level], lag)
        rows.append({
            "axis_level": level, "truth_range_m": range_m, "method": "truth",
            "realization_index": "", "n_realizations": "", "lag_m": lag,
            "model_semivariance_at_lag": model_v, "truth_semivariance_at_lag": truth_v,
            "value_semivariance": truth_v,
            "value_pct_of_model": 100.0 * truth_v / model_v,
            "value_pct_of_truth": 100.0,
        })
        print(f"{level:>6} {lag:>7.1f} {model_v:>7.3f} {'truth':>7} "
              f"{100 * truth_v / model_v:>7.1f} {'':>8} {'':>7}")
        per_level_report[level] = {
            "lag_m": lag, "model_semivariance": model_v,
            "truth_pct_of_model": 100.0 * truth_v / model_v,
        }
        for method in METHODS:
            vals = np.array([semivariance_at(v, lag) for v in varios[(level, method)]])
            for k, val in enumerate(vals):
                rows.append({
                    "axis_level": level, "truth_range_m": range_m, "method": method,
                    "realization_index": k, "n_realizations": len(vals), "lag_m": lag,
                    "model_semivariance_at_lag": model_v, "truth_semivariance_at_lag": truth_v,
                    "value_semivariance": float(val),
                    "value_pct_of_model": 100.0 * float(val) / model_v,
                    "value_pct_of_truth": 100.0 * float(val) / truth_v,
                })
            pct = 100.0 * vals / model_v
            print(f"{'':>6} {'':>7} {'':>7} {method:>7} {pct.min():>7.1f} "
                  f"{np.median(pct):>8.1f} {pct.max():>7.1f}")
            per_level_report[level][method] = {
                "min_pct_of_model": float(pct.min()),
                "median_pct_of_model": float(np.median(pct)),
                "max_pct_of_model": float(pct.max()),
            }

    summary_path = PROCESSED_DIR / SUMMARY_CSV_NAME
    new = pd.DataFrame(rows)
    if summary_path.exists() and set(levels) != set(LEVELS):
        old = pd.read_csv(summary_path, dtype={"axis_level": str},
                          float_precision="round_trip")
        old = old[~old["axis_level"].isin(set(levels))]
        new = pd.concat([old, new], ignore_index=True)
    new["_o"] = new["axis_level"].map({lv: i for i, lv in enumerate(LEVELS)})
    new = new.sort_values("_o", kind="stable").drop(columns="_o")
    new.to_csv(summary_path, index=False)
    print(f"\nsummary: {summary_path} ({len(new)} rows)")

    # --- seeds / provenance JSON --------------------------------------------
    total_seconds = time.time() - t_start
    json_path = PROCESSED_DIR / SEEDS_JSON_NAME
    existing = {}
    if json_path.exists() and set(levels) != set(LEVELS):
        existing = json.loads(json_path.read_text(encoding="utf-8")).get("levels", {})
    for level in levels:
        existing[level] = {
            "truth_range_m": LEVEL_RANGE_M[level],
            "n_samples_requested": N_SAMPLES,
            "n_samples_actual": data[level]["n_actual"],
            "gp_draw_seed": data[level]["gp_draw_seed"],
            "gp_draw_method": data[level]["gp_draw_method"],
            "gp_draw_seconds": round(data[level]["gp_draw_seconds"], 2),
            "level_runtime_seconds": data[level]["level_seconds"],
            "source_runs": data[level]["runs"],
            "sgs_out_of_range_values": data[level]["sgs_out_of_range"],
            "gp_reconstruction_validation": data[level]["gp_validation"],
            "report_lag_m": REPORT_LAG_BY_LEVEL[level],
            "report": per_level_report[level],
        }
    seeds = {
        "note": (
            "Seeds and provenance for range_axis/make_variogram_reproduction_figure.py (SGS vs. "
            "GP-MLE variogram reproduction across the 8 ground-truth range levels). Kriging and "
            "RBF+bootstrap are deliberately excluded -- see the script docstring and the figure "
            "footnotes. GP draws are re-generated on the fly (nothing written under results/raw): "
            "one sample_y(n_samples=10) call per level at gp_draw_seed_by_level[L]. "
            "GP_SAMPLE_SEED=55 is used only for the reconstruction-validation draw (the stored "
            "draw's own seed); it fixes a draw only for a bit-identical posterior covariance, "
            "which is why the seed-55 draw generally does NOT equal the stored one and the check "
            "is the chi-square consistency test instead (see corr_seed55_draw_vs_stored, "
            "n_eig_within_1e-6_rel_of_min and chi2_z_* per level)."
        ),
        "levels_run_last": list(levels),
        "methods": list(METHODS),
        "methods_excluded": {
            "kriging": "one smooth estimate, no realizations to reproduce a variogram with",
            "rbf_bootstrap": ("bootstrap replicate maps are resampled interpolations, not "
                              "conditional simulations of a spatial random function"),
        },
        "truth_seed": TRUTH_SEED,
        "sample_seed": SAMPLE_SEED,
        "n_samples_requested": N_SAMPLES,
        "variogram_pair_seed": VARIOGRAM_PAIR_SEED,
        "n_pairs_drawn": N_PAIRS_DRAWN,
        "lag_bin_width_m": LAG_BIN_WIDTH_M,
        "lag_max_m": LAG_MAX_M_RANGE_AXIS,
        "lag_window_rationale": (
            "0-1000 m (vs. 0-750 m on the density/replicate axes): the largest ground-truth "
            "range on this axis is 800 m and a spherical model reaches its sill only AT the "
            "range, so 750 m would cut the top level off before its plateau. 1000 m is the "
            "domain side length (50 x 20 m); beyond it only near-diagonal corner pairs remain "
            "(975-1000 m bin: 1709 of 199,905 drawn pairs; 1175-1200 m bin: 141). The bin width "
            "stays 25 m so curves remain comparable with the other axes, and every one of the 40 "
            "bins carries at least 345 pairs, far above MIN_PAIRS_PER_BIN."
        ),
        "min_pairs_per_bin_guard": MIN_PAIRS_PER_BIN,
        "gp_sample_seed_validation": GP_SAMPLE_SEED,
        "gp_draw_seed_base": GP_DRAW_SEED_BASE,
        "gp_draw_seed_by_level": GP_DRAW_SEED_BY_LEVEL,
        "gp_draw_seed_block_note": (
            "96xx is a NEW block reserved for the range axis; the sample-replicate axis uses "
            "9000+k (1%), 9100+k (2%), 9200+k (5%), 9300+k (10%), 9400+k (20%), so there is no "
            "collision."
        ),
        "n_gp_draws_per_level": N_GP_DRAWS,
        "gp_reconstruction_tolerances": {
            "mean_var_rtol": GP_MAP_RTOL, "mean_var_atol": GP_MAP_ATOL,
            "chi2_max_abs_z": GP_CHI2_MAX_ABS_Z,
            "eig_cluster_rel_tol": EIG_CLUSTER_REL_TOL,
        },
        "report_lag_rule": (
            "REPORT_LAG(R) = R - LAG_BIN_WIDTH_M/2, i.e. the centre of the last 25 m bin below "
            "that level's own ground-truth range (87.5 .. 787.5 m). This is the range-axis "
            "correspondent of the 287.5 m bin the density/replicate artifacts report against "
            "their fixed 300 m range; each value is expressed as a % of THAT level's own "
            "theoretical spherical model at the same lag."
        ),
        "theoretical_model": {
            "nugget_real_units": TRUTH_NUGGET_REAL,
            "structured_sill_real_units": TRUTH_STRUCT_SILL_REAL,
            "total_sill_real_units": TOTAL_SILL,
            "range_m": "per level (the axis variable)",
        },
        "ylim": list(YLIM),
        "ylim_rule": (
            f"y-axis fixed at 0 to Y_MAX_SILL_FACTOR ({Y_MAX_SILL_FACTOR:g}) x total sill "
            f"(POR_STDEV**2 = {TOTAL_SILL:g}) = {Y_MAX:g} in every panel, the same rule as the "
            "density/replicate variogram-reproduction figures; curves above it are cut off by "
            "the axis (data not clipped)."
        ),
        "max_plotted_value": float(max_plotted),
        "exceed_ylim_by_method": exceed,
        "exceed_ylim_truth": exceed_truth,
        "alpha": ALPHA,
        "pooled_alpha": POOLED_ALPHA_RANGE_AXIS,
        "pooled_alpha_replicate_axis_for_reference": POOLED_ALPHA,
        "figure_a": out_a.relative_to(_REPO_ROOT).as_posix(),
        "figure_b": out_b.relative_to(_REPO_ROOT).as_posix(),
        "summary_csv": summary_path.relative_to(_REPO_ROOT).as_posix(),
        "total_runtime_seconds_last_run": round(total_seconds, 1),
        "levels": {lv: existing[lv] for lv in LEVELS if lv in existing},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2)
    print(f"seeds: {json_path}")
    print(f"total runtime {total_seconds:.1f} s")
    return out_a, out_b, summary_path, json_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS),
                        help="range levels (m) to (re)generate; default: all eight")
    main(tuple(parser.parse_args().levels))
