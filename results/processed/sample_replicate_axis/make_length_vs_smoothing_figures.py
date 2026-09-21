"""Length scale vs. "smoothing" parameter, for GP-MLE and RBF+bootstrap
separately, across the 10 sample-seed replicates (src/experiments/
sample_replicate_axis.py) at all 3 sample-density-axis levels ("5"/"2"/"1"
-> n_samples_requested = 125/50/25).

WHAT this plots
----------------
Two figures, each ONE ROW of 3 scatter panels (5% / 2% / 1%, dense -> sparse
left to right, matching this project's standing panel-ordering convention --
e.g. make_length_and_variogram_figures.py's density-axis panels). Every panel
shows all 10 replicates (rep0..rep9, sample_seed 1001-1010) as points:
  x = length_scale_m, taken AS-IS from
      results/processed/sample_replicate_axis/length_scale_by_replicate.csv
      (already computed by evaluate_sample_replicate_axis.py's length-scale
      pipeline -- this script does NOT recompute or re-derive that
      conversion; see that CSV's own per-row ``length_definition`` /
      ``source`` columns for how each method's value was derived).
  y = a method-specific "smoothing" quantity, read FRESH from each run's own
      manifest.json (paths taken from source_runs.json, one manifest per
      (axis_level, replicate, method) cell -- NOT reused from any pre-joined
      table, since neither field lives in length_scale_by_replicate.csv):
        - rbf_bootstrap: params.best_smoothing -- the CV-selected
          regularization/smoothing parameter for scipy's RBFInterpolator,
          chosen from src/experiments/rbf_bootstrap.py's SMOOTHING_GRID.
          This is a LITERAL "smoothing" hyperparameter in this codebase.
        - gp_mle: params.fitted_hyperparameters.noise_variance_real_units --
          the MLE-fitted GP nugget/noise variance, converted to real
          (Porosity %^2) units.

WHY noise_variance_real_units is GP-MLE's y-axis (orchestrator decision,
stated explicitly so no reader is misled)
------------------------------------------------------------------------------
This codebase has NO hyperparameter literally named "smoothing" for GP-MLE.
The orchestrator judged noise_variance_real_units to be the closest analog to
RBF's best_smoothing: both trade off exact interpolation of the conditioning
data against a smoother fit -- RBF's smoothing parameter directly relaxes the
exact-interpolation constraint in the RBF system, and the GP's nugget/noise
term plays the equivalent role in a GP posterior (a larger noise variance
shrinks the posterior mean away from exact interpolation of the training
points and inflates predictive variance, exactly as CLAUDE.md's stated
Claim 2 describes -- "GPR ... risk: nugget을 신호로 잘못 학습해 불확실성을
과소평가"). This script's y-axis label and caption say "noise variance
(Porosity %^2) -- GP's smoothing-equivalent regularization term", NEVER a bare
"smoothing", so a reader is not misled into thinking a hyperparameter of that
exact name exists for GP-MLE in this codebase. The field itself comes
straight from gp_mle.py's manifest (params.fitted_hyperparameters, alongside
length_scale_m, signal_variance_normalized, noise_variance_normalized,
y_train_std, signal_variance_real_units) -- see any gp_mle manifest.json for
the full block.

AXIS SCALE CHOICES (both log, both panels of both figures)
------------------------------------------------------------------------------
Y-AXIS: log, in both figures. Observed ranges (see VERIFICATION output below,
printed by this script every run) span >=3 orders of magnitude within a
single figure:
  - rbf_bootstrap best_smoothing: 0.00215 to 1.0 across all 30 (17-value grid since 2026-09-21; was 0.001 to 1.0 on the old 7-value grid) (level,
    replicate) cells (SMOOTHING_GRID is itself log-like: values selected by
    CV from a small discrete grid, not a continuum).
  - gp_mle noise_variance_real_units: ~5.5e-5 to ~5.37 across all 30 cells --
    over 4.5 orders of magnitude. A linear y-axis would compress every level
    "5" replicate's noise variance (order 1, see below) into indistinguishable
    dots on the same scale as level "1" replicates near the MLE's lower
    bound (order 1e-5).
A linear y-axis for either quantity would make small-but-real differences
among the majority of points invisible next to the few largest values.

X-AXIS: log, in both figures, for the SAME length_scale_m quantity used
throughout this project's sample_replicate_axis outputs. Within a single
panel the fitted length scale already spans a large MULTIPLICATIVE range at
the sparser levels (this project's own case-study figure,
make_length_case_study_figures.py, already frames these swings
multiplicatively -- "~5x (gp_mle)" and "~11x (rbf_bootstrap)" at the 1%
level): this script's own re-tabulation from length_scale_by_replicate.csv
confirms a ~1.5x-2.4x within-panel spread at the 5% level growing to
~5.1x (gp_mle) / ~11.3x (rbf_bootstrap) at the 1% level. Since length scale is
a fundamentally multiplicative quantity here (and the y-axis is already log
for the reason above), a log x-axis keeps the sparsest-level panel's few very
large fitted length scales from stretching the panel so far that the tighter
majority of points collapse together, and keeps the two axes' visual
treatment consistent within each panel.

SHARED AXIS LIMITS -- WITHIN each figure, NEVER between the two figures
------------------------------------------------------------------------------
The point these figures are asked to make visually is that hyperparameter
optimization gets LESS STABLE as sample density drops. That comparison is
only readable if the 3 panels of a figure sit on identical axes, so a wider
cloud on screen means a genuinely wider spread and not merely a rescaled
panel. Therefore:

  * WITHIN a figure, all 3 panels share one x-limit pair and one y-limit pair,
    computed from ALL 30 points of that figure and applied EXPLICITLY with
    ax.set_xlim / ax.set_ylim (not left to sharex/sharey autoscaling, which
    would make the actual numbers implicit and non-reproducible). sharex=
    sharey=True is still passed to plt.subplots so panning/ticking stay
    locked together, but the limits themselves are the explicit ones below.
    y tick labels are drawn on the LEFTMOST panel only (they are identical by
    construction); x tick labels stay on all 3 panels.
  * BETWEEN the two figures, NOTHING is shared. GP-MLE's y is a noise
    variance in Porosity %^2 and RBF's y is a dimensionless smoothing
    parameter -- different physical quantities that must not be forced onto
    a common axis. The x axes (both in m) are likewise computed separately
    per figure, so each figure's panels are compared only against each other.

Padding: limits are the observed min/max padded SYMMETRICALLY IN LOG SPACE by
  pad_decades = max(AXIS_PAD_LOG_FRAC * log10_span, log10(AXIS_PAD_MIN_FACTOR))
                = max(5% of the axis' log10 span, a fixed x/ 1.15 factor)
i.e. the same number of decades is added below the min and above the max, so
the padding is visually symmetric on a log axis. The max() of the two rules is
used because neither alone works for both axes here: a pure 5%-of-span rule
degenerates to only ~1.09x of headroom on the narrowest axis (gp_mle x spans
just ~0.71 decades overall), too tight for the edge markers and their
replicate-id labels; while a pure fixed 1.15x is a negligible ~1.2% of the
gp_mle y axis' ~5-decade span. Taking the larger of the two guarantees both a
visible minimum margin and proportional headroom on wide axes.

PER-PANEL SPREAD ANNOTATION (numbers behind the visual claim)
------------------------------------------------------------------------------
Each panel carries a small text box reporting, for that panel's 10 points and
for x and y separately: the standard deviation of log10(value) (ddof=1, i.e.
the sample SD -- the natural dispersion measure on a log axis) and the
max/min ratio. The same numbers are printed to stdout in the VERIFICATION
block, so the figure's visual claim can always be checked against digits. The
box is placed in whichever corner of the panel contains the fewest points
(deterministic tie-break order: lower right, upper left, upper right, lower
left) so it never has to be nudged by hand.

NOTHING in the figures themselves is a hardcoded number: every value drawn on
a panel (the spread boxes, the shared limits quoted in the caption, the
caption's spread listings, the monotone/non-monotone wording, the list of
boundary-stop replicates) is formatted from the values just computed from the
manifests, so a caption cannot outlive the data it describes. The numbers
quoted IN THIS DOCSTRING below are prose and therefore cannot be dynamic --
they are instead mirrored in the DOCUMENTED_SPREAD / DOCUMENTED_NOISE_FLOOR /
DOCUMENTED_BOUNDARY_STOP_CELLS constants and re-checked against the freshly
computed values on every run (verify_documented_values(); the run FAILS with
an explicit ValueError if they ever drift, telling the maintainer to update
both the constants and this docstring).

Observed values (both figures, dense -> sparse), as of 2026-09-18:
  gp_mle x: log10-sd 0.051 / 0.170 / 0.227, ratio 1.51x / 3.97x / 5.12x
  gp_mle y: log10-sd 0.096 / 1.348 / 1.854
  rbf     x: log10-sd 0.113 / 0.208 / 0.340, ratio 2.42x / 4.69x / 11.34x
  rbf     y: log10-sd 0.316 / 0.738 / 0.675, ratio 10x / 1000x / 100x

TWO CAVEATS THAT MUST NOT BE OVERSTATED
------------------------------------------------------------------------------
1. The spread is NOT monotonically increasing with sparsity for every
   quantity. It is monotone for gp_mle's x AND y and for rbf_bootstrap's x,
   but NOT for rbf_bootstrap's smoothing parameter, whose spread is widest at
   the 2% level (log10-sd 0.738) and smaller again at 1% (0.675). Part of the
   reason is resolution: best_smoothing is chosen from a 7-value discrete
   SMOOTHING_GRID, so its per-panel spread moves in coarse jumps (ratios of
   exactly 10x / 1000x / 100x) and should not be read as a fine-grained
   trend. Do not describe these figures as "all spreads grow monotonically as
   density drops". (The captions state monotone vs. non-monotone from the
   computed numbers, not from a literal, so they stay correct if the data
   changes.)
2. gp_mle's y spread is a LOWER BOUND on the spread an unconstrained fit
   would show. Three of the 30 fitted noise variances (2% rep3, 1% rep0,
   1% rep5 -- the set is determined at runtime by at_optimizer_noise_floor(),
   not hardcoded in the plotting code) sit EXACTLY on the optimizer's lower
   bound for the WhiteKernel noise level (noise_level_bounds[0] = 1e-5 in
   normalized units; each run's own manifest records the bound it used, and
   this script reads it from there rather than hardcoding it). For those runs
   the constrained optimum is the bound itself, so the unconstrained optimum
   lies at or below it and the observed downward extent of the y cloud is
   truncated there.
   WHAT THAT MEANS IS NOT DECIDED BY THIS FIGURE. A boundary stop is equally
   consistent with (a) the nugget genuinely being ~0 for those replicates and
   (b) the nugget simply not being identifiable at that sample density (a
   marginal likelihood nearly flat in the nugget direction, letting the
   optimizer slide to the bound). State the observation, not a cause: these
   figures do not distinguish the two. Such points are drawn with an extra
   hollow black circle and counted in the panel's annotation box and in the
   VERIFICATION stdout.

VERIFICATION (same "verify before trusting" pattern this project's other
processed-data scripts already use, e.g. make_length_case_study_figures.py's
DECISIVE CORRECTNESS CHECK)
------------------------------------------------------------------------------
For each of the 2 methods x 3 axis levels x 10 replicates = 60 cells, this
script (a) reads that cell's run directory from source_runs.json, (b) opens
that run's OWN manifest.json and extracts the method-specific y-field
(raising KeyError loudly if the field is missing -- never silently defaulting),
and (c) looks up that cell's x-value (length_scale_m) from
length_scale_by_replicate.csv, requiring EXACTLY ONE matching
(axis_level, replicate, method) row (raising ValueError otherwise). After
assembly, this script asserts each (method, axis_level) group has EXACTLY 10
rows and each method has EXACTLY 30 rows (3 levels x 10 replicates) before
plotting anything -- so a missing or duplicated manifest can never be plotted
silently. The observed per-axis value ranges (both x and y, per level and
overall) plus the per-level log10-SD / max-min-ratio spread statistics are
printed to stdout every run so they can be sanity-checked against
already-known numbers (e.g. rbf_bootstrap length scale at the 1% level
spanning ~141-1604 m, matching make_length_case_study_figures.py's
independently-confirmed MIN/MAX replicates for that method/level).

Additionally, because the shared limits above are computed rather than
autoscaled, every figure runs a LIMIT CHECK after the limits are set. It
verifies exactly two things, and raises ValueError (not assert -- so `python
-O` cannot strip it, matching every other check in this file and in
results/processed/) on either:
  (a) all 3 panels ended up with the SAME final ax.get_xlim() and the SAME
      final ax.get_ylim() -- this is the whole contract of the shared-axes
      change, and nothing else in the script enforces it; and
  (b) every plotted (x, y) DATA POINT lies inside those final limits (1e-12
      relative tolerance, pure float round-trip slack).
Be honest about the limits of (b): given the padding rule above, the padded
limits always contain the data by construction, so in normal operation (b)
only catches a downstream bug that changed or overrode the limits after
set_xlim/set_ylim (e.g. a stray autoscale, or a future edit plotting extra
points). It does NOT check the things that could actually cross a spine:
marker radius (s=70, s=230 for the boundary-stop rings), the replicate-id
annotation text, and the per-panel spread box are all drawn in display space
and are not covered by any check here -- they were verified by eye only.

Run with:
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_length_vs_smoothing_figures
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
from matplotlib.ticker import LogFormatterSciNotation, LogLocator  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.base_case import NX, NY  # noqa: E402
from src.experiments.sample_density_axis import ALL_AXIS_LEVELS  # noqa: E402
from src.experiments.sample_replicate_axis import (  # noqa: E402
    AXIS_HMAJ1,
    N_SAMPLES_REQUESTED_BY_LEVEL,
    REPLICATE_IDS,
    TRUTH_SEED,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "sample_replicate_axis"

# Matches make_sample_replicate_figures.py's DPI, the most similar existing
# layout in this directory (also a bare 1x3 scatter grid, no imshow maps --
# unlike make_length_case_study_figures.py's 2x6 map-heavy layout, which
# instead uses 300 dpi).
FIG_DPI = 600

GRID_CELLS = NX * NY  # 2500; used only to state each level's requested % of
# the grid in panel titles (125/2500=5%, 50/2500=2%, 25/2500=1%, exactly, by
# construction of N_SAMPLES_REQUESTED_BY_LEVEL -- see sample_density_axis.py).

# Shared-limit padding, applied symmetrically in LOG space (both axes of both
# figures are log-scaled). pad_decades = max(5% of the axis' log10 span,
# log10(1.15)); see the module docstring's "SHARED AXIS LIMITS" section for
# why the max() of a proportional rule and a fixed-factor floor is used
# instead of either one alone.
AXIS_PAD_LOG_FRAC = 0.05
AXIS_PAD_MIN_FACTOR = 1.15

# Relative tolerance for the limit check (pure float round-trip slack through
# set_xlim -> get_xlim; NOT a real margin).
LIMIT_CHECK_RTOL = 1e-12

# --- Numbers quoted in the MODULE DOCSTRING (prose, so they cannot be
# generated) mirrored here as data and re-checked against the freshly computed
# values on every run by verify_documented_values(). Nothing drawn INTO the
# figures is taken from these constants -- the figures format everything from
# the just-computed stats -- these exist solely so the docstring cannot
# silently go stale if the source runs in source_runs.json change. Values are
# ordered dense -> sparse, i.e. ALL_AXIS_LEVELS order ("5", "2", "1").
DOCUMENTED_SPREAD = {
    "gp_mle": {
        "x_log10_sd": (0.051, 0.170, 0.227),
        "x_ratio": (1.51, 3.97, 5.12),
        "y_log10_sd": (0.096, 1.348, 1.854),
    },
    "rbf_bootstrap": {
        "x_log10_sd": (0.113, 0.208, 0.340),
        "x_ratio": (2.42, 4.69, 11.34),
        "y_log10_sd": (0.316, 0.738, 0.675),
        "y_ratio": (10.0, 1000.0, 100.0),
    },
}
# Rounding used when comparing: the docstring quotes log10-SDs to 3 decimals
# and ratios to 2 decimals, so agreement is checked at that same precision.
DOCUMENTED_DECIMALS = {"x_log10_sd": 3, "y_log10_sd": 3, "x_ratio": 2, "y_ratio": 2}
# gp_mle cells the docstring names as sitting on the optimizer's noise floor,
# and the normalized bound value it quotes.
DOCUMENTED_BOUNDARY_STOP_CELLS = (("2", "rep3"), ("1", "rep0"), ("1", "rep5"))
DOCUMENTED_NOISE_FLOOR = 1e-5

# Colors/markers redefined identically (NOT imported), following this
# directory's own established precedent (make_sample_replicate_figures.py's
# module docstring) of not importing make_sample_density_figures.py's
# METHOD_COLORS/METHOD_MARKERS to avoid that other module's file-reading
# import-time side effects. Only the 2 methods this script uses are included.
METHOD_COLORS = {"rbf_bootstrap": "tab:orange", "gp_mle": "tab:red"}
METHOD_MARKERS = {"rbf_bootstrap": "^", "gp_mle": "D"}
METHOD_LABELS = {"rbf_bootstrap": "RBF+bootstrap", "gp_mle": "GP-MLE"}

Y_FIELD_LABELS = {
    "rbf_bootstrap": "CV-selected RBF smoothing parameter\n(params.best_smoothing, log scale)",
    "gp_mle": (
        "Noise variance (Porosity %$^2$) -- GP's smoothing-equivalent\n"
        "regularization term (params.fitted_hyperparameters."
        "noise_variance_real_units), log scale"
    ),
}

X_LABEL = "Fitted length scale (m, log scale) -- from length_scale_by_replicate.csv"


def load_length_df() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "length_scale_by_replicate.csv", comment="#")
    df["axis_level"] = df["axis_level"].astype(str)
    return df


def load_source_runs() -> dict:
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_replicate_seeds() -> dict:
    with open(PROCESSED_DIR / "replicate_seeds.json", "r", encoding="utf-8") as f:
        return json.load(f)


def extract_y_value(manifest_params: dict, method: str) -> float:
    """Pull the method-specific "smoothing" analog straight out of a run's
    own manifest.json params block. Raises KeyError (loudly, not silently)
    if the expected field is missing -- see module docstring for exactly
    which field each method uses and why."""
    if method == "rbf_bootstrap":
        return float(manifest_params["best_smoothing"])
    if method == "gp_mle":
        return float(manifest_params["fitted_hyperparameters"]["noise_variance_real_units"])
    raise ValueError(f"unsupported method: {method}")


def at_optimizer_noise_floor(manifest_params: dict, method: str) -> bool:
    """True iff this gp_mle run's MLE stopped EXACTLY on the WhiteKernel
    noise-level lower bound, i.e. the reported noise variance is a boundary
    stop rather than an interior optimum (the true optimum may lie lower, so
    such points truncate the y spread from below -- see module docstring
    caveat 2). The bound is read from THIS run's own manifest
    (params.kernel_init.noise_level_bounds[0]) rather than hardcoded or
    imported, so a run fitted under different bounds is judged against the
    bounds it actually used. Always False for rbf_bootstrap, whose
    best_smoothing comes from a discrete CV grid, not a bounded optimizer."""
    if method != "gp_mle":
        return False
    noise_norm = float(manifest_params["fitted_hyperparameters"]["noise_variance_normalized"])
    floor = optimizer_noise_floor(manifest_params, method)
    # Relative comparison: sklearn returns the bound itself up to float noise
    # (e.g. 9.999999999999997e-06 for a 1e-5 bound).
    return abs(noise_norm - floor) <= 1e-9 * floor


def optimizer_noise_floor(manifest_params: dict, method: str):
    """The WhiteKernel noise-level LOWER BOUND this particular gp_mle run was
    fitted under, read from its own manifest (never hardcoded). None for
    rbf_bootstrap, which has no such bounded optimizer. Used both to decide
    which points are boundary stops and to quote the bound in the figure
    caption from the data rather than from a literal."""
    if method != "gp_mle":
        return None
    return float(manifest_params["kernel_init"]["noise_level_bounds"][0])


def build_dataset(method: str, length_df: pd.DataFrame, source_runs: dict) -> pd.DataFrame:
    """Assemble one method's (axis_level, replicate) -> (length_scale_m,
    y_value) table by reading each run's OWN manifest.json fresh (y_value)
    and cross-referencing length_scale_by_replicate.csv (x_value, NOT
    recomputed). Raises if any of the expected 3 levels x 10 replicates = 30
    cells is missing, duplicated, or mismatched."""
    rows = []
    for level in ALL_AXIS_LEVELS:
        for rep in REPLICATE_IDS:
            run_dir = _REPO_ROOT / source_runs[level][rep][method]
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"{method} axis_level={level} {rep}: manifest not found at {manifest_path}"
                )
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            y_value = extract_y_value(manifest["params"], method)
            at_floor = at_optimizer_noise_floor(manifest["params"], method)
            noise_floor = optimizer_noise_floor(manifest["params"], method)

            length_row = length_df[
                (length_df["axis_level"] == level)
                & (length_df["replicate"] == rep)
                & (length_df["method"] == method)
            ]
            if len(length_row) != 1:
                raise ValueError(
                    f"expected exactly 1 length_scale_by_replicate.csv row for "
                    f"(axis_level={level}, replicate={rep}, method={method}); "
                    f"found {len(length_row)}."
                )
            length_scale_m = float(length_row["length_scale_m"].iloc[0])

            rows.append(
                {
                    "axis_level": level,
                    "replicate": rep,
                    "method": method,
                    "length_scale_m": length_scale_m,
                    "y_value": y_value,
                    "at_noise_floor": at_floor,
                    "noise_floor_normalized": noise_floor,
                    "manifest_path": str(manifest_path.relative_to(_REPO_ROOT)),
                }
            )

    df = pd.DataFrame(rows)
    expected_total = len(ALL_AXIS_LEVELS) * len(REPLICATE_IDS)
    if len(df) != expected_total:
        raise ValueError(
            f"{method}: expected {expected_total} rows (({len(ALL_AXIS_LEVELS)} levels) x "
            f"({len(REPLICATE_IDS)} replicates)), found {len(df)}."
        )
    for level in ALL_AXIS_LEVELS:
        n_level = len(df[df["axis_level"] == level])
        if n_level != len(REPLICATE_IDS):
            raise ValueError(
                f"{method} axis_level={level}: expected {len(REPLICATE_IDS)} rows, "
                f"found {n_level}."
            )

    print(
        f"{method}: verified {len(df)} manifest reads "
        f"({len(ALL_AXIS_LEVELS)} levels x {len(REPLICATE_IDS)} replicates)."
    )
    for level in ALL_AXIS_LEVELS:
        sub = df[df["axis_level"] == level]
        stats = panel_spread_stats(sub)
        print(
            f"  axis_level={level}: length_scale_m in "
            f"[{sub['length_scale_m'].min():.2f}, {sub['length_scale_m'].max():.2f}] m "
            f"(ratio {sub['length_scale_m'].max() / sub['length_scale_m'].min():.2f}x); "
            f"y_value in [{sub['y_value'].min():.6g}, {sub['y_value'].max():.6g}]"
        )
        print(
            f"    spread (n=10): x log10-sd(ddof=1)={stats['x_log10_sd']:.3f}, "
            f"max/min={stats['x_ratio']:.2f}x | "
            f"y log10-sd(ddof=1)={stats['y_log10_sd']:.3f}, "
            f"max/min={stats['y_ratio']:.6g}x"
        )
        n_floor = int(sub["at_noise_floor"].sum())
        if n_floor:
            floored = ",".join(sorted(sub[sub["at_noise_floor"]]["replicate"],
                                      key=lambda r: int(r.replace("rep", ""))))
            floor_val = float(sub[sub["at_noise_floor"]]["noise_floor_normalized"].iloc[0])
            print(
                f"    NOTE: {n_floor}/10 point(s) ({floored}) stopped EXACTLY on the "
                f"optimizer's noise_level lower bound ({floor_val:g} normalized), so the "
                f"unconstrained optimum lies at or below it and this panel's y spread is a "
                f"LOWER BOUND on the spread an unconstrained fit would show. Whether the "
                f"nugget is genuinely ~0 for these replicates or simply not identifiable at "
                f"this density is NOT distinguished by this figure."
            )
    print(
        f"  overall: length_scale_m in "
        f"[{df['length_scale_m'].min():.2f}, {df['length_scale_m'].max():.2f}] m; "
        f"y_value in [{df['y_value'].min():.6g}, {df['y_value'].max():.6g}]"
    )
    return df


def panel_spread_stats(sub: pd.DataFrame) -> dict:
    """Dispersion of one panel's 10 points, per axis: SD of log10(value)
    (ddof=1, the sample SD -- the natural spread measure on a log axis) and
    the plain max/min ratio. Both are reported in the panel's annotation box
    AND in the VERIFICATION stdout so the figure's visual claim is always
    backed by printed digits."""
    stats = {}
    for key, col in (("x", "length_scale_m"), ("y", "y_value")):
        v = sub[col].to_numpy(dtype=float)
        if np.any(v <= 0):
            raise ValueError(f"non-positive value on a log axis in column {col}: min={v.min()!r}")
        stats[f"{key}_log10_sd"] = float(np.std(np.log10(v), ddof=1))
        stats[f"{key}_ratio"] = float(v.max() / v.min())
    return stats


def shared_log_limits(values) -> tuple:
    """Explicit shared (lo, hi) limits for one log-scaled axis, from ALL of a
    figure's points, padded symmetrically in log space by
    max(AXIS_PAD_LOG_FRAC * log10-span, log10(AXIS_PAD_MIN_FACTOR)) decades
    on each side (see module docstring for why both rules are combined)."""
    v = np.asarray(values, dtype=float)
    if np.any(v <= 0):
        raise ValueError(f"log-scaled axis requires strictly positive values; got min={v.min()!r}")
    log_lo, log_hi = float(np.log10(v.min())), float(np.log10(v.max()))
    pad_decades = max(AXIS_PAD_LOG_FRAC * (log_hi - log_lo), float(np.log10(AXIS_PAD_MIN_FACTOR)))
    return 10.0 ** (log_lo - pad_decades), 10.0 ** (log_hi + pad_decades)


def _trend_phrase(values) -> str:
    """Describe a dense -> sparse sequence of spread values WITHOUT asserting a
    trend that may not hold: returns "widens monotonically as density drops"
    only if the sequence is strictly increasing, otherwise names the level
    where it actually peaks. Captions use this instead of a literal phrase so
    they cannot outlive the data (rbf_bootstrap's smoothing spread, for
    instance, peaks at the 2% level rather than at 1%)."""
    vals = [float(v) for v in values]
    if all(b > a for a, b in zip(vals, vals[1:])):
        return "widens monotonically as density drops"
    widest = list(ALL_AXIS_LEVELS)[int(np.argmax(vals))]
    return f"does NOT widen monotonically (it is widest at the {widest}% level)"


def _fmt_seq(values, fmt: str) -> str:
    """'0.051 -> 0.170 -> 0.227' -- dense-to-sparse listing for captions."""
    return " -> ".join(format(float(v), fmt) for v in values)


def _emptiest_corner(sub: pd.DataFrame, xlim: tuple, ylim: tuple) -> tuple:
    """Pick the corner of a panel holding the fewest plotted points, so the
    spread-annotation box can be placed without hand-nudging. Deterministic:
    corners are tested in a fixed preference order (lower right first) and the
    first minimum wins. Returns (x, y, ha, va) in axes-fraction coords."""
    fx = (np.log10(sub["length_scale_m"].to_numpy(dtype=float)) - np.log10(xlim[0])) / (
        np.log10(xlim[1]) - np.log10(xlim[0])
    )
    fy = (np.log10(sub["y_value"].to_numpy(dtype=float)) - np.log10(ylim[0])) / (
        np.log10(ylim[1]) - np.log10(ylim[0])
    )
    box_w, box_h = 0.48, 0.34  # approximate footprint of the text box
    candidates = [
        ("lower right", 0.98, 0.03, "right", "bottom", (fx > 1 - box_w) & (fy < box_h)),
        ("upper left", 0.02, 0.97, "left", "top", (fx < box_w) & (fy > 1 - box_h)),
        ("upper right", 0.98, 0.97, "right", "top", (fx > 1 - box_w) & (fy > 1 - box_h)),
        ("lower left", 0.02, 0.03, "left", "bottom", (fx < box_w) & (fy < box_h)),
    ]
    best = min(candidates, key=lambda c: int(c[5].sum()))
    return best[1], best[2], best[3], best[4]


def _level_label(level: str, replicate_seeds: dict) -> str:
    """'5% level (n_requested=125, n_actual=118-124)' -- requested % (exact,
    by construction of N_SAMPLES_REQUESTED_BY_LEVEL against the 2500-cell
    grid) plus the ACTUAL post-dedup sample-count range across the 10
    replicates at this level, following the SAME requested-vs-actual
    reporting convention make_sample_replicate_figures.py's own caption
    already uses for this axis (actual count varies per replicate because
    random_interior_samples drops duplicate-cell draws)."""
    n_req = N_SAMPLES_REQUESTED_BY_LEVEL[level]
    pct_req = 100.0 * n_req / GRID_CELLS
    actual = replicate_seeds["n_samples_actual"][level]
    n_actual_vals = list(actual.values())
    return (
        f"{pct_req:.0f}% level (n_requested={n_req}, "
        f"n_actual={min(n_actual_vals)}-{max(n_actual_vals)})"
    )


def make_figure(method: str, df: pd.DataFrame, replicate_seeds: dict) -> Path:
    color = METHOD_COLORS[method]
    marker = METHOD_MARKERS[method]

    # --- Shared limits: computed ONCE from this figure's own 30 points and
    # applied explicitly to every panel. Never shared with the other figure
    # (different y quantity; see module docstring "SHARED AXIS LIMITS").
    xlim = shared_log_limits(df["length_scale_m"])
    ylim = shared_log_limits(df["y_value"])
    print(
        f"  shared axis limits for the {METHOD_LABELS[method]} figure (all 3 panels): "
        f"x=[{xlim[0]:.6g}, {xlim[1]:.6g}] m, y=[{ylim[0]:.6g}, {ylim[1]:.6g}]"
    )

    # Per-level spread stats and boundary-stop cells, collected while the
    # panels are drawn so the caption below can be FORMATTED FROM THEM instead
    # of quoting literals that would silently go stale if the source runs
    # changed (the shared limits quoted in the caption already worked this way).
    level_stats = {}
    floored_cells = []  # (axis_level, replicate) of every circled point
    noise_floor_values = set()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), sharex=True, sharey=True)
    for i_panel, (ax, level) in enumerate(zip(axes, ALL_AXIS_LEVELS)):
        sub = df[df["axis_level"] == level].sort_values("replicate")
        ax.scatter(
            sub["length_scale_m"], sub["y_value"], color=color, marker=marker,
            s=70, edgecolors="black", linewidths=0.7, alpha=0.85, zorder=3,
        )
        # Boundary stops (gp_mle only): the MLE stopped exactly ON the
        # WhiteKernel noise_level lower bound, so the unconstrained optimum
        # lies at or below it and this panel's y spread is truncated from
        # below. This says nothing about WHY (module docstring caveat 2).
        # Circled here and counted in the annotation box.
        floored = sub[sub["at_noise_floor"]]
        if len(floored):
            ax.scatter(
                floored["length_scale_m"], floored["y_value"],
                facecolors="none", edgecolors="black", marker="o",
                s=230, linewidths=1.1, zorder=4,
            )
            floored_cells.extend(
                (level, rep) for rep in sorted(
                    floored["replicate"], key=lambda r: int(r.replace("rep", ""))
                )
            )
            noise_floor_values.update(
                float(v) for v in floored["noise_floor_normalized"]
            )
        # Points that land on (or within a tight tolerance of) the same
        # (x, y) coordinate are REAL ties, not a plotting artifact:
        # best_smoothing and the length-scale conversion both come from
        # small discrete CV grids, so multiple replicates can legitimately
        # land on the exact same point (e.g. rep0/rep1/rep8 all at
        # (219.92 m, 0.1) in the RBF+bootstrap 5% panel). Annotating each
        # such point separately at the same fixed pixel offset stacks their
        # replicate-id labels on top of each other into an unreadable smear,
        # so points within TIE_LOG_TOL (~0.23% multiplicative, tight enough
        # to catch only true/near-exact ties, not merge genuinely distinct
        # nearby points) on BOTH axes (in log space, matching the log-log
        # panels) are combined into ONE label listing all their replicate
        # indices (e.g. "0,1,8") instead of one overlapping label per point.
        TIE_LOG_TOL = 3  # decimal places of log10(value) to round to
        sub = sub.copy()
        sub["_tie_key"] = list(
            zip(
                np.log10(sub["length_scale_m"]).round(TIE_LOG_TOL),
                np.log10(sub["y_value"]).round(TIE_LOG_TOL),
            )
        )
        for _, grp in sub.groupby("_tie_key", sort=False):
            reps_label = ",".join(
                sorted(
                    (r.replace("rep", "") for r in grp["replicate"]),
                    key=int,
                )
            )
            ax.annotate(
                reps_label,
                (grp["length_scale_m"].iloc[0], grp["y_value"].iloc[0]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=6.5, color="dimgray", alpha=0.9,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Explicit shared limits (NOT autoscaled) -- identical numbers on all
        # 3 panels so a wider cloud means a genuinely wider spread.
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(X_LABEL, fontsize=8.5)
        if i_panel == 0:
            ax.set_ylabel(Y_FIELD_LABELS[method], fontsize=8.5)
        else:
            # y tick labels only on the leftmost panel (all panels share the
            # exact same y limits, so repeating them adds nothing); x tick
            # labels are kept on all 3 panels.
            ax.tick_params(labelleft=False)
        ax.tick_params(labelbottom=True)
        # With the shared x limits above, a panel can span slightly more than
        # one decade and still contain at most ONE decade tick (e.g. the
        # RBF+bootstrap figure's 123-1844 m range contains only 10^3), which
        # would leave the x axis with a single labelled tick. Label the 2/3/5
        # minor ticks explicitly so the length-scale axis stays readable in
        # both figures. (The y axes span 3.3-5.5 decades, so their default
        # decade labelling is already dense enough and is left untouched.)
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0), numticks=20))
        ax.xaxis.set_minor_formatter(
            LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
        )
        ax.set_title(_level_label(level, replicate_seeds), fontsize=10.5)
        ax.grid(alpha=0.3, which="both")

        # --- Spread annotation: the numbers behind the visual claim.
        stats = panel_spread_stats(sub)
        level_stats[level] = stats
        box_lines = [
            "spread of these 10 points",
            f"x: log10-sd {stats['x_log10_sd']:.3f} | max/min {stats['x_ratio']:.2f}x",
            f"y: log10-sd {stats['y_log10_sd']:.3f} | max/min {stats['y_ratio']:.6g}x",
        ]
        if len(floored):
            box_lines.append(
                f"y sd is a LOWER BOUND: {len(floored)} circled pt(s)\n"
                f"sit ON the noise_level lower bound"
            )
        bx, by, ha, va = _emptiest_corner(sub, xlim, ylim)
        ax.text(
            bx, by, "\n".join(box_lines), transform=ax.transAxes,
            ha=ha, va=va, fontsize=7.0, color="black", family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="dimgray", alpha=0.88),
            zorder=6,
        )

    # --- LIMIT CHECK (two clauses; raises ValueError, never `assert`, so
    # `python -O` cannot strip it and so it matches every other check in this
    # file). WHAT IT COVERS:
    #   (a) all 3 panels really did end up with the SAME final x limits and
    #       the SAME final y limits -- this is the actual contract of the
    #       shared-axes layout and nothing else in the script enforces it;
    #   (b) every plotted DATA POINT lies inside those final limits.
    # WHAT IT DOES NOT COVER, stated plainly so nobody over-trusts it: the
    # padding rule guarantees by construction that the padded limits contain
    # the data, so in normal operation clause (b) can only fire if something
    # downstream changed or overrode the limits after set_xlim/set_ylim (a
    # stray autoscale, a future edit plotting extra points). It says nothing
    # about whether the drawn MARKERS (s=70, and the s=230 boundary-stop
    # rings), the replicate-id annotation text, or the per-panel spread box
    # overflow the spines -- those live in display space and were checked by
    # eye only.
    x_all = df["length_scale_m"].to_numpy(dtype=float)
    y_all = df["y_value"].to_numpy(dtype=float)
    ref_x, ref_y = tuple(axes[0].get_xlim()), tuple(axes[0].get_ylim())
    for ax, level in zip(axes, ALL_AXIS_LEVELS):
        got_x, got_y = tuple(ax.get_xlim()), tuple(ax.get_ylim())
        # (a) identical limits across panels
        for name, ref, got in (("x", ref_x, got_x), ("y", ref_y, got_y)):
            if not np.allclose(got, ref, rtol=LIMIT_CHECK_RTOL, atol=0.0):
                raise ValueError(
                    f"{method}: panels do not share identical {name} limits -- panel "
                    f"axis_level={level} has [{got[0]!r}, {got[1]!r}] but the first panel "
                    f"({ALL_AXIS_LEVELS[0]}) has [{ref[0]!r}, {ref[1]!r}]. The whole point "
                    f"of this layout is that the 3 panels are directly comparable, so this "
                    f"is a hard failure."
                )
        # (b) data points inside those limits
        for name, lim, vals in (("x", got_x, x_all), ("y", got_y, y_all)):
            lo, hi = lim[0] * (1 - LIMIT_CHECK_RTOL), lim[1] * (1 + LIMIT_CHECK_RTOL)
            outside = vals[(vals < lo) | (vals > hi)]
            if outside.size:
                raise ValueError(
                    f"{method} axis_level={level}: {outside.size} plotted data point(s) fall "
                    f"outside the final shared {name} limits [{lim[0]:.6g}, {lim[1]:.6g}] and "
                    f"would be clipped from the saved figure: {np.sort(outside).tolist()}"
                )
    print(
        f"  limit check passed: all {len(axes)} panels share identical x/y limits, and all "
        f"{len(df)} plotted data points lie inside them (marker extents, annotation text and "
        f"the spread boxes are NOT covered by this check)."
    )

    plt.suptitle(
        f"{METHOD_LABELS[method]}: fitted length scale vs. "
        f"{'smoothing parameter' if method == 'rbf_bootstrap' else 'noise-variance (smoothing-equivalent) parameter'} "
        "across 10 sample-location replicates, 3 sample-density levels",
        fontsize=13, y=1.01,
    )

    # --- Caption numbers are FORMATTED FROM level_stats / floored_cells (the
    # values just computed from the manifests), never typed in as literals, so
    # the caption cannot disagree with the points it sits under.
    x_sds = [level_stats[lv]["x_log10_sd"] for lv in ALL_AXIS_LEVELS]
    x_ratios = [level_stats[lv]["x_ratio"] for lv in ALL_AXIS_LEVELS]
    y_sds = [level_stats[lv]["y_log10_sd"] for lv in ALL_AXIS_LEVELS]
    y_ratios = [level_stats[lv]["y_ratio"] for lv in ALL_AXIS_LEVELS]

    if method == "rbf_bootstrap":
        y_field_caption = (
            "y = params.best_smoothing, the CV-selected RBFInterpolator smoothing/"
            "regularization parameter (chosen from src/experiments/rbf_bootstrap.py's "
            "SMOOTHING_GRID), read fresh from each run's own manifest.json."
        )
        spread_caption = (
            f"Length-scale spread {_trend_phrase(x_sds)} (log10-sd {_fmt_seq(x_sds, '.3f')}; "
            f"{_fmt_seq(x_ratios, '.2f')}x). The smoothing parameter's spread "
            f"{_trend_phrase(y_sds)}: log10-sd {_fmt_seq(y_sds, '.3f')}. Because "
            f"best_smoothing is picked from a 7-value discrete grid its max/min ratios move "
            f"in coarse jumps ({_fmt_seq(y_ratios, '.6g')}x) rather than as a fine-grained "
            f"trend."
        )
    else:
        y_field_caption = (
            "y = params.fitted_hyperparameters.noise_variance_real_units, the MLE-fitted GP "
            "nugget/noise variance in real (Porosity %^2) units -- labelled explicitly as "
            "GP's smoothing-EQUIVALENT regularization term because this codebase has no "
            "hyperparameter literally named 'smoothing' for GP-MLE; see module docstring for "
            "the analogy this y-axis represents (orchestrator decision, 2026-09-18)."
        )
        spread_caption = (
            f"Length-scale spread {_trend_phrase(x_sds)} (log10-sd {_fmt_seq(x_sds, '.3f')}; "
            f"{_fmt_seq(x_ratios, '.2f')}x), and the noise variance's spread "
            f"{_trend_phrase(y_sds)} (log10-sd {_fmt_seq(y_sds, '.3f')})."
        )
        if floored_cells:
            cells_label = ", ".join(f"{lv}% {rep}" for lv, rep in floored_cells)
            floor_label = ", ".join(f"{v:g}" for v in sorted(noise_floor_values))
            spread_caption += (
                f" CIRCLED points ({cells_label}) stopped EXACTLY on the optimizer's "
                f"noise_level lower bound ({floor_label} normalized, read from each run's own "
                f"manifest), so for those runs the unconstrained optimum lies at or below the "
                f"bound and the y spread drawn here is a LOWER BOUND on the spread an "
                f"unconstrained fit would show. Whether that means the nugget is genuinely "
                f"~0 for those replicates, or simply that the nugget is not identifiable at "
                f"that sample density, is NOT distinguished by this figure."
            )

    caption = (
        f"Each panel: all 10 sample-seed replicates (rep0-rep9, sample_seed 1001-1010, "
        f"labelled by replicate index next to each point) at ONE sample-density-axis level, "
        f"SAME ground-truth field for every replicate/level (TRUTH_SEED={TRUTH_SEED}, "
        f"range={AXIS_HMAJ1:g} m). Panels ordered dense -> sparse left to right (5% -> 2% -> "
        "1% requested; actual post-dedup counts noted per panel). x = length_scale_m taken "
        "AS-IS from length_scale_by_replicate.csv (NOT recomputed here -- see that file's own "
        "per-row length_definition/source columns). " + y_field_caption + " Both axes are "
        "log-scaled, and ALL 3 PANELS OF THIS FIGURE SHARE IDENTICAL x AND y LIMITS "
        f"(x=[{xlim[0]:.4g}, {xlim[1]:.4g}] m, y=[{ylim[0]:.4g}, {ylim[1]:.4g}], computed from "
        "this figure's own 30 points and padded symmetrically in log space), so a wider cloud "
        "means a genuinely wider spread; y tick labels are drawn on the leftmost panel only. "
        "Limits are NOT shared with the other figure of this pair, whose y is a different "
        "physical quantity. The box in each panel reports that panel's spread: SD of "
        "log10(value) (ddof=1) and max/min ratio, for x and y separately. " + spread_caption
    )

    # Caption wrap widened from 195 to 265 chars (still inside the 18in figure
    # width at fontsize 7.3) and the axes bottom raised from 0.22 to 0.24: the
    # caption grew by the shared-limits / spread-box / boundary-stop
    # explanations, and at the old wrap width its taller text block ran into
    # the panels' x-axis labels.
    fig.text(
        0.5, 0.005, textwrap.fill(caption, 265),
        ha="center", va="bottom", fontsize=7.3, color="dimgray",
    )
    plt.subplots_adjust(left=0.055, bottom=0.24, right=0.99, top=0.82, wspace=0.32)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_name = (
        "length_vs_noise_gp_mle.png" if method == "gp_mle" else "length_vs_smoothing_rbf_bootstrap.png"
    )
    out = FIGURES_DIR / out_name
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"{out.name}: {out} ({out.stat().st_size} bytes)")
    return out


def verify_documented_values(method: str, df: pd.DataFrame) -> None:
    """Re-check the numbers quoted in this module's DOCSTRING against the ones
    just computed from the manifests, and fail loudly if they have drifted.

    Nothing drawn into a figure comes from DOCUMENTED_SPREAD /
    DOCUMENTED_BOUNDARY_STOP_CELLS / DOCUMENTED_NOISE_FLOOR -- the panels and
    captions format everything from the freshly computed stats. These
    constants exist only because the module docstring is prose and therefore
    cannot be generated; this function is what stops that prose from silently
    outliving the data. Comparison is at the same precision the docstring
    quotes (DOCUMENTED_DECIMALS), so a drift smaller than the printed
    precision is not reported -- the docstring would still read correctly.

    Raises ValueError (not assert, so `python -O` cannot strip the check)
    listing every disagreement at once, with the fix instruction: update BOTH
    the constants and the docstring text they mirror.
    """
    problems = []

    documented = DOCUMENTED_SPREAD[method]
    computed_by_level = {
        level: panel_spread_stats(df[df["axis_level"] == level])
        for level in ALL_AXIS_LEVELS
    }
    for key, expected_seq in documented.items():
        decimals = DOCUMENTED_DECIMALS[key]
        if len(expected_seq) != len(ALL_AXIS_LEVELS):
            raise ValueError(
                f"DOCUMENTED_SPREAD[{method!r}][{key!r}] has {len(expected_seq)} entries "
                f"but there are {len(ALL_AXIS_LEVELS)} axis levels."
            )
        for level, expected in zip(ALL_AXIS_LEVELS, expected_seq):
            got = computed_by_level[level][key]
            if round(got, decimals) != round(float(expected), decimals):
                problems.append(
                    f"{method} {key} at the {level}% level: docstring says "
                    f"{float(expected):.{decimals}f}, computed {got:.6g} "
                    f"(compared rounded to {decimals} decimals)"
                )

    # Boundary stops are a gp_mle-only concept: rbf_bootstrap picks its
    # smoothing from a discrete grid with no bounded optimizer, so
    # at_noise_floor is False for all of its rows by construction and there is
    # nothing for the documented cell list to mirror.
    if method == "gp_mle":
        computed_cells = {
            (row.axis_level, row.replicate)
            for row in df[df["at_noise_floor"]].itertuples()
        }
        documented_cells = set(DOCUMENTED_BOUNDARY_STOP_CELLS)
        if computed_cells != documented_cells:
            def _fmt(cells):
                return ", ".join(
                    f"{lv}% {rep}"
                    for lv, rep in sorted(cells, key=lambda c: (ALL_AXIS_LEVELS.index(c[0]), c[1]))
                ) or "(none)"
            problems.append(
                f"{method} boundary-stop cells: docstring names {_fmt(documented_cells)}, "
                f"computed {_fmt(computed_cells)}"
            )
        floors = sorted({
            float(v) for v in df[df["at_noise_floor"]]["noise_floor_normalized"]
        })
        for floor in floors:
            if floor != DOCUMENTED_NOISE_FLOOR:
                problems.append(
                    f"{method} noise_level lower bound: docstring quotes "
                    f"{DOCUMENTED_NOISE_FLOOR:g}, a run's manifest gives {floor:g}"
                )

    if problems:
        bullet = "\n  "
        raise ValueError(
            "the numbers quoted in this module's docstring no longer match the data:"
            + bullet + bullet.join(problems)
            + bullet + bullet
            + "The figures themselves are still correct (they are generated from "
            "the computed values), but the docstring is now stale. Update BOTH the "
            "docstring prose AND the DOCUMENTED_* constants that mirror it, then re-run."
        )
    print(
        f"  docstring cross-check passed: {sum(len(v) for v in documented.values())} "
        f"documented spread value(s)"
        + (
            f" and the {len(DOCUMENTED_BOUNDARY_STOP_CELLS)} documented boundary-stop cell(s)"
            if method == "gp_mle" else ""
        )
        + " match the freshly computed ones."
    )


def main():
    length_df = load_length_df()
    source_runs = load_source_runs()
    replicate_seeds = load_replicate_seeds()

    saved = []
    for method in ["gp_mle", "rbf_bootstrap"]:
        print(f"\nBuilding length-vs-smoothing dataset for {method}...")
        df = build_dataset(method, length_df, source_runs)
        verify_documented_values(method, df)
        saved.append(make_figure(method, df, replicate_seeds))

    print("\nAll length-vs-smoothing figures:")
    for f in saved:
        print(" ", f)
    return saved


if __name__ == "__main__":
    main()
