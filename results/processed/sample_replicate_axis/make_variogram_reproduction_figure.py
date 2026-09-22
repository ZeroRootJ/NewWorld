"""Variogram reproduction of the REALIZATIONS of SGS, RBF+bootstrap and GP-MLE
at the three sample-density levels (axis_level '1' = 1%, n_requested 25;
'2' = 2%, n_requested 50; '5' = 5%, n_requested 125), organised by
random-sampling REPLICATE (sample-replicate axis; ONE fixed ground truth, only
the sample locations change across rep0..rep9). Two figures are made PER LEVEL
(L = 1, 2, 5); every level uses the identical procedure and styling.

WHAT THE FIGURES SHOW (per level L)
-----------------------------------
Figure A  results/figures/sample_replicate_axis/
          variogram_reproduction_by_replicate_level<L>.png
    10 rows (rep0..rep9, labelled 'rep k (sample_seed s)') x 3 columns
    (SGS | RBF+bootstrap | GP-MLE). Each panel = that replicate's 10
    realizations of that method plus the truth.
Figure B  results/figures/sample_replicate_axis/
          variogram_reproduction_pooled_level<L>.png
    ONE row, 3 panels (SGS | RBF+bootstrap | GP-MLE); each panel pools all
    10 replicates x 10 realizations = 100 curves (lower alpha, POOLED_ALPHA).
    Same axes (x and y limits) as Figure A OF THE SAME LEVEL. A figure-level
    legend sits above the panels in both figures.

Each panel draws
  * the truth's experimental variogram: SOLID thick black line with markers
    (identical in every panel: the truth is one field);
  * the experimental variogram of EVERY realization: thin line in the method's
    project colour (sgs tab:green, rbf_bootstrap tab:orange, gp_mle tab:red --
    the colours the existing figures use) at LOW alpha (ALPHA in Figure A,
    POOLED_ALPHA in Figure B);
  * NO mean / ensemble-mean variogram anywhere;
  * REFERENCE LINES ONLY (no estimation involved): the truth's theoretical
    spherical model (dashed; nugget 0.45, structured sill 8.55, range 300 m) and
    the total sill (dotted, 9.0).
x = 0-750 m (25 m bins). The y-axis is FIXED at 0 to Y_MAX_SILL_FACTOR x the
total sill = 1.3 x 9 = 11.7 (computed from the imported POR_STDEV**2) in EVERY
panel of EVERY figure (all three levels, by-replicate and pooled figures). Curves
exceeding it are simply cut off by the axis (the data are not clipped or
altered); the share of curve points above the limit is printed and recorded per
(level, method) in the seeds JSON. Each level's max plotted value is kept in the
JSON for reference.
Simple kriging is deliberately not shown (a smooth estimate, not a set of
realizations).

WHAT "REALIZATION" MEANS PER METHOD
-----------------------------------
  SGS            the 10 conditional simulations in sgs_realizations.npy
                 (shape (10, 50, 50)) of the run referenced by
                 results/processed/sample_replicate_axis/source_runs.json
                 (that level, that replicate; the live file is used for every
                 level, incl. the 2026-09-21 RBF runs at level '5', never the
                 frozen source_runs_level5_n125.json).
  RBF+bootstrap  the 10 bootstrap replicate maps in bootstrap_replicate_maps.npy
                 of the referenced run: RESAMPLED INTERPOLATIONS (the RBF refit
                 on a bootstrap resample of the conditioning samples), NOT
                 conditional simulations.
  GP-MLE         10 joint draws from the fitted GP posterior over the full
                 50x50 grid. Each run stores only ONE draw
                 (posterior_sample_map.npy), so the 10 draws per replicate are
                 produced here on the fly (nothing is written under results/raw;
                 nothing is cached on disk) by re-building the FITTED GP from the
                 run's manifest (fitted hyperparameters, normalize_y) and the
                 run's samples.csv WITHOUT re-optimising (fixed kernel,
                 optimizer=None), then drawing with the SAME procedure
                 src/experiments/gp_mle.py uses for the stored draw:
                 gpr.sample_y(grid_coords, n_samples=...) first, with the same
                 manual Cholesky fallback (identical N(y_mean, y_cov)).
                 gp_mle.py has no importable sampling function (it lives inside
                 main()), so that ~10-line procedure is mirrored in
                 ``draw_gp_posterior`` below; GP_SAMPLE_SEED, the grid, VCOL are
                 imported from the project modules, not re-declared.
                 NOTE (fact, not a design choice made here): sklearn's
                 predict(return_cov=True), which sample_y uses, includes the
                 fitted WhiteKernel noise on the diagonal of the covariance, so
                 each draw (like the stored one) carries that white-noise
                 component.

GP RECONSTRUCTION VALIDATION (every replicate of every level; raises on failure)
-----------------------------------------------------------------------
  (i)   reconstructed posterior mean / variance maps vs. the run's stored
        posterior_mean_map.npy / posterior_var_map.npy
        (np.allclose, rtol=GP_MAP_RTOL, atol=GP_MAP_ATOL);
  (ii)  n_samples=1 with GP_SAMPLE_SEED (=55, the run's own seed) through the
        same code path vs. the stored posterior_sample_map.npy: the max abs
        difference and correlation are PRINTED AND RECORDED. Exact equality is
        NOT required. Observed: the seed-55 draw reproduces the stored draw
        closely ONLY for some replicates (e.g. at level 1 rep0 and rep5, corr
        ~1.0, max|diff| ~0.03; the other 8 do not: corr 0.16-0.91, max|diff|
        4.7-19.4; per-level numbers for all levels are in the JSON).
        Cause, established empirically: the posterior covariance contains the
        white-noise floor, i.e. a numerically (near-)degenerate eigenvalue
        cluster whose size varies by replicate (n_eig_within_1e-6_rel_of_min in
        the JSON; e.g. level 1: 948 for rep0 up to 2415 for rep8, out of 2500), so the singular vectors numpy's SVD (inside sample_y) returns for that
        subspace are arbitrary rotations that change completely under
        perturbations of the covariance at the 1e-12 level (a 1e-12 random
        perturbation of the covariance changes the draw by the same O(10)
        amount as the stored-vs-reconstructed difference; the SVD factor itself
        reproduces the covariance to ~1e-13, so each draw is a valid sample).
        The seed therefore fixes the draw only for a bit-identical covariance
        matrix. Instead of exact equality, (ii) is checked as a
        distribution-consistency test: the stored draw x must be a plausible
        draw from the RECONSTRUCTED posterior N(m, C),
        q = (x-m)^T C^-1 (x-m) ~ chi2(2500); |q - 2500| / sqrt(2*2500) <=
        GP_CHI2_MAX_ABS_Z (default 4) is required (the same statistic is
        recorded for the reconstructed seed-55 draw). Because any draw
        x = m + A z with A A^T = C has q = |z|^2 for the same standard-normal
        vector z, the stored and the reconstructed seed-55 draws must give the
        SAME q (they do, to the printed precision); this is a sharp check that
        the rebuilt C is the covariance the stored draw came from, independent
        of the arbitrary SVD rotation;
  (iii) determinism: the new figure draws are reproducible run to run on this
        machine (same code, same inputs -> same SVD); this is asserted for
        rep0 by drawing the seed-55 sample twice.
The eigenvalue-cluster diagnostic (count of eigenvalues within 1e-6 relative of
the smallest; level 1: 948-2415 of 2500 depending on the replicate) is recorded
in the JSON for every (level, replicate).

SEEDS (all recorded; every one is deterministic; no other randomness)
---------------------------------------------------------------------
  TRUTH_SEED            = 101   (truth field)
  replicate rep k       = sample_seed 1001+k (conditioning samples)
  VARIOGRAM_PAIR_SEED   = 90    (200,000 random (i,j) pairs, shared estimator;
                                  every curve, truth included, is evaluated on
                                  the IDENTICAL pair set and 25 m bins)
  GP_SAMPLE_SEED        = 55    (validation draw only, same as the stored runs)
  GP_DRAW_SEED_BASE_BY_LEVEL = {'1': 9000, '2': 9100, '5': 9200}  (NEW; the GP
                                  draws of rep k at level L use seed
                                  base[L] + k, i.e. 9000..9009 / 9100..9109 /
                                  9200..9209, never colliding; one
                                  sample_y(n_samples=10) call per (level,
                                  replicate))
The seeds are also written to variogram_reproduction_seeds.json next to the
summary CSV.

VARIOGRAM ESTIMATOR
-------------------
The SAME estimator as the case-study figures (imported
``compute_experimental_variogram`` / ``spherical_semivariance``; 200,000 pairs
at VARIOGRAM_PAIR_SEED=90, 25 m bins, 0-750 m), applied to the FULL 50x50 grid
of each map (not the masked evaluation support). The semivariance at the 287.5 m
bin is summarised as raw value and as % of the truth's theoretical model value.

OUTPUTS
-------
  results/figures/sample_replicate_axis/variogram_reproduction_by_replicate_level{1,2,5}.png
  results/figures/sample_replicate_axis/variogram_reproduction_pooled_level{1,2,5}.png
  results/processed/sample_replicate_axis/variogram_reproduction_summary.csv
      tidy long: axis_level ('1','2','5'), replicate, method, n_realizations,
      lag_m, stat, value_semivariance, value_pct_of_model; stat in {min,
      median, max}; replicate = 'rep0'..'rep9' and 'pooled' (all 100
      realizations of a method); the truth is one row per level with
      method='truth', replicate='all', stat='value', n_realizations empty.
      Re-running a subset of levels replaces only those levels' rows.
  results/processed/sample_replicate_axis/variogram_reproduction_seeds.json
      (per-level block under "levels": seeds, source runs, GP validation,
      timings, exceed-y-limit statistics; fixed y-limit rule at top level; re-running a subset of levels replaces only those)

2026-09-21 EXTENSION: levels '10' (10%, n_requested 250) and '20' (20%, n_requested
500) were added with the IDENTICAL procedure/design and the same fixed y-axis 0-11.7
(GP draw seeds 9300+k for level '10', 9400+k for level '20'; levels 1/2/5 untouched:
9000/9100/9200+k). The LEVELS order is ("1", "2", "5", "10", "20"), so the new rows are
appended after the existing 300 rows of the summary CSV. Outputs for the new levels:
variogram_reproduction_{by_replicate,pooled}_level{10,20}.png. At these denser levels
the GP fits use n ~ 235 (10%) / ~445 (20%) conditioning points, so the posterior
covariance eigen-structure (size of the white-noise eigenvalue cluster) differs from
levels 1/2/5; the procedure is unchanged and every replicate is validated as before.
Every SGS realization is additionally scanned for values outside the physical range
POR_MEAN +- 4 POR_STDEV (= 3..27, the back-transform limits); any hits are recorded in
the seeds JSON ("sgs_out_of_range_values") and stated in that level's figure footnote
(nothing is altered or clipped; the variogram is computed from the stored map as is).
Known fact at level '10': rep4 (sample_seed 1005), SGS realization index 9 contains a
value of about -304.6, so that realization's variogram is huge and is cut off by the
fixed y-axis.

Footnote layout: a footnote with more wrapped lines than the levels 1/2/5 ones (10 / 11
lines for figure A / B; see FOOTNOTE_BASE_LINES, _footnote_layout) makes the figure taller
by the missing lines plus a small pad so the footnote never touches the x-axis label; with
no extra lines the layout is unchanged (levels 1, 2, 5 and 20 render as before; only the
level-10 figures, whose footnote carries the extra out-of-range NOTE sentence, are taller).

Run with (default: all five levels, ~5 min per level for the sparse ones, dominated by
the GP draws; the denser levels take longer):
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_variogram_reproduction_figure
Optional: --levels 10 20   (only those levels are regenerated / their CSV rows and JSON
blocks replaced; all other levels' figures, rows and blocks stay as they are)
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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.base_case import (  # noqa: E402
    NX, NY, XMN, YMN, XSIZ, YSIZ, CC1, HMAJ1, NUG, POR_MEAN, POR_STDEV,
)
from src.experiments.base_case_conditioning import (  # noqa: E402
    VCOL,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.gp_mle import GP_SAMPLE_SEED  # noqa: E402
from src.experiments.sample_replicate_axis import (  # noqa: E402
    AXIS_HMAJ1,
    N_SAMPLES_REQUESTED_BY_LEVEL,
    REPLICATE_IDS,
    REPLICATE_SEED,
    TRUTH_SEED,
)
from src.grid_utils import full_grid_coordinates  # noqa: E402

from results.processed.sample_replicate_axis import make_length_case_study_figures as base  # noqa: E402
from results.processed.sample_replicate_axis.make_length_case_study_figures import (  # noqa: E402
    CURVE_COLORS,
    FIG_DPI,
    LAG_BIN_WIDTH_M,
    LAG_MAX_M,
    MIN_PAIRS_PER_BIN,
    N_PAIRS_DRAWN,
    PROCESSED_DIR,
    SAMPLES_MATCH_ATOL,
    VARIOGRAM_PAIR_SEED,
    compute_experimental_variogram,
    spherical_semivariance,
)

FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "sample_replicate_axis"
FIGURE_A_NAME_TEMPLATE = "variogram_reproduction_by_replicate_level{level}.png"
FIGURE_B_NAME_TEMPLATE = "variogram_reproduction_pooled_level{level}.png"
SUMMARY_CSV_NAME = "variogram_reproduction_summary.csv"
SEEDS_JSON_NAME = "variogram_reproduction_seeds.json"

# --- design constants (all named; change here only) -------------------------
LEVELS = ("1", "2", "5", "10", "20")     # axis levels, in this order (new rows are appended)
# Text used in the figure titles: level 1 keeps its ORIGINAL wording so the
# level-1 PNGs stay byte-identical to the first version of this script.
LEVEL_TITLE = {
    "1": "1% level, n=25",
    "2": "2% level, n=50",
    "5": "5% level, n requested 125",
    "10": "10% level, n requested 250",
    "20": "20% level, n requested 500",
}
LEVEL_PERCENT = {"1": "1%", "2": "2%", "5": "5%", "10": "10%", "20": "20%"}
REPLICATES = tuple(REPLICATE_IDS)        # rep0 .. rep9
METHODS = ("sgs", "rbf_bootstrap", "gp_mle")   # figure columns, left to right
METHOD_LABELS = {"sgs": "SGS", "rbf_bootstrap": "RBF+bootstrap", "gp_mle": "GP-MLE"}
# Same per-method colours as the existing figures (make_fixed_range_case_study_figures:
# sgs tab:green; make_length_case_study_figures / make_length_vs_smoothing_figures:
# rbf_bootstrap tab:orange, gp_mle tab:red).
METHOD_COLORS = {"sgs": "tab:green", "rbf_bootstrap": "tab:orange", "gp_mle": "tab:red"}

ALPHA = 0.25                 # Figure A: low alpha of each realization's line
POOLED_ALPHA = 0.12          # Figure B: 100 curves per panel
LEGEND_SWATCH_ALPHA = 0.8    # legend swatch only (a 0.12-0.25 swatch is unreadable)
REALIZATION_LINEWIDTH = 1.0
POOLED_LINEWIDTH = 0.9
TRUTH_LINEWIDTH = 2.4
TRUTH_MARKERSIZE = 3.5
# The y-axis of ALL variogram-reproduction figures is 0 .. Y_MAX_SILL_FACTOR x the
# total sill (= POR_STDEV**2 = 9), i.e. 11.7; curves above it are cut off.
Y_MAX_SILL_FACTOR = 1.3
Y_MAX = Y_MAX_SILL_FACTOR * POR_STDEV ** 2
YLIM = (0.0, Y_MAX)

N_REALIZATIONS_EXPECTED = 10
N_GP_DRAWS = N_REALIZATIONS_EXPECTED
# NEW seed families, one per level (never colliding with each other or with
# TRUTH/sample/pair/GP_SAMPLE seeds); GP draw seed of rep k at level L =
# GP_DRAW_SEED_BASE_BY_LEVEL[L] + k  (1%: 9000..9009, 2%: 9100..9109, 5%: 9200..9209,
# 10%: 9300..9309, 20%: 9400..9409).
GP_DRAW_SEED_BASE_BY_LEVEL = {"1": 9000, "2": 9100, "5": 9200, "10": 9300, "20": 9400}

REPORT_LAG_M = 287.5         # lag bin reported in the summary (a 25 m bin centre)
REPORT_STATS = ("min", "median", "max")

# GP reconstruction validation (see docstring). Mean / variance are rebuilt from
# full-precision JSON hyperparameters, so they agree to ~1e-14; the tolerances are
# a stated safety margin, not a fit.
GP_MAP_RTOL = 1e-6
GP_MAP_ATOL = 1e-6
GP_CHI2_MAX_ABS_Z = 4.0      # |q-N| / sqrt(2N) bound for the stored draw under the rebuilt N(m, C)
EIG_CLUSTER_REL_TOL = 1e-6   # diagnostic only
# Physical range of the back-transformed porosity (base_case BACKTR_ZMIN/ZMAX =
# POR_MEAN -+ 4 POR_STDEV = 3..27); SGS values outside it are reported (not altered).
PHYS_RANGE = (POR_MEAN - 4.0 * POR_STDEV, POR_MEAN + 4.0 * POR_STDEV)

# Footnote layout: FOOTNOTE_BASE_LINES are the wrapped-line counts of the levels 1/2/5
# footnotes (figure A: 170 chars/line, figure B: 175 chars/line) for which the fixed
# layout below was tuned. A footnote with MORE lines than that (e.g. level 10's extra
# out-of-range NOTE sentence) makes the figure taller by exactly the missing lines
# (+ FOOTNOTE_GAP_PAD_IN of extra air), keeping panels, legend and the distance
# footnote <-> x-axis label as for the base case. With no extra lines nothing changes
# (the figures of levels 1/2/5/20 are unaffected).
FOOTNOTE_BASE_LINES = {"A": 10, "B": 11}
FOOTNOTE_FONTSIZE = 8
FOOTNOTE_LINE_HEIGHT_IN = FOOTNOTE_FONTSIZE * 1.25 / 72.0
FOOTNOTE_GAP_PAD_IN = 0.15


def _footnote_layout(kind, wrapped_footnote, height_in):
    """Returns (new_height_in, top(f), bottom(f), foot(f)) coordinate maps, all identity
    when the footnote needs no extra lines. top(f): a fraction measured from the bottom
    of the ORIGINAL figure that is anchored at the top (legend, suptitle, panels top);
    bottom(f): the tight_layout rect bottom; foot(f): the footnote text baseline."""
    extra = max(0, wrapped_footnote.count(chr(10)) + 1 - FOOTNOTE_BASE_LINES[kind])
    if extra == 0:
        ident = lambda f: f  # noqa: E731
        return height_in, ident, ident, ident, 0
    dh = extra * FOOTNOTE_LINE_HEIGHT_IN + FOOTNOTE_GAP_PAD_IN
    h2 = height_in + dh
    return (h2, lambda f: 1.0 - (1.0 - f) * height_in / h2,
            lambda f: (f * height_in + dh) / h2, lambda f: f * height_in / h2, extra)


FIG_A_SIZE = (15.0, 30.0)
FIG_B_SIZE = (15.0, 5.6)
# Figure-level legend placement (figure fractions): legend sits between the
# suptitle and the column titles; panels start below FIG_*_PANELS_TOP.
FIG_A_LEGEND_Y = 0.9845
FIG_A_PANELS_TOP = 0.968
FIG_B_LEGEND_Y = 0.925
FIG_B_PANELS_TOP = 0.85
TICK_FONTSIZE = 10
LABEL_FONTSIZE = 11


# ---------------------------------------------------------------------------
# GP reconstruction + drawing
# ---------------------------------------------------------------------------
def rebuild_fitted_gpr(run_dir: Path):
    """Rebuild the run's FITTED GaussianProcessRegressor from its manifest
    (params.fitted_hyperparameters, params.normalize_y) and samples.csv,
    without re-optimising: kernels are 'fixed' and optimizer=None, so fit()
    only factorises K + noise and computes the normalised targets."""
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        params = json.load(f)["params"]
    fh = params["fitted_hyperparameters"]
    kernel = (
        ConstantKernel(fh["signal_variance_normalized"], "fixed")
        * RBF(length_scale=fh["length_scale_m"], length_scale_bounds="fixed")
        + WhiteKernel(noise_level=fh["noise_variance_normalized"], noise_level_bounds="fixed")
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, optimizer=None, normalize_y=bool(params["normalize_y"]),
    )
    samples = pd.read_csv(run_dir / "samples.csv")
    gpr.fit(samples[["X", "Y"]].values, samples[VCOL].values)
    if not np.isclose(float(gpr._y_train_std), fh["y_train_std"], rtol=1e-12, atol=0.0):
        raise ValueError(
            f"{run_dir}: rebuilt y_train_std {float(gpr._y_train_std)!r} != manifest "
            f"{fh['y_train_std']!r}."
        )
    return gpr, params


def draw_gp_posterior(gpr, grid_coords, n_samples: int, seed: int):
    """n_samples joint posterior draws over the full grid, SAME procedure as
    src/experiments/gp_mle.py: gpr.sample_y() first; on numpy's SVD
    non-convergence fall back to a Cholesky draw from the identical full
    posterior N(y_mean, y_cov). Returns ((n_samples, NY, NX) array, method)."""
    try:
        draws = gpr.sample_y(grid_coords, n_samples=n_samples, random_state=seed)
        method = "sklearn_sample_y_svd"
    except np.linalg.LinAlgError as exc:
        print(
            f"WARNING: gpr.sample_y() raised {exc!r} -- falling back to a manual Cholesky "
            "draw from the identical full posterior (as gp_mle.py does)."
        )
        y_mean, y_cov = gpr.predict(grid_coords, return_cov=True)
        L = np.linalg.cholesky(y_cov)
        z = np.random.RandomState(seed).standard_normal((y_mean.shape[0], n_samples))
        draws = y_mean[:, None] + L @ z
        method = "manual_cholesky_fallback"
    return draws.T.reshape(n_samples, NY, NX), method


def _chi2_z(x_flat, mean, chol):
    """z-score of q = (x-m)^T C^-1 (x-m) against chi2(n): (q-n)/sqrt(2n)."""
    w = np.linalg.solve(chol, x_flat - mean)
    n = x_flat.shape[0]
    q = float(w @ w)
    return q, (q - n) / np.sqrt(2.0 * n)


def validate_gp_reconstruction(gpr, run_dir: Path, grid_coords, replicate_id: str,
                               check_determinism: bool) -> dict:
    """See module docstring, GP RECONSTRUCTION VALIDATION."""
    post_mean, post_std = gpr.predict(grid_coords, return_std=True)
    mean_map = post_mean.reshape(NY, NX)
    var_map = (post_std ** 2).reshape(NY, NX)
    stored_mean = np.load(run_dir / "posterior_mean_map.npy")
    stored_var = np.load(run_dir / "posterior_var_map.npy")
    stored_sample = np.load(run_dir / "posterior_sample_map.npy")
    for name, new, old in (
        ("posterior mean", mean_map, stored_mean),
        ("posterior variance", var_map, stored_var),
    ):
        if not np.allclose(new, old, rtol=GP_MAP_RTOL, atol=GP_MAP_ATOL):
            raise ValueError(
                f"{replicate_id}: reconstructed {name} does not match {run_dir.name} "
                f"(max abs diff {np.max(np.abs(new - old)):.3e}; rtol={GP_MAP_RTOL}, "
                f"atol={GP_MAP_ATOL})."
            )

    one_draw, method = draw_gp_posterior(gpr, grid_coords, 1, GP_SAMPLE_SEED)
    if check_determinism:
        again, _ = draw_gp_posterior(gpr, grid_coords, 1, GP_SAMPLE_SEED)
        if not np.array_equal(one_draw, again):
            raise ValueError(f"{replicate_id}: seed-{GP_SAMPLE_SEED} draw is not deterministic "
                             "run to run.")
    draw_diff = float(np.max(np.abs(one_draw[0] - stored_sample)))
    draw_corr = float(np.corrcoef(one_draw[0].ravel(), stored_sample.ravel())[0, 1])

    m_full, cov = gpr.predict(grid_coords, return_cov=True)
    chol = np.linalg.cholesky(cov)
    q_stored, z_stored = _chi2_z(stored_sample.ravel(), m_full, chol)
    q_new, z_new = _chi2_z(one_draw[0].ravel(), m_full, chol)
    if abs(z_stored) > GP_CHI2_MAX_ABS_Z:
        raise ValueError(
            f"{replicate_id}: the stored posterior draw is not a plausible draw from the "
            f"reconstructed posterior (chi2 z-score {z_stored:.2f}, limit {GP_CHI2_MAX_ABS_Z})."
        )
    eig = np.linalg.eigvalsh(cov)
    n_cluster = int(np.sum(eig <= eig.min() * (1.0 + EIG_CLUSTER_REL_TOL)))

    out = {
        "max_abs_diff_mean": float(np.max(np.abs(mean_map - stored_mean))),
        "max_abs_diff_var": float(np.max(np.abs(var_map - stored_var))),
        "max_abs_diff_seed55_draw_vs_stored": draw_diff,
        "corr_seed55_draw_vs_stored": draw_corr,
        "chi2_z_stored_draw": z_stored,
        "chi2_z_reconstructed_seed55_draw": z_new,
        "cov_eig_min": float(eig.min()),
        "cov_eig_max": float(eig.max()),
        "n_eig_within_1e-6_rel_of_min": n_cluster,
        "draw_method_reproduction": method,
        "draw_method_stored": json.load(open(run_dir / "manifest.json"))["params"][
            "posterior_sample_method"],
        "seed55_draw_deterministic_checked": bool(check_determinism),
    }
    print(
        f"  [GP {replicate_id}] mean/var match stored (max|diff| {out['max_abs_diff_mean']:.1e} / "
        f"{out['max_abs_diff_var']:.1e}); seed-{GP_SAMPLE_SEED} draw vs stored: max|diff|="
        f"{draw_diff:.3f}, corr={draw_corr:.5f} (not required equal, see docstring); chi2 z: "
        f"stored={z_stored:+.2f}, rebuilt={z_new:+.2f}; cov eig [{eig.min():.3g}, {eig.max():.3g}], "
        f"{n_cluster}/{eig.size} eigenvalues within {EIG_CLUSTER_REL_TOL:g} rel of the min"
    )
    return out


# ---------------------------------------------------------------------------
# Variograms
# ---------------------------------------------------------------------------
def quiet_variogram(map_2d: np.ndarray) -> pd.DataFrame:
    """The shared estimator with its per-call progress printing suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_experimental_variogram(map_2d)


def realization_variograms(maps: np.ndarray, truth_vario: pd.DataFrame, what: str):
    out = []
    for k in range(maps.shape[0]):
        v = quiet_variogram(maps[k])
        if not (np.array_equal(v["lag_bin_center_m"].values, truth_vario["lag_bin_center_m"].values)
                and np.array_equal(v["n_pairs"].values, truth_vario["n_pairs"].values)):
            raise ValueError(f"{what} #{k}: variogram not on the truth's lag bins / pair set.")
        out.append(v)
    return out


def semivariance_at(vario: pd.DataFrame, lag_m: float) -> float:
    centers = vario["lag_bin_center_m"].values
    b = int(np.argmin(np.abs(centers - lag_m)))
    if not np.isclose(centers[b], lag_m, rtol=0.0, atol=1e-9):
        raise ValueError(f"no lag bin centred at {lag_m} m (nearest {centers[b]}).")
    return float(vario["semivariance_empirical"].values[b])


# ---------------------------------------------------------------------------
# Data loading (one replicate)
# ---------------------------------------------------------------------------
def load_replicate(level: str, replicate_id: str, source_runs: dict, truth: np.ndarray,
                   grid_coords, check_determinism: bool):
    runs = {m: _REPO_ROOT / source_runs[level][replicate_id][m] for m in METHODS}
    sample_seed = int(REPLICATE_SEED[replicate_id])
    n_requested = N_SAMPLES_REQUESTED_BY_LEVEL[level]

    # Same conditioning samples in all three runs and equal to the regenerated ones.
    ref = get_conditioning_samples(truth, sample_seed=sample_seed, n_samples=n_requested)
    for m, rd in runs.items():
        rec = pd.read_csv(rd / "samples.csv")
        if len(rec) != len(ref) or not np.allclose(
            rec[["X", "Y", VCOL]].values, ref[["X", "Y", VCOL]].values, rtol=0.0,
            atol=SAMPLES_MATCH_ATOL,
        ):
            raise ValueError(
                f"level {level} {replicate_id} {m}: samples.csv does not match "
                f"sample_seed={sample_seed}, n_requested={n_requested}."
            )

    sgs = np.load(runs["sgs"] / "sgs_realizations.npy")
    rbf = np.load(runs["rbf_bootstrap"] / "bootstrap_replicate_maps.npy")
    for name, arr in (("sgs_realizations", sgs), ("bootstrap_replicate_maps", rbf)):
        if arr.shape != (N_REALIZATIONS_EXPECTED, NY, NX):
            raise ValueError(f"level {level} {replicate_id}: {name} has shape {arr.shape}, expected "
                             f"({N_REALIZATIONS_EXPECTED}, {NY}, {NX}) (full grid).")

    sgs_oor = []
    for k in range(sgs.shape[0]):
        lo, hi = float(np.min(sgs[k])), float(np.max(sgs[k]))
        if lo < PHYS_RANGE[0] or hi > PHYS_RANGE[1]:
            sgs_oor.append({"realization_index": k, "min": lo, "max": hi})
            print(f"  NOTE: SGS realization {k} of L{level} {replicate_id} has values outside the "
                  f"physical range {PHYS_RANGE}: min={lo:.4f}, max={hi:.4f} (kept as is)")

    gpr, _ = rebuild_fitted_gpr(runs["gp_mle"])
    validation = validate_gp_reconstruction(gpr, runs["gp_mle"], grid_coords,
                                            f"L{level} {replicate_id}", check_determinism)
    seed = GP_DRAW_SEED_BASE_BY_LEVEL[level] + REPLICATES.index(replicate_id)
    t0 = time.time()
    gp_draws, method = draw_gp_posterior(gpr, grid_coords, N_GP_DRAWS, seed)
    seconds = time.time() - t0
    print(f"  [GP L{level} {replicate_id}] drew {N_GP_DRAWS} posterior draws, seed={seed}, "
          f"method={method}, {seconds:.1f} s")

    return {
        "sample_seed": sample_seed,
        "n_requested": n_requested,
        "n_actual": len(ref),
        "maps": {"sgs": sgs, "rbf_bootstrap": rbf, "gp_mle": gp_draws},
        "runs": {m: str(source_runs[level][replicate_id][m]) for m in METHODS},
        "sgs_out_of_range": sgs_oor,
        "gp_validation": validation,
        "gp_draw_seed": seed,
        "gp_draw_method": method,
        "gp_draw_seconds": seconds,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _draw_reference_and_truth(ax, truth_vario, h_smooth, model_smooth, total_sill, ylim):
    well_t = truth_vario[truth_vario["n_pairs"] >= MIN_PAIRS_PER_BIN]
    sparse_t = truth_vario[truth_vario["n_pairs"] < MIN_PAIRS_PER_BIN]
    ax.plot(
        well_t["lag_bin_center_m"], well_t["semivariance_empirical"],
        color=CURVE_COLORS["truth"], marker="o", markersize=TRUTH_MARKERSIZE,
        linestyle="-", linewidth=TRUTH_LINEWIDTH, zorder=4,
    )
    if len(sparse_t) > 0:
        ax.scatter(sparse_t["lag_bin_center_m"], sparse_t["semivariance_empirical"],
                   color=CURVE_COLORS["truth"], marker="x", s=22, alpha=0.5, zorder=4)
    ax.plot(h_smooth, model_smooth, color=CURVE_COLORS["model"], linestyle="--",
            linewidth=1.8, zorder=3)
    ax.axhline(total_sill, color="gray", linestyle=":", linewidth=1.2, zorder=1)
    ax.set_xlim(0.0, LAG_MAX_M)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=TICK_FONTSIZE)


LEGEND_REALIZATION_SWATCH_COLOR = "0.35"   # neutral: realizations are coloured by method


def _figure_legend(fig, n_curves, alpha, linewidth, anchor_y):
    """ONE figure-level legend above the panels (never inside the data area)."""
    fig.legend(
        handles=_legend_handles(LEGEND_REALIZATION_SWATCH_COLOR, n_curves, alpha, linewidth),
        loc="upper center", bbox_to_anchor=(0.5, anchor_y), ncol=3, fontsize=10,
        framealpha=0.95,
    )


def _legend_handles(color, n_curves, alpha, linewidth):
    return [
        Line2D([], [], color=CURVE_COLORS["truth"], marker="o", markersize=TRUTH_MARKERSIZE,
               linewidth=TRUTH_LINEWIDTH, label="truth (experimental)"),
        Line2D([], [], color=color, linewidth=linewidth + 0.8, alpha=LEGEND_SWATCH_ALPHA,
               label=f"realizations (n={n_curves} per panel, alpha={alpha:g}; colour = method)"),
        Line2D([], [], color=CURVE_COLORS["model"], linestyle="--", linewidth=1.8,
               label="theoretical model"),
    ]


def _plot_curves(ax, curves, color, alpha, linewidth):
    for v in curves:
        well = v[v["n_pairs"] >= MIN_PAIRS_PER_BIN]
        ax.plot(well["lag_bin_center_m"], well["semivariance_empirical"], color=color,
                linewidth=linewidth, alpha=alpha, zorder=2)


def _common_footnote_text(level, data):
    total_sill = POR_STDEV ** 2
    n_actual = ", ".join(f"{r}: {data[r]['n_actual']}" for r in REPLICATES)
    ylim_sentence = (
        f"Y-axis fixed at 0-{Y_MAX:g} ({Y_MAX_SILL_FACTOR:g} x total sill {total_sill:g}) for "
        "every level and both figure types; curves exceeding it are cut off at the axis."
    )
    oor = [(r, e) for r in REPLICATES for e in data[r]["sgs_out_of_range"]]
    if oor:
        if len(oor) == 1:
            r, e = oor[0]
            bad = e["min"] if abs(e["min"] - POR_MEAN) >= abs(e["max"] - POR_MEAN) else e["max"]
            ylim_sentence += (
                f" NOTE: one SGS realization of {r} contains an out-of-range value ({bad:.1f}; "
                f"physical range {PHYS_RANGE[0]:g}-{PHYS_RANGE[1]:g}); its curve is cut off."
            )
        else:
            ylim_sentence += (
                f" NOTE: {len(oor)} SGS realizations ("
                + ", ".join(f"{r} #{e['realization_index']}" for r, e in oor)
                + f") contain out-of-range values (physical range {PHYS_RANGE[0]:g}-"
                f"{PHYS_RANGE[1]:g}); their curves are cut off."
            )
    return (
        f"{LEVEL_PERCENT[level]} sample-density level "
        f"(n_samples_requested={N_SAMPLES_REQUESTED_BY_LEVEL[level]}; "
        f"actual conditioning samples after grid snapping per replicate: {n_actual}); ONE fixed "
        f"ground truth (TRUTH_SEED={TRUTH_SEED}, range={AXIS_HMAJ1:g} m), only the sample "
        f"locations change across rep0-rep9 (sample_seed {REPLICATE_SEED[REPLICATES[0]]}-"
        f"{REPLICATE_SEED[REPLICATES[-1]]}). Every curve is an experimental variogram of a FULL "
        f"{NX}x{NY} grid map from the same estimator as the case-study figures: "
        f"{N_PAIRS_DRAWN:,} random cell pairs (VARIOGRAM_PAIR_SEED={VARIOGRAM_PAIR_SEED}, "
        f"identical pair set for all curves), {LAG_BIN_WIDTH_M:g} m lag bins over 0-{LAG_MAX_M:g} m. "
        "Solid black = truth field; thin coloured lines = one variogram per realization; no mean "
        f"variogram is drawn. Dashed = the truth's theoretical spherical model "
        f"(nugget={NUG * POR_STDEV ** 2:g}, structured sill={CC1 * POR_STDEV ** 2:g}, "
        f"range={HMAJ1:g} m) and dotted = total sill ({total_sill:g}); both are reference lines "
        "only, not estimates. Realizations: SGS = the 10 stored conditional simulations per "
        "replicate; RBF+bootstrap = the 10 stored bootstrap replicate maps per replicate "
        "(resampled interpolations, NOT conditional simulations); GP-MLE = 10 joint draws per "
        "replicate from the fitted GP posterior over the full grid (sklearn sample_y, includes "
        "the fitted white-noise term), re-drawn here from each run's fitted hyperparameters (not "
        "re-optimised; reconstruction checked against each run's stored posterior mean/variance), "
        f"draw seeds {GP_DRAW_SEED_BASE_BY_LEVEL[level]}+k for rep k. Legend swatches are drawn "
        "more opaque than the plotted lines for legibility. Simple kriging is not shown (a smooth "
        f"estimate, not a set of realizations). {ylim_sentence}"
    )


def make_figure_a(level, data, varios, truth_vario, ylim):
    h_smooth = np.linspace(0.0, LAG_MAX_M, 300)
    model_smooth = spherical_semivariance(h_smooth)
    total_sill = POR_STDEV ** 2
    n_rows = len(REPLICATES)
    footnote_wrapped = textwrap.fill(_common_footnote_text(level, data), 170)
    h_a, top_a, bottom_a, foot_a, extra_a = _footnote_layout("A", footnote_wrapped, FIG_A_SIZE[1])
    fig, axes = plt.subplots(n_rows, len(METHODS), figsize=(FIG_A_SIZE[0], h_a), sharex=True,
                             sharey=True)
    for i, rep in enumerate(REPLICATES):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            curves = varios[(rep, method)]
            _plot_curves(ax, curves, METHOD_COLORS[method], ALPHA, REALIZATION_LINEWIDTH)
            _draw_reference_and_truth(ax, truth_vario, h_smooth, model_smooth, total_sill, ylim)
            ax.text(0.03, 0.96, f"{len(curves)} realizations", transform=ax.transAxes,
                    fontsize=8.5, va="top", ha="left", color="dimgray")
            if i == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=13, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel("Lag h (m)", fontsize=LABEL_FONTSIZE)
            if j == 0:
                ax.set_ylabel(
                    f"{rep} (sample_seed {data[rep]['sample_seed']})\n"
                    "Semivariance (Porosity %$^2$)", fontsize=LABEL_FONTSIZE - 1,
                )
    fig.suptitle(
        "Variogram of the individual realizations vs. the truth, by sampling replicate "
        f"({LEVEL_TITLE[level]})", fontsize=15, y=top_a(0.9975),
    )
    fig.tight_layout(rect=(0.0, bottom_a(0.065), 1.0, top_a(FIG_A_PANELS_TOP)))
    _figure_legend(fig, N_REALIZATIONS_EXPECTED, ALPHA, REALIZATION_LINEWIDTH,
                   top_a(FIG_A_LEGEND_Y))
    fig.text(0.5, foot_a(0.004), footnote_wrapped, ha="center", va="bottom",
             fontsize=FOOTNOTE_FONTSIZE, color="dimgray")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / FIGURE_A_NAME_TEMPLATE.format(level=level)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def make_figure_b(level, data, varios, truth_vario, ylim):
    h_smooth = np.linspace(0.0, LAG_MAX_M, 300)
    model_smooth = spherical_semivariance(h_smooth)
    total_sill = POR_STDEV ** 2
    footnote = (
        "Pooled over all 10 sampling replicates (rep0-rep9) x 10 realizations per method = 100 "
        "curves per panel, drawn at alpha=" + f"{POOLED_ALPHA:g}. "
        + _common_footnote_text(level, data)
    )
    footnote_wrapped = textwrap.fill(footnote, 175)
    h_b, top_b, bottom_b, foot_b, extra_b = _footnote_layout("B", footnote_wrapped, FIG_B_SIZE[1])
    fig, axes = plt.subplots(1, len(METHODS), figsize=(FIG_B_SIZE[0], h_b), sharex=True,
                             sharey=True)
    for j, method in enumerate(METHODS):
        ax = axes[j]
        pooled = [v for rep in REPLICATES for v in varios[(rep, method)]]
        _plot_curves(ax, pooled, METHOD_COLORS[method], POOLED_ALPHA, POOLED_LINEWIDTH)
        _draw_reference_and_truth(ax, truth_vario, h_smooth, model_smooth, total_sill, ylim)
        ax.set_title(
            f"{METHOD_LABELS[method]}  ({len(pooled)} realizations = {len(REPLICATES)} "
            f"replicates x {N_REALIZATIONS_EXPECTED})", fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Lag h (m)", fontsize=LABEL_FONTSIZE)
        if j == 0:
            ax.set_ylabel("Semivariance (Porosity %$^2$)", fontsize=LABEL_FONTSIZE)
    fig.suptitle(
        "Variogram of the individual realizations vs. the truth, pooled over the 10 sampling "
        f"replicates ({LEVEL_TITLE[level]})", fontsize=14, y=top_b(0.995),
    )
    fig.tight_layout(rect=(0.0, bottom_b(0.25), 1.0, top_b(FIG_B_PANELS_TOP)))
    _figure_legend(fig, len(REPLICATES) * N_REALIZATIONS_EXPECTED, POOLED_ALPHA,
                   POOLED_LINEWIDTH, top_b(FIG_B_LEGEND_Y))
    fig.text(0.5, foot_b(0.005), footnote_wrapped, ha="center", va="bottom",
             fontsize=FOOTNOTE_FONTSIZE, color="dimgray")
    out = FIGURES_DIR / FIGURE_B_NAME_TEMPLATE.format(level=level)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def _summary_rows(level, rep_label, method, vals, model_at_report, n_real):
    stats = {"min": float(np.min(vals)), "median": float(np.median(vals)),
             "max": float(np.max(vals))}
    return [{
        "axis_level": level, "replicate": rep_label, "method": method,
        "n_realizations": n_real, "lag_m": REPORT_LAG_M, "stat": stat,
        "value_semivariance": stats[stat],
        "value_pct_of_model": 100.0 * stats[stat] / model_at_report,
    } for stat in REPORT_STATS]


def run_level(level, source_runs, truth, grid_coords, truth_vario):
    """Everything for ONE axis level: load + validate + draw, variograms, per-level
    y-limits, both figures, summary rows and the level's JSON block."""
    t_level = time.time()
    model_at_report = float(spherical_semivariance(REPORT_LAG_M))
    truth_report = semivariance_at(truth_vario, REPORT_LAG_M)
    print(
        f"\n======== axis_level={level} ({LEVEL_PERCENT[level]}, n_samples_requested="
        f"{N_SAMPLES_REQUESTED_BY_LEVEL[level]}), replicates={REPLICATES[0]}..{REPLICATES[-1]} "
        f"(sample_seed {REPLICATE_SEED[REPLICATES[0]]}..{REPLICATE_SEED[REPLICATES[-1]]}), "
        f"GP_DRAW_SEED_BASE={GP_DRAW_SEED_BASE_BY_LEVEL[level]} ========"
    )
    data, varios = {}, {}
    for k, rep in enumerate(REPLICATES):
        print(f"\nL{level} {rep}:")
        data[rep] = load_replicate(level, rep, source_runs, truth, grid_coords,
                                   check_determinism=(k == 0))
        for method in METHODS:
            varios[(rep, method)] = realization_variograms(
                data[rep]["maps"][method], truth_vario, f"L{level} {rep} {method}"
            )
        del data[rep]["maps"]  # free memory; only variograms are needed from here on

    # Fixed y-limits (YLIM) for every panel of every figure; max plotted value is only
    # recorded for reference.
    h_smooth = np.linspace(0.0, LAG_MAX_M, 300)
    all_max = max(
        float(np.max(truth_vario["semivariance_empirical"].values)),
        float(np.max(spherical_semivariance(h_smooth))),
        POR_STDEV ** 2,
        max(float(np.max(v["semivariance_empirical"].values)) for vs in varios.values() for v in vs),
    )
    ylim = YLIM
    print(f"\nL{level} fixed y-limits: {ylim} (level's max plotted value {all_max:.4f}, "
          "for reference)")
    # Share of plotted curve points (bins x realizations, well-populated bins only, as
    # drawn) above the y-limit, and number of realizations with >= 1 bin above it.
    exceed = {}
    for method in METHODS:
        n_pts = n_above = n_real = n_real_above = 0
        for rep_id in REPLICATES:
            for v in varios[(rep_id, method)]:
                y = v.loc[v["n_pairs"] >= MIN_PAIRS_PER_BIN, "semivariance_empirical"].values
                n_pts += y.size
                n_above += int(np.sum(y > ylim[1]))
                n_real += 1
                n_real_above += int(np.any(y > ylim[1]))
        exceed[method] = {
            "n_curve_points": n_pts, "n_points_above_ylim": n_above,
            "pct_points_above_ylim": 100.0 * n_above / n_pts,
            "n_realizations": n_real, "n_realizations_with_any_bin_above_ylim": n_real_above,
        }
        print(f"  L{level} {method:<14}: {exceed[method]['pct_points_above_ylim']:.2f}% of "
              f"{n_pts} curve points above {ylim[1]:g}; {n_real_above}/{n_real} realizations "
              "have >=1 bin above")

    out_a = make_figure_a(level, data, varios, truth_vario, ylim)
    print(f"figure A: {out_a} ({out_a.stat().st_size} bytes)")
    out_b = make_figure_b(level, data, varios, truth_vario, ylim)
    print(f"figure B: {out_b} ({out_b.stat().st_size} bytes)")

    # --- summary numbers -----------------------------------------------------
    rows = [{
        "axis_level": level, "replicate": "all", "method": "truth",
        "n_realizations": np.nan, "lag_m": REPORT_LAG_M, "stat": "value",
        "value_semivariance": truth_report,
        "value_pct_of_model": 100.0 * truth_report / model_at_report,
    }]
    print(f"\nL{level}: semivariance at the {REPORT_LAG_M:g} m bin as % of the truth's theoretical "
          f"model ({model_at_report:.3f}); truth experimental = {truth_report:.3f} "
          f"({100 * truth_report / model_at_report:.1f}%)")
    print(f"{'rep':>7} {'method':<14} {'n_real':>6} {'min%':>7} {'median%':>8} {'max%':>7}")
    for method in METHODS:
        pooled_vals = []
        for rep in REPLICATES:
            vals = np.array([semivariance_at(v, REPORT_LAG_M) for v in varios[(rep, method)]])
            pooled_vals.append(vals)
            rows += _summary_rows(level, rep, method, vals, model_at_report, len(vals))
            print(f"{rep:>7} {method:<14} {len(vals):>6d} "
                  f"{100 * vals.min() / model_at_report:7.1f} "
                  f"{100 * np.median(vals) / model_at_report:8.1f} "
                  f"{100 * vals.max() / model_at_report:7.1f}")
        allv = np.concatenate(pooled_vals)
        rows += _summary_rows(level, "pooled", method, allv, model_at_report, len(allv))
        print(f"{'pooled':>7} {method:<14} {len(allv):>6d} "
              f"{100 * allv.min() / model_at_report:7.1f} "
              f"{100 * np.median(allv) / model_at_report:8.1f} "
              f"{100 * allv.max() / model_at_report:7.1f}   (truth "
              f"{100 * truth_report / model_at_report:.1f}%)")

    val = {r: data[r]["gp_validation"] for r in REPLICATES}
    n_eig = [val[r]["n_eig_within_1e-6_rel_of_min"] for r in REPLICATES]
    corr = {r: val[r]["corr_seed55_draw_vs_stored"] for r in REPLICATES}
    level_seconds = time.time() - t_level
    block = {
        "n_samples_requested": N_SAMPLES_REQUESTED_BY_LEVEL[level],
        "n_samples_actual_by_replicate": {r: data[r]["n_actual"] for r in REPLICATES},
        "sample_seed_by_replicate": {r: int(REPLICATE_SEED[r]) for r in REPLICATES},
        "gp_draw_seed_base": GP_DRAW_SEED_BASE_BY_LEVEL[level],
        "gp_draw_seed_by_replicate": {r: data[r]["gp_draw_seed"] for r in REPLICATES},
        "sgs_out_of_range_values": {r: data[r]["sgs_out_of_range"] for r in REPLICATES
                                    if data[r]["sgs_out_of_range"]},
        "gp_draw_method_by_replicate": {r: data[r]["gp_draw_method"] for r in REPLICATES},
        "gp_draw_seconds_by_replicate": {r: round(data[r]["gp_draw_seconds"], 2) for r in REPLICATES},
        "level_runtime_seconds": round(level_seconds, 1),
        "source_runs": {r: data[r]["runs"] for r in REPLICATES},
        "gp_reconstruction_validation": val,
        "n_eig_within_1e-6_rel_of_min_range": [int(min(n_eig)), int(max(n_eig))],
        "replicates_seed55_draw_close_to_stored_corr_gt_0.99": [r for r in REPLICATES
                                                                if corr[r] > 0.99],
        "corr_seed55_draw_vs_stored_range": [float(min(corr.values())), float(max(corr.values()))],
        "ylim": list(ylim),
        "max_plotted_value": float(all_max),
        "exceed_ylim_by_method": exceed,
        "figure_a": out_a.relative_to(_REPO_ROOT).as_posix(),
        "figure_b": out_b.relative_to(_REPO_ROOT).as_posix(),
    }
    return rows, block


def _merge_csv(new_rows, levels_run):
    """Replace the run levels' rows of the existing summary CSV, keep the other
    levels' rows untouched, order by LEVELS."""
    path = PROCESSED_DIR / SUMMARY_CSV_NAME
    new = pd.DataFrame(new_rows)
    if path.exists():
        # float_precision="round_trip": the default C parser can change the last digit of
        # a float, which would alter the other levels' rows byte-wise on rewrite.
        old = pd.read_csv(path, dtype={"axis_level": str}, float_precision="round_trip")
        old = old[~old["axis_level"].isin(levels_run)]
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined["_o"] = combined["axis_level"].map({lv: i for i, lv in enumerate(LEVELS)})
    combined = combined.sort_values("_o", kind="stable").drop(columns="_o")
    combined.to_csv(path, index=False)
    return path, len(combined)


def main(levels=LEVELS):
    t_start = time.time()
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)
    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=base.AXIS_HMIN1)
    grid_coords = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)
    print(
        f"levels={list(levels)}, truth_seed={TRUTH_SEED}, "
        f"VARIOGRAM_PAIR_SEED={VARIOGRAM_PAIR_SEED}, GP_SAMPLE_SEED={GP_SAMPLE_SEED}, "
        f"GP_DRAW_SEED_BASE_BY_LEVEL={GP_DRAW_SEED_BASE_BY_LEVEL}"
    )
    truth_vario = quiet_variogram(truth)

    all_rows, blocks = [], {}
    for level in levels:
        rows, block = run_level(level, source_runs, truth, grid_coords, truth_vario)
        all_rows += rows
        blocks[level] = block

    summary_path, n_rows = _merge_csv(all_rows, set(levels))
    print(f"\nsummary: {summary_path} ({n_rows} rows total)")

    total_seconds = time.time() - t_start
    json_path = PROCESSED_DIR / SEEDS_JSON_NAME
    existing_levels = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            existing_levels = json.load(f).get("levels", {})
    existing_levels.update(blocks)
    seeds = {
        "note": (
            "Seeds used by make_variogram_reproduction_figure.py. GP draws are re-generated on "
            "the fly (not stored under results/raw): seed of replicate rep k at level L = "
            "gp_draw_seed_base_by_level[L] + k; one sample_y(n_samples=10) call per (level, "
            "replicate). GP_SAMPLE_SEED is used only for the reconstruction validation draw (the "
            "stored draw's own seed). The seed fixes a draw only for a bit-identical posterior "
            "covariance: the covariance has a near-degenerate white-noise eigenvalue cluster whose "
            "size varies by replicate (see n_eig_within_1e-6_rel_of_min in "
            "gp_reconstruction_validation and its range per level), so the seed-55 n_samples=1 "
            "draw reproduces the stored draw closely only for some replicates (see "
            "replicates_seed55_draw_close_to_stored_corr_gt_0.99 and "
            "corr_seed55_draw_vs_stored_range per level). The y-axis is FIXED for all figures (see "
            "ylim / ylim_rule); per-level max_plotted_value and exceed_ylim_by_method are kept for "
            "reference."
        ),
        "levels_run_last": list(levels),
        "ylim": list(YLIM),
        "ylim_rule": (
            f"y-axis fixed at 0 to Y_MAX_SILL_FACTOR ({Y_MAX_SILL_FACTOR:g}) x total sill "
            f"(POR_STDEV**2 = {POR_STDEV ** 2:g}) = {Y_MAX:g} for every panel of all "
            "variogram-reproduction figures; curves above it are cut off by the axis (data not clipped)."
        ),
        "truth_seed": TRUTH_SEED,
        "variogram_pair_seed": VARIOGRAM_PAIR_SEED,
        "n_pairs_drawn": N_PAIRS_DRAWN,
        "gp_sample_seed_validation": GP_SAMPLE_SEED,
        "gp_draw_seed_base_by_level": GP_DRAW_SEED_BASE_BY_LEVEL,
        "n_gp_draws_per_replicate": N_GP_DRAWS,
        "gp_reconstruction_tolerances": {
            "mean_var_rtol": GP_MAP_RTOL, "mean_var_atol": GP_MAP_ATOL,
            "chi2_max_abs_z": GP_CHI2_MAX_ABS_Z,
        },
        "alpha": ALPHA,
        "pooled_alpha": POOLED_ALPHA,
        "total_runtime_seconds_last_run": round(total_seconds, 1),
        "levels": {lv: existing_levels[lv] for lv in LEVELS if lv in existing_levels},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2)
    print(f"seeds: {json_path}")
    for lv in levels:
        print(f"L{lv} GP draw seconds by replicate: {blocks[lv]['gp_draw_seconds_by_replicate']}; "
              f"level runtime {blocks[lv]['level_runtime_seconds']} s")
    print(f"total runtime {total_seconds:.1f} s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS),
                        help="axis levels to (re)generate; default: all five")
    main(tuple(parser.parse_args().levels))
