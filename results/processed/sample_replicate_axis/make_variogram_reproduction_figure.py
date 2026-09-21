"""Variogram reproduction of the REALIZATIONS of SGS, RBF+bootstrap and GP-MLE
at the 1% sample-density level, organised by random-sampling REPLICATE
(sample-replicate axis; ONE fixed ground truth, only the sample locations change
across rep0..rep9).

WHAT THE FIGURES SHOW
---------------------
Figure A  results/figures/sample_replicate_axis/
          variogram_reproduction_by_replicate_level1.png
    10 rows (rep0..rep9, labelled 'rep k (sample_seed s)') x 3 columns
    (SGS | RBF+bootstrap | GP-MLE). Each panel = that replicate's 10
    realizations of that method plus the truth.
Figure B  results/figures/sample_replicate_axis/
          variogram_reproduction_pooled_level1.png
    ONE row, 3 panels (SGS | RBF+bootstrap | GP-MLE); each panel pools all
    10 replicates x 10 realizations = 100 curves (lower alpha, POOLED_ALPHA).
    Same axes (x and y limits) as Figure A.

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
x = 0-750 m (25 m bins); y-limits SHARED by every panel of both figures
(computed from everything plotted, then padded; nothing is clipped).
Simple kriging is deliberately not shown (a smooth estimate, not a set of
realizations).

WHAT "REALIZATION" MEANS PER METHOD
-----------------------------------
  SGS            the 10 conditional simulations in sgs_realizations.npy
                 (shape (10, 50, 50)) of the run referenced by
                 results/processed/sample_replicate_axis/source_runs.json
                 (level '1', that replicate).
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

GP RECONSTRUCTION VALIDATION (for ALL 10 replicates; raises on failure)
-----------------------------------------------------------------------
  (i)   reconstructed posterior mean / variance maps vs. the run's stored
        posterior_mean_map.npy / posterior_var_map.npy
        (np.allclose, rtol=GP_MAP_RTOL, atol=GP_MAP_ATOL);
  (ii)  n_samples=1 with GP_SAMPLE_SEED (=55, the run's own seed) through the
        same code path vs. the stored posterior_sample_map.npy: the max abs
        difference and correlation are PRINTED AND RECORDED. Exact equality is
        NOT required. Observed: the seed-55 draw reproduces the stored draw
        closely ONLY for rep0 and rep5 (corr ~1.0, max|diff| ~0.03); for the
        other 8 replicates it does not (corr 0.16-0.91, max|diff| 4.7-19.4).
        Cause, established empirically: the posterior covariance contains the
        white-noise floor, i.e. a numerically (near-)degenerate eigenvalue
        cluster whose size varies by replicate (n_eig_within_1e-6_rel_of_min in
        the JSON: 948 for rep0 up to 2415 for rep8, out of 2500), so the singular vectors numpy's SVD (inside sample_y) returns for that
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
the smallest; observed 948-2415 of 2500 depending on the replicate) is recorded
in the JSON.

SEEDS (all recorded; every one is deterministic; no other randomness)
---------------------------------------------------------------------
  TRUTH_SEED            = 101   (truth field)
  replicate rep k       = sample_seed 1001+k (conditioning samples)
  VARIOGRAM_PAIR_SEED   = 90    (200,000 random (i,j) pairs, shared estimator;
                                  every curve, truth included, is evaluated on
                                  the IDENTICAL pair set and 25 m bins)
  GP_SAMPLE_SEED        = 55    (validation draw only, same as the stored runs)
  GP_DRAW_SEED_BASE     = 9000  (NEW; the GP draws of rep k use seed
                                  GP_DRAW_SEED_BASE + k, i.e. 9000..9009; one
                                  sample_y(n_samples=10) call per replicate)
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
  results/figures/sample_replicate_axis/variogram_reproduction_by_replicate_level1.png
  results/figures/sample_replicate_axis/variogram_reproduction_pooled_level1.png
  results/processed/sample_replicate_axis/variogram_reproduction_summary.csv
      tidy long: axis_level, replicate, method, n_realizations, lag_m, stat,
      value_semivariance, value_pct_of_model; stat in {min, median, max};
      replicate = 'rep0'..'rep9' and 'pooled' (all 100 realizations of a
      method); the truth is one row with method='truth', replicate='all',
      stat='value', n_realizations empty.
  results/processed/sample_replicate_axis/variogram_reproduction_seeds.json

Run with:
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_variogram_reproduction_figure
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
    NX, NY, XMN, YMN, XSIZ, YSIZ, CC1, HMAJ1, NUG, POR_STDEV,
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
FIGURE_A_NAME = "variogram_reproduction_by_replicate_level1.png"
FIGURE_B_NAME = "variogram_reproduction_pooled_level1.png"
SUMMARY_CSV_NAME = "variogram_reproduction_summary.csv"
SEEDS_JSON_NAME = "variogram_reproduction_seeds.json"

# --- design constants (all named; change here only) -------------------------
AXIS_LEVEL = "1"                         # 1% level, n_samples_requested = 25
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
Y_PAD_FRACTION = 0.08        # headroom above the largest plotted value

N_REALIZATIONS_EXPECTED = 10
N_GP_DRAWS = N_REALIZATIONS_EXPECTED
GP_DRAW_SEED_BASE = 9000     # NEW seed family; GP draw seed of rep k = base + k

REPORT_LAG_M = 287.5         # lag bin reported in the summary (a 25 m bin centre)
REPORT_STATS = ("min", "median", "max")

# GP reconstruction validation (see docstring). Mean / variance are rebuilt from
# full-precision JSON hyperparameters, so they agree to ~1e-14; the tolerances are
# a stated safety margin, not a fit.
GP_MAP_RTOL = 1e-6
GP_MAP_ATOL = 1e-6
GP_CHI2_MAX_ABS_Z = 4.0      # |q-N| / sqrt(2N) bound for the stored draw under the rebuilt N(m, C)
EIG_CLUSTER_REL_TOL = 1e-6   # diagnostic only

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
def load_replicate(replicate_id: str, source_runs: dict, truth: np.ndarray, grid_coords,
                   check_determinism: bool):
    runs = {m: _REPO_ROOT / source_runs[AXIS_LEVEL][replicate_id][m] for m in METHODS}
    sample_seed = int(REPLICATE_SEED[replicate_id])
    n_requested = N_SAMPLES_REQUESTED_BY_LEVEL[AXIS_LEVEL]

    # Same conditioning samples in all three runs and equal to the regenerated ones.
    ref = get_conditioning_samples(truth, sample_seed=sample_seed, n_samples=n_requested)
    for m, rd in runs.items():
        rec = pd.read_csv(rd / "samples.csv")
        if len(rec) != len(ref) or not np.allclose(
            rec[["X", "Y", VCOL]].values, ref[["X", "Y", VCOL]].values, rtol=0.0,
            atol=SAMPLES_MATCH_ATOL,
        ):
            raise ValueError(
                f"{replicate_id} {m}: samples.csv does not match sample_seed={sample_seed}."
            )

    sgs = np.load(runs["sgs"] / "sgs_realizations.npy")
    rbf = np.load(runs["rbf_bootstrap"] / "bootstrap_replicate_maps.npy")
    for name, arr in (("sgs_realizations", sgs), ("bootstrap_replicate_maps", rbf)):
        if arr.shape != (N_REALIZATIONS_EXPECTED, NY, NX):
            raise ValueError(f"{replicate_id}: {name} has shape {arr.shape}, expected "
                             f"({N_REALIZATIONS_EXPECTED}, {NY}, {NX}) (full grid).")

    gpr, _ = rebuild_fitted_gpr(runs["gp_mle"])
    validation = validate_gp_reconstruction(gpr, runs["gp_mle"], grid_coords, replicate_id,
                                            check_determinism)
    seed = GP_DRAW_SEED_BASE + REPLICATES.index(replicate_id)
    t0 = time.time()
    gp_draws, method = draw_gp_posterior(gpr, grid_coords, N_GP_DRAWS, seed)
    seconds = time.time() - t0
    print(f"  [GP {replicate_id}] drew {N_GP_DRAWS} posterior draws, seed={seed}, "
          f"method={method}, {seconds:.1f} s")

    return {
        "sample_seed": sample_seed,
        "n_requested": n_requested,
        "n_actual": len(ref),
        "maps": {"sgs": sgs, "rbf_bootstrap": rbf, "gp_mle": gp_draws},
        "runs": {m: str(source_runs[AXIS_LEVEL][replicate_id][m]) for m in METHODS},
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


def _common_footnote_text(data):
    total_sill = POR_STDEV ** 2
    n_actual = ", ".join(f"{r}: {data[r]['n_actual']}" for r in REPLICATES)
    return (
        f"1% sample-density level (n_samples_requested={N_SAMPLES_REQUESTED_BY_LEVEL[AXIS_LEVEL]}; "
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
        f"draw seeds {GP_DRAW_SEED_BASE}+k for rep k. Legend swatches are drawn more opaque than "
        "the plotted lines for legibility. Simple kriging is not shown (a smooth estimate, not a "
        "set of realizations). Y-limits are shared by all panels of both level-1 variogram "
        "figures."
    )


def make_figure_a(data, varios, truth_vario, ylim):
    h_smooth = np.linspace(0.0, LAG_MAX_M, 300)
    model_smooth = spherical_semivariance(h_smooth)
    total_sill = POR_STDEV ** 2
    n_rows = len(REPLICATES)
    fig, axes = plt.subplots(n_rows, len(METHODS), figsize=FIG_A_SIZE, sharex=True, sharey=True)
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
    footnote = _common_footnote_text(data)
    fig.suptitle(
        "Variogram of the individual realizations vs. the truth, by sampling replicate "
        "(1% level, n=25)", fontsize=15, y=0.9975,
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, FIG_A_PANELS_TOP))
    _figure_legend(fig, N_REALIZATIONS_EXPECTED, ALPHA, REALIZATION_LINEWIDTH, FIG_A_LEGEND_Y)
    fig.text(0.5, 0.004, textwrap.fill(footnote, 170), ha="center", va="bottom",
             fontsize=8, color="dimgray")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / FIGURE_A_NAME
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def make_figure_b(data, varios, truth_vario, ylim):
    h_smooth = np.linspace(0.0, LAG_MAX_M, 300)
    model_smooth = spherical_semivariance(h_smooth)
    total_sill = POR_STDEV ** 2
    fig, axes = plt.subplots(1, len(METHODS), figsize=FIG_B_SIZE, sharex=True, sharey=True)
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
    footnote = (
        "Pooled over all 10 sampling replicates (rep0-rep9) x 10 realizations per method = 100 "
        "curves per panel, drawn at alpha=" + f"{POOLED_ALPHA:g}. " + _common_footnote_text(data)
    )
    fig.suptitle(
        "Variogram of the individual realizations vs. the truth, pooled over the 10 sampling "
        "replicates (1% level, n=25)", fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.25, 1.0, FIG_B_PANELS_TOP))
    _figure_legend(fig, len(REPLICATES) * N_REALIZATIONS_EXPECTED, POOLED_ALPHA,
                   POOLED_LINEWIDTH, FIG_B_LEGEND_Y)
    fig.text(0.5, 0.005, textwrap.fill(footnote, 175), ha="center", va="bottom",
             fontsize=8, color="dimgray")
    out = FIGURES_DIR / FIGURE_B_NAME
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def _summary_rows(rep_label, method, vals, model_at_report, n_real):
    stats = {"min": float(np.min(vals)), "median": float(np.median(vals)),
             "max": float(np.max(vals))}
    return [{
        "axis_level": AXIS_LEVEL, "replicate": rep_label, "method": method,
        "n_realizations": n_real, "lag_m": REPORT_LAG_M, "stat": stat,
        "value_semivariance": stats[stat],
        "value_pct_of_model": 100.0 * stats[stat] / model_at_report,
    } for stat in REPORT_STATS]


def main():
    t_start = time.time()
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)
    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=base.AXIS_HMIN1)
    grid_coords = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)

    print(
        f"axis_level={AXIS_LEVEL}, replicates={REPLICATES[0]}..{REPLICATES[-1]} "
        f"(sample_seed {REPLICATE_SEED[REPLICATES[0]]}..{REPLICATE_SEED[REPLICATES[-1]]}), "
        f"truth_seed={TRUTH_SEED}, VARIOGRAM_PAIR_SEED={VARIOGRAM_PAIR_SEED}, "
        f"GP_SAMPLE_SEED={GP_SAMPLE_SEED}, GP_DRAW_SEED_BASE={GP_DRAW_SEED_BASE}"
    )
    truth_vario = quiet_variogram(truth)
    model_at_report = float(spherical_semivariance(REPORT_LAG_M))
    truth_report = semivariance_at(truth_vario, REPORT_LAG_M)

    data, varios = {}, {}
    for k, rep in enumerate(REPLICATES):
        print(f"\n{rep}:")
        data[rep] = load_replicate(rep, source_runs, truth, grid_coords,
                                   check_determinism=(k == 0))
        for method in METHODS:
            varios[(rep, method)] = realization_variograms(
                data[rep]["maps"][method], truth_vario, f"{rep} {method}"
            )
        del data[rep]["maps"]  # free memory; only variograms are needed from here on

    # y-limits shared by ALL panels of both figures.
    h_smooth = np.linspace(0.0, LAG_MAX_M, 300)
    all_max = max(
        float(np.max(truth_vario["semivariance_empirical"].values)),
        float(np.max(spherical_semivariance(h_smooth))),
        POR_STDEV ** 2,
        max(float(np.max(v["semivariance_empirical"].values)) for vs in varios.values() for v in vs),
    )
    all_min = min(
        0.0,
        min(float(np.min(v["semivariance_empirical"].values)) for vs in varios.values() for v in vs),
    )
    ylim = (all_min, all_max * (1.0 + Y_PAD_FRACTION))
    print(f"\nshared y-limits: {ylim} (max plotted value {all_max:.4f}, min {all_min:.4f})")

    out_a = make_figure_a(data, varios, truth_vario, ylim)
    print(f"figure A: {out_a} ({out_a.stat().st_size} bytes)")
    out_b = make_figure_b(data, varios, truth_vario, ylim)
    print(f"figure B: {out_b} ({out_b.stat().st_size} bytes)")

    # --- summary numbers -----------------------------------------------------
    rows = [{
        "axis_level": AXIS_LEVEL, "replicate": "all", "method": "truth",
        "n_realizations": np.nan, "lag_m": REPORT_LAG_M, "stat": "value",
        "value_semivariance": truth_report,
        "value_pct_of_model": 100.0 * truth_report / model_at_report,
    }]
    print(f"\nSemivariance at the {REPORT_LAG_M:g} m bin as % of the truth's theoretical model "
          f"({model_at_report:.3f}); truth experimental = {truth_report:.3f} "
          f"({100 * truth_report / model_at_report:.1f}%)")
    print(f"{'rep':>7} {'method':<14} {'n_real':>6} {'min%':>7} {'median%':>8} {'max%':>7}")
    for method in METHODS:
        pooled_vals = []
        for rep in REPLICATES:
            vals = np.array([semivariance_at(v, REPORT_LAG_M) for v in varios[(rep, method)]])
            pooled_vals.append(vals)
            rows += _summary_rows(rep, method, vals, model_at_report, len(vals))
            print(f"{rep:>7} {method:<14} {len(vals):>6d} "
                  f"{100 * vals.min() / model_at_report:7.1f} "
                  f"{100 * np.median(vals) / model_at_report:8.1f} "
                  f"{100 * vals.max() / model_at_report:7.1f}")
        allv = np.concatenate(pooled_vals)
        rows += _summary_rows("pooled", method, allv, model_at_report, len(allv))
        print(f"{'pooled':>7} {method:<14} {len(allv):>6d} "
              f"{100 * allv.min() / model_at_report:7.1f} "
              f"{100 * np.median(allv) / model_at_report:8.1f} "
              f"{100 * allv.max() / model_at_report:7.1f}   (truth "
              f"{100 * truth_report / model_at_report:.1f}%)")
    summary = pd.DataFrame(rows)
    summary_path = PROCESSED_DIR / SUMMARY_CSV_NAME
    summary.to_csv(summary_path, index=False)
    print(f"summary: {summary_path} ({len(summary)} rows)")

    total_seconds = time.time() - t_start
    seeds = {
        "note": (
            "Seeds used by make_variogram_reproduction_figure.py. GP draws are re-generated on "
            "the fly (not stored under results/raw): seed of replicate rep k = GP_DRAW_SEED_BASE "
            "+ k; one sample_y(n_samples=10) call per replicate. GP_SAMPLE_SEED is used only for "
            "the reconstruction validation draw (the stored draw's own seed). The seed fixes a "
            "draw only for a bit-identical posterior covariance (see the script docstring: the "
            "covariance has a near-degenerate white-noise eigenvalue cluster whose size varies "
            "by replicate, 948-2415 of 2500 eigenvalues within 1e-6 relative of the minimum, "
            "see n_eig_within_1e-6_rel_of_min in gp_reconstruction_validation). The seed-55 "
            "n_samples=1 draw reproduces the stored draw closely only for rep0 and rep5 "
            "(corr ~1.0, max|diff| ~0.03) and not for the other 8 replicates (corr 0.16-0.91)."
        ),
        "axis_level": AXIS_LEVEL,
        "truth_seed": TRUTH_SEED,
        "sample_seed_by_replicate": {r: int(REPLICATE_SEED[r]) for r in REPLICATES},
        "variogram_pair_seed": VARIOGRAM_PAIR_SEED,
        "n_pairs_drawn": N_PAIRS_DRAWN,
        "gp_sample_seed_validation": GP_SAMPLE_SEED,
        "gp_draw_seed_base": GP_DRAW_SEED_BASE,
        "gp_draw_seed_by_replicate": {r: data[r]["gp_draw_seed"] for r in REPLICATES},
        "gp_draw_method_by_replicate": {r: data[r]["gp_draw_method"] for r in REPLICATES},
        "n_gp_draws_per_replicate": N_GP_DRAWS,
        "gp_draw_seconds_by_replicate": {r: round(data[r]["gp_draw_seconds"], 2) for r in REPLICATES},
        "total_runtime_seconds": round(total_seconds, 1),
        "source_runs": {r: data[r]["runs"] for r in REPLICATES},
        "gp_reconstruction_validation": {r: data[r]["gp_validation"] for r in REPLICATES},
        "gp_reconstruction_tolerances": {
            "mean_var_rtol": GP_MAP_RTOL, "mean_var_atol": GP_MAP_ATOL,
            "chi2_max_abs_z": GP_CHI2_MAX_ABS_Z,
        },
        "alpha": ALPHA,
        "pooled_alpha": POOLED_ALPHA,
        "shared_ylim": list(ylim),
    }
    with open(PROCESSED_DIR / SEEDS_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2)
    print(f"seeds: {PROCESSED_DIR / SEEDS_JSON_NAME}")
    print(f"GP draw seconds by replicate: "
          f"{ {r: round(data[r]['gp_draw_seconds'], 1) for r in REPLICATES} }; "
          f"total runtime {total_seconds:.1f} s")
    return out_a, out_b


if __name__ == "__main__":
    main()
