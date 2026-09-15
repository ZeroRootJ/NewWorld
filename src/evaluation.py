"""MSE + Uncertainty Model Goodness (UMG) evaluation library for the
base-case (and later, per-axis) method comparison (docs/experiment_context.md
section 5).

The UMG metric and the percentile-based "accuracy plot" fraction-in-interval
calculation are ported *exactly* from the reference codebase reviewed for
this task:

- ``calc_umg`` <- ``calc_UMG.py`` (reference repo, previously reviewed and
  confirmed bug-free)
- ``accuracy_plot_fraction_in`` <- ``dashboard.py``'s ``get_metrics`` /
  ``plot_dashboard`` accuracy-plot block (same reference repo -- only this
  specific calculation is ported; other parts of ``dashboard.py`` /
  ``Kriging_EXH.py`` / ``SGSIM-EXH.py`` are known from a prior session to
  contain unrelated nscore bugs and must NOT be used as a pattern elsewhere).

Design decisions confirmed by the orchestrator/user (not to be re-litigated
here, see task handoff 2026-09-14):
- RBF+bootstrap: MSE from ``point_estimate_map`` (single fit, not the
  bootstrap mean); UMG from ``(bootstrap_mean_map, bootstrap_var_map)``
  treated as a per-cell Gaussian predictive distribution.
- GP-MLE: ``posterior_mean_map`` / ``posterior_var_map`` used for both MSE
  and UMG.
- SGS: ``sgs_mean_map`` / ``sgs_var_map`` (ensemble statistics) used for both
  MSE and UMG.
- Kriging: MSE from ``kmap_physical``; UMG from ``kmap_ns`` + ``vmap_ns`` via
  exact quantile back-transform (``kriging_fraction_in`` below), NOT a
  variance back-transform approximation -- coverage is invariant under the
  monotonic normal-score back-transform, so this is the correct approach.
- All four methods apply the identical conditioning-sample-cell exclusion
  mask (``conditioning_cell_mask`` below) before computing any metric.

Added 2026-09-14 (range-axis task, user instruction): interval width and
CRPS, on top of -- never replacing -- MSE/UMG. Motivation (user's question,
recorded here so the metric choice is auditable): "GP-MLE's UMG is very
high; is it buying coverage by predicting loosely and just widening its
intervals?"

IMPORTANT -- do not overstate what UMG misses (an earlier version of this
docstring did, and the claim was wrong). UMG does NOT reward interval
inflation: ``calc_umg`` below is maximized at correct calibration and
penalizes over-wide predictions too. Measured directly on a synthetic
N(0,1) truth by scaling the predictive sigma (reviewer verification,
2026-09-14):

    sigma scale   0.5     1.0     1.6     2.0     3.0
    UMG           0.626   0.997   0.861   0.804   0.719
    CRPS          0.605   0.560   0.600   0.653   0.829

Both metrics peak/bottom at sigma scale 1.0. What UMG actually has is (a)
an ASYMMETRY -- the ``(3*a_p - 2)`` term weights under-coverage 2x and
over-coverage 1x, so over-dispersion is the cheaper way to be wrong -- and
(b) it is an AGGREGATE coverage statistic, so two models with the same
marginal coverage but different sharpness score identically. Mean interval
width reports sharpness directly, and CRPS scores calibration and sharpness
jointly per cell. That is the gap the two new metrics fill. Both are also
explicitly required by docs/experiment_context.md section 5.

Cross-method fairness convention for the two new metrics (important -- the
four methods express uncertainty in three different forms: analytic
Gaussian in physical units (GP-MLE), analytic Gaussian in normal-score
space (kriging), and a finite ensemble (SGS, RBF+bootstrap)):

  Every new metric is computed from the SAME predictive-quantile
  representation each method's UMG already uses, so no method is helped or
  hurt by the shape of its native output:

  - SGS / RBF+bootstrap / GP-MLE -> per-cell Gaussian N(mean, std), i.e.
    exactly what ``accuracy_plot_fraction_in`` assumes for their UMG
    (``gaussian_interval_widths`` / ``gaussian_crps`` below).
  - Kriging -> per-cell Gaussian in normal-score space whose quantile
    endpoints are back-transformed to physical units cell by cell, i.e.
    exactly what ``kriging_fraction_in`` does for its UMG. The kriging
    VARIANCE is never back-transformed (``kriging_interval_widths`` /
    ``kriging_crps`` below).

  ENSEMBLE-RESOLUTION CAVEAT (must be carried into any report): SGS and
  RBF+bootstrap only have N=10 replicates, so their (mean, variance) pair
  -- and therefore every quantile derived from it -- is estimated from 10
  samples. A ddof=1 variance from n=10 has ~24% relative standard error, so
  their interval widths and CRPS carry correspondingly more sampling noise
  than kriging's/GP-MLE's analytic values, and their true predictive
  distributions are not necessarily Gaussian. Empirical (order-statistic)
  quantiles are deliberately NOT used for them, because 10 replicates
  cannot resolve the tau=0.005/0.995 quantiles the CRPS integral needs --
  the Gaussian-moment representation is both the finer estimator here and
  the one that keeps these metrics consistent with their own UMG.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

import geostatspy.geostats as geostats

# Bin count for the percentile-based accuracy plot / UMG calculation --
# matches the reference codebase's ``bins = 20`` (dashboard.py get_metrics).
DEFAULT_BINS = 20

# Guard against std == 0 (or numerically indistinguishable from zero)
# producing 0/0 in norm.cdf. Conditioning-sample cells should already be
# excluded by conditioning_cell_mask before this module's functions are
# called (kriging variance is exactly 0 there), so this is a defensive
# fallback for any *other* reason a zero/near-zero std might appear (e.g. an
# unexpected degenerate GP/bootstrap variance) -- not expected to fire under
# normal use.
STD_EPS = 1e-12

# --- Interval-width / CRPS configuration (added 2026-09-14) --------------
# Nominal central-probability levels for the interval-width metric. These
# are the SAME 20 levels UMG/the accuracy plot use (np.linspace(0, 1,
# DEFAULT_BINS)), minus the final p = 1.0 level:
#
#   p = 1.0 means the [q_0, q_1] interval, whose width is +inf for any
#   Gaussian predictive distribution (SGS / RBF+bootstrap / GP-MLE) and, for
#   kriging, the constant BACKTR_ZMAX - BACKTR_ZMIN = 24 porosity % -- i.e.
#   an artifact of the finite normal-score back-transform bound
#   (src/experiments/kriging.py's BACKTR_ZMIN/ZMAX comment says exactly this
#   about p=1), not a property of the kriging model. Averaging it in would
#   make the metric either infinite (3 methods) or a constant (1 method), so
#   p=1.0 is excluded from the width average. This exclusion is a judgment
#   call made here and reported to the user, NOT silently applied: it is
#   recorded in the metric name/notes wherever the value is saved.
INTERVAL_WIDTH_NOMINAL_LEVELS = np.linspace(0.0, 1.0, DEFAULT_BINS)[:-1]

# The headline single-level interval width (central 95% prediction
# interval). Reported separately from the average above.
INTERVAL_WIDTH_HEADLINE_P = 0.95

# Default number of quantile levels used for the pinball/CRPS numerical
# integration -- see ``crps_tau_grid``. Chosen after the grid-refinement +
# analytic-Gaussian validation described in that function's docstring.
CRPS_DEFAULT_N_TAU = 199


def mse(truth: np.ndarray, pred: np.ndarray) -> float:
    """Mean squared error between two flattened (and pre-masked) arrays."""
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean((truth - pred) ** 2))


def calc_umg(p_intervals: np.ndarray, fraction_in: np.ndarray) -> float:
    """Uncertainty Model Goodness metric, ported as-is from calc_UMG.py."""
    p_intervals = np.asarray(p_intervals)
    fraction_in = np.asarray(fraction_in)
    if len(p_intervals) != len(fraction_in):
        raise ValueError("p_intervals and fraction_in must have the same length.")
    a_p = (fraction_in > p_intervals).astype(int)
    G = 1 - np.sum((3 * a_p - 2) * (fraction_in - p_intervals)) * (1 / len(p_intervals))
    return float(G)


def accuracy_plot_fraction_in(
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    bins: int = DEFAULT_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile-based fraction-in-interval curve, ported as-is from
    dashboard.py's get_metrics accuracy-plot block.

    Parameters
    ----------
    truth, mean, std : flattened 1-D arrays, same order, with
        conditioning-sample cells already excluded (via
        conditioning_cell_mask) by the caller.
    bins : number of probability-interval steps, np.linspace(0, 1, bins).

    Returns
    -------
    (p_intervals, fraction_in), each shape (bins,).
    """
    truth = np.asarray(truth, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    # Defensive filtering of std <= 0 (see STD_EPS docstring above) -- not
    # expected to remove anything once conditioning cells are excluded, but
    # guards against norm.cdf(x, loc, 0) producing nan/inf silently.
    valid = std > STD_EPS
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(
            f"WARNING accuracy_plot_fraction_in: dropped {n_dropped} of "
            f"{len(std)} cells with std <= {STD_EPS} before computing the "
            "accuracy plot (unexpected if conditioning cells were already "
            "excluded upstream)."
        )
    truth, mean, std = truth[valid], mean[valid], std[valid]

    percentiles = norm.cdf(truth, mean, std)
    p_intervals = np.linspace(0.0, 1.0, bins)
    fraction_in = np.zeros(bins)
    for i, p in enumerate(p_intervals):
        test_result = (percentiles > 0.5 - 0.5 * p) & (percentiles < 0.5 + 0.5 * p)
        fraction_in[i] = test_result.sum() / len(truth)
    return p_intervals, fraction_in


def kriging_fraction_in(
    truth_physical: np.ndarray,
    kmap_ns: np.ndarray,
    std_ns: np.ndarray,
    vr: np.ndarray,
    vrg: np.ndarray,
    zmin: float,
    zmax: float,
    ltail: int,
    ltpar: float,
    utail: int,
    utpar: float,
    bins: int = DEFAULT_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kriging-specific accuracy-plot curve via exact quantile
    back-transform (not a variance back-transform approximation): for each
    nominal probability interval p, the normal-score-space [z_lo, z_hi]
    interval is back-transformed to physical units *per cell* via
    ``geostats.backtr_value``, then truth is tested against those physical
    bounds. Coverage is invariant under the monotonic normal-score
    back-transform, so this is exact, not approximate.

    Parameters
    ----------
    truth_physical, kmap_ns, std_ns : flattened 1-D arrays, same order, same
        length, with conditioning-sample cells already excluded by the
        caller (via conditioning_cell_mask).
    vr, vrg : normal-score transform table (as returned by
        ``geostats.nscore``), same table used to fit the kriging model.
    zmin, zmax, ltail, ltpar, utail, utpar : back-transform tail parameters,
        identical to those used for the point-estimate back-transform in
        src/experiments/kriging.py (BACKTR_ZMIN/ZMAX, LTAIL/LTPAR,
        UTAIL/UTPAR).
    bins : number of probability-interval steps, np.linspace(0, 1, bins).

    Returns
    -------
    (p_intervals, fraction_in), each shape (bins,).

    Note on p=0: z_lo = z_hi = norm.ppf(0.5) = 0 exactly, so ns_lo == ns_hi
    and phys_lo == phys_hi for every cell; the strict-inequality test
    ``(truth > phys_lo) & (truth < phys_hi)`` is then always False (a
    continuous truth value essentially never exactly equals phys_lo), giving
    fraction_in[0] == 0. This matches the reference accuracy-plot's behavior
    at p=0 (same strict-inequality construction, mean==bounds there too) and
    is expected, not a bug.
    """
    truth_physical = np.asarray(truth_physical, dtype=float)
    kmap_ns = np.asarray(kmap_ns, dtype=float)
    std_ns = np.asarray(std_ns, dtype=float)

    valid = std_ns > STD_EPS
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(
            f"WARNING kriging_fraction_in: dropped {n_dropped} of "
            f"{len(std_ns)} cells with std_ns <= {STD_EPS} before computing "
            "the accuracy plot (unexpected if conditioning cells were "
            "already excluded upstream)."
        )
    truth_physical = truth_physical[valid]
    kmap_ns = kmap_ns[valid]
    std_ns = std_ns[valid]
    n = len(truth_physical)

    p_intervals = np.linspace(0.0, 1.0, bins)
    fraction_in = np.zeros(bins)
    for i, p in enumerate(p_intervals):
        z_lo, z_hi = norm.ppf(0.5 - 0.5 * p), norm.ppf(0.5 + 0.5 * p)
        ns_lo = kmap_ns + z_lo * std_ns
        ns_hi = kmap_ns + z_hi * std_ns
        phys_lo = np.array(
            [
                geostats.backtr_value(v, vr, vrg, zmin, zmax, ltail, ltpar, utail, utpar)
                for v in ns_lo
            ]
        )
        phys_hi = np.array(
            [
                geostats.backtr_value(v, vr, vrg, zmin, zmax, ltail, ltpar, utail, utpar)
                for v in ns_hi
            ]
        )
        test_result = (truth_physical > phys_lo) & (truth_physical < phys_hi)
        fraction_in[i] = test_result.sum() / n
    return p_intervals, fraction_in


# ---------------------------------------------------------------------------
# Interval width + CRPS (added 2026-09-14 -- see module docstring for the
# cross-method fairness convention these all follow). None of the functions
# above were modified.
#
# UNITS: both metrics are in physical porosity units.
#   - interval width: porosity % (larger = wider/less sharp; not "better" or
#     "worse" on its own -- it must be read together with UMG/coverage).
#   - CRPS: porosity % (LOWER is better; a proper scoring rule, so it is
#     minimized only by the true predictive distribution -- it cannot be
#     gamed by inflating the interval the way coverage-only UMG can).
# ---------------------------------------------------------------------------


def _filter_positive_std(arrays, std, name):
    """Apply the same std > STD_EPS filter ``accuracy_plot_fraction_in`` /
    ``kriging_fraction_in`` apply, so the new metrics are computed on exactly
    the same cell set as UMG. Returns the filtered arrays + filtered std."""
    valid = std > STD_EPS
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(
            f"WARNING {name}: dropped {n_dropped} of {len(std)} cells with "
            f"std <= {STD_EPS} (unexpected if conditioning cells were already "
            "excluded upstream)."
        )
    return [np.asarray(a, dtype=float)[valid] for a in arrays], std[valid]


def _central_interval_taus(p_levels: np.ndarray):
    """Central-interval endpoints (tau_lo, tau_hi) for nominal coverages p."""
    p_levels = np.asarray(p_levels, dtype=float)
    if np.any(p_levels < 0.0) or np.any(p_levels >= 1.0):
        raise ValueError(
            "p_levels must satisfy 0 <= p < 1; p = 1 has infinite width for a "
            "Gaussian predictive distribution (see INTERVAL_WIDTH_NOMINAL_LEVELS)."
        )
    return 0.5 - 0.5 * p_levels, 0.5 + 0.5 * p_levels


def crps_tau_grid(n_tau: int = CRPS_DEFAULT_N_TAU) -> np.ndarray:
    """Quantile levels for the CRPS numerical integration.

    MIDPOINT rule on a uniform grid: tau_i = (i + 0.5) / n_tau, i = 0..n_tau-1
    (e.g. n_tau=199 -> tau in [0.00251, 0.99749]). This integrates the FULL
    (0, 1) interval -- unlike a fixed tau = 0.01..0.99 grid, which silently
    truncates the two tail slabs [0, 0.01] and [0.99, 1] and therefore biases
    CRPS low. The pinball integrand tends to 0 at both endpoints (for a
    Gaussian, tau * |q_tau| ~ tau * sqrt(2 ln(1/tau)) -> 0), so the midpoint
    rule converges without needing the endpoints themselves (where the
    Gaussian quantile is +/- inf and cannot be evaluated at all).

    Grid resolution was checked empirically rather than assumed -- see the
    range-axis evaluation script, which recomputes CRPS at several n_tau and
    (for the three Gaussian-predictive methods) also compares against the
    exact analytic Gaussian CRPS (``crps_gaussian_analytic``).
    """
    if n_tau < 2:
        raise ValueError("n_tau must be >= 2.")
    return (np.arange(n_tau, dtype=float) + 0.5) / n_tau


def _crps_from_quantiles(truth: np.ndarray, quantiles: np.ndarray, taus: np.ndarray) -> float:
    """Mean CRPS from a per-cell predictive-quantile matrix.

    CRPS(F, y) = 2 * integral_0^1 pinball_tau(y, q_tau) dtau, with
    pinball_tau(y, q) = (tau - 1{y < q}) * (y - q)   (the standard quantile
    decomposition of the CRPS). Computing it this way -- rather than from
    each method's native parametric/ensemble form -- is what makes the four
    methods directly comparable: every one of them is reduced to the same
    object (a set of predictive quantiles in physical porosity units) before
    being scored.

    Parameters
    ----------
    truth : (n_cells,) physical-unit truth values.
    quantiles : (n_cells, n_tau) physical-unit predictive quantiles.
    taus : (n_tau,) the UNIFORM midpoint grid from ``crps_tau_grid``
        (the 2 * mean(...) below is the midpoint quadrature rule and is only
        valid for a uniform grid).

    Returns
    -------
    float -- mean CRPS over cells, porosity % (lower is better).
    """
    truth = np.asarray(truth, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    taus = np.asarray(taus, dtype=float)
    if quantiles.shape != (truth.shape[0], taus.shape[0]):
        raise ValueError(
            f"quantiles must have shape (n_cells, n_tau) = "
            f"({truth.shape[0]}, {taus.shape[0]}); got {quantiles.shape}."
        )
    diff = truth[:, None] - quantiles
    indicator = (diff < 0.0).astype(float)  # 1{y < q_tau}
    pinball = (taus[None, :] - indicator) * diff
    # Midpoint rule: integral_0^1 f dtau ~ mean_i f(tau_i) on a uniform grid.
    per_cell_crps = 2.0 * pinball.mean(axis=1)
    return float(np.mean(per_cell_crps))


def crps_gaussian_analytic(truth: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Exact mean CRPS for a per-cell Gaussian predictive distribution:

        CRPS(N(mu, sigma), y) = sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],
        z = (y - mu) / sigma.

    Used ONLY as an independent validation reference for the quantile-based
    ``_crps_from_quantiles`` path (the three Gaussian-predictive methods must
    agree with it to within the quadrature error) -- the reported CRPS for
    every method comes from the common quantile path, so that kriging (which
    has no closed form after the back-transform) and the others are scored by
    identical machinery.
    """
    (truth, mean), std = _filter_positive_std([truth, mean], np.asarray(std, dtype=float), "crps_gaussian_analytic")
    z = (truth - mean) / std
    per_cell = std * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(per_cell))


def gaussian_interval_widths(
    mean: np.ndarray,
    std: np.ndarray,
    p_levels: np.ndarray = INTERVAL_WIDTH_NOMINAL_LEVELS,
) -> np.ndarray:
    """Mean central-prediction-interval width (porosity %) per nominal level,
    for the per-cell Gaussian predictive distribution used by SGS,
    RBF+bootstrap and GP-MLE (the same N(mean, std) their UMG assumes).

    Parameters
    ----------
    mean, std : flattened 1-D arrays with conditioning-sample cells already
        excluded by the caller.
    p_levels : nominal central coverages, each in [0, 1).

    Returns
    -------
    (len(p_levels),) array of cell-averaged widths, in porosity %.
    """
    (mean,), std = _filter_positive_std([mean], np.asarray(std, dtype=float), "gaussian_interval_widths")
    tau_lo, tau_hi = _central_interval_taus(p_levels)
    z_lo, z_hi = norm.ppf(tau_lo), norm.ppf(tau_hi)
    # width = (mean + z_hi*std) - (mean + z_lo*std) = (z_hi - z_lo) * std
    widths = (z_hi - z_lo)[None, :] * std[:, None]
    return widths.mean(axis=0)


def gaussian_crps(
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_tau: int = CRPS_DEFAULT_N_TAU,
) -> float:
    """Mean CRPS (porosity %, lower better) for the per-cell Gaussian
    predictive distribution used by SGS, RBF+bootstrap and GP-MLE, computed
    through the common quantile/pinball path (NOT the closed form -- see
    ``crps_gaussian_analytic`` for the validation reference)."""
    (truth, mean), std = _filter_positive_std(
        [truth, mean], np.asarray(std, dtype=float), "gaussian_crps"
    )
    taus = crps_tau_grid(n_tau)
    quantiles = mean[:, None] + norm.ppf(taus)[None, :] * std[:, None]
    return _crps_from_quantiles(truth, quantiles, taus)


def kriging_quantiles_physical(
    kmap_ns: np.ndarray,
    std_ns: np.ndarray,
    taus: np.ndarray,
    vr: np.ndarray,
    vrg: np.ndarray,
    zmin: float,
    zmax: float,
    ltail: int,
    ltpar: float,
    utail: int,
    utpar: float,
    backtr_fn,
) -> np.ndarray:
    """(n_cells, n_tau) physical-unit kriging predictive quantiles.

    Exactly the construction ``kriging_fraction_in`` uses for UMG: the
    per-cell normal-score quantile ``kmap_ns + z_tau * std_ns`` is
    back-transformed to physical units through the normal-score transform
    table. The kriging VARIANCE is never back-transformed (a variance is not
    a monotone functional of the value, so it cannot be pushed through the
    back-transform pointwise); quantiles, being order statistics, ARE exact
    under the monotone back-transform.

    ``backtr_fn`` must be an array-capable back-transform with
    ``geostats.backtr_value``'s exact semantics/signature, i.e.
    ``src.experiments.kriging.backtr_value_vectorized``. It is a REQUIRED
    argument (no scalar-loop default) because this function evaluates
    n_cells * n_tau (~0.5M for n_tau=199) back-transforms -- a per-value
    Python loop over ``geostats.backtr_value`` would take minutes per method
    per axis level. The caller is responsible for having validated
    ``backtr_fn`` against the scalar reference (the range-axis evaluation
    script does this explicitly before calling in).
    """
    kmap_ns = np.asarray(kmap_ns, dtype=float)
    std_ns = np.asarray(std_ns, dtype=float)
    taus = np.asarray(taus, dtype=float)
    ns_quantiles = kmap_ns[:, None] + norm.ppf(taus)[None, :] * std_ns[:, None]
    return backtr_fn(ns_quantiles, vr, vrg, zmin, zmax, ltail, ltpar, utail, utpar)


def kriging_interval_widths(
    kmap_ns: np.ndarray,
    std_ns: np.ndarray,
    vr: np.ndarray,
    vrg: np.ndarray,
    zmin: float,
    zmax: float,
    ltail: int,
    ltpar: float,
    utail: int,
    utpar: float,
    backtr_fn,
    p_levels: np.ndarray = INTERVAL_WIDTH_NOMINAL_LEVELS,
) -> np.ndarray:
    """Mean central-prediction-interval width (PHYSICAL porosity %) per
    nominal level for kriging, via the exact normal-score quantile
    back-transform (same path as ``kriging_fraction_in``/UMG -- never a
    variance back-transform)."""
    (kmap_ns,), std_ns = _filter_positive_std(
        [kmap_ns], np.asarray(std_ns, dtype=float), "kriging_interval_widths"
    )
    tau_lo, tau_hi = _central_interval_taus(p_levels)
    taus = np.concatenate([tau_lo, tau_hi])
    q = kriging_quantiles_physical(
        kmap_ns, std_ns, taus, vr, vrg, zmin, zmax, ltail, ltpar, utail, utpar, backtr_fn
    )
    n_p = len(tau_lo)
    widths = q[:, n_p:] - q[:, :n_p]
    return widths.mean(axis=0)


def kriging_crps(
    truth_physical: np.ndarray,
    kmap_ns: np.ndarray,
    std_ns: np.ndarray,
    vr: np.ndarray,
    vrg: np.ndarray,
    zmin: float,
    zmax: float,
    ltail: int,
    ltpar: float,
    utail: int,
    utpar: float,
    backtr_fn,
    n_tau: int = CRPS_DEFAULT_N_TAU,
) -> float:
    """Mean CRPS (PHYSICAL porosity %, lower better) for kriging, scored
    against the physical-unit truth using back-transformed normal-score
    quantiles -- the identical quantile representation its UMG uses, and the
    identical pinball integration the other three methods go through.

    NOTE: the extreme quantiles this integral touches (tau ~ 0.0025 / 0.9975
    at the default n_tau=199) fall outside the normal-score transform table
    for many cells and are therefore produced by the LINEAR tail
    extrapolation toward BACKTR_ZMIN/ZMAX (see src/experiments/kriging.py).
    That is the same, already-accepted, project-wide back-transform
    convention that UMG's p -> 1 behavior depends on -- not a new
    approximation introduced here -- but it does mean kriging's CRPS tails
    inherit the finite-support back-transform's shape.
    """
    (truth_physical, kmap_ns), std_ns = _filter_positive_std(
        [truth_physical, kmap_ns], np.asarray(std_ns, dtype=float), "kriging_crps"
    )
    taus = crps_tau_grid(n_tau)
    q = kriging_quantiles_physical(
        kmap_ns, std_ns, taus, vr, vrg, zmin, zmax, ltail, ltpar, utail, utpar, backtr_fn
    )
    return _crps_from_quantiles(truth_physical, q, taus)


def conditioning_cell_mask(
    samples_df: pd.DataFrame,
    nx: int,
    ny: int,
    xmn: float,
    ymn: float,
    xsiz: float,
    ysiz: float,
) -> np.ndarray:
    """Boolean mask, shape (ny, nx), True = cell should be INCLUDED in
    evaluation (i.e. is NOT a conditioning-sample cell).

    Sample (X, Y) coordinates are assumed to be exact grid-cell centroids
    (as produced by src/sampling.py's random_interior_samples /
    regular_interior_samples, which snap every draw to its nearest cell
    centroid), so ``round((coord - origin) / cellsize)`` recovers the exact
    owning cell index -- same approach and row-flip convention as
    src/experiments/sgs.py's sample_ix/sample_iy/sample_row/sample_col (that
    computation was already reviewed and approved; reused/re-derived here,
    not re-invented).
    """
    ix = np.round((samples_df["X"].values - xmn) / xsiz).astype(int)
    iy = np.round((samples_df["Y"].values - ymn) / ysiz).astype(int)
    row = ny - 1 - iy  # row 0 = max-y row convention
    col = ix

    if np.any((row < 0) | (row >= ny) | (col < 0) | (col >= nx)):
        raise ValueError(
            "conditioning_cell_mask: at least one sample maps outside the "
            f"grid bounds (nx={nx}, ny={ny}) -- check samples_df/grid "
            "parameters passed in."
        )

    mask = np.ones((ny, nx), dtype=bool)
    mask[row, col] = False
    return mask
