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
