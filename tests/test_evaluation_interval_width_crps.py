"""Self-checks for the interval-width / CRPS metrics added to
src/evaluation.py (range-axis task, 2026-09-14).

These test the two things that can silently go wrong in a quantile-based
scoring implementation:

1. the pinball/CRPS quadrature actually converges to the true CRPS (checked
   against the closed-form Gaussian CRPS, which is exact), and
2. the interval-width definition is the central [(1-p)/2, (1+p)/2] interval
   in physical units (checked against the closed-form Gaussian width).

Run with: .venv/Scripts/python.exe -m pytest tests/test_evaluation_interval_width_crps.py -q
"""

import numpy as np
from scipy.stats import norm

from src.evaluation import (
    CRPS_DEFAULT_N_TAU,
    INTERVAL_WIDTH_NOMINAL_LEVELS,
    calc_umg,
    accuracy_plot_fraction_in,
    crps_gaussian_analytic,
    crps_tau_grid,
    gaussian_crps,
    gaussian_interval_widths,
    mse,
)

RNG = np.random.default_rng(12345)
N = 2000


def _synthetic():
    mean = RNG.normal(15.0, 2.0, size=N)
    std = RNG.uniform(0.5, 3.0, size=N)
    truth = mean + std * RNG.standard_normal(N)
    return truth, mean, std


def test_crps_tau_grid_is_uniform_midpoint():
    taus = crps_tau_grid(199)
    assert len(taus) == 199
    assert np.isclose(taus[0], 0.5 / 199)
    assert np.isclose(taus[-1], 1.0 - 0.5 / 199)
    assert np.allclose(np.diff(taus), 1.0 / 199)


def test_gaussian_crps_matches_closed_form():
    """The quantile/pinball CRPS must converge to the analytic Gaussian CRPS."""
    truth, mean, std = _synthetic()
    exact = crps_gaussian_analytic(truth, mean, std)
    prev_err = np.inf
    for n_tau in (49, 99, 199, 499, 999):
        approx = gaussian_crps(truth, mean, std, n_tau=n_tau)
        err = abs(approx - exact)
        assert err < prev_err  # monotone refinement
        prev_err = err
    # At the project default the residual quadrature error must be tiny
    # relative to the metric's own magnitude (porosity %).
    approx = gaussian_crps(truth, mean, std, n_tau=CRPS_DEFAULT_N_TAU)
    assert abs(approx - exact) / exact < 5e-3


def test_gaussian_interval_width_matches_closed_form():
    _, mean, std = _synthetic()
    widths = gaussian_interval_widths(mean, std, p_levels=np.array([0.5, 0.95]))
    expected_50 = (norm.ppf(0.75) - norm.ppf(0.25)) * std.mean()
    expected_95 = (norm.ppf(0.975) - norm.ppf(0.025)) * std.mean()
    assert np.isclose(widths[0], expected_50)
    assert np.isclose(widths[1], expected_95)


def test_interval_width_levels_exclude_p_equals_one():
    """p=1.0 (infinite Gaussian width) must not be in the averaged levels."""
    assert len(INTERVAL_WIDTH_NOMINAL_LEVELS) == 19
    assert INTERVAL_WIDTH_NOMINAL_LEVELS.max() < 1.0


def test_crps_penalizes_both_over_and_under_dispersion():
    """A proper scoring rule must be minimized at the TRUE predictive std --
    this is the property UMG/coverage lacks and the reason CRPS was added."""
    truth, mean, std = _synthetic()
    honest = gaussian_crps(truth, mean, std)
    too_wide = gaussian_crps(truth, mean, 3.0 * std)
    too_narrow = gaussian_crps(truth, mean, 0.3 * std)
    assert honest < too_wide
    assert honest < too_narrow
    # ...while an inflated interval scores BETTER than honest on plain
    # coverage-vs-nominal up to the point of over-coverage -- the exact
    # "cheating" concern that motivated adding CRPS/width.
    honest_umg = calc_umg(*accuracy_plot_fraction_in(truth, mean, std))
    wide_umg = calc_umg(*accuracy_plot_fraction_in(truth, mean, 3.0 * std))
    assert wide_umg < honest_umg  # over-coverage is still penalized by UMG...
    assert gaussian_interval_widths(mean, 3.0 * std)[-1] > gaussian_interval_widths(mean, std)[-1]


def test_existing_metrics_unchanged():
    """Regression guard: the pre-existing mse/UMG path must be untouched."""
    truth, mean, std = _synthetic()
    assert np.isclose(mse(truth, mean), float(np.mean((truth - mean) ** 2)))
    p, frac = accuracy_plot_fraction_in(truth, mean, std)
    assert len(p) == 20 and p[0] == 0.0 and p[-1] == 1.0
    assert 0.0 <= calc_umg(p, frac) <= 1.0
