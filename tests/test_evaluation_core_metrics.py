"""Regression guards for the MSE / UMG metrics in src/evaluation.py.

These two tests were kept from the former
tests/test_evaluation_interval_width_crps.py when the sharpness metrics
(interval width, CRPS) were removed on 2026-10-05 per user decision; they
only exercise the metrics that remain.

Run with: .venv/Scripts/python.exe -m pytest tests/test_evaluation_core_metrics.py -q
"""

import numpy as np

from src.evaluation import (
    calc_umg,
    accuracy_plot_fraction_in,
    mse,
)

RNG = np.random.default_rng(12345)
N = 2000


def _synthetic():
    mean = RNG.normal(15.0, 2.0, size=N)
    std = RNG.uniform(0.5, 3.0, size=N)
    truth = mean + std * RNG.standard_normal(N)
    return truth, mean, std


def test_umg_penalizes_over_dispersion():
    """UMG must score an inflated (3x std) predictive distribution below the
    honest one -- over-coverage is penalized, not rewarded."""
    truth, mean, std = _synthetic()
    honest_umg = calc_umg(*accuracy_plot_fraction_in(truth, mean, std))
    wide_umg = calc_umg(*accuracy_plot_fraction_in(truth, mean, 3.0 * std))
    assert wide_umg < honest_umg


def test_existing_metrics_unchanged():
    """Regression guard: the mse/UMG path must be untouched."""
    truth, mean, std = _synthetic()
    assert np.isclose(mse(truth, mean), float(np.mean((truth - mean) ** 2)))
    p, frac = accuracy_plot_fraction_in(truth, mean, std)
    assert len(p) == 20 and p[0] == 0.0 and p[-1] == 1.0
    assert 0.0 <= calc_umg(p, frac) <= 1.0
