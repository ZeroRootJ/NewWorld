"""Regression test for the "identical sample locations across methods"
requirement (CLAUDE.md / docs/experiment_context.md section 6).

This does NOT run rbf_bootstrap.py / gp_mle.py end-to-end (too slow for a
regression check, since both re-tune hyperparameters). Instead it checks the
actual mechanism that is supposed to guarantee identical conditioning data:

1. ``src.experiments.base_case_conditioning.get_base_case_conditioning_data``
   is deterministic -- calling it twice (independently, with no shared
   mutable state) yields bit-identical truth grids and sample DataFrames.
2. ``rbf_bootstrap.py`` and ``gp_mle.py`` import ``SAMPLE_SEED``/``N_SAMPLES``
   from ``base_case_conditioning`` rather than each re-declaring their own
   copies of those constants -- i.e. the two scripts are structurally tied
   to the same values, not just coincidentally equal literals.

No pytest dependency is used (not in requirements.txt / not installed in
this project's .venv) -- this is a plain assert-based script, runnable
directly:

    .venv/Scripts/python.exe -m tests.test_base_case_conditioning

It also works fine if pytest is later added to the project, since pytest
auto-discovers module-level ``test_*`` functions.
"""

import numpy as np

from src.experiments import base_case_conditioning
from src.experiments import gp_mle
from src.experiments import rbf_bootstrap


def test_conditioning_data_is_reproducible_across_independent_calls():
    """Two independent calls to get_base_case_conditioning_data() (the one
    and only place sample locations are generated, per
    base_case_conditioning.py's module docstring) must yield identical truth
    grids and identical sample locations/values.
    """
    truth_a, samples_a = base_case_conditioning.get_base_case_conditioning_data()
    truth_b, samples_b = base_case_conditioning.get_base_case_conditioning_data()

    assert np.array_equal(truth_a, truth_b), "truth grids differ between two calls"

    assert len(samples_a) == len(samples_b), (
        f"sample counts differ: {len(samples_a)} vs {len(samples_b)}"
    )
    assert list(samples_a.columns) == list(samples_b.columns)
    for col in samples_a.columns:
        assert np.array_equal(samples_a[col].values, samples_b[col].values), (
            f"column '{col}' differs between two calls"
        )


def test_rbf_bootstrap_and_gp_mle_share_sample_seed_and_n_samples():
    """rbf_bootstrap.py and gp_mle.py must import (not re-declare) SAMPLE_SEED
    and N_SAMPLES from base_case_conditioning.py, so the two scripts cannot
    silently drift apart. This checks both equality of values and that both
    modules' names are literally the same object as base_case_conditioning's
    (an `is` check would be over-strict for ints in general, but for small
    ints CPython caches them anyway; the equality check is the meaningful
    assertion here).
    """
    assert rbf_bootstrap.SAMPLE_SEED == base_case_conditioning.SAMPLE_SEED
    assert gp_mle.SAMPLE_SEED == base_case_conditioning.SAMPLE_SEED
    assert rbf_bootstrap.N_SAMPLES == base_case_conditioning.N_SAMPLES
    assert gp_mle.N_SAMPLES == base_case_conditioning.N_SAMPLES


def test_rbf_bootstrap_and_gp_mle_condition_on_identical_samples():
    """End-to-end (but cheap: no CV / GP fitting) check that both scripts'
    entry point for conditioning data -- get_base_case_conditioning_data,
    called with each script's SAMPLE_SEED -- produces identical samples.
    """
    truth_rbf, samples_rbf = base_case_conditioning.get_base_case_conditioning_data(
        sample_seed=rbf_bootstrap.SAMPLE_SEED
    )
    truth_gp, samples_gp = base_case_conditioning.get_base_case_conditioning_data(
        sample_seed=gp_mle.SAMPLE_SEED
    )

    assert np.array_equal(truth_rbf, truth_gp)
    assert len(samples_rbf) == len(samples_gp)
    for col in samples_rbf.columns:
        assert np.array_equal(samples_rbf[col].values, samples_gp[col].values), (
            f"column '{col}' differs between rbf_bootstrap's and gp_mle's conditioning data"
        )


def main():
    tests = [
        test_conditioning_data_is_reproducible_across_independent_calls,
        test_rbf_bootstrap_and_gp_mle_share_sample_seed_and_n_samples,
        test_rbf_bootstrap_and_gp_mle_condition_on_identical_samples,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
