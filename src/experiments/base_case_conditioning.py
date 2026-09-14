"""Shared conditioning-data generation for the base-case method comparison
(RBF+bootstrap vs. GP-MLE, and later kriging/SGS).

This module exists so that sample locations are generated in exactly one
place and reused by every method's experiment script -- the "identical
sample locations across methods" requirement in CLAUDE.md /
docs/experiment_context.md section 6 is otherwise easy to violate by
accident (e.g. two scripts each calling np.random with the same seed but a
slightly different call sequence).

Grid/variogram/distribution parameters are re-imported from
``src.experiments.base_case`` (not re-declared here) so the two stay in
sync by construction rather than by convention.
"""

from typing import Any, Dict, Tuple

import pandas as pd

import geostatspy.GSLIB as GSLIB

from src.experiments.base_case import (
    NX,
    NY,
    XSIZ,
    YSIZ,
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    XMN,
    YMN,
    POR_MEAN,
    POR_STDEV,
    NUG,
    IT1,
    CC1,
    AZI1,
    HMAJ1,
    HMIN1,
)
from src.sampling import random_interior_samples
from src.truth_model import make_porosity_truth

# Single ground-truth realization used for this method-comparison step (the
# first seed of the base_case.SEEDS list) -- CLAUDE.md task description:
# "Base case ground truth(단일 realization, seed=101 ...)".
TRUTH_SEED = 101

# 5% of the 50x50 = 2500-cell truth grid (2026-09-14, changed from 20%/500 --
# user decision after literature review, not to be re-litigated here). Basis:
# docs/references.md -- Fiedler et al. 2021 uses exactly 5% (50 of 1000 grid
# points); Trifonov et al. 2025 (SPE10 benchmark) uses ~5.3% (700 wells of
# ~13,200 areal cells) as their densest well count. This is the single source
# of truth for the sample count -- every method-comparison script
# (rbf_bootstrap.py, gp_mle.py, ...) must import N_SAMPLES from here rather
# than re-declaring the literal, so the value cannot silently drift between
# methods. The prior N_SAMPLES=500 (20%) run in results/raw/ is original data
# and is left untouched; this change only affects new runs going forward.
N_SAMPLES = 125

# Same interior-sampling margin convention as regular_interior_samples
# (docs/geostatspy_conventions.md / src/sampling.py).
MARGIN_FRAC = 0.05

VCOL = "Por"

# Single source of truth for the sample-location seed. All method-comparison
# scripts (rbf_bootstrap.py, gp_mle.py, ...) must import SAMPLE_SEED from
# here rather than each re-declaring their own SAMPLE_SEED constant -- with
# separate declarations, "identical sample locations across methods"
# (CLAUDE.md requirement) would only hold by the two literals happening to
# agree, not by construction.
SAMPLE_SEED = 20


def build_vario() -> Dict[str, Any]:
    """Build the (single, reused) base-case variogram dict.

    Reused as-is by every method that needs it (kriging/SGS baselines,
    later); RBF+bootstrap and GP-MLE do not consume this dict directly but
    it is built here for a single source of truth on variogram parameters.
    """
    return GSLIB.make_variogram(
        nug=NUG, nst=1, it1=IT1, cc1=CC1, azi1=AZI1, hmaj1=HMAJ1, hmin1=HMIN1
    )


def get_base_case_truth() -> "pd.DataFrame":
    """Regenerate the single base-case ground-truth field (seed=101).

    Returns
    -------
    np.ndarray of shape (NY, NX), same convention as
    src.truth_model.make_porosity_truth (row 0 = max-y row).
    """
    vario = build_vario()
    truth = make_porosity_truth(
        nx=NX,
        ny=NY,
        xsiz=XSIZ,
        ysiz=YSIZ,
        xmn=XMN,
        ymn=YMN,
        vario=vario,
        mean=POR_MEAN,
        stdev=POR_STDEV,
        seed=TRUTH_SEED,
    )
    return truth


def get_conditioning_samples(truth, sample_seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """Draw the (up to) N_SAMPLES random interior samples from the given
    truth field.

    Parameters
    ----------
    truth : np.ndarray, shape (NY, NX) -- exhaustive truth grid to sample
        from (typically the output of ``get_base_case_truth``)
    sample_seed : defaults to the module-level ``SAMPLE_SEED`` (the single
        source of truth for the sample-location seed). Callers should not
        pass a different value unless they explicitly intend to deviate from
        the shared base-case sample locations.

    Returns
    -------
    pd.DataFrame with columns ["X", "Y", VCOL]. May have fewer than
    N_SAMPLES rows after duplicate-cell removal -- callers must read
    len(df) rather than assume N_SAMPLES, and record the actual count in
    their manifest.
    """
    df = random_interior_samples(
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        n_samples=N_SAMPLES,
        margin_frac=MARGIN_FRAC,
        value_grid=truth,
        colname=VCOL,
        seed=sample_seed,
    )
    return df


def get_base_case_conditioning_data(sample_seed: int = SAMPLE_SEED) -> Tuple[Any, pd.DataFrame]:
    """Convenience wrapper: regenerate the base-case truth and draw samples
    from it in one call. Both rbf_bootstrap.py and gp_mle.py should call
    this rather than calling get_base_case_truth / get_conditioning_samples
    separately, so there is exactly one code path that ties truth generation
    to sampling. ``sample_seed`` defaults to the module-level ``SAMPLE_SEED``
    -- callers should import and pass ``SAMPLE_SEED`` from this module (or
    rely on the default) instead of re-declaring their own seed constant, so
    that "identical sample locations across methods" holds structurally
    rather than by two literals happening to agree.
    """
    truth = get_base_case_truth()
    samples = get_conditioning_samples(truth, sample_seed=sample_seed)
    return truth, samples
