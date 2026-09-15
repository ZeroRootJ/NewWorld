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

# --- Sample-density axis: X draws SHARED across levels, Y draws differ ---
# User decision 2026-09-15 (an earlier nested-subset instruction was
# explicitly CANCELLED -- do not re-introduce nesting here): each level of
# the sample-density axis (5% / 2% / 1% of the 2500-cell grid = n_samples
# 125 / 50 / 25) draws its samples from the full ground-truth field, i.e. by
# simply calling ``random_interior_samples`` with that level's own
# ``n_samples`` and the shared SAMPLE_SEED. Smaller sets are therefore NOT
# subsets of larger ones -- but they are NOT statistically independent of
# them either, and must not be described as "independent" (see below).
#
# This is NOT an accident of the implementation, and the consequence was
# measured rather than assumed (2026-09-15). ``random_interior_samples``
# draws ``xs = rng.uniform(..., n_samples)`` and then ``ys = rng.uniform(...,
# n_samples)`` as two separate calls on the same RandomState, so at a fixed
# seed the x-values of the n=25 draw match the first 25 x-values of the
# n=125 draw but the y-values do not (stream positions 25..49 vs. 125..149).
# Measured overlap of the resulting conditioning-CELL sets at SAMPLE_SEED=20:
#
#     |n25 ∩ n125| = 4 of 25,  |n50 ∩ n125| = 4 of 50,  |n25 ∩ n50| = 0
#
# KNOWN LIMITATION, must be carried into any report built on this axis:
# with one ground-truth realization per level, a difference between two
# sample-density levels mixes TWO effects -- "less data was given" and "the
# data landed in different places" -- and this design does not separate them.
#
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


def build_vario(hmaj1: float = HMAJ1, hmin1: float = HMIN1) -> Dict[str, Any]:
    """Build the (single, reused) base-case variogram dict.

    ``hmaj1``/``hmin1`` default to the base-case constants (300 m,
    isotropic) so calling this with no arguments reproduces the base case
    exactly. Passing different values is how the range axis (deliverable 2,
    docs/experiment_context.md) varies the variogram range while holding
    every other parameter (nugget, sill split, azimuth) fixed --
    one-factor-at-a-time.

    Reused as-is by every method that needs it (kriging/SGS baselines,
    later); RBF+bootstrap and GP-MLE do not consume this dict directly but
    it is built here for a single source of truth on variogram parameters.
    """
    return GSLIB.make_variogram(
        nug=NUG, nst=1, it1=IT1, cc1=CC1, azi1=AZI1, hmaj1=hmaj1, hmin1=hmin1
    )


def get_base_case_truth(
    truth_seed: int = TRUTH_SEED, hmaj1: float = HMAJ1, hmin1: float = HMIN1
) -> "pd.DataFrame":
    """Regenerate a base-case-style ground-truth field.

    Defaults (``truth_seed=TRUTH_SEED``, ``hmaj1=HMAJ1``, ``hmin1=HMIN1``)
    reproduce the exact base-case truth field (seed=101, range=300m). Passing
    a different ``hmaj1``/``hmin1`` (equal, for the isotropic range axis)
    regenerates the truth under a different variogram range while keeping
    every other parameter (grid, distribution, nugget, seed) identical --
    the range-axis experiment (docs/experiment_context.md deliverable 2).

    Returns
    -------
    np.ndarray of shape (NY, NX), same convention as
    src.truth_model.make_porosity_truth (row 0 = max-y row).
    """
    vario = build_vario(hmaj1=hmaj1, hmin1=hmin1)
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
        seed=truth_seed,
    )
    return truth


def get_conditioning_samples(
    truth, sample_seed: int = SAMPLE_SEED, n_samples: int = N_SAMPLES
) -> pd.DataFrame:
    """Draw the (up to) ``n_samples`` random interior samples from the given
    truth field.

    Parameters
    ----------
    truth : np.ndarray, shape (NY, NX) -- exhaustive truth grid to sample
        from (typically the output of ``get_base_case_truth``)
    sample_seed : defaults to the module-level ``SAMPLE_SEED`` (the single
        source of truth for the sample-location seed). Callers should not
        pass a different value unless they explicitly intend to deviate from
        the shared base-case sample locations.
    n_samples : defaults to the module-level ``N_SAMPLES`` (=125, the base
        case / 5% of the 2500-cell grid), which reproduces the base case
        exactly. The sample-density axis (docs/progress.md axis 5) passes 50
        (2%) and 25 (1%). The levels are neither nested subsets nor
        independent draws -- they share their X draws and differ only in Y;
        see the "X draws SHARED across levels" comment above for the measured
        overlap and the limitation this implies.

    Returns
    -------
    pd.DataFrame with columns ["X", "Y", VCOL]. May have fewer than
    ``n_samples`` rows after duplicate-cell removal (this is what happens at
    the default n_samples=125 -> 121 rows) -- callers must read len(df)
    rather than assume ``n_samples``, and record the actual count in their
    manifest.
    """
    df = random_interior_samples(
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        n_samples=n_samples,
        margin_frac=MARGIN_FRAC,
        value_grid=truth,
        colname=VCOL,
        seed=sample_seed,
    )
    return df


def get_base_case_conditioning_data(
    sample_seed: int = SAMPLE_SEED,
    truth_seed: int = TRUTH_SEED,
    hmaj1: float = HMAJ1,
    hmin1: float = HMIN1,
    n_samples: int = N_SAMPLES,
) -> Tuple[Any, pd.DataFrame]:
    """Convenience wrapper: regenerate the base-case truth and draw samples
    from it in one call. Every method-comparison script (kriging.py, sgs.py,
    rbf_bootstrap.py, gp_mle.py) should call this rather than calling
    get_base_case_truth / get_conditioning_samples separately, so there is
    exactly one code path that ties truth generation to sampling.
    ``sample_seed``/``truth_seed``/``hmaj1``/``hmin1``/``n_samples`` all
    default to the base-case constants -- calling with no arguments
    reproduces the base case exactly; passing ``truth_seed``/``hmaj1``/
    ``hmin1`` is how the range axis (docs/experiment_context.md deliverable
    2) varies the variogram range one-factor-at-a-time while keeping the
    sample locations (drawn with the same ``sample_seed``, from a truth field
    of identical shape/distribution) directly comparable across axis levels,
    and passing ``n_samples`` is how the sample-density axis varies the
    conditioning sample count one-factor-at-a-time (independent draws per
    level, NOT nested subsets -- see ``get_conditioning_samples``).
    """
    truth = get_base_case_truth(truth_seed=truth_seed, hmaj1=hmaj1, hmin1=hmin1)
    samples = get_conditioning_samples(
        truth, sample_seed=sample_seed, n_samples=n_samples
    )
    return truth, samples
