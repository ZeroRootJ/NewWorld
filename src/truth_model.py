"""Ground-truth porosity field generation via unconditional sequential
Gaussian simulation (SGS).

Follows the pattern from the reference notebook
(make_nonlinear_MV_spatial_data_v13.ipynb) and the project's geostatspy
conventions (docs/geostatspy_conventions.md):

- a single ``vario`` dict, built once with ``GSLIB.make_variogram`` and
  reused as-is (never rebuilt/mutated here) so the same variogram model can
  later be shared with kriging (kb2d) for the "identical conditions across
  methods" requirement,
- unconditional simulation: since there is no real conditioning data, a
  dummy dataset placed far outside the grid (and outside the variogram
  range) is passed to ``sgsim`` -- geostatspy's sgsim requires *some* input
  dataframe even for unconditional runs,
- ``itrans=0`` because the dummy data are already standard normal (mean 0,
  stdev 1) -- no need for sgsim's internal normal-score transform,
- ``ktype=0`` (simple kriging) inside sgsim, consistent with this project's
  simple-kriging baseline,
- the raw simulation is standard normal (mean ~0, variance ~1); the target
  mean/stdev are then imposed with ``GSLIB.affine`` (as in the reference
  notebook), not by scaling data going into sgsim.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

import geostatspy.GSLIB as GSLIB
import geostatspy.geostats as geostats

# Column name conventions (named constants, reused across calls -- see
# docs/geostatspy_conventions.md section 6).
XCOL = "X"
YCOL = "Y"
VCOL = "NVar"  # standard-normal dummy variable column used only to drive sgsim

# Number of dummy conditioning points and how far outside the grid they sit.
# sgsim requires a conditioning dataframe even for unconditional simulation;
# placing points at a coordinate far outside the grid and its variogram range
# ensures they have no influence on the realization.
N_DUMMY = 100
DUMMY_COORD = -9999.0

# Standard-normal truncation used by sgsim's internal back-transform limits.
# The simulation is generated in standard-normal space (mean 0, stdev 1)
# before the affine correction to the target mean/stdev, so +/-3 sigma is a
# safe, generous bound (matches the reference notebook).
# Note: with itrans=0 this clipping is NOT applied by sgsim internally (the
# zmin/zmax + ltail/utail back-transform path is only exercised when
# itrans=1); these bounds are currently unused in effect but kept here for
# documentation of the intended standard-normal range.
ZMIN_NS = -3.0
ZMAX_NS = 3.0


def make_porosity_truth(
    nx: int,
    ny: int,
    xsiz: float,
    ysiz: float,
    xmn: float,
    ymn: float,
    vario: Dict[str, Any],
    mean: float,
    stdev: float,
    seed: int,
) -> np.ndarray:
    """Generate one unconditional Gaussian-simulation porosity truth field.

    Parameters
    ----------
    nx, ny : number of grid cells in x and y
    xsiz, ysiz : cell size in x and y
    xmn, ymn : grid origin, i.e. the centroid of the first cell (GSLIB
        convention -- see docs/geostatspy_conventions.md section 5)
    vario : variogram dict built once via ``GSLIB.make_variogram`` and
        reused (not rebuilt here)
    mean, stdev : target univariate mean/stdev for the affine correction
    seed : random seed for this realization (drives both the dummy-data
        draw and sgsim itself, so a run is fully reproducible from `seed`
        alone)

    Returns
    -------
    np.ndarray of shape (ny, nx): the porosity truth field. Row 0
    corresponds to the maximum-y row (top row), matching geostatspy's
    sgsim/imshow convention.
    """
    rng = np.random.RandomState(seed)
    df_dummy = pd.DataFrame(
        {
            XCOL: np.full(N_DUMMY, DUMMY_COORD),
            YCOL: np.full(N_DUMMY, DUMMY_COORD),
            VCOL: rng.normal(0.0, 1.0, N_DUMMY),
        }
    )

    sim_ns = geostats.sgsim(
        df_dummy,
        XCOL,
        YCOL,
        VCOL,
        wcol=-1,
        scol=-1,
        tmin=-9999,
        tmax=9999,
        itrans=0,
        ismooth=0,
        dftrans=0,
        tcol=0,
        twtcol=0,
        zmin=ZMIN_NS,
        zmax=ZMAX_NS,
        ltail=1,
        ltpar=ZMIN_NS,
        utail=1,
        utpar=ZMAX_NS,
        nsim=1,
        nx=nx,
        xmn=xmn,
        xsiz=xsiz,
        ny=ny,
        ymn=ymn,
        ysiz=ysiz,
        seed=seed,
        ndmin=0,
        ndmax=100,
        # mults=0, nodmax=10 intentionally kept as-is per project decision
        # 2026-09-12, differs from docs/geostatspy_conventions.md example
        # values (mults=1, nodmax=20) -- do not "fix" to match the docs
        # example without re-confirming with the user.
        nodmax=10,
        mults=0,
        nmult=2,
        noct=-1,
        ktype=0,
        colocorr=0.0,
        sec_map=0,
        vario=vario,
    )[0]

    truth = GSLIB.affine(sim_ns, mean, stdev)
    return truth
