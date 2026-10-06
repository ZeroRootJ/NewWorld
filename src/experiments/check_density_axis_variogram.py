"""Diagnostic: does the variogram MODEL that kriging/SGS are handed on the
sample-density axis match what the conditioning DATA actually show?

On the sample-density axis every method that needs a variogram (kriging,
SGS) is given the exact model the truth was generated with
(``build_vario()``: nug=0.05 + spherical cc1=0.95, range 300 m, unit sill in
normal-score space) instead of a variogram inferred from the samples. This
script records, for several truth realizations and every density level,
the experimental variogram that would have been available from the samples
themselves, alongside that model, so the two can be compared directly
(user request 2026-10-05).

What is computed, per truth realization (TRUTH_SEEDS) and density level
(EXTENDED_AXIS_LEVELS, n_samples from SAMPLE_COUNTS):
  1. the truth field, regenerated with ``get_base_case_truth(truth_seed=s)``
     (seed 101 is the one the density axis actually uses; the others reuse
     every other base-case parameter unchanged);
  2. the conditioning samples, drawn exactly as the axis draws them
     (``get_conditioning_samples(truth, SAMPLE_SEED, n)``);
  3. the samples' normal scores via ``src.nscore.nscore`` -- the same
     transform kriging.py applies before kb2d, so the experimental variogram
     lives in the same space as the unit-sill model;
  4. ``geostats.gamv`` omnidirectional experimental variogram of those
     normal scores (isill=0: NOT re-standardized).
Also recorded per realization: the exhaustive (all 2500 cells) normal-score
grid semivariogram along x and y (``src.grid_variogram``), and once, the
model curve itself (``geostats.vmodel``-equivalent closed form below).

Output: one tidy long-format CSV (``variogram_long.csv``) with columns
truth_seed, axis_level, n_samples_actual, source, direction, lag_m, gamma,
npairs -- source in {"samples", "exhaustive", "model"}.

Run with: ../../../.venv/Scripts/python.exe -m src.experiments.check_density_axis_variogram
"""

import numpy as np
import pandas as pd

import geostatspy.geostats as geostats

from src.experiments.base_case import NX, XSIZ, XMN, YMN, NY, YSIZ
from src.experiments.base_case_conditioning import (
    HMAJ1,
    NUG,
    SAMPLE_SEED,
    VCOL,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.sample_density_axis import EXTENDED_AXIS_LEVELS, SAMPLE_COUNTS
from src.grid_variogram import grid_semivariogram
from src.io import make_run_dir, save_result
from src.nscore import nscore

EXPERIMENT_NAME = "density_axis_variogram_check"
CODE_ENTRYPOINT = "src/experiments/check_density_axis_variogram.py"

# 101 = the truth the density axis uses; 102-106 = additional realizations
# with every other base-case parameter unchanged.
TRUTH_SEEDS = [101, 102, 103, 104, 105, 106]

# gamv parameters (omnidirectional: isotropic base-case model).
LAG_DIST = 50.0
LAG_TOL = 25.0
NLAG = 12  # up to 600 m = 2x range
AZI = 0.0
ATOL = 90.0
BANDH = 9999.0
ISILL = 0  # do NOT rescale by the sample variance

# Exhaustive grid variogram: lags 1..30 cells (20..600 m).
GRID_NLAG = 30

# Model curve resolution.
MODEL_LAGS_M = np.linspace(0.0, 600.0, 121)


def spherical_model(h: np.ndarray, nug: float = NUG, a: float = HMAJ1) -> np.ndarray:
    """Unit-sill nugget + spherical model, as built by ``build_vario()``:
    gamma(0)=0, gamma(h>0)=nug + (1-nug)*sph(h/a)."""
    h = np.asarray(h, dtype=float)
    r = np.minimum(h / a, 1.0)
    g = nug + (1.0 - nug) * (1.5 * r - 0.5 * r ** 3)
    return np.where(h > 0, g, 0.0)


def _ns_field(truth: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({VCOL: truth.ravel()})
    ns = nscore(df, VCOL, verbose=False).ns
    return np.asarray(ns).reshape(truth.shape)


def main():
    rows = []
    n_actual = {}

    for h, g in zip(MODEL_LAGS_M, spherical_model(MODEL_LAGS_M)):
        rows.append(dict(truth_seed=-1, axis_level="", n_samples_actual=-1, source="model",
                         direction="omni", lag_m=float(h), gamma=float(g), npairs=-1))

    for seed in TRUTH_SEEDS:
        truth = get_base_case_truth(truth_seed=seed)

        ns_grid = _ns_field(truth)
        for d in ("x", "y"):
            gv = grid_semivariogram(ns_grid, d, GRID_NLAG, XSIZ if d == "x" else YSIZ)
            for h, g, npr in zip(gv["lag_m"], gv["gamma"], gv["npairs"]):
                rows.append(dict(truth_seed=seed, axis_level="", n_samples_actual=NX * NY,
                                 source="exhaustive", direction=d, lag_m=float(h),
                                 gamma=float(g), npairs=int(npr)))

        for level in EXTENDED_AXIS_LEVELS:
            samples = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED,
                                               n_samples=SAMPLE_COUNTS[level])
            n = len(samples)
            n_actual["%d_%s" % (seed, level)] = n
            samples = samples.assign(NS=nscore(samples, VCOL, verbose=False).ns)
            lag, gam, npair = geostats.gamv(samples, "X", "Y", "NS", -9999, 9999,
                                            LAG_DIST, LAG_TOL, NLAG, AZI, ATOL, BANDH, ISILL)
            for h, g, npr in zip(lag, gam, npair):
                if npr <= 0:
                    continue
                rows.append(dict(truth_seed=seed, axis_level=level, n_samples_actual=n,
                                 source="samples", direction="omni", lag_m=float(h),
                                 gamma=float(g), npairs=int(npr)))

    df = pd.DataFrame(rows)
    run_dir = make_run_dir(EXPERIMENT_NAME)
    out_csv = "variogram_long.csv"
    df.to_csv(run_dir / out_csv, index=False)

    params = dict(
        truth_seeds=TRUTH_SEEDS, sample_seed=SAMPLE_SEED,
        axis_levels=EXTENDED_AXIS_LEVELS,
        n_samples_requested={lv: SAMPLE_COUNTS[lv] for lv in EXTENDED_AXIS_LEVELS},
        model=dict(nug=NUG, cc1=1.0 - NUG, it1="spherical", range_m=HMAJ1, sill=1.0),
        gamv=dict(lag_dist=LAG_DIST, lag_tol=LAG_TOL, nlag=NLAG, azi=AZI, atol=ATOL,
                  bandh=BANDH, isill=ISILL, variable="normal score of samples (src.nscore)"),
        grid_variogram=dict(nlag_cells=GRID_NLAG, directions=["x", "y"],
                            variable="normal score of all 2500 truth cells"),
        grid=dict(nx=NX, ny=NY, xsiz=XSIZ, ysiz=YSIZ, xmn=XMN, ymn=YMN),
    )
    save_result(EXPERIMENT_NAME, params, TRUTH_SEEDS + [SAMPLE_SEED], run_dir=run_dir,
                code_entrypoint=CODE_ENTRYPOINT, output_files=[out_csv],
                extra=dict(n_samples_actual=n_actual))
    print(run_dir)


if __name__ == "__main__":
    main()
