"""Diagnose whether RBF+bootstrap's EPSILON_GRID (src/experiments/rbf_bootstrap.py)
is fine enough to resolve the range-axis's 8 levels (100..800 m), BEFORE any
grid change is made.

NOT REPRODUCIBLE AFTER THE GRID CHANGE THIS SCRIPT MOTIVATED (flagged by
reviewer, 2026-09-22): "production" below is read LIVE from
src.experiments.rbf_bootstrap.EPSILON_GRID. This script's own result led to
N_EPSILON going 25 -> 121 in that module, so re-running this script NOW makes
"production" and "fine" the SAME 121-point grid -- production_pct_worse_than_fine
will come back ~0 everywhere and the historical 25-vs-121 comparison below
(the actual evidence the N_EPSILON=121 decision was based on) can no longer be
reproduced by executing this file. That comparison is preserved instead in
results/processed/range_axis/epsilon_grid_diagnostic.csv (written when
EPSILON_GRID was still the 25-point grid) and in docs/progress.md. To redo this
exact before/after comparison against a DIFFERENT candidate resolution, pin
PRODUCTION_EPSILON_GRID to an explicit N_EPSILON=25 grid here instead of
importing the live module constant.

WHY THIS SCRIPT EXISTS
-----------------------
docs/progress.md (2026-09-17 entry) already recorded that with the 25-point
production EPSILON_GRID, 5 of the 8 range-axis levels (400/500/600/700/800 m)
do not all land on distinct grid points -- some collapse onto the same CV-
selected epsilon (measured directly from the pinned manifests below: 400 m
shares 300 m's pick at practical range 274.25 m, and 500/600/700/800 m all
share a single pick at 342.02 m). That collapse is consistent with two very
different explanations that look identical at 25-point resolution:

  (a) grid-resolution artifact -- the true (infinite-resolution) CV-MSE-
      minimizing epsilon genuinely DIFFERS across these levels, but the 25
      grid points are too coarse to land near each level's true optimum, so
      several levels snap to the same nearest grid point; or
  (b) genuinely flat CV surface -- RBF+bootstrap's CV-MSE, as a function of
      epsilon, truly does not distinguish these ranges (e.g. because 125
      samples over a 1000x1000 m domain cannot resolve correlation lengths
      beyond some point), so even an arbitrarily fine grid would keep
      returning close to the same argmin.

This script tells the two apart by REPLAYING THE IDENTICAL CV PROCEDURE
(same CV_FOLDS, same CV_SEED, same SMOOTHING_GRID, same RBF_KERNEL --
imported from src.experiments.rbf_bootstrap, never re-declared) against each
of the 8 range-axis levels' ACTUAL conditioning samples, using a MUCH finer
epsilon grid (N_EPSILON_FINE=121 points, log-uniform, same domain-geometry
bounds _EPS_LENGTH_MIN_M=10 m / _EPS_LENGTH_MAX_M=2000 m as the production
grid -- the bounds are NOT touched here, only the resolution between them).
This mirrors the same-shaped diagnostic already run once for the sample-
density axis's 3 levels on 2026-09-17 (45-point fine grid there), now applied
to the range axis's 8 levels with a denser fine grid (121 points) because 8
levels need to be told apart, not 3.

Conditioning samples used: regenerated via
``src.experiments.base_case_conditioning.get_base_case_conditioning_data``
with TRUTH_SEED (=101) and this axis's own hmaj1=hmin1=range (SAMPLE_SEED/
N_SAMPLES left at their base-case defaults, matching how
src/experiments/range_axis.py calls rbf_bootstrap.main() for every level).
Each level's regenerated samples.csv is verified (not assumed) against the
samples.csv actually recorded in that level's PINNED rbf_bootstrap run
(results/processed/range_axis/source_runs.json) before anything is computed
from them -- if they disagreed, this diagnostic would silently describe a
different conditioning dataset than the one the pinned run/production grid
decision was actually made from.

What is reported per level (results/processed/range_axis/epsilon_grid_diagnostic.csv)
----------------------------------------------------------------------------
  - the PRODUCTION 25-point grid's argmin (epsilon, smoothing, CV-MSE),
    recomputed here from scratch (not read off the pinned manifest) so the
    comparison below is apples-to-apples: identical fold splits, identical
    code path, only the epsilon grid differs.
  - the FINE 121-point grid's argmin (epsilon, smoothing, CV-MSE).
  - the practical-range (0.05-correlation-cutoff) conversion of each argmin
    epsilon, in metres, using the SAME conversion RBF_bootstrap.py's own
    comments and results/processed/*/length_scale_by_method.csv use
    (r = sqrt(-ln(0.05)) / epsilon).
  - how much worse (in relative CV-MSE) the production grid's pick is than
    the fine grid's pick at that level.

Run with: .venv/Scripts/python.exe -m src.experiments.diagnose_range_axis_epsilon_grid
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold

from src.experiments.base_case_conditioning import (
    TRUTH_SEED,
    VCOL,
    get_base_case_conditioning_data,
)
from src.experiments.range_axis import ALL_RANGE_VALUES
from src.experiments.rbf_bootstrap import (
    CV_FOLDS,
    CV_SEED,
    EPSILON_CUTOFF,
    EPSILON_GRID as PRODUCTION_EPSILON_GRID,
    RBF_KERNEL,
    SMOOTHING_GRID,
    _EPS_LENGTH_MAX_M,
    _EPS_LENGTH_MIN_M,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"

# --- Fine epsilon grid: SAME domain-geometry bounds as production, denser
# resolution (121 vs. 25 points -> 4.84x the resolution; the 2026-09-17
# density-axis diagnostic used 45 points for 3 levels, this uses more because
# 8 levels must be told apart, not 3).
N_EPSILON_FINE = 121
_EPSILON_LENGTHS_FINE_M = np.geomspace(_EPS_LENGTH_MAX_M, _EPS_LENGTH_MIN_M, N_EPSILON_FINE)
FINE_EPSILON_GRID = np.sqrt(-np.log(EPSILON_CUTOFF)) / _EPSILON_LENGTHS_FINE_M  # ascending

CUTOFF_FACTOR = float(np.sqrt(-np.log(EPSILON_CUTOFF)))  # 1.7308, gaussian RBFInterpolator kernel

SAMPLES_MATCH_ATOL = 1e-10


def cv_mse_grid_search_custom(X: np.ndarray, d: np.ndarray, epsilon_grid: np.ndarray):
    """Byte-for-byte the same procedure as rbf_bootstrap.cv_mse_grid_search
    (same KFold(CV_FOLDS, shuffle=True, random_state=CV_SEED), same
    SMOOTHING_GRID, same RBF_KERNEL) with the epsilon grid taken as an
    explicit argument instead of the module-level EPSILON_GRID constant, so
    the production and fine grids can be replayed against the identical fold
    splits within one run."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    rows = []
    for eps in epsilon_grid:
        for sm in SMOOTHING_GRID:
            fold_mses = []
            for train_idx, test_idx in kf.split(X):
                rbf = RBFInterpolator(
                    X[train_idx], d[train_idx], kernel=RBF_KERNEL, epsilon=eps, smoothing=sm
                )
                pred = rbf(X[test_idx])
                fold_mses.append(float(np.mean((pred - d[test_idx]) ** 2)))
            rows.append({"epsilon": float(eps), "smoothing": float(sm), "cv_mse": float(np.mean(fold_mses))})
    results_df = pd.DataFrame(rows)
    best_row = results_df.loc[results_df["cv_mse"].idxmin()]
    return float(best_row["epsilon"]), float(best_row["smoothing"]), float(best_row["cv_mse"]), results_df


def verify_against_pinned_run(rel_run_dir: str, r: float, samples_df: pd.DataFrame) -> None:
    """The regenerated conditioning samples for range ``r`` must match the
    samples.csv actually recorded in the PINNED rbf_bootstrap run for that
    level -- otherwise this diagnostic would silently be measuring a
    different dataset than the one the production grid decision concerns."""
    run_dir = _REPO_ROOT / rel_run_dir
    recorded = pd.read_csv(run_dir / "samples.csv")
    if len(recorded) != len(samples_df) or not np.allclose(
        recorded[["X", "Y", VCOL]].values,
        samples_df[["X", "Y", VCOL]].values,
        rtol=0.0,
        atol=SAMPLES_MATCH_ATOL,
    ):
        raise ValueError(
            f"range={r:g}m: regenerated conditioning samples do not match the samples.csv "
            f"recorded in the PINNED rbf_bootstrap run ({rel_run_dir}) -- this diagnostic "
            "would be measuring a different dataset than production."
        )


def main():
    source_runs = json.loads(
        (PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8")
    )

    print(
        f"Production grid: {len(PRODUCTION_EPSILON_GRID)} points, length bounds "
        f"[{_EPS_LENGTH_MIN_M:g}, {_EPS_LENGTH_MAX_M:g}] m.\n"
        f"Fine grid: {N_EPSILON_FINE} points, SAME length bounds.\n"
        f"CV_FOLDS={CV_FOLDS}, CV_SEED={CV_SEED}, SMOOTHING_GRID has {len(SMOOTHING_GRID)} "
        f"values, RBF_KERNEL={RBF_KERNEL!r}.\n"
    )

    rows = []
    t0 = time.time()
    for r in ALL_RANGE_VALUES:
        level = str(int(r))
        t_level = time.time()
        _, samples_df = get_base_case_conditioning_data(truth_seed=TRUTH_SEED, hmaj1=r, hmin1=r)
        verify_against_pinned_run(source_runs[level]["rbf_bootstrap"], r, samples_df)

        X = samples_df[["X", "Y"]].values
        d = samples_df[VCOL].values

        prod_eps, prod_sm, prod_mse, _ = cv_mse_grid_search_custom(X, d, PRODUCTION_EPSILON_GRID)
        fine_eps, fine_sm, fine_mse, _ = cv_mse_grid_search_custom(X, d, FINE_EPSILON_GRID)

        prod_range_m = CUTOFF_FACTOR / prod_eps
        fine_range_m = CUTOFF_FACTOR / fine_eps
        pct_worse = 100.0 * (prod_mse - fine_mse) / fine_mse

        rows.append(
            {
                "range_m": r,
                "n_samples_actual": len(samples_df),
                "production_best_epsilon": prod_eps,
                "production_practical_range_m": prod_range_m,
                "production_best_smoothing": prod_sm,
                "production_cv_mse": prod_mse,
                "fine_best_epsilon": fine_eps,
                "fine_practical_range_m": fine_range_m,
                "fine_best_smoothing": fine_sm,
                "fine_cv_mse": fine_mse,
                "production_pct_worse_than_fine": pct_worse,
            }
        )
        print(
            f"  range={r:g}m ({time.time() - t_level:.1f}s): "
            f"production -> eps={prod_eps:.6g} ({prod_range_m:.2f} m), cv_mse={prod_mse:.6f} | "
            f"fine -> eps={fine_eps:.6g} ({fine_range_m:.2f} m), cv_mse={fine_mse:.6f} | "
            f"production worse by {pct_worse:.2f}%"
        )

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "epsilon_grid_diagnostic.csv"
    df.to_csv(out_path, index=False)

    n_distinct_fine = df["fine_practical_range_m"].round(1).nunique()
    n_distinct_prod = df["production_practical_range_m"].round(1).nunique()
    print(
        f"\nDistinct practical-range picks (rounded to 0.1 m) across the 8 levels: "
        f"production={n_distinct_prod}/8, fine={n_distinct_fine}/8."
    )
    print(f"epsilon_grid_diagnostic.csv: {out_path}")
    print(f"Total: {time.time() - t0:.1f}s")
    return df


if __name__ == "__main__":
    main()
