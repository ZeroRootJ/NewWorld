"""One-off convergence check (NOT part of the permanent diagnostic suite,
underscore-prefixed so it is not mistaken for one): does going PAST the
N_EPSILON_FINE=121 grid used in diagnose_range_axis_epsilon_grid.py change
anything for the range=500/600/700/800 m levels, which still shared/tied at
121 points (500->313.09 m, 600 & 700 tied at 327.22 m, 800->342.00 m)?

Doubles the resolution to 241 points over the SAME domain-geometry bounds
(10-2000 m) and re-runs the identical CV procedure for just these 4 levels,
to see whether the 600/700 tie is a genuine near-degeneracy (stable under
further refinement) or itself still a resolution artifact.

Run with: .venv/Scripts/python.exe -m src.experiments._diagnose_range_axis_epsilon_convergence_check
"""

import time

import numpy as np

from src.experiments.base_case_conditioning import TRUTH_SEED, VCOL, get_base_case_conditioning_data
from src.experiments.diagnose_range_axis_epsilon_grid import (
    CUTOFF_FACTOR,
    cv_mse_grid_search_custom,
)
from src.experiments.rbf_bootstrap import EPSILON_CUTOFF, _EPS_LENGTH_MAX_M, _EPS_LENGTH_MIN_M

N_EPSILON_241 = 241
_LENGTHS_241 = np.geomspace(_EPS_LENGTH_MAX_M, _EPS_LENGTH_MIN_M, N_EPSILON_241)
GRID_241 = np.sqrt(-np.log(EPSILON_CUTOFF)) / _LENGTHS_241

LEVELS = [500.0, 600.0, 700.0, 800.0]


def main():
    for r in LEVELS:
        t0 = time.time()
        _, samples_df = get_base_case_conditioning_data(truth_seed=TRUTH_SEED, hmaj1=r, hmin1=r)
        X = samples_df[["X", "Y"]].values
        d = samples_df[VCOL].values
        eps, sm, cv_mse, _ = cv_mse_grid_search_custom(X, d, GRID_241)
        practical_range = CUTOFF_FACTOR / eps
        print(
            f"range={r:g}m ({time.time()-t0:.1f}s): 241-pt fine -> eps={eps:.6g} "
            f"({practical_range:.2f} m), smoothing={sm:g}, cv_mse={cv_mse:.6f}"
        )


if __name__ == "__main__":
    main()
