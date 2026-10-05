"""Experimental semivariogram of an exhaustive 2D grid along the x and y grid
axes (numpy only, so it imports under both the project .venv (py3.8) and the
separate ResMill .venv-obm (py3.11)).

WHY NOT ``geostats.gam``: geostatspy 0.0.79's ``gam`` has two 1-based ->
0-based porting artefacts visible in its source:
  * the head-row check is ``if 1 <= iy1 < ny`` (Fortran 1-based bound kept),
    so every pair whose head lies in row 0 is silently dropped;
  * the reported lag is ``ixd*xsiz*il`` with ``il`` starting at 0, so the
    first value (computed at a 1-cell offset) is labelled lag 0.
This mirrors the project's existing precedent of not calling geostatspy
routines with known porting bugs (``src/nscore.py``). Rather than wrap
``gam`` with corrections, this module computes the textbook grid
semivariogram directly so the SGS and the ALLUVSIM-based families are
measured by exactly the same, easily auditable code path.

Array convention: ``field`` is (ny, nx) in the project convention
(row 0 = max-y row; see src/truth_model.py). The x direction is therefore
along axis 1 (columns) and the y direction along axis 0 (rows). The
semivariogram is symmetric in h, so the row flip does not affect gamma(h).

    gamma(h) = 0.5 * mean over all pairs (z(u+h) - z(u))^2
"""

from typing import Dict

import numpy as np


def grid_semivariogram(field: np.ndarray, direction: str, nlag: int, cell_size: float) -> Dict[str, np.ndarray]:
    """Semivariogram of ``field`` along ``direction`` ('x' or 'y') for lags
    1..nlag cells.

    Returns a dict with ``lag_cells`` (int), ``lag_m`` (lag_cells *
    cell_size), ``gamma``, ``gamma_std`` (gamma / population variance of the
    whole field, ddof=0) and ``npairs``.
    """
    z = np.asarray(field, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("field must be 2D (ny, nx), got shape %s" % (z.shape,))
    if direction == "x":
        axis = 1
    elif direction == "y":
        axis = 0
    else:
        raise ValueError("direction must be 'x' or 'y', got %r" % (direction,))
    n_along = z.shape[axis]
    if not (1 <= nlag < n_along):
        raise ValueError("nlag must be in [1, %d), got %d" % (n_along, nlag))

    var = float(np.var(z))
    lags = np.arange(1, nlag + 1)
    gamma = np.empty(nlag)
    npairs = np.empty(nlag, dtype=np.int64)
    for i, h in enumerate(lags):
        if axis == 1:
            d = z[:, h:] - z[:, :-h]
        else:
            d = z[h:, :] - z[:-h, :]
        gamma[i] = 0.5 * float(np.mean(d * d))
        npairs[i] = d.size
    gamma_std = gamma / var if var > 0 else np.full(nlag, np.nan)
    return {
        "lag_cells": lags,
        "lag_m": lags * float(cell_size),
        "gamma": gamma,
        "gamma_std": gamma_std,
        "npairs": npairs,
    }
