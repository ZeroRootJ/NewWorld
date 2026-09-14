"""Shared exhaustive-grid coordinate helper.

Both the RBF+bootstrap and GP-MLE experiment scripts need to predict onto
every cell of the same 50x50 truth grid, and the resulting prediction array
must line up with the truth grid's row/column convention (row 0 = max-y
row, per geostatspy sgsim/imshow convention -- see
docs/geostatspy_conventions.md and src/truth_model.py). Implemented once
here so both scripts reshape predictions the same, verifiable way instead
of each re-deriving the row-flip.
"""

import numpy as np


def full_grid_coordinates(
    nx: int, ny: int, xmn: float, ymn: float, xsiz: float, ysiz: float
) -> np.ndarray:
    """Return the (ny*nx, 2) array of cell-centroid (X, Y) coordinates for
    every cell of an nx-by-ny grid, ordered so that reshaping a prediction
    vector as ``pred.reshape(ny, nx)`` matches the truth grid's row 0 =
    max-y convention (i.e. row r, column c of the reshaped array corresponds
    to the same cell as ``truth[r, c]``).

    Parameters
    ----------
    nx, ny : number of grid cells in x and y
    xmn, ymn : grid origin, i.e. the centroid of the first cell (GSLIB
        convention)
    xsiz, ysiz : cell size in x and y
    """
    xs = xmn + np.arange(nx) * xsiz
    # row r (0-indexed from the top, max-y row first) corresponds to
    # y = ymn + (ny - 1 - r) * ysiz
    ys_row_order = ymn + (ny - 1 - np.arange(ny)) * ysiz

    xx, yy = np.meshgrid(xs, ys_row_order)  # both shape (ny, nx)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return coords
