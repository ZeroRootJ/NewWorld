"""Regular interior sampling from an exhaustive truth-model grid.

Single-feature simplification of the reference notebook's
``regular_sample_MV`` (make_nonlinear_MV_spatial_data_v13.ipynb), which
handles up to 4 co-located arrays at once. This project only ever samples
one feature (porosity) at a time, so that generality is dropped rather than
copy-pasted.
"""

import numpy as np
import pandas as pd

import geostatspy.geostats as geostats


def regular_interior_samples(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_per_axis: int,
    margin_frac: float,
    value_grid: np.ndarray,
    colname: str,
) -> pd.DataFrame:
    """Draw an n_per_axis x n_per_axis regular grid of samples strictly
    inside the domain, leaving a margin so the sample convex hull does not
    touch the domain boundary.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : domain bounding box (not the grid xmn/ymn
        origin -- see docs/geostatspy_conventions.md section 6 on not
        confusing the two)
    n_per_axis : number of sample locations along each axis (total samples
        = n_per_axis ** 2)
    margin_frac : fraction of the domain extent left empty on *each* side
        along both axes, e.g. 0.05 keeps samples within
        [xmin + 0.05*(xmax-xmin), xmax - 0.05*(xmax-xmin)] (and similarly for y)
    value_grid : exhaustive truth-model array, shape (ny, nx), row 0 = max-y
        row (geostatspy sgsim/imshow convention)
    colname : name to give the sampled value column in the returned DataFrame

    Returns
    -------
    pd.DataFrame with columns ["X", "Y", colname]
    """
    ny, nx = value_grid.shape
    xsiz = (xmax - xmin) / nx
    ysiz = (ymax - ymin) / ny
    xmn = xmin + 0.5 * xsiz
    ymn = ymin + 0.5 * ysiz

    x_margin = margin_frac * (xmax - xmin)
    y_margin = margin_frac * (ymax - ymin)
    xs = np.linspace(xmin + x_margin, xmax - x_margin, n_per_axis)
    ys = np.linspace(ymin + y_margin, ymax - y_margin, n_per_axis)
    xx, yy = np.meshgrid(xs, ys)

    x_list, y_list, v_list = [], [], []
    for iy in range(n_per_axis):
        for ix in range(n_per_axis):
            xv = float(xx[iy, ix])
            yv = float(yy[iy, ix])
            iix = geostats.getindex(nx, xmn, xsiz, xv)
            iiy = geostats.getindex(ny, ymn, ysiz, yv)
            # sgsim fills row 0 = max-y row, so the row corresponding to a
            # given (0-based, increasing-with-y) index iiy is ny - iiy - 1
            # (same flip used in the reference notebook's regular_sample_MV).
            v_list.append(value_grid[ny - iiy - 1, iix])
            x_list.append(xv)
            y_list.append(yv)

    return pd.DataFrame({"X": x_list, "Y": y_list, colname: v_list})
