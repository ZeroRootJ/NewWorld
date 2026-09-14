"""Regular interior sampling from an exhaustive truth-model grid.

Single-feature simplification of the reference notebook's
``regular_sample_MV`` (make_nonlinear_MV_spatial_data_v13.ipynb), which
handles up to 4 co-located arrays at once. This project only ever samples
one feature (porosity) at a time, so that generality is dropped rather than
copy-pasted.
"""

import numpy as np
import pandas as pd


def _snap_to_grid_index(loc: float, cmn: float, csiz: float, n: int) -> int:
    """Snap a coordinate to its nearest grid cell index using GSLIB-standard
    rounding (``+0.5`` before truncation), clamped to ``[0, n - 1]``.

    This is deliberately *not* ``geostatspy.geostats.getindex``, which
    computes ``ic = int((loc - cmn) / csiz)`` with no rounding, i.e. it
    truncates toward the lower-index neighbor instead of rounding to the
    nearest cell centroid. Both ``regular_interior_samples`` and
    ``random_interior_samples`` use this helper so the correct rounding is
    implemented in exactly one place.
    """
    ic = int((loc - cmn) / csiz + 0.5)
    return min(max(ic, 0), n - 1)


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

    Notes
    -----
    Uses ``_snap_to_grid_index`` (GSLIB-standard ``+0.5`` rounding) rather
    than ``geostatspy.geostats.getindex`` to map each sample location to its
    nearest grid cell -- the same rounding logic used by
    ``random_interior_samples`` below, factored into one shared helper so it
    is implemented (and can be fixed) in exactly one place.

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
            iix = _snap_to_grid_index(xv, xmn, xsiz, nx)
            iiy = _snap_to_grid_index(yv, ymn, ysiz, ny)
            # sgsim fills row 0 = max-y row, so the row corresponding to a
            # given (0-based, increasing-with-y) index iiy is ny - iiy - 1
            # (same flip used in the reference notebook's regular_sample_MV).
            v_list.append(value_grid[ny - iiy - 1, iix])
            x_list.append(xv)
            y_list.append(yv)

    return pd.DataFrame({"X": x_list, "Y": y_list, colname: v_list})


def random_interior_samples(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_samples: int,
    margin_frac: float,
    value_grid: np.ndarray,
    colname: str,
    seed: int,
) -> pd.DataFrame:
    """Draw n_samples uniform-random sample locations strictly inside the
    domain (same margin convention as ``regular_interior_samples``), snap
    each to its nearest grid cell via ``_snap_to_grid_index`` (same helper
    ``regular_interior_samples`` uses, so both functions apply identical
    GSLIB-standard ``+0.5`` rounding rather than the truncating behavior of
    ``geostatspy.geostats.getindex``), and drop duplicate cell mappings.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : domain bounding box (not the grid xmn/ymn
        origin -- see docs/geostatspy_conventions.md section 6)
    n_samples : number of sample locations to attempt to draw. Because
        duplicate cell mappings are dropped, the returned DataFrame may have
        fewer than n_samples rows -- callers must read the actual row count
        rather than assume it, and record it in the run manifest.
    margin_frac : fraction of the domain extent left empty on *each* side
        along both axes (same convention as ``regular_interior_samples``)
    value_grid : exhaustive truth-model array, shape (ny, nx), row 0 = max-y
        row (geostatspy sgsim/imshow convention)
    colname : name to give the sampled value column in the returned DataFrame
    seed : random seed for the uniform draw (named constant at the call
        site, per project convention -- recorded in the manifest, not
        hardcoded here)

    Returns
    -------
    pd.DataFrame with columns ["X", "Y", colname], one row per unique grid
    cell sampled. X, Y are the snapped cell-centroid coordinates (not the
    raw uniform draw), so that duplicate-cell detection and downstream
    "identical sample locations across methods" comparisons are exact.
    """
    ny, nx = value_grid.shape
    xsiz = (xmax - xmin) / nx
    ysiz = (ymax - ymin) / ny
    xmn = xmin + 0.5 * xsiz
    ymn = ymin + 0.5 * ysiz

    x_margin = margin_frac * (xmax - xmin)
    y_margin = margin_frac * (ymax - ymin)

    rng = np.random.RandomState(seed)
    xs = rng.uniform(xmin + x_margin, xmax - x_margin, n_samples)
    ys = rng.uniform(ymin + y_margin, ymax - y_margin, n_samples)

    seen_cells = set()
    x_list, y_list, v_list = [], [], []
    for xv, yv in zip(xs, ys):
        ix = _snap_to_grid_index(xv, xmn, xsiz, nx)
        iy = _snap_to_grid_index(yv, ymn, ysiz, ny)

        cell = (ix, iy)
        if cell in seen_cells:
            continue
        seen_cells.add(cell)

        x_snap = xmn + ix * xsiz
        y_snap = ymn + iy * ysiz
        # sgsim fills row 0 = max-y row, so the row corresponding to a
        # given (0-based, increasing-with-y) index iy is ny - iy - 1 (same
        # flip used in regular_interior_samples / the reference notebook).
        v_list.append(value_grid[ny - iy - 1, ix])
        x_list.append(x_snap)
        y_list.append(y_snap)

    return pd.DataFrame({"X": x_list, "Y": y_list, colname: v_list})
