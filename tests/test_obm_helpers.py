"""Tests for the resmill-free helpers of obm_benchmark and src/grid_variogram.py.
Runs under the project .venv (pytest) and also under .venv-obm."""

import numpy as np
import pytest

from obm_benchmark.scripts._common import (
    affine_rescale,
    check_orientation_helper,
    slice_to_project_2d,
)
from src.grid_variogram import grid_semivariogram


def test_orientation_selfcheck():
    check_orientation_helper()


def test_orientation_explicit_corners():
    # ResMill layout [ix, iy, iz]; mark the cell at (max x, max y) and (min x, min y)
    nx, ny, nz = 5, 4, 3
    a = np.zeros((nx, ny, nz))
    a[nx - 1, ny - 1, 1] = 1.0     # NE corner
    a[0, 0, 1] = 2.0               # SW corner
    out = slice_to_project_2d(a, 1)
    assert out.shape == (ny, nx)
    assert out[0, nx - 1] == 1.0    # row 0 = max-y, last column = max-x
    assert out[ny - 1, 0] == 2.0    # last row = min-y, column 0 = min-x


def test_affine_rescale_matches_gslib_formula():
    x = np.random.RandomState(1).gamma(2.0, 0.05, size=(50, 50)).astype(np.float32)
    t, a, b = affine_rescale(x, 15.0, 3.0)
    assert np.mean(t) == pytest.approx(15.0, abs=1e-10)
    assert np.std(t) == pytest.approx(3.0, abs=1e-10)
    x64 = x.astype(np.float64)
    np.testing.assert_allclose(t, (3.0 / np.std(x64)) * (x64 - np.mean(x64)) + 15.0, atol=1e-12)
    np.testing.assert_allclose(t, a * x64 + b, atol=1e-12)


def test_grid_semivariogram_directions_and_values():
    # field varies only along x (columns): y-variogram must be 0
    z = np.tile(np.arange(10.0), (6, 1))           # shape (ny=6, nx=10)
    vx = grid_semivariogram(z, "x", 3, 20.0)
    vy = grid_semivariogram(z, "y", 3, 20.0)
    np.testing.assert_allclose(vy["gamma"], 0.0)
    np.testing.assert_allclose(vx["gamma"], 0.5 * np.array([1.0, 4.0, 9.0]))
    np.testing.assert_array_equal(vx["npairs"], [6 * 9, 6 * 8, 6 * 7])
    np.testing.assert_allclose(vx["lag_m"], [20.0, 40.0, 60.0])
    np.testing.assert_allclose(vx["gamma_std"], vx["gamma"] / np.var(z))
