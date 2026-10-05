"""Helpers shared by generate_ground_truth.py and run_qc.py.

Deliberately imports only numpy + PyYAML (no resmill), so the orientation
helper and its test also run under the project .venv (py3.8).

ARRAY CONVENTIONS (load-bearing, used throughout obm_benchmark):

* ResMill 3D arrays (``ChannelLayer.poro_mat``, ``ChannelLayer.facies``) are
  shape (nx, ny, nz), indexed [ix, iy, iz]; x increases with ix, y increases
  with iy (ResMill builds its grid with ``np.meshgrid(x, y, indexing='ij')``).
  The saved ``*_3d.npy`` files keep this NATIVE ResMill layout unchanged.
* The project's 2D convention (src/truth_model.py, src/grid_utils.py) is
  shape (ny, nx) with row 0 = max-y row and column c = x index c. All saved
  ``*_2d*.npy`` files use THIS convention, obtained by
  ``slice_to_project_2d`` = ``np.flipud(arr3d[:, :, k].T)``.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

OBM_ROOT = Path(__file__).resolve().parent.parent          # obm_benchmark/
REPO_ROOT = OBM_ROOT.parent
DEFAULT_CONFIG = OBM_ROOT / "config" / "cb_jigsaw.yaml"

# Facies with code >= NET_FACIES_MIN count as sand / net (CS, LV, LA, CH).
NET_FACIES_MIN = 1


def load_config(path=DEFAULT_CONFIG) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slice_index_from_rule(cfg: Dict[str, Any]) -> int:
    """The single fixed slice rule. Only ``nz // 2`` is implemented; any
    other rule string in the config raises instead of being silently
    reinterpreted."""
    rule = cfg["slice"]["rule"]
    if rule.replace(" ", "") != "nz//2":
        raise NotImplementedError("only slice rule 'nz // 2' is implemented, got %r" % rule)
    return int(cfg["grid"]["nz"]) // 2


def slice_to_project_2d(arr3d: np.ndarray, k: int) -> np.ndarray:
    """ResMill (nx, ny, nz)[ix, iy, iz] -> project 2D (ny, nx), row 0 = max-y.

    ``arr3d[:, :, k]`` is (nx, ny); ``.T`` makes it (ny, nx) with row = iy
    ascending (min-y first); ``np.flipud`` puts max-y on row 0.
    """
    return np.ascontiguousarray(np.flipud(arr3d[:, :, k].T))


def check_orientation_helper() -> None:
    """Self-test of ``slice_to_project_2d`` on a synthetic array whose value
    encodes its own (ix, iy, iz). Raises AssertionError on failure."""
    nx, ny, nz = 4, 3, 2
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    a = ix + 100 * iy + 10000 * iz            # a[ix, iy, iz] encodes indices
    out = slice_to_project_2d(a, 1)
    assert out.shape == (ny, nx), out.shape
    for r in range(ny):
        for c in range(nx):
            # column c <-> ix = c ; row r <-> iy = ny - 1 - r (row 0 = max y)
            assert out[r, c] == c + 100 * (ny - 1 - r) + 10000 * 1, (r, c, out[r, c])


def affine_rescale(x: np.ndarray, target_mean: float, target_stdev: float):
    """truth = a * x + b with a = target_stdev / std(x), b = target_mean - a*mean(x)
    (same as GSLIB.affine: np.average / np.std ddof=0). Computed in float64.
    Returns (truth, a, b)."""
    x64 = np.asarray(x, dtype=np.float64)
    m = float(np.mean(x64))
    s = float(np.std(x64))
    a = float(target_stdev) / s
    b = float(target_mean) - a * m
    return a * x64 + b, a, b


def array_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem_for(preset: str, seed: int) -> str:
    return "%s_seed_%03d" % (preset.lower(), seed)


# Per-seed file suffixes (file name = <stem><suffix>).
FILE_SUFFIXES = {
    "porosity_3d": "_porosity_3d.npy",       # float32 (nx, ny, nz) native ResMill layout, fraction
    "facies_3d": "_facies_3d.npy",           # int8    (nx, ny, nz) native ResMill layout
    "porosity_2d_frac": "_porosity_2d_frac.npy",  # float32 (ny, nx) project convention, fraction
    "facies_2d": "_facies_2d.npy",           # int8    (ny, nx) project convention
    "truth_2d": "_truth_2d.npy",             # float64 (ny, nx) project convention, rescaled %
}
