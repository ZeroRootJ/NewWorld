"""Interactive dashboard for exploring `make_porosity_truth()` parameters.

Scope: a single facies, single porosity field only -- this is deliberately a
much smaller cousin of the reference dashboard at
``D:\\WORKSPACE\\Geo-Dashboard\\ressim\\geo_gen\\dashboard\\app.py`` (which
handles multi-layer facies/perm cosimulation, faults and 3D). Only the
UI/UX pattern (dark sidebar with parameter cards + Simulate button, light
main area, save/load params as YAML, JSON success/error envelope) and the
project layout convention (``.claude/launch.json`` + ``python -m ...``-style
script entrypoint) are reused from that project. All code here is written
fresh for this project's single-field scope -- nothing is copy-pasted.

The actual truth-model generation logic is NOT reimplemented here: it is
imported unchanged from ``src/truth_model.py`` (``make_porosity_truth``), per
the SPACEX project's coder.md instruction to reuse existing simulation code
rather than duplicate it. geostatspy usage in this file (import style,
``vario`` dict via ``GSLIB.make_variogram``, sill bookkeeping, subplot +
``GSLIB.pixelplt_st`` plotting pattern) follows
``docs/geostatspy_conventions.md``.

Run (from the SPACEX project root):
    D:\\WORKSPACE\\SPACEX\\.venv\\Scripts\\python.exe subprojects/generation-dashboard/app.py
    -> http://127.0.0.1:5050

or via ``.claude/launch.json`` (configuration name "generation-dashboard").
"""

from __future__ import annotations

import base64
import io
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless: this process only ever renders to PNG bytes

import matplotlib.pyplot as plt
import numpy as np
import yaml
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Make the SPACEX project root importable so we can reuse src/truth_model.py
# as-is (no logic copy-paste). This script is run directly (not as a package
# import, per the task's "폴더명 하이픈 그대로" instruction), so sys.path
# must be patched manually before importing `src`.
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parents[1]  # SPACEX/subprojects/generation-dashboard -> SPACEX
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.truth_model import make_porosity_truth  # noqa: E402  (path patched above)

import geostatspy.GSLIB as GSLIB  # noqa: E402

PRESETS_DIR = _APP_DIR / "presets"
PRESETS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Base case parameters -- named constants per .claude/agents/coder.md
# ("base case 파라미터는 named constant로 명시", "black-box 금지").
# These MUST match presets/base_case.yaml and the base case described in
# docs/experiment_context.md section 3 (isotropic, moderate range,
# zero-to-low nugget). They are used only as fallback defaults when a
# request omits a field -- every actual simulation records the full
# resolved parameter set back in the API response.
# ---------------------------------------------------------------------------
DEFAULT_NX = 50
DEFAULT_NY = 50
DEFAULT_CELL_SIZE = 20.0  # xsiz = ysiz; this dashboard only supports square cells
DEFAULT_MEAN = 15.0
DEFAULT_STDEV = 3.0
DEFAULT_NUG = 0.05
DEFAULT_HMAJ1 = 300.0
DEFAULT_HMIN1 = 300.0
DEFAULT_AZI1 = 0.0
DEFAULT_SEED = 101
DEFAULT_N_REAL = 5

# The dashboard exposes exactly nug/hmaj1/hmin1/azi1 as tunable (per task
# scope), so the variogram is always a single spherical structure.
# docs/geostatspy_conventions.md section 3 requires the sill to be fully
# modeled (nug + sum(ccN) == sill); src/truth_model.py runs sgsim in
# standard-normal space (mean 0, variance 1), so sill == 1.0 here and
# cc1 is derived as (SILL - nug), never a second free parameter.
NST = 1
IT1_SPHERICAL = 1
SILL = 1.0

# Interactive realization-count cap (task requirement: 1-9).
MIN_N_REAL = 1
MAX_N_REAL = 9

# Rendering: an interactive-preview dpi, deliberately NOT the 600 dpi used
# for paper figures elsewhere in this project (docs/geostatspy_conventions.md
# section 6) -- this image is a live preview in a browser tab, not a
# publication figure.
RENDER_DPI = 110
COLORMAP = "viridis"

app = Flask(__name__)


def _preset_path(raw_name: str) -> Path:
    """Sanitize a user-supplied preset name into a safe path under PRESETS_DIR."""
    name = (raw_name or "preset").strip()
    name = Path(name).name  # strip any directory separators -> no path traversal
    if not name:
        name = "preset"
    if not name.lower().endswith((".yaml", ".yml")):
        name += ".yaml"
    return PRESETS_DIR / name


def _parse_params(body: Dict[str, Any]) -> Dict[str, Any]:
    """Read + validate simulation parameters from a request body, filling in
    base case defaults (the DEFAULT_* constants above) for anything missing.
    """
    nx = int(body.get("nx", DEFAULT_NX))
    ny = int(body.get("ny", DEFAULT_NY))
    cell_size = float(body.get("cell_size", DEFAULT_CELL_SIZE))
    mean = float(body.get("mean", DEFAULT_MEAN))
    stdev = float(body.get("stdev", DEFAULT_STDEV))
    nug = float(body.get("nug", DEFAULT_NUG))
    hmaj1 = float(body.get("hmaj1", DEFAULT_HMAJ1))
    hmin1 = float(body.get("hmin1", DEFAULT_HMIN1))
    azi1 = float(body.get("azi1", DEFAULT_AZI1))
    seed = int(body.get("seed", DEFAULT_SEED))
    n_real = int(body.get("n_real", DEFAULT_N_REAL))

    if nx < 2 or ny < 2:
        raise ValueError("nx, ny must be >= 2")
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    if stdev <= 0:
        raise ValueError("stdev must be > 0")
    if not (0.0 <= nug <= 1.0):
        raise ValueError("nug must be within [0, 1] (fraction of sill)")
    if hmaj1 <= 0 or hmin1 <= 0:
        raise ValueError("hmaj1, hmin1 must be > 0")
    if not (MIN_N_REAL <= n_real <= MAX_N_REAL):
        raise ValueError(f"n_real must be between {MIN_N_REAL} and {MAX_N_REAL}")

    return dict(
        nx=nx,
        ny=ny,
        cell_size=cell_size,
        mean=mean,
        stdev=stdev,
        nug=nug,
        hmaj1=hmaj1,
        hmin1=hmin1,
        azi1=azi1,
        seed=seed,
        n_real=n_real,
    )


def _simulate_fields(params: Dict[str, Any]) -> List[np.ndarray]:
    """Call src.truth_model.make_porosity_truth() once per realization,
    seed, seed+1, ..., seed+n_real-1 -- reusing the existing function
    unchanged, per the task requirement (logic 복붙 금지)."""
    nx, ny = params["nx"], params["ny"]
    xsiz = ysiz = params["cell_size"]
    # Grid origin = centroid of the first cell (GSLIB convention), assuming
    # the domain's bounding-box minimum is 0 -- see
    # docs/geostatspy_conventions.md section 5.
    xmn = xsiz / 2.0
    ymn = ysiz / 2.0

    vario = GSLIB.make_variogram(
        nug=params["nug"],
        nst=NST,
        it1=IT1_SPHERICAL,
        cc1=SILL - params["nug"],
        azi1=params["azi1"],
        hmaj1=params["hmaj1"],
        hmin1=params["hmin1"],
    )

    seed = params["seed"]
    fields = []
    for i in range(params["n_real"]):
        field = make_porosity_truth(
            nx=nx,
            ny=ny,
            xsiz=xsiz,
            ysiz=ysiz,
            xmn=xmn,
            ymn=ymn,
            vario=vario,
            mean=params["mean"],
            stdev=params["stdev"],
            seed=seed + i,
        )
        fields.append(field)
    return fields


def _render_fields_png_b64(fields: List[np.ndarray], nx: int, ny: int, cell_size: float) -> str:
    """Render all realizations as one PNG (subplot grid) and return it as a
    base64 string. Follows the subplot + GSLIB.pixelplt_st pattern from
    docs/geostatspy_conventions.md section 6."""
    n_real = len(fields)
    ncols = int(np.ceil(np.sqrt(n_real)))
    nrows = int(np.ceil(n_real / ncols))

    xmin, xmax = 0.0, nx * cell_size
    ymin, ymax = 0.0, ny * cell_size
    # Shared color scale across realizations so panels are visually
    # comparable to each other (not just to their own min/max).
    vmin = min(float(np.min(f)) for f in fields)
    vmax = max(float(np.max(f)) for f in fields)

    fig = plt.figure(figsize=(4.2 * ncols, 4.0 * nrows))
    for i, field in enumerate(fields):
        plt.subplot(nrows, ncols, i + 1)
        GSLIB.pixelplt_st(
            field,
            xmin,
            xmax,
            ymin,
            ymax,
            cell_size,
            vmin,
            vmax,
            f"Realization {i + 1}",
            "X (m)",
            "Y (m)",
            "Porosity",
            COLORMAP,
        )
    plt.subplots_adjust(wspace=0.35, hspace=0.35)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=RENDER_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _qc_stats(fields: List[np.ndarray], seed: int) -> List[Dict[str, float]]:
    return [
        {
            "realization": i + 1,
            "seed": seed + i,
            "mean": float(np.mean(f)),
            "stdev": float(np.std(f)),
        }
        for i, f in enumerate(fields)
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        body = request.get_json(force=True) or {}
        params = _parse_params(body)
        fields = _simulate_fields(params)
        image_b64 = _render_fields_png_b64(fields, params["nx"], params["ny"], params["cell_size"])
        qc = _qc_stats(fields, params["seed"])
        return jsonify({"success": True, "params": params, "image_base64": image_b64, "qc": qc})
    except Exception as e:  # noqa: BLE001 -- surface any failure to the UI as JSON
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/presets", methods=["GET"])
def api_list_presets():
    try:
        names = sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))
        return jsonify({"success": True, "presets": names})
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/presets/save", methods=["POST"])
def api_save_preset():
    try:
        body = request.get_json(force=True) or {}
        params = _parse_params(body.get("params", {}))
        path = _preset_path(body.get("name"))
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(params, f, allow_unicode=True, sort_keys=False)
        return jsonify({"success": True, "name": path.stem})
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/presets/load", methods=["GET"])
def api_load_preset():
    try:
        path = _preset_path(request.args.get("name"))
        if not path.exists():
            return jsonify({"success": False, "error": f"preset not found: {path.name}"}), 404
        with open(path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        return jsonify({"success": True, "name": path.stem, "params": params})
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("Truth model generation dashboard -> http://127.0.0.1:5050")
    print("Presets directory:", PRESETS_DIR)
    app.run(host="127.0.0.1", port=5050, debug=False)
