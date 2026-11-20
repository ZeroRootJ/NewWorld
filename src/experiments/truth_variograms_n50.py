"""Experimental variograms of the 50 ground-truth realizations per level.

User request 2026-11-20: check whether the scatter of the GP-MLE fits in
Fig. 4 (gp_realization_n50) comes from the realizations themselves. For every
(axis, level, truth_seed) of gp_realization_n50 this regenerates the SAME
truth and samples (get_base_case_conditioning_data, sample seed 20, same call
as fit_one in gp_realization_check_range_axis / _nugget_axis) and computes

  field   isotropic experimental variogram of the full 50 x 50 porosity map
          (all cell pairs), 20 m lag bins
  sample  the same from the conditioning samples the GP sees, 50 m lag bins

Semivariance is divided by POR_STDEV**2 (= 9, the model sill of the porosity
map), so the true model is nugget + (1 - nugget) * spherical(h; range).

Output (tidy): results/processed/truth_variograms_n50/variograms.csv with
columns axis, level, truth_seed, source, lag_m, gamma_norm, npairs; plus
provenance.json. No raw run: the truths are deterministic regenerations.

Run: .venv/Scripts/python.exe -m src.experiments.truth_variograms_n50 [--workers N]
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import gp_realization_check_range_axis as rng_mod
from src.experiments.base_case import NUG as BASE_NUG
from src.experiments.base_case import NX, NY, POR_STDEV, XMN, XSIZ, YMN, YSIZ
from src.experiments.base_case_conditioning import N_SAMPLES, SAMPLE_SEED, get_base_case_conditioning_data
from src.experiments.gp_realization_n50 import TRUTH_SEEDS_N50
from src.experiments.nugget_axis import ALL_NUGGET_VALUES, AXIS_HMAJ1
from src.experiments.range_axis import ALL_RANGE_VALUES

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "processed" / "truth_variograms_n50"
FIELD_BIN_M = 20.0
SAMPLE_BIN_M = 50.0
MAX_LAG_M = 700.0
SILL = POR_STDEV ** 2


def _bins(xy, width):
    i, j = np.triu_indices(len(xy), k=1)
    d = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
    keep = d < MAX_LAG_M
    return i[keep], j[keep], (d[keep] // width).astype(int)


_GX, _GY = np.meshgrid(XMN + XSIZ * np.arange(NX), YMN + YSIZ * np.arange(NY))
_FIELD = _bins(np.column_stack([_GX.ravel(), _GY.ravel()]), FIELD_BIN_M)


def _vario(z, i, j, b, width):
    nb = int(b.max()) + 1
    n = np.bincount(b, minlength=nb)
    g = np.bincount(b, weights=(z[i] - z[j]) ** 2, minlength=nb) / (2.0 * np.maximum(n, 1)) / SILL
    lag = (np.arange(nb) + 0.5) * width
    ok = n > 0
    return lag[ok], g[ok], n[ok]


def one(task):
    axis, level, seed = task
    hmaj = float(level) if axis == "range" else float(AXIS_HMAJ1)
    nug = float(BASE_NUG) if axis == "range" else float(level)
    truth, s = get_base_case_conditioning_data(
        sample_seed=SAMPLE_SEED, truth_seed=int(seed), hmaj1=hmaj, hmin1=hmaj, n_samples=N_SAMPLES, nug=nug)
    z = np.asarray(truth, dtype=float)
    if z.shape != (NY, NX):
        raise ValueError(f"unexpected truth shape {z.shape}")
    rows = []
    # GSLIB 2D arrays are stored top row first (largest y); flip so row k is y = YMN + k*YSIZ.
    zf = z[::-1, :].ravel()
    for lag, g, n in zip(*_vario(zf, *_FIELD, FIELD_BIN_M)):
        rows.append((axis, level, seed, "field", lag, g, int(n)))
    xy = s[["X", "Y"]].values
    for lag, g, n in zip(*_vario(s[rng_mod.VCOL].values.astype(float), *_bins(xy, SAMPLE_BIN_M), SAMPLE_BIN_M)):
        rows.append((axis, level, seed, "sample", lag, g, int(n)))
    return rows


def main(workers):
    t0 = time.time()
    tasks = [("range", float(r), s) for r in ALL_RANGE_VALUES for s in TRUTH_SEEDS_N50]
    tasks += [("nugget", float(n), s) for n in ALL_NUGGET_VALUES for s in TRUTH_SEEDS_N50]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = [r for rows in ex.map(one, tasks, chunksize=4) for r in rows]
    df = pd.DataFrame(out, columns=["axis", "level", "truth_seed", "source", "lag_m", "gamma_norm", "npairs"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "variograms.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (OUT_DIR / "provenance.json").write_text(json.dumps({
        "code_entrypoint": "src/experiments/truth_variograms_n50.py", "git_commit": commit,
        "truth_seeds": list(TRUTH_SEEDS_N50), "sample_seed": SAMPLE_SEED, "n_samples_requested": N_SAMPLES,
        "range_levels_m": list(map(float, ALL_RANGE_VALUES)), "range_axis_nugget": float(BASE_NUG),
        "nugget_levels": list(map(float, ALL_NUGGET_VALUES)), "nugget_axis_range_m": float(AXIS_HMAJ1),
        "field_bin_m": FIELD_BIN_M, "sample_bin_m": SAMPLE_BIN_M, "max_lag_m": MAX_LAG_M,
        "gamma_normalized_by": SILL, "n_tasks": len(tasks), "wall_s": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f"{len(tasks)} truths, {len(df)} rows, {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    main(ap.parse_args().workers)
