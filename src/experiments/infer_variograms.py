"""Sample-inferred variograms for every cell of the multi-truth run.

First step toward re-running kriging/SGS with a variogram INFERRED from the
conditioning samples instead of the true (oracle) one. This script only
produces data -- it does NOT re-run kriging/SGS and does not touch GP/RBF.
The fitted values are meant to be reviewed/edited by the user afterwards.

Inputs (read-only)
------------------
- results/processed/multi_truth_axes/evaluation_cells.csv : the 641 cells,
  each with ``run_dir_kriging_b`` (the kriging run that conditioned on the
  cell's samples).
- <multi-truth source run>/run_config.json : true hmaj1/hmin1/nug per cell,
  joined on experiment + config_key (same join as
  results/processed/paper_storyboard/make_storyboard_figures.py).
- <run_dir_kriging_b>/samples.csv : EXACTLY the conditioning samples kriging
  used (X, Y, Por, NPor). SGS conditions on the same rows (Por, itrans=1).

Per cell
--------
1. Normal-score transform of ``Por`` through ``src.nscore.nscore`` (never
   ``geostats.nscore``), exactly as src/experiments/kriging.py does; the
   result is asserted equal to the ``NPor`` column kriging saved.
2. Omnidirectional experimental variogram with ``geostats.gamv``
   (atol=90, isill=1). Lag rule (same rule for every cell, adapts to density):
     lag_dist = mean nearest-neighbour distance of the cell's samples
     lag_tol  = lag_dist / 2
     nlag     = floor(MAX_LAG_M / lag_dist) + 1  (bin centres 0, L, ..., <= MAX_LAG_M)
     MAX_LAG_M = 0.5 * domain extent (= 500 m)
   gamv bin layout (verified from geostatspy 0.0.79 source): returned index 0
   holds the zero-distance self-pairs, index k>=1 is centred on (k-1)*lag_dist
   with +/- lag_tol, index nlag+1 is always empty. Index 0 is dropped; empty
   bins are dropped. gamv loops over ORDERED pairs (i,j) and (j,i) and, when
   omnidirectional, adds each pair once more, so its ``npp`` is 4x the number
   of distinct point pairs; both are saved (``npairs_gamv`` and
   ``npairs`` = npp/4).
3. Fit of a single spherical structure with the total sill FIXED at 1.0
   (normal-score, matching build_vario's cc1 = 1 - nug convention):
     gamma(h) = nug + (1 - nug) * Sph(h; range)
   Bounds: nug in [0, NUG_UPPER] (NUG_UPPER = 0.999, i.e. [0, 1)),
           range in [lag_dist, DOMAIN_EXTENT_M].
   Lags with fewer than MIN_PAIRS_FOR_FIT distinct pairs are excluded.
   Weighted least squares with weights npairs / h^2 (gstat fit.method = 7,
   gstat's default), solved with scipy.optimize.least_squares (trf, bounded)
   from a fixed multi-start grid; the lowest-cost solution is kept, so the
   fit is deterministic. No random numbers are used anywhere.

Outputs: results/processed/variogram_inference/
  experimental_variograms.csv, fitted_variograms.csv, manifest.json
"""

import json
import time
from pathlib import Path

import geostatspy.geostats as geostats
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from src.experiments.base_case import NX, NY, XSIZ, YSIZ
from src.io import save_result
from src.nscore import nscore as nscore_corrected

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXPERIMENT = "variogram_inference"
CODE_ENTRYPOINT = "src/experiments/infer_variograms.py"

CELLS_CSV = _REPO_ROOT / "results/processed/multi_truth_axes/evaluation_cells.csv"
SOURCE_RUN = _REPO_ROOT / "results/raw/multi_truth_axes/20261116T222609599808Z"
OUT_DIR = _REPO_ROOT / "results/processed" / EXPERIMENT

XCOL, YCOL, VCOL, NSCOL = "X", "Y", "Por", "NPor"

# --- Domain --------------------------------------------------------------
DOMAIN_EXTENT_M = float(min(NX * XSIZ, NY * YSIZ))  # 1000 m (square domain)

# --- gamv parameters (named, no library defaults) -------------------------
GAMV_TMIN, GAMV_TMAX = -9999.0, 9999.0  # never trim normal-score data
AZI = 0.0  # irrelevant when omnidirectional
ATOL = 90.0  # >= 90 -> omnidirectional in gamv
BANDH = 9999.0  # no bandwidth restriction (omnidirectional)
ISILL = 1  # standardize gamma by the sample variance (sill -> 1)
MAX_LAG_FRACTION_OF_EXTENT = 0.5
MAX_LAG_M = MAX_LAG_FRACTION_OF_EXTENT * DOMAIN_EXTENT_M
LAG_TOL_FRACTION = 0.5
LAG_RULE = (
    "lag_dist = mean nearest-neighbour distance of the cell's samples; "
    "lag_tol = 0.5 * lag_dist; nlag = floor(MAX_LAG_M / lag_dist) + 1 "
    "(gamv bin centres 0, lag_dist, ..., <= MAX_LAG_M); "
    f"MAX_LAG_M = {MAX_LAG_FRACTION_OF_EXTENT} * domain extent = {MAX_LAG_M} m; "
    "omnidirectional (atol=90, bandh=9999, azi=0); isill=1"
)

# --- Fit parameters --------------------------------------------------------
SILL = 1.0
NUG_LOWER, NUG_UPPER = 0.0, 0.999
MIN_PAIRS_FOR_FIT = 30  # distinct point pairs (= gamv npp / 4)
MIN_LAGS_OK = 3  # >= this many usable lags -> status "ok"; 2 -> "few_lags"
MULTISTART_NUGGETS = (0.0, 0.2, 0.4, 0.6, 0.8)
MULTISTART_N_RANGES = 6  # geomspace(lag_dist, DOMAIN_EXTENT_M, n)
BOUND_RTOL = 1e-3  # a parameter within this relative distance of a bound is flagged
WEIGHT_RULE = "w_j = npairs_j / h_j^2 (gstat fit.method = 7, the gstat default)"
FIT_RULE = (
    "single spherical structure, total sill fixed 1.0: "
    "gamma(h) = nug + (1 - nug) * Sph(h; range); "
    f"nug in [{NUG_LOWER}, {NUG_UPPER}], range in [lag_dist, {DOMAIN_EXTENT_M}] m; "
    f"lags with < {MIN_PAIRS_FOR_FIT} distinct pairs excluded; "
    "weighted least squares (scipy.optimize.least_squares, method='trf') "
    f"with {WEIGHT_RULE}; deterministic multi-start grid nug0 in "
    f"{list(MULTISTART_NUGGETS)} x range0 in geomspace(lag_dist, {DOMAIN_EXTENT_M}, "
    f"{MULTISTART_N_RANGES}); lowest cost kept. fit_rmse = unweighted RMSE over used lags."
)


def spherical_sill1(h, nug, a):
    """Spherical model with total sill 1.0 (gamma(0+) = nug)."""
    h = np.asarray(h, dtype=float)
    r = np.minimum(h / a, 1.0)
    return nug + (1.0 - nug) * (1.5 * r - 0.5 * r ** 3)


def mean_nn_distance(x, y):
    tree = cKDTree(np.column_stack([x, y]))
    d, _ = tree.query(np.column_stack([x, y]), k=2)
    return float(np.mean(d[:, 1]))


def experimental_variogram(df_ns):
    """Omnidirectional gamv on NPor with the density-adaptive lag rule."""
    lag_dist = mean_nn_distance(df_ns[XCOL].values, df_ns[YCOL].values)
    lag_tol = LAG_TOL_FRACTION * lag_dist
    nlag = int(np.floor(MAX_LAG_M / lag_dist)) + 1
    lag, gamma, npp = geostats.gamv(
        df_ns, XCOL, YCOL, NSCOL, GAMV_TMIN, GAMV_TMAX,
        lag_dist, lag_tol, nlag, AZI, ATOL, BANDH, ISILL,
    )
    lag, gamma, npp = np.asarray(lag), np.asarray(gamma), np.asarray(npp)
    if npp[nlag + 1] != 0:
        raise RuntimeError("gamv's last bin (index nlag+1) is expected to be empty.")
    if npp[0] != 2 * len(df_ns):
        # self pairs: i==j counted once per i, doubled by the omni branch
        raise RuntimeError(f"unexpected gamv self-pair count {npp[0]} for nd={len(df_ns)}")
    idx = np.arange(1, nlag + 1)
    keep = idx[npp[idx] > 0]
    ev = pd.DataFrame({
        "lag_index": keep,
        "lag_center_m": (keep - 1) * lag_dist,
        "lag_m": lag[keep],
        "gamma": gamma[keep],
        "npairs_gamv": npp[keep].astype(int),
    })
    if (ev["npairs_gamv"] % 4 != 0).any():
        raise RuntimeError("gamv omnidirectional npp is not a multiple of 4.")
    ev["npairs"] = ev["npairs_gamv"] // 4
    return ev, lag_dist, lag_tol, nlag


def fit_spherical(ev, lag_dist):
    used = ev[(ev["npairs"] >= MIN_PAIRS_FOR_FIT) & (ev["lag_m"] > 0)]
    n_used = len(used)
    out = {"n_lags_used": n_used, "fit_nugget": np.nan, "fit_range_m": np.nan,
           "fit_rmse": np.nan, "fit_wcost": np.nan}
    if n_used < 2:
        out["status"] = "failed_too_few_lags"
        return out, used.index
    h = used["lag_m"].values
    g = used["gamma"].values
    sw = np.sqrt(used["npairs"].values / h ** 2)
    lo = np.array([NUG_LOWER, lag_dist])
    hi = np.array([NUG_UPPER, DOMAIN_EXTENT_M])

    def resid(p):
        return sw * (spherical_sill1(h, p[0], p[1]) - g)

    best = None
    for n0 in MULTISTART_NUGGETS:
        for a0 in np.geomspace(lag_dist, DOMAIN_EXTENT_M, MULTISTART_N_RANGES):
            x0 = np.clip([n0, a0], lo, hi)
            r = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                              x_scale=[1.0, DOMAIN_EXTENT_M])
            if r.success and (best is None or r.cost < best.cost - 1e-15):
                best = r
    if best is None:
        out["status"] = "failed_optimizer"
        return out, used.index
    nug, a = float(best.x[0]), float(best.x[1])
    out.update({
        "fit_nugget": nug,
        "fit_range_m": a,
        "fit_rmse": float(np.sqrt(np.mean((spherical_sill1(h, nug, a) - g) ** 2))),
        "fit_wcost": float(best.cost),
        "status": "ok" if n_used >= MIN_LAGS_OK else "few_lags",
    })
    return out, used.index


def bound_flags(nug, a, lag_dist):
    def near(v, b):
        return bool(np.isfinite(v) and abs(v - b) <= BOUND_RTOL * max(abs(b), 1.0))
    return {
        "nugget_at_lower": near(nug, NUG_LOWER),
        "nugget_at_upper": near(nug, NUG_UPPER),
        "range_at_lower": bool(np.isfinite(a) and abs(a - lag_dist) <= BOUND_RTOL * lag_dist),
        "range_at_upper": bool(np.isfinite(a) and abs(a - DOMAIN_EXTENT_M) <= BOUND_RTOL * DOMAIN_EXTENT_M),
    }


def load_cells():
    cells = pd.read_csv(CELLS_CSV, dtype={"axis_level": str})
    cfg = json.loads((SOURCE_RUN / "run_config.json").read_text(encoding="utf-8"))
    cfg_cells = pd.DataFrame(cfg["cells"])
    cells = cells.merge(cfg_cells[["experiment", "config_key", "hmaj1", "hmin1", "nug"]],
                        on=["experiment", "config_key"], how="left", validate="many_to_one")
    if cells[["hmaj1", "hmin1", "nug"]].isna().any().any():
        raise ValueError("evaluation_cells.csv rows without a run_config cell.")
    if not np.allclose(cells["hmaj1"], cells["hmin1"], rtol=0, atol=0):
        raise ValueError("Non-isotropic truth found (hmaj1 != hmin1); omnidirectional fit invalid.")
    if cells["run_dir_kriging_b"].isna().any():
        raise ValueError("cell without a kriging run dir.")
    return cells


def process_cell(row):
    samples_path = _REPO_ROOT / row["run_dir_kriging_b"] / "samples.csv"
    df = pd.read_csv(samples_path)
    if len(df) != int(row["n_samples_actual"]):
        raise ValueError(f"{samples_path}: {len(df)} rows != n_samples_actual {row['n_samples_actual']}")
    ns, _vr, _vrg, _corr = nscore_corrected(df, VCOL, verbose=False)
    if not np.allclose(ns, df[NSCOL].values, rtol=0, atol=1e-10):
        raise ValueError(f"{samples_path}: recomputed nscore != saved NPor")
    df_ns = df[[XCOL, YCOL]].copy()
    df_ns[NSCOL] = ns
    ev, lag_dist, lag_tol, nlag = experimental_variogram(df_ns)
    fit, used_idx = fit_spherical(ev, lag_dist)
    ev["used_in_fit"] = ev.index.isin(used_idx)
    fit.update(bound_flags(fit["fit_nugget"], fit["fit_range_m"], lag_dist))
    fit.update({"lag_dist_m": lag_dist, "lag_tol_m": lag_tol, "nlag": nlag,
                "ns_variance": float(np.var(ns))})
    return ev, fit


ID_COLS = ["experiment", "axis", "axis_level", "truth_seed", "sample_rep", "sample_seed", "config_key"]


def main():
    t0 = time.time()
    cells = load_cells()
    ev_rows, fit_rows = [], []
    for _, row in cells.iterrows():
        ev, fit = process_cell(row)
        ids = {k: row[k] for k in ID_COLS}
        ev_rows.append(ev.assign(**ids))
        fit_rows.append({**ids,
                         "n_samples": int(row["n_samples_actual"]),
                         "true_range_m": float(row["hmaj1"]),
                         "true_nugget": float(row["nug"]),
                         **fit,
                         "run_dir_kriging_b": row["run_dir_kriging_b"]})
    ev_all = pd.concat(ev_rows, ignore_index=True)
    ev_all = ev_all[ID_COLS + ["lag_index", "lag_center_m", "lag_m", "gamma", "npairs",
                               "npairs_gamv", "used_in_fit"]]
    fit_cols = ID_COLS + ["n_samples", "true_range_m", "true_nugget", "fit_range_m",
                          "fit_nugget", "fit_rmse", "fit_wcost", "n_lags_used", "lag_dist_m",
                          "lag_tol_m", "nlag", "ns_variance", "status", "nugget_at_lower",
                          "nugget_at_upper", "range_at_lower", "range_at_upper",
                          "run_dir_kriging_b"]
    fits = pd.DataFrame(fit_rows)[fit_cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = ["experimental_variograms.csv", "fitted_variograms.csv"]
    ev_all.to_csv(OUT_DIR / files[0], index=False)
    fits.to_csv(OUT_DIR / files[1], index=False)
    runtime = time.time() - t0

    params = {
        "source_cells_csv": str(CELLS_CSV.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "source_run": str(SOURCE_RUN.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "samples_source": "<run_dir_kriging_b>/samples.csv (exact kriging conditioning data)",
        "nscore": "src.nscore.nscore on Por (asserted == saved NPor, atol 1e-10)",
        "n_cells": int(len(fits)),
        "domain_extent_m": DOMAIN_EXTENT_M,
        "lag_rule": LAG_RULE,
        "gamv": {"tmin": GAMV_TMIN, "tmax": GAMV_TMAX, "azi": AZI, "atol": ATOL,
                 "bandh": BANDH, "isill": ISILL, "max_lag_m": MAX_LAG_M,
                 "lag_tol_fraction": LAG_TOL_FRACTION},
        "npairs_note": "npairs = distinct point pairs = gamv npp / 4 (ordered pairs x omni double count); "
                       "gamv index 0 (self pairs) and empty bins dropped",
        "fit_rule": FIT_RULE,
        "fit": {"sill": SILL, "nug_bounds": [NUG_LOWER, NUG_UPPER],
                "range_bounds": ["lag_dist_m", DOMAIN_EXTENT_M],
                "min_pairs_for_fit": MIN_PAIRS_FOR_FIT, "min_lags_ok": MIN_LAGS_OK,
                "multistart_nuggets": list(MULTISTART_NUGGETS),
                "multistart_n_ranges": MULTISTART_N_RANGES,
                "weights": WEIGHT_RULE, "bound_flag_rtol": BOUND_RTOL},
        "runtime_seconds": runtime,
    }
    save_result(experiment=EXPERIMENT, params=params, seed_or_seeds=None,
                run_dir=OUT_DIR, code_entrypoint=CODE_ENTRYPOINT, output_files=files)
    print(f"done: {len(fits)} cells, {len(ev_all)} lag rows, {runtime:.1f} s")
    print(fits["status"].value_counts().to_string())
    return fits, ev_all


if __name__ == "__main__":
    main()
