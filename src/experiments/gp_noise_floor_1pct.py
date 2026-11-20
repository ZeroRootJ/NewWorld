"""GP-MLE noise-floor diagnostic on the E4 1 % density cells.

Question (user, 2026-11-20): at 1 % density (E4 sample_replicate_axis level
'1', n_samples = 25) 16 of the 100 GP-MLE fits have the WhiteKernel noise
level stuck at the optimizer lower bound 1e-5 (normalized-y space). Did that
bound change GP-MLE's performance?

Test: refit GP-MLE ONLY on those 100 cells (10 truths x 10 sample sets),
identical in every respect to the production run (same truths / samples via
the evaluation's own ``_conditioning``, same ``gp_mle.build_gpr`` pipeline,
kernel, restarts, normalize_y, random_state; same prediction call as
``gp_mle.main``; same MSE / UMG code path ``_gaussian_metrics`` as
``evaluate_multi_truth_axes``) EXCEPT the WhiteKernel noise-level lower
bound, 1e-5 -> 1e-10 (upper bound unchanged).

Each cell is fitted twice:
  * bound_1e-05 : production bound (reproduction control -- must reproduce
                  the production E4 hyperparameters / UMG / MSE);
  * bound_1e-10 : lowered floor.

NOTE: sklearn draws the restart starting points uniformly in the LOG-bounds
box, so changing the lower bound also changes the noise-dimension starting
points of the 8 restarts (same random numbers, different box). This is
inherent to "same pipeline, different bound"; it is recorded, not corrected.

BLAS threading is left at its default (prior finding: default threading
reproduces the production fits exactly).

Outputs
-------
raw  : results/raw/gp_noise_floor_1pct/<timestamp>/
         fits.csv                     one row per (cell, bound)
         posterior_mean_maps_<b>.npy  (100, NY, NX), row order = fits.csv cells
         posterior_var_maps_<b>.npy   (100, NY, NX)
         cells.csv                    cell order of the npy stacks
         manifest.json
processed : results/processed/gp_noise_floor_1pct/
         comparison.csv, summary.csv, reproduction_check.json

Run: .venv/Scripts/python.exe -m src.experiments.gp_noise_floor_1pct
     (--processed-only <raw run dir> rebuilds processed from an existing run)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.base_case import NX, NY, XMN, YMN, XSIZ, YSIZ
from src.experiments.base_case_conditioning import VCOL
from src.experiments.evaluate_multi_truth_axes import (
    _conditioning,
    _gaussian_metrics,
    read_metrics_csv,
)
from src.experiments.gp_mle import (
    GP_RANDOM_STATE,
    N_RESTARTS_OPTIMIZER,
    NOISE_LEVEL_BOUNDS,
    build_gpr,
    extract_fitted_hyperparameters,
)
from src.experiments.multi_truth_axes import resolve_run_dir
from src.experiments.qc_nugget_axis import GP_PRACTICAL_RANGE_FACTOR
from src.grid_utils import full_grid_coordinates
from src.io import get_git_commit, get_git_status, make_run_dir, save_result

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXPERIMENT_NAME = "gp_noise_floor_1pct"
CODE_ENTRYPOINT = "src/experiments/gp_noise_floor_1pct.py"

SOURCE_RUN = "results/raw/multi_truth_axes/20261116T222609599808Z"
SOURCE_PROCESSED = _REPO_ROOT / "results" / "processed" / "multi_truth_axes"
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / EXPERIMENT_NAME

E4_EXPERIMENT = "E4"
DENSITY_LEVEL = "1"          # 1 % density, n_samples = 25
EXPECTED_N_CELLS = 100

OLD_NOISE_BOUNDS = tuple(NOISE_LEVEL_BOUNDS)                 # (1e-5, 1e3), production
NEW_NOISE_BOUNDS = (1e-10, NOISE_LEVEL_BOUNDS[1])            # only the floor changes
BOUND_TAGS = {"old": OLD_NOISE_BOUNDS, "new": NEW_NOISE_BOUNDS}

# a fit is "at the floor" if noise_variance_normalized is within this relative
# tolerance of the lower bound (production floor values are exactly 1e-5)
AT_FLOOR_RTOL = 1e-6
# reproduction tolerance of the 1e-5 refit vs production values
REPRO_RTOL = 1e-6


def _cells() -> list:
    cfg = json.loads((resolve_run_dir(SOURCE_RUN) / "run_config.json").read_text(encoding="utf-8"))
    cells = [c for c in cfg["cells"] if c["experiment"] == E4_EXPERIMENT and str(c["axis_level"]) == DENSITY_LEVEL]
    if len(cells) != EXPECTED_N_CELLS:
        raise ValueError(f"expected {EXPECTED_N_CELLS} E4 1 % cells, found {len(cells)}")
    if {c["n_samples"] for c in cells} != {25}:
        raise ValueError("E4 1 % cells do not all have n_samples = 25")
    return sorted(cells, key=lambda c: (c["truth_seed"], c["sample_rep"]))


def fit_cell(cell: dict, noise_bounds) -> dict:
    """Fit + predict exactly as gp_mle.main, evaluate exactly as
    evaluate_multi_truth_axes.evaluate_cell (gp_mle branch)."""
    truth, samples_df, mask = _conditioning(cell["truth_seed"], cell["sample_seed"], cell["hmaj1"],
                                            cell["hmin1"], cell["nug"], cell["n_samples"])
    X = samples_df[["X", "Y"]].values
    y = samples_df[VCOL].values
    gpr = build_gpr(noise_level_bounds=noise_bounds)
    t0 = time.time()
    gpr.fit(X, y)
    fit_s = time.time() - t0
    grid = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)
    post_mean, post_std = gpr.predict(grid, return_std=True)
    mean_map = post_mean.reshape(NY, NX)
    var_map = post_std.reshape(NY, NX) ** 2           # as gp_mle.main -> posterior_var_map.npy
    truth_m = truth[mask]
    met = _gaussian_metrics(truth_m, mean_map[mask], mean_map[mask], var_map[mask], f"gp {cell['config_key']}")
    fh = extract_fitted_hyperparameters(gpr)
    return {
        "metrics": met, "fh": fh, "lml": float(gpr.log_marginal_likelihood(gpr.kernel_.theta)),
        "kernel": str(gpr.kernel_), "fit_seconds": fit_s, "mean_map": mean_map, "var_map": var_map,
        "n_samples_actual": int(len(samples_df)), "n_cells_evaluated": int(mask.sum()),
    }


def run_raw() -> Path:
    t_start = time.time()
    cells = _cells()
    run_dir = make_run_dir(EXPERIMENT_NAME)
    rows, maps = [], {tag: ([], []) for tag in BOUND_TAGS}
    for i, c in enumerate(cells):
        for tag, bounds in BOUND_TAGS.items():
            r = fit_cell(c, bounds)
            fh = r["fh"]
            rows.append({
                "truth_seed": c["truth_seed"], "sample_rep": c["sample_rep"], "sample_seed": c["sample_seed"],
                "config_key": c["config_key"], "bound_tag": tag,
                "noise_lower_bound": bounds[0], "noise_upper_bound": bounds[1],
                "length_scale_m": fh["length_scale_m"],
                "practical_range_m": GP_PRACTICAL_RANGE_FACTOR * fh["length_scale_m"],
                "noise_variance_normalized": fh["noise_variance_normalized"],
                "noise_variance_real_units": fh["noise_variance_real_units"],
                "signal_variance_normalized": fh["signal_variance_normalized"],
                "signal_variance_real_units": fh["signal_variance_real_units"],
                "y_train_std": fh["y_train_std"],
                "log_marginal_likelihood": r["lml"], "fitted_kernel_str": r["kernel"],
                **{m: r["metrics"][m] for m in ("mse", "umg", "variance_mean", "coverage90")},
                "n_samples_actual": r["n_samples_actual"], "n_cells_evaluated": r["n_cells_evaluated"],
                "fit_seconds": r["fit_seconds"],
            })
            maps[tag][0].append(r["mean_map"])
            maps[tag][1].append(r["var_map"])
        print(f"[{i + 1}/{len(cells)}] {c['config_key']} done ({time.time() - t_start:.1f}s)", flush=True)

    output_files = []
    pd.DataFrame(rows).to_csv(run_dir / "fits.csv", index=False)
    output_files.append("fits.csv")
    pd.DataFrame([{k: c[k] for k in ("truth_seed", "sample_rep", "sample_seed", "config_key")} for c in cells]) \
        .to_csv(run_dir / "cells.csv", index=False)
    output_files.append("cells.csv")
    for tag, bounds in BOUND_TAGS.items():
        for kind, arrs in (("mean", maps[tag][0]), ("var", maps[tag][1])):
            name = f"posterior_{kind}_maps_bound_{bounds[0]:.0e}.npy"
            np.save(run_dir / name, np.stack(arrs))
            output_files.append(name)
    wall = time.time() - t_start
    params = {
        "question": "does the WhiteKernel noise lower bound (1e-5) change GP-MLE performance at E4 1 % density?",
        "source_run": SOURCE_RUN,
        "cells": f"{E4_EXPERIMENT} sample_replicate_axis axis_level={DENSITY_LEVEL} (n_samples=25), "
                 f"{len(cells)} cells (10 truth seeds x 10 sample reps)",
        "truth_seeds": sorted({c["truth_seed"] for c in cells}),
        "sample_seeds": sorted({c["sample_seed"] for c in cells}),
        "hmaj1": cells[0]["hmaj1"], "hmin1": cells[0]["hmin1"], "nug": cells[0]["nug"],
        "gp_pipeline": "src.experiments.gp_mle.build_gpr(noise_level_bounds=...) -- all other settings production",
        "noise_level_bounds_old": OLD_NOISE_BOUNDS,
        "noise_level_bounds_new": NEW_NOISE_BOUNDS,
        "n_restarts_optimizer": N_RESTARTS_OPTIMIZER,
        "gp_random_state": GP_RANDOM_STATE,
        "normalize_y": True,
        "evaluation": "evaluate_multi_truth_axes._conditioning + _gaussian_metrics (gp_mle branch)",
        "blas_threads": "default (not capped)",
        "restart_start_points_note": "sklearn samples restart starts uniformly in the log-bounds box; changing "
                                     "the lower bound shifts the noise-dimension start points",
        "wall_seconds": wall,
    }
    save_result(EXPERIMENT_NAME, params,
                {"truth_seeds": params["truth_seeds"], "sample_seeds": params["sample_seeds"],
                 "gp_random_state": GP_RANDOM_STATE},
                run_dir=run_dir, code_entrypoint=CODE_ENTRYPOINT, output_files=output_files)
    print(f"raw run: {run_dir}  wall {wall:.1f}s")
    return run_dir


def _production_tables():
    met = read_metrics_csv(SOURCE_PROCESSED / "metrics.csv")
    met = met[(met["experiment"] == E4_EXPERIMENT) & (met["axis_level"] == DENSITY_LEVEL)]
    hyp = pd.read_csv(SOURCE_PROCESSED / "hyperparameters.csv", dtype={"axis_level": str})
    hyp = hyp[(hyp["experiment"] == E4_EXPERIMENT) & (hyp["axis_level"] == DENSITY_LEVEL)
              & (hyp["method"] == "gp_mle")]
    key = ["truth_seed", "sample_rep"]

    def piv(df, col, sel):
        return df[sel].pivot_table(index=key, columns=col, values="value", aggfunc="first")

    gp = piv(met, "metric", (met["method"] == "gp_mle"))
    sgs = piv(met, "metric", (met["method"] == "sgs") & (met["batches"] == "1"))
    kr = piv(met, "metric", (met["method"] == "kriging"))
    hp = piv(hyp, "parameter", np.ones(len(hyp), bool))
    cells = pd.read_csv(SOURCE_PROCESSED / "evaluation_cells.csv", dtype={"axis_level": str})
    cells = cells[(cells["experiment"] == E4_EXPERIMENT) & (cells["axis_level"] == DENSITY_LEVEL)].set_index(key)
    for name, t in (("gp", gp), ("sgs", sgs), ("kriging", kr), ("hyp", hp), ("cells", cells)):
        if len(t) != EXPECTED_N_CELLS:
            raise ValueError(f"production {name} table has {len(t)} cells, expected {EXPECTED_N_CELLS}")
    return gp, sgs, kr, hp, cells


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


def build_processed(run_dir: Path) -> None:
    fits = pd.read_csv(run_dir / "fits.csv")
    old = fits[fits["bound_tag"] == "old"].set_index(["truth_seed", "sample_rep"])
    new = fits[fits["bound_tag"] == "new"].set_index(["truth_seed", "sample_rep"])
    gp, sgs, kr, hp, cells = _production_tables()
    idx = hp.index
    old, new, gp, sgs, kr, cells = (t.loc[idx] for t in (old, new, gp, sgs, kr, cells))

    # ---- reproduction check: 1e-5 refit vs production ---------------------
    repro = {}
    for ours, prod_tab, prod_col in (
        ("length_scale_m", hp, "gp_length_scale_m"),
        ("noise_variance_normalized", hp, "gp_noise_variance_normalized"),
        ("noise_variance_real_units", hp, "gp_noise_variance_real_units"),
        ("signal_variance_normalized", hp, "gp_signal_variance_normalized"),
        ("log_marginal_likelihood", hp, "gp_log_marginal_likelihood"),
        ("umg", gp, "umg"), ("mse", gp, "mse"), ("variance_mean", gp, "variance_mean"),
        ("coverage90", gp, "coverage90"),
    ):
        r = _rel(old[ours].values, prod_tab[prod_col].values)
        repro[ours] = {"max_rel_diff": float(r.max()), "n_exact": int((r == 0).sum()),
                       "n_within_rtol": int((r <= REPRO_RTOL).sum())}
    # posterior maps vs production npy, every cell
    mean_old = np.load(run_dir / f"posterior_mean_maps_bound_{OLD_NOISE_BOUNDS[0]:.0e}.npy")
    var_old = np.load(run_dir / f"posterior_var_maps_bound_{OLD_NOISE_BOUNDS[0]:.0e}.npy")
    order = pd.read_csv(run_dir / "cells.csv").set_index(["truth_seed", "sample_rep"]).index
    mm, vm = [], []
    for i, k in enumerate(order):
        gdir = resolve_run_dir(cells.loc[k, "run_dir_gp_mle_b"])
        mm.append(float(np.max(np.abs(mean_old[i] - np.load(gdir / "posterior_mean_map.npy")))))
        vm.append(float(np.max(np.abs(var_old[i] - np.load(gdir / "posterior_var_map.npy")))))
    repro["posterior_mean_map_max_abs_diff"] = max(mm)
    repro["posterior_var_map_max_abs_diff"] = max(vm)
    repro["n_cells_maps_bit_identical"] = int(sum((a == 0 and b == 0) for a, b in zip(mm, vm)))
    repro["rtol"] = REPRO_RTOL
    bad = [k for k, v in repro.items() if isinstance(v, dict) and v["n_within_rtol"] != EXPECTED_N_CELLS]
    repro["passed"] = not bad
    repro["failed_quantities"] = bad

    # ---- comparison table ------------------------------------------------
    lb_old, lb_new = OLD_NOISE_BOUNDS[0], NEW_NOISE_BOUNDS[0]
    comp = pd.DataFrame({
        "truth_seed": [k[0] for k in idx], "sample_rep": [k[1] for k in idx],
        "sample_seed": old["sample_seed"].values,
        "noise_old": hp["gp_noise_variance_normalized"].values,
        "noise_new": new["noise_variance_normalized"].values,
        "noise_real_old": hp["gp_noise_variance_real_units"].values,
        "noise_real_new": new["noise_variance_real_units"].values,
        "at_floor_old": np.isclose(hp["gp_noise_variance_normalized"].values, lb_old, rtol=AT_FLOOR_RTOL, atol=0),
        "at_floor_new": np.isclose(new["noise_variance_normalized"].values, lb_new, rtol=AT_FLOOR_RTOL, atol=0),
        "length_scale_old": hp["gp_length_scale_m"].values,
        "length_scale_new": new["length_scale_m"].values,
        "practical_range_old": hp["gp_practical_range_m"].values,
        "practical_range_new": new["practical_range_m"].values,
        "signal_var_real_old": hp["gp_signal_variance_real_units"].values,
        "signal_var_real_new": new["signal_variance_real_units"].values,
        "lml_old": hp["gp_log_marginal_likelihood"].values,
        "lml_new": new["log_marginal_likelihood"].values,
        "umg_old": gp["umg"].values, "umg_new": new["umg"].values,
        "mse_old": gp["mse"].values, "mse_new": new["mse"].values,
        "variance_mean_old": gp["variance_mean"].values, "variance_mean_new": new["variance_mean"].values,
        "coverage90_old": gp["coverage90"].values, "coverage90_new": new["coverage90"].values,
        "umg_sgs": sgs["umg"].values, "umg_kriging": kr["umg"].values,
    })
    comp["delta_umg"] = comp["umg_new"] - comp["umg_old"]
    comp["delta_mse"] = comp["mse_new"] - comp["mse_old"]
    comp["delta_lml"] = comp["lml_new"] - comp["lml_old"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(PROCESSED_DIR / "comparison.csv", index=False)

    # ---- summary ---------------------------------------------------------
    srows = []
    for subset, sel in (("all_100", np.ones(len(comp), bool)), ("floor_old", comp["at_floor_old"].values),
                        ("not_floor_old", ~comp["at_floor_old"].values)):
        d = comp[sel]
        for q in ("umg_old", "umg_new", "umg_sgs", "umg_kriging"):
            v = d[q].values
            srows.append({"subset": subset, "n": len(d), "quantity": q, "mean": v.mean(), "median": np.median(v),
                          "p10": np.percentile(v, 10), "min": v.min()})
        for q in ("mse_old", "mse_new"):
            srows.append({"subset": subset, "n": len(d), "quantity": q, "mean": d[q].mean(),
                          "median": np.nan, "p10": np.nan, "min": np.nan})
    summary = pd.DataFrame(srows)
    summary.to_csv(PROCESSED_DIR / "summary.csv", index=False)

    git_dirty, git_dirty_files = get_git_status()
    prov = {
        "generated_by": CODE_ENTRYPOINT, "git_commit": get_git_commit(), "git_dirty": git_dirty,
        "git_dirty_files": git_dirty_files, "raw_run": str(run_dir.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "production_source": str(SOURCE_PROCESSED.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "old_values": "production values from results/processed/multi_truth_axes (metrics.csv, "
                      "hyperparameters.csv); umg_sgs = sgs batches '1' (N=50)",
        "noise_columns": "noise_old/new = WhiteKernel noise_level in normalized-y space (the bound applies here)",
        "at_floor_rule": f"np.isclose(noise, lower_bound, rtol={AT_FLOOR_RTOL}, atol=0)",
        "reproduction_check": repro,
    }
    (PROCESSED_DIR / "reproduction_check.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps(repro, indent=2))
    print(summary.to_string(index=False))
    print(f"at_floor_old={int(comp['at_floor_old'].sum())} at_floor_new={int(comp['at_floor_new'].sum())}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-only", default=None, help="existing raw run dir; skip fitting")
    args = ap.parse_args(argv)
    run_dir = resolve_run_dir(args.processed_only) if args.processed_only else run_raw()
    build_processed(run_dir)


if __name__ == "__main__":
    main()
