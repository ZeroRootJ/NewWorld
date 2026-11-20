"""GP-MLE hyperparameters on 50 ground-truth realizations per level
(range axis: 8 levels, nugget axis: 6 levels) -- GP-MLE ONLY.

WHY THIS EXISTS
---------------
Fig. 4 of the paper (storyboard key F4) plots the GP-MLE fitted practical
range vs. the ground-truth range (range axis) and the GP-MLE fitted noise vs.
the ground-truth nugget (nugget axis), one point per ground-truth
realization. The existing tables

  results/processed/range_axis/gp_hyperparameters_by_realization.csv  (8 x 10)
  results/processed/nugget_axis/gp_hyperparameters_by_realization.csv (6 x 10)

carry 10 realizations per level. The user asked for ~50 per level. This
script adds 40 realizations per level and changes NOTHING else.

WHAT IS REUSED (by import, not re-declared)
-------------------------------------------
The per-cell work is the existing ``fit_one`` of
  src/experiments/gp_realization_check_range_axis.py   (range axis)
  src/experiments/gp_realization_check_nugget_axis.py  (nugget axis)
imported unchanged. So the base case, the sample design (SAMPLE_SEED=20,
N_SAMPLES, identical sample LOCATIONS across all cells), the truth-generation
function, the GP pipeline (``gp_mle.build_gpr``) and every recorded column are
exactly the ones that produced the 10-realization tables. Those two modules
are not edited (their docstrings say "do not extend this script"); this is a
separate driver around them. Level lists are imported (ALL_RANGE_VALUES,
ALL_NUGGET_VALUES). No kriging / SGS / RBF is run.

SEED RULE
---------
The existing block is (TRUTH_SEED=101, 7101, ..., 7109): the pinned seed
first, then the contiguous block 7100 + k. The SAME rule is continued:

  TRUTH_SEEDS_N50 = (101,) + tuple(7100 + k for k in range(1, 50))
                  = (101, 7101, ..., 7149)

The first 10 entries equal the existing TRUTH_SEED_REPLICATES of both
modules (asserted). The 40 new seeds 7110-7149 are asserted not to collide
with any seed pinned elsewhere in the project (the registries kept in
multi_truth_axes.py, ensemble_size_convergence.py, anisotropy_*.py, plus the
batch-seed ranges of multi_truth_axes). Both axes use the same 50 seeds, as
the 10-realization tables did.

REPRODUCTION CHECK (fatal on failure)
-------------------------------------
All 50 seeds are re-fitted, INCLUDING the original 10. The recomputed rows of
the original 10 seeds are compared against the existing CSVs column by column
(numeric: relative difference <= REPRO_RTOL = 1e-9; NaN must be NaN in both;
booleans/strings exact). Any mismatch raises before the processed CSVs are
written. The existing modules' own pinned-artifact checks
(``verify_against_pinned_samples`` / ``verify_against_pinned_hyperparameters``)
and sample-geometry check (``verify_sample_geometry``) are also run.

OUTPUTS
-------
raw run   results/raw/gp_realization_n50/<timestamp>/
            range_axis_fits.csv, nugget_axis_fits.csv   (all 50 seeds, recomputed)
            reproduction_check.csv                      (per-value comparison)
            manifest.json
processed results/processed/range_axis/gp_hyperparameters_by_realization_n50.csv
          results/processed/nugget_axis/gp_hyperparameters_by_realization_n50.csv
            Same columns / column order / sort order as the 10-realization
            CSVs. Rows of the ORIGINAL 10 seeds are copied VERBATIM from the
            existing CSV (so the n50 file is a strict superset of it); rows of
            the 40 new seeds come from the raw run.
The existing CSVs are never written.

Run with:
.venv/Scripts/python.exe -m src.experiments.gp_realization_n50 [--workers N]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import gp_realization_check_nugget_axis as nug_mod
from src.experiments import gp_realization_check_range_axis as rng_mod
from src.experiments.base_case_conditioning import N_SAMPLES, SAMPLE_SEED, TRUTH_SEED
from src.experiments.nugget_axis import ALL_NUGGET_VALUES, AXIS_HMAJ1, AXIS_HMIN1
from src.experiments.range_axis import ALL_RANGE_VALUES
from src.io import make_run_dir, save_result

EXPERIMENT = "gp_realization_n50"
CODE_ENTRYPOINT = "src/experiments/gp_realization_n50.py"

N_REALIZATIONS = 50
TRUTH_SEED_BLOCK_BASE = 7100
TRUTH_SEEDS_N50 = (TRUTH_SEED,) + tuple(
    TRUTH_SEED_BLOCK_BASE + k for k in range(1, N_REALIZATIONS)
)
EXISTING_SEEDS_10 = tuple(int(s) for s in rng_mod.TRUTH_SEED_REPLICATES)
NEW_SEEDS = tuple(s for s in TRUTH_SEEDS_N50 if s not in EXISTING_SEEDS_10)

REPRO_RTOL = 1e-9
REPRO_ZERO_ATOL = 1e-12  # only for compared values that are exactly 0 in the CSV

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RANGE_EXISTING_CSV = rng_mod.OUT_CSV
NUGGET_EXISTING_CSV = nug_mod.OUT_CSV
RANGE_OUT_CSV = rng_mod.PROCESSED_DIR / "gp_hyperparameters_by_realization_n50.csv"
NUGGET_OUT_CSV = nug_mod.PROCESSED_DIR / "gp_hyperparameters_by_realization_n50.csv"

_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


# --- seed-rule assertions (import time) -------------------------------------
if TRUTH_SEEDS_N50[:10] != EXISTING_SEEDS_10:
    raise ValueError(
        f"seed rule does not reproduce the existing block: {TRUTH_SEEDS_N50[:10]} "
        f"vs {EXISTING_SEEDS_10}"
    )
if tuple(int(s) for s in nug_mod.TRUTH_SEED_REPLICATES) != EXISTING_SEEDS_10:
    raise ValueError("range/nugget modules no longer share one truth-seed block.")
if len(set(TRUTH_SEEDS_N50)) != N_REALIZATIONS or len(NEW_SEEDS) != N_REALIZATIONS - 10:
    raise ValueError(f"unexpected seed list {TRUTH_SEEDS_N50}")


def _existing_project_seeds() -> set:
    """Union of the seed registries other experiments keep for their own
    non-collision assertions (imported, so this list cannot go stale silently
    with respect to them)."""
    from src.experiments import anisotropy_azimuth_sweep as aaz
    from src.experiments import anisotropy_variogram_check as avc
    from src.experiments import ensemble_size_convergence as esc
    from src.experiments import multi_truth_axes as mta

    seeds = (
        set(mta._EXISTING_SEEDS)
        | set(mta._ALL_BATCH_SEEDS)
        | set(esc._EXISTING_SEEDS)
        | set(esc.SGS_SEEDS)
        | set(esc.BOOTSTRAP_SEEDS)
        | set(avc._EXISTING_PINNED_SEEDS)
        | set(avc.TRUTH_SEEDS)
        | set(avc.SAMPLE_SEED_DRAWS)
        | set(aaz.TRUTH_SEEDS_SWEEP)
        | set(aaz.SAMPLE_SEED_DRAWS_SWEEP)
    )
    return {int(s) for s in seeds}


def assert_new_seeds_do_not_collide() -> None:
    existing = _existing_project_seeds()
    hit = sorted(set(NEW_SEEDS) & existing)
    if hit:
        raise ValueError(f"new truth seeds collide with existing project seeds: {hit}")
    print(
        f"seed check PASSED: new seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]} ({len(NEW_SEEDS)}) "
        f"collide with none of {len(existing)} registered project seeds."
    )


# --- worker -----------------------------------------------------------------
def _fit_cell(axis: str, level_value: float, truth_seed: int) -> dict:
    """Module-level (picklable under Windows spawn). Calls the UNCHANGED
    fit_one of the corresponding 10-realization module."""
    mod = rng_mod if axis == "range" else nug_mod
    row = mod.fit_one(float(level_value), int(truth_seed))
    row["_axis"] = axis
    return row


def run_all_cells(workers: int):
    cells = [("range", float(r), s) for r in ALL_RANGE_VALUES for s in TRUTH_SEEDS_N50]
    cells += [("nugget", float(n), s) for n in ALL_NUGGET_VALUES for s in TRUTH_SEEDS_N50]
    t0 = time.time()
    rows = []
    if workers <= 1:
        for c in cells:
            rows.append(_fit_cell(*c))
    else:
        # BLAS threads are deliberately NOT capped (unlike _axis_parallel):
        # the existing 10-realization CSVs were produced uncapped, and at least
        # one fit (range=100 m, seed 7106) sits on a flat likelihood optimum
        # where capping OpenBLAS to 1 thread moves the optimizer endpoint by
        # ~3e-6 relative (LML change ~1e-10), which breaks the 1e-9
        # reproduction check. Uncapped workers reproduce the CSVs.
        for v in _BLAS_THREAD_ENV_VARS:
            if v in os.environ:
                raise RuntimeError(
                    f"{v}={os.environ[v]} is set; unset it so fits use the same "
                    "default BLAS threading as the existing CSVs."
                )
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_fit_cell, *c) for c in cells]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % 25 == 0 or i == len(cells):
                    print(f"  [{i}/{len(cells)}] fits done  [{time.time() - t0:.0f}s]", flush=True)
    return rows, time.time() - t0


# --- table assembly -----------------------------------------------------------
def _to_df(rows, axis: str) -> pd.DataFrame:
    sel = [r for r in rows if r["_axis"] == axis]
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in sel])
    if axis == "range":
        df = df.sort_values(["range_m", "truth_seed"]).reset_index(drop=True)
        # same post-hoc column as gp_realization_check_range_axis.main()
        df["fitted_range_below_truth_range"] = df["gp_practical_range_m"] < df["truth_range_m"]
    else:
        df = df.sort_values(["nug_normalized", "truth_seed"]).reset_index(drop=True)
        # same post-hoc column as gp_realization_check_nugget_axis.main()
        df["noise_above_truth_nugget"] = (
            df["gp_noise_variance_real_units"] > df["truth_nugget_real_units"]
        )
    return df


def compare_to_existing(new_df: pd.DataFrame, existing_csv: Path, axis: str) -> pd.DataFrame:
    """Column-by-column comparison of the recomputed original-10 rows against
    the existing CSV. Raises on any mismatch."""
    old = pd.read_csv(existing_csv, dtype={"axis_level": str})
    if list(new_df.columns) != list(old.columns):
        raise ValueError(
            f"{axis}: column mismatch\n new={list(new_df.columns)}\n old={list(old.columns)}"
        )
    sub = new_df[new_df["truth_seed"].isin(EXISTING_SEEDS_10)].copy()
    sub["axis_level"] = sub["axis_level"].astype(str)
    key = ["axis_level", "truth_seed"]
    m = old.merge(sub, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    if (m["_merge"] != "both").any() or len(m) != len(old):
        raise ValueError(f"{axis}: row sets differ between existing CSV and recomputation")

    recs = []
    for col in old.columns:
        if col in key:
            continue
        for _, r in m.iterrows():
            a, b = r[f"{col}_old"], r[f"{col}_new"]
            if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)) or isinstance(a, str):
                ok = (bool(a) == bool(b)) if not isinstance(a, str) else (str(a) == str(b))
                rel = 0.0 if ok else np.inf
            else:
                a, b = float(a), float(b)
                if np.isnan(a) or np.isnan(b):
                    ok = np.isnan(a) and np.isnan(b)
                    rel = 0.0 if ok else np.inf
                elif a == 0.0:
                    rel = abs(b)
                    ok = rel <= REPRO_ZERO_ATOL
                else:
                    rel = abs(b - a) / abs(a)
                    ok = rel <= REPRO_RTOL
            recs.append({
                "axis": axis, "axis_level": r["axis_level"], "truth_seed": int(r["truth_seed"]),
                "column": col, "existing_value": a, "recomputed_value": b,
                "relative_difference": rel, "pass": bool(ok),
            })
    chk = pd.DataFrame(recs)
    bad = chk[~chk["pass"]]
    if len(bad):
        b = bad.iloc[0]
        raise ValueError(
            f"{axis}: REPRODUCTION FAILURE, {len(bad)} of {len(chk)} values; first: "
            f"level={b['axis_level']} seed={b['truth_seed']} {b['column']}: "
            f"existing {b['existing_value']!r} vs recomputed {b['recomputed_value']!r}"
        )
    print(
        f"{axis}: reproduction of existing CSV PASSED -- {len(chk)} values "
        f"({len(old)} rows x {len(old.columns) - 2} columns), max relative difference "
        f"{chk['relative_difference'].max():.3e} (tol {REPRO_RTOL:g})."
    )
    return chk


def build_processed(new_df: pd.DataFrame, existing_csv: Path) -> pd.DataFrame:
    """Original 10 rows verbatim from the existing CSV + the 40 new rows."""
    old = pd.read_csv(existing_csv, dtype={"axis_level": str})
    extra = new_df[~new_df["truth_seed"].isin(EXISTING_SEEDS_10)].copy()
    extra["axis_level"] = extra["axis_level"].astype(str)
    out = pd.concat([old, extra[old.columns]], ignore_index=True)
    sort_col = "range_m" if "range_m" in out.columns else "nug_normalized"
    out = out.sort_values([sort_col, "truth_seed"]).reset_index(drop=True)
    counts = out.groupby("axis_level").size()
    if not (counts == N_REALIZATIONS).all():
        raise ValueError(f"expected {N_REALIZATIONS} rows per level, got {counts.to_dict()}")
    return out


def main(workers: int = None):
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) // 2)
    for p in (RANGE_OUT_CSV, NUGGET_OUT_CSV):
        if p.exists():
            raise FileExistsError(f"{p} already exists -- refusing to overwrite.")
    assert_new_seeds_do_not_collide()
    rng_mod.verify_practical_range_factor()
    rng_mod.verify_against_pinned_samples()
    nug_mod.verify_against_pinned_samples()

    t_wall = time.time()
    rows, t_fit = run_all_cells(workers)

    geo_range = rng_mod.verify_sample_geometry([r for r in rows if r["_axis"] == "range"])
    geo_nug = nug_mod.verify_sample_geometry([r for r in rows if r["_axis"] == "nugget"])

    range_df = _to_df(rows, "range")
    nug_df = _to_df(rows, "nugget")

    # existing modules' own pinned-artifact checks (truth_seed=101 rows)
    rng_mod.verify_against_pinned_hyperparameters(range_df)
    nug_mod.verify_against_pinned_hyperparameters(nug_df)

    chk = pd.concat([
        compare_to_existing(range_df, RANGE_EXISTING_CSV, "range"),
        compare_to_existing(nug_df, NUGGET_EXISTING_CSV, "nugget"),
    ], ignore_index=True)

    # --- raw run ---
    run_dir = make_run_dir(EXPERIMENT)
    range_df.to_csv(run_dir / "range_axis_fits.csv", index=False)
    nug_df.to_csv(run_dir / "nugget_axis_fits.csv", index=False)
    chk.to_csv(run_dir / "reproduction_check.csv", index=False)
    wall = time.time() - t_wall
    params = {
        "description": "GP-MLE hyperparameter fits only (no kriging/SGS/RBF), "
                       "50 ground-truth realizations per level, range and nugget axes",
        "range_axis": {
            "levels_m": [float(r) for r in ALL_RANGE_VALUES],
            "nug_normalized": float(rng_mod.AXIS_NUG),
            "isotropic": True,
            "fit_function": "src.experiments.gp_realization_check_range_axis.fit_one",
        },
        "nugget_axis": {
            "levels_nug_normalized": [float(n) for n in ALL_NUGGET_VALUES],
            "hmaj1_m": float(AXIS_HMAJ1),
            "hmin1_m": float(AXIS_HMIN1),
            "fit_function": "src.experiments.gp_realization_check_nugget_axis.fit_one",
        },
        "truth_seeds": list(TRUTH_SEEDS_N50),
        "truth_seed_rule": "(TRUTH_SEED=101,) + (7100 + k for k in 1..49)",
        "existing_truth_seeds": list(EXISTING_SEEDS_10),
        "new_truth_seeds": list(NEW_SEEDS),
        "sample_seed": int(SAMPLE_SEED),
        "n_samples_requested": int(N_SAMPLES),
        "gp_config": "src.experiments.gp_mle.build_gpr() (imported, unchanged)",
        "reproduction_rtol": REPRO_RTOL,
        "reproduction_max_relative_difference": float(chk["relative_difference"].max()),
        "sample_geometry_range": geo_range,
        "sample_geometry_nugget": geo_nug,
        "workers": int(workers),
        "blas_threads_per_worker": "OpenBLAS default (uncapped), same as the existing 10-realization CSVs",
        "fit_wall_time_s": float(t_fit),
        "total_wall_time_s": float(wall),
    }
    outputs = ["range_axis_fits.csv", "nugget_axis_fits.csv", "reproduction_check.csv"]
    save_result(EXPERIMENT, params, list(TRUTH_SEEDS_N50), run_dir=run_dir,
                code_entrypoint=CODE_ENTRYPOINT, output_files=outputs)

    # --- processed ---
    range_out = build_processed(range_df, RANGE_EXISTING_CSV)
    nug_out = build_processed(nug_df, NUGGET_EXISTING_CSV)
    range_out.to_csv(RANGE_OUT_CSV, index=False)
    nug_out.to_csv(NUGGET_OUT_CSV, index=False)

    print(f"\nraw run: {run_dir}")
    print(f"processed: {RANGE_OUT_CSV}\n           {NUGGET_OUT_CSV}")
    print(f"fit wall time {t_fit:.0f}s, total {wall:.0f}s, workers={workers}")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    main(ap.parse_args().workers)
