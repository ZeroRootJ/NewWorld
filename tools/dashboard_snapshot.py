"""Write CSV snapshots for the experiment-progress dashboard artifact.

The dashboard (https://claude.ai/artifact/24XSk8GVRkCodgFAduuw7c) reads two
attached files that this script writes; the coordinator uploads them after
each run update (user request 2026-11-20: keep the dashboard on whatever
experiment is running).

  runs.csv  one row per experiment: experiment, status, n_done, n_total,
            started, updated, raw_dir, note  (from a JSON status file)
  fits.csv  GP-MLE fits for the current experiment, one row per
            (axis, level, truth_seed): axis, level, truth_seed, new,
            practical_range_m, noise_fraction, truth_range_m,
            truth_nugget_fraction, noise_at_floor

Usage:
  .venv/Scripts/python.exe tools/dashboard_snapshot.py <status.json> <out_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "results" / "processed"
SOURCES = {
    # axis: (original 10-truth file, extended file written by the n50 run)
    "range": (PROC / "range_axis" / "gp_hyperparameters_by_realization.csv",
              PROC / "range_axis" / "gp_hyperparameters_by_realization_n50.csv"),
    "nugget": (PROC / "nugget_axis" / "gp_hyperparameters_by_realization.csv",
               PROC / "nugget_axis" / "gp_hyperparameters_by_realization_n50.csv"),
}
TRUTH_RANGE_BASE_M = 300.0
TRUTH_NUGGET_BASE = 0.05


def _fits():
    out = []
    for axis, (orig, ext) in SOURCES.items():
        o = pd.read_csv(orig)
        d = pd.read_csv(ext) if ext.exists() else o
        old = set(zip(o["axis_level"].astype(str), o["truth_seed"]))
        f = pd.DataFrame({
            "axis": axis,
            "level": d["axis_level"],
            "truth_seed": d["truth_seed"],
            "practical_range_m": d["gp_practical_range_m"],
            "noise_fraction": d["gp_noise_fraction_of_fitted_sill"],
            "noise_at_floor": d["gp_noise_at_optimizer_floor"].astype(bool),
        })
        f["new"] = [(str(a), s) not in old for a, s in zip(d["axis_level"], d["truth_seed"])]
        if axis == "range":
            f["truth_range_m"] = d["truth_range_m"]
            f["truth_nugget_fraction"] = d["truth_nugget_normalized"]
        else:
            f["truth_range_m"] = TRUTH_RANGE_BASE_M
            f["truth_nugget_fraction"] = d["nug_normalized"]
        out.append(f)
    return pd.concat(out, ignore_index=True)


def main(status_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(json.loads(Path(status_path).read_text(encoding="utf-8")))
    fits = _fits()
    runs.to_csv(out / "runs.csv", index=False)
    fits.to_csv(out / "fits.csv", index=False)
    print(f"runs.csv {len(runs)} rows, fits.csv {len(fits)} rows "
          f"({int(fits['new'].sum())} new) -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
