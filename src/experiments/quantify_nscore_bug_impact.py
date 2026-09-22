"""Quantify how badly the ``geostats.nscore`` off-by-one (see src/nscore.py)
affected every kriging run this project's processed artifacts currently rely
on.

PURPOSE: this table is EVIDENCE FOR A SCOPE DECISION, not a fix. It re-runs
nothing and changes nothing. The orchestrator/user decide, from these
numbers, which axes (if any) beyond the nugget axis need their kriging arm
re-run.

METHOD (read-only, per pinned kriging run)
------------------------------------------
Each kriging run directory saves both of the things needed to detect the bug
without recomputing anything:

  samples.csv                   -> the ``NPor`` column actually used by kb2d
  nscore_transform_table.csv    -> the run's own (vr, vrg) transform table

The transform TABLE is unaffected by the bug -- only the per-datum lookup is
-- so ``vrg[0]`` is the correct normal score of that run's smallest datum by
construction (the smallest datum IS the table's first knot). Comparing it to
``min(NPor)`` therefore recovers exactly the error the run was computed with:

  npor_min_as_run   = samples.csv's minimum NPor
  npor_min_correct  = nscore_transform_table.csv's vrg[0]
  abs_error         = |as_run - correct|

For runs produced AFTER the 2026-09-22 fix this is 0 by construction, which
is itself a useful column (it separates repaired runs from contaminated ones
in the same table).

AXES COVERED
------------
The five processed axes' ``source_runs.json`` pins, plus the nugget axis's
SUPERSEDED (pre-fix) pins, recorded under the axis name
``nugget_axis_superseded`` so the size of the error that was actually fixed
stays visible after the re-pin. Each JSON has a different nesting shape
(flat method map for the base case; level -> method for range / sample
density / nugget; level -> replicate -> method for sample replicate), so the
kriging entries are located by a recursive walk rather than by four
hand-written loops.

Output: results/processed/nscore_bug_impact.csv (tidy, one row per pinned
kriging run per axis; a run pinned by two axes -- e.g. the base-case run,
which is also range_axis level 300 and sample_density_axis level 5 --
appears once per axis, which is intended, since the question is "which
artifacts are affected").

Run with:
.venv/Scripts/python.exe -m src.experiments.quantify_nscore_bug_impact
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = _REPO_ROOT / "results" / "processed"

# (axis name in the output, path to its source_runs.json)
AXIS_SOURCES = [
    ("base_case", PROCESSED / "base_case" / "source_runs.json"),
    ("range_axis", PROCESSED / "range_axis" / "source_runs.json"),
    ("sample_density_axis", PROCESSED / "sample_density_axis" / "source_runs.json"),
    ("sample_replicate_axis", PROCESSED / "sample_replicate_axis" / "source_runs.json"),
    ("nugget_axis", PROCESSED / "nugget_axis" / "source_runs.json"),
    (
        "nugget_axis_superseded",
        PROCESSED / "nugget_axis" / "source_runs_superseded_20260922_nscore_bug.json",
    ),
]

OUT_CSV = PROCESSED / "nscore_bug_impact.csv"

# A run is flagged as "grossly contaminated" above this absolute error in
# normal-score units. 0.5 NS units is far outside anything a correct
# transform could produce as roundoff, and is the scale at which the error
# starts moving kriged estimates by an appreciable fraction of a standard
# deviation. Reported as a convenience column only -- the scope decision is
# not this script's to make.
GROSS_ERROR_THRESHOLD = 0.5


def _walk_for_kriging(node, path):
    """Yield ``(label_parts, relative_run_dir)`` for every kriging pin found
    anywhere in a source_runs.json, whatever its nesting depth.

    Handles all three shapes in use:
      base_case              {"kriging": {"run_dir": "...", ...}, ...}
      level -> method        {"300": {"kriging": "...", ...}, ...}
      level -> rep -> method {"20": {"rep0": {"kriging": "..."}}, ...}
    """
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "kriging":
            if isinstance(value, str):
                yield path, value
            elif isinstance(value, dict) and "run_dir" in value:
                yield path, value["run_dir"]
            continue
        yield from _walk_for_kriging(value, path + [key])


def measure_run(rel_run_dir: str) -> dict:
    run_dir = _REPO_ROOT / rel_run_dir
    samples = pd.read_csv(run_dir / "samples.csv")
    table = pd.read_csv(run_dir / "nscore_transform_table.csv")

    npor_min_as_run = float(samples["NPor"].min())
    npor_min_correct = float(table["vrg"].iloc[0])
    por_min = float(samples["Por"].min())
    vr_min = float(table["vr"].iloc[0])
    if not np.isclose(por_min, vr_min, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{rel_run_dir}: samples.csv's minimum Por ({por_min}) does not match "
            f"the transform table's vr[0] ({vr_min}) -- the table does not belong "
            "to these samples, so vrg[0] is not the correct answer for them."
        )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    params = manifest.get("params", {})
    corrections = params.get("nscore_corrections")

    return {
        "run_dir": rel_run_dir,
        "n_samples": int(len(samples)),
        "por_min": por_min,
        "npor_min_as_run": npor_min_as_run,
        "npor_min_correct": npor_min_correct,
        "abs_error": abs(npor_min_as_run - npor_min_correct),
        "npor_max_as_run": float(samples["NPor"].max()),
        "npor_max_correct": float(table["vrg"].iloc[-1]),
        # True only for runs produced by the repaired code path (their
        # manifest carries src/nscore.py's audit record).
        "ran_with_nscore_fix": corrections is not None,
        "run_timestamp": manifest.get("timestamp"),
        "truth_seed": params.get("truth_seed"),
        "sample_seed": params.get("sample_seed"),
        "hmaj1": params.get("hmaj1"),
        "nug": params.get("nug"),
        "n_samples_requested": params.get("n_samples_requested"),
    }


def main():
    rows = []
    for axis, source_path in AXIS_SOURCES:
        if not source_path.exists():
            print(f"skipping {axis}: {source_path} does not exist")
            continue
        data = json.loads(source_path.read_text(encoding="utf-8"))
        found = list(_walk_for_kriging(data, []))
        if not found:
            raise ValueError(f"{source_path}: no kriging pin found")
        for label_parts, rel_run_dir in found:
            # label_parts is [] for the base case, ["300"] for a level-keyed
            # axis, ["20", "rep0"] for the replicate axis.
            axis_level = label_parts[0] if label_parts else "base_case"
            replicate = label_parts[1] if len(label_parts) > 1 else ""
            rows.append(
                {
                    "axis": axis,
                    "axis_level": axis_level,
                    "replicate": replicate,
                    "source_runs_json": str(source_path.relative_to(_REPO_ROOT)).replace(
                        "\\", "/"
                    ),
                    **measure_run(rel_run_dir),
                }
            )

    df = pd.DataFrame(rows)
    df["grossly_contaminated"] = df["abs_error"] > GROSS_ERROR_THRESHOLD
    df = df[
        [
            "axis", "axis_level", "replicate", "run_dir",
            "npor_min_as_run", "npor_min_correct", "abs_error", "n_samples",
            "grossly_contaminated", "ran_with_nscore_fix",
            "por_min", "npor_max_as_run", "npor_max_correct",
            "truth_seed", "sample_seed", "hmaj1", "nug", "n_samples_requested",
            "run_timestamp", "source_runs_json",
        ]
    ].sort_values(["axis", "axis_level", "replicate"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # The max-datum column is a control: the off-by-one does not affect the
    # top of the table, so this must be ~0 everywhere. If it ever is not,
    # the bug's characterization in src/nscore.py is incomplete.
    max_side_error = float(np.max(np.abs(df["npor_max_as_run"] - df["npor_max_correct"])))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 200)
    print(
        df[
            [
                "axis", "axis_level", "replicate", "run_dir",
                "npor_min_as_run", "npor_min_correct", "abs_error", "n_samples",
                "grossly_contaminated", "ran_with_nscore_fix",
            ]
        ].to_string(index=False)
    )
    print("\n--- Per-axis summary of |min-datum normal-score error| ---")
    print(
        df.groupby("axis")["abs_error"]
        .agg(
            n_runs="size",
            n_nonzero=lambda s: int((s > 0).sum()),
            median="median",
            mean="mean",
            max="max",
        )
        .to_string()
    )
    print(
        f"\ntotal pinned kriging runs examined: {len(df)}; "
        f"with nonzero error: {int((df['abs_error'] > 0).sum())}; "
        f"grossly contaminated (> {GROSS_ERROR_THRESHOLD}): "
        f"{int(df['grossly_contaminated'].sum())}"
    )
    print(
        "control check -- max-datum side error across all runs: "
        f"{max_side_error:.3e} (expected ~0: the off-by-one only affects the "
        "bottom of the transform table)"
    )
    print(f"\nnscore_bug_impact.csv: {OUT_CSV}")
    return df


if __name__ == "__main__":
    main()
