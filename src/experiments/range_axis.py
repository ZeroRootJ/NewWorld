"""Range axis experiment (Deliverable 2, docs/experiment_context.md):
one-factor-at-a-time variation of the ground-truth variogram range (hmaj1 ==
hmin1, isotropic), holding every other base-case parameter -- nugget, sill
split, grid, distribution, TRUTH_SEED, SAMPLE_SEED, N_SAMPLES, and every
method's own search/tuning constants -- fixed at their base-case values.

Axis definition (user decision 2026-09-14, NOT to be re-litigated here)
----------------------------------------------------------------------
ALL_RANGE_VALUES = 100, 200, ..., 800 m -- 8 evenly spaced levels, one
ground-truth realization each (TRUTH_SEED=101, as in the base case).

The earlier range=50 m level was DROPPED from the axis by the user ("too
short to be a meaningful level"). Its raw runs are left completely untouched
in results/raw/ (originals are never modified or deleted -- CLAUDE.md), and
their paths are recorded in results/processed/range_axis/excluded_runs.json
so the provenance is not lost, but they are excluded from source_runs.json,
metrics.csv, the figures and the report.

Run reuse (avoid recomputing identical runs)
--------------------------------------------
- range=300 m IS the base case. Its four pinned runs are reused verbatim
  from results/processed/base_case/source_runs.json.
- range=800 m was already run in the first range-axis execution. Its four
  runs are reused verbatim from the pre-existing
  results/processed/range_axis/source_runs.json.
- Only NEW_RANGE_VALUES = 100, 200, 400, 500, 600, 700 are actually executed
  by this script (6 levels x 4 methods = 24 fresh runs).

Every reused run is verified (not assumed) to belong to its axis level
before being written into the new source_runs.json: the manifest's recorded
range is checked where present, and -- decisively, because the base-case
rbf_bootstrap/gp_mle manifests predate the top-level hmaj1 field -- each
run's recorded samples.csv is compared against the conditioning samples
regenerated at that range. Sample VALUES are drawn from the truth field, so
they differ between ranges; an exact match therefore pins the run to the
right ground truth.

This script does NOT re-implement any method -- it simply calls each
method's already-parameterized main() (kriging.main / sgs.main /
rbf_bootstrap.main / gp_mle.main) once per new range value, in the fixed
order [kriging, sgs, rbf_bootstrap, gp_mle].

Run with: .venv/Scripts/python.exe -m src.experiments.range_axis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.base_case_conditioning import (
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_conditioning_data,
)
from src.experiments import gp_mle, kriging, rbf_bootstrap, sgs

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"
BASE_CASE_PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "base_case"

# Full axis: 100 m .. 800 m in 100 m steps (user decision, see docstring).
ALL_RANGE_VALUES = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]

# Levels this script actually executes (the rest are reused, see below).
NEW_RANGE_VALUES = [100.0, 200.0, 400.0, 500.0, 600.0, 700.0]

# axis_level -> where its already-existing runs come from.
BASE_CASE_AXIS_LEVEL = "300"
REUSED_FROM_RANGE_AXIS_LEVEL = "800"

# Axis level dropped by the user; its raw runs are preserved but excluded.
EXCLUDED_AXIS_LEVEL = "50"

METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]


def run_one_range(hmaj1: float) -> dict:
    """Run all 4 methods at variogram range hmaj1 (== hmin1, isotropic),
    truth_seed=TRUTH_SEED (base_case_conditioning's single source of truth),
    and every other parameter left at each method's own base-case default.

    Returns {"kriging": Path(...), ...}.
    """
    hmin1 = hmaj1  # isotropic range axis

    run_dirs = {}

    print(f"\n=== range = {hmaj1:g} m: kriging ===")
    kdir, _, _ = kriging.main(truth_seed=TRUTH_SEED, hmaj1=hmaj1, hmin1=hmin1)
    run_dirs["kriging"] = kdir

    print(f"\n=== range = {hmaj1:g} m: sgs ===")
    sdir, _, _ = sgs.main(truth_seed=TRUTH_SEED, hmaj1=hmaj1, hmin1=hmin1)
    run_dirs["sgs"] = sdir

    print(f"\n=== range = {hmaj1:g} m: rbf_bootstrap ===")
    rdir, _, _ = rbf_bootstrap.main(truth_seed=TRUTH_SEED, hmaj1=hmaj1, hmin1=hmin1)
    run_dirs["rbf_bootstrap"] = rdir

    print(f"\n=== range = {hmaj1:g} m: gp_mle ===")
    gdir, _, _ = gp_mle.main(truth_seed=TRUTH_SEED, hmaj1=hmaj1, hmin1=hmin1)
    run_dirs["gp_mle"] = gdir

    return run_dirs


def verify_reused_run(rel_run_dir: str, hmaj1: float, method: str) -> None:
    """Verify that an existing run directory really belongs to variogram
    range ``hmaj1`` (and to this axis's TRUTH_SEED/SAMPLE_SEED) before it is
    reused as an axis level. Raises on any mismatch -- a silently mislabeled
    axis level would be the single most damaging error this table can carry.
    """
    run_dir = _REPO_ROOT / rel_run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["params"]

    if params.get("truth_seed") != TRUTH_SEED or params.get("sample_seed") != SAMPLE_SEED:
        raise ValueError(
            f"{rel_run_dir}: truth_seed/sample_seed "
            f"({params.get('truth_seed')}/{params.get('sample_seed')}) do not match this "
            f"axis's ({TRUTH_SEED}/{SAMPLE_SEED})."
        )

    # Recorded range, where the manifest has it. The base-case rbf_bootstrap /
    # gp_mle runs predate the top-level hmaj1 field and carry no variogram
    # dict at all (those methods have no variogram of their own), so this
    # check is best-effort; the samples check below is the decisive one.
    recorded = params.get("hmaj1")
    if recorded is None:
        recorded = params.get("variogram", {}).get("hmaj1")
    if recorded is not None and not np.isclose(float(recorded), hmaj1):
        raise ValueError(
            f"{rel_run_dir}: manifest records range {recorded} m but is being "
            f"reused as the {hmaj1:g} m axis level."
        )

    # Decisive check: the conditioning sample VALUES are read off the truth
    # field, so they are range-specific. Regenerate them and require an exact
    # match against what this run recorded.
    _, samples_df = get_base_case_conditioning_data(
        sample_seed=SAMPLE_SEED, truth_seed=TRUTH_SEED, hmaj1=hmaj1, hmin1=hmaj1
    )
    recorded_samples = pd.read_csv(run_dir / "samples.csv")
    if len(recorded_samples) != len(samples_df) or not np.allclose(
        recorded_samples[["X", "Y", VCOL]].values, samples_df[["X", "Y", VCOL]].values
    ):
        raise ValueError(
            f"{rel_run_dir}: recorded samples.csv does not match the conditioning "
            f"samples regenerated at range={hmaj1:g} m -- this run was NOT produced "
            "from that ground truth."
        )
    print(f"  verified reused {method} run for range={hmaj1:g}m: {rel_run_dir}")


def _rel(p) -> str:
    return str(Path(p).relative_to(_REPO_ROOT)).replace("\\", "/")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    source_runs_path = PROCESSED_DIR / "source_runs.json"
    previous_source_runs = {}
    if source_runs_path.exists():
        previous_source_runs = json.loads(source_runs_path.read_text(encoding="utf-8"))

    # --- Preserve the provenance of the dropped range=50 level -------------
    # (raw runs themselves are never touched -- see module docstring)
    if EXCLUDED_AXIS_LEVEL in previous_source_runs:
        excluded_path = PROCESSED_DIR / "excluded_runs.json"
        excluded_path.write_text(
            json.dumps(
                {
                    "note": (
                        "Runs that exist in results/raw/ but are deliberately NOT part "
                        "of the range axis. range=50m was dropped from the axis by the "
                        "user on 2026-09-14 (too short to be a meaningful level); the "
                        "raw run directories below are preserved unmodified but are "
                        "excluded from source_runs.json, metrics.csv, the figures and "
                        "the report."
                    ),
                    "excluded_levels": {
                        EXCLUDED_AXIS_LEVEL: previous_source_runs[EXCLUDED_AXIS_LEVEL]
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Recorded dropped range={EXCLUDED_AXIS_LEVEL}m runs in {excluded_path}")

    source_runs = {}

    # --- Reuse: range=300 == the base case ---------------------------------
    base_case_source_runs = json.loads(
        (BASE_CASE_PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8")
    )
    print(f"\nReusing the base case as the range={BASE_CASE_AXIS_LEVEL}m axis level:")
    source_runs[BASE_CASE_AXIS_LEVEL] = {
        m: base_case_source_runs[m]["run_dir"] for m in METHODS
    }
    for m in METHODS:
        verify_reused_run(source_runs[BASE_CASE_AXIS_LEVEL][m], float(BASE_CASE_AXIS_LEVEL), m)

    # --- Reuse: range=800 from the previous range-axis execution ------------
    if REUSED_FROM_RANGE_AXIS_LEVEL not in previous_source_runs:
        raise ValueError(
            f"No existing range={REUSED_FROM_RANGE_AXIS_LEVEL}m runs found in "
            f"{source_runs_path} to reuse."
        )
    print(f"\nReusing existing range={REUSED_FROM_RANGE_AXIS_LEVEL}m runs:")
    source_runs[REUSED_FROM_RANGE_AXIS_LEVEL] = dict(
        previous_source_runs[REUSED_FROM_RANGE_AXIS_LEVEL]
    )
    for m in METHODS:
        verify_reused_run(
            source_runs[REUSED_FROM_RANGE_AXIS_LEVEL][m], float(REUSED_FROM_RANGE_AXIS_LEVEL), m
        )

    # --- Execute the new levels --------------------------------------------
    for r in NEW_RANGE_VALUES:
        run_dirs = run_one_range(r)
        source_runs[str(int(r))] = {m: _rel(run_dirs[m]) for m in METHODS}

    # Sort by numeric axis level for readability.
    source_runs = {str(int(r)): source_runs[str(int(r))] for r in ALL_RANGE_VALUES}

    missing = [lvl for lvl in source_runs if set(source_runs[lvl]) != set(METHODS)]
    if missing:
        raise ValueError(f"axis level(s) {missing} are missing a method run.")

    with open(source_runs_path, "w", encoding="utf-8") as f:
        json.dump(source_runs, f, indent=2)

    print("\nrange_axis source runs (8 levels x 4 methods):")
    print(json.dumps(source_runs, indent=2))
    print(f"\nsource_runs.json: {source_runs_path}")

    return source_runs


if __name__ == "__main__":
    main()
