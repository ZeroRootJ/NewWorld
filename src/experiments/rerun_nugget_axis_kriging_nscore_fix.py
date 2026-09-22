"""Re-run ONLY the kriging arm of the nugget axis after the
``geostats.nscore`` off-by-one repair (src/nscore.py, 2026-09-22), and
re-pin ``results/processed/nugget_axis/source_runs.json`` to the new runs.

WHY ONLY KRIGING, AND WHY ALL SIX LEVELS
----------------------------------------
``src/experiments/kriging.py`` was the only place in this repository that
called ``geostats.nscore`` (verified by grep over src/, results/, tests/ --
there is exactly one call site). SGS goes through ``geostats.sgsim`` with
``itrans=1``, which performs its own internal normal-score transform on a
separate code path and never touches the buggy function; RBF+bootstrap and
GP-MLE work in physical porosity units and have no normal-score step at all.
So sgs / rbf_bootstrap / gp_mle are provably unaffected and are NOT re-run --
their existing pins are carried over unchanged and are deep-compared
(byte-for-byte on the pinned path strings) between the old and new
``source_runs.json`` by this script.

All SIX kriging levels are re-run, not just the two where the symptom was
visible (nug=0.2 / 0.3). The off-by-one mis-assigns the minimum datum's
normal score in EVERY run unconditionally -- it is only spectacular when the
2nd and 3rd smallest sample values happen to be nearly tied. The other four
levels were wrong by a smaller amount, not right.

WHAT IS AND IS NOT TOUCHED
--------------------------
``results/raw/`` is append-only in this project: the superseded kriging runs
are left exactly where they are, and this script only adds new timestamped
run directories. The pre-fix pin set is preserved for provenance in
``results/processed/nugget_axis/source_runs_superseded_20260922_nscore_bug.json``
and is quantified in ``results/processed/nscore_bug_impact.csv``.

EXECUTION PATH
--------------
Runs through ``_axis_parallel.run_levels_parallel`` with ``methods=["kriging"]``
so the new runs are produced by the same spawned-worker / BLAS-capped path
that produced the original nugget-axis runs. (kriging is geostatspy/numba and
is bit-for-bit identical with or without BLAS thread capping -- see
_axis_parallel.py's 2026-09-16 note -- so this is about keeping the
provenance identical, not about the numbers.)

Run with:
.venv/Scripts/python.exe -m src.experiments.rerun_nugget_axis_kriging_nscore_fix
"""

import json
from pathlib import Path

from src.experiments._axis_parallel import run_levels_parallel
from src.experiments.base_case_conditioning import TRUTH_SEED
from src.experiments.nugget_axis import (
    ALL_NUGGET_VALUES,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    METHODS,
    level_key,
    verify_run,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"
SOURCE_RUNS_PATH = PROCESSED_DIR / "source_runs.json"
SUPERSEDED_PATH = PROCESSED_DIR / "source_runs_superseded_20260922_nscore_bug.json"

UNCHANGED_METHODS = [m for m in METHODS if m != "kriging"]


def _rel(p) -> str:
    return str(Path(p).relative_to(_REPO_ROOT)).replace("\\", "/")


def main():
    old = json.loads(SOURCE_RUNS_PATH.read_text(encoding="utf-8"))
    if not SUPERSEDED_PATH.exists():
        # Provenance snapshot of the pre-fix pins. Written once; never
        # overwritten, so re-running this script cannot lose the original.
        SUPERSEDED_PATH.write_text(json.dumps(old, indent=2), encoding="utf-8")
        print(f"wrote pre-fix pin snapshot: {SUPERSEDED_PATH}")

    level_kwargs = {
        level_key(nug): dict(
            truth_seed=TRUTH_SEED,
            hmaj1=AXIS_HMAJ1,
            hmin1=AXIS_HMIN1,
            nug=float(nug),
        )
        for nug in ALL_NUGGET_VALUES
    }
    new_run_dirs = run_levels_parallel(level_kwargs, methods=["kriging"])

    new = {}
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        entry = dict(old[level])  # carry sgs / rbf_bootstrap / gp_mle unchanged
        entry["kriging"] = _rel(new_run_dirs[level]["kriging"])
        new[level] = entry

    # --- Deep comparison: nothing but the kriging pin may change ----------
    for level in old:
        if set(old[level]) != set(new[level]):
            raise ValueError(
                f"level {level}: method set changed ({sorted(old[level])} -> "
                f"{sorted(new[level])})."
            )
        for m in UNCHANGED_METHODS:
            if old[level][m] != new[level][m]:
                raise ValueError(
                    f"level {level}: {m} pin changed ({old[level][m]} -> "
                    f"{new[level][m]}) -- only kriging may be re-pinned here."
                )
        if old[level]["kriging"] == new[level]["kriging"]:
            raise ValueError(
                f"level {level}: kriging pin did not change -- the re-run did not "
                "produce a new run directory."
            )
    print(
        f"deep comparison OK: {len(UNCHANGED_METHODS) * len(old)} non-kriging pins "
        "are byte-identical to the pre-fix source_runs.json; only the 6 kriging "
        "pins changed."
    )

    print("\nVerifying every new kriging run against its axis level:")
    for nug in ALL_NUGGET_VALUES:
        verify_run(new[level_key(nug)]["kriging"], float(nug), "kriging")

    # --- Report the nscore repairs that actually fired --------------------
    print("\nnscore off-by-one repairs applied in the new runs:")
    repair_rows = []
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        man = json.loads(
            (_REPO_ROOT / new[level]["kriging"] / "manifest.json").read_text(encoding="utf-8")
        )
        c = man["params"]["nscore_corrections"]
        repair_rows.append((level, c))
        print(
            f"  nug={level}: n_corrected={c['n_corrected']}, "
            f"raw={c['raw_normal_scores']}, corrected={c['corrected_normal_scores']}, "
            f"max_abs_correction={c['max_abs_correction']:.6g}"
        )

    SOURCE_RUNS_PATH.write_text(json.dumps(new, indent=2), encoding="utf-8")
    print(f"\nre-pinned: {SOURCE_RUNS_PATH}")
    print(json.dumps(new, indent=2))
    return new, repair_rows


if __name__ == "__main__":
    main()
