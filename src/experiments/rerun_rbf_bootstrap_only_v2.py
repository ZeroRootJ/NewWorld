"""Re-run ONLY the RBF+bootstrap runs of EVERY axis this project currently
has, and re-point ONLY the ``rbf_bootstrap`` entries of each processed
``source_runs.json`` at the new runs.

Why this exists instead of reusing rerun_rbf_bootstrap_only.py
----------------------------------------------------------------
``rerun_rbf_bootstrap_only.py`` (2026-09-17/2026-09-21) covers the base case
+ the 8-level range axis + the ORIGINAL 3 sample-density levels ("5"/"2"/"1")
+ their 3x10 replicate cells -- 40 runs. It predates the 10%/20% density
levels added on 2026-09-21/2026-09-22 (src/experiments/run_extra_density_levels.py),
so it does not touch those. This script is the SAME re-run, extended to also
cover the "20"/"10" density levels and their replicate cells, triggered by
the 2026-09-22 ``EPSILON_GRID`` resolution increase in
src/experiments/rbf_bootstrap.py (N_EPSILON 25 -> 121, see that module's
docstring for the measured justification) -- a change to RBF+bootstrap's OWN
tuning grid, so per this project's standing convention (2026-09-17/2026-09-21
entries in docs/progress.md) only RBF+bootstrap's pinned runs need to move;
kriging / SGS / GP-MLE runs, code and pins are all left untouched.

Runs (62 total):
    base case                        1   (= range=300 = density "5", shared)
    range axis (excl. 300, reused)    7   (100/200/400/500/600/700/800)
    density axis (excl. "5", reused)  4   ("20"/"10"/"2"/"1")
    replicate axis                   50   (5 density levels x 10 sample_seed
                                            replicates 1001-1010 each)
    -----------------------------------
    total                            62

Every new run is VERIFIED (not assumed) against its own axis level's
regenerated conditioning samples before being written into any
source_runs.json, reusing this project's ALREADY-TESTED verification
functions rather than re-implementing them:
  - base case / range axis levels -> src.experiments.range_axis.verify_reused_run
    (checks truth_seed/sample_seed/hmaj1 and an exact samples.csv match)
  - density axis levels (incl. "20"/"10")
        -> src.experiments.run_extra_density_levels.verify_density_run
  - replicate axis cells -> src.experiments.sample_replicate_axis.verify_run_against_replicate

TRAP THIS SCRIPT AVOIDS (documented in docs/progress.md 2026-09-21): the
sample-replicate axis's OWN ``main()`` rebuilds its source_runs.json from a
frozen snapshot ``source_runs_level5_n125.json`` (an old level-5 RBF pin) and
would silently revert the level-5 RBF entries back to a stale run. This
script never reads or writes that frozen file -- it merges new RBF run paths
directly into the CURRENT ``source_runs.json`` files, level by level, exactly
as ``rerun_rbf_bootstrap_only.py`` already does.

Run with: .venv/Scripts/python.exe -m src.experiments.rerun_rbf_bootstrap_only_v2
"""

import json
import time
from pathlib import Path

from src.experiments._axis_parallel import run_levels_parallel
from src.experiments.base_case_conditioning import TRUTH_SEED
from src.experiments.range_axis import ALL_RANGE_VALUES, verify_reused_run
from src.experiments.run_extra_density_levels import verify_density_run
from src.experiments.sample_density_axis import (
    AXIS_HMAJ1,
    AXIS_HMIN1,
    EXTENDED_AXIS_LEVELS,
    SAMPLE_COUNTS,
)
from src.experiments.sample_replicate_axis import (
    REPLICATE_IDS,
    REPLICATE_SEED,
    verify_run_against_replicate,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = _REPO_ROOT / "results" / "processed"

METHOD = "rbf_bootstrap"
BASE_CASE_RANGE = 300.0
BASE_KEY = "base"
BASE_DENSITY_LEVEL = "5"
DENSITY_LEVELS_TO_RUN = [lvl for lvl in EXTENDED_AXIS_LEVELS if lvl != BASE_DENSITY_LEVEL]  # 20,10,2,1


def _rel(p) -> str:
    return str(Path(p).relative_to(_REPO_ROOT)).replace("\\", "/")


def build_level_kwargs() -> dict:
    level_kwargs = {BASE_KEY: {}}  # rbf_bootstrap.main() defaults == base case

    for r in ALL_RANGE_VALUES:
        if r == BASE_CASE_RANGE:
            continue  # shares the base-case run
        level_kwargs[f"range__{int(r)}"] = dict(truth_seed=TRUTH_SEED, hmaj1=r, hmin1=r)

    for lvl in DENSITY_LEVELS_TO_RUN:
        level_kwargs[f"density__{lvl}"] = dict(
            truth_seed=TRUTH_SEED,
            hmaj1=AXIS_HMAJ1,
            hmin1=AXIS_HMIN1,
            n_samples=SAMPLE_COUNTS[lvl],
        )

    for lvl in EXTENDED_AXIS_LEVELS:  # 20, 10, 5, 2, 1 -- ALL 5 levels, incl. "5"
        for rep_id, seed in REPLICATE_SEED.items():
            level_kwargs[f"replicate__{lvl}__{rep_id}"] = dict(
                truth_seed=TRUTH_SEED,
                hmaj1=AXIS_HMAJ1,
                hmin1=AXIS_HMIN1,
                n_samples=SAMPLE_COUNTS[lvl],
                sample_seed=seed,
            )

    return level_kwargs


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    t0 = time.time()
    level_kwargs = build_level_kwargs()
    expected_total = 1 + (len(ALL_RANGE_VALUES) - 1) + len(DENSITY_LEVELS_TO_RUN) + (
        len(EXTENDED_AXIS_LEVELS) * len(REPLICATE_IDS)
    )
    if len(level_kwargs) != expected_total:
        raise ValueError(
            f"build_level_kwargs() produced {len(level_kwargs)} entries; expected {expected_total}."
        )
    print(f"{len(level_kwargs)} RBF+bootstrap runs to execute.")
    results = run_levels_parallel(level_kwargs, methods=(METHOD,))
    run = {k: _rel(v[METHOD]) for k, v in results.items()}

    # --- verify every new run against its own axis level's conditioning ----
    # samples BEFORE any source_runs.json is touched.
    print("\nVerifying every new run against its own axis level's regenerated conditioning samples...")
    verify_reused_run(run[BASE_KEY], BASE_CASE_RANGE, METHOD)
    for r in ALL_RANGE_VALUES:
        if r == BASE_CASE_RANGE:
            continue
        verify_reused_run(run[f"range__{int(r)}"], r, METHOD)
    for lvl in DENSITY_LEVELS_TO_RUN:
        verify_density_run(run[f"density__{lvl}"], lvl, METHOD)
    for lvl in EXTENDED_AXIS_LEVELS:
        for rep_id in REPLICATE_IDS:
            verify_run_against_replicate(run[f"replicate__{lvl}__{rep_id}"], lvl, rep_id, METHOD)
    print("all 62 new runs verified.")

    # --- base case -----------------------------------------------------
    base_path = PROCESSED / "base_case" / "source_runs.json"
    base = _load(base_path)
    base[METHOD] = {"run_dir": run[BASE_KEY], "timestamp": Path(run[BASE_KEY]).name}
    _dump(base_path, base)

    # --- range axis ------------------------------------------------------
    range_path = PROCESSED / "range_axis" / "source_runs.json"
    rng = _load(range_path)
    for r in ALL_RANGE_VALUES:
        key = BASE_KEY if r == BASE_CASE_RANGE else f"range__{int(r)}"
        rng[str(int(r))][METHOD] = run[key]
    _dump(range_path, rng)

    # --- density axis (all 5 levels: "20"/"10"/"5"/"2"/"1") ----------------
    dens_path = PROCESSED / "sample_density_axis" / "source_runs.json"
    dens = _load(dens_path)
    dens[BASE_DENSITY_LEVEL][METHOD] = run[BASE_KEY]
    for lvl in DENSITY_LEVELS_TO_RUN:
        dens[lvl][METHOD] = run[f"density__{lvl}"]
    _dump(dens_path, dens)

    # --- replicate axis (all 5 levels x 10 replicates; frozen
    # source_runs_level5_n125.json is NOT read/touched -- see module docstring)
    rep_path = PROCESSED / "sample_replicate_axis" / "source_runs.json"
    rep = _load(rep_path)
    for lvl in EXTENDED_AXIS_LEVELS:
        for rep_id in REPLICATE_IDS:
            rep[lvl][rep_id][METHOD] = run[f"replicate__{lvl}__{rep_id}"]
    _dump(rep_path, rep)

    print(json.dumps(run, indent=2))
    print(f"rerun_rbf_bootstrap_only_v2 total: {time.time() - t0:.1f}s")
    return run


if __name__ == "__main__":
    main()
