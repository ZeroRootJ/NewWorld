"""Run the 10% / 20% sample-density levels (user request 2026-09-21) for the
two density-related axes and ADD them to the pinned source_runs.json files,
leaving every existing level entry untouched.

Levels (sample_density_axis.EXTRA_DENSITY_LEVELS, registered in SAMPLE_COUNTS):
    "20" -> n_samples_requested = 500 (20% of the 2500-cell grid)
    "10" -> n_samples_requested = 250 (10%)

Two axes (all 4 methods -- kriging / SGS / RBF+bootstrap / GP-MLE -- with the
CURRENT code and tuning constants; nothing but n_samples varies relative to the
existing levels, one-factor-at-a-time):

  ``--axes density``    single-run sample-DENSITY axis: one run per new level
                        at that axis's own SAMPLE_SEED (=20) -> 2 x 4 = 8 runs.
                        Merged into results/processed/sample_density_axis/
                        source_runs.json.
  ``--axes replicate``  sample-REPLICATE axis: the same 10 sample_seed
                        replicates 1001..1010 at each new level, same ground
                        truth (TRUTH_SEED=101, range 300 m) -> 2 x 10 x 4 = 80
                        runs. Merged into results/processed/
                        sample_replicate_axis/source_runs.json.

Why this script exists instead of the axes' own ``main()``: the replicate
axis's ``main()`` rebuilds source_runs.json from the frozen file
source_runs_level5_n125.json (OLD level-5 RBF runs) and would silently revert
the current level-5 RBF pins; the density axis's ``main()`` rewrites
sample_overlap.json / re-verifies the base case. This script only ADDS the
entries of the new levels: the existing entries are re-read, must be preserved
exactly (checked below by comparing their serialized JSON before/after), and a
new-level key that already exists in source_runs.json aborts the run rather
than being overwritten. Raw run directories are new timestamped dirs; nothing
under results/raw is modified.

Every new run is verified (not assumed) against its own (level, seed)
regenerated conditioning samples (samples.csv atol=1e-10, manifest seeds,
n_samples_requested/actual) before it is written into source_runs.json. A
small record (seeds, requested/actual counts, run dirs, wall time) is written
to ``extra_density_levels_record.json`` in each axis's processed directory.

Run with (either axis, or both in one invocation):
    .venv/Scripts/python.exe -m src.experiments.run_extra_density_levels --axes density
    .venv/Scripts/python.exe -m src.experiments.run_extra_density_levels --axes replicate
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments._axis_parallel import run_levels_parallel
from src.experiments.base_case_conditioning import (
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.sample_density_axis import (
    ALL_AXIS_LEVELS,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    EXTENDED_AXIS_LEVELS,
    EXTRA_DENSITY_LEVELS,
    METHODS,
    SAMPLE_COUNTS,
    SAMPLES_MATCH_ATOL,
)
from src.experiments.sample_replicate_axis import (
    REPLICATE_IDS,
    REPLICATE_SEED,
    verify_run_against_replicate,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DENSITY_DIR = _REPO_ROOT / "results" / "processed" / "sample_density_axis"
REPLICATE_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"
RECORD_NAME = "extra_density_levels_record.json"


def _rel(p) -> str:
    return str(Path(p).relative_to(_REPO_ROOT)).replace("\\", "/")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def verify_density_run(rel_run_dir: str, axis_level: str, method: str) -> int:
    """Density-axis analogue of sample_replicate_axis.verify_run_against_replicate:
    the run must carry the axis's TRUTH_SEED/SAMPLE_SEED/range/n_samples_requested
    and its samples.csv must match the regenerated conditioning samples.
    Returns n_samples_actual."""
    n_requested = SAMPLE_COUNTS[axis_level]
    run_dir = _REPO_ROOT / rel_run_dir
    params = _load(run_dir / "manifest.json")["params"]
    if params.get("truth_seed") != TRUTH_SEED or params.get("sample_seed") != SAMPLE_SEED:
        raise ValueError(
            f"{rel_run_dir}: truth_seed/sample_seed "
            f"({params.get('truth_seed')}/{params.get('sample_seed')}) != "
            f"({TRUTH_SEED}/{SAMPLE_SEED})."
        )
    if params.get("n_samples_requested") is not None and int(
        params["n_samples_requested"]
    ) != n_requested:
        raise ValueError(
            f"{rel_run_dir}: n_samples_requested={params['n_samples_requested']} != {n_requested}."
        )
    if params.get("hmaj1") is not None and not np.isclose(float(params["hmaj1"]), AXIS_HMAJ1):
        raise ValueError(f"{rel_run_dir}: range {params['hmaj1']} != {AXIS_HMAJ1:g}.")
    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)
    samples_df = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED, n_samples=n_requested)
    recorded = pd.read_csv(run_dir / "samples.csv")
    if len(recorded) != len(samples_df) or not np.allclose(
        recorded[["X", "Y", VCOL]].values,
        samples_df[["X", "Y", VCOL]].values,
        rtol=0.0,
        atol=SAMPLES_MATCH_ATOL,
    ):
        raise ValueError(
            f"{rel_run_dir}: samples.csv does not match the conditioning samples regenerated "
            f"for level '{axis_level}' (n_requested={n_requested}, sample_seed={SAMPLE_SEED})."
        )
    if params.get("n_samples_actual") is not None and int(
        params["n_samples_actual"]
    ) != len(samples_df):
        raise ValueError(
            f"{rel_run_dir}: manifest n_samples_actual={params['n_samples_actual']} != "
            f"regenerated {len(samples_df)}."
        )
    print(
        f"  verified {method} density-axis run, level '{axis_level}' "
        f"(n_actual={len(samples_df)}): {rel_run_dir}"
    )
    return len(samples_df)


def _merge_source_runs(path: Path, new_entries: dict, replicate_axis: bool) -> None:
    """Add ``new_entries`` ({level: entry}) to ``path`` (a source_runs.json),
    preserving every existing level entry exactly and ordering levels dense ->
    sparse (EXTENDED_AXIS_LEVELS). Aborts if a new level already exists."""
    existing = _load(path)
    clash = [lvl for lvl in new_entries if lvl in existing]
    if clash:
        raise ValueError(f"{path}: level(s) {clash} already present -- refusing to overwrite.")
    unknown = set(existing) - set(EXTENDED_AXIS_LEVELS)
    if unknown:
        raise ValueError(f"{path}: unexpected level keys {sorted(unknown)}.")
    before = {lvl: json.dumps(existing[lvl], indent=2) for lvl in existing}

    merged = {**existing, **new_entries}
    merged = {lvl: merged[lvl] for lvl in EXTENDED_AXIS_LEVELS if lvl in merged}

    for lvl, txt in before.items():
        if json.dumps(merged[lvl], indent=2) != txt:  # pragma: no cover - safety net
            raise RuntimeError(f"{path}: existing level '{lvl}' entry changed during merge.")
    _dump(path, merged)


def _update_record(path: Path, key: str, payload: dict) -> None:
    rec = _load(path) if path.exists() else {}
    rec[key] = payload
    _dump(path, rec)


def run_density_axis(levels) -> None:
    t0 = time.time()
    source_path = DENSITY_DIR / "source_runs.json"
    if any(lvl in _load(source_path) for lvl in levels):
        raise ValueError(f"{source_path} already has one of {levels}; aborting before running.")
    level_kwargs = {
        lvl: dict(
            truth_seed=TRUTH_SEED,
            hmaj1=AXIS_HMAJ1,
            hmin1=AXIS_HMIN1,
            n_samples=SAMPLE_COUNTS[lvl],
        )
        for lvl in levels
    }
    run_dirs = run_levels_parallel(level_kwargs, methods=METHODS)
    entries = {lvl: {m: _rel(run_dirs[lvl][m]) for m in METHODS} for lvl in levels}

    n_actual = {}
    for lvl in levels:
        counts = {m: verify_density_run(entries[lvl][m], lvl, m) for m in METHODS}
        if len(set(counts.values())) != 1:
            raise ValueError(f"level '{lvl}': n_samples_actual differs across methods: {counts}")
        n_actual[lvl] = next(iter(counts.values()))

    _merge_source_runs(source_path, entries, replicate_axis=False)
    _update_record(
        DENSITY_DIR / RECORD_NAME,
        "density_axis",
        {
            "levels": list(levels),
            "sample_seed": SAMPLE_SEED,
            "truth_seed": TRUTH_SEED,
            "hmaj1": AXIS_HMAJ1,
            "n_samples_requested": {lvl: SAMPLE_COUNTS[lvl] for lvl in levels},
            "n_samples_actual": n_actual,
            "run_dirs": entries,
            "wall_time_s": round(time.time() - t0, 1),
        },
    )
    print(f"density axis: {len(levels)} levels x {len(METHODS)} methods done in "
          f"{time.time() - t0:.1f}s; n_samples_actual={n_actual}")


def run_replicate_axis(levels) -> None:
    t0 = time.time()
    source_path = REPLICATE_DIR / "source_runs.json"
    if any(lvl in _load(source_path) for lvl in levels):
        raise ValueError(f"{source_path} already has one of {levels}; aborting before running.")
    level_kwargs = {
        f"{lvl}__{rep_id}": dict(
            truth_seed=TRUTH_SEED,
            hmaj1=AXIS_HMAJ1,
            hmin1=AXIS_HMIN1,
            n_samples=SAMPLE_COUNTS[lvl],
            sample_seed=seed,
        )
        for lvl in levels
        for rep_id, seed in REPLICATE_SEED.items()
    }
    run_dirs = run_levels_parallel(level_kwargs, methods=METHODS)
    entries = {
        lvl: {
            rep_id: {m: _rel(run_dirs[f"{lvl}__{rep_id}"][m]) for m in METHODS}
            for rep_id in REPLICATE_IDS
        }
        for lvl in levels
    }

    print("\nVerifying every new run against its own (level, replicate) conditioning samples...")
    n_actual = {lvl: {} for lvl in levels}
    for lvl in levels:
        for rep_id in REPLICATE_IDS:
            counts = {
                m: verify_run_against_replicate(entries[lvl][rep_id][m], lvl, rep_id, m)[
                    "n_samples_actual"
                ]
                for m in METHODS
            }
            if len(set(counts.values())) != 1:
                raise ValueError(
                    f"level '{lvl}' {rep_id}: n_samples_actual differs across methods: {counts}"
                )
            n_actual[lvl][rep_id] = next(iter(counts.values()))

    _merge_source_runs(source_path, entries, replicate_axis=True)
    _update_record(
        REPLICATE_DIR / RECORD_NAME,
        "replicate_axis",
        {
            "levels": list(levels),
            "sample_seed_replicates": REPLICATE_SEED,
            "truth_seed": TRUTH_SEED,
            "hmaj1": AXIS_HMAJ1,
            "n_samples_requested": {lvl: SAMPLE_COUNTS[lvl] for lvl in levels},
            "n_samples_actual": n_actual,
            "run_dirs": entries,
            "wall_time_s": round(time.time() - t0, 1),
        },
    )
    print(f"replicate axis: {len(levels)} levels x {len(REPLICATE_IDS)} replicates x "
          f"{len(METHODS)} methods done in {time.time() - t0:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--axes", default="density,replicate",
                    help="comma list of: density, replicate")
    ap.add_argument("--levels", default=",".join(EXTRA_DENSITY_LEVELS),
                    help=f"comma list from {EXTRA_DENSITY_LEVELS}")
    args = ap.parse_args(argv)
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    if not set(axes) <= {"density", "replicate"}:
        raise ValueError(f"unknown axes {axes}")
    if not set(levels) <= set(EXTRA_DENSITY_LEVELS) or set(levels) & set(ALL_AXIS_LEVELS):
        raise ValueError(f"levels must come from {EXTRA_DENSITY_LEVELS}; got {levels}")
    if "density" in axes:
        run_density_axis(levels)
    if "replicate" in axes:
        run_replicate_axis(levels)


if __name__ == "__main__":
    main()
