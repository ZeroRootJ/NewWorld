"""Sample-seed replicate axis (user task 2026-09-18): how much does the
4-method comparison (kriging / SGS / RBF+bootstrap / GP-MLE) move when ONLY
the conditioning-sample LOCATIONS change, at a FIXED ground truth and a
FIXED requested sample count?

This is a different question from the sample-density axis
(src/experiments/sample_density_axis.py), which varies n_samples across 3
levels at ONE fixed sample_seed per level (so any level-to-level difference
there mixes "less data" with "the data landed in different places", as that
module's docstring documents). Here n_samples is held fixed at the base
case's 125 (the "5%" density level) and ONLY sample_seed varies, across 10
independent replicates -- isolating the sample-PLACEMENT effect on its own.

Held fixed across all 10 replicates (one-factor-at-a-time):
  - TRUTH_SEED = 101 (base_case_conditioning.TRUTH_SEED) -- literally the
    same ground-truth field for every replicate.
  - variogram range hmaj1 = hmin1 = 300 m (reused, not redeclared, from
    src.experiments.sample_density_axis.AXIS_HMAJ1/AXIS_HMIN1, which are
    themselves the base case's own HMAJ1/HMIN1).
  - n_samples REQUESTED = 125 (base_case_conditioning.N_SAMPLES, i.e. the
    sample-density axis's "5" level). ``random_interior_samples`` snaps
    draws to the nearest grid-cell centroid and drops duplicate-cell
    collisions, so the ACTUAL count used can be < 125 and can differ
    slightly between replicates (their raw draws collide on different
    cells) -- the actual count per replicate is measured and recorded
    below, never assumed to equal 125.
  - every method's own search/tuning constants (kriging's NDMIN/NDMAX/
    RADIUS, sgs's NDMAX/NODMAX/jitter, rbf_bootstrap's CV epsilon grid,
    gp_mle's kernel bounds/n_restarts, ...) stay at their base-case values
    -- untouched here, exactly as every other axis in this project requires.

The 10 sample-seed replicates
------------------------------
SAMPLE_SEED_REPLICATES = [1001, 1002, ..., 1010] -- ten consecutive literal
integers. No SeedSequence/derivation scheme is used; the exact values are
written here AND in this script's JSON output
(results/processed/sample_replicate_axis/replicate_seeds.json) for
auditability. They were chosen only to be trivially distinct from every
seed constant already pinned elsewhere in this project:
  - SAMPLE_SEED = 20 (base_case_conditioning.py, the base case's own sample
    locations at n_samples=125)
  - TRUTH_SEED = 101 (base_case_conditioning.py)
This module asserts at import time that none of SAMPLE_SEED_REPLICATES
collides with either of those, so this experiment cannot silently duplicate
or overwrite an existing pinned run's sample locations.

NOT one of these 10 (must not be folded in): the base case's own pinned
n_samples=125 run (results/processed/base_case/) uses SAMPLE_SEED=20 -- an
11th, PRE-EXISTING sample-location draw at the same requested n_samples. It
is a legitimate 11th point of reference for "how much does sample placement
move things at n=125", but it is deliberately NOT included in this
10-replicate set (no reuse shortcut here, unlike sample_density_axis.py's
5%-level base-case reuse): this experiment draws all 10 of its own sample
sets fresh, so it is a clean, uniformly-generated sample of "where do 125
samples land", uncontaminated by the specific SAMPLE_SEED=20 draw the rest
of the project is built around.

This script does not re-implement any method -- exactly like
sample_density_axis.py / range_axis.py, it calls each method's
already-parameterized main() (kriging.main / sgs.main / rbf_bootstrap.main /
gp_mle.main) once per (replicate x method). All 10 x 4 = 40 runs are fully
independent (different sample draws, disjoint output files, no shared
state), so they are submitted together to one process pool via
src/experiments/_axis_parallel.py's run_levels_parallel (2026-09-16
parallelization pattern, reused unchanged) rather than a sequential loop.

Every fresh run is VERIFIED (not assumed) to have actually used its
replicate's truth_seed/sample_seed/range/n_samples_requested, and its
recorded samples.csv is cross-checked against the conditioning samples
regenerated for that replicate's seed -- the same decisive check
sample_density_axis.py applies to its REUSED runs, applied here to every
FRESH run instead (a mislabeled replicate would be just as damaging).

Run with: .venv/Scripts/python.exe -m src.experiments.sample_replicate_axis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.base_case_conditioning import (
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.sample_density_axis import AXIS_HMAJ1, AXIS_HMIN1, SAMPLE_COUNTS
from src.experiments._axis_parallel import run_levels_parallel

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"

# The base case's own "5%" density level -- 125 requested samples. Reused
# (not redeclared) from sample_density_axis.SAMPLE_COUNTS so this axis's
# requested count cannot silently drift from the base case's.
N_SAMPLES_REQUESTED = SAMPLE_COUNTS["5"]

N_REPLICATES = 10
# See module docstring: ten literal, consecutive, auditable integers.
SAMPLE_SEED_REPLICATES = [1001 + i for i in range(N_REPLICATES)]
REPLICATE_IDS = [f"rep{i}" for i in range(N_REPLICATES)]
REPLICATE_SEED = dict(zip(REPLICATE_IDS, SAMPLE_SEED_REPLICATES))

if SAMPLE_SEED in SAMPLE_SEED_REPLICATES or TRUTH_SEED in SAMPLE_SEED_REPLICATES:
    raise ValueError(
        f"SAMPLE_SEED_REPLICATES={SAMPLE_SEED_REPLICATES} collides with an existing "
        f"pinned seed (base case SAMPLE_SEED={SAMPLE_SEED}, TRUTH_SEED={TRUTH_SEED}) -- "
        "refusing to silently duplicate or overwrite that run's sample locations."
    )
if len(set(SAMPLE_SEED_REPLICATES)) != N_REPLICATES:
    raise ValueError(f"SAMPLE_SEED_REPLICATES has duplicate values: {SAMPLE_SEED_REPLICATES}")

METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]

# Same float-text round-trip tolerance sample_density_axis.py uses for
# samples.csv comparisons.
SAMPLES_MATCH_ATOL = 1e-10


def _rel(p) -> str:
    return str(Path(p).relative_to(_REPO_ROOT)).replace("\\", "/")


def verify_fresh_run(rel_run_dir: str, replicate_id: str, method: str) -> dict:
    """Verify that a just-produced run really used this replicate's
    truth_seed/sample_seed/range/n_samples_requested, and that its recorded
    samples.csv matches the conditioning samples regenerated for this
    replicate's seed. Raises on any mismatch. Returns a small dict of facts
    (used to build replicate_seeds.json's n_samples_actual record)."""
    seed = REPLICATE_SEED[replicate_id]
    run_dir = _REPO_ROOT / rel_run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["params"]

    if params.get("truth_seed") != TRUTH_SEED or params.get("sample_seed") != seed:
        raise ValueError(
            f"{rel_run_dir}: truth_seed/sample_seed "
            f"({params.get('truth_seed')}/{params.get('sample_seed')}) do not match "
            f"replicate {replicate_id}'s expected ({TRUTH_SEED}/{seed})."
        )

    recorded_n_requested = params.get("n_samples_requested")
    if recorded_n_requested is not None and int(recorded_n_requested) != N_SAMPLES_REQUESTED:
        raise ValueError(
            f"{rel_run_dir}: manifest records n_samples_requested={recorded_n_requested}, "
            f"expected {N_SAMPLES_REQUESTED}."
        )

    recorded_range = params.get("hmaj1")
    if recorded_range is not None and not np.isclose(float(recorded_range), AXIS_HMAJ1):
        raise ValueError(
            f"{rel_run_dir}: manifest records range {recorded_range} m; this axis holds "
            f"the range fixed at {AXIS_HMAJ1:g} m."
        )

    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)
    samples_df = get_conditioning_samples(
        truth, sample_seed=seed, n_samples=N_SAMPLES_REQUESTED
    )
    recorded_n_actual = params.get("n_samples_actual")
    if recorded_n_actual is not None and int(recorded_n_actual) != len(samples_df):
        raise ValueError(
            f"{rel_run_dir}: manifest records n_samples_actual={recorded_n_actual}, but "
            f"regenerating replicate {replicate_id}'s (seed={seed}) conditioning samples "
            f"gives {len(samples_df)}."
        )

    recorded_samples = pd.read_csv(run_dir / "samples.csv")
    if len(recorded_samples) != len(samples_df) or not np.allclose(
        recorded_samples[["X", "Y", VCOL]].values,
        samples_df[["X", "Y", VCOL]].values,
        rtol=0.0,
        atol=SAMPLES_MATCH_ATOL,
    ):
        raise ValueError(
            f"{rel_run_dir}: recorded samples.csv does not match the conditioning samples "
            f"regenerated for replicate {replicate_id} (sample_seed={seed}) -- this run was "
            "NOT produced from that conditioning dataset."
        )

    print(
        f"  verified {method} run for {replicate_id} (sample_seed={seed}, "
        f"n_samples_actual={len(samples_df)}): {rel_run_dir}"
    )
    return {
        "replicate_id": replicate_id,
        "method": method,
        "sample_seed": seed,
        "n_samples_requested": N_SAMPLES_REQUESTED,
        "n_samples_actual": len(samples_df),
    }


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- 10 replicates x 4 methods (40 fully independent tasks) ------------
    level_kwargs = {
        rep_id: dict(
            truth_seed=TRUTH_SEED,
            hmaj1=AXIS_HMAJ1,
            hmin1=AXIS_HMIN1,
            n_samples=N_SAMPLES_REQUESTED,
            sample_seed=seed,
        )
        for rep_id, seed in REPLICATE_SEED.items()
    }
    run_dirs = run_levels_parallel(level_kwargs, methods=METHODS)

    source_runs = {}
    for rep_id in REPLICATE_IDS:
        source_runs[rep_id] = {m: _rel(run_dirs[rep_id][m]) for m in METHODS}

    missing = [r for r in source_runs if set(source_runs[r]) != set(METHODS)]
    if missing:
        raise ValueError(f"replicate(s) {missing} are missing a method run.")

    # --- Verify every fresh run against its replicate's own conditioning ---
    print("\nVerifying every fresh run against its replicate's conditioning samples...")
    verify_rows = []
    for rep_id in REPLICATE_IDS:
        for m in METHODS:
            verify_rows.append(verify_fresh_run(source_runs[rep_id][m], rep_id, m))
    verify_df = pd.DataFrame(verify_rows)

    # n_samples_actual must agree across the 4 methods within a replicate --
    # they were all fed the same sample_seed/n_samples_requested, so a
    # disagreement would mean one method's run does not actually belong to
    # this replicate.
    per_rep_actual = verify_df.groupby("replicate_id")["n_samples_actual"].nunique()
    if (per_rep_actual != 1).any():
        raise ValueError(
            "n_samples_actual disagrees across methods within a replicate:\n"
            + verify_df[["replicate_id", "method", "n_samples_actual"]].to_string(index=False)
        )

    source_runs_path = PROCESSED_DIR / "source_runs.json"
    source_runs_path.write_text(json.dumps(source_runs, indent=2), encoding="utf-8")

    n_samples_actual_by_rep = {
        rep_id: int(v)
        for rep_id, v in verify_df.drop_duplicates("replicate_id")
        .set_index("replicate_id")["n_samples_actual"]
        .items()
    }

    seeds_record = {
        "note": (
            "10 fresh sample-location replicates at the SAME ground truth "
            f"(TRUTH_SEED={TRUTH_SEED}, range={AXIS_HMAJ1:g} m) and the SAME requested "
            f"sample count (n_samples_requested={N_SAMPLES_REQUESTED}, the base case's "
            "own 5% density level), varying ONLY sample_seed. Distinct from the "
            f"project's pinned base-case SAMPLE_SEED={SAMPLE_SEED} (an 11th, pre-existing "
            "point of reference at the same n_samples, NOT folded into this set -- see "
            f"module docstring) and from TRUTH_SEED={TRUTH_SEED}. n_samples_actual can "
            "differ slightly between replicates because random_interior_samples snaps "
            "draws to grid-cell centroids and drops duplicate-cell collisions."
        ),
        "truth_seed": TRUTH_SEED,
        "hmaj1": AXIS_HMAJ1,
        "hmin1": AXIS_HMIN1,
        "n_samples_requested": N_SAMPLES_REQUESTED,
        "sample_seed_replicates": REPLICATE_SEED,
        "n_samples_actual": n_samples_actual_by_rep,
        "base_case_reference_sample_seed": SAMPLE_SEED,
        "methods": METHODS,
    }
    seeds_path = PROCESSED_DIR / "replicate_seeds.json"
    seeds_path.write_text(json.dumps(seeds_record, indent=2), encoding="utf-8")

    print("\nSample-seed replicate axis: 10 replicates x 4 methods, all verified.")
    print(json.dumps(seeds_record, indent=2))
    print(f"\nsource_runs.json: {source_runs_path}")
    print(f"replicate_seeds.json: {seeds_path}")

    return source_runs


if __name__ == "__main__":
    main()
