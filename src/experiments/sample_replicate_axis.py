"""Sample-seed replicate axis (user task 2026-09-18, extended 2026-09-18): how
much does the 4-method comparison (kriging / SGS / RBF+bootstrap / GP-MLE)
move when ONLY the conditioning-sample LOCATIONS change, at a FIXED ground
truth and a FIXED requested sample count -- now repeated at ALL THREE of the
sample-density axis's density levels (5% / 2% / 1%, i.e. n_samples_requested
= 125 / 50 / 25), using the SAME 10 sample_seed replicates at every level.

This is a different question from the sample-density axis
(src/experiments/sample_density_axis.py), which varies n_samples across 3
levels at ONE fixed sample_seed per level (so any level-to-level difference
there mixes "less data" with "the data landed in different places", as that
module's docstring documents). Here, AT EACH density level separately,
n_samples_requested is held fixed and ONLY sample_seed varies across 10
independent replicates -- isolating the sample-PLACEMENT effect on its own,
one level at a time.

Held fixed across all 3 levels x 10 replicates (one-factor-at-a-time):
  - TRUTH_SEED = 101 (base_case_conditioning.TRUTH_SEED) -- literally the
    same ground-truth field for every level and every replicate.
  - variogram range hmaj1 = hmin1 = 300 m (reused, not redeclared, from
    src.experiments.sample_density_axis.AXIS_HMAJ1/AXIS_HMIN1, which are
    themselves the base case's own HMAJ1/HMIN1).
  - n_samples REQUESTED per level = sample_density_axis.SAMPLE_COUNTS
    ("5"->125, "2"->50, "1"->25) -- the axis's own 3 density levels, reused
    (not redeclared) so this axis's requested counts cannot silently drift
    from sample_density_axis's. ``random_interior_samples`` snaps draws to
    the nearest grid-cell centroid and drops duplicate-cell collisions, so
    the ACTUAL count used can be < requested and can differ slightly between
    replicates (their raw draws collide on different cells) -- the actual
    count per (level, replicate) is measured and recorded below, never
    assumed to equal the requested count.
  - every method's own search/tuning constants (kriging's NDMIN/NDMAX/
    RADIUS, sgs's NDMAX/NODMAX/jitter, rbf_bootstrap's CV epsilon grid,
    gp_mle's kernel bounds/n_restarts, ...) stay at their base-case values
    -- untouched here, exactly as every other axis in this project requires.

ONLY sample_seed AND n_samples vary across the 3 x 10 = 30 (level,
replicate) cells of this design.

The 10 sample-seed replicates (SAME 10 seeds reused at every level)
---------------------------------------------------------------------
SAMPLE_SEED_REPLICATES = [1001, 1002, ..., 1010] -- ten consecutive literal
integers, reused UNCHANGED from this module's original (2026-09-18, n=125
only) version. No SeedSequence/derivation scheme is used; the exact values
are written here AND in this script's JSON output
(results/processed/sample_replicate_axis/replicate_seeds.json) for
auditability. They were chosen only to be trivially distinct from every seed
constant already pinned elsewhere in this project:
  - SAMPLE_SEED = 20 (base_case_conditioning.py, the base case's own sample
    locations at n_samples=125)
  - TRUTH_SEED = 101 (base_case_conditioning.py)
This module asserts at import time that none of SAMPLE_SEED_REPLICATES
collides with either of those, so this experiment cannot silently duplicate
or overwrite an existing pinned run's sample locations.

Reusing the SAME 10 seeds at n=50 and n=25 (rather than drawing 10 fresh
seeds per level) is a deliberate choice so that "rep0" identifies the SAME
underlying draw-stream position across all 3 levels -- comparable to how
sample_density_axis.py itself reuses one shared SAMPLE_SEED=20 across its 3
levels. Exactly like that module, this does NOT make the 3 levels'
rep-k sample sets nested subsets of each other or independent of one
another: ``random_interior_samples(n_samples=<level's n>, seed=<rep k's
seed>)`` draws xs = rng.uniform(..., n_samples) and then
ys = rng.uniform(..., n_samples) on ONE RandomState(seed), so for a FIXED
replicate seed the X draws are shared across that replicate's 3 levels
(the n=25 draw's 25 raw X values are exactly the first 25 raw X values of
the n=125 and n=50 draws) while the Y draws differ (stream positions
n..2n-1). This is measured and recorded, per replicate, in
sample_seed_level_overlap.json (see ``sample_seed_level_overlap()`` below) --
not assumed.

NOT one of these 10 (must not be folded in): the base case's own pinned
n_samples=125 run (results/processed/base_case/) uses SAMPLE_SEED=20 -- an
11th, PRE-EXISTING sample-location draw at the same requested n_samples. It
is a legitimate 11th point of reference for "how much does sample placement
move things at n=125", but it is deliberately NOT included in this
10-replicate set (no reuse shortcut here, unlike sample_density_axis.py's
5%-level base-case reuse): this experiment draws all 10 of its own sample
sets fresh, so it is a clean, uniformly-generated sample of "where do
<n> samples land", uncontaminated by the specific SAMPLE_SEED=20 draw the
rest of the project is built around.

Run reuse across the 3 density levels (avoid recomputing identical runs)
--------------------------------------------------------------------------
The "5" level (n_samples_requested=125) is the ORIGINAL 2026-09-18 version of
this experiment -- its 10 replicates x 4 methods = 40 runs were already
executed and are REUSED VERBATIM here, from the frozen snapshot
results/processed/sample_replicate_axis/source_runs_level5_n125.json (a
copy of this module's own source_runs.json exactly as it stood before this
extension, kept as a stable, self-contained reuse anchor -- the analogue of
how sample_density_axis.py reuses results/processed/base_case/
source_runs.json for its own "5" level, except here the anchor is a frozen
copy of this SAME module's prior output rather than a different
experiment's). Only "2" (n=50) and "1" (n=25) are actually executed by this
version of the script: 2 NEW levels x 10 replicates x 4 methods = 80 fresh
runs.

Every run -- REUSED (the "5" level's 40) and FRESH (the "2"/"1" levels' 80)
alike -- is VERIFIED (not assumed) against its own (level, replicate)'s
regenerated conditioning samples before being trusted; see
``verify_run_against_replicate`` below, used identically for both reused and
fresh runs (a mislabeled or drifted reused run would be just as damaging as
a mislabeled fresh one).

Parallel execution (2026-09-16 pattern, reused unchanged)
------------------------------------------------------------
All 2 (new levels) x 10 (replicates) x 4 (methods) = 80 fresh tasks are
fully independent (different sample draws, disjoint output files, no shared
state), so they are submitted together to one process pool via
src/experiments/_axis_parallel.py's run_levels_parallel rather than a
sequential loop -- exactly the pattern sample_density_axis.py and the
original (n=125-only) version of this module already use.

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
from src.experiments.sample_density_axis import (
    ALL_AXIS_LEVELS,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    SAMPLE_COUNTS,
)
from src.experiments._axis_parallel import run_levels_parallel

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"

# The 3 density levels, reused (not redeclared) from sample_density_axis so
# this axis's levels/requested counts cannot silently drift from that
# module's. "5" (n_samples_requested=125) is the level whose 40 runs already
# exist and are reused verbatim; "2"/"1" are executed fresh by this module.
N_SAMPLES_REQUESTED_BY_LEVEL = SAMPLE_COUNTS
REUSED_AXIS_LEVEL = "5"
NEW_AXIS_LEVELS = ["2", "1"]

N_REPLICATES = 10
# See module docstring: ten literal, consecutive, auditable integers, reused
# UNCHANGED from the original (n=125-only) version of this module, at every
# density level.
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

# Frozen snapshot of this module's OWN prior (n=125-only) source_runs.json --
# the reuse anchor for the "5" level. See module docstring.
LEVEL5_REUSE_ANCHOR_PATH = PROCESSED_DIR / "source_runs_level5_n125.json"

# Same float-text round-trip tolerance sample_density_axis.py uses for
# samples.csv comparisons.
SAMPLES_MATCH_ATOL = 1e-10


def _rel(p) -> str:
    return str(Path(p).relative_to(_REPO_ROOT)).replace("\\", "/")


def verify_run_against_replicate(
    rel_run_dir: str, axis_level: str, replicate_id: str, method: str
) -> dict:
    """Verify that a run (reused OR fresh) really used this (axis_level,
    replicate_id)'s truth_seed/sample_seed/range/n_samples_requested, and
    that its recorded samples.csv matches the conditioning samples
    regenerated for that (level, replicate). Raises on any mismatch. Returns
    a small dict of facts (used to build replicate_seeds.json's
    n_samples_actual record)."""
    seed = REPLICATE_SEED[replicate_id]
    n_requested = N_SAMPLES_REQUESTED_BY_LEVEL[axis_level]
    run_dir = _REPO_ROOT / rel_run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["params"]

    if params.get("truth_seed") != TRUTH_SEED or params.get("sample_seed") != seed:
        raise ValueError(
            f"{rel_run_dir}: truth_seed/sample_seed "
            f"({params.get('truth_seed')}/{params.get('sample_seed')}) do not match "
            f"level '{axis_level}' replicate {replicate_id}'s expected ({TRUTH_SEED}/{seed})."
        )

    recorded_n_requested = params.get("n_samples_requested")
    if recorded_n_requested is not None and int(recorded_n_requested) != n_requested:
        raise ValueError(
            f"{rel_run_dir}: manifest records n_samples_requested={recorded_n_requested}, "
            f"expected {n_requested} for level '{axis_level}'."
        )

    recorded_range = params.get("hmaj1")
    if recorded_range is not None and not np.isclose(float(recorded_range), AXIS_HMAJ1):
        raise ValueError(
            f"{rel_run_dir}: manifest records range {recorded_range} m; this axis holds "
            f"the range fixed at {AXIS_HMAJ1:g} m."
        )

    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)
    samples_df = get_conditioning_samples(
        truth, sample_seed=seed, n_samples=n_requested
    )
    recorded_n_actual = params.get("n_samples_actual")
    if recorded_n_actual is not None and int(recorded_n_actual) != len(samples_df):
        raise ValueError(
            f"{rel_run_dir}: manifest records n_samples_actual={recorded_n_actual}, but "
            f"regenerating level '{axis_level}' replicate {replicate_id}'s (seed={seed}) "
            f"conditioning samples gives {len(samples_df)}."
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
            f"regenerated for level '{axis_level}' replicate {replicate_id} (sample_seed="
            f"{seed}) -- this run was NOT produced from that conditioning dataset."
        )

    print(
        f"  verified {method} run for level '{axis_level}' {replicate_id} (sample_seed={seed}, "
        f"n_samples_actual={len(samples_df)}): {rel_run_dir}"
    )
    return {
        "axis_level": axis_level,
        "replicate_id": replicate_id,
        "method": method,
        "sample_seed": seed,
        "n_samples_requested": n_requested,
        "n_samples_actual": len(samples_df),
    }


def sample_seed_level_overlap() -> dict:
    """Per replicate, measure whether reusing the SAME sample_seed across the
    3 density levels reproduces the sample_density_axis.py-documented
    "X draws shared, Y draws differ" coupling -- checked, not assumed. Uses
    the exact same raw-pre-snapping-draw-stream replay technique
    sample_density_axis.sample_density_overlap() uses, applied per replicate
    seed instead of the single SAMPLE_SEED."""
    from src.experiments.base_case import XMAX, XMIN, YMAX, YMIN
    from src.experiments.base_case_conditioning import MARGIN_FRAC

    def _raw_uniform_draws(seed: int, n_samples: int):
        x_margin = MARGIN_FRAC * (XMAX - XMIN)
        y_margin = MARGIN_FRAC * (YMAX - YMIN)
        rng = np.random.RandomState(seed)
        xs = rng.uniform(XMIN + x_margin, XMAX - x_margin, n_samples)
        ys = rng.uniform(YMIN + y_margin, YMAX - y_margin, n_samples)
        return xs, ys

    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1)

    per_replicate = {}
    for rep_id, seed in REPLICATE_SEED.items():
        cells, xvals, yvals, counts = {}, {}, {}, {}
        for level in ALL_AXIS_LEVELS:
            df = get_conditioning_samples(
                truth, sample_seed=seed, n_samples=N_SAMPLES_REQUESTED_BY_LEVEL[level]
            )
            counts[level] = len(df)
            x_rounded = np.round(df["X"].values, 6).tolist()
            y_rounded = np.round(df["Y"].values, 6).tolist()
            cells[level] = set(zip(x_rounded, y_rounded))
            xvals[level] = set(x_rounded)
            yvals[level] = set(y_rounded)

        pairs = {}
        for i, a in enumerate(ALL_AXIS_LEVELS):
            for b in ALL_AXIS_LEVELS[i + 1:]:
                n_common = min(
                    N_SAMPLES_REQUESTED_BY_LEVEL[a], N_SAMPLES_REQUESTED_BY_LEVEL[b]
                )
                xa, ya = _raw_uniform_draws(seed, N_SAMPLES_REQUESTED_BY_LEVEL[a])
                xb, yb = _raw_uniform_draws(seed, N_SAMPLES_REQUESTED_BY_LEVEL[b])
                smaller_x, larger_x = (
                    (xvals[b], xvals[a]) if counts[b] <= counts[a] else (xvals[a], xvals[b])
                )
                smaller_y, larger_y = (
                    (yvals[b], yvals[a]) if counts[b] <= counts[a] else (yvals[a], yvals[b])
                )
                pairs[f"{a}pct_vs_{b}pct"] = {
                    "n_shared_cells": len(cells[a] & cells[b]),
                    f"n_cells_{a}pct": len(cells[a]),
                    f"n_cells_{b}pct": len(cells[b]),
                    "n_distinct_x_of_smaller_set": len(smaller_x),
                    "smaller_distinct_x_set_is_subset_of_larger": bool(smaller_x <= larger_x),
                    "n_distinct_y_of_smaller_set": len(smaller_y),
                    "smaller_distinct_y_set_is_subset_of_larger": bool(smaller_y <= larger_y),
                    "raw_draw_common_prefix_length": n_common,
                    "raw_x_draws_identical_over_common_prefix": bool(
                        np.array_equal(xa[:n_common], xb[:n_common])
                    ),
                    "raw_y_draws_identical_over_common_prefix": bool(
                        np.array_equal(ya[:n_common], yb[:n_common])
                    ),
                }
        per_replicate[rep_id] = {
            "sample_seed": seed,
            "n_samples_actual": counts,
            "pairwise_overlap": pairs,
        }

    # Aggregate a headline number: across all 10 replicates x 3 level-pairs,
    # is the "X shared, Y differs" coupling universal?
    all_x_subset = all(
        per_replicate[r]["pairwise_overlap"][pair]["smaller_distinct_x_set_is_subset_of_larger"]
        for r in REPLICATE_IDS
        for pair in per_replicate[r]["pairwise_overlap"]
    )
    all_y_not_subset = all(
        not per_replicate[r]["pairwise_overlap"][pair]["smaller_distinct_y_set_is_subset_of_larger"]
        for r in REPLICATE_IDS
        for pair in per_replicate[r]["pairwise_overlap"]
    )
    all_raw_x_identical = all(
        per_replicate[r]["pairwise_overlap"][pair]["raw_x_draws_identical_over_common_prefix"]
        for r in REPLICATE_IDS
        for pair in per_replicate[r]["pairwise_overlap"]
    )
    all_raw_y_not_identical = all(
        not per_replicate[r]["pairwise_overlap"][pair]["raw_y_draws_identical_over_common_prefix"]
        for r in REPLICATE_IDS
        for pair in per_replicate[r]["pairwise_overlap"]
    )

    return {
        "note": (
            "For EVERY one of the 10 replicates, reusing that replicate's sample_seed at "
            "all 3 density levels reproduces the same 'X draws shared across levels, Y "
            "draws differ' coupling sample_density_axis.py documents for its own single "
            "SAMPLE_SEED=20: see all_replicates_raw_x_draws_identical_over_common_prefix "
            "and all_replicates_raw_y_draws_NOT_identical_over_common_prefix below "
            "(measured, not assumed) plus the per-replicate detail in 'per_replicate'."
        ),
        "all_replicates_smaller_distinct_x_set_is_subset_of_larger": all_x_subset,
        "all_replicates_smaller_distinct_y_set_is_NOT_subset_of_larger": all_y_not_subset,
        "all_replicates_raw_x_draws_identical_over_common_prefix": all_raw_x_identical,
        "all_replicates_raw_y_draws_NOT_identical_over_common_prefix": all_raw_y_not_identical,
        "per_replicate": per_replicate,
    }


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- (1) Reuse: the "5" level's 40 runs already exist ------------------
    if not LEVEL5_REUSE_ANCHOR_PATH.exists():
        raise FileNotFoundError(
            f"{LEVEL5_REUSE_ANCHOR_PATH} not found -- this is the frozen snapshot of the "
            "original (n=125-only) version of this module's source_runs.json and is "
            "required to reuse the level-5 runs without re-running them."
        )
    level5_source_runs = json.loads(LEVEL5_REUSE_ANCHOR_PATH.read_text(encoding="utf-8"))
    if set(level5_source_runs) != set(REPLICATE_IDS):
        raise ValueError(
            f"{LEVEL5_REUSE_ANCHOR_PATH} has replicate ids {sorted(level5_source_runs)}; "
            f"expected {sorted(REPLICATE_IDS)}."
        )

    source_runs = {REUSED_AXIS_LEVEL: level5_source_runs}

    # --- (2) Execute the 2 NEW levels' 10 x 4 = 40 tasks EACH (80 total) ----
    # Flattened into one pool of 80 independent tasks, keyed by a composite
    # "<level>__<replicate_id>" string (run_levels_parallel does not
    # interpret level-key semantics at all -- see its docstring).
    level_kwargs = {
        f"{level}__{rep_id}": dict(
            truth_seed=TRUTH_SEED,
            hmaj1=AXIS_HMAJ1,
            hmin1=AXIS_HMIN1,
            n_samples=N_SAMPLES_REQUESTED_BY_LEVEL[level],
            sample_seed=seed,
        )
        for level in NEW_AXIS_LEVELS
        for rep_id, seed in REPLICATE_SEED.items()
    }
    run_dirs = run_levels_parallel(level_kwargs, methods=METHODS)

    for level in NEW_AXIS_LEVELS:
        source_runs[level] = {
            rep_id: {m: _rel(run_dirs[f"{level}__{rep_id}"][m]) for m in METHODS}
            for rep_id in REPLICATE_IDS
        }

    source_runs = {lvl: source_runs[lvl] for lvl in ALL_AXIS_LEVELS}

    missing = [
        (lvl, rep_id)
        for lvl in ALL_AXIS_LEVELS
        for rep_id in REPLICATE_IDS
        if set(source_runs[lvl].get(rep_id, {})) != set(METHODS)
    ]
    if missing:
        raise ValueError(f"(level, replicate) cell(s) {missing} are missing a method run.")

    # --- (3) Verify EVERY run (reused AND fresh) against its own (level, ---
    # replicate)'s regenerated conditioning samples -- 120 checks total.
    print(
        "\nVerifying every run (reused level-5 runs AND fresh level-2/1 runs) against its "
        "own (level, replicate)'s conditioning samples..."
    )
    verify_rows = []
    for level in ALL_AXIS_LEVELS:
        for rep_id in REPLICATE_IDS:
            for m in METHODS:
                verify_rows.append(
                    verify_run_against_replicate(
                        source_runs[level][rep_id][m], level, rep_id, m
                    )
                )
    verify_df = pd.DataFrame(verify_rows)

    # n_samples_actual must agree across the 4 methods within a (level,
    # replicate) cell -- they were all fed the same sample_seed/n_samples,
    # so a disagreement would mean one method's run does not actually
    # belong to this cell.
    per_cell_actual = verify_df.groupby(["axis_level", "replicate_id"])[
        "n_samples_actual"
    ].nunique()
    if (per_cell_actual != 1).any():
        raise ValueError(
            "n_samples_actual disagrees across methods within a (level, replicate) cell:\n"
            + verify_df[
                ["axis_level", "replicate_id", "method", "n_samples_actual"]
            ].to_string(index=False)
        )

    source_runs_path = PROCESSED_DIR / "source_runs.json"
    source_runs_path.write_text(json.dumps(source_runs, indent=2), encoding="utf-8")

    n_samples_actual_by_cell = {
        level: {
            rep_id: int(
                verify_df[
                    (verify_df["axis_level"] == level)
                    & (verify_df["replicate_id"] == rep_id)
                ]["n_samples_actual"].iloc[0]
            )
            for rep_id in REPLICATE_IDS
        }
        for level in ALL_AXIS_LEVELS
    }

    seeds_record = {
        "note": (
            "10 sample-location replicates (SAME 10 seeds reused at every level -- see "
            "module docstring) at the SAME ground truth (TRUTH_SEED="
            f"{TRUTH_SEED}, range={AXIS_HMAJ1:g} m), now repeated at all 3 sample-density "
            "axis levels ('5'/'2'/'1' -> n_samples_requested=125/50/25). ONLY sample_seed "
            "and n_samples vary. Distinct from the project's pinned base-case SAMPLE_SEED="
            f"{SAMPLE_SEED} (an 11th, pre-existing point of reference at n=125, NOT folded "
            f"into this set) and from TRUTH_SEED={TRUTH_SEED}. n_samples_actual can differ "
            "slightly between replicates AND between levels because random_interior_samples "
            "snaps draws to grid-cell centroids and drops duplicate-cell collisions."
        ),
        "truth_seed": TRUTH_SEED,
        "hmaj1": AXIS_HMAJ1,
        "hmin1": AXIS_HMIN1,
        "n_samples_requested_by_level": N_SAMPLES_REQUESTED_BY_LEVEL,
        "reused_axis_level": REUSED_AXIS_LEVEL,
        "new_axis_levels": NEW_AXIS_LEVELS,
        "sample_seed_replicates": REPLICATE_SEED,
        "n_samples_actual": n_samples_actual_by_cell,
        "base_case_reference_sample_seed": SAMPLE_SEED,
        "methods": METHODS,
    }
    seeds_path = PROCESSED_DIR / "replicate_seeds.json"
    seeds_path.write_text(json.dumps(seeds_record, indent=2), encoding="utf-8")

    # --- (4) Same-seed-across-levels X/Y coupling: measure, don't assume ---
    overlap = sample_seed_level_overlap()
    overlap_path = PROCESSED_DIR / "sample_seed_level_overlap.json"
    overlap_path.write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    print(
        "\nSame-10-seeds-across-levels X/Y draw coupling (measured per replicate, all 10): "
        f"X-subset holds for all replicates={overlap['all_replicates_smaller_distinct_x_set_is_subset_of_larger']}, "
        f"Y-subset does NOT hold for any replicate={overlap['all_replicates_smaller_distinct_y_set_is_NOT_subset_of_larger']}"
    )
    print(f"sample_seed_level_overlap.json: {overlap_path}")

    print("\nSample-seed replicate axis: 3 levels x 10 replicates x 4 methods, all verified.")
    print(f"\nsource_runs.json: {source_runs_path}")
    print(f"replicate_seeds.json: {seeds_path}")

    return source_runs


if __name__ == "__main__":
    main()
