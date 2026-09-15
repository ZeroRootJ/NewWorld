"""Sample-density axis experiment (docs/progress.md "axis 5 -- sample
count/sparsity"): one-factor-at-a-time variation of the number of
conditioning samples, holding EVERY other base-case parameter fixed --
variogram range (300 m, isotropic), nugget, sill split, grid, distribution,
TRUTH_SEED, SAMPLE_SEED, and every method's own search/tuning constants.

Axis definition (user decision 2026-09-15, NOT to be re-litigated here)
----------------------------------------------------------------------
Three levels, expressed as a percentage of the 2500-cell (50x50) truth grid:

    axis_level  "5"  -> n_samples = 125   (5%)  == THE BASE CASE
    axis_level  "2"  -> n_samples =  50   (2%)
    axis_level  "1"  -> n_samples =  25   (1%)

One ground-truth realization per level (TRUTH_SEED=101), same convention as
the range axis. Because the range is held at the base-case 300 m and the
truth seed is unchanged, THE TRUTH FIELD IS LITERALLY IDENTICAL AT ALL THREE
LEVELS -- only the conditioning data differs.

Sample locations: X draws SHARED across levels, Y draws differ
---------------------------------------------------------------
User decision 2026-09-15 (an earlier nested-subset instruction was
explicitly cancelled): each level draws its own samples directly from the
full truth field via ``random_interior_samples(n_samples=<level's n>,
seed=SAMPLE_SEED)``.

That call draws ``xs = rng.uniform(..., n_samples)`` and then
``ys = rng.uniform(..., n_samples)`` as two consecutive calls on ONE
``RandomState(SAMPLE_SEED)``. At a fixed seed this makes the three levels
neither nested subsets NOR independent draws -- do NOT describe them as
"independent" (the same coupling is documented at the source in
src/experiments/base_case_conditioning.py):

  - the X draws ARE shared: the n=25 level's 25 raw X values are exactly
    the first 25 raw X values of the n=125 level (and of the n=50 level) --
    verified, raw-stream prefixes compare equal element-for-element;
  - the Y draws are NOT shared: they are taken from stream positions
    n .. 2n-1, i.e. 25..49 for n=25 vs. 125..149 for n=125, so they differ.

Measured at SAMPLE_SEED=20 (facts, recorded rather than designed -- see
sample_density_overlap() below, which recomputes and saves all of them):

  distinct-X overlap  all 20 distinct X coordinates of the n=25 sample set
                      also occur in the n=125 set, and all 20 also occur in
                      the n=50 set (20 of 20 in both cases); all 30 distinct
                      X of the n=50 set occur in the n=125 set (30 of 30)
  distinct-Y overlap  16 of the n=25 set's 18 distinct Y coordinates occur
                      in the n=125 set (14 of 18 against the n=50 set)
  (X,Y) CELL overlap  |n25 ∩ n125| = 4 of 25, |n50 ∩ n125| = 4 of 50,
                      |n25 ∩ n50| = 0 of 25

The cell-overlap counts on their own (4/25, 4/50, 0/25) do NOT reveal this
structure: because the X coordinates are shared and only the Y coordinates
differ, the levels' sample sets stay strongly coupled along X even though
almost no complete (X, Y) cell is shared.

LIMITATION that must be carried into any report built on this axis: with a
single realization per level, a difference between two levels mixes two
effects -- "less data was given" and "the data landed in different places"
(chiefly along Y, since X is shared) -- and this design does not separate
them. This is recorded in
results/processed/sample_density_axis/sample_overlap.json, in the reused/new
run manifests (via ``n_samples_requested``/``n_samples_actual``), and in the
figure captions.

Run reuse (avoid recomputing identical runs)
--------------------------------------------
- 5% (n=125) IS the base case. Its four pinned runs are reused verbatim from
  results/processed/base_case/source_runs.json.
- Only the 2% and 1% levels are actually executed here (2 levels x 4 methods
  = 8 fresh runs).

Every reused run is VERIFIED (not assumed) to belong to its axis level
before being written into source_runs.json, using the same decisive check
the range axis uses: the run's recorded samples.csv is compared against the
conditioning samples regenerated at that level's n_samples. Sample locations
AND values are n_samples-specific (see the overlap numbers above), so an
exact match pins the run to the right level.

This script does NOT re-implement any method -- it calls each method's
already-parameterized main() (kriging.main / sgs.main / rbf_bootstrap.main /
gp_mle.main) once per new level, in the fixed order [kriging, sgs,
rbf_bootstrap, gp_mle].

Run with: .venv/Scripts/python.exe -m src.experiments.sample_density_axis

Add ``--overlap-only`` to regenerate just sample_overlap.json (a derived,
deterministic record) without re-executing any method run.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.base_case import NX, NY, XMAX, XMIN, YMAX, YMIN
from src.experiments.base_case_conditioning import (
    HMAJ1,
    HMIN1,
    MARGIN_FRAC,
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_conditioning_data,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments import gp_mle, kriging, rbf_bootstrap, sgs

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_density_axis"
BASE_CASE_PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "base_case"

# axis_level (percent of the 2500-cell grid, as a string) -> n_samples.
# "5" must map to N_SAMPLES so the base case is reused, not redefined here.
SAMPLE_COUNTS = {"5": N_SAMPLES, "2": 50, "1": 25}

# Ordered high -> low density, which is also the order the figures use.
ALL_AXIS_LEVELS = ["5", "2", "1"]

# The one level whose runs already exist (it IS the base case).
BASE_CASE_AXIS_LEVEL = "5"
NEW_AXIS_LEVELS = ["2", "1"]

# Held fixed across the whole axis (one-factor-at-a-time).
AXIS_HMAJ1 = HMAJ1
AXIS_HMIN1 = HMIN1

METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]

N_GRID_CELLS = NX * NY

# samples.csv is written as text, so a regenerated sample frame agrees with a
# recorded one only to float-text round-trip precision (measured max |diff| =
# 3.6e-15 on the Por column for the base case -- the same tolerance the range
# axis accepted). Bit-for-bit equality is NOT the right test here.
SAMPLES_MATCH_ATOL = 1e-10


def _raw_uniform_draws(n_samples: int):
    """Replay the two ``rng.uniform`` calls ``random_interior_samples`` makes
    for a given ``n_samples`` at SAMPLE_SEED, BEFORE grid snapping and
    duplicate-cell removal. Used only to state -- as a verified fact rather
    than an inference from reading the code -- that the X draw streams of two
    levels agree on their common prefix while the Y streams do not."""
    x_margin = MARGIN_FRAC * (XMAX - XMIN)
    y_margin = MARGIN_FRAC * (YMAX - YMIN)
    rng = np.random.RandomState(SAMPLE_SEED)
    xs = rng.uniform(XMIN + x_margin, XMAX - x_margin, n_samples)
    ys = rng.uniform(YMIN + y_margin, YMAX - y_margin, n_samples)
    return xs, ys


def sample_density_overlap() -> dict:
    """Measure how much the three levels' conditioning samples actually
    coincide -- at CELL level AND at individual-coordinate level. Recorded as
    a FACT about the chosen draw scheme, not as a design knob; see the module
    docstring ("X draws SHARED across levels, Y draws differ") and its
    LIMITATION note.

    The coordinate-level statistics exist because the cell-overlap counts
    alone are misleading here: they are small (4/25, 4/50, 0/25) even though
    the levels share their entire X draw stream.
    """
    truth = get_base_case_truth(
        truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1
    )
    cells = {}
    counts = {}
    xvals = {}
    yvals = {}
    for level in ALL_AXIS_LEVELS:
        df = get_conditioning_samples(
            truth, sample_seed=SAMPLE_SEED, n_samples=SAMPLE_COUNTS[level]
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
            n_common = min(SAMPLE_COUNTS[a], SAMPLE_COUNTS[b])
            xa, ya = _raw_uniform_draws(SAMPLE_COUNTS[a])
            xb, yb = _raw_uniform_draws(SAMPLE_COUNTS[b])
            smaller_x, larger_x = (
                (xvals[b], xvals[a]) if counts[b] <= counts[a] else (xvals[a], xvals[b])
            )
            smaller_y, larger_y = (
                (yvals[b], yvals[a]) if counts[b] <= counts[a] else (yvals[a], yvals[b])
            )
            pairs[f"{a}pct_vs_{b}pct"] = {
                # --- (X, Y) cell level ---------------------------------
                "n_shared_cells": len(cells[a] & cells[b]),
                f"n_cells_{a}pct": len(cells[a]),
                f"n_cells_{b}pct": len(cells[b]),
                "smaller_cell_set_is_subset_of_larger": bool(
                    cells[b] <= cells[a] if len(cells[b]) <= len(cells[a]) else cells[a] <= cells[b]
                ),
                # --- individual-coordinate level ------------------------
                # These are the numbers the cell counts hide: the snapped X
                # coordinates of the smaller sample set are fully contained
                # in the larger set's, because the underlying raw X draws
                # share a common stream prefix (checked below).
                f"n_distinct_x_{a}pct": len(xvals[a]),
                f"n_distinct_x_{b}pct": len(xvals[b]),
                "n_shared_distinct_x": len(xvals[a] & xvals[b]),
                "n_distinct_x_of_smaller_set": len(smaller_x),
                "smaller_distinct_x_set_is_subset_of_larger": bool(smaller_x <= larger_x),
                f"n_distinct_y_{a}pct": len(yvals[a]),
                f"n_distinct_y_{b}pct": len(yvals[b]),
                "n_shared_distinct_y": len(yvals[a] & yvals[b]),
                "n_distinct_y_of_smaller_set": len(smaller_y),
                "smaller_distinct_y_set_is_subset_of_larger": bool(smaller_y <= larger_y),
                # --- raw pre-snapping draw streams ----------------------
                "raw_draw_common_prefix_length": n_common,
                "raw_x_draws_identical_over_common_prefix": bool(
                    np.array_equal(xa[:n_common], xb[:n_common])
                ),
                "raw_y_draws_identical_over_common_prefix": bool(
                    np.array_equal(ya[:n_common], yb[:n_common])
                ),
            }

    return {
        "note": (
            "The three axis levels are NEITHER nested subsets NOR independent "
            "draws -- do not describe them as 'independent'. Each level calls "
            "random_interior_samples(n_samples=<level's n>, seed=SAMPLE_SEED), "
            "which draws xs = rng.uniform(..., n_samples) and then "
            "ys = rng.uniform(..., n_samples) on ONE RandomState(SAMPLE_SEED). "
            "At a fixed seed the X draws are therefore SHARED across levels "
            "(the n=25 level's 25 raw X values are exactly the first 25 raw X "
            "values of the n=125 and n=50 levels; see "
            "raw_x_draws_identical_over_common_prefix) while the Y draws come "
            "from stream positions n..2n-1 and differ. Measured consequence: "
            "all 20 distinct X coordinates of the n=25 set occur in the n=125 "
            "set and all 20 occur in the n=50 set, and all 30 distinct X of "
            "the n=50 set occur in the n=125 set, whereas only 16 of the n=25 "
            "set's 18 distinct Y coordinates occur in the n=125 set -- yet "
            "just 4 of 25 complete (X,Y) cells are shared between n=25 and "
            "n=125, 4 of 50 between n=50 and n=125, and 0 between n=25 and "
            "n=50. The cell counts alone hide the X-sharing. LIMITATION: with "
            "one ground-truth realization per level, a difference between "
            "levels mixes 'less data was given' with 'the data landed in "
            "different places' (chiefly along Y, since X is shared); this "
            "design does not separate the two."
        ),
        "sample_seed": SAMPLE_SEED,
        "truth_seed": TRUTH_SEED,
        "hmaj1": AXIS_HMAJ1,
        "n_samples_requested": {lvl: SAMPLE_COUNTS[lvl] for lvl in ALL_AXIS_LEVELS},
        "n_samples_actual": counts,
        "sample_fraction_actual_pct": {
            lvl: 100.0 * counts[lvl] / N_GRID_CELLS for lvl in ALL_AXIS_LEVELS
        },
        "n_distinct_x_actual": {lvl: len(xvals[lvl]) for lvl in ALL_AXIS_LEVELS},
        "n_distinct_y_actual": {lvl: len(yvals[lvl]) for lvl in ALL_AXIS_LEVELS},
        "pairwise_overlap": pairs,
    }


def write_overlap() -> dict:
    """Recompute sample_overlap.json and write it. Separated from main() so
    the (purely derived, deterministic) overlap record can be regenerated
    without re-executing any method run."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    overlap = sample_density_overlap()
    overlap_path = PROCESSED_DIR / "sample_overlap.json"
    overlap_path.write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    print(
        "Sample-location overlap across axis levels "
        "(X draws shared, Y draws differ -- NOT independent, NOT nested):"
    )
    print(json.dumps(overlap, indent=2))
    print(f"sample_overlap.json: {overlap_path}\n")
    return overlap


def run_one_level(axis_level: str) -> dict:
    """Run all 4 methods at this level's n_samples, with the variogram range
    (AXIS_HMAJ1/AXIS_HMIN1 = the base-case 300 m), TRUTH_SEED, SAMPLE_SEED
    and every method's own search/tuning constants left at their base-case
    values. Returns {"kriging": Path(...), ...}."""
    n = SAMPLE_COUNTS[axis_level]
    kwargs = dict(
        truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1, n_samples=n
    )
    run_dirs = {}

    for name, module in (
        ("kriging", kriging),
        ("sgs", sgs),
        ("rbf_bootstrap", rbf_bootstrap),
        ("gp_mle", gp_mle),
    ):
        print(f"\n=== sample fraction = {axis_level}% (n_samples={n}): {name} ===")
        run_dir, _, _ = module.main(**kwargs)
        run_dirs[name] = run_dir

    return run_dirs


def verify_reused_run(rel_run_dir: str, axis_level: str, method: str) -> None:
    """Verify that an existing run directory really belongs to this axis
    level (and to this axis's TRUTH_SEED/SAMPLE_SEED/range) before it is
    reused. Raises on any mismatch -- a silently mislabeled axis level would
    be the single most damaging error this table can carry."""
    n = SAMPLE_COUNTS[axis_level]
    run_dir = _REPO_ROOT / rel_run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["params"]

    if params.get("truth_seed") != TRUTH_SEED or params.get("sample_seed") != SAMPLE_SEED:
        raise ValueError(
            f"{rel_run_dir}: truth_seed/sample_seed "
            f"({params.get('truth_seed')}/{params.get('sample_seed')}) do not match this "
            f"axis's ({TRUTH_SEED}/{SAMPLE_SEED})."
        )

    recorded_n = params.get("n_samples_requested")
    if recorded_n is not None and int(recorded_n) != n:
        raise ValueError(
            f"{rel_run_dir}: manifest records n_samples_requested={recorded_n} but is "
            f"being reused as the {axis_level}% (n_samples={n}) axis level."
        )

    # Recorded range, where the manifest has it. The base-case rbf_bootstrap /
    # gp_mle manifests predate the top-level hmaj1 field, so this is
    # best-effort; the samples check below is the decisive one.
    recorded_range = params.get("hmaj1")
    if recorded_range is None:
        recorded_range = params.get("variogram", {}).get("hmaj1")
    if recorded_range is not None and not np.isclose(float(recorded_range), AXIS_HMAJ1):
        raise ValueError(
            f"{rel_run_dir}: manifest records range {recorded_range} m but this axis "
            f"holds the range fixed at {AXIS_HMAJ1:g} m."
        )

    # Decisive check: regenerate this level's conditioning samples and require
    # a match (to float-text round-trip precision, see SAMPLES_MATCH_ATOL).
    _, samples_df = get_base_case_conditioning_data(
        sample_seed=SAMPLE_SEED,
        truth_seed=TRUTH_SEED,
        hmaj1=AXIS_HMAJ1,
        hmin1=AXIS_HMIN1,
        n_samples=n,
    )
    recorded_samples = pd.read_csv(run_dir / "samples.csv")
    if len(recorded_samples) != len(samples_df) or not np.allclose(
        recorded_samples[["X", "Y", VCOL]].values,
        samples_df[["X", "Y", VCOL]].values,
        rtol=0.0,
        atol=SAMPLES_MATCH_ATOL,
    ):
        raise ValueError(
            f"{rel_run_dir}: recorded samples.csv does not match the conditioning "
            f"samples regenerated at n_samples={n} -- this run was NOT produced from "
            "that conditioning dataset."
        )
    max_abs_diff = float(
        np.max(
            np.abs(
                recorded_samples[["X", "Y", VCOL]].values - samples_df[["X", "Y", VCOL]].values
            )
        )
    )
    print(
        f"  verified reused {method} run for {axis_level}% (n={n}, actual="
        f"{len(samples_df)}): {rel_run_dir} (max |diff| vs. regenerated samples = "
        f"{max_abs_diff:.2e})"
    )


def _rel(p) -> str:
    return str(Path(p).relative_to(_REPO_ROOT)).replace("\\", "/")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Record how much the three levels' sample sets actually coincide ---
    write_overlap()

    source_runs = {}

    # --- Reuse: 5% == the base case ----------------------------------------
    base_case_source_runs = json.loads(
        (BASE_CASE_PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8")
    )
    print(f"Reusing the base case as the {BASE_CASE_AXIS_LEVEL}% axis level:")
    source_runs[BASE_CASE_AXIS_LEVEL] = {
        m: base_case_source_runs[m]["run_dir"] for m in METHODS
    }
    for m in METHODS:
        verify_reused_run(source_runs[BASE_CASE_AXIS_LEVEL][m], BASE_CASE_AXIS_LEVEL, m)

    # --- Execute the new levels --------------------------------------------
    for axis_level in NEW_AXIS_LEVELS:
        run_dirs = run_one_level(axis_level)
        source_runs[axis_level] = {m: _rel(run_dirs[m]) for m in METHODS}

    source_runs = {lvl: source_runs[lvl] for lvl in ALL_AXIS_LEVELS}

    missing = [lvl for lvl in source_runs if set(source_runs[lvl]) != set(METHODS)]
    if missing:
        raise ValueError(f"axis level(s) {missing} are missing a method run.")

    source_runs_path = PROCESSED_DIR / "source_runs.json"
    with open(source_runs_path, "w", encoding="utf-8") as f:
        json.dump(source_runs, f, indent=2)

    print("\nsample_density_axis source runs (3 levels x 4 methods):")
    print(json.dumps(source_runs, indent=2))
    print(f"\nsource_runs.json: {source_runs_path}")

    return source_runs


if __name__ == "__main__":
    # ``--overlap-only`` regenerates results/processed/sample_density_axis/
    # sample_overlap.json (a purely derived, deterministic record) without
    # re-executing any method run.
    if "--overlap-only" in sys.argv[1:]:
        write_overlap()
    else:
        main()
