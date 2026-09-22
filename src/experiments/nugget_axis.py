"""Nugget axis experiment (Deliverable 3, docs/experiment_context.md):
one-factor-at-a-time variation of the ground-truth variogram NUGGET, holding
every other base-case parameter -- variogram range (300 m isotropic), grid,
distribution, TRUTH_SEED, SAMPLE_SEED, N_SAMPLES, and every method's own
search/tuning constants -- fixed at its base-case value.

Axis definition (orchestrator-confirmed design, NOT to be re-litigated here)
---------------------------------------------------------------------------
ALL_NUGGET_VALUES = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 -- 6 levels of the
NORMALIZED nugget (fraction of the sill on standard-normal space). The sill
is held at 1.0 at every level, so ``cc1 = 1.0 - nug`` is derived, never
passed independently (see base_case_conditioning.build_vario's docstring for
why there is deliberately no ``cc1`` argument). In physical porosity units
the truth nugget is ``nug * POR_STDEV**2`` = 0 / 0.9 / 1.8 / 2.7 / 3.6 / 4.5
Porosity %^2, against a total sill of POR_STDEV**2 = 9.0 Porosity %^2. That
is the HEADLINE convention and it is not changing.

A SECOND, EQUALLY DEFENSIBLE CONVENTION IS ALSO RECORDED (reviewer 2026-09-22,
orchestrator decision: record both, headline unchanged). ``GSLIB.affine``
scales each realization by ``a = POR_STDEV / np.std(sim_ns)``, and a single
realization's ``np.std(sim_ns)`` is not exactly 1.0, so the sill the truth
field REALIZES is ``a**2`` = 7.5330 / 7.5553 / 7.7311 / 7.9710 / 8.2402 /
8.4950 rather than 9.0, giving an alternative truth nugget ``nug * a**2``.
Both appear as separate columns in
results/processed/nugget_axis/gp_fitted_nugget_vs_truth.csv; see
``truth_nugget_real_units`` / ``truth_nugget_real_units_affine`` below.

One ground-truth realization per level (TRUTH_SEED=101), as on the range
axis -- 6 levels x 4 methods = 24 runs.

KRIGING ARM RE-PINNED 2026-09-22 (nscore off-by-one)
-----------------------------------------------------
The 6 kriging runs originally produced by this script were computed with
``geostatspy.geostats.nscore``'s 1-based -> 0-based off-by-one, which gave
the smallest conditioning datum a wrong normal score (spectacularly so at
nug=0.3: -52.15 instead of -2.6411). They were replaced, WITHOUT touching
results/raw/, by ``src/experiments/rerun_nugget_axis_kriging_nscore_fix.py``
after the repair in ``src/nscore.py``; sgs / rbf_bootstrap / gp_mle are not
affected by that bug and keep their original pins. The pre-fix pin set is
preserved as source_runs_superseded_20260922_nscore_bug.json and quantified
in results/processed/nscore_bug_impact.csv. Re-running THIS script from
scratch would regenerate all 24 runs (kriging now repaired) and is the
normal path going forward.

WHY THE BASE CASE IS NOT REUSED AS A LEVEL (unlike the range axis)
------------------------------------------------------------------
The base case is NUG=0.05, which is not one of the 6 axis levels (it falls
between 0.0 and 0.1). There is therefore nothing to reuse and all 24 runs
are executed fresh here. The base case may be mentioned as a reference point
in figures/notes, but it is not an axis level and must not be spliced into
this axis's tables.

WHAT EACH METHOD IS GIVEN (this is the point of the axis)
---------------------------------------------------------
- kriging / SGS are handed the TRUE nugget as their input variogram, exactly
  as they are handed the true range on the range axis. They are the "correct
  answer" baselines (docs/experiment_context.md section 4), so their
  uncertainty model is allowed to know the generating statistics.
- GP-MLE learns its own WhiteKernel noise variance by marginal-likelihood
  maximization; nothing about the truth nugget is passed to it. Whether that
  learned noise tracks the true nugget is Claim 2's core question.
- RBF+bootstrap has no nugget concept at all; its CV-tuned smoothing
  (lambda) is the only parameter that can absorb nugget-scale variance.

Per-run verification (never assumed)
------------------------------------
Every produced run is verified to belong to its axis level before it is
written into source_runs.json: (a) the manifest's recorded ``nug``/``cc1``,
and (b) -- decisively -- the run's saved samples.csv is compared against the
conditioning samples regenerated at that level's nugget. Sample VALUES are
read off the truth field, and the truth field is regenerated per level, so
the sample values differ between nugget levels even though the sample
LOCATIONS are identical; an exact match therefore pins a run to one level.

Parallel execution
------------------
All 6 levels x 4 methods = 24 tasks are independent (different truth draws
per level, disjoint output files, no shared state) and are submitted together
to one process pool (src/experiments/_axis_parallel.py -- processes, not
threads, because geostatspy's sgsim mutates the global numpy RNG).

Run with: .venv/Scripts/python.exe -m src.experiments.nugget_axis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.base_case import (
    HMAJ1,
    HMIN1,
    NX,
    NY,
    POR_MEAN,
    POR_STDEV,
    XMN,
    XSIZ,
    YMN,
    YSIZ,
)
from src.experiments.base_case_conditioning import (
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    build_vario,
    get_base_case_conditioning_data,
)
from src.experiments._axis_parallel import run_levels_parallel
from src.truth_model import make_porosity_truth_with_sim_ns

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"

# Full axis: normalized nugget (fraction of the unit sill on NS space).
ALL_NUGGET_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

# Every other ground-truth parameter is pinned at its base-case value --
# imported, not re-declared, so a change to the base case cannot silently
# leave this axis behind.
AXIS_HMAJ1 = HMAJ1  # 300 m
AXIS_HMIN1 = HMIN1  # 300 m (isotropic)

METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]

CASE = "nugget_axis"
AXIS = "nugget_normalized"

# Float-text round-trip tolerance for the samples.csv comparison (same
# convention as sample_density_axis.py / evaluate_sample_density_axis.py).
SAMPLES_MATCH_ATOL = 1e-10


def level_key(nug: float) -> str:
    """Canonical string key for a nugget axis level.

    SINGLE SOURCE OF TRUTH for how a float nugget level is spelled in
    source_runs.json, metrics.csv, qc_summary.csv and every figure script --
    every one of those imports this function rather than formatting the float
    itself. Without that, "0.1" vs. "0.10" vs. repr(0.1) drift is a silent
    way to mislabel or lose an axis level (the range axis could get away with
    ``str(int(hmaj1))`` because its levels are integers; these are not).

    ROUND-TRIP GUARD (added after a test caught this, 2026-09-22): a bare
    ``f"{nug:.1f}"`` ROUNDS, so ``level_key(0.05)`` -- the BASE CASE's
    nugget -- would return "0.1" and silently collide with the genuine 0.1
    axis level. Any value that does not round-trip through the 1-decimal
    format therefore raises instead of being quietly snapped onto a
    neighbouring level's key.
    """
    key = f"{float(nug):.1f}"
    if float(key) != float(nug):
        raise ValueError(
            f"nug={nug!r} is not representable as a nugget-axis level key: "
            f"formatting it gives '{key}', which round-trips to {float(key)!r}, "
            "so it would be silently relabeled as a different axis level. The "
            f"nugget axis levels are {ALL_NUGGET_VALUES} (1 decimal place); the "
            "base case's nug=0.05 is deliberately NOT one of them."
        )
    return key


def level_file_token(nug: float) -> str:
    """Filename-safe spelling of a nugget axis level ("0.3" -> "0p3").

    Used for per-level figure file names (truth_field_nug0p3.png,
    qc_truth_predictions_kriging_nug0p3.png, ...). Derived from
    ``level_key`` rather than re-formatted, so the two can never disagree
    about which level a file belongs to. The dot is replaced because a
    filename like ``truth_field_nug0.3.png`` reads as if ".3" were part of
    the extension.
    """
    return level_key(nug).replace(".", "p")


def truth_nugget_real_units(nug: float) -> float:
    """The level's nugget in physical porosity units (Porosity %^2).
    HEADLINE convention -- this is the number used on every figure axis and
    in every ratio reported as "the" truth nugget.

    The variogram is defined on standard-normal space with a unit sill, and
    the target porosity distribution has stdev POR_STDEV, so the physical
    sill is POR_STDEV**2 and the physical nugget is that times the normalized
    nugget. Same convention as make_length_vs_noise_figure.py's
    ``NUG * POR_STDEV**2`` reference line on the range axis.

    SEE ALSO ``truth_nugget_real_units_affine`` for a second, equally
    defensible convention (reviewer, 2026-09-22) that uses the scale factor
    the affine correction ACTUALLY applied to this realization instead of the
    TARGET stdev. Both are recorded in
    results/processed/nugget_axis/gp_fitted_nugget_vs_truth.csv; this one
    remains the headline (orchestrator decision -- do not swap them).
    """
    return float(nug) * POR_STDEV ** 2


def affine_scale_factor(nug: float) -> float:
    """The multiplicative factor ``a`` that ``GSLIB.affine`` actually applied
    to this level's ground-truth realization.

    ``src/truth_model.py`` generates a standard-normal simulation ``sim_ns``
    under a UNIT-sill variogram and then calls
    ``GSLIB.affine(sim_ns, POR_MEAN, POR_STDEV)``, which computes
    ``a = POR_STDEV / np.std(sim_ns)`` and returns ``a * (sim_ns - mean) +
    POR_MEAN``. A single realization's sample stdev is not exactly 1.0
    (ergodic fluctuation), so ``a != POR_STDEV`` and the variogram sill the
    truth field REALIZES in physical units is ``a**2``, not ``POR_STDEV**2``.

    MEASURED at TRUTH_SEED=101, range 300 m isotropic (2026-09-22):
    std(sim_ns) = 1.093042 / 1.091430 / 1.078946 / 1.062590 / 1.045085 /
    1.029291 and hence a**2 = 7.5330 / 7.5553 / 7.7311 / 7.9710 / 8.2402 /
    8.4950 for nug = 0.0 / 0.1 / 0.2 / 0.3 / 0.4 / 0.5, against the headline
    POR_STDEV**2 = 9.0.

    COST: this regenerates the level's ground truth (one sgsim call, a few
    seconds), because ``std(sim_ns)`` is not stored anywhere. Callers that
    need it for several levels should cache the result.
    """
    vario = build_vario(hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1, nug=float(nug))
    _, sim_ns = make_porosity_truth_with_sim_ns(
        nx=NX, ny=NY, xsiz=XSIZ, ysiz=YSIZ, xmn=XMN, ymn=YMN,
        vario=vario, mean=POR_MEAN, stdev=POR_STDEV, seed=TRUTH_SEED,
    )
    return float(POR_STDEV / np.std(sim_ns))


def truth_nugget_real_units_affine(nug: float) -> float:
    """ALTERNATIVE (non-headline) physical-unit nugget: ``nug * a**2``, where
    ``a`` is the scale factor the affine correction actually applied to this
    level's realization (see ``affine_scale_factor``).

    Both this and ``truth_nugget_real_units`` are legitimate readings of "the
    truth nugget in physical units" -- one uses the TARGET sill the
    experiment asked for, the other the sill the single realization actually
    got -- so both are recorded rather than one being declared correct
    (reviewer + orchestrator decision 2026-09-22). The headline is unchanged.
    """
    a = affine_scale_factor(nug)
    return float(nug) * a * a


def run_one_nugget(nug: float) -> dict:
    """Run all 4 methods at normalized nugget ``nug``, with the range,
    seeds and sample count left at their base-case values.

    Kept as a single-level convenience wrapper (standalone/debug use) around
    ``run_levels_parallel``; ``main()`` does NOT call this once per level --
    it submits every (level x method) task to one pool together, since a
    per-level call would serialize the levels.
    """
    level = level_key(nug)
    results = run_levels_parallel(
        {
            level: dict(
                truth_seed=TRUTH_SEED,
                hmaj1=AXIS_HMAJ1,
                hmin1=AXIS_HMIN1,
                nug=float(nug),
            )
        },
        methods=METHODS,
    )
    return results[level]


def verify_run(rel_run_dir: str, nug: float, method: str) -> None:
    """Verify that ``rel_run_dir`` really belongs to nugget level ``nug``
    (and to this axis's TRUTH_SEED / SAMPLE_SEED / range / sample count).
    Raises on any mismatch -- a silently mislabeled axis level would be the
    single most damaging error this table can carry.
    """
    run_dir = _REPO_ROOT / rel_run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["params"]

    if params.get("truth_seed") != TRUTH_SEED or params.get("sample_seed") != SAMPLE_SEED:
        raise ValueError(
            f"{rel_run_dir}: truth_seed/sample_seed ({params.get('truth_seed')}/"
            f"{params.get('sample_seed')}) do not match this axis's "
            f"({TRUTH_SEED}/{SAMPLE_SEED})."
        )

    # Range must be the base-case range at EVERY nugget level
    # (one-factor-at-a-time).
    for field, expected in (("hmaj1", AXIS_HMAJ1), ("hmin1", AXIS_HMIN1)):
        recorded = params.get(field)
        if recorded is None or not np.isclose(float(recorded), expected):
            raise ValueError(
                f"{rel_run_dir}: manifest records {field}={recorded}, expected "
                f"{expected} (the nugget axis holds the range fixed at the base case)."
            )

    # Recorded nugget: top-level field written by all four methods, plus the
    # variogram dict for kriging/SGS (which actually consume it).
    recorded_nug = params.get("nug")
    if recorded_nug is None or not np.isclose(float(recorded_nug), nug):
        raise ValueError(
            f"{rel_run_dir}: manifest records nug={recorded_nug} but is being used as "
            f"the nug={nug:g} axis level."
        )
    recorded_cc1 = params.get("cc1")
    if recorded_cc1 is None or not np.isclose(float(recorded_cc1), 1.0 - nug):
        raise ValueError(
            f"{rel_run_dir}: manifest records cc1={recorded_cc1}, expected "
            f"{1.0 - nug} (unit sill: cc1 = 1 - nug)."
        )
    if method in ("kriging", "sgs"):
        vario = params.get("variogram")
        if vario is None:
            raise ValueError(f"{rel_run_dir}: {method} manifest has no variogram dict.")
        if not np.isclose(float(vario["nug"]), nug) or not np.isclose(
            float(vario["cc1"]), 1.0 - nug
        ):
            raise ValueError(
                f"{rel_run_dir}: {method}'s INPUT variogram has nug={vario['nug']}, "
                f"cc1={vario['cc1']}; expected {nug}/{1.0 - nug}. kriging/SGS must be "
                "handed this level's true nugget."
            )

    # Decisive check: the conditioning sample VALUES are read off the truth
    # field, which is regenerated per nugget level, so they are level-specific
    # even though the sample LOCATIONS are not. Regenerate and require a match.
    _, samples_df = get_base_case_conditioning_data(
        sample_seed=SAMPLE_SEED,
        truth_seed=TRUTH_SEED,
        hmaj1=AXIS_HMAJ1,
        hmin1=AXIS_HMIN1,
        n_samples=N_SAMPLES,
        nug=float(nug),
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
            f"samples regenerated at nug={nug:g} -- this run was NOT produced from "
            "that ground truth."
        )
    print(f"  verified {method} run for nug={nug:g}: {rel_run_dir}")


def _rel(p) -> str:
    return str(Path(p).relative_to(_REPO_ROOT)).replace("\\", "/")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity check that the levels really are distinct ground truths before
    # spending ~20 minutes of compute: the conditioning sample VALUES must
    # differ between levels (identical values would mean the nugget never
    # reached the truth generator).
    sample_fingerprints = {}
    for nug in ALL_NUGGET_VALUES:
        _, s = get_base_case_conditioning_data(
            sample_seed=SAMPLE_SEED,
            truth_seed=TRUTH_SEED,
            hmaj1=AXIS_HMAJ1,
            hmin1=AXIS_HMIN1,
            n_samples=N_SAMPLES,
            nug=float(nug),
        )
        sample_fingerprints[level_key(nug)] = s[VCOL].values
    keys = list(sample_fingerprints)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if np.allclose(sample_fingerprints[a], sample_fingerprints[b]):
                raise ValueError(
                    f"nugget levels {a} and {b} produce IDENTICAL conditioning sample "
                    "values -- the nugget is not reaching the truth generator."
                )
    print(
        f"pre-flight OK: all {len(keys)} nugget levels produce distinct conditioning "
        "sample values (truth fields genuinely differ)."
    )

    level_kwargs = {
        level_key(nug): dict(
            truth_seed=TRUTH_SEED,
            hmaj1=AXIS_HMAJ1,
            hmin1=AXIS_HMIN1,
            nug=float(nug),
        )
        for nug in ALL_NUGGET_VALUES
    }
    new_run_dirs = run_levels_parallel(level_kwargs, methods=METHODS)

    source_runs = {}
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        source_runs[level] = {m: _rel(new_run_dirs[level][m]) for m in METHODS}

    print("\nVerifying every run against its axis level:")
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        for m in METHODS:
            verify_run(source_runs[level][m], float(nug), m)

    missing = [lvl for lvl in source_runs if set(source_runs[lvl]) != set(METHODS)]
    if missing:
        raise ValueError(f"axis level(s) {missing} are missing a method run.")

    source_runs_path = PROCESSED_DIR / "source_runs.json"
    with open(source_runs_path, "w", encoding="utf-8") as f:
        json.dump(source_runs, f, indent=2)

    print(
        f"\nnugget_axis source runs ({len(ALL_NUGGET_VALUES)} levels x {len(METHODS)} methods):"
    )
    print(json.dumps(source_runs, indent=2))
    print("\nPhysical-unit nugget per level (Porosity %^2, total sill "
          f"{POR_STDEV ** 2:g}):")
    for nug in ALL_NUGGET_VALUES:
        print(f"  nug={level_key(nug)} -> {truth_nugget_real_units(nug):g}")
    print(f"\nsource_runs.json: {source_runs_path}")

    return source_runs


if __name__ == "__main__":
    main()
