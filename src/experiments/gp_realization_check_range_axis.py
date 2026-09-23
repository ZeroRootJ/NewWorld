"""Is the range axis's GP-MLE hyperparameter result a property of ONE
ground-truth realization?

WHY THIS EXISTS
---------------
results/processed/range_axis/ currently carries ONE ground-truth realization
per level (TRUTH_SEED=101). Two things were read off that single realization
and are currently stated as if they were properties of the method:

  1. SATURATION. The GP-MLE fitted practical range does not track the
     ground-truth variogram range: against truth 100 -> 800 m the fitted
     values are 103.16 / 182.48 / 284.26 / 335.13 / 360.74 / 374.31 / 384.69
     / 391.74 m -- visibly flat from 400 m on
     (results/figures/range_axis/00_length_scale_vs_range.png).
  2. THE NOISE TRAJECTORY. The fitted WhiteKernel noise variance FALLS from
     2.174 (100 m) to 1.245 (600 m) and then REBOUNDS to 1.397 at 800 m
     (results/figures/range_axis/length_vs_noise_gp_mle.png).

Both are single-realization observations. This script re-runs the SAME
hyperparameter fit on 10 ground-truth realizations per level so the spread
can be read off instead of assumed.

SCOPE -- HYPERPARAMETER OPTIMIZATION ONLY (do not extend this script)
---------------------------------------------------------------------
truth generation -> conditioning samples -> ``GaussianProcessRegressor.fit()``
-> record the learned hyperparameters. NOTHING ELSE: no prediction map, no
posterior draw, no variance map, no kriging/SGS/RBF, no accuracy or
calibration metric. This is a diagnostic, not a method run, so it writes NO
``results/raw/`` run directory -- its outputs are a CSV under
results/processed/range_axis/ and (via the companion figure script) two PNGs
under results/figures/range_axis/. It also never writes any pinned artifact:
length_scale_by_method.csv, gp_hyperparameter_scale_conversion.csv and
qc_summary.csv are opened READ-ONLY, as the reference the checks below
compare against.

WHAT IS VARIED AND WHAT IS HELD FIXED
-------------------------------------
VARIED (two factors, crossed, 8 x 10 = 80 fits):
  * the axis level -- ground-truth variogram range in ALL_RANGE_VALUES
    (imported from range_axis.py, not re-declared),
  * the ground-truth realization -- ``truth_seed`` in TRUTH_SEED_REPLICATES.
FIXED:
  * ``sample_seed`` = base_case_conditioning.SAMPLE_SEED (20). Sample
    LOCATIONS are drawn from the grid geometry and this seed alone, so they
    are IDENTICAL across all 80 cells; only the sample VALUES change, because
    they are read off a truth field regenerated per (level, realization).
    This script VERIFIES that (identical X/Y, differing values) rather than
    asserting it in prose -- see ``verify_sample_geometry``.
  * the nugget: base_case.NUG = 0.05 (normalized), i.e. 0.45 Porosity %^2 --
    the range axis is one-factor-at-a-time, only the range moves. It is
    passed EXPLICITLY below rather than left to a default, so the fixed
    factor is visible at the call site.
  * isotropy (hmin1 == hmaj1 at every level), N_SAMPLES, grid, porosity
    distribution,
  * every GP configuration constant: kernel structure, bounds, initial
    values, ``n_restarts_optimizer``, ``normalize_y`` and ``random_state``
    are taken from ``src.experiments.gp_mle.build_gpr()`` BY IMPORT. They are
    deliberately NOT re-declared here -- a re-declaration would let this
    diagnostic drift from the pinned production runs and make the comparison
    meaningless.

WHY THESE TRUTH SEEDS
---------------------
TRUTH_SEED_REPLICATES = (101, 7101 ... 7109) -- the SAME block the nugget
axis's equivalent diagnostic used
(src/experiments/gp_realization_check_nugget_axis.py), reused on purpose for
two reasons. First, comparability: the two axes then differ only in the
factor each varies, not in which realizations happened to be drawn, so a
statement like "the fitted noise spread is wider on the range axis than on
the nugget axis" is about the axis and not about seed choice. Second,
seed hygiene: minting a fresh block per axis would inflate the list of
"seeds used somewhere in this repo" with no gain, and that list is what
makes the non-collision argument checkable. The block itself was chosen (in
the nugget-axis task) to collide with none of 101, 20, 1001-1010,
30/40/50/55/60/70/80/85/86/90, the 9000/9100/9200/9300/9400 GP-draw
families, 73073, 12345, or the small literals 7/11/13/300/1000.
The axis's own pinned TRUTH_SEED is first, so the pinned realization is one
row of every level and its reproduction can be checked.

BUILT-IN REPRODUCTION CHECKS (the script stops if either fails)
---------------------------------------------------------------
1. CONDITIONING DATA. At truth_seed=101 the truth+sampling path used here
   must reproduce the PINNED gp_mle run's samples.csv (the run named in
   results/processed/range_axis/source_runs.json) exactly, at all 8 levels.
2. FITTED HYPERPARAMETERS. The truth_seed=101 rows must reproduce the PINNED
   fitted values in results/processed/range_axis/qc_summary.csv and
   gp_hyperparameter_scale_conversion.csv -- and, for the normalized
   quantities those two CSVs do not carry, the ``fitted_hyperparameters``
   block of the pinned gp_mle manifests -- to PINNED_RTOL (1e-9 relative;
   the fit is deterministic and observed agreement is ~1e-15, so this
   tolerance only absorbs float text round-trip and BLAS reduction-order
   noise).
A mismatch would mean the imported configuration is no longer the one the
pinned runs used, which invalidates the whole comparison -- so it raises
rather than writing a table.

Output
------
  results/processed/range_axis/gp_hyperparameters_by_realization.csv
    one row per (axis level, realization).
  results/processed/range_axis/gp_hyperparameters_by_realization_pinned_check.csv
    the observed relative differences of check 2.

Run with:
.venv/Scripts/python.exe -m src.experiments.gp_realization_check_range_axis
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.base_case import NUG as AXIS_NUG
from src.experiments.base_case_conditioning import (
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_conditioning_data,
)
from src.experiments.gp_mle import (
    CONSTANT_VALUE_BOUNDS,
    LENGTH_SCALE_BOUNDS,
    NOISE_LEVEL_BOUNDS,
    build_gpr,
    extract_fitted_hyperparameters,
)
from src.experiments.nugget_axis import truth_nugget_real_units
from src.experiments.qc_nugget_axis import GP_PRACTICAL_RANGE_FACTOR
from src.experiments.qc_range_axis import _at_bound
from src.experiments.range_axis import ALL_RANGE_VALUES

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"

# Ground-truth realizations -- see "WHY THESE TRUTH SEEDS" above. Identical
# to gp_realization_check_nugget_axis.TRUTH_SEED_REPLICATES by design; the
# equality is asserted at import time below so the two cannot silently drift.
TRUTH_SEED_REPLICATES = (TRUTH_SEED, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109)

# Relative tolerance for the truth_seed=101 reproduction check against the
# pinned artifacts. The fit is deterministic; observed agreement is ~1e-15
# relative, so this only absorbs CSV float round-trip / BLAS summation-order
# noise. Absolute tolerance is 0 -- every compared quantity is nonzero.
PINNED_RTOL = 1e-9

# Sample X/Y coordinates are floats produced by the identical RNG call at
# every cell, so they should be bit-identical; this is the tolerance used
# when asserting that (same convention as nugget_axis.SAMPLES_MATCH_ATOL).
SAMPLE_COORD_ATOL = 1e-12

# Float-text round-trip tolerance for the samples.csv comparison.
SAMPLES_MATCH_ATOL = 1e-10

OUT_CSV = PROCESSED_DIR / "gp_hyperparameters_by_realization.csv"
CHECK_CSV = PROCESSED_DIR / "gp_hyperparameters_by_realization_pinned_check.csv"


def _assert_shared_seed_block() -> None:
    """The docstring claims this block is the SAME one the nugget-axis
    diagnostic used (so the two axes are comparable). Check it instead of
    claiming it, and fail loudly if either side is ever edited alone."""
    from src.experiments.gp_realization_check_nugget_axis import (
        TRUTH_SEED_REPLICATES as NUGGET_SEEDS,
    )

    if tuple(TRUTH_SEED_REPLICATES) != tuple(NUGGET_SEEDS):
        raise ValueError(
            f"TRUTH_SEED_REPLICATES here {TRUTH_SEED_REPLICATES} differs from the "
            f"nugget axis's {tuple(NUGGET_SEEDS)}. The two diagnostics deliberately "
            "share one realization block so their spreads are comparable -- change "
            "both or neither."
        )


def fit_one(hmaj1: float, truth_seed: int) -> dict:
    """Generate one (level, realization) ground truth, draw the fixed-location
    conditioning samples from it, fit the production GP, and return one row of
    recorded hyperparameters.

    The truth+sampling pair comes from ``get_base_case_conditioning_data`` --
    the SAME call ``gp_mle.main()`` makes -- with the range passed as
    ``hmaj1 == hmin1`` (isotropic, as range_axis.py defines the axis) and the
    nugget pinned at the base-case value. Nothing about the truth is handed to
    the GP: the WhiteKernel noise level and the RBF length scale are learned
    by marginal-likelihood maximization from the samples alone, which is
    precisely the quantity under test.
    """
    truth, samples_df = get_base_case_conditioning_data(
        sample_seed=SAMPLE_SEED,
        truth_seed=int(truth_seed),
        hmaj1=float(hmaj1),
        hmin1=float(hmaj1),  # isotropic range axis
        n_samples=N_SAMPLES,
        nug=AXIS_NUG,  # fixed: only the range moves on this axis
    )

    X = samples_df[["X", "Y"]].values
    y = samples_df[VCOL].values

    gpr = build_gpr()
    gpr.fit(X, y)
    hp = extract_fitted_hyperparameters(gpr)

    signal_norm = hp["signal_variance_normalized"]
    noise_norm = hp["noise_variance_normalized"]
    noise_real = hp["noise_variance_real_units"]
    length_scale = hp["length_scale_m"]
    practical_range = GP_PRACTICAL_RANGE_FACTOR * length_scale

    truth_nug = truth_nugget_real_units(AXIS_NUG)

    # Exact boundary stop on the WhiteKernel LOWER bound -- same definition as
    # gp_realization_check_nugget_axis.py. There is no manifest to read the
    # bound from here (this is not a run), so the bound comes from the same
    # gp_mle constant the pinned runs were configured with.
    noise_floor = float(NOISE_LEVEL_BOUNDS[0])

    return {
        "axis_level": str(int(hmaj1)),
        "range_m": float(hmaj1),
        "truth_seed": int(truth_seed),
        "is_pinned_realization": int(truth_seed) == TRUTH_SEED,
        # --- fitted GP hyperparameters ---
        "gp_length_scale_m": length_scale,
        "gp_practical_range_m": practical_range,
        "gp_noise_variance_real_units": noise_real,
        "gp_noise_variance_normalized": noise_norm,
        "gp_signal_variance_normalized": signal_norm,
        "gp_signal_variance_real_units": hp["signal_variance_real_units"],
        "gp_noise_fraction_of_fitted_sill": noise_norm / (noise_norm + signal_norm),
        "gp_y_train_std": hp["y_train_std"],
        "gp_log_marginal_likelihood": float(gpr.log_marginal_likelihood(gpr.kernel_.theta)),
        # --- optimizer boundary flags ---
        "gp_noise_at_optimizer_floor": bool(
            abs(noise_norm - noise_floor) <= 1e-9 * noise_floor
        ),
        "gp_length_scale_at_bound": bool(_at_bound(length_scale, LENGTH_SCALE_BOUNDS)),
        "gp_signal_var_at_bound": bool(_at_bound(signal_norm, CONSTANT_VALUE_BOUNDS)),
        "gp_noise_var_at_bound": bool(_at_bound(noise_norm, NOISE_LEVEL_BOUNDS)),
        "gp_noise_level_lower_bound": noise_floor,
        # --- ground truth side ---
        "truth_range_m": float(hmaj1),
        "truth_nugget_normalized": float(AXIS_NUG),
        "truth_nugget_real_units": truth_nug,
        "truth_field_variance": float(np.var(truth)),
        # --- conditioning samples ---
        "n_samples_actual": int(len(samples_df)),
        "sample_mean": float(np.mean(y)),
        "sample_variance": float(np.var(y)),
        # --- derived comparisons (the headline question) ---
        "gp_practical_range_over_truth_range": practical_range / float(hmaj1),
        "gp_practical_range_minus_truth_range": practical_range - float(hmaj1),
        "gp_noise_over_truth_nugget": noise_real / truth_nug,
        # internal (dropped before writing): the sample geometry fingerprint
        "_sample_xy": samples_df[["X", "Y"]].values,
        "_sample_values": y,
    }


def verify_sample_geometry(rows) -> dict:
    """Verify -- not assume -- the design claim that sample LOCATIONS are
    identical across every (level, realization) while sample VALUES differ
    between realizations.

    Raises if the locations are not identical, or if any two realizations at
    the same level produced identical sample values (which would mean the
    truth seed never reached the generator).
    """
    ref_xy = rows[0]["_sample_xy"]
    ref_n = len(ref_xy)
    for r in rows:
        if len(r["_sample_xy"]) != ref_n or not np.allclose(
            r["_sample_xy"], ref_xy, rtol=0.0, atol=SAMPLE_COORD_ATOL
        ):
            raise ValueError(
                f"sample LOCATIONS differ at range={r['axis_level']} m, "
                f"truth_seed={r['truth_seed']} -- the design requires identical "
                "sample locations across all cells (sample_seed is fixed)."
            )
    max_coord_dev = max(float(np.max(np.abs(r["_sample_xy"] - ref_xy))) for r in rows)

    n_pairs = 0
    min_value_diff = np.inf
    by_level = {}
    for r in rows:
        by_level.setdefault(r["axis_level"], []).append(r)
    for level, level_rows in by_level.items():
        for i, a in enumerate(level_rows):
            for b in level_rows[i + 1:]:
                d = float(np.max(np.abs(a["_sample_values"] - b["_sample_values"])))
                min_value_diff = min(min_value_diff, d)
                n_pairs += 1
                if d == 0.0:
                    raise ValueError(
                        f"range={level} m: truth_seed {a['truth_seed']} and "
                        f"{b['truth_seed']} produced IDENTICAL sample values -- the "
                        "truth seed is not reaching the generator."
                    )
    result = {
        "n_cells": len(rows),
        "n_samples": ref_n,
        "max_abs_coordinate_deviation": max_coord_dev,
        "n_within_level_realization_pairs": n_pairs,
        "min_max_abs_value_difference_over_pairs": min_value_diff,
    }
    print(
        f"sample-geometry check PASSED: all {len(rows)} cells share the SAME "
        f"{ref_n} sample locations (max |X/Y| deviation {max_coord_dev:g}); across "
        f"{n_pairs} within-level realization pairs the sample VALUES always differ "
        f"(smallest max|dv| over any pair = {min_value_diff:.4f} Porosity %)."
    )
    return result


def verify_practical_range_factor() -> None:
    """The practical-range conversion used here must be the one the pinned
    range-axis artifact was written with -- it is imported (not re-derived),
    and compared against the factor recorded in the pinned CSV."""
    conv = pd.read_csv(PROCESSED_DIR / "gp_hyperparameter_scale_conversion.csv")
    pinned_factors = conv["practical_range_factor_exact"].unique()
    if len(pinned_factors) != 1 or not np.isclose(
        float(pinned_factors[0]), GP_PRACTICAL_RANGE_FACTOR, rtol=PINNED_RTOL, atol=0.0
    ):
        raise ValueError(
            f"practical-range factor mismatch: imported {GP_PRACTICAL_RANGE_FACTOR!r} vs. "
            f"pinned gp_hyperparameter_scale_conversion.practical_range_factor_exact "
            f"{pinned_factors!r}."
        )
    print(
        f"practical-range-factor check PASSED: imported factor "
        f"{GP_PRACTICAL_RANGE_FACTOR:.10f} == the pinned artifact's "
        "practical_range_factor_exact (0.05 correlation cutoff)."
    )


def verify_against_pinned_samples() -> None:
    """Check that the truth+sampling path used here reproduces the exact
    conditioning data of the PINNED gp_mle runs (samples.csv of the run in
    source_runs.json), at every level, for truth_seed=101."""
    source_runs = json.loads(
        (PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8")
    )
    for r in ALL_RANGE_VALUES:
        level = str(int(r))
        _, here = get_base_case_conditioning_data(
            sample_seed=SAMPLE_SEED,
            truth_seed=TRUTH_SEED,
            hmaj1=float(r),
            hmin1=float(r),
            n_samples=N_SAMPLES,
            nug=AXIS_NUG,
        )
        pinned = pd.read_csv(_REPO_ROOT / source_runs[level]["gp_mle"] / "samples.csv")
        if len(here) != len(pinned) or not np.allclose(
            here[["X", "Y", VCOL]].values,
            pinned[["X", "Y", VCOL]].values,
            rtol=0.0,
            atol=SAMPLES_MATCH_ATOL,
        ):
            raise ValueError(
                f"range={level} m: conditioning data regenerated here does not match the "
                f"pinned gp_mle run's samples.csv ({source_runs[level]['gp_mle']})."
            )
    print(
        f"conditioning-data check PASSED: at truth_seed={TRUTH_SEED} this script's "
        "truth+sampling path reproduces the pinned gp_mle runs' samples.csv exactly at "
        f"all {len(ALL_RANGE_VALUES)} levels."
    )


def verify_against_pinned_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    """The decisive configuration check: the truth_seed=101 rows must
    reproduce the PINNED fitted hyperparameters recorded in qc_summary.csv,
    gp_hyperparameter_scale_conversion.csv, and (for the normalized
    quantities neither CSV carries) the pinned gp_mle manifests' recorded
    ``fitted_hyperparameters`` block -- all to PINNED_RTOL.

    Raises on any mismatch (see module docstring for why this is fatal).
    Returns a per-level table of the observed relative differences.
    """
    qc = pd.read_csv(PROCESSED_DIR / "qc_summary.csv")
    qc["axis_level"] = qc["axis_level"].astype(str)
    conv = pd.read_csv(PROCESSED_DIR / "gp_hyperparameter_scale_conversion.csv")
    conv["axis_level"] = conv["axis_level"].astype(str)
    source_runs = json.loads(
        (PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8")
    )

    # (column here, pinned table, pinned column)
    csv_checks = [
        ("gp_length_scale_m", "qc_summary", "gp_length_scale_m"),
        ("gp_signal_variance_real_units", "qc_summary", "gp_signal_variance_real"),
        ("gp_noise_variance_real_units", "qc_summary", "gp_noise_variance_real"),
        ("gp_log_marginal_likelihood", "qc_summary", "gp_log_marginal_likelihood"),
        ("n_samples_actual", "qc_summary", "n_samples"),
        ("gp_length_scale_m", "scale_conversion", "gp_length_scale_m"),
        ("gp_practical_range_m", "scale_conversion", "practical_range_m"),
        (
            "gp_practical_range_over_truth_range",
            "scale_conversion",
            "practical_range_over_truth_range",
        ),
        (
            "gp_noise_variance_real_units",
            "scale_conversion",
            "gp_noise_variance_real_units",
        ),
        (
            "gp_signal_variance_real_units",
            "scale_conversion",
            "gp_signal_variance_real_units",
        ),
        ("truth_nugget_real_units", "scale_conversion", "truth_nugget_real_units"),
        ("gp_noise_over_truth_nugget", "scale_conversion", "gp_noise_over_truth_nugget"),
        ("truth_range_m", "scale_conversion", "truth_spherical_range_m"),
    ]
    pinned_tables = {"qc_summary": qc, "scale_conversion": conv}

    # Quantities no processed CSV carries -> read them from the pinned run's
    # own manifest (params.fitted_hyperparameters), which is where gp_mle.py
    # recorded them.
    manifest_checks = [
        ("gp_signal_variance_normalized", "signal_variance_normalized"),
        ("gp_noise_variance_normalized", "noise_variance_normalized"),
        ("gp_y_train_std", "y_train_std"),
        ("gp_length_scale_m", "length_scale_m"),
    ]

    recs = []
    for r in ALL_RANGE_VALUES:
        level = str(int(r))
        here = df[(df["axis_level"] == level) & (df["truth_seed"] == TRUTH_SEED)]
        if len(here) != 1:
            raise ValueError(
                f"expected exactly 1 pinned row for range={level} m, got {len(here)}"
            )
        here = here.iloc[0]

        for col, table_name, pinned_col in csv_checks:
            table = pinned_tables[table_name]
            prow = table[table["axis_level"] == level]
            if len(prow) != 1:
                raise ValueError(
                    f"{table_name}.csv: expected 1 row for axis_level={level}, got {len(prow)}"
                )
            recs.append(
                _compare(level, col, f"{table_name}.{pinned_col}",
                         float(prow[pinned_col].iloc[0]), float(here[col]))
            )

        manifest = json.loads(
            (_REPO_ROOT / source_runs[level]["gp_mle"] / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        fitted = manifest["params"]["fitted_hyperparameters"]
        for col, key in manifest_checks:
            recs.append(
                _compare(level, col, f"manifest.fitted_hyperparameters.{key}",
                         float(fitted[key]), float(here[col]))
            )

    check_df = pd.DataFrame(recs)
    bad = check_df[check_df["relative_difference"] > PINNED_RTOL]
    if len(bad):
        b = bad.iloc[0]
        raise ValueError(
            f"REPRODUCTION FAILURE at range={b['axis_level']} m, {b['quantity']}: "
            f"recomputed {b['recomputed_value']!r} vs. pinned {b['pinned_source']} = "
            f"{b['pinned_value']!r} (relative difference {b['relative_difference']:.3e} > "
            f"PINNED_RTOL={PINNED_RTOL:g}); {len(bad)} of {len(check_df)} comparisons "
            "failed. The GP configuration imported from gp_mle.py is no longer the one "
            "the pinned runs used -- STOPPING; the multi-realization table would not be "
            "comparable to the pinned result."
        )
    n_q = len(csv_checks) + len(manifest_checks)
    print(
        f"pinned-reproduction check PASSED: {len(check_df)} comparisons "
        f"({len(ALL_RANGE_VALUES)} levels x {n_q} quantities), max relative difference "
        f"{check_df['relative_difference'].max():.3e} (tolerance {PINNED_RTOL:g})."
    )
    return check_df


def _compare(level, col, source, pinned_val, here_val) -> dict:
    rel = abs(here_val - pinned_val) / abs(pinned_val) if pinned_val != 0 else abs(here_val)
    return {
        "axis_level": level,
        "quantity": col,
        "pinned_source": source,
        "pinned_value": pinned_val,
        "recomputed_value": here_val,
        "relative_difference": rel,
    }


def main():
    _assert_shared_seed_block()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    verify_practical_range_factor()
    verify_against_pinned_samples()

    rows = []
    n_cells = len(ALL_RANGE_VALUES) * len(TRUTH_SEED_REPLICATES)
    for r in ALL_RANGE_VALUES:
        for truth_seed in TRUTH_SEED_REPLICATES:
            row = fit_one(float(r), int(truth_seed))
            rows.append(row)
            print(
                f"  [{len(rows):>2}/{n_cells}] range={row['axis_level']:>3}m "
                f"seed={truth_seed:<5} length_scale={row['gp_length_scale_m']:8.3f} m  "
                f"practical={row['gp_practical_range_m']:8.2f} m "
                f"(fitted/GT {row['gp_practical_range_over_truth_range']:.3f})  "
                f"noise={row['gp_noise_variance_real_units']:7.4f}  "
                f"LML={row['gp_log_marginal_likelihood']:.3f}  "
                f"[{time.time() - t0:.0f}s]"
            )

    geometry = verify_sample_geometry(rows)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df = df.sort_values(["range_m", "truth_seed"]).reset_index(drop=True)

    check_df = verify_against_pinned_hyperparameters(df)

    # The headline question, recorded as a column so the figure script and any
    # later reader get the same boolean rather than recomputing the comparison.
    df["fitted_range_below_truth_range"] = (
        df["gp_practical_range_m"] < df["truth_range_m"]
    )

    df.to_csv(OUT_CSV, index=False)
    check_df.to_csv(CHECK_CSV, index=False)

    # --- Printed summary (numbers only) ----------------------------------
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 100)

    n_below = int(df["fitted_range_below_truth_range"].sum())
    print(
        f"\nfitted practical range < ground-truth range: {n_below} of {len(df)} fits "
        f"({n_below / len(df):.1%})"
    )
    print("\nfitted practical range (m) by level:")
    print(
        df.groupby("axis_level", sort=False)
        .agg(
            truth_range=("truth_range_m", "first"),
            n_below=("fitted_range_below_truth_range", "sum"),
            n=("fitted_range_below_truth_range", "size"),
            mean=("gp_practical_range_m", "mean"),
            std=("gp_practical_range_m", "std"),
            min=("gp_practical_range_m", "min"),
            max=("gp_practical_range_m", "max"),
        )
        .sort_values("truth_range")
        .to_string()
    )
    print("\nfitted/GT practical-range ratio by level:")
    print(
        df.groupby("axis_level", sort=False)
        .agg(
            truth_range=("truth_range_m", "first"),
            mean=("gp_practical_range_over_truth_range", "mean"),
            std=("gp_practical_range_over_truth_range", "std"),
            min=("gp_practical_range_over_truth_range", "min"),
            max=("gp_practical_range_over_truth_range", "max"),
        )
        .sort_values("truth_range")
        .to_string()
    )
    print("\nfitted noise variance (Porosity %^2) by level:")
    print(
        df.groupby("axis_level", sort=False)
        .agg(
            truth_range=("truth_range_m", "first"),
            mean=("gp_noise_variance_real_units", "mean"),
            std=("gp_noise_variance_real_units", "std"),
            min=("gp_noise_variance_real_units", "min"),
            max=("gp_noise_variance_real_units", "max"),
        )
        .sort_values("truth_range")
        .to_string()
    )

    # Where does the pinned realization sit inside each level's 10 values?
    print("\npinned realization's rank within each level (1 = smallest of 10):")
    rank_recs = []
    for level, grp in df.groupby("axis_level", sort=False):
        pin = grp[grp["is_pinned_realization"].astype(bool)].iloc[0]
        for q in ["gp_practical_range_m", "gp_noise_variance_real_units"]:
            rank_recs.append(
                {
                    "axis_level": level,
                    "truth_range_m": float(pin["truth_range_m"]),
                    "quantity": q,
                    "pinned_value": float(pin[q]),
                    "rank_of_10": int((grp[q] < pin[q]).sum()) + 1,
                    "level_min": float(grp[q].min()),
                    "level_max": float(grp[q].max()),
                }
            )
    rank_df = pd.DataFrame(rank_recs).sort_values(["quantity", "truth_range_m"])
    print(rank_df.to_string(index=False))

    bound_cols = [
        "gp_noise_at_optimizer_floor",
        "gp_length_scale_at_bound",
        "gp_signal_var_at_bound",
        "gp_noise_var_at_bound",
    ]
    print("\nfits touching an optimizer bound:")
    for c in bound_cols:
        hits = df[df[c]]
        print(f"  {c}: {len(hits)}")
        for _, h in hits.iterrows():
            print(f"      range={h['axis_level']}m seed={h['truth_seed']}")

    print(f"\nsample geometry: {geometry}")
    print(f"\ngp_hyperparameters_by_realization.csv: {OUT_CSV}")
    print(f"pinned check table: {CHECK_CSV}")
    print(f"total {time.time() - t0:.0f}s")
    return df


if __name__ == "__main__":
    main()
