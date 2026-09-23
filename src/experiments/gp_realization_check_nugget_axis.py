"""Is the nugget axis's GP-MLE hyperparameter result a property of ONE
ground-truth realization?

WHY THIS EXISTS
---------------
results/processed/nugget_axis/ currently carries ONE ground-truth realization
per level (TRUTH_SEED=101). On that single realization GP-MLE's
marginal-likelihood fit learned a WhiteKernel noise variance LARGER than the
truth nugget at all six levels (1.140 / 1.960 / 2.661 / 3.387 / 4.178 / 6.003
against truth 0 / 0.9 / 1.8 / 2.7 / 3.6 / 4.5 Porosity %^2), including 1.140
at the level whose true nugget is exactly zero; and the fitted length_scale
jumped from ~126 m to 557.5 m (practical range 1364.7 m) at nug=0.5. Both are
single-realization observations. This script re-runs the SAME hyperparameter
fit on 10 ground-truth realizations per level so the spread can be read off
instead of assumed.

SCOPE -- HYPERPARAMETER OPTIMIZATION ONLY (do not extend this script)
---------------------------------------------------------------------
truth generation -> conditioning samples -> ``GaussianProcessRegressor.fit()``
-> record the learned hyperparameters. NOTHING ELSE: no prediction map, no
posterior draw, no variance map, no kriging/SGS/RBF, no accuracy or
calibration metric. This is a diagnostic, not a method run, so it writes NO
``results/raw/`` run directory -- its outputs are a CSV under
results/processed/nugget_axis/ and (via the companion figure script) two PNGs
under results/figures/nugget_axis/.

WHAT IS VARIED AND WHAT IS HELD FIXED
-------------------------------------
VARIED (two factors, crossed, 6 x 10 = 60 fits):
  * the axis level -- normalized nugget in ALL_NUGGET_VALUES (imported from
    nugget_axis.py, not re-declared),
  * the ground-truth realization -- ``truth_seed`` in TRUTH_SEED_REPLICATES.
FIXED:
  * ``sample_seed`` = base_case_conditioning.SAMPLE_SEED (20). Sample
    LOCATIONS are drawn from the grid geometry and this seed alone, so they
    are IDENTICAL across all 60 cells; only the sample VALUES change, because
    they are read off a truth field regenerated per (level, realization).
    This script VERIFIES that (identical X/Y, differing values) rather than
    asserting it in prose -- see ``verify_sample_geometry``.
  * range 300 m isotropic (AXIS_HMAJ1 / AXIS_HMIN1), N_SAMPLES, grid,
    porosity distribution,
  * every GP configuration constant: kernel structure, bounds, initial
    values, ``n_restarts_optimizer``, ``normalize_y`` and ``random_state``
    are taken from ``src.experiments.gp_mle.build_gpr()`` BY IMPORT. They are
    deliberately NOT re-declared here -- a re-declaration would let this
    diagnostic drift from the pinned production runs and make the comparison
    meaningless.

WHY THESE TRUTH SEEDS
---------------------
TRUTH_SEED_REPLICATES = (101, 7101 ... 7109): the axis's own pinned
TRUTH_SEED first (so the pinned realization is one row of the table and its
reproduction can be checked), then a 9-seed block chosen to avoid every seed
already used anywhere in this repository -- 101, 20, 1001-1010, 30/40/50/55/
60/70/80/85/86/90, the 9000/9100/9200/9300/9400 GP-draw families, 73073,
12345, and the small literals 7/11/13/300/1000. The 7101-7109 block collides
with none of them and is contiguous, so "which realizations were used" is one
line to state in the paper.

BUILT-IN REPRODUCTION CHECK (the script stops if it fails)
----------------------------------------------------------
The truth_seed=101 row of every level must reproduce the PINNED values in
results/processed/nugget_axis/qc_summary.csv and
gp_fitted_nugget_vs_truth.csv to PINNED_RTOL (1e-9 relative; the fit is
deterministic and observed agreement is ~1e-15, so this tolerance only
absorbs float text round-trip and BLAS reduction-order noise). A mismatch
would mean the imported configuration is no longer the one the pinned runs
used, which invalidates the whole comparison -- so it raises rather than
writing a table.

Output
------
  results/processed/nugget_axis/gp_hyperparameters_by_realization.csv
    one row per (axis level, realization).

Run with:
.venv/Scripts/python.exe -m src.experiments.gp_realization_check_nugget_axis
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.base_case import (
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
    get_conditioning_samples,
)
from src.experiments.gp_mle import (
    CONSTANT_VALUE_BOUNDS,
    LENGTH_SCALE_BOUNDS,
    NOISE_LEVEL_BOUNDS,
    build_gpr,
    extract_fitted_hyperparameters,
)
from src.experiments.nugget_axis import (
    ALL_NUGGET_VALUES,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    affine_scale_factor_from_sim_ns,
    level_key,
    truth_nugget_real_units,
)
from src.experiments.qc_nugget_axis import (
    GP_PRACTICAL_RANGE_FACTOR,
    _at_bound,
)
from src.truth_model import make_porosity_truth_with_sim_ns

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"

# Ground-truth realizations. First entry is the axis's pinned TRUTH_SEED, so
# the pinned realization is row 0 of every level and its reproduction is
# checked against the pinned artifacts. The 7101-7109 block is chosen for
# non-collision with every seed already used in this repo (see the module
# docstring for the full list that was avoided).
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

OUT_CSV = PROCESSED_DIR / "gp_hyperparameters_by_realization.csv"


def fit_one(nug: float, truth_seed: int) -> dict:
    """Generate one (level, realization) ground truth, draw the fixed-location
    conditioning samples from it, fit the production GP, and return one row of
    recorded hyperparameters.

    The truth is generated with ``make_porosity_truth_with_sim_ns`` rather
    than ``get_base_case_conditioning_data`` for ONE reason: the raw
    standard-normal simulation ``sim_ns`` is needed for this realization's own
    affine scale factor ``a`` (``a`` differs per realization, so the
    ``nug * a**2`` alternative truth-nugget column cannot be taken from the
    pinned realization). Both entry points share a single sgsim call site
    (src/truth_model.py), and the sample draw below is the same
    ``get_conditioning_samples`` call ``get_base_case_conditioning_data``
    makes, so the conditioning data is identical to the production path --
    verified for truth_seed=101 in ``verify_against_pinned_samples``.
    """
    vario = build_vario(hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1, nug=float(nug))
    truth, sim_ns = make_porosity_truth_with_sim_ns(
        nx=NX, ny=NY, xsiz=XSIZ, ysiz=YSIZ, xmn=XMN, ymn=YMN,
        vario=vario, mean=POR_MEAN, stdev=POR_STDEV, seed=int(truth_seed),
    )
    samples_df = get_conditioning_samples(
        truth, sample_seed=SAMPLE_SEED, n_samples=N_SAMPLES
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

    a = affine_scale_factor_from_sim_ns(sim_ns)
    truth_nug = truth_nugget_real_units(nug)
    truth_nug_affine = float(nug) * a * a

    # Exact boundary stop on the WhiteKernel LOWER bound -- same definition as
    # qc_nugget_axis.py / make_nugget_fitted_vs_truth_figure.py. There is no
    # manifest to read the bound from here (this is not a run), so the bound
    # comes from the same gp_mle constant those runs were configured with.
    noise_floor = float(NOISE_LEVEL_BOUNDS[0])

    return {
        "axis_level": level_key(nug),
        "nug_normalized": float(nug),
        "truth_seed": int(truth_seed),
        "is_pinned_realization": int(truth_seed) == TRUTH_SEED,
        # --- fitted GP hyperparameters ---
        "gp_length_scale_m": length_scale,
        "gp_practical_range_m": GP_PRACTICAL_RANGE_FACTOR * length_scale,
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
        "truth_nugget_real_units": truth_nug,
        "affine_scale_factor_a": a,
        "affine_scale_factor_a_squared": a * a,
        "truth_nugget_real_units_affine": truth_nug_affine,
        "truth_field_variance": float(np.var(truth)),
        # --- conditioning samples ---
        "n_samples_actual": int(len(samples_df)),
        "sample_mean": float(np.mean(y)),
        "sample_variance": float(np.var(y)),
        # --- derived comparisons ---
        "gp_noise_over_truth_nugget": (noise_real / truth_nug if truth_nug > 0 else np.nan),
        "gp_noise_minus_truth_nugget": noise_real - truth_nug,
        "gp_noise_over_truth_nugget_affine": (
            noise_real / truth_nug_affine if truth_nug_affine > 0 else np.nan
        ),
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
                f"sample LOCATIONS differ at nug={r['axis_level']}, "
                f"truth_seed={r['truth_seed']} -- the design requires identical "
                "sample locations across all cells (sample_seed is fixed)."
            )
    max_coord_dev = max(
        float(np.max(np.abs(r["_sample_xy"] - ref_xy))) for r in rows
    )

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
                        f"nug={level}: truth_seed {a['truth_seed']} and "
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


def verify_against_pinned_samples() -> None:
    """Check that the truth+sampling path used here reproduces the exact
    conditioning data of the PINNED gp_mle runs (samples.csv of the run in
    source_runs.json), at every level, for truth_seed=101."""
    source_runs = json.loads(
        (PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8")
    )
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        vario = build_vario(hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1, nug=float(nug))
        truth, _ = make_porosity_truth_with_sim_ns(
            nx=NX, ny=NY, xsiz=XSIZ, ysiz=YSIZ, xmn=XMN, ymn=YMN,
            vario=vario, mean=POR_MEAN, stdev=POR_STDEV, seed=TRUTH_SEED,
        )
        here = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED, n_samples=N_SAMPLES)
        pinned = pd.read_csv(_REPO_ROOT / source_runs[level]["gp_mle"] / "samples.csv")
        if len(here) != len(pinned) or not np.allclose(
            here[["X", "Y", VCOL]].values,
            pinned[["X", "Y", VCOL]].values,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                f"nug={level}: conditioning data regenerated here does not match the "
                f"pinned gp_mle run's samples.csv ({source_runs[level]['gp_mle']})."
            )
    print(
        "conditioning-data check PASSED: at truth_seed=101 this script's truth+sampling "
        "path reproduces the pinned gp_mle runs' samples.csv exactly at all 6 levels."
    )


def verify_against_pinned_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    """The decisive configuration check: the truth_seed=101 rows must
    reproduce the PINNED fitted hyperparameters recorded in
    qc_summary.csv and gp_fitted_nugget_vs_truth.csv to PINNED_RTOL.

    Raises on any mismatch (see module docstring for why this is fatal).
    Returns a per-level table of the observed relative differences.
    """
    qc = pd.read_csv(PROCESSED_DIR / "qc_summary.csv")
    qc["axis_level"] = qc["axis_level"].astype(str)
    fv = pd.read_csv(PROCESSED_DIR / "gp_fitted_nugget_vs_truth.csv")
    fv["axis_level"] = fv["axis_level"].astype(str)

    # (column here, pinned file, pinned column)
    checks = [
        ("gp_length_scale_m", "qc_summary", "gp_length_scale_m"),
        ("gp_practical_range_m", "qc_summary", "gp_practical_range_m"),
        ("gp_noise_variance_real_units", "qc_summary", "gp_noise_variance_real"),
        ("gp_noise_variance_normalized", "qc_summary", "gp_noise_variance_normalized"),
        ("gp_signal_variance_real_units", "qc_summary", "gp_signal_variance_real"),
        ("gp_log_marginal_likelihood", "qc_summary", "gp_log_marginal_likelihood"),
        ("gp_noise_variance_real_units", "fitted_vs_truth", "gp_noise_variance_real_units"),
        ("gp_signal_variance_normalized", "fitted_vs_truth", "gp_signal_variance_normalized"),
        ("gp_noise_fraction_of_fitted_sill", "fitted_vs_truth", "gp_noise_fraction_of_fitted_sill"),
        ("gp_y_train_std", "fitted_vs_truth", "gp_y_train_std"),
        ("affine_scale_factor_a_squared", "fitted_vs_truth", "affine_scale_factor_a_squared"),
        ("truth_nugget_real_units_affine", "fitted_vs_truth", "truth_nugget_real_units_affine"),
    ]
    pinned_tables = {"qc_summary": qc, "fitted_vs_truth": fv}

    recs = []
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        here = df[(df["axis_level"] == level) & (df["truth_seed"] == TRUTH_SEED)]
        if len(here) != 1:
            raise ValueError(f"expected exactly 1 pinned row for nug={level}, got {len(here)}")
        here = here.iloc[0]
        for col, table_name, pinned_col in checks:
            table = pinned_tables[table_name]
            prow = table[table["axis_level"] == level]
            if len(prow) != 1:
                raise ValueError(
                    f"{table_name}.csv: expected 1 row for axis_level={level}, got {len(prow)}"
                )
            pinned_val = float(prow[pinned_col].iloc[0])
            here_val = float(here[col])
            rel = abs(here_val - pinned_val) / abs(pinned_val) if pinned_val != 0 else abs(here_val)
            recs.append(
                {
                    "axis_level": level,
                    "quantity": col,
                    "pinned_source": f"{table_name}.{pinned_col}",
                    "pinned_value": pinned_val,
                    "recomputed_value": here_val,
                    "relative_difference": rel,
                }
            )
            if rel > PINNED_RTOL:
                raise ValueError(
                    f"REPRODUCTION FAILURE at nug={level}, {col}: recomputed "
                    f"{here_val!r} vs. pinned {table_name}.{pinned_col} = {pinned_val!r} "
                    f"(relative difference {rel:.3e} > PINNED_RTOL={PINNED_RTOL:g}). The GP "
                    "configuration imported from gp_mle.py is no longer the one the pinned "
                    "runs used -- STOPPING; the multi-realization table would not be "
                    "comparable to the pinned result."
                )
    check_df = pd.DataFrame(recs)
    print(
        f"pinned-reproduction check PASSED: {len(check_df)} comparisons "
        f"(6 levels x {len(checks)} quantities), max relative difference "
        f"{check_df['relative_difference'].max():.3e} (tolerance {PINNED_RTOL:g})."
    )
    return check_df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    verify_against_pinned_samples()

    rows = []
    n_cells = len(ALL_NUGGET_VALUES) * len(TRUTH_SEED_REPLICATES)
    for nug in ALL_NUGGET_VALUES:
        for truth_seed in TRUTH_SEED_REPLICATES:
            r = fit_one(float(nug), int(truth_seed))
            rows.append(r)
            print(
                f"  [{len(rows):>2}/{n_cells}] nug={r['axis_level']} "
                f"seed={truth_seed:<5} length_scale={r['gp_length_scale_m']:9.3f} m  "
                f"noise={r['gp_noise_variance_real_units']:7.4f} "
                f"(truth nugget {r['truth_nugget_real_units']:.2f})  "
                f"LML={r['gp_log_marginal_likelihood']:.3f}  "
                f"[{time.time() - t0:.0f}s]"
            )

    geometry = verify_sample_geometry(rows)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df = df.sort_values(["nug_normalized", "truth_seed"]).reset_index(drop=True)

    check_df = verify_against_pinned_hyperparameters(df)

    # The headline question, recorded as a column so the figure script and any
    # later reader get the same boolean rather than recomputing the comparison.
    df["noise_above_truth_nugget"] = (
        df["gp_noise_variance_real_units"] > df["truth_nugget_real_units"]
    )

    df.to_csv(OUT_CSV, index=False)
    check_path = PROCESSED_DIR / "gp_hyperparameters_by_realization_pinned_check.csv"
    check_df.to_csv(check_path, index=False)

    # --- Printed summary (numbers only) ----------------------------------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)

    n_above = int(df["noise_above_truth_nugget"].sum())
    print(
        f"\nfitted noise variance > truth nugget: {n_above} of {len(df)} fits "
        f"({n_above / len(df):.1%})"
    )
    print(
        df.groupby("axis_level")
        .agg(
            truth_nugget=("truth_nugget_real_units", "first"),
            n_above=("noise_above_truth_nugget", "sum"),
            n=("noise_above_truth_nugget", "size"),
            noise_mean=("gp_noise_variance_real_units", "mean"),
            noise_std=("gp_noise_variance_real_units", "std"),
            noise_min=("gp_noise_variance_real_units", "min"),
            noise_max=("gp_noise_variance_real_units", "max"),
        )
        .to_string()
    )
    print("\nfitted practical range (m) by level:")
    print(
        df.groupby("axis_level")
        .agg(
            range_mean=("gp_practical_range_m", "mean"),
            range_std=("gp_practical_range_m", "std"),
            range_min=("gp_practical_range_m", "min"),
            range_max=("gp_practical_range_m", "max"),
            range_median=("gp_practical_range_m", "median"),
        )
        .to_string()
    )
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
            print(f"      nug={h['axis_level']} seed={h['truth_seed']}")

    print(f"\nsample geometry: {geometry}")
    print(f"\ngp_hyperparameters_by_realization.csv: {OUT_CSV}")
    print(f"pinned check table: {check_path}")
    print(f"total {time.time() - t0:.0f}s")
    return df


if __name__ == "__main__":
    main()
