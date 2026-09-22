"""Multi-initial-value refit check for the GP-MLE hyperparameters at a single
nugget-axis level (default: nug=0.5).

WHY
---
On the nugget axis the GP-MLE fitted length_scale is ~100-130 m at
nug = 0.0 .. 0.4 and then jumps to 557.5 m (practical range 1364.67 m) at
nug=0.5. The question this script answers -- and ONLY this question -- is
whether that jump is the GLOBAL marginal-likelihood optimum or an artifact of
where the optimizer happened to start. It reports numbers; it does not
interpret them.

This is the same procedure the reviewer applied to the sample-density axis's
n=25 length_scale collapse (docs/progress.md 2026-09-15: "480개 초기값
(restart 8~200 + 3차원 격자)으로 재적합해 세 레벨 모두 동일 해로 수렴"),
written down as a rerunnable script instead of an ad hoc check:

  A. RESTART SWEEP -- refit with sklearn's own ``n_restarts_optimizer`` set to
     8 (the production value), 16, 32, 64, 128 and 200. Each restart is an
     independent log-uniform draw inside the kernel bounds, so this alone is
     8 + 16 + ... + 200 = 448 optimizer starts.
  B. DETERMINISTIC 3-D INITIAL-VALUE GRID -- refit once per grid point with
     ``n_restarts_optimizer=0``, so the optimizer starts EXACTLY at the
     specified (constant_value, length_scale, noise_level) triple and nothing
     is left to the RNG. This covers the hyperparameter box systematically
     rather than randomly, which a pure restart sweep cannot guarantee.

Everything except the initial values / restart count is held at
src/experiments/gp_mle.py's production configuration (same kernel structure,
same bounds, ``normalize_y=True``, same conditioning data) -- the point is to
vary the OPTIMIZER's starting point and nothing else.

WHAT IS REPORTED
----------------
Per fit: the initial values used, the fitted length_scale / signal variance /
noise variance (normalized and in real porosity %^2 units), the
log-marginal-likelihood, and whether the fit landed on the same solution as
the pinned production run (all three hyperparameters within
``SOLUTION_RTOL``).

Aggregate: how many of the fits reached the pinned solution, the best LML
found anywhere, and -- the decisive number -- whether ANY starting point
found a HIGHER log marginal likelihood than the pinned run (which would mean
the production fit is a local, not global, optimum).

Outputs
-------
  results/processed/nugget_axis/gp_restart_check_nug<token>.csv          (one row per fit)
  results/processed/nugget_axis/gp_restart_check_nug<token>_summary.csv  (one row)

Run with:
.venv/Scripts/python.exe -m src.experiments.gp_restart_check_nugget_axis
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from src.experiments.base_case_conditioning import (
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_conditioning_data,
)
from src.experiments.gp_mle import (
    CONSTANT_VALUE_BOUNDS,
    GP_RANDOM_STATE,
    LENGTH_SCALE_BOUNDS,
    NOISE_LEVEL_BOUNDS,
    N_RESTARTS_OPTIMIZER,
)
from src.experiments.nugget_axis import (
    AXIS_HMAJ1,
    AXIS_HMIN1,
    level_file_token,
    level_key,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"

# The level under investigation (the length_scale jump).
CHECK_NUG = 0.5

# A. restart sweep. The first entry is gp_mle.py's production value, so the
# production configuration is itself one row of the table.
RESTART_COUNTS = (N_RESTARTS_OPTIMIZER, 16, 32, 64, 128, 200)

# B. deterministic 3-D initial-value grid, log-spaced strictly INSIDE each
# kernel hyperparameter's optimizer bounds (a start exactly on a bound is a
# degenerate starting point, not an informative one). 10 x 6 x 6 = 360 fits.
LENGTH_SCALE_INIT_GRID = np.logspace(np.log10(1.0), np.log10(5000.0), 10)
CONSTANT_VALUE_INIT_GRID = np.logspace(np.log10(1e-2), np.log10(1e2), 6)
NOISE_LEVEL_INIT_GRID = np.logspace(np.log10(1e-4), np.log10(1e1), 6)

# Two fits are called "the same solution" if all three hyperparameters agree
# to this relative tolerance. Loose enough to absorb L-BFGS-B stopping noise
# between different starting points, far tighter than the 4x gap between the
# 557 m solution and the ~126 m solutions at neighbouring levels.
SOLUTION_RTOL = 1e-3

# An LML is called "better than the pinned run" only if it exceeds it by more
# than this, so optimizer noise cannot be reported as a better optimum.
LML_TOL = 1e-6


def build_kernel(constant_value: float, length_scale: float, noise_level: float):
    """gp_mle.build_kernel's kernel STRUCTURE and BOUNDS, with the initial
    values supplied by the caller (that is the only thing this script varies)."""
    return (
        ConstantKernel(constant_value, CONSTANT_VALUE_BOUNDS)
        * RBF(length_scale=length_scale, length_scale_bounds=LENGTH_SCALE_BOUNDS)
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=NOISE_LEVEL_BOUNDS)
    )


def fit_once(X, y, constant_value, length_scale, noise_level, n_restarts):
    gpr = GaussianProcessRegressor(
        kernel=build_kernel(constant_value, length_scale, noise_level),
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=GP_RANDOM_STATE,
    )
    gpr.fit(X, y)
    k = gpr.kernel_
    y_train_std = float(gpr._y_train_std)
    signal_norm = float(k.k1.k1.constant_value)
    noise_norm = float(k.k2.noise_level)
    return {
        "init_constant_value": float(constant_value),
        "init_length_scale_m": float(length_scale),
        "init_noise_level": float(noise_level),
        "n_restarts_optimizer": int(n_restarts),
        "fitted_length_scale_m": float(k.k1.k2.length_scale),
        "fitted_signal_variance_normalized": signal_norm,
        "fitted_noise_variance_normalized": noise_norm,
        "fitted_signal_variance_real_units": signal_norm * y_train_std ** 2,
        "fitted_noise_variance_real_units": noise_norm * y_train_std ** 2,
        "log_marginal_likelihood": float(gpr.log_marginal_likelihood(k.theta)),
    }


def pinned_hyperparameters(nug: float) -> dict:
    """The production run's fitted hyperparameters for this level, read from
    the run currently pinned in source_runs.json (never hardcoded)."""
    source_runs = json.loads(
        (PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8")
    )
    rel = source_runs[level_key(nug)]["gp_mle"]
    params = json.loads(
        (_REPO_ROOT / rel / "manifest.json").read_text(encoding="utf-8")
    )["params"]
    hp = dict(params["fitted_hyperparameters"])
    hp["log_marginal_likelihood"] = float(params["log_marginal_likelihood"])
    hp["run_dir"] = rel
    return hp


def main(nug: float = CHECK_NUG):
    token = level_file_token(nug)
    pinned = pinned_hyperparameters(nug)
    print(
        f"Pinned gp_mle run for nug={level_key(nug)}: {pinned['run_dir']}\n"
        f"  length_scale = {pinned['length_scale_m']!r} m\n"
        f"  signal_variance_normalized = {pinned['signal_variance_normalized']!r}\n"
        f"  noise_variance_normalized  = {pinned['noise_variance_normalized']!r}\n"
        f"  log_marginal_likelihood    = {pinned['log_marginal_likelihood']!r}"
    )

    _, samples_df = get_base_case_conditioning_data(
        sample_seed=SAMPLE_SEED,
        truth_seed=TRUTH_SEED,
        hmaj1=AXIS_HMAJ1,
        hmin1=AXIS_HMIN1,
        n_samples=N_SAMPLES,
        nug=float(nug),
    )
    X = samples_df[["X", "Y"]].values
    y = samples_df[VCOL].values
    print(f"conditioning data: {len(samples_df)} samples at nug={level_key(nug)}")

    rows = []
    t0 = time.time()

    # --- A. restart sweep -------------------------------------------------
    from src.experiments.gp_mle import (
        CONSTANT_VALUE_INIT,
        LENGTH_SCALE_INIT,
        NOISE_LEVEL_INIT,
    )

    for n_restarts in RESTART_COUNTS:
        r = fit_once(
            X, y, CONSTANT_VALUE_INIT, LENGTH_SCALE_INIT, NOISE_LEVEL_INIT, n_restarts
        )
        r["block"] = "restart_sweep"
        rows.append(r)
        print(
            f"  restart sweep n_restarts={n_restarts:>3}: "
            f"length_scale={r['fitted_length_scale_m']:.4f} m, "
            f"LML={r['log_marginal_likelihood']:.6f}"
        )

    # --- B. deterministic 3-D initial-value grid --------------------------
    n_grid = (
        len(LENGTH_SCALE_INIT_GRID)
        * len(CONSTANT_VALUE_INIT_GRID)
        * len(NOISE_LEVEL_INIT_GRID)
    )
    print(f"\n  running {n_grid} deterministic grid fits (n_restarts_optimizer=0)...")
    done = 0
    for cv in CONSTANT_VALUE_INIT_GRID:
        for ls in LENGTH_SCALE_INIT_GRID:
            for nl in NOISE_LEVEL_INIT_GRID:
                r = fit_once(X, y, cv, ls, nl, 0)
                r["block"] = "initial_value_grid"
                rows.append(r)
                done += 1
                if done % 60 == 0:
                    print(f"    {done}/{n_grid} grid fits done ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df["axis_level"] = level_key(nug)
    df["nug_normalized"] = float(nug)

    # --- Agreement with the pinned production solution --------------------
    df["matches_pinned_solution"] = (
        np.isclose(df["fitted_length_scale_m"], pinned["length_scale_m"], rtol=SOLUTION_RTOL)
        & np.isclose(
            df["fitted_signal_variance_normalized"],
            pinned["signal_variance_normalized"],
            rtol=SOLUTION_RTOL,
        )
        & np.isclose(
            df["fitted_noise_variance_normalized"],
            pinned["noise_variance_normalized"],
            rtol=SOLUTION_RTOL,
        )
    )
    df["lml_minus_pinned"] = (
        df["log_marginal_likelihood"] - pinned["log_marginal_likelihood"]
    )
    df["better_than_pinned"] = df["lml_minus_pinned"] > LML_TOL

    total_optimizer_starts = int(sum(RESTART_COUNTS)) + n_grid
    df = df[
        [
            "axis_level", "nug_normalized", "block",
            "init_constant_value", "init_length_scale_m", "init_noise_level",
            "n_restarts_optimizer",
            "fitted_length_scale_m",
            "fitted_signal_variance_normalized", "fitted_noise_variance_normalized",
            "fitted_signal_variance_real_units", "fitted_noise_variance_real_units",
            "log_marginal_likelihood", "lml_minus_pinned",
            "matches_pinned_solution", "better_than_pinned",
        ]
    ]

    out_csv = PROCESSED_DIR / f"gp_restart_check_nug{token}.csv"
    df.to_csv(out_csv, index=False)

    best = df.loc[df["log_marginal_likelihood"].idxmax()]
    summary = pd.DataFrame(
        [
            {
                "axis_level": level_key(nug),
                "nug_normalized": float(nug),
                "pinned_run_dir": pinned["run_dir"],
                "pinned_length_scale_m": pinned["length_scale_m"],
                "pinned_noise_variance_normalized": pinned["noise_variance_normalized"],
                "pinned_signal_variance_normalized": pinned["signal_variance_normalized"],
                "pinned_log_marginal_likelihood": pinned["log_marginal_likelihood"],
                "n_fits": int(len(df)),
                "n_optimizer_starts_total": total_optimizer_starts,
                "n_matching_pinned_solution": int(df["matches_pinned_solution"].sum()),
                "frac_matching_pinned_solution": float(df["matches_pinned_solution"].mean()),
                "n_better_than_pinned": int(df["better_than_pinned"].sum()),
                "best_log_marginal_likelihood": float(best["log_marginal_likelihood"]),
                "best_length_scale_m": float(best["fitted_length_scale_m"]),
                "best_lml_minus_pinned": float(best["lml_minus_pinned"]),
                "fitted_length_scale_min_m": float(df["fitted_length_scale_m"].min()),
                "fitted_length_scale_max_m": float(df["fitted_length_scale_m"].max()),
                "fitted_length_scale_median_m": float(df["fitted_length_scale_m"].median()),
                "solution_rtol": SOLUTION_RTOL,
                "lml_tol": LML_TOL,
                "seconds": float(time.time() - t0),
            }
        ]
    )
    summary_csv = PROCESSED_DIR / f"gp_restart_check_nug{token}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)
    print(f"\n{len(df)} fits ({total_optimizer_starts} optimizer starts) in "
          f"{time.time() - t0:.0f}s")
    print("\nDistinct fitted length_scale solutions (rounded to 1e-4 m):")
    print(
        df.assign(ls=df["fitted_length_scale_m"].round(4))
        .groupby("ls")
        .agg(
            n=("ls", "size"),
            lml_min=("log_marginal_likelihood", "min"),
            lml_max=("log_marginal_likelihood", "max"),
            noise_norm=("fitted_noise_variance_normalized", "median"),
        )
        .sort_values("n", ascending=False)
        .to_string()
    )
    print("\nSummary:")
    print(summary.T.to_string())
    print(f"\ngp_restart_check_nug{token}.csv: {out_csv}")
    print(f"gp_restart_check_nug{token}_summary.csv: {summary_csv}")
    return df, summary


if __name__ == "__main__":
    main()
