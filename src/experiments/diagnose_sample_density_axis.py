"""Numeric diagnostics for the sample-density axis (3 levels, actual
n_samples = 121 / 50 / 25 = 4.84% / 2.00% / 1.00% of the 2500-cell grid).

This script RECORDS NUMBERS ONLY -- it deliberately contains no
interpretation of them. Nothing here changes any result: it reads the pinned
runs in results/processed/sample_density_axis/source_runs.json plus their
saved arrays/manifests, regenerates the (deterministic) truth field and
conditioning samples, and writes four small tidy tables next to them.

It is separate from qc_sample_density_axis.py (which answers "did each
method's machinery behave?") because these four tables answer a different
question -- "what are the properties of the conditioning data and of the
fitted GP, in units that can be compared against the ground truth?" -- and
because the slowest part of the QC script (re-running sgsim) is not needed
here.

Outputs (all in results/processed/sample_density_axis/)
------------------------------------------------------
1. conditioning_sample_bias.csv
   Per level: the mean of the conditioning samples' porosity values against
   the mean of the whole truth field, plus a NULL DISTRIBUTION of that same
   sample mean built by redrawing the same number of samples from the SAME
   truth field at N_NULL_SEEDS different sample seeds. Gives z and the
   percentile of the observed sample mean within the null.

2. mse_bias_variance_decomposition.csv
   Per level x method: MSE split into bias^2 + variance over the evaluated
   (non-conditioning) cells, where bias = mean(prediction - truth). Includes
   bias^2 as a percentage of MSE. The MSE recomputed here is cross-checked
   against the value recorded in metrics.csv.

3. gp_hyperparameter_scale_conversion.csv
   Per level: the fitted sklearn RBF-kernel ``length_scale`` converted to a
   practical range so it can be put next to the truth's SPHERICAL variogram
   range, plus the fitted noise variance next to the truth nugget in
   physical units. See the CORRELATION_CUTOFF comment below for the
   conversion and its basis.

4. kriging_backtransform_tail_sensitivity.csv
   Per level x back-transform bound: kriging's 95%-interval width and CRPS
   recomputed with BACKTR_ZMIN/ZMAX set to POR_MEAN -+ k*POR_STDEV for
   k in TAIL_BOUND_SIGMAS, with the three physical-unit Gaussian methods'
   (bound-independent) values from metrics.csv alongside. This is the
   provenance for the tail-assumption caption printed on
   results/figures/sample_density_axis/interval_width_p95_vs_sample_density.png.

Run with: .venv/Scripts/python.exe -m src.experiments.diagnose_sample_density_axis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import (
    INTERVAL_WIDTH_HEADLINE_P,
    conditioning_cell_mask,
    kriging_crps,
    kriging_interval_widths,
)
from src.experiments.base_case import (
    NUG,
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
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.kriging import LTAIL, UTAIL, backtr_value_vectorized
from src.experiments.sample_density_axis import (
    ALL_AXIS_LEVELS,
    AXIS_HMAJ1,
    METHODS,
    SAMPLE_COUNTS,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_density_axis"

# --- Null distribution of the conditioning-sample mean ----------------------
# Same truth field, same n_samples, only the SAMPLE SEED changes. The seed
# block is fixed and recorded (not drawn) so the null is reproducible; it is
# chosen to exclude the production SAMPLE_SEED=20, whose sample mean is the
# quantity being located within the null.
NULL_SEED_START = 1000
N_NULL_SEEDS = 300

# --- sklearn RBF length_scale -> "range"-like distance ----------------------
# sklearn's RBF kernel is k(h) = exp(-h^2 / (2 * length_scale^2)), which
# decays asymptotically and has NO finite range, unlike the truth model's
# SPHERICAL variogram, whose correlation reaches exactly 0 at its range.
# length_scale and a spherical range are therefore NOT the same quantity and
# must not be compared directly. The conversion used here is the usual
# "practical range": the lag h at which the correlation has decayed to
# CORRELATION_CUTOFF = 0.05, i.e.
#
#     exp(-h^2 / (2 l^2)) = 0.05  ->  h = l * sqrt(2 * ln 20) = 2.4477 * l
#
# which is what the common shorthand h = sqrt(6) * l = 2.4495 * l rounds to
# (2 * ln 20 = 5.9915 ~ 6). Both are tabulated below.
# Other conventions exist and give different numbers for the same fitted
# length_scale -- e.g. h = sqrt(3) * l, which is the lag at which this kernel's
# correlation is exp(-1.5) = 0.223, not 0.05 -- so the cutoff must always be
# stated alongside the converted value.
CORRELATION_CUTOFF = 0.05

# k values for BACKTR_ZMIN/ZMAX = POR_MEAN -+ k * POR_STDEV. The production
# value in src/experiments/kriging.py is 4.
TAIL_BOUND_SIGMAS = (2, 4, 6)
PRODUCTION_TAIL_BOUND_SIGMA = 4

# Point-estimate array each method's MSE is computed from -- identical to the
# choice made in src/experiments/evaluate_sample_density_axis.py.
POINT_ESTIMATE_FILE = {
    "kriging": "kriging_mean_map_physical.npy",
    "sgs": "sgs_mean_map.npy",
    "rbf_bootstrap": "point_estimate_map.npy",
    "gp_mle": "posterior_mean_map.npy",
}

MSE_CROSSCHECK_RTOL = 1e-9


def _manifest_params(rel_run_dir: str) -> dict:
    return json.loads(
        (_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8")
    )["params"]


def _truth():
    return get_base_case_truth(
        truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMAJ1
    )


def conditioning_sample_bias(truth: np.ndarray) -> pd.DataFrame:
    """Table 1: each level's conditioning-sample mean vs. the truth-field
    mean, located within a null distribution of the same statistic."""
    truth_field_mean = float(truth.mean())
    null_seeds = [
        s
        for s in range(NULL_SEED_START, NULL_SEED_START + N_NULL_SEEDS)
        if s != SAMPLE_SEED
    ]

    rows = []
    for level in ALL_AXIS_LEVELS:
        n_requested = SAMPLE_COUNTS[level]
        samples = get_conditioning_samples(
            truth, sample_seed=SAMPLE_SEED, n_samples=n_requested
        )
        observed = float(samples[VCOL].mean())

        null = np.array(
            [
                float(
                    get_conditioning_samples(
                        truth, sample_seed=seed, n_samples=n_requested
                    )[VCOL].mean()
                )
                for seed in null_seeds
            ]
        )
        null_mean, null_std = float(null.mean()), float(null.std(ddof=1))

        rows.append(
            {
                "axis_level": level,
                "n_samples_requested": n_requested,
                "n_samples_actual": len(samples),
                "sample_mean": observed,
                "truth_field_mean": truth_field_mean,
                "sample_mean_minus_truth_field_mean": observed - truth_field_mean,
                "null_seed_start": null_seeds[0],
                "null_seed_end": null_seeds[-1],
                "null_n_seeds": len(null_seeds),
                "null_mean": null_mean,
                "null_std_ddof1": null_std,
                "z_vs_null": (observed - null_mean) / null_std,
                "percentile_in_null_pct": 100.0 * float(np.mean(null < observed)),
                "null_min": float(null.min()),
                "null_max": float(null.max()),
            }
        )
    return pd.DataFrame(rows)


def mse_bias_variance(truth: np.ndarray, source_runs: dict) -> pd.DataFrame:
    """Table 2: MSE = bias^2 + variance over the evaluated cells, per level
    and method, with bias^2 expressed as a share of MSE."""
    recorded = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    recorded["axis_level"] = recorded["axis_level"].astype(str)

    rows = []
    for level in ALL_AXIS_LEVELS:
        samples = get_conditioning_samples(
            truth, sample_seed=SAMPLE_SEED, n_samples=SAMPLE_COUNTS[level]
        )
        mask = conditioning_cell_mask(samples, NX, NY, XMN, YMN, XSIZ, YSIZ)
        truth_masked = truth[mask]

        for method in METHODS:
            pred = np.load(
                _REPO_ROOT / source_runs[level][method] / POINT_ESTIMATE_FILE[method]
            )[mask]
            err = pred - truth_masked
            bias = float(err.mean())
            variance = float(err.var())  # ddof=0: exactly MSE - bias^2
            mse_here = float(np.mean(err ** 2))

            mse_recorded = float(
                recorded[
                    (recorded["axis_level"] == level)
                    & (recorded["method"] == method)
                    & (recorded["metric"] == "mse")
                ]["value"].iloc[0]
            )
            if not np.isclose(mse_here, mse_recorded, rtol=MSE_CROSSCHECK_RTOL, atol=0.0):
                raise ValueError(
                    f"{level}% {method}: MSE recomputed here ({mse_here}) disagrees with "
                    f"metrics.csv ({mse_recorded}) -- the decomposition would not describe "
                    "the reported MSE."
                )

            rows.append(
                {
                    "axis_level": level,
                    "n_samples_actual": len(samples),
                    "method": method,
                    "n_cells_evaluated": int(mask.sum()),
                    "mse": mse_here,
                    "mse_in_metrics_csv": mse_recorded,
                    "bias": bias,
                    "bias_squared": bias ** 2,
                    "variance_of_error": variance,
                    "bias_squared_share_of_mse_pct": 100.0 * bias ** 2 / mse_here,
                }
            )
    return pd.DataFrame(rows)


def gp_scale_conversion(source_runs: dict) -> pd.DataFrame:
    """Table 3: fitted GP hyperparameters converted into units comparable
    with the truth variogram (see the CORRELATION_CUTOFF comment above for
    the conversion and its basis)."""
    factor_exact = float(np.sqrt(-2.0 * np.log(CORRELATION_CUTOFF)))  # 2.4477
    truth_nugget_real = NUG * POR_STDEV ** 2  # NS-space nugget x physical sill

    rows = []
    for level in ALL_AXIS_LEVELS:
        hp = _manifest_params(source_runs[level]["gp_mle"])["fitted_hyperparameters"]
        ell = float(hp["length_scale_m"])
        noise_real = float(hp["noise_variance_real_units"])
        rows.append(
            {
                "axis_level": level,
                "gp_length_scale_m": ell,
                "correlation_cutoff": CORRELATION_CUTOFF,
                "practical_range_factor_exact": factor_exact,
                "practical_range_m": factor_exact * ell,
                "practical_range_sqrt6_shorthand_m": float(np.sqrt(6.0)) * ell,
                "truth_spherical_range_m": AXIS_HMAJ1,
                "practical_range_over_truth_range": factor_exact * ell / AXIS_HMAJ1,
                "gp_noise_variance_real_units": noise_real,
                "gp_signal_variance_real_units": float(
                    hp["signal_variance_real_units"]
                ),
                "truth_nugget_real_units": truth_nugget_real,
                "gp_noise_over_truth_nugget": noise_real / truth_nugget_real,
            }
        )
    return pd.DataFrame(rows)


def kriging_tail_sensitivity(truth: np.ndarray, source_runs: dict) -> pd.DataFrame:
    """Table 4: kriging's 95%-interval width and CRPS recomputed under
    several back-transform tail bounds, with the three bound-independent
    methods' recorded values alongside."""
    # interval_width_p95 / crps moved OUT of metrics.csv on 2026-09-15 when the
    # sharpness family was parked (EMIT_SHARPNESS_METRICS=False in
    # evaluate_sample_density_axis.py); their values were preserved verbatim in
    # metrics_parked_sharpness.csv. Read both and concatenate so this diagnostic
    # keeps working whichever side of that flag the axis is currently on --
    # otherwise this function raises IndexError on a parked axis.
    recorded = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    parked_path = PROCESSED_DIR / "metrics_parked_sharpness.csv"
    if parked_path.exists():
        parked = pd.read_csv(parked_path, comment="#")
        recorded = pd.concat([recorded, parked], ignore_index=True)
    recorded["axis_level"] = recorded["axis_level"].astype(str)

    def _rec(level, method, metric):
        return float(
            recorded[
                (recorded["axis_level"] == level)
                & (recorded["method"] == method)
                & (recorded["metric"] == metric)
            ]["value"].iloc[0]
        )

    rows = []
    for level in ALL_AXIS_LEVELS:
        samples = get_conditioning_samples(
            truth, sample_seed=SAMPLE_SEED, n_samples=SAMPLE_COUNTS[level]
        )
        mask = conditioning_cell_mask(samples, NX, NY, XMN, YMN, XSIZ, YSIZ)
        truth_masked = truth[mask]

        kdir = _REPO_ROOT / source_runs[level]["kriging"]
        kmap_ns = np.load(kdir / "kmap_ns.npy")[mask]
        std_ns = np.sqrt(
            np.clip(np.load(kdir / "kriging_var_map_ns.npy")[mask], 0.0, None)
        )
        table = pd.read_csv(kdir / "nscore_transform_table.csv")
        vr, vrg = table["vr"].values, table["vrg"].values

        for k in TAIL_BOUND_SIGMAS:
            zmin = POR_MEAN - k * POR_STDEV
            zmax = POR_MEAN + k * POR_STDEV
            # LTPAR/UTPAR follow the bound, exactly as kriging.py sets them.
            args = (
                vr, vrg, zmin, zmax, LTAIL, zmin, UTAIL, zmax, backtr_value_vectorized,
            )
            width_p95 = float(
                kriging_interval_widths(
                    kmap_ns, std_ns, *args,
                    p_levels=np.array([INTERVAL_WIDTH_HEADLINE_P]),
                )[0]
            )
            crps = float(kriging_crps(truth_masked, kmap_ns, std_ns, *args))

            row = {
                "axis_level": level,
                "n_samples_actual": len(samples),
                "backtr_bound_sigmas": k,
                "is_production_bound": k == PRODUCTION_TAIL_BOUND_SIGMA,
                "backtr_zmin": zmin,
                "backtr_zmax": zmax,
                "kriging_interval_width_p95": width_p95,
                "kriging_crps": crps,
            }
            for other in ("sgs", "rbf_bootstrap", "gp_mle"):
                # Physical-unit Gaussian predictives: independent of the
                # back-transform bound, repeated here only for ranking.
                row[f"{other}_interval_width_p95"] = _rec(
                    level, other, "interval_width_p95"
                )
                row[f"{other}_crps"] = _rec(level, other, "crps")
            rows.append(row)

    out = pd.DataFrame(rows)

    # The production rows must reproduce metrics.csv exactly.
    for level in ALL_AXIS_LEVELS:
        prod = out[(out["axis_level"] == level) & out["is_production_bound"]].iloc[0]
        for col, metric in (
            ("kriging_interval_width_p95", "interval_width_p95"),
            ("kriging_crps", "crps"),
        ):
            if not np.isclose(prod[col], _rec(level, "kriging", metric), rtol=1e-9, atol=0.0):
                raise ValueError(
                    f"{level}%: kriging {metric} recomputed at the production bound "
                    f"({prod[col]}) != metrics.csv ({_rec(level, 'kriging', metric)})."
                )
    return out


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    source_runs = json.loads(
        (PROCESSED_DIR / "source_runs.json").read_text(encoding="utf-8")
    )
    truth = _truth()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)

    bias_df = conditioning_sample_bias(truth)
    bias_path = PROCESSED_DIR / "conditioning_sample_bias.csv"
    bias_df.to_csv(bias_path, index=False)
    print("\n--- 1. Conditioning-sample mean vs. null distribution ---")
    print(
        bias_df[
            [
                "axis_level", "n_samples_actual", "sample_mean", "truth_field_mean",
                "null_mean", "null_std_ddof1", "z_vs_null", "percentile_in_null_pct",
            ]
        ].to_string(index=False)
    )

    decomp_df = mse_bias_variance(truth, source_runs)
    decomp_path = PROCESSED_DIR / "mse_bias_variance_decomposition.csv"
    decomp_df.to_csv(decomp_path, index=False)
    print("\n--- 2. MSE = bias^2 + variance (evaluated cells) ---")
    print(
        decomp_df[
            [
                "axis_level", "method", "mse", "bias", "bias_squared",
                "variance_of_error", "bias_squared_share_of_mse_pct",
            ]
        ].to_string(index=False)
    )

    gp_df = gp_scale_conversion(source_runs)
    gp_path = PROCESSED_DIR / "gp_hyperparameter_scale_conversion.csv"
    gp_df.to_csv(gp_path, index=False)
    print(
        f"\n--- 3. GP-MLE length_scale -> practical range (correlation cutoff "
        f"{CORRELATION_CUTOFF}) and noise vs. truth nugget ---"
    )
    print(
        gp_df[
            [
                "axis_level", "gp_length_scale_m", "practical_range_m",
                "practical_range_sqrt6_shorthand_m", "truth_spherical_range_m",
                "gp_noise_variance_real_units", "truth_nugget_real_units",
                "gp_noise_over_truth_nugget",
            ]
        ].to_string(index=False)
    )

    tail_df = kriging_tail_sensitivity(truth, source_runs)
    tail_path = PROCESSED_DIR / "kriging_backtransform_tail_sensitivity.csv"
    tail_df.to_csv(tail_path, index=False)
    print("\n--- 4. Kriging back-transform tail-bound sensitivity ---")
    print(
        tail_df[
            [
                "axis_level", "backtr_bound_sigmas", "is_production_bound",
                "backtr_zmin", "backtr_zmax", "kriging_interval_width_p95",
                "sgs_interval_width_p95", "gp_mle_interval_width_p95",
                "rbf_bootstrap_interval_width_p95",
                "kriging_crps", "gp_mle_crps",
            ]
        ].to_string(index=False)
    )

    print(f"\nconditioning_sample_bias.csv: {bias_path}")
    print(f"mse_bias_variance_decomposition.csv: {decomp_path}")
    print(f"gp_hyperparameter_scale_conversion.csv: {gp_path}")
    print(f"kriging_backtransform_tail_sensitivity.csv: {tail_path}")

    return bias_df, decomp_df, gp_df, tail_df


if __name__ == "__main__":
    main()
