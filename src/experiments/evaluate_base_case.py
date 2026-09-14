"""Evaluate the base-case 4-method comparison (kriging / SGS / RBF+bootstrap
/ GP-MLE) on MSE (accuracy) and UMG (uncertainty calibration), reported
separately per docs/experiment_context.md section 5.

Loads each method's *latest* run from results/raw/<method>/<timestamp>/ (no
re-simulation), applies the shared conditioning-sample-cell exclusion mask,
and writes a tidy long-format metrics table plus accuracy-plot curve data to
results/processed/base_case/ (git-tracked, deterministic overwrite -- not a
raw/timestamped artifact, see CLAUDE.md "결과 저장 컨벤션").

Run with: .venv/Scripts/python.exe -m src.experiments.evaluate_base_case
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation import (
    accuracy_plot_fraction_in,
    calc_umg,
    conditioning_cell_mask,
    kriging_fraction_in,
    mse,
)
from src.experiments.base_case import NX, NY, XMN, YMN, XSIZ, YSIZ
from src.experiments.base_case_conditioning import (
    SAMPLE_SEED,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.kriging import (
    BACKTR_ZMAX,
    BACKTR_ZMIN,
    LTAIL,
    LTPAR,
    UTAIL,
    UTPAR,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = _REPO_ROOT / "results" / "raw"
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "base_case"

METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]

CASE = "base_case"
AXIS = "none"
AXIS_LEVEL = "base"

# Numerical floor for variance maps before sqrt -- variance maps (esp.
# kriging's kb2d output) can carry tiny negative values (~1e-6 or smaller)
# from floating-point roundoff; anything more negative than this is a real
# bug, not roundoff, and should raise rather than be silently clipped.
VARIANCE_CLIP_TOLERANCE = 1e-6


def latest_run_dir(experiment: str) -> Path:
    """Return the most recent (by timestamp-sorted directory name)
    results/raw/<experiment>/<timestamp>/ directory."""
    exp_dir = RAW_ROOT / experiment
    run_dirs = sorted(p for p in exp_dir.iterdir() if p.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No runs found for experiment '{experiment}' under {exp_dir}")
    return run_dirs[-1]


def safe_sqrt_variance(var_map: np.ndarray, name: str) -> np.ndarray:
    """sqrt of a variance map, defensively clipping small-negative
    floating-point noise to 0 while raising on anything larger (a real bug).
    """
    min_val = float(np.min(var_map))
    if min_val < -VARIANCE_CLIP_TOLERANCE:
        raise ValueError(
            f"{name}: variance map has a value ({min_val}) more negative "
            f"than the roundoff tolerance ({-VARIANCE_CLIP_TOLERANCE}) -- "
            "this looks like a real bug, not floating-point noise."
        )
    n_clipped = int(np.sum(var_map < 0))
    if n_clipped > 0:
        print(f"  note: {name}: clipped {n_clipped} small-negative variance value(s) to 0 before sqrt.")
    return np.sqrt(np.clip(var_map, 0.0, None))


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = {m: latest_run_dir(m) for m in METHODS}
    print("Source runs (latest timestamp per method):")
    for m in METHODS:
        print(f"  {m}: {run_dirs[m]}")

    # --- Truth + shared conditioning mask, regenerated directly (not read
    # from any method's samples.csv) -- the safest single source of truth
    # per task instructions. ------------------------------------------------
    truth = get_base_case_truth()
    samples_df = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED)
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    n_excluded = int((~mask).sum())
    n_total = mask.size
    print(f"\nConditioning-sample cells excluded from evaluation: {n_excluded} of {n_total}")

    # --- Cross-check against each method's own recorded samples.csv, purely
    # as a drift-detection sanity check (does not affect the computation,
    # which always uses the regenerated samples_df above). -----------------
    for m in METHODS:
        recorded = pd.read_csv(run_dirs[m] / "samples.csv")
        if len(recorded) != len(samples_df) or not np.allclose(
            recorded[["X", "Y"]].values, samples_df[["X", "Y"]].values
        ):
            print(
                f"  WARNING: {m}'s recorded samples.csv does NOT match the "
                "regenerated conditioning samples -- identical-sample-"
                "locations assumption may be violated for this run."
            )

    truth_masked = truth[mask]

    # ------------------------------------------------------------------
    # Kriging: MSE from kmap_physical; UMG from kmap_ns + vmap_ns via exact
    # quantile back-transform.
    # ------------------------------------------------------------------
    kdir = run_dirs["kriging"]
    kmap_physical = np.load(kdir / "kriging_mean_map_physical.npy")
    kmap_ns = np.load(kdir / "kmap_ns.npy")
    vmap_ns = np.load(kdir / "kriging_var_map_ns.npy")
    transform_table = pd.read_csv(kdir / "nscore_transform_table.csv")
    vr, vrg = transform_table["vr"].values, transform_table["vrg"].values

    kriging_mse = mse(truth_masked, kmap_physical[mask])
    std_ns_masked = safe_sqrt_variance(vmap_ns[mask], "kriging vmap_ns")
    kriging_p, kriging_frac_in = kriging_fraction_in(
        truth_masked,
        kmap_ns[mask],
        std_ns_masked,
        vr,
        vrg,
        BACKTR_ZMIN,
        BACKTR_ZMAX,
        LTAIL,
        LTPAR,
        UTAIL,
        UTPAR,
    )
    kriging_umg = calc_umg(kriging_p, kriging_frac_in)

    # ------------------------------------------------------------------
    # SGS: ensemble mean/variance used for both MSE and UMG.
    # ------------------------------------------------------------------
    sdir = run_dirs["sgs"]
    sgs_mean_map = np.load(sdir / "sgs_mean_map.npy")
    sgs_var_map = np.load(sdir / "sgs_var_map.npy")

    sgs_mse = mse(truth_masked, sgs_mean_map[mask])
    sgs_std_masked = safe_sqrt_variance(sgs_var_map[mask], "sgs_var_map")
    sgs_p, sgs_frac_in = accuracy_plot_fraction_in(truth_masked, sgs_mean_map[mask], sgs_std_masked)
    sgs_umg = calc_umg(sgs_p, sgs_frac_in)

    # ------------------------------------------------------------------
    # RBF+bootstrap: MSE from the single-fit point_estimate_map; UMG from
    # (bootstrap_mean_map, bootstrap_var_map) as a per-cell Gaussian.
    # ------------------------------------------------------------------
    rdir = run_dirs["rbf_bootstrap"]
    point_estimate_map = np.load(rdir / "point_estimate_map.npy")
    bootstrap_mean_map = np.load(rdir / "bootstrap_mean_map.npy")
    bootstrap_var_map = np.load(rdir / "bootstrap_var_map.npy")

    rbf_mse = mse(truth_masked, point_estimate_map[mask])
    rbf_std_masked = safe_sqrt_variance(bootstrap_var_map[mask], "bootstrap_var_map")
    rbf_p, rbf_frac_in = accuracy_plot_fraction_in(truth_masked, bootstrap_mean_map[mask], rbf_std_masked)
    rbf_umg = calc_umg(rbf_p, rbf_frac_in)

    # ------------------------------------------------------------------
    # GP-MLE: posterior mean/variance used for both MSE and UMG.
    # ------------------------------------------------------------------
    gdir = run_dirs["gp_mle"]
    posterior_mean_map = np.load(gdir / "posterior_mean_map.npy")
    posterior_var_map = np.load(gdir / "posterior_var_map.npy")

    gp_mse = mse(truth_masked, posterior_mean_map[mask])
    gp_std_masked = safe_sqrt_variance(posterior_var_map[mask], "posterior_var_map")
    gp_p, gp_frac_in = accuracy_plot_fraction_in(truth_masked, posterior_mean_map[mask], gp_std_masked)
    gp_umg = calc_umg(gp_p, gp_frac_in)

    # ------------------------------------------------------------------
    # Assemble tidy long-format metrics table.
    # ------------------------------------------------------------------
    metrics = {
        "kriging": {"mse": kriging_mse, "umg": kriging_umg},
        "sgs": {"mse": sgs_mse, "umg": sgs_umg},
        "rbf_bootstrap": {"mse": rbf_mse, "umg": rbf_umg},
        "gp_mle": {"mse": gp_mse, "umg": gp_umg},
    }
    rows = []
    for method in METHODS:
        for metric_name, value in metrics[method].items():
            rows.append(
                {
                    "case": CASE,
                    "axis": AXIS,
                    "axis_level": AXIS_LEVEL,
                    "method": method,
                    "metric": metric_name,
                    "value": value,
                }
            )
    metrics_df = pd.DataFrame(rows)

    if metrics_df["value"].isna().any() or not np.all(np.isfinite(metrics_df["value"].values)):
        raise ValueError("metrics_df contains NaN/inf values -- see printed metrics above for details.")

    metrics_csv = PROCESSED_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # ------------------------------------------------------------------
    # Accuracy-plot curve data (for figure regeneration without rerunning).
    # ------------------------------------------------------------------
    curves = {
        "kriging": (kriging_p, kriging_frac_in),
        "sgs": (sgs_p, sgs_frac_in),
        "rbf_bootstrap": (rbf_p, rbf_frac_in),
        "gp_mle": (gp_p, gp_frac_in),
    }
    curve_rows = []
    for method, (p_intervals, fraction_in) in curves.items():
        for p, f in zip(p_intervals, fraction_in):
            curve_rows.append({"method": method, "p_interval": p, "fraction_in": f})
    curves_df = pd.DataFrame(curve_rows)
    curves_csv = PROCESSED_DIR / "accuracy_plot_curves.csv"
    curves_df.to_csv(curves_csv, index=False)

    # ------------------------------------------------------------------
    # QC figure: accuracy-plot curves for all 4 methods overlaid.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"kriging": "tab:blue", "sgs": "tab:green", "rbf_bootstrap": "tab:orange", "gp_mle": "tab:red"}
    markers = {"kriging": "o", "sgs": "s", "rbf_bootstrap": "^", "gp_mle": "D"}
    for method, (p_intervals, fraction_in) in curves.items():
        ax.plot(
            p_intervals,
            fraction_in,
            marker=markers[method],
            color=colors[method],
            label=f"{method} (UMG={metrics[method]['umg']:.3f})",
            alpha=0.85,
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", label="ideal (y=x)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal probability interval")
    ax.set_ylabel("Fraction of truth in interval")
    ax.set_title(f"Base case accuracy plot -- {CASE}, axis={AXIS}, level={AXIS_LEVEL}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig_path = PROCESSED_DIR / "accuracy_plot_comparison.png"
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Source-run provenance record.
    # ------------------------------------------------------------------
    source_runs = {
        m: {
            "run_dir": str(run_dirs[m].relative_to(_REPO_ROOT)).replace("\\", "/"),
            "timestamp": run_dirs[m].name,
        }
        for m in METHODS
    }
    source_runs_path = PROCESSED_DIR / "source_runs.json"
    with open(source_runs_path, "w", encoding="utf-8") as f:
        json.dump(source_runs, f, indent=2)

    # ------------------------------------------------------------------
    # Console report.
    # ------------------------------------------------------------------
    print("\nBase case metrics (MSE + UMG, tidy long-format table):")
    print(metrics_df.to_string(index=False))
    print(f"\nmetrics.csv: {metrics_csv}")
    print(f"accuracy_plot_curves.csv: {curves_csv}")
    print(f"accuracy_plot_comparison.png: {fig_path}")
    print(f"source_runs.json: {source_runs_path}")

    return metrics_df, curves_df


if __name__ == "__main__":
    main()
