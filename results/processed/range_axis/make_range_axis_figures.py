"""Build the range-axis figures (docs/experiment_context.md deliverable 2),
over the 8 axis levels range = 100, 200, ..., 800 m.

(a) MAIN FIGURE -- a 4-panel metric-vs-range plot (MSE / UMG / mean interval
    width / CRPS), one line per method. This is the report's headline
    figure: it puts accuracy, coverage calibration, sharpness and the proper
    scoring rule side by side, which is the only way to tell "well
    calibrated" apart from "calibrated because the intervals were widened".
(b) Calibration (accuracy-plot) curves for all 8 levels in one 2x4 grid.
(c) Per-range "truth & predictions" QC figures for all 8 levels, reusing the
    panel-layout/shared-colorbar logic of
    results/processed/base_case/make_truth_predictions_figures.py.

COLOR-SCALE CONVENTION for the QC figures (c)  -- flagged explicitly because
it constrains what the figures can be used for: the color scale is shared
WITHIN an axis level, NOT across axis levels. Concretely, within one range:
all porosity panels (truth / realization / mean) share that range's truth
min-max, and all variance panels share one scale anchored on the
SGS/RBF/GP-MLE variance maxima (kriging's physical-unit Monte Carlo variance
is annotated if it exceeds that shared max). Rationale: each range produces
its own ground-truth realization, so the fields' value ranges genuinely
differ between levels and a single global scale would wash out the
within-level method comparison these panels exist for. CONSEQUENCE: panels
from different ranges are NOT directly comparable by color -- read the
numbers in the metric figure (a) for cross-level comparison. This choice is
stated in the figure captions and was NOT decided silently.

Run dirs are read from results/processed/range_axis/source_runs.json.

Run with:
.venv/Scripts/python.exe -m results.processed.range_axis.make_range_axis_figures
(or: .venv/Scripts/python.exe results/processed/range_axis/make_range_axis_figures.py)
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.base_case import XMN, XMAX, XMIN, YMN, YMAX, YMIN  # noqa: E402
from src.experiments.base_case_conditioning import (  # noqa: E402
    SAMPLE_SEED,
    TRUTH_SEED,
    get_base_case_truth,
    get_conditioning_samples,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "range_axis"

METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]
RANGE_VALUES = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]

# QC figures are large (5 files x 8 levels); 300 dpi is plenty for on-screen
# QC review and keeps the directory to a reasonable size (was 600).
QC_FIG_DPI = 300
MAIN_FIG_DPI = 600

EXTENT = [XMIN, XMAX, YMIN, YMAX]

METHOD_COLORS = {"kriging": "tab:blue", "sgs": "tab:green", "rbf_bootstrap": "tab:orange", "gp_mle": "tab:red"}
METHOD_MARKERS = {"kriging": "o", "sgs": "s", "rbf_bootstrap": "^", "gp_mle": "D"}

# (metric key in metrics.csv, panel title, y-axis label)
MAIN_PANELS = [
    ("mse", "Accuracy: MSE vs. range", "MSE (Porosity %$^2$, lower better)"),
    ("umg", "Calibration: UMG vs. range", "UMG (1.0 = perfectly calibrated)"),
    (
        "interval_width_mean_nominal",
        "Sharpness: mean interval width vs. range",
        "Mean interval width (Porosity %)",
    ),
    ("crps", "Proper score: CRPS vs. range", "CRPS (Porosity %, lower better)"),
]


# ---------------------------------------------------------------------------
# (a) Main 4-panel metric-vs-range figure
# ---------------------------------------------------------------------------
def make_metric_vs_range_figure():
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    metrics_df["axis_level_numeric"] = metrics_df["axis_level"].astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (metric_name, title, ylabel) in zip(axes.ravel(), MAIN_PANELS):
        sub = metrics_df[metrics_df["metric"] == metric_name]
        for method in METHODS:
            m_sub = sub[sub["method"] == method].sort_values("axis_level_numeric")
            ax.plot(
                m_sub["axis_level_numeric"], m_sub["value"],
                marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                label=method, linewidth=2,
            )
        ax.set_xticks(RANGE_VALUES)
        ax.set_xlabel("Ground-truth variogram range (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if metric_name == "umg":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ideal (UMG=1.0)")
        ax.legend(fontsize=9)

    plt.suptitle(
        "Range axis: accuracy (MSE) vs. uncertainty quality (UMG / interval width / CRPS)\n"
        "one ground-truth realization per level (TRUTH_SEED=101), identical sample "
        "locations across methods",
        fontsize=12,
    )
    plt.subplots_adjust(left=0.08, bottom=0.07, right=0.98, top=0.90, wspace=0.25, hspace=0.30)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "metrics_vs_range.png"
    plt.savefig(out, dpi=MAIN_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"metrics_vs_range.png: {out} ({out.stat().st_size} bytes)")

    # Companion: the 95% interval width alone, which is the most directly
    # interpretable sharpness number (mean width of the central 95% interval).
    fig, ax = plt.subplots(figsize=(8, 6))
    sub = metrics_df[metrics_df["metric"] == "interval_width_p95"]
    for method in METHODS:
        m_sub = sub[sub["method"] == method].sort_values("axis_level_numeric")
        ax.plot(
            m_sub["axis_level_numeric"], m_sub["value"],
            marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=method, linewidth=2,
        )
    ax.set_xticks(RANGE_VALUES)
    ax.set_xlabel("Ground-truth variogram range (m)")
    ax.set_ylabel("Mean width of central 95% prediction interval (Porosity %)")
    ax.set_title("Range axis: 95% prediction-interval width vs. range")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.subplots_adjust(left=0.13, bottom=0.1, right=0.97, top=0.93)
    out95 = FIGURES_DIR / "interval_width_p95_vs_range.png"
    plt.savefig(out95, dpi=MAIN_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"interval_width_p95_vs_range.png: {out95} ({out95.stat().st_size} bytes)")

    return [out, out95]


# ---------------------------------------------------------------------------
# (b) Calibration curves, 2x4 grid over the 8 axis levels
# ---------------------------------------------------------------------------
def make_calibration_grid_figure():
    curves_df = pd.read_csv(PROCESSED_DIR / "accuracy_plot_curves.csv")
    curves_df["axis_level"] = curves_df["axis_level"].astype(str)
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    metrics_df["axis_level"] = metrics_df["axis_level"].astype(str)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for ax, hmaj1 in zip(axes.ravel(), RANGE_VALUES):
        axis_level = str(int(hmaj1))
        sub = curves_df[curves_df["axis_level"] == axis_level]
        for method in METHODS:
            m_sub = sub[sub["method"] == method]
            umg_val = metrics_df[
                (metrics_df["axis_level"] == axis_level)
                & (metrics_df["method"] == method)
                & (metrics_df["metric"] == "umg")
            ]["value"].iloc[0]
            ax.plot(
                m_sub["p_interval"], m_sub["fraction_in"],
                marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], markersize=4,
                label=f"{method} (UMG={umg_val:.3f})", alpha=0.85,
            )
        ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", label="ideal (y=x)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Nominal probability interval")
        ax.set_ylabel("Fraction of truth in interval")
        ax.set_title(f"range = {hmaj1:g} m")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle("Range axis calibration (accuracy) plots, 8 axis levels", fontsize=14)
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.92, wspace=0.28, hspace=0.25)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "calibration_curves_grid.png"
    plt.savefig(out, dpi=MAIN_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"calibration_curves_grid.png: {out} ({out.stat().st_size} bytes)")
    return out


# ---------------------------------------------------------------------------
# (c) Per-range truth & predictions QC figures (generalized from
# results/processed/base_case/make_truth_predictions_figures.py)
# ---------------------------------------------------------------------------
def _scatter_samples(ax, samples_df, color="red"):
    ax.scatter(samples_df["X"], samples_df["Y"], s=10, c=color, marker="+", label="conditioning samples")


def _panel(ax, arr, title, vmin, vmax, cmap, cbar_label, samples_df, scatter_color="red"):
    im = ax.imshow(arr, extent=EXTENT, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    _scatter_samples(ax, samples_df, color=scatter_color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, label=cbar_label)
    return im


# Caption appended to every QC figure, so the color-scale convention travels
# with the image rather than living only in this docstring.
SCALE_CAPTION = (
    "Color scales are shared WITHIN this range only (porosity panels: this range's "
    "truth min-max; variance panels: SGS/RBF/GP-MLE-anchored max). Panels from "
    "different ranges are NOT comparable by color."
)


def make_truth_predictions_figure_set(hmaj1: float, run_dirs: dict, out_dir: Path):
    """Generate the truth_field + 4 per-method QC figures for a single range
    value, following the panel layout / shared-colorbar convention of
    results/processed/base_case/make_truth_predictions_figures.py (see that
    script and this module's docstring for the full rationale).
    """
    hmin1 = hmaj1
    range_label = f"{hmaj1:g}m"

    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=hmaj1, hmin1=hmin1)
    samples_df = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED)

    kriging_dir = Path(_REPO_ROOT / run_dirs["kriging"])
    sgs_dir = Path(_REPO_ROOT / run_dirs["sgs"])
    rbf_dir = Path(_REPO_ROOT / run_dirs["rbf_bootstrap"])
    gp_dir = Path(_REPO_ROOT / run_dirs["gp_mle"])

    kriging_mean = np.load(kriging_dir / "kriging_mean_map_physical.npy")
    kriging_var = np.load(kriging_dir / "kriging_var_map_physical_mc.npy")

    sgs_realizations = np.load(sgs_dir / "sgs_realizations.npy")
    sgs_example = sgs_realizations[0]
    sgs_mean = np.load(sgs_dir / "sgs_mean_map.npy")
    sgs_var = np.load(sgs_dir / "sgs_var_map.npy")
    sgs_n_realizations = sgs_realizations.shape[0]

    bootstrap_replicates = np.load(rbf_dir / "bootstrap_replicate_maps.npy")
    bootstrap_example = bootstrap_replicates[0]
    bootstrap_mean = np.load(rbf_dir / "bootstrap_mean_map.npy")
    bootstrap_var = np.load(rbf_dir / "bootstrap_var_map.npy")
    bootstrap_n_replicates = bootstrap_replicates.shape[0]

    gp_sample = np.load(gp_dir / "posterior_sample_map.npy")
    gp_mean = np.load(gp_dir / "posterior_mean_map.npy")
    gp_var = np.load(gp_dir / "posterior_var_map.npy")

    # --- Color scales: WITHIN-level only (see module docstring) ------------
    mean_vmin = float(np.min(truth))
    mean_vmax = float(np.max(truth))

    kriging_var_actual_max = float(np.max(kriging_var))
    var_vmax = float(max(np.max(sgs_var), np.max(bootstrap_var), np.max(gp_var)))
    var_vmin = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def _finish(fig, filename, caption_ax=None):
        if caption_ax is not None:
            caption_ax.text(
                0.0, -0.22, SCALE_CAPTION, transform=caption_ax.transAxes,
                fontsize=7, color="dimgray", wrap=True,
            )
        out = out_dir / filename
        plt.savefig(out, dpi=QC_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
        print(f"{filename}: {out} ({out.stat().st_size} bytes)")

    # --- 1. truth_field ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    _panel(
        ax, truth, f"Range={range_label} truth ({truth.shape[0]}x{truth.shape[1]}, seed={TRUTH_SEED})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    plt.subplots_adjust(left=0.12, bottom=0.16, right=0.98, top=0.92)
    _finish(fig, f"truth_field_range{int(hmaj1)}.png", caption_ax=ax)

    # --- 2. kriging (2 panels: mean, variance) ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _panel(
        axes[0], kriging_mean, f"Simple kriging -- point estimate (range={range_label})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[1], kriging_var, "Simple kriging -- variance (physical units, Monte Carlo approx.)",
        var_vmin, var_vmax, "magma", "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    if kriging_var_actual_max > var_vmax:
        kvar_annotation = (
            f"Color scale clipped at {var_vmax:.2f} (shared with SGS/RBF/GP-MLE); "
            f"kriging's true max is {kriging_var_actual_max:.2f}."
        )
    else:
        kvar_annotation = (
            f"Not clipped: kriging's true max ({kriging_var_actual_max:.2f}) is below the "
            f"shared SGS/RBF/GP-MLE-anchored scale max ({var_vmax:.2f})."
        )
    axes[1].text(0.02, -0.14, kvar_annotation, transform=axes[1].transAxes, fontsize=8, color="dimgray")
    plt.subplots_adjust(left=0.06, bottom=0.2, right=0.98, top=0.9, wspace=0.35)
    _finish(fig, f"qc_truth_predictions_kriging_range{int(hmaj1)}.png", caption_ax=axes[0])

    # --- 3. sgs (3 panels) ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(
        axes[0], sgs_example, f"SGS -- example realization (#0 of {sgs_n_realizations}, range={range_label})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(axes[1], sgs_mean, "SGS -- ensemble mean", mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df)
    _panel(
        axes[2], sgs_var, "SGS -- ensemble variance", var_vmin, var_vmax, "magma",
        "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.98, top=0.9, wspace=0.35)
    _finish(fig, f"qc_truth_predictions_sgs_range{int(hmaj1)}.png", caption_ax=axes[0])

    # --- 4. rbf_bootstrap (3 panels) ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(
        axes[0], bootstrap_example,
        f"RBF+bootstrap -- example replicate (#0 of {bootstrap_n_replicates}, range={range_label})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(axes[1], bootstrap_mean, "RBF+bootstrap -- ensemble mean", mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df)
    _panel(
        axes[2], bootstrap_var, "RBF+bootstrap -- ensemble variance", var_vmin, var_vmax, "magma",
        "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.98, top=0.9, wspace=0.35)
    _finish(fig, f"qc_truth_predictions_rbf_bootstrap_range{int(hmaj1)}.png", caption_ax=axes[0])

    # --- 5. gp_mle (3 panels) ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(
        axes[0], gp_sample, f"GP-MLE -- posterior sample (1 draw, range={range_label})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(axes[1], gp_mean, "GP-MLE -- posterior mean", mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df)
    _panel(
        axes[2], gp_var, "GP-MLE -- posterior variance", var_vmin, var_vmax, "magma",
        "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.98, top=0.9, wspace=0.35)
    _finish(fig, f"qc_truth_predictions_gp_mle_range{int(hmaj1)}.png", caption_ax=axes[0])

    print(
        f"  range={range_label}: porosity scale=[{mean_vmin:.4f},{mean_vmax:.4f}], "
        f"variance scale (SGS/RBF/GP-anchored)=[{var_vmin:.4f},{var_vmax:.4f}], "
        f"kriging variance actual max={kriging_var_actual_max:.4f}"
    )
    return saved


def main():
    saved_files = []
    saved_files.extend(make_metric_vs_range_figure())
    saved_files.append(make_calibration_grid_figure())

    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    for hmaj1 in RANGE_VALUES:
        axis_level = str(int(hmaj1))
        print(f"\nBuilding truth/predictions QC figures for range={hmaj1:g}m...")
        saved_files.extend(
            make_truth_predictions_figure_set(hmaj1, source_runs[axis_level], FIGURES_DIR)
        )

    print("\nAll range-axis figures:")
    for f in saved_files:
        print(" ", f)
    return saved_files


if __name__ == "__main__":
    main()
