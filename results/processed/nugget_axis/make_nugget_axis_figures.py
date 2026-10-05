"""Build the nugget-axis figures (docs/experiment_context.md deliverable 3),
over the 6 axis levels normalized nugget = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5.

(a) MAIN FIGURE -- a 3-panel metric-vs-nugget plot (MSE / UMG /
    variance_mean), one line per method. The sharpness family (interval
    width, CRPS) is not plotted: it was removed project-wide on 2026-10-05
    (user decision: no longer computed or stored).
(b) Calibration (accuracy-plot) curves for all 6 levels in one 2x3 grid.
(c) Per-level "truth & predictions" QC figures for all 6 levels, reusing the
    panel-layout/shared-colorbar logic of
    results/processed/range_axis/make_range_axis_figures.py.

X-AXIS: the panels are plotted against the NORMALIZED nugget (fraction of
the unit sill on standard-normal space), which is the quantity the axis is
actually defined on. A secondary top x-axis gives the same levels in
physical units (nug * POR_STDEV**2, Porosity %^2, against a total sill of
POR_STDEV**2 = 9.0), computed via nugget_axis.truth_nugget_real_units rather
than hardcoded, and the caption repeats the conversion.

COLOR-SCALE CONVENTION for the QC figures (c) -- flagged explicitly because
it constrains what the figures can be used for: the color scale is shared
WITHIN an axis level, NOT across axis levels. Within one nugget level: all
porosity panels (truth / realization / mean) share that level's truth
min-max, and all variance panels share one scale anchored on the
SGS/RBF/GP-MLE variance maxima (kriging's physical-unit Monte Carlo variance
is annotated if it exceeds that shared max). Rationale: each nugget level
has its own ground-truth realization, so the fields' value ranges genuinely
differ between levels and a single global scale would wash out the
within-level method comparison these panels exist for. CONSEQUENCE: panels
from different nugget levels are NOT directly comparable by color -- read
the numbers in the metric figure (a) for cross-level comparison. This is
stated in the figure captions, not just here.

NOTE ON THE BASE CASE: the base case is NUG=0.05, which lies BETWEEN levels
0.0 and 0.1 and is therefore not an axis level. It is drawn as a light
reference line on the main figure purely for orientation; no base-case
number enters any of this axis's tables.

Run dirs are read from results/processed/nugget_axis/source_runs.json.

Run with:
.venv/Scripts/python.exe -m results.processed.nugget_axis.make_nugget_axis_figures
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.base_case import (  # noqa: E402
    NUG as BASE_CASE_NUG,
    POR_STDEV,
    XMAX,
    XMIN,
    YMAX,
    YMIN,
)
from src.experiments.base_case_conditioning import (  # noqa: E402
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.nugget_axis import (  # noqa: E402
    ALL_NUGGET_VALUES,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    METHODS,
    level_file_token,
    level_key,
    truth_nugget_real_units,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "nugget_axis"

# QC figures are large (5 files x 6 levels); 300 dpi matches the range axis.
QC_FIG_DPI = 300
MAIN_FIG_DPI = 600

EXTENT = [XMIN, XMAX, YMIN, YMAX]

METHOD_COLORS = {
    "kriging": "tab:blue", "sgs": "tab:green",
    "rbf_bootstrap": "tab:orange", "gp_mle": "tab:red",
}
METHOD_MARKERS = {"kriging": "o", "sgs": "s", "rbf_bootstrap": "^", "gp_mle": "D"}

NUGGET_LEVELS = [float(n) for n in ALL_NUGGET_VALUES]
TOTAL_SILL_REAL = POR_STDEV ** 2

# (metric key in metrics.csv, panel title, y-axis label).
MAIN_PANELS = [
    ("mse", "Accuracy: MSE vs. nugget", "MSE (Porosity %$^2$, lower better)"),
    ("umg", "Calibration: UMG vs. nugget", "UMG (1.0 = perfectly calibrated)"),
    (
        "variance_mean",
        "Predictive variance vs. nugget",
        "Mean predictive variance (Porosity %$^2$)",
    ),
]


def _add_physical_nugget_axis(ax):
    """Secondary top x-axis labelled in physical units (Porosity %^2).

    The conversion is nug * POR_STDEV**2, obtained from
    nugget_axis.truth_nugget_real_units (never hardcoded), so this axis
    cannot drift from the value the experiment actually used.
    """
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda v: np.asarray(v) * TOTAL_SILL_REAL,
            lambda v: np.asarray(v) / TOTAL_SILL_REAL,
        ),
    )
    secax.set_xticks([truth_nugget_real_units(n) for n in NUGGET_LEVELS])
    secax.set_xlabel(
        f"Ground-truth nugget (Porosity %$^2$; total sill = {TOTAL_SILL_REAL:g})",
        fontsize=9,
    )
    return secax


# ---------------------------------------------------------------------------
# (a) Main 3-panel metric-vs-nugget figure
# ---------------------------------------------------------------------------
def make_metric_vs_nugget_figure():
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    metrics_df["axis_level_numeric"] = metrics_df["axis_level"].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    for ax, (metric_name, title, ylabel) in zip(axes.ravel(), MAIN_PANELS):
        sub = metrics_df[metrics_df["metric"] == metric_name]
        if sub.empty:
            raise ValueError(f"metrics.csv has no rows for metric '{metric_name}'.")
        for method in METHODS:
            m_sub = sub[sub["method"] == method].sort_values("axis_level_numeric")
            if len(m_sub) != len(NUGGET_LEVELS):
                raise ValueError(
                    f"metric '{metric_name}', method '{method}': found {len(m_sub)} "
                    f"levels, expected {len(NUGGET_LEVELS)}."
                )
            ax.plot(
                m_sub["axis_level_numeric"], m_sub["value"],
                marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                label=method, linewidth=2,
            )
        ax.set_xticks(NUGGET_LEVELS)
        ax.set_xlabel("Ground-truth nugget (fraction of unit sill)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        # The base case (NUG=0.05) is NOT an axis level -- reference only.
        ax.axvline(
            BASE_CASE_NUG, color="gray", linestyle=":", linewidth=1.2,
            label=f"base case (nug={BASE_CASE_NUG:g}, not an axis level)",
        )
        if metric_name == "umg":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ideal (UMG=1.0)")
        ax.legend(fontsize=8)
        _add_physical_nugget_axis(ax)

    plt.suptitle(
        "Nugget axis: accuracy (MSE) vs. calibration (UMG) vs. predictive variance\n"
        "one ground-truth realization per level (TRUTH_SEED=101), range fixed at "
        f"{AXIS_HMAJ1:g} m isotropic, identical sample locations across methods and levels",
        fontsize=12,
    )
    fig.text(
        0.5, 0.005, textwrap.fill(
            "Bottom x-axis: normalized nugget (fraction of the unit sill on standard-normal "
            f"space). Top x-axis: the same levels in physical units, nug * POR_STDEV^2, against "
            f"a total sill of {TOTAL_SILL_REAL:g} Porosity %^2. kriging and SGS are handed the TRUE "
            "nugget as their input variogram (they are the reference baselines); GP-MLE learns "
            "its own noise variance by marginal likelihood and RBF+bootstrap has no nugget "
            "parameter at all.",
            170,
        ),
        ha="center", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.06, bottom=0.17, right=0.98, top=0.80, wspace=0.28)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "metrics_vs_nugget.png"
    plt.savefig(out, dpi=MAIN_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"metrics_vs_nugget.png: {out} ({out.stat().st_size} bytes)")
    return out


# ---------------------------------------------------------------------------
# (b) Calibration curves, 2x3 grid over the 6 axis levels
# ---------------------------------------------------------------------------
def make_calibration_grid_figure():
    curves_df = pd.read_csv(PROCESSED_DIR / "accuracy_plot_curves.csv")
    curves_df["axis_level"] = curves_df["axis_level"].astype(str)
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    metrics_df["axis_level"] = metrics_df["axis_level"].astype(str)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, nug in zip(axes.ravel(), NUGGET_LEVELS):
        axis_level = level_key(nug)
        sub = curves_df[curves_df["axis_level"] == axis_level]
        if sub.empty:
            raise ValueError(f"accuracy_plot_curves.csv has no rows for level {axis_level}.")
        for method in METHODS:
            m_sub = sub[sub["method"] == method]
            umg_rows = metrics_df[
                (metrics_df["axis_level"] == axis_level)
                & (metrics_df["method"] == method)
                & (metrics_df["metric"] == "umg")
            ]["value"]
            if len(umg_rows) != 1:
                raise ValueError(
                    f"expected exactly 1 UMG row for level {axis_level}, method {method}; "
                    f"found {len(umg_rows)}."
                )
            ax.plot(
                m_sub["p_interval"], m_sub["fraction_in"],
                marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], markersize=4,
                label=f"{method} (UMG={umg_rows.iloc[0]:.3f})", alpha=0.85,
            )
        ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", label="ideal (y=x)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Nominal probability interval")
        ax.set_ylabel("Fraction of truth in interval")
        ax.set_title(
            f"nugget = {nug:g} ({truth_nugget_real_units(nug):g} Porosity %$^2$)"
        )
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle(
        "Nugget axis calibration (accuracy) plots, 6 axis levels\n"
        "titles give the normalized nugget and, in parentheses, the same nugget in "
        f"physical units (total sill = {TOTAL_SILL_REAL:g} Porosity %$^2$)",
        fontsize=13,
    )
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.89, wspace=0.28, hspace=0.28)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "calibration_curves_grid.png"
    plt.savefig(out, dpi=MAIN_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"calibration_curves_grid.png: {out} ({out.stat().st_size} bytes)")
    return out


# ---------------------------------------------------------------------------
# (c) Per-level truth & predictions QC figures
# ---------------------------------------------------------------------------
def _scatter_samples(ax, samples_df, color="red"):
    ax.scatter(
        samples_df["X"], samples_df["Y"], s=10, c=color, marker="+",
        label="conditioning samples",
    )


def _panel(ax, arr, title, vmin, vmax, cmap, cbar_label, samples_df, scatter_color="red"):
    im = ax.imshow(arr, extent=EXTENT, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    _scatter_samples(ax, samples_df, color=scatter_color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, label=cbar_label)
    return im


SCALE_CAPTION = (
    "Color scales are shared WITHIN this nugget level only (porosity panels: this level's "
    "truth min-max; variance panels: SGS/RBF/GP-MLE-anchored max). Panels from different "
    "nugget levels are NOT comparable by color."
)


def make_truth_predictions_figure_set(nug: float, run_dirs: dict, out_dir: Path):
    """Generate the truth_field + 4 per-method QC figures for a single nugget
    level, following the panel layout / shared-colorbar convention of
    results/processed/range_axis/make_range_axis_figures.py."""
    token = level_file_token(nug)
    nug_label = f"{nug:g} ({truth_nugget_real_units(nug):g} Porosity %^2)"

    truth = get_base_case_truth(
        truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=AXIS_HMIN1, nug=float(nug)
    )
    samples_df = get_conditioning_samples(
        truth, sample_seed=SAMPLE_SEED, n_samples=N_SAMPLES
    )

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
        ax, truth,
        f"Nugget={nug_label} truth ({truth.shape[0]}x{truth.shape[1]}, seed={TRUTH_SEED})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    plt.subplots_adjust(left=0.12, bottom=0.16, right=0.98, top=0.92)
    _finish(fig, f"truth_field_nug{token}.png", caption_ax=ax)

    # --- 2. kriging (2 panels: mean, variance) ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _panel(
        axes[0], kriging_mean, f"Simple kriging -- point estimate (nugget={nug:g})",
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
    axes[1].text(
        0.02, -0.14, kvar_annotation, transform=axes[1].transAxes, fontsize=8, color="dimgray"
    )
    plt.subplots_adjust(left=0.06, bottom=0.2, right=0.98, top=0.9, wspace=0.35)
    _finish(fig, f"qc_truth_predictions_kriging_nug{token}.png", caption_ax=axes[0])

    # --- 3. sgs (3 panels) ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(
        axes[0], sgs_example,
        f"SGS -- example realization (#0 of {sgs_n_realizations}, nugget={nug:g})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[1], sgs_mean, "SGS -- ensemble mean", mean_vmin, mean_vmax, "viridis",
        "Porosity (%)", samples_df,
    )
    _panel(
        axes[2], sgs_var, "SGS -- ensemble variance", var_vmin, var_vmax, "magma",
        "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.98, top=0.9, wspace=0.35)
    _finish(fig, f"qc_truth_predictions_sgs_nug{token}.png", caption_ax=axes[0])

    # --- 4. rbf_bootstrap (3 panels) ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(
        axes[0], bootstrap_example,
        f"RBF+bootstrap -- example replicate (#0 of {bootstrap_n_replicates}, nugget={nug:g})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[1], bootstrap_mean, "RBF+bootstrap -- ensemble mean", mean_vmin, mean_vmax,
        "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[2], bootstrap_var, "RBF+bootstrap -- ensemble variance", var_vmin, var_vmax,
        "magma", "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.98, top=0.9, wspace=0.35)
    _finish(fig, f"qc_truth_predictions_rbf_bootstrap_nug{token}.png", caption_ax=axes[0])

    # --- 5. gp_mle (3 panels) ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(
        axes[0], gp_sample, f"GP-MLE -- posterior sample (1 draw, nugget={nug:g})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[1], gp_mean, "GP-MLE -- posterior mean", mean_vmin, mean_vmax, "viridis",
        "Porosity (%)", samples_df,
    )
    _panel(
        axes[2], gp_var, "GP-MLE -- posterior variance", var_vmin, var_vmax, "magma",
        "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.98, top=0.9, wspace=0.35)
    _finish(fig, f"qc_truth_predictions_gp_mle_nug{token}.png", caption_ax=axes[0])

    print(
        f"  nugget={nug:g}: porosity scale=[{mean_vmin:.4f},{mean_vmax:.4f}], "
        f"variance scale (SGS/RBF/GP-anchored)=[{var_vmin:.4f},{var_vmax:.4f}], "
        f"kriging variance actual max={kriging_var_actual_max:.4f}"
    )
    return saved


def main():
    saved_files = []
    saved_files.append(make_metric_vs_nugget_figure())
    saved_files.append(make_calibration_grid_figure())

    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    expected_levels = {level_key(n) for n in NUGGET_LEVELS}
    if set(source_runs) != expected_levels:
        raise ValueError(
            f"source_runs.json has levels {sorted(source_runs)}; expected "
            f"{sorted(expected_levels)}."
        )

    for nug in NUGGET_LEVELS:
        print(f"\nBuilding truth/predictions QC figures for nugget={nug:g}...")
        saved_files.extend(
            make_truth_predictions_figure_set(nug, source_runs[level_key(nug)], FIGURES_DIR)
        )

    print("\nAll nugget-axis figures:")
    for f in saved_files:
        print(" ", f)
    return saved_files


if __name__ == "__main__":
    main()
