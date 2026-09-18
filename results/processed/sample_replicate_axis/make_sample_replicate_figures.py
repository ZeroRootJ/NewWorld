"""Build the sample-seed replicate axis figure (src/experiments/
sample_replicate_axis.py, src/experiments/evaluate_sample_replicate_axis.py):
how much do mse / umg / variance_mean move across 10 sample-location
replicates drawn at the SAME ground truth, for each of the 4 methods
(kriging / SGS / RBF+bootstrap / GP-MLE)?

2026-09-18 EXTENSION: sample_replicate_axis.py now repeats this design at ALL
3 sample-density axis levels ("5"/"2"/"1" -> n_samples_requested = 125/50/25,
SAME 10 sample_seed replicates reused at each), not just n=125. This script
is updated to match: ONE figure PER density level (3 PNGs total,
metrics_replicate_spread_level5.png / _level2.png / _level1.png), each with
the SAME 3-panel layout as before (mse / umg / variance_mean, x-axis =
method) -- the per-level figure content/style is otherwise unchanged from
the original (n=125-only) version of this script.

For a cross-level comparison of the SAME data (mean +-1 std per method,
connected across levels rather than faceted by level), see
results/processed/sample_density_axis/make_sample_density_figures.py's
``metrics_vs_sample_density.png``, which was updated in this same task to
read this axis's 10-replicate output instead of a single pinned run.

ONE figure per level, 3 panels (mse / umg / variance_mean), x-axis = method,
y-axis = metric value: each replicate's value is plotted as a jittered point
per method, with an errorbar marker at that method's mean +- 1 std (ddof=1)
across the 10 replicates (from metrics_summary.csv, filtered to that level).

Colors/markers reused (redefined identically, not imported, to avoid running
make_sample_density_figures.py's module-level file reads as an import side
effect) from results/processed/sample_density_axis/make_sample_density_figures.py's
METHOD_COLORS / METHOD_MARKERS convention.

Run with:
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_sample_replicate_figures
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

from src.experiments.sample_density_axis import ALL_AXIS_LEVELS  # noqa: E402
from src.experiments.sample_replicate_axis import (  # noqa: E402
    AXIS_HMAJ1,
    METHODS,
    N_SAMPLES_REQUESTED_BY_LEVEL,
    REPLICATE_IDS,
    TRUTH_SEED,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "sample_replicate_axis"

FIG_DPI = 600

METHOD_COLORS = {
    "kriging": "tab:blue", "sgs": "tab:green",
    "rbf_bootstrap": "tab:orange", "gp_mle": "tab:red",
}
METHOD_MARKERS = {"kriging": "o", "sgs": "s", "rbf_bootstrap": "^", "gp_mle": "D"}
METHOD_LABELS = {
    "kriging": "Simple kriging", "sgs": "SGS",
    "rbf_bootstrap": "RBF+bootstrap", "gp_mle": "GP-MLE",
}

# Deterministic, purely cosmetic horizontal jitter so 10 overlapping
# replicate points per method are visible.
JITTER_WIDTH = 0.12
JITTER_SEED = 7

PANELS = [
    ("mse", "Accuracy: MSE across 10 sample-location replicates",
     "MSE (Porosity %$^2$, lower better)"),
    ("umg", "Calibration: UMG across 10 sample-location replicates",
     "UMG (1.0 = perfectly calibrated)"),
    ("variance_mean", "Predictive variance: mean over evaluated cells",
     "Mean predictive variance (Porosity %$^2$)"),
]

# axis_level ("5"/"2"/"1") -> output filename suffix / label, reused
# consistently with make_sample_density_figures.py's own level identifiers.
LEVEL_FILENAME_SUFFIX = {"5": "level5", "2": "level2", "1": "level1"}


def _caption(axis_level: str, seeds_record: dict) -> str:
    n_actual = list(seeds_record["n_samples_actual"][axis_level].values())
    seeds = list(seeds_record["sample_seed_replicates"].values())
    n_requested = N_SAMPLES_REQUESTED_BY_LEVEL[axis_level]
    return (
        f"Same ground-truth field for all {len(REPLICATE_IDS)} replicates at this level "
        f"(TRUTH_SEED={TRUTH_SEED}, range={AXIS_HMAJ1:g} m); only sample LOCATIONS vary, "
        f"via sample_seed in {seeds} -- the SAME 10 seeds reused at every one of the 3 "
        f"density levels (see sample_replicate_axis.py's module docstring for the measured "
        "X/Y draw coupling this reuse creates BETWEEN levels). n_samples requested="
        f"{n_requested} for every replicate at this level; actual count after grid-cell "
        f"dedup ranges {min(n_actual)}-{max(n_actual)} (see replicate_seeds.json). Each "
        "small marker is one replicate's value for that method (jittered horizontally for "
        "visibility, matching color); the larger black-edged marker is the mean across the "
        "10 replicates and the error bar is +-1 sample std (ddof=1) -- see "
        "metrics_summary.csv. variance_mean uses each method's own physical-unit variance "
        "array (kriging's is a Monte Carlo back-transform approximation; the other three "
        "are native) -- see variance_metric_sources.csv."
    )


def make_figure(axis_level: str, metrics_df: pd.DataFrame, summary_df: pd.DataFrame, seeds_record: dict):
    level_metrics = metrics_df[metrics_df["axis_level"] == axis_level]
    level_summary = summary_df[summary_df["axis_level"] == axis_level]
    n_requested = N_SAMPLES_REQUESTED_BY_LEVEL[axis_level]

    rng = np.random.default_rng(JITTER_SEED)
    x_positions = {m: i for i, m in enumerate(METHODS)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.0))
    for ax, (metric_name, title, ylabel) in zip(axes, PANELS):
        sub = level_metrics[level_metrics["metric"] == metric_name]
        for method in METHODS:
            m_sub = sub[sub["method"] == method].sort_values("replicate")
            if len(m_sub) != len(REPLICATE_IDS):
                raise ValueError(
                    f"level '{axis_level}' {method}/{metric_name}: expected "
                    f"{len(REPLICATE_IDS)} replicate rows in metrics.csv, found {len(m_sub)}."
                )
            x0 = x_positions[method]
            jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(m_sub))
            ax.scatter(
                x0 + jitter, m_sub["value"], color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method], alpha=0.55, s=45, zorder=2,
                label="_nolegend_",
            )

            row = level_summary[
                (level_summary["method"] == method) & (level_summary["metric"] == metric_name)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"level '{axis_level}' {method}/{metric_name}: expected 1 summary row, "
                    f"found {len(row)}."
                )
            row = row.iloc[0]
            ax.errorbar(
                [x0], [row["mean"]], yerr=[row["std"]], color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method], markersize=10, markeredgecolor="black",
                markeredgewidth=1.2, capsize=5, linewidth=2, zorder=3,
                label=f"{METHOD_LABELS[method]} (mean +-1 std)",
            )
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=12)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if metric_name == "umg":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ideal (UMG=1.0)")
        ax.legend(fontsize=7, loc="best")

    plt.suptitle(
        f"Sample-seed replicate axis (level '{axis_level}'): spread across 10 "
        f"sample-location draws at n_samples_requested={n_requested}",
        fontsize=13,
    )
    fig.text(
        0.5, 0.02, textwrap.fill(_caption(axis_level, seeds_record), 165),
        ha="center", va="top", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.06, bottom=0.28, right=0.98, top=0.86, wspace=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"metrics_replicate_spread_{LEVEL_FILENAME_SUFFIX[axis_level]}.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"{out.name}: {out} ({out.stat().st_size} bytes)")
    return out


def main():
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    metrics_df["axis_level"] = metrics_df["axis_level"].astype(str)
    summary_df = pd.read_csv(PROCESSED_DIR / "metrics_summary.csv")
    summary_df["axis_level"] = summary_df["axis_level"].astype(str)
    with open(PROCESSED_DIR / "replicate_seeds.json", "r", encoding="utf-8") as f:
        seeds_record = json.load(f)

    saved = [
        make_figure(axis_level, metrics_df, summary_df, seeds_record)
        for axis_level in ALL_AXIS_LEVELS
    ]
    print("\nAll sample-replicate-axis figures:")
    for f in saved:
        print(" ", f)
    return saved


if __name__ == "__main__":
    main()
