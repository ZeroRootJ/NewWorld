"""Build the sample-seed replicate axis figure (src/experiments/
sample_replicate_axis.py, src/experiments/evaluate_sample_replicate_axis.py):
how much do mse / umg / variance_mean move across 10 sample-location
replicates drawn at the SAME ground truth and the SAME requested sample
count (n_samples=125), for each of the 4 methods (kriging / SGS /
RBF+bootstrap / GP-MLE)?

ONE figure, 3 panels (mse / umg / variance_mean), x-axis = method, y-axis =
metric value: each replicate's value is plotted as a jittered point per
method, with an errorbar marker at that method's mean +- 1 std (ddof=1)
across the 10 replicates (from metrics_summary.csv).

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

from src.experiments.sample_replicate_axis import (  # noqa: E402
    AXIS_HMAJ1,
    METHODS,
    N_SAMPLES_REQUESTED,
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


def _caption(seeds_record: dict) -> str:
    n_actual = list(seeds_record["n_samples_actual"].values())
    seeds = list(seeds_record["sample_seed_replicates"].values())
    return (
        f"Same ground-truth field for all {len(REPLICATE_IDS)} replicates "
        f"(TRUTH_SEED={TRUTH_SEED}, range={AXIS_HMAJ1:g} m); only sample LOCATIONS vary, "
        f"via sample_seed in {seeds}. n_samples requested={N_SAMPLES_REQUESTED} for every "
        f"replicate; actual count after grid-cell dedup ranges {min(n_actual)}-"
        f"{max(n_actual)} (see replicate_seeds.json). Each small marker is one replicate's "
        "value for that method (jittered horizontally for visibility, matching color); the "
        "larger black-edged marker is the mean across the 10 replicates and the error bar is "
        "+-1 sample std (ddof=1) -- see metrics_summary.csv. variance_mean uses each method's "
        "own physical-unit variance array (kriging's is a Monte Carlo back-transform "
        "approximation; the other three are native) -- see variance_metric_sources.csv."
    )


def make_figure():
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    summary_df = pd.read_csv(PROCESSED_DIR / "metrics_summary.csv")
    with open(PROCESSED_DIR / "replicate_seeds.json", "r", encoding="utf-8") as f:
        seeds_record = json.load(f)

    rng = np.random.default_rng(JITTER_SEED)
    x_positions = {m: i for i, m in enumerate(METHODS)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.0))
    for ax, (metric_name, title, ylabel) in zip(axes, PANELS):
        sub = metrics_df[metrics_df["metric"] == metric_name]
        for method in METHODS:
            m_sub = sub[sub["method"] == method].sort_values("axis_level")
            if len(m_sub) != len(REPLICATE_IDS):
                raise ValueError(
                    f"{method}/{metric_name}: expected {len(REPLICATE_IDS)} replicate rows "
                    f"in metrics.csv, found {len(m_sub)}."
                )
            x0 = x_positions[method]
            jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(m_sub))
            ax.scatter(
                x0 + jitter, m_sub["value"], color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method], alpha=0.55, s=45, zorder=2,
                label="_nolegend_",
            )

            row = summary_df[
                (summary_df["method"] == method) & (summary_df["metric"] == metric_name)
            ]
            if len(row) != 1:
                raise ValueError(f"{method}/{metric_name}: expected 1 summary row, found {len(row)}.")
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
        "Sample-seed replicate axis: spread across 10 sample-location draws at fixed "
        f"n_samples_requested={N_SAMPLES_REQUESTED}",
        fontsize=13,
    )
    fig.text(
        0.5, 0.02, textwrap.fill(_caption(seeds_record), 165),
        ha="center", va="top", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.06, bottom=0.28, right=0.98, top=0.86, wspace=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "metrics_replicate_spread.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"metrics_replicate_spread.png: {out} ({out.stat().st_size} bytes)")
    return out


def main():
    saved = [make_figure()]
    print("\nAll sample-replicate-axis figures:")
    for f in saved:
        print(" ", f)
    return saved


if __name__ == "__main__":
    main()
