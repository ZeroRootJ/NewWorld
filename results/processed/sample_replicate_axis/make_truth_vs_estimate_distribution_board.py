"""2x3 TRUTH vs. POINT-ESTIMATE DISTRIBUTION COMPARISON BOARD (panels A-F), same
panel selection/layout as make_conditional_expectation_scatter_board.py, at the
1% sample-density level (axis_level="1") of the sample-seed replicate axis
(src/experiments/sample_replicate_axis.py, src/experiments/
evaluate_sample_replicate_axis.py).

THIS IS A DIAGNOSTIC-ONLY FIGURE, NOT a paper figure -- per 2026-10-16 user
instruction, it does NOT follow docs/figure_guidelines.md (Elsevier style).

Layout (identical panel selection to make_conditional_expectation_scatter_
board.py; see that script's docstring for the full rationale)
--------------------------------------------------------------------------
  Row 1:  A. SGS,            rep0, fixed input range 300 m
          B. GP-MLE,         MIN fitted-length replicate
          C. RBF+bootstrap,  MIN fitted-length replicate
  Row 2:  D. Simple kriging, rep0, fixed input range 300 m
          E. GP-MLE,         MAX fitted-length replicate
          F. RBF+bootstrap,  MAX fitted-length replicate

Each panel: overlaid histograms of (1) the ground-truth porosity values
("truth_masked") and (2) that method's point-estimate Z* values
("point_estimate_masked") over the SAME evaluated cells (conditioning cells
excluded) -- NOT a scatter/regression. Both histograms in a panel share the
SAME bin edges (built from the pooled [truth, estimate] range) so the two
shapes are directly comparable. Truth is drawn as a neutral black step
outline; the point estimate is drawn as a semi-transparent fill in the
method's project-wide color.

Data loading -- NOTHING new is implemented here
------------------------------------------------
Exactly the same loaders as make_conditional_expectation_scatter_board.py,
reused verbatim (not reimplemented):
  - make_fixed_range_case_study_figures.load_fixed_range_row() for SGS and
    simple kriging (CASE_STUDY_REPLICATE = "rep0").
  - make_length_case_study_figures.identify_length_extremes() +
    .load_case_study_row() for GP-MLE and RBF+bootstrap (MIN/MAX
    fitted-length replicates, recomputed from length_scale_by_replicate.csv).
Both loaders already recompute-and-assert (np.isclose) that replicate's
samples.csv / UMG / MSE / variance_mean against the pinned results/processed/
sample_replicate_axis/metrics.csv before returning, and return, among other
things, "truth_masked" and "point_estimate_masked" -- the exact two arrays
this script histograms. No slope/regression is computed here (unlike the
scatter board) and nothing is cross-checked against case_study_conditional_
expectation.csv, since this board does not touch that quantity at all.

Run with:
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_truth_vs_estimate_distribution_board
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

from results.processed.sample_replicate_axis import (  # noqa: E402
    make_length_case_study_figures as len_mod,
)
from results.processed.sample_replicate_axis import (  # noqa: E402
    make_fixed_range_case_study_figures as fr_mod,
)

PROCESSED_DIR = len_mod.PROCESSED_DIR
FIGURES_DIR = len_mod.FIGURES_DIR
AXIS_LEVEL = len_mod.AXIS_LEVEL
FIG_DPI = len_mod.FIG_DPI

OUTPUT_CSV = PROCESSED_DIR / "truth_vs_estimate_distribution_board.csv"
OUTPUT_FIG = FIGURES_DIR / "truth_vs_estimate_distribution_board.png"
OUTPUT_COLUMNS = [
    "panel", "method", "role", "replicate_id", "length_or_range_m",
    "truth_mean", "truth_std", "estimate_mean", "estimate_std", "n_cells",
]

PANEL_LABELS = ("A", "B", "C", "D", "E", "F")

# Same method colors as make_conditional_expectation_scatter_board.py (reused,
# not re-invented): kriging/sgs from the project-wide METHOD_COLORS dict,
# gp_mle/rbf_bootstrap from make_length_case_study_figures.make_case_study_figure().
METHOD_COLORS = dict(fr_mod.METHOD_COLORS)
METHOD_COLORS.update({"gp_mle": "tab:red", "rbf_bootstrap": "tab:orange"})

N_HIST_BINS = 30


def build_panel(method: str, role: str, row_data: dict) -> dict:
    """Package one panel's (truth_masked, point_estimate_masked) arrays plus
    their means/stds, straight from the existing loader's row_data -- no new
    recomputation beyond the two arrays' own mean/std."""
    truth_masked = np.asarray(row_data["truth_masked"], dtype=float)
    estimate_masked = np.asarray(row_data["point_estimate_masked"], dtype=float)
    n_cells = int(truth_masked.size)
    return {
        "method": method,
        "role": role,
        "replicate_id": row_data["replicate_id"],
        "truth_masked": truth_masked,
        "estimate_masked": estimate_masked,
        "truth_mean": float(np.mean(truth_masked)),
        "truth_std": float(np.std(truth_masked)),
        "estimate_mean": float(np.mean(estimate_masked)),
        "estimate_std": float(np.std(estimate_masked)),
        "n_cells": n_cells,
    }


def plot_panel(ax, panel: dict, color: str, title: str, letter: str) -> None:
    """Draw one of the 6 panels: overlaid Truth-vs-Estimate histograms on
    shared bin edges, density-normalized y-axis, legend with mean/std, and
    the same panel-letter convention as make_conditional_expectation_scatter_
    board.py."""
    truth = panel["truth_masked"]
    estimate = panel["estimate_masked"]

    pooled = np.concatenate([truth, estimate])
    bin_edges = np.histogram_bin_edges(pooled, bins=N_HIST_BINS)

    ax.hist(
        truth, bins=bin_edges, density=True, histtype="step", color="black",
        linewidth=1.6,
        label=f"Truth (GT): mean={panel['truth_mean']:.2f}, std={panel['truth_std']:.2f}",
    )
    ax.hist(
        estimate, bins=bin_edges, density=True, histtype="stepfilled", alpha=0.45,
        color=color, edgecolor=color, linewidth=1.2,
        label=f"Estimate Z*: mean={panel['estimate_mean']:.2f}, std={panel['estimate_std']:.2f}",
    )

    ax.set_xlabel("Porosity (%)")
    ax.set_ylabel("Density (normalized histogram)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)

    ax.text(
        -0.16, 1.12, letter, transform=ax.transAxes, ha="left", va="top",
        fontsize=16, fontweight="bold", clip_on=False,
    )


def main():
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    length_df = pd.read_csv(PROCESSED_DIR / "length_scale_by_replicate.csv", comment="#")

    truth = len_mod.get_base_case_truth(
        truth_seed=len_mod.TRUTH_SEED, hmaj1=len_mod.AXIS_HMAJ1, hmin1=len_mod.AXIS_HMIN1
    )
    print(
        f"Truth-vs-estimate distribution comparison board, axis_level={AXIS_LEVEL!r}: "
        f"truth_seed={len_mod.TRUTH_SEED}, hmaj1={len_mod.AXIS_HMAJ1:g} m"
    )

    # --- Load each panel's row through the EXISTING, unmodified loaders ----
    sgs_row = fr_mod.load_fixed_range_row(
        "sgs", fr_mod.CASE_STUDY_REPLICATE, truth, source_runs, metrics_df
    )
    kriging_row = fr_mod.load_fixed_range_row(
        "kriging", fr_mod.CASE_STUDY_REPLICATE, truth, source_runs, metrics_df
    )

    gp_extremes = len_mod.identify_length_extremes(length_df, "gp_mle")
    gp_min_row = len_mod.load_case_study_row(
        "gp_mle", "MIN", gp_extremes["min"], truth, source_runs, metrics_df
    )
    gp_max_row = len_mod.load_case_study_row(
        "gp_mle", "MAX", gp_extremes["max"], truth, source_runs, metrics_df
    )

    rbf_extremes = len_mod.identify_length_extremes(length_df, "rbf_bootstrap")
    rbf_min_row = len_mod.load_case_study_row(
        "rbf_bootstrap", "MIN", rbf_extremes["min"], truth, source_runs, metrics_df
    )
    rbf_max_row = len_mod.load_case_study_row(
        "rbf_bootstrap", "MAX", rbf_extremes["max"], truth, source_runs, metrics_df
    )

    panels = {
        "A": build_panel("sgs", fr_mod.CONDITIONAL_EXPECTATION_ROLE, sgs_row),
        "B": build_panel("gp_mle", "MIN", gp_min_row),
        "C": build_panel("rbf_bootstrap", "MIN", rbf_min_row),
        "D": build_panel("kriging", fr_mod.CONDITIONAL_EXPECTATION_ROLE, kriging_row),
        "E": build_panel("gp_mle", "MAX", gp_max_row),
        "F": build_panel("rbf_bootstrap", "MAX", rbf_max_row),
    }

    titles = {
        "A": f"SGS / {sgs_row['replicate_id']} / input range {fr_mod.AXIS_HMAJ1:g} m",
        "B": (
            f"GP-MLE / min fitted length, {gp_min_row['replicate_id']} "
            f"({gp_min_row['length_scale_m']:.0f} m)"
        ),
        "C": (
            f"RBF-bootstrap / min fitted length, {rbf_min_row['replicate_id']} "
            f"({rbf_min_row['length_scale_m']:.0f} m)"
        ),
        "D": f"Kriging / {kriging_row['replicate_id']} / input range {fr_mod.AXIS_HMAJ1:g} m",
        "E": (
            f"GP-MLE / max fitted length, {gp_max_row['replicate_id']} "
            f"({gp_max_row['length_scale_m']:.0f} m)"
        ),
        "F": (
            f"RBF-bootstrap / max fitted length, {rbf_max_row['replicate_id']} "
            f"({rbf_max_row['length_scale_m']:.0f} m)"
        ),
    }
    colors = {
        "A": METHOD_COLORS["sgs"], "B": METHOD_COLORS["gp_mle"], "C": METHOD_COLORS["rbf_bootstrap"],
        "D": METHOD_COLORS["kriging"], "E": METHOD_COLORS["gp_mle"], "F": METHOD_COLORS["rbf_bootstrap"],
    }
    length_or_range_m = {
        "A": float(fr_mod.AXIS_HMAJ1), "D": float(fr_mod.AXIS_HMAJ1),
        "B": float(gp_min_row["length_scale_m"]), "E": float(gp_max_row["length_scale_m"]),
        "C": float(rbf_min_row["length_scale_m"]), "F": float(rbf_max_row["length_scale_m"]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 10.2))
    for ax, letter in zip(axes.ravel(), PANEL_LABELS):
        plot_panel(ax, panels[letter], colors[letter], titles[letter], letter)

    fig.suptitle(
        "DIAGNOSTIC (not a paper figure): Truth vs. point-estimate distribution comparison "
        "(1% sample-density level, axis_level='1') -- overlaid histograms of truth_masked "
        "(black step outline) and point_estimate_masked (filled, method color) over the "
        "same evaluated cells, shared bin edges per panel",
        fontsize=12.0, y=0.998,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.95])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n{OUTPUT_FIG.name}: {OUTPUT_FIG} ({OUTPUT_FIG.stat().st_size} bytes)")

    print("\nPer-panel truth vs. estimate mean +/- std (diagnostic check):")
    rows = []
    for letter in PANEL_LABELS:
        p = panels[letter]
        print(
            f"  panel {letter} [{p['method']:<13s} {p['role']:<12s} {p['replicate_id']}] "
            f"truth: mean={p['truth_mean']:.3f} std={p['truth_std']:.3f}  |  "
            f"estimate: mean={p['estimate_mean']:.3f} std={p['estimate_std']:.3f}  |  "
            f"n_cells={p['n_cells']}"
        )
        rows.append({
            "panel": letter,
            "method": p["method"],
            "role": p["role"],
            "replicate_id": p["replicate_id"],
            "length_or_range_m": length_or_range_m[letter],
            "truth_mean": p["truth_mean"],
            "truth_std": p["truth_std"],
            "estimate_mean": p["estimate_mean"],
            "estimate_std": p["estimate_std"],
            "n_cells": p["n_cells"],
        })

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{OUTPUT_CSV.name}: {OUTPUT_CSV} ({len(out_df)} rows)")
    print("\nSummary:")
    print(out_df.to_string(index=False))

    return OUTPUT_FIG, OUTPUT_CSV


if __name__ == "__main__":
    main()
