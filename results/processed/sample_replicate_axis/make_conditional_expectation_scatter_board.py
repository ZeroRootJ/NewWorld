"""2x3 CONDITIONAL-EXPECTATION SCATTER COMPARISON BOARD (panels A-F) across the
four methods, at the 1% sample-density level (axis_level="1") of the
sample-seed replicate axis (src/experiments/sample_replicate_axis.py,
src/experiments/evaluate_sample_replicate_axis.py).

THIS IS A SEPARATE, NEW FIGURE -- NOT a replacement of the existing 9-panel
case-study boards' panel 7 (binned E[Z | Z*], make_length_case_study_figures.
compute_conditional_expectation / plot_conditional_expectation, written to
case_study_conditional_expectation.csv). That panel 7 and its CSV are
UNCHANGED by this script; this script only READS case_study_conditional_
expectation.csv, as a decisive cross-check (see below), and writes its own
separate CSV (conditional_expectation_scatter_board.csv).

Layout (matches a previous, now-lost notebook figure the user described)
--------------------------------------------------------------------------
  Row 1:  A. SGS,            rep0, fixed input range 300 m
          B. GP-MLE,         MIN fitted-length replicate
          C. RBF+bootstrap,  MIN fitted-length replicate
  Row 2:  D. Simple kriging, rep0, fixed input range 300 m
          E. GP-MLE,         MAX fitted-length replicate
          F. RBF+bootstrap,  MAX fitted-length replicate

Each panel: a raw scatter of (point estimate Z*, truth Z) over ALL of that
replicate's evaluated cells (conditioning cells excluded) -- NOT the 20
equal-count bins the existing panel 7 draws -- plus ONE OLS regression line
of truth on the point estimate over those same cells (color-matched to the
scatter), plus the 1:1 dashed gray reference line. The regression line is a
LINEAR APPROXIMATION of E[Truth | Estimate]; that is stated once in the
figure's suptitle and once per panel in the regression line's own legend
label, rather than silently implied.

Per 2026-10-16 user decision, the MACB statistic is DROPPED from this board's
per-panel stat box entirely -- only the OLS slope is shown (bottom-right box,
3 decimals). This is a presentation choice for THIS board only; it does not
touch make_length_case_study_figures.py / make_fixed_range_case_study_figures.py
or case_study_conditional_expectation.csv, which keep computing and storing
exactly what they already did (binned E[Z|Z*] curves; no MACB column exists
there either -- that script never computed MACB in the first place).

Data loading / recomputation -- NOTHING new is implemented here
------------------------------------------------------------------
All six panels reuse the EXISTING, already-reviewed loaders verbatim:
  - make_fixed_range_case_study_figures.load_fixed_range_row() for SGS and
    simple kriging (CASE_STUDY_REPLICATE = "rep0"), which already recomputes
    and asserts (np.isclose) that replicate's samples.csv / UMG / MSE /
    variance_mean against the pinned results/processed/sample_replicate_axis/
    metrics.csv before returning.
  - make_length_case_study_figures.identify_length_extremes() +
    .load_case_study_row() for GP-MLE and RBF+bootstrap (MIN/MAX
    fitted-length replicates, recomputed from length_scale_by_replicate.csv,
    never hardcoded), which does the equivalent samples.csv / UMG / MSE
    recompute-and-assert.
Both loaders return, among other things, "truth_masked" and
"point_estimate_masked" -- the SAME (truth, point-estimate) arrays the
existing panel 6 (accuracy crossplot) and panel 7 (binned conditional
expectation) already use. This script scatters exactly those two arrays and
calls the EXISTING make_length_case_study_figures.compute_conditional_
expectation(truth_masked, point_estimate_masked) to get the OLS slope (the
same "cov(Z*, Z) / var(Z*) over ALL evaluated cells" slope already used by
panel 7's title) -- the slope math itself is NOT reimplemented here. Only the
intercept is new, a single extra line this script adds:
    intercept = mean(truth_masked) - slope * mean(point_estimate_masked)
(the standard OLS intercept given that slope, since compute_conditional_
expectation does not return one).

DECISIVE CORRECTNESS CHECK (same recompute-and-assert pattern every other
script in this project uses): for each of the 6 panels, the recomputed slope
is compared (np.isclose) against the ALREADY-PINNED slope for the exact same
(method, level="1", replicate_id, role) combination in
case_study_conditional_expectation.csv (role="FIXED_RANGE" for kriging/sgs,
"MIN"/"MAX" for gp_mle/rbf_bootstrap -- every one of those 6 combinations
already has rows there, written by the two existing scripts). Any mismatch
raises; this script never silently plots a slope that disagrees with the
pinned value. Because both scripts load the data through the SAME loaders
and call the SAME compute_conditional_expectation(), these are expected to
match to full float precision, not just approximately -- a mismatch would
mean the two scripts have drifted apart.

The user also supplied, for context only (NOT an enforced target -- see
REFERENCE_IMAGE_SLOPES below), the six slope values read off the old,
now-lost notebook figure. Those are printed alongside each recomputed slope
as a "how close is this to the old figure" note; if a slope differs a lot
(plausible for rbf_bootstrap, whose MIN/MAX replicate selection changed after
the 2026-09-17 EPSILON_GRID redesign -- see docs/progress.md, 2026-09-17/18
entries) the real recomputed number and replicate id are used and printed,
never silently forced to match.

Run with:
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_conditional_expectation_scatter_board
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

# The existing, already-pinned panel-7 CSV this script cross-checks against
# (read-only here -- never written to by this script).
PINNED_CONDITIONAL_EXPECTATION_CSV = len_mod.CONDITIONAL_EXPECTATION_CSV

# This script's own, separate output files.
OUTPUT_CSV = PROCESSED_DIR / "conditional_expectation_scatter_board.csv"
OUTPUT_FIG = FIGURES_DIR / "conditional_expectation_scatter_comparison_board.png"
OUTPUT_COLUMNS = [
    "panel", "method", "role", "replicate_id", "length_or_range_m",
    "slope", "intercept", "n_cells",
]

# Same tolerance every other recompute-and-assert check in this project uses.
SLOPE_MATCH_RTOL = 1e-8
SLOPE_MATCH_ATOL = 1e-8

PANEL_LABELS = ("A", "B", "C", "D", "E", "F")

# Method colors: kriging/sgs reuse the project-wide METHOD_COLORS dict
# make_fixed_range_case_study_figures.py already defines; gp_mle/rbf_bootstrap
# reuse the exact inline colors make_length_case_study_figures.
# make_case_study_figure() already uses for its own conditional-expectation
# panel (`method_color = "tab:red" if method == "gp_mle" else "tab:orange"`)
# -- not a new color choice invented for this board.
METHOD_COLORS = dict(fr_mod.METHOD_COLORS)
METHOD_COLORS.update({"gp_mle": "tab:red", "rbf_bootstrap": "tab:orange"})

# Context-only reference values (NOT enforced -- see module docstring): the
# slopes read off the user's old, now-lost notebook figure, by panel letter.
REFERENCE_IMAGE_SLOPES = {
    "A": 0.771,  # SGS
    "B": 1.148,  # GP-MLE min
    "C": 1.104,  # RBF-bootstrap min
    "D": 1.199,  # Kriging
    "E": 0.900,  # GP-MLE max
    "F": 1.120,  # RBF-bootstrap max
}
# Loose, purely descriptive "close enough to flag" threshold for the stdout note.
REFERENCE_IMAGE_SLOPE_FLAG_ATOL = 0.15


def recompute_panel(method: str, role: str, row_data: dict, pinned_df: pd.DataFrame) -> dict:
    """Recompute this panel's OLS slope/intercept from ``row_data`` (the dict
    returned by load_case_study_row / load_fixed_range_row -- both carry
    "truth_masked" and "point_estimate_masked"), using the EXISTING
    compute_conditional_expectation() for the slope (not reimplemented), and
    cross-check the slope against the pinned case_study_conditional_
    expectation.csv row for this exact (method, level, replicate_id, role).
    Raises on any mismatch."""
    replicate_id = row_data["replicate_id"]
    truth_masked = np.asarray(row_data["truth_masked"], dtype=float)
    estimate_masked = np.asarray(row_data["point_estimate_masked"], dtype=float)

    _, slope = len_mod.compute_conditional_expectation(truth_masked, estimate_masked)
    intercept = float(np.mean(truth_masked) - slope * np.mean(estimate_masked))
    n_cells = int(truth_masked.size)

    pinned = pinned_df.loc[
        (pinned_df["method"] == method)
        & (pinned_df["level"].astype(str) == AXIS_LEVEL)
        & (pinned_df["replicate_id"] == replicate_id)
        & (pinned_df["role"] == role),
        "slope",
    ]
    if len(pinned) == 0:
        raise ValueError(
            f"no pinned case_study_conditional_expectation.csv rows found for "
            f"method={method}, level={AXIS_LEVEL}, replicate_id={replicate_id}, role={role}."
        )
    if not np.allclose(pinned.values, pinned.values[0]):
        raise ValueError(
            f"pinned slope is not constant across bins for method={method}, "
            f"replicate_id={replicate_id}, role={role}: {sorted(set(pinned.values))}."
        )
    pinned_slope = float(pinned.values[0])
    if not np.isclose(slope, pinned_slope, rtol=SLOPE_MATCH_RTOL, atol=SLOPE_MATCH_ATOL):
        raise ValueError(
            f"{method} {role} {replicate_id}: recomputed slope ({slope}) does not match "
            f"case_study_conditional_expectation.csv's pinned slope ({pinned_slope}) for this "
            "exact (method, level, replicate_id, role) -- refusing to plot a mismatched value."
        )
    print(
        f"  [{method:<13s} {role:<12s} {replicate_id}] slope recomputed={slope:.6f} == "
        f"pinned case_study_conditional_expectation.csv slope={pinned_slope:.6f} (match); "
        f"intercept={intercept:.6f}; n_cells={n_cells}"
    )

    return {
        "method": method,
        "role": role,
        "replicate_id": replicate_id,
        "truth_masked": truth_masked,
        "estimate_masked": estimate_masked,
        "slope": slope,
        "intercept": intercept,
        "n_cells": n_cells,
    }


def plot_panel(ax, panel: dict, lims: tuple, color: str, title: str, letter: str) -> None:
    """Draw one of the 6 panels: raw (estimate, truth) scatter, one OLS
    regression line (the linear approximation of E[Truth | Estimate]) spanning
    the full axis range, the 1:1 dashed gray reference line, shared axis
    limits + equal aspect, and a bottom-right stat box carrying ONLY the
    slope (MACB intentionally dropped, per 2026-10-16 user decision)."""
    truth = panel["truth_masked"]
    estimate = panel["estimate_masked"]

    ax.scatter(
        estimate, truth, s=7, alpha=0.25, color=color, edgecolors="none",
        label=f"evaluated cells (n={panel['n_cells']})",
    )
    xs = np.array(lims)
    ax.plot(
        xs, panel["intercept"] + panel["slope"] * xs, color=color, linewidth=2.2,
        label="OLS fit ($\\approx$ E[Truth | Estimate])",
    )
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1.3, label="1:1")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Estimate Z* (Porosity %)")
    ax.set_ylabel("Truth Z (Porosity %)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

    ax.text(
        0.97, 0.04, f"slope = {panel['slope']:.3f}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.88, linewidth=0.5),
    )
    ax.text(
        -0.16, 1.12, letter, transform=ax.transAxes, ha="left", va="top",
        fontsize=16, fontweight="bold", clip_on=False,
    )


def main():
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    length_df = pd.read_csv(PROCESSED_DIR / "length_scale_by_replicate.csv", comment="#")
    pinned_df = pd.read_csv(PINNED_CONDITIONAL_EXPECTATION_CSV, dtype={"level": str})

    truth = len_mod.get_base_case_truth(
        truth_seed=len_mod.TRUTH_SEED, hmaj1=len_mod.AXIS_HMAJ1, hmin1=len_mod.AXIS_HMIN1
    )
    print(
        f"Conditional-expectation scatter comparison board, axis_level={AXIS_LEVEL!r}: "
        f"truth_seed={len_mod.TRUTH_SEED}, hmaj1={len_mod.AXIS_HMAJ1:g} m"
    )

    porosity_vmin, porosity_vmax = float(np.min(truth)), float(np.max(truth))
    pad = 0.05 * (porosity_vmax - porosity_vmin)
    lims = (porosity_vmin - pad, porosity_vmax + pad)

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

    print(
        "\nRecomputing OLS slope/intercept per panel (reusing "
        "make_length_case_study_figures.compute_conditional_expectation) and cross-checking "
        "slope against the pinned case_study_conditional_expectation.csv row for the same "
        "(method, level, replicate_id, role) ..."
    )
    panels = {
        "A": recompute_panel("sgs", fr_mod.CONDITIONAL_EXPECTATION_ROLE, sgs_row, pinned_df),
        "B": recompute_panel("gp_mle", "MIN", gp_min_row, pinned_df),
        "C": recompute_panel("rbf_bootstrap", "MIN", rbf_min_row, pinned_df),
        "D": recompute_panel("kriging", fr_mod.CONDITIONAL_EXPECTATION_ROLE, kriging_row, pinned_df),
        "E": recompute_panel("gp_mle", "MAX", gp_max_row, pinned_df),
        "F": recompute_panel("rbf_bootstrap", "MAX", rbf_max_row, pinned_df),
    }

    print(
        "\nContext only (NOT enforced -- see module docstring): recomputed slope vs. the "
        "old, now-lost notebook figure's slope for the same panel."
    )
    for letter in PANEL_LABELS:
        ref = REFERENCE_IMAGE_SLOPES[letter]
        got = panels[letter]["slope"]
        diff = abs(got - ref)
        flag = "" if diff <= REFERENCE_IMAGE_SLOPE_FLAG_ATOL else "  <-- DIFFERS FROM reference image"
        print(f"  panel {letter}: recomputed={got:.3f}  reference_image={ref:.3f}  |diff|={diff:.3f}{flag}")

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
        plot_panel(ax, panels[letter], lims, colors[letter], titles[letter], letter)

    fig.suptitle(
        "Conditional-expectation scatter comparison (1% sample-density level, axis_level='1'): "
        "raw (estimate, truth) scatter per evaluated cell with one OLS regression line per "
        "panel -- a linear approximation of E[Truth | Estimate] -- vs. the 1:1 reference",
        fontsize=12.5, y=0.998,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.95])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n{OUTPUT_FIG.name}: {OUTPUT_FIG} ({OUTPUT_FIG.stat().st_size} bytes)")

    rows = []
    for letter in PANEL_LABELS:
        p = panels[letter]
        rows.append({
            "panel": letter,
            "method": p["method"],
            "role": p["role"],
            "replicate_id": p["replicate_id"],
            "length_or_range_m": length_or_range_m[letter],
            "slope": p["slope"],
            "intercept": p["intercept"],
            "n_cells": p["n_cells"],
        })
    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"{OUTPUT_CSV.name}: {OUTPUT_CSV} ({len(out_df)} rows)")
    print("\nSummary:")
    print(out_df.to_string(index=False))

    return OUTPUT_FIG, OUTPUT_CSV


if __name__ == "__main__":
    main()
