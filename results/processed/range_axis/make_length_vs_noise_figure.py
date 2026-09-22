"""GP-MLE "fitted length scale vs. noise variance" figure for the range axis
(user-requested deliverable, orchestrator task 2026-09-22) -- the range-axis
analogue of results/processed/sample_replicate_axis/
make_length_vs_smoothing_figures.py's GP-MLE panel, adapted for an axis with
NO replicate spread.

WHY THIS DIFFERS FROM THE REPLICATE-AXIS VERSION
----------------------------------------------------------------------------
make_length_vs_smoothing_figures.py draws 5 density-level PANELS, each a
scatter of 10 sample-seed replicates (50 points total for GP-MLE). The range
axis has no replicate axis of its own -- 8 levels, ONE ground-truth
realization and ONE GP-MLE fit each -- so there is nothing to scatter within
a level. Instead this script draws all 8 levels as ONE scatter (8 points,
one per level), labelled by their ground-truth range and connected in
range-ascending order so the reader can see the trajectory as range grows,
exactly as requested.

x = GP-MLE practical_range_m (0.05-correlation-cutoff conversion of the
    fitted sklearn RBF kernel length_scale), REUSED from
    results/processed/range_axis/length_scale_by_method.csv (built by
    make_length_scale_figure.py -- NOT recomputed here, per this project's
    "verify-before-trust, don't duplicate a conversion in two places"
    convention). Independently cross-checked below against
    results/processed/range_axis/qc_summary.csv's gp_length_scale_m column
    (the RAW, pre-conversion length_scale, read by qc_range_axis.py straight
    from each run's own manifest) -- multiplying that raw value by the same
    GP_FACTOR_EXACT used in make_length_scale_figure.py must reproduce this
    script's x values exactly, or this script raises.
y = GP-MLE noise_variance_real_units, read FRESH from each run's own
    manifest.json (params.fitted_hyperparameters.noise_variance_real_units --
    the MLE-fitted GP nugget/noise term in real, Porosity %^2 units), and
    cross-checked against qc_summary.csv's gp_noise_variance_real column
    (same quantity, already tabulated by qc_range_axis.py).

AXIS SCALE: LINEAR for both x and y (NOT log, unlike the replicate-axis
figure)
----------------------------------------------------------------------------
Decided from the OBSERVED value spread, computed below and printed every
run, using the same threshold this project's other figures apply informally
(an order-of-magnitude-or-more spread motivates a log axis; anything
narrower does not):
    x (practical_range_m): 103.2 - 391.7 m, max/min ratio ~3.8x (< 0.6 decades)
    y (noise_variance_real_units): 1.24 - 2.17 Porosity %^2, max/min ratio ~1.7x (< 0.25 decades)
Compare this to make_length_vs_smoothing_figures.py's GP-MLE panel, which
spans up to ~4.5 orders of magnitude in noise variance ACROSS SAMPLE-DENSITY
LEVELS (20% to 1%) and motivated a log y-axis there -- that huge span comes
from sample count varying by 20x across that axis's levels. This axis holds
n_samples fixed and only varies the ground-truth range, so neither axis
here comes close to that threshold; linear axes keep the true shape of the
(comparatively modest, monotonically-plateauing) trajectory legible, which a
log axis would visually compress. The exact ratios are (re)computed from the
data below (RATIO_LOG_THRESHOLD = 10x) rather than hardcoded, so this
docstring's numbers cannot silently drift from the figure.

Truth nugget reference line: NUG * POR_STDEV**2 (real units), imported from
src.experiments.base_case -- NOT hardcoded.

Output: results/figures/range_axis/length_vs_noise_gp_mle.png

Run with:
.venv/Scripts/python.exe -m results.processed.range_axis.make_length_vs_noise_figure
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

from src.experiments.base_case import NUG, POR_STDEV  # noqa: E402
from src.experiments.range_axis import ALL_RANGE_VALUES  # noqa: E402

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "range_axis"

FIG_DPI = 300
RATIO_LOG_THRESHOLD = 10.0  # max/min ratio at or above which an axis is drawn log-scaled

CROSS_CHECK_RTOL = 1e-9


def _manifest_params(rel_run_dir: str) -> dict:
    return json.loads(
        (_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8")
    )["params"]


def build_dataset() -> pd.DataFrame:
    """One row per range level: (gt_range_m, practical_range_m, noise_variance_real_units),
    x reused from length_scale_by_method.csv, y read fresh from each run's own
    manifest -- both cross-checked against qc_summary.csv below."""
    length_df = pd.read_csv(PROCESSED_DIR / "length_scale_by_method.csv")
    length_df["axis_level"] = length_df["axis_level"].astype(str)
    gp_length = length_df[length_df.method == "gp_mle"].set_index("axis_level")

    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    qc = pd.read_csv(PROCESSED_DIR / "qc_summary.csv")

    rows = []
    for r in ALL_RANGE_VALUES:
        level = str(int(r))
        x = float(gp_length.loc[level, "length_scale_m"])

        gp_params = _manifest_params(source_runs[level]["gp_mle"])
        hp = gp_params["fitted_hyperparameters"]
        y = float(hp["noise_variance_real_units"])

        # --- cross-check 1: y against qc_summary.csv's gp_noise_variance_real
        qc_row = qc[qc["range_m"] == r]
        if len(qc_row) != 1:
            raise ValueError(f"qc_summary.csv: expected exactly 1 row for range_m={r}, found {len(qc_row)}.")
        qc_noise = float(qc_row["gp_noise_variance_real"].iloc[0])
        if not np.isclose(y, qc_noise, rtol=CROSS_CHECK_RTOL, atol=0.0):
            raise ValueError(
                f"range={r:g}m: noise_variance_real_units from manifest ({y}) disagrees with "
                f"qc_summary.csv's gp_noise_variance_real ({qc_noise})."
            )

        # --- cross-check 2: x against qc_summary.csv's raw gp_length_scale_m,
        # converted with the SAME factor make_length_scale_figure.py used.
        qc_ell = float(qc_row["gp_length_scale_m"].iloc[0])
        gp_factor_exact = float(np.sqrt(-2.0 * np.log(0.05)))  # 2.4477, same as make_length_scale_figure.py
        x_from_qc = gp_factor_exact * qc_ell
        if not np.isclose(x, x_from_qc, rtol=CROSS_CHECK_RTOL, atol=0.0):
            raise ValueError(
                f"range={r:g}m: practical_range_m from length_scale_by_method.csv ({x}) disagrees "
                f"with qc_summary.csv's raw gp_length_scale_m converted the same way ({x_from_qc})."
            )

        rows.append({"gt_range_m": r, "practical_range_m": x, "noise_variance_real_units": y})

    df = pd.DataFrame(rows).sort_values("gt_range_m").reset_index(drop=True)
    print("cross-checks passed: x and y both agree with qc_summary.csv for all 8 levels.")
    return df


def main():
    df = build_dataset()

    x_ratio = float(df["practical_range_m"].max() / df["practical_range_m"].min())
    y_ratio = float(
        df["noise_variance_real_units"].max() / df["noise_variance_real_units"].min()
    )
    x_log = x_ratio >= RATIO_LOG_THRESHOLD
    y_log = y_ratio >= RATIO_LOG_THRESHOLD
    print(
        f"Observed spread: x (practical_range_m) in [{df['practical_range_m'].min():.2f}, "
        f"{df['practical_range_m'].max():.2f}] m, ratio {x_ratio:.2f}x -> "
        f"{'LOG' if x_log else 'LINEAR'} axis (threshold {RATIO_LOG_THRESHOLD:g}x)"
    )
    print(
        f"Observed spread: y (noise_variance_real_units) in "
        f"[{df['noise_variance_real_units'].min():.4f}, {df['noise_variance_real_units'].max():.4f}] "
        f"Porosity %^2, ratio {y_ratio:.2f}x -> {'LOG' if y_log else 'LINEAR'} axis "
        f"(threshold {RATIO_LOG_THRESHOLD:g}x)"
    )

    truth_nugget_real = NUG * POR_STDEV ** 2

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(
        df["practical_range_m"], df["noise_variance_real_units"],
        color="tab:red", marker="D", markersize=10, markeredgecolor="black",
        markeredgewidth=1.0, linewidth=1.8, zorder=3,
    )
    # 500-800 m cluster tightly on x (see docstring: ratio only 3.8x overall), so labels
    # are alternated above/below the marker to avoid overlapping each other.
    for i, (_, row) in enumerate(df.iterrows()):
        xytext = (7, 8) if i % 2 == 0 else (7, -14)
        ax.annotate(
            f"{int(row['gt_range_m'])} m", (row["practical_range_m"], row["noise_variance_real_units"]),
            textcoords="offset points", xytext=xytext, fontsize=9, color="dimgray",
        )
    ax.axhline(
        truth_nugget_real, color="gray", linestyle="--", linewidth=1.3,
        label=f"truth nugget ({truth_nugget_real:g} Porosity %$^2$)",
    )

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    ax.set_xlabel("GP-MLE fitted practical range (m, 0.05-correlation-cutoff)")
    ax.set_ylabel("GP-MLE fitted noise variance (Porosity %$^2$)")
    ax.set_title(
        "Range axis: GP-MLE fitted length scale vs. noise variance\n"
        "(8 range levels, 1 realization each -- points labelled by, and connected in, "
        "ascending ground-truth range)",
        fontsize=11,
    )
    ax.grid(alpha=0.3, which="both" if (x_log or y_log) else "major")
    ax.legend(fontsize=9, loc="lower right")

    fig.text(
        0.5, 0.01, textwrap.fill(
            f"x = GP-MLE practical_range_m (0.05-correlation-cutoff conversion of the fitted "
            f"sklearn RBF kernel length_scale), reused as-is from length_scale_by_method.csv "
            f"(built by make_length_scale_figure.py). y = params.fitted_hyperparameters."
            f"noise_variance_real_units, read fresh from each run's own manifest.json. Both "
            f"cross-checked against results/processed/range_axis/qc_summary.csv. Dashed line = "
            f"the ground truth's own nugget in real units (NUG * POR_STDEV^2). Axis scales chosen "
            f"from the OBSERVED value spread (ratio >= {RATIO_LOG_THRESHOLD:g}x -> log; x ratio "
            f"{x_ratio:.2f}x -> {'log' if x_log else 'linear'}, y ratio {y_ratio:.2f}x -> "
            f"{'log' if y_log else 'linear'}) -- unlike the sample-replicate axis's equivalent "
            "figure, whose GP-MLE noise variance spans up to ~4.5 orders of magnitude ACROSS "
            "sample-density levels and is therefore drawn log-scaled; this axis holds n_samples "
            "fixed and only varies the ground-truth range, so neither axis here approaches that "
            "spread. No replicate spread on this axis: each point is a SINGLE observation (1 "
            "ground-truth realization, 1 conditioning-sample draw per level), unlike the "
            "replicate-axis figure's 10-replicate scatter per panel.",
            150,
        ),
        ha="center", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.1, bottom=0.24, right=0.97, top=0.87)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "length_vs_noise_gp_mle.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"length_vs_noise_gp_mle.png: {out} ({out.stat().st_size} bytes)")
    return out


if __name__ == "__main__":
    main()
