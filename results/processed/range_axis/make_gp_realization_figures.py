"""Two figures for the multi-realization GP-hyperparameter diagnostic
(src/experiments/gp_realization_check_range_axis.py): does the range axis's
single-realization GP-MLE result survive a change of ground-truth realization?

Input: results/processed/range_axis/gp_hyperparameters_by_realization.csv
       (8 range levels x 10 ground-truth realizations = 80 GP fits; the
       truth_seed=101 row of each level is the realization every pinned
       range-axis artifact was built from, and that script refuses to write
       the CSV unless those rows reproduce the pinned values -- 136
       comparisons, max relative difference 1.7e-16).

Outputs
-------
1. results/figures/range_axis/length_scale_vs_range_realizations.png
   The multi-realization version of 00_length_scale_vs_range.png's GP-MLE
   series: x = ground-truth variogram range, y = GP-MLE fitted practical
   range, with the y = x reference line the saturation claim is read against.
   All 10 realizations per level are drawn as low-alpha points, the per-level
   mean +/- 1 std on top, and the pinned realization (truth_seed=101) with a
   distinct marker. The count of fits BELOW y = x is annotated per level and
   repeated in the caption, because that count IS the question the figure
   exists to answer ("the fitted range fails to track the ground-truth
   range").

2. results/figures/range_axis/length_vs_noise_gp_mle_realizations.png
   The multi-realization version of length_vs_noise_gp_mle.png: x = fitted
   practical range, y = fitted noise variance, all 80 fits coloured by their
   ground-truth range level (continuous colormap + colorbar), pinned
   realization marked, and the per-level MEAN trajectory drawn as a connected
   line so it can be compared directly against the pinned single-realization
   trajectory (which is also drawn, through the pinned markers).

AXIS SCALES
-----------
Both figures use LINEAR axes on both axes. This follows this project's
RATIO_LOG_THRESHOLD = 10x rule (introduced in
range_axis/make_length_vs_noise_figure.py: an axis is drawn log-scaled when
the observed max/min ratio reaches 10x) applied WITHOUT override -- the
observed spreads are ~5.5x for fitted practical range and ~7.0x for fitted
noise variance, both below the threshold. Every ratio is recomputed and
printed on every run, and written into the caption, so this docstring cannot
silently drift from what was drawn; if a future re-run pushes a ratio past
10x the code (not this text) switches that axis to log and says so in the
caption. Figure 1 additionally could not use a log y even if the rule asked
for one without breaking the y = x reference line, which is the entire point
of that panel -- that case is handled explicitly in the code.

Run with:
.venv/Scripts/python.exe -m results.processed.range_axis.make_gp_realization_figures
"""

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
from src.experiments.base_case_conditioning import (  # noqa: E402
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
)
from src.experiments.range_axis import ALL_RANGE_VALUES  # noqa: E402

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "range_axis"
IN_CSV = PROCESSED_DIR / "gp_hyperparameters_by_realization.csv"

FIG_DPI = 300
RATIO_LOG_THRESHOLD = 10.0  # project rule, from range_axis/make_length_vs_noise_figure.py

TRUTH_NUGGET_REAL = NUG * POR_STDEV ** 2  # 0.45 Porosity %^2, fixed on this axis

LEVEL_CMAP = plt.get_cmap("viridis")


def load() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    df["axis_level"] = df["axis_level"].astype(str)

    expected_levels = {str(int(r)) for r in ALL_RANGE_VALUES}
    if set(df["axis_level"]) != expected_levels:
        raise ValueError(
            f"{IN_CSV.name}: axis levels {sorted(set(df['axis_level']))} do not match "
            f"range_axis.ALL_RANGE_VALUES {sorted(expected_levels)}."
        )
    n_reps = df.groupby("range_m")["truth_seed"].nunique()
    if n_reps.nunique() != 1:
        raise ValueError(
            f"{IN_CSV.name}: levels do not all carry the same number of realizations "
            f"({n_reps.to_dict()})."
        )
    if int(df["is_pinned_realization"].sum()) != len(ALL_RANGE_VALUES):
        raise ValueError(
            f"{IN_CSV.name}: expected exactly one pinned realization per level, found "
            f"{int(df['is_pinned_realization'].sum())}."
        )
    # The x values of figure 1 must literally be the axis levels; check the
    # stored truth_range_m rather than trusting it.
    if not np.allclose(df["truth_range_m"], df["range_m"], rtol=0.0, atol=0.0):
        raise ValueError(f"{IN_CSV.name}: truth_range_m disagrees with range_m.")
    # The nugget is the FIXED factor on this axis -- verify it never moved.
    if not np.allclose(df["truth_nugget_real_units"], TRUTH_NUGGET_REAL, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"{IN_CSV.name}: truth_nugget_real_units is not constant at "
            f"{TRUTH_NUGGET_REAL} -- the range axis must hold the nugget fixed."
        )
    return df


def _scale_decision(values: pd.Series, name: str, tag: str):
    """Apply the project's 10x rule to one quantity, print the decision, and
    return (use_log, ratio) so the caption can state it."""
    ratio = float(values.max() / values.min())
    use_log = ratio >= RATIO_LOG_THRESHOLD
    print(
        f"[{tag}] {name} in [{values.min():.4g}, {values.max():.4g}], ratio {ratio:.2f}x "
        f"-> project {RATIO_LOG_THRESHOLD:g}x rule says {'LOG' if use_log else 'LINEAR'} "
        "(no override)"
    )
    return use_log, ratio


def _per_level(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df.groupby("range_m")
        .agg(
            mean=(col, "mean"),
            std=(col, "std"),
            min=(col, "min"),
            max=(col, "max"),
            n_below=("fitted_range_below_truth_range", "sum"),
            n=("fitted_range_below_truth_range", "size"),
        )
        .reset_index()
        .sort_values("range_m")
    )


def _shared_caption_prefix(df: pd.DataFrame) -> str:
    n_reps = int(df.groupby("range_m")["truth_seed"].nunique().iloc[0])
    n_samples_actual = sorted(df["n_samples_actual"].unique())
    seeds = sorted(df["truth_seed"].unique())
    return (
        f"WHAT VARIES: the ground-truth realization (truth_seed in "
        f"{{{', '.join(str(s) for s in seeds)}}}, {n_reps} per level) and the ground-truth "
        f"variogram range ({int(min(ALL_RANGE_VALUES))}-{int(max(ALL_RANGE_VALUES))} m, "
        f"isotropic). WHAT IS FIXED: sample_seed={SAMPLE_SEED}, so the sample LOCATIONS are "
        f"identical in all {len(df)} fits and only the sample VALUES change (verified: max "
        f"|X/Y| deviation 0 across all fits); the ground-truth nugget at the base-case value "
        f"{NUG:g} normalized = {TRUTH_NUGGET_REAL:g} Porosity %^2 at every level; n_samples "
        f"requested {N_SAMPLES} -> {'/'.join(str(int(n)) for n in n_samples_actual)} actual "
        f"after duplicate-cell removal; and the entire GP configuration (kernel structure, "
        f"bounds, initial values, n_restarts_optimizer, normalize_y, random_state), imported "
        f"from src/experiments/gp_mle.py rather than re-declared. truth_seed={TRUTH_SEED} is "
        f"the realization every pinned range-axis artifact was built from; those rows "
        f"reproduce the pinned values to <2e-16 relative."
    )


def figure_length_scale_vs_range(df: pd.DataFrame) -> Path:
    x_log, x_ratio = _scale_decision(df["truth_range_m"], "ground-truth range (m)", "fig1")
    y_rule_log, y_ratio = _scale_decision(
        df["gp_practical_range_m"], "fitted practical range (m)", "fig1"
    )
    # A log y would destroy the visual meaning of the y = x line, which is the
    # whole point of this panel, so it is only honoured if x is log too.
    y_log = y_rule_log and x_log
    if y_rule_log and not y_log:
        print(
            "[fig1]   NOTE: the rule asked for a LOG y but x is LINEAR; drawing y LINEAR "
            "so the y = x reference line stays a straight 45-degree line."
        )

    per_level = _per_level(df, "gp_practical_range_m")
    n_below = int(df["fitted_range_below_truth_range"].sum())

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    lo = 0.0
    hi = float(max(df["truth_range_m"].max(), df["gp_practical_range_m"].max()))
    pad = 0.06 * hi
    ax.plot(
        [lo, hi + pad], [lo, hi + pad], color="gray", linestyle="--", linewidth=1.0,
        zorder=1, label="y = x (fitted practical range = ground-truth range)",
    )

    ax.scatter(
        df["truth_range_m"], df["gp_practical_range_m"],
        s=18, marker="o", facecolors="tab:blue", edgecolors="none", alpha=0.35, zorder=2,
        label="individual GP fit (10 realizations per level)",
    )
    ax.errorbar(
        per_level["range_m"], per_level["mean"], yerr=per_level["std"],
        fmt="D", markersize=5, color="tab:red", markerfacecolor="tab:red",
        markeredgecolor="black", ecolor="tab:red", elinewidth=1.1, capsize=3,
        zorder=4, label="per-level mean $\\pm$ 1 std (10 realizations)",
    )
    ax.plot(per_level["range_m"], per_level["mean"], color="tab:red", linewidth=1.1, zorder=3)

    for _, row in per_level.iterrows():
        ax.annotate(
            f"{int(row['n_below'])}/{int(row['n'])}\nbelow",
            (row["range_m"], row["max"]),
            textcoords="offset points", xytext=(0, 5), fontsize=6.5, color="dimgray",
            ha="center",
        )

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if not x_log and not y_log:
        ax.set_xlim(lo, hi + pad)
        ax.set_ylim(lo, hi + pad)
    ax.set_xticks([float(r) for r in ALL_RANGE_VALUES])
    ax.set_xlabel("Ground-truth variogram range (m, isotropic)", fontsize=9)
    ax.set_ylabel("GP-MLE fitted practical range (m)", fontsize=9)
    ax.set_title(
        f"GP-MLE fitted range vs. ground-truth range "
        f"({n_below} of {len(df)} fits below y = x)", fontsize=9.5,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")

    per_level_txt = ", ".join(
        f"{int(r['range_m'])} m: {int(r['n_below'])}/{int(r['n'])}"
        for _, r in per_level.iterrows()
    )
    caption = (
        textwrap.fill(
            f"{n_below} of {len(df)} fits lie below the y = x line, i.e. the "
            f"marginal-likelihood fit learned a correlation length SHORTER than the ground "
            f"truth's. Per level ({per_level_txt}). Practical range = sqrt(2 ln 20) * fitted "
            f"length_scale (~2.4477x, 0.05 correlation cutoff, this project's standard "
            f"conversion, reused from gp_hyperparameter_scale_conversion.csv and verified "
            f"equal to it). " + _shared_caption_prefix(df) + f" Axis scales follow this "
            f"project's {RATIO_LOG_THRESHOLD:g}x rule with no override: x spread "
            f"{x_ratio:.2f}x -> {'log' if x_log else 'linear'}, y spread {y_ratio:.2f}x -> "
            f"{'log' if y_log else 'linear'}.",
            110,
        )
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "length_scale_vs_range_realizations.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    out.with_suffix(".caption.txt").write_text(caption, encoding="utf-8")
    plt.close(fig)
    print(f"[fig1] {out} ({out.stat().st_size} bytes)")
    return out


def figure_length_vs_noise(df: pd.DataFrame) -> Path:
    x_log, x_ratio = _scale_decision(
        df["gp_practical_range_m"], "fitted practical range (m)", "fig2"
    )
    y_log, y_ratio = _scale_decision(
        df["gp_noise_variance_real_units"], "fitted noise variance (Porosity %^2)", "fig2"
    )

    level_mean = (
        df.groupby("range_m")
        .agg(
            practical_range=("gp_practical_range_m", "mean"),
            noise=("gp_noise_variance_real_units", "mean"),
        )
        .reset_index()
        .sort_values("range_m")
    )
    pinned = df[df["is_pinned_realization"].astype(bool)].sort_values("range_m")

    fig, ax = plt.subplots(figsize=(10.0, 7.2))

    norm = plt.Normalize(vmin=min(ALL_RANGE_VALUES), vmax=max(ALL_RANGE_VALUES))
    non_pinned = df[~df["is_pinned_realization"].astype(bool)]
    sc = ax.scatter(
        non_pinned["gp_practical_range_m"], non_pinned["gp_noise_variance_real_units"],
        c=non_pinned["truth_range_m"], cmap=LEVEL_CMAP, norm=norm,
        s=62, marker="o", edgecolors="black", linewidths=0.4, alpha=0.9, zorder=3,
    )
    ax.scatter(
        pinned["gp_practical_range_m"], pinned["gp_noise_variance_real_units"],
        c=pinned["truth_range_m"], cmap=LEVEL_CMAP, norm=norm,
        s=230, marker="*", edgecolors="black", linewidths=1.2, zorder=5,
    )

    # The two trajectories the figure exists to compare.
    ax.plot(
        pinned["gp_practical_range_m"], pinned["gp_noise_variance_real_units"],
        color="black", linewidth=1.6, linestyle="--", zorder=4,
        label=f"pinned realization trajectory (truth_seed={TRUTH_SEED}, stars)",
    )
    ax.plot(
        level_mean["practical_range"], level_mean["noise"],
        color="tab:red", linewidth=2.0, marker="D", markersize=8,
        markeredgecolor="black", markeredgewidth=0.8, zorder=6,
        label="per-level MEAN over 10 realizations",
    )

    ax.axhline(
        TRUTH_NUGGET_REAL, color="gray", linestyle=":", linewidth=1.3, zorder=1,
        label=f"ground-truth nugget ({TRUTH_NUGGET_REAL:g} Porosity %$^2$, fixed at every level)",
    )

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    else:
        # Headroom for the legend: without it the top-right legend box sits on
        # top of the highest-noise fit (the 100 m level's largest), which is
        # one of the points the figure is about.
        ax.set_ylim(top=float(df["gp_noise_variance_real_units"].max()) * 1.25)

    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Ground-truth variogram range (m)")
    cbar.set_ticks([float(r) for r in ALL_RANGE_VALUES])

    ax.set_xlabel("GP-MLE fitted practical range (m)  [0.05 correlation cutoff]")
    ax.set_ylabel("GP-MLE fitted noise variance (Porosity %$^2$)")
    ax.set_title(
        f"Range axis: GP-MLE fitted length scale vs. fitted noise, {len(df)} fits\n"
        f"({len(ALL_RANGE_VALUES)} range levels x 10 ground-truth realizations; "
        "colour = ground-truth range)",
        fontsize=11.5,
    )
    ax.grid(alpha=0.3, which="both" if (x_log or y_log) else "major")
    ax.legend(fontsize=8.5, loc="upper right")

    pinned_lo = float(pinned["gp_noise_variance_real_units"].min())
    pinned_hi_level = int(pinned.loc[pinned["gp_noise_variance_real_units"].idxmin(), "range_m"])
    fig.text(
        0.5, 0.005,
        textwrap.fill(
            f"The single-realization figure (length_vs_noise_gp_mle.png) shows the pinned "
            f"trajectory falling to its minimum ({pinned_lo:.3f} Porosity %^2 at the "
            f"{pinned_hi_level} m level) and then rebounding; that pinned trajectory is the "
            f"dashed black line through the stars here, and the red line is the per-level "
            f"mean over all 10 realizations. "
            + _shared_caption_prefix(df)
            + f" Practical range = sqrt(2 ln 20) * fitted length_scale (~2.4477x, 0.05 "
            f"correlation cutoff). Axis scales follow this project's "
            f"{RATIO_LOG_THRESHOLD:g}x rule with no override: x spread {x_ratio:.2f}x -> "
            f"{'log' if x_log else 'linear'}, y spread {y_ratio:.2f}x -> "
            f"{'log' if y_log else 'linear'}.",
            150,
        ),
        ha="center", fontsize=7.2, color="dimgray",
    )
    plt.subplots_adjust(left=0.08, bottom=0.235, right=0.99, top=0.885)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "length_vs_noise_gp_mle_realizations.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] {out} ({out.stat().st_size} bytes)")
    return out


def main():
    df = load()
    print(f"loaded {len(df)} fits from {IN_CSV}")
    p1 = figure_length_scale_vs_range(df)
    p2 = figure_length_vs_noise(df)
    return p1, p2


if __name__ == "__main__":
    main()
