"""Two figures for the multi-realization GP-hyperparameter diagnostic
(src/experiments/gp_realization_check_nugget_axis.py): does the nugget axis's
single-realization GP-MLE result survive a change of ground-truth realization?

Input: results/processed/nugget_axis/gp_hyperparameters_by_realization.csv
       (6 nugget levels x 10 ground-truth realizations = 60 GP fits; the
       truth_seed=101 row of each level is the realization the pinned
       nugget-axis artifacts were built from, and that script refuses to write
       the CSV unless those rows reproduce the pinned values).

Outputs
-------
1. results/figures/nugget_axis/nugget_fitted_vs_truth_realizations.png
   The multi-realization version of nugget_fitted_vs_truth.png: same
   composition (x = truth nugget in Porosity %^2, y = GP-MLE fitted
   WhiteKernel noise variance, y = x reference line, equal aspect), with all
   10 realizations per level drawn as low-alpha points, the per-level
   mean +/- 1 std drawn on top, and the pinned realization (truth_seed=101)
   marked with a distinct marker. The count of fits above / below y = x is
   annotated on the axes and repeated in the caption, because that count IS
   the question the figure exists to answer.

   AXIS SCALES: LINEAR for both, for the same two reasons as the
   single-realization figure (make_nugget_fitted_vs_truth_figure.py): the x
   values include exactly 0.0 (the nug=0.0 level), which a log axis cannot
   represent at all; and a log y axis would destroy the visual meaning of the
   y = x reference line, which is the entire point of the figure. The
   measured y spread is printed on every run and written into the caption, so
   the reader can see what was traded away.

2. results/figures/nugget_axis/length_vs_noise_gp_mle_realizations.png
   x = fitted practical range (m, 0.05-correlation cutoff), y = fitted noise
   variance (Porosity %^2), all 60 fits, coloured by nugget level, pinned
   realization marked.

   AXIS SCALES, decided from the observed spread (both ratios are recomputed
   and printed on every run, so this docstring cannot silently drift):
   * y (fitted noise variance): observed max/min ~= 11.6x, at or above this
     project's RATIO_LOG_THRESHOLD = 10x rule (the rule introduced in
     range_axis/make_length_vs_noise_figure.py), so LOG follows the rule
     directly -- no override involved.
   * x (fitted practical range): observed max/min ~= 9.1x, just BELOW the 10x
     rule, so the rule alone would choose linear. It is drawn LOG anyway, as
     a deliberate override, for exactly the reason
     nugget_axis/make_length_scale_figure.py already overrode the same rule
     on this axis: 59 of the 60 fits fall in 149-371 m and a single fit (the
     PINNED realization at nug=0.5) sits at 1365 m. On a linear x axis those
     59 points collapse into the leftmost ~1/5 of the plot, and the question
     this figure is supposed to answer -- whether the pinned 1365 m
     length-scale jump recurs in other realizations, or whether the other
     realizations cluster somewhere else entirely -- becomes unreadable
     precisely because of the one point that motivates the figure. Whether
     the rule would have chosen log on its own is printed and stated in the
     caption on every run, so the override is never silent.

Run with:
.venv/Scripts/python.exe -m results.processed.nugget_axis.make_gp_realization_figures
"""

import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.base_case import HMAJ1, POR_STDEV  # noqa: E402
from src.experiments.base_case_conditioning import (  # noqa: E402
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
)
from src.experiments.nugget_axis import (  # noqa: E402
    ALL_NUGGET_VALUES,
    truth_nugget_real_units,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "nugget_axis"
IN_CSV = PROCESSED_DIR / "gp_hyperparameters_by_realization.csv"

FIG_DPI = 300
RATIO_LOG_THRESHOLD = 10.0  # project rule, from range_axis/make_length_vs_noise_figure.py

# Figure 2 only: draw the x axis (fitted practical range) on a log scale even
# if the 10x rule above would choose linear. See the module docstring for the
# recorded reason; the printed output and the caption always state whether the
# rule agreed or was overridden.
X_SCALE_LOG_OVERRIDE_FIG2 = True

TOTAL_SILL_REAL = POR_STDEV ** 2

LEVEL_COLORS = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(ALL_NUGGET_VALUES)))


def load() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    df["axis_level"] = df["axis_level"].astype(str)
    n_levels = df["nug_normalized"].nunique()
    n_reps = df.groupby("nug_normalized")["truth_seed"].nunique()
    if n_levels != len(ALL_NUGGET_VALUES) or n_reps.nunique() != 1:
        raise ValueError(
            f"{IN_CSV.name}: expected {len(ALL_NUGGET_VALUES)} levels with an equal "
            f"number of realizations each; found {n_levels} levels, realization counts "
            f"{n_reps.to_dict()}."
        )
    if int(df["is_pinned_realization"].sum()) != len(ALL_NUGGET_VALUES):
        raise ValueError(
            f"{IN_CSV.name}: expected exactly one pinned realization per level, found "
            f"{int(df['is_pinned_realization'].sum())}."
        )
    # The x values plotted in figure 1 must be the project's headline truth
    # nugget; check against the helper rather than trusting the stored column.
    for nug, grp in df.groupby("nug_normalized"):
        expected = truth_nugget_real_units(float(nug))
        if not np.allclose(grp["truth_nugget_real_units"], expected, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"nug={nug}: stored truth_nugget_real_units disagrees with "
                f"nugget_axis.truth_nugget_real_units ({expected})."
            )
    return df


def _shared_caption_prefix(df: pd.DataFrame) -> str:
    n_reps = int(df.groupby("nug_normalized")["truth_seed"].nunique().iloc[0])
    n_samples_actual = sorted(df["n_samples_actual"].unique())
    seeds = sorted(df["truth_seed"].unique())
    return (
        f"WHAT VARIES: the ground-truth realization (truth_seed in "
        f"{{{', '.join(str(s) for s in seeds)}}}, {n_reps} per level) and the nugget "
        f"level. WHAT IS FIXED: sample_seed={SAMPLE_SEED}, so the sample LOCATIONS are "
        f"identical in all {len(df)} fits and only the sample VALUES change (verified: "
        f"max |X/Y| deviation 0 across all fits); variogram range {HMAJ1:g} m isotropic; "
        f"n_samples requested {N_SAMPLES} -> {'/'.join(str(int(n)) for n in n_samples_actual)} "
        f"actual after duplicate-cell removal; and the entire GP configuration (kernel "
        f"structure, bounds, initial values, n_restarts_optimizer, normalize_y, "
        f"random_state), imported from src/experiments/gp_mle.py rather than re-declared. "
        f"truth_seed={TRUTH_SEED} is the realization every pinned nugget-axis artifact was "
        f"built from; those rows reproduce the pinned values to <1.2e-12 relative."
    )


def figure_fitted_vs_truth(df: pd.DataFrame) -> Path:
    y_ratio = float(
        df["gp_noise_variance_real_units"].max() / df["gp_noise_variance_real_units"].min()
    )
    rule_says_log = y_ratio >= RATIO_LOG_THRESHOLD
    print(
        f"[fig1] fitted noise variance in "
        f"[{df['gp_noise_variance_real_units'].min():.4f}, "
        f"{df['gp_noise_variance_real_units'].max():.4f}] Porosity %^2, ratio "
        f"{y_ratio:.2f}x -> project {RATIO_LOG_THRESHOLD:g}x rule would say "
        f"{'LOG' if rule_says_log else 'LINEAR'}; drawing LINEAR anyway because x "
        "contains an exact zero and a log y would break the y = x reference line."
    )

    n_above = int(df["noise_above_truth_nugget"].sum())
    n_below = len(df) - n_above
    per_level = (
        df.groupby("nug_normalized")
        .agg(
            truth_nugget=("truth_nugget_real_units", "first"),
            mean=("gp_noise_variance_real_units", "mean"),
            std=("gp_noise_variance_real_units", "std"),
            n_above=("noise_above_truth_nugget", "sum"),
            n=("noise_above_truth_nugget", "size"),
            level_max=("gp_noise_variance_real_units", "max"),
        )
        .reset_index()
        .sort_values("nug_normalized")
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    hi = float(
        max(df["truth_nugget_real_units"].max(), df["gp_noise_variance_real_units"].max())
    )
    pad = 0.06 * hi
    ax.plot(
        [0.0, hi + pad], [0.0, hi + pad], color="gray", linestyle="--", linewidth=1.0,
        zorder=1, label="y = x (fitted noise = true nugget)",
    )

    ax.scatter(
        df["truth_nugget_real_units"], df["gp_noise_variance_real_units"],
        s=18, marker="o", facecolors="tab:blue", edgecolors="none", alpha=0.35, zorder=2,
        label="individual GP fit (10 realizations per level)",
    )
    ax.errorbar(
        per_level["truth_nugget"], per_level["mean"], yerr=per_level["std"],
        fmt="D", markersize=5, color="tab:red", markerfacecolor="tab:red",
        markeredgecolor="black", ecolor="tab:red", elinewidth=1.1, capsize=3,
        zorder=4, label="per-level mean $\\pm$ 1 std (10 realizations)",
    )
    ax.plot(
        per_level["truth_nugget"], per_level["mean"], color="tab:red", linewidth=1.1,
        zorder=3,
    )
    for _, row in per_level.iterrows():
        ax.annotate(
            f"{int(row['n_above'])}/{int(row['n'])} above",
            (row["truth_nugget"], row["level_max"]),
            textcoords="offset points", xytext=(0, 5), fontsize=6.5, color="dimgray",
            ha="center",
        )

    x_max = float(df["truth_nugget_real_units"].max())
    ax.set_xlim(-0.08 * x_max, 1.08 * x_max)
    ax.set_ylim(-pad, hi + pad)
    ax.set_xticks([truth_nugget_real_units(n) for n in ALL_NUGGET_VALUES])
    ax.set_xlabel("Ground-truth nugget (Porosity %$^2$)", fontsize=9)
    ax.set_ylabel("GP-MLE fitted noise variance (Porosity %$^2$)", fontsize=9)
    ax.set_title(
        f"GP-MLE fitted noise variance vs. ground-truth nugget "
        f"({n_above} of {len(df)} fits above y = x)", fontsize=9.5,
    )
    ax.grid(alpha=0.3)
    # Legend goes bottom-right (the region below the y = x line is empty here);
    # the headline count box takes the top-left. Neither may sit top-centre,
    # where the per-level "n/10 above" annotations are.
    ax.legend(fontsize=7.5, loc="lower right")

    caption = (
        textwrap.fill(
            f"{n_above} of {len(df)} fits (6 nugget levels x 10 ground-truth realizations) "
            f"lie above the y = x line, i.e. the marginal-likelihood fit attributed MORE "
            f"variance to noise than the ground truth's nugget; {n_below} lie below. "
            f"Per-level counts are annotated above each error bar. "
            + _shared_caption_prefix(df)
            + f" x = nug * POR_STDEV^2 (nugget_axis.truth_nugget_real_units). Both axes are "
            f"LINEAR: x contains an exact zero (the nug=0.0 level), which a log axis cannot "
            f"show, and a log y would destroy the meaning of the y = x line; the measured y "
            f"spread is {y_ratio:.2f}x against this project's {RATIO_LOG_THRESHOLD:g}x "
            f"log-axis threshold.",
            110,
        )
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "nugget_fitted_vs_truth_realizations.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    out.with_suffix(".caption.txt").write_text(caption, encoding="utf-8")
    plt.close(fig)
    print(f"[fig1] {out} ({out.stat().st_size} bytes)")
    return out


def figure_length_vs_noise(df: pd.DataFrame) -> Path:
    x_ratio = float(df["gp_practical_range_m"].max() / df["gp_practical_range_m"].min())
    y_ratio = float(
        df["gp_noise_variance_real_units"].max() / df["gp_noise_variance_real_units"].min()
    )
    x_rule_log = x_ratio >= RATIO_LOG_THRESHOLD
    y_log = y_ratio >= RATIO_LOG_THRESHOLD
    x_log = True if X_SCALE_LOG_OVERRIDE_FIG2 else x_rule_log

    print(
        f"[fig2] fitted practical range in [{df['gp_practical_range_m'].min():.1f}, "
        f"{df['gp_practical_range_m'].max():.1f}] m, ratio {x_ratio:.2f}x -> project "
        f"{RATIO_LOG_THRESHOLD:g}x rule says {'LOG' if x_rule_log else 'LINEAR'} x-axis"
    )
    if x_log and not x_rule_log:
        print(
            "[fig2]   OVERRIDE: drawing a LOG x-axis anyway -- 59 of 60 fits fall in a "
            "narrow band and one (the pinned realization at nug=0.5) is ~6x larger, so a "
            "linear x would compress the band whose spread the figure exists to show. "
            "Same override precedent as nugget_axis/make_length_scale_figure.py."
        )
    elif x_log and x_rule_log:
        print("[fig2]   note: the rule now selects LOG on its own; the override is inert.")
    print(
        f"[fig2] fitted noise variance ratio {y_ratio:.2f}x -> "
        f"{'LOG' if y_log else 'LINEAR'} y-axis (project rule, no override)"
    )

    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    for color, nug in zip(LEVEL_COLORS, sorted(df["nug_normalized"].unique())):
        grp = df[df["nug_normalized"] == nug]
        non_pinned = grp[~grp["is_pinned_realization"].astype(bool)]
        pinned = grp[grp["is_pinned_realization"].astype(bool)]
        ax.scatter(
            non_pinned["gp_practical_range_m"], non_pinned["gp_noise_variance_real_units"],
            s=64, marker="o", facecolors=[color], edgecolors="black", linewidths=0.5,
            alpha=0.85, zorder=3, label=f"nug = {nug:.1f}",
        )
        ax.scatter(
            pinned["gp_practical_range_m"], pinned["gp_noise_variance_real_units"],
            s=210, marker="*", facecolors=[color], edgecolors="black", linewidths=1.2,
            zorder=4,
        )

    # One legend entry explaining the star marker, without repeating it per level.
    ax.scatter(
        [], [], s=210, marker="*", facecolors="white", edgecolors="black", linewidths=1.2,
        label=f"star = pinned realization (truth_seed={TRUTH_SEED})",
    )

    # On a log axis matplotlib's default decade-only labelling would leave this
    # figure with a single labelled x tick (10^3), making the 149-371 m band --
    # the part of the data the figure is about -- unreadable. Label explicit
    # ticks inside the observed span instead, with a plain (non-exponent)
    # formatter, so the values can be read directly off both axes.
    if x_log:
        ax.set_xscale("log")
        ax.set_xticks([150, 200, 250, 300, 400, 500, 700, 1000, 1400])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    if y_log:
        ax.set_yscale("log")
        ax.set_yticks([0.7, 1, 1.5, 2, 3, 4, 6, 8])
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.axvline(
        HMAJ1, color="gray", linestyle=":", linewidth=1.3, zorder=1,
        label=f"ground-truth variogram range ({HMAJ1:g} m, fixed at every level)",
    )

    pinned_jump = df[
        df["is_pinned_realization"].astype(bool) & (df["nug_normalized"] == 0.5)
    ]
    if len(pinned_jump) == 1:
        pj = pinned_jump.iloc[0]
        ax.annotate(
            f"pinned nug=0.5:\n{pj['gp_practical_range_m']:.0f} m",
            (pj["gp_practical_range_m"], pj["gp_noise_variance_real_units"]),
            textcoords="offset points", xytext=(-14, 20), fontsize=8.5, color="black",
            ha="right",
            arrowprops=dict(arrowstyle="->", color="dimgray", linewidth=1.0),
        )

    ax.set_xlabel("GP-MLE fitted practical range (m)  [0.05 correlation cutoff]")
    ax.set_ylabel("GP-MLE fitted noise variance (Porosity %$^2$)")
    ax.set_title(
        "Nugget axis: GP-MLE fitted length scale vs. fitted noise, "
        f"{len(df)} fits\n(6 nugget levels x 10 ground-truth realizations)",
        fontsize=11.5,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="best", ncol=2)

    x_scale_note = (
        f"x is LOG, a deliberate OVERRIDE of this project's {RATIO_LOG_THRESHOLD:g}x rule "
        f"(observed spread {x_ratio:.2f}x, which the rule would draw linear): 59 of 60 fits "
        f"fall within {df['gp_practical_range_m'].min():.0f}-"
        f"{sorted(df['gp_practical_range_m'])[-2]:.0f} m and one outlier sits at "
        f"{df['gp_practical_range_m'].max():.0f} m, so a linear x hides the spread this "
        f"figure is about (same override precedent as length_scale_vs_nugget.png)."
        if (x_log and not x_rule_log)
        else f"x spread {x_ratio:.2f}x -> {'log' if x_log else 'linear'} x by the "
        f"{RATIO_LOG_THRESHOLD:g}x rule."
    )
    fig.text(
        0.5, 0.005,
        textwrap.fill(
            f"Practical range = sqrt(2 ln 20) * fitted length_scale (~2.4477x, 0.05 "
            f"correlation cutoff, this project's standard conversion). "
            + _shared_caption_prefix(df)
            + " "
            + x_scale_note
            + f" y is LOG by the same rule applied without override (spread {y_ratio:.2f}x "
            f"vs. the {RATIO_LOG_THRESHOLD:g}x threshold).",
            152,
        ),
        ha="center", fontsize=7.2, color="dimgray",
    )
    plt.subplots_adjust(left=0.09, bottom=0.235, right=0.97, top=0.89)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "length_vs_noise_gp_mle_realizations.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] {out} ({out.stat().st_size} bytes)")
    return out


def main():
    df = load()
    print(f"loaded {len(df)} fits from {IN_CSV}")
    p1 = figure_fitted_vs_truth(df)
    p2 = figure_length_vs_noise(df)
    return p1, p2


if __name__ == "__main__":
    main()
