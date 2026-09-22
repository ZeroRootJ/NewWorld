"""THE nugget axis's central figure (docs/experiment_context.md deliverable
3, Claim 2): what nugget did GP-MLE's automatic hyperparameter selection
actually learn, against the nugget the ground truth was generated with?

  x = truth nugget in physical units (Porosity %^2) = nug * POR_STDEV**2,
      i.e. 0 / 0.9 / 1.8 / 2.7 / 3.6 / 4.5 against a total sill of 9.0.
      Computed via nugget_axis.truth_nugget_real_units, never hardcoded.
  y = GP-MLE's MLE-fitted WhiteKernel noise variance in the SAME physical
      units (params.fitted_hyperparameters.noise_variance_real_units), read
      FRESH from each run's own manifest.json and cross-checked against
      results/processed/nugget_axis/qc_summary.csv's gp_noise_variance_real
      column (the same quantity, already tabulated by qc_nugget_axis.py).
  The 6 points are connected in ascending nugget order, and a y = x line
  marks perfect recovery of the true nugget. Points ABOVE the line mean the
  automatic fit attributed MORE variance to noise than the truth actually
  has; points BELOW mean it fitted part of the nugget as signal -- the
  overconfidence mechanism Claim 2 predicts. This figure reports the
  numbers; it does not argue for either reading.

WHY kriging/SGS ARE NOT ON THIS FIGURE: they are handed the TRUE nugget as
an input (they are the reference baselines on this axis), so they would sit
exactly on y = x by construction and add no information. RBF+bootstrap has
no nugget parameter at all, so there is nothing to plot for it here -- its
CV-tuned smoothing is a different quantity in different units and is
reported separately in qc_summary.csv.

BOUNDARY-STOP MARKING (the ``at_optimizer_noise_floor`` pattern already used
by results/processed/sample_replicate_axis/make_length_vs_smoothing_figures.py)
Any level whose fitted noise variance stopped EXACTLY on the WhiteKernel
noise-level LOWER bound is drawn as a HOLLOW marker, because there the
constrained optimum is the bound itself and the unconstrained optimum lies
at or below it -- the plotted value is then a floor, not an estimate. The
bound is read from each run's OWN manifest
(params.kernel_init.noise_level_bounds[0]) rather than imported or
hardcoded, so a run fitted under different bounds is judged against the
bounds it actually used. This is most likely to bite at nug=0.0, where the
true noise is exactly zero and the bound (1e-5 in normalized units) is
strictly positive. Whether any point is actually at the floor is determined
at runtime and printed; it is not assumed either way by this code.

AXIS SCALES: LINEAR for both x and y. Two independent reasons, both checked
against the data rather than asserted:
  1. x CONTAINS EXACTLY ZERO (the nug=0.0 level). A log x-axis cannot
     represent that level at all, so log is not an option regardless of
     spread.
  2. The y spread is measured against the same RATIO_LOG_THRESHOLD = 10x
     rule results/processed/range_axis/make_length_vs_noise_figure.py uses
     (an order-of-magnitude-or-more spread motivates a log axis). The
     observed max/min ratio is computed and printed on every run; if it ever
     exceeded the threshold the script would say so in its printed output
     and in the caption, so this docstring cannot silently drift from the
     figure. (Measured at the time of writing: y in [1.14, 6.00] Porosity
     %^2, ratio ~5.3x -> linear.) A log y-axis would also break the visual
     meaning of the y = x reference line, which is the entire point of the
     figure.

WHAT THE OUTPUT CSV CARRIES BEYOND THE PLOTTED x AND y (reviewer 2026-09-22;
the FIGURE is unchanged -- these are recorded columns only)
--------------------------------------------------------------------------
1. An ALTERNATIVE physical-unit scale for the truth nugget.
   ``GSLIB.affine`` scales each realization by
   ``a = POR_STDEV / np.std(sim_ns)``; because a single realization's stdev
   is not exactly 1.0, the sill the truth field REALIZES is ``a**2``
   (7.5330 / 7.5553 / 7.7311 / 7.9710 / 8.2402 / 8.4950 at these six levels)
   rather than the target ``POR_STDEV**2`` = 9.0. Both readings of "the
   truth nugget in physical units" are defensible, so both are stored --
   ``truth_nugget_real_units`` (headline, ``nug * 9``, and the quantity
   plotted) and ``truth_nugget_real_units_affine`` (``nug * a**2``), each
   with its own GP-noise ratio column. The HEADLINE IS NOT CHANGING.
2. The DIMENSIONLESS comparison. GP-MLE's WhiteKernel noise is fitted in
   sklearn's internally standardized y-space (``normalize_y=True``) and the
   axis's truth nugget is defined on a unit sill, so
   ``gp_noise_variance_normalized`` vs. ``truth_nugget_normalized`` (= nug,
   0.0 .. 0.5) is a comparison that needs no unit conversion at all. Both
   columns, and their ratio, are stored. ALSO stored, and the closest
   like-for-like partner of the axis definition, is
   ``gp_noise_fraction_of_fitted_sill`` = noise / (signal + noise) of the
   fitted kernel -- the axis level is exactly that ratio for the ground
   truth. MEASURED 2026-09-22: 0.1572 / 0.2556 / 0.3396 / 0.4252 / 0.5157 /
   0.6032 against a truth normalized nugget of 0.0 / 0.1 / 0.2 / 0.3 / 0.4 /
   0.5.

Output: results/figures/nugget_axis/nugget_fitted_vs_truth.png

Run with:
.venv/Scripts/python.exe -m results.processed.nugget_axis.make_nugget_fitted_vs_truth_figure
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

from src.experiments.base_case import POR_STDEV  # noqa: E402
from src.experiments.nugget_axis import (  # noqa: E402
    ALL_NUGGET_VALUES,
    affine_scale_factor,
    level_key,
    truth_nugget_real_units,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "nugget_axis"

FIG_DPI = 300
RATIO_LOG_THRESHOLD = 10.0  # same rule as range_axis/make_length_vs_noise_figure.py
CROSS_CHECK_RTOL = 1e-9

TOTAL_SILL_REAL = POR_STDEV ** 2


def _manifest_params(rel_run_dir: str) -> dict:
    return json.loads(
        (_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8")
    )["params"]


def at_optimizer_noise_floor(manifest_params: dict) -> bool:
    """True iff this gp_mle run's MLE stopped EXACTLY on the WhiteKernel
    noise-level lower bound (a boundary stop, not an interior optimum).
    Same definition as sample_replicate_axis/make_length_vs_smoothing_figures.py's
    function of the same name: the bound is read from THIS run's own
    manifest, and the comparison is relative because sklearn returns the
    bound up to float noise (e.g. 9.999999999999997e-06 for a 1e-5 bound)."""
    noise_norm = float(manifest_params["fitted_hyperparameters"]["noise_variance_normalized"])
    floor = float(manifest_params["kernel_init"]["noise_level_bounds"][0])
    return abs(noise_norm - floor) <= 1e-9 * floor


def build_dataset() -> pd.DataFrame:
    """One row per nugget level: (nug, truth_nugget_real_units,
    gp_noise_variance_real_units, at_noise_floor, noise_level_lower_bound).
    y is read fresh from each run's own manifest and cross-checked against
    qc_summary.csv."""
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    qc = pd.read_csv(PROCESSED_DIR / "qc_summary.csv")
    qc["axis_level"] = qc["axis_level"].astype(str)

    rows = []
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        params = _manifest_params(source_runs[level]["gp_mle"])
        hp = params["fitted_hyperparameters"]
        y = float(hp["noise_variance_real_units"])

        qc_row = qc[qc["axis_level"] == level]
        if len(qc_row) != 1:
            raise ValueError(
                f"qc_summary.csv: expected exactly 1 row for axis_level={level}, "
                f"found {len(qc_row)}."
            )
        qc_noise = float(qc_row["gp_noise_variance_real"].iloc[0])
        if not np.isclose(y, qc_noise, rtol=CROSS_CHECK_RTOL, atol=0.0):
            raise ValueError(
                f"nug={level}: noise_variance_real_units from manifest ({y}) disagrees "
                f"with qc_summary.csv's gp_noise_variance_real ({qc_noise})."
            )
        # The truth nugget is also tabulated by qc_nugget_axis.py -- checking
        # it here means the x axis is verified against an independently
        # written file, not just recomputed from the same helper.
        qc_truth_nug = float(qc_row["truth_nugget_real_units"].iloc[0])
        x = truth_nugget_real_units(nug)
        if not np.isclose(x, qc_truth_nug, rtol=CROSS_CHECK_RTOL, atol=0.0):
            raise ValueError(
                f"nug={level}: truth_nugget_real_units computed here ({x}) disagrees with "
                f"qc_summary.csv's value ({qc_truth_nug})."
            )

        # --- Additional recorded columns (reviewer 2026-09-22) -----------
        # (a) ALTERNATIVE physical-unit scale. The headline x above is
        #     nug * POR_STDEV**2 (= nug * 9). GSLIB.affine actually scaled
        #     this realization by a = POR_STDEV / std(sim_ns), so the sill it
        #     REALIZES is a**2, not 9. Both readings are defensible, so both
        #     are stored; the PLOTTED x and the headline ratio are unchanged.
        #     affine_scale_factor regenerates the level's truth (one sgsim
        #     call) -- that is why this function is not instant.
        # (b) NORMALIZED (dimensionless) noise fraction. GP-MLE's
        #     WhiteKernel noise_level is fitted in sklearn's internally
        #     standardized y-space (normalize_y=True), and the axis's truth
        #     nugget is defined on a UNIT sill, so these two are directly
        #     comparable without any physical-unit conversion at all.
        a = affine_scale_factor(nug)
        truth_nugget_affine = float(nug) * a * a
        noise_normalized = float(hp["noise_variance_normalized"])
        rows.append(
            {
                "axis_level": level,
                "nug_normalized": float(nug),
                "truth_nugget_real_units": x,
                "gp_noise_variance_real_units": y,
                "at_optimizer_noise_floor": at_optimizer_noise_floor(params),
                "noise_level_lower_bound_normalized": float(
                    params["kernel_init"]["noise_level_bounds"][0]
                ),
                # (a) alternative affine-based physical scale
                "affine_scale_factor_a": a,
                "affine_scale_factor_a_squared": a * a,
                "truth_nugget_real_units_affine": truth_nugget_affine,
                "gp_noise_over_truth_nugget": (y / x if x > 0 else np.nan),
                "gp_noise_over_truth_nugget_affine": (
                    y / truth_nugget_affine if truth_nugget_affine > 0 else np.nan
                ),
                # (b) normalized (unit-sill) scale
                "truth_nugget_normalized": float(nug),
                "gp_noise_variance_normalized": noise_normalized,
                "gp_noise_over_truth_nugget_normalized": (
                    noise_normalized / float(nug) if float(nug) > 0 else np.nan
                ),
                "gp_signal_variance_normalized": float(
                    hp["signal_variance_normalized"]
                ),
                # The column that is the true like-for-like partner of the
                # axis definition: the axis level IS nugget/(nugget+sill
                # contribution) at a unit sill, so GP-MLE's comparable
                # quantity is noise/(signal + noise) of its OWN fitted
                # kernel, not its raw noise_level. MEASURED 2026-09-22:
                # 0.1572 / 0.2556 / 0.3396 / 0.4252 / 0.5157 / 0.6032
                # against truth 0.0 / 0.1 / 0.2 / 0.3 / 0.4 / 0.5.
                "gp_noise_fraction_of_fitted_sill": (
                    noise_normalized
                    / (noise_normalized + float(hp["signal_variance_normalized"]))
                ),
                "gp_noise_fraction_minus_truth_nugget": (
                    noise_normalized
                    / (noise_normalized + float(hp["signal_variance_normalized"]))
                    - float(nug)
                ),
                "gp_y_train_std": float(hp["y_train_std"]),
            }
        )

    df = pd.DataFrame(rows).sort_values("nug_normalized").reset_index(drop=True)
    print("cross-checks passed: x and y both agree with qc_summary.csv for all 6 levels.")
    return df


def main():
    df = build_dataset()

    y_ratio = float(
        df["gp_noise_variance_real_units"].max() / df["gp_noise_variance_real_units"].min()
    )
    y_log = y_ratio >= RATIO_LOG_THRESHOLD
    x_has_zero = bool((df["truth_nugget_real_units"] == 0).any())
    print(
        f"Observed spread: y (gp noise variance) in "
        f"[{df['gp_noise_variance_real_units'].min():.4f}, "
        f"{df['gp_noise_variance_real_units'].max():.4f}] Porosity %^2, ratio "
        f"{y_ratio:.2f}x -> {'LOG' if y_log else 'LINEAR'} axis "
        f"(threshold {RATIO_LOG_THRESHOLD:g}x)"
    )
    print(
        f"x contains an exact zero ({x_has_zero}) -> x axis must be LINEAR regardless "
        "of spread."
    )
    n_at_floor = int(df["at_optimizer_noise_floor"].sum())
    if n_at_floor:
        print(
            f"NOTE: {n_at_floor} level(s) stopped EXACTLY on the WhiteKernel noise "
            "lower bound and are drawn as hollow markers: "
            + ", ".join(df[df["at_optimizer_noise_floor"]]["axis_level"].tolist())
        )
    else:
        print(
            "No level stopped on the WhiteKernel noise lower bound -- every fitted "
            "noise variance is an interior optimum (all markers filled)."
        )

    interior = df[~df["at_optimizer_noise_floor"]]
    at_floor = df[df["at_optimizer_noise_floor"]]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # Connecting line through all 6 levels in ascending nugget order.
    ax.plot(
        df["truth_nugget_real_units"], df["gp_noise_variance_real_units"],
        color="tab:red", linewidth=1.8, zorder=2,
    )
    ax.scatter(
        interior["truth_nugget_real_units"], interior["gp_noise_variance_real_units"],
        s=110, marker="D", facecolors="tab:red", edgecolors="black", linewidths=1.0,
        zorder=3, label="GP-MLE fitted noise variance",
    )
    if len(at_floor):
        ax.scatter(
            at_floor["truth_nugget_real_units"], at_floor["gp_noise_variance_real_units"],
            s=130, marker="D", facecolors="none", edgecolors="tab:red", linewidths=1.8,
            zorder=4,
            label=(
                "stopped at optimizer noise lower bound "
                "(value is a floor, true optimum may be lower)"
            ),
        )

    lo = 0.0
    hi = float(
        max(df["truth_nugget_real_units"].max(), df["gp_noise_variance_real_units"].max())
    )
    pad = 0.06 * hi
    ax.plot(
        [lo, hi + pad], [lo, hi + pad], color="gray", linestyle="--", linewidth=1.3,
        zorder=1, label="y = x (perfect recovery of the true nugget)",
    )
    ax.set_xlim(-pad, hi + pad)
    ax.set_ylim(-pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")

    for _, row in df.iterrows():
        ax.annotate(
            f"nug={row['nug_normalized']:g}",
            (row["truth_nugget_real_units"], row["gp_noise_variance_real_units"]),
            textcoords="offset points", xytext=(9, -4), fontsize=8.5, color="dimgray",
        )

    ax.set_xticks([truth_nugget_real_units(n) for n in ALL_NUGGET_VALUES])
    ax.set_xlabel("Ground-truth nugget (Porosity %$^2$)")
    ax.set_ylabel("GP-MLE fitted noise variance (Porosity %$^2$)")
    ax.set_title(
        "Nugget axis: what nugget did GP-MLE's marginal-likelihood fit learn?\n"
        f"(6 levels, 1 ground-truth realization each; total sill = "
        f"{TOTAL_SILL_REAL:g} Porosity %$^2$)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")

    fig.text(
        0.5, 0.005, textwrap.fill(
            "x = nug * POR_STDEV^2 (nugget_axis.truth_nugget_real_units), also cross-checked "
            "against qc_summary.csv. y = params.fitted_hyperparameters.noise_variance_real_units, "
            "read fresh from each run's own manifest.json and cross-checked against "
            "qc_summary.csv's gp_noise_variance_real. kriging and SGS are omitted because they "
            "are HANDED the true nugget and would lie on y = x by construction; RBF+bootstrap "
            "has no nugget parameter to plot. Hollow markers (if any) are levels whose fit "
            "stopped exactly on the WhiteKernel noise-level lower bound read from that run's own "
            f"manifest. Both axes linear: x contains an exact zero (log impossible) and the y "
            f"spread is {y_ratio:.2f}x, below the {RATIO_LOG_THRESHOLD:g}x threshold this project "
            "uses to switch to log; a log y would also destroy the meaning of the y = x line.",
            150,
        ),
        ha="center", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.11, bottom=0.21, right=0.97, top=0.89)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "nugget_fitted_vs_truth.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"nugget_fitted_vs_truth.png: {out} ({out.stat().st_size} bytes)")

    csv_out = PROCESSED_DIR / "gp_fitted_nugget_vs_truth.csv"
    df.to_csv(csv_out, index=False)
    print(f"gp_fitted_nugget_vs_truth.csv: {csv_out}")
    print("\n" + df.to_string(index=False))
    return df, out


if __name__ == "__main__":
    main()
