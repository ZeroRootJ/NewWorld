"""Nugget-axis analogue of results/processed/range_axis/make_length_scale_figure.py
-- what "length"/"range" each of the 4 methods actually used or fitted, at
each of the 6 nugget-axis levels (normalized nugget 0.0 .. 0.5).

WHY THIS IS A SEPARATE SCRIPT, NOT A REUSE OF THE RANGE-AXIS ONE
-----------------------------------------------------------------
The range axis VARIES the ground-truth range, so kriging/SGS's plotted
"length" there tracks the x-axis 1:1 and the diagnostic question is whether
GP-MLE/RBF track that same line. This axis HOLDS the range fixed at the
base-case 300 m at every level and varies the nugget instead, so
kriging/SGS's input range is a FLAT line at 300 m (verified below from each
run's own manifest, never hardcoded) and the diagnostic question is
different: does increasing the ground truth's nugget pull GP-MLE's
MLE-fitted length scale and RBF+bootstrap's CV-tuned shape parameter AWAY
from that fixed, correct 300 m? Since the true spatial-correlation range is
identical at every level, ANY movement of those two curves is attributable
to the nugget alone (one-factor-at-a-time).

WHAT "LENGTH SCALE" MEANS FOR EACH METHOD -- conversions and caveats are
taken VERBATIM from results/processed/range_axis/make_length_scale_figure.py
(same CORRELATION_CUTOFF, same two factors) so the two axes' figures are
directly comparable:

  kriging / SGS
    The INPUT spherical variogram range (build_vario hmaj1/hmin1) handed to
    kb2d/sgsim for that level -- here constant at the base-case 300 m, read
    from each run's own manifest.json (params.variogram.hmaj1/hmin1) and
    REQUIRED to equal AXIS_HMAJ1 or this script raises.

  GP-MLE
    MLE-fitted sklearn RBF kernel length_scale converted to a "practical
    range" -- the lag at which the correlation has decayed to
    CORRELATION_CUTOFF=0.05 (h = l * sqrt(2*ln(20)) ~= 2.4477*l) -- read
    straight from each run's own manifest.json
    (params.fitted_hyperparameters.length_scale_m).

  RBF+bootstrap
    CV-tuned gaussian RBFInterpolator shape parameter epsilon, converted to
    the SAME 0.05-correlation-cutoff distance for visual comparability
    (r_0.05 = sqrt(-ln(0.05))/epsilon ~= 1.7308/epsilon -- note: no factor of
    2 in the exponent, because scipy's gaussian RBF kernel
    phi(r)=exp(-(epsilon*r)^2) is parameterized differently from sklearn's
    RBF kernel). See RBF_CAVEAT -- a geometric conversion of a curve-fitting
    shape parameter, NOT evidence RBF "learned" a spatial range.

X-AXIS SCALE: LINEAR. The 6 levels are linearly spaced by design (0.1 steps)
and include exactly 0.0, so a log axis is both unmotivated and impossible.

Y-AXIS SCALE: **LOG, as a deliberate OVERRIDE of this project's 10x rule**
(orchestrator decision, 2026-09-22). The project-wide convention -- the same
RATIO_LOG_THRESHOLD = 10x rule the range axis's
make_length_vs_noise_figure.py applies -- is still computed over ALL plotted
values and printed on every run, so the rule's verdict can never silently
drift from what is drawn. MEASURED: values span 240.22 m (RBF at nug=0.0) to
1364.67 m (GP-MLE at nug=0.5), a max/min ratio of 5.68x, which is BELOW the
10x threshold and would therefore select a LINEAR axis.

WHY THE OVERRIDE (this is the required justification, not a preference):
on a linear axis the single GP-MLE point at 1365 m stretches the range so
far that the remaining 23 points are compressed into an unreadable 240-320 m
band -- i.e. the rule's own purpose (making the plotted values legible) is
defeated here by one outlier rather than served by it. A log axis restores
the resolution of that band while keeping the outlier on the same panel. The
override applies to THIS figure only; the 10x rule stands everywhere else,
and if the observed ratio ever does cross 10x this figure's own printed
output will say so.

The x axis is unaffected (still LINEAR, see above).

That nug=0.5 GP-MLE point is an INTERIOR optimum, not a bound stop: the
optimizer's length_scale bounds are (1e-2, 1e4) m and the fitted value is
557.5 m (practical range 1364.67 m). This script re-checks the at-bound
condition from each run's OWN manifest bounds on every run, prints the
result, and annotates the caption accordingly -- so if a future run DID stop
on a bound, the figure would say so rather than presenting a clipped value
as a fit.

NO REPLICATE SPREAD: each of the 6 levels has exactly ONE ground-truth
realization (TRUTH_SEED=101) and ONE conditioning-sample draw
(SAMPLE_SEED=20, n_samples_requested=125 -> 121 actual). Every point is a
single observation, not a mean +- spread.

Outputs
-------
  results/processed/nugget_axis/length_scale_by_method.csv            (24 rows: 6 levels x 4 methods)
  results/processed/nugget_axis/gp_hyperparameter_scale_conversion.csv (6 rows)
  results/figures/nugget_axis/length_scale_vs_nugget.png

Run with:
.venv/Scripts/python.exe -m results.processed.nugget_axis.make_length_scale_figure
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
from src.experiments.gp_mle import LENGTH_SCALE_BOUNDS  # noqa: E402
from src.experiments.nugget_axis import (  # noqa: E402
    ALL_NUGGET_VALUES,
    AXIS_HMAJ1,
    AXIS_HMIN1,
    METHODS,
    level_key,
    truth_nugget_real_units,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "nugget_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "nugget_axis"

METHOD_COLORS = {
    "kriging": "tab:blue", "sgs": "tab:green",
    "rbf_bootstrap": "tab:orange", "gp_mle": "tab:red",
}
METHOD_MARKERS = {"kriging": "o", "sgs": "s", "rbf_bootstrap": "^", "gp_mle": "D"}
METHOD_LINESTYLES = {"kriging": "-", "sgs": ":", "rbf_bootstrap": "-", "gp_mle": "-"}

FIG_DPI = 300
RATIO_LOG_THRESHOLD = 10.0

# Orchestrator decision 2026-09-22: use a LOG y axis on this figure even
# though the project's RATIO_LOG_THRESHOLD rule would choose linear here
# (observed ratio 5.68x < 10x). Reason: a single GP-MLE outlier at 1365 m
# compresses the other 23 points into a ~240-320 m band on a linear axis. The
# rule's verdict is still computed and printed on every run, and the
# override is stated in the figure caption -- see the module docstring's
# "Y-AXIS SCALE" section for the full rationale. Set this back to False to
# return to the project default.
Y_SCALE_LOG_OVERRIDE = True

# Same 0.05-correlation-cutoff convention used throughout this project.
CORRELATION_CUTOFF = 0.05
GP_FACTOR_EXACT = float(np.sqrt(-2.0 * np.log(CORRELATION_CUTOFF)))  # 2.4477, sklearn RBF kernel
RBF_FACTOR_EXACT = float(np.sqrt(-np.log(CORRELATION_CUTOFF)))       # 1.7308, scipy gaussian RBF kernel

TOTAL_SILL_REAL = POR_STDEV ** 2

RBF_CAVEAT = (
    "RBF+bootstrap's value is NOT a fitted spatial correlation length: it is the CV-tuned "
    "shape parameter (epsilon) of a gaussian RBF INTERPOLATION kernel "
    "(phi(r)=exp(-(epsilon*r)^2)), converted here to an equivalent 0.05-correlation-cutoff "
    "distance (r_0.05 = sqrt(-ln(0.05))/epsilon) purely for visual comparability with the "
    "other three methods. Unlike kriging/SGS's input range (held fixed at the true 300 m on "
    "this axis) or GP-MLE's MLE-fitted length_scale, it is not an estimate of the data's "
    "spatial correlation structure."
)


def _manifest_params(rel_run_dir: str) -> dict:
    return json.loads(
        (_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8")
    )["params"]


def build_length_scale_table(source_runs: dict) -> pd.DataFrame:
    """Tidy long-format table: one row per (axis_level, method) with that
    method's "length scale" in metres and what kind of quantity it is.
    24 rows (6 nugget levels x 4 methods)."""
    rows = []
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        run_dirs = source_runs[level]
        n_actual_by_level = {}

        # --- kriging / SGS: the INPUT range, read from each run's own -------
        # manifest (never hardcoded). On THIS axis it must equal the FIXED
        # base-case range at every level -- verified, not assumed.
        k_params = _manifest_params(run_dirs["kriging"])
        s_params = _manifest_params(run_dirs["sgs"])
        k_hmaj1, k_hmin1 = float(k_params["variogram"]["hmaj1"]), float(k_params["variogram"]["hmin1"])
        s_hmaj1, s_hmin1 = float(s_params["variogram"]["hmaj1"]), float(s_params["variogram"]["hmin1"])
        if not (k_hmaj1 == k_hmin1 == s_hmaj1 == s_hmin1 == AXIS_HMAJ1 == AXIS_HMIN1):
            raise ValueError(
                f"nugget level {level}: kriging/SGS input ranges do not all equal the axis's "
                f"fixed range {AXIS_HMAJ1:g} m (kriging hmaj1={k_hmaj1}, hmin1={k_hmin1}; "
                f"sgs hmaj1={s_hmaj1}, hmin1={s_hmin1}) -- the nugget axis is documented as "
                "holding the range fixed at the base case."
            )
        # Also verify the nugget these two were handed IS this level's nugget
        # (kriging/SGS are the baselines that get the true value).
        for name, params in (("kriging", k_params), ("sgs", s_params)):
            if not np.isclose(float(params["variogram"]["nug"]), float(nug)):
                raise ValueError(
                    f"nugget level {level}: {name}'s input variogram nug="
                    f"{params['variogram']['nug']}, expected {nug}."
                )
        n_actual_by_level["kriging"] = int(k_params["n_samples_actual"])
        n_actual_by_level["sgs"] = int(s_params["n_samples_actual"])

        base = dict(
            axis_level=level,
            nug_normalized=float(nug),
            truth_nugget_real_units=truth_nugget_real_units(nug),
            gt_range_m=AXIS_HMAJ1,
        )
        fixed_range_definition = (
            "input spherical variogram range (build_vario hmaj1/hmin1); HELD FIXED at the "
            "base-case range at every level on this axis (the nugget is the only thing "
            "varying), NOT fitted from the conditioning data"
        )
        rows.append(dict(
            **base, method="kriging", length_scale_m=k_hmaj1,
            length_definition=fixed_range_definition,
            source=f"{run_dirs['kriging']}/manifest.json:params.variogram.hmaj1",
        ))
        rows.append(dict(
            **base, method="sgs", length_scale_m=s_hmaj1,
            length_definition=fixed_range_definition,
            source=f"{run_dirs['sgs']}/manifest.json:params.variogram.hmaj1",
        ))

        # --- GP-MLE ---------------------------------------------------------
        g_params = _manifest_params(run_dirs["gp_mle"])
        hp = g_params["fitted_hyperparameters"]
        ell = float(hp["length_scale_m"])
        n_actual_by_level["gp_mle"] = int(g_params["n_samples_actual"])
        rows.append(dict(
            **base, method="gp_mle", length_scale_m=GP_FACTOR_EXACT * ell,
            length_definition=(
                f"MLE-fitted sklearn RBF kernel length_scale converted to a practical range "
                f"at the {CORRELATION_CUTOFF} correlation cutoff "
                f"(factor=sqrt(2*ln(20))~={GP_FACTOR_EXACT:.4f})"
            ),
            source=f"{run_dirs['gp_mle']}/manifest.json:params.fitted_hyperparameters.length_scale_m",
        ))

        # --- RBF+bootstrap ---------------------------------------------------
        rbf_params = _manifest_params(run_dirs["rbf_bootstrap"])
        kernel = rbf_params["rbf_kernel"]
        if kernel != "gaussian":
            raise ValueError(
                f"nugget level {level}: rbf_bootstrap manifest records rbf_kernel="
                f"'{kernel}', not 'gaussian' -- the phi(r)=exp(-(epsilon*r)^2) conversion "
                "used here does not apply to a different kernel family."
            )
        epsilon = float(rbf_params["best_epsilon"])
        n_actual_by_level["rbf_bootstrap"] = int(rbf_params["n_samples_actual"])
        rows.append(dict(
            **base, method="rbf_bootstrap", length_scale_m=RBF_FACTOR_EXACT / epsilon,
            length_definition=(
                f"CV-tuned gaussian RBF interpolation kernel shape parameter epsilon "
                f"converted to an equivalent {CORRELATION_CUTOFF}-cutoff distance "
                f"(r=sqrt(-ln({CORRELATION_CUTOFF}))/epsilon); NOT a fitted spatial "
                "correlation length -- see RBF_CAVEAT"
            ),
            source=f"{run_dirs['rbf_bootstrap']}/manifest.json:params.best_epsilon",
        ))

        n_vals = set(n_actual_by_level.values())
        if len(n_vals) != 1:
            raise ValueError(
                f"nugget level {level}: n_samples_actual differs across methods: "
                f"{n_actual_by_level} -- the 4 methods must condition on the same sample count."
            )

    df = pd.DataFrame(rows)

    print("\n--- Length-scale-by-method inputs, verified from source files ---")
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        sub = df[df.axis_level == level]
        k_val = sub[sub.method == "kriging"]["length_scale_m"].iloc[0]
        s_val = sub[sub.method == "sgs"]["length_scale_m"].iloc[0]
        g_val = sub[sub.method == "gp_mle"]["length_scale_m"].iloc[0]
        r_val = sub[sub.method == "rbf_bootstrap"]["length_scale_m"].iloc[0]
        print(
            f"  nug={level}: kriging/SGS input={k_val:.1f}m/{s_val:.1f}m (fixed, == true "
            f"range), GP-MLE practical range={g_val:.2f}m, RBF practical range={r_val:.2f}m"
        )
    return df


def build_gp_conversion_table(source_runs: dict) -> pd.DataFrame:
    """Nugget-axis analogue of range_axis/make_length_scale_figure.py's
    build_gp_conversion_table: 6 rows (one per nugget level, no replicate axis
    here), SAME column schema as range_axis/gp_hyperparameter_scale_conversion.csv
    -- except that truth_spherical_range_m is now the constant 300 m and
    truth_nugget_real_units is what varies per row (the mirror image of the
    range axis, where the range varied and the nugget was constant).

    ``gp_noise_over_truth_nugget`` is NaN at nug=0.0: the truth nugget there
    is exactly 0, so the ratio is undefined. It is deliberately NOT replaced
    by inf or by a placeholder number."""
    rows = []
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        hp = _manifest_params(source_runs[level]["gp_mle"])["fitted_hyperparameters"]
        ell = float(hp["length_scale_m"])
        noise_real = float(hp["noise_variance_real_units"])
        truth_nugget_real = truth_nugget_real_units(nug)
        rows.append(
            {
                "axis_level": level,
                "gp_length_scale_m": ell,
                "correlation_cutoff": CORRELATION_CUTOFF,
                "practical_range_factor_exact": GP_FACTOR_EXACT,
                "practical_range_m": GP_FACTOR_EXACT * ell,
                "practical_range_sqrt6_shorthand_m": float(np.sqrt(6.0)) * ell,
                "truth_spherical_range_m": AXIS_HMAJ1,
                "practical_range_over_truth_range": GP_FACTOR_EXACT * ell / AXIS_HMAJ1,
                "gp_noise_variance_real_units": noise_real,
                "gp_signal_variance_real_units": float(hp["signal_variance_real_units"]),
                "truth_nugget_real_units": truth_nugget_real,
                "gp_noise_over_truth_nugget": (
                    noise_real / truth_nugget_real if truth_nugget_real > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def make_length_scale_figure(df: pd.DataFrame, source_runs: dict) -> Path:
    all_vals = df["length_scale_m"].to_numpy(dtype=float)
    y_ratio = float(all_vals.max() / all_vals.min())
    rule_says_log = y_ratio >= RATIO_LOG_THRESHOLD
    y_log = True if Y_SCALE_LOG_OVERRIDE else rule_says_log
    print(
        f"\nObserved spread: length_scale_m in [{all_vals.min():.2f}, {all_vals.max():.2f}] m, "
        f"ratio {y_ratio:.2f}x -> project {RATIO_LOG_THRESHOLD:g}x rule says "
        f"{'LOG' if rule_says_log else 'LINEAR'} y-axis"
    )
    if Y_SCALE_LOG_OVERRIDE and not rule_says_log:
        print(
            "  OVERRIDE (orchestrator decision 2026-09-22): drawing a LOG y-axis "
            "anyway -- the single GP-MLE outlier compresses the other 23 points "
            "into an unreadable band on a linear axis. See module docstring."
        )
    elif Y_SCALE_LOG_OVERRIDE and rule_says_log:
        print(
            "  note: Y_SCALE_LOG_OVERRIDE is on, but the project rule now selects "
            "LOG on its own -- the override is no longer doing anything."
        )

    # Which GP-MLE levels hit their length_scale optimizer bound (read from
    # each run's own manifest bounds, not the imported constant, so a run
    # fitted under different bounds is judged fairly). Reported so an extreme
    # point cannot be mistaken for a clipped one, or vice versa.
    at_bound_levels = []
    for nug in ALL_NUGGET_VALUES:
        level = level_key(nug)
        params = _manifest_params(source_runs[level]["gp_mle"])
        lo, hi = params["kernel_init"]["length_scale_bounds"]
        ell = float(params["fitted_hyperparameters"]["length_scale_m"])
        if ell <= float(lo) * 1.01 or ell >= float(hi) / 1.01:
            at_bound_levels.append(level)
    if at_bound_levels:
        print(
            f"NOTE: GP-MLE length_scale is AT its optimizer bound at level(s): "
            f"{at_bound_levels} (bounds {LENGTH_SCALE_BOUNDS})."
        )
    else:
        print(
            "GP-MLE length_scale is an interior optimum at every level "
            f"(optimizer bounds {LENGTH_SCALE_BOUNDS} m)."
        )

    fig, ax = plt.subplots(figsize=(9.5, 6.8))

    for method in METHODS:
        sub = df[df["method"] == method].sort_values("nug_normalized")
        ax.plot(
            sub["nug_normalized"], sub["length_scale_m"],
            marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method], linewidth=2, markersize=9,
            label=method, alpha=0.9,
        )

    ax.axhline(
        AXIS_HMAJ1, color="gray", linestyle="--", linewidth=1.3, zorder=1,
        label=f"true ground-truth range ({AXIS_HMAJ1:g} m, constant on this axis)",
    )

    if y_log:
        ax.set_yscale("log")
    ax.set_xticks([float(n) for n in ALL_NUGGET_VALUES])
    ax.set_xlabel("Ground-truth nugget (fraction of unit sill)")
    ax.set_ylabel("Length scale (m)")
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda v: np.asarray(v) * TOTAL_SILL_REAL,
            lambda v: np.asarray(v) / TOTAL_SILL_REAL,
        ),
    )
    secax.set_xticks([truth_nugget_real_units(n) for n in ALL_NUGGET_VALUES])
    secax.set_xlabel(
        f"Ground-truth nugget (Porosity %$^2$; total sill = {TOTAL_SILL_REAL:g})", fontsize=9
    )
    ax.set_title(
        "Nugget axis: each method's length scale vs. ground-truth nugget\n"
        "(range held fixed -- any movement is attributable to the nugget alone)",
        fontsize=11,
    )
    ax.grid(alpha=0.3, which="both" if y_log else "major")
    ax.legend(fontsize=8.5, loc="upper left")

    at_bound_note = (
        f" GP-MLE's length_scale is AT its optimizer bound at level(s) {at_bound_levels}."
        if at_bound_levels
        else " GP-MLE's length_scale is an interior optimum at every level (not clipped by "
             f"its optimizer bounds {LENGTH_SCALE_BOUNDS} m)."
    )
    if Y_SCALE_LOG_OVERRIDE and not rule_says_log:
        y_scale_note = (
            " Y-AXIS IS LOG, DELIBERATELY OVERRIDING this project's 10x rule: the observed "
            f"max/min ratio is only {y_ratio:.2f}x, which the rule would draw on a LINEAR "
            f"axis (threshold {RATIO_LOG_THRESHOLD:g}x). Override reason, recorded here and "
            "in this script's docstring: on a linear axis the single GP-MLE outlier at "
            f"{all_vals.max():.0f} m compresses the other {len(all_vals) - 1} points into an "
            "unreadable narrow band. The rule itself is unchanged for every other figure."
        )
    else:
        y_scale_note = (
            " y-axis scale chosen from the OBSERVED spread (max/min ratio "
            f"{y_ratio:.2f}x vs. a {RATIO_LOG_THRESHOLD:g}x threshold -> "
            f"{'log' if y_log else 'linear'})."
        )

    fig.text(
        0.5, 0.005, textwrap.fill(
            "kriging and SGS lie exactly on the dashed true-range line at every level (both are "
            f"handed hmaj1=hmin1={AXIS_HMAJ1:g} m as their INPUT variogram range, which on this "
            "axis IS the ground truth's range -- see line styles: kriging solid circles, SGS "
            "dotted squares). That is a structural identity, not something fitted. GP-MLE and "
            "RBF+bootstrap are genuinely fitted/tuned per level, so their movement as the nugget "
            "grows is the diagnostic content of this figure."
            + at_bound_note
            + y_scale_note
            + " NO REPLICATE SPREAD on this axis: each of the 6 "
            "levels has exactly ONE ground-truth realization (TRUTH_SEED=101) and ONE "
            "conditioning-sample draw (SAMPLE_SEED=20), so every point is a single observation, "
            "not a mean. " + RBF_CAVEAT,
            155,
        ),
        ha="center", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.1, bottom=0.27, right=0.97, top=0.84)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "length_scale_vs_nugget.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"length_scale_vs_nugget.png: {out} ({out.stat().st_size} bytes)")
    return out


def main():
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    expected_levels = {level_key(n) for n in ALL_NUGGET_VALUES}
    if set(source_runs) != expected_levels:
        raise ValueError(
            f"source_runs.json has levels {sorted(source_runs)}; expected "
            f"{sorted(expected_levels)}."
        )

    length_df = build_length_scale_table(source_runs)
    if len(length_df) != len(ALL_NUGGET_VALUES) * len(METHODS):
        raise ValueError(
            f"length_scale_by_method.csv has {len(length_df)} rows; expected "
            f"{len(ALL_NUGGET_VALUES) * len(METHODS)} (6 levels x 4 methods)."
        )
    length_csv = PROCESSED_DIR / "length_scale_by_method.csv"
    length_df.to_csv(length_csv, index=False)
    print(f"\nlength_scale_by_method.csv: {length_csv}")

    gp_df = build_gp_conversion_table(source_runs)
    gp_csv = PROCESSED_DIR / "gp_hyperparameter_scale_conversion.csv"
    gp_df.to_csv(gp_csv, index=False)
    print(f"gp_hyperparameter_scale_conversion.csv: {gp_csv}")

    # Cross-check: gp_mle rows of length_df must match gp_df's practical_range_m
    # exactly (the SAME formula computed in two different functions -- this
    # catches copy/paste drift between them).
    gp_from_length = length_df[length_df.method == "gp_mle"].sort_values("axis_level")
    gp_check = gp_df.sort_values("axis_level")
    if not np.allclose(
        gp_from_length["length_scale_m"].to_numpy(dtype=float),
        gp_check["practical_range_m"].to_numpy(dtype=float),
    ):
        raise ValueError(
            "gp_mle practical_range_m in length_scale_by_method.csv disagrees with "
            "gp_hyperparameter_scale_conversion.csv -- the two conversions have drifted apart."
        )
    print("cross-check passed: gp_mle practical_range_m agrees between both output tables.")

    saved = make_length_scale_figure(length_df, source_runs)
    print("\n" + gp_df.to_string(index=False))
    return length_df, gp_df, saved


if __name__ == "__main__":
    main()
