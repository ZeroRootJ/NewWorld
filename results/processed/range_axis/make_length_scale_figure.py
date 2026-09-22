"""Range-axis analogue of results/processed/sample_density_axis/
make_length_and_variogram_figures.py's "(a) length scale vs. density" figure
and of src/experiments/diagnose_sample_density_axis.py's
``gp_scale_conversion()`` -- what "length"/"range" each of the 4 methods
actually used or fitted, at each of the 8 range-axis levels (100..800 m).

WHY THIS IS A SEPARATE SCRIPT, NOT A REUSE OF THE DENSITY-AXIS ONE
--------------------------------------------------------------------
The density axis holds the ground-truth variogram range FIXED (300 m,
AXIS_HMAJ1) and varies n_samples; kriging/SGS's plotted "length" there is
therefore a single flat input constant across all levels. The range axis
holds n_samples fixed and instead varies the ground-truth range itself
(ALL_RANGE_VALUES, one realization per level -- unlike the density axis's
sample-replicate axis, there is NO 10-replicate spread here to average over:
each range level has exactly one ground-truth realization and one
conditioning-sample draw). kriging/SGS's input range is therefore NOT a flat
line here -- it tracks the x-axis EXACTLY (each level's kriging/SGS run is
handed hmaj1=hmin1=that level's own ground-truth range, verified below), so
the diagnostic value of this figure is different: it is whether RBF+
bootstrap's CV-tuned shape parameter and GP-MLE's MLE-fitted length_scale
track that same 1:1 line, not whether a fixed input constant is at least
consistent across levels.

WHAT "LENGTH SCALE" MEANS FOR EACH METHOD -- same conventions as
sample_density_axis/make_length_and_variogram_figures.py's docstring (READ
THAT FIRST if this is unfamiliar); summarized here:

  kriging / SGS
    The INPUT spherical variogram range (build_vario hmaj1/hmin1) handed to
    kb2d/sgsim for that level. On the range axis this literally EQUALS the
    ground-truth range at every level (unlike the density axis, where it is
    held fixed) -- verified below from each run's own manifest.json
    (params.variogram.hmaj1/hmin1), never hardcoded, and required to equal
    the level's own ground-truth range or this script raises.

  GP-MLE
    MLE-fitted sklearn RBF kernel length_scale, converted to a "practical
    range" -- the lag at which the correlation has decayed to
    CORRELATION_CUTOFF=0.05 (h = l * sqrt(2*ln(20)) ~= 2.4477*l) -- read
    straight from each run's own manifest.json
    (params.fitted_hyperparameters.length_scale_m), the SAME conversion
    src/experiments/diagnose_sample_density_axis.py uses.

  RBF+bootstrap
    CV-tuned gaussian RBFInterpolator shape parameter epsilon, converted to
    the SAME 0.05-correlation-cutoff distance for visual comparability
    (r_0.05 = sqrt(-ln(0.05))/epsilon ~= 1.7308/epsilon -- note: no factor of
    2 in the exponent, because scipy's gaussian RBF kernel
    phi(r)=exp(-(epsilon*r)^2) is parameterized differently from sklearn's
    RBF kernel). See RBF_CAVEAT below -- this is a geometric conversion of a
    curve-fitting shape parameter, NOT evidence RBF "learned" a spatial
    range. Values are read from the RE-RUN (2026-09-22 EPSILON_GRID
    resolution increase, N_EPSILON 25 -> 121) rbf_bootstrap runs pinned in
    source_runs.json, not from any earlier pin.

X-AXIS SCALE: LINEAR (not log, unlike the density axis's figure)
------------------------------------------------------------------
The density axis's figure uses a log x-axis because its 5 levels are
EXPONENTIALLY spaced (500/250/125/50/25 samples, ~1.3 decades). The range
axis's 8 levels are LINEARLY spaced BY DESIGN (100 m steps, 100..800 m, a
mere 8x/0.9-decade span) -- exactly matching the x-axis convention already
used by every other range-axis figure in this project
(results/processed/range_axis/make_range_axis_figures.py plots all of its
panels against RANGE_VALUES on a linear axis, no set_xscale call). A log
axis here would only compress the very range the reader most wants to read
precisely.

NO REPLICATE SPREAD (unlike the density axis's figure)
----------------------------------------------------------------------------
The sample-density axis's figure overlays a 10-sample_seed-replicate spread
for GP-MLE/RBF+bootstrap (results/processed/sample_replicate_axis/
length_scale_by_replicate.csv). The range axis has NO replicate axis of its
own: each of the 8 levels has exactly ONE ground-truth realization
(TRUTH_SEED=101) and ONE conditioning-sample draw (SAMPLE_SEED=20,
n_samples_requested=125). Every point in this figure is therefore a single
observation, not a mean +- spread -- stated explicitly in the figure caption
so it is not mistaken for the density-axis figure's replicate-averaged points.

Outputs
-------
  results/processed/range_axis/length_scale_by_method.csv      (32 rows: 8 levels x 4 methods)
  results/processed/range_axis/gp_hyperparameter_scale_conversion.csv  (8 rows)
  results/figures/range_axis/00_length_scale_vs_range.png

Run with:
.venv/Scripts/python.exe -m results.processed.range_axis.make_length_scale_figure
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
from src.experiments.range_axis import ALL_RANGE_VALUES, METHODS  # noqa: E402

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "range_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "range_axis"

METHOD_COLORS = {
    "kriging": "tab:blue", "sgs": "tab:green",
    "rbf_bootstrap": "tab:orange", "gp_mle": "tab:red",
}
METHOD_MARKERS = {"kriging": "o", "sgs": "s", "rbf_bootstrap": "^", "gp_mle": "D"}
METHOD_LINESTYLES = {"kriging": "-", "sgs": ":", "rbf_bootstrap": "-", "gp_mle": "-"}

FIG_DPI = 300

# Same 0.05-correlation-cutoff convention used throughout this project
# (src/experiments/diagnose_sample_density_axis.py's CORRELATION_CUTOFF and
# sample_density_axis/make_length_and_variogram_figures.py's identical
# constant of the same name).
CORRELATION_CUTOFF = 0.05
GP_FACTOR_EXACT = float(np.sqrt(-2.0 * np.log(CORRELATION_CUTOFF)))  # 2.4477, sklearn RBF kernel
RBF_FACTOR_EXACT = float(np.sqrt(-np.log(CORRELATION_CUTOFF)))       # 1.7308, scipy gaussian RBF kernel

RBF_CAVEAT = (
    "RBF+bootstrap's value is NOT a fitted spatial correlation length: it is the CV-tuned "
    "shape parameter (epsilon) of a gaussian RBF INTERPOLATION kernel "
    "(phi(r)=exp(-(epsilon*r)^2)), converted here to an equivalent 0.05-correlation-cutoff "
    "distance (r_0.05 = sqrt(-ln(0.05))/epsilon) purely for visual comparability with the "
    "other three methods. Unlike kriging/SGS's input range (which literally EQUALS the "
    "ground-truth range at every level on this axis) or GP-MLE's MLE-fitted length_scale, it "
    "is not an estimate of the data's spatial correlation structure."
)


def _manifest_params(rel_run_dir: str) -> dict:
    return json.loads(
        (_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8")
    )["params"]


def build_length_scale_table(source_runs: dict) -> pd.DataFrame:
    """Tidy long-format table: one row per (axis_level, method) with that
    method's "length scale" in metres and what kind of quantity it is.
    32 rows (8 range levels x 4 methods)."""
    rows = []
    n_actual_by_level = {}
    for r in ALL_RANGE_VALUES:
        level = str(int(r))
        run_dirs = source_runs[level]

        # --- kriging / SGS: the INPUT range, read from each run's own -------
        # manifest (never hardcoded). On THIS axis it must equal the
        # ground-truth range r itself at every level (unlike the density
        # axis, which holds it fixed) -- verified, not assumed.
        k_params = _manifest_params(run_dirs["kriging"])
        s_params = _manifest_params(run_dirs["sgs"])
        k_hmaj1, k_hmin1 = float(k_params["variogram"]["hmaj1"]), float(k_params["variogram"]["hmin1"])
        s_hmaj1, s_hmin1 = float(s_params["variogram"]["hmaj1"]), float(s_params["variogram"]["hmin1"])
        if not (k_hmaj1 == k_hmin1 == s_hmaj1 == s_hmin1 == r):
            raise ValueError(
                f"range level {level}: kriging/SGS input ranges do not all equal the "
                f"ground-truth range {r:g} m (kriging hmaj1={k_hmaj1}, hmin1={k_hmin1}; "
                f"sgs hmaj1={s_hmaj1}, hmin1={s_hmin1}) -- the range axis is documented as "
                "isotropic and each level's kriging/SGS input range MUST equal that level's "
                "ground-truth range."
            )
        n_actual_by_level["kriging"] = int(k_params["n_samples_actual"])
        n_actual_by_level["sgs"] = int(s_params["n_samples_actual"])

        rows.append(dict(
            axis_level=level, gt_range_m=r, method="kriging", length_scale_m=k_hmaj1,
            length_definition="input spherical variogram range (build_vario hmaj1/hmin1); "
                               "EQUALS the ground-truth range at every level on this axis, "
                               "NOT fitted from the conditioning data",
            source=f"{run_dirs['kriging']}/manifest.json:params.variogram.hmaj1",
        ))
        rows.append(dict(
            axis_level=level, gt_range_m=r, method="sgs", length_scale_m=s_hmaj1,
            length_definition="input spherical variogram range (build_vario hmaj1/hmin1); "
                               "EQUALS the ground-truth range at every level on this axis, "
                               "NOT fitted from the conditioning data",
            source=f"{run_dirs['sgs']}/manifest.json:params.variogram.hmaj1",
        ))

        # --- GP-MLE: convert the fitted length_scale here (verified against
        # gp_hyperparameter_scale_conversion.csv below, not just trusted).
        g_params = _manifest_params(run_dirs["gp_mle"])
        hp = g_params["fitted_hyperparameters"]
        ell = float(hp["length_scale_m"])
        n_actual_by_level["gp_mle"] = int(g_params["n_samples_actual"])
        rows.append(dict(
            axis_level=level, gt_range_m=r, method="gp_mle",
            length_scale_m=GP_FACTOR_EXACT * ell,
            length_definition=(
                f"MLE-fitted sklearn RBF kernel length_scale converted to a practical range "
                f"at the {CORRELATION_CUTOFF} correlation cutoff "
                f"(factor=sqrt(2*ln(20))~={GP_FACTOR_EXACT:.4f})"
            ),
            source=f"{run_dirs['gp_mle']}/manifest.json:params.fitted_hyperparameters.length_scale_m",
        ))

        # --- RBF+bootstrap: best_epsilon from the (2026-09-22 re-run) manifest.
        rbf_params = _manifest_params(run_dirs["rbf_bootstrap"])
        kernel = rbf_params["rbf_kernel"]
        if kernel != "gaussian":
            raise ValueError(
                f"range level {level}: rbf_bootstrap manifest records rbf_kernel="
                f"'{kernel}', not 'gaussian' -- the phi(r)=exp(-(epsilon*r)^2) conversion "
                "used here does not apply to a different kernel family."
            )
        epsilon = float(rbf_params["best_epsilon"])
        r_cutoff = RBF_FACTOR_EXACT / epsilon
        n_actual_by_level["rbf_bootstrap"] = int(rbf_params["n_samples_actual"])
        rows.append(dict(
            axis_level=level, gt_range_m=r, method="rbf_bootstrap",
            length_scale_m=r_cutoff,
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
                f"range level {level}: n_samples_actual differs across methods: "
                f"{n_actual_by_level} -- the 4 methods must condition on the same sample count."
            )

    df = pd.DataFrame(rows)

    print("\n--- Length-scale-by-method inputs, verified from source files ---")
    for r in ALL_RANGE_VALUES:
        level = str(int(r))
        sub = df[df.axis_level == level]
        k_val = sub[sub.method == "kriging"]["length_scale_m"].iloc[0]
        s_val = sub[sub.method == "sgs"]["length_scale_m"].iloc[0]
        g_val = sub[sub.method == "gp_mle"]["length_scale_m"].iloc[0]
        r_val = sub[sub.method == "rbf_bootstrap"]["length_scale_m"].iloc[0]
        print(
            f"  range={r:g}m: kriging/SGS input={k_val:.1f}m/{s_val:.1f}m (== GT range, by "
            f"construction), GP-MLE practical range={g_val:.2f}m, "
            f"RBF practical range={r_val:.2f}m"
        )
    return df


def build_gp_conversion_table(source_runs: dict) -> pd.DataFrame:
    """Range-axis analogue of diagnose_sample_density_axis.gp_scale_conversion():
    8 rows (one per range level, no replicate axis to average over here),
    same column schema as sample_density_axis/gp_hyperparameter_scale_conversion.csv
    except truth_spherical_range_m varies PER ROW (this axis's own ground-truth
    range) instead of being one fixed constant."""
    truth_nugget_real = NUG * POR_STDEV ** 2  # NS-space nugget x physical sill

    rows = []
    for r in ALL_RANGE_VALUES:
        level = str(int(r))
        hp = _manifest_params(source_runs[level]["gp_mle"])["fitted_hyperparameters"]
        ell = float(hp["length_scale_m"])
        noise_real = float(hp["noise_variance_real_units"])
        rows.append(
            {
                "axis_level": level,
                "gp_length_scale_m": ell,
                "correlation_cutoff": CORRELATION_CUTOFF,
                "practical_range_factor_exact": GP_FACTOR_EXACT,
                "practical_range_m": GP_FACTOR_EXACT * ell,
                "practical_range_sqrt6_shorthand_m": float(np.sqrt(6.0)) * ell,
                "truth_spherical_range_m": r,
                "practical_range_over_truth_range": GP_FACTOR_EXACT * ell / r,
                "gp_noise_variance_real_units": noise_real,
                "gp_signal_variance_real_units": float(hp["signal_variance_real_units"]),
                "truth_nugget_real_units": truth_nugget_real,
                "gp_noise_over_truth_nugget": noise_real / truth_nugget_real,
            }
        )
    return pd.DataFrame(rows)


def make_length_scale_figure(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for method in METHODS:
        sub = df[df["method"] == method].sort_values("gt_range_m")
        ax.plot(
            sub["gt_range_m"], sub["length_scale_m"],
            marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method], linewidth=2, markersize=9,
            label=method, alpha=0.9,
        )

    lo, hi = min(ALL_RANGE_VALUES), max(ALL_RANGE_VALUES)
    ax.plot(
        [lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1.3,
        label="y = x (ground-truth range)", zorder=1,
    )

    ax.set_xlim(lo - 30, hi + 30)
    ax.set_xticks(ALL_RANGE_VALUES)
    ax.set_xlabel("Ground-truth variogram range (m)")
    ax.set_ylabel("Length scale (m)")
    ax.set_title(
        "Range axis: each method's length scale vs. ground-truth range\n"
        "(1 realization per level, no replicate spread -- see caption)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    fig.text(
        0.5, 0.01, textwrap.fill(
            "kriging and SGS overlap the y=x line EXACTLY at every level (both are handed "
            "hmaj1=hmin1=that level's own ground-truth range as their INPUT variogram range, "
            "by one-factor-at-a-time design -- see line styles: kriging solid circles, SGS "
            "dotted squares) -- this is a structural identity, not something fitted from the "
            "conditioning data. GP-MLE and RBF+bootstrap are genuinely fitted/tuned per level "
            "and need not track y=x. Unlike the sample-density axis's equivalent figure, THIS "
            "AXIS HAS NO REPLICATE SPREAD: each of the 8 levels has exactly ONE ground-truth "
            "realization (TRUTH_SEED=101) and ONE conditioning-sample draw (SAMPLE_SEED=20), "
            "so every point plotted here is a single observation, not a mean. " + RBF_CAVEAT,
            150,
        ),
        ha="center", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.1, bottom=0.26, right=0.97, top=0.88)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "00_length_scale_vs_range.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"00_length_scale_vs_range.png: {out} ({out.stat().st_size} bytes)")
    return out


def main():
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    expected_levels = {str(int(r)) for r in ALL_RANGE_VALUES}
    if set(source_runs) != expected_levels:
        raise ValueError(
            f"source_runs.json has levels {sorted(source_runs)}; expected {sorted(expected_levels)}."
        )

    length_df = build_length_scale_table(source_runs)
    if len(length_df) != len(ALL_RANGE_VALUES) * len(METHODS):
        raise ValueError(
            f"length_scale_by_method.csv has {len(length_df)} rows; expected "
            f"{len(ALL_RANGE_VALUES) * len(METHODS)} (8 levels x 4 methods)."
        )
    length_csv = PROCESSED_DIR / "length_scale_by_method.csv"
    length_df.to_csv(length_csv, index=False)
    print(f"\nlength_scale_by_method.csv: {length_csv}")

    gp_df = build_gp_conversion_table(source_runs)
    gp_csv = PROCESSED_DIR / "gp_hyperparameter_scale_conversion.csv"
    gp_df.to_csv(gp_csv, index=False)
    print(f"gp_hyperparameter_scale_conversion.csv: {gp_csv}")

    # Cross-check: gp_mle rows of length_df must match gp_df's practical_range_m
    # exactly (both are the SAME formula computed in two different functions --
    # this catches a copy/paste drift between them).
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

    saved = make_length_scale_figure(length_df)
    print(f"\n00_length_scale_vs_range.png: {saved}")
    return length_df, gp_df, saved


if __name__ == "__main__":
    main()
