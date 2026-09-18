"""Two additional figures for the sample-density axis (docs/progress.md
"axis 5 -- sample count/sparsity"), requested by the user on top of the
figures already built by ``make_sample_density_figures.py``:

(a) 00_length_scale_vs_density.png -- what "length"/"range" each of the 4
    methods actually used or fitted, at each of the 3 density levels
    (5% / 2% / 1%, ACTUAL n_samples 121 / 50 / 25), so the reader can see at
    a glance that kriging/SGS/RBF's "range" does NOT respond to sample
    density the way GP-MLE's does.
(b) 00_experimental_variogram.png -- a SUBSAMPLED (not exhaustive) empirical
    variogram of the one shared ground-truth field, plotted against the
    known theoretical spherical model, so the reader can see the truth's
    actual spatial-correlation structure before looking at how each method
    recovered (or failed to recover) it.

Filenames are prefixed "00_" so they sort before every other file in
results/figures/sample_density_axis/ (calibration_curves_grid.png,
crossplot_*.png, metrics_vs_sample_density.png, qc_truth_predictions_*.png,
truth_field_all_levels.png) -- this repo has no separate written report, so
results/figures/<experiment>/ IS the report, and alphabetical order is how
"front of the report" is implemented (user instruction, 2026-09-17).

------------------------------------------------------------------------
(a) WHAT "LENGTH SCALE" MEANS FOR EACH METHOD -- READ BEFORE INTERPRETING
------------------------------------------------------------------------
The 4 methods do NOT all produce the same kind of quantity, and plotting
them on one axis risks implying they do. What is actually plotted:

  kriging / SGS
    Both call ``src.experiments.base_case_conditioning.build_vario(hmaj1,
    hmin1)`` with hmaj1=hmin1=HMAJ1=300.0 (the base-case constant) at EVERY
    axis level -- the sample-density axis is one-factor-at-a-time and holds
    the variogram range fixed while n_samples varies (see
    src/experiments/sample_density_axis.py's module docstring). This is the
    INPUT range handed to kb2d/sgsim, not something estimated from the
    conditioning data -- it is verified below to equal 300.0 in every
    level's own manifest.json (params.variogram.hmaj1), never hardcoded.

  GP-MLE
    Fits a sklearn RBF kernel by marginal-likelihood, whose length_scale has
    no finite range (the kernel decays asymptotically). Already converted
    (by src/experiments/diagnose_sample_density_axis.py, not recomputed
    here) to a "practical range" -- the lag at which the correlation has
    decayed to CORRELATION_CUTOFF=0.05 -- and read verbatim from
    results/processed/sample_density_axis/gp_hyperparameter_scale_conversion.csv
    (column practical_range_m).

  RBF+bootstrap
    Has no statistical correlation-length concept at all. Its only relevant
    tuning parameter is the CV-selected shape parameter ``epsilon`` of a
    gaussian interpolation kernel (scipy ``RBFInterpolator(kernel="gaussian")``:
    phi(r) = exp(-(epsilon*r)**2) -- confirmed directly, see manifest
    params.rbf_kernel == "gaussian"). Applying the SAME 0.05-cutoff
    definition used for GP-MLE for visual comparability: solving
    exp(-(epsilon*r)**2) = 0.05 gives
        r_0.05 = sqrt(-ln(0.05)) / epsilon  (approx. 1.7308 / epsilon)
    -- note the exponent has no factor of 2 here (unlike sklearn's RBF
    kernel exp(-d^2/(2*length_scale^2))), because scipy's gaussian RBF
    kernel is parameterized differently; the two "0.05-cutoff" conversions
    are NOT the same formula even though they share the same cutoff
    convention.

  *** CAVEAT THAT MUST TRAVEL WITH THIS FIGURE ***
  RBF's "practical range" is a GEOMETRIC CONVERSION of a curve-fitting
  shape parameter chosen by cross-validated MSE, not a fitted estimate of
  the data's spatial correlation length the way kriging/SGS's input range
  or GP-MLE's length_scale are. It is plotted here ONLY so the reader can
  see it on the same axis as the other three; it is not evidence that RBF
  "learned" a spatial range of that size. This caveat is printed on the
  figure itself, not just in this docstring.

------------------------------------------------------------------------
(b) EXPERIMENTAL VARIOGRAM OF THE SHARED TRUTH FIELD
------------------------------------------------------------------------
The sample-density axis's ground-truth field is IDENTICAL at all 3 levels
(TRUTH_SEED=101, HMAJ1=HMIN1=300.0 -- see make_sample_density_figures.py's
docstring), so there is exactly one experimental variogram to draw for the
whole axis.

Exhaustive is 2500 cells -> C(2500,2) ~= 3.1 million pairs; NOT computed.
Instead N_PAIRS_DRAWN=200,000 (i, j) cell-index pairs are drawn uniformly at
random (i != j) at a fixed, NEW seed (VARIOGRAM_PAIR_SEED=90 -- distinct
from every seed already used on this axis: TRUTH_SEED=101, SAMPLE_SEED=20,
NULL_SEED_START=1000..1300, and the CV/bootstrap seeds inside individual
method manifests). Each pair's Euclidean lag h and semivariance contribution
0.5*(z_i - z_j)^2 are computed, then aggregated into LAG_BIN_WIDTH_M=25 m
bins from 0 to LAG_MAX_M=750 m (30 bins) as a mean semivariance and a pair
count. Bins with fewer than MIN_PAIRS_PER_BIN=30 pairs are flagged (printed
as a warning and drawn with reduced opacity / a different marker) rather
than silently trusted.

Theoretical curve: nugget = NUG * POR_STDEV**2, structured sill = CC1 *
POR_STDEV**2, total sill = POR_STDEV**2, range = HMAJ1 -- all imported from
src.experiments.base_case (NUG, CC1, POR_STDEV, HMAJ1), not hardcoded, per
the derivation in the module docstring of
src/truth_model.py (GSLIB.affine is a LINEAR transform of the standard-
normal-space simulation, whose variogram sill is NUG + CC1 = 1.0, so scaling
by POR_STDEV multiplies every variogram value -- nugget, structured sill,
total sill -- by POR_STDEV**2 while leaving the range unchanged). The
truth field's own empirical mean/variance is logged below purely as a sanity
check that the affine step ran (GSLIB.affine rescales by the realization's
own sample mean/stdev to hit POR_MEAN/POR_STDEV**2 EXACTLY, to floating-point
precision, regardless of seed -- this is not an independent structural
check of the field, and no "finite-sample deviation" from the target is
expected here).

Run with:
.venv/Scripts/python.exe -m results.processed.sample_density_axis.make_length_and_variogram_figures
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

from src.experiments.base_case import (  # noqa: E402
    CC1,
    HMAJ1,
    NUG,
    NX,
    NY,
    POR_MEAN,
    POR_STDEV,
    XMN,
    YMN,
    XSIZ,
    YSIZ,
)
from src.experiments.base_case_conditioning import (  # noqa: E402
    TRUTH_SEED,
    get_base_case_truth,
)
from src.experiments.sample_density_axis import ALL_AXIS_LEVELS  # noqa: E402
from src.grid_utils import full_grid_coordinates  # noqa: E402

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_density_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "sample_density_axis"

METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]
METHOD_COLORS = {
    "kriging": "tab:blue", "sgs": "tab:green",
    "rbf_bootstrap": "tab:orange", "gp_mle": "tab:red",
}
METHOD_MARKERS = {"kriging": "o", "sgs": "s", "rbf_bootstrap": "^", "gp_mle": "D"}
METHOD_LINESTYLES = {"kriging": "-", "sgs": ":", "rbf_bootstrap": "-", "gp_mle": "-"}

FIG_DPI = 300

# Same 0.05-correlation-cutoff convention used by
# src/experiments/diagnose_sample_density_axis.py (CORRELATION_CUTOFF) for
# GP-MLE's length_scale -> practical range conversion; applied to RBF's
# epsilon here so both are visually comparable to the same yardstick. NOT
# recomputed for GP-MLE -- its values are read verbatim from
# gp_hyperparameter_scale_conversion.csv.
CORRELATION_CUTOFF = 0.05

RBF_CAVEAT = (
    "RBF+bootstrap's value is NOT a fitted spatial correlation length: it is the CV-tuned "
    "shape parameter (epsilon) of a gaussian RBF INTERPOLATION kernel "
    "(phi(r)=exp(-(epsilon*r)^2)), converted here to an equivalent 0.05-correlation-cutoff "
    "distance (r_0.05 = sqrt(-ln(0.05))/epsilon) purely for visual comparability with the "
    "other three methods. Unlike kriging/SGS's input range or GP-MLE's MLE-fitted "
    "length_scale, it is not an estimate of the data's spatial correlation structure."
)


# ---------------------------------------------------------------------------
# (a) Length-scale vs. density
# ---------------------------------------------------------------------------
def _manifest_params(rel_run_dir: str) -> dict:
    return json.loads(
        (_REPO_ROOT / rel_run_dir / "manifest.json").read_text(encoding="utf-8")
    )["params"]


def build_length_scale_table(source_runs: dict, pct_actual: dict) -> pd.DataFrame:
    """Tidy long-format table: one row per (axis_level, method) with that
    method's "length scale" in metres and what kind of quantity it is."""
    gp_df = pd.read_csv(PROCESSED_DIR / "gp_hyperparameter_scale_conversion.csv")
    gp_df["axis_level"] = gp_df["axis_level"].astype(str)

    rows = []
    kriging_ranges = {}
    for level in ALL_AXIS_LEVELS:
        # --- kriging / SGS: the INPUT range, read from each run's own -------
        # manifest (not hardcoded), and cross-checked for equality since both
        # call build_vario() with the same base-case HMAJ1/HMIN1 by design.
        k_params = _manifest_params(source_runs[level]["kriging"])
        s_params = _manifest_params(source_runs[level]["sgs"])
        k_hmaj1 = float(k_params["variogram"]["hmaj1"])
        k_hmin1 = float(k_params["variogram"]["hmin1"])
        s_hmaj1 = float(s_params["variogram"]["hmaj1"])
        s_hmin1 = float(s_params["variogram"]["hmin1"])
        if not (k_hmaj1 == k_hmin1 == s_hmaj1 == s_hmin1):
            raise ValueError(
                f"level '{level}': kriging/SGS input ranges disagree "
                f"(kriging hmaj1={k_hmaj1}, hmin1={k_hmin1}; sgs hmaj1={s_hmaj1}, "
                f"hmin1={s_hmin1}) -- this axis is documented as isotropic and range-fixed."
            )
        kriging_ranges[level] = k_hmaj1

        rows.append(dict(
            axis_level=level, density_pct_actual=pct_actual[level], method="kriging",
            length_scale_m=k_hmaj1,
            length_definition="input spherical variogram range (build_vario hmaj1/hmin1); "
                               "held fixed across this axis, NOT fitted from the conditioning data",
            source=f"{source_runs[level]['kriging']}/manifest.json:params.variogram.hmaj1",
        ))
        rows.append(dict(
            axis_level=level, density_pct_actual=pct_actual[level], method="sgs",
            length_scale_m=s_hmaj1,
            length_definition="input spherical variogram range (build_vario hmaj1/hmin1); "
                               "held fixed across this axis, NOT fitted from the conditioning data",
            source=f"{source_runs[level]['sgs']}/manifest.json:params.variogram.hmaj1",
        ))

        # --- GP-MLE: read verbatim from the diagnostic CSV -------------------
        gp_row = gp_df[gp_df["axis_level"] == level].iloc[0]
        rows.append(dict(
            axis_level=level, density_pct_actual=pct_actual[level], method="gp_mle",
            length_scale_m=float(gp_row["practical_range_m"]),
            length_definition=(
                f"MLE-fitted sklearn RBF kernel length_scale converted to a practical range "
                f"at the {CORRELATION_CUTOFF} correlation cutoff "
                f"(factor=sqrt(2*ln(20))~=2.4477)"
            ),
            source="results/processed/sample_density_axis/gp_hyperparameter_scale_conversion.csv",
        ))

        # --- RBF+bootstrap: read best_epsilon from the manifest, convert -----
        rbf_params = _manifest_params(source_runs[level]["rbf_bootstrap"])
        kernel = rbf_params["rbf_kernel"]
        if kernel != "gaussian":
            raise ValueError(
                f"level '{level}': rbf_bootstrap manifest records rbf_kernel="
                f"'{kernel}', not 'gaussian' -- the phi(r)=exp(-(epsilon*r)^2) conversion "
                "used here does not apply to a different kernel family."
            )
        epsilon = float(rbf_params["best_epsilon"])
        r_cutoff = float(np.sqrt(-np.log(CORRELATION_CUTOFF)) / epsilon)
        rows.append(dict(
            axis_level=level, density_pct_actual=pct_actual[level], method="rbf_bootstrap",
            length_scale_m=r_cutoff,
            length_definition=(
                f"CV-tuned gaussian RBF interpolation kernel shape parameter epsilon "
                f"converted to an equivalent {CORRELATION_CUTOFF}-cutoff distance "
                f"(r=sqrt(-ln({CORRELATION_CUTOFF}))/epsilon); NOT a fitted spatial "
                "correlation length -- see RBF_CAVEAT"
            ),
            source=f"{source_runs[level]['rbf_bootstrap']}/manifest.json:params.best_epsilon",
        ))

    df = pd.DataFrame(rows)

    print("\n--- Length-scale-by-method inputs, verified from source files ---")
    for level in ALL_AXIS_LEVELS:
        k_val = df[(df.axis_level == level) & (df.method == "kriging")]["length_scale_m"].iloc[0]
        s_val = df[(df.axis_level == level) & (df.method == "sgs")]["length_scale_m"].iloc[0]
        g_val = df[(df.axis_level == level) & (df.method == "gp_mle")]["length_scale_m"].iloc[0]
        r_val = df[(df.axis_level == level) & (df.method == "rbf_bootstrap")]["length_scale_m"].iloc[0]
        rbf_eps = float(_manifest_params(source_runs[level]["rbf_bootstrap"])["best_epsilon"])
        print(
            f"  level {level}% (density={pct_actual[level]:.2f}%): "
            f"kriging/SGS input range={k_val:.1f}m/{s_val:.1f}m, "
            f"GP-MLE practical range={g_val:.2f}m, "
            f"RBF best_epsilon={rbf_eps} -> practical range={r_val:.2f}m"
        )

    return df


# 2026-09-18 extension: GP-MLE / RBF+bootstrap now show the 10-replicate
# spread (sample_seed 1001-1010, reused unchanged from
# src/experiments/sample_replicate_axis.py) at each density level, read from
# results/processed/sample_replicate_axis/length_scale_by_replicate.csv --
# jittered points + mean+-1std errorbar, connected level-to-level by a line
# through the means, matching make_sample_density_figures.py's
# make_metric_vs_density_figure() treatment. kriging/SGS are LEFT EXACTLY AS
# THEY WERE (flat lines, no spread): their range is a FIXED INPUT CONSTANT
# (300 m) at every level and every replicate, never fitted from the
# conditioning data, so there is nothing to average over 10 replicates --
# this is stated explicitly in the caption below, not left to look like an
# oversight that only 2 of 4 lines carry error bars.
REPLICATE_PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"
SPREAD_METHODS = ("gp_mle", "rbf_bootstrap")
FLAT_METHODS = ("kriging", "sgs")
LENGTH_SCALE_JITTER_FRAC = 0.045
LENGTH_SCALE_JITTER_SEED = 13


def make_length_scale_figure(df: pd.DataFrame, pct_actual: dict, n_samples_actual: dict):
    axis_level_floats = [pct_actual[lvl] for lvl in ALL_AXIS_LEVELS]

    replicate_df = pd.read_csv(
        REPLICATE_PROCESSED_DIR / "length_scale_by_replicate.csv", comment="#"
    )
    replicate_df["axis_level"] = replicate_df["axis_level"].astype(str)
    n_replicates = replicate_df["replicate"].nunique()

    rng = np.random.default_rng(LENGTH_SCALE_JITTER_SEED)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # --- kriging / SGS: UNCHANGED, flat lines, no spread (see note above) --
    for method in FLAT_METHODS:
        sub = df[df["method"] == method].copy()
        sub["density_pct_actual"] = sub["axis_level"].map(pct_actual)
        sub = sub.sort_values("density_pct_actual", ascending=False)
        ax.plot(
            sub["density_pct_actual"], sub["length_scale_m"],
            marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method], linewidth=2, markersize=9,
            label=f"{method} (fixed input, no per-replicate spread)", alpha=0.9,
        )

    # --- GP-MLE / RBF+bootstrap: 10-replicate spread ------------------------
    for method in SPREAD_METHODS:
        mean_xs, mean_ys, mean_stds = [], [], []
        for lvl in ALL_AXIS_LEVELS:
            x0 = pct_actual[lvl]
            m_sub = replicate_df[
                (replicate_df["axis_level"] == lvl) & (replicate_df["method"] == method)
            ]
            if len(m_sub) != n_replicates:
                raise ValueError(
                    f"level '{lvl}' {method}: expected {n_replicates} replicate length-scale "
                    f"rows, found {len(m_sub)}."
                )
            jitter = rng.uniform(-LENGTH_SCALE_JITTER_FRAC, LENGTH_SCALE_JITTER_FRAC, size=len(m_sub))
            ax.scatter(
                x0 * (1.0 + jitter), m_sub["length_scale_m"],
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                alpha=0.4, s=30, zorder=2, edgecolors="none", label="_nolegend_",
            )
            mean_xs.append(x0)
            mean_ys.append(float(m_sub["length_scale_m"].mean()))
            mean_stds.append(float(m_sub["length_scale_m"].std(ddof=1)))
        ax.errorbar(
            mean_xs, mean_ys, yerr=mean_stds,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method], markersize=9, markeredgecolor="black",
            markeredgewidth=1.0, capsize=5, linewidth=2, zorder=3,
            label=f"{method} (mean +-1 std, n={n_replicates} replicates)", alpha=0.95,
        )

    truth_range = float(df[df["method"] == "kriging"]["length_scale_m"].iloc[0])
    ax.axhline(
        truth_range, color="gray", linestyle="--", linewidth=1.3,
        label=f"truth input range ({truth_range:g} m)",
    )

    ax.set_xscale("log")
    ax.set_xlim(max(axis_level_floats) * 1.35, min(axis_level_floats) / 1.35)
    ax.set_xticks(axis_level_floats)
    ax.set_xticklabels(
        [f"{pct_actual[lvl]:.2f}%\n(n={n_samples_actual[lvl]})" for lvl in ALL_AXIS_LEVELS]
    )
    ax.minorticks_off()
    ax.set_xlabel("Conditioning sample fraction of the 2500-cell grid (log scale, dense -> sparse)")
    ax.set_ylabel("Length scale (m)")
    ax.set_title(
        "Sample-density axis: each method's length scale vs. sample density\n"
        "(kriging/SGS: fixed input range, no spread; GP-MLE/RBF+bootstrap: 10-replicate "
        "spread -- see caption)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center left")

    fig.text(
        0.5, 0.01, textwrap.fill(
            "kriging and SGS overlap exactly at " + f"{truth_range:g} m at EVERY density "
            "level AND every replicate (both use the same fixed INPUT variogram range, by "
            "one-factor-at-a-time design -- see line styles: kriging solid circles, SGS "
            "dotted squares) -- this is a structural constant, not something fitted from "
            "the conditioning data, so there is nothing to average over the 10 sample-seed "
            "replicates and these two lines deliberately carry NO error bars. GP-MLE and "
            "RBF+bootstrap DO vary per replicate (small jittered points = each of the 10 "
            "replicates' value; solid marker = mean +-1 std, ddof=1, across them -- see "
            "results/processed/sample_replicate_axis/length_scale_by_replicate.csv). "
            + RBF_CAVEAT,
            150,
        ),
        ha="center", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.1, bottom=0.28, right=0.97, top=0.86)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "00_length_scale_vs_density.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"00_length_scale_vs_density.png: {out} ({out.stat().st_size} bytes)")
    return out


# ---------------------------------------------------------------------------
# (b) Experimental (subsampled) variogram of the shared ground-truth field
# ---------------------------------------------------------------------------
# New seed, distinct from every seed already used on this axis (TRUTH_SEED=
# 101, SAMPLE_SEED=20, NULL_SEED_START..+300=1000-1300, and the CV/bootstrap
# seeds inside individual method manifests -- 40/30/50/55/60/70/80/85 have
# also been used elsewhere in this project's other experiments).
VARIOGRAM_PAIR_SEED = 90
N_PAIRS_DRAWN = 200_000
LAG_BIN_WIDTH_M = 25.0
LAG_MAX_M = 750.0
MIN_PAIRS_PER_BIN = 30


def compute_experimental_variogram(truth: np.ndarray) -> pd.DataFrame:
    """Subsampled (NOT exhaustive) experimental variogram of ``truth``.

    Draws N_PAIRS_DRAWN random (i, j) cell-index pairs (i != j) from a fixed
    RandomState(VARIOGRAM_PAIR_SEED), computes each pair's lag and
    semivariance contribution, and aggregates into LAG_BIN_WIDTH_M-wide bins.
    """
    coords = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)
    z = truth.ravel()
    n_cells = coords.shape[0]
    if z.shape[0] != n_cells:
        raise ValueError(
            f"truth has {z.shape[0]} cells but full_grid_coordinates produced {n_cells} -- "
            "grid mismatch, pairs would not line up with the right coordinates."
        )

    rng = np.random.RandomState(VARIOGRAM_PAIR_SEED)
    idx_i = rng.randint(0, n_cells, N_PAIRS_DRAWN)
    idx_j = rng.randint(0, n_cells, N_PAIRS_DRAWN)
    keep = idx_i != idx_j
    n_dropped = int((~keep).sum())
    idx_i, idx_j = idx_i[keep], idx_j[keep]
    print(
        f"Drew {N_PAIRS_DRAWN} candidate (i,j) pairs at VARIOGRAM_PAIR_SEED={VARIOGRAM_PAIR_SEED}; "
        f"dropped {n_dropped} self-pairs (i==j); {idx_i.shape[0]} pairs retained "
        f"(exhaustive would be {n_cells * (n_cells - 1) // 2} unique pairs)."
    )

    h = np.sqrt(np.sum((coords[idx_i] - coords[idx_j]) ** 2, axis=1))
    gamma_contrib = 0.5 * (z[idx_i] - z[idx_j]) ** 2

    edges = np.arange(0.0, LAG_MAX_M + LAG_BIN_WIDTH_M, LAG_BIN_WIDTH_M)
    bin_idx = np.digitize(h, edges) - 1  # 0-based bin index

    rows = []
    n_bins = len(edges) - 1
    for b in range(n_bins):
        in_bin = bin_idx == b
        n_pairs = int(in_bin.sum())
        center = 0.5 * (edges[b] + edges[b + 1])
        if n_pairs == 0:
            print(f"  WARNING: lag bin centered at {center:.1f} m has 0 pairs -- omitted from the table.")
            continue
        semivariance_empirical = float(gamma_contrib[in_bin].mean())
        if n_pairs < MIN_PAIRS_PER_BIN:
            print(
                f"  WARNING: lag bin centered at {center:.1f} m has only {n_pairs} pairs "
                f"(< MIN_PAIRS_PER_BIN={MIN_PAIRS_PER_BIN}) -- flagged, drawn faded in the figure."
            )
        rows.append({
            "lag_bin_center_m": center,
            "n_pairs": n_pairs,
            "semivariance_empirical": semivariance_empirical,
        })

    df = pd.DataFrame(rows)
    df["semivariance_theoretical"] = spherical_semivariance(df["lag_bin_center_m"].values)
    return df


def spherical_semivariance(h) -> np.ndarray:
    """Theoretical spherical semivariance at the truth's own parameters
    (imported from src.experiments.base_case, never hardcoded here): nugget
    = NUG * POR_STDEV**2, structured sill = CC1 * POR_STDEV**2, range =
    HMAJ1. See the module docstring's affine-linearity derivation."""
    h = np.asarray(h, dtype=float)
    nugget = NUG * POR_STDEV ** 2
    structured_sill = CC1 * POR_STDEV ** 2
    rng = HMAJ1
    hr = np.clip(h / rng, 0.0, None)
    spherical_part = np.where(
        h < rng,
        1.5 * hr - 0.5 * hr ** 3,
        1.0,
    )
    return nugget + structured_sill * spherical_part


def make_experimental_variogram_figure(df: pd.DataFrame, truth: np.ndarray):
    total_sill = POR_STDEV ** 2

    fig, ax = plt.subplots(figsize=(9, 6.5))
    well_sampled = df[df["n_pairs"] >= MIN_PAIRS_PER_BIN]
    sparse = df[df["n_pairs"] < MIN_PAIRS_PER_BIN]

    sizes = 15 + 60 * (well_sampled["n_pairs"] / well_sampled["n_pairs"].max())
    ax.scatter(
        well_sampled["lag_bin_center_m"], well_sampled["semivariance_empirical"],
        s=sizes, color="tab:blue", alpha=0.75, edgecolors="k", linewidths=0.3,
        label="experimental (subsampled), marker size ~ n_pairs",
    )
    if len(sparse) > 0:
        ax.scatter(
            sparse["lag_bin_center_m"], sparse["semivariance_empirical"],
            s=25, color="tab:blue", alpha=0.25, marker="x",
            label=f"experimental, n_pairs<{MIN_PAIRS_PER_BIN} (unreliable)",
        )

    h_smooth = np.linspace(0, LAG_MAX_M, 300)
    ax.plot(
        h_smooth, spherical_semivariance(h_smooth), color="tab:red", linewidth=2,
        label="theoretical spherical model",
    )
    ax.axhline(total_sill, color="gray", linestyle="--", linewidth=1, label=f"sill ({total_sill:g})")

    ax.set_xlabel("Lag h (m)")
    ax.set_ylabel("Semivariance (Porosity %$^2$)")
    ax.set_title(
        f"Ground-truth experimental variogram (subsampled, not exhaustive)\n"
        f"N_PAIRS_DRAWN={N_PAIRS_DRAWN:,}, VARIOGRAM_PAIR_SEED={VARIOGRAM_PAIR_SEED}, "
        f"truth_seed={TRUTH_SEED}",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    truth_mean, truth_var = float(np.mean(truth)), float(np.var(truth))
    fig.text(
        0.5, 0.01, textwrap.fill(
            f"{N_PAIRS_DRAWN:,} random cell-index pairs drawn from the 2500-cell grid "
            f"(exhaustive would be {NX * NY * (NX * NY - 1) // 2:,} unique pairs), binned into "
            f"{LAG_BIN_WIDTH_M:g} m lag bins from 0-{LAG_MAX_M:g} m. Theoretical curve uses "
            f"nugget={NUG * POR_STDEV**2:g}, structured sill={CC1 * POR_STDEV**2:g}, range="
            f"{HMAJ1:g} m (truth parameters, not fitted). Truth field empirical mean/variance = "
            f"{truth_mean:.3f}/{truth_var:.3f} (targets: {POR_MEAN:g}/{POR_STDEV**2:g} -- these "
            "match to floating-point precision by construction of GSLIB.affine, which rescales "
            "the realization to hit the target mean/stdev exactly; this is not an independent "
            "check of spatial structure, only confirmation the affine step ran as intended).",
            150,
        ),
        ha="center", fontsize=7.5, color="dimgray",
    )
    plt.subplots_adjust(left=0.1, bottom=0.20, right=0.97, top=0.84)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "00_experimental_variogram.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"00_experimental_variogram.png: {out} ({out.stat().st_size} bytes)")
    return out


def main():
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)
    with open(PROCESSED_DIR / "sample_overlap.json", "r", encoding="utf-8") as f:
        overlap = json.load(f)
    pct_actual = {lvl: float(overlap["sample_fraction_actual_pct"][lvl]) for lvl in ALL_AXIS_LEVELS}
    n_samples_actual = {lvl: int(overlap["n_samples_actual"][lvl]) for lvl in ALL_AXIS_LEVELS}

    saved = []

    # --- (a) length scale vs. density --------------------------------------
    length_df = build_length_scale_table(source_runs, pct_actual)
    length_csv = PROCESSED_DIR / "length_scale_by_method.csv"
    length_df.to_csv(length_csv, index=False)
    print(f"\nlength_scale_by_method.csv: {length_csv}")
    saved.append(make_length_scale_figure(length_df, pct_actual, n_samples_actual))

    # --- (b) experimental variogram of the shared truth field --------------
    truth = get_base_case_truth(truth_seed=TRUTH_SEED)  # HMAJ1/HMIN1 default to base-case 300 m
    truth_mean, truth_var = float(np.mean(truth)), float(np.var(truth))
    print(
        f"\nTruth field sanity check (affine correction, single {NX}x{NY} realization, "
        f"truth_seed={TRUTH_SEED}): empirical mean={truth_mean:.4f} (target {POR_MEAN:g}), "
        f"empirical variance={truth_var:.4f} (target {POR_STDEV**2:g})"
    )

    vario_df = compute_experimental_variogram(truth)
    vario_csv = PROCESSED_DIR / "experimental_variogram_truth.csv"
    vario_df.to_csv(vario_csv, index=False)
    print(f"experimental_variogram_truth.csv: {vario_csv}")
    saved.append(make_experimental_variogram_figure(vario_df, truth))

    print("\nAll length-scale / experimental-variogram figures:")
    for f in saved:
        print(" ", f)
    return saved


if __name__ == "__main__":
    main()
