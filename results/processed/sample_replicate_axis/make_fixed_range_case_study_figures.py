"""Case-study figures for the two FIXED-RANGE methods (simple kriging and SGS)
within the sample-seed replicate axis (src/experiments/sample_replicate_axis.py,
src/experiments/evaluate_sample_replicate_axis.py), at the 1% sample-density
level (axis_level="1", n_samples_requested=25; all 10 replicates share the
SAME ground-truth field: TRUTH_SEED=101, range=300 m).

WHY ONE BLOCK PER METHOD (and not a MIN/MAX pair)
--------------------------------------------------
make_length_case_study_figures.py pulls out, for GP-MLE and RBF+bootstrap, the
replicates with the SHORTEST and LONGEST fitted length scale, because those two
methods FIT a length-scale-like quantity from the conditioning samples and it
swings by 5x-11x with sample placement alone. Simple kriging and SGS have no
such quantity: their variogram range (300 m, the truth's own range) is a
FIXED INPUT CONSTANT at every level and every replicate, never fitted from the
data (see the evaluate_sample_replicate_axis.py module docstring, and the
header note of length_scale_by_replicate.csv, which deliberately has no
kriging/SGS rows). There is therefore no "MIN vs. MAX length" replicate pair to
select -- ONE figure per method, containing ONE tinted 2x4 replicate block, is
shown. The block's layout, panel styles, fonts, DPI, footnote convention,
recomputation-and-assert pattern against metrics.csv and the COLOR/AXIS-RANGE
rules are REUSED from make_length_case_study_figures.py (its helpers are
IMPORTED, nothing is copy-pasted, and that script is untouched).

WHY rep0
--------
CASE_STUDY_REPLICATE = "rep0" (sample_seed=1001). Nothing else selects it here
(there is no fitted quantity to rank replicates by); rep0 was chosen because it
is the replicate that is COMMON to the GP-MLE case study (its MIN-length
replicate) and to the RBF+bootstrap case study (its MIN-length replicate after
the 2026-09-21 17-value smoothing-grid re-run), so all four case-study figures
share at least one sample-location set and the same truth field. It is a single
named constant so it is easy to change; every check below (samples.csv match,
UMG/MSE match against metrics.csv) is done for whichever replicate is set.

Block layout (identical to the existing figures' blocks)
--------------------------------------------------------
  ROW A (map panels):        1. truth + samples   2. [realization slot]
                             3. prediction map    4. predictive variance
  ROW B (diagnostic panels): 5. UMG plot          6. accuracy crossplot
                             7. variogram reproduction
                             8. distribution reproduction

Per-method arrays -- the SAME arrays evaluate_sample_replicate_axis.py's
metrics use (source runs from source_runs.json, level "1", rep0):

  kriging
    panel 3 (prediction map) and panel 6 (crossplot point estimate):
        kriging_mean_map_physical.npy (the MSE array).
    panel 4 (variance): kriging_var_map_physical_mc.npy (the variance_mean
        array; a Monte Carlo back-transform approximation of the normal-score
        predictive variance, NOT a native physical-unit variance).
    panel 5 (UMG): recomputed EXACTLY as evaluate_sample_replicate_axis.py does
        -- kmap_ns.npy + kriging_var_map_ns.npy (normal-score space) through
        src.evaluation.kriging_fraction_in, the exact per-cell quantile
        back-transform using nscore_transform_table.csv and the run's
        BACKTR_*/LTAIL/UTAIL constants -- NOT accuracy_plot_fraction_in on the
        physical-unit maps.
    panel 2: kriging produces a SINGLE estimate and NO realization, so there is
        no example-realization array. To keep the 4 map columns aligned with
        the other case-study figures (truth+samples | [realization slot] |
        prediction map | variance) the realization slot holds a plain TEXT panel
        saying so, and the variogram / distribution panels show the truth and
        the kriging estimate only (plus the conditioning samples in the
        distribution panel). Nothing is fabricated for the missing curve.
  sgs
    panel 2 (example realization): sgs_realizations.npy[0], the first stored
        realization (same choice make_sample_density_figures.py's
        make_predictions_figure_set() already uses for its QC panel).
    panel 3 (prediction map) and panel 6 (crossplot point estimate):
        sgs_mean_map.npy (the ensemble mean, which is also the MSE array).
    panel 4 (variance): sgs_var_map.npy (the variance_mean array).
    panel 5 (UMG): accuracy_plot_fraction_in(truth, sgs_mean, sqrt(sgs_var)),
        as evaluate_sample_replicate_axis.py does for sgs.

DECISIVE CORRECTNESS CHECK (same pattern as make_length_case_study_figures.py):
for the (method, replicate) cell this script (a) regenerates the replicate's
conditioning samples from its sample_seed and cross-checks them against the
run's own samples.csv (atol=1e-10), (b) recomputes UMG and requires it to match
(np.isclose) metrics.csv's (axis_level="1", replicate, method, "umg") row, (c)
recomputes MSE and requires it to match metrics.csv's "mse" row, and (d) also
recomputes variance_mean over the evaluated cells and requires it to match the
"variance_mean" row (an extra check beyond the existing script). Any failure
raises; nothing is plotted from a value that disagrees with the pinned output.

AXIS RANGES (so all four case-study figures are visually comparable)
--------------------------------------------------------------------
The GP-MLE and RBF+bootstrap figures set their variogram / distribution panel
axis ranges from their OWN (two-block) data. To make the new figures directly
comparable, the same limits are RECOMPUTED here (not hardcoded) by re-running
that script's own loaders on the GP-MLE and RBF+bootstrap length-extreme
replicates and applying the same formulas: the variogram y-range is identical
for the two existing figures; the distribution x-range differs slightly between
them, so the UNION of the two is used for both new figures (a judgment-free
tie-break, reported to the orchestrator). If a kriging/SGS map's values do not
fit inside those limits (an SGS realization has full variance, so it can), the
axis is WIDENED for that figure only (never clipped), with the same 3% padding
rule; this is stated in the figure footnote and printed. UMG [0,1]^2 and the
crossplot's truth-range+5% limits are the same as in the existing figures.
Porosity color scale is anchored on the truth's own [min, max] (as before);
variance color scale is [0, this map's max] (this figure only; stated under the
panel), as the existing per-block variance scale.

Stdout also prints, for each plotted map, the variogram value at the ~287.5 m
bin (as % of the theoretical model) and the mean / std (ddof=1) over the 2475
evaluated cells; these numbers feed a table on the published page.

Seeds recorded: TRUTH_SEED (truth), the replicate's sample_seed, and
VARIOGRAM_PAIR_SEED (the shared estimator's pair-drawing seed) are printed and
written on the figures. No randomness is introduced by this script itself; the
raw runs are only READ.

Run with:
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_fixed_range_case_study_figures
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

from src.evaluation import (  # noqa: E402
    accuracy_plot_fraction_in,
    calc_umg,
    conditioning_cell_mask,
    kriging_fraction_in,
    mse,
)
from src.experiments.base_case import (  # noqa: E402
    NX, NY, XMN, YMN, XSIZ, YSIZ, CC1, HMAJ1, NUG, POR_STDEV,
)
from src.experiments.base_case_conditioning import (  # noqa: E402
    VCOL,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.evaluate_sample_replicate_axis import (  # noqa: E402
    VARIANCE_SOURCE_FILES,
    safe_sqrt_variance,
)
from src.experiments.kriging import (  # noqa: E402
    BACKTR_ZMAX, BACKTR_ZMIN, LTAIL, LTPAR, UTAIL, UTPAR,
)
from src.experiments.sample_replicate_axis import (  # noqa: E402
    AXIS_HMAJ1,
    N_SAMPLES_REQUESTED_BY_LEVEL,
    REPLICATE_SEED,
    TRUTH_SEED,
)

# Everything below is IMPORTED from the existing case-study script (which is
# unchanged): constants, the panel/CDF/variogram helpers, the tolerance
# constants and the loaders used to recompute the existing figures' axis ranges.
from results.processed.sample_replicate_axis import make_length_case_study_figures as base  # noqa: E402
from results.processed.sample_replicate_axis.make_length_case_study_figures import (  # noqa: E402
    AXIS_LEVEL,
    BLOCK_BAND_FACECOLORS,
    BLOCK_GRID_KWARGS,
    CURVE_COLORS,
    DIST_STD_DDOF,
    FIG_DPI,
    LAG_BIN_WIDTH_M,
    LAG_MAX_M,
    MIN_PAIRS_PER_BIN,
    MSE_MATCH_ATOL,
    MSE_MATCH_RTOL,
    N_PAIRS_DRAWN,
    PROCESSED_DIR,
    SAMPLES_MATCH_ATOL,
    UMG_MATCH_ATOL,
    UMG_MATCH_RTOL,
    VARIOGRAM_PAIR_SEED,
    VARIOGRAM_REFERENCE_LAG_FRACTIONS,
    _curve_stats,
    _empirical_cdf,
    _panel,
    _variogram_at_reference_lags,
    compute_map_variogram,
    spherical_semivariance,
)

FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "sample_replicate_axis"

# The ONE replicate shown per method (see module docstring). Change here only.
CASE_STUDY_REPLICATE = "rep0"

FIXED_RANGE_METHODS = ("kriging", "sgs")
METHOD_LABELS = {"kriging": "Simple kriging", "sgs": "SGS"}
METHOD_COLORS = {"kriging": "tab:blue", "sgs": "tab:green"}  # project-wide method colors
# Same tolerance as the metrics.csv checks for the extra variance_mean check.
VARIANCE_MATCH_RTOL = 1e-8
VARIANCE_MATCH_ATOL = 1e-8

# One block on a taller-than-needed canvas: panel size matches the existing
# 21 x 22.5 in two-block figures (block height 0.355 * 22.5 in ~= 8.0 in).
FIG_SIZE = (21.0, 12.6)
_BLOCK_HEIGHT_IN = 0.355 * 22.5
GS_TOP = 0.855
GS_BOTTOM = GS_TOP - _BLOCK_HEIGHT_IN / FIG_SIZE[1]
BAND_PAD_TOP = 0.036 * 22.5 / FIG_SIZE[1]
BAND_PAD_BOTTOM = 0.058 * 22.5 / FIG_SIZE[1]
BLOCK_TITLE_OFFSET = 0.013 * 22.5 / FIG_SIZE[1]


def existing_case_study_axis_limits(source_runs, metrics_df, truth, truth_vario):
    """Recompute (not hardcode) the variogram y-range and distribution x-range
    the existing GP-MLE / RBF+bootstrap case-study figures use, by running
    their own loaders and formulas. Returns (vario_ylim, dist_xlim, detail)."""
    length_df = pd.read_csv(PROCESSED_DIR / "length_scale_by_replicate.csv", comment="#")
    h_smooth = np.linspace(0.0, LAG_MAX_M, 300)
    model_smooth = spherical_semivariance(h_smooth)
    detail, y_his, x_los, x_his = {}, [], [], []
    for method in ("gp_mle", "rbf_bootstrap"):
        ext = base.identify_length_extremes(length_df, method)
        rows = [
            base.load_case_study_row(method, role, ext[key], truth, source_runs, metrics_df)
            for key, role in (("min", "MIN"), ("max", "MAX"))
        ]
        stack = [truth_vario["semivariance_empirical"].values, model_smooth]
        for rd in rows:
            stack.append(rd["vario_example"]["semivariance_empirical"].values)
            stack.append(rd["vario_mean"]["semivariance_empirical"].values)
        y_hi = float(np.max(np.concatenate(stack))) * 1.10
        dist_all = np.concatenate([
            np.asarray(a, dtype=float).ravel()
            for rd in rows
            for a in (rd["truth_masked"], rd["example_masked"], rd["mean_masked"],
                      rd["sample_values"])
        ])
        pad = 0.03 * (float(dist_all.max()) - float(dist_all.min()))
        x_lo, x_hi = float(dist_all.min()) - pad, float(dist_all.max()) + pad
        detail[method] = {"vario_ymax": y_hi, "dist_xlim": (x_lo, x_hi)}
        y_his.append(y_hi)
        x_los.append(x_lo)
        x_his.append(x_hi)
    print(f"  existing case-study axis ranges (recomputed): {detail}")
    return (0.0, max(y_his)), (min(x_los), max(x_his)), detail


def _pinned(metrics_df, replicate_id, method, metric):
    sel = metrics_df.loc[
        (metrics_df["axis_level"].astype(str) == AXIS_LEVEL)
        & (metrics_df["replicate"] == replicate_id)
        & (metrics_df["method"] == method)
        & (metrics_df["metric"] == metric),
        "value",
    ]
    if len(sel) != 1:
        raise ValueError(
            f"expected exactly 1 pinned {metric} row for {method} {replicate_id} at "
            f"axis_level={AXIS_LEVEL}; found {len(sel)}."
        )
    return float(sel.iloc[0])


def load_fixed_range_row(method, replicate_id, truth, source_runs, metrics_df):
    """Load + verify everything the one block of one method's figure needs.
    Raises on any samples.csv / UMG / MSE / variance_mean mismatch."""
    sample_seed = int(REPLICATE_SEED[replicate_id])
    n_requested = N_SAMPLES_REQUESTED_BY_LEVEL[AXIS_LEVEL]
    run_dir = _REPO_ROOT / source_runs[AXIS_LEVEL][replicate_id][method]

    # (a) samples cross-check
    samples_df = get_conditioning_samples(truth, sample_seed=sample_seed, n_samples=n_requested)
    recorded = pd.read_csv(run_dir / "samples.csv")
    if len(recorded) != len(samples_df) or not np.allclose(
        recorded[["X", "Y", VCOL]].values, samples_df[["X", "Y", VCOL]].values,
        rtol=0.0, atol=SAMPLES_MATCH_ATOL,
    ):
        raise ValueError(
            f"{method} {replicate_id}: regenerated conditioning samples (sample_seed="
            f"{sample_seed}, n={n_requested}) do not match {run_dir}/samples.csv."
        )
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    truth_masked = truth[mask]

    var_map = np.load(run_dir / VARIANCE_SOURCE_FILES[method])
    if method == "kriging":
        mean_map = np.load(run_dir / "kriging_mean_map_physical.npy")
        point_estimate_map = mean_map
        example_map = None  # kriging produces no realization
        # UMG: exact quantile back-transform path (as evaluate_sample_replicate_axis.py).
        kmap_ns = np.load(run_dir / "kmap_ns.npy")
        vmap_ns = np.load(run_dir / "kriging_var_map_ns.npy")
        table = pd.read_csv(run_dir / "nscore_transform_table.csv")
        std_ns_masked = safe_sqrt_variance(vmap_ns[mask], "kriging vmap_ns")
        p_intervals, fraction_in = kriging_fraction_in(
            truth_masked, kmap_ns[mask], std_ns_masked, table["vr"].values, table["vrg"].values,
            BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR,
        )
        mean_label = "Simple kriging -- point estimate"
        var_label = "Simple kriging -- variance (physical units, Monte Carlo approx.)"
        example_label = None
        example_short_label = None
        mean_short_label = "kriging point estimate (panel 3)"
        point_estimate_label = "Simple kriging point estimate"
    else:  # sgs
        realizations = np.load(run_dir / "sgs_realizations.npy")
        example_map = realizations[0]
        mean_map = np.load(run_dir / "sgs_mean_map.npy")
        point_estimate_map = mean_map
        std_masked = safe_sqrt_variance(var_map[mask], "sgs variance map")
        p_intervals, fraction_in = accuracy_plot_fraction_in(
            truth_masked, mean_map[mask], std_masked
        )
        example_label = f"SGS -- example realization (#0 of {realizations.shape[0]}, {replicate_id})"
        mean_label = "SGS -- ensemble mean"
        var_label = "SGS -- ensemble variance"
        example_short_label = "SGS realization #0 (panel 2)"
        mean_short_label = "SGS ensemble mean (panel 3)"
        point_estimate_label = "SGS ensemble mean"
    umg_recomputed = calc_umg(p_intervals, fraction_in)
    umg_pinned = _pinned(metrics_df, replicate_id, method, "umg")
    if not np.isclose(umg_recomputed, umg_pinned, rtol=UMG_MATCH_RTOL, atol=UMG_MATCH_ATOL):
        raise ValueError(
            f"{method} {replicate_id}: recomputed UMG ({umg_recomputed}) != metrics.csv "
            f"({umg_pinned})."
        )

    point_estimate_masked = point_estimate_map[mask]
    mse_recomputed = mse(truth_masked, point_estimate_masked)
    mse_pinned = _pinned(metrics_df, replicate_id, method, "mse")
    if not np.isclose(mse_recomputed, mse_pinned, rtol=MSE_MATCH_RTOL, atol=MSE_MATCH_ATOL):
        raise ValueError(
            f"{method} {replicate_id}: recomputed MSE ({mse_recomputed}) != metrics.csv "
            f"({mse_pinned})."
        )

    variance_mean_recomputed = float(np.mean(var_map[mask]))
    variance_mean_pinned = _pinned(metrics_df, replicate_id, method, "variance_mean")
    if not np.isclose(variance_mean_recomputed, variance_mean_pinned,
                      rtol=VARIANCE_MATCH_RTOL, atol=VARIANCE_MATCH_ATOL):
        raise ValueError(
            f"{method} {replicate_id}: recomputed variance_mean ({variance_mean_recomputed}) "
            f"!= metrics.csv ({variance_mean_pinned})."
        )

    # Panel 7/8 inputs (same arrays panels 2/3 display).
    vario_mean = compute_map_variogram(mean_map, f"{method} {replicate_id} prediction map")
    vario_example = (
        compute_map_variogram(example_map, f"{method} {replicate_id} example realization")
        if example_map is not None else None
    )
    mean_masked = mean_map[mask]
    example_masked = example_map[mask] if example_map is not None else None
    sample_values = samples_df[VCOL].values.astype(float)
    dist_stats = {
        "truth": _curve_stats(truth_masked),
        "mean": _curve_stats(mean_masked),
        "samples": _curve_stats(sample_values),
    }
    if example_masked is not None:
        dist_stats["example"] = _curve_stats(example_masked)

    return {
        "method": method,
        "replicate_id": replicate_id,
        "sample_seed": sample_seed,
        "samples_df": samples_df,
        "example_map": example_map,
        "example_label": example_label,
        "example_short_label": example_short_label,
        "mean_map": mean_map,
        "mean_label": mean_label,
        "mean_short_label": mean_short_label,
        "point_estimate_label": point_estimate_label,
        "var_map": var_map,
        "var_label": var_label,
        "var_max": float(np.max(var_map)),
        "p_intervals": p_intervals,
        "fraction_in": fraction_in,
        "umg_recomputed": umg_recomputed,
        "umg_pinned": umg_pinned,
        "truth_masked": truth_masked,
        "point_estimate_masked": point_estimate_masked,
        "mse_recomputed": mse_recomputed,
        "mse_pinned": mse_pinned,
        "variance_mean_recomputed": variance_mean_recomputed,
        "variance_mean_pinned": variance_mean_pinned,
        "n_evaluated_cells": int(mask.sum()),
        "vario_mean": vario_mean,
        "vario_example": vario_example,
        "mean_masked": mean_masked,
        "example_masked": example_masked,
        "sample_values": sample_values,
        "dist_stats": dist_stats,
    }


def make_fixed_range_figure(method, truth, source_runs, metrics_df, truth_vario,
                            vario_ylim, dist_xlim):
    rd = load_fixed_range_row(method, CASE_STUDY_REPLICATE, truth, source_runs, metrics_df)
    label = METHOD_LABELS[method]
    color = METHOD_COLORS[method]
    has_example = rd["example_map"] is not None
    n_samples = len(rd["samples_df"])

    porosity_vmin, porosity_vmax = float(np.min(truth)), float(np.max(truth))
    crossplot_pad = 0.05 * (porosity_vmax - porosity_vmin)
    crossplot_lims = (porosity_vmin - crossplot_pad, porosity_vmax + crossplot_pad)

    # Variogram like-for-like check (same bins / pair counts as the truth curve).
    curves = [("prediction map", rd["vario_mean"])]
    if has_example:
        curves.append(("example realization", rd["vario_example"]))
    for what, vdf in curves:
        if not (np.array_equal(vdf["lag_bin_center_m"].values, truth_vario["lag_bin_center_m"].values)
                and np.array_equal(vdf["n_pairs"].values, truth_vario["n_pairs"].values)):
            raise ValueError(
                f"{method}: the {what} variogram was not computed on the same lag bins / pair "
                "set as the truth variogram."
            )
    t_refs = _variogram_at_reference_lags(truth_vario)
    m_refs = _variogram_at_reference_lags(rd["vario_mean"])
    e_refs = _variogram_at_reference_lags(rd["vario_example"]) if has_example else None
    vario_reference = []
    for k, frac in enumerate(VARIOGRAM_REFERENCE_LAG_FRACTIONS):
        entry = {
            "frac_of_range": frac,
            "bin_center_m": t_refs[k]["bin_center_m"],
            "theoretical": t_refs[k]["theoretical"],
            "truth": t_refs[k]["empirical"],
            "mean": m_refs[k]["empirical"],
            "example": e_refs[k]["empirical"] if has_example else None,
        }
        vario_reference.append(entry)

    # The data must fit inside the (existing-figure) distribution / variogram limits.
    dist_arrays = [rd["truth_masked"], rd["mean_masked"], rd["sample_values"]]
    if has_example:
        dist_arrays.append(rd["example_masked"])
    dmin = min(float(np.min(a)) for a in dist_arrays)
    dmax = max(float(np.max(a)) for a in dist_arrays)
    axis_range_note = ""
    if dmin < dist_xlim[0] or dmax > dist_xlim[1]:
        # Widen (never clip) using the same 3% padding rule as the existing figures.
        pad = 0.03 * (dmax - dmin)
        widened = (min(dist_xlim[0], dmin - pad), max(dist_xlim[1], dmax + pad))
        axis_range_note += (
            f" NOTE: this figure's plotted porosity values [{dmin:.2f}, {dmax:.2f}] exceed the "
            f"GP-MLE/RBF+bootstrap distribution range {dist_xlim[0]:.2f}-{dist_xlim[1]:.2f}, "
            f"so the distribution x-axis is widened to {widened[0]:.2f}-{widened[1]:.2f} "
            "(same 3% padding rule)."
        )
        print(f"    widening distribution x-range {dist_xlim} -> {widened} to fit {method} data")
        dist_xlim = widened
    vario_max = max(float(np.max(v["semivariance_empirical"].values)) for _, v in curves)
    if vario_max > vario_ylim[1]:
        widened_y = (vario_ylim[0], vario_max * 1.10)
        axis_range_note += (
            f" NOTE: a variogram value ({vario_max:.2f}) exceeds the GP-MLE/RBF+bootstrap "
            f"y-limit, so the y-axis is widened to 0-{widened_y[1]:.2f}."
        )
        print(f"    widening variogram y-range {vario_ylim} -> {widened_y}")
        vario_ylim = widened_y

    h_smooth = np.linspace(0.0, LAG_MAX_M, 300)
    model_smooth = spherical_semivariance(h_smooth)

    fig = plt.figure(figsize=FIG_SIZE)
    fig.patches.append(plt.Rectangle(
        (0.008, GS_BOTTOM - BAND_PAD_BOTTOM), 0.984,
        (GS_TOP + BAND_PAD_TOP) - (GS_BOTTOM - BAND_PAD_BOTTOM),
        transform=fig.transFigure, facecolor=BLOCK_BAND_FACECOLORS[0],
        edgecolor="0.55", linewidth=1.2, zorder=-1,
    ))
    gs = fig.add_gridspec(2, 4, top=GS_TOP, bottom=GS_BOTTOM, **BLOCK_GRID_KWARGS)
    ax_truth = fig.add_subplot(gs[0, 0])
    ax_example = fig.add_subplot(gs[0, 1])
    ax_mean = fig.add_subplot(gs[0, 2])
    ax_var = fig.add_subplot(gs[0, 3])
    ax_umg = fig.add_subplot(gs[1, 0])
    ax_crossplot = fig.add_subplot(gs[1, 1])
    ax_vario = fig.add_subplot(gs[1, 2])
    ax_dist = fig.add_subplot(gs[1, 3])

    tag = f"[{CASE_STUDY_REPLICATE}]"
    fig.text(
        0.5, GS_TOP + BLOCK_TITLE_OFFSET,
        f"{tag} (sample_seed={rd['sample_seed']}, n={n_samples} conditioning samples) -- "
        f"variogram range = {AXIS_HMAJ1:g} m (fixed input, not fitted)  |  "
        f"UMG={rd['umg_recomputed']:.4f}, MSE={rd['mse_recomputed']:.4f}  |  top sub-row: "
        "maps, bottom sub-row: diagnostics",
        ha="center", va="bottom", fontsize=13, fontweight="bold", color="black",
    )

    _panel(
        ax_truth, truth,
        f"{tag} Truth + samples\n{rd['replicate_id']} (sample_seed={rd['sample_seed']}, "
        f"n={n_samples}); range={AXIS_HMAJ1:g} m (fixed)",
        porosity_vmin, porosity_vmax, "viridis", "Porosity (%)", rd["samples_df"],
    )
    if has_example:
        _panel(
            ax_example, rd["example_map"], rd["example_label"],
            porosity_vmin, porosity_vmax, "viridis", "Porosity (%)", rd["samples_df"],
        )
    else:
        ax_example.set_axis_off()
        ax_example.set_title("Simple kriging -- realization slot", fontsize=9.5)
        ax_example.text(
            0.5, 0.5,
            "Simple kriging produces a single\nestimate (panel 3), no realization.\n\n"
            "Kriging returns the conditional mean\nand variance at each cell; it does\n"
            "not simulate a field, so there is no\nexample realization to show here.\n"
            "Panels 7-8 therefore compare the\ntruth with the estimate only.",
            transform=ax_example.transAxes, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
        )
    _panel(
        ax_mean, rd["mean_map"], rd["mean_label"],
        porosity_vmin, porosity_vmax, "viridis", "Porosity (%)", rd["samples_df"],
    )
    _panel(
        ax_var, rd["var_map"], rd["var_label"], 0.0, rd["var_max"], "magma",
        "Variance (Porosity %^2)", rd["samples_df"], scatter_color="cyan",
    )
    ax_var.text(
        0.02, -0.16,
        f"Variance scale: [0, {rd['var_max']:.2f}] %^2 (THIS FIGURE ONLY -- NOT shared with "
        "the other case-study figures; see module docstring).",
        transform=ax_var.transAxes, fontsize=7, color="dimgray", wrap=True,
    )

    ax_umg.plot(
        rd["p_intervals"], rd["fraction_in"], marker="o", markersize=4, color=color,
        label=f"{label} (UMG={rd['umg_recomputed']:.3f})",
    )
    ax_umg.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", label="ideal (y=x)")
    ax_umg.set_xlim(0.0, 1.0)
    ax_umg.set_ylim(0.0, 1.0)
    ax_umg.set_xlabel("Nominal probability interval")
    ax_umg.set_ylabel("Fraction of truth in interval")
    ax_umg.set_title(f"{tag} UMG plot (UMG={rd['umg_recomputed']:.3f})")
    ax_umg.legend(loc="upper left", fontsize=8)
    ax_umg.grid(alpha=0.3)

    ax_crossplot.scatter(
        rd["truth_masked"], rd["point_estimate_masked"], s=10, alpha=0.5, color=color,
        edgecolors="none",
    )
    ax_crossplot.plot(crossplot_lims, crossplot_lims, color="gray", linestyle="--", label="1:1")
    ax_crossplot.set_xlim(crossplot_lims)
    ax_crossplot.set_ylim(crossplot_lims)
    ax_crossplot.set_xlabel("Truth (Porosity %)")
    ax_crossplot.set_ylabel(f"{rd['point_estimate_label']} (Porosity %)")
    ax_crossplot.set_title(f"{tag} Accuracy crossplot (MSE={rd['mse_recomputed']:.3f})")
    ax_crossplot.legend(loc="upper left", fontsize=8)
    ax_crossplot.grid(alpha=0.3)
    ax_crossplot.set_aspect("equal", adjustable="box")

    # ---- 7. Variogram reproduction ----
    vario_curves = [("truth field", truth_vario, CURVE_COLORS["truth"], "o", "-")]
    if has_example:
        vario_curves.append(
            (rd["example_short_label"], rd["vario_example"], CURVE_COLORS["example"], "s", "-")
        )
    vario_curves.append((rd["mean_short_label"], rd["vario_mean"], CURVE_COLORS["mean"], "^", "-"))
    for curve_label, vdf, ccolor, marker, linestyle in vario_curves:
        well = vdf[vdf["n_pairs"] >= MIN_PAIRS_PER_BIN]
        sparse = vdf[vdf["n_pairs"] < MIN_PAIRS_PER_BIN]
        ax_vario.plot(
            well["lag_bin_center_m"], well["semivariance_empirical"], color=ccolor,
            marker=marker, linestyle=linestyle, markersize=3.5, linewidth=1.4,
            label=curve_label, alpha=0.9,
        )
        if len(sparse) > 0:
            ax_vario.scatter(
                sparse["lag_bin_center_m"], sparse["semivariance_empirical"], color=ccolor,
                marker="x", s=22, alpha=0.35, label=f"{curve_label}, n_pairs<{MIN_PAIRS_PER_BIN}",
            )
    ax_vario.plot(
        h_smooth, model_smooth, color=CURVE_COLORS["model"], linestyle="--", linewidth=2,
        label="truth theoretical spherical model",
    )
    ax_vario.axhline(
        POR_STDEV ** 2, color="gray", linestyle=":", linewidth=1.2,
        label=f"truth total sill ({POR_STDEV ** 2:g})",
    )
    ax_vario.set_xlim(0.0, LAG_MAX_M)
    ax_vario.set_ylim(vario_ylim)
    ax_vario.set_xlabel("Lag h (m)")
    ax_vario.set_ylabel("Semivariance (Porosity %$^2$)")
    ax_vario.set_title(
        f"{tag} Variogram of each map (full 50x50 grid)\n"
        f"all curves: SAME {N_PAIRS_DRAWN:,} (i,j) pairs, VARIOGRAM_PAIR_SEED="
        f"{VARIOGRAM_PAIR_SEED}, {LAG_BIN_WIDTH_M:g} m bins",
        fontsize=8.5,
    )
    ax_vario.legend(loc="lower right", fontsize=6.8)
    ax_vario.grid(alpha=0.3)
    ref_lines = []
    for ref in vario_reference:
        line = (
            f"h~{ref['bin_center_m']:.0f} m bin ({ref['frac_of_range']:g}x range "
            f"{HMAJ1:g} m): model {ref['theoretical']:.2f} | truth {ref['truth']:.2f} "
            f"({ref['truth'] / ref['theoretical']:.2f}x model) | "
        )
        if has_example:
            line += f"example {ref['example']:.2f} ({ref['example'] / ref['theoretical']:.2f}x) | "
        line += f"pred map {ref['mean']:.2f} ({ref['mean'] / ref['theoretical']:.2f}x)"
        ref_lines.append(line)
    ax_vario.text(
        0.02, 0.985, "\n".join(textwrap.fill(line, 58) for line in ref_lines),
        transform=ax_vario.transAxes, fontsize=6.2, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, linewidth=0.4),
    )

    # ---- 8. Distribution reproduction ----
    dist_curves = [("truth field", rd["truth_masked"], CURVE_COLORS["truth"], "-")]
    if has_example:
        dist_curves.append(
            (rd["example_short_label"], rd["example_masked"], CURVE_COLORS["example"], "-")
        )
    dist_curves.append((rd["mean_short_label"], rd["mean_masked"], CURVE_COLORS["mean"], "-"))
    dist_curves.append(("conditioning samples", rd["sample_values"], CURVE_COLORS["samples"], "--"))
    for curve_label, values, ccolor, linestyle in dist_curves:
        xs, ps = _empirical_cdf(values)
        ax_dist.plot(xs, ps, color=ccolor, linestyle=linestyle, linewidth=1.6,
                     label=curve_label, alpha=0.9)
    ax_dist.set_xlim(dist_xlim)
    ax_dist.set_ylim(0.0, 1.0)
    ax_dist.set_xlabel("Porosity (%)")
    ax_dist.set_ylabel("Cumulative probability")
    ax_dist.set_title(
        f"{tag} Distribution of each map (CDF)\n"
        f"maps on the {rd['n_evaluated_cells']} evaluated cells; samples = the "
        f"{len(rd['sample_values'])} excluded conditioning cells",
        fontsize=8.5,
    )
    ax_dist.legend(loc="lower right", fontsize=6.8)
    ax_dist.grid(alpha=0.3)
    stats_lines = [f"mean / std (ddof={DIST_STD_DDOF})"]
    stat_keys = [("truth", "truth")]
    if has_example:
        stat_keys.append(("example", "example"))
    stat_keys += [("mean", "pred map"), ("samples", "samples")]
    for key, stats_label in stat_keys:
        s = rd["dist_stats"][key]
        stats_lines.append(f"{stats_label:<10s}{s['mean']:6.2f} / {s['std']:5.2f}  (n={s['n']})")
    ax_dist.text(
        0.02, 0.985, "\n".join(stats_lines), transform=ax_dist.transAxes, fontsize=6.8,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, linewidth=0.4),
    )

    # ---- stdout numbers (feed a table on the published page) ----
    print(
        f"  {method} {rd['replicate_id']} (sample_seed={rd['sample_seed']}): "
        f"UMG recomputed={rd['umg_recomputed']:.6f} vs. pinned metrics.csv="
        f"{rd['umg_pinned']:.6f} (match); MSE recomputed={rd['mse_recomputed']:.6f} vs. "
        f"pinned={rd['mse_pinned']:.6f} (match); variance_mean recomputed="
        f"{rd['variance_mean_recomputed']:.6f} vs. pinned={rd['variance_mean_pinned']:.6f} (match)"
    )
    print(
        f"    mean/std (ddof={DIST_STD_DDOF}) on the {rd['n_evaluated_cells']} evaluated cells: "
        + ", ".join(
            f"{k}={rd['dist_stats'][k]['mean']:.3f}/{rd['dist_stats'][k]['std']:.3f}"
            for k, _ in stat_keys
        )
    )
    for ref in vario_reference:
        msg = (
            f"    variogram @ h~{ref['bin_center_m']:.1f} m bin ({ref['frac_of_range']:g}x range "
            f"{HMAJ1:g} m): model={ref['theoretical']:.3f}, truth={ref['truth']:.3f} "
            f"({100 * ref['truth'] / ref['theoretical']:.1f}% of model)"
        )
        if has_example:
            msg += f", example={ref['example']:.3f} ({100 * ref['example'] / ref['theoretical']:.1f}%)"
        msg += f", pred_map={ref['mean']:.3f} ({100 * ref['mean'] / ref['theoretical']:.1f}%)"
        print(msg)

    realization_sentence = (
        "The example realization is sgs_realizations.npy[0] (first stored realization) and "
        "the prediction map is the SGS ensemble mean (sgs_mean_map.npy, also the MSE array); "
        if has_example else
        "Kriging produces a single estimate and NO realization, so the realization slot "
        "(panel 2) is a text panel and the variogram / distribution panels show the truth and "
        "the estimate only; the prediction map is kriging_mean_map_physical.npy (the MSE array) "
        "and the UMG plot is recomputed through the normal-score quantile back-transform "
        "(kriging_fraction_in on kmap_ns.npy / kriging_var_map_ns.npy), exactly as "
        "evaluate_sample_replicate_axis.py does; the variance panel is the Monte Carlo "
        "back-transform approximation kriging_var_map_physical_mc.npy; "
    )
    caption = (
        f"{label} at the 1% sample-density level (axis_level='1', n_samples_requested=25), "
        f"replicate {rd['replicate_id']} (sample_seed={rd['sample_seed']}) of the 10 "
        f"sample-location replicates (sample_seed 1001-1010); same ground-truth field for every "
        f"replicate (TRUTH_SEED={TRUTH_SEED}, range={AXIS_HMAJ1:g} m). {label}'s variogram range "
        f"({AXIS_HMAJ1:g} m) is a FIXED INPUT at every level and replicate (not fitted from the "
        "data), so there is no min/max-fitted-length pair and ONE 2x4 block is shown; rep0 is the "
        "replicate common to the GP-MLE and RBF+bootstrap case-study figures. Top sub-row = maps "
        "(truth+samples / realization slot / prediction map / predictive variance), bottom "
        "sub-row = diagnostics (UMG / accuracy crossplot / variogram / distribution). "
        f"{realization_sentence}"
        "Porosity color scale (truth/example/prediction panels) is anchored on the truth's own "
        "[min, max]; variance color scale is [0, this map's max] (this figure only). UMG, MSE and "
        "variance_mean shown are recomputed here from this run's own arrays and verified "
        "(np.isclose) to match results/processed/sample_replicate_axis/metrics.csv for the same "
        f"(axis_level, replicate, method) rows (UMG={rd['umg_recomputed']:.6f}, "
        f"MSE={rd['mse_recomputed']:.6f}, variance_mean={rd['variance_mean_recomputed']:.6f}). "
        "The accuracy crossplot plots truth vs. the point estimate on the replicate's evaluated "
        "cells (conditioning cells excluded), axis limits = truth [min, max] +5%. VARIOGRAM panel: "
        "experimental variogram of the truth and of the plotted map(s), plus the truth's "
        f"theoretical spherical model (nugget={NUG * POR_STDEV ** 2:g}, structured "
        f"sill={CC1 * POR_STDEV ** 2:g}, range={HMAJ1:g} m, imported from src.experiments.base_case) "
        f"and total sill ({POR_STDEV ** 2:g}); IDENTICAL {N_PAIRS_DRAWN:,} randomly drawn (i,j) "
        f"pairs (VARIOGRAM_PAIR_SEED={VARIOGRAM_PAIR_SEED}) and {LAG_BIN_WIDTH_M:g} m bins over "
        f"0-{LAG_MAX_M:g} m on the FULL {NX}x{NY} grid (a different cell set from the masked "
        "UMG/MSE/distribution support). DISTRIBUTION panel: empirical CDFs (plotting position "
        "(i-0.5)/n) on the evaluated cells plus the conditioning-sample values (the excluded "
        f"cells); mean/std (ddof={DIST_STD_DDOF}) printed in the panel. Axis ranges of the "
        "variogram and distribution panels are the SAME as the GP-MLE / RBF+bootstrap case-study "
        "figures (variogram y-max and the union of the two figures' distribution x-ranges, "
        "recomputed from those figures' own data) so the four figures are visually comparable"
        f"{'.' + axis_range_note if axis_range_note else '.'} "
        "A prediction map is an estimator, whose smoothing (lower variance than the truth) is a "
        "generic property of estimation; these panels state what was compared with what and make "
        "no claim about cause."
    )
    plt.suptitle(
        f"{label}: fixed-range ({AXIS_HMAJ1:g} m) replicate {rd['replicate_id']} "
        "at the 1% sample-density level",
        fontsize=16, y=0.985,
    )
    fig.text(0.5, GS_BOTTOM - BAND_PAD_BOTTOM - 0.012, textwrap.fill(caption, 210),
             ha="center", va="top",
             fontsize=8, color="dimgray")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"case_study_{method}_level1_replicate.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"{out.name}: {out} ({out.stat().st_size} bytes)")
    return out


def main():
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    truth = get_base_case_truth(truth_seed=TRUTH_SEED, hmaj1=AXIS_HMAJ1, hmin1=base.AXIS_HMIN1)

    print(
        f"\nCase-study replicate: {CASE_STUDY_REPLICATE} (sample_seed="
        f"{REPLICATE_SEED[CASE_STUDY_REPLICATE]}); truth_seed={TRUTH_SEED}, "
        f"VARIOGRAM_PAIR_SEED={VARIOGRAM_PAIR_SEED}"
    )
    truth_vario = compute_map_variogram(truth, "ground-truth field")
    vario_ylim, dist_xlim, _ = existing_case_study_axis_limits(
        source_runs, metrics_df, truth, truth_vario
    )
    print(f"  axis limits used: variogram y={vario_ylim}, distribution x={dist_xlim}")

    saved = []
    for method in FIXED_RANGE_METHODS:
        print(f"\nBuilding fixed-range case-study figure for {method}...")
        saved.append(
            make_fixed_range_figure(
                method, truth, source_runs, metrics_df, truth_vario, vario_ylim, dist_xlim
            )
        )
    print("\nAll fixed-range case-study figures:")
    for f in saved:
        print(" ", f)
    return saved


if __name__ == "__main__":
    main()
