"""Paper STORYBOARD figure set F1-F7: the paper's argument laid out with the
data that already exists, plus clearly marked placeholders where no experiment
has been run yet.

NO SIMULATION IS RUN HERE. This script only READS existing CSV/JSON files under
results/processed/ (plus, read-only, the multi_truth_axes run_config.json named
in results/processed/multi_truth_axes/evaluation_provenance.json, for the
ground-truth parameters of each cell, and the hyperparam_grid_surface
run_config.json named in results/processed/hyperparam_grid_surface/
provenance.json, for F7's grid / truth parameters) and draws. results/raw/ is never modified.

Figures (results/figures/paper_storyboard/, each as vector PDF + 300 dpi PNG +
a ``.status.txt`` sidecar):

  F1 (1-a)  accuracy (MSE) vs calibration (UMG) along range / nugget / density
            [multi_truth_axes E2 / E3 / E4]
  F2 (1-a)  base-case accuracy plot [E5] + mean predictive variance vs density [E4]
  F3 (1-b)  kriging vs GP-MLE along sample density (MSE, UMG, 90% coverage)
            + GP-MLE fitted range [E4]
  F3-2      F3 restricted to the rep0 cells (10 realizations x 1 sampling per
            level), row 2 = kriging / SGS with the user-picked variogram
            [E4 rep0 + results/processed/picked_variogram_density/]
  F4 (1-b)  GP-MLE fitted range / nugget vs ground truth (10 realizations/level)
  F5 (1-b)  anisotropy: true range vs azimuth, variogram check, PLACEHOLDER metrics
  F6 (1-b)  hyperparameter stability vs density: real (isotropic) + PLACEHOLDER
  F7 (2)    MSE and UMG over fixed-hyperparameter grids, 4 methods x 2 surfaces
            [hyperparam_grid_surface, via results/processed/hyperparam_grid_surface/]

Conventions
-----------
* Method colors / markers / labels / order are redefined IDENTICALLY to
  results/processed/sample_replicate_axis/make_sample_replicate_figures.py
  (METHOD_COLORS / METHOD_MARKERS / METHOD_LABELS / METHODS). Not imported, to
  avoid that module's import-time file reads.
* docs/figure_guidelines.md: canvas built at the FINAL width (190 mm), Arial
  7 pt, panel labels A/B/C, no caption inside the artwork, fonts embedded
  (pdf.fonttype 42). The canvas is saved as-is (no bbox_inches="tight") so the
  physical width is exactly the intended one.
* Every figure carries a small ``DATA STATUS:`` note in the bottom margin,
  OUTSIDE all axes. Every number in those notes is derived at run time from the
  input files / source constants (never typed in by hand).
* PLACEHOLDER panels (no experiment exists) have a grey background, a
  ``PLACEHOLDER - schematic, no data`` label above the panel, no numeric tick
  labels, and dashed/hollow artists. Their curve shapes are arbitrary seeded
  smooth curves that carry NO expected-result meaning (see _schematic_curves).

Run with:
.venv/Scripts/python.exe -m results.processed.paper_storyboard.make_storyboard_figures
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.ticker  # noqa: F401  (F3 panel D log-axis labels)
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation import calc_umg  # noqa: E402
from src.experiments.base_case import NX, NY  # noqa: E402
from src.experiments.evaluate_multi_truth_axes import read_metrics_csv  # noqa: E402
from src.experiments.evaluate_sample_replicate_axis import (  # noqa: E402
    VARIANCE_SOURCE_FILES,
    VARIANCE_SOURCE_IS_MC_BACKTRANSFORM,
)

PROCESSED = _REPO_ROOT / "results" / "processed"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "paper_storyboard"

# --- inputs (all read-only) ---------------------------------------------------
# F1-F3: multi-truth evaluation (src/experiments/evaluate_multi_truth_axes.py)
MTA_DIR = PROCESSED / "multi_truth_axes"
MTA_METRICS = MTA_DIR / "metrics.csv"
MTA_SUMMARY = MTA_DIR / "metrics_summary.csv"
MTA_HYPER = MTA_DIR / "hyperparameters.csv"
MTA_CELLS = MTA_DIR / "evaluation_cells.csv"
MTA_PROV = MTA_DIR / "evaluation_provenance.json"
MTA_CURVES = MTA_DIR / "accuracy_plot_curves.csv"   # written by make_base_case_curves.py
MTA_EXP = {"range": "E2", "nugget": "E3", "density": "E4", "base": "E5"}
# F4-F6: single-truth / per-axis inputs (unchanged)
REPL_SUMMARY = PROCESSED / "sample_replicate_axis" / "metrics_summary.csv"
REPL_SEEDS = PROCESSED / "sample_replicate_axis" / "replicate_seeds.json"
REPL_EXTRA = PROCESSED / "sample_replicate_axis" / "extra_density_levels_record.json"
REPL_LENGTH = PROCESSED / "sample_replicate_axis" / "length_scale_by_replicate.csv"
REPL_VAR_SOURCES = PROCESSED / "sample_replicate_axis" / "variance_metric_sources.csv"
RANGE_GP_REAL = PROCESSED / "range_axis" / "gp_hyperparameters_by_realization.csv"
NUGGET_GP_REAL = PROCESSED / "nugget_axis" / "gp_hyperparameters_by_realization.csv"
ANISO_AZIMUTH = PROCESSED / "anisotropy_azimuth_sweep" / "true_range_by_azimuth.csv"
ANISO_VARIO = PROCESSED / "anisotropy_variogram_check" / "variogram_comparison.csv"

# --- method styling: identical to the existing per-axis figure scripts ---------
METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]
METHOD_COLORS = {
    "kriging": "tab:blue", "sgs": "tab:green",
    "rbf_bootstrap": "tab:orange", "gp_mle": "tab:red",
}
METHOD_MARKERS = {"kriging": "o", "sgs": "s", "rbf_bootstrap": "^", "gp_mle": "D"}
METHOD_LABELS = {
    "kriging": "Simple kriging", "sgs": "SGS",
    "rbf_bootstrap": "RBF+bootstrap", "gp_mle": "GP-MLE",
}

# --- layout constants -----------------------------------------------------------
MM = 1.0 / 25.4
FULL_WIDTH_MM = 190.0
PNG_DPI = 300
FONT_PT = 7
STATUS_PT = 6          # guideline minimum for small text
PANEL_LABEL_PT = 9
STATUS_COLOR = "0.30"
PLACEHOLDER_FACE = "0.92"
PLACEHOLDER_TEXT = "PLACEHOLDER – schematic, no data"
STATUS_WRAP_CHARS = 186  # 190 mm canvas at 6 pt Arial

# Same cosmetic multiplicative x offset as
# sample_density_axis/make_sample_density_figures.py (METHOD_X_OFFSET_FRAC).
METHOD_X_OFFSET_FRAC = 0.07
# Same order-of-magnitude rule the project already uses to switch to a log axis.
RATIO_LOG_THRESHOLD = 10.0
DENSITY_LEVELS = ["20", "10", "5", "2", "1"]  # dense -> sparse

# F4 panel B: the nugget quantity plotted. "real" reuses EXACTLY the columns of
# results/processed/nugget_axis/make_nugget_fitted_vs_truth_figure.py and
# make_gp_realization_figures.py (y = gp_noise_variance_real_units,
# x = truth_nugget_real_units, both Porosity %^2). "fraction" switches to the
# dimensionless pair those scripts only RECORD (y = gp_noise_fraction_of_fitted_sill,
# x = nug_normalized).
F4_NUGGET_MODE = "real"
# F5 panel B: representative ratio level of anisotropy_variogram_check.
F5_VARIOGRAM_RATIO = "3:1"
# F5 panel B y-axis upper limit (display only; means use all points and the
# number of individual points above it is counted and printed in the note).
F5_VARIOGRAM_YMAX = 2.5
PLACEHOLDER_SEED = 20261116
# F3 is 1x4 at 190 mm; plot-area height (mm) and legend columns.
F3_PLOT_HEIGHT_MM = 50.0
F3_LEGEND_NCOL = 7
# wider right margin so panel D's x label is not clipped at 190 mm
F3_RIGHT_MM = 8.0
# F3 panel D (100 points per level): deterministic horizontal spread (+- fraction of x),
# marker area (pt^2) and alpha, chosen for readability only.
F3_D_SPREAD_FRAC = 0.12
# methods left out of F3 panels B (UMG) and C (coverage90) - user request 2026-11-17
F3_BC_OMIT = ("rbf_bootstrap",)
F3_D_MARKER_S = 3.0
F3_D_ALPHA = 0.35
F3_D_LOG_TICKS = (50, 100, 200, 300, 500, 1000, 2000)
# F3 row 2 (B-2 UMG, C-2 coverage90) + panel D second series: kriging / SGS re-run
# on E4 with the sample-inferred (auto-fit) variogram - user request 2026-11-17.
INF_DIR = PROCESSED / "inferred_variogram_density"
INF_METRICS = INF_DIR / "metrics.csv"
INF_SUMMARY = INF_DIR / "metrics_summary.csv"
INF_CELLS = INF_DIR / "evaluation_cells.csv"
INF_FITS = PROCESSED / "variogram_inference" / "fitted_variograms.csv"
INF_METHODS = ["kriging_inferred", "sgs_inferred"]
# inferred-variogram variants: same colour/marker/x offset as the oracle method,
# hollow marker + dashed line
INF_BASE = {"kriging_inferred": "kriging", "sgs_inferred": "sgs"}
INF_LABELS = {"kriging_inferred": "Simple kriging (inferred variogram)",
              "sgs_inferred": "SGS (inferred variogram)"}
INF_LINESTYLE = (0, (3, 1.5))
# panel D: GP-MLE points shifted left, inferred spherical ranges shifted right
# (multiplicative x shift), each spread +-F3_D_SPLIT_SPREAD
F3_D_SPLIT_SHIFT = 0.10
F3_D_SPLIT_SPREAD = 0.07
F3_D_INF_COLOR = "tab:blue"
# F3-2: F3 layout on the 50 rep0 cells only (10 truths x 1 sampling per level),
# row 2 = kriging / SGS re-run with the user-picked variogram (Variogram Review
# board) - user request 2026-11-18.
PICK_DIR = PROCESSED / "picked_variogram_density"
PICK_METRICS = PICK_DIR / "metrics.csv"
PICK_SUMMARY = PICK_DIR / "metrics_summary.csv"
PICK_CELLS = PICK_DIR / "evaluation_cells.csv"
PICK_PROV = PICK_DIR / "evaluation_provenance.json"
PICK_PICKS = PROCESSED / "variogram_inference" / "variogram_picks.csv"
PICK_METHODS = ["kriging_picked", "sgs_picked"]
# F3-2: GP-MLE is drawn from all 10 truths x 10 samplings per level (user request 2026-11-18)
F3_2_GP_N_CELLS = 100
PICK_SAMPLE_REP = "rep0"
INF_BASE.update({"kriging_picked": "kriging", "sgs_picked": "sgs"})
INF_LABELS.update({"kriging_picked": "Simple kriging (user-picked variogram)",
                   "sgs_picked": "SGS (user-picked variogram)"})
# panel D has 10 points per level per series (F3: 100) -> larger, less transparent markers
F3_2_D_MARKER_S = 6.0
F3_2_D_ALPHA = 0.7
# F3-2 panel D: three series per level (GP-MLE / user-picked / auto-fit), x shift and spread
F3_2_D_SHIFT = 0.15
F3_2_D_SPREAD = 0.05
F3_2_D_AUTO_COLOR = "tab:purple"
# F3-2b: one-realization truth map + rep0 samples per density (user request 2026-11-18)
F3_2_MAP_TRUTH_SEED = 101
F3_2_MAP_XMIN = 0.0
F3_2_MAP_CELL = 20.0  # == base_case XSIZ = YSIZ
F3_2_MAP_HEIGHT_MM = 38.0
F3_2_MAP_MARKER_S = 3.0
F3_2_MAP_CMAP = "viridis"

MSE_LABEL = "MSE (porosity %²)"
UMG_LABEL = "UMG (1 = calibrated)"
COV90_LABEL = "90% interval coverage"
COV90_NOMINAL = 0.90
DENSITY_XLABEL = "Sample density (% of grid cells sampled, dense → sparse)"
DENSITY_XLABEL_SHORT = "Sample density (%, dense → sparse)"


def _setup_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "grid.linewidth": 0.25,
        "grid.color": "0.8",
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,   # embed TrueType (editable live text)
        "ps.fonttype": 42,
        "savefig.dpi": PNG_DPI,
        "hatch.linewidth": 0.3,
    })


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _new_figure(plot_height_mm: float, status_text: str, top_mm: float = 9.0,
                bottom_axis_mm: float = 11.0, left_mm: float = 13.0, right_mm: float = 3.0):
    """Canvas at the FINAL 190 mm width. Height = plot area + a bottom strip
    sized to the wrapped DATA STATUS note. Returns (fig, rect) where rect is the
    (left, bottom, right, top) figure-fraction box available to the axes."""
    wrapped = textwrap.fill("DATA STATUS: " + status_text, STATUS_WRAP_CHARS)
    n_lines = wrapped.count("\n") + 1
    line_mm = STATUS_PT * 1.25 * (25.4 / 72.0)
    status_mm = n_lines * line_mm + 3.0
    height_mm = top_mm + plot_height_mm + bottom_axis_mm + status_mm
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, height_mm * MM))
    fig.text(
        1.5 / FULL_WIDTH_MM, (status_mm - 1.2) / height_mm, wrapped,
        ha="left", va="top", fontsize=STATUS_PT, color=STATUS_COLOR, linespacing=1.25,
    )
    rect = (
        left_mm / FULL_WIDTH_MM,
        (status_mm + bottom_axis_mm) / height_mm,
        1.0 - right_mm / FULL_WIDTH_MM,
        1.0 - top_mm / height_mm,
    )
    return fig, rect


def _panel_label(ax, letter: str, dx_pt: float = -24.0, dy_pt: float = 4.0):
    """Bold uppercase panel label at the upper-left, outside the axes."""
    ax.annotate(
        letter, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(dx_pt, dy_pt),
        textcoords="offset points", fontsize=PANEL_LABEL_PT, fontweight="bold",
        ha="left", va="bottom",
    )


def _mark_placeholder(ax, strip_ticks: bool = True, label: bool = True):
    """Make a panel impossible to mistake for data: grey hatched background,
    label above (outside) the panel, no numeric tick labels."""
    ax.set_facecolor(PLACEHOLDER_FACE)
    ax.patch.set_hatch("////")
    ax.patch.set_edgecolor("0.84")
    for spine in ax.spines.values():
        spine.set_linestyle((0, (4, 2)))
        spine.set_edgecolor("0.35")
    if strip_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
    if label:
        ax.annotate(
            PLACEHOLDER_TEXT, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 4),
            textcoords="offset points", ha="center", va="bottom", fontsize=FONT_PT,
            fontweight="bold", color="0.25",
        )


def _schematic_curves(n_curves: int, n_points: int, seed: int) -> np.ndarray:
    """Arbitrary smooth curves for placeholder panels, in [0.2, 0.8].

    Deliberately NOT an expected-result pattern: each curve is a seeded random
    low-order sinusoid mix, so curves cross and have no consistent ordering or
    trend. Deterministic (fixed seed) so reruns give identical output."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_points)
    out = np.empty((n_curves, n_points))
    for i in range(n_curves):
        a1, a2 = rng.uniform(-1.0, 1.0, size=2)
        p1, p2 = rng.uniform(0.0, 2.0 * np.pi, size=2)
        raw = a1 * np.sin(1.5 * np.pi * t + p1) + 0.6 * a2 * np.sin(3.0 * np.pi * t + p2)
        raw = raw - raw.mean()
        scale = np.abs(raw).max()
        out[i] = 0.5 + 0.3 * (raw / scale if scale > 0 else raw)
    return out


def _save(fig, stem: str, status_lines) -> list:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = FIGURES_DIR / f"{stem}.png"
    txt = FIGURES_DIR / f"{stem}.status.txt"
    # CreationDate=None -> byte-identical PDF on rerun.
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(png, dpi=PNG_DPI, metadata={"Software": None})
    plt.close(fig)
    txt.write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    for p in (pdf, png, txt):
        print(f"  wrote {_rel(p)} ({p.stat().st_size} bytes)")
    return [pdf, png, txt]


def _status_file(stem, inputs, panels, in_figure_note, limits=(), needed=()):
    w = FULL_WIDTH_MM
    lines = [
        f"{stem}",
        "generated by: results/processed/paper_storyboard/make_storyboard_figures.py",
        f"canvas: {w:g} mm wide (final size), Arial {FONT_PT} pt, vector PDF (fonts embedded, "
        f"fonttype 42) + PNG {PNG_DPI} dpi",
        "",
        "INPUT FILES (read-only):",
    ]
    lines += [f"  - {i}" for i in inputs] if inputs else ["  - (none: no data is read for this figure)"]
    lines += ["", "PANELS:"]
    lines += [f"  {k}: {v}" for k, v in panels]
    lines += ["", "DATA STATUS note printed in the figure margin:", f"  {in_figure_note}"]
    if limits:
        lines += ["", "DATA LIMITATIONS / FACTS ABOUT THE INPUTS:"]
        lines += [f"  - {x}" for x in limits]
    if needed:
        lines += ["", "EXPERIMENT NEEDED TO REPLACE PLACEHOLDER(S):"]
        lines += [f"  - {x}" for x in needed]
    return lines


# ---------------------------------------------------------------------------
# Data loading (with the checks the status notes rely on)
# ---------------------------------------------------------------------------
def _load_summary() -> pd.DataFrame:
    df = pd.read_csv(REPL_SUMMARY)
    df["axis_level"] = df["axis_level"].astype(str)
    if set(df["axis_level"]) != set(DENSITY_LEVELS):
        raise ValueError(f"{_rel(REPL_SUMMARY)}: unexpected levels {sorted(set(df['axis_level']))}.")
    return df


def _density_info() -> dict:
    """ACTUAL (post grid-cell dedup) sample counts per density level, per
    replicate, from the replicate axis's own records. x positions on every
    density axis are the REQUESTED fraction of the nx*ny grid
    (user decision 2026-11-18; actual post-dedup counts are reported in status files)."""
    seeds = json.loads(REPL_SEEDS.read_text(encoding="utf-8"))
    extra = json.loads(REPL_EXTRA.read_text(encoding="utf-8"))["replicate_axis"]
    n_actual = {}
    n_requested = {}
    for lvl, d in seeds["n_samples_actual"].items():
        n_actual[lvl] = [int(v) for v in d.values()]
        n_requested[lvl] = int(seeds["n_samples_requested_by_level"][lvl])
    for lvl, d in extra["n_samples_actual"].items():
        n_actual[lvl] = [int(v) for v in d.values()]
        n_requested[lvl] = int(extra["n_samples_requested"][lvl])
    if set(n_actual) != set(DENSITY_LEVELS):
        raise ValueError(f"density levels in seed records: {sorted(n_actual)}")
    # nx*ny from the evaluated-cell bookkeeping: n_cells_averaged + n_samples = grid.
    src = pd.read_csv(REPL_VAR_SOURCES)
    src["axis_level"] = src["axis_level"].astype(str)
    one = src[(src["axis_level"] == "20") & (src["replicate"] == "rep0")]["n_cells_averaged"].iloc[0]
    n_grid = int(one) + int(extra["n_samples_actual"]["20"]["rep0"])
    pct = {lvl: 100.0 * n_requested[lvl] / n_grid for lvl in DENSITY_LEVELS}  # requested density
    return {
        "n_actual": n_actual, "n_requested": n_requested, "n_grid": n_grid, "pct": pct,
        "truth_seed": int(seeds["truth_seed"]), "truth_range_m": float(seeds["hmaj1"]),
        "sample_seeds": list(seeds["sample_seed_replicates"].values()),
    }


def _style_density_axis(ax, dens, xlabel=True):
    xs = [dens["pct"][lvl] for lvl in DENSITY_LEVELS]
    ax.set_xscale("log")
    ax.set_xlim(max(xs) * 1.4, min(xs) / 1.4)  # dense -> sparse, as in the existing figure
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.minorticks_off()
    if xlabel:
        ax.set_xlabel(DENSITY_XLABEL)


def _plot_density_metric(ax, summary, dens, metric, methods=METHODS, faded=(), errorbars=True):
    """mean +- 1 std across replicates, methods offset in x so bars do not overlap.
    Inferred-variogram methods (INF_METHODS) take their oracle method's colour,
    marker and x offset, drawn with a hollow marker and dashed line.
    errorbars=False draws the means only (F3-2, user request 2026-11-18)."""
    for method in METHODS + INF_METHODS + PICK_METHODS:
        if method not in methods:
            continue
        base = INF_BASE.get(method, method)
        m_idx = METHODS.index(base)
        off = (m_idx - (len(METHODS) - 1) / 2) * METHOD_X_OFFSET_FRAC
        sub = summary[(summary["method"] == method) & (summary["metric"] == metric)]
        sub = sub.set_index("axis_level").loc[DENSITY_LEVELS]
        x = [dens["pct"][lvl] * (1.0 + off) for lvl in DENSITY_LEVELS]
        is_faded = method in faded
        ax.errorbar(
            x, sub["mean"].values, yerr=sub["std"].values if errorbars else None,
            color=METHOD_COLORS[base], marker=METHOD_MARKERS[base],
            markersize=2.6 if is_faded else 3.6,
            linewidth=0.6 if is_faded else 1.0,
            linestyle=INF_LINESTYLE if method in INF_BASE else "-",
            markerfacecolor="white" if method in INF_BASE else METHOD_COLORS[base],
            elinewidth=0.5, capsize=1.5, capthick=0.5,
            alpha=0.32 if is_faded else 1.0,
            zorder=2 if is_faded else 3,
        )


def _method_handles(methods=METHODS, faded=()):
    out = []
    for m in methods:
        base = INF_BASE.get(m, m)
        out.append(Line2D(
            [0], [0], color=METHOD_COLORS[base], marker=METHOD_MARKERS[base], markersize=3.6,
            linewidth=1.0, alpha=0.32 if m in faded else 1.0,
            linestyle=INF_LINESTYLE if m in INF_BASE else "-",
            markerfacecolor="white" if m in INF_BASE else METHOD_COLORS[base],
            label=INF_LABELS.get(m, METHOD_LABELS.get(m))))
    return out


def _load_inferred_density(mt) -> dict:
    """Inferred-variogram kriging / SGS on E4 (results/processed/inferred_variogram_density)
    and the per-cell spherical fit it used. Checks: pooled summary re-derived from
    metrics.csv; same cells / levels / counts as the oracle E4 evaluation; the
    variogram recorded per cell equals the fitted_variograms.csv row."""
    ed = mt["exp"]["density"]
    metrics = read_metrics_csv(INF_METRICS)
    summary = read_metrics_csv(INF_SUMMARY)
    for name, df in (("metrics", metrics), ("summary", summary)):
        if set(df["method"]) != set(INF_METHODS) or set(df["experiment"]) != {ed["experiment"]}:
            raise ValueError(f"inferred_variogram_density {name}: unexpected methods/experiments.")
        if set(df["axis_level"]) != set(DENSITY_LEVELS):
            raise ValueError(f"inferred_variogram_density {name}: unexpected levels.")
    key = ["experiment", "axis_level", "method", "metric"]
    re_ = metrics.groupby(key)["value"].agg(mean="mean", std=lambda v: float(np.std(v, ddof=1)),
                                            n="count")
    chk = summary.set_index(key)[["mean", "std", "n"]].join(re_, rsuffix="_re")
    if not (np.allclose(chk["mean"], chk["mean_re"], rtol=1e-12, atol=0)
            and np.allclose(chk["std"], chk["std_re"], rtol=1e-9, atol=1e-15)
            and (chk["n"] == chk["n_re"]).all() and set(chk["n"]) == {ed["n_cells"]}):
        raise ValueError(f"{_rel(INF_SUMMARY)} does not match {_rel(INF_METRICS)} / {ed['n_cells']} cells.")
    ens = summary[summary["method"] == "sgs_inferred"]
    if set(ens["n_ensemble"]) != {mt["n_ens"]}:
        raise ValueError(f"{_rel(INF_SUMMARY)}: SGS ensemble N differs from the oracle runs.")

    cells = pd.read_csv(INF_CELLS, dtype={"axis_level": str, "sample_rep": str})
    fits = pd.read_csv(INF_FITS, dtype={"axis_level": str, "sample_rep": str})
    fits = fits[fits["experiment"] == ed["experiment"]]
    j = cells.merge(fits[["config_key", "fit_range_m", "fit_nugget"]], on="config_key", how="left",
                    validate="one_to_one")
    if len(j) != len(fits) or j["fit_range_m"].isna().any():
        raise ValueError(f"{_rel(INF_CELLS)} and {_rel(INF_FITS)} cells differ.")
    if not (np.allclose(j["vario_hmaj1"], j["fit_range_m"], rtol=1e-12, atol=0)
            and np.allclose(j["vario_hmin1"], j["fit_range_m"], rtol=1e-12, atol=0)
            and np.allclose(j["vario_nug"], j["fit_nugget"], rtol=1e-12, atol=0)):
        raise ValueError(f"{_rel(INF_CELLS)} variogram differs from {_rel(INF_FITS)}.")
    counts = j.groupby("axis_level").size()
    if set(counts) != {ed["n_cells"]}:
        raise ValueError(f"inferred E4 cells per level {counts.to_dict()}")
    warn = {lvl: (int((j.loc[j["axis_level"] == lvl, "sgs_grid_node_warnings_b1"] > 0).sum()),
                  int(j.loc[j["axis_level"] == lvl, "sgs_grid_node_warnings_b1"].sum()))
            for lvl in DENSITY_LEVELS}
    bound = {lvl: int(((fits["axis_level"] == lvl) & (fits["range_at_upper"] | fits["range_at_lower"])).sum())
             for lvl in DENSITY_LEVELS}
    return {"summary": summary, "fits": j, "warn": warn, "range_bound_hits": bound}


def _replicate_n_text(dens) -> str:
    return "; ".join(
        f"{dens['pct'][lvl]:g}% = n {min(dens['n_actual'][lvl])}-{max(dens['n_actual'][lvl])} "
        f"(requested {dens['n_requested'][lvl]})"
        for lvl in DENSITY_LEVELS
    )


# ---------------------------------------------------------------------------
# Multi-truth evaluation data (F1-F3)
# ---------------------------------------------------------------------------
def _load_multi_truth() -> dict:
    """Everything F1-F3 need from results/processed/multi_truth_axes/, with the
    checks every DATA STATUS number relies on. All counts are derived here.

    Error bars everywhere in F1-F3 are the evaluator's POOLED mean / std
    (ddof=1) over all cells of a level (metrics_summary.csv); this is
    re-derived from metrics.csv and asserted equal."""
    prov = json.loads(MTA_PROV.read_text(encoding="utf-8"))
    if len(prov["batch_sets"]) != 1 or len(prov["sources"]) != 1:
        raise ValueError(f"{_rel(MTA_PROV)}: expected one source and one batch set.")
    batches = ",".join(str(b) for b in prov["batch_sets"][0])
    src_dir = _REPO_ROOT / prov["sources"][0]["source"]
    cfg = json.loads((src_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg_cells = pd.DataFrame(cfg["cells"])

    metrics = read_metrics_csv(MTA_METRICS)
    summary = read_metrics_csv(MTA_SUMMARY)
    for name, df in (("metrics", metrics), ("summary", summary)):
        if set(df["method"]) != set(METHODS):
            raise ValueError(f"multi_truth_axes {name}: unexpected methods {sorted(set(df['method']))}.")
    ens = summary[summary["method"].isin(["sgs", "rbf_bootstrap"])]
    if set(ens["batches"]) != {batches} or ens["n_ensemble"].nunique() != 1:
        raise ValueError(f"{_rel(MTA_SUMMARY)}: ensemble rows not all batch set '{batches}' with one N.")
    n_ens = int(ens["n_ensemble"].iloc[0])
    key = ["experiment", "axis_level", "method", "metric"]
    if summary.groupby(key).size().max() != 1:
        raise ValueError(f"{_rel(MTA_SUMMARY)}: more than one row per {key}.")
    # pooled mean / std (ddof=1) / n re-derived from the per-cell table
    re = metrics.groupby(key)["value"].agg(mean="mean", std=lambda v: float(np.std(v, ddof=1)),
                                          n="count")
    chk = summary.set_index(key)[["mean", "std", "n"]].join(re, rsuffix="_re")
    if not (np.allclose(chk["mean"], chk["mean_re"], rtol=1e-12, atol=0)
            and np.allclose(chk["std"].fillna(0), chk["std_re"].fillna(0), rtol=1e-9, atol=1e-15)
            and (chk["n"] == chk["n_re"]).all()):
        raise ValueError("metrics_summary.csv does not equal the pooled mean/std/n of metrics.csv.")

    cells = pd.read_csv(MTA_CELLS, dtype={"axis_level": str})
    cells = cells.merge(cfg_cells[["experiment", "config_key", "hmaj1", "hmin1", "nug"]],
                        on=["experiment", "config_key"], how="left", validate="many_to_one")
    if cells[["hmaj1", "hmin1", "nug"]].isna().any().any():
        raise ValueError("evaluation_cells.csv rows without a run_config cell.")
    n_grid = NX * NY
    if not ((cells["n_cells_evaluated"] + cells["n_samples_actual"]) == n_grid).all():
        raise ValueError(f"n_cells_evaluated + n_samples_actual != NX*NY = {n_grid} for some cell.")

    exp = {}
    for axis, e in MTA_EXP.items():
        c = cells[cells["experiment"] == e]
        per = c.groupby("axis_level").agg(n_cells=("truth_seed", "size"), n_truth=("truth_seed", "nunique"),
                                          n_sets=("sample_seed", "nunique"))
        if per.nunique().max() != 1:
            raise ValueError(f"{e}: unequal cell counts per level {per.to_dict()}")
        s = summary[summary["experiment"] == e]
        if set(s["n"]) != {int(per["n_cells"].iloc[0])}:
            raise ValueError(f"{e}: summary n {sorted(set(s['n']))} != cells per level {per['n_cells'].iloc[0]}")
        exp[axis] = {
            "experiment": e, "summary": s, "cells": c,
            "n_cells": int(per["n_cells"].iloc[0]), "n_truth": int(per["n_truth"].iloc[0]),
            "n_sets": int(per["n_sets"].iloc[0]), "n_levels": int(len(per)),
            "truth_seeds": sorted(int(v) for v in c["truth_seed"].unique()),
            "sample_seeds": sorted(int(v) for v in c["sample_seed"].unique()),
            "n_req": sorted(int(v) for v in c["n_samples_requested"].unique()),
            "n_act": sorted(int(v) for v in c["n_samples_actual"].unique()),
            "hmaj1": sorted(float(v) for v in c["hmaj1"].unique()),
            "hmin1": sorted(float(v) for v in c["hmin1"].unique()),
            "nug": sorted(float(v) for v in c["nug"].unique()),
        }

    # density axis: x = REQUESTED sample fraction (user decision 2026-11-18)
    d = exp["density"]["cells"]
    if set(d["axis_level"]) != set(DENSITY_LEVELS):
        raise ValueError(f"E4 levels {sorted(set(d['axis_level']))} != {DENSITY_LEVELS}")
    n_actual, n_requested, warn = {}, {}, {}
    for lvl in DENSITY_LEVELS:
        sub = d[d["axis_level"] == lvl]
        n_actual[lvl] = [int(v) for v in sub["n_samples_actual"]]
        if sub["n_samples_requested"].nunique() != 1:
            raise ValueError(f"E4 level {lvl}: more than one n_samples_requested.")
        n_requested[lvl] = int(sub["n_samples_requested"].iloc[0])
        w = sub["sgs_grid_node_warnings_b1"]
        warn[lvl] = (int((w > 0).sum()), int(np.nansum(w)), int(w.isna().sum()))
    if len(exp["density"]["hmaj1"]) != 1 or exp["density"]["hmaj1"] != exp["density"]["hmin1"]:
        raise ValueError("E4 ground truth is not a single isotropic range.")
    dens = {
        "n_actual": n_actual, "n_requested": n_requested, "n_grid": n_grid,
        "pct": {lvl: 100.0 * n_requested[lvl] / n_grid for lvl in DENSITY_LEVELS},
        "truth_range_m": exp["density"]["hmaj1"][0], "sgs_warnings": warn,
    }

    hyper = pd.read_csv(MTA_HYPER, dtype={"axis_level": str})
    gp_range = hyper[(hyper["experiment"] == MTA_EXP["density"]) & (hyper["method"] == "gp_mle")
                     & (hyper["parameter"] == "gp_practical_range_m")].copy()
    counts = gp_range.groupby("axis_level").size()
    if set(counts.index) != set(DENSITY_LEVELS) or set(counts) != {exp["density"]["n_cells"]}:
        raise ValueError(f"{_rel(MTA_HYPER)}: unexpected E4 gp_practical_range_m rows {counts.to_dict()}")

    curves = pd.read_csv(MTA_CURVES, dtype={"batches": str}, keep_default_na=False,
                         na_values={"n_ensemble": [""]})
    return {
        "prov": prov, "src_dir": src_dir, "batches": batches, "n_ens": n_ens,
        "project_default_n_ensemble": cfg.get("project_default_n_ensemble"),
        "exp": exp, "dens": dens, "gp_range": gp_range, "curves": curves,
    }


def _mt_ensemble_text(mt) -> str:
    return (f"SGS and RBF+bootstrap ensemble N = {mt['n_ens']} (batch {mt['batches']}); kriging and GP-MLE "
            "have no ensemble")


def _mt_bar_text() -> str:
    return "points = mean, bars = ±1 std (ddof = 1) pooled over all cells of a level"


def _mt_axis_text(a) -> str:
    """'10 ground-truth realizations x 1 sample set (n = 121 of 125 requested) = 10 cells per level'."""
    n = (f"n = {a['n_act'][0]} of {a['n_req'][0]} requested" if len(a["n_act"]) == 1
         else f"n = {a['n_act'][0]}-{a['n_act'][-1]}")
    sets = f"1 sample set ({n})" if a["n_sets"] == 1 else f"{a['n_sets']} sample sets"
    return f"{a['n_truth']} ground-truth realizations × {sets} = {a['n_cells']} cells per level"


def _plot_level_metric(ax, summ, metric):
    """mean +- 1 std on a LINEAR numeric level axis; methods offset additively
    by a fraction of the level spacing so error bars do not overlap."""
    levels = np.array(sorted(summ["axis_level"].astype(float).unique()))
    step = float(np.min(np.diff(levels)))
    for m_idx, method in enumerate(METHODS):
        off = (m_idx - (len(METHODS) - 1) / 2) * METHOD_X_OFFSET_FRAC * step
        sub = summ[(summ["method"] == method) & (summ["metric"] == metric)].copy()
        sub["x"] = sub["axis_level"].astype(float)
        sub = sub.sort_values("x")
        ax.errorbar(sub["x"] + off, sub["mean"], yerr=sub["std"], color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method], markersize=3.6, linewidth=1.0,
                    elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)
    ax.set_xticks(levels)
    return levels


def _level_table(summ, metric) -> str:
    t = summ[summ["metric"] == metric].copy()
    t["x"] = t["axis_level"].astype(float)
    parts = []
    for method in METHODS:
        s = t[t["method"] == method].sort_values("x")
        parts.append(f"{method}: " + ", ".join(
            f"{r.axis_level} {r.mean:.3g}±{r.std:.2g}" for r in s.itertuples()))
    return "; ".join(parts)


def _density_table(summ, metric, methods=METHODS) -> str:
    t = summ[summ["metric"] == metric].set_index(["method", "axis_level"])
    return "; ".join(
        f"{m}: " + ", ".join(f"{lvl}% {t.loc[(m, lvl), 'mean']:.3g}±{t.loc[(m, lvl), 'std']:.2g}"
                             for lvl in DENSITY_LEVELS)
        for m in methods)


def _mt_inputs(mt, extra=()):
    return [
        f"{_rel(MTA_SUMMARY)} (read with read_metrics_csv; pooled mean, std (ddof=1), n per experiment, "
        "axis_level, method, n_ensemble, batches, metric) - re-derived from "
        f"{_rel(MTA_METRICS)} at run time and asserted equal",
        f"{_rel(MTA_CELLS)} (n_samples_requested, n_samples_actual, n_cells_evaluated, sample_seed, "
        "sgs_grid_node_warnings_b1, config_key)",
        f"{_rel(MTA_PROV)} (source run folder, batch set)",
        f"{_rel(mt['src_dir'])}/run_config.json (read-only; hmaj1 / hmin1 / nug of every cell, "
        "joined on experiment + config_key)",
        *extra,
    ]


def _mt_common_limits(mt):
    p = mt["prov"]
    return [
        f"evaluation: {p['generated_by']} at git {p['git_commit'][:7]} (git_dirty={p['git_dirty']}, dirty files "
        f"{p['git_dirty_files']}); {p['n_cells_evaluated']} cells evaluated, {p['n_cells_skipped_incomplete']} "
        f"skipped; source {p['sources'][0]['source']} (is_production_configuration="
        f"{p['sources'][0]['is_production_configuration']}).",
        f"ensemble size: SGS and RBF+bootstrap N = {mt['n_ens']} from batch set '{mt['batches']}' "
        f"(run_config project_default_n_ensemble = {mt['project_default_n_ensemble']} is NOT what these runs "
        "used); kriging / GP-MLE rows have no ensemble.",
        "error bars: POOLED mean +- 1 std (ddof=1) over all cells of a level, exactly metrics_summary.csv. "
        "On the density axis the cells are truth x sample-set combinations, so the std mixes between-truth "
        "and between-sample-set variation (no nested decomposition).",
    ]


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------
def figure_f1(mt):
    stem = "F1_accuracy_vs_calibration_three_axes"
    ex = mt["exp"]
    dens = mt["dens"]
    range_txt, nugget_txt = _mt_axis_text(ex["range"]), _mt_axis_text(ex["nugget"])
    if range_txt == nugget_txt:
        rn_txt = (f"Range axis ({ex['range']['n_levels']} levels) and nugget axis "
                  f"({ex['nugget']['n_levels']} levels): {range_txt}")
    else:
        rn_txt = (f"Range axis ({ex['range']['n_levels']} levels): {range_txt}. Nugget axis "
                  f"({ex['nugget']['n_levels']} levels): {nugget_txt}")
    note = (
        f"{rn_txt}. Density axis: {_mt_axis_text(ex['density'])}. "
        f"{_mt_bar_text().capitalize()}. {_mt_ensemble_text(mt)}."
    )
    fig, rect = _new_figure(plot_height_mm=84.0, status_text=note, top_mm=13.0)
    gs = fig.add_gridspec(2, 3, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3],
                          wspace=0.24, hspace=0.16)
    axes = np.empty((2, 3), dtype=object)
    for c in range(3):
        axes[0, c] = fig.add_subplot(gs[0, c])
        axes[1, c] = fig.add_subplot(gs[1, c], sharex=axes[0, c],
                                     sharey=axes[1, 0] if c > 0 else None)

    level_cols = [
        (ex["range"]["summary"], "Ground-truth variogram range (m)"),
        (ex["nugget"]["summary"], "Ground-truth nugget (fraction of sill)"),
    ]
    level_values = []
    for c, (summ, xlabel) in enumerate(level_cols):
        for r, metric in enumerate(["mse", "umg"]):
            lv = _plot_level_metric(axes[r, c], summ, metric)
        level_values.append(lv)
        axes[1, c].set_xlabel(xlabel)
    for r, metric in enumerate(["mse", "umg"]):
        _plot_density_metric(axes[r, 2], ex["density"]["summary"], dens, metric)
    _style_density_axis(axes[1, 2], dens, xlabel=False)
    axes[1, 2].set_xlabel(DENSITY_XLABEL_SHORT)

    for c in range(3):
        axes[1, c].axhline(1.0, color="gray", linestyle="--", linewidth=0.6, zorder=1)
        plt.setp(axes[0, c].get_xticklabels(), visible=False)
        if c > 0:
            plt.setp(axes[1, c].get_yticklabels(), visible=False)
        for r in range(2):
            axes[r, c].grid(True)
    axes[0, 0].set_ylabel(MSE_LABEL)
    axes[1, 0].set_ylabel(UMG_LABEL)
    for ax, letter in zip(axes.ravel(), "ABCDEF"):
        _panel_label(ax, letter, dx_pt=-20.0 if letter in "AD" else -8.0)

    handles = _method_handles() + [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.6, label="Ideal (UMG = 1)")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.8)

    er, en, ed = ex["range"], ex["nugget"], ex["density"]
    lines = _status_file(
        stem,
        inputs=_mt_inputs(mt),
        panels=[
            ("A", f"REAL DATA - MSE vs ground-truth range ({er['experiment']}), 4 methods, mean +-1 std over "
                  f"{er['n_cells']} cells per level"),
            ("B", f"REAL DATA - MSE vs ground-truth nugget ({en['experiment']}), mean +-1 std over "
                  f"{en['n_cells']} cells per level"),
            ("C", f"REAL DATA - MSE vs sample density ({ed['experiment']}), mean +-1 std over "
                  f"{ed['n_cells']} cells per level"),
            ("D", "REAL DATA - UMG vs ground-truth range, same statistics as A"),
            ("E", "REAL DATA - UMG vs ground-truth nugget, same statistics as B"),
            ("F", "REAL DATA - UMG vs sample density, same statistics as C"),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=_mt_common_limits(mt) + [
            f"range axis {er['experiment']}: levels {[float(v) for v in level_values[0]]} m (hmaj1 = hmin1), "
            f"nug {er['nug']}; truth seeds {er['truth_seeds']}; sample seed(s) {er['sample_seeds']} shared by "
            f"every truth and level; n_samples_actual {er['n_act']} (requested {er['n_req']}).",
            f"nugget axis {en['experiment']}: levels {[float(v) for v in level_values[1]]} (fraction of sill), "
            f"range {en['hmaj1']} m; truth seeds {en['truth_seeds']}; sample seed(s) {en['sample_seeds']}; "
            f"n_samples_actual {en['n_act']} (requested {en['n_req']}).",
            f"density axis {ed['experiment']}: truth range {ed['hmaj1']} m, nug {ed['nug']}; truth seeds "
            f"{ed['truth_seeds']} x sample seeds {ed['sample_seeds']} (the same sample seeds at every truth "
            "and level).",
            "density x positions are the REQUESTED sample fraction of the "
            f"{NX}x{NY} = {dens['n_grid']} grid (actual post-dedup n per level, evaluation_cells.csv): "
            + _replicate_n_text(dens) + ".",
            "methods are offset in x only so error bars do not overlap (range / nugget: additive "
            f"+-{METHOD_X_OFFSET_FRAC:g} x level spacing steps; density: multiplicative +-"
            f"{METHOD_X_OFFSET_FRAC:.0%} steps); tick positions are the true level values.",
            "UMG row (D-F) shares one y axis; MSE row (A-C) has independent y axes.",
            "plotted values (level mean+-std), MSE range axis: " + _level_table(er["summary"], "mse"),
            "UMG range axis: " + _level_table(er["summary"], "umg"),
            "MSE nugget axis: " + _level_table(en["summary"], "mse"),
            "UMG nugget axis: " + _level_table(en["summary"], "umg"),
            "MSE density axis: " + _density_table(ed["summary"], "mse"),
            "UMG density axis: " + _density_table(ed["summary"], "umg"),
        ],
    )
    return _save(fig, stem, lines)


# ---------------------------------------------------------------------------
# F2
# ---------------------------------------------------------------------------
def figure_f2(mt):
    stem = "F2_base_case_calibration_and_variance"
    ex = mt["exp"]
    dens = mt["dens"]
    eb, ed = ex["base"], ex["density"]
    curves = mt["curves"]
    if set(curves["method"]) != set(METHODS) or set(curves["experiment"]) != {eb["experiment"]}:
        raise ValueError(f"{_rel(MTA_CURVES)}: unexpected methods / experiment.")
    c_ens = curves[curves["method"].isin(["sgs", "rbf_bootstrap"])]
    if set(c_ens["n_ensemble"].astype(int)) != {mt["n_ens"]} or set(c_ens["batches"]) != {mt["batches"]}:
        raise ValueError(f"{_rel(MTA_CURVES)}: ensemble N / batches differ from metrics_summary.csv.")
    n_p = int(curves.groupby("method").size().iloc[0])
    # UMG of the stored curves vs metrics_summary (E5 has 1 cell -> mean = the cell value)
    eb_umg = eb["summary"][eb["summary"]["metric"] == "umg"].set_index("method")["mean"]
    for method in METHODS:
        sub = curves[curves["method"] == method].sort_values("p_interval")
        g = calc_umg(sub["p_interval"].values, sub["fraction_in"].values)
        if not np.isclose(g, eb_umg[method], rtol=0.0, atol=1e-12):
            raise ValueError(f"{_rel(MTA_CURVES)} {method}: UMG {g} != metrics_summary {eb_umg[method]}")
    mc_methods = sorted(m for m in METHODS if VARIANCE_SOURCE_IS_MC_BACKTRANSFORM[m])
    # The in-figure wording below names exactly one Monte Carlo back-transformed method.
    if len(mc_methods) != 1:
        raise ValueError(f"expected exactly one MC back-transform variance method, got {mc_methods}")
    n_native = len(METHODS) - 1
    mc_txt = (f"{mc_methods[0]}'s is a Monte Carlo back-transform of its normal-score variance, the other "
              f"{ {2: 'two', 3: 'three', 4: 'four'}.get(n_native, str(n_native)) } are native")
    # E5 cell vs any other cell with the same configuration (shared run folders)
    all_cells = pd.concat([a["cells"] for a in ex.values()], ignore_index=True)
    e5_cell = eb["cells"]
    if len(e5_cell) != 1:
        raise ValueError(f"expected 1 {eb['experiment']} cell, got {len(e5_cell)}")
    e5_cell = e5_cell.iloc[0]
    run_cols = [c for c in all_cells.columns if c.startswith("run_dir_")]
    twins = all_cells[(all_cells["config_key"] == e5_cell["config_key"])
                      & (all_cells["experiment"] != eb["experiment"])]
    for _, t in twins.iterrows():
        if any(t[c] != e5_cell[c] for c in run_cols):
            raise ValueError(f"{eb['experiment']} and {t['experiment']} share config_key "
                             f"{e5_cell['config_key']} but not all {run_cols}.")
    twin_txt = (
        "panel A: the " + eb["experiment"] + " cell (config_key " + e5_cell["config_key"] + ") has the same "
        "configuration as " + "; ".join(
            f"{t['experiment']} level {t['axis_level']} / truth {t['truth_seed']} / sample_rep {t['sample_rep']}"
            for _, t in twins.iterrows())
        + f" and identical run folders ({', '.join(run_cols)} all equal, checked at run time)."
        if len(twins) else
        f"panel A: no other evaluated cell shares the {eb['experiment']} config_key {e5_cell['config_key']}."
    )
    if "variance_mean" not in set(ed["summary"]["metric"]):
        raise ValueError(f"{_rel(MTA_SUMMARY)} has no variance_mean for {ed['experiment']}.")

    note = (
        f"A: base case ({eb['experiment']}) = 1 ground-truth realization (truth seed {eb['truth_seeds'][0]}), "
        f"1 sample set (n = {eb['n_act'][0]} of {eb['n_req'][0]} requested). "
        f"B: {_mt_axis_text(ed)}; {_mt_bar_text()}. All variances in porosity %² – {mc_txt}. "
        f"{_mt_ensemble_text(mt)}."
    )
    fig, rect = _new_figure(plot_height_mm=62.0, status_text=note, top_mm=12.0)
    gs = fig.add_gridspec(1, 2, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3],
                          wspace=0.24, width_ratios=[1.0, 1.35])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    ax_a.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.6, zorder=1)
    for method in METHODS:
        sub = curves[curves["method"] == method].sort_values("p_interval")
        ax_a.plot(sub["p_interval"], sub["fraction_in"], color=METHOD_COLORS[method],
                  marker=METHOD_MARKERS[method], markersize=2.6)
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Nominal probability interval")
    ax_a.set_ylabel("Fraction of truth in interval")
    ax_a.grid(True)

    _plot_density_metric(ax_b, ed["summary"], dens, "variance_mean")
    _style_density_axis(ax_b, dens)
    ax_b.set_ylabel("Mean predictive variance (porosity %²)")
    ax_b.grid(True)

    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")
    handles = _method_handles() + [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.6, label="Ideal (1:1), panel A")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.8)

    lines = _status_file(
        stem,
        inputs=[
            f"{_rel(MTA_CURVES)} (columns method, n_ensemble, batches, p_interval, fraction_in) - written by "
            "results/processed/multi_truth_axes/make_base_case_curves.py from the evaluator's own "
            "evaluate_cell calc_umg inputs",
        ] + _mt_inputs(mt, extra=[
            "src/experiments/evaluate_sample_replicate_axis.py VARIANCE_SOURCE_FILES / "
            "VARIANCE_SOURCE_IS_MC_BACKTRANSFORM (variance provenance text)",
        ]),
        panels=[
            ("A", f"REAL DATA - base-case ({eb['experiment']}) accuracy plot (fraction of truth inside each "
                  f"nominal interval), {n_p} probability levels per method, 1:1 reference line"),
            ("B", f"REAL DATA - variance_mean vs sample density ({ed['experiment']}), mean +-1 std over "
                  f"{ed['n_cells']} cells per level"),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=_mt_common_limits(mt) + [
            f"panel A cell: {eb['experiment']} truth seed {eb['truth_seeds'][0]}, sample seed "
            f"{eb['sample_seeds'][0]}, range {eb['hmaj1']} m, nug {eb['nug']}. The UMG of each stored curve "
            "is recomputed here with src.evaluation.calc_umg and asserted equal (atol 1e-12) to the "
            f"{eb['experiment']} umg in metrics_summary.csv.",
            f"panel A UMG per method: " + ", ".join(f"{m} {eb_umg[m]:.3f}" for m in METHODS) + ".",
            twin_txt,
            "panel B variance sources (VARIANCE_SOURCE_FILES): " + "; ".join(
                f"{m}: {VARIANCE_SOURCE_FILES[m]}" for m in METHODS)
            + f"; SGS / RBF+bootstrap variance recomputed by the evaluator from the N = {mt['n_ens']} "
            "ensemble (ddof=1) and asserted equal to the saved map. Monte Carlo back-transform "
            f"approximation: {mc_methods}. Units are not stored in the multi_truth_axes outputs; "
            "porosity %² follows those source definitions.",
            "panel B x positions are the REQUESTED sample fraction (actual n): " + _replicate_n_text(dens) + ".",
            "panel B y axis is linear and autoscaled; an error bar (mean - 1 std) may extend below 0 "
            "where std > mean - this is the plotted +-1 std, not a negative variance.",
            "panel B plotted values (level mean+-std): " + _density_table(ed["summary"], "variance_mean"),
        ],
    )
    return _save(fig, stem, lines)


# ---------------------------------------------------------------------------
# Single-truth replicate-axis loader (used by F6 only)
# ---------------------------------------------------------------------------
def _load_gp_lengths() -> pd.DataFrame:
    df = pd.read_csv(REPL_LENGTH, comment="#")
    df["axis_level"] = df["axis_level"].astype(str)
    gp = df[df["method"] == "gp_mle"].copy()
    counts = gp.groupby("axis_level").size()
    if set(counts.index) != set(DENSITY_LEVELS) or counts.nunique() != 1:
        raise ValueError(f"{_rel(REPL_LENGTH)}: unexpected gp_mle rows per level: {counts.to_dict()}")
    return gp


# ---------------------------------------------------------------------------
# F3
# ---------------------------------------------------------------------------
def figure_f3(mt):
    stem = "F3_gp_vs_sample_density"
    ed = mt["exp"]["density"]
    dens = mt["dens"]
    summ = ed["summary"]
    gp = mt["gp_range"]
    n_per = ed["n_cells"]
    inf = _load_inferred_density(mt)
    fits = inf["fits"]
    spread = float(max(gp["value"].max(), fits["fit_range_m"].max())
                   / min(gp["value"].min(), fits["fit_range_m"].min()))
    use_log = spread >= RATIO_LOG_THRESHOLD
    faded = ("sgs", "rbf_bootstrap", "sgs_inferred")
    if "coverage90" not in set(summ["metric"]):
        raise ValueError(f"{_rel(MTA_SUMMARY)} has no coverage90 for {ed['experiment']}.")
    # row 2 data: inferred kriging / SGS + the SAME GP-MLE rows as row 1
    summ2 = pd.concat([summ[summ["method"] == "gp_mle"], inf["summary"]], ignore_index=True)
    row2_methods = INF_METHODS + ["gp_mle"]

    note = (
        f"{_mt_axis_text(ed)} (ground-truth range {dens['truth_range_m']:g} m); A–C, B-2, C-2: {_mt_bar_text()}; "
        f"row 1 kriging/SGS use the true (oracle) variogram; B-2, C-2: kriging/SGS re-run with the variogram "
        f"fitted to each cell's samples (spherical, automatic fit, same samples and SGS seed), GP-MLE as in row 1; "
        f"B-2/C-2 y axes equal B/C; "
        f"C: fraction of evaluated cells inside the central {COV90_NOMINAL:.0%} predictive interval (kriging: "
        f"normal-score interval back-transformed; SGS, RBF+bootstrap, GP-MLE: Gaussian mean ± z·std interval); "
        f"B, C: RBF+bootstrap not drawn (values in status file); "
        f"D: one point per cell per series ({n_per} per level); GP-MLE practical range = RBF-kernel length "
        f"scale at the 0.05 correlation cutoff; kriging/SGS inferred range = spherical range (correlation 0) – "
        f"the two range definitions differ; max/min over all points = {spread:.1f}×, "
        f"{'log' if use_log else 'linear'} y axis. {_mt_ensemble_text(mt)}."
    )
    fig, rect = _new_figure(plot_height_mm=2 * F3_PLOT_HEIGHT_MM + 16.0, status_text=note, top_mm=14.0,
                            right_mm=F3_RIGHT_MM)
    gs = fig.add_gridspec(2, 4, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3], wspace=0.42,
                          hspace=16.0 / F3_PLOT_HEIGHT_MM)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[0, i]) for i in range(4))
    ax_b2, ax_c2 = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    ax_leg2 = fig.add_subplot(gs[1, 3])
    ax_leg2.axis("off")

    _plot_density_metric(ax_a, summ, dens, "mse", faded=faded)
    # B and C omit RBF+bootstrap (its UMG ~0.5 / coverage ~0.45 compresses the
    # y range of the other methods); its values are in the status file.
    bc_methods = [m for m in METHODS if m not in F3_BC_OMIT]
    _plot_density_metric(ax_b, summ, dens, "umg", methods=bc_methods, faded=faded)
    _plot_density_metric(ax_c, summ, dens, "coverage90", methods=bc_methods, faded=faded)
    _plot_density_metric(ax_b2, summ2, dens, "umg", methods=row2_methods, faded=faded)
    _plot_density_metric(ax_c2, summ2, dens, "coverage90", methods=row2_methods, faded=faded)
    for ax in (ax_b, ax_b2):
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, zorder=1)
        ax.set_ylabel(UMG_LABEL)
    for ax in (ax_c, ax_c2):
        ax.axhline(COV90_NOMINAL, color="gray", linestyle="--", linewidth=0.6, zorder=1)
        ax.set_ylabel(COV90_LABEL)
    ax_a.set_ylabel(MSE_LABEL)
    # B-2 / C-2 share B / C's y range (union of the autoscaled ranges)
    for pair in ((ax_b, ax_b2), (ax_c, ax_c2)):
        lo = min(a.get_ylim()[0] for a in pair)
        hi = max(a.get_ylim()[1] for a in pair)
        for a in pair:
            a.set_ylim(lo, hi)

    # Deterministic (non-random) horizontal spread of the per-cell points,
    # ordered by (truth_seed, sample_rep); GP-MLE shifted left, inferred
    # spherical range shifted right.
    spread_frac = np.linspace(-F3_D_SPLIT_SPREAD, F3_D_SPLIT_SPREAD, n_per)
    for lvl in DENSITY_LEVELS:
        sub = gp[gp["axis_level"] == lvl].sort_values(["truth_seed", "sample_rep"])
        ax_d.scatter(
            dens["pct"][lvl] * (1.0 - F3_D_SPLIT_SHIFT + spread_frac), sub["value"],
            s=F3_D_MARKER_S, marker=METHOD_MARKERS["gp_mle"], facecolors=METHOD_COLORS["gp_mle"],
            edgecolors="none", alpha=F3_D_ALPHA, zorder=3,
        )
        fsub = fits[fits["axis_level"] == lvl].sort_values(["truth_seed", "sample_rep"])
        ax_d.scatter(
            dens["pct"][lvl] * (1.0 + F3_D_SPLIT_SHIFT + spread_frac), fsub["fit_range_m"],
            s=F3_D_MARKER_S, marker="o", facecolors=F3_D_INF_COLOR,
            edgecolors="none", alpha=F3_D_ALPHA, zorder=3,
        )
    ax_d.axhline(dens["truth_range_m"], color="black", linestyle="--", linewidth=0.7, zorder=2)
    if use_log:
        ax_d.set_yscale("log")
        # plain-number major ticks inside the data range (default log ticks
        # label only 10^2 here)
        lo, hi = ax_d.get_ylim()
        ticks = [t for t in F3_D_LOG_TICKS if lo <= t <= hi]
        ax_d.set_yticks(ticks)
        ax_d.set_yticklabels([f"{t:g}" for t in ticks])
        ax_d.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax_d.set_ylim(bottom=0)
    ax_d.set_ylabel("Fitted range (m)")

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d, ax_b2, ax_c2), ("A", "B", "C", "D", "B-2", "C-2")):
        _style_density_axis(ax, dens, xlabel=False)
        ax.set_xlabel(DENSITY_XLABEL_SHORT)
        ax.grid(True)
        _panel_label(ax, letter)

    # Panel y axes are autoscaled (errorbar includes the +-1 std bars) then
    # unioned per pair; verify nothing is clipped.
    for ax, s_, ms, metric, name in (
            (ax_b, summ, bc_methods, "umg", "B"), (ax_c, summ, bc_methods, "coverage90", "C"),
            (ax_b2, summ2, row2_methods, "umg", "B-2"), (ax_c2, summ2, row2_methods, "coverage90", "C-2")):
        r = s_[(s_["metric"] == metric) & (s_["method"].isin(ms))]
        lo_, hi_ = float((r["mean"] - r["std"]).min()), float((r["mean"] + r["std"]).max())
        yl = ax.get_ylim()
        if yl[0] > lo_ or yl[1] < hi_:
            raise ValueError(f"panel {name} y range {yl} clips the mean +-1 std range [{lo_}, {hi_}].")

    handles = _method_handles(faded=faded) + [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.6, label="Ideal (UMG = 1)"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.6,
               label=f"Nominal ({COV90_NOMINAL:.2f})"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.7,
               label=f"Ground-truth range ({dens['truth_range_m']:g} m)"),
        Line2D([0], [0], color=F3_D_INF_COLOR, marker="o", linestyle="none", markersize=3.0,
               label="D: kriging/SGS inferred range (spherical)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.4)
    ax_leg2.legend(handles=_method_handles(methods=row2_methods, faded=faded),
                   loc="center", title="B-2, C-2", frameon=False)

    per_level = gp.groupby("axis_level")["value"].describe(percentiles=[0.5]).loc[DENSITY_LEVELS]
    n_above = {lvl: int((gp[gp["axis_level"] == lvl]["value"] > dens["truth_range_m"]).sum())
               for lvl in DENSITY_LEVELS}
    inf_level = fits.groupby("axis_level")["fit_range_m"].describe(percentiles=[0.5]).loc[DENSITY_LEVELS]
    inf_above = {lvl: int((fits[fits["axis_level"] == lvl]["fit_range_m"] > dens["truth_range_m"]).sum())
                 for lvl in DENSITY_LEVELS}
    lines = _status_file(
        stem,
        inputs=_mt_inputs(mt, extra=[
            f"{_rel(MTA_HYPER)} (rows experiment == {ed['experiment']}, method == gp_mle, parameter == "
            "gp_practical_range_m; column value)",
            f"{_rel(INF_SUMMARY)} (kriging_inferred, sgs_inferred; pooled mean, std (ddof=1), n) - re-derived "
            f"from {_rel(INF_METRICS)} at run time and asserted equal",
            f"{_rel(INF_CELLS)} (vario_hmaj1 / vario_hmin1 / vario_nug per cell, sgs_grid_node_warnings_b1)",
            f"{_rel(INF_FITS)} (rows experiment == {ed['experiment']}; fit_range_m, fit_nugget, range_at_* ; "
            "asserted equal to the variogram recorded per cell)",
        ]),
        panels=[
            ("A", "REAL DATA - MSE vs density; kriging and GP-MLE at full strength, SGS and RBF+bootstrap "
                  f"drawn faded (same data, alpha 0.32); mean +-1 std over {n_per} cells per level"),
            ("B", "REAL DATA - UMG vs density, same emphasis and statistics; RBF+bootstrap NOT drawn "
                  "(user request 2026-11-17; its values are listed below)"),
            ("C", f"REAL DATA - 90% interval coverage vs density, same emphasis and statistics; RBF+bootstrap "
                  "NOT drawn (as B); "
                  f"dashed gray line = nominal {COV90_NOMINAL:.2f}"),
            ("D", f"REAL DATA - fitted range, one point per cell per series ({n_per} per level = "
                  f"{ed['n_truth']} truths x {ed['n_sets']} sample sets): GP-MLE practical range (red, shifted "
                  f"left) and the spherical range of the sample-inferred variogram used by kriging/SGS in B-2/C-2 "
                  f"(blue circles, shifted right; one series - kriging and SGS use the same variogram); dashed "
                  f"line = ground-truth range {dens['truth_range_m']:g} m"),
            ("B-2", "REAL DATA - UMG vs density: kriging / SGS re-run with the sample-inferred variogram "
                    "(hollow markers, dashed line; SGS faded) + GP-MLE (identical to B); y axis equal to B; "
                    f"mean +-1 std over {n_per} cells per level (user request 2026-11-17)"),
            ("C-2", "REAL DATA - 90% interval coverage vs density, same series and statistics as B-2; y axis "
                    "equal to C"),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=_mt_common_limits(mt) + [
            "panel D practical range = sqrt(-2 ln 0.05) x fitted RBF length scale "
            "(evaluate_multi_truth_axes.py, qc_nugget_axis.GP_PRACTICAL_RANGE_FACTOR).",
            f"panel C coverage90 = accuracy-plot fraction_in at exactly p = {COV90_NOMINAL:.2f} from the SAME "
            "call that produces UMG in src/experiments/evaluate_multi_truth_axes.py:evaluate_cell (p grid = "
            "default 20 levels + 0.90; strict inequality; conditioning-sample cells excluded).",
            "panel C kriging: normal-score central interval back-transformed per cell (kriging_fraction_in, "
            "same tail parameters as the point back-transform).",
            "panel C SGS, RBF+bootstrap and GP-MLE: Gaussian percentile method, interval = mean +- z*std from "
            f"the N = {mt['n_ens']} ensemble mean / variance (SGS, RBF+bootstrap) or the GP posterior mean / "
            "variance - NOT empirical quantiles of the SGS realizations or bootstrap replicates.",
            "panel C y axis autoscaled to include every mean +-1 std bar (checked at run time; no clipping).",
            f"panel D y-axis rule: log if max/min >= {RATIO_LOG_THRESHOLD:g}x; observed {spread:.2f}x -> "
            f"{'log' if use_log else 'linear'}.",
            "panel D per-level min / median / max (m) and count above the true range: " + "; ".join(
                f"{lvl}%: {per_level.loc[lvl, 'min']:.0f} / {per_level.loc[lvl, '50%']:.0f} / "
                f"{per_level.loc[lvl, 'max']:.0f}, {n_above[lvl]} of {n_per} above"
                for lvl in DENSITY_LEVELS),
            f"panel D points are spread horizontally by a fixed (non-random) +-{F3_D_SPREAD_FRAC:.0%} pattern "
            "ordered by (truth_seed, sample_rep), for visibility only; marker alpha "
            f"{F3_D_ALPHA:g}.",
            f"truth seeds {ed['truth_seeds']} x sample seeds {ed['sample_seeds']}; the same sample seeds at "
            "every truth and level.",
            "SGS grid-node warnings (evaluation_cells.csv sgs_grid_node_warnings_b1; cells with >0 / total "
            "warnings / cells without the field): " + "; ".join(
                f"{lvl}%: {dens['sgs_warnings'][lvl][0]} / {dens['sgs_warnings'][lvl][1]} / "
                f"{dens['sgs_warnings'][lvl][2]}" for lvl in DENSITY_LEVELS) + ".",
            "x positions are the REQUESTED sample fraction (actual n): " + _replicate_n_text(dens) + ".",
            "plotted values (level mean+-std), MSE: " + _density_table(summ, "mse"),
            "UMG: " + _density_table(summ, "umg"),
            "coverage90: " + _density_table(summ, "coverage90"),
            "B-2 / C-2 inferred-variogram runs: results/raw/inferred_variogram_density/ (runner "
            "src/experiments/inferred_variogram_density.py, evaluator "
            "src/experiments/evaluate_inferred_variogram_density.py); same truth, conditioning samples, "
            f"kb2d/sgsim settings, SGS N = {mt['n_ens']} and SGS seed as the row-1 oracle runs; only the "
            "kriging/SGS variogram differs: build_vario(hmaj1 = hmin1 = fit_range_m, nug = fit_nugget), "
            "spherical, unit sill, automatic WLS fit (no manual edits).",
            "B-2 / C-2 / D: GP-MLE values are the row-1 values (not re-run).",
            "panel D range definitions differ: GP-MLE practical range = correlation 0.05; spherical range = "
            "correlation 0 (where the spherical variogram reaches the sill).",
            "panel D inferred spherical range per-level min / median / max (m), count above the true range, "
            "fits at a range bound: " + "; ".join(
                f"{lvl}%: {inf_level.loc[lvl, 'min']:.0f} / {inf_level.loc[lvl, '50%']:.0f} / "
                f"{inf_level.loc[lvl, 'max']:.0f}, {inf_above[lvl]} of {n_per} above, "
                f"{inf['range_bound_hits'][lvl]} at bound"
                for lvl in DENSITY_LEVELS),
            "inferred SGS grid-node warnings (cells with >0 / total): " + "; ".join(
                f"{lvl}%: {inf['warn'][lvl][0]} / {inf['warn'][lvl][1]}" for lvl in DENSITY_LEVELS) + ".",
            "inferred MSE (not drawn): " + _density_table(inf["summary"], "mse", INF_METHODS),
            "inferred UMG: " + _density_table(inf["summary"], "umg", INF_METHODS),
            "inferred coverage90: " + _density_table(inf["summary"], "coverage90", INF_METHODS),
        ],
    )
    return _save(fig, stem, lines)


# ---------------------------------------------------------------------------
# F3-2 (F3 on the rep0 cells, row 2 = user-picked variogram)
# ---------------------------------------------------------------------------
def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    key = ["experiment", "axis_level", "method", "metric"]
    return (metrics.groupby(key)["value"]
            .agg(mean="mean", std=lambda v: float(np.std(v, ddof=1)), n="count").reset_index())


def _load_rep0_density(mt) -> dict:
    """E4 restricted to sample_rep == rep0 (10 truths x 1 sampling per level):
    oracle 4-method summary re-derived from multi_truth_axes/metrics.csv, GP-MLE
    practical ranges, and the user-picked-variogram kriging / SGS evaluation
    (summary re-derived from its metrics.csv). Checks: identical (level, truth,
    sample seed) cells in every source; picked variogram per cell == pick row
    (vario_source == pick) or the auto fit (vario_source == auto)."""
    ed = mt["exp"]["density"]
    e = ed["experiment"]
    m = read_metrics_csv(MTA_METRICS)
    m = m[(m["experiment"] == e) & (m["sample_rep"] == PICK_SAMPLE_REP)]
    if set(m["axis_level"]) != set(DENSITY_LEVELS) or set(m["method"]) != set(METHODS):
        raise ValueError(f"{_rel(MTA_METRICS)}: E4 {PICK_SAMPLE_REP} rows incomplete.")
    ens = m[m["method"].isin(["sgs", "rbf_bootstrap"])]
    if set(ens["batches"]) != {mt["batches"]} or set(ens["n_ensemble"]) != {mt["n_ens"]}:
        raise ValueError(f"{_rel(MTA_METRICS)}: E4 {PICK_SAMPLE_REP} ensemble rows not batch "
                         f"'{mt['batches']}' / N {mt['n_ens']}.")
    summ = _summarize(m)
    cells_o = ed["cells"][ed["cells"]["sample_rep"] == PICK_SAMPLE_REP]
    keys_o = set(zip(cells_o["axis_level"], cells_o["truth_seed"], cells_o["sample_seed"]))
    n_per = len(cells_o) // len(DENSITY_LEVELS)
    if (len(keys_o) != len(cells_o) or set(cells_o.groupby("axis_level").size()) != {n_per}
            or set(summ["n"]) != {n_per} or cells_o["sample_seed"].nunique() != 1):
        raise ValueError(f"E4 {PICK_SAMPLE_REP}: unequal / duplicate cells per level.")
    if set(zip(m["axis_level"], m["truth_seed"], m["sample_seed"])) != keys_o:
        raise ValueError(f"E4 {PICK_SAMPLE_REP}: metrics.csv cells != evaluation_cells.csv cells.")

    gp = mt["gp_range"]
    gp = gp[gp["sample_rep"] == PICK_SAMPLE_REP]
    if set(zip(gp["axis_level"], gp["truth_seed"], gp["sample_seed"])) != keys_o or len(gp) != len(keys_o):
        raise ValueError(f"{_rel(MTA_HYPER)}: E4 {PICK_SAMPLE_REP} gp_practical_range_m cells differ.")

    pm = read_metrics_csv(PICK_METRICS)
    ps = read_metrics_csv(PICK_SUMMARY)
    for name, df in (("metrics", pm), ("summary", ps)):
        if set(df["method"]) != set(PICK_METHODS) or set(df["experiment"]) != {e}:
            raise ValueError(f"picked_variogram_density {name}: unexpected methods/experiments.")
    if set(pm["sample_rep"]) != {PICK_SAMPLE_REP} or \
            set(zip(pm["axis_level"], pm["truth_seed"], pm["sample_seed"])) != keys_o:
        raise ValueError(f"{_rel(PICK_METRICS)}: cells differ from the oracle E4 {PICK_SAMPLE_REP} cells.")
    key = ["experiment", "axis_level", "method", "metric"]
    chk = ps.set_index(key)[["mean", "std", "n"]].join(_summarize(pm).set_index(key), rsuffix="_re")
    if not (np.allclose(chk["mean"], chk["mean_re"], rtol=1e-12, atol=0)
            and np.allclose(chk["std"], chk["std_re"], rtol=1e-9, atol=1e-15)
            and (chk["n"] == chk["n_re"]).all() and set(chk["n"]) == {n_per}):
        raise ValueError(f"{_rel(PICK_SUMMARY)} does not match {_rel(PICK_METRICS)} / {n_per} cells.")
    if set(ps.loc[ps["method"] == "sgs_picked", "n_ensemble"]) != {mt["n_ens"]}:
        raise ValueError(f"{_rel(PICK_SUMMARY)}: SGS ensemble N differs from the oracle runs.")

    pc = pd.read_csv(PICK_CELLS, dtype={"axis_level": str, "sample_rep": str})
    if set(zip(pc["axis_level"], pc["truth_seed"], pc["sample_seed"])) != keys_o or len(pc) != len(keys_o):
        raise ValueError(f"{_rel(PICK_CELLS)}: cells differ from the oracle E4 {PICK_SAMPLE_REP} cells.")
    picks = pd.read_csv(PICK_PICKS, dtype={"axis_level": str, "sample_rep": str}).set_index("cell_id")
    fits = pd.read_csv(INF_FITS, dtype={"axis_level": str, "sample_rep": str})
    fits = fits[fits["experiment"] == e].set_index("config_key")
    for r in pc.itertuples():
        cid = f"{r.experiment}_{r.config_key}"
        if r.vario_source == "pick":
            want = (picks.loc[cid, "range_m"], picks.loc[cid, "nugget"])
        elif r.vario_source == "auto" and cid not in picks.index:
            want = (fits.loc[r.config_key, "fit_range_m"], fits.loc[r.config_key, "fit_nugget"])
        else:
            raise ValueError(f"{_rel(PICK_CELLS)} {cid}: vario_source {r.vario_source!r} inconsistent with picks.")
        if not (r.vario_hmaj1 == r.vario_hmin1 and np.isclose(r.vario_hmaj1, want[0], rtol=1e-12, atol=0)
                and np.isclose(r.vario_nug, want[1], rtol=1e-12, atol=0)):
            raise ValueError(f"{_rel(PICK_CELLS)} {cid}: variogram differs from its pick / auto fit.")
    warn_o = {lvl: (int((cells_o.loc[cells_o["axis_level"] == lvl, "sgs_grid_node_warnings_b1"] > 0).sum()),
                    int(np.nansum(cells_o.loc[cells_o["axis_level"] == lvl, "sgs_grid_node_warnings_b1"])))
              for lvl in DENSITY_LEVELS}
    warn_p = {lvl: (int((pc.loc[pc["axis_level"] == lvl, "sgs_grid_node_warnings_b1"] > 0).sum()),
                    int(pc.loc[pc["axis_level"] == lvl, "sgs_grid_node_warnings_b1"].sum()))
              for lvl in DENSITY_LEVELS}
    a = {
        "experiment": e, "n_cells": n_per, "n_truth": int(cells_o["truth_seed"].nunique()),
        "truth_seeds": sorted(int(v) for v in cells_o["truth_seed"].unique()),
        "sample_seed": int(cells_o["sample_seed"].iloc[0]),
        "n_actual": {lvl: sorted(int(v) for v in cells_o.loc[cells_o["axis_level"] == lvl, "n_samples_actual"])
                     for lvl in DENSITY_LEVELS},
    }
    prov = json.loads(PICK_PROV.read_text(encoding="utf-8"))
    return {"summary": summ, "gp": gp, "pick_summary": ps, "pick_cells": pc, "axis": a,
            "warn_oracle": warn_o, "warn_picked": warn_p, "prov": prov}


def figure_f3_2(mt):
    """Density axis, one row (user request 2026-11-18): kriging and SGS with the
    user-picked variogram on the rep0 cells (10 per level); GP-MLE over ALL cells
    of a level (10 truths x 10 samplings = 100). Means only, no error bars.
    D: GP-MLE practical range of all 100 cells, user-picked range of the 10 rep0 cells."""
    stem = "F3-2_density_rep0_picked_variogram"
    dens = mt["dens"]
    ed = mt["exp"]["density"]
    r0 = _load_rep0_density(mt)
    a = r0["axis"]
    pc = r0["pick_cells"]
    n_pick = a["n_cells"]
    gp_all = mt["gp_range"]
    gp_summ = ed["summary"][ed["summary"]["method"] == "gp_mle"]
    if set(gp_summ["n"]) != {F3_2_GP_N_CELLS} or set(gp_summ["axis_level"]) != set(DENSITY_LEVELS):
        raise ValueError(f"F3-2: GP-MLE summary is not {F3_2_GP_N_CELLS} cells per density level.")
    if set(gp_all.groupby("axis_level").size()) != {F3_2_GP_N_CELLS}:
        raise ValueError(f"F3-2: GP-MLE practical range is not {F3_2_GP_N_CELLS} cells per density level.")
    # plotted with the regular kriging / SGS styles (no oracle series in this figure)
    pk = r0["pick_summary"].copy()
    pk["method"] = pk["method"].map({"kriging_picked": "kriging", "sgs_picked": "sgs"})
    summ = pd.concat([pk, gp_summ[pk.columns.intersection(gp_summ.columns)]], ignore_index=True)
    methods = ["kriging", "sgs", "gp_mle"]
    # B, C extra series (user request 2026-11-18): kriging / SGS with the AUTOMATIC-fit
    # variogram on the SAME rep0 cells (dashed, hollow markers)
    im = read_metrics_csv(INF_METRICS)
    im = im[(im["experiment"] == a["experiment"]) & (im["sample_rep"] == PICK_SAMPLE_REP)]
    if set(im["method"]) != set(INF_METHODS) or \
            set(zip(im["axis_level"], im["truth_seed"], im["sample_seed"])) != \
            set(zip(pc["axis_level"], pc["truth_seed"], pc["sample_seed"])):
        raise ValueError(f"{_rel(INF_METRICS)}: rep0 auto-fit cells differ from the user-picked cells.")
    inf_summ = _summarize(im)
    if set(inf_summ["n"]) != {n_pick}:
        raise ValueError(f"{_rel(INF_METRICS)}: expected {n_pick} rep0 cells per level.")
    summ_bc = pd.concat([summ, inf_summ[summ.columns.intersection(inf_summ.columns)]], ignore_index=True)
    methods_bc = ["kriging", "kriging_inferred", "sgs", "sgs_inferred", "gp_mle"]
    n_src = pc["vario_source"].value_counts().to_dict()
    auto_cells = pc[pc["vario_source"] == "auto"]
    unconf = pc[pc["pick_confirmed"].astype(str) == "False"]
    spread = float(max(gp_all["value"].max(), pc["vario_hmaj1"].max(), pc["auto_fit_range_m"].max())
                   / min(gp_all["value"].min(), pc["vario_hmaj1"].min(), pc["auto_fit_range_m"].min()))
    use_log = spread >= RATIO_LOG_THRESHOLD
    ax_scale = "log" if use_log else "linear"

    note = (
        f"{_mt_axis_text(ed)} (ground-truth range {dens['truth_range_m']:g} m); points = level means, no error "
        f"bars; kriging and SGS: user-picked variogram (spherical, set by hand on the Variogram Review board for "
        f"{n_src.get('pick', 0)} cells; automatic fit for {n_src.get('auto', 0)} cell without a pick), "
        f"{a['n_truth']} realizations × 1 sampling ({PICK_SAMPLE_REP}, sample seed {a['sample_seed']}) = "
        f"{n_pick} cells per level; GP-MLE: all {F3_2_GP_N_CELLS} cells per level (10 realizations × 10 "
        f"samplings); B, C dashed / hollow: kriging and SGS with the automatic-fit variogram on the same "
        f"{n_pick} {PICK_SAMPLE_REP} cells; C: fraction of evaluated cells inside the central {COV90_NOMINAL:.0%} predictive interval "
        f"(kriging: normal-score interval back-transformed; SGS, GP-MLE: Gaussian mean ± z·std interval); "
        f"D: one point per cell ({F3_2_GP_N_CELLS} GP-MLE, {n_pick} user-picked, {n_pick} auto-fit per level); GP-MLE practical "
        f"range = RBF-kernel length scale at the 0.05 correlation cutoff; user-picked range = spherical range "
        f"(correlation 0) – the two range definitions differ; max/min over all points = {spread:.1f}×, "
        f"{ax_scale} y axis. SGS ensemble N = {mt['n_ens']}; kriging and GP-MLE have no ensemble."
    )
    fig, rect = _new_figure(plot_height_mm=F3_PLOT_HEIGHT_MM, status_text=note, top_mm=19.0,
                            right_mm=F3_RIGHT_MM)
    gs = fig.add_gridspec(1, 4, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3], wspace=0.42)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[0, i]) for i in range(4))

    _plot_density_metric(ax_a, summ, dens, "mse", methods=methods, errorbars=False)
    _plot_density_metric(ax_b, summ_bc, dens, "umg", methods=methods_bc, errorbars=False)
    _plot_density_metric(ax_c, summ_bc, dens, "coverage90", methods=methods_bc, errorbars=False)
    ax_b.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, zorder=1)
    ax_c.axhline(COV90_NOMINAL, color="gray", linestyle="--", linewidth=0.6, zorder=1)
    ax_a.set_ylabel(MSE_LABEL)
    ax_b.set_ylabel(UMG_LABEL)
    ax_c.set_ylabel(COV90_LABEL)

    # D: deterministic horizontal spread; GP-MLE (all cells) left, user-picked range (rep0)
    # centre, automatic-fit range of the same rep0 cells right (user request 2026-11-18)
    sp_gp = np.linspace(-F3_2_D_SPREAD, F3_2_D_SPREAD, F3_2_GP_N_CELLS)
    sp_pk = np.linspace(-F3_2_D_SPREAD, F3_2_D_SPREAD, n_pick)
    for lvl in DENSITY_LEVELS:
        sub = gp_all[gp_all["axis_level"] == lvl].sort_values(["truth_seed", "sample_rep"])
        ax_d.scatter(
            dens["pct"][lvl] * (1.0 - F3_2_D_SHIFT + sp_gp), sub["value"],
            s=F3_D_MARKER_S, marker=METHOD_MARKERS["gp_mle"], facecolors=METHOD_COLORS["gp_mle"],
            edgecolors="none", alpha=F3_D_ALPHA, zorder=3,
        )
        psub = pc[pc["axis_level"] == lvl].sort_values("truth_seed")
        ax_d.scatter(
            dens["pct"][lvl] * (1.0 + sp_pk), psub["vario_hmaj1"],
            s=F3_2_D_MARKER_S, marker="o", facecolors=F3_D_INF_COLOR,
            edgecolors="none", alpha=F3_2_D_ALPHA, zorder=3,
        )
        ax_d.scatter(
            dens["pct"][lvl] * (1.0 + F3_2_D_SHIFT + sp_pk), psub["auto_fit_range_m"],
            s=F3_2_D_MARKER_S, marker="s", facecolors=F3_2_D_AUTO_COLOR,
            edgecolors="none", alpha=F3_2_D_ALPHA, zorder=3,
        )
    ax_d.axhline(dens["truth_range_m"], color="black", linestyle="--", linewidth=0.7, zorder=2)
    if use_log:
        ax_d.set_yscale("log")
        lo, hi = ax_d.get_ylim()
        ticks = [t for t in F3_D_LOG_TICKS if lo <= t <= hi]
        ax_d.set_yticks(ticks)
        ax_d.set_yticklabels([f"{t:g}" for t in ticks])
        ax_d.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax_d.set_ylim(bottom=0)
    ax_d.set_ylabel("Fitted / picked range (m)")

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), ("A", "B", "C", "D")):
        _style_density_axis(ax, dens, xlabel=False)
        ax.set_xlabel(DENSITY_XLABEL_SHORT)
        ax.grid(True)
        _panel_label(ax, letter)

    for ax, metric, name in ((ax_a, "mse", "A"), (ax_b, "umg", "B"), (ax_c, "coverage90", "C")):
        r = (summ if name == "A" else summ_bc)
        r = r[r["metric"] == metric]
        lo_, hi_ = float(r["mean"].min()), float(r["mean"].max())
        yl = ax.get_ylim()
        if yl[0] > lo_ or yl[1] < hi_:
            raise ValueError(f"panel {name} y range {yl} clips the means [{lo_}, {hi_}].")

    labels = {
        "kriging": f"Simple kriging (user-picked variogram, {n_pick} cells/level)",
        "sgs": f"SGS (user-picked variogram, {n_pick} cells/level)",
        "gp_mle": f"GP-MLE ({F3_2_GP_N_CELLS} cells/level)",
        "kriging_inferred": f"B, C: Simple kriging (auto-fit variogram, {n_pick} cells/level)",
        "sgs_inferred": f"B, C: SGS (auto-fit variogram, {n_pick} cells/level)",
    }
    leg_methods = ["kriging", "kriging_inferred", "sgs", "sgs_inferred", "gp_mle"]
    handles = _method_handles(methods=leg_methods)
    for h, m in zip(handles, leg_methods):
        h.set_label(labels[m])
    handles += [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.6,
               label=f"Ideal UMG (1) / nominal coverage ({COV90_NOMINAL:.2f})"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.7,
               label=f"Ground-truth range ({dens['truth_range_m']:g} m)"),
        Line2D([0], [0], color=F3_D_INF_COLOR, marker="o", linestyle="none", markersize=3.0,
               label="D: user-picked range (spherical)"),
        Line2D([0], [0], color=F3_2_D_AUTO_COLOR, marker="s", linestyle="none", markersize=3.0,
               label="D: auto-fit range (spherical)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4)

    gp_level = gp_all.groupby("axis_level")["value"].describe(percentiles=[0.5]).loc[DENSITY_LEVELS]
    gp_above = {lvl: int((gp_all[gp_all["axis_level"] == lvl]["value"] > dens["truth_range_m"]).sum())
                for lvl in DENSITY_LEVELS}
    pk_level = pc.groupby("axis_level")["vario_hmaj1"].describe(percentiles=[0.5]).loc[DENSITY_LEVELS]
    pk_above = {lvl: int((pc[pc["axis_level"] == lvl]["vario_hmaj1"] > dens["truth_range_m"]).sum())
                for lvl in DENSITY_LEVELS}
    fits = pd.read_csv(INF_FITS, dtype={"axis_level": str, "sample_rep": str})
    fits = fits[fits["experiment"] == a["experiment"]].set_index("config_key")
    if not np.allclose(pc["auto_fit_range_m"].values, fits.loc[pc["config_key"], "fit_range_m"].values,
                       rtol=1e-12, atol=0):
        raise ValueError(f"{_rel(PICK_CELLS)} auto_fit_range_m differs from {_rel(INF_FITS)} fit_range_m.")
    au_level = pc.groupby("axis_level")["auto_fit_range_m"].describe(percentiles=[0.5]).loc[DENSITY_LEVELS]
    au_above = {lvl: int((pc[pc["axis_level"] == lvl]["auto_fit_range_m"] > dens["truth_range_m"]).sum())
                for lvl in DENSITY_LEVELS}
    prov = r0["prov"]
    lines = _status_file(
        stem,
        inputs=_mt_inputs(mt, extra=[
            f"{_rel(PICK_CELLS)} auto_fit_range_m (panel D purple; asserted equal to {_rel(INF_FITS)} "
            "fit_range_m)",
            f"{_rel(MTA_SUMMARY)} (rows experiment == {a['experiment']}, method == gp_mle; {F3_2_GP_N_CELLS} "
            "cells per level, pooled mean)",
            f"{_rel(MTA_HYPER)} (rows experiment == {a['experiment']}, method == gp_mle, parameter == "
            f"gp_practical_range_m; all {F3_2_GP_N_CELLS} cells per level)",
            f"{_rel(PICK_SUMMARY)} (kriging_picked, sgs_picked; mean, n) - re-derived from "
            f"{_rel(PICK_METRICS)} at run time and asserted equal",
            f"{_rel(PICK_CELLS)} (vario_hmaj1 / vario_hmin1 / vario_nug, vario_source, pick_confirmed per cell)",
            f"{_rel(PICK_PROV)} (source run)",
            f"{_rel(PICK_PICKS)} and {_rel(INF_FITS)} (per-cell variogram asserted equal to the pick row, or "
            "to the automatic fit for cells without a pick)",
        ]),
        panels=[
            ("A", f"REAL DATA - MSE vs density; level means, no error bars; kriging / SGS = user-picked variogram "
                  f"({n_pick} rep0 cells per level), GP-MLE = all {F3_2_GP_N_CELLS} cells per level"),
            ("B", "REAL DATA - UMG vs density, same series and statistics; dashed gray line = 1"),
            ("C", "REAL DATA - 90% interval coverage vs density, same series and statistics; dashed gray line = "
                  f"nominal {COV90_NOMINAL:.2f}"),
            ("D", f"REAL DATA - range, one point per cell: GP-MLE practical range of all {F3_2_GP_N_CELLS} cells "
                  f"per level (red diamonds, shifted left) and the spherical range of the user-picked variogram "
                  f"of the {n_pick} rep0 cells used by kriging and SGS (blue circles, centre) and the automatic-fit "
                  f"spherical range of the same cells (purple squares, shifted right); dashed line = "
                  f"ground-truth range {dens['truth_range_m']:g} m"),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=[
            "user request 2026-11-18: single row (no oracle-variogram row), no error bars, kriging/SGS on the "
            f"rep0 cells with the user-picked variogram, GP-MLE over all {F3_2_GP_N_CELLS} cells per level; "
            "RBF+bootstrap not drawn.",
            f"kriging / SGS cells: sample_rep == {PICK_SAMPLE_REP} (sample seed {a['sample_seed']}), "
            f"{a['n_truth']} truth seeds {a['truth_seeds']} x {len(DENSITY_LEVELS)} levels. GP-MLE cells: the "
            f"same truth seeds x 10 sample sets ({F3_2_GP_N_CELLS} per level) - the GP-MLE means are NOT on "
            "the same cell set as kriging / SGS.",
            "user-picked variogram: range_m / nugget of variogram_picks.csv where a row exists, else the automatic "
            f"fit (fit_range_m / fit_nugget of fitted_variograms.csv). Cells with vario_source = pick: "
            f"{n_src.get('pick', 0)}; auto fallback: {n_src.get('auto', 0)} (" + ", ".join(
                f"{r.axis_level}% truth {r.truth_seed}" for r in auto_cells.itertuples()) + "). Unconfirmed picks "
            f"used as-is: {len(unconf)} (" + ", ".join(
                f"{r.axis_level}% truth {r.truth_seed}" for r in unconf.itertuples()) + "). The Variogram Review "
            "board showed the true (oracle) variogram curve while picking.",
            f"user-picked-variogram evaluation: {prov['generated_by']} at git {prov['git_commit'][:7]} "
            f"(git_dirty={prov['git_dirty']}); source run {prov['source_run']} (status "
            f"{prov['source_run_status']}); same samples and SGS seed as the oracle runs.",
            "panel D practical range = sqrt(-2 ln 0.05) x fitted RBF length scale "
            "(evaluate_multi_truth_axes.py, qc_nugget_axis.GP_PRACTICAL_RANGE_FACTOR); spherical range = "
            "correlation 0 - the two definitions differ.",
            f"panel C coverage90 = accuracy-plot fraction_in at exactly p = {COV90_NOMINAL:.2f} (as F3).",
            "A/B/C y axes autoscaled to include every plotted mean (checked at run time; no clipping).",
            f"panel D y-axis rule: log if max/min >= {RATIO_LOG_THRESHOLD:g}x; observed {spread:.2f}x -> "
            f"{ax_scale}; points spread horizontally by a fixed (non-random) +-{F3_D_SPLIT_SPREAD:.0%} pattern "
            f"ordered by (truth_seed, sample_rep); GP-MLE alpha {F3_D_ALPHA:g}, user-picked alpha "
            f"{F3_2_D_ALPHA:g}.",
            "panel D GP-MLE practical range per-level min / median / max (m), count above the true range: " +
            "; ".join(f"{lvl}%: {gp_level.loc[lvl, 'min']:.0f} / {gp_level.loc[lvl, '50%']:.0f} / "
                      f"{gp_level.loc[lvl, 'max']:.0f}, {gp_above[lvl]} of {F3_2_GP_N_CELLS} above"
                      for lvl in DENSITY_LEVELS),
            "panel D user-picked spherical range per-level min / median / max (m), count above the true range: " +
            "; ".join(f"{lvl}%: {pk_level.loc[lvl, 'min']:.0f} / {pk_level.loc[lvl, '50%']:.0f} / "
                      f"{pk_level.loc[lvl, 'max']:.0f}, {pk_above[lvl]} of {n_pick} above" for lvl in DENSITY_LEVELS),
            "panel D auto-fit spherical range per-level min / median / max (m), count above the true range: " +
            "; ".join(f"{lvl}%: {au_level.loc[lvl, 'min']:.0f} / {au_level.loc[lvl, '50%']:.0f} / "
                      f"{au_level.loc[lvl, 'max']:.0f}, {au_above[lvl]} of {n_pick} above" for lvl in DENSITY_LEVELS),
            "SGS grid-node warnings, user-picked (cells with >0 / total): " + "; ".join(
                f"{lvl}%: {r0['warn_picked'][lvl][0]} / {r0['warn_picked'][lvl][1]}" for lvl in DENSITY_LEVELS) + ".",
            "x positions are the REQUESTED sample fraction; actual n per level (rep0): " + "; ".join(
                f"{dens['pct'][lvl]:g}% = n {min(a['n_actual'][lvl])}-{max(a['n_actual'][lvl])} "
                f"(requested {dens['n_requested'][lvl]})" for lvl in DENSITY_LEVELS) + ".",
            "plotted values (level mean+-std; std not drawn), user-picked MSE: "
            + _density_table(r0["pick_summary"], "mse", PICK_METHODS),
            "user-picked UMG: " + _density_table(r0["pick_summary"], "umg", PICK_METHODS),
            "user-picked coverage90: " + _density_table(r0["pick_summary"], "coverage90", PICK_METHODS),
            f"B, C auto-fit variogram ({_rel(INF_METRICS)} rows sample_rep == {PICK_SAMPLE_REP}, re-summarized) "
            "UMG: " + _density_table(inf_summ, "umg", INF_METHODS),
            "auto-fit coverage90: " + _density_table(inf_summ, "coverage90", INF_METHODS),
            "auto-fit MSE (not drawn): " + _density_table(inf_summ, "mse", INF_METHODS),
            f"GP-MLE ({F3_2_GP_N_CELLS} cells) MSE: " + _density_table(gp_summ, "mse", ["gp_mle"]),
            f"GP-MLE ({F3_2_GP_N_CELLS} cells) UMG: " + _density_table(gp_summ, "umg", ["gp_mle"]),
            f"GP-MLE ({F3_2_GP_N_CELLS} cells) coverage90: " + _density_table(gp_summ, "coverage90", ["gp_mle"]),
        ],
    )
    return _save(fig, stem, lines)


def figure_f3_2_maps(mt):
    """Ground-truth (oracle) porosity map of ONE realization (truth seed
    F3_2_MAP_TRUTH_SEED) with the rep0 conditioning samples at each density level
    (user request 2026-11-18). The truth and samples are regenerated with the
    experiment's own generators and asserted equal to the samples.csv recorded by
    the user-picked kriging run of each cell."""
    from src.experiments.base_case_conditioning import (  # local: only this figure needs the generators
        get_base_case_truth, get_conditioning_samples)

    stem = "F3-2b_density_truth_and_samples"
    dens = mt["dens"]
    r0 = _load_rep0_density(mt)
    pc = r0["pick_cells"]
    t = F3_2_MAP_TRUTH_SEED
    cells = pc[pc["truth_seed"] == t].set_index("axis_level")
    if set(cells.index) != set(DENSITY_LEVELS):
        raise ValueError(f"F3-2b: truth seed {t} lacks a rep0 cell at some density level.")
    fits = pd.read_csv(INF_FITS, dtype={"axis_level": str, "sample_rep": str})
    fits = fits[fits["experiment"] == r0["axis"]["experiment"]].set_index("config_key")
    tv = fits.loc[cells["config_key"], ["true_range_m", "true_nugget"]].drop_duplicates()
    if len(tv) != 1 or float(tv["true_range_m"].iloc[0]) != float(dens["truth_range_m"]):
        raise ValueError("F3-2b: truth variogram differs between density levels / from the axis range.")
    rng_true, nug_true = float(tv["true_range_m"].iloc[0]), float(tv["true_nugget"].iloc[0])
    truth = get_base_case_truth(truth_seed=t, hmaj1=rng_true, hmin1=rng_true, nug=nug_true)
    if truth.shape != (NY, NX):
        raise ValueError(f"F3-2b: truth shape {truth.shape} != {(NY, NX)}")
    samples = {}
    for lvl in DENSITY_LEVELS:
        c = cells.loc[lvl]
        s = get_conditioning_samples(truth, sample_seed=int(c["sample_seed"]), n_samples=int(c["n_samples_requested"]))
        rec = pd.read_csv(_REPO_ROOT / c["run_dir_kriging_picked"] / "samples.csv")
        if len(rec) != len(s) or not np.allclose(rec[["X", "Y"]].values, s[["X", "Y"]].values, rtol=0, atol=1e-9):
            raise ValueError(f"F3-2b: regenerated samples differ from {c['run_dir_kriging_picked']}/samples.csv")
        samples[lvl] = s
    vmin, vmax = float(np.min(truth)), float(np.max(truth))
    ext = (F3_2_MAP_XMIN, F3_2_MAP_XMIN + NX * F3_2_MAP_CELL, F3_2_MAP_XMIN, F3_2_MAP_XMIN + NY * F3_2_MAP_CELL)

    note = (
        f"one ground-truth realization (truth seed {t}; range {rng_true:g} m, nugget {nug_true:g}) shown at every "
        f"density level; points = the conditioning samples of the rep0 sample set (sample seed "
        f"{int(cells['sample_seed'].iloc[0])}) used by kriging and SGS in F3-2; the same truth field in every "
        f"panel; n = actual number of distinct sample locations. Colour scale = min/max of the truth field."
    )
    fig, rect = _new_figure(plot_height_mm=F3_2_MAP_HEIGHT_MM, status_text=note, top_mm=6.0, right_mm=22.0)
    gs = fig.add_gridspec(1, len(DENSITY_LEVELS), left=rect[0], bottom=rect[1], right=rect[2], top=rect[3],
                          wspace=0.30)
    im = None
    for i, lvl in enumerate(DENSITY_LEVELS):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(truth, extent=ext, origin="upper", cmap=F3_2_MAP_CMAP, vmin=vmin, vmax=vmax)
        s = samples[lvl]
        ax.scatter(s["X"], s["Y"], s=F3_2_MAP_MARKER_S, facecolors="white", edgecolors="black",
                   linewidths=0.4, zorder=3)
        ax.set_title(f"{dens['pct'][lvl]:g}% (n = {len(s)})", fontsize=7)
        ax.set_xticks([0, 500, 1000])
        ax.set_yticks([0, 500, 1000])
        ax.set_xlabel("X (m)")
        if i == 0:
            ax.set_ylabel("Y (m)")
        else:
            ax.set_yticklabels([])
        ax.set_aspect("equal")
    cax = fig.add_axes([rect[2] + 0.03, rect[1], 0.012, rect[3] - rect[1]])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Porosity (%)")

    lines = _status_file(
        stem,
        inputs=[
            f"{_rel(PICK_CELLS)} (rows truth_seed == {t}: sample_seed, n_samples_requested, run_dir_kriging_picked)",
            "truth regenerated with src.experiments.base_case_conditioning.get_base_case_truth(truth_seed, "
            "hmaj1 = hmin1 = ground-truth range, nug); samples with get_conditioning_samples(truth, sample_seed, "
            "n_samples) - asserted equal (X, Y) to each run's samples.csv",
        ],
        panels=[(f"{dens['pct'][lvl]:g}%", f"REAL DATA - truth seed {t} porosity map + rep0 samples "
                 f"(n = {len(samples[lvl])}, requested {int(cells.loc[lvl, 'n_samples_requested'])})")
                for lvl in DENSITY_LEVELS],
        in_figure_note="DATA STATUS: " + note,
        limits=[f"single realization (truth seed {t}) chosen as the first truth seed; the other 9 realizations "
                "are not shown.",
                f"colour scale vmin/vmax = truth min/max ({vmin:.2f} / {vmax:.2f} %), colormap {F3_2_MAP_CMAP}."],
    )
    return _save(fig, stem, lines)


# ---------------------------------------------------------------------------
# F4
# ---------------------------------------------------------------------------
def _scatter_fitted_vs_truth(ax, df, xcol, ycol, level_col):
    color = METHOD_COLORS["gp_mle"]
    ax.scatter(df[xcol], df[ycol], s=9, marker=METHOD_MARKERS["gp_mle"], facecolors=color,
               edgecolors="none", alpha=0.45, zorder=3)
    mean = df.groupby(level_col).agg(x=(xcol, "first"), y=(ycol, "mean")).sort_values("x")
    ax.plot(mean["x"], mean["y"], color=color, marker=METHOD_MARKERS["gp_mle"], markersize=4.2,
            markeredgecolor="black", markeredgewidth=0.5, linewidth=1.0, zorder=4)
    hi = float(max(df[xcol].max(), df[ycol].max())) * 1.06
    ax.plot([0, hi], [0, hi], color="gray", linestyle="--", linewidth=0.6, zorder=1)
    ax.set_xlim(-0.03 * hi, hi)
    ax.set_ylim(-0.03 * hi, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(sorted(df[xcol].unique()))
    ax.grid(True)


def figure_f4():
    stem = "F4_gp_fitted_vs_truth_hyperparameters"
    rng = pd.read_csv(RANGE_GP_REAL)
    nug = pd.read_csv(NUGGET_GP_REAL)
    n_rng = rng.groupby("range_m")["truth_seed"].nunique()
    n_nug = nug.groupby("nug_normalized")["truth_seed"].nunique()
    if n_rng.nunique() != 1 or n_nug.nunique() != 1:
        raise ValueError("unequal realization counts per level.")
    n_real_r, n_real_n = int(n_rng.iloc[0]), int(n_nug.iloc[0])
    if rng["n_samples_actual"].nunique() != 1 or nug["n_samples_actual"].nunique() != 1:
        raise ValueError("n_samples_actual is not constant.")
    n_samp_r = int(rng["n_samples_actual"].iloc[0])
    n_samp_n = int(nug["n_samples_actual"].iloc[0])
    bound_cols = ["gp_noise_at_optimizer_floor", "gp_length_scale_at_bound",
                  "gp_signal_var_at_bound", "gp_noise_var_at_bound"]
    n_bound_r = int(rng[bound_cols].astype(bool).any(axis=1).sum())
    n_bound_n = int(nug[bound_cols].astype(bool).any(axis=1).sum())

    if F4_NUGGET_MODE == "real":
        nx, ny = "truth_nugget_real_units", "gp_noise_variance_real_units"
        nxl = "Ground-truth nugget (porosity %²)"
        nyl = "GP-MLE fitted noise variance (porosity %²)"
    elif F4_NUGGET_MODE == "fraction":
        nx, ny = "nug_normalized", "gp_noise_fraction_of_fitted_sill"
        nxl = "Ground-truth nugget (fraction of sill)"
        nyl = "GP-MLE fitted noise / (signal + noise)"
    else:
        raise ValueError(f"F4_NUGGET_MODE={F4_NUGGET_MODE!r}")

    note = (
        f"GP-MLE only. A: {n_real_r} ground-truth realizations per range level ({n_rng.size} levels); "
        f"B: {n_real_n} ground-truth realizations per nugget level ({n_nug.size} levels); the same sample "
        f"locations for every realization (n = {n_samp_r} in A, n = {n_samp_n} in B). Small markers = individual "
        "realizations, large markers = per-level mean. Fits stopped at an optimizer bound: "
        f"{n_bound_r} of {len(rng)} (A), {n_bound_n} of {len(nug)} (B). Single sample density only."
    )
    fig, rect = _new_figure(plot_height_mm=70.0, status_text=note, top_mm=11.0)
    gs = fig.add_gridspec(1, 2, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3], wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _scatter_fitted_vs_truth(ax_a, rng, "range_m", "gp_practical_range_m", "range_m")
    ax_a.set_xlabel("Ground-truth variogram range (m)")
    ax_a.set_ylabel("GP-MLE fitted practical range (m)")
    _scatter_fitted_vs_truth(ax_b, nug, nx, ny, "nug_normalized")
    ax_b.set_xlabel(nxl)
    ax_b.set_ylabel(nyl)
    if F4_NUGGET_MODE == "real":
        ax_b.set_xticklabels([f"{v:.1f}" for v in sorted(nug[nx].unique())])
    fig.align_xlabels([ax_a, ax_b])
    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")

    c = METHOD_COLORS["gp_mle"]
    mk = METHOD_MARKERS["gp_mle"]
    handles = [
        Line2D([0], [0], linestyle="none", marker=mk, markersize=3, markerfacecolor=c,
               markeredgecolor="none", alpha=0.45, label="GP-MLE, one ground-truth realization"),
        Line2D([0], [0], color=c, marker=mk, markersize=4.2, markeredgecolor="black",
               markeredgewidth=0.5, label="GP-MLE, mean per level"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.6, label="y = x"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.8)

    lines = _status_file(
        stem,
        inputs=[
            f"{_rel(RANGE_GP_REAL)} (x = range_m, y = gp_practical_range_m; truth_seed, n_samples_actual, "
            "*_at_bound / gp_noise_at_optimizer_floor flags)",
            f"{_rel(NUGGET_GP_REAL)} (x = {nx}, y = {ny}; level key nug_normalized; same flag columns)",
        ],
        panels=[
            ("A", f"REAL DATA - GP-MLE fitted practical range vs ground-truth range, {n_real_r} realizations "
                  "per level, y = x line"),
            ("B", f"REAL DATA - GP-MLE fitted nugget vs ground-truth nugget, {n_real_n} realizations per "
                  f"level, y = x line (F4_NUGGET_MODE = '{F4_NUGGET_MODE}')"),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=[
            f"truth seeds per level (both files): {sorted(rng['truth_seed'].unique())}.",
            "panel B columns: the existing nugget_axis/make_nugget_fitted_vs_truth_figure.py and "
            "make_gp_realization_figures.py PLOT y = gp_noise_variance_real_units against "
            "x = truth_nugget_real_units (= nug * 9, porosity %^2), not a normalized ratio; the "
            "dimensionless pair (gp_noise_fraction_of_fitted_sill vs nug_normalized) is only a recorded "
            "column there. This figure follows the PLOTTED definition; set F4_NUGGET_MODE = 'fraction' "
            "in the script for the dimensionless pair.",
            "panel B x tick positions "
            f"{[round(float(v), 2) for v in sorted(nug[nx].unique())]} correspond to nug_normalized "
            f"{sorted(nug['nug_normalized'].unique())}.",
            "only GP-MLE has per-realization hyperparameter tables; kriging/SGS are given the true "
            "variogram and RBF+bootstrap has no nugget parameter, so they are not on this figure.",
            f"range-axis file truth_nugget_normalized = {sorted(rng['truth_nugget_normalized'].unique())}; "
            "nugget-axis runs use the base-case range.",
        ],
    )
    return _save(fig, stem, lines)


# ---------------------------------------------------------------------------
# F5
# ---------------------------------------------------------------------------
def figure_f5():
    stem = "F5_anisotropy"
    azi = pd.read_csv(ANISO_AZIMUTH)
    vario = pd.read_csv(ANISO_VARIO)
    v = vario[vario["ratio_label"] == F5_VARIOGRAM_RATIO]
    if v.empty:
        raise ValueError(f"ratio {F5_VARIOGRAM_RATIO} not in {_rel(ANISO_VARIO)}")
    exp = v[v["source"] == "experimental"]
    n_truth = int(exp["truth_seed"].nunique())
    n_draw = int(exp["sample_draw_seed"].nunique())
    v_hmaj, v_hmin = float(v["hmaj1"].iloc[0]), float(v["hmin1"].iloc[0])
    n_off = int((exp["gamma"] > F5_VARIOGRAM_YMAX).sum())
    exp_max = float(exp["gamma"].max())
    azi_levels = azi.groupby("ratio_label", sort=False)[["hmaj1", "hmin1"]].first()
    azi_txt = ", ".join(
        f"'{k}' = hmaj1 {r.hmaj1:g} m / hmin1 {r.hmin1:g} m" for k, r in azi_levels.iterrows())

    note = (
        "A: analytic true practical range by azimuth, no prediction method involved; ratio labels and "
        f"ranges exactly as stored ({azi_txt}). B: variogram check (5 % sample density, 125 samples requested) for the '{F5_VARIOGRAM_RATIO}' level "
        f"of a DIFFERENT parameterization (hmaj1 {v_hmaj:g} m / hmin1 {v_hmin:g} m), normal-score "
        f"variogram standardized to unit sill, {n_truth} ground-truth realizations × {n_draw} sample "
        f"draws; y axis cut at {F5_VARIOGRAM_YMAX:g}: {n_off} of {len(exp)} individual experimental points lie "
        f"above it (max {exp_max:.1f}), means are computed from all points. C: PLACEHOLDER – the anisotropy-axis experiment (4 methods, MSE and UMG vs "
        "anisotropy ratio) has not been run; curve shapes are arbitrary."
    )
    fig, rect = _new_figure(plot_height_mm=58.0, status_text=note, top_mm=22.0)
    gs = fig.add_gridspec(2, 3, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3],
                          wspace=0.34, hspace=0.12)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c1 = fig.add_subplot(gs[0, 2])
    ax_c2 = fig.add_subplot(gs[1, 2], sharex=ax_c1)

    # A --- neutral greys (these are ground-truth models, not prediction methods)
    styles = ["-", "--", ":"]
    h_a = []
    for ls, (label, row) in zip(styles, azi_levels.iterrows()):
        sub = azi[azi["ratio_label"] == label].sort_values("azimuth_deg")
        ax_a.plot(sub["azimuth_deg"], sub["true_practical_range"], color="black", linestyle=ls)
        h_a.append(Line2D([0], [0], color="black", linestyle=ls,
                          label=f"{label}: {row.hmaj1:g} m / {row.hmin1:g} m"))
    ax_a.set_xlabel("Azimuth (°)")
    ax_a.set_ylabel("True practical range (m)")
    ax_a.set_xticks(np.arange(0, 181, 45))
    ax_a.set_ylim(bottom=0)
    ax_a.grid(True)
    ax_a.legend(handles=h_a, title="Label: hmaj1 / hmin1", loc="lower center",
                bbox_to_anchor=(0.5, 1.02), ncol=1, borderaxespad=0.0, title_fontsize=FONT_PT)

    # B --- colors as in src/experiments/anisotropy_variogram_check.py
    # (experimental tab:blue, model black); major solid, minor dashed.
    dir_ls = {"major": "-", "minor": "--"}
    dir_mk = {"major": "o", "minor": "s"}
    max_mean = 0.0
    for direction in ("major", "minor"):
        d_exp = exp[exp["direction"] == direction].copy()
        d_exp["lag_idx"] = d_exp.groupby(["truth_seed", "sample_draw_seed"]).cumcount()
        for _, grp in d_exp.groupby(["truth_seed", "sample_draw_seed"]):
            ax_b.plot(grp["lag"], grp["gamma"], color="tab:blue", linewidth=0.3, alpha=0.22,
                      linestyle=dir_ls[direction], zorder=1)
        mean = d_exp.groupby("lag_idx")[["lag", "gamma"]].mean()
        max_mean = max(max_mean, float(mean["gamma"].max()))
        ax_b.plot(mean["lag"], mean["gamma"], color="tab:blue", linestyle=dir_ls[direction],
                  marker=dir_mk[direction], markersize=2.6, zorder=3)
        mod = v[(v["source"] == "model") & (v["direction"] == direction)].sort_values("lag")
        ax_b.plot(mod["lag"], mod["gamma"], color="black", linestyle=dir_ls[direction], zorder=4)
    ax_b.axhline(1.0, color="gray", linestyle=":", linewidth=0.4, zorder=0)
    ax_b.set_xlabel("Lag distance (m)")
    ax_b.set_ylabel("Variogram (standardized, sill = 1)")
    ax_b.set_xlim(left=0)
    ax_b.set_ylim(0, F5_VARIOGRAM_YMAX)
    ax_b.grid(True)
    h_b = [
        Line2D([0], [0], color="black", linestyle="-", label="Model, major"),
        Line2D([0], [0], color="black", linestyle="--", label="Model, minor"),
        Line2D([0], [0], color="tab:blue", linestyle="-", marker="o", markersize=2.6,
               label="Experimental mean, major"),
        Line2D([0], [0], color="tab:blue", linestyle="--", marker="s", markersize=2.6,
               label="Experimental mean, minor"),
        Line2D([0], [0], color="tab:blue", linewidth=0.3, alpha=0.5,
               label=f"Individual ({n_truth} truths × {n_draw} draws)"),
    ]
    ax_b.legend(handles=h_b, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
                borderaxespad=0.0)

    # C --- placeholder: arbitrary seeded curves, no numeric ticks
    t = np.linspace(0.0, 1.0, 5)
    for ax, seed_off, ylabel in ((ax_c1, 0, "MSE"), (ax_c2, 1, "UMG")):
        curves = _schematic_curves(len(METHODS), len(t), PLACEHOLDER_SEED + seed_off)
        for method, y in zip(METHODS, curves):
            ax.plot(t, y, color=METHOD_COLORS[method], linestyle=(0, (3, 2)), linewidth=0.8,
                    marker=METHOD_MARKERS[method], markersize=3.2, markerfacecolor="none",
                    markeredgewidth=0.6, alpha=0.8)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(0.0, 1.0)
        _mark_placeholder(ax, label=(ax is ax_c1))
        ax.set_ylabel(ylabel + " (schematic)")
    ax_c2.set_xlabel("Anisotropy ratio (isotropic → strong)")
    h_c = [
        Line2D([0], [0], color=METHOD_COLORS[m], linestyle=(0, (3, 2)), linewidth=0.8,
               marker=METHOD_MARKERS[m], markersize=3.2, markerfacecolor="none",
               markeredgewidth=0.6, label=METHOD_LABELS[m] + " (schematic, no data)")
        for m in METHODS
    ]
    ax_c1.legend(handles=h_c, loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=1,
                 borderaxespad=0.0)

    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")
    _panel_label(ax_c1, "C")

    lines = _status_file(
        stem,
        inputs=[
            f"{_rel(ANISO_AZIMUTH)} (columns ratio_label, hmaj1, hmin1, azimuth_deg, true_practical_range)",
            f"{_rel(ANISO_VARIO)} (rows ratio_label == '{F5_VARIOGRAM_RATIO}'; columns direction "
            "[major, minor], source [model, experimental], lag, gamma, truth_seed, sample_draw_seed)",
        ],
        panels=[
            ("A", "REAL DATA (analytic) - true practical range vs azimuth for the 3 stored ratio levels"),
            ("B", f"REAL DATA - experimental vs model variogram, major and minor direction, ratio level "
                  f"'{F5_VARIOGRAM_RATIO}' (thin lines = individual truth x draw combinations, markers = "
                  "mean over them by lag index)"),
            ("C", "PLACEHOLDER - synthetic illustration, NO DATA. Two stacked axes (MSE, UMG) vs anisotropy "
                  "ratio for the 4 methods. Curves are arbitrary seeded smooth curves "
                  f"(seed {PLACEHOLDER_SEED}, {PLACEHOLDER_SEED + 1}) with no expected-result meaning; no "
                  "numeric ticks."),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=[
            f"panel A levels exactly as stored in true_range_by_azimuth.csv: {azi_txt}. Note the level "
            "labelled '3:1' has hmin1 (600 m) LARGER than hmaj1 (300 m) and '2:1' has hmin1 150 m.",
            "panel B comes from anisotropy_variogram_check, whose ratio levels are parameterized "
            "differently: " + ", ".join(
                f"'{k}' = hmaj1 {r.hmaj1:g} m / hmin1 {r.hmin1:g} m"
                for k, r in vario.groupby('ratio_label', sort=False)[['hmaj1', 'hmin1']].first().iterrows())
            + ". Panels A and B therefore do NOT show the same ground-truth models.",
            f"panel B: {n_truth} truth seeds {sorted(int(s) for s in exp['truth_seed'].unique())} x "
            f"{n_draw} sample draws {sorted(int(s) for s in exp['sample_draw_seed'].unique())}; the "
            "top-level variogram_comparison.csv is used (the density_10/ subfolder holds a separate, "
            "different file that is not used here).",
            "panel B experimental mean: lags differ slightly between runs, so runs are averaged by lag "
            "INDEX (mean lag, mean gamma); npair weighting is not applied.",
            f"panel B y axis is cut at {F5_VARIOGRAM_YMAX:g}; {n_off} of {len(exp)} individual experimental "
            f"(lag, gamma) points exceed it (max gamma {exp_max:.2f}); no mean marker is cut "
            f"(max mean-by-lag-index gamma {max_mean:.2f}).",
            "no prediction method (kriging/SGS/RBF/GP) has been run on any anisotropic ground truth.",
        ],
        needed=[
            "Anisotropy axis proper: all 4 methods on anisotropic ground truths at several ratios "
            "(non-axis-aligned direction per docs/experiment_context.md axis 4), MSE and UMG per level, "
            "same sample locations across methods, multiple realizations.",
        ],
    )
    return _save(fig, stem, lines)


# ---------------------------------------------------------------------------
# F6
# ---------------------------------------------------------------------------
def figure_f6(dens):
    stem = "F6_hyperparameter_stability_vs_density"
    gp = _load_gp_lengths()
    n_rep = int(gp.groupby("axis_level").size().iloc[0])
    stats = gp.groupby("axis_level")["length_scale_m"].agg(["mean", "std"]).loc[DENSITY_LEVELS]
    stats["cv"] = stats["std"] / stats["mean"]  # pandas std -> ddof=1

    # Cross-check against the independently tabulated summary.
    summ = _load_summary()
    chk = summ[(summ["method"] == "gp_mle") & (summ["metric"] == "length_scale_practical_range_m")]
    chk = chk.set_index("axis_level").loc[DENSITY_LEVELS]
    if not (np.allclose(chk["mean"], stats["mean"], rtol=1e-9)
            and np.allclose(chk["std"], stats["std"], rtol=1e-9)):
        raise ValueError("length_scale mean/std disagree with metrics_summary.csv.")

    note = (
        f"A: isotropic GP-MLE (one length scale), 1 ground-truth realization, CV = std/mean (ddof = 1) of "
        f"the fitted practical range over {n_rep} sample-location replicates per density level. "
        "B: PLACEHOLDER – no anisotropic-kernel GP (axis-aligned, or with rotation angle) has been "
        "fitted; curve shapes and ordering are arbitrary."
    )
    fig, rect = _new_figure(plot_height_mm=56.0, status_text=note, top_mm=19.0)
    gs = fig.add_gridspec(1, 2, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3], wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    xs = [dens["pct"][lvl] for lvl in DENSITY_LEVELS]
    ax_a.plot(xs, stats["cv"].values, color=METHOD_COLORS["gp_mle"],
              marker=METHOD_MARKERS["gp_mle"], markersize=3.6)
    _style_density_axis(ax_a, dens)
    ax_a.set_ylim(bottom=0)
    ax_a.set_ylabel("CV of fitted practical range (std / mean)")
    ax_a.grid(True)
    ax_a.legend(
        handles=[Line2D([0], [0], color=METHOD_COLORS["gp_mle"], marker=METHOD_MARKERS["gp_mle"],
                        markersize=3.6, label="GP-MLE, isotropic kernel (1 length scale)")],
        loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0)

    t = np.linspace(0.0, 1.0, 5)
    curves = _schematic_curves(2, len(t), PLACEHOLDER_SEED + 2)
    specs = [
        ((0, (4, 2)), "Axis-aligned anisotropic kernel (2 length scales)"),
        ((0, (1, 1.5)), "Rotated anisotropic kernel (2 length scales + angle)"),
        # legend title carries the "(schematic, no data)" tag for both entries
    ]
    h_b = []
    for (ls, label), y in zip(specs, curves):
        ax_b.plot(t, y, color="0.35", linestyle=ls, linewidth=0.9)
        h_b.append(Line2D([0], [0], color="0.35", linestyle=ls, linewidth=0.9,
                          label=label))
    ax_b.set_xlim(-0.08, 1.08)
    ax_b.set_ylim(0.0, 1.0)
    _mark_placeholder(ax_b)
    ax_b.set_xlabel("Sample density (dense → sparse)")
    ax_b.set_ylabel("CV of each fitted hyperparameter (schematic)")
    ax_b.legend(handles=h_b, loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=1,
                borderaxespad=0.0, title="Schematic, no data:", title_fontsize=FONT_PT)

    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")

    lines = _status_file(
        stem,
        inputs=[
            f"{_rel(REPL_LENGTH)} (read with comment='#'; rows method == gp_mle; column length_scale_m)",
            f"{_rel(REPL_SUMMARY)} (metric length_scale_practical_range_m, gp_mle: cross-check of mean/std only)",
            f"{_rel(REPL_SEEDS)} + {_rel(REPL_EXTRA)} (n_samples_actual -> x positions)",
        ],
        panels=[
            ("A", f"REAL DATA - CV (std/mean, ddof=1, {n_rep} replicates) of the isotropic GP-MLE fitted "
                  "practical range vs sample density"),
            ("B", "PLACEHOLDER - synthetic illustration, NO DATA. Same axes format for kernels with more "
                  "hyperparameters. Curves are arbitrary seeded smooth curves "
                  f"(seed {PLACEHOLDER_SEED + 2}); their shape, crossing and ordering carry no meaning; "
                  "no numeric ticks."),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=[
            "panel A values (level: mean m, std m, CV): " + "; ".join(
                f"{lvl}: {stats.loc[lvl, 'mean']:.1f}, {stats.loc[lvl, 'std']:.1f}, {stats.loc[lvl, 'cv']:.3f}"
                for lvl in DENSITY_LEVELS),
            "panel A: mean/std cross-checked (rtol 1e-9) against metrics_summary.csv's "
            "length_scale_practical_range_m rows.",
            f"panel A: single ground-truth realization (truth_seed {dens['truth_seed']}, isotropic, range "
            f"{dens['truth_range_m']:g} m); variation is across sample-location replicates only.",
            "the real and placeholder series were split into separate panels (A real / B placeholder) "
            "instead of sharing one axis.",
            "x positions (panel A) are the REQUESTED sample fraction (actual n): " + _replicate_n_text(dens) + ".",
        ],
        needed=[
            "GP-MLE with (i) an axis-aligned anisotropic (ARD) kernel and (ii) a rotated anisotropic kernel, "
            "fitted over the same sample-location replicates and density levels (and on anisotropic ground "
            "truths), recording every fitted hyperparameter per replicate so its CV can be computed.",
        ],
    )
    return _save(fig, stem, lines)


# ---------------------------------------------------------------------------
# F7
# ---------------------------------------------------------------------------
F7_SUMMARY = PROCESSED / "hyperparam_grid_surface" / "metrics_summary.csv"
F7_OPTIMA = PROCESSED / "hyperparam_grid_surface" / "optima.csv"
F7_PROV = PROCESSED / "hyperparam_grid_surface" / "provenance.json"
F7_METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp"]
F7_TITLES = {"kriging": "Simple kriging", "sgs": "SGS", "rbf_bootstrap": "RBF+bootstrap",
             "gp": "GP (fixed hyperparameters)"}
F7_YLABELS = {"kriging": "Nugget (fraction of sill)", "sgs": "Nugget (fraction of sill)",
              "rbf_bootstrap": "Smoothing λ", "gp": "Noise variance (fraction of sill)"}
F7_YLOG = {"kriging": False, "sgs": False, "rbf_bootstrap": True, "gp": True}
F7_YTICKS = {"kriging": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "sgs": [0.0, 0.1, 0.2, 0.3, 0.4],
             "rbf_bootstrap": [1e-5, 1e-3, 1e-1, 1e1], "gp": [1e-5, 1e-3, 1e-1]}
F7_XTICKS = [106, 212, 424, 849]
F7_BEST = {"mse": "min", "umg": "max"}
F7_CMAP = {"mse": "viridis_r", "umg": "viridis"}   # bright = better on both surfaces
F7_CBAR_LABEL = {"mse": MSE_LABEL, "umg": "UMG"}
F7_TRUE_POINT = (300.0, 0.05)   # base-case truth (range m, nugget fraction); asserted vs provenance below
# vertical layout (mm): colourbar strip, gap, panel; between rows: x ticks + next colourbar ticks/label
F7_PANEL_MM = 30.0
F7_CBAR_MM = 1.5
F7_CBAR_GAP_MM = 1.2
F7_ROW_GAP_MM = 12.5
F7_COL_GAP_MM = 14.0


def _f7_edges(vals, log):
    """Cell edges around the grid nodes: midpoints (in log10 if ``log``) and
    half a step beyond the outer nodes. Cells only -- no interpolation."""
    v = np.log10(np.asarray(vals, float)) if log else np.asarray(vals, float)
    mid = (v[1:] + v[:-1]) / 2.0
    e = np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])
    return 10.0 ** e if log else e


F7_TRUE_MARKER = {"kriging": "o", "sgs": "o", "rbf_bootstrap": "D", "gp": "D"}


def _f7_true_point(method):
    """Base-case truth on each method's axes. Kriging/SGS: (range, nugget) exactly.
    RBF/GP: APPROXIMATE equivalent for visual comparison only (user request) --
    x = 300 m on the same range-equivalent axis; GP noise = nugget (unit sill,
    signal 1 - noise, same units); RBF smoothing = nugget / (1 - nugget), the
    noise-to-signal ratio added to the unit-diagonal Gaussian kernel matrix."""
    r, nug = F7_TRUE_POINT
    if method == "rbf_bootstrap":
        return r, nug / (1.0 - nug)
    return r, nug


F7_N_LEVELS = 14     # filled-contour bands per panel
F7_FINE_N = 200       # display grid per axis for the bilinear upsampling


def _f7_fine(xs, ys, z, ylog):
    """Bilinear interpolation of the node means onto a fine display grid, done in
    the plotted coordinates (log10 range; log10 of param2 where the y axis is log),
    so contour bands are straight between nodes as drawn. Node values are kept
    exactly; the domain is the node hull (no extrapolation)."""
    from scipy.interpolate import RegularGridInterpolator
    tx = np.log10(xs)
    ty = np.log10(ys) if ylog else np.asarray(ys, float)
    fx = np.linspace(tx[0], tx[-1], F7_FINE_N)
    fy = np.linspace(ty[0], ty[-1], F7_FINE_N)
    gy, gx = np.meshgrid(fy, fx, indexing="ij")
    fz = RegularGridInterpolator((ty, tx), z, method="linear")(np.stack([gy, gx], axis=-1))
    return 10.0 ** fx, (10.0 ** fy if ylog else fy), fz


def _f7_grid(summ, method, metric):
    g = summ[(summ["method"] == method) & (summ["metric"] == metric)]
    xs = np.sort(g["range_equiv_m"].unique())
    ys = np.sort(g["param2_value"].unique())
    z = np.full((len(ys), len(xs)), np.nan)
    for _, r in g.iterrows():
        z[np.searchsorted(ys, r["param2_value"]), np.searchsorted(xs, r["range_equiv_m"])] = r["mean"]
    if np.isnan(z).any():
        raise ValueError(f"F7 {method} {metric}: incomplete grid")
    return xs, ys, z


def figure_f7():
    from matplotlib import patheffects
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogFormatterMathtext, NullLocator

    stem = "F7_response_surfaces_accuracy_vs_uncertainty"
    summ = pd.read_csv(F7_SUMMARY)
    opt = pd.read_csv(F7_OPTIMA)
    prov = json.loads(F7_PROV.read_text(encoding="utf-8"))
    n_truths = prov["checks"]["n_truths_per_node"]
    if len(n_truths) != 1:
        raise ValueError(f"F7: unequal truth counts per node {n_truths}")
    n_truths = n_truths[0]
    run_cfg = json.loads((_REPO_ROOT / prov["source_run"] / "run_config.json").read_text(encoding="utf-8"))
    truth = run_cfg["truth"]
    if (truth["hmaj1"], truth["nug"]) != F7_TRUE_POINT:
        raise ValueError(f"F7: base-case truth {truth} != {F7_TRUE_POINT}")
    n_ens = run_cfg["n_ensemble"]
    grid_txt = ", ".join(
        f"{F7_TITLES[m].split(' (')[0]} {len(run_cfg['grids'][m]['range_levels_m'])}×"
        f"{len(run_cfg['grids'][m]['param2_levels'])}" for m in F7_METHODS)

    def opt_row(m, met):
        r = opt[(opt["method"] == m) & (opt["metric"] == met)]
        if len(r) != 1:
            raise ValueError(f"F7 optima: {m} {met}")
        return r.iloc[0]

    note = (
        f"Real data. Colour = mean over {n_truths} ground truths (base case: range {truth['hmaj1']:g} m, nugget "
        f"{truth['nug']:g}, n = {truth['n_samples_requested']} requested) at each grid node, drawn as filled "
        f"contours of a bilinear interpolation between nodes (log axes in log space). Fixed hyperparameters, no tuning; SGS and bootstrap N = {n_ens}. Grids: {grid_txt}. "
        "x = range-equivalent: kriging/SGS spherical range; RBF ε = √(−ln 0.05)/x and GP length scale = "
        "x/√(−2 ln 0.05) (Gaussian kernel = 0.05 at x). GP: signal = 1 − noise in normalized units. Optima from grid "
        f"nodes (MSE min, UMG max). Diamond = approximate true-case equivalent for comparison only: GP noise = nugget, "
        f"RBF λ = nugget/(1 − nugget) (noise-to-signal ratio). Source: {prov['source_run']} (cross-check vs multi-truth run: "
        f"{prov['checks']['crosscheck_verdict']})."
    )
    plot_h = 2 * (F7_CBAR_MM + F7_CBAR_GAP_MM + F7_PANEL_MM) + F7_ROW_GAP_MM
    fig, rect = _new_figure(plot_height_mm=plot_h, status_text=note, top_mm=19.0, left_mm=14.0)
    ratios = [F7_CBAR_MM, F7_CBAR_GAP_MM, F7_PANEL_MM, F7_ROW_GAP_MM, F7_CBAR_MM, F7_CBAR_GAP_MM, F7_PANEL_MM]
    panel_w = ((rect[2] - rect[0]) * FULL_WIDTH_MM - 3 * F7_COL_GAP_MM) / 4.0
    gs = fig.add_gridspec(7, 4, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3],
                          height_ratios=ratios, hspace=0.0, wspace=F7_COL_GAP_MM / panel_w)
    halo = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    letters = iter("ABCDEFGH")
    panel_info = []
    for row, met in enumerate(("mse", "umg")):
        other = "umg" if met == "mse" else "mse"
        for col, m in enumerate(F7_METHODS):
            cax = fig.add_subplot(gs[4 * row, col])
            ax = fig.add_subplot(gs[4 * row + 2, col])
            xs, ys, z = _f7_grid(summ, m, met)
            zmin, zmax = float(np.min(z)), float(np.max(z))
            use_log = met == "mse" and zmax / zmin >= RATIO_LOG_THRESHOLD
            norm = LogNorm(zmin, zmax) if use_log else Normalize(zmin, zmax)
            levels = (np.geomspace(zmin, zmax, F7_N_LEVELS + 1) if use_log
                      else np.linspace(zmin, zmax, F7_N_LEVELS + 1))
            fx, fy, fz = _f7_fine(xs, ys, z, F7_YLOG[m])
            pc = ax.contourf(fx, fy, fz, levels=levels, cmap=F7_CMAP[met], norm=norm, extend="neither")
            ax.contour(fx, fy, fz, levels=levels[1:-1], colors="black", linewidths=0.25, alpha=0.45)
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
            ax.set_xscale("log")
            if F7_YLOG[m]:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.set_xticks(F7_XTICKS)
            ax.set_xticklabels([str(t) for t in F7_XTICKS])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_yticks(F7_YTICKS[m])
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylabel(F7_YLABELS[m])
            if row == 1:
                ax.set_xlabel("Range equivalent (m)")
            # optima (grid nodes)
            own, oth = opt_row(m, met), opt_row(m, other)
            ax.plot(oth["range_equiv_m"], oth["param2_value"], linestyle="none", marker="*", markersize=8,
                    markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.8, path_effects=halo,
                    zorder=6, clip_on=False)
            ax.plot(own["range_equiv_m"], own["param2_value"], linestyle="none", marker="*", markersize=8,
                    markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.4, zorder=7, clip_on=False)
            ax.plot(*_f7_true_point(m), linestyle="none", marker=F7_TRUE_MARKER[m], markersize=4,
                    markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.8, path_effects=halo,
                    zorder=5, clip_on=False)
            cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            cb.set_label(F7_CBAR_LABEL[met], labelpad=1.5)
            cb.outline.set_linewidth(0.5)
            cax.tick_params(length=1.5, pad=1.0)
            # contourf colourbars default to one tick per band edge -> pick a few readable ticks
            if use_log:
                tk = matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)).tick_values(zmin, zmax)
            else:
                tk = matplotlib.ticker.MaxNLocator(nbins=4).tick_values(zmin, zmax)
            tk = [t for t in tk if zmin <= t <= zmax]
            cb.set_ticks(tk)
            cb.set_ticklabels([f"{t:g}" for t in tk])
            cax.xaxis.set_minor_locator(NullLocator())
            if row == 0:
                cax.annotate(F7_TITLES[m], xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 21),
                             textcoords="offset points", ha="center", va="bottom", fontweight="bold")
            letter = next(letters)
            _panel_label(ax, letter, dx_pt=-30.0, dy_pt=12.0)
            panel_info.append((letter, m, met, use_log, zmin, zmax, own))
    fig.legend(
        handles=[
            Line2D([0], [0], linestyle="none", marker="*", markersize=7, markerfacecolor="black",
                   markeredgecolor="black", label="Optimum, this surface"),
            Line2D([0], [0], linestyle="none", marker="*", markersize=7, markerfacecolor="none",
                   markeredgecolor="black", markeredgewidth=0.8, label="Optimum, other surface"),
            Line2D([0], [0], linestyle="none", marker="o", markersize=4, markerfacecolor="none",
                   markeredgecolor="black", markeredgewidth=0.8,
                   label=f"True base case ({truth['hmaj1']:g} m, {truth['nug']:g})"),
            Line2D([0], [0], linestyle="none", marker="D", markersize=4, markerfacecolor="none",
                   markeredgecolor="black", markeredgewidth=0.8,
                   label="Approx. true-case equivalent (RBF, GP)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, borderaxespad=0.3,
    )

    panels = []
    for letter, m, met, use_log, zmin, zmax, own in panel_info:
        panels.append((letter, (
            f"{F7_TITLES[m]}, {met.upper()} surface: mean over {n_truths} truths at each of the "
            f"{len(run_cfg['grids'][m]['range_levels_m'])}x{len(run_cfg['grids'][m]['param2_levels'])} grid nodes "
            f"(filled contours, {F7_N_LEVELS} bands, bilinear interpolation between nodes); colour {'log' if use_log else 'linear'} scale "
            f"{zmin:.4g}-{zmax:.4g}, colormap {F7_CMAP[met]} (bright = better). Filled star = {F7_BEST[met]} node "
            f"({own['range_equiv_m']:g} m, {own['param2_name']} {own['param2_value']:.4g}, {own['mean']:.4f}); "
            "hollow star = the other surface's optimum"
            + ("; open circle = true base-case point." if m in ("kriging", "sgs") else
               f"; open diamond = APPROXIMATE base-case equivalent ({_f7_true_point(m)[0]:g} m, "
               f"{_f7_true_point(m)[1]:.4g}), for visual comparison only.")
        )))
    limits = [
        f"source run {prov['source_run']}: status {prov['checks']['manifest_status']}, failed points "
        f"{prov['checks']['n_failed_total']}, sgsim warnings {prov['checks']['sgsim_warnings_total']:g}, "
        f"cross-check {prov['checks']['crosscheck_verdict']}.",
        "colour scales are per panel (each panel has its own colourbar); MSE uses a log scale where "
        f"max/min >= {RATIO_LOG_THRESHOLD:g}.",
        "contours: node means bilinearly interpolated on a {0}x{0} display grid in the plotted coordinates "
        "(log10 on log axes); values between nodes are interpolated, not computed; domain = outer nodes, no "
        "extrapolation; thin lines = band boundaries.".format(F7_FINE_N),
        "optima (results/processed/hyperparam_grid_surface/optima.csv), best vs runner-up node:",
    ]
    for _, r in opt.iterrows():
        limits.append(
            f"  {r['method']} {r['metric']}: best ({r['range_equiv_m']:g} m, {r['param2_value']:.4g}) "
            f"{r['mean']:.4f} (std over truths {r['std_over_truths']:.4f}); runner-up "
            f"({r['runner_up_range_equiv_m']:g} m, {r['runner_up_param2_value']:.4g}) {r['runner_up_mean']:.4f}; "
            f"gap {r['gap_best_minus_runner_up']:+.4f}"
        )
    lines = _status_file(stem, inputs=[_rel(F7_SUMMARY), _rel(F7_OPTIMA), _rel(F7_PROV),
                                       prov["source_run"] + "/run_config.json"],
                         panels=panels, in_figure_note="DATA STATUS: " + note, limits=limits)
    return _save(fig, stem, lines)


def main():
    _setup_style()
    dens = _density_info()   # F6 (single-truth replicate axis)
    mt = _load_multi_truth()  # F1-F3 (multi_truth_axes E2-E5)
    written = []
    for name, fn, args in (
        ("F1", figure_f1, (mt,)),
        ("F2", figure_f2, (mt,)),
        ("F3", figure_f3, (mt,)),
        ("F3-2", figure_f3_2, (mt,)),
        ("F3-2b", figure_f3_2_maps, (mt,)),
        ("F4", figure_f4, ()),
        ("F5", figure_f5, ()),
        ("F6", figure_f6, (dens,)),
        ("F7", figure_f7, ()),
    ):
        print(name)
        written += fn(*args)
    print(f"{len(written)} files written to {_rel(FIGURES_DIR)}")
    return written


if __name__ == "__main__":
    main()
