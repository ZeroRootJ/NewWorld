"""Paper STORYBOARD figure set F1-F7: the paper's argument laid out with the
data that already exists, plus clearly marked placeholders where no experiment
has been run yet.

NO SIMULATION IS RUN HERE. This script only READS existing CSV/JSON files under
results/processed/ (and two integer constants from src/experiments/) and draws.
results/raw/ is never touched.

Figures (results/figures/paper_storyboard/, each as vector PDF + 300 dpi PNG +
a ``.status.txt`` sidecar):

  F1 (1-a)  accuracy (MSE) vs calibration (UMG) along range / nugget / density
  F2 (1-a)  base-case accuracy plot + mean predictive variance vs density
  F3 (1-b)  kriging vs GP-MLE along sample density (MSE, UMG, 90% coverage)
            + GP-MLE fitted range
  F4 (1-b)  GP-MLE fitted range / nugget vs ground truth (10 realizations/level)
  F5 (1-b)  anisotropy: true range vs azimuth, variogram check, PLACEHOLDER metrics
  F6 (1-b)  hyperparameter stability vs density: real (isotropic) + PLACEHOLDER
  F7 (2)    response surfaces of MSE and UMG over (range, nugget): PLACEHOLDER

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
  The single exception is F7, where the coordinator's spec asks for two bowls
  with their optima at different places; that is stated in F7's status file.

Run with:
.venv/Scripts/python.exe -m results.processed.paper_storyboard.make_storyboard_figures
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.rbf_bootstrap import N_BOOTSTRAP  # noqa: E402
from src.experiments.sgs import N_REALIZATIONS  # noqa: E402

PROCESSED = _REPO_ROOT / "results" / "processed"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "paper_storyboard"

# --- inputs (all read-only) ---------------------------------------------------
RANGE_METRICS = PROCESSED / "range_axis" / "metrics.csv"
NUGGET_METRICS = PROCESSED / "nugget_axis" / "metrics.csv"
REPL_SUMMARY = PROCESSED / "sample_replicate_axis" / "metrics_summary.csv"
REPL_SEEDS = PROCESSED / "sample_replicate_axis" / "replicate_seeds.json"
REPL_EXTRA = PROCESSED / "sample_replicate_axis" / "extra_density_levels_record.json"
REPL_LENGTH = PROCESSED / "sample_replicate_axis" / "length_scale_by_replicate.csv"
REPL_VAR_SOURCES = PROCESSED / "sample_replicate_axis" / "variance_metric_sources.csv"
REPL_COV90_SUMMARY = PROCESSED / "sample_replicate_axis" / "coverage90_summary.csv"
REPL_COV90 = PROCESSED / "sample_replicate_axis" / "coverage90.csv"
BASE_CURVES = PROCESSED / "base_case" / "accuracy_plot_curves.csv"
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
def _load_tidy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    counts = df.groupby(["axis_level", "method", "metric"]).size()
    if counts.max() != 1:
        raise ValueError(f"{_rel(path)}: expected 1 row per (level, method, metric).")
    if set(df["method"]) != set(METHODS):
        raise ValueError(f"{_rel(path)}: unexpected method set {sorted(set(df['method']))}.")
    return df


def _load_summary() -> pd.DataFrame:
    df = pd.read_csv(REPL_SUMMARY)
    df["axis_level"] = df["axis_level"].astype(str)
    if set(df["axis_level"]) != set(DENSITY_LEVELS):
        raise ValueError(f"{_rel(REPL_SUMMARY)}: unexpected levels {sorted(set(df['axis_level']))}.")
    return df


def _density_info() -> dict:
    """ACTUAL (post grid-cell dedup) sample counts per density level, per
    replicate, from the replicate axis's own records. x positions on every
    density axis are the replicate-MEAN actual fraction of the nx*ny grid."""
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
    pct = {lvl: 100.0 * float(np.mean(n_actual[lvl])) / n_grid for lvl in DENSITY_LEVELS}
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
    ax.set_xticklabels([f"{x:.1f}" for x in xs])
    ax.minorticks_off()
    if xlabel:
        ax.set_xlabel(DENSITY_XLABEL)


def _plot_density_metric(ax, summary, dens, metric, methods=METHODS, faded=()):
    """mean +- 1 std across replicates, methods offset in x so bars do not overlap."""
    for m_idx, method in enumerate(METHODS):
        if method not in methods:
            continue
        off = (m_idx - (len(METHODS) - 1) / 2) * METHOD_X_OFFSET_FRAC
        sub = summary[(summary["method"] == method) & (summary["metric"] == metric)]
        sub = sub.set_index("axis_level").loc[DENSITY_LEVELS]
        x = [dens["pct"][lvl] * (1.0 + off) for lvl in DENSITY_LEVELS]
        is_faded = method in faded
        ax.errorbar(
            x, sub["mean"].values, yerr=sub["std"].values,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            markersize=2.6 if is_faded else 3.6,
            linewidth=0.6 if is_faded else 1.0,
            elinewidth=0.5, capsize=1.5, capthick=0.5,
            alpha=0.32 if is_faded else 1.0,
            zorder=2 if is_faded else 3,
        )


def _method_handles(methods=METHODS, faded=()):
    return [
        Line2D([0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], markersize=3.6,
               linewidth=1.0, alpha=0.32 if m in faded else 1.0, label=METHOD_LABELS[m])
        for m in methods
    ]


def _replicate_n_text(dens) -> str:
    return "; ".join(
        f"{dens['pct'][lvl]:.1f}% = n {min(dens['n_actual'][lvl])}-{max(dens['n_actual'][lvl])} "
        f"(requested {dens['n_requested'][lvl]})"
        for lvl in DENSITY_LEVELS
    )


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------
def figure_f1(dens, summary):
    stem = "F1_accuracy_vs_calibration_three_axes"
    rng_df = _load_tidy(RANGE_METRICS)
    nug_df = _load_tidy(NUGGET_METRICS)
    n_rep = int(summary["n_replicates"].unique().item())
    n_range_lvls = rng_df["axis_level"].nunique()
    n_nug_lvls = nug_df["axis_level"].nunique()

    note = (
        f"range axis ({n_range_lvls} levels) and nugget axis ({n_nug_lvls} levels): 1 ground-truth "
        f"realization and 1 sample set per level (no error bars possible). Density axis: 1 ground-truth "
        f"realization, {n_rep} sample-location replicates per level (points = mean, bars = ±1 std). "
        f"RBF+bootstrap N_BOOTSTRAP = {N_BOOTSTRAP}; SGS N_REALIZATIONS = {N_REALIZATIONS} – "
        "to be re-run at production settings."
    )
    fig, rect = _new_figure(plot_height_mm=84.0, status_text=note, top_mm=13.0)
    gs = fig.add_gridspec(2, 3, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3],
                          wspace=0.24, hspace=0.16)
    axes = np.empty((2, 3), dtype=object)
    for c in range(3):
        axes[0, c] = fig.add_subplot(gs[0, c])
        axes[1, c] = fig.add_subplot(gs[1, c], sharex=axes[0, c],
                                     sharey=axes[1, 0] if c > 0 else None)

    tidy_cols = [
        (rng_df, "Ground-truth variogram range (m)"),
        (nug_df, "Ground-truth nugget (fraction of sill)"),
    ]
    for c, (df, xlabel) in enumerate(tidy_cols):
        for r, metric in enumerate(["mse", "umg"]):
            ax = axes[r, c]
            for method in METHODS:
                sub = df[(df["method"] == method) & (df["metric"] == metric)].sort_values("axis_level")
                ax.plot(sub["axis_level"], sub["value"], color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method], markersize=3.6)
            ax.set_xticks(sorted(df["axis_level"].unique()))
        axes[1, c].set_xlabel(xlabel)
    for r, metric in enumerate(["mse", "umg"]):
        _plot_density_metric(axes[r, 2], summary, dens, metric)
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

    lines = _status_file(
        stem,
        inputs=[
            f"{_rel(RANGE_METRICS)} (columns axis_level, method, metric, value; metrics mse, umg)",
            f"{_rel(NUGGET_METRICS)} (columns axis_level, method, metric, value; metrics mse, umg)",
            f"{_rel(REPL_SUMMARY)} (columns axis_level, method, metric, mean, std, n_replicates; metrics mse, umg)",
            f"{_rel(REPL_SEEDS)} + {_rel(REPL_EXTRA)} (n_samples_actual -> x positions)",
            f"{_rel(REPL_VAR_SOURCES)} (n_cells_averaged, only to recover grid size {dens['n_grid']})",
            f"src/experiments/rbf_bootstrap.py N_BOOTSTRAP = {N_BOOTSTRAP}; "
            f"src/experiments/sgs.py N_REALIZATIONS = {N_REALIZATIONS}",
        ],
        panels=[
            ("A", "REAL DATA - MSE vs ground-truth range, 4 methods"),
            ("B", "REAL DATA - MSE vs ground-truth nugget, 4 methods"),
            ("C", "REAL DATA - MSE vs sample density, 4 methods, mean +-1 std (ddof=1 per metrics_summary.csv)"),
            ("D", "REAL DATA - UMG vs ground-truth range"),
            ("E", "REAL DATA - UMG vs ground-truth nugget"),
            ("F", "REAL DATA - UMG vs sample density, mean +-1 std"),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=[
            f"range_axis/metrics.csv: exactly 1 row per (level, method, metric) -> 1 realization per level; "
            f"levels {sorted(rng_df['axis_level'].unique())} m.",
            f"nugget_axis/metrics.csv: exactly 1 row per (level, method, metric) -> 1 realization per level; "
            f"levels {sorted(nug_df['axis_level'].unique())}.",
            f"density axis: n_replicates = {n_rep} at every level; same truth (truth_seed "
            f"{dens['truth_seed']}, range {dens['truth_range_m']:g} m); sample seeds {dens['sample_seeds']} "
            "reused at every level.",
            "density x positions are the replicate-MEAN actual sample fraction: " + _replicate_n_text(dens) + ".",
            "density-axis methods are offset multiplicatively in x (+-7% steps) only so error bars do not "
            "overlap; tick positions are the true level values.",
            "UMG row (D-F) shares one y axis; MSE row (A-C) has independent y axes.",
        ],
    )
    return _save(fig, stem, lines)


# ---------------------------------------------------------------------------
# F2
# ---------------------------------------------------------------------------
def figure_f2(dens, summary):
    stem = "F2_base_case_calibration_and_variance"
    curves = pd.read_csv(BASE_CURVES)
    if set(curves["method"]) != set(METHODS):
        raise ValueError(f"{_rel(BASE_CURVES)}: unexpected methods.")
    if "variance_mean" not in set(summary["metric"]):
        raise ValueError(f"{_rel(REPL_SUMMARY)} has no variance_mean metric.")
    src = pd.read_csv(REPL_VAR_SOURCES)
    units = sorted(src["units"].unique())
    if units != ["porosity %^2"]:
        raise ValueError(f"variance units are not uniform: {units}")
    src_files = src.groupby("method")["variance_source_file"].unique()
    if any(len(v) != 1 for v in src_files):
        raise ValueError("more than one variance source file per method.")
    mc_methods = sorted(src[src["is_monte_carlo_backtransform_approximation"]]["method"].unique())
    n_rep = int(summary["n_replicates"].unique().item())

    note = (
        "A: base case = 1 ground-truth realization, 1 sample set. B: 1 ground-truth realization, "
        f"{n_rep} sample-location replicates per density level (mean ±1 std); all variances in "
        f"porosity %² – kriging's is a Monte Carlo back-transform of its normal-score variance, "
        f"the other three are native. RBF+bootstrap N_BOOTSTRAP = {N_BOOTSTRAP}; SGS N_REALIZATIONS = "
        f"{N_REALIZATIONS} – to be re-run at production settings."
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

    _plot_density_metric(ax_b, summary, dens, "variance_mean")
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
            f"{_rel(BASE_CURVES)} (columns method, p_interval, fraction_in)",
            f"{_rel(REPL_SUMMARY)} (metric variance_mean; columns mean, std, n_replicates)",
            f"{_rel(REPL_VAR_SOURCES)} (columns units, variance_source_file, "
            "is_monte_carlo_backtransform_approximation - unit/provenance check)",
            f"{_rel(REPL_SEEDS)} + {_rel(REPL_EXTRA)} (n_samples_actual -> x positions)",
            f"src/experiments/rbf_bootstrap.py N_BOOTSTRAP = {N_BOOTSTRAP}; "
            f"src/experiments/sgs.py N_REALIZATIONS = {N_REALIZATIONS}",
        ],
        panels=[
            ("A", "REAL DATA - base-case accuracy plot (fraction of truth inside each nominal interval), "
                  f"{curves.groupby('method').size().iloc[0]} probability levels per method, 1:1 reference line"),
            ("B", "REAL DATA - variance_mean vs sample density, mean +-1 std over replicates"),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=[
            "panel A: single ground-truth realization and single sample set (base case); the curves "
            "come from base_case/accuracy_plot_curves.csv as stored.",
            f"panel B: variance_mean WAS present in sample_replicate_axis/metrics_summary.csv, so the "
            f"{n_rep}-replicate mean/std is used (sample_density_axis/metrics.csv single-run values are not used).",
            "panel B variance source arrays (variance_metric_sources.csv): "
            + "; ".join(f"{m}: {src_files[m][0]}" for m in METHODS)
            + f". units all '{units[0]}'. Monte Carlo back-transform approximation: {mc_methods}.",
            "panel B x positions are the replicate-MEAN actual sample fraction: " + _replicate_n_text(dens) + ".",
            "panel B y axis is linear and autoscaled; an error bar (mean - 1 std) may extend below 0 "
            "where std > mean - this is the plotted +-1 std, not a negative variance.",
        ],
    )
    return _save(fig, stem, lines)


# ---------------------------------------------------------------------------
# F3
# ---------------------------------------------------------------------------
def _load_gp_lengths() -> pd.DataFrame:
    df = pd.read_csv(REPL_LENGTH, comment="#")
    df["axis_level"] = df["axis_level"].astype(str)
    gp = df[df["method"] == "gp_mle"].copy()
    counts = gp.groupby("axis_level").size()
    if set(counts.index) != set(DENSITY_LEVELS) or counts.nunique() != 1:
        raise ValueError(f"{_rel(REPL_LENGTH)}: unexpected gp_mle rows per level: {counts.to_dict()}")
    return gp


def _load_cov90_summary() -> pd.DataFrame:
    """coverage90_summary.csv (written by
    results/processed/sample_replicate_axis/make_coverage90.py) in the same
    (axis_level, method, metric, mean, std, n_replicates) shape as
    metrics_summary.csv, so _plot_density_metric can draw it unchanged."""
    df = pd.read_csv(REPL_COV90_SUMMARY, comment="#")
    df["axis_level"] = df["axis_level"].astype(str)
    if set(df["metric"]) != {"coverage90"}:
        raise ValueError(f"{_rel(REPL_COV90_SUMMARY)}: unexpected metrics {sorted(set(df['metric']))}.")
    if set(df["axis_level"]) != set(DENSITY_LEVELS) or set(df["method"]) != set(METHODS):
        raise ValueError(f"{_rel(REPL_COV90_SUMMARY)}: unexpected levels/methods.")
    if df.groupby(["axis_level", "method"]).size().max() != 1:
        raise ValueError(f"{_rel(REPL_COV90_SUMMARY)}: duplicate (level, method) rows.")
    return df


def figure_f3(dens, summary):
    stem = "F3_gp_vs_sample_density"
    gp = _load_gp_lengths()
    n_rep = int(gp.groupby("axis_level").size().iloc[0])
    spread = float(gp["length_scale_m"].max() / gp["length_scale_m"].min())
    use_log = spread >= RATIO_LOG_THRESHOLD
    faded = ("sgs", "rbf_bootstrap")
    cov = _load_cov90_summary()
    n_rep_cov = int(cov["n_replicates"].unique().item())
    if n_rep_cov != n_rep:
        raise ValueError(f"coverage90 n_replicates {n_rep_cov} != GP length rows per level {n_rep}.")

    note = (
        f"1 ground-truth realization (range {dens['truth_range_m']:g} m), {n_rep} sample-location "
        f"replicates per density level; A–C: mean ±1 std; C: fraction of evaluated cells inside the "
        f"central {COV90_NOMINAL:.0%} predictive interval (kriging: normal-score interval back-transformed; "
        f"SGS, RBF+bootstrap, GP-MLE: Gaussian mean ± z·std interval); D: one point per replicate (fitted "
        f"practical range = RBF-kernel length scale converted at the 0.05 correlation cutoff; max/min "
        f"over all points = {spread:.1f}×, {'log' if use_log else 'linear'} y axis). "
        f"RBF+bootstrap N_BOOTSTRAP = {N_BOOTSTRAP}; SGS N_REALIZATIONS = {N_REALIZATIONS} – to be "
        "re-run at production settings."
    )
    fig, rect = _new_figure(plot_height_mm=F3_PLOT_HEIGHT_MM, status_text=note, top_mm=14.0,
                            right_mm=F3_RIGHT_MM)
    gs = fig.add_gridspec(1, 4, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3], wspace=0.42)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[0, i]) for i in range(4))

    _plot_density_metric(ax_a, summary, dens, "mse", faded=faded)
    _plot_density_metric(ax_b, summary, dens, "umg", faded=faded)
    ax_b.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, zorder=1)
    _plot_density_metric(ax_c, cov, dens, "coverage90", faded=faded)
    ax_c.axhline(COV90_NOMINAL, color="gray", linestyle="--", linewidth=0.6, zorder=1)
    ax_a.set_ylabel(MSE_LABEL)
    ax_b.set_ylabel(UMG_LABEL)
    ax_c.set_ylabel(COV90_LABEL)

    # Deterministic (non-random) horizontal spread of the replicate points.
    spread_frac = np.linspace(-0.10, 0.10, n_rep)
    for lvl in DENSITY_LEVELS:
        sub = gp[gp["axis_level"] == lvl].sort_values("replicate")
        ax_d.scatter(
            dens["pct"][lvl] * (1.0 + spread_frac), sub["length_scale_m"],
            s=9, marker=METHOD_MARKERS["gp_mle"], facecolors=METHOD_COLORS["gp_mle"],
            edgecolors="none", alpha=0.75, zorder=3,
        )
    ax_d.axhline(dens["truth_range_m"], color="black", linestyle="--", linewidth=0.7, zorder=2)
    if use_log:
        ax_d.set_yscale("log")
    else:
        ax_d.set_ylim(bottom=0)
    ax_d.set_ylabel("GP-MLE fitted practical range (m)")

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
        _style_density_axis(ax, dens, xlabel=False)
        ax.set_xlabel(DENSITY_XLABEL_SHORT)
        ax.grid(True)
        _panel_label(ax, letter)

    # Panel C y axis is autoscaled (errorbar includes the +-1 std bars); verify
    # nothing is clipped.
    cov_lo = float((cov["mean"] - cov["std"]).min())
    cov_hi = float((cov["mean"] + cov["std"]).max())
    ylim_c = ax_c.get_ylim()
    if ylim_c[0] > cov_lo or ylim_c[1] < cov_hi:
        raise ValueError(f"panel C y range {ylim_c} clips the mean +-1 std range [{cov_lo}, {cov_hi}].")

    handles = _method_handles(faded=faded) + [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.6, label="Ideal (UMG = 1)"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.6,
               label=f"Nominal ({COV90_NOMINAL:.2f})"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.7,
               label=f"Ground-truth range ({dens['truth_range_m']:g} m)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=F3_LEGEND_NCOL, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.4)

    per_level = gp.groupby("axis_level")["length_scale_m"].agg(["min", "max"]).loc[DENSITY_LEVELS]
    lines = _status_file(
        stem,
        inputs=[
            f"{_rel(REPL_SUMMARY)} (metrics mse, umg; columns mean, std, n_replicates)",
            f"{_rel(REPL_COV90_SUMMARY)} (read with comment='#'; metric coverage90; columns mean, std, "
            "n_replicates) - written by results/processed/sample_replicate_axis/make_coverage90.py; "
            f"per-replicate values in {_rel(REPL_COV90)}",
            f"{_rel(REPL_LENGTH)} (read with comment='#'; rows method == gp_mle; column length_scale_m)",
            f"{_rel(REPL_SEEDS)} + {_rel(REPL_EXTRA)} (n_samples_actual -> x positions; hmaj1 -> truth range)",
            f"src/experiments/rbf_bootstrap.py N_BOOTSTRAP = {N_BOOTSTRAP}; "
            f"src/experiments/sgs.py N_REALIZATIONS = {N_REALIZATIONS}",
        ],
        panels=[
            ("A", "REAL DATA - MSE vs density; kriging and GP-MLE at full strength, SGS and RBF+bootstrap "
                  "drawn faded (same data, alpha 0.32)"),
            ("B", "REAL DATA - UMG vs density, same emphasis"),
            ("C", f"REAL DATA - 90% interval coverage vs density, same emphasis, mean +-1 std (ddof=1); "
                  f"dashed gray line = nominal {COV90_NOMINAL:.2f}"),
            ("D", f"REAL DATA - GP-MLE fitted practical range, one point per replicate ({n_rep} per level), "
                  f"dashed line = ground-truth range {dens['truth_range_m']:g} m"),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=[
            "length_definition in the CSV: " + str(gp["length_definition"].unique()[0]),
            f"panel C coverage90 = accuracy-plot fraction_in evaluated at exactly p = {COV90_NOMINAL:.2f} "
            "(strict inequality, conditioning-sample cells excluded), computed through the SAME per-method "
            "path as UMG in src/experiments/evaluate_sample_replicate_axis.py:evaluate_one_cell; "
            "make_coverage90.py re-derives UMG from the same calls and asserts it equals metrics.csv (atol 1e-12).",
            "panel C kriging: normal-score central interval back-transformed per cell (kriging_fraction_in, "
            "same tail parameters as the point back-transform).",
            "panel C SGS, RBF+bootstrap and GP-MLE: Gaussian percentile method, interval = mean +- z*std from "
            "(sgs_mean_map, sgs_var_map) / (bootstrap_mean_map, bootstrap_var_map) / (posterior_mean_map, "
            "posterior_var_map) - NOT empirical quantiles of the SGS realizations or bootstrap replicates.",
            f"panel C SGS ensemble size N_REALIZATIONS = {N_REALIZATIONS} (src/experiments/sgs.py); "
            f"RBF+bootstrap N_BOOTSTRAP = {N_BOOTSTRAP}.",
            "panel C y axis autoscaled to include every mean +-1 std bar (checked at run time; no clipping).",
            f"panel D y-axis rule: log if max/min >= {RATIO_LOG_THRESHOLD:g}x; observed {spread:.2f}x -> "
            f"{'log' if use_log else 'linear'}.",
            "panel D per-level min-max (m): " + "; ".join(
                f"level {lvl}: {per_level.loc[lvl, 'min']:.0f}-{per_level.loc[lvl, 'max']:.0f}"
                for lvl in DENSITY_LEVELS),
            "panel D points are spread horizontally by a fixed (non-random) +-10% pattern for visibility only.",
            "single ground-truth realization (truth_seed "
            f"{dens['truth_seed']}); only sample locations vary between replicates.",
            "x positions are the replicate-MEAN actual sample fraction: " + _replicate_n_text(dens) + ".",
        ],
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
            "x positions (panel A) are the replicate-MEAN actual sample fraction: " + _replicate_n_text(dens) + ".",
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
def figure_f7():
    stem = "F7_response_surfaces_accuracy_vs_uncertainty"
    note = (
        "A, B: PLACEHOLDER – no hyperparameter grid search has been run. Both surfaces are "
        "synthetic quadratic bowls; contour shapes and the positions of both optima are arbitrary and "
        "were not derived from any result."
    )
    fig, rect = _new_figure(plot_height_mm=66.0, status_text=note, top_mm=17.0)
    gs = fig.add_gridspec(1, 2, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3], wspace=0.20)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    g = np.linspace(0.0, 1.0, 121)
    xx, yy = np.meshgrid(g, g)

    def bowl(x0, y0, a, b, rot):
        dx, dy = xx - x0, yy - y0
        u = np.cos(rot) * dx + np.sin(rot) * dy
        w = -np.sin(rot) * dx + np.cos(rot) * dy
        return a * u ** 2 + b * w ** 2

    # Arbitrary, fixed positions (spec: the two panels' optima at different places).
    opt_a = (0.38, 0.36)
    opt_b = (0.64, 0.62)
    z_a = bowl(*opt_a, 1.0, 2.2, 0.5)
    z_b = bowl(*opt_b, 2.0, 1.0, -0.4)
    levels = np.linspace(0.02, 0.9, 8)
    for ax, z, own, other, own_lbl, other_lbl in (
        (ax_a, z_a, opt_a, opt_b, "MSE optimum", "UMG optimum (from B)"),
        (ax_b, z_b, opt_b, opt_a, "UMG optimum", "MSE optimum (from A)"),
    ):
        _mark_placeholder(ax)
        ax.contour(xx, yy, z, levels=levels, colors="0.35", linewidths=0.5, linestyles="dashed")
        ax.plot(*own, linestyle="none", marker="*", markersize=9, markerfacecolor="black",
                markeredgecolor="black", zorder=5)
        ax.plot(*other, linestyle="none", marker="*", markersize=9, markerfacecolor="none",
                markeredgecolor="black", markeredgewidth=0.7, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Range hyperparameter (low → high)")
        ax.set_ylabel("Nugget hyperparameter (low → high)")
        ax.legend(
            handles=[
                Line2D([0], [0], linestyle="none", marker="*", markersize=7, markerfacecolor="black",
                       markeredgecolor="black", label=own_lbl + " (schematic, no data)"),
                Line2D([0], [0], linestyle="none", marker="*", markersize=7, markerfacecolor="none",
                       markeredgecolor="black", markeredgewidth=0.7,
                       label=other_lbl + " (schematic, no data)"),
            ],
            loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=1, borderaxespad=0.0,
            title="Accuracy surface: MSE" if ax is ax_a else "Uncertainty-goodness surface: UMG",
            title_fontsize=FONT_PT,
        )
    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")

    lines = _status_file(
        stem,
        inputs=[],
        panels=[
            ("A", "PLACEHOLDER - synthetic illustration, NO DATA. Contours of an accuracy (MSE) response "
                  "surface over a (range, nugget) hyperparameter grid; filled star = this panel's optimum, "
                  "hollow star = panel B's optimum."),
            ("B", "PLACEHOLDER - synthetic illustration, NO DATA. Same for uncertainty goodness (UMG); "
                  "filled star = this panel's optimum, hollow star = panel A's optimum."),
        ],
        in_figure_note="DATA STATUS: " + note,
        limits=[
            "no input file is read. Both surfaces are analytic quadratic bowls with hard-coded, arbitrary "
            f"centres {opt_a} and {opt_b} in unit-square axis coordinates; no random numbers are used.",
            "that the two optima sit at different positions is the layout requested for this storyboard "
            "slot; it is NOT a result and is not supported by any data in this repository.",
            "no numeric tick labels; contour levels are unlabeled.",
        ],
        needed=[
            "Hyperparameter grid (range x nugget) for the predictor(s) under study, evaluating BOTH MSE and "
            "UMG at every grid node on the same ground truth(s) and sample set(s), so the two response "
            "surfaces and their optima can be drawn from data.",
        ],
    )
    return _save(fig, stem, lines)


def main():
    _setup_style()
    dens = _density_info()
    summary = _load_summary()
    written = []
    for name, fn, args in (
        ("F1", figure_f1, (dens, summary)),
        ("F2", figure_f2, (dens, summary)),
        ("F3", figure_f3, (dens, summary)),
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
