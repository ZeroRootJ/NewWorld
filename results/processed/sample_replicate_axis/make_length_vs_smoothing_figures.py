"""Length scale vs. "smoothing" parameter, for GP-MLE and RBF+bootstrap
separately, across the 10 sample-seed replicates (src/experiments/
sample_replicate_axis.py) at all 3 sample-density-axis levels ("5"/"2"/"1"
-> n_samples_requested = 125/50/25).

WHAT this plots
----------------
Two figures, each ONE ROW of 3 scatter panels (5% / 2% / 1%, dense -> sparse
left to right, matching this project's standing panel-ordering convention --
e.g. make_length_and_variogram_figures.py's density-axis panels). Every panel
shows all 10 replicates (rep0..rep9, sample_seed 1001-1010) as points:
  x = length_scale_m, taken AS-IS from
      results/processed/sample_replicate_axis/length_scale_by_replicate.csv
      (already computed by evaluate_sample_replicate_axis.py's length-scale
      pipeline -- this script does NOT recompute or re-derive that
      conversion; see that CSV's own per-row ``length_definition`` /
      ``source`` columns for how each method's value was derived).
  y = a method-specific "smoothing" quantity, read FRESH from each run's own
      manifest.json (paths taken from source_runs.json, one manifest per
      (axis_level, replicate, method) cell -- NOT reused from any pre-joined
      table, since neither field lives in length_scale_by_replicate.csv):
        - rbf_bootstrap: params.best_smoothing -- the CV-selected
          regularization/smoothing parameter for scipy's RBFInterpolator,
          chosen from src/experiments/rbf_bootstrap.py's SMOOTHING_GRID.
          This is a LITERAL "smoothing" hyperparameter in this codebase.
        - gp_mle: params.fitted_hyperparameters.noise_variance_real_units --
          the MLE-fitted GP nugget/noise variance, converted to real
          (Porosity %^2) units.

WHY noise_variance_real_units is GP-MLE's y-axis (orchestrator decision,
stated explicitly so no reader is misled)
------------------------------------------------------------------------------
This codebase has NO hyperparameter literally named "smoothing" for GP-MLE.
The orchestrator judged noise_variance_real_units to be the closest analog to
RBF's best_smoothing: both trade off exact interpolation of the conditioning
data against a smoother fit -- RBF's smoothing parameter directly relaxes the
exact-interpolation constraint in the RBF system, and the GP's nugget/noise
term plays the equivalent role in a GP posterior (a larger noise variance
shrinks the posterior mean away from exact interpolation of the training
points and inflates predictive variance, exactly as CLAUDE.md's stated
Claim 2 describes -- "GPR ... risk: nugget을 신호로 잘못 학습해 불확실성을
과소평가"). This script's y-axis label and caption say "noise variance
(Porosity %^2) -- GP's smoothing-equivalent regularization term", NEVER a bare
"smoothing", so a reader is not misled into thinking a hyperparameter of that
exact name exists for GP-MLE in this codebase. The field itself comes
straight from gp_mle.py's manifest (params.fitted_hyperparameters, alongside
length_scale_m, signal_variance_normalized, noise_variance_normalized,
y_train_std, signal_variance_real_units) -- see any gp_mle manifest.json for
the full block.

AXIS SCALE CHOICES (both log, both panels of both figures)
------------------------------------------------------------------------------
Y-AXIS: log, in both figures. Observed ranges (see VERIFICATION output below,
printed by this script every run) span >=3 orders of magnitude within a
single figure:
  - rbf_bootstrap best_smoothing: 0.001 to 1.0 across all 30 (level,
    replicate) cells (SMOOTHING_GRID is itself log-like: values selected by
    CV from a small discrete grid, not a continuum).
  - gp_mle noise_variance_real_units: ~5.5e-5 to ~5.37 across all 30 cells --
    over 4.5 orders of magnitude. A linear y-axis would compress every level
    "5" replicate's noise variance (order 1, see below) into indistinguishable
    dots on the same scale as level "1" replicates near the MLE's lower
    bound (order 1e-5).
A linear y-axis for either quantity would make small-but-real differences
among the majority of points invisible next to the few largest values.

X-AXIS: log, in both figures, for the SAME length_scale_m quantity used
throughout this project's sample_replicate_axis outputs. Within a single
panel the fitted length scale already spans a large MULTIPLICATIVE range at
the sparser levels (this project's own case-study figure,
make_length_case_study_figures.py, already frames these swings
multiplicatively -- "~5x (gp_mle)" and "~11x (rbf_bootstrap)" at the 1%
level): this script's own re-tabulation from length_scale_by_replicate.csv
confirms a ~1.5x-2.4x within-panel spread at the 5% level growing to
~5.1x (gp_mle) / ~11.3x (rbf_bootstrap) at the 1% level. Since length scale is
a fundamentally multiplicative quantity here (and the y-axis is already log
for the reason above), a log x-axis keeps the sparsest-level panel's few very
large fitted length scales from stretching the panel so far that the tighter
majority of points collapse together, and keeps the two axes' visual
treatment consistent within each panel.

VERIFICATION (same "verify before trusting" pattern this project's other
processed-data scripts already use, e.g. make_length_case_study_figures.py's
DECISIVE CORRECTNESS CHECK)
------------------------------------------------------------------------------
For each of the 2 methods x 3 axis levels x 10 replicates = 60 cells, this
script (a) reads that cell's run directory from source_runs.json, (b) opens
that run's OWN manifest.json and extracts the method-specific y-field
(raising KeyError loudly if the field is missing -- never silently defaulting),
and (c) looks up that cell's x-value (length_scale_m) from
length_scale_by_replicate.csv, requiring EXACTLY ONE matching
(axis_level, replicate, method) row (raising ValueError otherwise). After
assembly, this script asserts each (method, axis_level) group has EXACTLY 10
rows and each method has EXACTLY 30 rows (3 levels x 10 replicates) before
plotting anything -- so a missing or duplicated manifest can never be plotted
silently. The observed per-axis value ranges (both x and y, per level and
overall) are printed to stdout every run so they can be sanity-checked against
already-known numbers (e.g. rbf_bootstrap length scale at the 1% level
spanning ~141-1604 m, matching make_length_case_study_figures.py's
independently-confirmed MIN/MAX replicates for that method/level).

Run with:
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_length_vs_smoothing_figures
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

from src.experiments.base_case import NX, NY  # noqa: E402
from src.experiments.sample_density_axis import ALL_AXIS_LEVELS  # noqa: E402
from src.experiments.sample_replicate_axis import (  # noqa: E402
    AXIS_HMAJ1,
    N_SAMPLES_REQUESTED_BY_LEVEL,
    REPLICATE_IDS,
    TRUTH_SEED,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "sample_replicate_axis"

# Matches make_sample_replicate_figures.py's DPI, the most similar existing
# layout in this directory (also a bare 1x3 scatter grid, no imshow maps --
# unlike make_length_case_study_figures.py's 2x6 map-heavy layout, which
# instead uses 300 dpi).
FIG_DPI = 600

GRID_CELLS = NX * NY  # 2500; used only to state each level's requested % of
# the grid in panel titles (125/2500=5%, 50/2500=2%, 25/2500=1%, exactly, by
# construction of N_SAMPLES_REQUESTED_BY_LEVEL -- see sample_density_axis.py).

# Colors/markers redefined identically (NOT imported), following this
# directory's own established precedent (make_sample_replicate_figures.py's
# module docstring) of not importing make_sample_density_figures.py's
# METHOD_COLORS/METHOD_MARKERS to avoid that other module's file-reading
# import-time side effects. Only the 2 methods this script uses are included.
METHOD_COLORS = {"rbf_bootstrap": "tab:orange", "gp_mle": "tab:red"}
METHOD_MARKERS = {"rbf_bootstrap": "^", "gp_mle": "D"}
METHOD_LABELS = {"rbf_bootstrap": "RBF+bootstrap", "gp_mle": "GP-MLE"}

Y_FIELD_LABELS = {
    "rbf_bootstrap": "CV-selected RBF smoothing parameter\n(params.best_smoothing, log scale)",
    "gp_mle": (
        "Noise variance (Porosity %$^2$) -- GP's smoothing-equivalent\n"
        "regularization term (params.fitted_hyperparameters."
        "noise_variance_real_units), log scale"
    ),
}

X_LABEL = "Fitted length scale (m, log scale) -- from length_scale_by_replicate.csv"


def load_length_df() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "length_scale_by_replicate.csv", comment="#")
    df["axis_level"] = df["axis_level"].astype(str)
    return df


def load_source_runs() -> dict:
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_replicate_seeds() -> dict:
    with open(PROCESSED_DIR / "replicate_seeds.json", "r", encoding="utf-8") as f:
        return json.load(f)


def extract_y_value(manifest_params: dict, method: str) -> float:
    """Pull the method-specific "smoothing" analog straight out of a run's
    own manifest.json params block. Raises KeyError (loudly, not silently)
    if the expected field is missing -- see module docstring for exactly
    which field each method uses and why."""
    if method == "rbf_bootstrap":
        return float(manifest_params["best_smoothing"])
    if method == "gp_mle":
        return float(manifest_params["fitted_hyperparameters"]["noise_variance_real_units"])
    raise ValueError(f"unsupported method: {method}")


def build_dataset(method: str, length_df: pd.DataFrame, source_runs: dict) -> pd.DataFrame:
    """Assemble one method's (axis_level, replicate) -> (length_scale_m,
    y_value) table by reading each run's OWN manifest.json fresh (y_value)
    and cross-referencing length_scale_by_replicate.csv (x_value, NOT
    recomputed). Raises if any of the expected 3 levels x 10 replicates = 30
    cells is missing, duplicated, or mismatched."""
    rows = []
    for level in ALL_AXIS_LEVELS:
        for rep in REPLICATE_IDS:
            run_dir = _REPO_ROOT / source_runs[level][rep][method]
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"{method} axis_level={level} {rep}: manifest not found at {manifest_path}"
                )
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            y_value = extract_y_value(manifest["params"], method)

            length_row = length_df[
                (length_df["axis_level"] == level)
                & (length_df["replicate"] == rep)
                & (length_df["method"] == method)
            ]
            if len(length_row) != 1:
                raise ValueError(
                    f"expected exactly 1 length_scale_by_replicate.csv row for "
                    f"(axis_level={level}, replicate={rep}, method={method}); "
                    f"found {len(length_row)}."
                )
            length_scale_m = float(length_row["length_scale_m"].iloc[0])

            rows.append(
                {
                    "axis_level": level,
                    "replicate": rep,
                    "method": method,
                    "length_scale_m": length_scale_m,
                    "y_value": y_value,
                    "manifest_path": str(manifest_path.relative_to(_REPO_ROOT)),
                }
            )

    df = pd.DataFrame(rows)
    expected_total = len(ALL_AXIS_LEVELS) * len(REPLICATE_IDS)
    if len(df) != expected_total:
        raise ValueError(
            f"{method}: expected {expected_total} rows (({len(ALL_AXIS_LEVELS)} levels) x "
            f"({len(REPLICATE_IDS)} replicates)), found {len(df)}."
        )
    for level in ALL_AXIS_LEVELS:
        n_level = len(df[df["axis_level"] == level])
        if n_level != len(REPLICATE_IDS):
            raise ValueError(
                f"{method} axis_level={level}: expected {len(REPLICATE_IDS)} rows, "
                f"found {n_level}."
            )

    print(
        f"{method}: verified {len(df)} manifest reads "
        f"({len(ALL_AXIS_LEVELS)} levels x {len(REPLICATE_IDS)} replicates)."
    )
    for level in ALL_AXIS_LEVELS:
        sub = df[df["axis_level"] == level]
        print(
            f"  axis_level={level}: length_scale_m in "
            f"[{sub['length_scale_m'].min():.2f}, {sub['length_scale_m'].max():.2f}] m "
            f"(ratio {sub['length_scale_m'].max() / sub['length_scale_m'].min():.2f}x); "
            f"y_value in [{sub['y_value'].min():.6g}, {sub['y_value'].max():.6g}]"
        )
    print(
        f"  overall: length_scale_m in "
        f"[{df['length_scale_m'].min():.2f}, {df['length_scale_m'].max():.2f}] m; "
        f"y_value in [{df['y_value'].min():.6g}, {df['y_value'].max():.6g}]"
    )
    return df


def _level_label(level: str, replicate_seeds: dict) -> str:
    """'5% level (n_requested=125, n_actual=118-124)' -- requested % (exact,
    by construction of N_SAMPLES_REQUESTED_BY_LEVEL against the 2500-cell
    grid) plus the ACTUAL post-dedup sample-count range across the 10
    replicates at this level, following the SAME requested-vs-actual
    reporting convention make_sample_replicate_figures.py's own caption
    already uses for this axis (actual count varies per replicate because
    random_interior_samples drops duplicate-cell draws)."""
    n_req = N_SAMPLES_REQUESTED_BY_LEVEL[level]
    pct_req = 100.0 * n_req / GRID_CELLS
    actual = replicate_seeds["n_samples_actual"][level]
    n_actual_vals = list(actual.values())
    return (
        f"{pct_req:.0f}% level (n_requested={n_req}, "
        f"n_actual={min(n_actual_vals)}-{max(n_actual_vals)})"
    )


def make_figure(method: str, df: pd.DataFrame, replicate_seeds: dict) -> Path:
    color = METHOD_COLORS[method]
    marker = METHOD_MARKERS[method]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
    for ax, level in zip(axes, ALL_AXIS_LEVELS):
        sub = df[df["axis_level"] == level].sort_values("replicate")
        ax.scatter(
            sub["length_scale_m"], sub["y_value"], color=color, marker=marker,
            s=70, edgecolors="black", linewidths=0.7, alpha=0.85, zorder=3,
        )
        # Points that land on (or within a tight tolerance of) the same
        # (x, y) coordinate are REAL ties, not a plotting artifact:
        # best_smoothing and the length-scale conversion both come from
        # small discrete CV grids, so multiple replicates can legitimately
        # land on the exact same point (e.g. rep0/rep1/rep8 all at
        # (219.92 m, 0.1) in the RBF+bootstrap 5% panel). Annotating each
        # such point separately at the same fixed pixel offset stacks their
        # replicate-id labels on top of each other into an unreadable smear,
        # so points within TIE_LOG_TOL (~0.23% multiplicative, tight enough
        # to catch only true/near-exact ties, not merge genuinely distinct
        # nearby points) on BOTH axes (in log space, matching the log-log
        # panels) are combined into ONE label listing all their replicate
        # indices (e.g. "0,1,8") instead of one overlapping label per point.
        TIE_LOG_TOL = 3  # decimal places of log10(value) to round to
        sub = sub.copy()
        sub["_tie_key"] = list(
            zip(
                np.log10(sub["length_scale_m"]).round(TIE_LOG_TOL),
                np.log10(sub["y_value"]).round(TIE_LOG_TOL),
            )
        )
        for _, grp in sub.groupby("_tie_key", sort=False):
            reps_label = ",".join(
                sorted(
                    (r.replace("rep", "") for r in grp["replicate"]),
                    key=int,
                )
            )
            ax.annotate(
                reps_label,
                (grp["length_scale_m"].iloc[0], grp["y_value"].iloc[0]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=6.5, color="dimgray", alpha=0.9,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(X_LABEL, fontsize=8.5)
        ax.set_ylabel(Y_FIELD_LABELS[method], fontsize=8.5)
        ax.set_title(_level_label(level, replicate_seeds), fontsize=10.5)
        ax.grid(alpha=0.3, which="both")

    plt.suptitle(
        f"{METHOD_LABELS[method]}: fitted length scale vs. "
        f"{'smoothing parameter' if method == 'rbf_bootstrap' else 'noise-variance (smoothing-equivalent) parameter'} "
        "across 10 sample-location replicates, 3 sample-density levels",
        fontsize=13, y=1.01,
    )

    if method == "rbf_bootstrap":
        y_field_caption = (
            "y = params.best_smoothing, the CV-selected RBFInterpolator smoothing/"
            "regularization parameter (chosen from src/experiments/rbf_bootstrap.py's "
            "SMOOTHING_GRID), read fresh from each run's own manifest.json."
        )
    else:
        y_field_caption = (
            "y = params.fitted_hyperparameters.noise_variance_real_units, the MLE-fitted GP "
            "nugget/noise variance in real (Porosity %^2) units -- labelled explicitly as "
            "GP's smoothing-EQUIVALENT regularization term because this codebase has no "
            "hyperparameter literally named 'smoothing' for GP-MLE; see module docstring for "
            "the analogy this y-axis represents (orchestrator decision, 2026-09-18)."
        )

    caption = (
        f"Each panel: all 10 sample-seed replicates (rep0-rep9, sample_seed 1001-1010, "
        f"labelled by replicate index next to each point) at ONE sample-density-axis level, "
        f"SAME ground-truth field for every replicate/level (TRUTH_SEED={TRUTH_SEED}, "
        f"range={AXIS_HMAJ1:g} m). Panels ordered dense -> sparse left to right (5% -> 2% -> "
        "1% requested; actual post-dedup counts noted per panel). x = length_scale_m taken "
        "AS-IS from length_scale_by_replicate.csv (NOT recomputed here -- see that file's own "
        "per-row length_definition/source columns). " + y_field_caption + " Both axes are "
        "log-scaled: y spans >=3 orders of magnitude across the 30 (level, replicate) cells "
        "plotted here (see this script's stdout for the exact observed ranges), and x (length "
        "scale) is a multiplicative quantity that already spans up to ~11x within a single "
        "panel at the sparsest level -- see module docstring for the full rationale."
    )

    fig.text(
        0.5, 0.005, textwrap.fill(caption, 195),
        ha="center", va="bottom", fontsize=7.3, color="dimgray",
    )
    plt.subplots_adjust(left=0.055, bottom=0.22, right=0.99, top=0.82, wspace=0.32)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_name = (
        "length_vs_noise_gp_mle.png" if method == "gp_mle" else "length_vs_smoothing_rbf_bootstrap.png"
    )
    out = FIGURES_DIR / out_name
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"{out.name}: {out} ({out.stat().st_size} bytes)")
    return out


def main():
    length_df = load_length_df()
    source_runs = load_source_runs()
    replicate_seeds = load_replicate_seeds()

    saved = []
    for method in ["gp_mle", "rbf_bootstrap"]:
        print(f"\nBuilding length-vs-smoothing dataset for {method}...")
        df = build_dataset(method, length_df, source_runs)
        saved.append(make_figure(method, df, replicate_seeds))

    print("\nAll length-vs-smoothing figures:")
    for f in saved:
        print(" ", f)
    return saved


if __name__ == "__main__":
    main()
