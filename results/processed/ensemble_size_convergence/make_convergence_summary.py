"""E1 ensemble-size convergence: tidy table, N-selection table and figure.

Input (read only): results/raw/ensemble_size_convergence/<ts>/nested_metrics.csv
and timings.csv, written by src/experiments/ensemble_size_convergence.py. The
raw run directory is pinned in SOURCE_RAW_RUN below (pass --raw-run-dir to
override; the pin used is written into every output's header comment).

Outputs
-------
results/processed/ensemble_size_convergence/
  convergence.csv     tidy long: case, method, seed, n_ensemble, metric, value
                      (seed = that method's own seed: sgsim seed for sgs,
                      bootstrap RandomState seed for rbf_bootstrap)
  deltas_by_n.csv     per (case, method, seed, n_ensemble): value(N) and
                      value(N_MAX) for every metric, plus
                        abs_delta_umg            = |UMG(N) - UMG(N_MAX)|
                        rel_delta_variance_mean  = |vm(N) - vm(N_MAX)| / vm(N_MAX)
                        abs_delta_coverage90     = |cov90(N) - cov90(N_MAX)|
                        rel_delta_mse            = |MSE(N) - MSE(N_MAX)| / MSE(N_MAX)
  n_selection.csv     user-approved rule, per (case, method, seed):
                        n_min = smallest N in the grid with
                                |UMG(N) - UMG(N_MAX)| <= UMG_TOL (0.005)
                      plus, as reference values at n_min: rel_delta_variance_mean,
                      abs_delta_coverage90; and the FACT column
                      ``all_larger_n_also_satisfy`` (whether every grid N >= n_min
                      also satisfies the tolerance). N_MAX satisfies trivially
                      (delta 0), so n_min always exists.
  n_selection_overall.csv
                        max_of_cell_n_min      = max over all (case, method, seed)
                                                 of n_min
                        smallest_n_all_cells   = smallest grid N at which EVERY
                                                 (case, method, seed) satisfies
                                                 the tolerance
                      (the two coincide unless some cell is non-monotone)
  timings.csv         copy of the raw timings (N_MAX runs)

results/figures/ensemble_size_convergence/
  ensemble_size_convergence.pdf / .png : 4 rows (UMG, variance_mean,
      coverage90, MSE) x 4 columns (case x method); one line per seed; x = N
      (log scale, ticks at the N grid). 190 mm wide, Arial 7 pt
      (docs/figure_guidelines.md); method colors identical to the paper
      storyboard (sgs tab:green, rbf_bootstrap tab:orange).

Run with:
.venv/Scripts/python.exe -m results.processed.ensemble_size_convergence.make_convergence_summary
"""

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "ensemble_size_convergence"
FIGURE_DIR = _REPO_ROOT / "results" / "figures" / "ensemble_size_convergence"

# Pinned raw run (set after the production run; None -> latest directory).
SOURCE_RAW_RUN = "results/raw/ensemble_size_convergence/20261116T202735769759Z"

UMG_TOL = 0.005  # user-approved selection rule
UNLABELED_TICKS = {150}
METRICS = ["umg", "variance_mean", "coverage90", "mse"]
CASES = ["base_case", "density_1pct_rep0"]
METHODS = ["sgs", "rbf_bootstrap"]

# --- styling: identical to results/processed/paper_storyboard ----------------
METHOD_COLORS = {"sgs": "tab:green", "rbf_bootstrap": "tab:orange"}
METHOD_LABELS = {"sgs": "SGS", "rbf_bootstrap": "RBF+bootstrap"}
CASE_LABELS = {"base_case": "Base case (n = 121)", "density_1pct_rep0": "1% density, rep0 (n = 25)"}
METRIC_LABELS = {
    "umg": "UMG (1 = calibrated)",
    "variance_mean": "Mean predictive variance\n(porosity %²)",
    "coverage90": "90% interval coverage",
    "mse": "MSE (porosity %²)",
}
SEED_STYLES = [("-", "o"), ("--", "s"), (":", "^")]
MM = 1.0 / 25.4
FULL_WIDTH_MM = 190.0
HEIGHT_MM = 150.0
FONT_PT = 7
PANEL_LABEL_PT = 9
PNG_DPI = 600


def _setup_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3, "ytick.minor.width": 0.3,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "grid.linewidth": 0.25, "grid.color": "0.8",
        "legend.frameon": False, "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(_REPO_ROOT).as_posix()


def _resolve_raw_run(arg):
    if arg:
        return Path(arg) if Path(arg).is_absolute() else _REPO_ROOT / arg
    if SOURCE_RAW_RUN:
        return _REPO_ROOT / SOURCE_RAW_RUN
    runs = sorted(p for p in (_REPO_ROOT / "results" / "raw" / "ensemble_size_convergence").iterdir() if p.is_dir())
    return runs[-1]


def _write_with_header(df: pd.DataFrame, path: Path, header_lines):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)


def build_tables(raw: pd.DataFrame):
    n_grid = sorted(raw["n_ensemble"].unique())
    n_max = n_grid[-1]
    tidy = raw[["case", "method", "seed", "n_ensemble", "metric", "value"]].copy()

    wide = raw.pivot_table(index=["case", "method", "seed", "n_ensemble"], columns="metric",
                           values="value", aggfunc="first").reset_index()
    ref = wide[wide["n_ensemble"] == n_max].set_index(["case", "method", "seed"])
    d = wide.copy()
    for m in METRICS:
        d[f"{m}_at_nmax"] = [ref.loc[(c, me, s), m] for c, me, s in zip(d["case"], d["method"], d["seed"])]
    d["abs_delta_umg"] = (d["umg"] - d["umg_at_nmax"]).abs()
    d["rel_delta_variance_mean"] = (d["variance_mean"] - d["variance_mean_at_nmax"]).abs() / d["variance_mean_at_nmax"]
    d["abs_delta_coverage90"] = (d["coverage90"] - d["coverage90_at_nmax"]).abs()
    d["rel_delta_mse"] = (d["mse"] - d["mse_at_nmax"]).abs() / d["mse_at_nmax"]
    d["umg_within_tol"] = d["abs_delta_umg"] <= UMG_TOL
    d = d.sort_values(["case", "method", "seed", "n_ensemble"]).reset_index(drop=True)

    sel_rows = []
    for (case, method, seed), g in d.groupby(["case", "method", "seed"], sort=False):
        g = g.sort_values("n_ensemble")
        ok = g[g["umg_within_tol"]]
        n_min = int(ok["n_ensemble"].iloc[0])
        r = g[g["n_ensemble"] == n_min].iloc[0]
        sel_rows.append({
            "case": case, "method": method, "seed": int(seed), "n_min": n_min,
            "umg_at_n_min": r["umg"], "umg_at_nmax": r["umg_at_nmax"],
            "abs_delta_umg_at_n_min": r["abs_delta_umg"],
            "all_larger_n_also_satisfy": bool(g[g["n_ensemble"] >= n_min]["umg_within_tol"].all()),
            "rel_delta_variance_mean_at_n_min": r["rel_delta_variance_mean"],
            "abs_delta_coverage90_at_n_min": r["abs_delta_coverage90"],
            "abs_delta_umg_at_smallest_grid_n": float(g[g["n_ensemble"] == n_grid[0]]["abs_delta_umg"].iloc[0]),
        })
    sel = pd.DataFrame(sel_rows)

    all_ok = d.groupby("n_ensemble")["umg_within_tol"].all()
    smallest_all = int(min(n for n, v in all_ok.items() if v))
    overall = pd.DataFrame([{
        "umg_tol": UMG_TOL, "n_max": int(n_max), "n_grid": " ".join(str(int(n)) for n in n_grid),
        "n_cells": len(sel),
        "max_of_cell_n_min": int(sel["n_min"].max()),
        "smallest_n_all_cells": smallest_all,
    }])
    return tidy, d, sel, overall, n_grid


def make_figure(d: pd.DataFrame, n_grid, out_stem: Path):
    _setup_style()
    fig, axes = plt.subplots(len(METRICS), len(CASES) * len(METHODS),
                             figsize=(FULL_WIDTH_MM * MM, HEIGHT_MM * MM), sharex=True)
    cols = [(c, m) for c in CASES for m in METHODS]
    letters = iter("ABCDEFGHIJKLMNOP")
    for j, (case, method) in enumerate(cols):
        sub = d[(d["case"] == case) & (d["method"] == method)]
        seeds = sorted(sub["seed"].unique())
        for i, metric in enumerate(METRICS):
            ax = axes[i, j]
            for k, seed in enumerate(seeds):
                s = sub[sub["seed"] == seed].sort_values("n_ensemble")
                ls, mk = SEED_STYLES[k]
                ax.plot(s["n_ensemble"], s[metric], color=METHOD_COLORS[method], linestyle=ls, marker=mk,
                        markersize=3.0, linewidth=0.9, markeredgewidth=0.4)
            if metric == "umg":
                ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, zorder=0)
            if metric == "coverage90":
                ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.5, zorder=0)
            ax.set_xscale("log")
            ax.set_xticks(n_grid)
            # all grid N get a tick; 150 is left unlabeled (overlaps 100/200 on log axis at 190 mm)
            ax.set_xticklabels(["" if int(n) in UNLABELED_TICKS else str(int(n)) for n in n_grid])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.grid(True, axis="y")
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            if i == 0:
                ax.set_title(f"{CASE_LABELS[case]}\n{METHOD_LABELS[method]}")
            if i == len(METRICS) - 1:
                ax.set_xlabel("Ensemble size N")
            ax.text(-0.02, 1.02, next(letters), transform=ax.transAxes, fontsize=PANEL_LABEL_PT,
                    fontweight="bold", ha="right", va="bottom")
            ax.tick_params(axis="x", labelrotation=0)
    handles = [Line2D([0], [0], color="0.3", linestyle=ls, marker=mk, markersize=3.0, linewidth=0.9,
                      label=f"Seed {k + 1}") for k, (ls, mk) in enumerate(SEED_STYLES)]
    handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=0.5,
                          label="Nominal (UMG = 1; coverage = 0.90)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.10, wspace=0.42, hspace=0.18)
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=PNG_DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-run-dir", default=None)
    args = parser.parse_args()
    raw_dir = _resolve_raw_run(args.raw_run_dir)
    raw = pd.read_csv(raw_dir / "nested_metrics.csv")
    if set(raw["case"]) != set(CASES) or set(raw["method"]) != set(METHODS) or set(raw["metric"]) != set(METRICS):
        raise ValueError("nested_metrics.csv cases/methods/metrics differ from expected.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    tidy, deltas, sel, overall, n_grid = build_tables(raw)
    src = f"source: {_rel(raw_dir)}/nested_metrics.csv (written by src/experiments/ensemble_size_convergence.py)"
    _write_with_header(tidy, PROCESSED_DIR / "convergence.csv", [
        src,
        "seed = method's own seed (sgsim seed for sgs, bootstrap seed for rbf_bootstrap); "
        "n_ensemble = first-N nested subset of one N_MAX run",
        "rbf_bootstrap mse uses point_estimate_map (N-independent), as in evaluate_one_cell",
    ])
    _write_with_header(deltas, PROCESSED_DIR / "deltas_by_n.csv", [src, f"reference = N_MAX = {int(n_grid[-1])}"])
    _write_with_header(sel, PROCESSED_DIR / "n_selection.csv", [
        src,
        f"n_min = smallest grid N with |UMG(N) - UMG({int(n_grid[-1])})| <= {UMG_TOL} (user-approved rule)",
        "rel_delta_variance_mean / abs_delta_coverage90 at n_min are reference values only",
    ])
    _write_with_header(overall, PROCESSED_DIR / "n_selection_overall.csv", [src])
    shutil.copyfile(raw_dir / "timings.csv", PROCESSED_DIR / "timings.csv")
    make_figure(deltas, n_grid, FIGURE_DIR / "ensemble_size_convergence")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"raw run: {_rel(raw_dir)}")
    print(sel.to_string(index=False))
    print(overall.to_string(index=False))
    print(deltas[["case", "method", "seed", "n_ensemble", "umg", "abs_delta_umg", "rel_delta_variance_mean",
                  "abs_delta_coverage90", "rel_delta_mse"]].to_string(index=False))


if __name__ == "__main__":
    main()
