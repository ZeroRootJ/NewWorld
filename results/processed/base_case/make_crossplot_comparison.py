"""Build the 4-panel truth-vs-predicted crossplot comparison figure for the
base-case 4-method comparison (kriging / SGS / RBF+bootstrap / GP-MLE).

Reads exclusively from the raw run directories recorded in
results/processed/base_case/source_runs.json (NOT "latest run" -- unlike
evaluate_base_case.py, this script is a data-curator artifact written after
that source_runs.json snapshot was taken, so it must pin to the exact
recorded runs rather than re-resolving "latest", to stay consistent with
metrics.csv/accuracy_plot_curves.csv even if a newer raw run is added later).

Per-method arrays used (point estimate / mean, physical porosity units;
must match the arrays evaluate_base_case.py uses for MSE, for consistency
with the MSE annotated in each subplot title):
- kriging:       kriging_mean_map_physical.npy
- sgs:           sgs_mean_map.npy
- rbf_bootstrap: point_estimate_map.npy   (single-fit estimate, NOT
                                            bootstrap_mean_map.npy)
- gp_mle:        posterior_mean_map.npy

The same conditioning_cell_mask (src/evaluation.py) used by
evaluate_base_case.py is re-derived here (regenerated truth + conditioning
samples, not read from any method's samples.csv) so all 4 panels exclude
exactly the same 121 conditioning cells.

Output: results/figures/base_case/crossplot_comparison.png

Run with: .venv/Scripts/python.exe -m results.processed.base_case.make_crossplot_comparison
(or: .venv/Scripts/python.exe results/processed/base_case/make_crossplot_comparison.py)
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation import conditioning_cell_mask, mse  # noqa: E402
from src.experiments.base_case import NX, NY, XMN, YMN, XSIZ, YSIZ  # noqa: E402
from src.experiments.base_case_conditioning import (  # noqa: E402
    SAMPLE_SEED,
    get_base_case_truth,
    get_conditioning_samples,
)

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "base_case"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "base_case"

METHODS = ["kriging", "sgs", "rbf_bootstrap", "gp_mle"]
METHOD_LABELS = {
    "kriging": "Simple kriging",
    "sgs": "SGS (ensemble mean)",
    "rbf_bootstrap": "RBF + bootstrap",
    "gp_mle": "GP-MLE",
}
POINT_ESTIMATE_FILE = {
    "kriging": "kriging_mean_map_physical.npy",
    "sgs": "sgs_mean_map.npy",
    "rbf_bootstrap": "point_estimate_map.npy",
    "gp_mle": "posterior_mean_map.npy",
}


def main():
    with open(PROCESSED_DIR / "source_runs.json", "r", encoding="utf-8") as f:
        source_runs = json.load(f)

    run_dirs = {m: _REPO_ROOT / source_runs[m]["run_dir"] for m in METHODS}
    print("Source runs (pinned via source_runs.json):")
    for m in METHODS:
        print(f"  {m}: {run_dirs[m]}")

    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")

    # --- Shared truth + conditioning-cell mask, regenerated (not read from
    # any method's samples.csv) -- identical to evaluate_base_case.py's
    # approach, so the mask used here is guaranteed the same one that
    # produced metrics.csv's MSE values. ------------------------------------
    truth = get_base_case_truth()
    samples_df = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED)
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    truth_masked = truth[mask]

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    axes = axes.ravel()

    vmin = float(np.min(truth))
    vmax = float(np.max(truth))
    pad = 0.05 * (vmax - vmin)
    lims = (vmin - pad, vmax + pad)

    for ax, method in zip(axes, METHODS):
        pred_map = np.load(run_dirs[method] / POINT_ESTIMATE_FILE[method])
        pred_masked = pred_map[mask]

        # Cross-check against metrics.csv's recorded MSE for this method --
        # should match to float precision since both are computed from the
        # same arrays/mask; a mismatch would indicate the mask or array
        # choice here has drifted from evaluate_base_case.py.
        recorded_mse = metrics_df.loc[
            (metrics_df["method"] == method) & (metrics_df["metric"] == "mse"), "value"
        ].iloc[0]
        computed_mse = mse(truth_masked, pred_masked)
        if not np.isclose(computed_mse, recorded_mse, rtol=1e-8, atol=1e-8):
            raise ValueError(
                f"{method}: crossplot-computed MSE ({computed_mse}) does not match "
                f"metrics.csv's recorded MSE ({recorded_mse}) -- mask or source array "
                "has drifted from evaluate_base_case.py; do not silently proceed."
            )

        ax.scatter(truth_masked, pred_masked, s=10, alpha=0.5, color="tab:blue", edgecolors="none")
        ax.plot(lims, lims, color="gray", linestyle="--", label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Truth (Porosity %)")
        ax.set_ylabel(f"{METHOD_LABELS[method]} estimate (Porosity %)")
        ax.set_title(f"{METHOD_LABELS[method]} (MSE={computed_mse:.3f})")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Base case: truth vs. point-estimate/mean crossplot "
        f"(conditioning cells excluded, n={truth_masked.size} of {truth.size})",
        fontsize=12,
    )
    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.98, top=0.92, wspace=0.3, hspace=0.35)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "crossplot_comparison.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"\ncrossplot_comparison.png: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
