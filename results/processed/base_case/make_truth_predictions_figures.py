"""Build the "Truth & predictions" figures for the base-case 4-method
comparison (kriging / SGS / RBF+bootstrap / GP-MLE), replacing the earlier
per-method 3-panel [truth, mean, variance] layout with a leaner set of
figures: one truth-only reference figure, plus per-method figures showing
only [example realization, ensemble mean, ensemble variance] (2 panels for
kriging, which has no realization concept -- see task description,
2026-09-14 design decision).

This script is INDEPENDENT of results/processed/base_case/source_runs.json
and metrics.csv -- it does NOT read or write either, and must not be
combined with make_crossplot_comparison.py's pipeline (that pipeline is
pinned to specific historical runs for MSE consistency and is out of scope
here). Instead, the four input run directories are hardcoded below:

- kriging:       a NEW run (produced by this task) with the physical-units
                  Monte Carlo variance approximation
                  (kriging_var_map_physical_mc.npy) -- kriging has no
                  ensemble/"realization" concept (deterministic estimator),
                  so its figure has 2 panels only (mean, variance).
- gp_mle:        a NEW run (produced by this task) with a single posterior
                  realization draw (posterior_sample_map.npy).
- sgs:            the existing pinned run results/raw/sgs/20260914T141516Z
                  (unchanged by this task).
- rbf_bootstrap:  results/raw/rbf_bootstrap/20260917T230913430600Z -- the
                  post-EPSILON_GRID-redesign base-case run (2026-09-17);
                  re-pointed from the superseded 20260914T135501Z run.

Color bar unification (project decision 2026-09-14, see task description):
- All "mean" and "realization" panels (physical porosity units) share ONE
  common scale: vmin=truth.min(), vmax=truth.max(), cmap="viridis" --
  identical to truth_field.png's own scale.
- All "variance" panels (physical porosity^2 units: kriging's is a Monte
  Carlo approximation, see kriging.py's kriging_var_map_physical_mc_note)
  share ONE common scale: vmin=0, vmax=max over SGS/RBF+bootstrap/GP-MLE's
  three variance arrays ONLY (2026-09-14 revised decision -- kriging's own
  variance has a normal-score tail-extrapolation artifact that pushes its
  max an order of magnitude above the other three methods; including it in
  the shared vmax washed out all three other variance panels to near-black,
  defeating the point of a shared scale. kriging's variance panel uses this
  same shared scale and lets its extreme cells saturate/clip at the top
  color -- its true (unclipped) max is printed at runtime and annotated
  directly on its panel so the clipping is disclosed, not hidden), cmap=
  "magma". UPDATE (2026-09-14, later same day): after BACKTR_ZMIN/BACKTR_ZMAX
  in src/experiments/kriging.py was reverted from +/-10 to +/-4 stdev, this
  particular run's kriging max (~9.8) is actually now BELOW the shared
  SGS/RBF/GP-MLE-anchored scale (~18), i.e. no clipping currently occurs --
  but the exclusion-from-vmax-computation logic below is left in place
  unchanged (still architecturally robust to a future run/case where
  kriging's max is large again), and the panel annotation is written to
  state whichever is actually true at runtime rather than assuming clipping.
Each panel gets its own colorbar (since figures are viewed independently in
the report), but the vmin/vmax passed to every imshow() call of a given
panel type are the SAME global values computed once below -- never a
panel's own local min/max.

Run with: .venv/Scripts/python.exe -m results.processed.base_case.make_truth_predictions_figures
(or: .venv/Scripts/python.exe results/processed/base_case/make_truth_predictions_figures.py)
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.base_case import NX, NY, XMN, XMAX, XMIN, YMN, YMAX, YMIN, XSIZ, YSIZ  # noqa: E402
from src.experiments.base_case_conditioning import (  # noqa: E402
    SAMPLE_SEED,
    get_base_case_truth,
    get_conditioning_samples,
)

FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "base_case"

# --- Hardcoded source run directories -------------------------------------
# kriging / gp_mle: NEW runs produced by this task (see final report for how
# these timestamps were chosen -- they are the runs that produced
# kriging_var_map_physical_mc.npy / posterior_sample_map.npy respectively).
# kriging: re-pointed 2026-09-14 (second update same day) to the run
# produced after BACKTR_ZMIN/BACKTR_ZMAX was reverted from +/-10 stdev back
# to +/-4 stdev in src/experiments/kriging.py (see that file's comment for
# the full history/tradeoff) -- the prior run (20260914T180655Z) used the
# now-superseded +/-10 stdev bounds and is stale for this figure.
KRIGING_RUN_DIR = _REPO_ROOT / "results" / "raw" / "kriging" / "20260914T184654Z"
GP_MLE_RUN_DIR = _REPO_ROOT / "results" / "raw" / "gp_mle" / "20260914T180412Z"
# sgs: existing pinned run, unchanged.
SGS_RUN_DIR = _REPO_ROOT / "results" / "raw" / "sgs" / "20260914T141516Z"
# rbf_bootstrap: re-pointed 2026-09-17 to the run produced after
# EPSILON_GRID in src/experiments/rbf_bootstrap.py was redesigned (the old
# 9-point grid had no point between 173.1 m and 576.9 m and pinned every
# density level to 173.1 m -- see that file's comment for the measured
# evidence). The prior run (20260914T135501Z) used the superseded grid and
# is stale for this figure; leaving it would make this figure disagree with
# results/processed/base_case/metrics.csv. kriging / sgs / gp_mle are
# untouched (their code did not change).
RBF_BOOTSTRAP_RUN_DIR = _REPO_ROOT / "results" / "raw" / "rbf_bootstrap" / "20260917T230913430600Z"

EXTENT = [XMIN, XMAX, YMIN, YMAX]


def _scatter_samples(ax, samples_df, color="red"):
    ax.scatter(samples_df["X"], samples_df["Y"], s=10, c=color, marker="+", label="conditioning samples")


def _panel(ax, arr, title, vmin, vmax, cmap, cbar_label, samples_df, scatter_color="red"):
    im = ax.imshow(arr, extent=EXTENT, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    _scatter_samples(ax, samples_df, color=scatter_color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, label=cbar_label)
    return im


def main():
    truth = get_base_case_truth()
    samples_df = get_conditioning_samples(truth, sample_seed=SAMPLE_SEED)

    # --- Load per-method arrays -------------------------------------------
    kriging_mean = np.load(KRIGING_RUN_DIR / "kriging_mean_map_physical.npy")
    kriging_var = np.load(KRIGING_RUN_DIR / "kriging_var_map_physical_mc.npy")

    sgs_realizations = np.load(SGS_RUN_DIR / "sgs_realizations.npy")
    sgs_example = sgs_realizations[0]
    sgs_mean = np.load(SGS_RUN_DIR / "sgs_mean_map.npy")
    sgs_var = np.load(SGS_RUN_DIR / "sgs_var_map.npy")
    sgs_n_realizations = sgs_realizations.shape[0]

    bootstrap_replicates = np.load(RBF_BOOTSTRAP_RUN_DIR / "bootstrap_replicate_maps.npy")
    bootstrap_example = bootstrap_replicates[0]
    bootstrap_mean = np.load(RBF_BOOTSTRAP_RUN_DIR / "bootstrap_mean_map.npy")
    bootstrap_var = np.load(RBF_BOOTSTRAP_RUN_DIR / "bootstrap_var_map.npy")
    bootstrap_n_replicates = bootstrap_replicates.shape[0]

    gp_sample = np.load(GP_MLE_RUN_DIR / "posterior_sample_map.npy")
    gp_mean = np.load(GP_MLE_RUN_DIR / "posterior_mean_map.npy")
    gp_var = np.load(GP_MLE_RUN_DIR / "posterior_var_map.npy")

    # --- Global color scales (project decision 2026-09-14) -----------------
    mean_vmin = float(np.min(truth))
    mean_vmax = float(np.max(truth))

    # Shared variance scale is anchored to SGS/RBF+bootstrap/GP-MLE only --
    # kriging's Monte Carlo physical-unit variance is excluded from this max
    # (see module docstring): its own panel still uses this scale, so its
    # tail cells clip/saturate at the top color rather than dominating it.
    kriging_var_actual_max = float(np.max(kriging_var))
    var_vmax = float(max(np.max(sgs_var), np.max(bootstrap_var), np.max(gp_var)))
    var_vmin = 0.0

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. truth_field.png -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    _panel(
        ax, truth, f"Base-case truth (single reference field, {truth.shape[0]}x{truth.shape[1]})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.98, top=0.92)
    out = FIGURES_DIR / "truth_field.png"
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"truth_field.png: {out} ({out.stat().st_size} bytes)")

    # --- 2. qc_truth_predictions_kriging.png (2 panels: mean, variance) -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _panel(
        axes[0], kriging_mean, "Simple kriging -- point estimate (mean)",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    ax_kvar = _panel(
        axes[1], kriging_var, "Simple kriging -- variance (physical units, Monte Carlo approx.)",
        var_vmin, var_vmax, "magma", "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    ).axes
    # Whether kriging's true max actually exceeds (and is therefore clipped
    # by) the shared SGS/RBF/GP-MLE-anchored scale depends on
    # BACKTR_ZMIN/BACKTR_ZMAX in src/experiments/kriging.py at the time
    # KRIGING_RUN_DIR was produced -- with the +/-10 stdev bounds this used
    # to be true (true max ~61, clipped at the shared scale's ~18); after the
    # 2026-09-14 revert to +/-4 stdev it is generally no longer true (true
    # max ~9.8, below the shared scale), so this annotation is written to
    # state whichever is actually the case rather than assuming clipping.
    if kriging_var_actual_max > var_vmax:
        kvar_annotation = (
            f"Color scale clipped at {var_vmax:.2f} (shared with SGS/RBF/GP-MLE); "
            f"kriging's true max is {kriging_var_actual_max:.2f}, in the tail-extrapolated cell(s)."
        )
    else:
        kvar_annotation = (
            f"Not clipped: kriging's true max ({kriging_var_actual_max:.2f}) is below the "
            f"shared SGS/RBF/GP-MLE-anchored scale max ({var_vmax:.2f})."
        )
    ax_kvar.text(
        0.02, -0.14,
        kvar_annotation,
        transform=ax_kvar.transAxes, fontsize=8, color="dimgray",
    )
    plt.subplots_adjust(left=0.06, bottom=0.16, right=0.98, top=0.9, wspace=0.35)
    out = FIGURES_DIR / "qc_truth_predictions_kriging.png"
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"qc_truth_predictions_kriging.png: {out} ({out.stat().st_size} bytes)")

    # --- 3. qc_truth_predictions_sgs.png (3 panels) -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(
        axes[0], sgs_example, f"SGS -- example realization (#0 of {sgs_n_realizations})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[1], sgs_mean, "SGS -- ensemble mean",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[2], sgs_var, "SGS -- ensemble variance",
        var_vmin, var_vmax, "magma", "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.9, wspace=0.35)
    out = FIGURES_DIR / "qc_truth_predictions_sgs.png"
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"qc_truth_predictions_sgs.png: {out} ({out.stat().st_size} bytes)")

    # --- 4. qc_truth_predictions_rbf_bootstrap.png (3 panels) ---------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(
        axes[0], bootstrap_example, f"RBF+bootstrap -- example replicate (#0 of {bootstrap_n_replicates})",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[1], bootstrap_mean, "RBF+bootstrap -- ensemble mean",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[2], bootstrap_var, "RBF+bootstrap -- ensemble variance",
        var_vmin, var_vmax, "magma", "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.9, wspace=0.35)
    out = FIGURES_DIR / "qc_truth_predictions_rbf_bootstrap.png"
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"qc_truth_predictions_rbf_bootstrap.png: {out} ({out.stat().st_size} bytes)")

    # --- 5. qc_truth_predictions_gp_mle.png (3 panels) ----------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(
        axes[0], gp_sample, "GP-MLE -- posterior sample (1 draw)",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[1], gp_mean, "GP-MLE -- posterior mean",
        mean_vmin, mean_vmax, "viridis", "Porosity (%)", samples_df,
    )
    _panel(
        axes[2], gp_var, "GP-MLE -- posterior variance",
        var_vmin, var_vmax, "magma", "Variance (Porosity %^2)", samples_df, scatter_color="cyan",
    )
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.9, wspace=0.35)
    out = FIGURES_DIR / "qc_truth_predictions_gp_mle.png"
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"qc_truth_predictions_gp_mle.png: {out} ({out.stat().st_size} bytes)")

    print(f"\nGlobal mean/realization scale: vmin={mean_vmin:.4f}, vmax={mean_vmax:.4f}")
    print(f"Global variance scale (SGS/RBF/GP-MLE-anchored): vmin={var_vmin:.4f}, vmax={var_vmax:.4f}")
    print(f"Kriging variance actual max (clipped in panel): {kriging_var_actual_max:.4f}")


if __name__ == "__main__":
    main()
