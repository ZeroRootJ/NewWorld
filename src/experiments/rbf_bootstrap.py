"""RBF + bootstrap uncertainty quantification for the base-case ground
truth (docs/experiment_context.md section 4, first of the four methods to
be compared; kriging/SGS are implemented separately).

Pipeline
--------
1. Reuse the shared base-case conditioning data (truth field regenerated
   with seed=TRUTH_SEED, N_SAMPLES random interior samples drawn with
   SAMPLE_SEED) from ``src.experiments.base_case_conditioning`` -- this is
   the *only* place sample locations are generated, and TRUTH_SEED,
   N_SAMPLES, and SAMPLE_SEED are all imported from there (not re-declared
   here), so gp_mle.py (which must use identical sample locations, per
   CLAUDE.md) is structurally guaranteed to match rather than relying on two
   scripts independently declaring the same literal.
2. Tune the RBF shape parameter (epsilon) and smoothing parameter (lambda)
   by k-fold cross-validated MSE grid search on the 500 samples (automatic
   procedure per docs/experiment_context.md section 4 -- "Tune these by the
   automatic procedure the paper is critiquing", not hand-tuned).
3. Fit once on all samples with the tuned hyperparameters -> point estimate
   map (the RBF's native prediction).
4. Bootstrap: resample the samples with replacement N_BOOTSTRAP times,
   refit with the *same* tuned hyperparameters each time, predict onto the
   full 50x50 grid -> N_BOOTSTRAP replicate maps.
5. Cell-wise mean and variance across the replicate maps -- this is RBF+
   bootstrap's uncertainty model, and is the artifact used later to test
   Claim 1 (that it under-estimates uncertainty relative to kriging/SGS).

Run with: .venv/Scripts/python.exe -m src.experiments.rbf_bootstrap
"""

import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold

from src.experiments.base_case import NX, NY, XMN, XMAX, XMIN, YMN, YMAX, YMIN, XSIZ, YSIZ
from src.experiments.base_case_conditioning import (
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    get_base_case_conditioning_data,
)
from src.grid_utils import full_grid_coordinates
from src.io import make_run_dir, save_result

# --- RBF kernel choice -------------------------------------------------
# 'gaussian' (exp(-r**2), with r scaled by epsilon) chosen over
# 'multiquadric' (-sqrt(1+r**2)): gaussian decays to zero with distance
# (bounded, well-behaved far from any sample -- e.g. near domain corners
# sparsely covered by the 500 interior samples), whereas multiquadric grows
# unboundedly with distance from data, which risks poorly-behaved
# bootstrap replicate maps at grid cells far from any given bootstrap
# resample's support. Both kernels have a shape parameter (epsilon), which
# the task requires.
RBF_KERNEL = "gaussian"

# --- CV hyperparameter grid search --------------------------------------
# 5-fold CV is a standard default (not hand-picked to favor any outcome);
# flagged here as a reasonable-default choice rather than a value the paper
# depends on.
CV_FOLDS = 5
CV_SEED = 40  # controls the KFold shuffle only

# Grid spans distances relevant to this domain: sample spacing for 500
# points over the ~900m x 900m interior sampling region is ~40m, and the
# variogram range used to generate the truth is 300m, so epsilon (acting
# as an inverse length scale) is swept from well below 1/900 to above 1/40.
EPSILON_GRID = np.array([0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0])
# Smoothing (lambda) swept from 0 (exact interpolation) across several
# orders of magnitude relative to the porosity variance (stdev=3 -> var=9).
SMOOTHING_GRID = np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])

# --- Bootstrap ------------------------------------------------------------
N_BOOTSTRAP = 10
BOOTSTRAP_SEED = 30

EXPERIMENT_NAME = "rbf_bootstrap"
CODE_ENTRYPOINT = "src/experiments/rbf_bootstrap.py"


def cv_mse_grid_search(X: np.ndarray, d: np.ndarray):
    """Grid search over (epsilon, smoothing) minimizing k-fold CV MSE.

    Returns (best_epsilon, best_smoothing, results_df) where results_df has
    one row per (epsilon, smoothing) combination with its mean CV MSE.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    rows = []
    for eps in EPSILON_GRID:
        for sm in SMOOTHING_GRID:
            fold_mses = []
            for train_idx, test_idx in kf.split(X):
                rbf = RBFInterpolator(
                    X[train_idx], d[train_idx], kernel=RBF_KERNEL, epsilon=eps, smoothing=sm
                )
                pred = rbf(X[test_idx])
                fold_mses.append(float(np.mean((pred - d[test_idx]) ** 2)))
            rows.append({"epsilon": float(eps), "smoothing": float(sm), "cv_mse": float(np.mean(fold_mses))})

    results_df = pd.DataFrame(rows)
    best_row = results_df.loc[results_df["cv_mse"].idxmin()]
    return float(best_row["epsilon"]), float(best_row["smoothing"]), results_df


def main():
    t_start = time.time()

    truth, samples_df = get_base_case_conditioning_data(sample_seed=SAMPLE_SEED)
    n_actual_samples = len(samples_df)

    X = samples_df[["X", "Y"]].values
    d = samples_df[VCOL].values

    t_cv_start = time.time()
    best_epsilon, best_smoothing, cv_results_df = cv_mse_grid_search(X, d)
    cv_seconds = time.time() - t_cv_start

    grid_coords = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)

    # --- Point estimate: single fit on all samples with tuned hyperparams --
    rbf_point = RBFInterpolator(X, d, kernel=RBF_KERNEL, epsilon=best_epsilon, smoothing=best_smoothing)
    point_estimate_map = rbf_point(grid_coords).reshape(NY, NX)

    # --- Bootstrap replicates ------------------------------------------
    # Note (Claim 1 context): bootstrap resampling here only varies *which*
    # of the 500 sample locations/values are used to fit each replicate --
    # it does not add any independent uncertainty about the field at
    # locations far from all samples, which is exactly the mechanism the
    # paper argues under-estimates true spatial uncertainty.
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    replicate_maps = np.empty((N_BOOTSTRAP, NY, NX))
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n_actual_samples, size=n_actual_samples, replace=True)
        rbf_b = RBFInterpolator(
            X[idx], d[idx], kernel=RBF_KERNEL, epsilon=best_epsilon, smoothing=best_smoothing
        )
        replicate_maps[b] = rbf_b(grid_coords).reshape(NY, NX)

    bootstrap_mean_map = replicate_maps.mean(axis=0)
    bootstrap_var_map = replicate_maps.var(axis=0, ddof=1)

    total_seconds = time.time() - t_start

    # --- Save outputs --------------------------------------------------
    run_dir = make_run_dir(EXPERIMENT_NAME)
    output_files = []

    samples_csv = "samples.csv"
    samples_df.to_csv(run_dir / samples_csv, index=False)
    output_files.append(samples_csv)

    cv_csv = "cv_results.csv"
    cv_results_df.to_csv(run_dir / cv_csv, index=False)
    output_files.append(cv_csv)

    for name, arr in [
        ("point_estimate_map.npy", point_estimate_map),
        ("bootstrap_replicate_maps.npy", replicate_maps),
        ("bootstrap_mean_map.npy", bootstrap_mean_map),
        ("bootstrap_var_map.npy", bootstrap_var_map),
    ]:
        np.save(run_dir / name, arr)
        output_files.append(name)

    # --- QC figure: truth / point estimate / bootstrap variance ---------
    vmin = float(np.min(truth))
    vmax = float(np.max(truth))
    fig = plt.figure(figsize=(18, 6))

    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(truth, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.scatter(samples_df["X"], samples_df["Y"], s=8, c="red", marker="+", label="samples")
    ax1.set_title(f"Truth (seed={TRUTH_SEED})")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    plt.colorbar(im1, ax=ax1, label="Porosity (%)")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(point_estimate_map, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.scatter(samples_df["X"], samples_df["Y"], s=8, c="red", marker="+")
    ax2.set_title(f"RBF point estimate (eps={best_epsilon:g}, sm={best_smoothing:g})")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    plt.colorbar(im2, ax=ax2, label="Porosity (%)")

    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(bootstrap_var_map, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="magma")
    ax3.scatter(samples_df["X"], samples_df["Y"], s=8, c="cyan", marker="+")
    ax3.set_title(f"Bootstrap variance ({N_BOOTSTRAP} replicates)")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    plt.colorbar(im3, ax=ax3, label="Variance (Porosity %^2)")

    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.9, wspace=0.35)
    qc_fig_name = "qc_rbf_bootstrap.png"
    plt.savefig(run_dir / qc_fig_name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    output_files.append(qc_fig_name)

    params = {
        "grid": {"nx": NX, "ny": NY, "xsiz": XSIZ, "ysiz": YSIZ, "xmn": XMN, "ymn": YMN},
        "truth_seed": TRUTH_SEED,
        "sample_seed": SAMPLE_SEED,
        "n_samples_requested": N_SAMPLES,
        "n_samples_actual": n_actual_samples,
        "rbf_kernel": RBF_KERNEL,
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "epsilon_grid": EPSILON_GRID.tolist(),
        "smoothing_grid": SMOOTHING_GRID.tolist(),
        "best_epsilon": best_epsilon,
        "best_smoothing": best_smoothing,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cv_seconds": cv_seconds,
        "total_seconds": total_seconds,
    }

    manifest_path = save_result(
        experiment=EXPERIMENT_NAME,
        params=params,
        seed_or_seeds={"truth_seed": TRUTH_SEED, "sample_seed": SAMPLE_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "cv_seed": CV_SEED},
        run_dir=run_dir,
        code_entrypoint=CODE_ENTRYPOINT,
        output_files=output_files,
    )

    print(f"Run directory: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"n_samples_actual = {n_actual_samples}")
    print(f"best_epsilon = {best_epsilon}, best_smoothing = {best_smoothing}")
    print(f"CV grid search took {cv_seconds:.2f}s, total {total_seconds:.2f}s")

    return run_dir, manifest_path, samples_df


if __name__ == "__main__":
    main()
