"""GP-MLE (Gaussian Process Regression with marginal-likelihood-tuned
hyperparameters) for the base-case ground truth (docs/experiment_context.md
section 4).

Pipeline
--------
1. Reuse the shared base-case conditioning data (truth field regenerated
   with seed=TRUTH_SEED, N_SAMPLES random interior samples drawn with
   SAMPLE_SEED) from ``src.experiments.base_case_conditioning`` -- the same
   function rbf_bootstrap.py uses, with TRUTH_SEED, N_SAMPLES, and
   SAMPLE_SEED all imported from there (not re-declared here), so both
   methods condition on identical sample locations structurally, not by
   coincidence (CLAUDE.md requirement).
2. Fit a GaussianProcessRegressor with kernel
   ConstantKernel() * RBF(length_scale) + WhiteKernel(noise_level), i.e.
   signal variance x isotropic length scale + nugget/noise. Hyperparameters
   are tuned automatically by marginal likelihood maximization (sklearn's
   default ``.fit()`` behavior) -- not hand-set -- with
   ``n_restarts_optimizer`` to reduce the risk of a poor local optimum.
3. Unlike RBF+bootstrap, the GP gives posterior mean AND variance
   analytically in a single ``predict(..., return_std=True)`` call over the
   full 50x50 grid -- no repeated refitting / bootstrap ensemble is needed,
   because the Bayesian posterior *is* the uncertainty model.

Run with: .venv/Scripts/python.exe -m src.experiments.gp_mle
"""

import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

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

# --- GP kernel initial values / bounds ----------------------------------
# Initial guesses are order-of-magnitude starting points only -- the
# optimizer (marginal likelihood maximization, via multiple restarts) is
# what actually selects the final hyperparameters, per
# docs/experiment_context.md section 4 ("do not hand-tune"). Bounds are
# wide relative to the domain (1000m extent, samples spaced ~40m) and the
# porosity distribution (stdev=3 -> variance=9) so the optimizer is not
# artificially constrained near the true generating values.
CONSTANT_VALUE_INIT = 1.0
CONSTANT_VALUE_BOUNDS = (1e-3, 1e3)
LENGTH_SCALE_INIT = 100.0
LENGTH_SCALE_BOUNDS = (1e-2, 1e4)
NOISE_LEVEL_INIT = 1.0
NOISE_LEVEL_BOUNDS = (1e-5, 1e3)

# Number of random restarts for the marginal-likelihood optimizer, per task
# spec range (5-10) to reduce risk of a poor local optimum.
N_RESTARTS_OPTIMIZER = 8

GP_RANDOM_STATE = 50  # controls the optimizer's random restarts

EXPERIMENT_NAME = "gp_mle"
CODE_ENTRYPOINT = "src/experiments/gp_mle.py"


def build_kernel():
    return (
        ConstantKernel(CONSTANT_VALUE_INIT, CONSTANT_VALUE_BOUNDS)
        * RBF(length_scale=LENGTH_SCALE_INIT, length_scale_bounds=LENGTH_SCALE_BOUNDS)
        + WhiteKernel(noise_level=NOISE_LEVEL_INIT, noise_level_bounds=NOISE_LEVEL_BOUNDS)
    )


def main():
    t_start = time.time()

    truth, samples_df = get_base_case_conditioning_data(sample_seed=SAMPLE_SEED)
    n_actual_samples = len(samples_df)

    X = samples_df[["X", "Y"]].values
    y = samples_df[VCOL].values

    kernel = build_kernel()
    # normalize_y=True: standard sklearn practice for GP regression when the
    # target has a non-zero mean (porosity ~15) -- the GP prior itself is
    # zero-mean, so y is internally centered/scaled for the optimizer, and
    # sklearn's predict() automatically reverses this so the returned mean
    # and std are already in the original porosity units.
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        normalize_y=True,
        random_state=GP_RANDOM_STATE,
    )

    t_fit_start = time.time()
    gpr.fit(X, y)
    fit_seconds = time.time() - t_fit_start

    grid_coords = full_grid_coordinates(NX, NY, XMN, YMN, XSIZ, YSIZ)

    t_pred_start = time.time()
    post_mean, post_std = gpr.predict(grid_coords, return_std=True)
    predict_seconds = time.time() - t_pred_start

    post_mean_map = post_mean.reshape(NY, NX)
    post_std_map = post_std.reshape(NY, NX)
    post_var_map = post_std_map ** 2

    total_seconds = time.time() - t_start

    # --- Extract fitted kernel hyperparameters --------------------------
    # gpr.kernel_ == Sum(Product(ConstantKernel, RBF), WhiteKernel); pull out
    # the three pieces by structure (k1 = left branch, k2 = right branch of
    # each combinator), matching how `kernel` was built above.
    fitted_kernel = gpr.kernel_
    constant_kernel = fitted_kernel.k1.k1
    rbf_kernel = fitted_kernel.k1.k2
    white_kernel = fitted_kernel.k2

    # normalize_y=True means these raw values are in the *internally
    # standardized* y-space; convert signal/noise variance back to real
    # porosity^2 units for reporting (length_scale is a function of X only,
    # so it needs no conversion). See sklearn GaussianProcessRegressor
    # source: gpr._y_train_std is the std used for that internal scaling.
    y_train_std = float(gpr._y_train_std)
    signal_variance_normalized = float(constant_kernel.constant_value)
    noise_variance_normalized = float(white_kernel.noise_level)
    signal_variance_real = signal_variance_normalized * (y_train_std ** 2)
    noise_variance_real = noise_variance_normalized * (y_train_std ** 2)
    length_scale = float(rbf_kernel.length_scale)

    # --- Save outputs --------------------------------------------------
    run_dir = make_run_dir(EXPERIMENT_NAME)
    output_files = []

    samples_csv = "samples.csv"
    samples_df.to_csv(run_dir / samples_csv, index=False)
    output_files.append(samples_csv)

    for name, arr in [
        ("posterior_mean_map.npy", post_mean_map),
        ("posterior_var_map.npy", post_var_map),
    ]:
        np.save(run_dir / name, arr)
        output_files.append(name)

    # --- QC figure: truth / GP posterior mean / GP posterior variance ---
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
    im2 = ax2.imshow(post_mean_map, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.scatter(samples_df["X"], samples_df["Y"], s=8, c="red", marker="+")
    ax2.set_title(f"GP posterior mean (l={length_scale:.1f}m)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    plt.colorbar(im2, ax=ax2, label="Porosity (%)")

    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(post_var_map, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="magma")
    ax3.scatter(samples_df["X"], samples_df["Y"], s=8, c="cyan", marker="+")
    ax3.set_title("GP posterior variance")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    plt.colorbar(im3, ax=ax3, label="Variance (Porosity %^2)")

    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.9, wspace=0.35)
    qc_fig_name = "qc_gp_mle.png"
    plt.savefig(run_dir / qc_fig_name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    output_files.append(qc_fig_name)

    params = {
        "grid": {"nx": NX, "ny": NY, "xsiz": XSIZ, "ysiz": YSIZ, "xmn": XMN, "ymn": YMN},
        "truth_seed": TRUTH_SEED,
        "sample_seed": SAMPLE_SEED,
        "n_samples_requested": N_SAMPLES,
        "n_samples_actual": n_actual_samples,
        "kernel_init": {
            "constant_value_init": CONSTANT_VALUE_INIT,
            "constant_value_bounds": CONSTANT_VALUE_BOUNDS,
            "length_scale_init": LENGTH_SCALE_INIT,
            "length_scale_bounds": LENGTH_SCALE_BOUNDS,
            "noise_level_init": NOISE_LEVEL_INIT,
            "noise_level_bounds": NOISE_LEVEL_BOUNDS,
        },
        "n_restarts_optimizer": N_RESTARTS_OPTIMIZER,
        "gp_random_state": GP_RANDOM_STATE,
        "normalize_y": True,
        "fitted_kernel_str": str(fitted_kernel),
        "fitted_hyperparameters": {
            "length_scale_m": length_scale,
            "signal_variance_normalized": signal_variance_normalized,
            "noise_variance_normalized": noise_variance_normalized,
            "y_train_std": y_train_std,
            "signal_variance_real_units": signal_variance_real,
            "noise_variance_real_units": noise_variance_real,
        },
        "log_marginal_likelihood": float(gpr.log_marginal_likelihood(gpr.kernel_.theta)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
    }

    manifest_path = save_result(
        experiment=EXPERIMENT_NAME,
        params=params,
        seed_or_seeds={"truth_seed": TRUTH_SEED, "sample_seed": SAMPLE_SEED, "gp_random_state": GP_RANDOM_STATE},
        run_dir=run_dir,
        code_entrypoint=CODE_ENTRYPOINT,
        output_files=output_files,
    )

    print(f"Run directory: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"n_samples_actual = {n_actual_samples}")
    print(f"Fitted kernel: {fitted_kernel}")
    print(
        f"length_scale={length_scale:.3f}m, "
        f"signal_variance(real)={signal_variance_real:.4f}, "
        f"noise_variance(real)={noise_variance_real:.4f}"
    )
    print(f"Fit took {fit_seconds:.2f}s, predict {predict_seconds:.2f}s, total {total_seconds:.2f}s")

    return run_dir, manifest_path, samples_df


if __name__ == "__main__":
    main()
