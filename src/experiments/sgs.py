"""Sequential Gaussian Simulation (SGS, via ``geostats.sgsim``) conditioned
on the base-case sample data (docs/experiment_context.md section 4, fourth
of the four methods to be compared; RBF+bootstrap, GP-MLE and simple kriging
are implemented separately).

Pipeline
--------
1. Reuse the shared base-case conditioning data (truth field regenerated
   with seed=TRUTH_SEED, N_SAMPLES random interior samples drawn with
   SAMPLE_SEED) from ``src.experiments.base_case_conditioning`` -- the same
   function every other method-comparison script uses, with TRUTH_SEED,
   N_SAMPLES, SAMPLE_SEED, and ``build_vario()`` all imported from there (not
   re-declared here), so all methods condition on identical sample locations
   and share the identical variogram model structurally (CLAUDE.md
   requirement).
2. ``geostats.sgsim`` is called directly on the *physical-units* porosity
   column (``VCOL`` = 'Por') with ``itrans=1`` -- sgsim performs its own
   internal normal-score transform/back-transform. Per this task's explicit
   bug-avoidance instructions, samples are NOT manually normal-score
   transformed before this call (that would be a double transform --
   docs/geostatspy_conventions.md section 6 gotcha, reproduced as a bug in
   the external reference script this task's instructions explicitly warn
   against copying). The X, Y columns passed to this call are a jittered
   *copy* of the shared conditioning samples (see SGS_JITTER_* constants
   below) -- a project decision (confirmed with user, 2026-09-14) to work
   around a geostatspy edge case triggered by conditioning data sitting
   exactly on grid-cell centroids; samples.csv / the manifest's sample
   coordinates remain the original, un-jittered, snapped-centroid locations
   shared with every other method.
3. ``N_REALIZATIONS`` conditional realizations are drawn (imported from
   ``rbf_bootstrap.N_BOOTSTRAP`` so the replicate count matches RBF+
   bootstrap exactly, per docs/geostatspy_conventions.md section 7 point 3).
   Each realization is already back-transformed to physical porosity units
   by sgsim itself (itrans=1) -- no separate back-transform step is needed
   here (unlike kriging.py, where the back-transform had to be done
   manually via geostats.backtr_value on the kriged normal-score map).
4. Cell-wise mean and variance across the realizations are this method's
   point estimate and uncertainty model, computed directly in physical
   units (no nonlinear back-transform-of-a-moment issue here, unlike
   kriging's variance map -- these are moments of an ensemble of
   already-physical-unit realizations).

Run with: .venv/Scripts/python.exe -m src.experiments.sgs
"""

import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geostatspy.geostats as geostats

from src.experiments.base_case import (
    NX,
    NY,
    XMN,
    XMAX,
    XMIN,
    YMN,
    YMAX,
    YMIN,
    XSIZ,
    YSIZ,
    POR_MEAN,
    POR_STDEV,
    HMAJ1,
)
from src.experiments.base_case_conditioning import (
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    build_vario,
    get_base_case_conditioning_data,
)
from src.experiments.rbf_bootstrap import N_BOOTSTRAP
from src.io import make_run_dir, save_result

# Number of conditional realizations, matched to RBF+bootstrap's replicate
# count so all methods being compared use the same number of replicates
# (docs/geostatspy_conventions.md section 7, point 3). NOT re-declared as an
# independent literal -- imported directly from rbf_bootstrap.py above.
N_REALIZATIONS = N_BOOTSTRAP

# New named seed (does not collide with TRUTH_SEED=101, SAMPLE_SEED=20,
# CV_SEED=40, BOOTSTRAP_SEED=30, GP_RANDOM_STATE=50 used elsewhere in this
# comparison).
SGS_SEED = 60

# --- Local coordinate jitter (bug-avoidance, confirmed with user 2026-09-14)
# -----------------------------------------------------------------------
# geostatspy.geostats.sgsim raises `numpy.linalg.LinAlgError: Matrix is
# singular to machine precision` inside its internal simple-kriging solve
# for some grid nodes, when conditioning data sit exactly on the simulation
# grid's node centroids (this project's samples are snapped to cell
# centroids by src/sampling.py, per docs/geostatspy_conventions.md). Working
# hypothesis, consistent with everything observed while debugging this (not
# independently verified against sgsim's internal kriging-matrix values):
# with data exactly grid-aligned, sgsim's per-node local search -- combining
# nearby original data with nearby *already-simulated grid nodes* (also
# grid-aligned, and, with this script's mults=1/nmult=3 multi-grid search
# strategy, visited in a coarse-to-fine order that revisits regular
# sub-grids) -- has a substantially higher chance of assembling a
# conditioning set with an exact or near-exact geometric symmetry, which for
# an isotropic covariance model can produce a singular/near-singular
# simple-kriging matrix. This reproduces even for very sparse (as few as 1),
# well-separated-but-grid-aligned synthetic datasets tested in isolation
# while debugging, i.e. it is not specific to this project's dense,
# possibly-collinear 454-sample dataset. Below exact singularity, many other
# nodes still get a near-singular solve whose result is numerically garbage
# (see the `WARNING: grid node location ...` / absurd conditional mean
# magnitudes observed before this fix).
#
# Fix (project decision, confirmed with user, NOT independently chosen
# here): apply a deterministic, negligible-relative-to-cell-size jitter to
# the X, Y columns of a *copy* of samples_df immediately before the
# geostats.sgsim call only. This is NOT applied to samples_df itself, so
# the "official" sample coordinates recorded in samples.csv and the
# manifest (and used identically by kriging/RBF/GPR) are the original
# snapped centroids, unperturbed -- only sgsim's internal solve sees the
# jittered copy. SGS_JITTER_SEED is a new named seed (distinct from
# SGS_SEED, which drives sgsim's own internal random path) so the jitter
# draw is reproducible independently of sgsim's internal RNG.
SGS_JITTER_SEED = 70
# Magnitude chosen by empirical sweep over SGS_JITTER_MAGNITUDE in
# {1e-3, 2e-3, 5e-3, 7.5e-3, 1e-2, 2e-2, 5e-2} m, run against a conditioning
# dataset of n=454 samples (this project's N_SAMPLES was later changed to
# 125 requested / 121 actual, 2026-09-14 -- see base_case_conditioning.py)
# with the production N_REALIZATIONS=10 and search/kriging parameters below.
# Magnitudes <= 5e-3 m left 1-2127 grid nodes per run with a near-singular
# simple-kriging solve (the `WARNING: grid node location ...` console
# messages, with absurd conditional-mean magnitudes up to ~1e21 before
# zmin/zmax clipping bounded the visible damage); 1e-2 m (5e-4 of the 20 m
# grid cell size, and 3.3e-5 of the 300 m variogram range -- negligible on
# both scales) was the smallest magnitude tested that produced zero such
# warnings across all 10 realizations, with realization mean/stdev matching
# the target distribution (mean~15, stdev~3) rather than the inflated values
# seen at smaller magnitudes. This sweep predates the n=454 -> n=121 sample
# count change; it has NOT been re-run at n=121, but reviewer re-verified
# (2026-09-14) that this same 1e-2 m magnitude still produces zero
# near-singular-solve warnings and correct realization statistics at n=121,
# so no change to SGS_JITTER_MAGNITUDE or a re-sweep is needed.
SGS_JITTER_MAGNITUDE = 1e-2  # meters, +/- half-width of the uniform jitter

# --- Post-hoc honoring re-enforcement (bug-avoidance, discovered + reported
# 2026-09-14, NOT part of the originally-confirmed jitter-only design --
# flagged here for orchestrator/user awareness) -------------------------
# geostatspy.geostats.sgsim's own conditioning-data honoring mechanism (the
# "Reassigning data to nodes" step, geostats.py ~line 4081-4088) only
# overwrites a grid node with a datum's exact value if
# `abs(xx - x[iid]) + abs(yy - y[iid]) <= TINY`, where `xx, yy` is the exact
# grid-cell centroid recomputed from the datum's (possibly jittered) x[iid]
# via sgsim's own (truncating) `getindex`, and `TINY = 1.0e-4` (0.0001 m) is
# hard-coded inside sgsim. Our SGS_JITTER_MAGNITUDE=1e-2 m is 100x larger
# than this TINY tolerance, so for most jittered samples `test > TINY` and
# sgsim's internal honoring check silently fails to fire -- the grid node
# is left holding whatever value the main simulation loop happened to
# simulate there (not necessarily close to the true datum value), which is
# what produced the ~3 porosity-unit (roughly 1 sigma) honor violations
# observed when jitter was applied but this section's fix was not yet in
# place. Empirically, a jitter magnitude small enough to satisfy sgsim's
# `<= TINY` check (i.e. ~1e-4 m or smaller) is NOT large enough to reliably
# avoid the singular/near-singular kriging matrix this jitter exists to
# avoid in the first place (see SGS_JITTER_MAGNITUDE sweep above -- 5e-3 m,
# 50x TINY, still left near-singular nodes) -- i.e. no single jitter
# magnitude satisfies both requirements simultaneously with this library.
#
# Fix: after sgsim returns, explicitly re-enforce honoring ourselves --
# overwrite each realization's value at every conditioning sample's grid
# cell with that sample's true (un-jittered) recorded value. This is
# exactly the operation sgsim's own "Reassigning data to nodes" step is
# supposed to perform; it does not change how sgsim's internal solve
# incorporates the data during simulation (jitter still does that, as
# originally confirmed), it only corrects the *output* at conditioning
# locations to guarantee exact honoring is not silently undermined by the
# jitter workaround. QC below reports the honor error both *before*
# (`qc_honor_raw_sgsim_output`, i.e. as sgsim itself returned it) and
# *after* (`qc_honor`) this correction, so the TINY-threshold effect
# remains visible/auditable rather than hidden.
APPLY_HONOR_REENFORCEMENT = True

# --- sgsim back-transform parameters (physical-units column, itrans=1) ---
# Same +/-4 stdev convention used elsewhere in this comparison
# (kriging.py's BACKTR_ZMIN/ZMAX, base_case.py's QC plot vmin/vmax).
SGSIM_ZMIN = POR_MEAN - 4 * POR_STDEV
SGSIM_ZMAX = POR_MEAN + 4 * POR_STDEV
SGSIM_LTAIL, SGSIM_LTPAR = 1, SGSIM_ZMIN
SGSIM_UTAIL, SGSIM_UTPAR = 1, SGSIM_ZMAX

# --- sgsim search / kriging-type parameters ------------------------------
# Values below are the docs/geostatspy_conventions.md section 5 example
# values (NOT truth_model.py's unconditional-simulation values of
# mults=0/nodmax=10, which that script's docstring explicitly documents as
# intentionally different for the unconditional case -- this is a
# conditional-simulation script, so the documented example values apply
# here instead).
NDMIN = 0
NDMAX = 20
NODMAX = 20
MULTS = 1
NMULT = 3
NOCT = -1
KTYPE = 0  # simple kriging inside sgsim, this project's baseline
COLOCORR = 0.0
SEC_MAP = 0

EXPERIMENT_NAME = "sgs"
CODE_ENTRYPOINT = "src/experiments/sgs.py"

# --- QC variogram check parameters (gamv on one example realization) -----
QC_REALIZATION_INDEX = 0
QC_LAG_DIST = 40.0  # comparable to XSIZ/YSIZ=20m cell size
QC_LAG_TOL = 20.0
QC_NLAG = 15
QC_AZI = 0.0
QC_ATOL = 90.0  # omnidirectional-ish tolerance, isotropic base-case variogram
QC_BANDH = 9999.0
QC_ISILL = 1


def main():
    t_start = time.time()

    truth, samples_df = get_base_case_conditioning_data(sample_seed=SAMPLE_SEED)
    n_actual_samples = len(samples_df)
    vario = build_vario()

    # --- Local jitter, sgsim-call copy only (see SGS_JITTER_* comment above)
    # samples_df itself (saved to samples.csv / used by every other method)
    # is left untouched -- only this copy, passed to geostats.sgsim below,
    # is perturbed.
    jitter_rng = np.random.RandomState(SGS_JITTER_SEED)
    sgsim_input_df = samples_df.copy()
    sgsim_input_df["X"] = sgsim_input_df["X"] + jitter_rng.uniform(
        -SGS_JITTER_MAGNITUDE, SGS_JITTER_MAGNITUDE, size=n_actual_samples
    )
    sgsim_input_df["Y"] = sgsim_input_df["Y"] + jitter_rng.uniform(
        -SGS_JITTER_MAGNITUDE, SGS_JITTER_MAGNITUDE, size=n_actual_samples
    )

    t_sgsim_start = time.time()
    sim = geostats.sgsim(
        sgsim_input_df,
        "X",
        "Y",
        VCOL,
        wcol=-1,
        scol=-1,
        tmin=-9999,
        tmax=9999,
        itrans=1,
        ismooth=0,
        dftrans=0,
        tcol=0,
        twtcol=0,
        zmin=SGSIM_ZMIN,
        zmax=SGSIM_ZMAX,
        ltail=SGSIM_LTAIL,
        ltpar=SGSIM_LTPAR,
        utail=SGSIM_UTAIL,
        utpar=SGSIM_UTPAR,
        nsim=N_REALIZATIONS,
        nx=NX,
        xmn=XMN,
        xsiz=XSIZ,
        ny=NY,
        ymn=YMN,
        ysiz=YSIZ,
        seed=SGS_SEED,
        ndmin=NDMIN,
        ndmax=NDMAX,
        nodmax=NODMAX,
        mults=MULTS,
        nmult=NMULT,
        noct=NOCT,
        ktype=KTYPE,
        colocorr=COLOCORR,
        sec_map=SEC_MAP,
        vario=vario,
    )
    sgsim_seconds = time.time() - t_sgsim_start

    # sim indexed by realization: sim[0], sim[1], ... (docs/
    # geostatspy_conventions.md section 5); each is already in physical
    # porosity units (itrans=1 handles the back-transform internally) and
    # already follows this project's row 0 = max-y convention (same sgsim
    # call used unconditionally in src/truth_model.py, whose docstring
    # documents that convention).
    realizations = np.stack([sim[r] for r in range(N_REALIZATIONS)], axis=0)  # (N_REALIZATIONS, NY, NX)

    # --- Sample -> grid-cell mapping (used by both honor re-enforcement and
    # QC 1 below). Computed from the original (un-jittered) samples_df, via
    # the same nearest-cell rounding convention src/sampling.py used to
    # generate these coordinates in the first place (samples are exact cell
    # centroids, so this recovers each sample's true owning cell exactly,
    # independent of jitter or of sgsim's own internal indexing).
    sample_ix = np.round((samples_df["X"].values - XMN) / XSIZ).astype(int)
    sample_iy = np.round((samples_df["Y"].values - YMN) / YSIZ).astype(int)
    sample_row = NY - 1 - sample_iy  # row 0 = max-y convention
    sample_col = sample_ix

    # --- QC 1a: honoring as returned directly by geostats.sgsim, i.e.
    # BEFORE this script's post-hoc honor re-enforcement (see
    # APPLY_HONOR_REENFORCEMENT comment above) -- kept purely as a
    # diagnostic so the TINY-threshold effect of the jitter workaround stays
    # visible/auditable rather than silently hidden by the correction below.
    honor_abs_errors_raw = np.abs(
        realizations[:, sample_row, sample_col] - samples_df[VCOL].values[None, :]
    )  # (N_REALIZATIONS, n_actual_samples)
    honor_raw_max_abs_error = float(honor_abs_errors_raw.max())
    honor_raw_mean_abs_error = float(honor_abs_errors_raw.mean())

    # --- Post-hoc honor re-enforcement (see APPLY_HONOR_REENFORCEMENT
    # comment above): force every realization's value at each conditioning
    # sample's grid cell to that sample's true, un-jittered recorded value.
    if APPLY_HONOR_REENFORCEMENT:
        realizations[:, sample_row, sample_col] = samples_df[VCOL].values[None, :]

    sgs_mean_map = realizations.mean(axis=0)
    sgs_var_map = realizations.var(axis=0, ddof=1)

    total_seconds = time.time() - t_start

    # --- QC 1b: honoring of conditioning data, final (post re-enforcement)
    # output -- for each sample location, compare its recorded Por value
    # against the grid-cell value in each realization. With
    # APPLY_HONOR_REENFORCEMENT=True this is exact by construction (kept as
    # a QC check anyway, to catch indexing mistakes rather than to test
    # sgsim's own behavior -- see honor_raw_* above for that).
    honor_abs_errors = np.abs(
        realizations[:, sample_row, sample_col] - samples_df[VCOL].values[None, :]
    )  # (N_REALIZATIONS, n_actual_samples)
    honor_max_abs_error = float(honor_abs_errors.max())
    honor_mean_abs_error = float(honor_abs_errors.mean())

    # --- QC 2: histogram reproduction ------------------------------------
    realization_means = realizations.reshape(N_REALIZATIONS, -1).mean(axis=1)
    realization_stdevs = realizations.reshape(N_REALIZATIONS, -1).std(axis=1)

    # --- QC 3: variogram structure (gamv on one example realization) -----
    qc_realization = realizations[QC_REALIZATION_INDEX]
    ny_idx, nx_idx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    qc_row = ny_idx.ravel()
    qc_col = nx_idx.ravel()
    qc_x = XMN + qc_col * XSIZ
    qc_y = YMN + (NY - 1 - qc_row) * YSIZ
    qc_df = pd.DataFrame({"X": qc_x, "Y": qc_y, VCOL: qc_realization[qc_row, qc_col]})
    qc_lag, qc_gamma, qc_npair = geostats.gamv(
        qc_df, "X", "Y", VCOL, -9999, 9999, QC_LAG_DIST, QC_LAG_TOL, QC_NLAG, QC_AZI, QC_ATOL, QC_BANDH, QC_ISILL
    )

    # --- Save outputs ------------------------------------------------------
    run_dir = make_run_dir(EXPERIMENT_NAME)
    output_files = []

    samples_csv = "samples.csv"
    samples_df.to_csv(run_dir / samples_csv, index=False)
    output_files.append(samples_csv)

    for name, arr in [
        ("sgs_realizations.npy", realizations),
        ("sgs_mean_map.npy", sgs_mean_map),
        ("sgs_var_map.npy", sgs_var_map),
    ]:
        np.save(run_dir / name, arr)
        output_files.append(name)

    qc_variogram_csv = "qc_variogram.csv"
    pd.DataFrame({"lag": qc_lag, "gamma": qc_gamma, "npair": qc_npair}).to_csv(
        run_dir / qc_variogram_csv, index=False
    )
    output_files.append(qc_variogram_csv)

    # --- QC figure: truth / 2 example realizations / mean / variance ----
    vmin = float(np.min(truth))
    vmax = float(np.max(truth))
    fig = plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(truth, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.scatter(samples_df["X"], samples_df["Y"], s=8, c="red", marker="+", label="samples")
    ax1.set_title(f"Truth (seed={TRUTH_SEED})")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    plt.colorbar(im1, ax=ax1, label="Porosity (%)")
    ax1.legend(loc="upper right", fontsize=8)

    for i, r in enumerate([0, 1]):
        ax = plt.subplot(2, 3, 2 + i)
        im = ax.imshow(realizations[r], extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.scatter(samples_df["X"], samples_df["Y"], s=8, c="red", marker="+")
        ax.set_title(f"SGS realization {r}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.colorbar(im, ax=ax, label="Porosity (%)")

    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(sgs_mean_map, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    ax4.scatter(samples_df["X"], samples_df["Y"], s=8, c="red", marker="+")
    ax4.set_title(f"SGS mean ({N_REALIZATIONS} realizations)")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    plt.colorbar(im4, ax=ax4, label="Porosity (%)")

    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(sgs_var_map, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="magma")
    ax5.scatter(samples_df["X"], samples_df["Y"], s=8, c="cyan", marker="+")
    ax5.set_title(f"SGS variance ({N_REALIZATIONS} realizations)")
    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")
    plt.colorbar(im5, ax=ax5, label="Variance (Porosity %^2)")

    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(qc_lag, qc_gamma, "o-", label=f"realization {QC_REALIZATION_INDEX} experimental")
    ax6.axhline(1.0, color="gray", linestyle="--", label="sill (standardized=1.0)")
    ax6.axvline(HMAJ1, color="red", linestyle=":", label=f"base-case range={HMAJ1:g}m")
    ax6.set_title("QC: experimental variogram vs. base-case range")
    ax6.set_xlabel("Lag distance (m)")
    ax6.set_ylabel("Semivariance (standardized)")
    ax6.legend(fontsize=7)

    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.93, wspace=0.4, hspace=0.4)
    qc_fig_name = "qc_sgs.png"
    plt.savefig(run_dir / qc_fig_name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    output_files.append(qc_fig_name)

    params = {
        "grid": {"nx": NX, "ny": NY, "xsiz": XSIZ, "ysiz": YSIZ, "xmn": XMN, "ymn": YMN},
        "truth_seed": TRUTH_SEED,
        "sample_seed": SAMPLE_SEED,
        "n_samples_requested": N_SAMPLES,
        "n_samples_actual": n_actual_samples,
        "variogram": vario,
        "n_realizations": N_REALIZATIONS,
        "sgs_seed": SGS_SEED,
        "sgsim_input_jitter": {
            "applied": True,
            "reason": (
                "geostats.sgsim raised numpy.linalg.LinAlgError (singular/"
                "near-singular simple-kriging matrix) when conditioning on "
                "samples snapped exactly to grid-cell centroids; a "
                "deterministic, cell-size-negligible jitter is applied to "
                "X, Y in the DataFrame copy passed to geostats.sgsim only. "
                "samples.csv and this manifest's sample coordinates below "
                "are the original, un-jittered, snapped-centroid locations "
                "-- identical to those used by kriging/RBF/GPR."
            ),
            "jitter_seed": SGS_JITTER_SEED,
            "jitter_magnitude_m": SGS_JITTER_MAGNITUDE,
            "jitter_magnitude_frac_of_cell_size": SGS_JITTER_MAGNITUDE / XSIZ,
            "applies_to": "geostats.sgsim call input DataFrame copy only, not samples_df/samples.csv",
        },
        "honor_reenforcement": {
            "applied": APPLY_HONOR_REENFORCEMENT,
            "reason": (
                "geostats.sgsim's own data-honoring step only reassigns a "
                "grid node to a datum's exact value if "
                "abs(xx-x)+abs(yy-y) <= TINY (TINY=1e-4 m, hard-coded in "
                "geostats.sgsim); SGS_JITTER_MAGNITUDE=1e-2 m is 100x this "
                "tolerance, so sgsim's own honoring silently fails to fire "
                "for most jittered samples (see qc_honor_raw_sgsim_output "
                "below for the resulting, non-negligible violation). This "
                "script therefore re-enforces honoring itself post-hoc: "
                "every realization's value at each conditioning sample's "
                "grid cell is overwritten with that sample's true, "
                "un-jittered recorded value after geostats.sgsim returns."
            ),
        },
        "ndmin": NDMIN,
        "ndmax": NDMAX,
        "nodmax": NODMAX,
        "mults": MULTS,
        "nmult": NMULT,
        "noct": NOCT,
        "ktype": KTYPE,
        "colocorr": COLOCORR,
        "sec_map": SEC_MAP,
        "sgsim_backtransform": {
            "zmin": SGSIM_ZMIN,
            "zmax": SGSIM_ZMAX,
            "ltail": SGSIM_LTAIL,
            "ltpar": SGSIM_LTPAR,
            "utail": SGSIM_UTAIL,
            "utpar": SGSIM_UTPAR,
        },
        "qc_honor": {
            "max_abs_error": honor_max_abs_error,
            "mean_abs_error": honor_mean_abs_error,
        },
        "qc_honor_raw_sgsim_output": {
            "note": (
                "Honor error as geostats.sgsim itself returned it, BEFORE "
                "this script's post-hoc honor re-enforcement (see "
                "honor_reenforcement above) -- diagnostic only, not the "
                "final saved realizations."
            ),
            "max_abs_error": honor_raw_max_abs_error,
            "mean_abs_error": honor_raw_mean_abs_error,
        },
        "qc_histogram": {
            "realization_means": realization_means.tolist(),
            "realization_stdevs": realization_stdevs.tolist(),
            "target_mean": POR_MEAN,
            "target_stdev": POR_STDEV,
        },
        "qc_variogram_params": {
            "lag_dist": QC_LAG_DIST,
            "lag_tol": QC_LAG_TOL,
            "nlag": QC_NLAG,
            "azi": QC_AZI,
            "atol": QC_ATOL,
            "bandh": QC_BANDH,
            "isill": QC_ISILL,
            "realization_index": QC_REALIZATION_INDEX,
        },
        "sgsim_seconds": sgsim_seconds,
        "total_seconds": total_seconds,
    }

    manifest_path = save_result(
        experiment=EXPERIMENT_NAME,
        params=params,
        seed_or_seeds={"truth_seed": TRUTH_SEED, "sample_seed": SAMPLE_SEED, "sgs_seed": SGS_SEED},
        run_dir=run_dir,
        code_entrypoint=CODE_ENTRYPOINT,
        output_files=output_files,
    )

    print(f"Run directory: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"n_samples_actual = {n_actual_samples}")
    print(
        f"QC honor (raw sgsim output, pre-reenforcement): "
        f"max_abs_error={honor_raw_max_abs_error:.6f}, mean_abs_error={honor_raw_mean_abs_error:.6f}"
    )
    print(f"QC honor (final, post-reenforcement): max_abs_error={honor_max_abs_error:.6f}, mean_abs_error={honor_mean_abs_error:.6f}")
    print(f"QC histogram: realization means={np.round(realization_means, 3)}")
    print(f"QC histogram: realization stdevs={np.round(realization_stdevs, 3)}")
    print(f"sgsim took {sgsim_seconds:.2f}s, total {total_seconds:.2f}s")

    return run_dir, manifest_path, samples_df


if __name__ == "__main__":
    main()
