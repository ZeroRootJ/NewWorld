"""Simple kriging (via ``geostats.kb2d``) for the base-case ground truth
(docs/experiment_context.md section 4, third of the four methods to be
compared; RBF+bootstrap and GP-MLE are implemented separately, SGS in
``src/experiments/sgs.py``).

Pipeline
--------
1. Reuse the shared base-case conditioning data (truth field regenerated
   with seed=TRUTH_SEED, N_SAMPLES random interior samples drawn with
   SAMPLE_SEED) from ``src.experiments.base_case_conditioning`` -- the same
   function rbf_bootstrap.py / gp_mle.py use, with TRUTH_SEED, N_SAMPLES,
   SAMPLE_SEED, and the shared ``build_vario()`` all imported from there (not
   re-declared here), so all methods condition on identical sample locations
   and share the identical variogram model structurally (CLAUDE.md
   requirement).
2. Normal-score transform the sample porosity values
   (``ns, vr, vrg = geostats.nscore(df, VCOL)``) -- kb2d/the shared variogram
   dict is built on a standard-normal sill of 1.0 (nug + cc1 = 1.0), so
   kriging must be performed in normal-score space, not on the raw physical
   values.
3. ``geostats.kb2d`` with ``ktype=0`` (simple kriging, this project's
   baseline per docs/geostatspy_conventions.md section 4) and ``skmean=0.0``
   (the simple-kriging mean in normal-score space is 0 by construction --
   NOT the physical-units porosity mean).
4. Back-transform the point-estimate map (``kmap_ns``) to physical porosity
   units, cell by cell, via ``geostats.backtr_value`` using the same
   normal-score transform table (``vr``, ``vrg``) produced in step 2.
5. The kriging *variance* map (``vmap_ns``) is deliberately left in
   normal-score units -- back-transforming a variance (a second moment)
   through the nonlinear normal-score back-transform is not a simple
   pointwise operation the way the mean is, and is out of scope for this
   script. It is saved as-is, tagged as "normal-score units" in both the
   file name and manifest, for later calibration-metric use directly in
   normal-score space.

Run with: .venv/Scripts/python.exe -m src.experiments.kriging
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
)
from src.experiments.base_case_conditioning import (
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
    VCOL,
    build_vario,
    get_base_case_conditioning_data,
)
from src.io import make_run_dir, save_result

# --- kb2d search / kriging-type parameters -------------------------------
# ktype=0 (simple kriging) is this project's baseline
# (docs/geostatspy_conventions.md section 4) -- never left at a library
# default.
KTYPE = 0
# Simple-kriging mean in normal-score space is 0.0 by construction (the
# normal-score transform centers the data distribution on 0) -- NOT
# POR_MEAN (15.0), which is the physical-units mean and would be the wrong
# value to pass here.
SKMEAN_NS = 0.0
NXDIS, NYDIS = 1, 1  # point kriging (no block discretization)
NDMIN = 0
# ndmax=50: reused from the reference GeostatsPy demo code (Kriging_EXH.py)
# reviewed for this task; with radius effectively unlimited (below) this
# caps each kriged cell's neighborhood at the nearest 50 of the N_SAMPLES
# (125 requested / 121 actual) samples -- so for this base case ndmax=50 is
# not actually a binding constraint (fewer than 50 conditioning points
# exist), but is left in place unchanged since it is harmless and matches
# the reference convention.
NDMAX = 50
RADIUS = 9999  # effectively unlimited search radius, combined with ndmax above

# kb2d's tmin/tmax filter conditioning data to tmin <= NPor <= tmax (GSLIB's
# original intent was to drop missing-value sentinels). These are NOT the
# same thing as truth_model.py's ZMIN_NS/ZMAX_NS (which bound an unconditional
# simulation back-transform and are unused there since itrans=0) -- reusing
# those (-3.0/3.0) here silently dropped real normal-score conditioning
# values above 3.0 (e.g. NPor=3.0615 for one base-case sample), breaking the
# "identical sample locations across methods" requirement. Use the same
# effectively-unbounded literals src/experiments/sgs.py already uses for
# geostats.sgsim's tmin/tmax so no real conditioning data is ever trimmed.
KB2D_TMIN, KB2D_TMAX = -9999, 9999

# --- Back-transform (normal-score -> physical porosity units) parameters -
# +/-10 stdev (widened from +/-4, project decision 2026-09-14, reviewer-
# flagged issue): src/evaluation.py's kriging_fraction_in back-transforms the
# per-cell NS-space interval [kmap_ns + z_lo*std_ns, kmap_ns + z_hi*std_ns]
# to physical units for every nominal probability p, and z_lo/z_hi -> +/-inf
# as p -> 1. With only +/-4 stdev of "reach", any cell with a moderately
# large kriging variance already saturates BACKTR_ZMIN/ZMAX at a p well
# below 1, which pins its UMG accuracy-plot curve to the [ZMIN, ZMAX] clip
# instead of tracking that cell's actual (possibly poorly calibrated) NS
# interval -- an artifact of the clip's tightness, not of the kriging model
# itself. This risk grows on the nugget/range axes planned in
# docs/experiment_context.md (larger kriging variance cells), so the bound
# is widened to +/-10 stdev to push the clip point much further out into
# genuinely negligible-probability territory before it can bias fraction_in.
#
# NOT solved by any finite bound, and not intended to be: exactly at p=1
# (z_lo=-inf, z_hi=+inf), the NS-space interval is unbounded regardless of
# how large this constant is, so geostats.backtr_value *always* clips
# ns_lo/ns_hi to [BACKTR_ZMIN, BACKTR_ZMAX] and returns those two physical
# values at p=1, for every method that goes through this same quantile
# back-transform. This is the mathematical nature of a finite-support
# quantile back-transform table applied to an unbounded normal-score
# interval, not a bug -- do not re-investigate "why kriging's curve/behavior
# at p=1 looks different" without first re-reading this comment.
BACKTR_ZMIN = POR_MEAN - 10 * POR_STDEV
BACKTR_ZMAX = POR_MEAN + 10 * POR_STDEV
LTAIL, LTPAR = 1, BACKTR_ZMIN
UTAIL, UTPAR = 1, BACKTR_ZMAX

EXPERIMENT_NAME = "kriging"
CODE_ENTRYPOINT = "src/experiments/kriging.py"


def main():
    t_start = time.time()

    truth, samples_df = get_base_case_conditioning_data(sample_seed=SAMPLE_SEED)
    n_actual_samples = len(samples_df)
    vario = build_vario()

    # --- Normal-score transform of the sample data ----------------------
    # Exact variable names per project decision (do not rename vr/vrg --
    # they are the back-transform lookup table, not a mean/stdev pair).
    ns, vr, vrg = geostats.nscore(samples_df, VCOL)
    samples_df = samples_df.copy()
    samples_df["NPor"] = ns

    # --- Simple kriging in normal-score space ----------------------------
    t_kb2d_start = time.time()
    kmap_ns, vmap_ns = geostats.kb2d(
        samples_df,
        "X",
        "Y",
        "NPor",
        KB2D_TMIN,
        KB2D_TMAX,
        NX,
        XMN,
        XSIZ,
        NY,
        YMN,
        YSIZ,
        nxdis=NXDIS,
        nydis=NYDIS,
        ndmin=NDMIN,
        ndmax=NDMAX,
        radius=RADIUS,
        ktype=KTYPE,
        skmean=SKMEAN_NS,
        vario=vario,
    )
    kb2d_seconds = time.time() - t_kb2d_start

    # Independently reproduce kb2d's own tmin/tmax conditioning-data filter
    # (df.loc[(df[vcol] >= tmin) & (df[vcol] <= tmax)]) so a silent drop of
    # samples (as previously happened when ZMIN_NS/ZMAX_NS=-3/3 clipped a
    # real NPor=3.0615 value) is caught by the manifest, not just by manual
    # inspection.
    n_samples_used_by_kb2d = int(
        ((samples_df["NPor"] >= KB2D_TMIN) & (samples_df["NPor"] <= KB2D_TMAX)).sum()
    )
    if n_samples_used_by_kb2d != n_actual_samples:
        raise RuntimeError(
            f"kb2d tmin/tmax dropped {n_actual_samples - n_samples_used_by_kb2d} "
            f"conditioning sample(s): used {n_samples_used_by_kb2d} of "
            f"{n_actual_samples}. This violates the identical-sample-locations "
            "requirement across methods."
        )

    # kb2d's returned kmap/vmap already follow this project's row 0 = max-y
    # convention: verified directly from the geostatspy source
    # (`kmap[ny-iy-1, ix] = est` where `yloc = ymn + iy*ysiz` increases with
    # iy), i.e. iy=ny-1 (max yloc) is written to row 0. This matches
    # src/grid_utils.py's full_grid_coordinates row ordering exactly, so NO
    # row flip is applied here (unlike the external reference script's
    # `por_kmap[::-1]`, which is not needed for this project's convention).

    # --- Back-transform the point estimate to physical units -------------
    # Variance is intentionally NOT back-transformed here (see module
    # docstring) -- vmap_ns is saved as-is, in normal-score units.
    kmap_physical = np.empty((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            kmap_physical[iy, ix] = geostats.backtr_value(
                kmap_ns[iy, ix],
                vr,
                vrg,
                zmin=BACKTR_ZMIN,
                zmax=BACKTR_ZMAX,
                ltail=LTAIL,
                ltpar=LTPAR,
                utail=UTAIL,
                utpar=UTPAR,
            )

    total_seconds = time.time() - t_start

    # --- Save outputs ------------------------------------------------------
    run_dir = make_run_dir(EXPERIMENT_NAME)
    output_files = []

    samples_csv = "samples.csv"
    samples_df.to_csv(run_dir / samples_csv, index=False)
    output_files.append(samples_csv)

    transform_table_csv = "nscore_transform_table.csv"
    pd.DataFrame({"vr": vr, "vrg": vrg}).to_csv(run_dir / transform_table_csv, index=False)
    output_files.append(transform_table_csv)

    for name, arr in [
        ("kriging_mean_map_physical.npy", kmap_physical),
        ("kriging_var_map_ns.npy", vmap_ns),
        ("kmap_ns.npy", kmap_ns),
    ]:
        np.save(run_dir / name, arr)
        output_files.append(name)

    # --- QC figure: truth / kriging point estimate (physical) / kriging
    # variance (normal-score units) -----------------------------------------
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
    im2 = ax2.imshow(kmap_physical, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.scatter(samples_df["X"], samples_df["Y"], s=8, c="red", marker="+")
    ax2.set_title("Simple kriging point estimate (physical units)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    plt.colorbar(im2, ax=ax2, label="Porosity (%)")

    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(vmap_ns, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper", cmap="magma")
    ax3.scatter(samples_df["X"], samples_df["Y"], s=8, c="cyan", marker="+")
    ax3.set_title("Simple kriging variance (NS units, not back-transformed)")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    plt.colorbar(im3, ax=ax3, label="Variance (normal-score units)")

    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.9, wspace=0.35)
    qc_fig_name = "qc_kriging.png"
    plt.savefig(run_dir / qc_fig_name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    output_files.append(qc_fig_name)

    params = {
        "grid": {"nx": NX, "ny": NY, "xsiz": XSIZ, "ysiz": YSIZ, "xmn": XMN, "ymn": YMN},
        "truth_seed": TRUTH_SEED,
        "sample_seed": SAMPLE_SEED,
        "n_samples_requested": N_SAMPLES,
        "n_samples_actual": n_actual_samples,
        "n_samples_used_by_kb2d": n_samples_used_by_kb2d,
        "variogram": vario,
        "ktype": KTYPE,
        "skmean_ns": SKMEAN_NS,
        "nxdis": NXDIS,
        "nydis": NYDIS,
        "ndmin": NDMIN,
        "ndmax": NDMAX,
        "radius": RADIUS,
        "nscore_trim": {"tmin": KB2D_TMIN, "tmax": KB2D_TMAX},
        "backtransform": {
            "zmin": BACKTR_ZMIN,
            "zmax": BACKTR_ZMAX,
            "ltail": LTAIL,
            "ltpar": LTPAR,
            "utail": UTAIL,
            "utpar": UTPAR,
        },
        "kriging_variance_units": "normal-score (not back-transformed)",
        "kb2d_seconds": kb2d_seconds,
        "total_seconds": total_seconds,
    }

    manifest_path = save_result(
        experiment=EXPERIMENT_NAME,
        params=params,
        seed_or_seeds={"truth_seed": TRUTH_SEED, "sample_seed": SAMPLE_SEED},
        run_dir=run_dir,
        code_entrypoint=CODE_ENTRYPOINT,
        output_files=output_files,
    )

    print(f"Run directory: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"n_samples_actual = {n_actual_samples}")
    print(f"n_samples_used_by_kb2d = {n_samples_used_by_kb2d}")
    print(f"kmap_physical range: [{kmap_physical.min():.4f}, {kmap_physical.max():.4f}]")
    print(f"vmap_ns range: [{vmap_ns.min():.4f}, {vmap_ns.max():.4f}]")
    print(f"kb2d took {kb2d_seconds:.2f}s, total {total_seconds:.2f}s")

    return run_dir, manifest_path, samples_df


if __name__ == "__main__":
    main()
