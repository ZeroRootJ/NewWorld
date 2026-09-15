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
    HMAJ1 as _COND_HMAJ1,
    HMIN1 as _COND_HMIN1,
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
# +/-4 stdev (reverted from +/-10, project decision 2026-09-14, second
# reversal same day -- see docs/progress.md for both). History:
#
# (a) Originally +/-4 stdev. A 2026-09-14 review found that
#     src/evaluation.py's kriging_fraction_in back-transforms the per-cell
#     NS-space interval [kmap_ns + z_lo*std_ns, kmap_ns + z_hi*std_ns] to
#     physical units for every nominal probability p, and z_lo/z_hi -> +/-inf
#     as p -> 1. With only +/-4 stdev of "reach", any cell with a moderately
#     large kriging variance already saturates BACKTR_ZMIN/ZMAX at a p well
#     below 1, which pins its UMG accuracy-plot curve to the [ZMIN, ZMAX]
#     clip instead of tracking that cell's actual (possibly poorly
#     calibrated) NS interval -- an artifact of the clip's tightness, not of
#     the kriging model itself. The bound was widened to +/-10 stdev
#     ([-15, 45]) to push this clip point further into negligible-probability
#     territory.
# (b) Later the same day, adding the physical-units Monte Carlo kriging
#     variance approximation (kriging_var_map_physical_mc.npy, see below)
#     exposed the opposite failure mode of (a)'s fix: with zmax=45, cells
#     whose predictive distribution (in normal-score space) has appreciable
#     mass beyond the normal-score transform table's range (vrg in
#     [-2.64, 2.64], i.e. physical [8.35, 22.07]) get back-transformed via
#     the LINEAR tail extrapolation (LTAIL=UTAIL=1) all the way out to the
#     now-much-more-permissive 45% bound, producing physically implausible
#     draws and inflating that single cell's Monte Carlo physical-unit
#     variance to ~61 %^2 (vs. <=5 %^2 everywhere else) and distorting the
#     shared variance color scale in the truth/predictions figures.
# (c) Project decision 2026-09-14: revert to +/-4 stdev ([3, 27]), accepting
#     that (a)'s tail-clipping-bias risk in kriging_fraction_in/UMG may
#     reappear (especially on the larger-kriging-variance nugget/range axes
#     planned in docs/experiment_context.md) as the lesser of the two
#     tradeoffs. Do not re-litigate this choice -- it has already been
#     explained to and confirmed by the user; if the tail-clipping symptom in
#     (a) resurfaces on a later axis, that is an accepted, known consequence
#     of this reversion, not a new bug to silently "fix" by re-widening this
#     constant again.
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
BACKTR_ZMIN = POR_MEAN - 4 * POR_STDEV
BACKTR_ZMAX = POR_MEAN + 4 * POR_STDEV
LTAIL, LTPAR = 1, BACKTR_ZMIN
UTAIL, UTPAR = 1, BACKTR_ZMAX

# --- Monte Carlo physical-units kriging variance (project decision
# 2026-09-14, see task description for the "Truth & predictions" figure
# rework) ---------------------------------------------------------------
# geostats.backtr_value only back-transforms a single scalar at a time, so
# calling it directly inside a per-cell x per-MC-draw Python loop
# (NX*NY*N_MC calls) would be far too slow. Since this project always uses
# LTAIL=1, UTAIL=1 (linear tail extrapolation, see BACKTR_ZMIN/ZMAX/LTAIL/
# UTAIL above), geostats.dpowint is always called with cpow=1.0 (pure linear
# interpolation) -- backtr_value_vectorized below is a numpy-vectorized
# reimplementation valid ONLY for that ltail=1/utail=1 case, verified
# against the reference geostats.backtr_value in main() (see
# "backtr_vectorized_max_abs_diff_vs_reference" in the manifest).
KRIGING_VAR_MC_SEED = 80  # does not collide with TRUTH_SEED=101, SAMPLE_SEED=20,
# CV_SEED=40, BOOTSTRAP_SEED=30, GP_RANDOM_STATE=50, GP_SAMPLE_SEED=55,
# SGS_SEED=60, SGS_JITTER_SEED=70 used elsewhere in this comparison.
BACKTR_VALIDATION_SEED = 85  # separate seed, only used to pick the random
# kmap_ns test values compared against the reference geostats.backtr_value
# below -- does not affect any saved result array, kept fixed/recorded
# anyway for full reproducibility of the validation step itself.
N_BACKTR_VALIDATION_SAMPLES = 500  # >= the 200 minimum specified in the task
N_MC = 5000  # Monte Carlo draws per grid cell


def _gcum_vectorized(x):
    """Numpy-vectorized standard normal CDF, written to be numerically
    identical to ``geostatspy.geostats.gcum`` (validated in main() below via
    comparison against the scalar reference).

    NOTE: this is deliberately NOT ``scipy.stats.norm.cdf`` -- geostats.gcum
    is itself only a polynomial *approximation* to the normal CDF (accurate
    to ~5 decimal places, per its own docstring), and geostats.backtr_value
    calls THIS approximation, not the exact CDF. Substituting scipy's exact
    CDF here would introduce a small but real discrepancy against the
    reference this function must reproduce for the mandatory validation
    below to pass at a tight tolerance. (Confirmed close to double-precision
    agreement with geostats.gcum in ad hoc testing while developing this
    function, well within the ~1e-5 accuracy geostats.gcum itself claims.)
    """
    x = np.asarray(x, dtype=float)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.231_641_9 * z)
    poly = t * (
        0.319_381_53
        + t * (-0.356_563_782 + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429)))
    )
    e2 = np.where(z <= 6, np.exp(-z * z / 2.0) * 0.398_942_280_3, 0.0)
    gcum_pos = 1.0 - e2 * poly
    return np.where(x >= 0.0, gcum_pos, 1.0 - gcum_pos)


def backtr_value_vectorized(vrgs, vr, vrg, zmin, zmax, ltail, ltpar, utail, utpar):
    """Numpy-vectorized equivalent of ``geostatspy.geostats.backtr_value``,
    valid ONLY for ``ltail == 1 and utail == 1`` (linear tail extrapolation,
    ``cpow=1.0`` in GSLIB's ``dpowint``) -- the only case this project's
    kriging/SGS/RBF/GP back-transforms ever use. Raises ``NotImplementedError``
    for any other ltail/utail rather than silently returning a wrong answer.

    ``vrgs`` may be a scalar or an array of any shape; the return value has
    the same shape.

    Reproduces geostats.backtr_value's THREE distinct branches exactly
    (verified against the scalar reference in main(), not merely assumed
    equivalent):

    1. Interior (``vrg[0] < vrgs < vrg[-1]``): GSLIB's own ``backtr_value``
       source calls ``dlocate`` (whose ``bisect`` is applied to
       ``vrg[1:nt-1]`` -- i.e. excluding BOTH endpoints -- with the resulting
       0-indexed position then used, unmodified, as an index into the FULL
       ``vrg``/``vr`` arrays, then clamped to ``[1, nt-2]``) to find the
       bracketing table segment, then does a **plain linear interpolation
       directly in raw normal-score (``vrg``) space** between that segment's
       two table points -- it does NOT go through ``gcum`` at all for
       interior points. This is intentionally reproduced below via
       ``np.searchsorted(vrg[1:nt-1], vrgs, side="right")`` (``side="right"``
       matches Python's ``bisect.bisect`` == ``bisect_right``, which is what
       ``dlocate`` calls), clamped the same way, rather than via a single
       global ``np.interp(vrgs, vrg, vr)`` -- a naive full-table
       ``np.interp`` does NOT reproduce this because it uses the true
       bracketing segment (including segments touching index 0 or nt-1),
       whereas the reference's clamp silently discards the first and last
       table segments as valid brackets. This was confirmed empirically
       while implementing this function: naive ``np.interp`` disagreed with
       the reference by up to ~0.75 (physical units) for a handful of test
       points near the table's extremes, exactly the cells where this
       clamping quirk changes which segment is used.
    2. Lower tail (``vrgs <= vrg[0]``): linear interpolation in *gcum(vrgs)*
       (CDF) space between (0.0, zmin) and (gcum(vrg[0]), vr[0]).
    3. Upper tail (``vrgs >= vrg[-1]``): linear interpolation in *gcum(vrgs)*
       space between (gcum(vrg[-1]), vr[-1]) and (1.0, zmax).
    """
    if ltail != 1 or utail != 1:
        raise NotImplementedError(
            "backtr_value_vectorized only implements/validates the ltail=1, "
            f"utail=1 (linear tail) case used by this project; got "
            f"ltail={ltail}, utail={utail}."
        )
    vrgs = np.asarray(vrgs, dtype=float)
    nt = len(vr)

    # --- Interior: replicate geostats.dlocate's exact (quirky) indexing --
    sub = vrg[1 : nt - 1]
    j_raw = np.searchsorted(sub, vrgs, side="right")
    j = np.clip(j_raw, 1, nt - 2)
    x0, x1 = vrg[j], vrg[j + 1]
    y0, y1 = vr[j], vr[j + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        interior = y0 + (y1 - y0) * ((vrgs - x0) / (x1 - x0))

    # --- Tails: linear extrapolation in gcum (CDF) space ------------------
    cdfbt = _gcum_vectorized(vrgs)
    cdflo = _gcum_vectorized(vrg[0])
    lower = zmin + (vr[0] - zmin) * (cdfbt / cdflo)
    cdfhi = _gcum_vectorized(vrg[nt - 1])
    upper = vr[nt - 1] + (zmax - vr[nt - 1]) * ((cdfbt - cdfhi) / (1.0 - cdfhi))

    return np.where(vrgs <= vrg[0], lower, np.where(vrgs >= vrg[nt - 1], upper, interior))


EXPERIMENT_NAME = "kriging"
CODE_ENTRYPOINT = "src/experiments/kriging.py"


def main(
    truth_seed: int = TRUTH_SEED,
    sample_seed: int = SAMPLE_SEED,
    hmaj1: float = _COND_HMAJ1,
    hmin1: float = _COND_HMIN1,
    n_samples: int = N_SAMPLES,
):
    """Run the simple-kriging base-case pipeline.

    Defaults (``truth_seed=TRUTH_SEED``, ``sample_seed=SAMPLE_SEED``,
    ``hmaj1``/``hmin1`` = the base-case 300m range, ``n_samples=N_SAMPLES``)
    reproduce the exact base case. Passing a different ``hmaj1``/``hmin1``
    (equal, isotropic) is how the range axis (docs/experiment_context.md
    deliverable 2) varies the variogram range; passing a different
    ``n_samples`` is how the sample-density axis varies the conditioning
    sample count (each level drawn INDEPENDENTLY, not as a nested subset --
    see base_case_conditioning.get_conditioning_samples). In both cases every
    search/tuning constant below
    (NDMAX, RADIUS, BACKTR_ZMIN/ZMAX, etc.) stays fixed at its base-case
    value (one-factor-at-a-time -- do not vary those here).
    """
    t_start = time.time()

    truth, samples_df = get_base_case_conditioning_data(
        sample_seed=sample_seed,
        truth_seed=truth_seed,
        hmaj1=hmaj1,
        hmin1=hmin1,
        n_samples=n_samples,
    )
    n_actual_samples = len(samples_df)
    vario = build_vario(hmaj1=hmaj1, hmin1=hmin1)

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
    # docstring) -- vmap_ns is saved as-is, in normal-score units. This loop
    # is left unchanged (still the scalar geostats.backtr_value, not the new
    # vectorized version below) so kmap_physical always goes through the
    # reference (non-vectorized) implementation directly -- the vectorized
    # version is validated against this same reference call below before it
    # is trusted for the Monte Carlo variance step. NOTE: kmap_physical's
    # actual numeric values DO depend on BACKTR_ZMIN/BACKTR_ZMAX (see above
    # for their current value and history) since cells whose NS estimate
    # falls in either tail of the transform table are extrapolated out to
    # those bounds -- so this map is NOT expected to be bit-for-bit identical
    # across runs that used different BACKTR_ZMIN/ZMAX values (e.g. the
    # earlier +/-10 stdev pinned run results/raw/kriging/20260914T144604Z).
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

    # --- Validate backtr_value_vectorized against the reference scalar
    # geostats.backtr_value (mandatory, per task spec -- do not proceed on
    # mismatch) -----------------------------------------------------------
    # Draw N_BACKTR_VALIDATION_SAMPLES (>= 200) values from the ACTUAL
    # kmap_ns array (not synthetic values) -- these are exactly the kind of
    # value the Monte Carlo step below will feed through this function millions
    # of times, so validating on this array's own value range is the most
    # direct test of correctness for this run.
    validation_rng = np.random.default_rng(BACKTR_VALIDATION_SEED)
    validation_values = validation_rng.choice(
        kmap_ns.ravel(), size=N_BACKTR_VALIDATION_SAMPLES, replace=True
    )
    reference_backtr = np.array(
        [
            geostats.backtr_value(
                v, vr, vrg, zmin=BACKTR_ZMIN, zmax=BACKTR_ZMAX,
                ltail=LTAIL, ltpar=LTPAR, utail=UTAIL, utpar=UTPAR,
            )
            for v in validation_values
        ]
    )
    vectorized_backtr = backtr_value_vectorized(
        validation_values, vr, vrg, BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR
    )
    backtr_vectorized_max_abs_diff_vs_reference = float(
        np.max(np.abs(reference_backtr - vectorized_backtr))
    )
    if not np.allclose(reference_backtr, vectorized_backtr, rtol=1e-8, atol=1e-8):
        raise RuntimeError(
            "backtr_value_vectorized does not match the reference "
            "geostats.backtr_value within tolerance (max abs diff = "
            f"{backtr_vectorized_max_abs_diff_vs_reference}) -- refusing to "
            "proceed with the Monte Carlo variance step on an unvalidated "
            "back-transform. See backtr_value_vectorized's docstring for the "
            "known dlocate-indexing quirk this function must reproduce."
        )
    print(
        f"backtr_value_vectorized validated against reference geostats.backtr_value "
        f"on {N_BACKTR_VALIDATION_SAMPLES} samples drawn from kmap_ns "
        f"(max abs diff = {backtr_vectorized_max_abs_diff_vs_reference:.3e})."
    )

    # --- Monte Carlo physical-units kriging variance (project decision
    # 2026-09-14) -----------------------------------------------------------
    # This is an APPROXIMATION (see kriging_var_map_physical_mc_note below),
    # not an exact analytic back-transform of the variance: for each cell we
    # assume NPor ~ N(kmap_ns[cell], sqrt(vmap_ns[cell])) in normal-score
    # space (exactly the simple-kriging Gaussian posterior at that cell,
    # ignoring spatial correlation across cells -- fine for a per-cell
    # variance-only estimate), draw N_MC samples, back-transform every draw
    # to physical units via the validated vectorized function, and take the
    # per-cell sample variance of the back-transformed draws.
    t_mc_start = time.time()
    # vmap_ns can be a tiny negative number (~1e-15) at cells with essentially
    # zero kriging variance, from floating-point roundoff in kb2d -- clip to
    # 0 before sqrt (verified: only ever negative at magnitude ~1e-15, never a
    # real negative variance).
    std_ns_flat = np.sqrt(np.clip(vmap_ns, 0.0, None)).ravel()
    kmap_ns_flat = kmap_ns.ravel()
    n_cells = kmap_ns_flat.shape[0]

    mc_rng = np.random.default_rng(KRIGING_VAR_MC_SEED)
    # Shape (n_cells, N_MC): fully vectorized draw + back-transform, no
    # per-cell/per-draw Python loop.
    ns_draws = mc_rng.normal(
        loc=kmap_ns_flat[:, None], scale=std_ns_flat[:, None], size=(n_cells, N_MC)
    )
    physical_draws = backtr_value_vectorized(
        ns_draws, vr, vrg, BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR
    )
    kriging_var_map_physical_mc = physical_draws.var(axis=1, ddof=1).reshape(NY, NX)
    mc_seconds = time.time() - t_mc_start
    print(
        f"Monte Carlo physical-units variance: {n_cells} cells x {N_MC} draws "
        f"took {mc_seconds:.2f}s"
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
        ("kriging_var_map_physical_mc.npy", kriging_var_map_physical_mc),
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
    ax1.set_title(f"Truth (seed={truth_seed})")
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
        "truth_seed": truth_seed,
        "sample_seed": sample_seed,
        # hmaj1/hmin1 duplicated top-level (also present inside "variogram"
        # below) for range-axis lookup convenience (task instruction).
        "hmaj1": hmaj1,
        "hmin1": hmin1,
        "n_samples_requested": n_samples,
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
        "backtr_validation": {
            "seed": BACKTR_VALIDATION_SEED,
            "n_samples": N_BACKTR_VALIDATION_SAMPLES,
            "backtr_vectorized_max_abs_diff_vs_reference": backtr_vectorized_max_abs_diff_vs_reference,
        },
        "kriging_var_mc": {
            "seed": KRIGING_VAR_MC_SEED,
            "n_mc": N_MC,
            "backtr_vectorized_max_abs_diff_vs_reference": backtr_vectorized_max_abs_diff_vs_reference,
            "mc_seconds": mc_seconds,
        },
        "kriging_var_map_physical_mc_note": (
            "kriging_var_map_physical_mc.npy is a Monte Carlo APPROXIMATION, "
            "not an exact analytic back-transform of the kriging variance: "
            "for each cell it assumes NPor ~ N(kmap_ns[cell], "
            "sqrt(vmap_ns[cell])) independently in normal-score space (the "
            "simple-kriging Gaussian posterior at that cell, ignoring "
            "spatial correlation of the error across cells), draws N_MC "
            "samples, back-transforms them to physical porosity units via "
            "the validated backtr_value_vectorized, and takes the per-cell "
            "sample variance of the back-transformed draws. It is provided "
            "for visualization/comparison against SGS/RBF/GP physical-unit "
            "variance maps (project decision 2026-09-14), not as a "
            "replacement for kriging_var_map_ns.npy in any calibration-metric "
            "computation."
        ),
        "total_seconds": total_seconds,
    }

    manifest_path = save_result(
        experiment=EXPERIMENT_NAME,
        params=params,
        seed_or_seeds={
            "truth_seed": truth_seed,
            "sample_seed": sample_seed,
            "backtr_validation_seed": BACKTR_VALIDATION_SEED,
            "kriging_var_mc_seed": KRIGING_VAR_MC_SEED,
        },
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
    print(
        f"kriging_var_map_physical_mc range: "
        f"[{kriging_var_map_physical_mc.min():.4f}, {kriging_var_map_physical_mc.max():.4f}]"
    )
    print(f"kb2d took {kb2d_seconds:.2f}s, MC variance took {mc_seconds:.2f}s, total {total_seconds:.2f}s")

    return run_dir, manifest_path, samples_df


if __name__ == "__main__":
    main()
