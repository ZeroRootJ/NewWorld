"""Pre-axis-4 diagnostic: does the EXPERIMENTAL variogram estimated from a
sparse sample set recover the TRUE (model) anisotropic variogram used to
generate the ground truth?

This is NOT axis 4 itself (the 4-method kriging/SGS/RBF-bootstrap/GPR
comparison) -- it is the sanity check the user wants to run before axis 4,
because anisotropy is a "ratio + direction" parameter rather than the single
scalar range axis 1 varied, and the earlier axes (range/nugget) never fit a
variogram from samples at all: they fed kriging/SGS the EXACT truth-model
variogram dict directly (see base_case_conditioning.build_vario, reused
as-is, no gamv call). This script is the first place in the project that
actually computes an experimental (gamv) variogram and compares it against
the model (vmodel) curve. kriging.py / sgs.py / rbf_bootstrap.py / gp_mle.py
are not touched by this script.

Run with: .venv/Scripts/python.exe -m src.experiments.anisotropy_variogram_check

Design (user-confirmed 2026-10-16, do not re-litigate here)
-------------------------------------------------------------
- 3 anisotropy ratios, MINOR range held fixed at 300 m, MAJOR range scaled by
  the ratio (minor range is NOT the base case's own isotropic range by
  coincidence here -- it is pinned to 300 m by this axis's own design, which
  happens to equal base_case.HMAJ1/HMIN1):
      "1:1" -> hmaj1=300, hmin1=300
      "2:1" -> hmaj1=600, hmin1=300
      "3:1" -> hmaj1=900, hmin1=300
- nugget = 0.1 for ALL THREE ratios (one-factor-at-a-time: only anisotropy
  varies). THIS IS A NEW, LOCAL CONSTANT (NUG_ANISO below), deliberately NOT
  base_case.NUG (0.05) -- the two axes are independent and must not drift
  into each other by accident. cc1 = 1.0 - 0.1 = 0.9 by this project's
  unit-sill invariant (nug + cc1 = 1.0 on standard-normal space; see
  base_case_conditioning.build_vario's docstring, reused here, not
  reimplemented).
- azi1 = 0 degrees (axis-aligned), nst=1, it1=1 (spherical) -- same structure
  as the base case, reusing base_case.AZI1 (asserted below to equal 0.0) so
  the "axis-aligned" claim is enforced, not just stated.
- Grid/distribution: reused unmodified from src.experiments.base_case (NX,
  NY, XSIZ, YSIZ, XMIN, XMAX, YMIN, YMAX, XMN, YMN, POR_MEAN, POR_STDEV).
- Ground truth: src.truth_model.make_porosity_truth via
  base_case_conditioning.get_base_case_truth (same affine-corrected
  unconditional-SGS pattern as every other axis in this project).
- 10 truth realizations (TRUTH_SEEDS = 201..210) x 2 sample draws per
  realization (SAMPLE_SEED_DRAWS = [2001, 2002], the SAME 2 seeds reused at
  every truth_seed and every ratio, so "draw1"/"draw2" mean the same thing
  everywhere -- the sample_replicate_axis.py pattern of reusing one small set
  of seeds across all levels, applied here to 2 draws instead of 10
  replicates). Both seed blocks are asserted at import time not to collide
  with any seed already pinned elsewhere in this project (base_case.SEEDS,
  base_case_conditioning.SAMPLE_SEED/TRUTH_SEED,
  sample_replicate_axis.SAMPLE_SEED_REPLICATES) -- see the assertion block
  below.
- Sample density: 5% (N_SAMPLES = base_case_conditioning.N_SAMPLES = 125
  requested, imported rather than redeclared -- single source of truth for
  the project's "5%" density level), margin_frac = 0.05
  (base_case_conditioning.MARGIN_FRAC, likewise imported), drawn via
  src.sampling.random_interior_samples (through
  base_case_conditioning.get_conditioning_samples, the project's single entry
  point for this).
- Direction is ASSUMED KNOWN (user decision): only azi=0 (major) and
  azi=90 (minor, "azi1 + 90.0" per docs/geostatspy_conventions.md section 3)
  are computed. No azimuth scan/search.
- Variogram space: the model variogram (nug=0.1, cc1=0.9) is defined on
  standard-normal UNIT-SILL space. The truth field (and hence the samples
  drawn from it) is in PHYSICAL porosity units after GSLIB.affine(mean,
  stdev) -- see src/truth_model.py. Comparing a physical-unit experimental
  variogram against a unit-sill model variogram would not be a fair
  comparison (different sills entirely), so every sample set is
  normal-score-transformed via the project's corrected wrapper
  (src/nscore.py, NOT geostats.nscore directly -- see
  docs/geostatspy_conventions.md section 3.1 for why geostats.nscore is
  banned in this project) before any gamv call. gamv's isill=1 then
  standardizes by the SAMPLE variance of the normal scores (which is close
  to, but not exactly, 1.0 for a finite sample -- see geostats.gamv's own
  "Standardize sill to one by dividing ... by the variance" step), not by an
  assumed/forced value of 1.0.

KNOWN CAVEAT -- FLAG FOR THE USER, NOT RESOLVED HERE (GSLIB azimuth
convention vs. the task note's "X axis" wording)
---------------------------------------------------------------------------
The task note that specified this design states "azi1=0 -> major axis is the
X direction". This was CHECKED EMPIRICALLY against the installed
geostatspy 0.0.79 (geostats.setup_rotmat / geostats.variogram_loop both use
``azmuth = (90 - azi) * pi/180``, i.e. the GSLIB convention where azimuth 0
points along +Y ("north") and azimuth 90 points along +X ("east")) and found
to be the OPPOSITE of what the note says: with azi1=0, hmaj1=900,
hmin1=300, ``geostats.vmodel(nlag, xlag, azm=0, vario)`` reaches its sill at
h~900 (slow -> MAJOR direction) while ``azm=90`` reaches the sill at h~300
(fast -> MINOR direction). I.e. with azi1=0 the MAJOR axis of this model lies
along Y, not X. This is the single most common GSLIB azimuth gotcha
(azimuth is a compass bearing from north/+Y, not a mathematical angle from
+X) and is almost certainly just a mislabeling in the task note's prose, not
a request to actually orient the model along X. It does NOT affect anything
this script computes: the truth field is generated with azi1=0, and the
experimental variogram is computed at azm=0 ("major", matching azi1 by
definition) and azm=90 ("minor", azi1+90), exactly mirroring what
geostats.vmodel/geostats.sgsim/geostats.gamv already do internally with that
same azi1 -- so model and experimental curves are compared in a mutually
consistent convention regardless of which physical compass direction you
call "X". Still, since the paper's prose may eventually say "major axis
along X", this mismatch should be confirmed with the user before axis 4
reuses this note's language.

gamv bin-indexing subtlety (why bin 0 and the last bin are dropped)
---------------------------------------------------------------------------
Read directly out of geostatspy.geostats.variogram_loop's source (the
numba-jitted function gamv calls): for il in range(1, nlag+1), each bin's
NOMINAL center is ``(il - 1) * xlag`` -- i.e. gamv's bin index 1 is centered
at h=0 (same nominal location as the "h<=EPSLON self-pair" bin index 0), bin
index 2 at h=xlag, ..., bin index nlag at h=(nlag-1)*xlag. The array gamv
returns has length nlag+2; index 0 holds the degenerate zero-separation
"self pair" bin (meaningless as a variogram point) and index nlag+1 is never
written by the loop (it stays at its zero-initialized value) -- both are
sliced off below (``dis[1:-1]`` etc.) before anything is plotted or stored.
This was verified empirically (not assumed) against the installed library;
see the coder's handback report for the numbers. Because each kept point
still carries its OWN measured mean pair distance (``dis``), not the nominal
bin center, no further index arithmetic vs. ``geostats.vmodel`` (whose
``h[k] = k * xlag`` convention does NOT line up 1-for-1 with gamv's bins) is
needed: the model curve and the experimental points are plotted against
their own correct lag (x) values on a shared axis, not matched up by array
index.

gamv/vmodel lag parameters (named constants below) -- rationale
---------------------------------------------------------------------------
- GAMV_LAG_DIST = 60.0 m, GAMV_NLAG = 16 -> covers lags 0..960 m. The
  interior sample domain (margin_frac=0.05 on a 1000 m x 1000 m box) spans
  roughly [50, 950] along each axis, i.e. ~900 m -- so 16 x 60 m = 960 m lags
  comfortably reaches past the largest range tested (ratio 3:1 -> major
  range 900 m) and shows the model curve's plateau at the sill, while 60 m
  is small enough (less than the ~82 m mean nearest-neighbour spacing of 121
  samples spread over ~900 m x 900 m) to resolve structure near the origin.
- GAMV_LAG_TOL = 30.0 m = 0.5 * GAMV_LAG_DIST, the standard GSLIB default
  when xltol is left unset (gamv's own fallback is ``0.5 * xlag``), applied
  explicitly per this project's "no implicit defaults" convention
  (docs/geostatspy_conventions.md section 6, "Black-box 금지").
- GAMV_ATOL = 22.5 degrees, the conventional GSLIB half-angle tolerance used
  when two roughly-orthogonal principal directions are being resolved
  (Deutsch & Journel's GSLIB default directional tolerance; distinct from
  the isotropic QC gamv call in sgs.py, which deliberately uses atol=90,
  i.e. omnidirectional, because that check has no anisotropy to resolve). An
  isotropic check (ratio "1:1") run through this SAME 22.5-degree tolerance
  is itself a useful internal check: both directions should recover
  statistically the same (noisy) curve.
- GAMV_BANDH = 9999.0 (effectively unrestricted band width), matching this
  project's one existing gamv precedent (sgs.py's QC_BANDH).
- GAMV_ISILL = 1 (standardize to the sample variance of the normal-scored
  values), matching this project's unit-sill variogram convention.
Empirically verified (not assumed) on one (ratio, truth_seed, draw) cell
that these settings give non-degenerate pair counts (tens to a few hundred
pairs per bin, never systematically zero) and qualitatively recover the
correct ordering (major variogram rises slower than minor) -- see the
coder's handback report.

Raw / processed / figure outputs
---------------------------------------------------------------------------
- Raw (results/raw/anisotropy_variogram_check/<timestamp>/, via
  src.io.make_run_dir/save_result): the 30 truth fields (3 ratios x 10
  truth_seeds, shared across both sample draws) and the 60 sample draws (3
  ratios x 10 truth_seeds x 2 draws), each sample CSV carrying both the
  physical ("Por") and normal-scored ("NPor") columns actually fed to gamv.
  manifest.json's params carry every named constant above plus the
  per-(ratio, truth_seed, draw) nscore correction audit record (see
  src/nscore.py) -- this script does not call geostats.nscore directly.
- Processed (results/processed/anisotropy_variogram_check/
  variogram_comparison.csv): tidy long format, one row per (ratio_label,
  direction, source, lag) point. Columns: ratio_label ("1:1"/"2:1"/"3:1"),
  hmaj1, hmin1, truth_seed, sample_draw_seed, direction ("major"/"minor"),
  source ("experimental"/"model"), lag, gamma, npair.
  MODEL rows have truth_seed=NaN and sample_draw_seed=NaN (the model curve
  does not depend on either -- it is the SAME curve for a given
  (ratio_label, direction) regardless of realization or draw) and npair=NaN
  (pair count is not a model-curve concept). EXPERIMENTAL rows always have a
  concrete truth_seed/sample_draw_seed/npair. This mixed-NaN design means
  truth_seed/sample_draw_seed/npair are float columns, not int, when read
  back with pandas -- documented here rather than discovered by surprise.
- Figures (results/figures/anisotropy_variogram_check/
  variogram_qc_<1to1|2to1|3to1>.pdf): one 2-panel (major, minor) figure per
  ratio, vector PDF, built per the project's FIGURE_GUIDELINES file (Arial,
  7pt text, embedded fonts, double-column 190mm width given the panel
  density). Each panel overlays all 20 experimental curves (10 truth_seed x
  2 draws) at alpha~0.3 against the single bold model curve.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import geostatspy.geostats as geostats

from src.experiments.base_case import (
    AZI1,
    NX,
    NY,
    POR_MEAN,
    POR_STDEV,
    XMAX,
    XMIN,
    XMN,
    XSIZ,
    YMAX,
    YMIN,
    YMN,
    YSIZ,
)
from src.experiments.base_case import SEEDS as BASE_CASE_TRUTH_SEEDS
from src.experiments.base_case_conditioning import (
    MARGIN_FRAC,
    N_SAMPLES,
    SAMPLE_SEED as BASE_SAMPLE_SEED,
    TRUTH_SEED as BASE_TRUTH_SEED,
    VCOL,
    build_vario,
    get_base_case_truth,
    get_conditioning_samples,
)
from src.experiments.sample_replicate_axis import SAMPLE_SEED_REPLICATES
from src.io import make_run_dir, save_result
from src.nscore import nscore as nscore_corrected

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "anisotropy_variogram_check"
FIGURES_DIR = _REPO_ROOT / "results" / "figures" / "anisotropy_variogram_check"

EXPERIMENT_NAME = "anisotropy_variogram_check"
CODE_ENTRYPOINT = "src/experiments/anisotropy_variogram_check.py"

# --- Variogram parameters for THIS axis's checks (NOT base_case.NUG/CC1) --
# nugget=0.1 (cc1=0.9) is a LOCAL constant for the anisotropy axis, held
# fixed across all 3 ratio levels (one-factor-at-a-time) -- see module
# docstring. Deliberately distinct from base_case.NUG (0.05).
NUG_ANISO = 0.1
CC1_ANISO = 1.0 - NUG_ANISO
if abs((NUG_ANISO + CC1_ANISO) - 1.0) > 1e-12:
    raise ValueError(
        f"NUG_ANISO ({NUG_ANISO}) + CC1_ANISO ({CC1_ANISO}) != 1.0; this axis's "
        "variogram must keep the project's unit-sill invariant (see module docstring)."
    )

if AZI1 != 0.0:
    raise AssertionError(
        f"base_case.AZI1 = {AZI1!r}, expected 0.0 -- this script's 'axis-aligned, "
        "azi1=0' design (see module docstring) assumes the project's shared AZI1 "
        "constant is 0.0; it has apparently changed elsewhere in the project."
    )
AZI1_ANISO = AZI1  # == 0.0, imported (not redeclared) so this can never silently drift

MINOR_RANGE_M = 300.0  # held fixed across all 3 ratio levels (user decision 2026-10-16)
RATIO_ORDER = ["1:1", "2:1", "3:1"]
RATIO_LEVELS = {
    "1:1": {"hmaj1": 300.0, "hmin1": MINOR_RANGE_M, "filename_label": "1to1"},
    "2:1": {"hmaj1": 600.0, "hmin1": MINOR_RANGE_M, "filename_label": "2to1"},
    "3:1": {"hmaj1": 900.0, "hmin1": MINOR_RANGE_M, "filename_label": "3to1"},
}

# --- Ground-truth realizations and sample draws ---------------------------
# 10 truth realizations, a fresh seed range not used anywhere else in the
# project (asserted below, not merely claimed).
TRUTH_SEEDS = list(range(201, 211))  # 201..210 inclusive
# 2 sample draws, SAME 2 seeds reused at every truth_seed and every ratio
# (sample_replicate_axis.py's "same seeds reused across levels" pattern,
# applied here across ratios/truth_seeds instead of density levels).
DRAW_LABELS = ["draw1", "draw2"]
SAMPLE_SEED_DRAWS = [2001, 2002]

_EXISTING_PINNED_SEEDS = (
    set(BASE_CASE_TRUTH_SEEDS) | {BASE_SAMPLE_SEED, BASE_TRUTH_SEED} | set(SAMPLE_SEED_REPLICATES)
)
_NEW_SEEDS = set(TRUTH_SEEDS) | set(SAMPLE_SEED_DRAWS)
_seed_collision = _EXISTING_PINNED_SEEDS & _NEW_SEEDS
if _seed_collision:
    raise ValueError(
        f"TRUTH_SEEDS/SAMPLE_SEED_DRAWS collide with already-pinned project seed(s) "
        f"{sorted(_seed_collision)} (base_case.SEEDS, base_case_conditioning.SAMPLE_SEED/"
        "TRUTH_SEED, sample_replicate_axis.SAMPLE_SEED_REPLICATES) -- refusing to "
        "silently reuse/overwrite an existing pinned draw's seed."
    )

# Sample density: 5% level, imported (not redeclared) from
# base_case_conditioning -- single source of truth for the project's "5%"
# sample count. MARGIN_FRAC is likewise imported and used implicitly inside
# get_conditioning_samples.
N_SAMPLES_ANISO = N_SAMPLES

# --- gamv / vmodel lag parameters (see module docstring for rationale) ----
GAMV_TMIN, GAMV_TMAX = -9999.0, 9999.0  # no value trimming
GAMV_LAG_DIST = 60.0
GAMV_LAG_TOL = 30.0
GAMV_NLAG = 16
GAMV_ATOL = 22.5
GAMV_BANDH = 9999.0
GAMV_ISILL = 1

MAJOR_AZI = AZI1_ANISO
MINOR_AZI = AZI1_ANISO + 90.0  # docs/geostatspy_conventions.md section 3 convention
DIRECTIONS = [("major", MAJOR_AZI), ("minor", MINOR_AZI)]

NSCOL = "NPor"  # normal-scored porosity column fed to gamv

COMPARISON_COLUMNS = [
    "ratio_label",
    "hmaj1",
    "hmin1",
    "truth_seed",
    "sample_draw_seed",
    "direction",
    "source",
    "lag",
    "gamma",
    "npair",
]


def model_variogram_rows(ratio_label: str, hmaj1: float, hmin1: float) -> list:
    """The single true model variogram curve for (ratio_label, direction),
    evaluated at GAMV_NLAG+1 lags (0, GAMV_LAG_DIST, ..., GAMV_NLAG *
    GAMV_LAG_DIST) via geostats.vmodel. Independent of truth_seed/draw --
    callers must not call this once per (truth_seed, draw)."""
    vario = build_vario(hmaj1=hmaj1, hmin1=hmin1, nug=NUG_ANISO)
    rows = []
    for direction_name, azi in DIRECTIONS:
        _, h, gam, _, _ = geostats.vmodel(GAMV_NLAG, GAMV_LAG_DIST, azi, vario)
        for lag_h, gamma_v in zip(h, gam):
            rows.append(
                {
                    "ratio_label": ratio_label,
                    "hmaj1": hmaj1,
                    "hmin1": hmin1,
                    "truth_seed": np.nan,
                    "sample_draw_seed": np.nan,
                    "direction": direction_name,
                    "source": "model",
                    "lag": float(lag_h),
                    "gamma": float(gamma_v),
                    "npair": np.nan,
                }
            )
    return rows


def experimental_variogram_rows(
    samples_ns: pd.DataFrame,
    ratio_label: str,
    hmaj1: float,
    hmin1: float,
    truth_seed: int,
    draw_seed: int,
) -> list:
    """gamv-computed experimental variogram for one (ratio, truth_seed, draw),
    both directions. ``samples_ns`` must already carry the normal-scored
    column NSCOL (see module docstring -- gamv is never called on raw
    physical porosity here)."""
    rows = []
    for direction_name, azi in DIRECTIONS:
        dis, gamma, npair = geostats.gamv(
            samples_ns,
            "X",
            "Y",
            NSCOL,
            GAMV_TMIN,
            GAMV_TMAX,
            GAMV_LAG_DIST,
            GAMV_LAG_TOL,
            GAMV_NLAG,
            azi,
            GAMV_ATOL,
            GAMV_BANDH,
            GAMV_ISILL,
        )
        # Drop index 0 (degenerate zero-separation "self pair" bin) and the
        # trailing index (GAMV_NLAG + 1, structurally never filled by this
        # GSLIB port's variogram_loop) -- see module docstring.
        for lag_h, gamma_v, n_v in zip(dis[1:-1], gamma[1:-1], npair[1:-1]):
            rows.append(
                {
                    "ratio_label": ratio_label,
                    "hmaj1": hmaj1,
                    "hmin1": hmin1,
                    "truth_seed": truth_seed,
                    "sample_draw_seed": draw_seed,
                    "direction": direction_name,
                    "source": "experimental",
                    "lag": float(lag_h),
                    "gamma": float(gamma_v),
                    "npair": float(n_v),
                }
            )
    return rows


def build_params(nscore_corrections_by_combo: dict) -> dict:
    return {
        "grid": {
            "nx": NX,
            "ny": NY,
            "xsiz": XSIZ,
            "ysiz": YSIZ,
            "xmin": XMIN,
            "xmax": XMAX,
            "ymin": YMIN,
            "ymax": YMAX,
            "xmn": XMN,
            "ymn": YMN,
        },
        "porosity_distribution": {"mean": POR_MEAN, "stdev": POR_STDEV},
        "variogram": {
            "nug": NUG_ANISO,
            "nst": 1,
            "it1": 1,
            "cc1": CC1_ANISO,
            "azi1": AZI1_ANISO,
            "minor_range_m": MINOR_RANGE_M,
            "ratio_levels": RATIO_LEVELS,
            "note": (
                "nug=0.1/cc1=0.9 is a LOCAL constant for this axis, distinct from "
                "base_case.NUG=0.05/CC1=0.95 -- see module docstring."
            ),
        },
        "truth_seeds": TRUTH_SEEDS,
        "sample_seed_draws": dict(zip(DRAW_LABELS, SAMPLE_SEED_DRAWS)),
        "n_samples_requested": N_SAMPLES_ANISO,
        "margin_frac": MARGIN_FRAC,
        "directions": {"major_azi": MAJOR_AZI, "minor_azi": MINOR_AZI},
        "gamv_params": {
            "tmin": GAMV_TMIN,
            "tmax": GAMV_TMAX,
            "lag_dist": GAMV_LAG_DIST,
            "lag_tol": GAMV_LAG_TOL,
            "nlag": GAMV_NLAG,
            "atol": GAMV_ATOL,
            "bandh": GAMV_BANDH,
            "isill": GAMV_ISILL,
        },
        "azimuth_convention_caveat": (
            "Empirically verified against installed geostatspy 0.0.79: with azi1=0, "
            "the MAJOR range axis lies along +Y (GSLIB compass-bearing convention), "
            "NOT +X. See module docstring 'KNOWN CAVEAT' section. Does not affect any "
            "computation here (model and experimental curves use the same azi1/azi1+90 "
            "convention throughout), but the original task note's 'major axis = X "
            "direction' phrasing should be confirmed with the user before axis 4 reuses it."
        ),
        "nscore_corrections_by_combo": nscore_corrections_by_combo,
    }


def make_figures(comparison_df: pd.DataFrame) -> list:
    """One 2-panel (major, minor) figure per ratio level: the single model
    curve (bold) vs. all 20 experimental curves (alpha~0.3). Vector PDF,
    styled per the project's FIGURE_GUIDELINES file (Arial, 7pt text,
    embedded TrueType fonts -- docs/figure_guidelines.md does not exist yet
    in this repo, so this follows the repo-root FIGURE_GUIDELINES file,
    which this user's standing preference applies to all figures)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.titlesize": 7,
            "figure.titlesize": 8,
            "pdf.fonttype": 42,  # embed as TrueType, not Type 3 (FIGURE_GUIDELINES 2.2)
        }
    )

    fig_width_mm = 190.0  # double-column (FIGURE_GUIDELINES 2.1): dense 2-panel, 20-curve overlay
    fig_width_in = fig_width_mm / 25.4
    fig_height_in = fig_width_in * 0.42

    # Shared y-axis limit across every panel/figure of this script, computed
    # from the full dataset (not hand-picked), so the 3 ratios are visually
    # comparable on the same scale.
    y_max = float(comparison_df["gamma"].max()) * 1.05
    x_max = GAMV_NLAG * GAMV_LAG_DIST

    paths = []
    for ratio_label in RATIO_ORDER:
        info = RATIO_LEVELS[ratio_label]
        fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in))
        legend_handles = None
        for ax, (direction_name, _) in zip(axes, DIRECTIONS):
            sub = comparison_df[
                (comparison_df["ratio_label"] == ratio_label)
                & (comparison_df["direction"] == direction_name)
            ]
            exp = sub[sub["source"] == "experimental"]
            mod = sub[sub["source"] == "model"].sort_values("lag")

            first = True
            for (_ts, _ds), g in exp.groupby(["truth_seed", "sample_draw_seed"]):
                g = g.sort_values("lag")
                ax.plot(
                    g["lag"],
                    g["gamma"],
                    color="tab:blue",
                    alpha=0.3,
                    linewidth=0.8,
                    label="Experimental (10 seeds x 2 draws)" if first else None,
                )
                first = False

            ax.plot(mod["lag"], mod["gamma"], color="black", linewidth=1.2, label="Model (true)")
            ax.axhline(1.0, color="grey", linewidth=0.4, linestyle=":")
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Lag distance h (m)")
            ax.set_ylabel(r"Semivariogram $\gamma(h)$ (unit sill)")
            ax.set_title(f"{direction_name} (azi={dict(DIRECTIONS)[direction_name]:g} deg)")
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        # One shared legend BELOW both panels (not inside either axes), so it
        # never overlaps the 20 overlaid experimental curves -- FIGURE_GUIDELINES
        # 2.4 ("ensure legend does not cover data").
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.04),
        )
        fig.suptitle(
            f"Anisotropy ratio {ratio_label} -- hmaj1={info['hmaj1']:g} m, "
            f"hmin1={info['hmin1']:g} m, nug={NUG_ANISO:g}"
        )
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        out_path = FIGURES_DIR / f"variogram_qc_{info['filename_label']}.pdf"
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)
    return paths


def main():
    run_dir = make_run_dir(EXPERIMENT_NAME)
    output_files = []
    all_rows = []
    nscore_corrections_by_combo = {}

    print("Model variogram curves (3 ratios x 2 directions)...")
    for ratio_label in RATIO_ORDER:
        info = RATIO_LEVELS[ratio_label]
        all_rows.extend(model_variogram_rows(ratio_label, info["hmaj1"], info["hmin1"]))

    print("Truth realizations, sample draws, and experimental variograms...")
    for ratio_label in RATIO_ORDER:
        info = RATIO_LEVELS[ratio_label]
        hmaj1, hmin1, fname_label = info["hmaj1"], info["hmin1"], info["filename_label"]
        for truth_seed in TRUTH_SEEDS:
            truth = get_base_case_truth(
                truth_seed=truth_seed, hmaj1=hmaj1, hmin1=hmin1, nug=NUG_ANISO
            )
            truth_fname = f"truth_{fname_label}_seed{truth_seed}.npy"
            np.save(run_dir / truth_fname, truth)
            output_files.append(truth_fname)

            for draw_label, draw_seed in zip(DRAW_LABELS, SAMPLE_SEED_DRAWS):
                samples = get_conditioning_samples(
                    truth, sample_seed=draw_seed, n_samples=N_SAMPLES_ANISO
                )
                ns, _vr, _vrg, corrections = nscore_corrected(samples, VCOL, verbose=False)
                samples = samples.copy()
                samples[NSCOL] = ns

                combo_key = f"{ratio_label}|truth_seed={truth_seed}|{draw_label}"
                nscore_corrections_by_combo[combo_key] = corrections

                samples_fname = f"samples_{fname_label}_seed{truth_seed}_{draw_label}.csv"
                samples.to_csv(run_dir / samples_fname, index=False)
                output_files.append(samples_fname)

                all_rows.extend(
                    experimental_variogram_rows(
                        samples, ratio_label, hmaj1, hmin1, truth_seed, draw_seed
                    )
                )

        print(f"  ratio {ratio_label} (hmaj1={hmaj1:g}, hmin1={hmin1:g}) done")

    comparison_df = pd.DataFrame(all_rows)[COMPARISON_COLUMNS]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / "variogram_comparison.csv"
    comparison_df.to_csv(processed_path, index=False)

    manifest_path = save_result(
        experiment=EXPERIMENT_NAME,
        params=build_params(nscore_corrections_by_combo),
        seed_or_seeds=sorted(_NEW_SEEDS),
        run_dir=run_dir,
        code_entrypoint=CODE_ENTRYPOINT,
        output_files=output_files,
    )

    print(f"\nRun directory: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"variogram_comparison.csv: {processed_path} ({len(comparison_df)} rows)")

    n_corrected_total = sum(
        c["n_corrected"] for c in nscore_corrections_by_combo.values()
    )
    print(
        f"nscore off-by-one repairs applied across all {len(nscore_corrections_by_combo)} "
        f"(ratio, truth_seed, draw) combos: {n_corrected_total} total"
    )

    figure_paths = make_figures(comparison_df)
    for p in figure_paths:
        print(f"Figure: {p}")

    return run_dir, manifest_path, processed_path, figure_paths


if __name__ == "__main__":
    main()
