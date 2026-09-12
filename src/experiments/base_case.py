"""Base case: reproduce the reference single ground-truth porosity model
within the new codebase (Deliverable Order step 1, docs/experiment_context.md
section 7).

Base case parameters (named constants below, per docs/geostatspy_conventions.md
"black-box 금지" principle -- agreed with the user, do not re-litigate here):

- Grid: nx=ny=50, xsiz=ysiz=20.0 m, domain 1000m x 1000m (xmin=ymin=0,
  xmax=ymax=1000, xmn=ymn=10.0)
- Porosity target distribution: mean=15.0, stdev=3.0
- Variogram: isotropic, spherical (it1=1), nugget=0.05, cc1=0.95
  (nug + cc1 = 1.0 sill on standard-normal space, before the affine
  correction), range (hmaj1 = hmin1) = 300 m
- Ground truth: 5 realizations, seeds 101-105, unconditional simulation.

This script produces only the 5 exhaustive truth maps (50x50 each) and their
QC statistics. Sample extraction (e.g. regular_interior_samples) is
deliberately NOT part of this generator -- per project decision 2026-09-12,
sampling will be implemented separately per axis (see src/sampling.py).

Run with: .venv/Scripts/python.exe -m src.experiments.base_case
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geostatspy.GSLIB as GSLIB

from src.io import make_run_dir, save_result
from src.truth_model import make_porosity_truth

# --- Grid parameters --------------------------------------------------
NX, NY = 50, 50
XSIZ, YSIZ = 20.0, 20.0
XMIN, XMAX = 0.0, 1000.0
YMIN, YMAX = 0.0, 1000.0
XMN, YMN = 10.0, 10.0  # first-cell centroid, GSLIB convention (xmn = xmin + 0.5*xsiz)

# --- Target univariate distribution ------------------------------------
POR_MEAN = 15.0
POR_STDEV = 3.0

# --- Variogram parameters (isotropic base case) -------------------------
NUG = 0.05
IT1 = 1  # spherical
CC1 = 0.95  # nug + cc1 = 1.0 sill on standard-normal space
AZI1 = 0.0  # irrelevant for isotropic case, kept explicit
HMAJ1 = 300.0
HMIN1 = 300.0  # == HMAJ1 -> isotropic

# --- Ground truth realizations --------------------------------------------
SEEDS = [101, 102, 103, 104, 105]

EXPERIMENT_NAME = "base_case"
CODE_ENTRYPOINT = "src/experiments/base_case.py"


def build_params() -> dict:
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
            "nug": NUG,
            "nst": 1,
            "it1": IT1,
            "cc1": CC1,
            "azi1": AZI1,
            "hmaj1": HMAJ1,
            "hmin1": HMIN1,
        },
        "seeds": SEEDS,
    }


def main():
    vario = GSLIB.make_variogram(
        nug=NUG, nst=1, it1=IT1, cc1=CC1, azi1=AZI1, hmaj1=HMAJ1, hmin1=HMIN1
    )

    truths = {}
    qc_stats = []

    for seed in SEEDS:
        truth = make_porosity_truth(
            nx=NX,
            ny=NY,
            xsiz=XSIZ,
            ysiz=YSIZ,
            xmn=XMN,
            ymn=YMN,
            vario=vario,
            mean=POR_MEAN,
            stdev=POR_STDEV,
            seed=seed,
        )
        truths[seed] = truth

        qc_stats.append(
            {
                "seed": seed,
                "truth_mean": float(np.mean(truth)),
                "truth_stdev": float(np.std(truth)),
            }
        )

    # --- Save outputs --------------------------------------------------
    run_dir = make_run_dir(EXPERIMENT_NAME)
    output_files = []

    for seed in SEEDS:
        fname = f"truth_seed{seed}.npy"
        np.save(run_dir / fname, truths[seed])
        output_files.append(fname)

    # --- QC figure: 5 realizations, truth maps only ---------------------
    vmin = POR_MEAN - 4 * POR_STDEV
    vmax = POR_MEAN + 4 * POR_STDEV
    fig = plt.figure(figsize=(15, 10))
    for i, seed in enumerate(SEEDS, start=1):
        plt.subplot(2, 3, i)
        GSLIB.pixelplt_st(
            truths[seed],
            XMIN,
            XMAX,
            YMIN,
            YMAX,
            XSIZ,
            vmin,
            vmax,
            f"Base Case Truth - seed {seed}",
            "X (m)",
            "Y (m)",
            "Porosity (%)",
            plt.cm.viridis,
        )
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.95, wspace=0.35, hspace=0.3)
    qc_fig_name = "qc_five_realizations.png"
    plt.savefig(run_dir / qc_fig_name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    output_files.append(qc_fig_name)

    manifest_path = save_result(
        experiment=EXPERIMENT_NAME,
        params=build_params(),
        seed_or_seeds=SEEDS,
        run_dir=run_dir,
        code_entrypoint=CODE_ENTRYPOINT,
        output_files=output_files,
    )

    print(f"Run directory: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"QC figure: {run_dir / qc_fig_name}")
    print()
    print("Per-realization QC stats (target mean=%.2f, stdev=%.2f):" % (POR_MEAN, POR_STDEV))
    for row in qc_stats:
        print(
            "  seed {seed}: truth mean={truth_mean:.4f} stdev={truth_stdev:.4f}".format(**row)
        )

    return run_dir, manifest_path


if __name__ == "__main__":
    main()
