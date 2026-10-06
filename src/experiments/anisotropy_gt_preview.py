"""Anisotropy axis (deliverable 4, docs/experiment_context.md) -- ground-truth
preview only: no method is run here.

Axis definition (user request 2026-10-06)
-----------------------------------------
Anisotropy ratio major:minor in {1:1, 1:2, 1:3}, with the "1" pinned to the
base-case range of 300 m: hmin1 = 300 m, hmaj1 = 300 * ratio (300 / 600 /
900 m). 1:1 is therefore the base-case isotropic variogram. Everything else
(nugget 0.05, unit sill, spherical, grid, distribution) is the base case.

Three ground-truth realizations per ratio, each with a RANDOM major-axis
azimuth (GSLIB convention: degrees clockwise from north, drawn uniformly on
[0, 180) from AZI_SEED). Realization r uses the SAME (truth_seed, azimuth)
pair at every ratio, so across rows only the ratio changes. Exception: at
1:1 the azimuth is meaningless, so the base-case AZI1 is used there (user
decision 2026-10-06) -- a different azimuth changes sgsim's rotation matrix in
floating point and would make the seed-101 1:1 field differ slightly from the
base-case truth; with AZI1 it is bit-identical (asserted in main()).

Run with: .venv/Scripts/python.exe -m src.experiments.anisotropy_gt_preview
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from src.experiments.base_case import XMIN, XMAX, YMIN, YMAX, NUG, IT1, AZI1
from src.experiments.base_case_conditioning import TRUTH_SEED, get_base_case_truth
from src.io import make_run_dir, save_result

EXPERIMENT = "anisotropy_gt_preview"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = _REPO_ROOT / "results" / "figures" / "anisotropy_axis"

HMIN1 = 300.0
RATIOS = [1, 2, 3]
TRUTH_SEEDS = [TRUTH_SEED, TRUTH_SEED + 1, TRUTH_SEED + 2]  # 101 = base case
AZI_SEED = 4
AZIMUTHS = [
    float(a) for a in np.round(
        np.random.RandomState(AZI_SEED).uniform(0.0, 180.0, len(TRUTH_SEEDS)), 1
    )
]

MM = 1.0 / 25.4


def main():
    run_dir = make_run_dir(EXPERIMENT)
    fields = {}
    records = []
    for ratio in RATIOS:
        for r, (seed, azi_drawn) in enumerate(zip(TRUTH_SEEDS, AZIMUTHS)):
            azi = AZI1 if ratio == 1 else azi_drawn
            truth = get_base_case_truth(
                truth_seed=seed, hmaj1=HMIN1 * ratio, hmin1=HMIN1, azi1=azi
            )
            fname = f"truth_ratio{ratio}_real{r}.npy"
            np.save(run_dir / fname, truth)
            fields[(ratio, r)] = truth
            records.append(
                {
                    "ratio": ratio,
                    "realization": r,
                    "truth_seed": seed,
                    "azi1": azi,
                    "hmaj1": HMIN1 * ratio,
                    "hmin1": HMIN1,
                    "file": fname,
                    "mean": float(truth.mean()),
                    "std": float(truth.std()),
                }
            )
            print(f"ratio 1:{ratio} real {r} seed {seed} azi {azi:5.1f}  "
                  f"mean {truth.mean():.3f} std {truth.std():.3f}")

    if not np.array_equal(fields[(1, 0)], get_base_case_truth()):
        raise AssertionError("1:1 seed-101 field is not bit-identical to the base case")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42})
    vmin = min(f.min() for f in fields.values())
    vmax = max(f.max() for f in fields.values())
    fig, axes = plt.subplots(3, 3, figsize=(190 * MM, 190 * MM), constrained_layout=True)
    xc, yc = 0.5 * (XMIN + XMAX), 0.5 * (YMIN + YMAX)
    for i, ratio in enumerate(RATIOS):
        for j, azi in enumerate(AZIMUTHS):
            ax = axes[i, j]
            im = ax.imshow(fields[(ratio, j)], extent=[XMIN, XMAX, YMIN, YMAX],
                           origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
            # Range ellipse at domain centre: semi-axes hmaj1 / hmin1, major axis
            # along the GSLIB azimuth (clockwise from north -> matplotlib angle
            # counter-clockwise from +x is 90 - azi).
            ax.add_patch(Ellipse((xc, yc), 2 * HMIN1 * ratio, 2 * HMIN1,
                                 angle=90.0 - azi, fill=False, ec="white", lw=1.0))
            azi_txt = "isotropic" if ratio == 1 else f"azimuth {azi:.1f}°"
            ax.set_title(f"1:{ratio}, {azi_txt}, seed {TRUTH_SEEDS[j]}")
            ax.set_xlim(XMIN, XMAX)
            ax.set_ylim(YMIN, YMAX)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.text(-0.30, 1.04, "ABCDEFGHI"[3 * i + j], transform=ax.transAxes,
                    fontweight="bold", fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, location="right")
    cb.set_label("Porosity (%)")
    outs = []
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"anisotropy_gt_realizations.{ext}"
        fig.savefig(p, dpi=300)
        outs.append(str(p.relative_to(_REPO_ROOT)))
    plt.close(fig)

    save_result(
        EXPERIMENT,
        params={
            "hmin1": HMIN1, "ratios": RATIOS, "nug": NUG, "it1": IT1,
            "truth_seeds": TRUTH_SEEDS, "azi_seed": AZI_SEED, "azimuths": AZIMUTHS,
            "azimuth_convention": "GSLIB, degrees clockwise from north, U[0,180)",
        },
        seed_or_seeds=TRUTH_SEEDS + [AZI_SEED],
        run_dir=run_dir,
        code_entrypoint="src/experiments/anisotropy_gt_preview.py",
        output_files=[rec["file"] for rec in records],
        extra={"realizations": records, "figures": outs},
    )
    print("run dir:", run_dir)


if __name__ == "__main__":
    main()
