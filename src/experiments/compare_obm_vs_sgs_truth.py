"""Descriptive comparison of the two ground-truth families on the same 50x50
grid, in the same units (porosity %):

* ``sgs``  : the 5 base-case SGS truths (seeds 101-105), regenerated here
  exactly as src/experiments/base_case.py does (its named constants and
  src.truth_model.make_porosity_truth are imported, not copied);
* ``obm``  : the ALLUVSIM-based ground truths (ResMill ChannelLayer,
  CB_JIGSAW), i.e. the ``*_truth_2d.npy`` files (affine-rescaled to mean 15 /
  std 3 per realization) of one results/raw/obm_ground_truth run. Only
  realizations with status "included" in that run's manifest are used.
* ``obm_mud``: the same CB_JIGSAW realizations generated with
  obm_benchmark/config/cb_jigsaw_mud_sd.yaml (mud facies FF/FFCH given ResMill's
  per-facies ``poro_sd``; facies and sand porosity identical to ``obm``), from
  one results/raw/obm_ground_truth_mud_sd run. Skipped if no such run exists.

Runs in the project .venv (py3.8, geostatspy). From the repository root::

    .venv/Scripts/python.exe -m src.experiments.compare_obm_vs_sgs_truth [--obm-run-dir DIR] [--obm-mud-run-dir DIR]

Default OBM run: the latest (timestamp-sorted) directory under
results/raw/obm_ground_truth/.

Writes to results/processed/obm_vs_sgs_truth/:
  summary_stats.csv    tidy long (family, seed, statistic, value)
  histogram_bins.csv   (family, bin_left, bin_right, density), pooled per family, common bins
  variogram.csv        (family, seed, direction, lag_cells, lag_m, gamma, gamma_std, npairs)
  example_fields.json  compact 50x50 fields for embedding (truth %, OBM facies)
  source_runs.json     provenance (OBM manifest path + its git commit, this run's git commit, rules)
  truth_families_comparison.png  side-by-side figure

Descriptive only -- no sampling, no estimation, no evaluation metrics.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geostatspy.GSLIB as GSLIB

from src.experiments import base_case as bc
from src.grid_variogram import grid_semivariogram
from src.io import _REPO_ROOT, get_git_commit, get_git_status
from src.truth_model import make_porosity_truth

OUT_DIR = _REPO_ROOT / "results" / "processed" / "obm_vs_sgs_truth"
OBM_RAW_ROOT = _REPO_ROOT / "results" / "raw" / "obm_ground_truth"
OBM_MUD_RAW_ROOT = _REPO_ROOT / "results" / "raw" / "obm_ground_truth_mud_sd"
OBM_FAMILIES = ("obm", "obm_mud")
CODE_ENTRYPOINT = "src/experiments/compare_obm_vs_sgs_truth.py"

NLAG = 25                    # lags 1..25 cells along x and y
HIST_BIN_WIDTH = 0.5         # %, common bin set for both families
EXAMPLE_SGS_SEEDS = [101, 102, 103]
EXAMPLE_OBM_SEEDS = [0, 10, 20]
PERCENTILES = [10, 50, 90]
NET_FACIES_MIN = 1           # facies >= 1 (CS, LV, LA, CH) = net; matches obm_benchmark


def latest_obm_run_dir(root: Path = OBM_RAW_ROOT) -> Path:
    runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if not runs:
        raise FileNotFoundError("no OBM runs under %s" % root)
    return runs[-1]


def sgs_truths():
    """Regenerate the base-case truths with base_case.py's own constants."""
    vario = GSLIB.make_variogram(
        nug=bc.NUG, nst=1, it1=bc.IT1, cc1=bc.CC1, azi1=bc.AZI1, hmaj1=bc.HMAJ1, hmin1=bc.HMIN1
    )
    out = {}
    for seed in bc.SEEDS:
        out[seed] = make_porosity_truth(
            nx=bc.NX, ny=bc.NY, xsiz=bc.XSIZ, ysiz=bc.YSIZ, xmn=bc.XMN, ymn=bc.YMN,
            vario=vario, mean=bc.POR_MEAN, stdev=bc.POR_STDEV, seed=seed,
        )
    return out


def obm_truths(run_dir: Path):
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        man = json.load(f)
    truths, facies, ntg = {}, {}, {}
    for rec in man["realizations"]:
        if rec["status"] != "included":
            continue
        seed = int(rec["geological_seed"])
        stem = rec["id"]
        truths[seed] = np.load(run_dir / (stem + "_truth_2d.npy"))
        facies[seed] = np.load(run_dir / (stem + "_facies_2d.npy"))
        ntg[seed] = float(np.mean(facies[seed] >= NET_FACIES_MIN))
        # Cross-check against the value recorded at generation time.
        assert abs(ntg[seed] - rec["summary"]["net_to_gross_2d"]) < 1e-12
    return man, truths, facies, ntg


def field_stats(z: np.ndarray):
    v = np.asarray(z, dtype=np.float64).ravel()
    rows = [
        ("mean", float(np.mean(v))),
        ("std", float(np.std(v))),                                   # population, ddof=0
        ("min", float(np.min(v))),
        ("max", float(np.max(v))),
        ("skewness", float(stats.skew(v, bias=True))),
        ("excess_kurtosis", float(stats.kurtosis(v, fisher=True, bias=True))),  # normal = 0
    ]
    for p in PERCENTILES:
        rows.append(("p%d" % p, float(np.percentile(v, p))))
    # Share of cells holding the single most frequent (exactly equal) value.
    _, counts = np.unique(v, return_counts=True)
    rows.append(("largest_tie_fraction", float(counts.max() / v.size)))
    return rows


def write_csv(path: Path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obm-run-dir", default=None)
    ap.add_argument("--obm-mud-run-dir", default=None)
    args = ap.parse_args(argv)
    obm_dirs = {"obm": Path(args.obm_run_dir).resolve() if args.obm_run_dir else latest_obm_run_dir()}
    if args.obm_mud_run_dir:
        obm_dirs["obm_mud"] = Path(args.obm_mud_run_dir).resolve()
    elif OBM_MUD_RAW_ROOT.is_dir():
        obm_dirs["obm_mud"] = latest_obm_run_dir(OBM_MUD_RAW_ROOT)

    sgs = sgs_truths()
    families = {"sgs": sgs}
    mans, facs, ntgs = {}, {}, {}
    for fam, d in obm_dirs.items():
        mans[fam], families[fam], facs[fam], ntgs[fam] = obm_truths(d)
    man, obm, obm_fac = mans["obm"], families["obm"], facs["obm"]

    for fam, d in families.items():
        for seed, z in d.items():
            assert z.shape == (bc.NY, bc.NX), (fam, seed, z.shape)
    g = man["params"]["config"]["grid"]
    assert int(g["nx"]) == bc.NX and int(g["ny"]) == bc.NY
    assert abs(float(g["x_len"]) / int(g["nx"]) - bc.XSIZ) < 1e-12
    assert abs(float(g["y_len"]) / int(g["ny"]) - bc.YSIZ) < 1e-12

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- summary_stats.csv --------------------------------------------------
    rows = []
    for fam, d in families.items():
        for seed in sorted(d):
            for name, val in field_stats(d[seed]):
                rows.append((fam, seed, name, val))
            if fam in OBM_FAMILIES:
                rows.append((fam, seed, "net_to_gross", ntgs[fam][seed]))
    write_csv(OUT_DIR / "summary_stats.csv", ["family", "seed", "statistic", "value"], rows)

    # --- histogram_bins.csv (pooled per family, common bins) ----------------
    allv = np.concatenate([z.ravel() for d in families.values() for z in d.values()])
    lo = np.floor(allv.min() / HIST_BIN_WIDTH) * HIST_BIN_WIDTH
    hi = np.ceil(allv.max() / HIST_BIN_WIDTH) * HIST_BIN_WIDTH
    edges = np.arange(lo, hi + HIST_BIN_WIDTH / 2, HIST_BIN_WIDTH)
    hist_rows, hist = [], {}
    for fam, d in families.items():
        pooled = np.concatenate([z.ravel() for z in d.values()])
        dens, _ = np.histogram(pooled, bins=edges, density=True)
        hist[fam] = dens
        for i in range(len(dens)):
            hist_rows.append((fam, round(float(edges[i]), 6), round(float(edges[i + 1]), 6), float(dens[i])))
    write_csv(OUT_DIR / "histogram_bins.csv", ["family", "bin_left", "bin_right", "density"], hist_rows)

    # --- variogram.csv -------------------------------------------------------
    vrows, vario_store = [], {}
    for fam, d in families.items():
        for seed in sorted(d):
            for direction, cell in (("x", bc.XSIZ), ("y", bc.YSIZ)):
                r = grid_semivariogram(d[seed], direction, NLAG, cell)
                vario_store[(fam, seed, direction)] = r
                for i in range(NLAG):
                    vrows.append((fam, seed, direction, int(r["lag_cells"][i]), float(r["lag_m"][i]),
                                  float(r["gamma"][i]), float(r["gamma_std"][i]), int(r["npairs"][i])))
    write_csv(OUT_DIR / "variogram.csv",
              ["family", "seed", "direction", "lag_cells", "lag_m", "gamma", "gamma_std", "npairs"], vrows)

    # --- example_fields.json -------------------------------------------------
    missing = [(f, s) for f in obm_dirs for s in EXAMPLE_OBM_SEEDS if s not in families[f]]
    if missing:
        raise RuntimeError("example OBM seeds not included in the run: %s" % missing)
    examples = {
        "grid": {"nx": bc.NX, "ny": bc.NY, "xsiz": bc.XSIZ, "ysiz": bc.YSIZ,
                 "xmin": bc.XMIN, "xmax": bc.XMAX, "ymin": bc.YMIN, "ymax": bc.YMAX,
                 "orientation": "arrays are [row][col], shape (ny, nx); row 0 = max-y (top), col 0 = min-x"},
        "units": {"truth": "porosity %", "facies": "ResMill facies code"},
        "facies_codes": {"-1": "FF overbank fines", "0": "FFCH mud plug", "1": "CS crevasse splay",
                         "2": "LV levee", "3": "LA lateral accretion", "4": "CH channel"},
        "sgs": {str(s): {"truth": np.round(sgs[s], 2).tolist()} for s in EXAMPLE_SGS_SEEDS},
        "obm": {str(s): {"truth": np.round(obm[s], 2).tolist(), "facies": obm_fac[s].astype(int).tolist()}
                for s in EXAMPLE_OBM_SEEDS},
    }
    if "obm_mud" in families:
        examples["obm_mud"] = {str(s): {"truth": np.round(families["obm_mud"][s], 2).tolist()}
                               for s in EXAMPLE_OBM_SEEDS}
    with open(OUT_DIR / "example_fields.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, separators=(",", ":"))

    # --- figure --------------------------------------------------------------
    fig_name = "truth_families_comparison.png"
    vmin = bc.POR_MEAN - 4 * bc.POR_STDEV
    vmax = bc.POR_MEAN + 4 * bc.POR_STDEV
    nrow = len(families) + 1
    fig = plt.figure(figsize=(16, 4 * nrow))
    panels = [("sgs", s) for s in EXAMPLE_SGS_SEEDS] + [(f, s) for f in obm_dirs for s in EXAMPLE_OBM_SEEDS]
    for i, (fam, s) in enumerate(panels, start=1):
        plt.subplot(nrow, 3, i)
        GSLIB.pixelplt_st(families[fam][s], bc.XMIN, bc.XMAX, bc.YMIN, bc.YMAX, bc.XSIZ, vmin, vmax,
                          "%s truth - seed %d" % (fam.upper(), s), "X (m)", "Y (m)", "Porosity (%)",
                          plt.cm.viridis)
    plt.subplot(nrow, 3, 3 * nrow - 2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for fam in families:
        plt.step(centers, hist[fam], where="mid", label="%s (n=%d)" % (fam.upper(), len(families[fam])))
    plt.xlabel("Porosity (%)"); plt.ylabel("Density"); plt.title("Pooled histogram"); plt.legend()
    for j, direction in enumerate(("x", "y")):
        plt.subplot(nrow, 3, 3 * nrow - 1 + j)
        for fam, color in zip(families, ("C0", "C1", "C2")):
            curves = np.array([vario_store[(fam, s, direction)]["gamma_std"] for s in sorted(families[fam])])
            lag_m = vario_store[(fam, sorted(families[fam])[0], direction)]["lag_m"]
            for c in curves:
                plt.plot(lag_m, c, color=color, alpha=0.2, lw=0.8)
            plt.plot(lag_m, curves.mean(axis=0), color=color, lw=2, label="%s mean" % fam.upper())
        plt.axhline(1.0, color="k", lw=0.8, ls="--")
        plt.xlabel("Lag (m)"); plt.ylabel("gamma / variance"); plt.title("Variogram, %s direction" % direction)
        plt.legend()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.95, wspace=0.35, hspace=0.35)
    plt.savefig(OUT_DIR / fig_name, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # --- source_runs.json ----------------------------------------------------
    git_dirty, git_dirty_files = get_git_status()
    def rel(path):
        try:
            return str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
        except ValueError:
            return str(path)
    source = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_entrypoint": CODE_ENTRYPOINT,
        "git_commit": get_git_commit(),
        "git_dirty": git_dirty,
        "git_dirty_files": git_dirty_files,
        **{fam: {
            "manifest_path": rel(obm_dirs[fam] / "manifest.json"),
            "manifest_git_commit": mans[fam]["git_commit"],
            "manifest_git_dirty": mans[fam].get("git_dirty"),
            "config_path": mans[fam]["params"].get("config_path"),
            "resmill": mans[fam]["software"]["resmill"],
            "seeds_used": sorted(families[fam]),
            "seeds_excluded": [e["geological_seed"] for e in mans[fam]["excluded_realizations"]],
            "file_used": "<stem>_truth_2d.npy (affine-rescaled per realization)",
        } for fam in obm_dirs},
        "sgs": {
            "regenerated_by": "src.truth_model.make_porosity_truth with src/experiments/base_case.py constants",
            "seeds": list(bc.SEEDS),
            "params": bc.build_params(),
        },
        "rules": {
            "std": "population (ddof=0)",
            "skewness": "scipy.stats.skew(bias=True)",
            "excess_kurtosis": "scipy.stats.kurtosis(fisher=True, bias=True); normal = 0",
            "percentiles": "numpy.percentile default (linear)",
            "largest_tie_fraction": "max count of exactly-equal values / number of cells",
            "net_to_gross": "fraction of 2D slice cells with facies code >= 1 (OBM only)",
            "histogram": "pooled over all realizations of a family, common bins of width %.2f %% from %.2f to %.2f, density=True"
                         % (HIST_BIN_WIDTH, lo, hi),
            "variogram": "src.grid_variogram.grid_semivariogram, lags 1..%d cells along x (columns) and y (rows); gamma_std = gamma / field variance (ddof=0)" % NLAG,
        },
        "output_files": ["summary_stats.csv", "histogram_bins.csv", "variogram.csv",
                         "example_fields.json", fig_name, "source_runs.json"],
    }
    with open(OUT_DIR / "source_runs.json", "w", encoding="utf-8") as f:
        json.dump(source, f, indent=2, default=str)

    for fam, d in obm_dirs.items():
        print("%s run: %s" % (fam, d))
    print("Output: %s" % OUT_DIR)
    return OUT_DIR


if __name__ == "__main__":
    main()
