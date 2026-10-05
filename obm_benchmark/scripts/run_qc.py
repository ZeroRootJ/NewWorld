"""Automated QC for the ALLUVSIM-based ground truth (brief section 9).

Two uses:

1. Imported by ``generate_ground_truth.py``, which calls these functions
   in-process during generation, writes ``qc_summary.csv`` + diagnostic PNGs
   into the run directory and records exclusions in manifest.json.
2. Standalone, READ-ONLY re-verification of an existing run directory
   (results/raw is never modified after the fact)::

       .venv-obm/Scripts/python.exe -m obm_benchmark.scripts.run_qc [RUN_DIR]

   It reloads every saved array, recomputes all QC checks from the files on
   disk and compares them to manifest.json / qc_summary.csv. Exit code 0 if
   everything agrees, 1 otherwise. It writes nothing.

All rules are defined in obm_benchmark/config/cb_jigsaw.yaml (qc section),
fixed before any full-run result was inspected.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from obm_benchmark.scripts._common import (
    FILE_SUFFIXES,
    NET_FACIES_MIN,
    REPO_ROOT,
    array_sha256,
    file_sha256,
    slice_to_project_2d,
)
from src.grid_variogram import grid_semivariogram

EXCLUSION_RULE_IDS = ["shape", "finite", "porosity_range", "facies_codes", "slice_nondegenerate", "duplicate"]

QC_SUMMARY_FILE = "qc_summary.csv"
QC_SUMMARY_COLUMNS = [
    "realization_id",
    "geological_seed",
    "check_shape",
    "check_finite",
    "check_porosity_range",
    "check_facies_codes",
    "check_slice_nondegenerate",
    "check_duplicate",
    "check_slice_rule",
    "check_files_match",
    "por3d_min",
    "por3d_max",
    "por2d_std",
    "n_invalid_facies_3d",
    "duplicate_of",
    "status",
    "exclusion_reasons",
]


# ---------------------------------------------------------------------------
# Per-realization checks
# ---------------------------------------------------------------------------
def realization_checks(
    por3d: np.ndarray,
    fac3d: np.ndarray,
    por2d: np.ndarray,
    fac2d: np.ndarray,
    truth2d: Optional[np.ndarray],
    cfg: Dict[str, Any],
    valid_codes: List[int],
) -> Dict[str, Dict[str, Any]]:
    """Exclusion-rule checks except ``duplicate`` (needs all seeds).
    Returns {rule_id: {"passed": bool, "detail": str}}."""
    g = cfg["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    vmin, vmax = float(cfg["porosity"]["valid_min"]), float(cfg["porosity"]["valid_max"])
    out = {}

    shape_ok = (
        por3d.shape == (nx, ny, nz)
        and fac3d.shape == (nx, ny, nz)
        and por2d.shape == (ny, nx)
        and fac2d.shape == (ny, nx)
        and (truth2d is None or truth2d.shape == (ny, nx))
    )
    out["shape"] = {
        "passed": bool(shape_ok),
        "detail": "por3d %s fac3d %s por2d %s fac2d %s truth2d %s"
        % (por3d.shape, fac3d.shape, por2d.shape, fac2d.shape, None if truth2d is None else truth2d.shape),
    }

    arrays = [por3d, por2d] + ([truth2d] if truth2d is not None else [])
    n_nonfinite = int(sum(np.count_nonzero(~np.isfinite(a)) for a in arrays))
    out["finite"] = {"passed": n_nonfinite == 0, "detail": "n_nonfinite=%d" % n_nonfinite}

    finite = por3d[np.isfinite(por3d)]
    n_out = int(np.count_nonzero((finite < vmin) | (finite > vmax)))
    out["porosity_range"] = {
        "passed": n_out == 0,
        "detail": "n_outside[%g,%g]=%d (min=%.6g max=%.6g)"
        % (vmin, vmax, n_out, float(finite.min()) if finite.size else np.nan, float(finite.max()) if finite.size else np.nan),
    }

    n_bad = int(np.count_nonzero(~np.isin(fac3d, valid_codes)))
    out["facies_codes"] = {"passed": n_bad == 0, "detail": "n_invalid_facies_3d=%d" % n_bad}

    s = float(np.std(por2d.astype(np.float64)))
    thr = float(cfg["qc"]["min_slice_std"])
    out["slice_nondegenerate"] = {"passed": bool(s > thr), "detail": "slice_std=%.6g (rule: > %g)" % (s, thr)}
    return out


def duplicate_map(hashes: Dict[int, str]) -> Dict[int, Optional[int]]:
    """{seed: lowest earlier seed with the same hash, or None}."""
    first_seen: Dict[str, int] = {}
    dup_of: Dict[int, Optional[int]] = {}
    for seed in sorted(hashes):
        h = hashes[seed]
        if h in first_seen:
            dup_of[seed] = first_seen[h]
        else:
            first_seen[h] = seed
            dup_of[seed] = None
    return dup_of


def slice_rule_check(run_dir: Path, stem: str, slice_index: int, truth_coeffs: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """FATAL check: reload the saved 3D and 2D files from disk and confirm
    the 2D files equal the transformed 3D[:, :, slice_index] (and that the
    truth equals a * porosity_2d + b with the recorded coefficients)."""
    por3d = np.load(run_dir / (stem + FILE_SUFFIXES["porosity_3d"]))
    fac3d = np.load(run_dir / (stem + FILE_SUFFIXES["facies_3d"]))
    por2d = np.load(run_dir / (stem + FILE_SUFFIXES["porosity_2d_frac"]))
    fac2d = np.load(run_dir / (stem + FILE_SUFFIXES["facies_2d"]))
    problems = []
    if not np.array_equal(por2d, slice_to_project_2d(por3d, slice_index)):
        problems.append("porosity_2d_frac != flipud(por3d[:,:,k].T)")
    if not np.array_equal(fac2d, slice_to_project_2d(fac3d, slice_index)):
        problems.append("facies_2d != flipud(fac3d[:,:,k].T)")
    truth_path = run_dir / (stem + FILE_SUFFIXES["truth_2d"])
    if truth_coeffs is not None:
        truth = np.load(truth_path)
        expect = truth_coeffs["a"] * por2d.astype(np.float64) + truth_coeffs["b"]
        if not np.allclose(truth, expect, rtol=0, atol=1e-9):
            problems.append("truth_2d != a*porosity_2d_frac + b")
    elif truth_path.exists():
        problems.append("truth_2d exists although slice is degenerate")
    return {"passed": not problems, "detail": "; ".join(problems) if problems else "ok"}


def files_match_check(run_dir: Path, file_records: Dict[str, str], expected_extra: List[str]) -> Dict[str, Any]:
    """FATAL check: every recorded file exists with matching sha256; every
    file on disk is either recorded or in ``expected_extra``."""
    problems = []
    for name, sha in file_records.items():
        p = run_dir / name
        if not p.exists():
            problems.append("missing %s" % name)
        elif file_sha256(p) != sha:
            problems.append("sha256 mismatch %s" % name)
    on_disk = {p.name for p in run_dir.iterdir() if p.is_file()}
    unexpected = sorted(on_disk - set(file_records) - set(expected_extra))
    if unexpected:
        problems.append("unlisted files: %s" % unexpected)
    return {"passed": not problems, "detail": "; ".join(problems) if problems else "ok"}


# ---------------------------------------------------------------------------
# Summary statistics for manifest records
# ---------------------------------------------------------------------------
def _stats(a: np.ndarray) -> Dict[str, float]:
    a64 = np.asarray(a, dtype=np.float64)
    return {"min": float(a64.min()), "max": float(a64.max()), "mean": float(a64.mean()), "std": float(a64.std())}


def facies_proportions(fac: np.ndarray, codes: List[int]) -> Dict[str, float]:
    n = fac.size
    return {str(c): float(np.count_nonzero(fac == c)) / n for c in codes}


def porosity_mean_by_facies(por: np.ndarray, fac: np.ndarray, codes: List[int]) -> Dict[str, Optional[float]]:
    out = {}
    for c in codes:
        m = fac == c
        out[str(c)] = float(np.mean(por[m].astype(np.float64))) if np.any(m) else None
    return out


def realization_summary(por3d, fac3d, por2d, fac2d, codes: List[int]) -> Dict[str, Any]:
    return {
        "porosity_3d": _stats(por3d),
        "porosity_2d_frac": _stats(por2d),
        "facies_proportions_3d": facies_proportions(fac3d, codes),
        "facies_proportions_2d": facies_proportions(fac2d, codes),
        "net_to_gross_3d": float(np.mean(fac3d >= NET_FACIES_MIN)),
        "net_to_gross_2d": float(np.mean(fac2d >= NET_FACIES_MIN)),
        "porosity_mean_by_facies_3d": porosity_mean_by_facies(por3d, fac3d, codes),
        "porosity_mean_by_facies_2d": porosity_mean_by_facies(por2d, fac2d, codes),
        "hash_3d_porosity_facies": array_sha256(por3d, fac3d),
        "hash_2d_porosity": array_sha256(por2d),
    }


# ---------------------------------------------------------------------------
# Diagnostics (predefined seeds only)
# ---------------------------------------------------------------------------
FACIES_LABELS = {-1: "FF", 0: "FFCH", 1: "CS", 2: "LV", 3: "LA", 4: "CH"}


def connectivity_stats(por2d: np.ndarray, cfg: Dict[str, Any]):
    from scipy import ndimage

    c = cfg["diagnostics"]["connectivity"]
    if int(c["structure"]) != 4:
        raise NotImplementedError("only 4-connectivity is implemented")
    mask = por2d >= float(c["threshold"])
    labels, n = ndimage.label(mask)  # default structure = 4-connectivity cross in 2D
    sizes = np.bincount(labels.ravel())[1:] if n > 0 else np.array([], dtype=int)
    return labels, {
        "threshold": float(c["threshold"]),
        "structure": 4,
        "fraction_above_threshold": float(mask.mean()),
        "n_components": int(n),
        "largest_component_cells": int(sizes.max()) if n > 0 else 0,
        "largest_component_fraction_of_grid": float(sizes.max() / mask.size) if n > 0 else 0.0,
    }


def make_diagnostic_plot(run_dir: Path, stem: str, seed: int, por2d: np.ndarray, fac2d: np.ndarray,
                         cfg: Dict[str, Any], codes: List[int]) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    g = cfg["grid"]
    extent = (0.0, float(g["x_len"]), 0.0, float(g["y_len"]))  # row 0 = max-y -> imshow origin='upper'
    cell = float(g["x_len"]) / int(g["nx"])
    assert abs(cell - float(g["y_len"]) / int(g["ny"])) < 1e-12
    nlag = int(cfg["diagnostics"]["variogram_nlag"])
    vx = grid_semivariogram(por2d, "x", nlag, cell)
    vy = grid_semivariogram(por2d, "y", nlag, cell)
    labels, conn = connectivity_stats(por2d, cfg)

    fig = plt.figure(figsize=(15, 9))
    plt.subplot(2, 3, 1)
    im = plt.imshow(por2d, extent=extent, origin="upper", cmap="viridis", vmin=0.0, vmax=float(cfg["porosity"]["valid_max"]))
    plt.colorbar(im, label="Porosity (fraction)")
    plt.title("Porosity slice k=%d - seed %d" % (int(g["nz"]) // 2, seed))
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")

    plt.subplot(2, 3, 2)
    cmap = ListedColormap(["#6b4f2a", "#3b3b3b", "#e6c229", "#9acd32", "#ff8c00", "#1f77b4"])
    norm = BoundaryNorm(np.arange(min(codes) - 0.5, max(codes) + 1.5), cmap.N)
    im = plt.imshow(fac2d, extent=extent, origin="upper", cmap=cmap, norm=norm)
    cb = plt.colorbar(im, ticks=codes)
    cb.ax.set_yticklabels(["%d %s" % (c, FACIES_LABELS.get(c, "?")) for c in codes])
    plt.title("Facies slice - seed %d" % seed)
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")

    plt.subplot(2, 3, 3)
    plt.hist(por2d.ravel(), bins=40, range=(0.0, float(cfg["porosity"]["valid_max"])), color="grey", edgecolor="k")
    plt.xlabel("Porosity (fraction)"); plt.ylabel("Count"); plt.title("Slice porosity histogram")

    plt.subplot(2, 3, 4)
    present = [c for c in codes if np.any(fac2d == c)]
    plt.boxplot([por2d[fac2d == c] for c in present])
    plt.xticks(np.arange(1, len(present) + 1), [FACIES_LABELS.get(c, str(c)) for c in present])
    plt.ylabel("Porosity (fraction)"); plt.title("Slice porosity by facies")

    plt.subplot(2, 3, 5)
    plt.plot(vx["lag_m"], vx["gamma_std"], "o-", label="x direction")
    plt.plot(vy["lag_m"], vy["gamma_std"], "s-", label="y direction")
    plt.axhline(1.0, color="k", lw=0.8, ls="--")
    plt.xlabel("Lag (m)"); plt.ylabel("gamma / variance"); plt.title("Directional variogram (slice)")
    plt.legend()

    plt.subplot(2, 3, 6)
    shown = np.where(labels > 0, labels, np.nan)
    plt.imshow(shown, extent=extent, origin="upper", cmap="tab20", interpolation="nearest")
    plt.title("Connected bodies, por >= %.2f (4-conn)\nn=%d, largest=%.3f of grid"
              % (conn["threshold"], conn["n_components"], conn["largest_component_fraction_of_grid"]))
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")

    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.97, top=0.93, wspace=0.35, hspace=0.4)
    fname = "diag_%s.png" % stem
    plt.savefig(run_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "file": fname,
        "connectivity": conn,
        "variogram_x_gamma_std": [float(v) for v in vx["gamma_std"]],
        "variogram_y_gamma_std": [float(v) for v in vy["gamma_std"]],
        "variogram_lag_m": [float(v) for v in vx["lag_m"]],
    }


def write_qc_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=QC_SUMMARY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in QC_SUMMARY_COLUMNS})


# ---------------------------------------------------------------------------
# Standalone read-only re-verification
# ---------------------------------------------------------------------------
def latest_run_dir() -> Path:
    root = REPO_ROOT / "results" / "raw" / "obm_ground_truth"
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError("no runs under %s" % root)
    return runs[-1]


def verify_run(run_dir: Path) -> List[str]:
    """Recompute every QC check from files on disk; return a list of
    disagreements (empty = consistent)."""
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        man = json.load(f)
    cfg = man["params"]["config"]
    codes = sorted(int(k) for k in man["params"]["facies_props"])
    k = int(man["params"]["slice_index"])
    problems = []

    file_records = {}
    for rec in man["realizations"]:
        file_records.update(rec["files_sha256"])
    shared = man["params"]["shared_files_sha256"]
    file_records.update(shared)
    fm = files_match_check(run_dir, file_records, expected_extra=["manifest.json"])
    if not fm["passed"]:
        problems.append("files_match: " + fm["detail"])
    listed = set(man["output_files"])
    if listed != set(file_records):
        problems.append("output_files != recorded files: %s" % sorted(listed ^ set(file_records)))

    with open(run_dir / QC_SUMMARY_FILE, newline="", encoding="utf-8") as f:
        qc_rows = {int(r["geological_seed"]): r for r in csv.DictReader(f)}

    hashes = {}
    recomputed_status = {}
    for rec in man["realizations"]:
        seed, stem = int(rec["geological_seed"]), rec["id"]
        por3d = np.load(run_dir / (stem + FILE_SUFFIXES["porosity_3d"]))
        fac3d = np.load(run_dir / (stem + FILE_SUFFIXES["facies_3d"]))
        por2d = np.load(run_dir / (stem + FILE_SUFFIXES["porosity_2d_frac"]))
        fac2d = np.load(run_dir / (stem + FILE_SUFFIXES["facies_2d"]))
        tp = run_dir / (stem + FILE_SUFFIXES["truth_2d"])
        truth = np.load(tp) if tp.exists() else None
        chk = realization_checks(por3d, fac3d, por2d, fac2d, truth, cfg, codes)
        hashes[seed] = array_sha256(por3d, fac3d)
        if hashes[seed] != rec["summary"]["hash_3d_porosity_facies"]:
            problems.append("seed %d: 3D hash differs from manifest" % seed)
        sr = slice_rule_check(run_dir, stem, k, rec.get("affine"))
        if not sr["passed"]:
            problems.append("seed %d slice_rule: %s" % (seed, sr["detail"]))
        recomputed_status[seed] = chk
    dup = duplicate_map(hashes)
    excluded_manifest = {int(e["geological_seed"]) for e in man["excluded_realizations"]}
    for seed, chk in recomputed_status.items():
        failed = [rid for rid in EXCLUSION_RULE_IDS[:-1] if not chk[rid]["passed"]]
        if dup[seed] is not None:
            failed.append("duplicate")
        should_exclude = bool(failed)
        if should_exclude != (seed in excluded_manifest):
            problems.append("seed %d: recomputed exclusion=%s, manifest=%s" % (seed, failed, seed in excluded_manifest))
        row = qc_rows.get(seed)
        if row is None:
            problems.append("seed %d missing from qc_summary.csv" % seed)
        elif (row["status"] == "excluded") != should_exclude:
            problems.append("seed %d: qc_summary status %s disagrees" % (seed, row["status"]))
    return problems


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    run_dir = Path(argv[0]) if argv else latest_run_dir()
    problems = verify_run(run_dir)
    print("Run directory: %s" % run_dir)
    if problems:
        print("QC re-verification FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print("QC re-verification passed (all checks recomputed from disk agree with manifest.json and qc_summary.csv).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
