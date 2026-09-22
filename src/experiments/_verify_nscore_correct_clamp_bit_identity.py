"""One-off verification that rewriting ``src/nscore.py``'s transform loop
with ``sgsim``'s correct 0-based clamp changes no stored number.

Background
----------
``src/nscore.py`` originally repaired ``geostats.nscore``'s 1-based clamp by
re-evaluating only the data at or below the table's first knot
(``values <= vr[0]``) on the first segment. That patch is wrong when the
minimum value is TIED (``vr[0] == vr[1]`` makes the first segment degenerate,
so ``dpowint`` returns ``(vrg[0] + vrg[1]) / 2``). It was replaced by a
re-run of the whole transform loop using the clamp geostatspy's own ``sgsim``
uses, ``j = min(max(0, j), nd - 2)``.

This script answers the only question that matters for already-saved results:
does the new implementation reproduce, BIT FOR BIT, the ``NPor`` column of the
kriging runs that the nugget-axis artifacts are pinned to? If yes, no run has
to be re-executed.

Checks
------
1. For each of the 6 nugget-axis levels pinned in
   ``results/processed/nugget_axis/source_runs.json``: re-run
   ``src.nscore.nscore`` on that run's ``samples.csv`` ``Por`` column and
   require ``np.array_equal`` against the stored ``NPor``.
2. For EVERY run directory under ``results/raw/kriging/``: re-run the wrapper
   and require it not to raise (no guard false positive). Runs produced
   before the fix still carry buggy stored ``NPor``; for those only the guard
   outcome is checked, and the number of differing values is reported.

Run:
    .venv/Scripts/python.exe -m src.experiments._verify_nscore_correct_clamp_bit_identity

Read-only: writes nothing, touches no run directory.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import geostatspy.geostats as geostats

from src.nscore import nscore

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUNS = REPO_ROOT / "results" / "processed" / "nugget_axis" / "source_runs.json"
KRIGING_RAW = REPO_ROOT / "results" / "raw" / "kriging"
VCOL = "Por"
NSCOL = "NPor"


def _read_csv_exact(path: Path) -> pd.DataFrame:
    """Read a run's CSV WITHOUT losing the last bit of each float.

    pandas' default C-parser float conversion is not round-trip exact: on
    this data it reads e.g. the (correctly written) text
    ``-1.6098160784042201`` back as ``-1.60981607840422``, one ULP away. That
    is invisible for every normal use, but it makes a bit-identity check give
    a false negative -- a freshly recomputed transform table "differed" from
    the stored one at 41/121 entries purely because of the reader.
    ``float_precision="round_trip"`` removes it (verified: with it, a fresh
    ``geostats.nscore`` reproduces the stored table exactly).
    """
    return pd.read_csv(path, float_precision="round_trip")


def _recompute(samples_csv: Path):
    df = _read_csv_exact(samples_csv)
    ns, vr, vrg, corrections = nscore(df[[VCOL]], VCOL, verbose=False)
    return df, ns


def check_pinned_nugget_runs() -> bool:
    print("=" * 72)
    print("CHECK 1: bit-identity against the 6 pinned nugget-axis kriging runs")
    print("=" * 72)
    source_runs = json.loads(SOURCE_RUNS.read_text(encoding="utf-8"))
    ok = True
    for level in sorted(source_runs, key=float):
        run_dir = REPO_ROOT / source_runs[level]["kriging"]
        df, ns = _recompute(run_dir / "samples.csv")
        stored = df[NSCOL].values
        identical = np.array_equal(ns, stored)
        max_abs = float(np.max(np.abs(ns - stored))) if len(ns) else 0.0
        print(
            f"  nug={level:>4}  n={len(df):>4}  {run_dir.name}  "
            f"array_equal={identical}  max|diff|={max_abs:.3e}"
        )
        ok = ok and identical
    print(f"  -> all 6 levels bit-identical: {ok}")
    return ok


def check_all_kriging_runs() -> bool:
    print()
    print("=" * 72)
    print("CHECK 2: guard false positives over every results/raw/kriging run")
    print("=" * 72)
    run_dirs = sorted(d for d in KRIGING_RAW.iterdir() if d.is_dir())
    n_fail = 0
    n_identical = 0
    n_differ = 0
    worst = (0.0, None)
    for run_dir in run_dirs:
        samples_csv = run_dir / "samples.csv"
        if not samples_csv.exists():
            print(f"  SKIP (no samples.csv): {run_dir.name}")
            continue
        try:
            df, ns = _recompute(samples_csv)
        except Exception as exc:  # guard fired -> false positive
            n_fail += 1
            print(f"  GUARD FAILED: {run_dir.name}: {type(exc).__name__}: {exc}")
            continue
        stored = df[NSCOL].values
        if np.array_equal(ns, stored):
            n_identical += 1
        else:
            n_differ += 1
            d = float(np.max(np.abs(ns - stored)))
            if d > worst[0]:
                worst = (d, run_dir.name)
    print(f"  runs inspected           : {len(run_dirs)}")
    print(f"  guard failures           : {n_fail}")
    print(f"  stored NPor bit-identical: {n_identical}")
    print(f"  stored NPor differs      : {n_differ}  (pre-fix runs; expected)")
    if worst[1]:
        print(f"  largest stored-vs-recomputed difference: {worst[0]:.6g} in {worst[1]}")
    return n_fail == 0


def _legacy_wrapper_ns(values: np.ndarray, vr: np.ndarray, vrg: np.ndarray, ns_raw) -> np.ndarray:
    """The ORIGINAL src/nscore.py algorithm, reproduced here for comparison:
    keep the library's ``ns`` and re-evaluate only data with
    ``value <= vr[0]`` on the first table segment."""
    ns = np.asarray(ns_raw, dtype=float).copy()
    affected = np.flatnonzero(values <= vr[0])
    for i in affected:
        ns[i] = geostats.dpowint(vr[0], vr[1], vrg[0], vrg[1], values[i], 1.0)
    return ns


def check_new_vs_legacy_algorithm() -> bool:
    """The question the rewrite has to answer: on the real data, does the
    correct 0-based clamp produce anything different from the value-based
    patch it replaces? (It can only differ when the minimum is tied.)"""
    print()
    print("=" * 72)
    print("CHECK 3: new whole-loop clamp vs the legacy `values <= vr[0]` patch")
    print("=" * 72)
    run_dirs = sorted(d for d in KRIGING_RAW.iterdir() if d.is_dir())
    n_same = 0
    n_diff = 0
    for run_dir in run_dirs:
        samples_csv = run_dir / "samples.csv"
        if not samples_csv.exists():
            continue
        df = _read_csv_exact(samples_csv)
        ns_raw, vr, vrg = geostats.nscore(df[[VCOL]], VCOL)
        legacy = _legacy_wrapper_ns(np.asarray(df[VCOL].values, dtype=float), vr, vrg, ns_raw)
        new, _, _, _ = nscore(df[[VCOL]], VCOL, verbose=False)
        if np.array_equal(new, legacy):
            n_same += 1
        else:
            n_diff += 1
            print(f"  DIFFERS: {run_dir.name} max|diff|={np.max(np.abs(new - legacy)):.6g}")
    print(f"  runs identical to the legacy algorithm: {n_same}")
    print(f"  runs that differ                      : {n_diff}")
    return n_diff == 0


def main() -> int:
    ok1 = check_pinned_nugget_runs()
    ok2 = check_all_kriging_runs()
    ok3 = check_new_vs_legacy_algorithm()
    print()
    print("RESULT:", "PASS" if (ok1 and ok2 and ok3) else "FAIL")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
