"""F7 processed tables from the fixed-hyperparameter grid run
(src/experiments/hyperparam_grid_surface.py). Read-only on results/raw.

Writes (this directory):
  metrics_summary.csv  copy of <run>/metrics_summary.csv (mean / std / n over
                       the 10 truths per method x grid node x metric), unchanged
  optima.csv           per (method, metric in {mse, umg}): the grid NODE with the
                       best truth-mean (mse: minimum, umg: maximum), plus the
                       runner-up node and the gap between them. Taken from grid
                       nodes only -- no interpolation.
  provenance.json      source run dir, sha256 of the source files, checks

Run with:
.venv/Scripts/python.exe -m results.processed.hyperparam_grid_surface.make_f7_tables
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.io import get_git_commit, get_git_status  # noqa: E402

SOURCE_RUN = _REPO_ROOT / "results" / "raw" / "hyperparam_grid_surface" / "20261117T071222580635Z"
OUT_DIR = Path(__file__).resolve().parent
BEST = {"mse": "min", "umg": "max"}
EXPECTED_GRID = {"kriging": (8, 9), "sgs": (5, 5), "rbf_bootstrap": (8, 13), "gp": (8, 11)}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    src = SOURCE_RUN / "metrics_summary.csv"
    manifest = json.loads((SOURCE_RUN / "manifest.json").read_text(encoding="utf-8"))
    cross = json.loads((SOURCE_RUN / "crosscheck.json").read_text(encoding="utf-8"))
    s = pd.read_csv(src)

    checks = {"manifest_status": manifest["params"]["status"], "crosscheck_verdict": cross["verdict"],
              "n_truths_per_node": sorted(int(v) for v in s["n"].unique()),
              "n_failed_total": int(s["n_failed"].sum()),
              "sgsim_warnings_total": float(s["sgsim_warnings_total"].fillna(0).sum())}
    if checks["manifest_status"] != "complete" or checks["n_failed_total"] != 0:
        raise ValueError(f"source run not complete / has failed points: {checks}")
    for m, (n1, n2) in EXPECTED_GRID.items():
        g = s[(s["method"] == m) & (s["metric"] == "mse")]
        if (g["grid_i"].nunique(), g["grid_j"].nunique()) != (n1, n2) or len(g) != n1 * n2:
            raise ValueError(f"{m}: grid shape differs from {n1}x{n2}")

    rows = []
    for (m, met), g in s[s["metric"].isin(list(BEST))].groupby(["method", "metric"], sort=False):
        g = g.sort_values("mean", ascending=(BEST[met] == "min"), kind="mergesort")
        b, r = g.iloc[0], g.iloc[1]
        rows.append({
            "method": m, "metric": met, "criterion": BEST[met],
            "range_equiv_m": b["range_equiv_m"], "param1_name": b["param1_name"],
            "param1_value": b["param1_value"], "param2_name": b["param2_name"],
            "param2_value": b["param2_value"], "mean": b["mean"], "std_over_truths": b["std"], "n": int(b["n"]),
            "runner_up_range_equiv_m": r["range_equiv_m"], "runner_up_param2_value": r["param2_value"],
            "runner_up_mean": r["mean"], "gap_best_minus_runner_up": b["mean"] - r["mean"],
        })
    optima = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    optima.to_csv(OUT_DIR / "optima.csv", index=False)
    dirty, dirty_files = get_git_status()
    prov = {
        "generated_by": "results/processed/hyperparam_grid_surface/make_f7_tables.py",
        "git_commit": get_git_commit(), "git_dirty": dirty, "git_dirty_files": dirty_files,
        "source_run": SOURCE_RUN.relative_to(_REPO_ROOT).as_posix(),
        "source_files_sha256": {f: _sha256(SOURCE_RUN / f) for f in
                                ("metrics_summary.csv", "metrics.csv", "manifest.json", "crosscheck.json")},
        "source_run_git_commit": manifest.get("git_commit"),
        "checks": checks,
        "optimum_rule": "grid node with the best truth-mean: mse minimum, umg maximum (no interpolation); "
                        "ties broken by sort order (mergesort, stable)",
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(optima.to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
