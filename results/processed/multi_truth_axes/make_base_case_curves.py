"""Accuracy-plot curves (fraction of truth inside each nominal probability
interval, 20 p levels = default np.linspace(0, 1, 20) grid) for the E5
base-case cell of the multi-truth evaluation, for storyboard figure F2 panel A.

src/experiments/evaluate_multi_truth_axes.py stores only scalar metrics, not
the curves. This script does NOT reimplement the per-method path: it calls the
evaluator's own ``evaluate_cell`` on the E5 cell and records the exact
(p_intervals, fraction_in) arrays that ``evaluate_cell`` passes to
``calc_umg`` (one call per method per batch set, in code order kriging,
gp_mle, sgs, rbf_bootstrap). The recording wrapper is installed only for the
duration of that call and returns calc_umg's own result unchanged.

Checks (raise on failure):
  * exactly one E5 cell in the run config; batch set = the single batch set of
    evaluation_provenance.json;
  * exactly 4 calc_umg calls, each with the 20-point default p grid;
  * calc_umg(recorded curve) == the umg row evaluate_cell returned for that
    method (exact) and == the E5 umg row in metrics.csv (atol 1e-12).

Inputs (read-only): results/processed/multi_truth_axes/evaluation_provenance.json
(source run folder + batch sets), metrics.csv; the source run's run_config.json /
run_registry.jsonl and the per-method run folders under results/raw/ (read only,
via the evaluator's own loaders).

Output: results/processed/multi_truth_axes/accuracy_plot_curves.csv
  columns: experiment, case, truth_seed, sample_seed, config_key, method,
           n_ensemble, batches, p_interval, fraction_in

Run with:
.venv/Scripts/python.exe -m results.processed.multi_truth_axes.make_base_case_curves
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import src.experiments.evaluate_multi_truth_axes as ev  # noqa: E402
from src.evaluation import DEFAULT_BINS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PROVENANCE = OUT_DIR / "evaluation_provenance.json"
METRICS = OUT_DIR / "metrics.csv"
OUT_CSV = OUT_DIR / "accuracy_plot_curves.csv"
EXPERIMENT = "E5"
UMG_ATOL = 1e-12
# Order in which evaluate_cell calls calc_umg (kriging, gp_mle, then the
# ensemble methods in ev.ENSEMBLE order for each batch set).
CALL_ORDER = ("kriging", "gp_mle") + tuple(ev.ENSEMBLE)


def main():
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    sources = [s["source"] for s in prov["sources"]]
    batch_sets = [tuple(b) for b in prov["batch_sets"]]
    if len(batch_sets) != 1:
        raise ValueError(f"expected exactly one batch set in {PROVENANCE.name}, got {batch_sets}")
    batches_str = ",".join(str(b) for b in batch_sets[0])

    cells, runs, _ = ev.load_sources(sources)
    e5 = [c for c in cells.values() if c["experiment"] == EXPERIMENT]
    if len(e5) != 1:
        raise ValueError(f"expected 1 {EXPERIMENT} cell, found {len(e5)}")
    cell = e5[0]

    recorded = []
    original = ev.calc_umg

    def _recording_calc_umg(p, frac):
        g = original(p, frac)
        recorded.append((np.array(p, dtype=float), np.array(frac, dtype=float), g))
        return g

    ev.calc_umg = _recording_calc_umg
    try:
        rows, _hyper, info, _rec = ev.evaluate_cell(cell, runs[cell["config_key"]], batch_sets)
    finally:
        ev.calc_umg = original

    if len(recorded) != len(CALL_ORDER):
        raise AssertionError(f"expected {len(CALL_ORDER)} calc_umg calls, got {len(recorded)}")
    p_default = np.linspace(0.0, 1.0, DEFAULT_BINS)
    row_umg = {r["method"]: r["value"] for r in rows if r["metric"] == "umg"}
    row_nens = {r["method"]: r["n_ensemble"] for r in rows if r["metric"] == "umg"}

    metrics = ev.read_metrics_csv(METRICS)
    ref = metrics[(metrics["experiment"] == EXPERIMENT) & (metrics["metric"] == "umg")]

    out = []
    for method, (p, frac, g) in zip(CALL_ORDER, recorded):
        if not np.array_equal(p, p_default):
            raise AssertionError(f"{method}: recorded p grid is not the default {DEFAULT_BINS}-point grid")
        if g != row_umg[method]:
            raise AssertionError(f"{method}: recorded calc_umg {g} != evaluate_cell umg {row_umg[method]}")
        r = ref[ref["method"] == method]
        if len(r) != 1:
            raise ValueError(f"{METRICS.name}: {len(r)} {EXPERIMENT} umg rows for {method}")
        if not np.isclose(g, float(r["value"].iloc[0]), rtol=0.0, atol=UMG_ATOL):
            raise AssertionError(f"{method}: recomputed UMG {g!r} != metrics.csv {float(r['value'].iloc[0])!r}")
        nens = row_nens[method]
        is_ens = method in ev.ENSEMBLE
        for pi, fi in zip(p, frac):
            out.append({
                "experiment": EXPERIMENT, "case": cell["case"], "truth_seed": cell["truth_seed"],
                "sample_seed": cell["sample_seed"], "config_key": cell["config_key"], "method": method,
                "n_ensemble": int(nens) if is_ens else pd.NA, "batches": batches_str if is_ens else "",
                "p_interval": float(pi), "fraction_in": float(fi),
            })
        print(f"  {method:14s} UMG {g:.12f} (metrics.csv {float(r['value'].iloc[0]):.12f})")

    df = pd.DataFrame(out)
    df["n_ensemble"] = df["n_ensemble"].astype("Int64")
    df.to_csv(OUT_CSV, index=False)
    print(f"cell {cell['config_key']}: n_samples_actual {info['n_samples_actual']}, "
          f"n_cells_evaluated {info['n_cells_evaluated']}")
    print(f"wrote {OUT_CSV.relative_to(_REPO_ROOT).as_posix()} ({len(df)} rows)")
    return OUT_CSV


if __name__ == "__main__":
    main()
