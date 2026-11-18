"""Evaluate the inferred-variogram kriging / SGS re-run of the density axis
(src/experiments/inferred_variogram_density.py) with the SAME code path as
src/experiments/evaluate_multi_truth_axes.py:

  * truth / conditioning samples / mask : evaluate_multi_truth_axes._conditioning
  * run verification                    : evaluate_multi_truth_axes._verify_run
                                          (truth_seed, sample_seed, n_samples,
                                          TRUE hmaj1/hmin1/nug, sgs batch-1 seed,
                                          samples.csv == regenerated samples)
  * kriging metrics                     : evaluate_multi_truth_axes.kriging_metrics
  * SGS metrics                         : ensemble mean / var(ddof=1) recomputed
                                          from sgs_realizations.npy, checked against
                                          the saved maps (MAP_MATCH_TOL), then
                                          evaluate_multi_truth_axes._gaussian_metrics
                                          (point = ensemble mean), as evaluate_cell
                                          does for a single-batch set.

Extra checks (raise on mismatch): each run's manifest variogram ==
build_vario(fit_range_m, fit_range_m, fit_nugget) of its cell and carries
params.variogram_source; samples.csv equals the ORIGINAL oracle run's
samples.csv; SGS n_realizations == 50. As a code-path regression, the ORIGINAL
oracle kriging / SGS runs of the same cells are re-evaluated with these same
functions and compared to results/processed/multi_truth_axes/metrics.csv
(max abs difference recorded in the provenance; raises above ORACLE_REPRO_ATOL).

Outputs (--out-dir, default results/processed/inferred_variogram_density/),
column conventions of results/processed/multi_truth_axes/:
  metrics.csv          experiment, case, axis, axis_level, truth_seed, sample_rep,
                       sample_seed, method, n_ensemble, batches, metric, value
                       (method = kriging_inferred | sgs_inferred)
  metrics_summary.csv  per (experiment, case, axis, axis_level, method, n_ensemble,
                       batches, metric): mean, std (ddof=1), n
  evaluation_cells.csv per cell: fitted variogram, n samples, n cells evaluated,
                       sgs grid-node warnings, new and original run dirs
  evaluation_provenance.json

Picked-variogram run (inferred_variogram_density.py --variogram-source picks,
added 2026-11-18): detected from run_config.variogram_source_option == "picks";
methods kriging_picked | sgs_picked, default out dir
results/processed/picked_variogram_density/; evaluation_cells.csv additionally
carries vario_source / pick_confirmed / pick_edited / auto_fit_*. The variogram
check always compares against the per-cell variogram recorded in run_config
(= the automatic fit for the original run). Auto-run outputs are unchanged.
For the picked run pass --source results/raw/picked_variogram_density/<ts>.

Usage:
  .venv/Scripts/python.exe -m src.experiments.evaluate_inferred_variogram_density \
      --source results/raw/inferred_variogram_density/<ts>
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.experiments.base_case import NX, NY
from src.experiments.base_case_conditioning import VCOL, build_vario
from src.experiments.evaluate_multi_truth_axes import (
    CELL_ID_COLS,
    DEFAULT_BINS,
    MAP_MATCH_TOL,
    METRICS,
    NOMINAL_P_COVERAGE,
    SAMPLES_MATCH_ATOL,
    _conditioning,
    _gaussian_metrics,
    _verify_run,
    kriging_metrics,
    read_metrics_csv,
)
from src.experiments.multi_truth_axes import read_jsonl, rel_or_abs, resolve_run_dir
from src.io import get_git_commit, get_git_status

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "processed" / "inferred_variogram_density"
ORACLE_METRICS_CSV = _REPO_ROOT / "results" / "processed" / "multi_truth_axes" / "metrics.csv"
METHOD_LABEL = {"kriging": "kriging_inferred", "sgs": "sgs_inferred"}
# Per-run variant, selected from the source run_config (variogram_source_option,
# absent = "auto" for the original inferred_variogram_density run).
VARIANTS = {
    "auto": {"labels": {"kriging": "kriging_inferred", "sgs": "sgs_inferred"},
             "out_dir": DEFAULT_OUT_DIR, "vario_desc": "sample-inferred variogram (automatic fit)"},
    "picks": {"labels": {"kriging": "kriging_picked", "sgs": "sgs_picked"},
              "out_dir": _REPO_ROOT / "results" / "processed" / "picked_variogram_density",
              "vario_desc": "user-picked variogram (Variogram Review board; auto fit where no pick)"},
}
SGS_BATCH = 1
N_ENSEMBLE = 50
ORACLE_REPRO_ATOL = 1e-12


def _sgs_metrics(rdir: Path, p: dict, truth_m, mask, label) -> dict:
    ens = np.load(rdir / "sgs_realizations.npy")
    if ens.shape != (p["n_realizations"], NY, NX):
        raise ValueError(f"{rdir}: ensemble shape {ens.shape}")
    mean_map = ens.mean(axis=0)
    var_map = ens.var(axis=0, ddof=1)
    if not (np.allclose(mean_map, np.load(rdir / "sgs_mean_map.npy"), rtol=MAP_MATCH_TOL, atol=MAP_MATCH_TOL)
            and np.allclose(var_map, np.load(rdir / "sgs_var_map.npy"), rtol=MAP_MATCH_TOL, atol=MAP_MATCH_TOL)):
        raise AssertionError(f"{label}: recomputed maps != saved maps")
    return _gaussian_metrics(truth_m, mean_map[mask], mean_map[mask], var_map[mask], label)


def evaluate_cell(cell: dict, runs: Dict[str, str], labels: Dict[str, str] = None):
    labels = labels or METHOD_LABEL
    truth, samples_df, mask = _conditioning(cell["truth_seed"], cell["sample_seed"], cell["hmaj1"],
                                            cell["hmin1"], cell["nug"], cell["n_samples"])
    truth_m = truth[mask]
    ident = {k: cell[k] for k in CELL_ID_COLS}
    expected_vario = build_vario(hmaj1=cell["vario_hmaj1"], hmin1=cell["vario_hmin1"], nug=cell["vario_nug"])
    rows, oracle_rows, info = [], [], {}

    for method in ("kriging", "sgs"):
        rdir = resolve_run_dir(runs[method])
        batch = SGS_BATCH if method == "sgs" else None
        p = _verify_run(cell, method, rdir, samples_df, batch=batch)
        if p["variogram"] != expected_vario or "variogram_source" not in p:
            raise ValueError(f"{rdir}: variogram {p['variogram']} != run_config cell variogram {expected_vario}")
        odir = resolve_run_dir(cell["orig_run_dir_kriging"] if method == "kriging" else cell["orig_run_dir_sgs_b1"])
        a, b = pd.read_csv(rdir / "samples.csv"), pd.read_csv(odir / "samples.csv")
        if len(a) != len(b) or not np.allclose(a[["X", "Y", VCOL]].values, b[["X", "Y", VCOL]].values,
                                               rtol=0.0, atol=SAMPLES_MATCH_ATOL):
            raise ValueError(f"{rdir}: samples.csv differs from original run {odir}")
        label = f"{method} {cell['config_key']}"
        if method == "kriging":
            vals = kriging_metrics(rdir, truth_m, mask)
            ovals = kriging_metrics(odir, truth_m, mask)
            n_ens, batches = np.nan, ""
        else:
            if p["n_realizations"] != N_ENSEMBLE:
                raise ValueError(f"{rdir}: n_realizations {p['n_realizations']} != {N_ENSEMBLE}")
            op = _verify_run({**cell}, "sgs", odir, samples_df, batch=SGS_BATCH)
            vals = _sgs_metrics(rdir, p, truth_m, mask, label + " " + labels[method])
            ovals = _sgs_metrics(odir, op, truth_m, mask, label + " oracle")
            n_ens, batches = N_ENSEMBLE, str(SGS_BATCH)
            info["sgs_grid_node_warnings_b1"] = (p.get("sgsim_warnings") or {}).get("n_grid_node_warnings", np.nan)
            info["orig_sgs_grid_node_warnings_b1"] = (op.get("sgsim_warnings") or {}).get("n_grid_node_warnings",
                                                                                         np.nan)
        for metric in METRICS:
            rows.append({**ident, "method": labels[method], "n_ensemble": n_ens, "batches": batches,
                         "metric": metric, "value": float(vals[metric])})
            oracle_rows.append({**ident, "method": method, "metric": metric, "value": float(ovals[metric])})
        info[f"run_dir_{labels[method]}" + ("_b1" if method == "sgs" else "")] = runs[method]

    info = {**ident, "n_samples_requested": cell["n_samples"], "n_samples_actual": int(len(samples_df)),
            "n_cells_evaluated": int(mask.sum()), "config_key": cell["config_key"],
            "vario_hmaj1": cell["vario_hmaj1"], "vario_hmin1": cell["vario_hmin1"], "vario_nug": cell["vario_nug"],
            **({k: cell[k] for k in ("vario_source", "pick_confirmed", "pick_edited", "auto_fit_range_m",
                                     "auto_fit_nugget")} if "vario_source" in cell else {}),
            **info, "orig_run_dir_kriging": cell["orig_run_dir_kriging"],
            "orig_run_dir_sgs_b1": cell["orig_run_dir_sgs_b1"]}
    return rows, oracle_rows, info


def _evaluate_group(args):
    cells, runs_by_key, labels = args
    return [evaluate_cell(c, runs_by_key[c["config_key"]], labels) for c in cells]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: results/processed/inferred_variogram_density (auto run) or "
                         "results/processed/picked_variogram_density (--variogram-source picks run)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args(argv)

    src = resolve_run_dir(args.source)
    cfg = json.loads((src / "run_config.json").read_text(encoding="utf-8"))
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    variant_key = cfg.get("variogram_source_option", "auto")
    variant = VARIANTS[variant_key]
    labels = variant["labels"]
    out_dir_default = variant["out_dir"]
    reg = read_jsonl(src / "run_registry.jsonl")
    runs: Dict[str, Dict[str, str]] = {}
    for r in reg:
        if r["method"] in runs.get(r["config_key"], {}):
            raise ValueError(f"two runs registered for {r['config_key']} {r['method']}")
        runs.setdefault(r["config_key"], {})[r["method"]] = r["run_dir"]
    cells = cfg["cells"]
    missing = [c["config_key"] for c in cells if set(runs.get(c["config_key"], {})) != {"kriging", "sgs"}]
    if missing:
        raise ValueError(f"{len(missing)} cells lack runs (e.g. {missing[0]})")

    groups: Dict[tuple, list] = {}
    for c in cells:
        groups.setdefault((c["truth_seed"], c["hmaj1"], c["hmin1"], c["nug"]), []).append(c)
    tasks = []
    for g in groups.values():
        g = sorted(g, key=lambda c: (c["sample_seed"], c["n_samples"]))
        tasks.append((g, {c["config_key"]: runs[c["config_key"]] for c in g}, labels))
    results = []
    if args.workers <= 1:
        for t in tasks:
            results.extend(_evaluate_group(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for out in ex.map(_evaluate_group, tasks):
                results.extend(out)

    metrics = pd.DataFrame([r for res in results for r in res[0]])
    oracle = pd.DataFrame([r for res in results for r in res[1]])
    info = pd.DataFrame([res[2] for res in results])
    metrics["n_ensemble"] = metrics["n_ensemble"].astype("Int64")
    if not np.all(np.isfinite(metrics["value"].values)):
        raise ValueError("non-finite metric values")
    if len(metrics) != len(cells) * len(METRICS) * 2:
        raise ValueError(f"metrics has {len(metrics)} rows")

    # --- code-path regression against the existing oracle evaluation -------------
    ref = read_metrics_csv(ORACLE_METRICS_CSV)
    ref = ref[ref["method"].isin(["kriging", "sgs"]) & ref["batches"].isin(["", "1"])]
    key = ["experiment", "axis_level", "truth_seed", "sample_rep", "method", "metric"]
    merged = oracle.merge(ref[key + ["value"]], on=key, how="left", suffixes=("", "_ref"))
    if merged["value_ref"].isna().any():
        raise ValueError("oracle rows missing from multi_truth_axes/metrics.csv")
    oracle_max_abs_diff = float(np.max(np.abs(merged["value"] - merged["value_ref"])))
    if oracle_max_abs_diff > ORACLE_REPRO_ATOL:
        raise AssertionError(f"re-evaluated oracle metrics differ from {ORACLE_METRICS_CSV} "
                             f"(max abs diff {oracle_max_abs_diff:.3e})")

    sort_cols = ["experiment", "case", "axis_level", "truth_seed", "sample_rep", "method", "batches", "metric"]
    metrics = metrics.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    info = info.sort_values(["experiment", "case", "axis_level", "truth_seed", "sample_rep"],
                            kind="mergesort").reset_index(drop=True)
    grp = ["experiment", "case", "axis", "axis_level", "method", "n_ensemble", "batches", "metric"]
    summary = (metrics.groupby(grp, dropna=False, sort=False)["value"]
               .agg(mean="mean", std=lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, n="count")
               .reset_index())

    out_dir = Path(args.out_dir) if args.out_dir else out_dir_default
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    info.to_csv(out_dir / "evaluation_cells.csv", index=False)
    git_dirty, git_dirty_files = get_git_status()
    run_commits: Dict[str, Dict[str, int]] = {}
    for c in cells:
        for m in ("kriging", "sgs"):
            man = json.loads((resolve_run_dir(runs[c["config_key"]][m]) / "manifest.json").read_text(encoding="utf-8"))
            k = f"{man.get('git_commit')} (dirty={man.get('git_dirty')})"
            run_commits.setdefault(labels[m], {})
            run_commits[labels[m]][k] = run_commits[labels[m]].get(k, 0) + 1
    w = info["sgs_grid_node_warnings_b1"]
    prov = {
        "generated_by": "src/experiments/evaluate_inferred_variogram_density.py",
        "git_commit": get_git_commit(), "git_dirty": git_dirty, "git_dirty_files": git_dirty_files,
        "source_run": rel_or_abs(src),
        "source_run_status": src_manifest.get("params", {}).get("status"),
        "source_run_git_commit": src_manifest.get("git_commit"),
        "run_git_commits_by_method": run_commits,
        "oracle_source_run": cfg["source_run"],
        "variogram_source": cfg["variogram_source"],
        "n_cells_evaluated": len(cells),
        **({"variogram_source_option": variant_key, "cell_filter": cfg.get("cell_filter"),
            "vario_source_counts": info["vario_source"].value_counts().to_dict(),
            "unconfirmed_pick_cells": cfg["variogram_source"].get("unconfirmed_pick_cells")}
           if variant_key != "auto" else {}),
        "methods": ({"kriging_inferred": "simple kriging (kb2d) with the sample-inferred variogram",
                     "sgs_inferred": f"SGS, N={N_ENSEMBLE}, sgsim seed {cfg['sgs_seed']} (= oracle batch 1), "
                                     "inferred variogram"} if variant_key == "auto" else
                    {labels["kriging"]: f"simple kriging (kb2d) with the {variant['vario_desc']}",
                     labels["sgs"]: f"SGS, N={N_ENSEMBLE}, sgsim seed {cfg['sgs_seed']} (= oracle batch 1), "
                                    f"{variant['vario_desc']}"}),
        "metrics": list(METRICS),
        "coverage_nominal_p": NOMINAL_P_COVERAGE,
        "umg_bins": DEFAULT_BINS,
        "evaluation_code_path": "evaluate_multi_truth_axes._conditioning / _verify_run / kriging_metrics / "
                                "_gaussian_metrics (identical functions to the oracle evaluation)",
        "summary_definition": "mean / std(ddof=1) / n over all (truth_seed, sample_rep) rows of a "
                              "(experiment, axis_level, method, n_ensemble, batches, metric) group",
        "batches_column": f"'1' for {labels['sgs']}, '' for {labels['kriging']}; read with "
                          "evaluate_multi_truth_axes.read_metrics_csv",
        "checks": {
            "oracle_reevaluation_vs_multi_truth_axes_metrics_csv": {
                "reference": rel_or_abs(ORACLE_METRICS_CSV), "n_values_compared": int(len(merged)),
                "max_abs_diff": oracle_max_abs_diff, "tolerance": ORACLE_REPRO_ATOL},
            "samples_equal_to_original_runs": True,
            ("manifest_variogram_equals_inferred_fit" if variant_key == "auto"
             else "manifest_variogram_equals_run_config_cell_variogram"): True,
            ("sgs_grid_node_warnings_inferred" if variant_key == "auto" else "sgs_grid_node_warnings_picked"): {"total": float(np.nansum(w)), "runs_with_warnings": int((w > 0).sum())},
            "sgs_grid_node_warnings_oracle_same_cells": {
                "total": float(np.nansum(info["orig_sgs_grid_node_warnings_b1"])),
                "runs_with_warnings": int((info["orig_sgs_grid_node_warnings_b1"] > 0).sum())},
        },
    }
    (out_dir / "evaluation_provenance.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))
    print(json.dumps(prov["checks"], indent=2))
    print(f"wrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
