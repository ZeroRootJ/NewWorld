"""Density axis (E4 of multi_truth_axes) re-run of simple kriging and SGS with
the SAMPLE-INFERRED variogram instead of the true (oracle) one
(coordinator task 2026-11-17).

What is re-run, and what is identical to the original
-----------------------------------------------------
Source: the production batch-1 multi_truth_axes run
(SOURCE_RUN below), experiment E4 = 5 density levels x 10 truth seeds x 10
sample seeds = 500 cells. For every cell, kriging and SGS are run again with

    vario = build_vario(hmaj1=fit_range_m, hmin1=fit_range_m, nug=fit_nugget)

(isotropic, spherical, unit sill -- the same build_vario kriging.py / sgs.py
use for the true variogram), where fit_range_m / fit_nugget are the
automatic fits of results/processed/variogram_inference/fitted_variograms.csv
(rows experiment == "E4", used as-is). The variogram is passed through the
``vario_override`` argument of kriging.main / sgs.main; with it, hmaj1 / hmin1
/ nug keep defining the GROUND TRUTH. Everything else is the unchanged
method code: same truth realization (truth_seed, true range/nugget), same
conditioning samples (sample_seed, n_samples), same grid, same kb2d / sgsim
search settings, same nscore handling, SGS N = 50 with the SAME sgsim seed as
the original batch 1 (sgs_batch_seed(1) = 4101, shared by all cells as in the
original), same jitter seed. The cell list is taken from the source
run_config.json and asserted equal to multi_truth_axes.build_cells(("E4",)).

GP-MLE and RBF+bootstrap do not use a variogram and are NOT re-run.

Output layout (results/raw convention)
--------------------------------------
results/raw/inferred_variogram_density/<timestamp>/
  run_config.json     cells (+ fitted variogram + original run dirs per cell),
                      seeds, N, source provenance
  run_registry.jsonl  one line per COMPLETED method run (resume ledger)
  failures.jsonl      one line per failed task (retried on --resume)
  progress.log        completed / total, elapsed, ETA
  source_runs.json    {case: {axis_level: {truth_seed: {sample_rep: {method: run_dir}}}}}
  manifest.json       src.io.save_result
  runs/kriging/<ts>/, runs/sgs/<ts>/   the method run folders (own manifests)

Usage
-----
  .venv/Scripts/python.exe -m src.experiments.inferred_variogram_density --max-workers 20
  .venv/Scripts/python.exe -m src.experiments.inferred_variogram_density --resume results/raw/inferred_variogram_density/<ts> --max-workers 20
Smoke (scratch output only): add --raw-root <scratch dir> --limit-cells K
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.experiments._axis_parallel import default_max_workers, run_levels_parallel
from src.experiments.base_case_conditioning import build_vario
from src.experiments.multi_truth_axes import (
    _append_jsonl,
    _fmt_seconds,
    _now,
    build_cells,
    build_source_runs,
    read_jsonl,
    rel_or_abs,
    resolve_run_dir,
    sgs_batch_seed,
)
from src.io import get_git_commit, get_git_status, make_run_dir, save_result

EXPERIMENT_NAME = "inferred_variogram_density"
CODE_ENTRYPOINT = "src/experiments/inferred_variogram_density.py"

SOURCE_RUN = "results/raw/multi_truth_axes/20261116T222609599808Z"
SOURCE_EXPERIMENT = "E4"
SOURCE_BATCH = 1
FITTED_VARIOGRAMS_CSV = "results/processed/variogram_inference/fitted_variograms.csv"
VARIOGRAM_INFERENCE_MANIFEST = "results/processed/variogram_inference/manifest.json"

METHODS = ("kriging", "sgs")
N_ENSEMBLE = 50  # == source batch-1 n_ensemble_this_batch (asserted)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_config(raw_root: str, limit_cells=None) -> dict:
    src = resolve_run_dir(SOURCE_RUN)
    src_cfg = json.loads((src / "run_config.json").read_text(encoding="utf-8"))
    if src_cfg["batch"] != SOURCE_BATCH or src_cfg["n_ensemble_this_batch"] != N_ENSEMBLE:
        raise ValueError("source run is not batch 1 with N=50")
    sgs_seed = sgs_batch_seed(SOURCE_BATCH)
    if src_cfg["sgs_seed"] != sgs_seed:
        raise ValueError(f"source sgs_seed {src_cfg['sgs_seed']} != {sgs_seed}")

    cells = [c for c in src_cfg["cells"] if c["experiment"] == SOURCE_EXPERIMENT]
    if cells != build_cells(experiments=(SOURCE_EXPERIMENT,)):
        raise ValueError("source E4 cells differ from multi_truth_axes.build_cells(('E4',))")
    if len(cells) != 500 or len({c["config_key"] for c in cells}) != 500:
        raise ValueError("expected 500 distinct E4 cells")

    # Original kriging / sgs batch-1 run per cell (for provenance and the
    # paired sample / seed checks).
    reg = read_jsonl(src / "run_registry.jsonl")
    orig = {}
    for r in reg:
        if r["method"] in METHODS:
            orig.setdefault(r["config_key"], {})[r["method"]] = r
    for c in cells:
        o = orig.get(c["config_key"], {})
        if set(o) != set(METHODS):
            raise ValueError(f"source registry lacks kriging/sgs for {c['config_key']}")
        if o["sgs"]["method_seed"] != sgs_seed or o["sgs"]["n_ensemble"] != N_ENSEMBLE or o["sgs"]["batch"] != 1:
            raise ValueError(f"source sgs run of {c['config_key']} is not seed {sgs_seed} / N {N_ENSEMBLE}: {o['sgs']}")

    fits = pd.read_csv(resolve_run_dir(FITTED_VARIOGRAMS_CSV), dtype={"axis_level": str, "sample_rep": str})
    fits = fits[fits["experiment"] == SOURCE_EXPERIMENT]
    if fits["config_key"].duplicated().any():
        raise ValueError("duplicate config_key in fitted_variograms.csv E4 rows")
    fits = fits.set_index("config_key")
    out_cells = []
    for c in cells:
        if c["config_key"] not in fits.index:
            raise ValueError(f"no fitted variogram for {c['config_key']}")
        f = fits.loc[c["config_key"]]
        for k in ("axis_level", "sample_rep"):
            if str(f[k]) != str(c[k]):
                raise ValueError(f"{c['config_key']}: fitted {k}={f[k]} != {c[k]}")
        for k in ("truth_seed", "sample_seed"):
            if int(f[k]) != int(c[k]):
                raise ValueError(f"{c['config_key']}: fitted {k}={f[k]} != {c[k]}")
        if not (float(f["true_range_m"]) == c["hmaj1"] == c["hmin1"] and float(f["true_nugget"]) == c["nug"]):
            raise ValueError(f"{c['config_key']}: fitted true_range/true_nugget do not match the cell")
        if f["status"] != "ok":
            raise ValueError(f"{c['config_key']}: fit status {f['status']}")
        if resolve_run_dir(f["run_dir_kriging_b"]) != resolve_run_dir(orig[c["config_key"]]["kriging"]["run_dir"]):
            raise ValueError(f"{c['config_key']}: fitted_variograms run_dir_kriging_b != source registry kriging run")
        fit_range, fit_nug = float(f["fit_range_m"]), float(f["fit_nugget"])
        build_vario(hmaj1=fit_range, hmin1=fit_range, nug=fit_nug)  # validates nug in [0, 1]
        out_cells.append({
            **c,
            "vario_hmaj1": fit_range,
            "vario_hmin1": fit_range,
            "vario_nug": fit_nug,
            "orig_run_dir_kriging": orig[c["config_key"]]["kriging"]["run_dir"],
            "orig_run_dir_sgs_b1": orig[c["config_key"]]["sgs"]["run_dir"],
        })
    if limit_cells is not None:
        # smoke only: the K cheapest cells (smallest n_samples)
        out_cells = sorted(out_cells, key=lambda c: c["n_samples"])[: int(limit_cells)]

    vi_manifest = json.loads(resolve_run_dir(VARIOGRAM_INFERENCE_MANIFEST).read_text(encoding="utf-8"))
    return {
        "source_run": SOURCE_RUN,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_batch": SOURCE_BATCH,
        "methods": list(METHODS),
        "n_ensemble": N_ENSEMBLE,
        # Same keys as multi_truth_axes' run_config.json, so tools/run_dashboard.py
        # (reads experiments / batch / n_ensemble_this_batch / methods / cells) works unchanged.
        "experiments": [SOURCE_EXPERIMENT],
        "batch": SOURCE_BATCH,
        "n_ensemble_this_batch": N_ENSEMBLE,
        "sgs_seed": sgs_seed,
        "sgs_seed_note": "same sgsim seed as the original multi_truth_axes batch 1 (sgs_batch_seed(1)), "
                         "shared by all cells -> paired comparison with the oracle SGS runs",
        "variogram_source": {
            "file": FITTED_VARIOGRAMS_CSV,
            "file_sha256": _sha256(resolve_run_dir(FITTED_VARIOGRAMS_CSV)),
            "rows": f"experiment == '{SOURCE_EXPERIMENT}'",
            "columns_used": {"hmaj1": "fit_range_m", "hmin1": "fit_range_m", "nug": "fit_nugget"},
            "model": "build_vario(hmaj1=fit_range_m, hmin1=fit_range_m, nug=fit_nugget): isotropic, "
                     "spherical (it1=1), unit sill (cc1 = 1 - nug), azi1 = 0",
            "values_note": "automatic fits used as-is (user reviewed in the review UI and reset all to auto-fit)",
            "variogram_inference_manifest": VARIOGRAM_INFERENCE_MANIFEST,
            "variogram_inference_git_commit": vi_manifest.get("git_commit"),
            "variogram_inference_git_dirty": vi_manifest.get("git_dirty"),
            "variogram_inference_timestamp": vi_manifest.get("timestamp"),
            "variogram_inference_code_entrypoint": vi_manifest.get("code_entrypoint"),
        },
        "unchanged_vs_source": "truth (truth_seed, true hmaj1/hmin1/nug), conditioning samples (sample_seed, "
                               "n_samples), grid, kb2d/sgsim search settings, nscore handling, SGS N, SGS seed, "
                               "jitter seed -- only the method variogram differs",
        "raw_root": rel_or_abs(raw_root if Path(raw_root).is_absolute() else resolve_run_dir(raw_root)),
        "is_production_configuration": limit_cells is None,
        "n_cells": len(out_cells),
        "n_tasks_total": len(out_cells) * len(METHODS),
        "cells": out_cells,
    }


def _write_manifest(exp_dir: Path, config: dict, status: str, sessions: list, n_done: int, n_failures: int,
                    output_files: list) -> Path:
    params = {k: v for k, v in config.items()}
    params.update({"status": status, "sessions": sessions, "n_tasks_completed": n_done,
                   "n_task_failures_logged": n_failures})
    seeds = {
        "truth_seeds": sorted({c["truth_seed"] for c in config["cells"]}),
        "sample_seeds": sorted({c["sample_seed"] for c in config["cells"]}),
        "sgs_seed": config["sgs_seed"],
        "fixed_in_method_code": "sgs jitter seed 70, kriging MC seed 80, backtr validation seed 85",
    }
    return save_result(experiment=EXPERIMENT_NAME, params=params, seed_or_seeds=seeds, run_dir=exp_dir,
                       code_entrypoint=CODE_ENTRYPOINT, output_files=output_files)


def run(config: dict, exp_dir: Path, max_workers: int) -> int:
    registry_path = exp_dir / "run_registry.jsonl"
    failures_path = exp_dir / "failures.jsonl"
    progress_path = exp_dir / "progress.log"
    output_files = ["run_config.json", "run_registry.jsonl", "failures.jsonl", "progress.log", "source_runs.json"]
    cells = config["cells"]
    runs_root = exp_dir / "runs"

    registry = read_jsonl(registry_path)
    done = {(r["config_key"], r["method"]) for r in registry}
    for r in registry:
        if not (resolve_run_dir(r["run_dir"]) / "manifest.json").exists():
            raise FileNotFoundError(f"registry entry without manifest: {r}")
    n_total = config["n_tasks_total"]

    prev = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8")) \
        if (exp_dir / "manifest.json").exists() else {}
    sessions = list(prev.get("params", {}).get("sessions", []))
    git_dirty, git_dirty_files = get_git_status()
    session = {"started_utc": _now(), "git_commit": get_git_commit(), "git_dirty": git_dirty,
               "git_dirty_files": git_dirty_files, "max_workers": max_workers, "os_cpu_count": os.cpu_count(),
               "n_tasks_already_done": len(done)}
    sessions.append(session)
    (exp_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    registry_path.touch()
    failures_path.touch()
    _write_manifest(exp_dir, config, "in_progress", sessions, len(done), len(read_jsonl(failures_path)),
                    output_files)

    # Longest tasks first (SGS cost grows with n_samples), for better pool packing.
    ordered = sorted(cells, key=lambda c: -c["n_samples"])
    level_kwargs = {}
    for c in ordered:
        level_kwargs[c["config_key"]] = dict(
            truth_seed=c["truth_seed"], sample_seed=c["sample_seed"], hmaj1=c["hmaj1"], hmin1=c["hmin1"],
            nug=c["nug"], n_samples=c["n_samples"], output_dir=str(runs_root),
            vario_override=build_vario(hmaj1=c["vario_hmaj1"], hmin1=c["vario_hmin1"], nug=c["vario_nug"]),
        )
    method_kwargs = {k: {"sgs": {"sgs_seed": config["sgs_seed"]}} for k in level_kwargs}

    t0 = time.time()
    state = {"done": 0, "failed": 0}
    remaining = {m: sum(1 for k in level_kwargs if (k, m) not in done) for m in METHODS}
    secs: Dict[str, List[float]] = {}

    def _log(msg):
        line = f"{_now()} {msg}"
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def on_done(key, method, run_dir):
        p = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))["params"]
        entry = {"config_key": key, "method": method,
                 "batch": SOURCE_BATCH if method == "sgs" else None,
                 "n_ensemble": p.get("n_realizations") if method == "sgs" else None,
                 "method_seed": p.get("sgs_seed") if method == "sgs" else None,
                 "variogram": p["variogram"],
                 "sgs_grid_node_warnings": (p.get("sgsim_warnings") or {}).get("n_grid_node_warnings")
                 if method == "sgs" else None,
                 "run_dir": rel_or_abs(run_dir), "total_seconds": p.get("total_seconds"),
                 "completed_utc": _now()}
        _append_jsonl(registry_path, entry)
        state["done"] += 1
        remaining[method] -= 1
        secs.setdefault(method, []).append(float(entry["total_seconds"] or 0.0))
        allv = [s for v in secs.values() for s in v]
        overall = sum(allv) / len(allv)
        est = sum(n * (sum(secs[m]) / len(secs[m]) if secs.get(m) else overall) for m, n in remaining.items())
        _log(f"[{len(done) + state['done']}/{n_total}] done {method:8s} {key} ({entry['total_seconds']:.0f}s) "
             f"| elapsed {_fmt_seconds(time.time() - t0)} | ETA ~{_fmt_seconds(est / max_workers)} "
             f"| failures {state['failed']}")

    def on_error(key, method, exc):
        state["failed"] += 1
        _append_jsonl(failures_path, {
            "config_key": key, "method": method, "error": repr(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "failed_utc": _now()})
        _log(f"FAILED {method} {key}: {exc!r}")

    _log(f"session start: tasks total={n_total}, already done={len(done)}, workers={max_workers}")
    status = "incomplete"
    try:
        run_levels_parallel(level_kwargs, methods=METHODS, max_workers=max_workers, n_ensemble=config["n_ensemble"],
                            method_kwargs=method_kwargs, skip=done, on_done=on_done, on_error=on_error)
    finally:
        registry = read_jsonl(registry_path)
        n_done = len({(r["config_key"], r["method"]) for r in registry})
        status = "complete" if n_done == n_total else "incomplete"
        session.update({"ended_utc": _now(), "wall_seconds": time.time() - t0,
                        "completed_this_session": state["done"], "failed_this_session": state["failed"]})
        (exp_dir / "source_runs.json").write_text(json.dumps(build_source_runs(cells, registry), indent=2),
                                                  encoding="utf-8")
        mp = _write_manifest(exp_dir, config, status, sessions, n_done, len(read_jsonl(failures_path)),
                             output_files)
        _log(f"session end: status={status} completed={n_done}/{n_total} failures this session={state['failed']} "
             f"manifest={mp}")
    return 0 if status == "complete" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-root", default="results/raw")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--limit-cells", type=int, default=None, help="smoke tests only: the K cells with the smallest n_samples")
    args = ap.parse_args(argv)
    max_workers = args.max_workers if args.max_workers is not None else default_max_workers()
    if args.resume:
        exp_dir = resolve_run_dir(args.resume)
        config = json.loads((exp_dir / "run_config.json").read_text(encoding="utf-8"))
        print(f"resuming {exp_dir}")
    else:
        config = make_config(args.raw_root, args.limit_cells)
        exp_dir = make_run_dir(EXPERIMENT_NAME, args.raw_root)
        print(f"new run folder {exp_dir}")
    return run(config, exp_dir, max_workers)


if __name__ == "__main__":
    raise SystemExit(main())
