"""E2-E5 -- multi-truth re-run of the range / nugget / density axes and the
base case with a batched (appendable) SGS / RBF+bootstrap ensemble
(user-approved 2026-11-16: stabilize storyboard figures F1-F4).

What changes relative to the existing axis experiments
------------------------------------------------------
* 10 ground-truth realizations per axis level instead of 1 (TRUTH_SEEDS
  below), so every level-to-level difference can be read against the
  truth-to-truth spread.
* The two ENSEMBLE methods (SGS realizations, RBF bootstrap replicates) are
  run in BATCHES of N_PER_BATCH = 50 members. Batch 1 is the first
  production batch; further batches (k = 2, 3, ...) can be appended later
  WITHOUT touching batch 1 -- each batch is its own run directory with its
  own batch seed, and the evaluation script
  (src/experiments/evaluate_multi_truth_axes.py) concatenates whichever
  batches it is told to.
* Everything else (truth generator, sampling rule, true-variogram kriging,
  the automatic RBF CV / GP marginal-likelihood procedures, every method
  constant incl. the project-default N_BOOTSTRAP = N_REALIZATIONS = 10) is
  UNCHANGED. The ensemble size is passed at the call site through
  ``_axis_parallel.run_levels_parallel(n_ensemble=...)``, which injects the
  SAME value into sgs and rbf_bootstrap.

Experiments (cells = (case, axis_level, truth_seed, sample_rep))
---------------------------------------------------------------
E5 base_case             : base-case settings (range 300 m, nug 0.05,
                           N_SAMPLES=125 requested, SAMPLE_SEED=20),
                           truth_seed 101 only -> 1 cell.
E2 range_axis            : range_axis.ALL_RANGE_VALUES (8 levels) x 10 truths,
                           nug = base 0.05, SAMPLE_SEED=20 -> 80 cells.
E3 nugget_axis           : nugget_axis.ALL_NUGGET_VALUES (6 levels) x 10
                           truths, range 300 m, SAMPLE_SEED=20 -> 60 cells.
E4 sample_replicate_axis : sample_density_axis.EXTENDED_AXIS_LEVELS
                           (20/10/5/2/1 -> n_requested 500/250/125/50/25) x
                           10 truths x 10 sample seeds (1001-1010, the
                           existing replicate seeds) -> 500 cells; range
                           300 m, nug 0.05.
x 4 methods each. Execution order of the single process pool follows the
order E5 -> E2 -> E3 -> E4 (tasks are submitted in that order, so the small
experiments finish first and can be evaluated while E4 is still running).

RUN SHARING (E5 == E2 range=300 m, truth 101): the base case IS the
range=300 m level at truth_seed=101 (identical truth_seed, sample_seed,
range, nugget, n_samples). Tasks are keyed by their full configuration
(``config_key``), so that configuration is run ONCE and both cells point at
the same run directories -- the same sharing the project already used
("base case = range300 = density5% share one run", docs/progress.md
2026-09-17). Recorded in source_runs.json (both cells list the same dirs).

Truth seeds (reused, NOT new)
-----------------------------
TRUTH_SEEDS = (101, 7101, ..., 7109) = the TRUTH_SEED_REPLICATES of
src/experiments/gp_realization_check_range_axis.py and
gp_realization_check_nugget_axis.py (asserted equal at import), i.e. the
realizations behind results/processed/{range,nugget}_axis/
gp_hyperparameters_by_realization.csv (figure F4). Since the truth
generator, SAMPLE_SEED and N_SAMPLES are the same, the range/nugget cells
here reproduce F4's ground truths and conditioning data exactly (the
evaluation script cross-checks the fitted GP hyperparameters against that
CSV). The density axis uses the same 10 truth seeds.

Batch seeds (new; asserted not to collide with any existing project seed)
------------------------------------------------------------------------
batch k (1 <= k <= MAX_BATCH = 99):
    sgs_seed       = SGS_BATCH_SEED_BASE       + k = 4100 + k  (4101, 4102, ...)
    bootstrap_seed = BOOTSTRAP_BATCH_SEED_BASE + k = 4200 + k  (4201, 4202, ...)
The batch seed is the same for every cell of a batch (the project's existing
convention: SGS_SEED=60 / BOOTSTRAP_SEED=30 were likewise shared across all
cells of an axis). SGS_JITTER_SEED (70) and the RBF CV_SEED (40) are part of
each method's fixed configuration and are NOT varied by batch -- in
particular the RBF CV fit, and hence best_epsilon / best_smoothing /
point_estimate_map, are identical in every batch (asserted at evaluation).
kriging and gp_mle have no ensemble: they are run once, in batch 1 only.

Output layout (results/raw convention)
--------------------------------------
Each method run goes to its usual results/raw/<method>/<timestamp>/ (with
its own manifest). This invocation's folder
results/raw/multi_truth_axes/<timestamp>/ holds:
  run_config.json     : the full cell list / batch / seeds / N (written at
                        start; used by --resume and by the evaluation).
  run_registry.jsonl  : one line per COMPLETED method run (append-only,
                        flushed per task) -- the resume ledger.
  failures.jsonl      : one line per failed task (retried on --resume).
  progress.log        : completed / total, elapsed, ETA per completion.
  source_runs.json    : {case: {axis_level: {truth_seed: {sample_rep:
                        {method: run_dir}}}}} for this invocation's batch.
  manifest.json       : src.io.save_result (written at start with status
                        "in_progress", rewritten at the end of every session).

Resume: re-invoke with ``--resume <that folder>``: the stored run_config is
used (cell/batch CLI options are ignored), tasks already in the registry are
skipped, everything else (incl. failures) is (re-)submitted.

What an interruption leaves behind (be precise -- results/raw is never
cleaned up): if the process tree is killed outright (observed in the smoke
test: the pool's workers died with it), a worker killed mid-task leaves at
most a partial
results/raw/<method>/<ts>/ folder (make_run_dir is called only at save time).
If instead the parent raises (Ctrl+C, an error inside the registry/progress
callbacks, a re-raised task error), run_levels_parallel cancels every
not-yet-started task, but tasks already started or already pre-queued by the
executor (at most 2 * max_workers + 1) still RUN TO COMPLETION and write a
complete run folder that is NOT in run_registry.jsonl. Such orphan folders
are never referenced by source_runs.json / the evaluation; on --resume the
same configuration is simply run again into a new folder.

Usage
-----
Production batch 1 (E5 -> E2 -> E3 -> E4, N=50):
  .venv/Scripts/python.exe -m src.experiments.multi_truth_axes --batch 1 --max-workers 20
Resume an interrupted invocation:
  .venv/Scripts/python.exe -m src.experiments.multi_truth_axes --resume results/raw/multi_truth_axes/<ts> --max-workers 20
Append batch 2 later (SGS + RBF only, new folder):
  .venv/Scripts/python.exe -m src.experiments.multi_truth_axes --batch 2 --max-workers 20
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.experiments import ensemble_size_convergence as _e1
from src.experiments import rbf_bootstrap, sgs
from src.experiments._axis_parallel import (
    DEFAULT_METHODS,
    ENSEMBLE_METHODS,
    default_max_workers,
    run_levels_parallel,
)
from src.experiments.base_case_conditioning import (
    HMAJ1,
    HMIN1,
    NUG,
    N_SAMPLES,
    SAMPLE_SEED,
    TRUTH_SEED,
)
from src.experiments.gp_mle import GP_RANDOM_STATE, GP_SAMPLE_SEED
from src.experiments.gp_realization_check_nugget_axis import (
    TRUTH_SEED_REPLICATES as _NUGGET_GP_TRUTH_SEEDS,
)
from src.experiments.gp_realization_check_range_axis import (
    TRUTH_SEED_REPLICATES as _RANGE_GP_TRUTH_SEEDS,
)
from src.experiments.kriging import BACKTR_VALIDATION_SEED, KRIGING_VAR_MC_SEED
from src.experiments.nugget_axis import ALL_NUGGET_VALUES
from src.experiments.nugget_axis import level_key as nugget_level_key
from src.experiments.range_axis import ALL_RANGE_VALUES
from src.experiments.sample_density_axis import EXTENDED_AXIS_LEVELS, SAMPLE_COUNTS
from src.experiments.sample_replicate_axis import (
    AXIS_HMAJ1 as DENSITY_HMAJ1,
    AXIS_HMIN1 as DENSITY_HMIN1,
    REPLICATE_IDS,
    REPLICATE_SEED,
)
from src.io import get_git_commit, get_git_status, make_run_dir, save_result

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXPERIMENT_NAME = "multi_truth_axes"
CODE_ENTRYPOINT = "src/experiments/multi_truth_axes.py"

# --- Ensemble batching ----------------------------------------------------
N_PER_BATCH = 50
SGS_BATCH_SEED_BASE = 4100
BOOTSTRAP_BATCH_SEED_BASE = 4200
MAX_BATCH = 99

# --- Ground-truth realizations (reused from the F4 GP-realization check) --
TRUTH_SEEDS = tuple(int(s) for s in _RANGE_GP_TRUTH_SEEDS)
if TRUTH_SEEDS != tuple(int(s) for s in _NUGGET_GP_TRUTH_SEEDS):
    raise ValueError("range/nugget GP-realization checks no longer share one truth-seed block.")
if TRUTH_SEEDS[0] != TRUTH_SEED or len(TRUTH_SEEDS) != 10:
    raise ValueError(f"unexpected TRUTH_SEEDS {TRUTH_SEEDS}")

# Sample-rep label for the cells that use the base SAMPLE_SEED (E2/E3/E5).
BASE_SAMPLE_REP = "base"

EXPERIMENTS = ("E5", "E2", "E3", "E4")  # execution order
EXPERIMENT_CASE = {
    "E5": ("base_case", "none"),
    "E2": ("range_axis", "range"),
    "E3": ("nugget_axis", "nugget_normalized"),
    "E4": ("sample_replicate_axis", "sample_seed_replicate"),
}
METHOD_SEED_KW = {"sgs": "sgs_seed", "rbf_bootstrap": "bootstrap_seed"}


def sgs_batch_seed(batch: int) -> int:
    _check_batch(batch)
    return SGS_BATCH_SEED_BASE + int(batch)


def bootstrap_batch_seed(batch: int) -> int:
    _check_batch(batch)
    return BOOTSTRAP_BATCH_SEED_BASE + int(batch)


def batch_seed(method: str, batch: int) -> int:
    return sgs_batch_seed(batch) if method == "sgs" else bootstrap_batch_seed(batch)


def _check_batch(batch: int) -> None:
    if not (1 <= int(batch) <= MAX_BATCH):
        raise ValueError(f"batch must be in 1..{MAX_BATCH}; got {batch}")


# Every pinned seed constant of the project (grep of src/ and
# results/processed/ on 2026-11-16, extending E1's list with E1's own seeds,
# the anisotropy experiments' 201-210 / 301-305 / 4001-4005 and the truth
# block 7101-7109). The full batch-seed ranges must not touch any of them.
_EXISTING_SEEDS = (
    set(_e1._EXISTING_SEEDS)
    | set(_e1.SGS_SEEDS) | set(_e1.BOOTSTRAP_SEEDS)
    | {sgs.SGS_SEED, sgs.SGS_JITTER_SEED, rbf_bootstrap.BOOTSTRAP_SEED, rbf_bootstrap.CV_SEED,
       GP_RANDOM_STATE, GP_SAMPLE_SEED, KRIGING_VAR_MC_SEED, BACKTR_VALIDATION_SEED,
       SAMPLE_SEED, 86, 12345}
    | set(TRUTH_SEEDS)
    | set(REPLICATE_SEED.values())
    | set(range(201, 211)) | set(range(301, 306)) | set(range(4001, 4006))
    | set(range(9000, 9500))  # GP-draw seed families 9000/9100/.../9400
)
_ALL_BATCH_SEEDS = (
    [SGS_BATCH_SEED_BASE + k for k in range(1, MAX_BATCH + 1)]
    + [BOOTSTRAP_BATCH_SEED_BASE + k for k in range(1, MAX_BATCH + 1)]
)
if len(set(_ALL_BATCH_SEEDS)) != len(_ALL_BATCH_SEEDS):
    raise ValueError("SGS and bootstrap batch-seed ranges overlap.")
if set(_ALL_BATCH_SEEDS) & _EXISTING_SEEDS:
    raise ValueError(f"batch seeds collide with existing seeds: {sorted(set(_ALL_BATCH_SEEDS) & _EXISTING_SEEDS)}")
if set(ENSEMBLE_METHODS) != set(METHOD_SEED_KW):
    raise ValueError(f"ENSEMBLE_METHODS {ENSEMBLE_METHODS} != {tuple(METHOD_SEED_KW)}")


# --- Cells ------------------------------------------------------------------
def config_key(truth_seed: int, sample_seed: int, hmaj1: float, hmin1: float, nug: float, n_samples: int) -> str:
    """Full run configuration of one (truth, samples) setting -- the identity
    under which a method run is executed and registered. Two cells with the
    same key (E5 and E2 range=300 m truth 101) share their runs."""
    return (f"t{int(truth_seed)}_s{int(sample_seed)}_h{float(hmaj1)!r}x{float(hmin1)!r}"
            f"_nug{float(nug)!r}_n{int(n_samples)}")


def _cell(experiment, axis_level, truth_seed, sample_rep, sample_seed, hmaj1, hmin1, nug, n_samples) -> dict:
    case, axis = EXPERIMENT_CASE[experiment]
    return {
        "experiment": experiment,
        "case": case,
        "axis": axis,
        "axis_level": str(axis_level),
        "truth_seed": int(truth_seed),
        "sample_rep": str(sample_rep),
        "sample_seed": int(sample_seed),
        "hmaj1": float(hmaj1),
        "hmin1": float(hmin1),
        "nug": float(nug),
        "n_samples": int(n_samples),
        "config_key": config_key(truth_seed, sample_seed, hmaj1, hmin1, nug, n_samples),
    }


def build_cells(
    experiments=EXPERIMENTS,
    truth_seeds=TRUTH_SEEDS,
    range_levels: Optional[List[float]] = None,
    nugget_levels: Optional[List[float]] = None,
    density_levels: Optional[List[str]] = None,
    sample_reps: Optional[List[str]] = None,
) -> List[dict]:
    """All cells of the selected experiments, in execution order. The
    optional level / replicate subsets exist for smoke tests only; the
    production defaults are the full axes."""
    range_levels = list(ALL_RANGE_VALUES) if range_levels is None else [float(r) for r in range_levels]
    nugget_levels = list(ALL_NUGGET_VALUES) if nugget_levels is None else [float(v) for v in nugget_levels]
    density_levels = list(EXTENDED_AXIS_LEVELS) if density_levels is None else [str(v) for v in density_levels]
    sample_reps = list(REPLICATE_IDS) if sample_reps is None else [str(v) for v in sample_reps]
    for r in range_levels:
        if r not in ALL_RANGE_VALUES:
            raise ValueError(f"range level {r} not in {ALL_RANGE_VALUES}")
    for v in nugget_levels:
        if v not in ALL_NUGGET_VALUES:
            raise ValueError(f"nugget level {v} not in {ALL_NUGGET_VALUES}")
    for v in density_levels:
        if v not in EXTENDED_AXIS_LEVELS:
            raise ValueError(f"density level {v} not in {EXTENDED_AXIS_LEVELS}")
    for v in sample_reps:
        if v not in REPLICATE_SEED:
            raise ValueError(f"sample rep {v} not in {list(REPLICATE_SEED)}")
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"unknown experiments {unknown}")

    cells = []
    for exp in EXPERIMENTS:  # fixed execution order, independent of CLI order
        if exp not in experiments:
            continue
        if exp == "E5":
            cells.append(_cell("E5", "base", TRUTH_SEED, BASE_SAMPLE_REP, SAMPLE_SEED, HMAJ1, HMIN1, NUG, N_SAMPLES))
        elif exp == "E2":
            for r in range_levels:
                for t in truth_seeds:
                    cells.append(_cell("E2", str(int(r)), t, BASE_SAMPLE_REP, SAMPLE_SEED, r, r, NUG, N_SAMPLES))
        elif exp == "E3":
            for v in nugget_levels:
                for t in truth_seeds:
                    cells.append(_cell("E3", nugget_level_key(v), t, BASE_SAMPLE_REP, SAMPLE_SEED,
                                       HMAJ1, HMIN1, v, N_SAMPLES))
        elif exp == "E4":
            for lvl in density_levels:
                for t in truth_seeds:
                    for rep in sample_reps:
                        cells.append(_cell("E4", lvl, t, rep, REPLICATE_SEED[rep],
                                           DENSITY_HMAJ1, DENSITY_HMIN1, NUG, SAMPLE_COUNTS[lvl]))
    return cells


def unique_configs(cells: List[dict]) -> Dict[str, dict]:
    """config_key -> method-main kwargs, first-seen order (= execution order)."""
    out = {}
    for c in cells:
        out.setdefault(c["config_key"], dict(
            truth_seed=c["truth_seed"], sample_seed=c["sample_seed"], hmaj1=c["hmaj1"],
            hmin1=c["hmin1"], nug=c["nug"], n_samples=c["n_samples"],
        ))
    return out


def methods_for_batch(batch: int):
    """Batch 1 runs all four methods; later batches only add ensemble members."""
    return tuple(DEFAULT_METHODS) if int(batch) == 1 else tuple(ENSEMBLE_METHODS)


# --- Bookkeeping helpers ------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel_or_abs(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_run_dir(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else _REPO_ROOT / p


def _append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _fmt_seconds(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def build_source_runs(cells: List[dict], registry: List[dict]) -> dict:
    by_key = {(r["config_key"], r["method"]): r["run_dir"] for r in registry}
    out: dict = {}
    for c in cells:
        node = (out.setdefault(c["case"], {}).setdefault(c["axis_level"], {})
                .setdefault(str(c["truth_seed"]), {}).setdefault(c["sample_rep"], {}))
        for (key, method), run_dir in by_key.items():
            if key == c["config_key"]:
                node[method] = run_dir
    return out


def _write_manifest(exp_dir: Path, config: dict, status: str, sessions: list, registry: list,
                    failures: list, output_files: list) -> Path:
    batch = config["batch"]
    params = {
        **config,
        "status": status,
        "sessions": sessions,
        "n_tasks_total": config["n_tasks_total"],
        "n_tasks_completed": len(registry),
        "n_task_failures_logged": len(failures),
    }
    seeds = {
        "truth_seeds": config["truth_seeds"],
        "sample_seed_base": SAMPLE_SEED,
        "density_sample_seeds": {r: REPLICATE_SEED[r] for r in config["sample_reps"]},
        "sgs_seed": config["sgs_seed"],
        "bootstrap_seed": config["bootstrap_seed"],
        "fixed_not_varied_by_batch": config["fixed_seeds_not_varied"],
    }
    return save_result(
        experiment=EXPERIMENT_NAME,
        params=params,
        seed_or_seeds=seeds,
        run_dir=exp_dir,
        code_entrypoint=CODE_ENTRYPOINT,
        output_files=output_files,
    )


# --- Main -------------------------------------------------------------------
def make_config(args) -> dict:
    batch = int(args.batch)
    _check_batch(batch)
    n_ensemble = int(args.n_ensemble)
    if n_ensemble < 2:
        raise ValueError("n_ensemble must be >= 2")
    experiments = tuple(args.experiments)
    truth_seeds = tuple(int(t) for t in (args.truth_seeds or TRUTH_SEEDS))
    cells = build_cells(
        experiments=experiments,
        truth_seeds=truth_seeds,
        range_levels=args.range_levels,
        nugget_levels=args.nugget_levels,
        density_levels=args.density_levels,
        sample_reps=args.sample_reps,
    )
    configs = unique_configs(cells)
    methods = methods_for_batch(batch)
    is_production = (
        n_ensemble == N_PER_BATCH and truth_seeds == TRUTH_SEEDS and args.range_levels is None
        and args.nugget_levels is None and args.density_levels is None and args.sample_reps is None
    )
    return {
        "batch": batch,
        "n_ensemble_this_batch": n_ensemble,
        "n_per_batch_production": N_PER_BATCH,
        "is_production_configuration": bool(is_production),
        "project_default_n_ensemble": rbf_bootstrap.N_BOOTSTRAP,
        "methods": list(methods),
        "sgs_seed": sgs_batch_seed(batch),
        "bootstrap_seed": bootstrap_batch_seed(batch),
        "batch_seed_rule": {
            "sgs_seed": f"{SGS_BATCH_SEED_BASE} + batch",
            "bootstrap_seed": f"{BOOTSTRAP_BATCH_SEED_BASE} + batch",
            "max_batch": MAX_BATCH,
            "shared_across_cells": True,
        },
        "fixed_seeds_not_varied": {
            "sgs_jitter_seed": sgs.SGS_JITTER_SEED,
            "rbf_cv_seed": rbf_bootstrap.CV_SEED,
            "gp_random_state": GP_RANDOM_STATE,
            "gp_sample_seed": GP_SAMPLE_SEED,
            "kriging_var_mc_seed": KRIGING_VAR_MC_SEED,
            "kriging_backtr_validation_seed": BACKTR_VALIDATION_SEED,
        },
        "experiments": [e for e in EXPERIMENTS if e in experiments],
        "truth_seeds": list(truth_seeds),
        "truth_seed_source": "src/experiments/gp_realization_check_range_axis.py TRUTH_SEED_REPLICATES "
                             "(== gp_realization_check_nugget_axis.py)",
        "range_levels": args.range_levels,
        "nugget_levels": args.nugget_levels,
        "density_levels": args.density_levels,
        "sample_reps": list(args.sample_reps) if args.sample_reps is not None else list(REPLICATE_IDS),
        "raw_root": rel_or_abs(args.raw_root if Path(args.raw_root).is_absolute() else _REPO_ROOT / args.raw_root),
        "n_cells": len(cells),
        "n_unique_configs": len(configs),
        "n_tasks_total": len(configs) * len(methods),
        "cells": cells,
    }


def run(config: dict, exp_dir: Path, max_workers: int) -> int:
    registry_path = exp_dir / "run_registry.jsonl"
    failures_path = exp_dir / "failures.jsonl"
    progress_path = exp_dir / "progress.log"
    config_path = exp_dir / "run_config.json"
    output_files = ["run_config.json", "run_registry.jsonl", "failures.jsonl", "progress.log",
                    "source_runs.json"]

    cells = config["cells"]
    configs = unique_configs(cells)
    methods = tuple(config["methods"])
    batch = config["batch"]
    raw_root = str(resolve_run_dir(config["raw_root"]))

    registry = read_jsonl(registry_path)
    failures = read_jsonl(failures_path)
    done = {(r["config_key"], r["method"]) for r in registry}
    for r in registry:
        if not (resolve_run_dir(r["run_dir"]) / "manifest.json").exists():
            raise FileNotFoundError(f"registry entry without manifest: {r}")
    n_total = len(configs) * len(methods)

    manifest_payload = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8")) \
        if (exp_dir / "manifest.json").exists() else {}
    sessions = list(manifest_payload.get("params", {}).get("sessions", []))
    git_dirty, git_dirty_files = get_git_status()
    session = {"started_utc": _now(), "git_commit": get_git_commit(), "git_dirty": git_dirty,
               "git_dirty_files": git_dirty_files, "max_workers": max_workers,
               "os_cpu_count": os.cpu_count(), "n_tasks_already_done": len(done)}
    sessions.append(session)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    registry_path.touch()
    failures_path.touch()
    _write_manifest(exp_dir, config, "in_progress", sessions, registry, failures, output_files)

    level_kwargs = {k: dict(kw, output_dir=raw_root) for k, kw in configs.items()}
    method_kwargs = {
        k: {m: {METHOD_SEED_KW[m]: batch_seed(m, batch)} for m in methods if m in METHOD_SEED_KW}
        for k in configs
    }
    t0 = time.time()
    state = {"done_session": 0, "failed_session": 0}
    remaining_by_method = {m: sum(1 for k in configs if (k, m) not in done) for m in methods}
    secs_by_method: Dict[str, List[float]] = {}

    def _log(msg: str) -> None:
        line = f"{_now()} {msg}"
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def on_done(key, method, run_dir):
        manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
        p = manifest["params"]
        entry = {
            "config_key": key, "method": method,
            "batch": batch if method in ENSEMBLE_METHODS else None,
            "n_ensemble": (p.get("n_realizations") if method == "sgs"
                           else p.get("n_bootstrap") if method == "rbf_bootstrap" else None),
            "method_seed": p.get(METHOD_SEED_KW[method]) if method in METHOD_SEED_KW else None,
            "run_dir": rel_or_abs(run_dir),
            "total_seconds": p.get("total_seconds"),
            "completed_utc": _now(),
        }
        _append_jsonl(registry_path, entry)
        state["done_session"] += 1
        n_done = len(done) + state["done_session"]
        elapsed = time.time() - t0
        remaining_by_method[method] -= 1
        secs_by_method.setdefault(method, []).append(float(entry["total_seconds"] or 0.0))
        # ETA: remaining tasks x mean observed seconds of THAT method (methods
        # not yet observed use the overall mean), divided by the pool size.
        all_secs = [s for v in secs_by_method.values() for s in v]
        overall = float(sum(all_secs) / len(all_secs))
        est_cpu = sum(
            n_rem * (sum(secs_by_method[m]) / len(secs_by_method[m]) if secs_by_method.get(m) else overall)
            for m, n_rem in remaining_by_method.items()
        )
        eta = est_cpu / max_workers
        _log(f"[{n_done}/{n_total}] done {method:13s} {key} "
             f"({entry['total_seconds']:.0f}s) | session elapsed {_fmt_seconds(elapsed)} "
             f"| ETA ~{_fmt_seconds(eta)} (remaining {sum(remaining_by_method.values())} tasks x per-method "
             f"mean / {max_workers} workers) | failures this session {state['failed_session']}")

    def on_error(key, method, exc):
        state["failed_session"] += 1
        _append_jsonl(failures_path, {
            "config_key": key, "method": method, "batch": batch, "error": repr(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "failed_utc": _now(),
        })
        _log(f"FAILED {method} {key}: {exc!r}")

    _log(f"session start: batch={batch} N={config['n_ensemble_this_batch']} methods={list(methods)} "
         f"tasks total={n_total}, already done={len(done)}, to run={n_total - len(done)}, "
         f"workers={max_workers}")
    try:
        run_levels_parallel(
            level_kwargs,
            methods=methods,
            max_workers=max_workers,
            n_ensemble=config["n_ensemble_this_batch"],
            method_kwargs=method_kwargs,
            skip=done,
            on_done=on_done,
            on_error=on_error,
        )
    finally:
        registry = read_jsonl(registry_path)
        failures = read_jsonl(failures_path)
        n_done = len({(r["config_key"], r["method"]) for r in registry})
        status = "complete" if n_done == n_total else "incomplete"
        session.update({"ended_utc": _now(), "wall_seconds": time.time() - t0,
                        "completed_this_session": state["done_session"],
                        "failed_this_session": state["failed_session"]})
        (exp_dir / "source_runs.json").write_text(
            json.dumps(build_source_runs(cells, registry), indent=2), encoding="utf-8")
        manifest_path = _write_manifest(exp_dir, config, status, sessions, registry, failures, output_files)
        _log(f"session end: status={status} completed={n_done}/{n_total} "
             f"failures this session={state['failed_session']} manifest={manifest_path}")
    return 0 if status == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n-ensemble", type=int, default=N_PER_BATCH,
                        help=f"members per batch (production: {N_PER_BATCH}; smaller only for smoke tests)")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--truth-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--range-levels", nargs="+", type=float, default=None)
    parser.add_argument("--nugget-levels", nargs="+", type=float, default=None)
    parser.add_argument("--density-levels", nargs="+", default=None)
    parser.add_argument("--sample-reps", nargs="+", default=None)
    parser.add_argument("--raw-root", default="results/raw",
                        help="base directory for method runs AND this experiment's folder")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--resume", default=None, help="existing multi_truth_axes run folder to continue")
    args = parser.parse_args(argv)

    max_workers = args.max_workers if args.max_workers is not None else default_max_workers()
    if args.resume:
        exp_dir = resolve_run_dir(args.resume)
        config = json.loads((exp_dir / "run_config.json").read_text(encoding="utf-8"))
        print(f"resuming {exp_dir} (stored run_config used; cell/batch CLI options ignored)")
    else:
        config = make_config(args)
        exp_dir = make_run_dir(EXPERIMENT_NAME, args.raw_root)
        print(f"new run folder {exp_dir}")
    return run(config, exp_dir, max_workers)


if __name__ == "__main__":
    raise SystemExit(main())
