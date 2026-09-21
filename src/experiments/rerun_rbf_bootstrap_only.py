"""Re-run ONLY the RBF+bootstrap runs of the base case and the three axes
(range / sample density / sample-seed replicate), and re-point ONLY the
``rbf_bootstrap`` entries of each processed ``source_runs.json`` at the new
runs.

Why this exists
---------------
Each axis runner's ``main()`` re-executes all 4 methods (kriging / SGS /
RBF+bootstrap / GP-MLE). When only RBF's own tuning grid changes
(2026-09-17 ``EPSILON_GRID``; 2026-09-21 ``SMOOTHING_GRID``), the other three
methods' pinned runs must stay untouched (results/raw is immutable, seeds and
sample locations unchanged). This script builds exactly the same per-level
kwargs the axis runners build, but submits ``methods=("rbf_bootstrap",)`` to
``src.experiments._axis_parallel.run_levels_parallel``.

Runs (40 total): base case 1 (= range300 = density5%, one shared run),
range axis 7 (100/200/400/500/600/700/800), density axis 2 (2%, 1%), replicate
axis 30 (3 density levels x 10 sample_seed replicates).

Run with: .venv/Scripts/python.exe -m src.experiments.rerun_rbf_bootstrap_only
"""

import json
import time
from pathlib import Path

from src.experiments._axis_parallel import run_levels_parallel
from src.experiments.base_case_conditioning import TRUTH_SEED
from src.experiments.range_axis import ALL_RANGE_VALUES
from src.experiments.sample_density_axis import (
    AXIS_HMAJ1,
    AXIS_HMIN1,
    SAMPLE_COUNTS,
)
from src.experiments.sample_replicate_axis import (
    REPLICATE_IDS,
    REPLICATE_SEED,
    verify_run_against_replicate,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = _REPO_ROOT / "results" / "processed"

METHOD = "rbf_bootstrap"
BASE_CASE_RANGE = 300.0
BASE_KEY = "base"


def _rel(p) -> str:
    return str(Path(p).relative_to(_REPO_ROOT)).replace("\\", "/")


def build_level_kwargs() -> dict:
    level_kwargs = {BASE_KEY: {}}  # rbf_bootstrap.main() defaults == base case
    for r in ALL_RANGE_VALUES:
        if r == BASE_CASE_RANGE:
            continue  # shares the base-case run
        level_kwargs[f"range__{int(r)}"] = dict(truth_seed=TRUTH_SEED, hmaj1=r, hmin1=r)
    for lvl in ("2", "1"):  # "5" == base case
        level_kwargs[f"density__{lvl}"] = dict(
            truth_seed=TRUTH_SEED,
            hmaj1=AXIS_HMAJ1,
            hmin1=AXIS_HMIN1,
            n_samples=SAMPLE_COUNTS[lvl],
        )
    for lvl in ("5", "2", "1"):
        for rep_id, seed in REPLICATE_SEED.items():
            level_kwargs[f"replicate__{lvl}__{rep_id}"] = dict(
                truth_seed=TRUTH_SEED,
                hmaj1=AXIS_HMAJ1,
                hmin1=AXIS_HMIN1,
                n_samples=SAMPLE_COUNTS[lvl],
                sample_seed=seed,
            )
    return level_kwargs


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    t0 = time.time()
    level_kwargs = build_level_kwargs()
    print(f"{len(level_kwargs)} RBF+bootstrap runs to execute.")
    results = run_levels_parallel(level_kwargs, methods=(METHOD,))
    run = {k: _rel(v[METHOD]) for k, v in results.items()}

    # --- base case: pinned entry has {"run_dir", "timestamp"} -------------
    base_path = PROCESSED / "base_case" / "source_runs.json"
    base = _load(base_path)
    base[METHOD] = {"run_dir": run[BASE_KEY], "timestamp": Path(run[BASE_KEY]).name}
    _dump(base_path, base)

    # --- range axis --------------------------------------------------------
    range_path = PROCESSED / "range_axis" / "source_runs.json"
    rng = _load(range_path)
    for r in ALL_RANGE_VALUES:
        key = BASE_KEY if r == BASE_CASE_RANGE else f"range__{int(r)}"
        rng[str(int(r))][METHOD] = run[key]
    _dump(range_path, rng)

    # --- density axis ------------------------------------------------------
    dens_path = PROCESSED / "sample_density_axis" / "source_runs.json"
    dens = _load(dens_path)
    dens["5"][METHOD] = run[BASE_KEY]
    for lvl in ("2", "1"):
        dens[lvl][METHOD] = run[f"density__{lvl}"]
    _dump(dens_path, dens)

    # --- replicate axis (frozen source_runs_level5_n125.json NOT touched) --
    rep_path = PROCESSED / "sample_replicate_axis" / "source_runs.json"
    rep = _load(rep_path)
    for lvl in ("5", "2", "1"):
        for rep_id in REPLICATE_IDS:
            rep[lvl][rep_id][METHOD] = run[f"replicate__{lvl}__{rep_id}"]
    _dump(rep_path, rep)

    # --- verify replicate runs against their own (level, replicate) --------
    for lvl in ("5", "2", "1"):
        for rep_id in REPLICATE_IDS:
            verify_run_against_replicate(rep[lvl][rep_id][METHOD], lvl, rep_id, METHOD)
    print("replicate-axis runs verified against regenerated conditioning samples.")
    print(json.dumps(run, indent=2))
    print(f"rerun_rbf_bootstrap_only total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
