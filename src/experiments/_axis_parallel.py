"""Shared helper for parallelizing axis-experiment (level x method) runs
across OS processes.

Background (2026-09-16 orchestrator task, scoped to range_axis.py /
sample_density_axis.py only): each axis experiment runs several ground-truth
levels x 4 methods (kriging / sgs / rbf_bootstrap / gp_mle), all fully
independent of each other -- different truth/sample draws, disjoint output
files, no shared mutable state between calls. Measured sequentially this
costs ~60-85s per (level x method) task (sgs dominates, ~70% of that), so the
whole level x method grid is executed as one flat pool of independent tasks
instead of sequential python loops.

Why PROCESSES, not threads
---------------------------
geostatspy's ``sgsim`` mutates the *global* numpy RNG (``np.random.seed`` /
``np.random`` module-level state) internally. Running it from multiple
threads in one interpreter would race on that shared global state and could
silently produce run B's simulation contaminated by run A's RNG state (or
vice versa) -- a correctness bug, not just a performance one. A process pool
sidesteps this entirely: each process has its own interpreter, its own numpy
RNG, no shared memory -- so there is nothing to race on.

Why the worker function is a plain module-level function
----------------------------------------------------------
On Windows, ``ProcessPoolExecutor`` uses the ``spawn`` start method: each
worker is a *fresh* Python interpreter that must be able to import
(unpickle-by-reference) whatever callable was submitted. Closures, lambdas,
and bound methods are not picklable this way -- only plain module-level
functions/classes are -- hence ``_run_one_method`` below.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Sequence

from src.experiments import gp_mle, kriging, rbf_bootstrap, sgs

METHOD_MODULES = {
    "kriging": kriging,
    "sgs": sgs,
    "rbf_bootstrap": rbf_bootstrap,
    "gp_mle": gp_mle,
}

DEFAULT_METHODS = ("kriging", "sgs", "rbf_bootstrap", "gp_mle")

# Env vars that cap BLAS/OpenMP thread pools (numpy in this environment is
# OpenBLAS-backed). Set on the PARENT process before the pool is created so
# spawned children inherit them at process-creation time, before they import
# numpy -- see default_max_workers() docstring for why this matters.
_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _run_one_method(method: str, kwargs: dict):
    """Worker: call ``<method>.main(**kwargs)``, return ``(method, run_dir)``.

    Must stay a plain module-level function (see module docstring) so it is
    picklable under Windows' ``spawn`` start method.
    """
    module = METHOD_MODULES[method]
    run_dir, _, _ = module.main(**kwargs)
    return method, run_dir


def default_max_workers() -> int:
    """Conservative worker-count default.

    numpy here is OpenBLAS-backed, so gp_mle's GP fit and rbf_bootstrap's
    linear-algebra-heavy CV/bootstrap loops may *already* be internally
    multithreaded. Naively using one process per logical CPU would multiply
    "process-level parallelism" by "BLAS-thread-level parallelism" and
    oversubscribe the machine (more runnable threads than cores), which can
    make the parallel run slower than sequential rather than faster. We
    therefore do both of the two mitigations the task called out rather than
    picking just one: (1) cap processes at half the logical CPUs here, AND
    (2) cap each child process's BLAS/OpenMP thread pool to 1 in
    ``run_levels_parallel`` below -- so total runnable compute threads stays
    bounded by (CPU count // 2) even in the worst case where every task is
    BLAS-heavy at once.
    """
    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count // 2)


# Reviewer-verified (2026-09-16): capping BLAS threads to 1 makes gp_mle's
# and rbf_bootstrap's outputs (both go through scipy/sklearn OpenBLAS linalg
# -- Cholesky/SVD) differ from an uncapped run at the ~1e-14 (double
# machine-epsilon) level -- CV-selected hyperparameters and all downstream
# metrics are unaffected. kriging/sgs (geostatspy, numba-backed, no BLAS
# dependency) are exactly bit-for-bit regardless of thread capping. User
# decision 2026-09-16: scientifically inert, document only, no code change.


def run_levels_parallel(
    level_kwargs: Dict[str, dict],
    methods: Sequence[str] = DEFAULT_METHODS,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Path]]:
    """Run ``methods x level_kwargs`` as one flat pool of independent tasks.

    Parameters
    ----------
    level_kwargs : maps an axis-level key (e.g. "100" or "2") to the kwargs
        dict to pass as ``**kwargs`` to each method's ``main()`` for that
        level (e.g. ``dict(truth_seed=..., hmaj1=..., hmin1=..., n_samples=...)``,
        already fully resolved by the caller -- this helper does not
        interpret axis semantics at all).
    methods : method names to run at every level; must be keys of
        ``METHOD_MODULES``.
    max_workers : process-pool size; defaults to ``default_max_workers()``.

    Returns
    -------
    ``{level_key: {method: run_dir}}``, matching the shape the sequential
    ``run_one_range`` / ``run_one_level`` helpers previously returned per
    level, just assembled from parallel tasks instead of a sequential loop.
    """
    if max_workers is None:
        max_workers = default_max_workers()

    # Cap BLAS/OpenMP thread pools in the CHILD processes (see
    # default_max_workers() docstring). Set on the parent's environ before
    # the pool starts so `spawn`-ed children inherit it pre-numpy-import;
    # restored afterwards so this function has no lasting side effect on the
    # calling process's environment.
    _prev_env = {var: os.environ.get(var) for var in _BLAS_THREAD_ENV_VARS}
    for var in _BLAS_THREAD_ENV_VARS:
        os.environ[var] = "1"

    tasks = [
        (level_key, method, kwargs)
        for level_key, kwargs in level_kwargs.items()
        for method in methods
    ]

    results: Dict[str, Dict[str, Path]] = {level_key: {} for level_key in level_kwargs}

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_one_method, method, kwargs): (level_key, method)
                for level_key, method, kwargs in tasks
            }
            print(
                f"Submitted {len(futures)} independent (level x method) tasks to a "
                f"{max_workers}-process pool (BLAS threads capped to 1/process)."
            )
            for future in as_completed(futures):
                level_key, method = futures[future]
                _, run_dir = future.result()
                results[level_key][method] = run_dir
                print(f"  done: level={level_key} method={method} -> {run_dir}")
    finally:
        for var, prev in _prev_env.items():
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev

    return results
