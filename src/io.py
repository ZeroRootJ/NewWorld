"""Shared result-saving utilities.

Every simulation/experiment script in this project must save its outputs
through this module (see CLAUDE.md "결과 저장 컨벤션" and
.claude/agents/coder.md) instead of re-implementing manifest handling
per-script.

Typical usage from an experiment script::

    from src.io import make_run_dir, save_result

    run_dir = make_run_dir("base_case")
    # ... write output files (npy, csv, png, ...) into run_dir ...
    save_result(
        experiment="base_case",
        params={...},
        seed_or_seeds=[101, 102, 103, 104, 105],
        run_dir=run_dir,
        code_entrypoint="src/experiments/base_case.py",
        output_files=["truth_seed101.npy", "samples.csv", "qc_five_realizations.png"],
    )
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Repo root = parent of the directory this file lives in (src/ -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_git_commit(repo_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Return the current git commit hash, or None if it cannot be determined.

    Never raises: if the directory is not a git repo, git is unavailable, or
    there are no commits yet, this returns None instead of erroring out so
    that experiment runs are never blocked by an incomplete git setup.
    """
    repo_dir = Path(repo_dir) if repo_dir is not None else _REPO_ROOT
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit if commit else None


def make_run_dir(experiment: str, output_dir: Union[str, Path] = "results/raw") -> Path:
    """Create and return results/raw/<experiment>/<timestamp>/.

    The timestamp is UTC, filesystem-safe (no colons), e.g. 20260911T153000Z.
    """
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / experiment / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_result(
    experiment: str,
    params: Dict[str, Any],
    seed_or_seeds: Union[int, List[int]],
    output_dir: Union[str, Path] = "results/raw",
    run_dir: Optional[Union[str, Path]] = None,
    code_entrypoint: Optional[str] = None,
    output_files: Optional[List[str]] = None,
) -> Path:
    """Write manifest.json into the run directory and return the manifest path.

    If ``run_dir`` is not given, a fresh timestamped directory is created via
    ``make_run_dir`` (useful for scripts that have no outputs to write before
    the manifest). If ``run_dir`` is given (e.g. from a prior ``make_run_dir``
    call), the manifest is written there, after the caller has already saved
    its output files -- this lets ``output_files`` accurately reflect what
    was written.

    Parameters
    ----------
    experiment : name of the experiment (matches the results/raw/<experiment> folder)
    params : full dict of parameters used for this run (grid, variogram, etc.)
    seed_or_seeds : a single seed, or a list of seeds if the run has multiple
        realizations (e.g. multiple ground-truth realizations)
    output_dir : base output directory, only used when run_dir is None
    run_dir : an existing run directory (e.g. from make_run_dir) to write the
        manifest into
    code_entrypoint : path (relative to repo root) of the script that produced
        this run
    output_files : list of output file names (relative to run_dir) that this
        run produced
    """
    if run_dir is None:
        target_dir = make_run_dir(experiment, output_dir)
    else:
        target_dir = Path(run_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "params": params,
        "seed": seed_or_seeds,
        "code_entrypoint": code_entrypoint,
        "output_files": output_files if output_files is not None else [],
    }

    manifest_path = target_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest_path
