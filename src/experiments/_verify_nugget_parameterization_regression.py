"""One-off regression check for the nugget parameterization of
``build_vario`` / the four method ``main()`` functions (nugget axis,
docs/experiment_context.md deliverable 3, step 1).

WHAT IT CHECKS AND WHY
----------------------
Adding a ``nug`` argument to ``build_vario`` (deriving ``cc1 = 1.0 - nug``
instead of reading the module constant ``CC1``) and threading it through
``get_base_case_truth`` / ``get_base_case_conditioning_data`` /
kriging.main / sgs.main / rbf_bootstrap.main / gp_mle.main must be
BEHAVIOR-PRESERVING at the defaults: calling any of the four ``main()``
functions with NO arguments has to reproduce the pinned base-case run
bit-for-bit. This is the same regression gate the range-axis and
sample-density-axis parameterizations were held to.

It runs each of the four methods once at its defaults (producing four new,
additive run directories in results/raw/ -- existing runs are never touched)
and compares every saved .npy array against the corresponding array in the
base case's pinned run (results/processed/base_case/source_runs.json),
reporting the max absolute difference per array. It then runs rbf_bootstrap
ONE more time through ``_axis_parallel.run_levels_parallel`` (see the
"BLAS FOLLOW-UP" section below) -- so a full invocation produces FIVE new run
directories, not four.

HOW MANY RUN DIRECTORIES THIS PRODUCES, AND HOW MANY THE 2026-09-22 SESSION
PRODUCED (recorded at the reviewer's request, 2026-09-22)
---------------------------------------------------------------------------
One invocation of this script now writes 5 new run directories (kriging 1,
sgs 1, rbf_bootstrap 2 -- plain + BLAS follow-up -- gp_mle 1). The 2026-09-22
nugget-parameterization verification session as a whole left **10** new
regression-byproduct runs in results/raw/, NOT 6: kriging 2, sgs 2,
rbf_bootstrap 3, gp_mle 3. (The session included ad hoc re-invocations of
individual methods in addition to the full sweep.) None of those 10 is
pinned by any source_runs.json; they are inert byproducts kept because
results/raw/ is append-only in this project.

KNOWN NON-BIT-EXACT PATHS (measured and reported, NOT silently tolerated)
-------------------------------------------------------------------------
This script deliberately does NOT manipulate BLAS/OpenMP thread-count
environment variables: it runs each method in a plain interpreter, which is
how the pinned kriging / sgs / gp_mle base-case runs were produced.

1. rbf_bootstrap. Its pinned base-case run
   (results/raw/rbf_bootstrap/20260922T142648066028Z-1) was produced by
   src/experiments/rerun_rbf_bootstrap_only_v2.py, which goes through
   ``_axis_parallel.run_levels_parallel`` -- i.e. inside a spawned worker
   with BLAS threads capped to 1. Per _axis_parallel.py's
   reviewer-verified note (2026-09-16, user decision: "scientifically inert,
   document only"), scipy/sklearn OpenBLAS linear algebra differs at the
   ~1e-14 (double machine-epsilon) level between capped and uncapped thread
   pools. A residual of that magnitude here is therefore expected and is
   attributable to the thread configuration, not to the nugget
   parameterization. It is printed, never rounded away.

   BLAS FOLLOW-UP (now automated -- see below). To demonstrate rather than
   assert that attribution, this script ALSO re-runs rbf_bootstrap at its
   defaults through ``_axis_parallel.run_levels_parallel`` -- the exact path
   that produced the pinned run -- and compares again. That second
   comparison is expected to be exactly 0.0 on all four arrays. Its result
   is written into ``base_case_regression_check.json`` under
   ``rbf_bootstrap.blas_thread_followup`` BY THIS SCRIPT. It used to be
   appended to that file by hand after the fact, which meant re-running the
   script silently deleted it (reviewer finding, 2026-09-22); the block is
   now regenerated on every run so the JSON is always reproducible from the
   repository alone.
2. kriging, FROM 2026-09-22 ONWARDS. ``src/experiments/kriging.py`` now
   normal-score transforms through ``src/nscore.py``, which repairs a
   1-based -> 0-based off-by-one in ``geostatspy.geostats.nscore`` that
   mis-assigned the SMALLEST conditioning datum's normal score. Every
   kriging run produced BEFORE that fix -- including the pinned base-case
   run this script compares against -- carries the bad value, so kriging can
   no longer reproduce it bit-for-bit, BY DESIGN. A non-zero kriging diff
   here is therefore the nscore repair, not a nugget-parameterization
   regression. The pre-fix results (kriging 0.0 on all four arrays, which
   DID establish the parameterization was behavior-preserving at the time)
   are preserved in
   results/processed/nugget_axis/base_case_regression_check_pre_nscore_fix.json,
   written once by this script before it first overwrites the main file.
   The base case's own pinned runs are deliberately NOT re-run here -- the
   scope of re-running contaminated axes is an orchestrator/user decision,
   informed by results/processed/nscore_bug_impact.csv.
3. gp_mle's ``posterior_sample_map.npy``. gp_mle.main() first tries
   ``gpr.sample_y`` (numpy SVD) and falls back to a Cholesky draw when
   numpy's SVD fails to converge -- a documented, environment-specific numpy
   quirk (see gp_mle.py's long comment). SVD and Cholesky are two different
   square roots of the SAME posterior covariance, so when the fallback
   branch taken differs between two runs the resulting single posterior DRAW
   differs completely (O(10) porosity %), while the posterior mean/variance
   are unaffected. Whether SVD converges was observed to depend on the BLAS
   thread configuration, so this script prints each run's recorded
   ``posterior_sample_method`` alongside the diffs; a large
   posterior_sample_map diff accompanied by a method mismatch is that
   environment quirk, not a regression.

Run with:
.venv/Scripts/python.exe -m src.experiments._verify_nugget_parameterization_regression
"""

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_CASE_SOURCE_RUNS = _REPO_ROOT / "results" / "processed" / "base_case" / "source_runs.json"

# Arrays compared per method (every .npy each method saves).
ARRAYS = {
    "kriging": [
        "kriging_mean_map_physical.npy",
        "kriging_var_map_ns.npy",
        "kmap_ns.npy",
        "kriging_var_map_physical_mc.npy",
    ],
    "sgs": ["sgs_realizations.npy", "sgs_mean_map.npy", "sgs_var_map.npy"],
    "rbf_bootstrap": [
        "point_estimate_map.npy",
        "bootstrap_replicate_maps.npy",
        "bootstrap_mean_map.npy",
        "bootstrap_var_map.npy",
    ],
    "gp_mle": ["posterior_mean_map.npy", "posterior_var_map.npy", "posterior_sample_map.npy"],
}


def main(methods=None):
    from src.experiments import gp_mle, kriging, rbf_bootstrap, sgs

    modules = {
        "kriging": kriging,
        "sgs": sgs,
        "rbf_bootstrap": rbf_bootstrap,
        "gp_mle": gp_mle,
    }
    if methods is None:
        methods = list(modules)

    pinned = json.loads(BASE_CASE_SOURCE_RUNS.read_text(encoding="utf-8"))

    results = {}
    for method in methods:
        print(f"\n=== {method}: re-running main() at defaults ===")
        new_run_dir, _, _ = modules[method].main()
        pinned_dir = _REPO_ROOT / pinned[method]["run_dir"]
        per_array = {}
        for name in ARRAYS[method]:
            a = np.load(pinned_dir / name)
            b = np.load(Path(new_run_dir) / name)
            if a.shape != b.shape:
                raise ValueError(f"{method}/{name}: shape {a.shape} vs {b.shape}")
            per_array[name] = float(np.max(np.abs(a - b)))
        entry = {
            "new_run_dir": str(Path(new_run_dir).relative_to(_REPO_ROOT)).replace("\\", "/"),
            "pinned_run_dir": pinned[method]["run_dir"],
            "max_abs_diff": per_array,
        }
        if method == "gp_mle":
            # See module docstring note 2 -- the posterior DRAW depends on
            # which square root of the covariance was used, so record it.
            entry["posterior_sample_method_pinned"] = json.loads(
                (pinned_dir / "manifest.json").read_text(encoding="utf-8")
            )["params"].get("posterior_sample_method")
            entry["posterior_sample_method_new"] = json.loads(
                (Path(new_run_dir) / "manifest.json").read_text(encoding="utf-8")
            )["params"].get("posterior_sample_method")
        results[method] = entry
        for name, d in per_array.items():
            print(f"  {name}: max abs diff = {d!r}")
        if method == "gp_mle":
            print(
                f"  posterior_sample_method: pinned="
                f"{entry['posterior_sample_method_pinned']}, "
                f"new={entry['posterior_sample_method_new']}"
            )

    # --- BLAS thread follow-up for rbf_bootstrap -------------------------
    # Regenerated on every run (see module docstring) rather than hand-added
    # afterwards. Re-runs rbf_bootstrap at its defaults through the SAME
    # spawned-worker / BLAS-capped path that produced the pinned run, so the
    # ~1e-14 residual reported above can be attributed to the thread
    # configuration by measurement instead of by assertion.
    if "rbf_bootstrap" in methods:
        print(
            "\n=== rbf_bootstrap: BLAS-thread follow-up "
            "(re-run through _axis_parallel.run_levels_parallel) ==="
        )
        from src.experiments._axis_parallel import run_levels_parallel

        followup = run_levels_parallel(
            {"base_case_defaults": {}}, methods=["rbf_bootstrap"], max_workers=1
        )
        followup_dir = Path(followup["base_case_defaults"]["rbf_bootstrap"])
        pinned_dir = _REPO_ROOT / pinned["rbf_bootstrap"]["run_dir"]
        followup_diffs = {}
        for name in ARRAYS["rbf_bootstrap"]:
            a = np.load(pinned_dir / name)
            b = np.load(followup_dir / name)
            followup_diffs[name] = float(np.max(np.abs(a - b)))
            print(f"  {name}: max abs diff = {followup_diffs[name]!r}")
        results["rbf_bootstrap"]["blas_thread_followup"] = {
            "note": (
                "The plain-interpreter residual reported in max_abs_diff above was "
                "produced by calling rbf_bootstrap.main() directly, while the pinned "
                "run was produced inside _axis_parallel.run_levels_parallel (spawned "
                "worker, BLAS threads capped to 1). This block re-runs the SAME "
                "defaults through that SAME path; a diff of 0.0 here attributes the "
                "residual to the documented OpenBLAS thread-count effect "
                "(_axis_parallel.py, 2026-09-16) rather than to the nugget "
                "parameterization. Written by "
                "src/experiments/_verify_nugget_parameterization_regression.py on "
                "every run -- do not hand-edit."
            ),
            "rerun_through_axis_parallel_run_dir": str(
                followup_dir.relative_to(_REPO_ROOT)
            ).replace("\\", "/"),
            "max_abs_diff": followup_diffs,
        }

    print("\n=== SUMMARY (max abs diff vs. pinned base-case run) ===")
    for method, info in results.items():
        worst = max(info["max_abs_diff"].values())
        print(f"  {method}: worst = {worst!r}  (new run: {info['new_run_dir']})")

    out = _REPO_ROOT / "results" / "processed" / "nugget_axis"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "base_case_regression_check.json"

    # One-time provenance snapshot of the PRE-nscore-fix results (see module
    # docstring, known non-bit-exact path 2). Written once and never
    # overwritten, so the evidence that the nugget parameterization itself
    # was behavior-preserving for kriging is not lost when the nscore repair
    # legitimately changes kriging's output.
    pre_fix_path = out / "base_case_regression_check_pre_nscore_fix.json"
    if out_path.exists() and not pre_fix_path.exists():
        pre_fix_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"snapshotted pre-nscore-fix results: {pre_fix_path}")

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten: {out_path}")
    return results


if __name__ == "__main__":
    main(sys.argv[1:] or None)
