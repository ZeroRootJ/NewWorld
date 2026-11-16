"""90% interval coverage per (density level, replicate, method) for the
sample-replicate axis (1 ground truth, range 300 m; density levels 20/10/5/2/1;
10 sample-location replicates per level; 4 methods).

Definition
----------
coverage90 = the accuracy-plot ``fraction_in`` evaluated at EXACTLY the nominal
probability interval p = 0.90, i.e. the fraction of evaluated cells whose truth
lies strictly inside the central 90% predictive interval.

It is computed through the SAME per-method code path that
src/experiments/evaluate_sample_replicate_axis.py's ``evaluate_one_cell`` uses
for UMG:
  * evaluation cells: ``conditioning_cell_mask`` of that (level, replicate)'s
    regenerated conditioning samples (via that module's
    ``_level_replicate_mask``), identical for all 4 methods;
  * kriging: ``kriging_fraction_in`` on (kmap_ns, sqrt(kriging_var_map_ns))
    with the run's own nscore_transform_table.csv and the same back-transform
    tail parameters (BACKTR_ZMIN/ZMAX, LTAIL/LTPAR, UTAIL/UTPAR) -- the
    normal-score interval is back-transformed per cell, exactly;
  * sgs / rbf_bootstrap / gp_mle: ``accuracy_plot_fraction_in`` (Gaussian
    percentile of truth under N(mean, std)) on
      sgs           sgs_mean_map.npy        / sgs_var_map.npy
      rbf_bootstrap bootstrap_mean_map.npy  / bootstrap_var_map.npy
      gp_mle        posterior_mean_map.npy  / posterior_var_map.npy
    i.e. a mean +- z*std interval (NOT empirical quantiles of the SGS
    realizations / bootstrap replicates).

0.90 is not on the default np.linspace(0, 1, 20) grid, so both functions are
called with the optional ``p_intervals`` argument (added to src/evaluation.py
on 2026-11-16; default behavior unchanged) as
``p_intervals = [linspace(0, 1, 20)..., 0.90]`` in ONE call per method. The
first 20 entries are then fed to ``calc_umg`` and asserted equal (atol 1e-12)
to the umg stored in metrics.csv for every (level, replicate, method); the
script raises and writes nothing if any cell disagrees. For the three Gaussian
methods the 20-point prefix is additionally asserted bit-identical to a
default (p_intervals=None) call.

Provenance
----------
Run directories per (axis_level, replicate, method) come from
results/processed/sample_replicate_axis/source_runs.json (the same file
evaluate_sample_replicate_axis.py reads). Each run's samples.csv is
cross-checked against the regenerated conditioning samples, as in
evaluate_one_cell. results/raw/ is only read.

Outputs (results/processed/sample_replicate_axis/):
  coverage90.csv          tidy: case,axis,axis_level,replicate,method,metric,value
                          (metric = "coverage90"); leading '#' comment lines
                          record the definition and source_runs.json path
                          (read with pandas comment='#').
  coverage90_summary.csv  per (axis_level, method): mean, std (ddof=1, as in
                          metrics_summary.csv), min, max, n_replicates.

Run with:
.venv/Scripts/python.exe -m results.processed.sample_replicate_axis.make_coverage90
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation import (  # noqa: E402
    DEFAULT_BINS,
    accuracy_plot_fraction_in,
    calc_umg,
    kriging_fraction_in,
)
from src.experiments.evaluate_sample_replicate_axis import (  # noqa: E402
    AXIS,
    CASE,
    SAMPLES_MATCH_ATOL,
    SOURCE_RUNS_PATH,
    _level_replicate_mask,
    safe_sqrt_variance,
)
from src.experiments.kriging import (  # noqa: E402
    BACKTR_ZMAX, BACKTR_ZMIN, LTAIL, LTPAR, UTAIL, UTPAR,
)
from src.experiments.sample_density_axis import EXTENDED_AXIS_LEVELS  # noqa: E402
from src.experiments.sample_replicate_axis import METHODS, REPLICATE_IDS  # noqa: E402

PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "sample_replicate_axis"
METRICS_CSV = PROCESSED_DIR / "metrics.csv"
OUT_CSV = PROCESSED_DIR / "coverage90.csv"
OUT_SUMMARY_CSV = PROCESSED_DIR / "coverage90_summary.csv"

NOMINAL_P = 0.90
METRIC_NAME = "coverage90"
UMG_ATOL = 1e-12

# Same (point/unc-mean, unc-var) source files as evaluate_one_cell's
# gaussian_specs (only the uncertainty mean/var enter fraction_in).
GAUSSIAN_SOURCES = {
    "sgs": ("sgs_mean_map.npy", "sgs_var_map.npy"),
    "rbf_bootstrap": ("bootstrap_mean_map.npy", "bootstrap_var_map.npy"),
    "gp_mle": ("posterior_mean_map.npy", "posterior_var_map.npy"),
}


def _p_grid() -> np.ndarray:
    return np.append(np.linspace(0.0, 1.0, DEFAULT_BINS), NOMINAL_P)


def curves_for_cell(axis_level: str, replicate_id: str, run_dirs: dict) -> dict:
    """{method: (p_intervals, fraction_in)} on _p_grid(), same path as
    evaluate_one_cell's UMG computation."""
    truth, samples_df, mask = _level_replicate_mask(axis_level, replicate_id)
    for m in METHODS:
        recorded = pd.read_csv(_REPO_ROOT / run_dirs[m] / "samples.csv")
        if len(recorded) != len(samples_df) or not np.allclose(
            recorded[["X", "Y"]].values, samples_df[["X", "Y"]].values,
            rtol=0.0, atol=SAMPLES_MATCH_ATOL,
        ):
            raise ValueError(
                f"{m}: samples.csv does not match regenerated conditioning samples "
                f"(level '{axis_level}' {replicate_id})."
            )
    truth_masked = truth[mask]
    p_grid = _p_grid()
    out = {}

    kdir = _REPO_ROOT / run_dirs["kriging"]
    kmap_ns = np.load(kdir / "kmap_ns.npy")
    vmap_ns = np.load(kdir / "kriging_var_map_ns.npy")
    tt = pd.read_csv(kdir / "nscore_transform_table.csv")
    std_ns_masked = safe_sqrt_variance(vmap_ns[mask], "kriging vmap_ns")
    out["kriging"] = kriging_fraction_in(
        truth_masked, kmap_ns[mask], std_ns_masked, tt["vr"].values, tt["vrg"].values,
        BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR, p_intervals=p_grid,
    )

    for method, (mean_file, var_file) in GAUSSIAN_SOURCES.items():
        d = _REPO_ROOT / run_dirs[method]
        mean_masked = np.load(d / mean_file)[mask]
        std_masked = safe_sqrt_variance(np.load(d / var_file)[mask], f"{method} variance map")
        p, frac = accuracy_plot_fraction_in(truth_masked, mean_masked, std_masked, p_intervals=p_grid)
        p_def, frac_def = accuracy_plot_fraction_in(truth_masked, mean_masked, std_masked)
        if not (np.array_equal(p[:DEFAULT_BINS], p_def) and np.array_equal(frac[:DEFAULT_BINS], frac_def)):
            raise AssertionError(
                f"{method} level '{axis_level}' {replicate_id}: explicit-p_intervals prefix is not "
                "bit-identical to the default call."
            )
        out[method] = (p, frac)
    return out


def main():
    source_runs = json.loads(SOURCE_RUNS_PATH.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS_CSV)
    metrics["axis_level"] = metrics["axis_level"].astype(str)
    umg_ref = metrics[metrics["metric"] == "umg"].set_index(["axis_level", "replicate", "method"])["value"]
    if set(metrics["case"]) != {CASE} or set(metrics["axis"]) != {AXIS}:
        raise ValueError(f"metrics.csv case/axis {set(metrics['case'])}/{set(metrics['axis'])} unexpected.")

    rows, mismatches, max_abs_diff = [], [], 0.0
    for axis_level in EXTENDED_AXIS_LEVELS:
        for replicate_id in REPLICATE_IDS:
            print(f"level '{axis_level}' {replicate_id} ...", flush=True)
            curves = curves_for_cell(axis_level, replicate_id, source_runs[axis_level][replicate_id])
            for method in METHODS:
                p, frac = curves[method]
                if p[-1] != NOMINAL_P:
                    raise AssertionError("last p_interval is not the nominal 0.90.")
                umg = calc_umg(p[:DEFAULT_BINS], frac[:DEFAULT_BINS])
                ref = float(umg_ref.loc[(axis_level, replicate_id, method)])
                diff = abs(umg - ref)
                max_abs_diff = max(max_abs_diff, diff)
                if diff > UMG_ATOL:
                    mismatches.append((axis_level, replicate_id, method, umg, ref, diff))
                rows.append({
                    "case": CASE, "axis": AXIS, "axis_level": axis_level,
                    "replicate": replicate_id, "method": method,
                    "metric": METRIC_NAME, "value": float(frac[-1]),
                })

    n_checked = len(rows)
    print(f"\nUMG reproduction: {n_checked} (level, replicate, method) cells checked, "
          f"max |umg - metrics.csv umg| = {max_abs_diff:.3e}, mismatches (> {UMG_ATOL:g}) = {len(mismatches)}")
    if mismatches:
        for m in mismatches:
            print("  MISMATCH", m)
        raise AssertionError("UMG reproduction failed -- coverage90 NOT written.")

    df = pd.DataFrame(rows)
    expected = len(EXTENDED_AXIS_LEVELS) * len(REPLICATE_IDS) * len(METHODS)
    if len(df) != expected or not np.all(np.isfinite(df["value"].values)):
        raise ValueError(f"coverage90 table has {len(df)} rows (expected {expected}) or non-finite values.")

    header = [
        f"# coverage90 = accuracy-plot fraction_in at nominal p = {NOMINAL_P} (strict inequality, "
        "conditioning-sample cells excluded), same per-method path as UMG in "
        "src/experiments/evaluate_sample_replicate_axis.py:evaluate_one_cell",
        "# kriging: normal-score interval back-transformed per cell (kriging_fraction_in); "
        "sgs/rbf_bootstrap/gp_mle: Gaussian percentile of truth under N(mean, std) (accuracy_plot_fraction_in)",
        f"# raw run directory per (axis_level, replicate, method): {SOURCE_RUNS_PATH.relative_to(_REPO_ROOT).as_posix()}",
        f"# UMG reproduction check: {n_checked} cells, max abs diff vs metrics.csv = {max_abs_diff:.3e} (atol {UMG_ATOL:g})",
        "# generated by: results/processed/sample_replicate_axis/make_coverage90.py",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f, index=False)

    summary_rows = []
    for axis_level in EXTENDED_AXIS_LEVELS:
        for method in METHODS:
            vals = df.loc[(df["axis_level"] == axis_level) & (df["method"] == method), "value"].values
            summary_rows.append({
                "axis_level": axis_level, "method": method, "metric": METRIC_NAME,
                "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)), "max": float(np.max(vals)), "n_replicates": len(vals),
            })
    summary = pd.DataFrame(summary_rows)
    with open(OUT_SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header[:3]) + "\n# std uses ddof=1 (same as metrics_summary.csv)\n")
        summary.to_csv(f, index=False)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(f"wrote {OUT_CSV}\nwrote {OUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
