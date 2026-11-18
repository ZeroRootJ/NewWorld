"""Evaluate the E2-E5 multi-truth runs (src/experiments/multi_truth_axes.py)
for one or more ensemble BATCH SETS.

For every cell (case, axis_level, truth_seed, sample_rep) and every requested
batch set (e.g. "1" -> N=50, later "1,2" -> N=100), the SGS realizations /
RBF bootstrap replicates of the listed batches are CONCATENATED and the
ensemble mean / variance (ddof=1, as everywhere in the project) are
recomputed from them. kriging and gp_mle have no ensemble and are evaluated
once per cell (their rows carry n_ensemble = NaN and batches = "").

Metric definitions are those of
src/experiments/evaluate_sample_replicate_axis.py:evaluate_one_cell and
results/processed/sample_replicate_axis/make_coverage90.py, unchanged:
  * mask          : conditioning_cell_mask of the regenerated conditioning
                    samples (identical for all methods within a cell);
  * mse           : kriging kriging_mean_map_physical; sgs ensemble mean;
                    rbf_bootstrap point_estimate_map (single full-data fit,
                    batch-independent); gp_mle posterior mean;
  * umg           : calc_umg on the default np.linspace(0, 1, 20) grid;
                    kriging via kriging_fraction_in (exact per-cell
                    normal-score back-transform), the others via
                    accuracy_plot_fraction_in under N(mean, sqrt(var)) with
                    mean = sgs ensemble mean / rbf bootstrap mean / gp
                    posterior mean;
  * coverage90    : fraction_in at p = 0.90 from the SAME call, via
                    p_intervals = [linspace(0, 1, 20)..., 0.90]; for the
                    Gaussian methods the 20-point prefix is asserted
                    bit-identical to a default call;
  * variance_mean : mean predictive variance over the mask (kriging: the
                    Monte-Carlo physical-unit map kriging_var_map_physical_mc).

Fitted hyperparameters (hyperparameters.csv, long format) use the existing
definitions: GP practical range = sqrt(-2 ln 0.05) * length_scale
(qc_nugget_axis.GP_PRACTICAL_RANGE_FACTOR), noise / signal variance in real
units from params.fitted_hyperparameters; RBF r_0.05 = sqrt(-ln 0.05) /
best_epsilon and best_smoothing. For range/nugget cells the GP values are
cross-checked against results/processed/{range,nugget}_axis/
gp_hyperparameters_by_realization.csv (same truth seed / level); the max
relative difference is reported in evaluation_provenance.json and the script
raises if it exceeds GP_CROSSCHECK_RTOL.

Verification per run (raises on any mismatch): manifest truth_seed /
sample_seed / hmaj1 / hmin1 / nug / n_samples_requested equal the cell's;
samples.csv equals the regenerated conditioning samples; ensemble runs carry
the expected batch seed; batch seeds within one batch set are distinct; for
a single-batch set the recomputed mean/var equal the run's saved maps; for
RBF, best_epsilon / best_smoothing are equal and point_estimate_map is
identical (atol RBF_POINT_ATOL) across all batches of a set.

Outputs (--out-dir, default results/processed/multi_truth_axes/):
  metrics.csv            tidy: experiment, case, axis, axis_level,
                         truth_seed, sample_rep, sample_seed, method,
                         n_ensemble, batches, metric, value
  metrics_summary.csv    per (experiment, case, axis, axis_level, method,
                         n_ensemble, batches, metric): mean, std (ddof=1), n
                         -- pooled over all (truth_seed, sample_rep) rows.
  hyperparameters.csv    experiment, case, axis, axis_level, truth_seed,
                         sample_rep, method, parameter, value
  evaluation_cells.csv   per cell: n_samples_actual, n_cells_evaluated,
                         sgs_grid_node_warnings_b<k> (count of geostats.sgsim
                         "WARNING: grid node location" lines of the batch-k
                         SGS run, from its manifest params.sgsim_warnings;
                         empty for runs that predate that field), run dirs
  evaluation_provenance.json : sources, batch sets, checks, git commit of this
                         evaluation and of every run per (method, batch)

The ``batches`` column is ALWAYS a string ('1', '1,2', ...; '' for
kriging/gp_mle). Read the CSVs with ``read_metrics_csv`` (below), which keeps
it a string -- a plain pd.read_csv turns '1' into 1 and '' into NaN.

Batch-combination guards (coordinator decision J3 (a)): the SGS and RBF
configuration params listed in BATCH_INVARIANT_PARAMS (jitter, search,
back-transform / epsilon & smoothing grids, CV folds & seed, ...) must be
identical across every batch that is concatenated, else AssertionError.
git_commit is not asserted, only recorded per batch. In a production
evaluation (all sources production, no skipped cell) the GP cross-check must
cover exactly 80 range + 60 nugget cells (140).

Usage:
  .venv/Scripts/python.exe -m src.experiments.evaluate_multi_truth_axes \
      --sources results/raw/multi_truth_axes/<batch1 ts> --batch-sets 1
  later:  --sources <b1> <b2> --batch-sets 1 1,2
  partial (while E4 is still running): add --allow-incomplete
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.evaluation import (
    DEFAULT_BINS,
    accuracy_plot_fraction_in,
    calc_umg,
    conditioning_cell_mask,
    kriging_fraction_in,
    mse,
)
from src.experiments.base_case import NX, NY, XMN, YMN, XSIZ, YSIZ
from src.experiments.base_case_conditioning import VCOL, get_base_case_truth, get_conditioning_samples
from src.experiments.evaluate_sample_replicate_axis import (
    CORRELATION_CUTOFF,
    VARIANCE_SOURCE_FILES,
    safe_sqrt_variance,
)
from src.experiments.kriging import BACKTR_ZMAX, BACKTR_ZMIN, LTAIL, LTPAR, UTAIL, UTPAR
from src.experiments.multi_truth_axes import (
    METHOD_SEED_KW,
    batch_seed,
    read_jsonl,
    rel_or_abs,
    resolve_run_dir,
)
from src.experiments.qc_nugget_axis import GP_PRACTICAL_RANGE_FACTOR
from src.io import get_git_commit, get_git_status

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "processed" / "multi_truth_axes"

METHODS = ("kriging", "sgs", "rbf_bootstrap", "gp_mle")
ENSEMBLE = ("sgs", "rbf_bootstrap")
METRICS = ("mse", "umg", "variance_mean", "coverage90")
NOMINAL_P_COVERAGE = 0.90

SAMPLES_MATCH_ATOL = 1e-10
MAP_MATCH_TOL = 1e-10
RBF_POINT_ATOL = 1e-10
# 1e-5 (was 1e-6, changed 2026-11-17): in the batch-1 production run one of the 140
# cells (range_axis 100 m, truth 7106) reproduced the F4 GP fit to 2.9e-6 relative on all
# three parameters (L-BFGS stopping point); the other 139 matched to <= 3e-11.
GP_CROSSCHECK_RTOL = 1e-5

# Configuration params that must be IDENTICAL across all batches whose
# ensembles are concatenated (coordinator decision J3 (a), 2026-11-16). Only
# the batch seed (and n, timing, QC numbers) may differ. git_commit is NOT
# asserted -- it is recorded per batch in evaluation_provenance.json.
BATCH_INVARIANT_PARAMS = {
    "sgs": ("grid", "variogram", "sgsim_input_jitter", "honor_reenforcement", "ndmin", "ndmax",
            "nodmax", "mults", "nmult", "noct", "ktype", "colocorr", "sec_map", "sgsim_backtransform"),
    "rbf_bootstrap": ("grid", "rbf_kernel", "cv_folds", "cv_seed", "epsilon_grid", "smoothing_grid"),
}

# Production evaluation must compare every range/nugget cell (8 x 10 + 6 x 10).
EXPECTED_GP_CROSSCHECK_CELLS = {"range_axis": 80, "nugget_axis": 60}

GP_REALIZATION_CSV = {
    "range_axis": _REPO_ROOT / "results/processed/range_axis/gp_hyperparameters_by_realization.csv",
    "nugget_axis": _REPO_ROOT / "results/processed/nugget_axis/gp_hyperparameters_by_realization.csv",
}

CELL_ID_COLS = ["experiment", "case", "axis", "axis_level", "truth_seed", "sample_rep", "sample_seed"]


BATCHES_COLUMN_NOTE = (
    "'batches' is always written as a STRING: the comma-separated batch list ('1', '1,2', ...) for "
    "sgs / rbf_bootstrap, '' for kriging / gp_mle. pandas would otherwise parse '1' as an integer "
    "and '' as NaN; read with read_metrics_csv() (dtype={'batches': str}, '' kept as '') ."
)


def read_metrics_csv(path) -> pd.DataFrame:
    """Read metrics.csv / metrics_summary.csv preserving the string 'batches'
    column ('1', '1,2', and '' for kriging/gp_mle) and the integer
    n_ensemble (<NA> for kriging/gp_mle)."""
    df = pd.read_csv(path, dtype={"batches": str, "axis_level": str}, keep_default_na=False,
                     na_values={"n_ensemble": [""], "std": [""], "value": [""], "mean": [""]})
    if "n_ensemble" in df.columns:
        df["n_ensemble"] = df["n_ensemble"].astype("Int64")
    return df


def _p_grid() -> np.ndarray:
    return np.append(np.linspace(0.0, 1.0, DEFAULT_BINS), NOMINAL_P_COVERAGE)


@lru_cache(maxsize=2)
def _truth(truth_seed, hmaj1, hmin1, nug):
    """Ground-truth field, cached PER WORKER PROCESS (lru_cache is not shared
    between processes). Keyed by the truth parameters only -- not by the
    sample seed / count -- so within one evaluation group (all cells of one
    truth, see main()) the field is generated once and reused by every cell
    of that group. Same generator call as get_base_case_conditioning_data
    (get_base_case_truth); callers must not mutate the returned array."""
    return get_base_case_truth(truth_seed=truth_seed, hmaj1=hmaj1, hmin1=hmin1, nug=nug)


def _conditioning(truth_seed, sample_seed, hmaj1, hmin1, nug, n_samples):
    """(truth, samples_df, mask) for one cell. Equivalent to
    get_base_case_conditioning_data (= get_base_case_truth followed by
    get_conditioning_samples), with only the truth field cached."""
    truth = _truth(truth_seed, hmaj1, hmin1, nug)
    samples_df = get_conditioning_samples(truth, sample_seed=sample_seed, n_samples=n_samples)
    mask = conditioning_cell_mask(samples_df, NX, NY, XMN, YMN, XSIZ, YSIZ)
    return truth, samples_df, mask


def _params(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["params"]


def _verify_run(cell: dict, method: str, run_dir: Path, samples_df: pd.DataFrame, batch=None) -> dict:
    p = _params(run_dir)
    for key, expected in (("truth_seed", cell["truth_seed"]), ("sample_seed", cell["sample_seed"]),
                          ("n_samples_requested", cell["n_samples"])):
        if p.get(key) != expected:
            raise ValueError(f"{run_dir}: params.{key}={p.get(key)} != {expected}")
    for key in ("hmaj1", "hmin1", "nug"):
        if not np.isclose(float(p[key]), float(cell[key]), rtol=0, atol=1e-12):
            raise ValueError(f"{run_dir}: params.{key}={p[key]} != {cell[key]}")
    if method in METHOD_SEED_KW:
        seed_key = METHOD_SEED_KW[method]
        if p.get(seed_key) != batch_seed(method, batch):
            raise ValueError(f"{run_dir}: params.{seed_key}={p.get(seed_key)} != batch {batch} seed "
                             f"{batch_seed(method, batch)}")
    recorded = pd.read_csv(run_dir / "samples.csv")
    if len(recorded) != len(samples_df) or not np.allclose(
        recorded[["X", "Y", VCOL]].values, samples_df[["X", "Y", VCOL]].values,
        rtol=0.0, atol=SAMPLES_MATCH_ATOL,
    ):
        raise ValueError(f"{run_dir}: samples.csv != regenerated conditioning samples for {cell['config_key']}")
    return p


def _gaussian_metrics(truth_m, point_m, mean_m, var_m, label) -> dict:
    std_m = safe_sqrt_variance(var_m, label)
    p, frac = accuracy_plot_fraction_in(truth_m, mean_m, std_m, p_intervals=_p_grid())
    p_def, frac_def = accuracy_plot_fraction_in(truth_m, mean_m, std_m)
    if not (np.array_equal(p[:DEFAULT_BINS], p_def) and np.array_equal(frac[:DEFAULT_BINS], frac_def)):
        raise AssertionError(f"{label}: explicit p_intervals prefix not bit-identical to default call.")
    if p[-1] != NOMINAL_P_COVERAGE:
        raise AssertionError("last p_interval is not 0.90")
    return {
        "mse": mse(truth_m, point_m),
        "umg": calc_umg(p_def, frac_def),
        "variance_mean": float(np.mean(var_m)),
        "coverage90": float(frac[-1]),
    }


def kriging_metrics(kdir: Path, truth_m: np.ndarray, mask: np.ndarray) -> dict:
    """mse / umg / variance_mean / coverage90 of one kriging run directory,
    evaluated on the cells selected by ``mask``. Moved verbatim out of
    evaluate_cell (2026-11-17) so src/experiments/
    evaluate_inferred_variogram_density.py runs the identical code path."""
    kmap_phys = np.load(kdir / "kriging_mean_map_physical.npy")
    kmap_ns = np.load(kdir / "kmap_ns.npy")
    vmap_ns = np.load(kdir / "kriging_var_map_ns.npy")
    tt = pd.read_csv(kdir / "nscore_transform_table.csv")
    std_ns_m = safe_sqrt_variance(vmap_ns[mask], "kriging vmap_ns")
    kp, kfrac = kriging_fraction_in(truth_m, kmap_ns[mask], std_ns_m, tt["vr"].values, tt["vrg"].values,
                                    BACKTR_ZMIN, BACKTR_ZMAX, LTAIL, LTPAR, UTAIL, UTPAR,
                                    p_intervals=_p_grid())
    if not np.array_equal(kp[:DEFAULT_BINS], np.linspace(0.0, 1.0, DEFAULT_BINS)) or kp[-1] != NOMINAL_P_COVERAGE:
        raise AssertionError("kriging p grid mismatch")
    kvar = np.load(kdir / VARIANCE_SOURCE_FILES["kriging"])[mask]
    safe_sqrt_variance(kvar, "kriging variance_mean")
    return {
        "mse": mse(truth_m, kmap_phys[mask]),
        "umg": calc_umg(kp[:DEFAULT_BINS], kfrac[:DEFAULT_BINS]),
        "variance_mean": float(np.mean(kvar)),
        "coverage90": float(kfrac[-1]),
    }


def evaluate_cell(cell: dict, runs: Dict[Tuple[str, object], str], batch_sets: List[Tuple[int, ...]]):
    """runs: {(method, batch or None): run_dir}. Returns (metric_rows,
    hyper_rows, cell_info_row)."""
    truth, samples_df, mask = _conditioning(cell["truth_seed"], cell["sample_seed"], cell["hmaj1"],
                                            cell["hmin1"], cell["nug"], cell["n_samples"])
    truth_m = truth[mask]
    ident = {k: cell[k] for k in CELL_ID_COLS}
    rows, hyper = [], []

    def add(method, n_ens, batches, values):
        for metric in METRICS:
            rows.append({**ident, "method": method, "n_ensemble": n_ens, "batches": batches,
                         "metric": metric, "value": float(values[metric])})

    # --- kriging -----------------------------------------------------------
    kdir = resolve_run_dir(runs[("kriging", None)])
    _verify_run(cell, "kriging", kdir, samples_df)
    add("kriging", np.nan, "", kriging_metrics(kdir, truth_m, mask))

    # --- gp_mle --------------------------------------------------------------
    gdir = resolve_run_dir(runs[("gp_mle", None)])
    gp = _verify_run(cell, "gp_mle", gdir, samples_df)
    gmean = np.load(gdir / "posterior_mean_map.npy")[mask]
    gvar = np.load(gdir / "posterior_var_map.npy")[mask]
    add("gp_mle", np.nan, "", _gaussian_metrics(truth_m, gmean, gmean, gvar, "gp_mle"))
    fh = gp["fitted_hyperparameters"]
    for name, val in (
        ("gp_length_scale_m", fh["length_scale_m"]),
        ("gp_practical_range_m", GP_PRACTICAL_RANGE_FACTOR * fh["length_scale_m"]),
        ("gp_noise_variance_real_units", fh["noise_variance_real_units"]),
        ("gp_noise_variance_normalized", fh["noise_variance_normalized"]),
        ("gp_signal_variance_real_units", fh["signal_variance_real_units"]),
        ("gp_signal_variance_normalized", fh["signal_variance_normalized"]),
        ("gp_log_marginal_likelihood", gp["log_marginal_likelihood"]),
    ):
        hyper.append({**ident, "method": "gp_mle", "parameter": name, "value": float(val)})

    # --- ensemble methods, per batch set ----------------------------------------
    rbf_reference = None
    config_reference = {}   # method -> (params subset, run_dir)
    run_records = []        # (method, batch, git_commit, git_dirty) per run, for provenance
    sgs_warnings = {}       # batch -> n_grid_node_warnings
    for _m, _d in (("kriging", kdir), ("gp_mle", gdir)):
        _man = json.loads((_d / "manifest.json").read_text(encoding="utf-8"))
        run_records.append((_m, None, _man.get("git_commit"), _man.get("git_dirty")))
    for bset in batch_sets:
        seeds_seen = {m: [] for m in ENSEMBLE}
        for method in ENSEMBLE:
            members, saved, point_map = [], None, None
            for b in bset:
                rdir = resolve_run_dir(runs[(method, b)])
                p = _verify_run(cell, method, rdir, samples_df, batch=b)
                seeds_seen[method].append(p[METHOD_SEED_KW[method]])
                cfg_subset = {k: p.get(k, "<missing>") for k in BATCH_INVARIANT_PARAMS[method]}
                if method not in config_reference:
                    config_reference[method] = (cfg_subset, rdir)
                elif json.dumps(cfg_subset, sort_keys=True) != json.dumps(config_reference[method][0], sort_keys=True):
                    diff = [k for k in cfg_subset if cfg_subset[k] != config_reference[method][0][k]]
                    raise AssertionError(f"{method} configuration differs across batches ({diff}): {rdir} vs "
                                         f"{config_reference[method][1]}")
                man = json.loads((rdir / "manifest.json").read_text(encoding="utf-8"))
                rec = (method, b, man.get("git_commit"), man.get("git_dirty"))
                if rec not in run_records:
                    run_records.append(rec)
                if method == "sgs":
                    w = p.get("sgsim_warnings", {})
                    sgs_warnings[b] = w.get("n_grid_node_warnings", np.nan) if isinstance(w, dict) else np.nan
                if method == "sgs":
                    ens = np.load(rdir / "sgs_realizations.npy")
                    n_rec = p["n_realizations"]
                    saved = (rdir / "sgs_mean_map.npy", rdir / "sgs_var_map.npy")
                else:
                    ens = np.load(rdir / "bootstrap_replicate_maps.npy")
                    n_rec = p["n_bootstrap"]
                    saved = (rdir / "bootstrap_mean_map.npy", rdir / "bootstrap_var_map.npy")
                    pm = np.load(rdir / "point_estimate_map.npy")
                    hp = (float(p["best_epsilon"]), float(p["best_smoothing"]))
                    if rbf_reference is None:
                        rbf_reference = (hp, pm, rdir)
                    else:
                        if hp != rbf_reference[0]:
                            raise AssertionError(f"RBF CV hyperparameters differ across batches: {rdir} {hp} "
                                                 f"vs {rbf_reference[2]} {rbf_reference[0]}")
                        if not np.allclose(pm, rbf_reference[1], rtol=0.0, atol=RBF_POINT_ATOL):
                            raise AssertionError(f"RBF point_estimate_map differs across batches: {rdir}")
                    point_map = rbf_reference[1]
                if ens.shape != (n_rec, NY, NX):
                    raise ValueError(f"{rdir}: ensemble shape {ens.shape} != {(n_rec, NY, NX)}")
                members.append(ens)
            if len(set(seeds_seen[method])) != len(seeds_seen[method]):
                raise AssertionError(f"{method}: repeated batch seed in batch set {bset}")
            ens_all = np.concatenate(members, axis=0)
            mean_map = ens_all.mean(axis=0)
            var_map = ens_all.var(axis=0, ddof=1)
            if len(bset) == 1 and not (
                np.allclose(mean_map, np.load(saved[0]), rtol=MAP_MATCH_TOL, atol=MAP_MATCH_TOL)
                and np.allclose(var_map, np.load(saved[1]), rtol=MAP_MATCH_TOL, atol=MAP_MATCH_TOL)
            ):
                raise AssertionError(f"{method} {cell['config_key']}: recomputed maps != saved maps")
            pt = mean_map if method == "sgs" else point_map
            label = f"{method} {cell['config_key']} batches {bset}"
            add(method, int(ens_all.shape[0]), ",".join(str(b) for b in bset),
                _gaussian_metrics(truth_m, pt[mask], mean_map[mask], var_map[mask], label))

    rp = _params(resolve_run_dir(runs[("rbf_bootstrap", batch_sets[0][0])]))
    if rp["rbf_kernel"] != "gaussian":
        raise ValueError("rbf_kernel is not gaussian; r_0.05 conversion does not apply")
    for name, val in (
        ("rbf_best_epsilon", rp["best_epsilon"]),
        ("rbf_length_r005_m", np.sqrt(-np.log(CORRELATION_CUTOFF)) / float(rp["best_epsilon"])),
        ("rbf_best_smoothing", rp["best_smoothing"]),
    ):
        hyper.append({**ident, "method": "rbf_bootstrap", "parameter": name, "value": float(val)})

    info = {**ident, "n_samples_requested": cell["n_samples"], "n_samples_actual": int(len(samples_df)),
            "n_cells_evaluated": int(mask.sum()), "config_key": cell["config_key"],
            **{f"sgs_grid_node_warnings_b{b}": sgs_warnings[b] for b in sorted(sgs_warnings)},
            **{f"run_dir_{m}_b{'' if b is None else b}": v for (m, b), v in sorted(
                runs.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1]))}}
    return rows, hyper, info, run_records


def _evaluate_group(args):
    cells, runs_by_cell, batch_sets = args
    out = []
    for c in cells:
        out.append(evaluate_cell(c, runs_by_cell[c["config_key"]], batch_sets))
    return out


def load_sources(sources: List[str]):
    """Merge the cells and registries of several multi_truth_axes folders.
    Returns (cells keyed by cell id, {config_key: {(method, batch): run_dir}},
    source_info)."""
    cells, runs, info = {}, {}, []
    for s in sources:
        d = resolve_run_dir(s)
        cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        reg = read_jsonl(d / "run_registry.jsonl")
        for c in cfg["cells"]:
            cid = tuple(c[k] for k in CELL_ID_COLS)
            if cid in cells and cells[cid] != c:
                raise ValueError(f"cell {cid} defined differently in two sources")
            cells[cid] = c
        for r in reg:
            key = (r["method"], r["batch"])
            prev = runs.setdefault(r["config_key"], {}).get(key)
            if prev is not None and prev != r["run_dir"]:
                raise ValueError(f"two runs registered for {r['config_key']} {key}: {prev} / {r['run_dir']}")
            runs[r["config_key"]][key] = r["run_dir"]
        info.append({"source": rel_or_abs(d), "batch": cfg["batch"],
                     "n_ensemble_this_batch": cfg["n_ensemble_this_batch"],
                     "is_production_configuration": cfg["is_production_configuration"],
                     "n_registry_entries": len(reg), "n_tasks_total": cfg["n_tasks_total"]})
    return cells, runs, info


def gp_crosscheck(hyper: pd.DataFrame) -> dict:
    """Compare GP hyperparameters of range/nugget cells with the F4
    gp_hyperparameters_by_realization.csv (same truth seed, same level)."""
    result = {}
    for case, path in GP_REALIZATION_CSV.items():
        sub = hyper[(hyper["case"] == case) & (hyper["method"] == "gp_mle")]
        if sub.empty or not path.exists():
            continue
        ref = pd.read_csv(path, dtype={"axis_level": str})
        ref["axis_level"] = ref["axis_level"].astype(str)
        ref = ref.set_index(["axis_level", "truth_seed"])
        max_rel, n_values, cells_compared = 0.0, 0, set()
        for param in ("gp_length_scale_m", "gp_practical_range_m", "gp_noise_variance_real_units"):
            s = sub[sub["parameter"] == param]
            for _, r in s.iterrows():
                key = (str(r["axis_level"]), int(r["truth_seed"]))
                if key not in ref.index:
                    continue
                rv = float(ref.loc[key, param])
                rel = abs(r["value"] - rv) / abs(rv)
                max_rel = max(max_rel, rel)
                n_values += 1
                cells_compared.add(key)
        result[case] = {"reference": rel_or_abs(path), "n_compared": len(cells_compared),
                        "n_values_compared": n_values, "max_rel_diff": max_rel}
        if max_rel > GP_CROSSCHECK_RTOL:
            raise AssertionError(f"GP hyperparameters differ from {path} (max rel diff {max_rel:.3e})")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--batch-sets", nargs="+", default=["1"],
                    help='each a comma-separated batch list, e.g. 1  1,2  1,2,3')
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="skip cells whose runs are not all registered yet (default: raise)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args(argv)

    batch_sets = [tuple(sorted(int(b) for b in s.split(","))) for s in args.batch_sets]
    cells, runs, source_info = load_sources(args.sources)

    ready, missing = [], []
    for cid, c in cells.items():
        r = runs.get(c["config_key"], {})
        need = [("kriging", None), ("gp_mle", None)] + [(m, b) for bs in batch_sets for b in bs for m in ENSEMBLE]
        lack = [k for k in need if k not in r]
        (missing if lack else ready).append((c, lack))
    if missing and not args.allow_incomplete:
        raise ValueError(f"{len(missing)} cells lack runs (e.g. {missing[0][0]['config_key']} {missing[0][1]}); "
                         "use --allow-incomplete for a partial evaluation")
    print(f"cells ready: {len(ready)}, incomplete (skipped): {len(missing)}")
    if not ready:
        raise ValueError("no cell has all required runs for the requested batch sets")

    # Group by truth: one group = one task = one worker, and _truth's cache
    # (keyed by truth parameters) then generates each truth field once per group.
    groups: Dict[tuple, list] = {}
    for c, _ in ready:
        groups.setdefault((c["truth_seed"], c["hmaj1"], c["hmin1"], c["nug"]), []).append(c)
    tasks = []
    for gcells in groups.values():
        gcells = sorted(gcells, key=lambda c: (c["sample_seed"], c["n_samples"]))
        tasks.append((gcells, {c["config_key"]: runs[c["config_key"]] for c in gcells}, batch_sets))

    results = []
    if args.workers <= 1:
        for t in tasks:
            results.extend(_evaluate_group(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for out in ex.map(_evaluate_group, tasks):
                results.extend(out)

    metric_rows = [r for res in results for r in res[0]]
    hyper_rows = [r for res in results for r in res[1]]
    info_rows = [res[2] for res in results]
    # git commit / dirty flag of every run, per (method, batch) -- recorded,
    # NOT asserted (coordinator decision J3 (a)).
    git_by_batch: Dict[str, Dict[str, int]] = {}
    for res in results:
        for method, b, commit, dirty in res[3]:
            k = f"{method}|batch={'none' if b is None else b}"
            ck = f"{commit} (dirty={dirty})"
            git_by_batch.setdefault(k, {})
            git_by_batch[k][ck] = git_by_batch[k].get(ck, 0) + 1
    metrics = pd.DataFrame(metric_rows)
    metrics["n_ensemble"] = metrics["n_ensemble"].astype("Int64")  # NaN (kriging/gp) -> <NA>, 50 not 50.0
    hyper = pd.DataFrame(hyper_rows)
    info = pd.DataFrame(info_rows)
    if not np.all(np.isfinite(metrics["value"].values)):
        raise ValueError("non-finite metric values")
    expected_rows = len(ready) * len(METRICS) * (2 + len(ENSEMBLE) * len(batch_sets))
    if len(metrics) != expected_rows:
        raise ValueError(f"metrics has {len(metrics)} rows, expected {expected_rows}")

    sort_cols = ["experiment", "case", "axis_level", "truth_seed", "sample_rep", "method", "batches", "metric"]
    metrics = metrics.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    hyper = hyper.sort_values(["experiment", "case", "axis_level", "truth_seed", "sample_rep", "method",
                               "parameter"], kind="mergesort").reset_index(drop=True)
    info = info.sort_values(["experiment", "case", "axis_level", "truth_seed", "sample_rep"],
                            kind="mergesort").reset_index(drop=True)

    grp = ["experiment", "case", "axis", "axis_level", "method", "n_ensemble", "batches", "metric"]
    summary = (metrics.groupby(grp, dropna=False, sort=False)["value"]
               .agg(mean="mean", std=lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, n="count")
               .reset_index())

    checks = {"gp_crosscheck_vs_f4_csv": gp_crosscheck(hyper)}
    # Production evaluation (every source is the production configuration and
    # no cell was skipped) must have compared every range / nugget cell.
    is_full_production = all(si["is_production_configuration"] for si in source_info) and not missing
    checks["gp_crosscheck_full_coverage_required"] = is_full_production
    if is_full_production:
        for case, n_exp in EXPECTED_GP_CROSSCHECK_CELLS.items():
            n_got = checks["gp_crosscheck_vs_f4_csv"].get(case, {}).get("n_compared", 0)
            if n_got != n_exp:
                raise AssertionError(f"GP cross-check compared {n_got} {case} cells, expected {n_exp} "
                                     "(sum 140) in a production evaluation")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    hyper.to_csv(out_dir / "hyperparameters.csv", index=False)
    info.to_csv(out_dir / "evaluation_cells.csv", index=False)
    git_dirty, git_dirty_files = get_git_status()
    prov = {
        "generated_by": "src/experiments/evaluate_multi_truth_axes.py",
        "git_commit": get_git_commit(), "git_dirty": git_dirty, "git_dirty_files": git_dirty_files,
        "sources": source_info,
        "batch_sets": [list(b) for b in batch_sets],
        "n_cells_evaluated": len(ready),
        "n_cells_skipped_incomplete": len(missing),
        "skipped_cells": [c["config_key"] + f" ({c['case']} {c['axis_level']})" for c, _ in missing][:50],
        "metrics": list(METRICS),
        "coverage_nominal_p": NOMINAL_P_COVERAGE,
        "umg_bins": DEFAULT_BINS,
        "summary_definition": "mean / std(ddof=1) / n over all (truth_seed, sample_rep) rows of a "
                              "(experiment, axis_level, method, n_ensemble, batches, metric) group",
        "kriging_gp_rows": "n_ensemble = NaN, batches = '' (no ensemble; evaluated once per cell)",
        "batches_column": BATCHES_COLUMN_NOTE,
        "run_git_commits_by_method_batch": git_by_batch,
        "batch_invariant_params_asserted": {m: list(v) for m, v in BATCH_INVARIANT_PARAMS.items()},
        "checks": checks,
    }
    (out_dir / "evaluation_provenance.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))
    print(json.dumps(checks, indent=2))
    print(f"wrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
