"""Generate the ALLUVSIM-based ground-truth family (event-based fluvial
realizations, ResMill ``ChannelLayer``, preset CB_JIGSAW) and run the
automated QC in-process.

Runs in the SEPARATE venv .venv-obm (Python 3.11, ResMill pinned to a git
commit, see obm_benchmark/requirements-obm.txt). From the repository root::

    C:/Users/yj7563/Desktop/NewWorld/.venv-obm/Scripts/python.exe -m obm_benchmark.scripts.generate_ground_truth

Optional: ``--config PATH`` (default obm_benchmark/config/cb_jigsaw.yaml).

Output: results/raw/obm_ground_truth/<timestamp>/ (via src/io.py) containing,
per geological seed (stem = cb_jigsaw_seed_NNN):
  <stem>_porosity_3d.npy       float32 (nx, ny, nz) NATIVE ResMill layout [ix, iy, iz], fraction
  <stem>_facies_3d.npy         int8    (nx, ny, nz) NATIVE ResMill layout
  <stem>_porosity_2d_frac.npy  float32 (ny, nx) project convention (row 0 = max-y), fraction
  <stem>_facies_2d.npy         int8    (ny, nx) project convention
  <stem>_truth_2d.npy          float64 (ny, nx) project convention, affine-rescaled to
                               mean 15 / std 3 (%), per realization
plus qc_summary.csv, diag_<stem>.png for the predefined diagnostic seeds,
config_used.yaml (verbatim copy of the config) and manifest.json.

GaussianLayer is not used anywhere in this workflow (brief section 5.1).
"""

import argparse
import json
import platform
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import resmill as rm
import resmill.layers.channel as rm_channel

from obm_benchmark.scripts import run_qc
from obm_benchmark.scripts._common import (
    DEFAULT_CONFIG,
    FILE_SUFFIXES,
    REPO_ROOT,
    affine_rescale,
    array_sha256,
    check_orientation_helper,
    file_sha256,
    load_config,
    slice_index_from_rule,
    slice_to_project_2d,
    stem_for,
)
from src.io import make_run_dir, save_result

CODE_ENTRYPOINT = "obm_benchmark/scripts/generate_ground_truth.py"
CONFIG_COPY_NAME = "config_used.yaml"


def resmill_provenance() -> Dict[str, Any]:
    dist = metadata.distribution("resmill")
    direct_url = dist.read_text("direct_url.json")
    info = json.loads(direct_url) if direct_url else None
    commit = info.get("vcs_info", {}).get("commit_id") if info else None
    return {
        "version": dist.version,
        "module_version": getattr(rm, "__version__", None),
        "git_commit": commit,
        "direct_url": info,
    }


def generate_one(seed: int, cfg: Dict[str, Any], preset_kwargs: Dict[str, Any],
                 facies_props_arg: Optional[Dict[int, Dict[str, float]]] = None):
    """One ChannelLayer realization. Returns (porosity_3d float32, facies_3d int8)
    in the native ResMill (nx, ny, nz) layout."""
    g = cfg["grid"]
    model = rm.ChannelLayer(
        nx=int(g["nx"]), ny=int(g["ny"]), nz=int(g["nz"]),
        x_len=float(g["x_len"]), y_len=float(g["y_len"]), z_len=float(g["z_len"]),
        top_depth=float(g["top_depth"]),
    )
    model.create_geology(
        seed=int(seed),
        poro_noise_std=float(cfg["porosity"]["poro_noise_std"]),
        poro_noise_range=float(cfg["porosity"]["poro_noise_range"]),
        # None = ResMill's own default table, and the argument is then not passed
        # at all, so runs without facies_poro_sd are bit-identical to before.
        **({} if facies_props_arg is None else {"facies_props": facies_props_arg}),
        **preset_kwargs,
    )
    por3d = np.array(model.poro_mat, dtype=np.float32, copy=True)
    fac3d = np.array(model.facies, dtype=np.int8, copy=True)
    # The int8 cast must be lossless.
    assert np.array_equal(fac3d.astype(np.int64), np.asarray(model.facies).astype(np.int64))
    return por3d, fac3d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    if cfg["generator"]["layer_class"] != "ChannelLayer":
        raise ValueError("only ChannelLayer is allowed (GaussianLayer must not be used)")
    if cfg["generator"]["facies_props"] != "default":
        raise NotImplementedError("only the default ResMill FACIES_PROPS table is supported")

    check_orientation_helper()  # synthetic-array assertion of the 3D -> project 2D mapping

    preset_name = cfg["generator"]["preset"]
    preset_kwargs = dict(getattr(rm_channel, preset_name))
    facies_props = {int(k): dict(v) for k, v in rm_channel.FACIES_PROPS.items()}
    codes = sorted(facies_props)
    # Optional per-facies porosity spread (ResMill ``poro_sd``: log-normal relative
    # spread, correlated over poro_noise_range cells, drawn after the geometry).
    # Absent from the config = default table untouched and not passed to ResMill.
    poro_sd_cfg = cfg["generator"].get("facies_poro_sd") or {}
    facies_props_arg = None
    if poro_sd_cfg:
        for code, sd in poro_sd_cfg.items():
            if int(code) not in facies_props:
                raise ValueError("facies_poro_sd: unknown facies code %r" % code)
            facies_props[int(code)]["poro_sd"] = float(sd)
        facies_props_arg = facies_props
    k = slice_index_from_rule(cfg)
    seeds = list(range(int(cfg["geological_seeds"]["start"]), int(cfg["geological_seeds"]["stop_exclusive"])))
    diag_seeds = [int(s) for s in cfg["diagnostics"]["seeds"]]
    tgt_mean = float(cfg["rescale"]["target_mean"])
    tgt_std = float(cfg["rescale"]["target_stdev"])

    # --- FATAL: determinism check on the first diagnostic seed ------------
    d0 = diag_seeds[0]
    t0 = time.time()
    a1 = generate_one(d0, cfg, preset_kwargs, facies_props_arg)
    a2 = generate_one(d0, cfg, preset_kwargs, facies_props_arg)
    h1, h2 = array_sha256(*a1), array_sha256(*a2)
    determinism = {"seed": d0, "hash_run1": h1, "hash_run2": h2, "passed": h1 == h2}
    if not determinism["passed"]:
        raise RuntimeError("ResMill realization for seed %d is not reproducible: %s" % (d0, determinism))
    print("Determinism check seed %d passed (%.1f s for 2 runs)" % (d0, time.time() - t0))
    cache = {d0: a1}

    run_dir = make_run_dir(cfg["experiment"])
    print("Run directory: %s" % run_dir)

    records = []
    arrays2d = {}
    hashes = {}
    checks = {}
    for seed in seeds:
        t = time.time()
        por3d, fac3d = cache.pop(seed) if seed in cache else generate_one(seed, cfg, preset_kwargs, facies_props_arg)
        por2d = slice_to_project_2d(por3d, k)
        fac2d = slice_to_project_2d(fac3d, k)
        stem = stem_for(preset_name, seed)

        slice_std = float(np.std(por2d.astype(np.float64)))
        if slice_std > float(cfg["qc"]["min_slice_std"]):
            truth, a, b = affine_rescale(por2d, tgt_mean, tgt_std)
            affine = {"a": a, "b": b}
        else:
            truth, affine = None, None

        files = {
            stem + FILE_SUFFIXES["porosity_3d"]: por3d,
            stem + FILE_SUFFIXES["facies_3d"]: fac3d,
            stem + FILE_SUFFIXES["porosity_2d_frac"]: por2d,
            stem + FILE_SUFFIXES["facies_2d"]: fac2d,
        }
        if truth is not None:
            files[stem + FILE_SUFFIXES["truth_2d"]] = truth
        files_sha = {}
        for name, arr in files.items():
            np.save(run_dir / name, arr)
            files_sha[name] = file_sha256(run_dir / name)

        chk = run_qc.realization_checks(por3d, fac3d, por2d, fac2d, truth, cfg, codes)
        checks[seed] = chk
        hashes[seed] = array_sha256(por3d, fac3d)
        arrays2d[seed] = (por2d, fac2d)
        rec = {
            "id": stem,
            "geological_seed": seed,
            "preset": preset_name,
            "slice_index": k,
            "native_dtypes": {"porosity_3d": "float32", "facies_3d": "int8"},
            "summary": run_qc.realization_summary(por3d, fac3d, por2d, fac2d, codes),
            "affine": affine,
            "truth_2d": run_qc._stats(truth) if truth is not None else None,
            "files_sha256": files_sha,
        }
        records.append(rec)
        print("  seed %3d  %.1f s  slice mean=%.4f std=%.4f  NTG2d=%.3f"
              % (seed, time.time() - t, rec["summary"]["porosity_2d_frac"]["mean"],
                 rec["summary"]["porosity_2d_frac"]["std"], rec["summary"]["net_to_gross_2d"]))

    # --- Duplicate rule (needs all seeds) ---------------------------------
    dup = run_qc.duplicate_map(hashes)
    for seed in seeds:
        checks[seed]["duplicate"] = {
            "passed": dup[seed] is None,
            "detail": "ok" if dup[seed] is None else "identical 3D porosity+facies to seed %d" % dup[seed],
        }

    # --- FATAL: slice rule, re-read from disk ------------------------------
    for rec in records:
        sr = run_qc.slice_rule_check(run_dir, rec["id"], k, rec["affine"])
        checks[rec["geological_seed"]]["slice_rule"] = sr
        if not sr["passed"]:
            raise RuntimeError("slice rule check failed for %s: %s" % (rec["id"], sr["detail"]))

    # --- Exclusions ----------------------------------------------------------
    excluded = []
    for rec in records:
        seed = rec["geological_seed"]
        failed = [rid for rid in run_qc.EXCLUSION_RULE_IDS if not checks[seed][rid]["passed"]]
        rec["qc"] = checks[seed]
        rec["status"] = "excluded" if failed else "included"
        rec["exclusion_reasons"] = ["%s: %s" % (rid, checks[seed][rid]["detail"]) for rid in failed]
        if failed:
            excluded.append({"id": rec["id"], "geological_seed": seed, "rules_failed": failed,
                             "reasons": rec["exclusion_reasons"]})

    # --- Diagnostics (predefined seeds, regardless of status) -------------
    shared_files = []
    for seed in diag_seeds:
        rec = next(r for r in records if r["geological_seed"] == seed)
        por2d, fac2d = arrays2d[seed]
        rec["diagnostics"] = run_qc.make_diagnostic_plot(run_dir, rec["id"], seed, por2d, fac2d, cfg, codes)
        shared_files.append(rec["diagnostics"]["file"])

    shutil.copyfile(config_path, run_dir / CONFIG_COPY_NAME)
    shared_files.append(CONFIG_COPY_NAME)

    # --- FATAL: files on disk match the per-file records ------------------
    file_records = {}
    for rec in records:
        file_records.update(rec["files_sha256"])
    pre = run_qc.files_match_check(
        run_dir, file_records, expected_extra=shared_files + [run_qc.QC_SUMMARY_FILE])
    if not pre["passed"]:
        raise RuntimeError("files_match check failed: %s" % pre["detail"])

    qc_rows = []
    for rec in records:
        q = rec["qc"]
        s = rec["summary"]
        qc_rows.append({
            "realization_id": rec["id"],
            "geological_seed": rec["geological_seed"],
            **{"check_%s" % rid: q[rid]["passed"] for rid in run_qc.EXCLUSION_RULE_IDS},
            "check_slice_rule": q["slice_rule"]["passed"],
            "check_files_match": all(n in file_records for n in rec["files_sha256"]) and pre["passed"],
            "por3d_min": s["porosity_3d"]["min"],
            "por3d_max": s["porosity_3d"]["max"],
            "por2d_std": s["porosity_2d_frac"]["std"],
            "n_invalid_facies_3d": int(q["facies_codes"]["detail"].split("=")[1]),
            "duplicate_of": "" if dup[rec["geological_seed"]] is None else dup[rec["geological_seed"]],
            "status": rec["status"],
            "exclusion_reasons": " | ".join(rec["exclusion_reasons"]),
        })
    run_qc.write_qc_summary(run_dir / run_qc.QC_SUMMARY_FILE, qc_rows)
    shared_files.append(run_qc.QC_SUMMARY_FILE)
    shared_sha = {name: file_sha256(run_dir / name) for name in shared_files}

    output_files = sorted(list(file_records) + shared_files)
    import matplotlib
    import scipy
    import yaml

    params = {
        "config": cfg,
        "config_path": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "slice_index": k,
        "create_geology_kwargs": {
            "poro_noise_std": float(cfg["porosity"]["poro_noise_std"]),
            "poro_noise_range": float(cfg["porosity"]["poro_noise_range"]),
            "preset_name": preset_name,
            "preset_values": preset_kwargs,
        },
        "facies_props": {str(c): facies_props[c] for c in codes},
        "net_definition": "facies code >= 1 (CS, LV, LA, CH)",
        "array_conventions": {
            "3d": "native ResMill (nx, ny, nz) indexed [ix, iy, iz]; x increases with ix, y with iy",
            "2d": "project convention (ny, nx), row 0 = max-y, column = ix; = np.flipud(arr3d[:, :, k].T)",
        },
        "rescale_formula": "truth = (x - mean(x)) * (target_stdev / std(x)) + target_mean, std ddof=0, per realization",
        "shared_files_sha256": shared_sha,
    }
    extra = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "software": {
            "resmill": resmill_provenance(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__,
        },
        "seed_types": {
            "geological_seed": seeds,
            "sampling_seed": "not applicable (ground-truth generation only)",
            "method_seed": "not applicable (ground-truth generation only)",
        },
        "determinism_check": determinism,
        "qc_counts": {
            "n_generated": len(records),
            "n_included": sum(r["status"] == "included" for r in records),
            "n_excluded": len(excluded),
        },
        "excluded_realizations": excluded,
        "realizations": records,
    }
    manifest_path = save_result(
        experiment=cfg["experiment"],
        params=params,
        seed_or_seeds=seeds,
        run_dir=run_dir,
        code_entrypoint=CODE_ENTRYPOINT,
        output_files=output_files,
        extra=extra,
    )

    # --- FATAL: post-manifest consistency (output_files <-> disk) ----------
    problems = run_qc.verify_run(run_dir)
    if problems:
        raise RuntimeError("post-manifest QC re-verification failed: %s" % problems)

    print("Manifest: %s" % manifest_path)
    print("QC: %d generated, %d included, %d excluded" % (
        extra["qc_counts"]["n_generated"], extra["qc_counts"]["n_included"], extra["qc_counts"]["n_excluded"]))
    for e in excluded:
        print("  excluded %s: %s" % (e["id"], e["reasons"]))
    return run_dir


if __name__ == "__main__":
    main()
