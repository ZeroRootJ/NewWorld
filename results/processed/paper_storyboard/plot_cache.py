"""Plot-data cache for the storyboard figures (make_storyboard_figures.py and
make_storyboard_diagnostics.py).

Every storyboard figure is split into
  * a COMPUTE step (reads the input files, runs every consistency check and
    calculation) that returns a plain dict of plot data, and
  * a DRAW step that only takes that dict and draws with matplotlib.

The dict is stored here, per figure, under
results/processed/paper_storyboard/plot_data/:

  <stem>.json          scalars / strings / lists / dicts (incl. the DATA STATUS
                       note and the .status.txt lines), the tables' column
                       dtypes, and a "meta" block: git commit, compute time and
                       every input file read during the compute step
                       (repo-relative path, size, mtime, sha256)
  <stem>.<name>.csv    one CSV per pandas table (index not stored)
  <stem>.npz           numpy arrays (only if the figure has any)

Round trip is exact: floats are written with their shortest round-trip repr
(JSON, CSV) and read back with float_precision="round_trip"; ``save`` reloads
what it wrote and asserts it equals the in-memory dict (values AND types) before
returning. Input files are recorded automatically by ``track_reads`` (every
file under the repo opened for reading while the compute step runs), not by a
hand-written list.
"""

import builtins
import contextlib
import datetime
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PLOT_DATA_DIR = Path(__file__).resolve().parent / "plot_data"
FORMAT_VERSION = 1

# read-tracking: only data files inside the repo, never code / caches / our own outputs
_IGNORE_PARTS = {".venv", "__pycache__", ".git", ".idea"}
_IGNORE_SUFFIXES = {".py", ".pyc", ".pyd", ".nbi", ".nbc"}


class CacheMissingError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# input tracking
# ---------------------------------------------------------------------------
def _is_tracked_path(p: Path) -> bool:
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        return False
    if set(rel.parts) & _IGNORE_PARTS or p.suffix.lower() in _IGNORE_SUFFIXES:
        return False
    if rel.parts[:2] == ("results", "figures"):
        return False
    try:
        p.relative_to(PLOT_DATA_DIR)
        return False
    except ValueError:
        return True


@contextlib.contextmanager
def track_reads(sink: set):
    """Record every repo data file opened for reading (builtins.open / io.open,
    which pandas.read_csv, numpy.load and Path.read_text all go through) into
    ``sink`` as absolute Paths."""
    orig_builtin, orig_io = builtins.open, io.open

    def _wrap(fn):
        def opener(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and not any(c in mode for c in "wax+"):
                p = Path(os.fsdecode(os.fspath(file))).resolve()
                if _is_tracked_path(p):
                    sink.add(p)
            return fn(file, mode, *args, **kwargs)
        return opener

    builtins.open, io.open = _wrap(orig_builtin), _wrap(orig_io)
    try:
        yield sink
    finally:
        builtins.open, io.open = orig_builtin, orig_io


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(p: Path) -> dict:
    st = p.stat()
    return {"path": p.relative_to(REPO_ROOT).as_posix(), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "sha256": _sha256(p)}


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.rstrip("\n")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stale_inputs(meta: dict) -> list:
    """Inputs recorded in a cache's meta block that changed since it was computed:
    missing, or a different size / content (sha256 checked only when size or mtime
    differ). Returns a list of (path, reason)."""
    out = []
    for rec in meta.get("inputs", []):
        p = REPO_ROOT / rec["path"]
        if not p.exists():
            out.append((rec["path"], "missing"))
            continue
        st = p.stat()
        if st.st_size != rec["size"]:
            out.append((rec["path"], "size changed"))
        elif st.st_mtime_ns != rec["mtime_ns"] and _sha256(p) != rec["sha256"]:
            out.append((rec["path"], "content changed"))
    return out


# ---------------------------------------------------------------------------
# (de)serialization
# ---------------------------------------------------------------------------
def _encode(obj, path, stem, tables, arrays):
    """Replace tables / arrays by references; tag the Python types JSON cannot
    represent (tuple, non-string dict keys) so they come back unchanged."""
    if isinstance(obj, pd.DataFrame):
        name = path or "table"
        if not all(isinstance(c, str) for c in obj.columns):
            raise TypeError(f"plot cache {stem}:{name}: non-string column names {list(obj.columns)}")
        fname = f"{stem}.{name}.csv"
        tables[fname] = obj.reset_index(drop=True)
        return {"__table__": fname, "dtypes": {c: str(t) for c, t in obj.dtypes.items()}}
    if isinstance(obj, pd.Series):
        raise TypeError(f"plot cache {stem}:{path}: store a DataFrame / list, not a Series")
    if isinstance(obj, np.ndarray):
        key = path or "array"
        if key in arrays:
            raise ValueError(f"plot cache {stem}: duplicate array key {key}")
        arrays[key] = obj
        return {"__array__": key}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, f"{path}.{i}", stem, tables, arrays) for i, v in enumerate(obj)]}
    if isinstance(obj, list):
        return [_encode(v, f"{path}.{i}", stem, tables, arrays) for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            if any(k.startswith("__") for k in obj):
                raise ValueError(f"plot cache {stem}:{path}: keys starting with '__' are reserved")
            return {k: _encode(v, f"{path}.{k}" if path else k, stem, tables, arrays) for k, v in obj.items()}
        return {"__items__": [[_encode(k, f"{path}.k{i}", stem, tables, arrays),
                               _encode(v, f"{path}.{i}", stem, tables, arrays)]
                              for i, (k, v) in enumerate(obj.items())]}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"plot cache {stem}:{path}: unsupported type {type(obj).__name__}")


def _read_table(fname: str, dtypes: dict) -> pd.DataFrame:
    text_cols = [c for c, t in dtypes.items() if t == "object"]
    num_cols = [c for c in dtypes if c not in text_cols]
    df = pd.read_csv(PLOT_DATA_DIR / fname, dtype={c: str for c in text_cols}, keep_default_na=False,
                     na_values={c: [""] for c in num_cols}, float_precision="round_trip")
    if list(df.columns) != list(dtypes):
        raise ValueError(f"{fname}: columns {list(df.columns)} != {list(dtypes)}")
    for c, t in dtypes.items():
        if t != "object" and str(df[c].dtype) != t:
            df[c] = df[c].astype(t)
    return df


def _decode(obj, arrays):
    if isinstance(obj, dict):
        if "__table__" in obj:
            return _read_table(obj["__table__"], obj["dtypes"])
        if "__array__" in obj:
            return arrays[obj["__array__"]]
        if "__tuple__" in obj:
            return tuple(_decode(v, arrays) for v in obj["__tuple__"])
        if "__items__" in obj:
            return {_decode(k, arrays): _decode(v, arrays) for k, v in obj["__items__"]}
        return {k: _decode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    return obj


def _assert_same(a, b, where):
    """Exact equality of an in-memory value and its reloaded copy (types included)."""
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b, check_exact=True, obj=where)
        return
    if isinstance(a, np.ndarray):
        same = (isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
        if not same:
            raise AssertionError(f"plot cache round trip: array {where} differs")
        return
    if isinstance(a, np.generic):
        a = a.item()
    if type(a) is not type(b):
        raise AssertionError(f"plot cache round trip: {where} type {type(a).__name__} -> {type(b).__name__}")
    if isinstance(a, dict):
        if list(a) != list(b):
            raise AssertionError(f"plot cache round trip: {where} keys differ")
        for k in a:
            _assert_same(a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"plot cache round trip: {where} length differs")
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{where}[{i}]")
    elif isinstance(a, float) and np.isnan(a):
        if not np.isnan(b):
            raise AssertionError(f"plot cache round trip: {where} NaN lost")
    elif a != b or (isinstance(a, float) and repr(a) != repr(b)):
        raise AssertionError(f"plot cache round trip: {where} {a!r} -> {b!r}")


def _files_of(stem: str) -> list:
    return sorted(p for p in PLOT_DATA_DIR.glob(f"{stem}.*") if p.name.startswith(stem + "."))


def save(key: str, stem: str, data: dict, inputs, generated_by: str) -> list:
    """Write the plot data of one figure, verify the exact round trip, return the files written."""
    PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in _files_of(stem):
        old.unlink()
    tables, arrays = {}, {}
    payload = _encode(data, "", stem, tables, arrays)
    for fname, df in tables.items():
        df.to_csv(PLOT_DATA_DIR / fname, index=False, lineterminator="\n")
    if arrays:
        np.savez_compressed(PLOT_DATA_DIR / f"{stem}.npz", **arrays)
    meta = {
        "format_version": FORMAT_VERSION,
        "figure": key, "stem": stem, "generated_by": generated_by,
        "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_files": [ln[3:] for ln in _git("status", "--porcelain").splitlines() if ln.strip()],
        "inputs": [_fingerprint(p) for p in sorted(inputs)],
        "tables": sorted(tables), "arrays_file": f"{stem}.npz" if arrays else None,
        "array_keys": sorted(arrays),
    }
    with open(PLOT_DATA_DIR / f"{stem}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"meta": meta, "data": payload}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    back, _ = load(stem)
    _assert_same(data, back, stem)
    return _files_of(stem)


def load(stem: str):
    """(data, meta) of one figure; CacheMissingError if it was never computed."""
    jpath = PLOT_DATA_DIR / f"{stem}.json"
    if not jpath.exists():
        raise CacheMissingError(
            f"no plot cache for {stem}: {jpath.relative_to(REPO_ROOT).as_posix()} does not exist. Run the "
            "compute step first (the same command without --draw-only).")
    with open(jpath, encoding="utf-8") as f:
        doc = json.load(f)
    meta = doc["meta"]
    if meta.get("format_version") != FORMAT_VERSION:
        raise CacheMissingError(f"{jpath.name}: cache format {meta.get('format_version')} != {FORMAT_VERSION}; "
                                "recompute (run without --draw-only).")
    arrays = {}
    if meta.get("arrays_file"):
        with np.load(PLOT_DATA_DIR / meta["arrays_file"], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    return _decode(doc["data"], arrays), meta


def cache_size(stem: str) -> int:
    return sum(p.stat().st_size for p in _files_of(stem))
