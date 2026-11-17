"""Build the Variogram Review artifact HTML (interactive range/nugget picker).

Reads results/processed/variogram_inference/{experimental,fitted}_variograms.csv
and writes one self-contained HTML page. The user's edited picks are stored in
the artifact's db (collection "picks", doc id = cell id), not in this file.

usage: python build_variogram_review_artifact.py <out.html>
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])

fit = pd.read_csv(HERE / "fitted_variograms.csv", dtype={"axis_level": str, "sample_rep": str})
exp = pd.read_csv(HERE / "experimental_variograms.csv", dtype={"axis_level": str, "sample_rep": str})
manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

LEVEL_LABEL = {
    "E2": lambda l: f"range {float(l):g} m",
    "E3": lambda l: f"nugget {float(l):g}",
    "E4": lambda l: f"density {float(l):g}%",
    "E5": lambda l: "base case",
}
EXP_LABEL = {"E2": "Range axis", "E3": "Nugget axis", "E4": "Density axis", "E5": "Base case"}


def cell_id(r):
    return f"{r['experiment']}_{r['config_key']}"


exp["cid"] = exp.apply(cell_id, axis=1)
groups = {k: g for k, g in exp.groupby("cid")}

cells = []
for _, r in fit.iterrows():
    cid = cell_id(r)
    g = groups[cid].sort_values("lag_m")
    cells.append({
        "id": cid,
        "e": r["experiment"],
        "lv": str(r["axis_level"]),
        "lvl": LEVEL_LABEL[r["experiment"]](r["axis_level"]),
        "t": int(r["truth_seed"]),
        "s": str(r["sample_rep"]),
        "n": int(r["n_samples"]),
        "tr": float(r["true_range_m"]),
        "tn": float(r["true_nugget"]),
        "fr": round(float(r["fit_range_m"]), 1),
        "fn": round(float(r["fit_nugget"]), 3),
        "rmse": round(float(r["fit_rmse"]), 3),
        "lag": round(float(r["lag_dist_m"]), 1),
        "b": [k for k in ("nugget_at_lower", "nugget_at_upper", "range_at_lower", "range_at_upper") if bool(r[k])],
        # points: [lag_m, gamma, npairs, used]
        "p": [[round(float(a), 1), round(float(b), 4), int(c), 1 if bool(d) else 0]
              for a, b, c, d in zip(g["lag_m"], g["gamma"], g["npairs"], g["used_in_fit"])],
    })


def level_sort(e, lv):
    try:
        return float(lv)
    except ValueError:
        return 0.0


order = {"E2": 0, "E3": 1, "E4": 2, "E5": 3}
cells.sort(key=lambda c: (order[c["e"]], level_sort(c["e"], c["lv"]), c["t"], c["s"]))

params = manifest.get("params", {})
lag_rule = params.get("lag_rule") or params.get("LAG_RULE") or {}
fit_rule = params.get("fit_rule") or params.get("FIT_RULE") or {}

data = {
    "cells": cells,
    "expLabel": EXP_LABEL,
    "commit": manifest.get("git_commit", ""),
    "lagRule": lag_rule,
    "fitRule": fit_rule,
}

TEMPLATE = (HERE / "variogram_review_template.html").read_text(encoding="utf-8")
html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
OUT.write_text(html, encoding="utf-8")
print(OUT, OUT.stat().st_size, len(cells), "cells")
