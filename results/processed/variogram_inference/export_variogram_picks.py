"""Export the user's manual variogram picks (Variogram Review artifact db) to a
tracked CSV so later experiments can reuse them without the artifact.

The picks live in the artifact's db, collection "picks", one document per cell
(doc id = cell id = f"{experiment}_{config_key}"). They are dumped to local JSON
files first (coordinator: ArtifactData list, collection "picks", out_dir=<dump>),
then this script joins them to fitted_variograms.csv and writes:

  variogram_picks.csv        one row per picked cell (range_m / nugget = user value)
  variogram_picks_meta.json  source artifact, dump time, row counts

Downstream rule: use the pick when a row exists, otherwise the automatic fit
(fit_range_m / fit_nugget in fitted_variograms.csv). `confirmed` says whether the
user pressed confirm; `edited` whether the value differs from the auto fit.

usage: python export_variogram_picks.py <dump_dir>/picks
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ARTIFACT_URL = "https://claude.ai/artifact/BzfP9XfxxioiWpdUq6ntBC"
DUMP = Path(sys.argv[1])

fit = pd.read_csv(HERE / "fitted_variograms.csv", dtype={"axis_level": str, "sample_rep": str})
fit["cell_id"] = fit["experiment"] + "_" + fit["config_key"]
fit = fit.set_index("cell_id")

rows = []
for f in sorted(DUMP.glob("*.json")):
    doc = json.loads(f.read_text(encoding="utf-8"))
    d = doc.get("data", doc)  # tolerate both {data: {...}} and bare documents
    cid = f.stem
    if cid not in fit.index:
        raise KeyError(f"pick {cid} has no matching cell in fitted_variograms.csv")
    r = fit.loc[cid]
    rng, nug = float(d["range_m"]), float(d["nugget"])
    rows.append({
        "cell_id": cid,
        "experiment": r["experiment"],
        "axis": r["axis"],
        "axis_level": r["axis_level"],
        "truth_seed": int(r["truth_seed"]),
        "sample_rep": r["sample_rep"],
        "sample_seed": int(r["sample_seed"]),
        "n_samples": int(r["n_samples"]),
        "range_m": rng,
        "nugget": nug,
        "confirmed": bool(d.get("confirmed", False)),
        "edited": abs(rng - round(float(r["fit_range_m"]), 1)) > 0.05
                  or abs(nug - round(float(r["fit_nugget"]), 3)) > 0.0005,
        "auto_range_m": float(r["fit_range_m"]),
        "auto_nugget": float(r["fit_nugget"]),
        "true_range_m": float(r["true_range_m"]),
        "true_nugget": float(r["true_nugget"]),
        "updated": d.get("updated", ""),
    })

cols = ["cell_id", "experiment", "axis", "axis_level", "truth_seed", "sample_rep", "sample_seed",
        "n_samples", "range_m", "nugget", "confirmed", "edited", "auto_range_m", "auto_nugget",
        "true_range_m", "true_nugget", "updated"]
out = pd.DataFrame(rows, columns=cols).sort_values(["experiment", "axis_level", "truth_seed", "sample_rep"])
out.to_csv(HERE / "variogram_picks.csv", index=False)
meta = {
    "source_artifact": ARTIFACT_URL,
    "db_collection": "picks",
    "exported_at": datetime.now(timezone.utc).isoformat(),
    "n_picks": len(out),
    "n_confirmed": int(out["confirmed"].sum()),
    "n_edited": int(out["edited"].sum()),
    "by_experiment": out.groupby("experiment").size().to_dict(),
    "rule": "use range_m/nugget here when a row exists, else fit_range_m/fit_nugget from fitted_variograms.csv",
}
(HERE / "variogram_picks_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(len(out), "picks ->", HERE / "variogram_picks.csv")
