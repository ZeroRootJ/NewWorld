"""Build the "Density Axis Variogram Check" artifact HTML from a
check_density_axis_variogram run (results/raw is read, never modified).

Writes results/processed/density_axis_variogram_check/variogram_check.html by
injecting the run's tidy CSV (as JSON) into artifact_template.html in the
same folder.

Run with: ../../../.venv/Scripts/python.exe -m src.experiments.build_density_variogram_artifact [run_dir]
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

from src.experiments.base_case_conditioning import HMAJ1, NUG, SAMPLE_SEED
from src.experiments.check_density_axis_variogram import (
    EXPERIMENT_NAME, LAG_DIST, LAG_TOL, NLAG,
)
from src.experiments.sample_density_axis import EXTENDED_AXIS_LEVELS

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = _REPO_ROOT / "results" / "processed" / EXPERIMENT_NAME


def main(run_dir: Path):
    df = pd.read_csv(run_dir / "variogram_long.csv", dtype={"axis_level": str})
    df["axis_level"] = df["axis_level"].fillna("").str.replace(r"\.0$", "", regex=True)

    model = df[df.source == "model"][["lag_m", "gamma"]].values.round(4).tolist()

    smp = df[(df.source == "samples") & (df.lag_m > 0)]
    seeds = sorted(int(s) for s in smp.truth_seed.unique())
    samples = {
        s: {lv: g[["lag_m", "gamma", "npairs"]].values.round(4).tolist()
            for lv, g in smp[smp.truth_seed == s].groupby("axis_level")}
        for s in seeds
    }
    levels = [dict(lv=lv, n=int(smp[(smp.axis_level == lv)].n_samples_actual.iloc[0]))
              for lv in EXTENDED_AXIS_LEVELS]

    exh = {}
    ex = df[df.source == "exhaustive"]
    for s in seeds:
        e = ex[ex.truth_seed == s].groupby("lag_m").gamma.mean().reset_index()
        exh[s] = e.values.round(4).tolist()

    ymax = math.ceil(max(smp.gamma.max(), ex.gamma.max()) / 0.5) * 0.5

    data = dict(
        model=model, seeds=seeds, levels=levels, samples=samples, exh=exh,
        nug=NUG, range=HMAJ1, ymax=ymax,
        meta=[
            ["model", "nug %.2f + sph %.2f, a = %g m, sill 1" % (NUG, 1 - NUG, HMAJ1)],
            ["samples", "SAMPLE_SEED %d → nscore → gamv omni" % SAMPLE_SEED],
            ["lag", "%g ± %g m × %d" % (LAG_DIST, LAG_TOL, NLAG)],
            ["run", "%s/%s" % (run_dir.parent.name, run_dir.name)],
        ],
    )
    tpl = (OUT_DIR / "artifact_template.html").read_text(encoding="utf-8")
    out = OUT_DIR / "variogram_check.html"
    out.write_text(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rd = Path(sys.argv[1])
    else:
        rd = sorted((_REPO_ROOT / "results" / "raw" / EXPERIMENT_NAME).iterdir())[-1]
    main(rd)
