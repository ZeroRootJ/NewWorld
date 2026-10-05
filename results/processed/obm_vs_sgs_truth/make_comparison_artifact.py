"""Build the self-contained HTML page comparing the existing SGS truths with
the ALLUVSIM-based (ResMill ChannelLayer) truths, from the processed tables in
this directory. No recomputation: it only reshapes summary_stats.csv,
histogram_bins.csv, variogram.csv and example_fields.json for embedding.

Run with: .venv/Scripts/python.exe results/processed/obm_vs_sgs_truth/make_comparison_artifact.py
Writes:   .artifact_tmp/obm_vs_sgs_truth.html (gitignored; the published artifact is the original)
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TEMPLATE = HERE / "comparison_artifact_template.html"
OUT = REPO / ".artifact_tmp" / "obm_vs_sgs_truth.html"

STATS_ORDER = ["min", "p10", "p50", "p90", "max", "skewness", "excess_kurtosis",
               "largest_tie_fraction", "net_to_gross"]


def main():
    stats = pd.read_csv(HERE / "summary_stats.csv")
    agg = stats.groupby(["family", "statistic"]).value.agg(["mean", "min", "max"]).reset_index()
    stats_rows = []
    for st in STATS_ORDER:
        row = {"statistic": st}
        for fam in ("sgs", "obm", "obm_mud"):
            r = agg[(agg.family == fam) & (agg.statistic == st)]
            row[fam] = None if r.empty else [round(float(r[c].iloc[0]), 3) for c in ("mean", "min", "max")]
        stats_rows.append(row)
    n_seeds = stats.groupby("family").seed.nunique().to_dict()

    hist = pd.read_csv(HERE / "histogram_bins.csv")
    hist_out = {
        fam: g[["bin_left", "bin_right", "density"]].round(5).values.tolist()
        for fam, g in hist.groupby("family")
    }

    vario = pd.read_csv(HERE / "variogram.csv")
    va = vario.groupby(["family", "direction", "lag_m"]).gamma_std.agg(["mean", "min", "max"]).reset_index()
    vario_out = {}
    for (fam, d), g in va.groupby(["family", "direction"]):
        vario_out.setdefault(fam, {})[d] = g[["lag_m", "mean", "min", "max"]].round(4).values.tolist()

    fields = json.loads((HERE / "example_fields.json").read_text(encoding="utf-8"))
    sources = json.loads((HERE / "source_runs.json").read_text(encoding="utf-8"))

    data = {
        "n_seeds": n_seeds,
        "stats": stats_rows,
        "hist": hist_out,
        "vario": vario_out,
        "fields": fields,
        "obm_manifest": sources["obm"]["manifest_path"],
        "resmill_commit": sources["obm"]["resmill"]["git_commit"],
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":"))
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
