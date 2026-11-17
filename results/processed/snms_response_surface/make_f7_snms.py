"""F7 alternative: MSE / UMG response surfaces from the OLD SNMS grid experiment
(pre-NewWorld project "Kriging VS GP", Dec 2025 - Feb 2026), drawn in the same
layout as the storyboard F7 (4 methods x {MSE, UMG}, one pcolormesh cell per
grid node, no interpolation).

Source (read-only, outside this repo):
  <OLD_ROOT>/Results/Kriging/*.pkl        ordinary kriging (kb2d ktype=1), 19 range x 10 nugget
  <OLD_ROOT>/Results/SGSIM3/pkls/19.pkl   SGS, 9 range x 10 nugget (file N is cumulative: seeds 1..N)
  <OLD_ROOT>/Results/RBF2/*.pkl           RBF (gaussian, eps = 2.146/range) + bootstrap, 10 x 20 smoothing
  <OLD_ROOT>/Results/GP4/*.pkl            GP C(1)*RBF(ls = range/2.146), fixed, 10 x 10 alpha
Data: SNMSv5.csv (real porosity), normal-score, random 70/30 train/test split per
replicate (sklearn random_state = replicate id). Metrics are on the 30 % test points.

Step 1 (cached): pull only the scalar columns (range, param2, mse, umg) out of the
pkls -> old_grid_long.csv. The pkls were written with numpy 2.x; the numpy 1.x
.venv reads them through a numpy._core -> numpy.core alias.
Step 2: mean over replicates per node -> metrics_summary.csv, optima.csv, figure.
"""
import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "results" / "processed" / "paper_storyboard"))
import make_storyboard_figures as sb  # noqa: E402  (style helpers only)

OLD_ROOT_DEFAULT = (r"C:\Users\yj7563\OneDrive - The University of Texas at Austin\Documents"
                    r"\Exchange Student\GeoStats_Study\Kriging VS GP")
LONG_CSV = HERE / "old_grid_long.csv"
SUMMARY_CSV = HERE / "metrics_summary.csv"
OPTIMA_CSV = HERE / "optima.csv"
STEM = "F7_snms_response_surfaces"

METHODS = ["kriging", "sgs", "rbf", "gp"]
TITLES = {"kriging": "Ordinary kriging", "sgs": "SGS", "rbf": "RBF+bootstrap", "gp": "GP (fixed hyperparameters)"}
YLABEL = {"kriging": "Nugget (fraction of sill)", "sgs": "Nugget (fraction of sill)",
          "rbf": "Smoothing λ", "gp": "Noise α (normalized)"}
YLOG = {"kriging": False, "sgs": False, "rbf": True, "gp": True}
YTICKS = {"kriging": [0.0, 0.3, 0.6, 0.9], "sgs": [0.0, 0.3, 0.6, 0.9],
          "rbf": [1e-6, 1e-4, 1e-2, 1e0], "gp": [1e-3, 1e-2, 1e-1, 1e0]}
XTICKS = [500, 1000, 2000, 5000]
CMAP = {"mse": "viridis_r", "umg": "viridis"}
BEST = {"mse": "min", "umg": "max"}


def _numpy2_alias():
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)


def extract(old_root: str) -> pd.DataFrame:
    _numpy2_alias()
    rows = []

    def take(df, method, p2, rep, f):
        s = df[["range", p2, "mse", "umg"]].rename(columns={p2: "param2"}).copy()
        s.insert(0, "replicate", rep)
        s.insert(0, "method", method)
        s["param2_name"] = p2
        s["source_file"] = os.path.relpath(f, old_root)
        rows.append(s)

    for method, sub, p2 in (("kriging", "Kriging", "nugget"), ("rbf", "RBF2", "smoothing"), ("gp", "GP4", "alpha")):
        for f in sorted(glob.glob(os.path.join(old_root, "Results", sub, "*.pkl"))):
            rep = int(Path(f).stem.split("_")[-1])        # file index = train/test split seed
            take(pickle.load(open(f, "rb")), method, p2, rep, f)
    # SGS: pkls/N.pkl holds seeds 1..N back to back (results list never reset) -> use the largest only
    sgs_files = glob.glob(os.path.join(old_root, "Results", "SGSIM3", "pkls", "*.pkl"))
    f = max(sgs_files, key=lambda p: int(Path(p).stem))
    df = pickle.load(open(f, "rb")).reset_index(drop=True)
    n_nodes = df[["range", "nugget"]].drop_duplicates().shape[0]
    if len(df) % n_nodes:
        raise ValueError("SGS cumulative file not a whole number of seed blocks")
    nodes = df[["range", "nugget"]].to_numpy().reshape(-1, n_nodes, 2)
    if not (nodes == nodes[0]).all():
        raise ValueError("SGS node order differs between seed blocks")
    for b in range(len(df) // n_nodes):
        take(df.iloc[b * n_nodes:(b + 1) * n_nodes], "sgs", "nugget", b + 1, f)
    out = pd.concat(rows, ignore_index=True)
    out["param2"] = out["param2"].round(12)
    return out


def summarize(long: pd.DataFrame):
    summ = (long.melt(id_vars=["method", "replicate", "range", "param2", "param2_name"],
                      value_vars=["mse", "umg"], var_name="metric")
            .groupby(["method", "param2_name", "range", "param2", "metric"])["value"]
            .agg(mean="mean", std="std", n="count").reset_index())
    opt = []
    for (m, met), g in summ.groupby(["method", "metric"]):
        g = g.sort_values("mean", ascending=(met == "mse"))
        b, r = g.iloc[0], g.iloc[1]
        opt.append(dict(method=m, metric=met, criterion=BEST[met], range=b["range"], param2_name=b["param2_name"],
                        param2=b["param2"], mean=b["mean"], std_over_replicates=b["std"], n=b["n"],
                        runner_up_range=r["range"], runner_up_param2=r["param2"], runner_up_mean=r["mean"]))
    return summ, pd.DataFrame(opt)


def grid(summ, m, met):
    g = summ[(summ.method == m) & (summ.metric == met)]
    xs, ys = np.sort(g["range"].unique()), np.sort(g["param2"].unique())
    z = g.pivot(index="param2", columns="range", values="mean").loc[ys, xs].to_numpy()
    if np.isnan(z).any():
        raise ValueError(f"{m} {met}: incomplete grid")
    return xs, ys, z


def figure(summ, opt, nrep):
    import matplotlib
    from matplotlib import patheffects
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterMathtext, NullLocator

    sb._setup_style()
    grids = {m: grid(summ, m, "mse")[:2] for m in METHODS}
    gtxt = ", ".join(f"{TITLES[m].split(' (')[0]} {len(grids[m][0])}×{len(grids[m][1])}" for m in METHODS)
    ntxt = ", ".join(f"{TITLES[m].split(' (')[0]} {nrep[m]}" for m in METHODS)
    note = (
        "Old SNMS experiment (real data SNMSv5, normal score; pre-NewWorld code). Colour = mean over random 70/30 "
        f"train/test splits, evaluated at the 30 % test points; each cell is one grid node (no interpolation). Grids: "
        f"{gtxt}. Splits: {ntxt}. Kriging = ordinary (kb2d ktype 1). x = range parameter: kriging/SGS spherical range; "
        "RBF ε = 2.146/x (Gaussian); GP length scale = x/2.146. Optima from grid nodes (MSE min, UMG max). "
        "SGS: cumulative pkl 19.pkl split into 19 seed blocks."
    )
    P, CB, CG, RG, COLG = 30.0, 1.5, 1.2, 12.5, 14.0
    fig, rect = sb._new_figure(plot_height_mm=2 * (CB + CG + P) + RG, status_text=note, top_mm=19.0, left_mm=14.0)
    panel_w = ((rect[2] - rect[0]) * sb.FULL_WIDTH_MM - 3 * COLG) / 4.0
    gs = fig.add_gridspec(7, 4, left=rect[0], bottom=rect[1], right=rect[2], top=rect[3],
                          height_ratios=[CB, CG, P, RG, CB, CG, P], hspace=0.0, wspace=COLG / panel_w)
    halo = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    letters = iter("ABCDEFGH")

    def orow(m, met):
        return opt[(opt.method == m) & (opt.metric == met)].iloc[0]

    for row, met in enumerate(("mse", "umg")):
        other = "umg" if met == "mse" else "mse"
        for col, m in enumerate(METHODS):
            cax, ax = fig.add_subplot(gs[4 * row, col]), fig.add_subplot(gs[4 * row + 2, col])
            xs, ys, z = grid(summ, m, met)
            zmin, zmax = float(z.min()), float(z.max())
            use_log = met == "mse" and zmax / zmin >= sb.RATIO_LOG_THRESHOLD
            pc = ax.pcolormesh(sb._f7_edges(xs, True), sb._f7_edges(ys, YLOG[m]), z, cmap=CMAP[met],
                               norm=LogNorm(zmin, zmax) if use_log else Normalize(zmin, zmax),
                               shading="flat", edgecolors="face", linewidth=0.0)
            ax.set_xscale("log")
            if YLOG[m]:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.set_xticks(XTICKS)
            ax.set_xticklabels([str(t) for t in XTICKS])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_yticks(YTICKS[m])
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylabel(YLABEL[m])
            if row == 1:
                ax.set_xlabel("Range parameter (m)")
            own, oth = orow(m, met), orow(m, other)
            ax.plot(oth["range"], oth["param2"], ls="none", marker="*", ms=8, mfc="none", mec="black", mew=0.8,
                    path_effects=halo, zorder=6, clip_on=False)
            ax.plot(own["range"], own["param2"], ls="none", marker="*", ms=8, mfc="black", mec="white", mew=0.4,
                    zorder=7, clip_on=False)
            cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            cb.set_label("MSE (normal score)" if met == "mse" else "UMG", labelpad=1.5)
            cb.outline.set_linewidth(0.5)
            cax.tick_params(length=1.5, pad=1.0)
            if use_log:
                cax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if row == 0:
                cax.annotate(TITLES[m], xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 21),
                             textcoords="offset points", ha="center", va="bottom", fontweight="bold")
            sb._panel_label(ax, next(letters), dx_pt=-30.0, dy_pt=12.0)
    fig.legend(handles=[
        Line2D([0], [0], ls="none", marker="*", ms=7, mfc="black", mec="black", label="Optimum of this surface (grid node)"),
        Line2D([0], [0], ls="none", marker="*", ms=7, mfc="none", mec="black", mew=0.8,
               label="Optimum of the other surface")],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.3)
    out = REPO / "results" / "figures" / "paper_storyboard"
    for ext, kw in (("pdf", {}), ("png", {"dpi": sb.PNG_DPI})):
        fig.savefig(out / f"{STEM}.{ext}", **kw)
    return [out / f"{STEM}.pdf", out / f"{STEM}.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", default=OLD_ROOT_DEFAULT)
    ap.add_argument("--re-extract", action="store_true")
    a = ap.parse_args()
    if a.re_extract or not LONG_CSV.exists():
        extract(a.old_root).to_csv(LONG_CSV, index=False)
    long = pd.read_csv(LONG_CSV)
    summ, opt = summarize(long)
    summ.to_csv(SUMMARY_CSV, index=False)
    opt.to_csv(OPTIMA_CSV, index=False)
    nrep = long.groupby("method")["replicate"].nunique().to_dict()
    print(opt.to_string(index=False))
    print(figure(summ, opt, nrep))


if __name__ == "__main__":
    main()
