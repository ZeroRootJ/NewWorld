"""Wireframe schematic of the one-factor-at-a-time experimental design.

Purely illustrative: sample positions are seeded random draws, not the actual
experiment samples. Levels for range / nugget / sample density are the ones
implemented in src/experiments; the heterogeneity (anisotropy ellipse) and
concentration axes are schematic (no implementation yet).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

OUT = Path(__file__).resolve().parent / "design_wireframe.png"

INK = "#2b3040"
MUTED = "#8a90a0"
ACCENT = "#2a6fdb"
BASE_BG = "#eaf1fd"

N_BASE = 125
RANGE_BASE = 300.0
NUG_BASE = 0.1
L = 1000.0


def uniform_pts(n, seed=7):
    return np.random.default_rng(seed).uniform(0, L, (n, 2))


def frame(ax, base=False):
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(ACCENT if base else INK)
        s.set_linewidth(2.0 if base else 1.0)
    if base:
        ax.set_facecolor(BASE_BG)


def dots(ax, pts, color=INK):
    ax.scatter(pts[:, 0], pts[:, 1], s=5, facecolors="none", edgecolors=color, linewidths=0.6)


def map_panel(ax, pts, base=False):
    frame(ax, base)
    dots(ax, pts)


def range_panel(ax, r, base=False):
    map_panel(ax, uniform_pts(N_BASE), base)
    for cx, cy in [(300, 300), (700, 650)]:
        ax.add_patch(Circle((cx, cy), r, fill=False, ec=ACCENT, lw=1.2, ls="--"))
        ax.plot([cx, cx + r], [cy, cy], color=ACCENT, lw=1.2)
        ax.plot(cx, cy, "o", color=ACCENT, ms=2.5)


def vario_panel(ax, nug, base=False):
    """Spherical variogram, sill = 1, with the nugget intercept."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(ACCENT if base else INK)
        s.set_linewidth(2.0 if base else 1.0)
    if base:
        ax.set_facecolor(BASE_BG)
    h = np.linspace(0, 1, 200)
    a = 0.6
    sph = np.where(h < a, 1.5 * h / a - 0.5 * (h / a) ** 3, 1.0)
    g = nug + (1 - nug) * sph
    g[0] = 0
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.plot(h[1:], g[1:], color=INK, lw=1.4)
    ax.plot([0, 0], [0, nug], color=ACCENT, lw=3.0, solid_capstyle="butt")
    ax.plot(0, nug, "o", color=ACCENT, ms=4)


def hetero_panel(ax, ratio, base=False, major=300.0, azimuth=40.0):
    """Spatial-continuity heterogeneity: anisotropic range ellipse (major : minor = ratio : 1)."""
    map_panel(ax, uniform_pts(N_BASE, seed=11), base)
    minor = major / ratio
    th = np.deg2rad(azimuth)
    for cx, cy in [(300, 300), (700, 650)]:
        ax.add_patch(Ellipse((cx, cy), 2 * major, 2 * minor, angle=azimuth,
                             fill=False, ec=ACCENT, lw=1.2, ls="--"))
        ax.plot([cx, cx + major * np.cos(th)], [cy, cy + major * np.sin(th)], color=ACCENT, lw=1.2)
        ax.plot(cx, cy, "o", color=ACCENT, ms=2.5)


def conc_panel(ax, strength, base=False):
    """Same n, a fraction of the samples pulled toward one cluster centre."""
    rng = np.random.default_rng(23)
    n_c = int(round(N_BASE * strength))
    clus = rng.normal([330, 330], 90, (n_c, 2))
    uni = uniform_pts(N_BASE - n_c, seed=5)
    pts = np.clip(np.vstack([clus, uni]), 0, L)
    map_panel(ax, pts, base)


def density_panel(ax, pct, base=False):
    n = int(round(2500 * pct / 100))
    map_panel(ax, uniform_pts(n, seed=3), base)
    if n > 200:
        for c in ax.collections:
            c.set_sizes([2.5])


def main():
    rows = [
        ("① Range axis", "range 100 – 800 m", [
            (f"{int(r)} m", (lambda r: lambda ax, b: range_panel(ax, r, b))(r), r == RANGE_BASE)
            for r in [100, 200, 300, 400, 500, 600, 700, 800]]),
        ("② Nugget axis", "nugget 0 – 0.5 of sill", [
            (f"{n:.1f}", (lambda n: lambda ax, b: vario_panel(ax, n, b))(n), n == 0.1)
            for n in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]),
        ("③ Density axis", "sample count 1 – 20 %", [
            (f"{p} %\n(n={int(2500 * p / 100)})", (lambda p: lambda ax, b: density_panel(ax, p, b))(p), p == 5)
            for p in [20, 10, 5, 2, 1]]),
        ("④ Heterogeneity axis", "spatial continuity varies with direction\n(anisotropy ellipse, major : minor)", [
            ("1 : 1\n(isotropic)", lambda ax, b: hetero_panel(ax, 1, b), True),
            ("2 : 1", lambda ax, b: hetero_panel(ax, 2, b), False),
            ("3 : 1", lambda ax, b: hetero_panel(ax, 3, b), False)]),
        ("⑤ Concentration axis", "same n = 125, sampling locations\nclustered vs. random", [
            ("uniform\n(random sampled)", lambda ax, b: conc_panel(ax, 0.0, b), True),
            ("cluster\n(40 % clustered)", lambda ax, b: conc_panel(ax, 0.4, b), False),
            ("cluster\n(80 % clustered)", lambda ax, b: conc_panel(ax, 0.8, b), False)]),
    ]

    fig_w, fig_h = 13.333, 7.5  # 16:9 slide
    panel = 0.84       # panel edge (in)
    pitch_x = 1.13
    pitch_y = 1.25
    label_w = 3.0
    top = 0.95
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200, facecolor="white")

    fig.text(0.02, 1 - 0.30 / fig_h, "Experimental design: one base case, one factor varied at a time",
             fontsize=18, fontweight="bold", color=INK, va="center")
    fig.text(0.02, 1 - 0.62 / fig_h,
             "4 methods (simple kriging, SGS, RBF + bootstrap, GPR) on identical samples   |   "
             "blue = factor being varied, shaded = base-case level",
             fontsize=10.5, color=MUTED, va="center")

    # base-case block (right side, next to the axis rows)
    bx, by, bs = 10.6, 2.55, 1.9
    fig.text((bx + bs / 2) / fig_w, 1 - 2.25 / fig_h, "Base case", fontsize=14, fontweight="bold",
             color=INK, ha="center", va="top")
    bax = fig.add_axes([bx / fig_w, 1 - (by + bs) / fig_h, bs / fig_w, bs / fig_h])
    map_panel(bax, uniform_pts(N_BASE), True)
    fig.text((bx + bs / 2) / fig_w, 1 - (by + bs + 0.08) / fig_h,
             "isotropic, range 300 m, nugget 0.1\nn = 125 (5 %), uniform (random sampled)",
             fontsize=9.5, color=MUTED, ha="center", va="top", linespacing=1.35)

    for i, (title, sub, panels) in enumerate(rows):
        y_top = top + i * pitch_y
        fig.text(0.02, 1 - (y_top + 0.05) / fig_h, title, fontsize=14, fontweight="bold", color=INK, va="top")
        fig.text(0.02, 1 - (y_top + 0.36) / fig_h, sub, fontsize=10, color=MUTED, va="top", linespacing=1.35)
        for j, p in enumerate(panels):
            label, fn = p[0], p[1]
            base = p[2] if len(p) > 2 else (i == 0)
            x0 = label_w + j * pitch_x
            ax = fig.add_axes([x0 / fig_w, 1 - (y_top + panel) / fig_h, panel / fig_w, panel / fig_h])
            fn(ax, base)
            if label != "base":
                fig.text((x0 + panel / 2) / fig_w, 1 - (y_top + panel + 0.03) / fig_h, label,
                         fontsize=8.5, color=INK, ha="center", va="top", linespacing=1.15)
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
