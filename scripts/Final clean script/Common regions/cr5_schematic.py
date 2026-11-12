"""
Left panel of Figures 3C and 5B: how the common regions are obtained.

1. one grey strip per event = ASD group-level event map, regions ordered by Dev-Atlas
   system, with a system bar under the last strip
2. bracket to one strip = equal-weight mean across events (same grey scale: 2nd-98th
   percentile of all strip values, values outside clipped)
3. ranked bars of the averaged map; top 12 coloured by system; dashed line at the 12th value
4. arrow towards the surface maps

Output: <CR_OUT_DIR>/figures/schematic_<film>.jpg (600 dpi)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrow  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from cr_common import DATA_IDS, FIG, INK, INK2, K, OUT, SYS_COLOR, SYSTEM_ORDER, films_with, regions, save_jpg  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8.5, "font.weight": "bold", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
BG, BAR_GREY = "#d7e4ee", "#bdbbb5"
FS, FS2, FSUP = 8.5, 8.0, 6.2
R = regions()
ORDER = sorted(DATA_IDS, key=lambda i: int(R.loc[i, "circle_order"]))
cs = pd.read_csv(OUT / "common_regions.csv") if (OUT / "common_regions.csv").exists() else None
FIG.mkdir(parents=True, exist_ok=True)


def rot_label(fig, x, ycen, text, size):
    t = fig.text(0, 0, text, rotation=90, fontsize=size, color=INK, ha="center", va="bottom")
    h = t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).height
    t.set_position((x, ycen - h / 2))


def unit(fig, x, y, ha, va):
    """'×10' with a hand-placed superscript '−3'."""
    t = fig.text(x, y, "×10", ha=ha, va=va, fontsize=FS2, color=INK)
    b = t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(b.x1, b.y1 + 0.004, "−3", ha="left", va="top", fontsize=FSUP, color=INK)


for film in films_with("event_maps_{film}.npz"):
    if cs is None or cs[cs.film == film].empty:
        continue
    z = np.load(OUT / f"event_maps_{film}.npz")
    em, grp = z["maps"].astype(float), z["group"].astype(str)
    M = np.stack([em[grp == "ASD", e, :].mean(axis=0) for e in range(em.shape[1])])
    avg = M.mean(axis=0)
    rank = sorted(DATA_IDS, key=lambda i: -avg[i])
    if set(rank[:K]) != set(cs[cs.film == film].brainnetome_id.astype(int)):
        raise ValueError(f"{film}: top {K} of the averaged map differ from common_regions.csv")
    n_ev = M.shape[0]
    S, A = M[:, ORDER] * 1e3, avg[ORDER] * 1e3
    vmin, vmax = (float(round(v, 2)) for v in np.percentile(np.concatenate([S.ravel(), A]), [2, 98]))
    cmap, norm = plt.get_cmap("Greys"), mcolors.Normalize(vmin, vmax)

    W = 3.2
    sh, sg, mh, hb = 0.085, 0.022, 0.12, 0.52
    y_sys_bot = 0.06 + n_ev * (sh + sg) - sg + 0.012 + 0.045
    yb = y_sys_bot + 0.20
    y_mean = yb + 0.30
    y_bar = y_mean + mh + 0.20
    H = y_bar + hb + 0.10
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    xL, xR = 0.46, 2.56

    def ax_in(x0, y_top, w, h):
        return fig.add_axes([x0 / W, 1 - (y_top + h) / H, w / W, h / H])

    y = 0.06
    for e in range(n_ev):
        ax = ax_in(xL, y, xR - xL, sh)
        ax.imshow(S[e][None, :], aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
            sp.set_color(INK2)
        fig.text((xL - 0.05) / W, 1 - (y + sh / 2) / H, f"E{e + 1}", ha="right", va="center", fontsize=FS, color=INK)
        y += sh + sg
    strips_top, strips_bot = 0.06, y - sg
    ax = ax_in(xL, y - sg + 0.012, xR - xL, 0.045)
    ax.imshow(np.array([[mcolors.to_rgb(SYS_COLOR[R.loc[i, "system"]]) for i in ORDER]]), aspect="auto",
              interpolation="nearest")
    ax.axis("off")
    for s_ in SYSTEM_ORDER:
        ix = [k for k, i in enumerate(ORDER) if R.loc[i, "system"] == s_]
        xc = xL + (np.mean(ix) + 0.5) / len(ORDER) * (xR - xL)
        fig.text(xc / W, 1 - (y_sys_bot + 0.015) / H, s_, ha="center", va="top", fontsize=FS2,
                 color=SYS_COLOR[s_], fontweight="bold")
    cax = ax_in(2.64, strips_top + 0.01, 0.055, strips_bot - strips_top - 0.02)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both", extendfrac=0.06)
    cb.set_ticks([vmin, vmax])
    cb.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    cb.ax.tick_params(labelsize=FS2, length=2, width=0.5, pad=1)
    cb.outline.set_linewidth(0.4)
    fig.canvas.draw()
    rot_label(fig, 3.10 / W, 1 - ((strips_top + strips_bot) / 2) / H, "|IG|", FS2)
    unit(fig, 2.62 / W, 1 - (strips_bot + 0.05) / H, "left", "top")
    xc, lw = (xL + xR) / 2, 0.8
    bx = fig.add_axes([0, 0, 1, 1], facecolor="none")
    bx.set_xlim(0, W)
    bx.set_ylim(H, 0)
    bx.axis("off")
    bx.plot([xL, xL, xR, xR], [yb - 0.05, yb, yb, yb - 0.05], color=INK, lw=lw, solid_capstyle="butt")
    bx.annotate("", xy=(xc, y_mean - 0.02), xytext=(xc, yb),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=INK, mutation_scale=7, shrinkA=0, shrinkB=0))
    bx.text(xc + 0.06, (yb + y_mean) / 2, "mean across events\n(equal weight)", ha="left", va="center", fontsize=FS,
            color=INK, linespacing=1.1)
    ax = ax_in(xL, y_mean, xR - xL, mh)
    ax.imshow(A[None, :], aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color(INK)
    fig.text((xL - 0.05) / W, 1 - (y_mean + mh / 2) / H, "mean", ha="right", va="center", fontsize=FS, color=INK)
    ax = ax_in(xL, y_bar, xR - xL, hb)
    ax.set_facecolor(BG)
    vals = np.array([avg[i] * 1e3 for i in rank])
    ax.bar(np.arange(len(rank)), vals, width=1.0, linewidth=0,
           color=[SYS_COLOR[R.loc[i, "system"]] if k < K else BAR_GREY for k, i in enumerate(rank)])
    thr = vals[K - 1]
    ax.axhline(thr, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(len(rank) - 1, thr + vals.max() * 0.03, "top 5%", ha="right", va="bottom", fontsize=FS, color=INK)
    ax.set_xlim(-0.5, len(rank) - 0.5)
    ax.set_ylim(0, vals.max() * 1.08)
    ax.set_xticks([])
    fig.canvas.draw()
    rot_label(fig, (xL - 0.36) / W, 1 - (y_bar + hb / 2) / H, "mean |IG|", FS)
    unit(fig, (xL - 0.02) / W, 1 - (y_bar - 0.03) / H, "center", "bottom")
    ax.tick_params(axis="y", labelsize=FS2, length=2, pad=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.spines[["top", "right"]].set_visible(False)
    bx.add_patch(FancyArrow(xR + 0.10, y_bar + hb * 0.25, 0.36, 0, width=0.035, head_width=0.13, head_length=0.12,
                            length_includes_head=True, color=INK, lw=0))
    save_jpg(fig, FIG / f"schematic_{film}.jpg", dpi=600, facecolor=BG)
    plt.close(fig)
    print(f"{film}: schematic_{film}.jpg (grey scale {vmin}-{vmax} x1e-3)")
