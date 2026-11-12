"""
Radar plots of the ASD group-level attribution (Figures 3C, 5B, S5, S10).

radar_events_<film>.jpg          one panel per event; spokes = the event's top 5% regions,
                                 radius = mean over ASD participants of the per-participant
                                 median |IG| in the event; common regions boxed (S5, S10)
radar_common_<film>.jpg          spokes = the 12 common regions, radius = across-event mean |IG|;
                                 per system, dashed arc and band = network average ± 1 SD
                                 across all regions of the system (3C, 5B)
radar_common_events_<film>.jpg   the same with each event's value as a thin grey line

Spokes are grouped by Dev-Atlas system; the empty sector at 12 o'clock holds the radial
tick labels. Panel spacing is set from the measured text extents; ROI labels shrink in
0.25 pt steps while any two overlap.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from cr_common import (FIG, FILMS, GRID, INK, INK2, OUT, SURFACE, SYS_COLOR, SYSTEM_LONG,  # noqa: E402
                       SYSTEM_ORDER, regions, save_jpg)

TICK_OUTLINE = [pe.withStroke(linewidth=3.0, foreground="white")]
COMMON_FILL, EVENT_GREY, REF_GREY = "#fff1b8", "#9a9893", "#7d7b76"
N_RINGS, SYSTEM_GAP, PAD, LABEL_MIN, DPI = 4, 2.0, 0.06, 4.0, 300
EVENTS_STYLE = dict(font="Verdana", panel=2.3, label=6.5, title=9.0, tick=6.5, legend=8.0, marker=20.0,
                    line=1.2, arc=4.5, gap_deg=10.0, nrows=2)
AVG_STYLE = dict(font="Arial", panel=4.6, label=17.0, title=16.0, tick=10.0, legend=10.5, marker=46.0,
                 line=2.0, arc=7.0, gap_deg=10.0, nrows=1)

R = regions()
FIG.mkdir(parents=True, exist_ok=True)


def thetas(systems, gap, gap_deg):
    """Angle per spoke, clockwise from 12 o'clock; `gap` spoke widths between systems and an
    empty sector of gap_deg centred on 12 o'clock."""
    pos, x, prev = [], 0.0, None
    for s in systems:
        if prev is not None and s != prev:
            x += gap
        pos.append(x)
        x += 1.0
        prev = s
    wrap = max(gap, (x - 1.0) * gap_deg / (360.0 - gap_deg)) if gap_deg > 0 else gap
    total = x - 1.0 + wrap
    return (np.array(pos) + wrap / 2) / total * 2 * np.pi, 2 * np.pi / total


def two_line(rid):
    """(line 1, line 2): short anatomical label and Brainnetome code; one line if no short label
    or if the code is itself the abbreviation."""
    r = R.loc[rid]
    if not r.short_label:
        return r.label, None
    return r.short_label, (None if r.short_label.replace(" ", "_") == r.label else r.label)


def draw(panels, st, L, hi, title_y=1.02, kind="events", ref=None):
    P = st["panel"]
    n = len(panels)
    ncols = min(n, int(np.ceil(n / st["nrows"])))
    nrows = int(np.ceil(n / ncols))
    H = nrows * P + (nrows - 1) * L["row_gap"] + L["top"] + L["bottom"]
    W = ncols * P + (ncols - 1) * L["col_gap"] + 2 * L["side"]
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_subplot(nrows, ncols, i + 1, projection="polar") for i in range(n)]
    fig.subplots_adjust(top=1 - L["top"] / H, bottom=L["bottom"] / H, left=L["side"] / W,
                        right=1 - L["side"] / W, hspace=L["row_gap"] / P, wspace=L["col_gap"] / P)
    lo, span = 0.0, hi
    for ax, pnl in zip(axes, panels):
        ids, systems, vals, intop = pnl["ids"], pnl["systems"], pnl["values"], pnl["intop"]
        th, step = thetas(systems, SYSTEM_GAP * 0.5, st["gap_deg"])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(lo, hi)
        ax.set_facecolor(SURFACE)
        ax.spines["polar"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        rings = np.linspace(lo, hi, N_RINGS + 1)[1:]
        for r in rings:
            ax.plot(np.linspace(0, 2 * np.pi, 361), np.full(361, r), color=GRID, lw=0.6, zorder=0)
        for r in rings[:-1]:
            ax.text(0.0, r, f"{r * 1e3:.2g}", ha="center", va="center", gid="tick", fontsize=st["tick"],
                    color=INK2, fontweight="bold", zorder=20, path_effects=TICK_OUTLINE, clip_on=False)
        for t in th:
            ax.plot([t, t], [lo, hi], color=GRID, lw=0.5, zorder=0)
        s_arr = np.array(systems)
        if ref is not None:
            for sy in SYSTEM_ORDER:
                ix = np.where(s_arr == sy)[0]
                if not len(ix):
                    continue
                tt = np.linspace(max(th[ix[0]] - 0.45 * step, th[0]), min(th[ix[-1]] + 0.45 * step, th[-1]), 60)
                m, sd = ref[sy]
                b0, b1 = max(0.0, m - sd), m + sd
                ax.fill_between(tt, np.full_like(tt, b0), np.full_like(tt, b1), color=SYS_COLOR[sy], alpha=0.18,
                                linewidth=0, zorder=3.2).set_gid("ref")
                for b in (b0, b1):
                    ax.plot(tt, np.full_like(tt, b), color=SYS_COLOR[sy], lw=0.7, ls=(0, (2.5, 2)), alpha=0.9,
                            zorder=3.4, gid="ref")
                ax.plot(tt, np.full_like(tt, m), color=SYS_COLOR[sy], lw=2.4, ls=(0, (5, 2.5)),
                        dash_capstyle="butt", zorder=3.6, gid="ref")
        for v in pnl.get("event_lines", []):
            ax.plot(th, v, color=EVENT_GREY, lw=0.7, zorder=4)
        ax.plot(th, vals, color=INK, lw=st["line"], zorder=5)
        for t, v, sy, top in zip(th, vals, systems, intop):
            ax.scatter([t], [v], s=st["marker"], zorder=6 + (top and kind != "events"), linewidths=1.3,
                       facecolors=SYS_COLOR[sy] if top else SURFACE, edgecolors=SYS_COLOR[sy])
        for sy in SYSTEM_ORDER:
            idx = np.where(s_arr == sy)[0]
            if not len(idx):
                continue
            tt = np.linspace(th[idx[0]] - 0.45 * step, th[idx[-1]] + 0.45 * step, max(80, len(idx) * 20))
            ax.plot(tt, np.full_like(tt, hi + span * 0.04), color=SYS_COLOR[sy], lw=st["arc"],
                    solid_capstyle="butt", clip_on=False, zorder=3)
        for t, rid in zip(th, ids):
            if kind == "events":
                deg = np.degrees(t)
                box = (dict(boxstyle="round,pad=0.18", facecolor=COMMON_FILL, edgecolor=INK, linewidth=0.7)
                       if rid in pnl["common"] else None)
                ax.text(t, hi + span * 0.085, R.loc[rid, "anatomical_label"], rotation=90 - deg if deg <= 180 else 270 - deg,
                        rotation_mode="anchor", gid=f"roi{rid}", ha="left" if deg <= 180 else "right", va="center",
                        fontsize=st["label"], color=INK, fontweight="bold", clip_on=False, bbox=box)
                continue
            l1, l2 = two_line(rid)
            sn, cs_ = np.sin(t), np.cos(t)
            ha = "left" if sn > 0.2 else ("right" if sn < -0.2 else "center")
            f1, f2 = st["label"], st["label"] - 2.0
            g1, g2 = f1 * 1.2, f2 * 1.2
            if cs_ > 0.45:
                y1, v1, y2, v2 = (g2, "bottom", 0, "bottom")
            elif cs_ < -0.45:
                y1, v1, y2, v2 = (0, "top", -g1, "top")
            else:
                y1, v1, y2, v2 = ((1, "bottom", -1, "top") if l2 else (0, "center", 0, "center"))
            if not l2 and cs_ > 0.45:
                y1 = 0
            pos = (t, hi + span * 0.10)
            ax.text(*pos, l1, ha=ha, va=v1, fontsize=f1, color=INK, fontweight="bold", clip_on=False, gid=f"roi{rid}",
                    transform=offset_copy(ax.transData, fig=fig, x=0, y=y1, units="points"))
            if l2:
                ax.text(*pos, l2, ha=ha, va=v2, fontsize=f2, color="#3a3936", fontweight="normal", clip_on=False,
                        gid=f"roi{rid}", transform=offset_copy(ax.transData, fig=fig, x=0, y=y2, units="points"))
        ax.set_title(pnl["title"], fontsize=st["title"], color=INK, fontweight="bold", y=title_y, va="bottom")
    return fig, axes


def overhangs(fig, axes):
    fig.canvas.draw()
    rr, dpi = fig.canvas.get_renderer(), fig.dpi
    out = []
    for ax in axes:
        ab = ax.get_window_extent(rr)
        bbs = [a.get_window_extent(rr) for a in ax.texts if a.get_text()]
        out.append(dict(left=max(0, ab.x0 - min(b.x0 for b in bbs)) / dpi,
                        right=max(0, max(b.x1 for b in bbs) - ab.x1) / dpi,
                        bottom=max(0, ab.y0 - min(b.y0 for b in bbs)) / dpi,
                        top=max(0, max(b.y1 for b in bbs) - ab.y1) / dpi))
    return out


def rot_rect(t, rr):
    """Corners (display px) of a text's rotated box, including its bbox padding."""
    w, h, _ = rr.get_text_width_height_descent(t.get_text(), t.get_fontproperties(), ismath=False)
    pad = 0.18 * t.get_fontsize() * t.figure.dpi / 72 + 1.0 if t.get_bbox_patch() is not None else 1.0
    x, y = t.get_transform().transform(t.get_position())
    a = np.radians(t.get_rotation())
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-u[1], u[0]])
    s0, s1 = (-pad, w + pad) if t.get_horizontalalignment() == "left" else (-w - pad, pad)
    hh = h / 2 + pad
    return np.array([[x, y] + u * s0 - v * hh, [x, y] + u * s1 - v * hh,
                     [x, y] + u * s1 + v * hh, [x, y] + u * s0 + v * hh])


def overlap(Pq, Qq):
    for Rq in (Pq, Qq):
        for i in range(4):
            e = Rq[(i + 1) % 4] - Rq[i]
            nrm = np.array([-e[1], e[0]])
            a, b = Pq @ nrm, Qq @ nrm
            if a.max() <= b.min() or b.max() <= a.min():
                return False
    return True


def n_overlaps(fig, axes):
    """Overlapping pairs among ROI labels and panel titles."""
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    rects = [rot_rect(t, rr) for ax in axes for t in ax.texts if (t.get_gid() or "").startswith("roi")]
    for ax in axes:
        b = ax.title.get_window_extent(rr)
        rects.append(np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]]))
    return sum(overlap(rects[i], rects[j]) for i in range(len(rects)) for j in range(i + 1, len(rects)))


def drop_covered_tick(fig, axes, hi):
    """Remove the one radial tick label that a data point or data line sits on, if exactly one does."""
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    pts = []
    for ax in axes:
        for col in ax.collections:
            if col.get_gid() == "ref":
                continue
            for (t_, r_), s_ in zip(col.get_offsets(), np.resize(col.get_sizes(), len(col.get_offsets()))):
                if r_ <= hi:
                    x, y = ax.transData.transform((t_, r_))
                    pts.append((x, y, np.sqrt(s_) / 2 * fig.dpi / 72))
        for ln in ax.lines:
            if ln.get_color() not in (INK, EVENT_GREY) or ln.get_gid() == "ref" or len(ln.get_xdata()) < 2:
                continue
            Pd = ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float))
            for a_, b_ in zip(Pd[:-1], Pd[1:]):
                if np.all(np.isfinite(a_)) and np.all(np.isfinite(b_)) and np.hypot(*(b_ - a_)) < 4 * fig.dpi:
                    for f_ in np.linspace(0, 1, 30):
                        x, y = a_ + f_ * (b_ - a_)
                        pts.append((x, y, ln.get_linewidth() / 2 * fig.dpi / 72))
    ticks = [t for ax in axes for t in ax.texts if t.get_gid() == "tick"]
    covered = [t for t in ticks if any(b.x0 - rad < x < b.x1 + rad and b.y0 - rad < y < b.y1 + rad
                                       for b in [t.get_window_extent(rr)] for (x, y, rad) in pts)]
    if len(covered) == 1 and len(ticks) >= 3:
        covered[0].remove()


def figure(panels, st, hi, stem, handles, labels=None, kind="events", ref=None, leg_cols=4):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [st["font"], "DejaVu Sans"],
                         "font.size": 7, "axes.edgecolor": GRID})
    st = dict(st)
    while True:
        P = st["panel"]
        title_h = st["title"] / 72 * 1.25
        fig, axes = draw(panels, st, dict(col_gap=3.0, row_gap=3.0, top=3.0, bottom=3.0, side=3.0), hi, kind=kind, ref=ref)
        oh = overhangs(fig, axes)
        plt.close(fig)
        n = len(axes)
        ncols = min(n, int(np.ceil(n / st["nrows"])))
        nrows = int(np.ceil(n / ncols))
        lab_top = max(o["top"] for o in oh)
        for o in oh:
            o["top"] = lab_top + PAD + title_h
        grid = [[oh[r * ncols + c] if r * ncols + c < n else None for c in range(ncols)] for r in range(nrows)]
        col_gap = max([grid[r][c]["right"] + grid[r][c + 1]["left"] for r in range(nrows) for c in range(ncols - 1)
                       if grid[r][c] and grid[r][c + 1]] or [0]) + PAD
        row_gap = max([grid[r][c]["bottom"] + grid[r + 1][c]["top"] for r in range(nrows - 1) for c in range(ncols)
                       if grid[r][c] and grid[r + 1][c]] or [0]) + PAD
        top_need = max(o["top"] for o in grid[0] if o)
        bot_need = max(o["bottom"] for o in grid[-1] if o)
        side = max(o[k] for o in oh for k in ("left", "right")) + PAD
        legend_h = int(np.ceil(len(handles) / leg_cols)) * st["legend"] / 72 * 1.45
        L = dict(col_gap=col_gap, row_gap=row_gap, top=top_need + PAD, bottom=bot_need + legend_h + 2 * PAD, side=side)
        fig, axes = draw(panels, st, L, hi, title_y=1 + (lab_top + PAD) / P, kind=kind, ref=ref)
        H = fig.get_figheight()
        kw = dict(handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, handlelength=2.2) if labels else \
            dict(handlelength=1.5)
        fig.legend(handles=handles, **({"labels": labels} if labels else {}), loc="upper center", ncol=leg_cols,
                   frameon=False, prop={"size": st["legend"], "weight": "bold"},
                   bbox_to_anchor=(0.5, (L["bottom"] - bot_need - PAD) / H), columnspacing=1.0, borderaxespad=0,
                   handletextpad=0.4, labelspacing=0.3, **kw)
        if n_overlaps(fig, axes) and st["label"] - 0.25 >= LABEL_MIN:
            plt.close(fig)
            st["label"] -= 0.25
            continue
        break
    if kind != "events":
        drop_covered_tick(fig, axes, hi)
    save_jpg(fig, FIG / f"{stem}.jpg", dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  {stem}.jpg (labels {st['label']:.2f} pt)")


def sys_patches():
    return [Patch(facecolor=SYS_COLOR[s], edgecolor="none", label=f"{s}  {SYSTEM_LONG[s]}") for s in SYSTEM_ORDER]


ev_all = pd.read_csv(OUT / "event_values.csv") if (OUT / "event_values.csv").exists() else None
cs_all = pd.read_csv(OUT / "common_regions.csv") if (OUT / "common_regions.csv").exists() else None
net_all = pd.read_csv(OUT / "network_average.csv") if (OUT / "network_average.csv").exists() else None
if ev_all is None or cs_all is None:
    print("cr2 outputs not found, radars skipped")
    raise SystemExit(0)

order = lambda ids: sorted(ids, key=lambda r: int(R.loc[r, "circle_order"]))  # noqa: E731
for film in FILMS:
    d = ev_all[(ev_all.film == film) & (ev_all.group == "ASD")]
    if d.empty:
        continue
    print(film)
    common = set(cs_all[cs_all.film == film].brainnetome_id.astype(int))
    events = sorted(d.event.unique(), key=lambda e: int(e[1:]))
    val = d.pivot(index="brainnetome_id", columns="event", values="value")

    panels = []
    for ev in events:
        ids = order(d[(d.event == ev) & d.in_top5_ASD].brainnetome_id.tolist())
        n_c = sum(r in common for r in ids)
        panels.append(dict(ids=ids, systems=R.loc[ids, "system"].values, values=val.loc[ids, ev].values,
                           intop=np.ones(len(ids), bool), common=common,
                           title=f"{ev}  ({n_c} common + {len(ids) - n_c} event-specific)"))
    hi = max(p["values"].max() for p in panels)
    hs = sys_patches() + [Line2D([], [], color=INK, lw=1.2, label="ASD"),
                          Patch(facecolor=COMMON_FILL, edgecolor=INK, linewidth=0.7,
                                label="cross-event common region (unboxed = event-specific)")]
    ncols = min(len(panels), int(np.ceil(len(panels) / EVENTS_STYLE["nrows"])))
    figure(panels, EVENTS_STYLE, hi, f"radar_events_{film}", hs, leg_cols=5 if ncols >= 3 else 4)

    ids = order(common)
    avg = val.loc[ids, events].mean(axis=1).values
    net = net_all[net_all.film == film].set_index("system")
    ref = {s: (float(net.loc[s, "mean"]), float(net.loc[s, "sd"])) for s in SYSTEM_ORDER}
    drawn = set(R.loc[ids, "system"])
    for kind in ("common", "common_events"):
        lines = [val.loc[ids, ev].values for ev in events] if kind == "common_events" else []
        top = [avg.max()] + [v.max() for v in lines] + [ref[s][0] + ref[s][1] for s in drawn]
        pnl = [dict(ids=ids, systems=R.loc[ids, "system"].values, values=avg, intop=np.ones(len(ids), bool),
                    common=common, event_lines=lines, title=FILMS[film]["title"])]
        hs = sys_patches() + [Line2D([], [], color=INK, lw=1.2)]
        labels = [h.get_label() for h in hs[:-1]] + ["twelve cross-event common regions"]
        if lines:
            hs.append(Line2D([], [], color=EVENT_GREY, lw=0.9))
            labels.append(f"single event (E1–E{len(events)})")
        hs.append((Patch(facecolor=REF_GREY, alpha=0.18, edgecolor="none"),
                   Line2D([], [], color=REF_GREY, lw=2.4, ls=(0, (5, 2.5)))))
        labels.append("network average ± 1 SD across that system's regions")
        figure(pnl, AVG_STYLE, max(top) * 1.08, f"radar_{kind}_{film}", hs, labels=labels, kind="avg", ref=ref,
               leg_cols=3)
