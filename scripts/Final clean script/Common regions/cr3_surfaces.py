"""
Cortical surface maps of the common and event-specific regions (fsaverage5, inflated).

surface_network_common_<film>.jpg   common regions coloured by Dev-Atlas system; lightness
                                    within the hue = across-event mean |IG| (Figures 3C, 5B);
                                    subcortical common regions listed below the surfaces
surface_common_<film>.jpg           common regions only, red
surface_common_vs_event_<film>.jpg  common regions (red) and each event's event-specific
                                    regions (blue) (Figures 3D, 5C); colour = per-region
                                    weight: per fold, mean |IG| over ASD participants and
                                    TRs; median over folds; divided by its maximum

Region with Brainnetome ID b is painted with data column b. Needs the Brainnetome
246 volume (BN_ATLAS, default assets/atlas/BN_Atlas_246_1mm.nii).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from cr_common import (ATLAS, FIG, FILMS, INK, OUT, SYS_COLOR, SYSTEM_LONG, SYSTEM_ORDER,  # noqa: E402
                       TEST_MODE, film_path, save_jpg)

if not ATLAS.exists():
    msg = f"Brainnetome volume not found at {ATLAS}; set BN_ATLAS"
    if TEST_MODE:
        print(msg + ", surface maps skipped")
        raise SystemExit(0)
    raise FileNotFoundError(msg)
if not (OUT / "common_regions.csv").exists():
    print("common_regions.csv not found, surface maps skipped")
    raise SystemExit(0)

import nibabel as nib  # noqa: E402
from nilearn import datasets, plotting, surface  # noqa: E402

S_MIN, RADIUS, DARKNESS, VMAX, DPI = 0.42, 2.0, 0.35, 0.6, 300
TILE_W, TILE_H, ROW_H, VIEW_ZOOM = 3.0, 3.0, 2.0, 1.45
VIEWS = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]
FIG.mkdir(parents=True, exist_ok=True)
cs = pd.read_csv(OUT / "common_regions.csv")
spec = pd.read_csv(OUT / "event_specific_regions.csv") if (OUT / "event_specific_regions.csv").exists() else None
atlas_img = nib.load(str(ATLAS))
atlas = np.asarray(atlas_img.dataobj).astype(int)
fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")


def tint(hex_, s):
    rgb = np.array(mcolors.to_rgb(hex_))
    return tuple(1.0 - s * (1.0 - rgb))


def network_map(film, c):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Verdana", "DejaVu Sans"],
                         "font.weight": "bold"})
    ig = c.mean_ig_x1000.to_numpy(float)
    lo, hi = float(ig.min()), float(ig.max())
    strength = S_MIN + (1 - S_MIN) * (ig - lo) / ((hi - lo) or 1.0)
    P = pd.DataFrame(dict(k=np.arange(1, len(c) + 1), id=c.brainnetome_id.astype(int), label=c.anatomical_label,
                          system=c.system, color=[tint(SYS_COLOR[s], v) for s, v in zip(c.system, strength)]))
    P["cortical"] = P.id <= 210
    vol = np.zeros(atlas.shape, dtype=np.float32)
    for p in P[P.cortical].itertuples():
        vol[atlas == p.id] = p.k
    img = nib.Nifti1Image(vol, atlas_img.affine)
    tex = {h: surface.vol_to_surf(img, fs[f"pial_{h}"], radius=RADIUS, interpolation="nearest_most_frequent")
           for h in ("left", "right")}
    K = len(P)
    cmap = mcolors.ListedColormap(list(P.color))
    W, Hv, Wv = 6.4, 1.75, 3.2
    H = 0.7 + 2 * Hv + 0.3 + 0.95
    fig = plt.figure(figsize=(W, H))
    fig.text(0.5, 1 - 0.1 / H, FILMS[film]["title"], ha="center", va="top", fontsize=14, color=INK)
    fig.text(0.5, 1 - 0.42 / H, f"Cross-event common regions ({K})", ha="center", va="top", fontsize=10, color=INK)
    for i, (h, v) in enumerate(VIEWS):
        ax = fig.add_axes([(i % 2) * Wv / W, 1 - (0.7 + (i // 2 + 1) * Hv) / H, Wv / W, Hv / H], projection="3d")
        plotting.plot_surf_roi(fs[f"infl_{h}"], roi_map=tex[h], hemi=h, view=v, bg_map=fs[f"sulc_{h}"],
                               bg_on_data=False, darkness=DARKNESS, cmap=cmap, vmin=0.5, vmax=K + 0.5,
                               threshold=0.5, colorbar=False, axes=ax, figure=fig, avg_method="median")
        ax.set_box_aspect(None, zoom=1.45)
        ax.text2D(0.04, 0.93, f"{'L' if h == 'left' else 'R'} {v}", transform=ax.transAxes, fontsize=7.5, color=INK)
    y = 1 - (0.7 + 2 * Hv + 0.12) / H
    fig.text(0.25 / W, y, "Subcortical:", ha="left", va="center", fontsize=8.5, color=INK)
    x = 1.25
    for p in P[~P.cortical].itertuples():
        fig.patches.append(Rectangle((x / W, y - 0.08 / H), 0.2 / W, 0.16 / H, transform=fig.transFigure,
                                     facecolor=p.color, edgecolor=INK, linewidth=0.6))
        fig.text((x + 0.27) / W, y, f"{p.label} [{p.system}]", ha="left", va="center", fontsize=8.5, color=INK)
        x += 2.55
    hs = [Patch(facecolor=SYS_COLOR[s], edgecolor="none", label=f"{s}  {SYSTEM_LONG[s]}") for s in SYSTEM_ORDER]
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, (y * H - 0.18) / H),
               prop={"size": 8.5, "weight": "bold"}, handlelength=1.4, columnspacing=1.2, handletextpad=0.4,
               labelspacing=0.35)
    yk = 0.1 / H
    kax = fig.add_axes([0.3 / W, yk, 1.9 / W, 0.13 / H])
    kax.imshow(np.array([tint("#52514e", s) for s in np.linspace(S_MIN, 1, 256)])[None], aspect="auto",
               extent=[lo, hi, 0, 1])
    kax.set_yticks([])
    kax.set_xticks([lo, hi])
    kax.set_xticklabels([f"{lo:.2f}", f"{hi:.2f}"], fontsize=7)
    for sp in kax.spines.values():
        sp.set_linewidth(0.5)
    fig.text((0.3 + 1.9 + 0.12) / W, yk + 0.065 / H, "lightness within hue = mean IG across events (shown ×1000)",
             ha="left", va="center", fontsize=7.5, color=INK)
    save_jpg(fig, FIG / f"surface_network_common_{film}.jpg", dpi=DPI, facecolor="white",
             bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    P.assign(color=[mcolors.to_hex(x) for x in P.color]).to_csv(OUT / f"surface_network_parcels_{film}.csv", index=False)


def roi_weights(fold_root):
    out = []
    for f in range(5):
        ig = np.load(fold_root / f"fold{f}" / "ig_roi_milout.npy")
        lab = np.load(fold_root / f"fold{f}" / "labels.npy")
        out.append(np.mean(np.abs(ig[lab == 0]), axis=(0, 1)))
    w = np.median(np.stack(out), axis=0)
    return w / w.max()


def weighted_img(common_ids, specific_ids, w):
    out = np.zeros_like(atlas, dtype=np.float32)
    for ids, sign in ((common_ids, 1.0), (specific_ids, -1.0)):
        for b in ids:
            out[atlas == b] = sign * w[b]                         # region b <- column b
    return nib.Nifti1Image(out, atlas_img.affine)


def tile(fig, gs, img, title, compact=False):
    inner = gs.subgridspec(2, 2, hspace=-0.12 if compact else -0.05, wspace=-0.08 if compact else -0.05)
    for i, (hemi, view) in enumerate(VIEWS):
        ax = fig.add_subplot(inner[i // 2, i % 2], projection="3d")
        tex = surface.vol_to_surf(img, fs[f"pial_{hemi}"], interpolation="linear", radius=RADIUS)
        plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], tex, hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"],
                                    bg_on_data=True, darkness=DARKNESS, cmap="RdBu_r", colorbar=False,
                                    threshold=1e-8, vmax=VMAX, symmetric_cbar=True, axes=ax)
        if compact:
            ax.set_box_aspect(None, zoom=VIEW_ZOOM)
        else:
            ax.text2D(0.04, 0.88, "L" if hemi == "left" else "R", transform=ax.transAxes, fontsize=8, fontweight="bold")
        if i == 0:
            ax.set_title(title, fontsize=10, fontweight="bold", loc="left", x=0.0, pad=4)


def event_maps(film, c):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Verdana", "DejaVu Sans"],
                         "font.weight": "normal"})
    wroot = film_path(film, "weight_folds")
    if wroot is None:
        print(f"{film}: weight folds not found, red/blue maps skipped")
        return
    w = roi_weights(wroot)
    common_ids = c.brainnetome_id.astype(int).tolist()
    s = spec[spec.film == film].sort_values(["event", "rank_in_event"]) if spec is not None else pd.DataFrame()
    s = s[s.brainnetome_id.notna()]
    ev_spec = {ev: g.brainnetome_id.astype(int).tolist() for ev, g in s.groupby("event")}
    title = FILMS[film]["title"]
    fig = plt.figure(figsize=(TILE_W, TILE_H + 0.42))
    gs = fig.add_gridspec(1, 1, bottom=0.07, top=0.875)
    fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=12, fontweight="bold")
    tile(fig, gs[0], weighted_img(common_ids, [], w), f"Common regions ({len(common_ids)})")
    save_jpg(fig, FIG / f"surface_common_{film}.jpg", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    events = sorted(ev_spec, key=lambda e: int(e[1:]))
    n = 1 + len(events)
    fig = plt.figure(figsize=(TILE_W * n, ROW_H + 0.6))
    gs = fig.add_gridspec(1, n, wspace=0.02, bottom=0.04, top=0.80, left=0.01, right=0.91)
    fig.text(0.46, 0.985, title, ha="center", va="top", fontsize=13, fontweight="bold")
    tile(fig, gs[0], weighted_img(common_ids, [], w), f"Common ({len(common_ids)})", compact=True)
    for j, ev in enumerate(events, start=1):
        tile(fig, gs[j], weighted_img(common_ids, ev_spec[ev], w), f"{ev}  (+{len(ev_spec[ev])} specific)",
             compact=True)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-VMAX, vmax=VMAX))
    cb = fig.colorbar(sm, cax=fig.add_axes([0.925, 0.3, 0.006, 0.45]))
    cb.set_label("IG weight (normalised)\nred: common | blue: event-specific", fontsize=7, fontweight="bold")
    cb.ax.tick_params(labelsize=6)
    save_jpg(fig, FIG / f"surface_common_vs_event_{film}.jpg", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


for film in FILMS:
    c = cs[cs.film == film].sort_values("rank").reset_index(drop=True)
    if c.empty:
        continue
    network_map(film, c)
    event_maps(film, c)
    print(f"{film}: surface maps written")
