"""
Automatic valence against human valence ratings (movieDM).

The ratings file holds one column per rating series (20 raters x 3 viewings, TR
grid). The human series is the mean over all columns. Each automatic valence
series is correlated with it, with circular-shift and phase-randomisation
p values; no HRF shift is applied. The column-to-rater layout is chosen by which
grouping gives higher within-group than between-group correlation, and the
primary series is also correlated with each rater's mean.

Outputs: tables/valence_vs_human.csv, tables/valence_vs_human_per_rater.csv,
         valence_validation_log.json
"""
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat

from sf_common import MOVIES, N_SURROGATE, OUT, SEED, TABLES, ensure_dirs, load_channel, media
from sf_stats import surrogate_corr

SERIES = ["v_rank_argmax_nrc", "v_rank_argmax_warriner", "v_3way_shotwise"]
PRIMARY = "v_rank_argmax_nrc"
KEYS = ["label", "n", "r", "p_naive", "p_circshift", "p_phaserand"]


def main():
    ensure_dirs()
    m = loadmat(str(media(MOVIES["movieDM"]["ratings_mat"])))
    hv = np.asarray(m[[k for k in m if not k.startswith("__")][0]], dtype=float)
    v_hum = hv.mean(axis=1)

    rows = []
    for nm in SERIES:
        s = load_channel("movieDM", nm)
        r = surrogate_corr(s, v_hum, N_SURROGATE, SEED, f"movieDM|{nm}|v_human")
        rows.append({**{k: r.get(k) for k in KEYS}, "series": nm})
    pd.DataFrame(rows).to_csv(TABLES / "valence_vs_human.csv", index=False)

    C = np.corrcoef(hv, rowvar=False)
    layouts = {"consecutive_triples": np.repeat(np.arange(20), 3), "blocks_of_20": np.tile(np.arange(20), 3)}
    diff = {}
    for nm, g in layouts.items():
        same = g[:, None] == g[None, :]
        diff[nm] = float(C[np.triu(same, 1)].mean() - C[np.triu(~same, 1)].mean())
    groups = layouts[max(diff, key=diff.get)]
    rater = np.column_stack([hv[:, groups == i].mean(axis=1) for i in range(20)])
    prim = load_channel("movieDM", PRIMARY)
    per = [{**{k: v for k, v in surrogate_corr(prim, rater[:, i], N_SURROGATE, SEED + i,
                                               f"movieDM|{PRIMARY}|rater{i + 1}").items() if k in KEYS},
            "rater": i + 1} for i in range(20)]
    pd.DataFrame(per).to_csv(TABLES / "valence_vs_human_per_rater.csv", index=False)
    rr = np.array([p["r"] for p in per], dtype=float)
    Cr = np.corrcoef(rater, rowvar=False)
    mp = float(Cr[np.triu_indices(20, 1)].mean())
    log = {"layout_within_minus_between_r": diff, "layout_used": max(diff, key=diff.get),
           "per_rater_r": {"mean": float(rr.mean()), "min": float(rr.min()), "max": float(rr.max()),
                           "n_positive": int((rr > 0).sum())},
           "mean_pairwise_r_between_raters": mp,
           "spearman_brown_20_raters": float(20 * mp / (1 + 19 * mp))}
    (OUT / "valence_validation_log.json").write_text(json.dumps(log, indent=2))
    for r in rows:
        print(f"{r['series']} vs human valence: r = {r['r']:.3f}, p_circ = {r['p_circshift']:.4f}")


if __name__ == "__main__":
    main()
