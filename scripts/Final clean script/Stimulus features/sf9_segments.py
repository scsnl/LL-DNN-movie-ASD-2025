"""
movieDM Segment 2 against the rest of the clip, and event windows against the
rest of their analysis window (Table S7).

Segments: TR [0, 237), [372, 543), [547, 750); TR 237-371 and 543-546 belong to
no segment, so two contrasts are computed: Segment 2 vs Segments 1+3 (Table S7)
and Segment 2 vs the rest of the movie. Each gets Cohen's d (pooled SD of the
two windows) and circular-shift and phase-randomisation p values.
Events: runs of significant attention TRs (movieTP E1-E6 within the 250 TR;
movieDM E1-E4 within Segment 2), inside vs outside.

Outputs: tables/segment_tests_movieDM.csv, tables/segment_descriptives_movieDM.csv,
         tables/event_tests.csv, tables/table_s7.csv
"""
import numpy as np
import pandas as pd

from sf_common import DM_SEGMENTS, MOVIES, N_SURROGATE, OUT, SEED, TABLES, ensure_dirs, load_channel, media, sig_clusters
from sf_stats import surrogate_mean_difference

PHYSICAL = ["loudness_mom", "loudness_short", "spectral_flux", "speech_prob", "salience",
            "motion_absdiff", "cutrate_5s", "cutrate_10s"]
SEMANTIC = ["a_rank_argmax_nrc", "a_rank_argmax_warriner", "v_rank_argmax_nrc", "v_rank_argmax_warriner",
            "v_3way_shotwise"]
COVARIATES = ["lum_mean", "rms_contrast"]
KIND = {**{s: "physical" for s in PHYSICAL}, **{s: "semantic" for s in SEMANTIC},
        **{s: "low-level covariate" for s in COVARIATES}}

TABLE_S7 = [("Physical", "Integrated loudness", "loudness_mom"),
            ("Physical", "Short-term loudness", "loudness_short"),
            ("Physical", "Spectral flux", "spectral_flux"),
            ("Physical", "Speech presence", "speech_prob"),
            ("Physical", "Composite audiovisual salience", "salience"),
            ("Physical", "Visual motion", "motion_absdiff"),
            ("Physical", "Cut rate", "cutrate_5s"),
            ("Physical", "RMS luminance contrast", "rms_contrast"),
            ("Semantic", "Valence rank, NRC lexicon", "v_rank_argmax_nrc"),
            ("Semantic", "Valence rank, Warriner lexicon", "v_rank_argmax_warriner"),
            ("Semantic", "Valence, published three-way annotation", "v_3way_shotwise")]


def segment_tests():
    n_tr = MOVIES["movieDM"]["n_tr"]
    masks = {}
    for k, (a, b) in DM_SEGMENTS.items():
        masks[k] = np.zeros(n_tr, dtype=bool)
        masks[k][a:b] = True
    other = masks["Segment1"] | masks["Segment3"]
    sub = masks["Segment2"] | other
    desc, tests = [], []
    for s in PHYSICAL + SEMANTIC + COVARIATES:
        x = load_channel("movieDM", s)
        for k, m in list(masks.items()) + [("WholeMovie", np.ones(n_tr, dtype=bool))]:
            desc.append({"series": s, "kind": KIND[s], "segment": k, "n_tr": int(m.sum()),
                         "mean": float(x[m].mean()), "sd": float(x[m].std(ddof=1))})
        for label, xv, mv in [("Seg2 vs Seg1+Seg3", x[sub], masks["Segment2"][sub]),
                              ("Seg2 vs rest of movie", x, masks["Segment2"])]:
            r = surrogate_mean_difference(xv, mv, N_SURROGATE, SEED, f"movieDM|{s}|{label}")
            pooled = np.sqrt((r["sd_in"] ** 2 + r["sd_out"] ** 2) / 2)
            r.update(series=s, kind=KIND[s], contrast=label, cohens_d=r["diff"] / pooled if pooled else np.nan)
            tests.append(r)
    pd.DataFrame(desc).to_csv(TABLES / "segment_descriptives_movieDM.csv", index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(TABLES / "segment_tests_movieDM.csv", index=False)
    return tests


def event_tests():
    windows = {"movieTP": (0, MOVIES["movieTP"]["n_tr"]), "movieDM": DM_SEGMENTS["Segment2"]}
    rows = []
    for movie, (a0, a1) in windows.items():
        off = MOVIES[movie]["attn_offset_tr"]
        mask = np.zeros(a1 - a0, dtype=bool)
        for r0, r1 in sig_clusters(media(MOVIES[movie]["attn_csv"])):
            b0, b1 = r0 + off, r1 + off
            if b1 < a0 or b0 >= a1:
                continue
            mask[max(b0, a0) - a0: min(b1, a1 - 1) - a0 + 1] = True
        for s in PHYSICAL + SEMANTIC + COVARIATES:
            xs = load_channel(movie, s)[a0:a1]
            r = surrogate_mean_difference(xs, mask, N_SURROGATE, SEED, f"{movie}|{s}|event")
            r.update(movie=movie, series=s, kind=KIND[s],
                     std_diff_vs_window_sd=r["diff"] / xs.std(ddof=1) if xs.std(ddof=1) else np.nan)
            rows.append(r)
    pd.DataFrame(rows).to_csv(TABLES / "event_tests.csv", index=False)


def main():
    ensure_dirs()
    tests = segment_tests()
    event_tests()
    t = tests[tests.contrast == "Seg2 vs Seg1+Seg3"].set_index("series")
    rows = [{"level": lv, "measure": name, "series": s, "mean_in": t.loc[s, "mean_in"],
             "mean_out": t.loc[s, "mean_out"], "difference": t.loc[s, "diff"], "d": t.loc[s, "cohens_d"],
             "p_circshift": t.loc[s, "p_circshift"], "p_phaserand": t.loc[s, "p_phaserand"]}
            for lv, name, s in TABLE_S7]
    var = pd.read_csv(OUT / "segment2_composition_robustness.csv")
    head = var[(var.lexicon == "nrc") & (var.split_shift_ranks == 0) & (var.share_method == "by_TR_argmax")].iloc[0]
    rows.append({"level": "Semantic", "measure": "Low- to high-arousal category ratio, negatively valenced scenes",
                 "series": "segment2_composition_robustness.csv", "difference": head.ratio_low_over_high})
    s7 = pd.DataFrame(rows)
    s7.to_csv(TABLES / "table_s7.csv", index=False)
    print(s7[["measure", "difference", "d", "p_circshift"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
