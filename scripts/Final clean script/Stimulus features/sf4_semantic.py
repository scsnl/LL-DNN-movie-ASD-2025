"""
Semantic series from the emotion labels, and the category composition of
movieDM Segment 2.

Series per TR (both movies):
  {a,v}_rank_argmax_{nrc,warriner}  normalised lexicon rank (rank / n ranked) of the
                                    argmax CLIP category of each frame, averaged per TR
  v_3way_shotwise                   published construction: shot label from step 5
                                    -> positive / neutral / negative = +1 / 0 / -1

Composition: share of Segment 2 taken by negatively valenced categories (step-5
grouping) of high vs low arousal, for 2 lexicons x 3 split positions (median and
one rank either side) x 4 share methods = 24 variants.

Outputs: channels/*, segment2_composition.csv, segment2_composition_robustness.csv
"""
import numpy as np
import pandas as pd

from sf_common import (CATEGORY_SCORE, DM_SEGMENTS, MOVIES, OUT, PROMPT_TO_TERM, TR, ensure_dirs, frames_to_tr,
                       load_category_map, load_prompts, save_channel, scene_csv)

LEXICONS = ["nrc", "warriner"]
DIMS = {"a": "arousal", "v": "valence"}


def rank_series(movie, rank_tab):
    df = pd.read_csv(OUT / f"clip_probs_{movie}.csv")
    P = df[[c for c in df.columns if c.startswith("p") and c[1:].isdigit()]].to_numpy()
    argmax_idx = np.argsort(-P, axis=1)[:, 0]
    for lex in LEXICONS:
        for d, dim in DIMS.items():
            ranks = rank_tab[f"{lex}_{dim}_rank"].to_numpy(dtype=float)
            s = (ranks / int(np.isfinite(ranks).sum()))[argmax_idx]
            # a category without a norm carries the last valid value forward
            s = pd.Series(s).ffill().bfill().to_numpy()
            save_channel(movie, f"{d}_rank_argmax_{lex}", frames_to_tr(s), s, df["time_sec"].to_numpy())


def three_way(movie, cat_map):
    n_tr = MOVIES[movie]["n_tr"]
    centres = (np.arange(n_tr) + 0.5) * TR
    v3 = np.full(n_tr, np.nan)
    for _, r in pd.read_csv(scene_csv(movie)).sort_values("start_sec").iterrows():
        v3[(centres >= r["start_sec"]) & (centres < r["end_sec"])] = CATEGORY_SCORE[cat_map[r["emotion"]]]
    save_channel(movie, "v_3way_shotwise", pd.Series(v3).ffill().bfill().to_numpy())


def composition(movie, rank_tab, prompts, cat_map):
    s0, s1 = DM_SEGMENTS["Segment2"]
    t0, t1 = s0 * TR, s1 * TR
    dur = t1 - t0
    trp = pd.read_csv(OUT / f"clip_probs_{movie}_tr.csv")
    n_tr = len(trp)
    seg = trp.iloc[s0:s1]
    pcols = [c for c in trp.columns if c.startswith("p") and c[1:].isdigit()]
    mean_p_seg, mean_p_all = seg[pcols].mean().to_numpy(), trp[pcols].mean().to_numpy()
    cnt_seg, cnt_all = seg["argmax_prompt"].value_counts(), trp["argmax_prompt"].value_counts()

    shares = {}
    for tag, kind in [("shot", "visual"), ("semantic", "semantic")]:
        d = pd.read_csv(scene_csv(movie, kind))
        ov = (d["end_sec"].clip(upper=t1) - d["start_sec"].clip(lower=t0)).clip(lower=0)
        d = d.assign(overlap_sec=ov, length_sec=d["end_sec"] - d["start_sec"])
        shares[tag] = {"in_seg2": d.groupby("emotion")["overlap_sec"].sum().to_dict(),
                       "whole": d.groupby("emotion")["length_sec"].sum().to_dict(),
                       "total_whole": float(d["length_sec"].sum())}

    rows = []
    for j, p in enumerate(prompts):
        r = {"prompt": p, "lookup_term": PROMPT_TO_TERM[p], "step5_valence_group": cat_map[p]}
        for lex in LEXICONS:
            r[f"{lex}_arousal_rank"] = rank_tab.iloc[j][f"{lex}_arousal_rank"]
            r[f"{lex}_arousal_group_median"] = rank_tab.iloc[j][f"{lex}_arousal_group"]
        n_seg, n_all = int(cnt_seg.get(p, 0)), int(cnt_all.get(p, 0))
        r.update(tr_count_seg2=n_seg, tr_share_seg2=n_seg / (s1 - s0), tr_count_whole=n_all,
                 tr_share_whole=n_all / n_tr, mean_prob_seg2=float(mean_p_seg[j]),
                 mean_prob_whole=float(mean_p_all[j]))
        for tag in ("shot", "semantic"):
            ins = float(shares[tag]["in_seg2"].get(p, 0.0))
            r[f"{tag}wise_share_seg2"] = ins / dur
            r[f"{tag}wise_share_whole"] = float(shares[tag]["whole"].get(p, 0.0)) / shares[tag]["total_whole"]
        rows.append(r)
    comp = pd.DataFrame(rows)

    variants = []
    neg = comp["step5_valence_group"].to_numpy() == "negative"
    for lex in LEXICONS:
        base_low = int((rank_tab[f"{lex}_arousal_group"] == "low").sum())
        r_ = rank_tab[f"{lex}_arousal_rank"]
        for shift in (-1, 0, +1):
            n_low = base_low + shift
            grp = np.where(r_.isna(), "missing", np.where(r_ <= n_low, "low", "high"))
            comp[f"{lex}_group_shift{shift:+d}"] = grp
            hi, lo = neg & (grp == "high"), neg & (grp == "low")
            for method, col in [("by_TR_argmax", "tr_share_seg2"), ("by_shot_length", "shotwise_share_seg2"),
                                ("by_semantic_segment_length", "semanticwise_share_seg2"),
                                ("mean_probability", "mean_prob_seg2")]:
                h, l_ = float(comp.loc[hi, col].sum()), float(comp.loc[lo, col].sum())
                variants.append({"lexicon": lex, "split_shift_ranks": shift, "n_low_categories": n_low,
                                 "share_method": method, "high_arousal_negative": h,
                                 "low_arousal_negative": l_, "ratio_low_over_high": l_ / h if h > 0 else np.inf})
    comp.to_csv(OUT / "segment2_composition.csv", index=False)
    var = pd.DataFrame(variants)
    var.to_csv(OUT / "segment2_composition_robustness.csv", index=False)
    return var


def main():
    ensure_dirs()
    prompts, cat_map = load_prompts(), load_category_map()
    rank_tab = pd.read_csv(OUT / "lexicon_ranking.csv").sort_values("prompt_index").reset_index(drop=True)
    assert list(rank_tab["prompt"]) == prompts
    for m in MOVIES:
        rank_series(m, rank_tab)
        three_way(m, cat_map)
    var = composition("movieDM", rank_tab, prompts, cat_map)
    head = var[(var.lexicon == "nrc") & (var.split_shift_ranks == 0) & (var.share_method == "by_TR_argmax")].iloc[0]
    print(f"Segment 2 low/high-arousal negative ratio {head.ratio_low_over_high:.2f} "
          f"(range {var.ratio_low_over_high.min():.2f}-{var.ratio_low_over_high.max():.2f}, {len(var)} variants)")


if __name__ == "__main__":
    main()
