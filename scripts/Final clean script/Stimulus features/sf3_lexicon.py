"""
Valence and arousal norms of the 30 prompt categories, used ordinally.

Each category is looked up in two lexicons, ranked (1 = lowest, average ranks for
ties), and split at the median of its own norms. Missing terms are reported and
not replaced.
  NRC VAD Lexicon (Mohammad 2018), scale [0, 1]
  Warriner, Kuperman & Brysbaert (2013), scale [1, 9]
The lexicon files are read from STIM_LEXICON_DIR and downloaded there if absent.
With TEST_MODE=1 small seeded tables are written instead.

Outputs: lexicon_ranking.csv, lexicon_ranking_log.json
"""
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

from sf_common import LEXICON_DIR, OUT, PROMPT_TO_TERM, SEED, TEST_MODE, ensure_dirs, load_category_map, load_prompts

NRC_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip"
NRC_MEMBER = "NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt"
WARRINER_URL = "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv"
NRC_TXT = "NRC-VAD-Lexicon.txt"
WARRINER_CSV = "Ratings_Warriner_et_al.csv"


def write_test_lexicons():
    rng = np.random.default_rng(SEED)
    terms = sorted(set(PROMPT_TO_TERM.values()))
    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"term": terms, "V": rng.uniform(0, 1, len(terms)), "A": rng.uniform(0, 1, len(terms)),
                  "D": rng.uniform(0, 1, len(terms))}).to_csv(LEXICON_DIR / NRC_TXT, sep="\t",
                                                               header=False, index=False)
    pd.DataFrame({"Word": terms, "V.Mean.Sum": rng.uniform(1, 9, len(terms)),
                  "A.Mean.Sum": rng.uniform(1, 9, len(terms))}).to_csv(LEXICON_DIR / WARRINER_CSV, index=False)


def fetch_lexicons():
    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    if not (LEXICON_DIR / NRC_TXT).exists():
        z = LEXICON_DIR / "NRC-VAD-Lexicon.zip"
        urllib.request.urlretrieve(NRC_URL, z)
        with zipfile.ZipFile(z) as zf, zf.open(NRC_MEMBER) as src:
            (LEXICON_DIR / NRC_TXT).write_bytes(src.read())
    if not (LEXICON_DIR / WARRINER_CSV).exists():
        urllib.request.urlretrieve(WARRINER_URL, LEXICON_DIR / WARRINER_CSV)


def load_lexicons():
    nrc = pd.read_csv(LEXICON_DIR / NRC_TXT, sep="\t", header=None, names=["term", "V", "A", "D"])
    nrc["term"] = nrc["term"].astype(str)
    w = pd.read_csv(LEXICON_DIR / WARRINER_CSV)
    war = pd.DataFrame({"term": w["Word"].astype(str), "A": w["A.Mean.Sum"].astype(float),
                        "V": w["V.Mean.Sum"].astype(float)})
    return {"nrc": nrc[["term", "A", "V"]].drop_duplicates("term").set_index("term"),
            "warriner": war.drop_duplicates("term").set_index("term")}


def main():
    ensure_dirs()
    write_test_lexicons() if TEST_MODE else fetch_lexicons()
    prompts, cat_map, lex = load_prompts(), load_category_map(), load_lexicons()

    rows, missing = [], {k: [] for k in lex}
    for i, p in enumerate(prompts):
        t = PROMPT_TO_TERM[p]
        r = {"prompt_index": i, "prompt": p, "lookup_term": t, "step5_valence_group": cat_map[p]}
        for k, tab in lex.items():
            present = t in tab.index
            r[f"{k}_arousal_norm"] = float(tab.loc[t, "A"]) if present else np.nan
            r[f"{k}_valence_norm"] = float(tab.loc[t, "V"]) if present else np.nan
            r[f"{k}_present"] = present
            if not present:
                missing[k].append(t)
        rows.append(r)
    tab = pd.DataFrame(rows)

    split = {}
    for k in lex:
        a, v = tab[f"{k}_arousal_norm"], tab[f"{k}_valence_norm"]
        tab[f"{k}_arousal_rank"] = a.rank(method="average")
        tab[f"{k}_valence_rank"] = v.rank(method="average")
        a_med = float(np.nanmedian(a))
        tab[f"{k}_arousal_group"] = np.where(a.isna(), "missing", np.where(a > a_med, "high", "low"))
        split[k] = {"arousal_median": a_med, "valence_median": float(np.nanmedian(v)),
                    "n_with_norm": int(a.notna().sum()),
                    "n_high": int((tab[f"{k}_arousal_group"] == "high").sum()),
                    "n_low": int((tab[f"{k}_arousal_group"] == "low").sum())}
    both = tab["nrc_present"] & tab["warriner_present"]
    rho_a, p_a = stats.spearmanr(tab.loc[both, "nrc_arousal_norm"], tab.loc[both, "warriner_arousal_norm"])
    rho_v, p_v = stats.spearmanr(tab.loc[both, "nrc_valence_norm"], tab.loc[both, "warriner_valence_norm"])
    tab.to_csv(OUT / "lexicon_ranking.csv", index=False)
    log = {"split": split, "missing_terms": missing,
           "agreement": {"n": int(both.sum()), "spearman_arousal": float(rho_a), "p_arousal": float(p_a),
                         "spearman_valence": float(rho_v), "p_valence": float(p_v)}}
    (OUT / "lexicon_ranking_log.json").write_text(json.dumps(log, indent=2))
    for k, s in split.items():
        print(f"{k}: arousal median {s['arousal_median']:.4f}, {s['n_high']} high / {s['n_low']} low, "
              f"missing {missing[k]}")
    print(f"Spearman arousal NRC vs Warriner = {rho_a:.3f} (n = {int(both.sum())})")


if __name__ == "__main__":
    main()
