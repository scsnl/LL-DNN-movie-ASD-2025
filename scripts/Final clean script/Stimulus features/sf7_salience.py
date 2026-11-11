"""
Composite low-level audiovisual salience.

Each input channel is z-scored over the whole movie, PCA is run across channels,
and PC1 is the composite. One channel per family:
  loudness_mom, spectral_flux, speech_prob, motion_absdiff, cutrate_5s
Luminance and contrast are covariates and are not entered. PC1 is sign-flipped
if needed so that it correlates positively with the mean of its z-scored inputs.

Outputs: channels/<movie>__salience.csv, tables/pca_weights_<movie>.csv,
         tables/pca_variance_<movie>.csv
"""
import numpy as np
import pandas as pd

from sf_common import MOVIES, TABLES, ensure_dirs, load_channel, save_channel

INPUTS = ["loudness_mom", "spectral_flux", "speech_prob", "motion_absdiff", "cutrate_5s"]


def zsc(x):
    x = np.asarray(x, dtype=float)
    s = x.std(ddof=1)
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def main():
    ensure_dirs()
    for m in MOVIES:
        used = [c for c in INPUTS if load_channel(m, c) is not None and len(load_channel(m, c)) == MOVIES[m]["n_tr"]]
        Z = np.column_stack([zsc(load_channel(m, c)) for c in used])
        vals, vecs = np.linalg.eigh(np.cov(Z, rowvar=False))
        o = np.argsort(vals)[::-1]
        vals, vecs = vals[o], vecs[:, o]
        pc1 = (Z @ vecs)[:, 0]
        if np.corrcoef(pc1, Z.mean(axis=1))[0, 1] < 0:
            pc1, vecs[:, 0] = -pc1, -vecs[:, 0]
        save_channel(m, "salience", pc1)
        cols = [f"PC{i + 1}" for i in range(len(used))]
        pd.DataFrame(vecs, index=used, columns=cols).to_csv(TABLES / f"pca_weights_{m}.csv")
        expl = vals / vals.sum()
        pd.DataFrame({"component": cols, "eigenvalue": vals, "explained_variance": expl,
                      "cumulative": np.cumsum(expl)}).to_csv(TABLES / f"pca_variance_{m}.csv", index=False)
        print(f"{m}: salience PC1 explains {expl[0] * 100:.1f}% ({', '.join(used)})")


if __name__ == "__main__":
    main()
