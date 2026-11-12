"""
Per-fold model outputs keyed by subject ID.

The training scripts write one directory per outer fold (saved_models/fold0 ...
fold4). Each array in a fold directory has one row per validation participant of
that fold, in the order of subject_ids.npy. Concatenating the folds gives an
order that differs from the input pickle, so rows are always matched to
participants by subject ID, never by position.

For fold directories written before subject_ids.npy was saved, the IDs are
recovered by repeating the training split (StratifiedKFold, 5 folds, shuffle,
random_state 42) on the same QC-filtered sample, and the recovered labels are
checked against labels.npy.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
CV_SEED = 42


def norm_id(s):
    return str(s).strip().lower()


def qc_filter(datao):
    """Sample used for training: motion QC, modal time-series shape, no NaN."""
    datao = datao[(datao["percentofvolsrepaired"] <= 10) & (datao["mean_fd"] <= 0.5)]
    shapes = [np.asarray(d).shape for d in datao.data]
    main_shape = max(set(shapes), key=shapes.count)
    datao = datao.iloc[[i for i, s in enumerate(shapes) if s == main_shape]].reset_index(drop=True)
    fmri_all = np.stack([np.asarray(d) for d in datao.data])
    nan_subs = np.unique(np.argwhere(np.isnan(fmri_all))[:, 0])
    return datao.drop(index=nan_subs).reset_index(drop=True)


def load_sample(data_path):
    """QC-filtered sample with normalised subject_id and y (0 = ASD, 1 = TD)."""
    d = qc_filter(pd.read_pickle(data_path))
    d["subject_id"] = d["subject_id"].map(norm_id)
    d["y"] = d["label"].str.lower().map({"asd": 0, "td": 1, "tdc": 1}).astype(int)
    if d["subject_id"].duplicated().any():
        raise ValueError("duplicate subject IDs in the QC-filtered sample")
    return d


def recover_fold_ids(data_path, n_folds=N_FOLDS, seed=CV_SEED):
    """Validation IDs per fold from the training split."""
    d = load_sample(data_path)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dummy = np.zeros((len(d), 1))
    return [d["subject_id"].to_numpy()[va] for _, va in skf.split(dummy, d["y"].to_numpy())], d


def fold_ids(fold_root, data_path=None, n_folds=N_FOLDS):
    """List of ID arrays, one per fold, checked against labels.npy."""
    out, rec, d = [], None, None
    for f in range(n_folds):
        fdir = os.path.join(fold_root, f"fold{f}")
        p = os.path.join(fdir, "subject_ids.npy")
        if os.path.exists(p):
            out.append(np.array([norm_id(s) for s in np.load(p)]))
            continue
        if data_path is None:
            raise FileNotFoundError(f"{p} missing and no data_path given to recover fold IDs")
        if rec is None:
            rec, d = recover_fold_ids(data_path, n_folds)
        out.append(rec[f])
    lab_of = None if d is None else dict(zip(d["subject_id"], d["y"]))
    for f, ids in enumerate(out):
        lbl = np.load(os.path.join(fold_root, f"fold{f}", "labels.npy"))
        if len(lbl) != len(ids):
            raise ValueError(f"fold{f}: {len(ids)} IDs vs {len(lbl)} labels")
        if lab_of is not None and not all(lab_of[s] == int(y) for s, y in zip(ids, lbl)):
            raise ValueError(f"fold{f}: recovered IDs do not match labels.npy")
    allids = np.concatenate(out)
    if len(set(allids)) != len(allids):
        raise ValueError("a subject ID appears in more than one fold")
    return out


def load_fold_array(fold_root, fname, data_path=None, n_folds=N_FOLDS):
    """(ids, array, labels), folds concatenated; row i belongs to ids[i]."""
    ids = fold_ids(fold_root, data_path, n_folds)
    arr = [np.load(os.path.join(fold_root, f"fold{f}", fname)) for f in range(n_folds)]
    lbl = [np.load(os.path.join(fold_root, f"fold{f}", "labels.npy")) for f in range(n_folds)]
    for f in range(n_folds):
        if arr[f].shape[0] != len(ids[f]):
            raise ValueError(f"fold{f}/{fname}: {arr[f].shape[0]} rows vs {len(ids[f])} IDs")
    return np.concatenate(ids), np.concatenate(arr), np.concatenate(lbl).astype(int)


def reindex(ids, arr, target_ids):
    """Rows of arr reordered to target_ids; every target ID must be present."""
    pos = {s: i for i, s in enumerate(ids)}
    missing = [s for s in target_ids if s not in pos]
    if missing:
        raise KeyError(f"{len(missing)} IDs not found, e.g. {missing[:3]}")
    return arr[[pos[s] for s in target_ids]]


def save_fold_ids(fold_dir, ids):
    np.save(os.path.join(fold_dir, "subject_ids.npy"), np.array([norm_id(s) for s in ids], dtype=str))


# ---------------------------------------------------------------------------
# Region labels, saliency indices, covariates
# ---------------------------------------------------------------------------
TOP_K = int(246 * 0.05)  # top 5% of 246 regions = 12


def column_to_bn_id(c):
    """Data column c is Brainnetome region ID c; column 0 has no region label."""
    return int(c) if int(c) > 0 else None


def global_group_map(fold_root, fname, label, n_folds=N_FOLDS):
    """Per-region group attribution: mean |IG| over a group's participants within
    each fold, median across folds, then mean over time."""
    per_fold = []
    for f in range(n_folds):
        ig = np.load(os.path.join(fold_root, f"fold{f}", fname))
        lbl = np.load(os.path.join(fold_root, f"fold{f}", "labels.npy"))
        g = ig[lbl == label]
        if len(g):
            per_fold.append(np.mean(np.abs(g), axis=0))
    return np.median(np.stack(per_fold), axis=0).mean(axis=0)


def top_regions(group_map, k=TOP_K):
    """Indices of the k regions with the largest group attribution, largest first."""
    return np.argsort(group_map)[::-1][:k]


def global_saliency_index(ig, regions):
    """Per participant: mean |IG| over the whole viewing period, averaged over regions."""
    return np.abs(ig).mean(axis=1)[:, regions].mean(axis=1)


def event_maps(ig, trs):
    """Per participant and region: median |IG| over the event's TRs."""
    trs = [t for t in trs if t < ig.shape[1]]
    return np.median(np.abs(ig[:, trs, :]), axis=1)


def neural_saliency_index(ig, trs, k=TOP_K):
    """Per participant Neural Saliency Index for one event and the event's top regions
    (top k of the group mean of the per-participant event maps)."""
    m = event_maps(ig, trs)
    regions = top_regions(m.mean(axis=0), k)
    return m[:, regions].mean(axis=1), regions


def event_windows(attn_csv, col="sig_raw"):
    """Contiguous runs of significant TRs in an attention table."""
    from itertools import groupby
    d = pd.read_csv(attn_csv)
    sig = sorted(d.loc[d[col].astype(str).str.upper().isin(["TRUE", "1"]), "timepoint"].astype(int))
    return [[t for _, t in g] for _, g in groupby(enumerate(sig), lambda x: x[1] - x[0])]


def covariates(d):
    """Age, sex, mean framewise displacement and site (indicator coded), keyed by subject_id."""
    out = pd.DataFrame({"subject_id": d["subject_id"].map(norm_id).values,
                        "age": pd.to_numeric(d["age"], errors="coerce").values,
                        "sex": d["gender"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0}).values,
                        "mean_fd": pd.to_numeric(d["mean_fd"], errors="coerce").values})
    site = pd.get_dummies(d["site"].astype(str).str.strip().str.lower(), prefix="site",
                          drop_first=True, dtype=float).reset_index(drop=True)
    return pd.concat([out, site], axis=1)


def partial_corr(x, y, Z):
    """Pearson correlation of x and y after regressing Z (with intercept) from both."""
    from scipy.stats import pearsonr
    Z1 = np.column_stack([np.ones(len(x)), Z])
    rx = x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]
    ry = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
    return pearsonr(rx, ry)


def correlate_with_behaviour(index_df, cov_df, behav, cols, min_n):
    """Partial correlations of every column in index_df (except subject_id) with each
    behavioural column, adjusting for the covariates; all joins on subject_id."""
    behav = behav.copy()
    behav["subject_id"] = behav["Identifiers"].map(norm_id)
    zcols = [c for c in cov_df.columns if c != "subject_id"]
    rows = []
    for feat in [c for c in index_df.columns if c != "subject_id"]:
        for col in cols:
            if col in ("Identifiers", "subject_id"):
                continue
            b = behav[["subject_id", col]].assign(score=pd.to_numeric(behav[col], errors="coerce"))
            df = index_df[["subject_id", feat]].merge(cov_df, on="subject_id").merge(
                b[["subject_id", "score"]], on="subject_id").dropna()
            if len(df) < min_n:
                continue
            r, p = partial_corr(df[feat].to_numpy(float), df["score"].to_numpy(float), df[zcols].to_numpy(float))
            rows.append({"feature": feat, "scale": col, "partial_r": r, "partial_p": p, "n": len(df)})
    return pd.DataFrame(rows)
