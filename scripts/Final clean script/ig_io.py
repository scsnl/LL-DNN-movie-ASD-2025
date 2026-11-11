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
