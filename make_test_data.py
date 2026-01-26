#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a small test dataset (10 subjects) for GitHub demo runs.

Priority Strategy:
1) If source paths are provided via environment variables, extract a subset 
   of 10 subjects from real datasets.
2) Otherwise, generate structurally consistent synthetic data (for pipeline validation).

Output (written to data/):
- combined_asd_td_movieDM_data.pklz
- combined_asd_td_movieTP_data.pklz
- all subject behavior data.csv

Optional Environment Variables:
- SOURCE_MOVIEDM_PKLZ
- SOURCE_MOVIETP_PKLZ
- SOURCE_BEHAVIOR_CSV
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SUB = 10
R = 246
T = 600  # Length compatible with seg2 indices (372-543) used in scripts


def _pick_10_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Select 5 ASD and 5 TD subjects if possible, otherwise first 10."""
    df = df.copy()
    df["label"] = df["label"].astype(str).str.lower()
    asd = df[df["label"] == "asd"]
    td = df[df["label"].isin(["td", "tdc"])]
    
    if len(asd) >= 5 and len(td) >= 5:
        out = pd.concat([asd.head(5), td.head(5)], axis=0).reset_index(drop=True)
    else:
        out = df.head(N_SUB).reset_index(drop=True)
    return out


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required metadata columns exist in the dataframe."""
    df = df.copy()
    if "subject_id" not in df.columns:
        df["subject_id"] = [f"subj{i:03d}" for i in range(len(df))]
    df["subject_id"] = df["subject_id"].astype(str).str.strip().str.lower()
    
    if "site" not in df.columns:
        df["site"] = np.random.choice(["ru", "cuny", "cbic"], size=len(df))
    if "gender" not in df.columns:
        df["gender"] = np.random.choice(["male", "female"], size=len(df))
    if "age" not in df.columns:
        df["age"] = np.random.uniform(6, 18, size=len(df)).round(2)
    if "percentofvolsrepaired" not in df.columns:
        df["percentofvolsrepaired"] = 0.0
    if "mean_fd" not in df.columns:
        df["mean_fd"] = 0.0
    if "label" not in df.columns:
        df["label"] = np.random.choice(["asd", "td"], size=len(df))
    return df


def _make_synthetic_dataset(name: str) -> pd.DataFrame:
    """Generate a fully synthetic fMRI ROI dataset."""
    rng = np.random.default_rng(42 if name == "MovieDM" else 43)
    labels = ["asd"] * 5 + ["td"] * 5
    rng.shuffle(labels)
    df = pd.DataFrame(
        {
            "subject_id": [f"{name.lower()}_{i:03d}" for i in range(N_SUB)],
            "label": labels,
            "site": rng.choice(["ru", "cuny", "cbic"], size=N_SUB),
            "gender": rng.choice(["male", "female"], size=N_SUB),
            "age": rng.uniform(6, 18, size=N_SUB).round(2),
            "percentofvolsrepaired": rng.uniform(0, 2, size=N_SUB).round(3),
            "mean_fd": rng.uniform(0, 0.2, size=N_SUB).round(4),
        }
    )
    # Generate synthetic time series: (T, R) array per subject
    base = rng.normal(0, 1, size=(N_SUB, 1, R)).astype(np.float32)
    noise = rng.normal(0, 0.5, size=(N_SUB, T, R)).astype(np.float32)
    ts = base + noise
    df["data"] = [ts[i] for i in range(N_SUB)]
    return df


def _make_behavior_csv(subject_ids: list[str], out_path: Path):
    """Generate a synthetic behavior CSV based on the provided identifiers."""
    # Attempt to load column names from assets/behavior/behavior-list.csv
    beh_list_path = HERE / "assets" / "behavior" / "behavior-list.csv"
    if not beh_list_path.exists():
        # Fallback to default total scores if asset is missing
        cols = ["CBCL,CBCL_Total_T", "RBS,RBS_Total", "SRS,SRS_Total_T"]
    else:
        cols = pd.read_csv(beh_list_path, header=None).iloc[:, 0].astype(str).tolist()
        cols = [c.strip().strip('"') for c in cols]

    rng = np.random.default_rng(123)
    df = pd.DataFrame({"Identifiers": subject_ids})
    for c in cols:
        # Generate random numeric scores
        df[c] = rng.normal(50, 10, size=len(subject_ids)).round(3)
    df.to_csv(out_path, index=False)


def main():
    src_dm = os.environ.get("SOURCE_MOVIEDM_PKLZ")
    src_tp = os.environ.get("SOURCE_MOVIETP_PKLZ")
    src_beh = os.environ.get("SOURCE_BEHAVIOR_CSV")

    # Process MovieDM
    if src_dm and Path(src_dm).exists():
        dm = pd.read_pickle(src_dm)
        dm = _pick_10_subjects(_ensure_columns(dm))
    else:
        dm = _make_synthetic_dataset("MovieDM")

    # Process MovieTP
    if src_tp and Path(src_tp).exists():
        tp = pd.read_pickle(src_tp)
        tp = _pick_10_subjects(_ensure_columns(tp))
    else:
        tp = _make_synthetic_dataset("MovieTP")

    out_dm = OUT_DIR / "combined_asd_td_movieDM_data.pklz"
    out_tp = OUT_DIR / "combined_asd_td_movieTP_data.pklz"
    dm.to_pickle(out_dm)
    tp.to_pickle(out_tp)

    # Behavior Table: Combine unique IDs from DM and TP to create a unified test file
    ids = sorted(set(dm["subject_id"].astype(str).str.lower().tolist() + 
                     tp["subject_id"].astype(str).str.lower().tolist()))
    out_beh = OUT_DIR / "all subject behavior data.csv"
    
    if src_beh and Path(src_beh).exists():
        beh = pd.read_csv(src_beh, low_memory=False)
        beh["Identifiers"] = beh["Identifiers"].astype(str).str.strip().str.lower()
        beh = beh[beh["Identifiers"].isin(ids)].copy()
        
        # Fallback to synthetic if identifiers are not found in the source CSV
        if beh.empty:
            _make_behavior_csv(ids, out_beh)
        else:
            beh.to_csv(out_beh, index=False)
    else:
        _make_behavior_csv(ids, out_beh)

    print("Saved test files to:", OUT_DIR)
    print(" -", out_dm.name)
    print(" -", out_tp.name)
    print(" -", out_beh.name)


if __name__ == "__main__":
    main()
