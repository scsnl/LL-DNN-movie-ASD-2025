"""
Final Step 19: IG-Only Partial Least Squares Correlation (PLSC)

This script performs multivariate association between Event-level Neural Saliency 
and the three core behavioral total scores: CBCL Total, RBS Total, and SRS Total.
Analysis Flow:
1. Calculates subject-level saliency indices for each significant attention cluster.
2. Applies Rank-based Inverse Normal Transform (Rank-INT) to features and targets.
3. Residualizes data by regressing out Age, Sex, and Site.
4. Conducts PLSC to extract Latent Variables (LVs) maximizing the covariance 
   between brain events and behavioral totals.
5. Evaluates significance via permutation tests and stability via bootstrap BSR.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import groupby
from operator import itemgetter
from sklearn.cross_decomposition import PLSCanonical
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import resample
from scipy.stats import norm
from tqdm import tqdm

# Environment configuration
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
N_PERM = int(os.environ.get("N_PERM", "5000"))
N_BOOT = int(os.environ.get("N_BOOT", "5000"))
MIN_SUBJECTS = 10 if TEST_MODE else 30
TOP_PERCENT = 0.05

def _rank_int(x: np.ndarray) -> np.ndarray:
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    return qt.fit_transform(x)

def _residualize(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    if Z is None or Z.size == 0: return Y
    Y_res = np.zeros_like(Y, dtype=float); Z_ = np.hstack([np.ones((Z.shape[0], 1)), Z])
    for j in range(Y.shape[1]):
        beta, *_ = np.linalg.lstsq(Z_, Y[:, j], rcond=None)
        Y_res[:, j] = Y[:, j] - Z_ @ beta
    return Y_res

def main():
    OUT_DIR = Path("ig_only_plsc_totals"); OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS = [
        {
            "name": "MovieDM_Seg2",
            "fmri_file": os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz") if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz",
            "behavior_file": os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv",
            "ig_root": os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models",
            "attn_csv": os.path.join(TEST_OUTPUT_DIR, "figures", "attn_significance.csv") if TEST_OUTPUT_DIR else "figures/attn_significance.csv",
        }
    ]
    targets = ["CBCL,CBCL_Total_T", "RBS,RBS_Total", "SRS,SRS_Total_T"]

    for conf in CONFIGS:
        print(f"Processing {conf['name']}...")
        if not os.path.exists(conf["attn_csv"]): continue
        df_attn = pd.read_csv(conf["attn_csv"])
        sig_trs = sorted(df_attn.loc[df_attn["sig_raw"].astype(bool), "timepoint"].astype(int).tolist())
        clusters = [list(map(itemgetter(1), g)) for _, g in groupby(enumerate(sig_trs), lambda ix: ix[0] - ix[1])]
        if not clusters: continue

        datao = pd.read_pickle(conf["fmri_file"])
        datao = datao[(datao["percentofvolsrepaired"] <= 10) & (datao["mean_fd"] <= 0.5)]
        datao["subject_id"] = datao["subject_id"].astype(str).str.strip().str.lower()
        asd = datao[datao["label"].str.lower() == "asd"].reset_index(drop=True)
        
        all_ig = []
        for fold in range(5):
            p_ig = os.path.join(conf["ig_root"], f"fold{fold}", "ig_roi_milout.npy")
            p_lbl = os.path.join(conf["ig_root"], f"fold{fold}", "labels.npy")
            if os.path.exists(p_ig): all_ig.append(np.load(p_ig)[np.load(p_lbl) == 0])
        if not all_ig: continue
        ig_asd = np.vstack(all_ig)

        X_list = []
        for trs in clusters[:6]:
            valid_trs = [t for t in trs if t < ig_asd.shape[1]]
            if not valid_trs: continue
            sub_abs_med = np.median(np.abs(ig_asd[:, valid_trs, :]), axis=1)
            top_idx = np.argsort(np.mean(sub_abs_med, axis=0))[-int(246*TOP_PERCENT):]
            X_list.append(np.mean(sub_abs_med[:, top_idx], axis=1).reshape(-1, 1))
        if not X_list: continue
        X = np.hstack(X_list)

        behav = pd.read_csv(conf["behavior_file"], low_memory=False)
        behav["Identifiers"] = behav["Identifiers"].astype(str).str.strip().str.lower()
        cols = [c for c in targets if c in behav.columns]
        df_final = pd.DataFrame({"subject_id": asd["subject_id"]}).join(pd.DataFrame(X, columns=[f"E{i+1}" for i in range(X.shape[1])])).merge(behav[["Identifiers"] + cols], left_on="subject_id", right_on="Identifiers", how="inner")
        
        age = pd.to_numeric(asd["age"], errors="coerce").values.reshape(-1, 1)
        gender = asd["gender"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0}).values.reshape(-1, 1)
        site = pd.get_dummies(asd["site"].astype(str).str.strip().str.lower(), dtype=float).values
        Z = np.concatenate([age, gender, site], axis=1)[df_final.index]
        
        df_final = df_final.dropna(subset=cols)
        if len(df_final) < MIN_SUBJECTS: continue
        X_in, Y_in = df_final[[f"E{i+1}" for i in range(X.shape[1])]].values, df_final[cols].values
        
        # PLSC logic
        Xr, Yr = _residualize(_rank_int(X_in), _rank_int(Z)), _residualize(_rank_int(Y_in), _rank_int(Z))
        plsc = PLSCanonical(n_components=1, scale=False).fit(Xr, Yr)
        r_lv1 = np.corrcoef(plsc.transform(Xr, Yr)[0][:, 0], plsc.transform(Xr, Yr)[1][:, 0])[0, 1]
        
        null = np.array([np.corrcoef(PLSCanonical(n_components=1, scale=False).fit(Xr, Yr[np.random.permutation(Yr.shape[0])]).transform(Xr, Yr[np.random.permutation(Yr.shape[0])])[0][:, 0], PLSCanonical(n_components=1, scale=False).fit(Xr, Yr[np.random.permutation(Yr.shape[0])]).transform(Xr, Yr[np.random.permutation(Yr.shape[0])])[1][:, 0])[0, 1] for _ in range(N_PERM)])
        p_perm = np.mean(null >= r_lv1)
        
        boot_xw, boot_yw = np.zeros((N_BOOT, X_in.shape[1])), np.zeros((N_BOOT, Y_in.shape[1]))
        xw0, yw0 = plsc.x_weights_[:, 0], plsc.y_weights_[:, 0]
        for b in range(N_BOOT):
            idx = resample(np.arange(Xr.shape[0]))
            pb = PLSCanonical(n_components=1, scale=False).fit(Xr[idx], Yr[idx])
            xw, yw = pb.x_weights_[:, 0], pb.y_weights_[:, 0]
            if np.dot(xw, xw0) < 0: xw, yw = -xw, -yw
            boot_xw[b], boot_yw[b] = xw, yw
        
        x_bsr = boot_xw.mean(0) / (boot_xw.std(0, ddof=1) + 1e-12)
        y_bsr = boot_yw.mean(0) / (boot_yw.std(0, ddof=1) + 1e-12)
        
        out = OUT_DIR / conf["name"]; out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"event": [f"Event_{i+1}" for i in range(X_in.shape[1])], "weight": xw0, "bsr": x_bsr, "p": 2*norm.sf(np.abs(x_bsr))}).to_csv(out / "brain_weights.csv", index=False)
        pd.DataFrame({"behavior": cols, "weight": yw0, "bsr": y_bsr, "p": 2*norm.sf(np.abs(y_bsr))}).to_csv(out / "behavior_weights.csv", index=False)
        print(f"  LV1 r = {r_lv1:.4f}, p = {p_perm:.4f}")

if __name__ == "__main__":
    main()
