"""
Final Step 18: Integrated Gradients (IG) Weighted CCA Analysis

This script implements a multivariate analysis to map neural saliency across 
all identified events to clinical behavioral profiles.
Key Features:
1. Canonical Correlation Analysis (CCA) between Event-level Neural Saliency 
   and Behavioral scores.
2. Freedman-Lane Permutation: A robust strategy for significance testing 
   that properly handles covariates by permuting residuals.
3. Bootstrap Resampling: Estimates the stability of feature loadings (structure 
   coefficients) and calculates Bootstrap Ratios (BSR).
4. Rank-INT Transformation: Normalizes feature distributions to improve 
   CCA stability and handle potential outliers.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from scipy.stats import norm, pearsonr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from itertools import groupby
from operator import itemgetter
from tqdm import tqdm
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore")

# Environment configuration
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
N_PERM = int(os.environ.get("N_PERM", "5000"))
N_BOOT = int(os.environ.get("N_BOOT", "5000"))
MIN_SUBJECTS = 10 if TEST_MODE else 30
TOP_PERCENT = 0.05

class PermCCA:
    def __init__(self, n_components=1, n_perm=5000):
        self.n_components = n_components
        self.n_perm = n_perm
        self.cca = CCA(n_components=n_components, scale=False)
        self.result = {}

    def _regress_out(self, target, confound):
        if confound is None or confound.size == 0: return target
        beta, _, _, _ = np.linalg.lstsq(confound, target, rcond=None)
        return target - confound @ beta

    def fit(self, X, Y, Z=None):
        X_z = (X - X.mean(0)) / (X.std(0, ddof=1) + 1e-10)
        Y_z = (Y - Y.mean(0)) / (Y.std(0, ddof=1) + 1e-10)
        Z_z = np.hstack([np.ones((Z.shape[0], 1)), (Z - Z.mean(0)) / (Z.std(0, ddof=1) + 1e-10)]) if Z is not None else None
        
        X_res = self._regress_out(X_z, Z_z)
        Y_res = self._regress_out(Y_z, Z_z)
        self.cca.fit(X_res, Y_res)
        x_sc, y_sc = self.cca.transform(X_res, Y_res)
        n_comps = min(self.n_components, x_sc.shape[1])
        
        true_corrs = np.array([np.corrcoef(x_sc[:, i], y_sc[:, i])[0, 1] for i in range(n_comps)])
        true_x_loadings = np.array([[np.corrcoef(X_res[:, f], x_sc[:, c])[0, 1] for c in range(n_comps)] for f in range(X.shape[1])])
        true_y_loadings = np.array([[np.corrcoef(Y_res[:, f], y_sc[:, c])[0, 1] for c in range(n_comps)] for f in range(Y.shape[1])])

        null_corrs = np.zeros((self.n_perm, n_comps))
        for i in range(self.n_perm):
            Y_res_p = Y_res[np.random.permutation(Y_res.shape[0])]
            cp = CCA(n_components=n_comps, scale=False); cp.fit(X_res, Y_res_p)
            xs, ys = cp.transform(X_res, Y_res_p)
            null_corrs[i, :] = [np.corrcoef(xs[:, c], ys[:, c])[0, 1] for c in range(n_comps)]

        self.result = {'true_corrs': true_corrs, 'p_vals': [np.mean(null_corrs[:, c] >= true_corrs[c]) for c in range(n_comps)],
                       'x_loadings': true_x_loadings, 'y_loadings': true_y_loadings, 'x_scores': x_sc, 'y_scores': y_sc, 'X_res': X_res, 'Y_res': Y_res}
        return self

    def bootstrap_loadings(self, n_boot=5000):
        Xr, Yr = self.result['X_res'], self.result['Y_res']
        txl, tyl = self.result['x_loadings'], self.result['y_loadings']
        n_comps = txl.shape[1]
        boot_x, boot_y = np.zeros((n_boot, Xr.shape[1], n_comps)), np.zeros((n_boot, Yr.shape[1], n_comps))
        for b in range(n_boot):
            idx = resample(np.arange(Xr.shape[0]))
            Xb, Yb = Xr[idx], Yr[idx]
            cb = CCA(n_components=n_comps, scale=False); cb.fit(Xb, Yb)
            xb, yb = cb.transform(Xb, Yb)
            for i in range(min(n_comps, xb.shape[1])):
                xl = np.array([np.corrcoef(Xb[:, f], xb[:, i])[0, 1] for f in range(Xr.shape[1])])
                yl = np.array([np.corrcoef(Yb[:, f], yb[:, i])[0, 1] for f in range(Yr.shape[1])])
                if np.corrcoef(xl, txl[:, i])[0, 1] < 0: xl, yl = -xl, -yl
                boot_x[b, :, i], boot_y[b, :, i] = xl, yl
        self.result['x_bsr'] = np.mean(boot_x, 0) / (np.std(boot_x, 0, ddof=1) + 1e-10)
        self.result['y_bsr'] = np.mean(boot_y, 0) / (np.std(boot_y, 0, ddof=1) + 1e-10)
        return self

def main():
    OUT_DIR = "ig_weighted_cca_analysis"; os.makedirs(OUT_DIR, exist_ok=True)
    CONFIGS = [
        {"name": "MovieDM_Seg2", 
         "fmri_file": os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz") if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz",
         "behavior_file": os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv",
         "ig_root": os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models",
         "attn_csv": os.path.join(TEST_OUTPUT_DIR, "figures", "attn_significance.csv") if TEST_OUTPUT_DIR else "figures/attn_significance.csv",
         "tr_start": 0, "tr_end": 9999}
    ]
    
    for conf in CONFIGS:
        print(f"Processing {conf['name']}...")
        if not os.path.exists(conf['attn_csv']): continue
        df_attn = pd.read_csv(conf['attn_csv'])
        sig_trs = df_attn.loc[df_attn['sig_raw'], 'timepoint'].tolist()
        clusters = [list(map(itemgetter(1), g)) for _, g in groupby(enumerate(sorted(sig_trs)), lambda ix: ix[0]-ix[1])]
        
        datao = pd.read_pickle(conf['fmri_file'])
        datao = datao[(datao['percentofvolsrepaired']<=10) & (datao['mean_fd']<=0.5)]
        datao['subject_id'] = datao['subject_id'].astype(str).str.strip().str.lower()
        asd = datao[datao['label'].str.lower()=='asd'].reset_index(drop=True)
        
        all_ig = []
        for f in range(5):
            p = os.path.join(conf['ig_root'], f"fold{f}", "ig_roi_milout.npy")
            if os.path.exists(p): all_ig.append(np.load(p)[np.load(os.path.join(conf['ig_root'], f"fold{f}", "labels.npy"))==0])
        if not all_ig: continue
        ig_asd = np.vstack(all_ig)
        
        X_list = []
        for clus in clusters[:6]:
            sub_med = np.median(np.abs(ig_asd[:, [t for t in clus if t < ig_asd.shape[1]], :]), axis=1)
            top_idx = np.argsort(np.mean(sub_med, 0))[-int(246*TOP_PERCENT):]
            X_list.append(np.mean(sub_med[:, top_idx], axis=1).reshape(-1, 1))
        if not X_list: continue
        X = np.hstack(X_list)
        
        behav = pd.read_csv(conf['behavior_file'], low_memory=False)
        behav['Identifiers'] = behav['Identifiers'].astype(str).str.strip().str.lower()
        targets = ["CBCL,CBCL_Total_T", "RBS,RBS_Total", "SRS,SRS_Total_T"]
        df_all = pd.DataFrame({'subject_id': asd['subject_id']}).join(pd.DataFrame(X, columns=[f'E{i+1}' for i in range(X.shape[1])])).merge(behav[['Identifiers']+targets], left_on='subject_id', right_on='Identifiers', how='inner')
        df_all = df_all.merge(asd[['subject_id', 'age', 'gender', 'site']], on='subject_id').dropna()
        if len(df_all) < MIN_SUBJECTS: continue
        
        X_final = QuantileTransformer(output_distribution='normal', n_quantiles=len(df_all)).fit_transform(df_all[[f'E{i+1}' for i in range(X.shape[1])]].values)
        Y_final = StandardScaler().fit_transform(df_all[targets].values)
        Cov = df_all[['age', 'gender', 'site']].apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        n_comps = min(X_final.shape[1], Y_final.shape[1], X_final.shape[0])
        pcca = PermCCA(n_components=n_comps, n_perm=N_PERM).fit(X_final, Y_final, Z=Cov).bootstrap_loadings(n_boot=N_BOOT)
        
        res_dir = os.path.join(OUT_DIR, conf['name']); os.makedirs(res_dir, exist_ok=True)
        for i in range(n_comps):
            print(f"  CV{i+1}: p={pcca.result['p_vals'][i]:.4f}, r={pcca.result['true_corrs'][i]:.4f}")
            pd.DataFrame({'Feature': [f'Event {j+1}' for j in range(X_final.shape[1])], 'Loading': pcca.result['x_loadings'][:, i], 'BSR': pcca.result['x_bsr'][:, i]}).to_csv(os.path.join(res_dir, f"brain_cv{i+1}.csv"), index=False)
            pd.DataFrame({'Feature': targets, 'Loading': pcca.result['y_loadings'][:, i], 'BSR': pcca.result['y_bsr'][:, i]}).to_csv(os.path.join(res_dir, f"behav_cv{i+1}.csv"), index=False)

if __name__ == "__main__":
    main()
