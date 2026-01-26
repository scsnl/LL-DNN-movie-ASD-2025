"""
Final Step 13 (MIL Path): Cluster-Based Brain-Behavior Association

This script investigates neural associations during specific model-identified events.
Analysis Pipeline:
1. Detects timepoints where ASD and TDC attention scores differ significantly.
2. Groups these timepoints into "Clusters" (neural events).
3. Automatically classifies clusters into "High Attention" or "Low Attention" 
   events based on their mean magnitude relative to the global average.
4. Calculates subject-level Neural Saliency Indices (mean activation of Top-K ROIs) 
   during these specific events.
5. Performs partial correlation with clinical phenotypes, adjusting for 
   Age, Sex, and Scanning Site.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
from operator import itemgetter

# Execution configuration
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")

OUT_DIR = "cluster_split_correlation_auto"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    {
        "name": "MovieDM_Seg2",
        "fmri_file": os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz") if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz",
        "behavior_file": os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv",
        "ig_root": os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models",
        "attn_csv": os.path.join(TEST_OUTPUT_DIR, "figures", "attn_significance.csv") if TEST_OUTPUT_DIR else "figures/attn_significance.csv",
        "tr_start": 0, "tr_end": 9999
    }
]

THRESHOLDS = [0.05, 0.10, 0.20, 0.30]

def get_clusters(attn_csv, tr_start, tr_end):
    if not os.path.exists(attn_csv): return [], []
    df = pd.read_csv(attn_csv)
    seg_mask = (df['timepoint'] >= tr_start) & (df['timepoint'] < tr_end)
    df_seg = df.loc[seg_mask].copy()
    if df_seg.empty: return [], []
    col_td = 'mean_td' if 'mean_td' in df.columns else 'mean_tdc'
    curve = (df_seg['mean_asd'].values + df_seg[col_td].values) / 2.0
    global_threshold = np.mean(curve)
    sig_mask = df_seg['sig_raw']
    sig_trs_global = df_seg.loc[sig_mask, 'timepoint'].values
    sig_trs_rel = sorted([t - tr_start for t in sig_trs_global])
    clusters = []
    for k, g in groupby(enumerate(sig_trs_rel), lambda ix: ix[0] - ix[1]):
        clusters.append(list(map(itemgetter(1), g)))
    high_trs, low_trs = [], []
    for clus in clusters:
        valid_idx = [t for t in clus if t < len(curve)]
        if not valid_idx: continue
        if np.mean(curve[valid_idx]) >= global_threshold: high_trs.extend(valid_idx)
        else: low_trs.extend(valid_idx)
    return sorted(list(set(high_trs))), sorted(list(set(low_trs)))

def get_top_rois_constrained(root_dir, percent, tr_indices, label_val=0):
    fold_means = []
    for fold in range(5):
        p_ig = os.path.join(root_dir, f"fold{fold}", "ig_roi_milout.npy")
        p_lbl = os.path.join(root_dir, f"fold{fold}", "labels.npy")
        if os.path.exists(p_ig) and os.path.exists(p_lbl):
            ig = np.load(p_ig); lbl = np.load(p_lbl)
            ig_group = ig[lbl == label_val]
            if len(ig_group) == 0: continue
            valid_t = [t for t in tr_indices if t < ig_group.shape[1]]
            if not valid_t: continue
            sub_median = np.median(ig_group[:, valid_t, :], axis=1)
            fold_mean = np.mean(np.abs(sub_median), axis=0)
            fold_means.append(fold_mean)
    if not fold_means: return []
    grand_mean = np.mean(np.stack(fold_means), axis=0)
    top_k = int(len(grand_mean) * percent)
    return np.argsort(grand_mean)[-top_k:][::-1]

def load_fmri_data(path, tr_start, tr_end):
    if not os.path.exists(path): return None, None
    datao = pd.read_pickle(path)
    datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
    shapes = [np.asarray(d).shape for d in datao.data]
    main_shape = max(set(shapes), key=shapes.count)
    datao = datao.iloc[[i for i, s in enumerate(shapes) if s == main_shape]].reset_index(drop=True)
    fmri = np.stack([np.asarray(d) for d in datao.data])
    if tr_start > 0 or (tr_end < 9999 and fmri.shape[1] >= tr_end):
        end = min(tr_end, fmri.shape[1]); fmri = fmri[:, tr_start:end, :]
    datao['label'] = datao['label'].str.lower().map({'asd': 0, 'td': 1, 'tdc': 1})
    datao['subject_id'] = datao['subject_id'].astype(str).str.strip().str.lower()
    return fmri, datao

def main():
    configs_to_run = [c for c in CONFIGS if (c["name"] == "MovieDM_Seg2" if TEST_MODE else True)]
    for conf in configs_to_run:
        print(f"Processing {conf['name']}...")
        high_trs, low_trs = get_clusters(conf['attn_csv'], conf['tr_start'], conf['tr_end'])
        fmri, meta = load_fmri_data(conf['fmri_file'], conf['tr_start'], conf['tr_end'])
        if fmri is None: continue
        meta_asd = meta[meta['label'] == 0].reset_index(drop=True)
        fmri_asd = fmri[meta['label'] == 0]
        behav = pd.read_csv(conf['behavior_file'], low_memory=False)
        behav['Identifiers'] = behav['Identifiers'].astype(str).str.strip().str.lower()
        
        for pct in THRESHOLDS:
            for subset_name, trs in [("HighCluster", high_trs), ("LowCluster", low_trs), ("AllClusters", sorted(list(set(high_trs+low_trs))))]:
                if not trs: continue
                top_rois = get_top_rois_constrained(conf['ig_root'], pct, trs)
                if len(top_rois) == 0: continue
                roi_activity = np.mean(fmri_asd[:, [t for t in trs if t < fmri_asd.shape[1]], :][:, :, top_rois], axis=(1, 2))
                
                results = []
                for col in behav.columns:
                    if col == 'Identifiers': continue
                    df_final = pd.DataFrame({'id': meta_asd['subject_id'], 'roi': roi_activity}).merge(behav[['Identifiers', col]].dropna(), left_on='id', right_on='Identifiers', how='inner')
                    m = meta_asd.copy()
                    m['age'] = pd.to_numeric(m['age'], errors='coerce')
                    m['gender'] = m['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
                    m['site'] = m['site'].astype(str).str.strip().str.lower().map({'ru': 0, 'cuny': 1, 'cbic': 2})
                    df_final = df_final.merge(m[['subject_id', 'age', 'gender', 'site']], left_on='id', right_on='subject_id', how='inner').dropna()
                    if len(df_final) < (5 if TEST_MODE else 15): continue
                    
                    covars = df_final[['age', 'gender', 'site']].values
                    def get_resid(y, X): return y - LinearRegression().fit(X, y).predict(X)
                    par_r, par_p = pearsonr(get_resid(df_final['roi'].values, covars), get_resid(pd.to_numeric(df_final[col], errors='coerce').values, covars))
                    results.append({'scale': col, 'partial_r': par_r, 'partial_p': par_p})
                
                res_df = pd.DataFrame(results).sort_values('partial_p')
                res_df.to_csv(os.path.join(OUT_DIR, f"{conf['name']}_{subset_name}_Top{int(pct*100)}pct_results.csv"), index=False)

if __name__ == "__main__":
    main()
