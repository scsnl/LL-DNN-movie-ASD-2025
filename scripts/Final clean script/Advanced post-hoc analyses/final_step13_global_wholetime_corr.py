"""
Final Step 13 (Global Mean Path): Whole-Time Correlation Analysis

This script explores the association between clinical traits and the model's 
global attention map.
Methodology:
1. Identifies the Top-K most important brain regions based on Integrated Gradients (IG) 
   calculated across the entire movie duration.
2. Extracts subject-level mean fMRI activation from these Top-ROIs.
3. Computes partial Pearson correlations with clinical behavioral scores (SRS, RBS, CBCL).
4. Controls for Age, Sex, and Scanning Site as covariates.
5. Generates high-resolution scatter plots with regression lines and 95% confidence intervals.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Path and execution mode configuration
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")

OUT_DIR = "correlation_results_global_wholetime_final"
os.makedirs(OUT_DIR, exist_ok=True)

# Configuration
CONFIGS = [
    {
        "name": "MovieDM_Seg2",
        "fmri_file": os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz") if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz",
        "behavior_file": os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv",
        "ig_root": os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models",
        "tr_start": 0,
        "tr_end": 9999
    }
]

THRESHOLDS = [0.05, 0.10, 0.20, 0.30]

def get_top_rois_wholetime(root_dir, percent, label_val=0):
    fold_means = []
    for fold in range(5):
        p_ig = os.path.join(root_dir, f"fold{fold}", "ig_roi_outmean.npy")
        p_lbl = os.path.join(root_dir, f"fold{fold}", "labels.npy")
        if os.path.exists(p_ig) and os.path.exists(p_lbl):
            ig = np.load(p_ig); lbl = np.load(p_lbl)
            ig_group = ig[lbl == label_val]
            if len(ig_group) == 0: continue
            sub_median = np.median(ig_group, axis=1)
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
        end = min(tr_end, fmri.shape[1])
        fmri = fmri[:, tr_start:end, :]
    datao['label'] = datao['label'].str.lower().map({'asd': 0, 'td': 1, 'tdc': 1})
    datao['subject_id'] = datao['subject_id'].astype(str).str.strip().str.lower()
    return fmri, datao

def get_resid(target, X):
    try: return target - LinearRegression().fit(X, target).predict(X)
    except: return target

def plot_correlation_custom(df, x_col, y_col, partial_r, partial_p, title_prefix, out_path):
    plt.figure(figsize=(6, 5), dpi=300)
    sns.set_style("ticks")
    ax = sns.regplot(x=x_col, y=y_col, data=df,
                     scatter_kws={'s': 50, 'edgecolor': 'black', 'linewidths': 0.5, 'alpha': 0.7, 'color': '#1f77b4'},
                     line_kws={'color': '#003366'})
    textstr = f'Partial r = {partial_r:.2f}\np = {partial_p:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    plt.xlabel("Mean Activation", fontsize=14, fontweight='bold')
    plt.ylabel(y_col.replace("_", " "), fontsize=14, fontweight='bold')
    plt.title(f"{title_prefix} | {y_col}", fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def main():
    configs_to_run = [c for c in CONFIGS if (c["name"] == "MovieDM_Seg2" if TEST_MODE else True)]
    for conf in configs_to_run:
        print(f"Processing {conf['name']}...")
        fmri, meta = load_fmri_data(conf['fmri_file'], conf['tr_start'], conf['tr_end'])
        if fmri is None: continue
        asd_mask = meta['label'] == 0
        fmri_asd = fmri[asd_mask]; meta_asd = meta[asd_mask].reset_index(drop=True)
        behav = pd.read_csv(conf['behavior_file'], low_memory=False)
        behav['Identifiers'] = behav['Identifiers'].astype(str).str.strip().str.lower()
        
        for pct in THRESHOLDS:
            top_rois = get_top_rois_wholetime(conf['ig_root'], pct)
            if len(top_rois) == 0: continue
            roi_activity = np.mean(fmri_asd[:, :, top_rois], axis=(1, 2))
            
            results = []
            for col in behav.columns:
                if col == 'Identifiers': continue
                df_merged = pd.DataFrame({'id': meta_asd['subject_id'], 'roi': roi_activity}).merge(behav[['Identifiers', col]].dropna(), left_on='id', right_on='Identifiers', how='inner')
                m = meta_asd.copy()
                m['age'] = pd.to_numeric(m['age'], errors='coerce')
                m['gender'] = m['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
                m['site'] = m['site'].astype(str).str.strip().str.lower().map({'ru': 0, 'cuny': 1, 'cbic': 2})
                df_final = df_merged.merge(m[['subject_id', 'age', 'gender', 'site']], left_on='id', right_on='subject_id', how='inner').dropna()
                if len(df_final) < (5 if TEST_MODE else 15): continue
                
                covars = df_final[['age', 'gender', 'site']].values
                x_resid = get_resid(df_final['roi'].values, covars)
                y_resid = get_resid(pd.to_numeric(df_final[col], errors='coerce').values, covars)
                par_r, par_p = pearsonr(x_resid, y_resid)
                results.append({'scale': col, 'partial_r': par_r, 'partial_p': par_p})
                
            res_df = pd.DataFrame(results).sort_values('partial_p')
            res_df.to_csv(os.path.join(OUT_DIR, f"{conf['name']}_Top{int(pct*100)}pct_results.csv"), index=False)

if __name__ == "__main__":
    main()
