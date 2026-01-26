"""
Step 6-2: MIL Path Brain-Behavior Association

This script performs partial Pearson correlation analysis between event-specific 
Neural Saliency Indices (from the MIL path) and clinical behavior scores, 
controlling for Age, Sex, and Site.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

# === Configuration & Test Mode ===
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TARGET_BEHAVIORS = ["CBCL,CBCL_Total_T", "RBS,RBS_Total", "SRS,SRS_Total_T"]

def main():
    base_dir = "./"
    roi_file = os.path.join(base_dir, "top_roi_summary/mil_asd_top_roi.csv")
    fmri_file = os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz") if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz"
    behavior_file = os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv"
    attn_sig_file = os.path.join(base_dir, "figures/attn_significance.csv")
    ig_sig_file = os.path.join(base_dir, "ig_group_level_results/ig_curve_group_mil_topattn.csv")

    if not all(os.path.exists(f) for f in [roi_file, fmri_file, behavior_file, attn_sig_file, ig_sig_file]):
        print("Missing required dependency files. Skipping.")
        return

    # Data Loading
    roi_df = pd.read_csv(roi_file)
    datao = pd.read_pickle(fmri_file)
    behavior_df = pd.read_csv(behavior_file)
    attn_df = pd.read_csv(attn_sig_file)
    ig_df = pd.read_csv(ig_sig_file)

    # QC and cleaning
    datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
    shapes = [np.asarray(d).shape for d in datao.data]
    main_shape = max(set(shapes), key=shapes.count)
    datao = datao.iloc[[i for i, s in enumerate(shapes) if s == main_shape]].reset_index(drop=True)
    fmri = np.stack([np.asarray(d) for d in datao.data])
    nan_subs = np.unique(np.argwhere(np.isnan(fmri))[:, 0])
    datao = datao.drop(index=nan_subs).reset_index(drop=True)
    fmri = np.stack([np.asarray(d) for d in datao.data])

    # ID Normalization
    datao['label'] = datao['label'].str.lower().map({'asd': 0, 'td': 1, 'tdc': 1})
    datao['subject_id'] = datao['subject_id'].astype(str).str.strip().str.lower()
    behavior_df['Identifiers'] = behavior_df['Identifiers'].astype(str).str.strip().str.lower()

    # ASD sub-group
    asd_mask = datao['label'] == 0
    fmri_asd = fmri[asd_mask.values]
    asd_ids = datao.loc[asd_mask, 'subject_id'].values

    # Determine Top ROIs and Intersecting Timepoints
    roi_col = 'roi_index' if 'roi_index' in roi_df.columns else ('Unnamed: 0' if 'Unnamed: 0' in roi_df.columns else roi_df.columns[0])
    top_roi_indices = roi_df.sort_values('count', ascending=False).head(10)[roi_col].astype(int).values
    
    attn_timepoints = attn_df.loc[attn_df['sig_raw'], 'timepoint'].values
    ig_timepoints = ig_df.loc[ig_df['is_top_asd'], 'timepoint'].values
    intersect_timepoints = np.intersect1d(attn_timepoints, ig_timepoints)

    if len(intersect_timepoints) < 1:
        print(" No valid intersecting timepoints found.")
        if TEST_MODE: return
        exit()

    mil_subset = fmri_asd[:, intersect_timepoints, :][:, :, top_roi_indices]
    mil_avg_activation_all = mil_subset.mean(axis=(1, 2))

    plot_dir = "milpath_partial_plots_intersection"
    os.makedirs(plot_dir, exist_ok=True)
    results = []
    
    cols_to_test = behavior_df.columns if not TEST_MODE else [c for c in behavior_df.columns if any(t in c for t in TARGET_BEHAVIORS)]

    for col in cols_to_test:
        if col == 'Identifiers': continue
        try:
            values_all = pd.to_numeric(behavior_df[col], errors='coerce')
            if values_all.notna().sum() < (5 if TEST_MODE else 10): continue

            common_ids = np.intersect1d(asd_ids, behavior_df.loc[values_all.notna(), 'Identifiers'].values)
            if len(common_ids) < (5 if TEST_MODE else 10): continue

            fmri_idx = [np.where(asd_ids == sid)[0][0] for sid in common_ids]
            behav_idx = [behavior_df[behavior_df['Identifiers'] == sid].index[0] for sid in common_ids]

            meta_df = datao[asd_mask].copy().reset_index(drop=True).loc[fmri_idx].reset_index(drop=True)
            meta_df['age'] = pd.to_numeric(meta_df['age'], errors='coerce')
            meta_df['gender'] = meta_df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
            meta_df['site'] = meta_df['site'].astype(str).str.strip().str.lower().map({'ru': 0, 'cuny': 1, 'cbic': 2})

            combined_df = pd.DataFrame({
                'roi': mil_avg_activation_all[fmri_idx],
                'behavior': pd.to_numeric(behavior_df.loc[behav_idx, col], errors='coerce').values,
                'age': meta_df['age'], 'gender': meta_df['gender'], 'site': meta_df['site']
            }).dropna()

            if len(combined_df) < (5 if TEST_MODE else 10): continue

            X_cov = combined_df[['age', 'gender', 'site']].values
            def get_resid(y, X): return y - LinearRegression().fit(X, y).predict(X)
            
            roi_res = get_resid(combined_df['roi'].values, X_cov)
            beh_res = get_resid(combined_df['behavior'].values, X_cov)
            r, p = pearsonr(roi_res, beh_res)
            results.append({'scale': col, 'partial_r': r, 'partial_p': p, 'n': len(combined_df)})

        except Exception as e:
            print(f"Skipped {col}: {e}")

    results_df = pd.DataFrame(results).sort_values(by='partial_p').reset_index(drop=True)
    results_df.to_csv("milpath_partial_correlation_intersection.csv", index=False)
    print(f" Analysis completed. Results saved to 'milpath_partial_correlation_intersection.csv'")

if __name__ == "__main__":
    main()
