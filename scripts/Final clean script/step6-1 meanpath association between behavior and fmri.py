"""
Step 6-1: Global Path Brain-Behavior Association

This script performs partial Pearson correlation analysis between the Global 
Saliency Index (mean fMRI activation across top attribution ROIs) and clinical 
behavior scores, controlling for Age, Sex, and Site.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# === Configuration & Test Mode ===
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TARGET_BEHAVIORS = ["CBCL,CBCL_Total_T", "RBS,RBS_Total", "SRS,SRS_Total_T"]

# === Function: Pearson + Partial Correlation ===
def compute_pearson_and_partial_corr(x, y, covariates):
    pearson_r, pearson_p = pearsonr(x, y)
    def get_residuals(target, X):
        model = LinearRegression().fit(X, target)
        return target - model.predict(X)
    x_resid = get_residuals(x, covariates)
    y_resid = get_residuals(y, covariates)
    partial_r, partial_p = pearsonr(x_resid, y_resid)
    return {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'partial_r': partial_r, 'partial_p': partial_p
    }

def main():
    base_dir = "./"
    roi_dir = "./top_roi_summary/"
    roi_file = os.path.join(roi_dir, "outmean_asd_top_roi.csv")
    fmri_file = os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz") if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz"
    behavior_file = os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv"

    if not os.path.exists(roi_file):
        print(f"ROI file not found: {roi_file}")
        return

    # Data Loading
    roi_df = pd.read_csv(roi_file)
    datao = pd.read_pickle(fmri_file)
    behavior_df = pd.read_csv(behavior_file, low_memory=False)

    # QC and fMRI cleaning
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

    # Filter ASD subgroup
    asd_mask = datao['label'] == 0
    fmri_asd = fmri[asd_mask.values]
    asd_ids = datao.loc[asd_mask, 'subject_id'].values

    # Identify Top 10 ROIs from IG attribution
    roi_col = 'roi_index' if 'roi_index' in roi_df.columns else ('Unnamed: 0' if 'Unnamed: 0' in roi_df.columns else roi_df.columns[0])
    top_roi_indices = roi_df.sort_values('count', ascending=False).head(10)[roi_col].astype(int).values

    results = []
    cols_to_test = behavior_df.columns if not TEST_MODE else [c for c in behavior_df.columns if any(t in c for t in TARGET_BEHAVIORS)]

    for col in cols_to_test:
        if col == 'Identifiers': continue
        try:
            values_all = pd.to_numeric(behavior_df[col], errors='coerce')
            if values_all.notna().sum() < (5 if TEST_MODE else 10): continue

            valid_behavior_ids = behavior_df.loc[values_all.notna(), 'Identifiers'].values
            common_ids = np.intersect1d(asd_ids, valid_behavior_ids)
            if len(common_ids) < (5 if TEST_MODE else 10): continue

            fmri_idx = [np.where(asd_ids == sid)[0][0] for sid in common_ids]
            behav_idx = [behavior_df[behavior_df['Identifiers'] == sid].index[0] for sid in common_ids]

            fmri_subset = fmri_asd[fmri_idx][:, :, top_roi_indices]
            roi_avg_activation = fmri_subset.mean(axis=(1, 2))
            
            # Metadata for covariates
            meta_df = datao[asd_mask].copy().reset_index(drop=True).loc[fmri_idx].reset_index(drop=True)
            meta_df['age'] = pd.to_numeric(meta_df['age'], errors='coerce')
            meta_df['gender'] = meta_df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
            meta_df['site'] = meta_df['site'].astype(str).str.strip().str.lower().map({'ru': 0, 'cuny': 1, 'cbic': 2})

            combined_df = pd.DataFrame({
                'roi': roi_avg_activation,
                'behavior': pd.to_numeric(behavior_df.loc[behav_idx, col], errors='coerce').values,
                'age': meta_df['age'], 'gender': meta_df['gender'], 'site': meta_df['site']
            }).dropna()

            if len(combined_df) < (5 if TEST_MODE else 10): continue

            res = compute_pearson_and_partial_corr(combined_df['roi'].values, combined_df['behavior'].values, combined_df[['age', 'gender', 'site']].values)
            results.append({'scale': col, 'partial_r': res['partial_r'], 'partial_p': res['partial_p'], 'n': len(combined_df)})

        except Exception as e:
            print(f"Error processing {col}: {e}")

    # Output results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='partial_p').reset_index(drop=True)
        results_df.to_csv("meanpath_correlation_results_top10roi.csv", index=False)
        print(" Analysis completed. Results saved to 'meanpath_correlation_results_top10roi.csv'")
    else:
        print(" No valid correlations found.")

if __name__ == "__main__":
    main()
