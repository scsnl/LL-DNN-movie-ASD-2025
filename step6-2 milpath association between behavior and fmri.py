#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Wed May 21 14:34:35 2025
Updated for partial correlation + plotting
@author: leili
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

# === Step 0:  ===

base_dir = "./"
roi_file = os.path.join(base_dir, "# TODO: specify your data path")

fmri_file = os.path.join(base_dir, "combined_asd_td_movieDM_data.pklz")
behavior_file = os.path.join(base_dir, "all subject behavior data.csv")
attn_sig_file = os.path.join(base_dir, "# TODO: specify your data path")

ig_sig_file = os.path.join(base_dir, "# TODO: specify your data path")


# === Step 1:  ===

roi_df = pd.read_csv(roi_file)
datao = pd.read_pickle(fmri_file)
behavior_df = pd.read_csv(behavior_file)
attn_df = pd.read_csv(attn_sig_file)
ig_df = pd.read_csv(ig_sig_file)

# === Step 2: fMRI  ASD  ===

datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
shapes = [np.asarray(d).shape for d in datao.data]
main_shape = max(set(shapes), key=shapes.count)
datao = datao.iloc[[i for i, s in enumerate(shapes) if s == main_shape]].reset_index(drop=True)
fmri = np.stack([np.asarray(d) for d in datao.data])
nan_subs = np.unique(np.argwhere(np.isnan(fmri))[:, 0])
datao = datao.drop(index=nan_subs).reset_index(drop=True)
fmri = np.stack([np.asarray(d) for d in datao.data])

datao['label'] = datao['label'].str.lower().map({'asd': 0, 'td': 1})
datao['subject_id'] = datao['subject_id'].astype(str).str.strip().str.lower()
behavior_df['Identifiers'] = behavior_df['Identifiers'].astype(str).str.strip().str.lower()

asd_mask = datao['label'] == 0
fmri_asd = fmri[asd_mask.values]
asd_ids = datao.loc[asd_mask, 'subject_id'].values

# === Step 3:  ROI  ===

top_roi_indices = roi_df.iloc[:10, 0].astype(int).values
attn_timepoints = attn_df.loc[attn_df['sig_raw'], 'timepoint'].values
ig_timepoints = ig_df.loc[ig_df['is_top_asd'], 'timepoint'].values
intersect_timepoints = np.intersect1d(attn_timepoints, ig_timepoints)

if len(intersect_timepoints) < 1:
    print(" No valid intersecting timepoints found.")
    exit()

mil_subset = fmri_asd[:, intersect_timepoints, :][:, :, top_roi_indices]
mil_avg_activation = mil_subset.mean(axis=(1, 2))  # shape: (N,)


# === Step 4:  age, gender, site+  ===

plot_dir = "milpath_partial_plots_intersection"
os.makedirs(plot_dir, exist_ok=True)
results = []

for col in behavior_df.columns:
    if col == 'Identifiers':
        continue
    try:
        values_all = pd.to_numeric(behavior_df[col], errors='coerce')
        if values_all.notna().sum() < 5:
            continue

        valid_behavior_mask = values_all.notna()
        valid_behavior_ids = behavior_df.loc[valid_behavior_mask, 'Identifiers'].values
        common_ids = np.intersect1d(asd_ids, valid_behavior_ids)
        if len(common_ids) < 10:
            continue

        fmri_idx = [np.where(asd_ids == sid)[0][0] for sid in common_ids]
        behav_idx = [behavior_df[behavior_df['Identifiers'] == sid].index[0] for sid in common_ids]

        meta_df = datao[asd_mask].copy().reset_index(drop=True).loc[fmri_idx].reset_index(drop=True)
        meta_df['age'] = pd.to_numeric(meta_df['age'], errors='coerce')
        meta_df['gender'] = meta_df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
        meta_df['site'] = meta_df['site'].astype(str).str.strip().str.lower().map({'ru': 0, 'cuny': 1, 'cbic': 2})

        combined_df = pd.DataFrame({
            'roi': mil_avg_activation[fmri_idx],
            'behavior': behavior_df.loc[behav_idx, col].values,
            'age': meta_df['age'],
            'gender': meta_df['gender'],
            'site': meta_df['site']
        }).dropna()

        if len(combined_df) < 10:
            continue

        # ===  ===

        def get_residuals(y, X):
            return y - LinearRegression().fit(X, y).predict(X)

        X_cov = combined_df[['age', 'gender', 'site']].values
        roi_resid = get_residuals(combined_df['roi'].values, X_cov)
        behav_resid = get_residuals(pd.to_numeric(combined_df['behavior']), X_cov)
        r, p = pearsonr(roi_resid, behav_resid)

        results.append({'scale': col, 'partial_r': r, 'partial_p': p, 'n': len(combined_df)})

        # ===  ===

        if p < 0.05:
            x = combined_df['roi'].values
            y = pd.to_numeric(combined_df['behavior'])

            # 

            slope, intercept, *_ = stats.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = slope * x_fit + intercept
            y_pred = slope * x + intercept
            residuals = y - y_pred
            dof = len(x) - 2
            t_val = stats.t.ppf(0.975, dof)
            s_err = np.sqrt(np.sum(residuals**2) / dof)
            ci = t_val * s_err * np.sqrt(1/len(x) + (x_fit - x.mean())**2 / np.sum((x - x.mean())**2))

            # 

            plt.figure(figsize=(6, 4), dpi=300)
            plt.scatter(x, y, color='#1f77b4', s=40, alpha=0.75, edgecolor='k', linewidth=0.5)

            plt.plot(x_fit, y_fit, color='#0e3e66', linewidth=2)

            plt.fill_between(x_fit, y_fit - ci, y_fit + ci, color='#0e3e66', alpha=0.15)


            plt.text(0.05, 0.95, f"Partial r = {r:.2f}\np = {p:.3g}",
                     transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

            plt.xlabel("Mean Activation (Top ROIs @ intersection)", fontsize=11)
            plt.ylabel(col, fontsize=11)
            plt.title(col, fontsize=13, weight='bold')
            plt.grid(alpha=0.3, linestyle='--')

            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            plt.tight_layout()

            save_path = os.path.join(plot_dir, f"{col.replace('/', '_')}.png")
            plt.savefig(save_path)
            plt.close()

    except Exception as e:
        print(f" Skipped {col}: {e}")
        continue

# ===  ===

results_df = pd.DataFrame(results).sort_values(by='partial_p').reset_index(drop=True)
results_df.to_csv("milpath_partial_correlation_intersection.csv", index=False)
print(f" Done. {len(results_df)} scales tested. Significant plots saved to '{plot_dir}'")
print(results_df.head(5))
