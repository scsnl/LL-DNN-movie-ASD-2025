#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Extract Top ROI from IG Attribution with Frequency Count and Normalized Weight
@author: leili
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from nilearn import plotting, surface, datasets
from nilearn.image import new_img_like

# ========== Compute ROI Attribution Weight ==========

def compute_roi_attribution_weights(attr_median):
    roi_importance = np.mean(np.abs(attr_median), axis=0)  # shape: (246,)

    roi_weights = roi_importance / np.sum(roi_importance)  # normalize to sum = 1

    return roi_weights

# ========== Load Fold-wise IG Attribution ==========

def load_foldwise_ig(group: str):
    group_ig_per_fold = []
    target_label = 0 if 'asd' in group else 1
    ig_key = 'ig_roi_outmean.npy' if 'outmean' in group else 'ig_roi_milout.npy'

    for fold in range(5):
        fold_dir = f'./saved_models/fold{fold}'
        ig_path = os.path.join(fold_dir, ig_key)
        label_path = os.path.join(fold_dir, 'labels.npy')

        if os.path.exists(ig_path) and os.path.exists(label_path):
            ig_data = np.load(ig_path)  # shape: (N, T, R)

            labels = np.load(label_path)
            group_data = ig_data[labels == target_label]
            mean_attr = np.mean(np.abs(group_data), axis=0)  # shape: (T, R)

            group_ig_per_fold.append(mean_attr)
        else:
            print(f"Skipping fold {fold}: missing files.")

    return np.median(np.stack(group_ig_per_fold), axis=0) if group_ig_per_fold else np.zeros((250, 246))

# ========== Extract Top ROI (index only) ==========

def extract_top_roi(attr_median, top_percent=5):
    T, R = attr_median.shape
    flat = attr_median.flatten()
    threshold = np.percentile(flat, 100 - top_percent)
    top_mask = attr_median >= threshold
    top_coords = np.argwhere(top_mask)
    top_rois = top_coords[:, 1]
    roi_counts = pd.Series(top_rois).value_counts().sort_values(ascending=False)

    roi_weights = compute_roi_attribution_weights(attr_median)

    roi_df = roi_counts.reset_index()
    roi_df.columns = ['roi_index', 'count']
    roi_df['weight'] = roi_df['roi_index'].apply(lambda x: roi_weights[x])
    return roi_df

# ========== Main Execution ==========

top_roi_dir = "top_roi_summary"
os.makedirs(top_roi_dir, exist_ok=True)
groups = ['outmean_asd', 'outmean_tdc', 'mil_asd', 'mil_tdc']

for group in groups:
    print(f"\nProcessing {group} ...")
    attr_median = load_foldwise_ig(group)
    top_rois_df = extract_top_roi(attr_median, top_percent=5)
    top_rois_df.to_csv(f"# TODO: specify your data path", index=False)

    print(top_rois_df.head(10))
