"""
Step 3-3: Extract Top ROIs from IG Attribution Maps

This script identifies the most influential brain regions (ROIs) according 
to the Integrated Gradients attribution. 
Analysis flow:
1. Loads fold-wise IG maps for each subject and group.
2. Computes the group-level median absolute attribution per ROI.
3. Selects the Top 5% of ROIs based on their overall contribution.
4. Calculates normalized weights and frequency counts for each ROI.
5. Saves the resulting Top-ROI lists to CSV files for post-hoc analysis.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

def compute_roi_attribution_weights(attr_median):
    """Normalize ROI importance scores to sum to 1."""
    # Mean across timepoints to get spatial importance vector
    roi_importance = np.mean(np.abs(attr_median), axis=0)
    roi_weights = roi_importance / (np.sum(roi_importance) + 1e-12)
    return roi_weights

def load_foldwise_ig(group: str):
    """Compute the group-level median attribution map across CV folds."""
    group_ig_per_fold = []
    # Identify target group label and which path (Global Mean vs MIL) to load
    target_label = 0 if 'asd' in group else 1
    ig_key = 'ig_roi_outmean.npy' if 'outmean' in group else 'ig_roi_milout.npy'

    for fold in range(5):
        fold_dir = f'./saved_models/fold{fold}'
        ig_path = os.path.join(fold_dir, ig_key)
        label_path = os.path.join(fold_dir, 'labels.npy')

        if os.path.exists(ig_path) and os.path.exists(label_path):
            ig_data = np.load(ig_path)  # (N, T, R)
            labels = np.load(label_path)
            group_data = ig_data[labels == target_label]
            if len(group_data) > 0:
                # Mean across subjects in this fold
                mean_attr = np.mean(np.abs(group_data), axis=0)
                group_ig_per_fold.append(mean_attr)

    if not group_ig_per_fold:
        return np.zeros((250, 246))
    # Final stable map is the median of group maps across folds
    return np.median(np.stack(group_ig_per_fold), axis=0)

def extract_top_roi(attr_median, top_percent=5):
    """Extract ROIs above the specified percentile threshold."""
    flat = attr_median.flatten()
    threshold = np.percentile(flat, 100 - top_percent)
    
    # Identify coordinates (Time, ROI) above threshold
    top_mask = attr_median >= threshold
    top_coords = np.argwhere(top_mask)
    top_rois = top_coords[:, 1]
    
    # Count how many timepoints each ROI remained in the Top-K
    roi_counts = pd.Series(top_rois).value_counts().sort_values(ascending=False)

    # Compute normalized weights
    roi_weights = compute_roi_attribution_weights(attr_median)
    roi_df = roi_counts.reset_index()
    roi_df.columns = ['roi_index', 'count']
    # Add weight metadata
    roi_df['weight'] = roi_df['roi_index'].apply(lambda x: roi_weights[x])
    return roi_df

def main():
    # Setup output directory
    top_roi_dir = "top_roi_summary"
    os.makedirs(top_roi_dir, exist_ok=True)
    
    # Process both paths and both groups
    groups = ['outmean_asd', 'outmean_tdc', 'mil_asd', 'mil_tdc']

    for group in groups:
        print(f"Processing Top-ROIs for group: {group}...")
        # Get stable group map
        attr_median = load_foldwise_ig(group)
        # Select top regions
        top_rois_df = extract_top_roi(attr_median, top_percent=5)
        # Export result
        out_path = os.path.join(top_roi_dir, f"{group}_top_roi.csv")
        top_rois_df.to_csv(out_path, index=False)
        print(f"Top 10 ROIs for {group}:")
        print(top_rois_df.head(10))

if __name__ == "__main__":
    main()
