#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Mon May 19 15:32:32 2025

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

# ===== Step 1: Load IG by fold from saved_models/fold{K} =====

attr_out_asd, attr_out_tdc = [], []
attr_mil_asd, attr_mil_tdc = [], []

for fold in range(5):
    fold_dir = f'./saved_models/fold{fold}'
    out_path = os.path.join(fold_dir, 'ig_roi_outmean.npy')
    mil_path = os.path.join(fold_dir, 'ig_roi_milout.npy')
    label_path = os.path.join(fold_dir, 'labels.npy')

    if os.path.exists(out_path) and os.path.exists(mil_path) and os.path.exists(label_path):
        out = np.load(out_path)  # shape: (N_val, T, R)

        mil = np.load(mil_path)
        labels = np.load(label_path)

        attr_out_asd.append(out[labels == 0])
        attr_out_tdc.append(out[labels == 1])
        attr_mil_asd.append(mil[labels == 0])
        attr_mil_tdc.append(mil[labels == 1])
    else:
        print(f"Missing files for fold {fold}, skipping...")

# ===== Step 2: Median across folds (per group) =====

def median_attr(attr_list):
    fold_means = [np.mean(np.abs(a), axis=0) for a in attr_list]  # (T, R) per fold

    return np.median(np.stack(fold_means), axis=0) if fold_means else np.zeros((250, 246))

median_out_asd = median_attr(attr_out_asd)
median_out_tdc = median_attr(attr_out_tdc)
median_mil_asd = median_attr(attr_mil_asd)
median_mil_tdc = median_attr(attr_mil_tdc)

# ===== Step 3: Map to brain volume using BN atlas =====

atlas_path = './BN_Atlas_246_1mm.nii.gz'
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()

def map_to_volume(atlas_data, roi_values):
    vol = np.zeros_like(atlas_data)
    for i in range(246):
        vol[atlas_data == i + 1] = roi_values[i]
    return vol

volume_dict = {
    "outmean_asd": map_to_volume(atlas_data, median_out_asd.mean(axis=0)),
    "outmean_tdc": map_to_volume(atlas_data, median_out_tdc.mean(axis=0)),
    "mil_asd": map_to_volume(atlas_data, median_mil_asd.mean(axis=0)),
    "mil_tdc": map_to_volume(atlas_data, median_mil_tdc.mean(axis=0))
}

# ===== Step 4: Project to fsaverage5 and plot views =====

output_dir = "./surface_figures"
os.makedirs(output_dir, exist_ok=True)
fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
view_angles = ['left_lateral', 'left_medial', 'right_lateral', 'right_medial']

def plot_views(group_name, volume):
    img = new_img_like(atlas_img, volume)
    nii_path = os.path.join(output_dir, f"{group_name}.nii.gz")
    nib.save(img, nii_path)

    tex_l = surface.vol_to_surf(img, fsaverage.pial_left)
    tex_r = surface.vol_to_surf(img, fsaverage.pial_right)

    view_map = {
        "left_lateral": ("left", "lateral", tex_l),
        "left_medial": ("left", "medial", tex_l),
        "right_lateral": ("right", "lateral", tex_r),
        "right_medial": ("right", "medial", tex_r)
    }

    nonzero_vals = np.concatenate([tex_l[tex_l > 0], tex_r[tex_r > 0]])
    vmin = np.min(nonzero_vals) if len(nonzero_vals) > 0 else 1e-6
    vmax = np.percentile(nonzero_vals, 98) if len(nonzero_vals) > 0 else 1.0

    for view_name, (hemi, view, tex) in view_map.items():
        out_file = os.path.join(output_dir, f"{group_name}_{view_name}.png")
        plotting.plot_surf_stat_map(
            surf_mesh=fsaverage.infl_left if hemi == "left" else fsaverage.infl_right,
            stat_map=tex,
            hemi=hemi,
            view=view,
            bg_map=fsaverage.sulc_left if hemi == "left" else fsaverage.sulc_right,
            threshold=vmin,
            cmap="autumn_r",
            colorbar=True,
            output_file=out_file,
            title=f"{group_name.upper()} - {view_name.replace('_', ' ').title()}",
            vmax=vmax,
            symmetric_cbar=False
        )

# ===== Step 5: Generate multi-view panels =====

for group_name, volume in volume_dict.items():
    plot_views(group_name, volume)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(group_name.upper(), fontsize=16)
    view_files = [os.path.join(output_dir, f"{group_name}_{v}.png") for v in view_angles]

    for i, path in enumerate(view_files):
        img = Image.open(path) if os.path.exists(path) else None
        ax = axs[i // 2][i % 2]
        if img:
            ax.imshow(img)
            ax.set_title(view_angles[i].replace('_', ' ').capitalize())
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    panel_path = os.path.join(output_dir, f"{group_name}_panel.png")
    plt.savefig(panel_path, dpi=300)
    plt.close()
    print(f"Saved: {panel_path}")
