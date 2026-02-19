"""
Step 1: Generate Nested CV indices (outer 5-fold + inner K-fold).
Stratified by labels. Subject-level safe: StratifiedGroupKFold ensures no subject
appears in both train and test.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")

combined_data_path = (
    os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz")
    if TEST_DATA_DIR
    else "# TODO: combined dataset path (.pklz)"
)
output_npz_path = (
    os.path.join(TEST_OUTPUT_DIR, "cv_indices", "nested_indices_seed42.npz")
    if TEST_OUTPUT_DIR
    else "# TODO: output path/nested_indices_seed42.npz"
)

OUTER_SPLITS = 5
INNER_SPLITS = 4
SEED = 42

datao = pd.read_pickle(combined_data_path)
print("Original shape:", datao.shape)

datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
print("After motion/repair filtering:", datao.shape)

shapes = [np.asarray(d).shape for d in datao.data]
main_shape = max(set(shapes), key=shapes.count)
valid_indices = [i for i, s in enumerate(shapes) if s == main_shape]
datao = datao.iloc[valid_indices].reset_index(drop=True)
print("After shape filtering:", datao.shape)

fmri_data = np.stack([np.asarray(d) for d in datao.data])
nan_subjects = np.unique(np.argwhere(np.isnan(fmri_data))[:, 0])
valid_indices = [i for i in range(fmri_data.shape[0]) if i not in nan_subjects]
datao = datao.iloc[valid_indices].reset_index(drop=True)
print("After NaN filtering:", datao.shape)

labels = datao['label'].apply(lambda x: 0 if x == 'asd' else 1).values
subjid = datao['subject_id'].values
n_samples = len(labels)

try:
    outer_sgkf = StratifiedGroupKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    outer_folds = list(outer_sgkf.split(np.arange(n_samples), labels, groups=subjid))
    print("Using StratifiedGroupKFold (subject-level independence)")
except (ImportError, TypeError):
    outer_skf = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    outer_folds = list(outer_skf.split(np.arange(n_samples), labels))
    print("Using StratifiedKFold (upgrade sklearn>=1.1 for subject-level safety)")

nested_arrays = {
    "y": labels,
    "n_outer": np.array(OUTER_SPLITS, dtype=np.int32),
    "n_inner": np.array(INNER_SPLITS, dtype=np.int32),
}

for i, (outer_train_idx, outer_test_idx) in enumerate(outer_folds):
    nested_arrays[f"outer_train_{i}"] = outer_train_idx
    nested_arrays[f"outer_test_{i}"] = outer_test_idx
    inner_labels = labels[outer_train_idx]
    inner_groups = subjid[outer_train_idx]
    try:
        inner_sgkf = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
        inner_splits_iter = inner_sgkf.split(outer_train_idx, inner_labels, groups=inner_groups)
    except (ImportError, TypeError):
        inner_skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
        inner_splits_iter = inner_skf.split(outer_train_idx, inner_labels)
    for j, (inner_train_global, inner_val_global) in enumerate(inner_splits_iter):
        inner_train_idx = outer_train_idx[inner_train_global]
        inner_val_idx = outer_train_idx[inner_val_global]
        nested_arrays[f"inner_train_{i}_{j}"] = inner_train_idx
        nested_arrays[f"inner_val_{i}_{j}"] = inner_val_idx

os.makedirs(os.path.dirname(output_npz_path) or ".", exist_ok=True)
np.savez_compressed(output_npz_path, **nested_arrays)
print(f"Nested CV indices saved to {output_npz_path}")
