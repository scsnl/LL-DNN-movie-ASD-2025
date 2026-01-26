"""
Step 1: Generate Cross-Validation Indices

This script prepares the dataset for model training and evaluation. It performs
the following steps:
1. Loads the combined fMRI dataset (ROI time series).
2. Applies quality control filters (motion and repair thresholds).
3. Ensures consistent temporal dimensions across subjects.
4. Removes subjects containing NaN values in signals.
5. Generates multiple sets of stratified 5-fold cross-validation indices to 
   ensure robust performance estimation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Environment variables for test/dummy runs
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")

# Define path to the input dataset
combined_data_path = (
    os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz")
    if TEST_DATA_DIR
    else "combined_asd_td_movieDM_data.pklz"
)

# Load the combined dataset (pickle format)
datao = pd.read_pickle(combined_data_path)
print(f"Original dataset shape: {datao.shape}")

# Quality Control: Filter out subjects with >10% repaired volumes or high motion (mean FD > 0.5)
datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
print(f"Shape after QC filtering: {datao.shape}")

# Consistency Check: Keep only subjects matching the majority temporal shape
shapes = [np.asarray(d).shape for d in datao.data]
main_shape = max(set(shapes), key=shapes.count)
valid_indices = [i for i, s in enumerate(shapes) if s == main_shape]
datao = datao.iloc[valid_indices].reset_index(drop=True)
print(f"Shape after temporal consistency filtering: {datao.shape}")

# Clean Data: Drop subjects with NaN values in their fMRI time series
fmri_data = np.stack([np.asarray(d) for d in datao.data])
nan_subjects = np.unique(np.argwhere(np.isnan(fmri_data))[:, 0])
valid_indices = [i for i in range(fmri_data.shape[0]) if i not in nan_subjects]
datao = datao.iloc[valid_indices].reset_index(drop=True)
print(f"Final shape after NaN removal: {datao.shape}")

# Prepare features and labels for CV split
# labels: 'asd' -> 0, 'td' -> 1
data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
labels = datao['label'].apply(lambda x: 0 if x == 'asd' else 1).values
subjid = datao['subject_id']

# Generate repeated cross-validation splits
# In TEST_MODE, we run only 1 iteration to save time
n_repeats = 1 if TEST_MODE else 10
for i in range(n_repeats):
    base_out = TEST_OUTPUT_DIR if TEST_OUTPUT_DIR else "."
    output_folder = os.path.join(base_out, "cv_indices", f"cv_dataset_{i}")
    output_f_trainlist_index = os.path.join(output_folder, 'train_list_index.npy')
    output_f_testlist_index = os.path.join(output_folder, 'test_list_index.npy')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use StratifiedKFold to maintain ASD/TD ratio in each fold
    kf = StratifiedKFold(n_splits=5, random_state=i, shuffle=True)
    train_index_list = []
    test_index_list = []

    for train_idx, test_idx in kf.split(subjid, labels):
        train_index_list.append(train_idx)
        test_index_list.append(test_idx)

    # Save indices as numpy arrays for training scripts
    np.save(output_f_trainlist_index, np.array(train_index_list, dtype=object))
    np.save(output_f_testlist_index, np.array(test_index_list, dtype=object))

    print(f"Split {i} indices saved to {output_folder}")
