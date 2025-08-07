import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

# Path to the combined dataset
combined_data_path = '# TODO: specify your combined dataset path (.pklz)'

# Load combined dataset
datao = pd.read_pickle(combined_data_path)
print("Original shape:", datao.shape)

# Step 1: Drop bad quality samples
datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
print("After motion/repair filtering:", datao.shape)

# Step 2: Drop inconsistent timepoints (non-majority shape)
shapes = [np.asarray(d).shape for d in datao.data]
main_shape = max(set(shapes), key=shapes.count)
valid_indices = [i for i, s in enumerate(shapes) if s == main_shape]
datao = datao.iloc[valid_indices].reset_index(drop=True)
print("After shape filtering:", datao.shape)

# Step 3: Drop subjects with NaNs in fMRI data
fmri_data = np.stack([np.asarray(d) for d in datao.data])  # (N, T, R)
nan_subjects = np.unique(np.argwhere(np.isnan(fmri_data))[:, 0])
valid_indices = [i for i in range(fmri_data.shape[0]) if i not in nan_subjects]
datao = datao.iloc[valid_indices].reset_index(drop=True)
print("After NaN filtering:", datao.shape)

# Prepare data and labels
data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
labels = datao['label'].apply(lambda x: 0 if x == 'asd' else 1).values  # 'asd' -> 0, 'td' -> 1
subjid = datao['subject_id']

# Generate 10 different 5-fold splits
for i in range(10):
    output_folder = f'# TODO: specify your output directory for CV indices/cv_dataset_{i}/'
    output_f_trainlist_index = os.path.join(output_folder, 'train_list_index.npy')
    output_f_testlist_index = os.path.join(output_folder, 'test_list_index.npy')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    kf = StratifiedKFold(n_splits=5, random_state=i, shuffle=True)
    train_index_list = []
    test_index_list = []

    for train_idx, test_idx in kf.split(subjid, labels):
        train_index_list.append(train_idx)
        test_index_list.append(test_idx)

    np.save(output_f_trainlist_index, np.array(train_index_list, dtype=object))
    np.save(output_f_testlist_index, np.array(test_index_list, dtype=object))

    print(f"Split {i} saved in {output_folder}")
