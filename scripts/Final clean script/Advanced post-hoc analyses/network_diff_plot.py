"""
Network-Level IG Weight Comparison (ASD vs TDC)

This script maps Brainnetome 246 ROIs to Yeo 7 Networks, computes subject-level 
network importance based on IG, and performs T-tests to compare ASD vs TDC groups.
Covers: MovieDM (Seg2) and MovieTP, both GlobalMean and MIL paths.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from pathlib import Path

# Configuration
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")

MAPPING_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "mapping", "subregion_func_network_Yeo_updated (1).csv")

TASKS = [
    {
        "name": "MovieDM_Seg2",
        "path": os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models"
    }
]

PATHS = {
    "GlobalMean": "ig_roi_outmean.npy",
    "MIL": "ig_roi_milout.npy"
}

OUT_DIR = "network_analysis_figs"; os.makedirs(OUT_DIR, exist_ok=True)

NETWORK_NAMES = ["Visual", "Somatomotor", "Dorsal Attention", "Ventral Attention", "Limbic", "Frontoparietal", "Default"]

def load_roi_mapping(csv_path):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"Mapping file not found: {csv_path}")
    df = pd.read_csv(csv_path, nrows=246)
    if "Label" not in df.columns: df = pd.read_csv(csv_path, skiprows=1, nrows=246)
    mapping = {}
    for _, row in df.iterrows():
        try:
            roi_id, net_id = int(row['Label']), int(row['Yeo_7network'])
            if 1 <= net_id <= 7: mapping[roi_id] = net_id
        except: continue
    return mapping

def load_task_data(task_root, ig_filename):
    asd_list, tdc_list = [], []
    for fold in range(5):
        fold_dir = os.path.join(task_root, f"fold{fold}")
        ig_p, lbl_p = os.path.join(fold_dir, ig_filename), os.path.join(fold_dir, "labels.npy")
        if os.path.exists(ig_p) and os.path.exists(lbl_p):
            ig, labels = np.load(ig_p), np.load(lbl_p)
            ig_imp = np.mean(np.abs(ig), axis=1)
            asd_list.append(ig_imp[labels == 0]); tdc_list.append(ig_imp[labels == 1])
    return (np.concatenate(asd_list, axis=0), np.concatenate(tdc_list, axis=0)) if asd_list else (None, None)

def aggregate_to_networks(X_roi, mapping):
    N = X_roi.shape[0]; X_net = np.zeros((N, 7))
    for net_id in range(1, 8):
        roi_indices = [r-1 for r, n in mapping.items() if n == net_id]
        if roi_indices: X_net[:, net_id-1] = np.mean(X_roi[:, roi_indices], axis=1)
    return X_net

def plot_network_diff(X_asd, X_tdc, title, filename):
    means_asd, means_tdc = np.mean(X_asd, 0), np.mean(X_tdc, 0)
    se_asd, se_tdc = np.std(X_asd, 0) / np.sqrt(X_asd.shape[0]), np.std(X_tdc, 0) / np.sqrt(X_tdc.shape[0])
    p_values = [ttest_ind(X_asd[:, i], X_tdc[:, i], equal_var=False).pvalue for i in range(7)]
    
    fig, ax = plt.subplots(figsize=(12, 6)); x = np.arange(7); width = 0.35
    ax.bar(x - width/2, means_asd, width, yerr=se_asd, label='ASD', capsize=5, color='#fc8d62', alpha=0.8)
    ax.bar(x + width/2, means_tdc, width, yerr=se_tdc, label='TDC', capsize=5, color='#8da0cb', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(NETWORK_NAMES, rotation=30, ha='right'); ax.legend(); ax.set_title(title)
    for i, p in enumerate(p_values):
        if p < 0.05: ax.text(i, max(means_asd[i]+se_asd[i], means_tdc[i]+se_tdc[i]) + 0.0001, "**" if p < 0.01 else "*", ha='center', va='bottom', fontsize=12)
    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.close()

def main():
    try: mapping = load_roi_mapping(MAPPING_FILE)
    except Exception as e: print(f"Warning: {e}"); return
    for task in TASKS:
        for path_name, filename in PATHS.items():
            X_asd, X_tdc = load_task_data(task['path'], filename)
            if X_asd is None: continue
            plot_network_diff(aggregate_to_networks(X_asd, mapping), aggregate_to_networks(X_tdc, mapping), f"{task['name']} - {path_name}", os.path.join(OUT_DIR, f"{task['name']}_{path_name}_NetworkDiff.png"))

if __name__ == "__main__":
    main()
