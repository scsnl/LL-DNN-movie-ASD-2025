"""
Step 7: Baseline Network Analysis (20 Functional Networks)

This script performs a group-level comparison of model attribution across 
20 functional networks without temporal windowing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from pathlib import Path

# Configuration
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
IG_ROOT = os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models"
MAPPING_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "mapping", "subregion_func_network_Yeo_updated (1).csv")
OUT_DIR = "network20_baseline_stats"; os.makedirs(OUT_DIR, exist_ok=True)

NETWORK_NAMES = ["Visual_Cent", "Visual_Peri", "Somatomotor_A", "Somatomotor_B", "DorsalAttn_A", "DorsalAttn_B", "VentralAttn_A", "VentralAttn_B", "Limbic_A", "Limbic_B", "Control_A", "Control_B", "Control_C", "Default_A", "Default_B", "Default_C", "TempPar", "Amygdala_Hipp", "Basal_Ganglia", "Thalamus"]

def load_mapping(csv_path):
    if not os.path.exists(csv_path): return None, None
    df = pd.read_csv(csv_path); n_rois, n_nets = 246, 20; mapping_mat = np.zeros((n_rois, n_nets))
    for _, row in df.iterrows():
        try:
            roi_idx, net_val = int(float(row['Label'])) - 1, int(float(row['Yeo_17network']))
            if 1 <= net_val <= 20: mapping_mat[roi_idx, net_val-1] = 1
        except: continue
    counts = mapping_mat.sum(axis=0); counts[counts == 0] = 1
    return mapping_mat, counts

def load_mil_ig_data(root_dir):
    asd_data, tdc_data = [], []
    for fold in range(5):
        p_ig, p_lbl = os.path.join(root_dir, f"fold{fold}", "ig_roi_milout.npy"), os.path.join(root_dir, f"fold{fold}", "labels.npy")
        if os.path.exists(p_ig) and os.path.exists(p_lbl):
            ig, lbl = np.load(p_ig), np.load(p_lbl)
            ig_mean = np.mean(np.abs(ig), axis=1)
            asd_data.append(ig_mean[lbl == 0]); tdc_data.append(ig_mean[lbl == 1])
    return (np.concatenate(asd_data, 0), np.concatenate(tdc_data, 0)) if asd_data else (None, None)

def main():
    map_mat, counts = load_mapping(MAPPING_FILE)
    X_asd_roi, X_tdc_roi = load_mil_ig_data(IG_ROOT)
    if X_asd_roi is None or map_mat is None: return
    
    X_asd_net = np.dot(X_asd_roi, map_mat) / counts
    X_tdc_net = np.dot(X_tdc_roi, map_mat) / counts
    
    means_asd, means_tdc = X_asd_net.mean(0), X_tdc_net.mean(0)
    se_asd, se_tdc = X_asd_net.std(0)/np.sqrt(X_asd_net.shape[0]), X_tdc_net.std(0)/np.sqrt(X_tdc_net.shape[0])
    p_vals = [ttest_ind(X_asd_net[:, i], X_tdc_net[:, i], equal_var=False).pvalue for i in range(20)]
    
    plt.figure(figsize=(16, 6))
    x = np.arange(20); width = 0.35
    plt.bar(x - width/2, means_asd, width, yerr=se_asd, label='ASD', color='#d62728', alpha=0.8)
    plt.bar(x + width/2, means_tdc, width, yerr=se_tdc, label='TDC', color='#1f77b4', alpha=0.8)
    plt.xticks(x, NETWORK_NAMES, rotation=45, ha='right'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "MIL_Baseline_20Net_Diff.png")); plt.close()

if __name__ == "__main__":
    main()
