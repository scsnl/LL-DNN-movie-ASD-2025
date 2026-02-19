"""
Step 16: Network-Level Group Difference on Top 10% ROIs

This script identifies the most important ROIs identified by the model 
cluster within functional networks and compares groups.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
from operator import itemgetter
from pathlib import Path

# Configuration
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")

OUT_DIR = "network_diff_top10_analysis"; os.makedirs(OUT_DIR, exist_ok=True)
MAPPING_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "mapping", "subregion_func_network_Yeo_updated (1).csv")

NETWORK_NAMES = {1: "VisCent", 2: "VisPeri", 3: "SomMotA", 4: "SomMotB", 5: "DorsAttnA", 6: "DorsAttnB", 7: "SalVentAttnA", 8: "SalVentAttnB", 9: "LimbicB", 10: "LimbicA", 11: "ContC", 12: "ContA", 13: "ContB", 14: "TempPar", 15: "DefaultC", 16: "DefaultA", 17: "DefaultB", 18: "Amyg/Hipp", 19: "Basal Ganglia", 20: "Thalamus"}

CONFIGS = [{"name": "MovieDM_Seg2", "mil_cluster_type": "Low", "global_ig": "ig_roi_outmean.npy", "mil_ig": "ig_roi_milout.npy", "ig_root": os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models", "attn_csv": os.path.join(TEST_OUTPUT_DIR, "figures", "attn_significance.csv") if TEST_OUTPUT_DIR else "figures/attn_significance.csv", "tr_start": 0, "tr_end": 9999, "has_mil": True}]

def load_mapping():
    if not os.path.exists(MAPPING_FILE): return None
    df = pd.read_csv(MAPPING_FILE, header=1); mapping = np.zeros(246, dtype=int)
    for _, row in df.iterrows():
        try:
            val = row['Label']
            if pd.isna(val): continue
            idx = int(float(val)) - 1
            net = int(float(row['Yeo_17network']))
            if 0 <= idx < 246: mapping[idx] = net
        except: continue
    return mapping

def get_clusters(attn_csv, tr_start, tr_end):
    if not os.path.exists(attn_csv): return [], []
    df = pd.read_csv(attn_csv); seg_mask = (df['timepoint'] >= tr_start) & (df['timepoint'] < tr_end)
    df_seg = df.loc[seg_mask].copy()
    if df_seg.empty: return [], []
    col_td = 'mean_td' if 'mean_td' in df.columns else 'mean_tdc'
    curve = (df_seg['mean_asd'].values + df_seg[col_td].values) / 2.0; global_threshold = np.mean(curve)
    sig_trs = sorted([t - tr_start for t in df_seg.loc[df_seg['sig_raw'], 'timepoint'].values])
    clusters = [list(map(itemgetter(1), g)) for _, g in groupby(enumerate(sig_trs), lambda ix: ix[0] - ix[1])]
    high_trs, low_trs = [], []
    for clus in clusters:
        v = [t for t in clus if t < len(curve)]
        if not v: continue
        if np.mean(curve[v]) >= global_threshold: high_trs.extend(v)
        else: low_trs.extend(v)
    return high_trs, low_trs

def load_subject_data(root_dir, ig_filename, tr_indices):
    groups = {0: [], 1: []}
    for fold in range(5):
        p_ig, p_lbl = os.path.join(root_dir, f"fold{fold}", ig_filename), os.path.join(root_dir, f"fold{fold}", "labels.npy")
        if os.path.exists(p_ig) and os.path.exists(p_lbl):
            ig, lbl = np.load(p_ig), np.load(p_lbl)
            if ig.ndim == 3:
                if tr_indices is not None:
                    vt = [t for t in tr_indices if t < ig.shape[1]]
                    if not vt: continue
                    ig_subset = ig[:, vt, :]
                else:
                    ig_subset = ig
                sub_abs = np.abs(np.median(ig_subset, axis=1))
            else:
                sub_abs = np.abs(ig) # (N, 246)
            
            for i, label in enumerate(lbl): groups[int(label)].append(sub_abs[i])
    
    # Ensure 2D (N, 246) even if 0 or 1 subjects
    res = {}
    for k, v in groups.items():
        if not v: res[k] = np.zeros((0, 246))
        else: res[k] = np.vstack(v)
    return res

def main():
    mapping = load_mapping()
    if mapping is None: return
    for conf in CONFIGS:
        print(f"Processing {conf['name']}...")
        high_trs, low_trs = get_clusters(conf['attn_csv'], conf['tr_start'], conf['tr_end'])
        mil_trs = high_trs if conf['mil_cluster_type'] == 'High' else low_trs
        for label, ig_file, trs in [("GlobalPath", conf['global_ig'], None), ("MILPath", conf['mil_ig'], mil_trs)]:
            data_dict = load_subject_data(conf['ig_root'], ig_file, trs)
            if data_dict[0].shape[0] == 0 or data_dict[1].shape[0] == 0:
                print(f"  Skipping {label}: missing group data.")
                continue
            
            asd_data, tdc_data = data_dict[0], data_dict[1]
            # asd_data is (N, 246)
            mean_asd_roi = np.mean(asd_data, axis=0)
            top_idx = np.argsort(mean_asd_roi)[-int(246*0.10):]
            top_mask = np.zeros(246, dtype=bool); top_mask[top_idx] = True
            
            # Aggregate and T-test
            asd_net_means, tdc_net_means, p_vals, valid_nets = [], [], [], []
            for nid in sorted(NETWORK_NAMES.keys()):
                roi_idx = np.where((mapping == nid) & top_mask)[0]
                if len(roi_idx) == 0: continue
                
                valid_nets.append(nid)
                a = np.mean(asd_data[:, roi_idx], axis=1) # (N_asd,)
                t = np.mean(tdc_data[:, roi_idx], axis=1) # (N_tdc,)
                asd_net_means.append(np.mean(a))
                tdc_net_means.append(np.mean(t))
                _, p = ttest_ind(a, t, equal_var=False)
                p_vals.append(p)
            
            if not valid_nets: continue
            plt.figure(figsize=(12, 6), dpi=300)
            x = np.arange(len(valid_nets)); width = 0.35
            plt.bar(x - width/2, asd_net_means, width, label='ASD', color='#d62728', alpha=0.8)
            plt.bar(x + width/2, tdc_net_means, width, label='TDC', color='#1f77b4', alpha=0.8)
            plt.xticks(x, [NETWORK_NAMES[n] for n in valid_nets], rotation=45, ha='right'); plt.legend(); plt.title(f"{conf['name']} {label}")
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"{conf['name']}_{label}_Top10_NetDiff.png")); plt.close()

if __name__ == "__main__":
    main()
