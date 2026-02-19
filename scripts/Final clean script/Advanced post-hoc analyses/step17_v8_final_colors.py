"""
Step 17: Attention Score Curve Visualization (Standardized Style)

This script plots attention score curves for ASD and TDC groups, highlighting 
High and Low attention clusters with specific colors and hatching styles.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from itertools import groupby
from operator import itemgetter

# Configuration
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
OUT_DIR = "final_curve_visualization"; os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    {
        "name": "MovieDM_Seg2",
        "attn_csv": os.path.join(TEST_OUTPUT_DIR, "figures", "attn_significance.csv") if TEST_OUTPUT_DIR else "figures/attn_significance.csv",
        "tr_start": 0, "tr_end": 9999
    }
]

COLOR_LOW, COLOR_HIGH = '#006400', '#d62728'

def get_clusters_classification(attn_csv, tr_start, tr_end, config_name):
    if not os.path.exists(attn_csv): return [], []
    df = pd.read_csv(attn_csv); mask = (df['timepoint'] >= tr_start) & (df['timepoint'] < tr_end)
    df_seg = df.loc[mask].copy()
    if df_seg.empty: return [], []
    sig_trs = sorted([t - tr_start for t in df_seg.loc[df_seg['sig_raw'], 'timepoint'].values])
    clusters = [list(map(itemgetter(1), g)) for _, g in groupby(enumerate(sig_trs), lambda ix: ix[0] - ix[1])]
    high_cls, low_cls = [], []
    if config_name == "MovieDM_Seg2":
        if clusters: low_cls.append(clusters[0]); high_cls.extend(clusters[1:])
    return high_cls, low_cls

def main():
    for c in CONFIGS:
        if not os.path.exists(c['attn_csv']): continue
        df = pd.read_csv(c['attn_csv']); mask = (df['timepoint'] >= c['tr_start']) & (df['timepoint'] < c['tr_end'])
        df_seg = df.loc[mask].copy()
        if df_seg.empty: continue
        col_td = 'mean_td' if 'mean_td' in df.columns else 'mean_tdc'
        high_cls, low_cls = get_clusters_classification(c['attn_csv'], c['tr_start'], c['tr_end'], c['name'])
        
        plt.figure(figsize=(15, 4), dpi=300)
        for clus in high_cls: plt.axvspan(clus[0]-0.5, clus[-1]+0.5, facecolor=COLOR_HIGH, alpha=0.3, hatch='|||')
        for clus in low_cls: plt.axvspan(clus[0]-0.5, clus[-1]+0.5, facecolor=COLOR_LOW, alpha=0.2, hatch='|||')
        plt.plot(df_seg['mean_asd'].values, color='blue', linewidth=2, label='ASD')
        plt.plot(df_seg[col_td].values, color='orange', linewidth=2, label='TDC')
        plt.xlabel("Timepoint"); plt.ylabel("Attention Score"); plt.title(f"{c['name']} Attention"); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{c['name']}_AttnCurve.png")); plt.close()

if __name__ == "__main__":
    main()
