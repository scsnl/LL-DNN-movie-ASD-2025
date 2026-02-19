"""
Supply Step 3-6 (Seg2): Multi-Feature Event Definition and Integration

This script defines neural events for Segment 2 by integrating multiple signals:
1. Calculates group-level Top-20% timepoints based on IG attribution.
2. Combines these with statistically significant attention TRs.
3. Overlays the MIL path accuracy difference curve.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
BASE = Path(TEST_OUTPUT_DIR) / 'seg2' if TEST_OUTPUT_DIR else Path('results/seg2')

def compute_ig_group():
    ig_asd, ig_tdc = [], []
    fold_dir_root = BASE / 'saved_models'
    for k in range(5):
        f_ig, f_lbl = fold_dir_root / f'fold{k}' / 'ig_curve_milout.npy', fold_dir_root / f'fold{k}' / 'labels.npy'
        if f_ig.exists() and f_lbl.exists():
            ig, lbl = np.load(f_ig), np.load(f_lbl)
            ig_asd.append(np.mean(np.abs(ig[lbl == 0]), 0)); ig_tdc.append(np.mean(np.abs(ig[lbl == 1]), 0))
    if not ig_asd: return None
    m_asd, m_tdc = np.mean(np.stack(ig_asd), 0), np.mean(np.stack(ig_tdc), 0)
    thr_asd, thr_tdc = np.percentile(m_asd, 80), np.percentile(m_tdc, 80)
    return pd.DataFrame({'timepoint': np.arange(len(m_asd)), 'is_top_asd': m_asd >= thr_asd, 'is_top_tdc': m_tdc >= thr_tdc})

def main():
    acc_p, attn_p = BASE / 'runs' / 'mil_accuracy_difference_curve.csv', BASE / 'figures' / 'attn_significance.csv'
    if not acc_p.exists() or not attn_p.exists(): 
        print("Missing Seg2 dependencies. Ensure training and aggregation steps are complete.")
        return
    ig_df = compute_ig_group()
    if ig_df is None: return
    df = pd.read_csv(acc_p).merge(ig_df, on='timepoint', how='inner').merge(pd.read_csv(attn_p)[['timepoint', 'sig_raw']], on='timepoint', how='left')
    df['sig_raw'] = df['sig_raw'].fillna(False)
    
    # Save Overlap Events
    overlap_asd = df.loc[df['is_top_asd'] & df['sig_raw'], 'timepoint'].values
    overlap_tdc = df.loc[df['is_top_tdc'] & df['sig_raw'], 'timepoint'].values
    pd.DataFrame({'overlap_asd': pd.Series(overlap_asd), 'overlap_tdc': pd.Series(overlap_tdc)}).to_csv(BASE / 'overlap_events.csv', index=False)
    print(f"Overlap events saved to {BASE / 'overlap_events.csv'}")

if __name__ == '__main__':
    main()
