"""
Step 3-1: MIL Path Attention Weight Plotting

This script generates a specialized visualization of the attention weights 
focusing on the MIL (Multi-Instance Learning) path component. It helps in 
identifying specific time windows that the model considers critical for 
diagnosing ASD.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.special import softmax
from statsmodels.stats.multitest import multipletests

def analyze_attention_and_prediction_from_saved(folds_dir='saved_models', alpha=0.08):
    """
    Summarizes and plots MIL-specific attention scores across folds.
    """
    attn_all, outmean_all, label_all = [], [], []

    # Iterate through saved folds
    for fold in range(5):
        fold_dir = os.path.join(folds_dir, f"fold{fold}")
        attn_path = os.path.join(fold_dir, "attn_weights.npy")
        out_path = os.path.join(fold_dir, "out_mean.npy")
        label_path = os.path.join(fold_dir, "labels.npy")
        
        if not all(os.path.exists(p) for p in [attn_path, out_path, label_path]):
            continue
            
        attn_all.append(np.load(attn_path))
        outmean_all.append(np.load(out_path))
        label_all.append(np.load(label_path))

    if not attn_all:
        print("No MIL output data found.")
        return

    # Aggregate data across folds
    attn_all = np.vstack(attn_all)
    outmean_all = np.vstack(outmean_all)
    label_all = np.concatenate(label_all)

    os.makedirs("figures", exist_ok=True)
    attn_asd = attn_all[label_all == 0]
    attn_td = attn_all[label_all == 1]

    # Calculate timepoint means
    mean_asd = attn_asd.mean(axis=0)
    mean_td = attn_td.mean(axis=0)
    
    # Point-wise t-test for group differences
    pvals = np.array([ttest_ind(attn_asd[:, t], attn_td[:, t], equal_var=False).pvalue for t in range(attn_asd.shape[1])])
    sig_raw = pvals < alpha
    reject_fdr, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    sig_fdr = reject_fdr

    # Visualizing the MIL Attention Profile
    plt.figure(figsize=(12, 5))
    plt.plot(mean_asd, label='ASD', color='blue', linewidth=2)
    plt.plot(mean_td, label='TD', color='orange', linewidth=2)
    # Highlight significant TRs
    for t in np.where(sig_raw)[0]:
        plt.axvline(x=t, color='green', linestyle='--', alpha=0.3)
    plt.title('MIL Path Attention Profile (ASD vs TD)')
    plt.xlabel('Timepoint')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/attn_profile_raw_mil.png", dpi=300)
    plt.close()

    # Save data for further statistical analysis
    df = pd.DataFrame({
        'timepoint': np.arange(len(pvals)),
        'mean_asd': mean_asd,
        'mean_td': mean_td,
        'p_raw': pvals,
        'p_fdr': pvals_fdr,
        'sig_raw': sig_raw,
        'sig_fdr': sig_fdr
    })
    df.to_csv(os.path.join("figures", "attn_significance_mil.csv"), index=False)

    # ---------------- MIL Softmax Confidence ----------------
    probs = softmax(outmean_all, axis=1)
    asd_probs = probs[label_all == 0][:, 0]
    td_probs = probs[label_all == 1][:, 0]
    
    plt.figure(figsize=(8, 5))
    plt.boxplot([asd_probs, td_probs], tick_labels=['ASD', 'TD'], patch_artist=True,
                boxprops=dict(facecolor='skyblue'), medianprops=dict(color='black'))
    plt.title('MIL Path Softmax Confidence Distribution')
    plt.ylabel('Confidence (ASD Class)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/mean_prob_boxplot_mil.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    folds = os.environ.get("TEST_OUTPUT_DIR", "")
    folds_root = os.path.join(folds, "saved_models") if folds else "saved_models"
    analyze_attention_and_prediction_from_saved(folds_dir=folds_root)
