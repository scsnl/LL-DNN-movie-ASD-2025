"""
Step 3: Attention Profile Analysis

This script processes the saved attention weights and model predictions to 
characterize the temporal focus of the model.
Key functions:
1. Aggregates attention weights across all CV folds for ASD and TDC groups.
2. Performs a point-wise t-test to identify timepoints where attention focus 
   significantly differs between groups.
3. Applies FDR correction for multiple comparisons.
4. Generates temporal profile plots highlighting significant windows.
5. Evaluates model prediction confidence (softmax probabilities) for both groups.
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
    Load saved arrays from 5-fold CV and perform statistical comparison of 
    temporal attention profiles.
    """
    attn_all, outmean_all, label_all = [], [], []

    # Load data from each fold's result directory
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
        print("No data found for attention analysis. Run training script first.")
        return

    # Stack results across all subjects in all folds
    # attn_all: (N_total, T, 1)
    attn_all = np.vstack(attn_all)
    outmean_all = np.vstack(outmean_all)
    label_all = np.concatenate(label_all)

    os.makedirs("figures", exist_ok=True)
    
    # Split subjects into groups
    attn_asd = attn_all[label_all == 0]
    attn_td = attn_all[label_all == 1]

    # Calculate group-level mean attention per timepoint
    mean_asd = attn_asd.mean(axis=0)
    mean_td = attn_td.mean(axis=0)
    
    # Point-wise independent t-test (ASD vs TD)
    pvals = np.array([ttest_ind(attn_asd[:, t], attn_td[:, t], equal_var=False).pvalue for t in range(attn_asd.shape[1])])
    
    # Identify significant timepoints (uncorrected)
    sig_raw = pvals < alpha
    # Identify significant timepoints (FDR corrected)
    reject_fdr, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    sig_fdr = reject_fdr

    # Visualization 1: Raw significance (p < 0.05 / custom alpha)
    plt.figure(figsize=(12, 5))
    plt.plot(mean_asd, label='ASD', color='blue', linewidth=2)
    plt.plot(mean_td, label='TD', color='orange', linewidth=2)
    # Highlight significant TRs with vertical lines
    for t in np.where(sig_raw)[0]:
        plt.axvline(x=t, color='green', linestyle='--', alpha=0.3)
    plt.title(f'Attention Profile (ASD vs TD) - Uncorrected p < {alpha}')
    plt.xlabel('Timepoint (TR)')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/attn_profile_raw.png", dpi=300)
    plt.close()

    # Visualization 2: FDR-corrected significance
    plt.figure(figsize=(12, 5))
    plt.plot(mean_asd, label='ASD', color='blue', linewidth=2)
    plt.plot(mean_td, label='TD', color='orange', linewidth=2)
    for t in np.where(sig_fdr)[0]:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.3)
    plt.title(f'Attention Profile (ASD vs TD) - FDR-corrected p < {alpha}')
    plt.xlabel('Timepoint (TR)')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/attn_profile_fdr.png", dpi=300)
    plt.close()

    # Save statistical results to CSV for event definition
    df = pd.DataFrame({
        'timepoint': np.arange(len(pvals)),
        'mean_asd': mean_asd,
        'mean_td': mean_td,
        'p_raw': pvals,
        'p_fdr': pvals_fdr,
        'sig_raw': sig_raw,
        'sig_fdr': sig_fdr
    })
    df.to_csv(os.path.join("figures", "attn_significance.csv"), index=False)

    # ---------------- Prediction Confidence Analysis ----------------
    # Compute probabilities using softmax on final logits
    probs = softmax(outmean_all, axis=1)
    # Extract probability of being ASD (class 0)
    asd_probs = probs[label_all == 0][:, 0]
    td_probs = probs[label_all == 1][:, 0]
    
    t_stat, p_val = ttest_ind(asd_probs, td_probs, equal_var=False)
    print(f"ASD Mean Prediction Confidence: {asd_probs.mean():.4f}")
    print(f"TD Mean Prediction Confidence: {td_probs.mean():.4f}")
    print(f"Confidence Diff: t={t_stat:.4f}, p={p_val:.4e}")

    # Plot boxplot of prediction probabilities
    plt.figure(figsize=(8, 5))
    plt.boxplot([asd_probs, td_probs], tick_labels=['ASD', 'TD'], patch_artist=True,
                boxprops=dict(facecolor='skyblue'), medianprops=dict(color='black'))
    plt.title('Mean Path Class-0 Prediction Confidence')
    plt.ylabel('Predicted Probability (ASD)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/mean_prob_boxplot.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Path configuration for standard or test runs
    folds = os.environ.get("TEST_OUTPUT_DIR", "")
    folds_root = os.path.join(folds, "saved_models") if folds else "saved_models"
    
    analyze_attention_and_prediction_from_saved(folds_dir=folds_root)
