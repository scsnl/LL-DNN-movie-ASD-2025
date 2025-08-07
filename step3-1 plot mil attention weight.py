#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Mon May 19 15:06:39 2025

@author: leili
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.special import softmax
from statsmodels.stats.multitest import multipletests

def analyze_attention_and_prediction_from_saved(folds_dir='saved_models', alpha=0.08):
    attn_all, outmean_all, label_all = [], [], []

    # Load data from each fold

    for fold in range(5):
        fold_dir = os.path.join(folds_dir, f"fold{fold}")
        attn = np.load(os.path.join(fold_dir, "attn_weights.npy"))
        out = np.load(os.path.join(fold_dir, "out_mean.npy"))
        label = np.load(os.path.join(fold_dir, "labels.npy"))
        attn_all.append(attn)
        outmean_all.append(out)
        label_all.append(label)

    attn_all = np.vstack(attn_all)
    outmean_all = np.vstack(outmean_all)
    label_all = np.concatenate(label_all)

    os.makedirs("figures", exist_ok=True)
    attn_asd = attn_all[label_all == 0]
    attn_td = attn_all[label_all == 1]

    mean_asd = attn_asd.mean(axis=0)
    mean_td = attn_td.mean(axis=0)
    pvals = np.array([ttest_ind(attn_asd[:, t], attn_td[:, t], equal_var=False).pvalue for t in range(attn_asd.shape[1])])
    sig_raw = pvals < alpha
    reject_fdr, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    sig_fdr = reject_fdr

    # Plot raw p < 0.05

    plt.figure(figsize=(12, 5))
    plt.plot(mean_asd, label='ASD', color='blue', linewidth=2)
    plt.plot(mean_td, label='TD', color='orange', linewidth=2)
    for t in np.where(sig_raw)[0]:
        plt.axvline(x=t, color='green', linestyle='--', alpha=0.3)
    plt.title('Attention Profile (ASD vs TD)\nGreen = Raw p < 0.05')
    plt.xlabel('Timepoint')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/attn_profile_raw.png", dpi=300)
    plt.close()

    # Plot FDR-significant p < 0.05

    plt.figure(figsize=(12, 5))
    plt.plot(mean_asd, label='ASD', color='blue', linewidth=2)
    plt.plot(mean_td, label='TD', color='orange', linewidth=2)
    for t in np.where(sig_fdr)[0]:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.3)
    plt.title('Attention Profile (ASD vs TD)\n = FDR-significant p < 0.05')
    plt.xlabel('Timepoint')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/attn_profile_fdr.png", dpi=300)
    plt.close()

    df = pd.DataFrame({
        'timepoint': np.arange(len(pvals)),
        'mean_asd': mean_asd,
        'mean_td': mean_td,
        'p_raw': pvals,
        'p_fdr': pvals_fdr,
        'sig_raw': sig_raw,
        'sig_fdr': sig_fdr
    })
    df.to_csv("# TODO: specify your data path", index=False)


    # Prediction confidence

    probs = softmax(outmean_all, axis=1)
    asd_probs = probs[label_all == 0][:, 0]
    td_probs = probs[label_all == 1][:, 0]
    t_stat, p_val = ttest_ind(asd_probs, td_probs, equal_var=False)
    print(f"ASD prob: {asd_probs.mean():.4f}, TD prob: {td_probs.mean():.4f}, t={t_stat:.4f}, p={p_val:.4e}")

    plt.figure(figsize=(8, 5))
    plt.boxplot([asd_probs, td_probs], labels=['ASD', 'TD'], patch_artist=True,
                boxprops=dict(facecolor='skyblue'), medianprops=dict(color='black'))
    plt.title('Mean Path Softmax Probabilities')
    plt.ylabel('Predicted Probability')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/mean_prob_boxplot.png", dpi=300)
    plt.close()

# Run the analysis

if __name__ == "__main__":
    analyze_attention_and_prediction_from_saved()
