"""
Step 3-6: Integrated Event Visualization

This script integrates multiple temporal features into a unified visualization:
1. Accuracy Bias Curve (MIL performance over time).
2. Top IG Attribution timepoints (Regions where model is most 'active').
3. Attention Significant timepoints (TRs where groups statistically differ).
By overlapping these features, we can identify robust "Events" that drive the 
model's decision-making process.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Test mode check
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"

def main():
    # File path configuration
    mil_dir = "./mil_logits_per_fold"
    group_dir = "./ig_group_level_results"
    fig_path = "./figures/"
    save_path = os.path.join(fig_path, "combined_accuracy_bias_with_events.png")
    os.makedirs(fig_path, exist_ok=True)

    # Dependency check: Load previously generated CSVs
    acc_csv = os.path.join(mil_dir, "mil_accuracy_difference_curve.csv")
    ig_csv = os.path.join(group_dir, "ig_curve_group_mil_topattn.csv")
    attn_csv = os.path.join(fig_path, "attn_significance.csv")

    if not all(os.path.exists(p) for p in [acc_csv, ig_csv, attn_csv]):
        print("[WARNING] Missing required data files. Ensure steps 3-1 through 3-5 are complete.")
        return

    # Load integrated datasets
    df_acc = pd.read_csv(acc_csv)
    df_ig = pd.read_csv(ig_csv)
    df_attn = pd.read_csv(attn_csv)

    # Extract relevant vectors
    time = df_acc['timepoint']
    diff = df_acc['accuracy_diff']
    asd_ig_top = df_ig[df_ig['is_top_asd']]['timepoint'].values
    tdc_ig_top = df_ig[df_ig['is_top_tdc']]['timepoint'].values
    attn_sig = df_attn[df_attn['sig_raw']]['timepoint'].values

    # Unified Temporal Plotting
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # 1. Prediction Accuracy Bias (Smooth curve)
    ax.plot(time, diff, label="Accuracy Bias (ASD - TDC)", color="purple", linewidth=2)

    # 2. Top-K Attribution regions (Vertical shaded lines)
    for tp in asd_ig_top:
        ax.axvline(x=tp, color='mediumblue', linestyle='-', alpha=0.4, linewidth=1)
    for tp in tdc_ig_top:
        ax.axvline(x=tp, color='darkorange', linestyle='-', alpha=0.4, linewidth=1)

    # 3. Attention Statistical significance (Dashed vertical lines)
    for tp in attn_sig:
        ax.axvline(x=tp, color='forestgreen', linestyle='--', alpha=0.6, linewidth=1)

    # Styling and Labels
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Timepoint (TR)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Group Bias Score", fontsize=12, fontweight='bold')
    ax.set_title("Temporal Event Fingerprinting: Accuracy Bias vs. Attribution & Attention", fontsize=14, fontweight='bold')

    # Legend Construction
    custom_legend = [
        Patch(facecolor='mediumblue', edgecolor='mediumblue', alpha=0.4, label='ASD Top 20% IG'),
        Patch(facecolor='darkorange', edgecolor='darkorange', alpha=0.4, label='TDC Top 20% IG'),
        Patch(facecolor='forestgreen', edgecolor='forestgreen', alpha=0.6, label='Attention sig (p < 0.05)'),
        plt.Line2D([0], [0], color='purple', lw=2, label='Accuracy Difference')
    ]
    ax.legend(handles=custom_legend, loc='best', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Analyze and print temporal intersections
    overlap_asd = np.intersect1d(asd_ig_top, attn_sig)
    overlap_tdc = np.intersect1d(tdc_ig_top, attn_sig)
    print(f"Number of overlapping ASD timepoints: {len(overlap_asd)}")
    print(f"Number of overlapping TDC timepoints: {len(overlap_tdc)}")
    print(f"Consolidated visualization saved to: {save_path}")

if __name__ == "__main__":
    main()
