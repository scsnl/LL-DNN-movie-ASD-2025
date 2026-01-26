"""
Step 3-4: Integrated Gradients (IG) Temporal Group Difference

This script analyzes the temporal variation of ROI importance by comparing 
IG attribution curves between ASD and TDC groups. 
Main objectives:
1. Calculates the point-wise group difference in IG magnitude.
2. Identifies timepoints where the difference is maximal (Top 10% diff).
3. Detects timepoints where the model's attention is most concentrated (Top 20% attention).
4. Generates temporal curves with highlighted regions of interest.
5. Exports statistical summaries for event integration.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Path and test configuration
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
save_fig = not TEST_MODE
save_csv = True

def plot_top_diff(asd_curve, tdc_curve, title, save_name, output_dir, top_pct=10):
    """Highlight timepoints with the largest statistical differences between groups."""
    mean_asd = np.mean(asd_curve, axis=0)
    mean_tdc = np.mean(tdc_curve, axis=0)
    std_asd = np.std(asd_curve, axis=0)
    std_tdc = np.std(tdc_curve, axis=0)
    T = mean_asd.shape[0]

    # Independent t-test at each timepoint
    p_vals = np.array([ttest_ind(asd_curve[:, t], tdc_curve[:, t])[1] for t in range(T)])
    _, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")

    # Select timepoints with most significant differences
    n_top = max(1, int(T * (top_pct / 100)))
    top_diff_idx = np.argsort(p_vals)[:n_top]

    plt.figure(figsize=(12, 6))
    plt.plot(mean_asd, label="ASD", color="red")
    plt.fill_between(range(T), mean_asd - std_asd, mean_asd + std_asd, color="red", alpha=0.2)
    plt.plot(mean_tdc, label="TDC", color="blue")
    plt.fill_between(range(T), mean_tdc - std_tdc, mean_tdc + std_tdc, color="blue", alpha=0.2)
    
    # Shade top-diff timepoints
    for t in top_diff_idx:
        plt.axvline(x=t, color="red", linestyle="--", alpha=0.4)
        
    plt.title(title)
    plt.xlabel("Timepoint (TR)")
    plt.ylabel("IG Attribution Magnitude")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(output_dir, f"{save_name}_topdiff.png"), dpi=300)
    plt.close()

    if save_csv:
        df = pd.DataFrame({
            "timepoint": np.arange(T),
            "mean_asd": mean_asd,
            "std_asd": std_asd,
            "mean_tdc": mean_tdc,
            "std_tdc": std_tdc,
            "p_raw": p_vals,
            "p_fdr": p_fdr,
            "is_topdiff": np.isin(np.arange(T), top_diff_idx)
        })
        df.to_csv(os.path.join(output_dir, f"{save_name}_topdiff.csv"), index=False)

def plot_top_attention(asd_curve, tdc_curve, title, save_name, output_dir, top_pct=20):
    """Highlight timepoints with the highest absolute attribution within each group."""
    mean_asd = np.mean(asd_curve, axis=0)
    mean_tdc = np.mean(tdc_curve, axis=0)
    std_asd = np.std(asd_curve, axis=0)
    std_tdc = np.std(tdc_curve, axis=0)
    T = mean_asd.shape[0]

    n_top = max(1, int(T * (top_pct / 100)))
    top_asd = np.argsort(mean_asd)[-n_top:]
    top_tdc = np.argsort(mean_tdc)[-n_top:]

    plt.figure(figsize=(12, 6))
    plt.plot(mean_asd, label="ASD", color="blue")
    plt.fill_between(range(T), mean_asd - std_asd, mean_asd + std_asd, color="blue", alpha=0.2)
    plt.plot(mean_tdc, label="TDC", color="orange")
    plt.fill_between(range(T), mean_tdc - std_tdc, mean_tdc + std_tdc, color="orange", alpha=0.2)

    # Marker highest attribution timepoints per group
    for t in top_asd:
        plt.axvline(x=t, color="blue", linestyle="--", alpha=0.3)
    for t in top_tdc:
        plt.axvline(x=t, color="orange", linestyle="--", alpha=0.3)

    plt.title(title)
    plt.xlabel("Timepoint (TR)")
    plt.ylabel("IG Attribution")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(output_dir, f"{save_name}_topattn.png"), dpi=300)
    plt.close()

    if save_csv:
        df = pd.DataFrame({
            "timepoint": np.arange(T),
            "mean_asd": mean_asd,
            "std_asd": std_asd,
            "mean_tdc": mean_tdc,
            "std_tdc": std_tdc,
            "is_top_asd": np.isin(np.arange(T), top_asd),
            "is_top_tdc": np.isin(np.arange(T), top_tdc)
        })
        df.to_csv(os.path.join(output_dir, f"{save_name}_topattn.csv"), index=False)

def main():
    output_dir = "./ig_group_level_results"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize curve collectors
    curve_out_asd, curve_out_tdc = [], []
    curve_mil_asd, curve_mil_tdc = [], []

    # Aggregate IG curves from all folds
    for fold in range(5):
        fold_dir = f"./saved_models/fold{fold}"
        out_path = os.path.join(fold_dir, "ig_curve_outmean.npy")
        mil_path = os.path.join(fold_dir, "ig_curve_milout.npy")
        label_path = os.path.join(fold_dir, "labels.npy")

        if os.path.exists(out_path) and os.path.exists(mil_path) and os.path.exists(label_path):
            out_curve = np.load(out_path)
            mil_curve = np.load(mil_path)
            labels = np.load(label_path)
            # Append subject curves based on label
            curve_out_asd.append(out_curve[labels == 0])
            curve_out_tdc.append(out_curve[labels == 1])
            curve_mil_asd.append(mil_curve[labels == 0])
            curve_mil_tdc.append(mil_curve[labels == 1])

    if not curve_out_asd:
        print("No IG curve data found. Skipping temporal analysis.")
        return

    # Stack results
    curve_out_asd = np.vstack(curve_out_asd); curve_out_tdc = np.vstack(curve_out_tdc)
    curve_mil_asd = np.vstack(curve_mil_asd); curve_mil_tdc = np.vstack(curve_mil_tdc)

    # Run analysis for both paths
    plot_top_diff(curve_out_asd, curve_out_tdc, "Group Difference (Mean Path)", "ig_curve_group_outmean", output_dir)
    plot_top_attention(curve_out_asd, curve_out_tdc, "Concentrated Importance (Mean Path)", "ig_curve_group_outmean", output_dir)
    plot_top_diff(curve_mil_asd, curve_mil_tdc, "Group Difference (MIL Path)", "ig_curve_group_mil", output_dir)
    plot_top_attention(curve_mil_asd, curve_mil_tdc, "Concentrated Importance (MIL Path)", "ig_curve_group_mil", output_dir)

if __name__ == "__main__":
    main()
