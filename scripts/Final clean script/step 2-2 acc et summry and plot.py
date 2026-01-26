"""
Step 2-2: Performance Metrics Summary and Visualization

This script aggregates performance metrics across all cross-validation folds. 
Specifically, it:
1. Calculates accuracy, precision, recall, and F1-score for each fold based on 
   the saved model logits and true labels.
2. Computes the mean and standard deviation of these metrics.
3. Saves a summary CSV for documentation.
4. Generates bar plots showing each metric across folds to visualize performance 
   consistency.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Check for specified metrics file or compute from scratch
metrics_csv = os.environ.get("METRICS_CSV", "")
if metrics_csv and os.path.exists(metrics_csv):
    df = pd.read_csv(metrics_csv)
else:
    # Automatically compute metrics from saved model outputs in the results directory
    base_out = os.environ.get("TEST_OUTPUT_DIR", "")
    folds_root = Path(base_out) / "saved_models" if base_out else Path("saved_models")
    rows = []
    
    # Iterate through 5 folds
    for fold in range(5):
        fold_dir = folds_root / f"fold{fold}"
        out_path = fold_dir / "out_mean.npy"
        lbl_path = fold_dir / "labels.npy"
        
        # Skip if results for this fold are missing
        if not out_path.exists() or not lbl_path.exists():
            continue
            
        logits = np.load(out_path)
        labels = np.load(lbl_path)
        preds = np.argmax(logits, axis=1)
        
        # Append performance metrics for this fold
        rows.append({
            "type": "fusion",
            "fold": fold,
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, average="macro"),
        })
        
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No model outputs found. Please run training (step 2-1) first.")

# Drop unnamed columns if any
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

# Define target metrics for aggregation
metrics = ['accuracy', 'precision', 'recall', 'f1']
type_list = df["type"].unique()

# Generate and save summary statistics (Mean and Std)
summary_df = df.groupby("type")[metrics].agg(['mean', 'std'])
summary_df.columns = [f"{metric}_{stat}" for metric in metrics for stat in ['mean', 'std']]
summary_df.to_csv("cv_metrics_summary.csv")
print("Performance summary saved to cv_metrics_summary.csv")

# Visualization Configuration
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

# Create bar plots for each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))

    # Prepare data for plotting (long format)
    plot_data = []
    for t in type_list:
        vals = df[df["type"] == t][metric].values
        for fold_idx, val in enumerate(vals):
            plot_data.append({
                'Fold': f"Fold {fold_idx+1}",
                'Value': val,
                'Type': t
            })
    plot_df = pd.DataFrame(plot_data)

    # Plot metrics across folds
    sns.barplot(data=plot_df, x="Fold", y="Value", hue="Type", palette=palette)

    # Labeling and Formatting
    plt.title(f"{metric.capitalize()} Across Folds", fontsize=16, fontweight='bold')
    plt.xlabel("CV Fold", fontsize=14, fontweight='bold')
    plt.ylabel(metric.capitalize(), fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend(title=None, fontsize=12)

    plt.tight_layout()
    output_fig = f"{metric}_fold_barplot.png"
    plt.savefig(output_fig, dpi=300)
    plt.close()

print("Metric visualization plots completed.")
