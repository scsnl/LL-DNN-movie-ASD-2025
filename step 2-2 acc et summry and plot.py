import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===== Step 1:  meanstd  =====

df = pd.read_csv("# TODO: specify your data path")

df = df.loc[:, ~df.columns.str.contains("Unnamed")]

metrics = ['accuracy', 'precision', 'recall', 'f1']
type_list = df["type"].unique()

#  mean  std

summary_df = df.groupby("type")[metrics].agg(['mean', 'std'])
summary_df.columns = [f"{metric}_{stat}" for metric in metrics for stat in ['mean', 'std']]
summary_df.to_csv("cv_metrics_summary.csv")
print(" Saved summary to cv_metrics_summary.csv")

# ===== Step 2: Fold Type  =====

sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

for metric in metrics:
    plt.figure(figsize=(10, 6))

    # 

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

    # 

    sns.barplot(data=plot_df, x="Fold", y="Value", hue="Type", palette=palette)

    # 

    plt.title(f"{metric.capitalize()} Across Folds by Type", fontsize=16, fontweight='bold')
    plt.xlabel("Fold", fontsize=14, fontweight='bold')
    plt.ylabel(metric.capitalize(), fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend(title=None, fontsize=12)  #  


    plt.tight_layout()
    plt.savefig(f"{metric}_fold_barplot_bold.png", dpi=300)
    plt.close()

print(" All plots saved without legend title, fonts bolded.")
