"""
step 2-2: Read nested_cv_summary.json from step 2-1 nested cv runner and plot 5-fold results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SUMMARY_PATH = os.environ.get(
    "NESTED_CV_SUMMARY",
    "nested_results/nested_cv_summary.json",
)
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
OUTPUT_DIR = "figures"


def main():
    if not os.path.exists(SUMMARY_PATH):
        if TEST_OUTPUT_DIR:
            dummy_path = os.path.join(TEST_OUTPUT_DIR, "nested_results", "nested_cv_summary.json")
            if not os.path.exists(dummy_path):
                print("[step 2-2] nested_cv_summary not found; skipping (run step 2-1 nested cv runner for full pipeline)")
                return
            path_to_use = dummy_path
        else:
            raise FileNotFoundError(
                f"nested_cv_summary.json not found at {SUMMARY_PATH}. "
                "Please run step 2-1 nested cv runner.py first."
            )
    else:
        path_to_use = SUMMARY_PATH

    with open(path_to_use, "r", encoding="utf-8") as f:
        summary = json.load(f)

    outer_results = summary["outer_results"]
    mean_acc = summary.get("mean_acc", np.mean([r["acc"] for r in outer_results]))
    std_acc = summary.get("std_acc", np.std([r["acc"] for r in outer_results]))
    mean_f1 = summary.get("mean_f1", np.mean([r["f1"] for r in outer_results]))
    std_f1 = summary.get("std_f1", np.std([r["f1"] for r in outer_results]))

    # Build plot data: 5 folds x 2 metrics (acc, f1)
    plot_data = []
    for r in outer_results:
        plot_data.append({"Fold": f"Fold {r['outer_fold']+1}", "Metric": "Accuracy", "Value": r["acc"]})
        plot_data.append({"Fold": f"Fold {r['outer_fold']+1}", "Metric": "F1 (macro)", "Value": r["f1"]})

    df = pd.DataFrame(plot_data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2", 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Fold", y="Value", hue="Metric", palette=palette, ax=ax)

    ax.set_title(
        f"Nested CV 5-Fold Results (Mean Acc: {mean_acc:.4f}±{std_acc:.4f}, F1: {mean_f1:.4f}±{std_f1:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Outer Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(title=None, fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "nested_cv_5fold_barplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"5折结果图已保存: {out_path}")

    # Save summary CSV
    summary_rows = [
        {"fold": r["outer_fold"], "accuracy": r["acc"], "f1": r["f1"]}
        for r in outer_results
    ]
    summary_rows.append({"fold": "mean±std", "accuracy": f"{mean_acc:.4f}±{std_acc:.4f}", "f1": f"{mean_f1:.4f}±{std_f1:.4f}"})
    csv_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, "nested_cv_metrics_summary.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"指标汇总已保存: {csv_path}")


if __name__ == "__main__":
    main()
