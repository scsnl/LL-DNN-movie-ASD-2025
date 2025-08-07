#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Evaluate saved model outputs across 5 folds
Metrics: Accuracy, Precision, Recall, F1 + Mean and Std Summary
Save CSV with fold-wise and average  std summary
@author: leili
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_saved_predictions(folds_dir='saved_models'):
    metrics = {'fold': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold in range(5):
        fold_dir = os.path.join(folds_dir, f"fold{fold}")
        out_path = os.path.join(fold_dir, "out_mean.npy")
        label_path = os.path.join(fold_dir, "labels.npy")

        if not os.path.exists(out_path) or not os.path.exists(label_path):
            print(f"Missing data in fold {fold}, skipping...")
            continue

        logits = np.load(out_path)
        labels = np.load(label_path)
        preds = np.argmax(logits, axis=1)

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average='binary')
        rec = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')

        metrics['fold'].append(fold)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)

        print(f"Fold {fold} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Convert to DataFrame

    df = pd.DataFrame(metrics)
    mean_vals = df.mean(numeric_only=True)
    std_vals = df.std(numeric_only=True)

    print("\nAverage Performance Across 5 Folds:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"{metric.capitalize()}: {mean_vals[metric]:.4f}  {std_vals[metric]:.4f}")

    # Create summary row

    summary_row = {
        'fold': 'mean  std',
        'accuracy': f"{mean_vals['accuracy']:.4f}  {std_vals['accuracy']:.4f}",
        'precision': f"{mean_vals['precision']:.4f}  {std_vals['precision']:.4f}",
        'recall': f"{mean_vals['recall']:.4f}  {std_vals['recall']:.4f}",
        'f1': f"{mean_vals['f1']:.4f}  {std_vals['f1']:.4f}"
    }

    # Append and save

    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    os.makedirs("figures", exist_ok=True)
    df.to_csv("# TODO: specify your data path", index=False)


    return df, mean_vals, std_vals

# Run evaluation

if __name__ == "__main__":
    evaluate_saved_predictions()
