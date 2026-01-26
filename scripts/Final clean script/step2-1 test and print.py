"""
Step 2-1 (Evaluation): Cross-Validation Performance Summary

This script provides a quick summary of model performance after training. 
It performs the following:
1. Iterates through the saved model output files for each of the 5 folds.
2. Calculates classification metrics (Accuracy, Precision, Recall, F1) for each fold.
3. Computes and prints the average performance and standard deviation.
4. Generates a consolidated CSV file containing all fold-wise and mean metrics.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_saved_predictions(folds_dir='saved_models'):
    # Initialize metrics container
    metrics = {'fold': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # Iterate through each cross-validation fold
    for fold in range(5):
        fold_dir = os.path.join(folds_dir, f"fold{fold}")
        out_path = os.path.join(fold_dir, "out_mean.npy")
        label_path = os.path.join(fold_dir, "labels.npy")

        # Check if fold data exists
        if not os.path.exists(out_path) or not os.path.exists(label_path):
            print(f"Skipping fold {fold}: missing saved data.")
            continue

        # Load logits and true labels
        logits = np.load(out_path)
        labels = np.load(label_path)
        # Classify based on highest logit
        preds = np.argmax(logits, axis=1)

        # Calculate metrics (Binary classification ASD vs TD)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average='binary', zero_division=0)
        rec = recall_score(labels, preds, average='binary', zero_division=0)
        f1 = f1_score(labels, preds, average='binary', zero_division=0)

        # Store results
        metrics['fold'].append(fold)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)

        print(f"Fold {fold} | Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    if not metrics['fold']:
        print("No evaluation data found. Run training script first.")
        return

    # Calculate aggregate statistics
    df = pd.DataFrame(metrics)
    mean_vals = df.mean(numeric_only=True)
    std_vals = df.std(numeric_only=True)

    print("\n" + "="*40)
    print("AVERAGE PERFORMANCE (5 FOLDS):")
    print("="*40)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"{metric.capitalize():<10}: {mean_vals[metric]:.4f} ± {std_vals[metric]:.4f}")

    # Prepare summary for CSV export
    summary_row = {
        'fold': 'mean ± std',
        'accuracy': f"{mean_vals['accuracy']:.4f} ± {std_vals['accuracy']:.4f}",
        'precision': f"{mean_vals['precision']:.4f} ± {std_vals['precision']:.4f}",
        'recall': f"{mean_vals['recall']:.4f} ± {std_vals['recall']:.4f}",
        'f1': f"{mean_vals['f1']:.4f} ± {std_vals['f1']:.4f}"
    }

    # Concatenate individual results with summary row
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save to file
    os.makedirs("figures", exist_ok=True)
    output_path = os.path.join("figures", "cv_eval_metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"\nConsolidated results saved to: {output_path}")

    return df, mean_vals, std_vals

if __name__ == "__main__":
    # Use environment variable if provided (for test runs)
    folds_dir = os.environ.get("TEST_OUTPUT_DIR", "")
    folds_root = os.path.join(folds_dir, "saved_models") if folds_dir else "saved_models"
    
    evaluate_saved_predictions(folds_dir=folds_root)
