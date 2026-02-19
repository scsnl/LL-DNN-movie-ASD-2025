"""
Supply Step 2-1: Performance Evaluation of Time-Filtered Models

This script evaluates the classification performance of models trained on 
specific movie segments (seg1, seg2, seg3).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_segment(seg_root: Path):
    folds_dir = seg_root / 'saved_models'
    metrics = {'fold': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for fold in range(5):
        fold_dir = folds_dir / f"fold{fold}"
        out_path, lbl_path = fold_dir / "out_mean.npy", fold_dir / "labels.npy"
        if out_path.exists() and lbl_path.exists():
            logits, labels = np.load(out_path), np.load(lbl_path)
            preds = np.argmax(logits, axis=1)
            metrics['fold'].append(fold)
            metrics['accuracy'].append(accuracy_score(labels, preds))
            metrics['precision'].append(precision_score(labels, preds, average='binary', zero_division=0))
            metrics['recall'].append(recall_score(labels, preds, average='binary', zero_division=0))
            metrics['f1'].append(f1_score(labels, preds, average='binary', zero_division=0))
    if not metrics['fold']: return
    df = pd.DataFrame(metrics)
    summary = df.mean(numeric_only=True)
    print(f"\n[{seg_root.name}] Acc: {summary['accuracy']:.4f}, F1: {summary['f1']:.4f}")
    df.to_csv(seg_root / 'figures' / 'cv_eval_metrics.csv', index=False)

def main():
    base = Path('results')
    for seg in ['seg1', 'seg2', 'seg3']:
        if (base / seg).exists(): evaluate_segment(base / seg)

if __name__ == '__main__':
    main()
