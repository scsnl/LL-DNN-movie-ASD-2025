"""
Step 4: Feature Embedding Visualization (UMAP/t-SNE)

This script generates 2D embeddings of the high-dimensional latent features 
learned by the DualPathNet model. 
Objectives:
1. Load features from the Global Mean path and MIL Attention path for all subjects.
2. Concatenate features from all cross-validation folds.
3. Apply UMAP (Uniform Manifold Approximation and Projection) and t-SNE 
   (t-distributed Stochastic Neighbor Embedding) to reduce dimensionality.
4. Visualize group separation (ASD vs TDC) in the feature space.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Check for test mode to avoid time-consuming manifold learning
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
if TEST_MODE:
    print("[TEST_MODE] Skipping manifold embedding visualization.")
    raise SystemExit(0)

# Import UMAP only when needed
try:
    import umap
except ImportError:
    umap = None

def plot_umap(X, y, title, save_path):
    if umap is None:
        print("UMAP not installed, skipping UMAP plot.")
        return
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_embedded = reducer.fit_transform(X)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.scatter(X_embedded[np.array(y) == 0, 0], X_embedded[np.array(y) == 0, 1], 
                label='ASD', alpha=0.7, color='#d62728', edgecolor='white', linewidth=0.5)
    plt.scatter(X_embedded[np.array(y) == 1, 0], X_embedded[np.array(y) == 1, 1], 
                label='TDC', alpha=0.7, color='#1f77b4', edgecolor='white', linewidth=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP-1', fontsize=12); plt.ylabel('UMAP-2', fontsize=12)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout()
    plt.savefig(save_path); plt.close()

def plot_tsne(X, y, title, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.scatter(X_embedded[np.array(y) == 0, 0], X_embedded[np.array(y) == 0, 1], 
                label='ASD', alpha=0.7, color='#d62728', edgecolor='white', linewidth=0.5)
    plt.scatter(X_embedded[np.array(y) == 1, 0], X_embedded[np.array(y) == 1, 1], 
                label='TDC', alpha=0.7, color='#1f77b4', edgecolor='white', linewidth=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE-1', fontsize=12); plt.ylabel('t-SNE-2', fontsize=12)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout()
    plt.savefig(save_path); plt.close()

def main():
    save_dir = "./saved_models"
    output_dir = "./feature_embeddings"
    os.makedirs(output_dir, exist_ok=True)

    feat_mean_all, attn_feat_all, fused_feat_all, label_all = [], [], [], []

    for fold in range(5):
        fold_path = os.path.join(save_dir, f"fold{fold}")
        mean_path = os.path.join(fold_path, "feat_mean.npy")
        attn_path = os.path.join(fold_path, "attn_feat.npy")
        label_path = os.path.join(fold_path, "labels.npy")

        if all(os.path.exists(p) for p in [mean_path, attn_path, label_path]):
            mean_feat = np.load(mean_path)
            attn_feat = np.load(attn_path)
            labels = np.load(label_path)
            feat_mean_all.append(mean_feat)
            attn_feat_all.append(attn_feat)
            fused_feat_all.append(np.concatenate([mean_feat, attn_feat], axis=1))
            label_all.append(labels)

    if not feat_mean_all:
        print("No feature files found.")
        return

    feat_mean_all = np.concatenate(feat_mean_all, axis=0)
    attn_feat_all = np.concatenate(attn_feat_all, axis=0)
    fused_feat_all = np.concatenate(fused_feat_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)

    plot_umap(feat_mean_all, label_all, "Global Mean Features (UMAP)", os.path.join(output_dir, "feat_mean_umap.png"))
    plot_umap(attn_feat_all, label_all, "MIL Attention Features (UMAP)", os.path.join(output_dir, "attn_feat_umap.png"))
    plot_umap(fused_feat_all, label_all, "Fused Features (UMAP)", os.path.join(output_dir, "fused_feat_umap.png"))
    
    plot_tsne(feat_mean_all, label_all, "Global Mean Features (t-SNE)", os.path.join(output_dir, "feat_mean_tsne.png"))
    plot_tsne(attn_feat_all, label_all, "MIL Attention Features (t-SNE)", os.path.join(output_dir, "attn_feat_tsne.png"))
    plot_tsne(fused_feat_all, label_all, "Fused Features (t-SNE)", os.path.join(output_dir, "fused_feat_tsne.png"))

if __name__ == "__main__":
    main()
