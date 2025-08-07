#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Mon May 19 15:45:24 2025

@author: leili
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

# ========= CONFIG ========= #

save_dir = "./saved_models"
n_folds = 5
output_dir = "./feature_embeddings"
os.makedirs(output_dir, exist_ok=True)

# ========= INIT ========= #

feat_mean_all, attn_feat_all, fused_feat_all, label_all = [], [], [], []

# ========= Load Saved Features ========= #

for fold in range(n_folds):
    fold_path = os.path.join(save_dir, f"fold{fold}")
    mean_path = os.path.join(fold_path, "feat_mean.npy")
    attn_path = os.path.join(fold_path, "attn_feat.npy")
    label_path = os.path.join(fold_path, "labels.npy")

    if os.path.exists(mean_path) and os.path.exists(attn_path) and os.path.exists(label_path):
        mean_feat = np.load(mean_path)
        attn_feat = np.load(attn_path)
        labels = np.load(label_path)
        fused_feat = np.concatenate([mean_feat, attn_feat], axis=1)

        feat_mean_all.append(mean_feat)
        attn_feat_all.append(attn_feat)
        fused_feat_all.append(fused_feat)
        label_all.append(labels)
    else:
        print(f"Missing data in fold {fold}, skipping.")

# ========= Concatenate All Folds ========= #

feat_mean_all = np.concatenate(feat_mean_all, axis=0)
attn_feat_all = np.concatenate(attn_feat_all, axis=0)
fused_feat_all = np.concatenate(fused_feat_all, axis=0)
label_all = np.concatenate(label_all, axis=0)

# ========= Visualization Functions ========= #

def plot_umap(X, y, title, save_name):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_embedded = reducer.fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_embedded[np.array(y) == 0, 0], X_embedded[np.array(y) == 0, 1], label='ASD', alpha=0.6)
    plt.scatter(X_embedded[np.array(y) == 1, 0], X_embedded[np.array(y) == 1, 1], label='TDC', alpha=0.6)
    plt.title(title)
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{save_name}_umap.png"), dpi=300)
    plt.close()

def plot_tsne(X, y, title, save_name, perplexity=30):
    X = np.array(X)
    y = np.array(y)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], label='ASD', alpha=0.6)
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], label='TDC', alpha=0.6)
    plt.title(title)
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{save_name}_tsne.png"), dpi=300)
    plt.close()

# ========= Plot All ========= #

plot_umap(feat_mean_all, label_all, "Mean Path Features (UMAP)", "feat_mean")
plot_umap(attn_feat_all, label_all, "Attention Path Features (UMAP)", "attn_feat")
plot_umap(fused_feat_all, label_all, "Fused Features (UMAP)", "fused_feat")

plot_tsne(feat_mean_all, label_all, "Mean Path Features (t-SNE)", "feat_mean")
plot_tsne(attn_feat_all, label_all, "Attention Path Features (t-SNE)", "attn_feat")
plot_tsne(fused_feat_all, label_all, "Fused Features (t-SNE)", "fused_feat")
