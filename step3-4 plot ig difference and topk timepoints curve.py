#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Mon May 19 15:39:07 2025

@author: leili
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# -------- CONFIG --------

n_folds = 5
output_dir = "./ig_group_level_results"
os.makedirs(output_dir, exist_ok=True)
save_fig = True
save_csv = True

# -------- Load all folds --------

curve_out_asd, curve_out_tdc = [], []
curve_mil_asd, curve_mil_tdc = [], []

for fold in range(n_folds):
    fold_dir = f"./saved_models/fold{fold}"
    out_path = os.path.join(fold_dir, "ig_curve_outmean.npy")
    mil_path = os.path.join(fold_dir, "ig_curve_milout.npy")
    label_path = os.path.join(fold_dir, "labels.npy")

    if not all(os.path.exists(p) for p in [out_path, mil_path, label_path]):
        print(f"Skipping fold {fold}: missing files")
        continue

    out_curve = np.load(out_path)
    mil_curve = np.load(mil_path)
    labels = np.load(label_path)

    curve_out_asd.append(out_curve[labels == 0])
    curve_out_tdc.append(out_curve[labels == 1])
    curve_mil_asd.append(mil_curve[labels == 0])
    curve_mil_tdc.append(mil_curve[labels == 1])

curve_out_asd = np.vstack(curve_out_asd)
curve_out_tdc = np.vstack(curve_out_tdc)
curve_mil_asd = np.vstack(curve_mil_asd)
curve_mil_tdc = np.vstack(curve_mil_tdc)

# -------- Plot: Top 10% group difference (p-value) --------

def plot_top_diff(asd_curve, tdc_curve, title, save_name, top_pct=10):
    mean_asd = np.mean(asd_curve, axis=0)
    mean_tdc = np.mean(tdc_curve, axis=0)
    std_asd = np.std(asd_curve, axis=0)
    std_tdc = np.std(tdc_curve, axis=0)
    T = mean_asd.shape[0]

    p_vals = np.array([ttest_ind(asd_curve[:, t], tdc_curve[:, t])[1] for t in range(T)])
    _, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")

    n_top = max(1, int(T * (top_pct / 100)))
    top_diff_idx = np.argsort(p_vals)[:n_top]

    plt.figure(figsize=(12, 6))
    plt.plot(mean_asd, label="ASD", color="red")
    plt.fill_between(range(T), mean_asd - std_asd, mean_asd + std_asd, color="red", alpha=0.2)
    plt.plot(mean_tdc, label="TDC", color="blue")
    plt.fill_between(range(T), mean_tdc - std_tdc, mean_tdc + std_tdc, color="blue", alpha=0.2)
    for t in top_diff_idx:
        plt.axvline(x=t, color="red", linestyle="--", alpha=0.4)
    plt.title(title)
    plt.xlabel("Timepoint")
    plt.ylabel("IG Attribution (|mean|)")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(output_dir, f"{save_name}_topdiff2.png"), dpi=300)
    plt.show()

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
    print(f"[Saved] {save_name}_topdiff with {len(top_diff_idx)} red lines.")

# -------- Plot: Top 20% attention per group --------

def plot_top_attention(asd_curve, tdc_curve, title, save_name, top_pct=20):
    mean_asd = np.mean(asd_curve, axis=0)
    mean_tdc = np.mean(tdc_curve, axis=0)
    std_asd = np.std(asd_curve, axis=0)
    std_tdc = np.std(asd_curve, axis=0)
    T = mean_asd.shape[0]

    n_top = max(1, int(T * (top_pct / 100)))
    top_asd = np.argsort(mean_asd)[-n_top:]
    top_tdc = np.argsort(mean_tdc)[-n_top:]

    plt.figure(figsize=(12, 6))
    plt.plot(mean_asd, label="ASD", color="blue")
    plt.fill_between(range(T), mean_asd - std_asd, mean_asd + std_asd, color="blue", alpha=0.2)
    plt.plot(mean_tdc, label="TDC", color="orange")
    plt.fill_between(range(T), mean_tdc - std_tdc, mean_tdc + std_tdc, color="orange", alpha=0.2)

    for t in top_asd:
        plt.axvline(x=t, color="blue", linestyle="--", alpha=0.3)
    for t in top_tdc:
        plt.axvline(x=t, color="orange", linestyle="--", alpha=0.3)

    plt.title(title)
    plt.xlabel("Timepoint")
    plt.ylabel("IG Attribution")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(output_dir, f"{save_name}_topattn2.png"), dpi=300)
    plt.show()

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
    print(f"[Saved] {save_name}_topattn with {len(top_asd)} blue, {len(top_tdc)} orange lines.")

# -------- Run: Mean path --------

plot_top_diff(curve_out_asd, curve_out_tdc,
              title="Top 10% Group Difference (Mean Path)",
              save_name="ig_curve_group_outmean")

plot_top_attention(curve_out_asd, curve_out_tdc,
                   title="Top 20% Attention (Mean Path)",
                   save_name="ig_curve_group_outmean")

# -------- Run: MIL path --------

plot_top_diff(curve_mil_asd, curve_mil_tdc,
              title="Top 10% Group Difference (MIL Path)",
              save_name="ig_curve_group_mil")

plot_top_attention(curve_mil_asd, curve_mil_tdc,
                   title="Top 20% Attention (MIL Path)",
                   save_name="ig_curve_group_mil")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# -------- CONFIG --------

n_folds = 5
output_dir = "./ig_group_level_results"
os.makedirs(output_dir, exist_ok=True)
save_fig = True
save_csv = True

# -------- Load all folds --------

curve_out_asd, curve_out_tdc = [], []
curve_mil_asd, curve_mil_tdc = [], []

for fold in range(n_folds):
    fold_dir = f"./saved_models/fold{fold}"
    out_path = os.path.join(fold_dir, "ig_curve_outmean.npy")
    mil_path = os.path.join(fold_dir, "ig_curve_milout.npy")
    label_path = os.path.join(fold_dir, "labels.npy")

    if not all(os.path.exists(p) for p in [out_path, mil_path, label_path]):
        print(f"Skipping fold {fold}: missing files")
        continue

    out_curve = np.load(out_path)
    mil_curve = np.load(mil_path)
    labels = np.load(label_path)

    curve_out_asd.append(out_curve[labels == 0])
    curve_out_tdc.append(out_curve[labels == 1])
    curve_mil_asd.append(mil_curve[labels == 0])
    curve_mil_tdc.append(mil_curve[labels == 1])

curve_out_asd = np.vstack(curve_out_asd)
curve_out_tdc = np.vstack(curve_out_tdc)
curve_mil_asd = np.vstack(curve_mil_asd)
curve_mil_tdc = np.vstack(curve_mil_tdc)

# -------- Plot: Top 10% group difference (p-value) --------

def plot_top_diff(asd_curve, tdc_curve, title, save_name, top_pct=10):
    mean_asd = np.mean(asd_curve, axis=0)
    mean_tdc = np.mean(tdc_curve, axis=0)
    T = mean_asd.shape[0]

    p_vals = np.array([ttest_ind(asd_curve[:, t], tdc_curve[:, t])[1] for t in range(T)])
    _, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")

    n_top = max(1, int(T * (top_pct / 100)))
    top_diff_idx = np.argsort(p_vals)[:n_top]

    plt.figure(figsize=(12, 6))
    plt.plot(mean_asd, label="ASD", color="red")
    plt.plot(mean_tdc, label="TDC", color="blue")
    for t in top_diff_idx:
        plt.axvline(x=t, color="red", linestyle="--", alpha=0.4)
    plt.title(title)
    plt.xlabel("Timepoint")
    plt.ylabel("IG Attribution (|mean|)")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(output_dir, f"{save_name}_topdiff.png"), dpi=300)
    plt.show()

    if save_csv:
        df = pd.DataFrame({
            "timepoint": np.arange(T),
            "mean_asd": mean_asd,
            "mean_tdc": mean_tdc,
            "p_raw": p_vals,
            "p_fdr": p_fdr,
            "is_topdiff": np.isin(np.arange(T), top_diff_idx)
        })
        df.to_csv(os.path.join(output_dir, f"{save_name}_topdiff.csv"), index=False)
    print(f"[Saved] {save_name}_topdiff with {len(top_diff_idx)} red lines.")

# -------- Plot: Top 20% attention per group --------

def plot_top_attention(asd_curve, tdc_curve, title, save_name, top_pct=20):
    mean_asd = np.mean(asd_curve, axis=0)
    mean_tdc = np.mean(tdc_curve, axis=0)
    T = mean_asd.shape[0]

    n_top = max(1, int(T * (top_pct / 100)))
    top_asd = np.argsort(mean_asd)[-n_top:]
    top_tdc = np.argsort(mean_tdc)[-n_top:]

    plt.figure(figsize=(12, 6))
    plt.plot(mean_asd, label="ASD", color="blue")
    plt.plot(mean_tdc, label="TDC", color="orange")

    for t in top_asd:
        plt.axvline(x=t, color="blue", linestyle="--", alpha=0.3)
    for t in top_tdc:
        plt.axvline(x=t, color="orange", linestyle="--", alpha=0.3)

    plt.title(title)
    plt.xlabel("Timepoint")
    plt.ylabel("IG Attribution")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(output_dir, f"{save_name}_topattn.png"), dpi=300)
    plt.show()

    if save_csv:
        df = pd.DataFrame({
            "timepoint": np.arange(T),
            "mean_asd": mean_asd,
            "mean_tdc": mean_tdc,
            "is_top_asd": np.isin(np.arange(T), top_asd),
            "is_top_tdc": np.isin(np.arange(T), top_tdc)
        })
        df.to_csv(os.path.join(output_dir, f"{save_name}_topattn.csv"), index=False)
    print(f"[Saved] {save_name}_topattn with {len(top_asd)} blue, {len(top_tdc)} orange lines.")

# -------- Run: Mean path --------

plot_top_diff(curve_out_asd, curve_out_tdc,
              title="Top 10% Group Difference (Mean Path)",
              save_name="ig_curve_group_outmean")

plot_top_attention(curve_out_asd, curve_out_tdc,
                   title="Top 20% Attention (Mean Path)",
                   save_name="ig_curve_group_outmean")

# -------- Run: MIL path --------

plot_top_diff(curve_mil_asd, curve_mil_tdc,
              title="Top 10% Group Difference (MIL Path)",
              save_name="ig_curve_group_mil")

plot_top_attention(curve_mil_asd, curve_mil_tdc,
                   title="Top 20% Attention (MIL Path)",
                   save_name="ig_curve_group_mil")
