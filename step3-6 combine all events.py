import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== CONFIG ==========

mil_dir = "./mil_logits_per_fold"
group_dir = "./ig_group_level_results"
save_path = "./figures/combined_accuracy_bias_with_events.png"
fig_path = "./figures/"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ========== Load All Required Data ==========

#  accuracy curve

df_acc = pd.read_csv(os.path.join(mil_dir, "mil_accuracy_difference_curve.csv"))

# IG group-level results

df_ig = pd.read_csv(os.path.join(group_dir, "ig_curve_group_mil_topattn.csv"))

# attention significance from attn_significance.csv

df_attn = pd.read_csv(os.path.join(fig_path, "attn_significance.csv"))

# ========== Check Alignment ==========

assert all(df_acc['timepoint'] == df_ig['timepoint']), "Mismatch: accuracy vs IG timepoints"
assert all(df_acc['timepoint'] == df_attn['timepoint']), "Mismatch: accuracy vs attention timepoints"

# ========== Extract Data ==========

time = df_acc['timepoint']
diff = df_acc['accuracy_diff']

asd_ig_top = df_ig[df_ig['is_top_asd']]['timepoint'].values
tdc_ig_top = df_ig[df_ig['is_top_tdc']]['timepoint'].values
attn_sig = df_attn[df_attn['sig_raw']]['timepoint'].values

# ========== Plot ==========

fig, ax = plt.subplots(figsize=(14, 5))

#  Accuracy curve

ax.plot(time, diff, label=" Accuracy (ASD - TDC)", color="purple", linewidth=2)

# Add IG highlights

for tp in asd_ig_top:
    ax.axvline(x=tp, color='mediumblue', linestyle='-', alpha=0.4, linewidth=1)
for tp in tdc_ig_top:
    ax.axvline(x=tp, color='darkorange', linestyle='-', alpha=0.4, linewidth=1)

# Add attention significance

for tp in attn_sig:
    ax.axvline(x=tp, color='forestgreen', linestyle='--', alpha=0.6, linewidth=1)

# Reference line

ax.axhline(0, color='gray', linestyle='--', linewidth=1)

# Labels and legend

ax.set_xlabel("Timepoint", fontsize=12)
ax.set_ylabel(" Predicted [ASD]", fontsize=12)
ax.set_title("Timepoint-wise Accuracy Bias with IG & Attention Highlights", fontsize=14)

# Custom legend

from matplotlib.patches import Patch
custom_lines = [
    Patch(facecolor='mediumblue', edgecolor='mediumblue', alpha=0.4, label='ASD Top 20% IG'),
    Patch(facecolor='darkorange', edgecolor='darkorange', alpha=0.4, label='TDC Top 20% IG'),
    Patch(facecolor='forestgreen', edgecolor='forestgreen', alpha=0.6, label='Attention Significant (p < 0.05)'),
    plt.Line2D([0], [0], color='purple', lw=2, label=' Accuracy (ASD - TDC)')
]
ax.legend(handles=custom_lines, loc='best', frameon=True)

# Grid and layout

ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

# ========== Print Overlaps ==========

overlap_asd = np.intersect1d(asd_ig_top, attn_sig)
overlap_tdc = np.intersect1d(tdc_ig_top, attn_sig)
print(f"Overlap Timepoints (ASD IG  Attention sig_raw): {overlap_asd.tolist()}")
print(f"Overlap Timepoints (TDC IG  Attention sig_raw): {overlap_tdc.tolist()}")
