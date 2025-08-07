#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Mon May 19 15:41:33 2025

@author: leili
"""

# Part 1: Train model and save MIL frame_logits + labels per fold


import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
warnings.filterwarnings("ignore")

# ---------------- Load & Preprocess ----------------

data_path = '# TODO: specify your data path'

datao = pd.read_pickle(data_path)
datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
shapes = [np.asarray(d).shape for d in datao.data]
main_shape = max(set(shapes), key=shapes.count)
datao = datao.iloc[[i for i, s in enumerate(shapes) if s == main_shape]].reset_index(drop=True)
fmri_all = np.stack([np.asarray(d) for d in datao.data])
nan_subs = np.unique(np.argwhere(np.isnan(fmri_all))[:, 0])
datao = datao.drop(index=nan_subs).reset_index(drop=True)

fmri = np.stack([np.asarray(d) for d in datao.data])  # (N, T, R)

labels = datao['label'].apply(lambda x: 0 if x == 'asd' else 1).values
site = pd.get_dummies(datao['site'], dtype=float).values
gender = pd.get_dummies(datao['gender'], dtype=float).values
age = datao['age'].astype(float).values.reshape(-1, 1)
meta = np.concatenate([site, gender, age], axis=1)

# ---------------- Model Definition ----------------

class DualPathNet(nn.Module):
    def __init__(self, metadata_dim=6, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv1 = nn.Sequential(nn.Conv1d(246, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.ReLU())
        self.out_mean = nn.Linear(512 + metadata_dim, 2)
        self.frame_fc = nn.Linear(512 + metadata_dim, 2)
        self.attn_fc = nn.Linear(512 + metadata_dim, 1)

    def forward(self, x, meta):
        B, T, R = x.shape
        x = x.reshape(-1, R).unsqueeze(1).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(-1).reshape(B, T, -1)
        x = self.dropout(x)
        meta_exp = meta.unsqueeze(1).expand(-1, T, -1)
        x_cat = torch.cat([x, meta_exp], dim=-1)
        x_mean = torch.mean(x, dim=1)
        feat_mean = torch.cat([x_mean, meta], dim=1)
        out_mean = self.out_mean(feat_mean)
        frame_logits = self.frame_fc(x_cat)
        attn_scores = self.attn_fc(x_cat)
        attn_weights = torch.softmax(torch.clamp(attn_scores, -30, 30), dim=1)
        mil_out = torch.sum(attn_weights * frame_logits, dim=1)
        return out_mean, mil_out, attn_weights, frame_logits

# ---------------- Lightning Wrapper ----------------

class DualPathPL(pl.LightningModule):
    def __init__(self, metadata_dim, lr=1e-4, alpha=0.5, dropout_rate=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = DualPathNet(metadata_dim, dropout_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, meta):
        return self.model(x, meta)

    def training_step(self, batch, batch_idx):
        x, meta, y = batch
        out_mean, mil_out, _, _ = self(x, meta)
        loss = self.hparams.alpha * self.criterion(out_mean, y) + \
               (1 - self.hparams.alpha) * self.criterion(mil_out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        out_mean, mil_out, _, _ = self(x, meta)
        fusion_logits = self.hparams.alpha * out_mean + (1 - self.hparams.alpha) * mil_out
        pred = torch.argmax(fusion_logits, dim=1)
        acc = accuracy_score(y.cpu(), pred.cpu())
        f1 = f1_score(y.cpu(), pred.cpu(), average='macro')
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ---------------- 5-Fold Training & Save MIL Outputs ----------------

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_preds_asd, all_preds_tdc = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(fmri, labels)):
    print(f"\n=== Fold {fold + 1}/5 ===")
    x_train, x_val = fmri[train_idx], fmri[val_idx]
    x_meta_train, x_meta_val = meta[train_idx], meta[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                             torch.tensor(x_meta_train, dtype=torch.float32),
                             torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                           torch.tensor(x_meta_val, dtype=torch.float32),
                           torch.tensor(y_val))

    config = {'lr': 1e-5, 'alpha': 0.4, 'dropout_rate': 0.5, 'batch_size': 16, 'max_epochs': 50, 'patience': 5}
    ckpt_dir = f"./checkpoints/fold{fold}"
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_dir, filename="dualpath-fold-best", monitor="val_acc", mode="max"),
            EarlyStopping(monitor="val_acc", mode="max", patience=config['patience'])
        ]
    )

    model = DualPathPL(metadata_dim=meta.shape[1], lr=config['lr'], alpha=config['alpha'], dropout_rate=config['dropout_rate'])
    trainer.fit(model, DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True),
                      DataLoader(val_ds, batch_size=config['batch_size']))

    best_model = DualPathPL.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path,
        metadata_dim=meta.shape[1], lr=config['lr'], alpha=config['alpha'], dropout_rate=config['dropout_rate']
    )
    best_model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    frame_logits_all = []
    labels_all = []

    with torch.no_grad():
        for xb, mb, yb in DataLoader(val_ds, batch_size=32):
            xb, mb = xb.to(best_model.device), mb.to(best_model.device)
            _, _, _, frame_logits = best_model(xb, mb)
            frame_logits_all.append(frame_logits.cpu().numpy())
            labels_all.append(yb.cpu().numpy())

    frame_logits_all = np.concatenate(frame_logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    out_dir = "./mil_logits_per_fold"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"frame_logits_fold{fold}.npy"), frame_logits_all)
    np.save(os.path.join(out_dir, f"labels_fold{fold}.npy"), labels_all)
    print(f" Saved MIL frame_logits and labels for fold {fold} to {out_dir}")

    # Group-level accuracy per timepoint

    probs = softmax(torch.tensor(frame_logits_all), dim=2).numpy()
    preds = np.argmax(probs, axis=-1)
    asd_idx = labels_all == 0
    tdc_idx = labels_all == 1
    acc_asd = np.mean(preds[asd_idx] == labels_all[asd_idx, None], axis=0)
    acc_tdc = np.mean(preds[tdc_idx] == labels_all[tdc_idx, None], axis=0)
    all_preds_asd.append(acc_asd)
    all_preds_tdc.append(acc_tdc)

# Plot combined accuracy curves

mean_acc_asd = np.mean(all_preds_asd, axis=0)
mean_acc_tdc = np.mean(all_preds_tdc, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(mean_acc_asd, label='ASD', color='blue')
plt.plot(mean_acc_tdc, label='TDC', color='orange')
plt.axhline(0.5, color='gray', linestyle='--')
plt.xlabel('Timepoint')
plt.ylabel('Accuracy')
plt.title('Timepoint-wise MIL Frame Classification Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig("./mil_logits_per_fold/mil_group_accuracy_by_time.png", dpi=300)
plt.show()


# ---------------- Group-level Predicted P[ASD] Difference Analysis ----------------

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

all_probs_asd, all_probs_tdc = [], []

# Load saved logits per fold

for fold in range(5):
    probs = np.load(f"# TODO: specify your data path")  # (N, T, 2)

    labels_fold = np.load(f"# TODO: specify your data path")  # (N,)

    probs = softmax(torch.tensor(probs), dim=2).numpy()  # (N, T, 2)

    
    asd_idx = labels_fold == 0
    tdc_idx = labels_fold == 1
    all_probs_asd.append(probs[asd_idx, :, 0])  # class=0 -> ASD

    all_probs_tdc.append(probs[tdc_idx, :, 0])

# Stack across folds

probs_asd_all = np.vstack(all_probs_asd)  # (n_asd, T)

probs_tdc_all = np.vstack(all_probs_tdc)  # (n_tdc, T)


mean_asd = np.mean(probs_asd_all, axis=0)
mean_tdc = np.mean(probs_tdc_all, axis=0)
se_asd = np.std(probs_asd_all, axis=0) / np.sqrt(probs_asd_all.shape[0])
se_tdc = np.std(probs_tdc_all, axis=0) / np.sqrt(probs_tdc_all.shape[0])

# Difference & standard error

diff_curve = mean_asd - mean_tdc
se_diff = np.sqrt(se_asd**2 + se_tdc**2)

# Statistical test

tvals, pvals = ttest_ind(probs_asd_all, probs_tdc_all, axis=0, equal_var=False)
rejected, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

# Plot

plt.figure(figsize=(10, 5))
plt.plot(diff_curve, label="P[ASD]", color="purple")
plt.fill_between(range(len(diff_curve)), diff_curve - se_diff, diff_curve + se_diff, alpha=0.3, color="purple")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Timepoint")
plt.ylabel(" Predicted [ASD]")
plt.title("Timepoint-wise Prediction Bias: ASD vs TDC")
plt.legend()
plt.tight_layout()
plt.savefig("./mil_logits_per_fold/mil_confidence_diff_with_significance.png", dpi=300)
plt.show()

# Save CSV

df = pd.DataFrame({
    "timepoint": np.arange(len(diff_curve)),
    "mean_asd": mean_asd,
    "mean_tdc": mean_tdc,
    "diff": diff_curve,
    "se_diff": se_diff,
    "pval": pvals,
    "pval_fdr": pvals_fdr,
    "sig_fdr": rejected
})
df.to_csv("# TODO: specify your data path", index=False)






import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from torch.nn.functional import softmax

# ---------------- Accuracy  ----------------

mean_acc_asd = np.mean(all_preds_asd, axis=0)
mean_acc_tdc = np.mean(all_preds_tdc, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(mean_acc_asd, label='ASD', color='blue')
plt.plot(mean_acc_tdc, label='TDC', color='orange')
plt.axhline(0.5, color='gray', linestyle='--')
plt.xlabel('Timepoint')
plt.ylabel('Accuracy')
plt.title('Timepoint-wise MIL Frame Classification Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("./mil_logits_per_fold/mil_group_accuracy_by_time.png", dpi=300)
plt.close()


# ---------------- Accuracy Difference Curve (ASD - TDC) ----------------

accuracy_diff = mean_acc_asd - mean_acc_tdc

plt.figure(figsize=(10, 5))
plt.plot(accuracy_diff, color='purple', label=' Predicted P[ASD]')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Timepoint')
plt.ylabel(' Predicted [ASD]')
plt.title('Difference in Accuracy: ASD vs TDC')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('./mil_logits_per_fold/mil_accuracy_difference_curve.png', dpi=300)
plt.show()

# Save as CSV

df_diff = pd.DataFrame({
    'timepoint': np.arange(len(accuracy_diff)),
    'accuracy_asd': mean_acc_asd,
    'accuracy_tdc': mean_acc_tdc,
    'accuracy_diff': accuracy_diff
})
df_diff.to_csv('# TODO: specify your data path', index=False)

