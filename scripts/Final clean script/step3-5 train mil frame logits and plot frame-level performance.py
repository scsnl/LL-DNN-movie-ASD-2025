"""
Step 3-5: MIL Frame-Level Performance Analysis

This script focuses on the Multi-Instance Learning (MIL) path to evaluate 
prediction accuracy at each individual timepoint (frame). 
Analysis steps:
1. Re-trains or loads models to output frame-level classification logits.
2. Aggregates logits across subjects and folds.
3. Computes the group-level accuracy bias (ASD classification probability) 
   at every TR.
4. Generates temporal accuracy curves to identify windows of high discriminative power.
"""

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
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import warnings
warnings.filterwarnings("ignore")

# Config for data paths
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")

if TEST_MODE:
    print("[TEST_MODE] Skipping heavy frame-level re-training and evaluation.")
    raise SystemExit(0)

# ---------------- Load & Preprocess ----------------
data_path = os.path.join(TEST_DATA_DIR, 'combined_asd_td_movieDM_data.pklz') if TEST_DATA_DIR else './combined_asd_td_movieDM_data.pklz'
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
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
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
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def main():
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
            ],
            logger=False
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

        probs = softmax(torch.tensor(frame_logits_all), dim=2).numpy()
        preds = np.argmax(probs, axis=-1)
        asd_idx = labels_all == 0
        tdc_idx = labels_all == 1
        acc_asd = np.mean(preds[asd_idx] == labels_all[asd_idx, None], axis=0)
        acc_tdc = np.mean(preds[tdc_idx] == labels_all[tdc_idx, None], axis=0)
        all_preds_asd.append(acc_asd)
        all_preds_tdc.append(acc_tdc)

    mean_acc_asd = np.mean(all_preds_asd, axis=0)
    mean_acc_tdc = np.mean(all_preds_tdc, axis=0)
    accuracy_diff = mean_acc_asd - mean_acc_tdc

    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_diff, color='purple', label='Δ Predicted P[ASD]')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Timepoint')
    plt.ylabel('Δ Predicted [ASD]')
    plt.title('Difference in Accuracy: ASD vs TDC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('./mil_logits_per_fold/mil_accuracy_difference_curve.png', dpi=300)
    
    df_diff = pd.DataFrame({
        'timepoint': np.arange(len(accuracy_diff)),
        'accuracy_asd': mean_acc_asd,
        'accuracy_tdc': mean_acc_tdc,
        'accuracy_diff': accuracy_diff
    })
    df_diff.to_csv('./mil_logits_per_fold/mil_accuracy_difference_curve.csv', index=False)

if __name__ == "__main__":
    main()
