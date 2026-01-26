"""
Step 2-1: DualPathNet Training and Feature Attribution (IG)

This script implements the core training and interpretation pipeline:
1. Loads fMRI ROI time series and metadata (Age, Gender, Site).
2. Defines the DualPathNet architecture (Global Pooling Path + MIL Attention Path).
3. Performs Stratified 5-Fold Cross-Validation.
4. Saves the best model weights based on validation accuracy.
5. Performs feature attribution using Integrated Gradients (IG) to estimate 
   the importance of each ROI at each timepoint.
6. Exports model predictions, attention weights, and attribution maps for downstream analysis.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

# Configuration for test/dummy environments
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
DUMMY_MODE = os.environ.get("DUMMY_MODE", "0") == "1"

# Deep learning framework imports
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = nn = DataLoader = TensorDataset = None

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
except ImportError:
    pl = ModelCheckpoint = EarlyStopping = None

# Fallback stub for Pytorch Lightning if not installed (allows non-training code to run)
if pl is None:
    class _LightningModuleStub(object): pass
    class _PLStub(object): LightningModule = _LightningModuleStub
    pl = _PLStub()

warnings.filterwarnings("ignore")
device = torch.device("cuda" if (torch and torch.cuda.is_available() and not TEST_MODE) else "cpu") if torch else "cpu"

# XAI tool: Integrated Gradients for feature attribution
try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None

# Input path handling
data_path = (
    os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz")
    if TEST_DATA_DIR
    else "combined_asd_td_movieDM_data.pklz"
)

# Load and preprocess data (consistent with index generation)
datao = pd.read_pickle(data_path)
datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
shapes = [np.asarray(d).shape for d in datao.data]
main_shape = max(set(shapes), key=shapes.count)
datao = datao.iloc[[i for i, s in enumerate(shapes) if s == main_shape]].reset_index(drop=True)
fmri_all = np.stack([np.asarray(d) for d in datao.data])
nan_subs = np.unique(np.argwhere(np.isnan(fmri_all))[:, 0])
datao = datao.drop(index=nan_subs).reset_index(drop=True)
fmri = np.stack([np.asarray(d) for d in datao.data])

# Process labels and metadata (One-hot for site/gender, standardized age)
labels = datao['label'].apply(lambda x: 0 if x == 'asd' else 1).values
site = pd.get_dummies(datao['site'], dtype=float).values
gender = pd.get_dummies(datao['gender'], dtype=float).values
age = datao['age'].astype(float).values.reshape(-1, 1)
meta = np.concatenate([site, gender, age], axis=1)

# Model Definition: DualPathNet with Global pooling and Attention paths
class DualPathNet(nn.Module):
    def __init__(self, metadata_dim=6, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # 1D Convolutions to extract temporal features
        self.conv1 = nn.Sequential(nn.Conv1d(246, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.ReLU())
        
        # Prediction heads
        self.out_mean = nn.Linear(512 + metadata_dim, 2)  # Global Pool path
        self.frame_fc = nn.Linear(512 + metadata_dim, 2)  # MIL path
        self.attn_fc = nn.Linear(512 + metadata_dim, 1)   # Attention mechanism

    def forward(self, x, meta):
        B, T, R = x.shape
        x = x.reshape(-1, R).unsqueeze(1).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(-1).reshape(B, T, -1)
        x = self.dropout(x)
        
        # Fuse temporal features with subject metadata
        meta_exp = meta.unsqueeze(1).expand(-1, T, -1)
        x_cat = torch.cat([x, meta_exp], dim=-1)
        
        # Path 1: Mean pooling over time (Trait-level features)
        x_mean = torch.mean(x, dim=1)
        feat_mean = torch.cat([x_mean, meta], dim=1)
        out_mean = self.out_mean(feat_mean)
        
        # Path 2: Multi-Instance Learning with Attention (Event-level features)
        frame_logits = self.frame_fc(x_cat)
        attn_scores = torch.clamp(self.attn_fc(x_cat), -30, 30)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_feat = torch.sum(attn_weights * x_cat, dim=1)
        mil_out = self.frame_fc(attn_feat)
        
        return out_mean, mil_out, attn_weights, feat_mean, attn_feat, frame_logits

# Pytorch Lightning wrapper for training logic
class DualPathPL(pl.LightningModule):
    def __init__(self, metadata_dim, lr=1e-4, alpha=0.6, dropout_rate=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = DualPathNet(metadata_dim, dropout_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, meta):
        return self.model(x, meta)

    def training_step(self, batch, batch_idx):
        x, meta, y = batch
        out_mean, mil_out, *_ = self(x, meta)
        # Multi-task loss: weighted sum of Mean Path and MIL Path cross-entropy
        loss = self.hparams.alpha * self.criterion(out_mean, y) + (1 - self.hparams.alpha) * self.criterion(mil_out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        out_mean, mil_out, *_ = self(x, meta)
        # Validation prediction uses fusion of both paths
        fusion_logits = self.hparams.alpha * out_mean + (1 - self.hparams.alpha) * mil_out
        pred = torch.argmax(fusion_logits, dim=1)
        acc = accuracy_score(y.cpu(), pred.cpu())
        f1 = f1_score(y.cpu(), pred.cpu(), average='macro')
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# Wrapper for Integrated Gradients attribution
class IGWrapper(nn.Module):
    def __init__(self, model, use_mean=True):
        super().__init__()
        self.model = model.model.eval()
        self.use_mean = use_mean

    def forward(self, x):
        meta = self.meta.expand(x.size(0), -1)
        out_mean, mil_out, *_ = self.model(x, meta)
        return (out_mean if self.use_mean else mil_out)[:, self.target_class]

    def set_context(self, meta, target_class):
        self.meta = meta
        self.target_class = target_class

def main():
    # ---------------- DUMMY_MODE: Generate placeholder outputs ----------------
    if DUMMY_MODE or pl is None or torch is None or nn is None:
        print("[INFO] Generating dummy saved_models outputs (no active training).")
        base_out = TEST_OUTPUT_DIR if TEST_OUTPUT_DIR else "."
        folds_root = os.path.join(base_out, "saved_models")
        os.makedirs(folds_root, exist_ok=True)

        rng = np.random.default_rng(42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (_, val_idx) in enumerate(skf.split(fmri, labels)):
            fold_dir = os.path.join(folds_root, f"fold{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            yv = labels[val_idx]; Nv = len(yv)
            T, R = fmri.shape[1], fmri.shape[2]

            # Placeholder arrays for downstream scripts
            np.save(os.path.join(fold_dir, "out_mean.npy"), rng.normal(size=(Nv, 2)).astype(np.float32))
            np.save(os.path.join(fold_dir, "labels.npy"), yv.astype(int))
            np.save(os.path.join(fold_dir, "attn_weights.npy"), rng.random(size=(Nv, T)).astype(np.float32))
            np.save(os.path.join(fold_dir, "ig_roi_outmean.npy"), rng.normal(scale=0.01, size=(Nv, T, R)).astype(np.float32))
            np.save(os.path.join(fold_dir, "ig_roi_milout.npy"), rng.normal(scale=0.01, size=(Nv, T, R)).astype(np.float32))
            np.save(os.path.join(fold_dir, "ig_curve_outmean.npy"), rng.normal(size=(Nv, T)).astype(np.float32))
            np.save(os.path.join(fold_dir, "ig_curve_milout.npy"), rng.normal(size=(Nv, T)).astype(np.float32))
            np.save(os.path.join(fold_dir, "frame_logits.npy"), rng.normal(size=(Nv, T, 2)).astype(np.float32))

            with open(os.path.join(fold_dir, "config.json"), "w") as f:
                json.dump({"test_mode": True, "fold": fold}, f, indent=2)

        print(f"Dummy folds saved to: {folds_root}")
        return

    # ---------------- Training Logic ----------------
    config = {
        'lr': 5e-4,
        'alpha': 0.6,
        'dropout_rate': 0.5,
        'batch_size': 1 if TEST_MODE else 32,
        'max_epochs': 1 if TEST_MODE else 50,
        'patience': 1 if TEST_MODE else 5
    }
    model_kwargs = {k: config[k] for k in ['lr', 'alpha', 'dropout_rate']}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(fmri, labels)):
        print(f"--- Training Fold {fold + 1}/5 ---")
        base_out = TEST_OUTPUT_DIR if TEST_OUTPUT_DIR else "."
        save_dir = os.path.join(base_out, "saved_models", f"fold{fold}")
        os.makedirs(save_dir, exist_ok=True)

        # Prepare datasets
        train_ds = TensorDataset(torch.tensor(fmri[train_idx], dtype=torch.float32),
                                 torch.tensor(meta[train_idx], dtype=torch.float32),
                                 torch.tensor(labels[train_idx]))
        val_ds = TensorDataset(torch.tensor(fmri[val_idx], dtype=torch.float32),
                               torch.tensor(meta[val_idx], dtype=torch.float32),
                               torch.tensor(labels[val_idx]))

        # Initialize trainer and model
        model = DualPathPL(metadata_dim=meta.shape[1], **model_kwargs)
        ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best")
        trainer = pl.Trainer(default_root_dir=save_dir, max_epochs=config['max_epochs'],
                             callbacks=[ckpt, EarlyStopping(monitor="val_acc", patience=config['patience'])],
                             accelerator="cpu" if TEST_MODE else ("gpu" if torch.cuda.is_available() else "cpu"),
                             enable_checkpointing=True,
                             logger=False)
        trainer.fit(model, DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True),
                    DataLoader(val_ds, batch_size=config['batch_size']))

        # Inference and Attribution
        best_model = DualPathPL.load_from_checkpoint(ckpt.best_model_path, metadata_dim=meta.shape[1], **model_kwargs).to(device)
        best_model.eval()

        all_logits, all_labels, all_attn, feat_mean_all, attn_feat_all, frame_logits_all = [], [], [], [], [], []
        ig_roi_mean, ig_roi_mil = [], []

        ig_wrapper_mean = IGWrapper(best_model, use_mean=True)
        ig_wrapper_mil = IGWrapper(best_model, use_mean=False)
        ig_mean = IntegratedGradients(ig_wrapper_mean) if IntegratedGradients is not None else None
        ig_mil = IntegratedGradients(ig_wrapper_mil) if IntegratedGradients is not None else None

        for xb, mb, yb in DataLoader(val_ds, batch_size=1):
            xb, mb = xb.to(device), mb.to(device)
            yb_int = int(yb.item())

            out_mean, mil_out, attn, feat_mean, attn_feat, frame_logits = best_model(xb, mb)
            fusion_logits = config['alpha'] * out_mean + (1 - config['alpha']) * mil_out

            all_logits.append(fusion_logits.squeeze(0).detach().cpu().numpy())
            all_labels.append(yb_int)
            all_attn.append(attn.squeeze(0).squeeze(-1).detach().cpu().numpy())
            feat_mean_all.append(feat_mean.squeeze(0).detach().cpu().numpy())
            attn_feat_all.append(attn_feat.squeeze(0).detach().cpu().numpy())
            frame_logits_all.append(frame_logits.squeeze(0).detach().cpu().numpy())

            # IG Attribution: baseline is zero signal
            if ig_mean is None or ig_mil is None:
                attr_out_np = np.zeros_like(xb.squeeze(0).detach().cpu().numpy())
                attr_mil_np = np.zeros_like(xb.squeeze(0).detach().cpu().numpy())
            else:
                ig_wrapper_mean.set_context(mb, yb_int); ig_wrapper_mil.set_context(mb, yb_int)
                attr_mean = ig_mean.attribute(inputs=xb, baselines=torch.zeros_like(xb))
                attr_mil = ig_mil.attribute(inputs=xb, baselines=torch.zeros_like(xb))
                attr_out_np = attr_mean.squeeze(0).detach().cpu().numpy()
                attr_mil_np = attr_mil.squeeze(0).detach().cpu().numpy()
            
            ig_roi_mean.append(attr_out_np)
            ig_roi_mil.append(attr_mil_np)

        # Save all results for this fold
        np.save(os.path.join(save_dir, "out_mean.npy"), np.array(all_logits))
        np.save(os.path.join(save_dir, "labels.npy"), np.array(all_labels))
        np.save(os.path.join(save_dir, "attn_weights.npy"), np.array(all_attn))
        np.save(os.path.join(save_dir, "ig_roi_outmean.npy"), np.array(ig_roi_mean))
        np.save(os.path.join(save_dir, "ig_roi_milout.npy"), np.array(ig_roi_mil))
        np.save(os.path.join(save_dir, "ig_curve_outmean.npy"), np.mean(np.abs(ig_roi_mean), axis=2))
        np.save(os.path.join(save_dir, "ig_curve_milout.npy"), np.mean(np.abs(ig_roi_mil), axis=2))
        np.save(os.path.join(save_dir, "frame_logits.npy"), np.array(frame_logits_all))

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Fold {fold + 1} completed. Results in {save_dir}")

if __name__ == "__main__":
    main()
