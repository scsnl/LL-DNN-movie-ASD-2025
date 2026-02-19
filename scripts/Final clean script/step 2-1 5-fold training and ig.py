# step 2-1 5-fold training and ig.py
# Provides fit_one_split, eval_on_idx, load_and_preprocess_data for nested CV.
# github_upload: TEST_MODE uses minimal params (epochs=1) to run flow; DUMMY_MODE skips training.

import os
import json
import warnings
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
DUMMY_MODE = os.environ.get("DUMMY_MODE", "0") == "1"

try:
    import torch
except ImportError:
    torch = None
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
except ImportError:
    pl = ModelCheckpoint = EarlyStopping = None
    class _LightningModuleStub: pass
    pl = type("PLStub", (), {"LightningModule": _LightningModuleStub})()
try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None

warnings.filterwarnings("ignore")
device = torch.device("cuda" if (torch and torch.cuda.is_available() and not TEST_MODE) else "cpu") if torch else "cpu"


def load_and_preprocess_data(data_path):
    """Load data with same QC as step 1. Returns (fmri, meta, labels)."""
    datao = pd.read_pickle(data_path)
    datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
    shapes = [np.asarray(d).shape for d in datao.data]
    main_shape = max(set(shapes), key=shapes.count)
    datao = datao.iloc[[i for i, s in enumerate(shapes) if s == main_shape]].reset_index(drop=True)
    fmri_all = np.stack([np.asarray(d) for d in datao.data])
    nan_subs = np.unique(np.argwhere(np.isnan(fmri_all))[:, 0])
    datao = datao.drop(index=nan_subs).reset_index(drop=True)
    fmri = np.stack([np.asarray(d) for d in datao.data])
    labels = datao['label'].apply(lambda x: 0 if x == 'asd' else 1).values
    site = pd.get_dummies(datao['site'], dtype=float).values
    gender = pd.get_dummies(datao['gender'], dtype=float).values
    age = datao['age'].astype(float).values.reshape(-1, 1)
    meta = np.concatenate([site, gender, age], axis=1)
    return fmri, meta, labels


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
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        meta_exp = meta.unsqueeze(1).expand(-1, T, -1)
        x_cat = torch.cat([x, meta_exp], dim=-1)
        x_mean = torch.mean(x, dim=1)
        feat_mean = torch.cat([x_mean, meta], dim=1)
        out_mean = self.out_mean(feat_mean)
        frame_logits = self.frame_fc(x_cat)
        attn_scores = torch.clamp(self.attn_fc(x_cat), -30, 30)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_feat = torch.sum(attn_weights * x_cat, dim=1)
        mil_out = self.frame_fc(attn_feat)
        return out_mean, mil_out, attn_weights, feat_mean, attn_feat, frame_logits


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
        loss = self.hparams.alpha * self.criterion(out_mean, y) + (1 - self.hparams.alpha) * self.criterion(mil_out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        out_mean, mil_out, *_ = self(x, meta)
        fusion_logits = self.hparams.alpha * out_mean + (1 - self.hparams.alpha) * mil_out
        pred = torch.argmax(fusion_logits, dim=1)
        acc = accuracy_score(y.cpu(), pred.cpu())
        f1 = f1_score(y.cpu(), pred.cpu(), average='macro')
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


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


def fit_one_split(X, meta, y, train_idx, val_idx, hp, metadata_dim, seed, save_dir):
    """Train on train_idx, validate on val_idx. Returns (best_val_acc, best_ckpt_path)."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, x_val = X[train_idx], X[val_idx]
    m_train, m_val = meta[train_idx], meta[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(m_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(m_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    model_kwargs = {k: hp[k] for k in ['lr', 'alpha', 'dropout_rate'] if k in hp}
    model = DualPathPL(metadata_dim=metadata_dim, **model_kwargs)
    ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best")
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=hp.get('max_epochs', 50),
        callbacks=[
            ckpt,
            EarlyStopping(monitor="val_acc", patience=hp.get('patience', 5), mode="max"),
        ],
        accelerator="cpu" if TEST_MODE else ("gpu" if torch.cuda.is_available() else "cpu"),
        enable_progress_bar=False,
    )
    trainer.fit(
        model,
        DataLoader(train_ds, batch_size=hp.get('batch_size', 32), shuffle=True),
        DataLoader(val_ds, batch_size=hp.get('batch_size', 32)),
    )
    best_val_acc = float(ckpt.best_model_score.item()) if ckpt.best_model_score is not None else 0.0
    best_ckpt = ckpt.best_model_path
    if best_ckpt is None or not os.path.exists(best_ckpt):
        raise RuntimeError(f"No valid checkpoint saved in {save_dir}")
    return best_val_acc, best_ckpt


def eval_on_idx(ckpt_path, X, meta, y, test_idx, hp, metadata_dim):
    """Load model from ckpt_path, evaluate on test_idx. Returns (acc, f1, preds, logits)."""
    model_kwargs = {k: hp[k] for k in ['lr', 'alpha', 'dropout_rate'] if k in hp}
    model = DualPathPL.load_from_checkpoint(
        ckpt_path,
        metadata_dim=metadata_dim,
        **model_kwargs,
    ).to(device)
    model.eval()

    x_test = X[test_idx]
    m_test = meta[test_idx]
    y_test = y[test_idx]

    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(m_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    loader = DataLoader(test_ds, batch_size=hp.get('batch_size', 32))

    preds, logits_list, targets = [], [], []
    with torch.no_grad():
        for xb, mb, yb in loader:
            xb, mb = xb.to(device), mb.to(device)
            out_mean, mil_out, *_ = model(xb, mb)
            alpha = hp.get('alpha', 0.6)
            fusion_logits = alpha * out_mean + (1 - alpha) * mil_out
            pred = torch.argmax(fusion_logits, dim=1)
            preds.extend(pred.cpu().numpy())
            logits_list.append(fusion_logits.cpu().numpy())
            targets.extend(yb.numpy())

    preds = np.array(preds)
    logits = np.concatenate(logits_list, axis=0)
    targets = np.array(targets)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    return acc, f1, preds, logits


def main():
    data_path = (
        os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz")
        if TEST_DATA_DIR
        else "combined_asd_td_movieDM_data.pklz"
    )
    base_out = TEST_OUTPUT_DIR if TEST_OUTPUT_DIR else "."
    fmri, meta, labels = load_and_preprocess_data(data_path)

    if DUMMY_MODE or pl is None or torch is None:
        print("[INFO] Generating dummy saved_models outputs (no active training).")
        folds_root = os.path.join(base_out, "saved_models")
        os.makedirs(folds_root, exist_ok=True)
        rng = np.random.default_rng(42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (_, val_idx) in enumerate(skf.split(fmri, labels)):
            fold_dir = os.path.join(folds_root, f"fold{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            yv = labels[val_idx]
            Nv, T, R = len(yv), fmri.shape[1], fmri.shape[2]
            np.save(os.path.join(fold_dir, "out_mean.npy"), rng.normal(size=(Nv, 2)).astype(np.float32))
            np.save(os.path.join(fold_dir, "labels.npy"), yv.astype(int))
            np.save(os.path.join(fold_dir, "attn_weights.npy"), rng.random(size=(Nv, T)).astype(np.float32))
            np.save(os.path.join(fold_dir, "feat_mean.npy"), rng.normal(size=(Nv, 512 + meta.shape[1])).astype(np.float32))
            np.save(os.path.join(fold_dir, "attn_feat.npy"), rng.normal(size=(Nv, 512 + meta.shape[1])).astype(np.float32))
            np.save(os.path.join(fold_dir, "ig_roi_outmean.npy"), rng.normal(scale=0.01, size=(Nv, T, R)).astype(np.float32))
            np.save(os.path.join(fold_dir, "ig_roi_milout.npy"), rng.normal(scale=0.01, size=(Nv, T, R)).astype(np.float32))
            np.save(os.path.join(fold_dir, "ig_curve_outmean.npy"), rng.normal(size=(Nv, T)).astype(np.float32))
            np.save(os.path.join(fold_dir, "ig_curve_milout.npy"), rng.normal(size=(Nv, T)).astype(np.float32))
            np.save(os.path.join(fold_dir, "frame_logits.npy"), rng.normal(size=(Nv, T, 2)).astype(np.float32))
            with open(os.path.join(fold_dir, "config.json"), "w") as f:
                json.dump({"lr": 5e-4, "alpha": 0.6, "dropout_rate": 0.5}, f, indent=2)
        print(f"Dummy folds saved to: {folds_root}")
        return

    config = {
        'lr': 5e-4,
        'alpha': 0.6,
        'dropout_rate': 0.5,
        'batch_size': 1 if TEST_MODE else 32,
        'max_epochs': 1 if TEST_MODE else 50,
        'patience': 1 if TEST_MODE else 5,
    }
    model_kwargs = {k: config[k] for k in ['lr', 'alpha', 'dropout_rate']}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(fmri, labels)):
        print(f"--- Training Fold {fold + 1}/5 ---")
        save_dir = os.path.join(base_out, "saved_models", f"fold{fold}")

        _, best_ckpt = fit_one_split(
            fmri, meta, labels, train_idx, val_idx,
            hp=config, metadata_dim=meta.shape[1], seed=42, save_dir=save_dir,
        )

        x_val = fmri[val_idx]
        m_val = meta[val_idx]
        y_val = labels[val_idx]
        val_ds = TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(m_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        best_model = DualPathPL.load_from_checkpoint(
            best_ckpt,
            metadata_dim=meta.shape[1],
            **model_kwargs
        ).to(device)
        best_model.eval()

        all_logits, all_labels, all_attn = [], [], []
        feat_mean_all, attn_feat_all, fused_feat_all = [], [], []
        frame_logits_all = []
        inputs_pt, logits_pt, labels_pt = [], [], []
        ig_curve_mean, ig_curve_mil = [], []
        ig_roi_mean, ig_roi_mil = [], []

        ig_wrapper_mean = IGWrapper(best_model, use_mean=True)
        ig_wrapper_mil = IGWrapper(best_model, use_mean=False)
        ig_mean = IntegratedGradients(ig_wrapper_mean) if IntegratedGradients else None
        ig_mil = IntegratedGradients(ig_wrapper_mil) if IntegratedGradients else None

        for xb, mb, yb in DataLoader(val_ds, batch_size=1):
            xb, mb = xb.to(device), mb.to(device)
            yb_int = int(yb.item())

            out_mean, mil_out, attn, feat_mean, attn_feat, frame_logits = best_model(xb, mb)
            fusion_logits = config['alpha'] * out_mean + (1 - config['alpha']) * mil_out
            pred_class = int(torch.argmax(fusion_logits, dim=1).item())

            all_logits.append(fusion_logits.squeeze(0).detach().cpu().numpy())
            all_labels.append(yb_int)
            all_attn.append(attn.squeeze(0).squeeze(-1).detach().cpu().numpy())
            feat_mean_all.append(feat_mean.squeeze(0).detach().cpu().numpy())
            attn_feat_all.append(attn_feat.squeeze(0).detach().cpu().numpy())
            fused_feat_all.append(torch.cat([feat_mean, attn_feat], dim=1).squeeze(0).detach().cpu().numpy())
            frame_logits_all.append(frame_logits.squeeze(0).detach().cpu().numpy())

            if ig_mean and ig_mil:
                ig_wrapper_mean.set_context(mb, pred_class)
                ig_wrapper_mil.set_context(mb, pred_class)
                attr_mean = ig_mean.attribute(inputs=xb, baselines=torch.zeros_like(xb))
                attr_mil = ig_mil.attribute(inputs=xb, baselines=torch.zeros_like(xb))
                attr_out_np = attr_mean.squeeze(0).detach().cpu().numpy()
                attr_mil_np = attr_mil.squeeze(0).detach().cpu().numpy()
            else:
                attr_out_np = np.zeros_like(xb.squeeze(0).detach().cpu().numpy())
                attr_mil_np = np.zeros_like(xb.squeeze(0).detach().cpu().numpy())

            ig_roi_mean.append(attr_out_np)
            ig_roi_mil.append(attr_mil_np)
            ig_curve_mean.append(np.mean(np.abs(attr_out_np), axis=1))
            ig_curve_mil.append(np.mean(np.abs(attr_mil_np), axis=1))

            inputs_pt.append(xb.squeeze(0).cpu())
            logits_pt.append(fusion_logits.squeeze(0).detach().cpu())
            labels_pt.append(yb_int)

        np.save(os.path.join(save_dir, "out_mean.npy"), np.array(all_logits))
        np.save(os.path.join(save_dir, "labels.npy"), np.array(all_labels))
        np.save(os.path.join(save_dir, "attn_weights.npy"), np.array(all_attn))
        np.save(os.path.join(save_dir, "feat_mean.npy"), np.array(feat_mean_all))
        np.save(os.path.join(save_dir, "attn_feat.npy"), np.array(attn_feat_all))
        np.save(os.path.join(save_dir, "fused_feat.npy"), np.array(fused_feat_all))
        np.save(os.path.join(save_dir, "frame_logits.npy"), np.array(frame_logits_all))
        np.save(os.path.join(save_dir, "ig_curve_outmean.npy"), np.array(ig_curve_mean))
        np.save(os.path.join(save_dir, "ig_curve_milout.npy"), np.array(ig_curve_mil))
        np.save(os.path.join(save_dir, "ig_roi_outmean.npy"), np.array(ig_roi_mean))
        np.save(os.path.join(save_dir, "ig_roi_milout.npy"), np.array(ig_roi_mil))

        torch.save(torch.stack(inputs_pt), os.path.join(save_dir, "ig_inputs.pt"))
        torch.save(torch.stack(logits_pt), os.path.join(save_dir, "ig_logits.pt"))
        torch.save(torch.tensor(labels_pt), os.path.join(save_dir, "ig_labels.pt"))

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Fold {fold + 1} completed. Results in {save_dir}")


if __name__ == "__main__":
    main()
