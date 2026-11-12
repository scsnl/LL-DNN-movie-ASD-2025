"""
Supply Step 2: time-filtered DualPathNet training with 5-fold CV and IG export.

Segments of movieDM (TR = 0.8 s):
  seg1  0-190 s    -> TR [0, 237)
  seg2  298-435 s  -> TR [372, 543)
  seg3  TR 547-end -> TR [547, T)

Hyperparameters: lr 5e-4, alpha 0.8, dropout 0.3, batch size 32,
max 50 epochs, early stopping patience 5 on validation accuracy.
IG: zero baseline, Captum default 50 steps, target = true class,
attributions of the mean-path logit and the MIL-path logit.

Outputs: results/<seg>/saved_models/fold*/ (TEST_MODE: $TEST_OUTPUT_DIR/<seg>/...).
Each fold directory holds subject_ids.npy giving the row order of every array.

TEST_MODE=1: seg2 only, one epoch, batch size 1. DUMMY_MODE=1: random outputs.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from captum.attr import IntegratedGradients
except ImportError:
    pl = IntegratedGradients = None

if pl is None:
    class _LightningModuleStub(nn.Module):
        pass

    class _PLStub:
        LightningModule = _LightningModuleStub
    pl_base = _PLStub
else:
    pl_base = pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ig_io import save_fold_ids  # noqa: E402

warnings.filterwarnings("ignore")

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
DUMMY_MODE = os.environ.get("DUMMY_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")

TR = 0.8
DATA_PATH = os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz") if TEST_DATA_DIR else "./combined_asd_td_movieDM_data.pklz"
OUT_BASE = Path(TEST_OUTPUT_DIR) if TEST_OUTPUT_DIR else Path("results")
SEED = 42
MAX_EPOCHS = 1 if TEST_MODE else 50
PATIENCE = 1 if TEST_MODE else 5
BEST_CFG = dict(lr=5e-4, alpha=0.8, dropout_rate=0.3, batch_size=1 if TEST_MODE else 32)
NUM_WORKERS = 0 if TEST_MODE else 4
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


def segment_indices(T, seg, tr=TR):
    if seg == "seg1":
        start, end = 0, int(190 / tr)
    elif seg == "seg2":
        start, end = int(298 / tr), int(435 / tr)
    elif seg == "seg3":
        start, end = 547, T
    else:
        raise ValueError(f"unknown segment: {seg}")
    start, end = max(0, min(T, start)), max(0, min(T, end))
    if end <= start:
        raise ValueError(f"invalid window for {seg}: [{start}, {end}) with T={T}")
    return start, end


def filter_segment(fmri, seg, tr=TR):
    T = fmri.shape[1]
    s, e = segment_indices(T, seg, tr)
    info = {"segment": seg, "tr": tr,
            "segment_seconds": (0, 190) if seg == "seg1" else ((298, 435) if seg == "seg2" else (int(547 * tr), "end")),
            "segment_idx": [int(s), int(e)], "T_segment": int(e - s), "T_original": int(T)}
    return fmri[:, s:e, :], info


def load_data_and_meta(data_path):
    datao = pd.read_pickle(data_path)
    datao = datao[(datao["percentofvolsrepaired"] <= 10) & (datao["mean_fd"] <= 0.5)]
    shapes = [np.asarray(d).shape for d in datao.data]
    main_shape = max(set(shapes), key=shapes.count)
    datao = datao.iloc[[i for i, s in enumerate(shapes) if s == main_shape]].reset_index(drop=True)
    fmri_all = np.stack([np.asarray(d) for d in datao.data])
    nan_subs = np.unique(np.argwhere(np.isnan(fmri_all))[:, 0])
    datao = datao.drop(index=nan_subs).reset_index(drop=True)
    fmri = np.stack([np.asarray(d) for d in datao.data])
    labels = datao["label"].str.lower().map({"asd": 0, "td": 1}).values
    site = pd.get_dummies(datao["site"], dtype=float).values
    gender = pd.get_dummies(datao["gender"], dtype=float).values
    age = datao["age"].astype(float).values.reshape(-1, 1)
    meta = np.concatenate([site, gender, age], axis=1)
    return fmri, labels, meta, datao["subject_id"].astype(str).values


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
        self.out_mil = nn.Linear(512 + metadata_dim, 2)

    def forward(self, x, meta):
        B, T, R = x.shape
        x = self.conv3(self.conv2(self.conv1(x.permute(0, 2, 1)))).permute(0, 2, 1)
        x = self.dropout(x)
        x_cat = torch.cat([x, meta.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        feat_mean = torch.cat([torch.mean(x, dim=1), meta], dim=1)
        out_mean = self.out_mean(feat_mean)
        frame_logits = self.frame_fc(x_cat)
        attn_weights = torch.softmax(torch.clamp(self.attn_fc(x_cat), -30, 30), dim=1)
        attn_feat = torch.sum(attn_weights * x_cat, dim=1)
        mil_out = self.out_mil(attn_feat)
        return out_mean, mil_out, attn_weights, feat_mean, attn_feat, frame_logits


class DualPathPL(pl_base.LightningModule):
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
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        out_mean, mil_out, *_ = self(x, meta)
        preds = torch.argmax(self.hparams.alpha * out_mean + (1 - self.hparams.alpha) * mil_out, dim=1)
        y_np, p_np = y.detach().cpu().numpy(), preds.detach().cpu().numpy()
        self.log("val_acc", accuracy_score(y_np, p_np), prog_bar=True, on_epoch=True)
        self.log("val_f1", f1_score(y_np, p_np, average="macro"), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class IGWrapper(nn.Module):
    def __init__(self, model, use_mean=True):
        super().__init__()
        self.model = model.model.eval()
        self.use_mean = use_mean
        self.meta = None
        self.target_class = 0

    def forward(self, x):
        out_mean, mil_out, *_ = self.model(x, self.meta.expand(x.size(0), -1))
        return (out_mean if self.use_mean else mil_out)[:, self.target_class]

    def set_context(self, meta, target_class):
        self.meta = meta
        self.target_class = target_class


def write_dummy(seg, fmri_seg, labels, meta, ids, out_root):
    rng = np.random.default_rng(SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    N, T, R = fmri_seg.shape
    D = 512 + meta.shape[1]
    for fold, (_, va) in enumerate(skf.split(fmri_seg, labels)):
        d = out_root / "saved_models" / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        n = len(va)
        np.save(d / "out_mean.npy", rng.normal(size=(n, 2)).astype(np.float32))
        np.save(d / "labels.npy", labels[va].astype(int))
        save_fold_ids(str(d), ids[va])
        np.save(d / "attn_weights.npy", rng.random(size=(n, T)).astype(np.float32))
        np.save(d / "feat_mean.npy", rng.normal(size=(n, D)).astype(np.float32))
        np.save(d / "attn_feat.npy", rng.normal(size=(n, D)).astype(np.float32))
        np.save(d / "frame_logits.npy", rng.normal(size=(n, T, 2)).astype(np.float32))
        for k in ("outmean", "milout"):
            np.save(d / f"ig_roi_{k}.npy", rng.normal(scale=0.01, size=(n, T, R)).astype(np.float32))
            np.save(d / f"ig_curve_{k}.npy", rng.random(size=(n, T)).astype(np.float32))
    print(f"[{seg}] random fold outputs written to {out_root / 'saved_models'}")


def run_for_segment(seg):
    out_root = OUT_BASE / seg
    out_root.mkdir(parents=True, exist_ok=True)
    fmri, labels, meta, ids = load_data_and_meta(DATA_PATH)
    fmri_seg, info = filter_segment(fmri, seg, TR)
    (out_root / "time_filtering_info.json").write_text(json.dumps(info, indent=2))
    if DUMMY_MODE or pl is None or IntegratedGradients is None:
        write_dummy(seg, fmri_seg, labels, meta, ids, out_root)
        return

    X = torch.tensor(fmri_seg, dtype=torch.float32)
    M = torch.tensor(meta, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    metadata_dim = M.shape[1]
    model_kw = dict(metadata_dim=metadata_dim, lr=BEST_CFG["lr"], alpha=BEST_CFG["alpha"],
                    dropout_rate=BEST_CFG["dropout_rate"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"[{seg}] fold {fold + 1}/5")
        fold_dir = out_root / "saved_models" / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_ds = TensorDataset(X[tr_idx], M[tr_idx], y[tr_idx])
        val_ds = TensorDataset(X[va_idx], M[va_idx], y[va_idx])

        ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best")
        es = EarlyStopping(monitor="val_acc", patience=PATIENCE, mode="max")
        trainer = pl.Trainer(default_root_dir=str(fold_dir), max_epochs=MAX_EPOCHS, accelerator=DEVICE,
                             devices=1, callbacks=[ckpt, es], log_every_n_steps=10, deterministic=True,
                             enable_progress_bar=not TEST_MODE)
        trainer.fit(DualPathPL(**model_kw),
                    DataLoader(train_ds, batch_size=BEST_CFG["batch_size"], shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True),
                    DataLoader(val_ds, batch_size=BEST_CFG["batch_size"], shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True))

        best_model = DualPathPL.load_from_checkpoint(ckpt.best_model_path, **model_kw)
        best_model.eval()

        logits_all, labels_all, attn_all = [], [], []
        feat_mean_all, attn_feat_all, fused_feat_all, frame_logits_all = [], [], [], []
        inputs_pt, logits_pt = [], []
        with torch.no_grad():
            for xb, mb, yb in DataLoader(val_ds, batch_size=1, shuffle=False):
                out_mean, mil_out, attn, feat_mean, attn_feat, frame_logits = best_model(xb, mb)
                fusion = BEST_CFG["alpha"] * out_mean + (1 - BEST_CFG["alpha"]) * mil_out
                logits_all.append(fusion.squeeze(0).cpu().numpy())
                labels_all.append(int(yb.item()))
                attn_all.append(attn.squeeze(0).squeeze(-1).cpu().numpy())
                feat_mean_all.append(feat_mean.squeeze(0).cpu().numpy())
                attn_feat_all.append(attn_feat.squeeze(0).cpu().numpy())
                fused_feat_all.append(torch.cat([feat_mean, attn_feat], dim=1).squeeze(0).cpu().numpy())
                frame_logits_all.append(frame_logits.squeeze(0).cpu().numpy())
                inputs_pt.append(xb.squeeze(0).cpu())
                logits_pt.append(fusion.squeeze(0).cpu())

        wrap_mean, wrap_mil = IGWrapper(best_model, use_mean=True), IGWrapper(best_model, use_mean=False)
        ig_mean, ig_mil = IntegratedGradients(wrap_mean), IntegratedGradients(wrap_mil)
        ig_roi_mean, ig_roi_mil, ig_curve_mean, ig_curve_mil = [], [], [], []
        for xb, mb, yb in DataLoader(val_ds, batch_size=1, shuffle=False):
            target = int(yb.item())
            wrap_mean.set_context(mb, target)
            wrap_mil.set_context(mb, target)
            a_mean = ig_mean.attribute(inputs=xb, baselines=torch.zeros_like(xb)).squeeze(0).detach().cpu().numpy()
            a_mil = ig_mil.attribute(inputs=xb, baselines=torch.zeros_like(xb)).squeeze(0).detach().cpu().numpy()
            ig_roi_mean.append(a_mean)
            ig_roi_mil.append(a_mil)
            ig_curve_mean.append(np.mean(np.abs(a_mean), axis=1))
            ig_curve_mil.append(np.mean(np.abs(a_mil), axis=1))

        arrays = {"out_mean": logits_all, "labels": labels_all, "attn_weights": attn_all,
                  "feat_mean": feat_mean_all, "attn_feat": attn_feat_all, "fused_feat": fused_feat_all,
                  "frame_logits": frame_logits_all, "ig_curve_outmean": ig_curve_mean,
                  "ig_curve_milout": ig_curve_mil, "ig_roi_outmean": ig_roi_mean, "ig_roi_milout": ig_roi_mil}
        for name, a in arrays.items():
            np.save(fold_dir / f"{name}.npy", np.array(a))
        save_fold_ids(str(fold_dir), ids[va_idx])
        torch.save(torch.stack(inputs_pt), fold_dir / "ig_inputs.pt")
        torch.save(torch.stack(logits_pt), fold_dir / "ig_logits.pt")
        torch.save(torch.tensor(labels_all), fold_dir / "ig_labels.pt")
        with open(fold_dir / "config.json", "w") as f:
            json.dump({"segment": seg, "tr": TR, **BEST_CFG, "max_epochs": MAX_EPOCHS, "patience": PATIENCE}, f, indent=2)
        print(f"[{seg}] fold {fold + 1} saved to {fold_dir}")


def main():
    if pl is not None:
        pl.seed_everything(SEED, workers=True)
    for seg in (["seg2"] if TEST_MODE else ["seg1", "seg2", "seg3"]):
        run_for_segment(seg)


if __name__ == "__main__":
    main()
