# Final version of train_and_save_all.py with clean parameter management


import os
import json
import torch
import warnings
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from captum.attr import IntegratedGradients
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data

data_path = '# TODO: specify your data path'

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

# Model definition

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

def main():
    config = {
        'lr': 5e-4,
        'alpha': 0.6,
        'dropout_rate': 0.5,
        'batch_size': 32,
        'max_epochs': 50,
        'patience': 5
    }
    model_kwargs = {k: config[k] for k in ['lr', 'alpha', 'dropout_rate']}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(fmri, labels)):
        print(f"Running fold {fold + 1}/5")
        save_dir = f"saved_models/fold{fold}"
        os.makedirs(save_dir, exist_ok=True)

        x_train, x_val = fmri[train_idx], fmri[val_idx]
        m_train, m_val = meta[train_idx], meta[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                 torch.tensor(m_train, dtype=torch.float32),
                                 torch.tensor(y_train))
        val_ds = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                               torch.tensor(m_val, dtype=torch.float32),
                               torch.tensor(y_val))

        model = DualPathPL(metadata_dim=meta.shape[1], **model_kwargs)
        ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best")
        trainer = pl.Trainer(default_root_dir=save_dir, max_epochs=config['max_epochs'],
                             callbacks=[ckpt, EarlyStopping(monitor="val_acc", patience=config['patience'])],
                             accelerator="gpu" if torch.cuda.is_available() else "cpu")
        trainer.fit(model,
                    DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True),
                    DataLoader(val_ds, batch_size=config['batch_size']))

        best_model = DualPathPL.load_from_checkpoint(
            ckpt.best_model_path,
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
        ig_mean = IntegratedGradients(ig_wrapper_mean)
        ig_mil = IntegratedGradients(ig_wrapper_mil)

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
            fused_feat_all.append(torch.cat([feat_mean, attn_feat], dim=1).squeeze(0).detach().cpu().numpy())
            frame_logits_all.append(frame_logits.squeeze(0).detach().cpu().numpy())

            ig_wrapper_mean.set_context(mb, yb_int)
            ig_wrapper_mil.set_context(mb, yb_int)
            attr_mean = ig_mean.attribute(inputs=xb, baselines=torch.zeros_like(xb))
            attr_mil = ig_mil.attribute(inputs=xb, baselines=torch.zeros_like(xb))

            attr_out_np = attr_mean.squeeze(0).detach().cpu().numpy()
            attr_mil_np = attr_mil.squeeze(0).detach().cpu().numpy()
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

        print(f"Fold {fold + 1} completed and saved to {save_dir}")


if __name__ == "__main__":
    main()
