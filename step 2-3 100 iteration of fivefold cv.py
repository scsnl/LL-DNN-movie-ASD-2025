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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Load and preprocess data ========

data_path = '# TODO: specify your data path'  # 

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

# ======== Define model ========

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
        attn_scores = torch.clamp(self.attn_fc(x_cat), -30, 30)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_feat = torch.sum(attn_weights * x_cat, dim=1)
        mil_out = self.frame_fc(attn_feat)
        return out_mean, mil_out

# ======== Lightning module ========

class DualPathPL(pl.LightningModule):
    def __init__(self, metadata_dim, lr=1e-4, alpha=0.6, dropout_rate=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = DualPathNet(metadata_dim, dropout_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, meta):
        self.model.to(x.device)  #   device

        return self.model(x, meta)

    def training_step(self, batch, batch_idx):
        x, meta, y = batch
        self.model.to(x.device)
        out_mean, mil_out = self(x, meta)
        loss = self.hparams.alpha * self.criterion(out_mean, y) + (1 - self.hparams.alpha) * self.criterion(mil_out, y)
        pred = torch.argmax((self.hparams.alpha * out_mean + (1 - self.hparams.alpha) * mil_out), dim=1)
        acc = (pred == y).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        self.model.to(x.device)
        out_mean, mil_out = self(x, meta)
        fusion_logits = self.hparams.alpha * out_mean + (1 - self.hparams.alpha) * mil_out
        loss = self.criterion(fusion_logits, y)
        pred = torch.argmax(fusion_logits, dim=1)
        acc = accuracy_score(y.cpu(), pred.cpu())
        f1 = f1_score(y.cpu(), pred.cpu(), average='macro')
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_f1", f1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ======== Training loop ========

def main():
    config = {
        'lr': 5e-5,
        'alpha': 0.5,
        'dropout_rate': 0.5,
        'batch_size': 32,
        'max_epochs': 16,
        'patience': 5
    }
    model_kwargs = {k: config[k] for k in ['lr', 'alpha', 'dropout_rate']}
    results = []

    for run in range(100):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
        for fold, (train_idx, val_idx) in enumerate(skf.split(fmri, labels)):
            print(f"[Run {run+1}/100 | Fold {fold+1}/5]")
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
            trainer = pl.Trainer(
                max_epochs=config['max_epochs'],
                callbacks=[EarlyStopping(monitor="val_acc", patience=config['patience'], mode="max")],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                logger=False,
                enable_checkpointing=False
            )
            trainer.fit(model,
                        DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True),
                        DataLoader(val_ds, batch_size=config['batch_size']))

            # Evaluation

            model.eval()
            val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
            preds, targets = [], []
            for xb, mb, yb in val_loader:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                out_mean, mil_out = model(xb, mb)
                fusion_logits = config['alpha'] * out_mean + (1 - config['alpha']) * mil_out
                pred = torch.argmax(fusion_logits, dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())

            acc = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average='macro')
            results.append({"run": run, "fold": fold, "acc": acc, "f1": f1})

    # Save results

    os.makedirs("crossval_results", exist_ok=True)
    with open("crossval_results/dualpathnet_100runs5fold.json", "w") as f:
        json.dump(results, f, indent=2)

    # Accuracy distribution plot

    accs = [r["acc"] * 100 for r in results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    plt.figure(figsize=(6, 6))
    sns.boxplot(y=accs, color='lightblue', width=0.3)
    plt.ylim(65, 80)
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Distribution (DualPathNet: 100x5-Fold)")
    plt.text(0, mean_acc, f"Mean: {mean_acc:.2f}%\nStd: {std_acc:.2f}%", ha='center', va='center', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray'))
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("crossval_results/dualpathnet_accuracy_boxplot.png")
    plt.close()

if __name__ == "__main__":
    main()
