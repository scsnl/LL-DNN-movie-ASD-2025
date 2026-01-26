"""
Model Export: ONNX Format

This script exports the DualPathNet model to ONNX format for visualization in 
tools like Netron.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn

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

def main():
    T, R, D_META = 156, 246, 6
    model = DualPathNet(metadata_dim=D_META, dropout_rate=0.5).eval()
    x = torch.randn(1, T, R)
    meta = torch.randn(1, D_META)
    out = Path("model_viz"); out.mkdir(exist_ok=True)

    try:
        torch.onnx.export(
            model, (x, meta), str(out / "dualpathnet.onnx"),
            input_names=["x", "meta"],
            output_names=["out_mean","mil_out","attn_weights","feat_mean","attn_feat","frame_logits"],
            dynamic_axes={"x": {0: "batch"}, "meta": {0: "batch"}},
            opset_version=17, do_constant_folding=True
        )
        print(f"Saved: {out / 'dualpathnet.onnx'}")
    except Exception as e:
        print(f"ONNX export failed: {e}")

if __name__ == "__main__":
    main()
