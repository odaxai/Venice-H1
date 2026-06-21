#!/usr/bin/env python3
"""
Train Venice-H1 re-ranker on pre-extracted DeRIS feature caches.

Matches paper Section 3.5 exactly:
  - Loss: L = L_gate + λ·L_gain (λ=5)
  - L_gate: focal BCE (γ=2, auto w_pos)
  - L_gain: smooth-L1 IoU regression on ALL samples
  - AdamW (lr=5e-4, wd=1e-4), cosine + 3-epoch warmup
  - 20 epochs, batch 512, FP16
  - Training time: ~3 min on a single GPU

Usage:
    python train.py --config venice_h1/configs/default.yaml
    python train.py --config venice_h1/configs/default.yaml --no_grid
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

try:
    import yaml
except ImportError:
    raise SystemExit("pip install pyyaml")

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from venice_h1.model.reranker import VeniceH1Reranker


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FeatureCacheDataset(Dataset):
    """Loads a pre-extracted .pt feature cache from extract_features.py."""

    def __init__(self, path: str, use_grid: bool = True):
        data = torch.load(path, map_location="cpu")
        self.use_grid = use_grid

        self.query_feat  = data["query_feat"].float()
        self.det_scores  = data["det_scores"].float()
        self.query_ious  = data["query_ious"].float()
        self.oracle_idx  = data["oracle_idx"].long()

        self.mask_mean   = data["mask_mean"].float()
        self.mask_max    = data["mask_max"].float()
        self.mask_area   = data["mask_area"].float()
        self.mask_std    = data["mask_std"].float()

        if use_grid:
            self.grid_mean_4  = data["grid_mean_4"].float()
            self.grid_max_4   = data["grid_max_4"].float()
            self.boundary_4   = data["boundary_4"].float()
            self.grid_mean_8  = data["grid_mean_8"].float()
            self.grid_max_8   = data["grid_max_8"].float()
            self.boundary_8   = data["boundary_8"].float()
            self.grid_mean_16 = data["grid_mean_16"].float()
            self.grid_max_16  = data["grid_max_16"].float()
            self.boundary_16  = data["boundary_16"].float()

        self.failure_flag = (self.oracle_idx != 0).float()

    def __len__(self):
        return len(self.oracle_idx)

    def __getitem__(self, idx):
        N = self.query_feat.shape[1]
        qf = self.query_feat[idx]                         # [N, D]
        ds = self.det_scores[idx].unsqueeze(-1)           # [N, 1]
        mm = self.mask_mean[idx].unsqueeze(-1)            # [N, 1]
        mx = self.mask_max[idx].unsqueeze(-1)             # [N, 1]
        ma = self.mask_area[idx].unsqueeze(-1)            # [N, 1]
        ms = self.mask_std[idx].unsqueeze(-1)             # [N, 1]

        parts = [qf, ds, mm, mx, ma, ms]                 # [N, D+5]

        if self.use_grid:
            gm4 = self.grid_mean_4[idx]                   # [N, 16]
            gx4 = self.grid_max_4[idx]                    # [N, 16]
            b4  = self.boundary_4[idx].unsqueeze(-1)      # [N, 1]
            gm8 = self.grid_mean_8[idx]                   # [N, 64]
            gx8 = self.grid_max_8[idx]                    # [N, 64]
            b8  = self.boundary_8[idx].unsqueeze(-1)      # [N, 1]
            gm16 = self.grid_mean_16[idx]                 # [N, 256]
            gx16 = self.grid_max_16[idx]                  # [N, 256]
            b16  = self.boundary_16[idx].unsqueeze(-1)    # [N, 1]
            parts += [gm4, gx4, b4, gm8, gx8, b8, gm16, gx16, b16]

        features = torch.cat(parts, dim=-1)               # [N, 936]

        return {
            "features":     features,
            "det_scores":   self.det_scores[idx],
            "mask_means":   self.mask_mean[idx],
            "oracle_idx":   self.oracle_idx[idx],
            "failure_flag": self.failure_flag[idx],
            "query_ious":   self.query_ious[idx],
        }


# ---------------------------------------------------------------------------
# Loss (Section 3.5)
# ---------------------------------------------------------------------------

def focal_bce_loss(pred: torch.Tensor, target: torch.Tensor,
                   gamma: float = 2.0, pos_weight: float | None = None):
    """Focal binary cross-entropy (Eq. in Section 3.5)."""
    if pos_weight is not None:
        weight = torch.where(target > 0.5, pos_weight, 1.0)
    else:
        weight = None

    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = pred * target + (1 - pred) * (1 - target)
    focal = ((1 - pt) ** gamma) * bce

    if weight is not None:
        focal = focal * weight
    return focal.mean()


def compute_loss(out: dict, batch: dict, cfg: dict) -> dict:
    """L = L_gate + λ·L_gain (Section 3.5)."""
    device = out["p_fail"].device
    tcfg = cfg["training"]

    failure_flag = batch["failure_flag"].to(device)
    query_ious   = batch["query_ious"].to(device)

    # Auto positive weight for focal BCE
    n_pos = failure_flag.sum().clamp(min=1)
    n_neg = (1 - failure_flag).sum().clamp(min=1)
    wpos = (n_neg / n_pos).clamp(max=50.0) if tcfg.get("auto_wpos") else None

    # L_gate: focal BCE
    loss_gate = focal_bce_loss(
        out["p_fail"], failure_flag,
        gamma=tcfg["focal_gamma"], pos_weight=wpos)

    # L_gain: smooth-L1 on IoU gain (all samples, dense supervision)
    gain_target = query_ious - query_ious[:, 0:1]          # relative to Q0
    loss_gain = F.smooth_l1_loss(out["gain_logits"], gain_target)

    # Total loss
    total = loss_gate + tcfg["lambda_gain"] * loss_gain

    return {"total": total, "gate": loss_gate, "gain": loss_gain}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, tau: float = 0.05) -> dict:
    model.eval()
    total = failures = reranked_count = correct = harmful = 0
    all_pfail, all_flag = [], []

    for batch in loader:
        features     = batch["features"].to(device)
        det_scores   = batch["det_scores"].to(device)
        mask_means   = batch["mask_means"].to(device)
        oracle_idx   = batch["oracle_idx"].to(device)
        failure_flag = batch["failure_flag"].to(device)
        query_ious   = batch["query_ious"].to(device)

        out = model(features, det_scores, mask_means)
        selected = model.rerank(features, det_scores, mask_means, tau=tau)

        B = len(oracle_idx)
        total += B
        fail_mask = failure_flag.bool()
        failures += fail_mask.sum().item()

        reranked_mask = selected != 0
        reranked_count += reranked_mask.sum().item()
        correct += (selected[fail_mask] == oracle_idx[fail_mask]).sum().item()

        q0_iou  = query_ious[:, 0]
        sel_iou = query_ious.gather(1, selected.unsqueeze(1)).squeeze(1)
        harmful += (reranked_mask & (sel_iou < q0_iou - 0.01)).sum().item()

        all_pfail.append(out["p_fail"].cpu())
        all_flag.append(failure_flag.cpu())

    all_pfail = torch.cat(all_pfail).numpy()
    all_flag  = torch.cat(all_flag).numpy()
    auc = roc_auc_score(all_flag, all_pfail) if _HAS_SKLEARN and all_flag.sum() > 0 else 0.0

    return {
        "failure_rate":    failures / max(total, 1),
        "rerank_rate":     reranked_count / max(total, 1),
        "correct_rerank":  correct / max(failures, 1),
        "harmful_switch":  harmful / max(total, 1),
        "gate_auc":        auc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Venice-H1 re-ranker")
    parser.add_argument("--config", default="venice_h1/configs/default.yaml")
    parser.add_argument("--no_grid", action="store_true",
                        help="Ablation: BASE features only (Df=261)")
    parser.add_argument("--tau", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_grid:
        cfg["model"]["use_grid"] = False

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_grid = cfg["model"]["use_grid"]
    tau = args.tau if args.tau is not None else cfg["model"]["tau"]

    # ---- Data ----
    train_paths = [p for p in cfg["data"]["train_splits"] if Path(p).exists()]
    val_paths   = [p for p in cfg["data"]["val_splits"] if Path(p).exists()]

    if not train_paths:
        print("ERROR: No training data found. Run extract_features.py first.")
        print("Expected paths:", cfg["data"]["train_splits"])
        return

    train_ds = ConcatDataset([FeatureCacheDataset(p, use_grid) for p in train_paths])
    val_ds   = ConcatDataset([FeatureCacheDataset(p, use_grid) for p in val_paths]) \
               if val_paths else None

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True) if val_ds else None

    print(f"Train: {len(train_ds):,} samples  |  "
          f"Val: {len(val_ds) if val_ds else 0:,} samples")
    print(f"Feature set: {'BASE+GRID (Df=936)' if use_grid else 'BASE (Df=261)'}")

    # ---- Model ----
    model = VeniceH1Reranker(**cfg["model"]).to(device)
    print(f"Venice-H1: {model.num_parameters():,} parameters (~11.3M)")

    # ---- Optimizer (Section 3.5: AdamW, lr=5e-4, wd=1e-4) ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"])

    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"]["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # FP16
    scaler = GradScaler() if cfg["training"].get("fp16") else None

    # Logging
    writer = SummaryWriter(cfg["output"]["log_dir"]) if _HAS_TB else None
    ckpt_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_auc = 0.0

    # ---- Training Loop ----
    print(f"\nTraining for {epochs} epochs (batch={bs}, lr={cfg['training']['lr']})")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            features   = batch["features"].to(device)
            det_scores = batch["det_scores"].to(device)
            mask_means = batch["mask_means"].to(device)

            optimizer.zero_grad()

            if scaler:
                with autocast():
                    out = model(features, det_scores, mask_means)
                    losses = compute_loss(out, batch, cfg)
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(features, det_scores, mask_means)
                losses = compute_loss(out, batch, cfg)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += losses["total"].item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        metrics = evaluate(model, val_loader, device, tau=tau) if val_loader else {}

        if writer:
            writer.add_scalar("train/loss", avg_loss, epoch)
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        auc = metrics.get("gate_auc", 0)
        harm = metrics.get("harmful_switch", 0) * 100
        print(f"  Epoch {epoch:2d}/{epochs} | loss={avg_loss:.4f} | "
              f"AUC={auc:.3f} | harmful={harm:.2f}%")

        if auc > best_auc:
            best_auc = auc
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "config": cfg, "metrics": metrics},
                os.path.join(ckpt_dir, "best.pt"))

    if writer:
        writer.close()

    print(f"\nDone. Best Gate AUC: {best_auc:.4f}")
    print(f"Checkpoint: {os.path.join(ckpt_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
