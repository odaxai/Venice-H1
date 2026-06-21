#!/usr/bin/env python3
"""
Feature extraction from frozen DeRIS backbone (Section 3.1-3.3 of Venice-H1).

Produces cached .pt files containing per-sample features:
  - query_feat:   [N_samples, N, D]   query embeddings (D=256)
  - det_scores:   [N_samples, N]      detection scores
  - query_ious:   [N_samples, N]      per-query IoU vs GT
  - oracle_idx:   [N_samples]         best query index
  - mask_mean:    [N_samples, N]      μ_i = mean(P_i)
  - mask_max:     [N_samples, N]      p̂_i = max(P_i)
  - mask_area:    [N_samples, N]      a_i = mean(P_i > 0.5)
  - mask_std:     [N_samples, N]      σ_i = std(P_i)
  - grid_mean_4:  [N_samples, N, 16]  AvgPool 4×4
  - grid_max_4:   [N_samples, N, 16]  MaxPool 4×4
  - boundary_4:   [N_samples, N]      boundary energy at 4×4
  - grid_mean_8:  [N_samples, N, 64]  AvgPool 8×8
  - grid_max_8:   [N_samples, N, 64]  MaxPool 8×8
  - boundary_8:   [N_samples, N]      boundary energy at 8×8
  - grid_mean_16: [N_samples, N, 256] AvgPool 16×16
  - grid_max_16:  [N_samples, N, 256] MaxPool 16×16
  - boundary_16:  [N_samples, N]      boundary energy at 16×16

Usage:
    python scripts/extract_features.py \\
        --deris_checkpoint /path/to/deris_l.pth \\
        --data_root /path/to/refcoco/ \\
        --dataset refcoco --split val \\
        --output data/
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def compute_mask_statistics(mask_probs: torch.Tensor) -> dict:
    """
    Compute scalar mask statistics (Section 3.2, Eq. 1).

    Args:
        mask_probs: [N, H, W] sigmoid mask probabilities

    Returns:
        dict with mask_mean, mask_max, mask_area, mask_std (each [N])
    """
    N, H, W = mask_probs.shape
    flat = mask_probs.reshape(N, -1)

    return {
        "mask_mean": flat.mean(dim=1),              # μ_i
        "mask_max":  flat.max(dim=1).values,        # p̂_i
        "mask_area": (flat > 0.5).float().mean(1),  # a_i
        "mask_std":  flat.std(dim=1),               # σ_i
    }


def compute_grid_signatures(mask_probs: torch.Tensor, grid_size: int) -> dict:
    """
    Compute multi-scale grid signatures (Section 3.3, Eqs. 2-4).

    Args:
        mask_probs: [N, H, W] mask probabilities
        grid_size:  G (one of 4, 8, 16)

    Returns:
        dict with grid_mean, grid_max, boundary (per query)
    """
    N = mask_probs.shape[0]
    G = grid_size

    # Reshape for pooling: [N, 1, H, W]
    x = mask_probs.unsqueeze(1)

    # Eq. 2: grid mean (AvgPool)
    grid_mean = F.adaptive_avg_pool2d(x, (G, G)).reshape(N, G * G)

    # Eq. 3: grid max (MaxPool)
    grid_max = F.adaptive_max_pool2d(x, (G, G)).reshape(N, G * G)

    # Eq. 4: boundary energy (mean absolute gradient of grid_mean)
    grid_2d = grid_mean.reshape(N, G, G)
    dx = (grid_2d[:, :, 1:] - grid_2d[:, :, :-1]).abs().mean(dim=(1, 2))
    dy = (grid_2d[:, 1:, :] - grid_2d[:, :-1, :]).abs().mean(dim=(1, 2))
    boundary = 0.5 * (dx + dy)

    return {
        f"grid_mean_{G}":  grid_mean,
        f"grid_max_{G}":   grid_max,
        f"boundary_{G}":   boundary,
    }


def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute IoU between binary masks."""
    pred = (pred_mask > 0.5).float()
    gt = gt_mask.float()
    intersection = (pred * gt).sum()
    union = (pred + gt).clamp(0, 1).sum()
    if union < 1:
        return 0.0
    return (intersection / union).item()


def extract_sample_features(
    mask_logits: torch.Tensor,
    query_embeddings: torch.Tensor,
    det_scores: torch.Tensor,
    gt_mask: torch.Tensor,
) -> dict:
    """
    Extract all Venice-H1 features for one sample.

    Args:
        mask_logits:      [N, H, W] raw mask logits from DeRIS
        query_embeddings: [N, D] query embeddings
        det_scores:       [N] detection scores
        gt_mask:          [H_gt, W_gt] ground-truth binary mask

    Returns:
        dict with all features for this sample
    """
    N = mask_logits.shape[0]

    # Eq. 1: mask probabilities
    mask_probs = torch.sigmoid(mask_logits)  # [N, H, W]

    # Mask statistics (Section 3.2)
    stats = compute_mask_statistics(mask_probs)

    # Multi-scale grid signatures (Section 3.3)
    grid_4  = compute_grid_signatures(mask_probs, 4)
    grid_8  = compute_grid_signatures(mask_probs, 8)
    grid_16 = compute_grid_signatures(mask_probs, 16)

    # Compute IoU for each query vs GT
    H_gt, W_gt = gt_mask.shape
    mask_probs_resized = F.interpolate(
        mask_probs.unsqueeze(1), size=(H_gt, W_gt),
        mode='bilinear', align_corners=False
    ).squeeze(1)

    query_ious = torch.tensor([
        compute_iou(mask_probs_resized[i], gt_mask) for i in range(N)
    ])
    oracle_idx = query_ious.argmax().item()

    return {
        "query_feat":  query_embeddings,   # [N, D]
        "det_scores":  det_scores,         # [N]
        "query_ious":  query_ious,         # [N]
        "oracle_idx":  oracle_idx,
        **stats,
        **grid_4,
        **grid_8,
        **grid_16,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract Venice-H1 features from frozen DeRIS.")
    parser.add_argument("--deris_checkpoint", type=str, required=True,
                        help="Path to frozen DeRIS-L/B checkpoint")
    parser.add_argument("--data_root", type=str, default="data/",
                        help="Root directory for RefCOCO data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "val", "testA", "testB", "test"])
    parser.add_argument("--output", type=str, default="data/",
                        help="Output directory for cached features")
    parser.add_argument("--n_queries", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    print(f"Venice-H1 Feature Extraction")
    print(f"  Backbone: {args.deris_checkpoint}")
    print(f"  Dataset:  {args.dataset} / {args.split}")
    print(f"  Device:   {device}")
    print()

    # ---- Load DeRIS model ----
    # NOTE: Adapt this import to your DeRIS installation path
    # from deris.model import build_deris
    # model = build_deris(args.deris_checkpoint).to(device).eval()
    print("=" * 60)
    print("IMPORTANT: You must adapt the model loading section below")
    print("to your DeRIS installation. See comments in this script.")
    print("=" * 60)
    print()
    print("Expected DeRIS outputs per sample:")
    print("  - query_embeddings: [N, 256] (N=10 candidate queries)")
    print("  - mask_logits:      [N, H, W] (mask predictions)")
    print("  - det_scores:       [N] (detection confidence scores)")
    print()
    print("Once you have DeRIS producing these outputs, the feature")
    print("extraction loop below handles everything else automatically.")
    print()

    # ---- Placeholder: replace with your data loader ----
    # dataloader = build_refcoco_loader(args.data_root, args.dataset,
    #                                    args.split, batch_size=1)
    #
    # all_features = []
    # for batch in tqdm(dataloader, desc=f"Extracting {args.split}"):
    #     img = batch["image"].to(device)
    #     expr = batch["expression"]
    #     gt_mask = batch["gt_mask"].to(device)
    #
    #     with torch.no_grad():
    #         outputs = model(img, expr)
    #         mask_logits = outputs["pred_masks"][:args.n_queries]
    #         query_emb = outputs["query_embeddings"][:args.n_queries]
    #         scores = outputs["det_scores"][:args.n_queries]
    #
    #     feats = extract_sample_features(
    #         mask_logits.squeeze(0), query_emb.squeeze(0),
    #         scores.squeeze(0), gt_mask.squeeze(0))
    #     all_features.append(feats)
    #
    # ---- Stack and save ----
    # output_path = os.path.join(
    #     args.output,
    #     f"cached_{args.split}_{args.dataset}_unc_feats.pt")
    # stacked = {k: torch.stack([f[k] for f in all_features])
    #            for k in all_features[0].keys()
    #            if k != "oracle_idx"}
    # stacked["oracle_idx"] = torch.tensor(
    #     [f["oracle_idx"] for f in all_features])
    # torch.save(stacked, output_path)
    # print(f"Saved {len(all_features)} samples → {output_path}")

    print("Feature extraction template ready.")
    print("Uncomment the dataloader section above and adapt to your setup.")


if __name__ == "__main__":
    main()
