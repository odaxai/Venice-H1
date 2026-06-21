"""
Loss functions for Venice-H1 v2 training.

Two modes:
  1. Full model (seg head + grid cells): seg mask loss + cell localization loss
  2. Ablation (seg head only): seg mask loss only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0):
    """Dice loss for binary masks."""
    pred = pred.sigmoid().flatten(1)
    target = target.flatten(1).float()
    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    return 1 - (2 * intersection + smooth) / (union + smooth)


def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """Focal loss for imbalanced binary classification."""
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1).float()
    bce = F.binary_cross_entropy_with_logits(
        pred_flat, target_flat, reduction='none'
    )
    p = pred_flat.sigmoid()
    pt = p * target_flat + (1 - p) * (1 - target_flat)
    focal = ((1 - pt) ** gamma) * bce
    if alpha >= 0:
        alpha_t = alpha * target_flat + (1 - alpha) * (1 - target_flat)
        focal = alpha_t * focal
    return focal.mean(1)


def boundary_loss(pred, target):
    """Boundary loss using Laplacian edge detection."""
    laplacian = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)

    pred_s = pred.sigmoid()
    if pred_s.dim() == 3:
        pred_s = pred_s.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1).float()

    pred_edge = F.conv2d(pred_s, laplacian, padding=1).abs()
    gt_edge = F.conv2d(target, laplacian, padding=1).abs()

    return F.mse_loss(pred_edge, gt_edge, reduction='none').mean([1, 2, 3])


def compute_grid_cell_losses(grid_out, target_bbox, target_mask,
                              l_mask=1.0, l_ref=1.0, l_aux=0.2,
                              l_bce=2.0, l_dice=5.0, l_focal=2.0,
                              l_boundary=1.0):
    """
    Compute losses for Venice-H1 v2.

    Supports both v1 (round_outputs) and v2 (seg_mask + cell_logits) formats.

    For v2:
      - Seg mask loss (dice + focal + bce + boundary) on the final seg_mask
      - Cell localization loss (CE on cell_logits vs GT cell)
    """
    device = target_bbox.device
    B = target_bbox.shape[0]
    losses = {}
    total_loss = torch.tensor(0.0, device=device)

    # ===== V1 path: round_outputs exist (legacy) =====
    if 'round_outputs' in grid_out:
        round_outputs = grid_out['round_outputs']
        num_rounds = len(round_outputs)
        H_out = grid_out['H_out']
        grid_size = int(round_outputs[0]['ref_logits_all'].shape[1] ** 0.5)

        gt_mask_resized = F.interpolate(
            target_mask.float(), size=(H_out, H_out),
            mode='bilinear', align_corners=False
        ).squeeze(1)

        cx = target_bbox[:, 0]
        cy = target_bbox[:, 1]
        gt_gx = (cx * grid_size).clamp(0, grid_size - 1e-6).long()
        gt_gy = (cy * grid_size).clamp(0, grid_size - 1e-6).long()
        gt_cells = gt_gy * grid_size + gt_gx

        for r, rout in enumerate(round_outputs):
            weight = 1.0 if r == num_rounds - 1 else l_aux
            masks_per_cell = rout['masks_per_cell']
            ref_logits = rout['ref_logits']
            topk_idx = rout['topk_idx']
            K = masks_per_cell.shape[1]

            gt_cells_exp = gt_cells.unsqueeze(1)
            gt_in_topk = (topk_idx == gt_cells_exp)

            round_mask_loss = torch.tensor(0.0, device=device)
            round_ref_loss = torch.tensor(0.0, device=device)

            for b in range(B):
                gt_mask_b = gt_mask_resized[b]
                gt_positions = gt_in_topk[b].nonzero(as_tuple=True)[0]
                if len(gt_positions) > 0:
                    gt_k = gt_positions[0].item()
                else:
                    with torch.no_grad():
                        pred_binary = (masks_per_cell[b].sigmoid() > 0.5).float()
                        ious = (pred_binary * gt_mask_b.unsqueeze(0)).sum([1, 2])
                        ious = ious / ((pred_binary + gt_mask_b.unsqueeze(0)).clamp(0, 1).sum([1, 2]) + 1e-6)
                        gt_k = ious.argmax().item()

                pred_mask_b = masks_per_cell[b, gt_k].clamp(-20, 20)
                d = dice_loss(pred_mask_b.unsqueeze(0), gt_mask_b.unsqueeze(0)).mean()
                f = focal_loss(pred_mask_b.unsqueeze(0), gt_mask_b.unsqueeze(0)).mean()
                bce = F.binary_cross_entropy_with_logits(pred_mask_b, gt_mask_b, reduction='mean')
                bl = boundary_loss(pred_mask_b.unsqueeze(0), gt_mask_b.unsqueeze(0)).mean()
                sample_loss = l_bce * bce + l_dice * d + l_focal * f + l_boundary * bl
                if torch.isnan(sample_loss) or torch.isinf(sample_loss):
                    sample_loss = torch.tensor(0.0, device=device)
                round_mask_loss = round_mask_loss + sample_loss

                ref_target = torch.zeros(K, device=device)
                if len(gt_positions) > 0:
                    ref_target[gt_positions[0]] = 1.0
                ref_loss_b = F.binary_cross_entropy_with_logits(
                    ref_logits[b].clamp(-20, 20), ref_target, reduction='mean')
                if torch.isnan(ref_loss_b) or torch.isinf(ref_loss_b):
                    ref_loss_b = torch.tensor(0.0, device=device)
                round_ref_loss = round_ref_loss + ref_loss_b

            round_mask_loss = round_mask_loss / B
            round_ref_loss = round_ref_loss / B
            losses[f'mask_r{r}'] = round_mask_loss.item()
            losses[f'ref_r{r}'] = round_ref_loss.item()
            total_loss = total_loss + weight * (l_mask * round_mask_loss + l_ref * round_ref_loss)

        if 'ref_exists_logits' in grid_out:
            ref_exists_target = torch.ones(B, device=device)
            ref_exists_loss = F.binary_cross_entropy_with_logits(
                grid_out['ref_exists_logits'].clamp(-20, 20),
                ref_exists_target, reduction='mean')
            if not (torch.isnan(ref_exists_loss) or torch.isinf(ref_exists_loss)):
                total_loss = total_loss + ref_exists_loss
                losses['ref_exists'] = ref_exists_loss.item()

    # ===== Seg mask output loss (works for both v1 and v2) =====
    if 'seg_mask' in grid_out and grid_out['seg_mask'] is not None:
        seg_mask_out = grid_out['seg_mask']
        H_seg = seg_mask_out.shape[2]
        gt_seg = F.interpolate(
            target_mask.float(), size=(H_seg, H_seg),
            mode='bilinear', align_corners=False
        )
        seg_d = dice_loss(seg_mask_out.squeeze(1), gt_seg.squeeze(1)).mean()
        seg_bce = F.binary_cross_entropy_with_logits(
            seg_mask_out.squeeze(1).clamp(-20, 20),
            gt_seg.squeeze(1), reduction='mean')
        seg_f = focal_loss(seg_mask_out.squeeze(1), gt_seg.squeeze(1)).mean()
        seg_b = boundary_loss(seg_mask_out.squeeze(1), gt_seg.squeeze(1)).mean()
        seg_loss = l_dice * seg_d + l_bce * seg_bce + l_focal * seg_f + l_boundary * seg_b
        if torch.isnan(seg_loss) or torch.isinf(seg_loss):
            seg_loss = torch.tensor(0.0, device=device)
        losses['seg_out'] = seg_loss.item()
        total_loss = total_loss + l_mask * seg_loss

    # ===== Cell localization loss (v2: cell_logits directly) =====
    if 'cell_logits' in grid_out and 'round_outputs' not in grid_out:
        cell_logits = grid_out['cell_logits']  # [B, C]
        grid_size = int(cell_logits.shape[1] ** 0.5)
        cx = target_bbox[:, 0]
        cy = target_bbox[:, 1]
        gt_gx = (cx * grid_size).clamp(0, grid_size - 1e-6).long()
        gt_gy = (cy * grid_size).clamp(0, grid_size - 1e-6).long()
        gt_cells = gt_gy * grid_size + gt_gx
        cell_loss = F.cross_entropy(cell_logits, gt_cells)
        if not (torch.isnan(cell_loss) or torch.isinf(cell_loss)):
            losses['cell_loc'] = cell_loss.item()
            total_loss = total_loss + l_ref * cell_loss

    # Final NaN guard
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    losses['total'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss

    return losses, total_loss
