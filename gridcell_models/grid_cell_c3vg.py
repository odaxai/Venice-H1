"""
GridCellC3VG: Multi-Scale Grid Cells on top of frozen C3VG (AAAI'25)

IDENTICAL grid cell integration pattern as GridCellOneRef:
  1. Backbone extracts: spatial_feat [B,H,W,768], language [B,L,768], lang_cls [B,768]
  2. Grid cells: offset = MultiScaleGridCells(spatial_feat, language) → [B,H,W,768]
  3. Enhanced: enhanced_lang = lang_cls + offset → [B,H,W,768]
  4. GC mask: gc_mask = (enhanced_lang * F.normalize(spatial_feat)).sum(-1) → [B,1,H,W]
  5. Final: pred_mask = head_pred_mask + gc_mask  (additive correction)

The C3VG head runs frozen and produces its own mask. Grid cells produce an
additive correction via per-position language offsets — same mechanism as OneRef.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridcell_models.grid_cell_oneref import MultiScaleGridCells


class GridCellC3VG(nn.Module):
    """
    Venice-H1 Grid Cells on C3VG backbone.

    Uses the SAME per-position offset pattern as GridCellOneRef:
    - MultiScaleGridCells produces spatial query offsets [B,H,W,D]
    - Offsets enhance language_cls per-position
    - Dot product with normalized spatial features → mask correction
    - Added to C3VG head's predicted mask (gated, additive)
    """

    def __init__(self, c3vg_model, embed_dim=768, ablation_seg_only=False,
                 unfreeze_layers=0, grid_cfg=None):
        super().__init__()
        self.c3vg = c3vg_model
        self._ablation_seg_only = ablation_seg_only
        self._unfreeze_layers = unfreeze_layers
        self._embed_dim = embed_dim
        self.patch_size = self.c3vg.patch_size

        # FREEZE entire C3VG model
        for param in self.c3vg.parameters():
            param.requires_grad = False

        # Multi-Scale Grid Cells (our contribution) — SAME as OneRef
        if not ablation_seg_only:
            gc_heads = grid_cfg.get('num_heads', 8) if grid_cfg else 8
            gc_dropout = grid_cfg.get('dropout', 0.05) if grid_cfg else 0.05
            self.multi_grid = MultiScaleGridCells(
                embed_dim=embed_dim, num_heads=gc_heads, dropout=gc_dropout,
            )
        else:
            self.multi_grid = None

        # Optionally unfreeze last N BEiT-3 encoder layers
        if unfreeze_layers > 0:
            self._unfreeze_backbone_layers(unfreeze_layers)

    def _unfreeze_backbone_layers(self, n_unfreeze):
        """Unfreeze last N encoder layers of BEiT-3 inside C3VG."""
        encoder_layers = self.c3vg.vis_enc.beit3.encoder.layers
        n_total = len(encoder_layers)
        start_idx = n_total - n_unfreeze
        unfrozen_count = 0
        for i in range(start_idx, n_total):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True
                unfrozen_count += 1
        print(f"  C3VG backbone layers {start_idx}-{n_total-1}: "
              f"UNFROZEN ({unfrozen_count} param tensors)")

        encoder = self.c3vg.vis_enc.beit3.encoder
        if hasattr(encoder, 'layer_norm') and encoder.layer_norm is not None:
            for param in encoder.layer_norm.parameters():
                param.requires_grad = True

    def get_gate_value(self):
        if self.multi_grid is not None:
            return torch.tanh(self.multi_grid.scale).item()
        return 0.0

    def get_param_groups(self, lr_backbone=1e-5, lr_grid=3e-4):
        """Return param groups with different LRs for dual-LR optimizer."""
        backbone_params = []
        grid_params = []

        for name, param in self.c3vg.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)

        if self.multi_grid is not None:
            for param in self.multi_grid.parameters():
                grid_params.append(param)

        groups = []
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': lr_backbone,
                           'name': 'backbone'})
        if grid_params:
            groups.append({'params': grid_params, 'lr': lr_grid,
                           'name': 'grid_cells'})
        return groups

    def print_summary(self):
        c3vg_frozen = sum(p.numel() for p in self.c3vg.parameters()
                          if not p.requires_grad)
        c3vg_trainable = sum(p.numel() for p in self.c3vg.parameters()
                             if p.requires_grad)
        grid_p = sum(p.numel() for p in self.multi_grid.parameters()) \
            if self.multi_grid is not None else 0
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n[Venice-H1 on C3VG] Architecture:")
        print(f"  C3VG FROZEN:             {c3vg_frozen:,}")
        print(f"  C3VG TRAINABLE:          {c3vg_trainable:,} "
              f"(last {self._unfreeze_layers} layers)")
        print(f"  MultiScale Grid TRAINABLE: {grid_p:,}")
        print(f"  Total TRAINABLE:         {total:,}")

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def _grid_cell_mask(self, img_feat_raw, text_feat, cls_feat):
        """
        Produce grid cell mask correction — IDENTICAL to OneRef pattern.

        Args:
            img_feat_raw: [B, 768, H, W] raw backbone features (768-dim)
            text_feat: [B, L, 768] language token features
            cls_feat: [B, 768] CLS token

        Returns:
            gc_mask: [B, 1, H', W'] grid cell mask correction
                     (H' = img_feat_raw spatial size)
        """
        if self.multi_grid is None:
            return None

        B, C, H, W = img_feat_raw.shape

        # Convert to [B, H, W, D] format (same as OneRef)
        spatial_2d = img_feat_raw.detach().permute(0, 2, 3, 1)  # [B, H, W, 768]
        spatial_norm = F.normalize(spatial_2d, dim=-1)

        # Grid cells produce per-position offset — SAME call as OneRef
        query_offset = self.multi_grid(
            spatial_norm,                   # [B, H, W, 768] — like seg_feat_norm
            text_feat.detach(),             # [B, L, 768]    — like language_feat
        )  # → [B, H, W, 768]

        # Enhanced language per-position — SAME as OneRef
        enhanced_lang = cls_feat.detach().reshape(B, 1, 1, -1) + query_offset

        # Dot product → mask — SAME as OneRef
        gc_mask = torch.mul(
            enhanced_lang.expand(-1, H, W, -1),
            spatial_norm
        ).sum(dim=-1).unsqueeze(1)  # [B, 1, H, W]

        return gc_mask

    def forward_train(self, img, ref_expr_inds, img_metas,
                      text_attention_mask=None, gt_bbox=None,
                      gt_mask_vertices=None, mass_center=None,
                      gt_mask=None, rescale=False, epoch=None,
                      visual=True, **kwargs):
        B, _, H, W = img.shape

        # ===== 1. Extract features from C3VG backbone (frozen) =====
        with torch.no_grad():
            img_feat, text_feat, cls_feat = self.c3vg.extract_visual_language(
                img, ref_expr_inds, text_attention_mask)

        # Raw backbone features at 768-dim [B, 768, H/16, W/16]
        img_feat_raw = img_feat.transpose(-1, -2).reshape(
            B, -1, H // self.patch_size, W // self.patch_size)

        # ===== 2. Grid cell mask correction (per-position, same as OneRef) =====
        gc_mask = self._grid_cell_mask(img_feat_raw, text_feat, cls_feat)

        # ===== 3. Run C3VG head (frozen, produces its own mask) =====
        targets = {"mask": gt_mask, "bbox": gt_bbox,
                   "img_metas": img_metas, "epoch": epoch}

        # Head runs with detached inputs (frozen)
        with torch.no_grad():
            losses_dict, pred_dict, extra_dict = self.c3vg.head.forward_train(
                img_feat_raw.detach(), targets, cls_feat.detach(),
                text_feat.detach(), text_attention_mask, img)

        # ===== 4. Add grid cell correction to head's mask =====
        if gc_mask is not None:
            # Interpolate gc_mask to match pred_mask size
            pred_mask = pred_dict['pred_mask']  # [B, 1, H, W]
            gc_mask_up = F.interpolate(
                gc_mask, size=pred_mask.shape[-2:],
                mode='bilinear', align_corners=False)

            # Additive correction (gated by MultiScaleGridCells.scale via tanh)
            pred_mask_enhanced = pred_mask.detach() + gc_mask_up
            pred_dict['pred_mask'] = pred_mask_enhanced

            # Recompute losses with enhanced mask
            import pycocotools.mask as maskUtils
            import numpy as np
            device = img.device
            target_mask_tensor = torch.from_numpy(
                np.concatenate([maskUtils.decode(t)[None]
                                for t in gt_mask])
            ).to(device).float().unsqueeze(1)

            # Dice + BCE loss on enhanced mask
            mask_pred_sig = pred_mask_enhanced.sigmoid()
            mask_flat = mask_pred_sig.flatten(1)
            target_flat = target_mask_tensor.flatten(1)

            # Dice loss
            numerator = 2 * (mask_flat * target_flat).sum(1)
            denominator = mask_flat.sum(-1) + target_flat.sum(-1)
            dice_loss = 1 - (numerator + 1) / (denominator + 1)

            # BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_mask_enhanced, target_mask_tensor, reduction='none'
            ).mean()

            # Replace mask loss (keep det and cons from frozen head)
            gc_mask_loss = dice_loss.mean() + bce_loss
            losses_dict['loss_mask'] = gc_mask_loss
            losses_dict['loss_det'] = losses_dict['loss_det'].detach()
            losses_dict['loss_cons'] = losses_dict.get(
                'loss_cons', torch.tensor(0.0, device=device)).detach()

        # Get predictions for metrics
        with torch.no_grad():
            predictions = self.c3vg.get_predictions(
                pred_dict, img_metas, rescale=rescale,
                threshold=getattr(self.c3vg, 'threshold', 0.5))

        return losses_dict, predictions

    @torch.no_grad()
    def forward_test(self, img, ref_expr_inds, img_metas,
                     text_attention_mask=None, with_bbox=False,
                     with_mask=False, gt_bbox=None, gt_mask=None,
                     rescale=False, visual=True, **kwargs):
        B, _, H, W = img.shape

        # Extract features
        img_feat, text_feat, cls_feat = self.c3vg.extract_visual_language(
            img, ref_expr_inds, text_attention_mask)

        img_feat_raw = img_feat.transpose(-1, -2).reshape(
            B, -1, H // self.patch_size, W // self.patch_size)

        # Grid cell mask correction
        gc_mask = self._grid_cell_mask(img_feat_raw, text_feat, cls_feat)

        # Run head
        pred_dict, extra_dict = self.c3vg.head.forward_test(
            img_feat_raw, cls_feat, text_feat,
            text_attention_mask, img)

        # Add grid cell correction
        if gc_mask is not None:
            pred_mask = pred_dict['pred_mask']
            gc_mask_up = F.interpolate(
                gc_mask, size=pred_mask.shape[-2:],
                mode='bilinear', align_corners=False)
            pred_dict['pred_mask'] = pred_mask + gc_mask_up

        predictions = self.c3vg.get_predictions(
            pred_dict, img_metas, rescale=rescale,
            threshold=getattr(self.c3vg, 'threshold', 0.5))

        return predictions
