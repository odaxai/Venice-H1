"""
GridCellDeRIS: Multi-Scale Grid Cells on top of frozen DeRIS (ICCV'25 SOTA)

DeRIS architecture:
  - Understanding branch (BEiT-3 Large) → img_feat [B,256,H/16,W/16], text_feat, cls_feat
  - Perception branch (Swin-B + Mask2Former decoder) → pred_masks [B,N,H,W]
  - Grid cells enhance pred_masks via additive correction

IDENTICAL grid cell integration pattern as OneRef/C3VG:
  1. Backbone extracts: spatial_feat [B,H,W,256], language [B,L,256], lang_cls [B,256]
  2. Grid cells: offset = MultiScaleGridCells(spatial_feat, language) → [B,H,W,256]
  3. Enhanced: enhanced_lang = lang_cls + offset → [B,H,W,256]
  4. GC mask: gc_mask = (enhanced_lang * F.normalize(spatial_feat)).sum(-1) → [B,1,H,W]
  5. Final: pred_masks = head_pred_masks + gc_mask  (additive correction)

embed_dim=256 (DeRIS projects everything to 256-dim hidden channels)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridcell_models.grid_cell_oneref import MultiScaleGridCells


class GridCellDeRIS(nn.Module):
    """
    Venice-H1 Grid Cells on DeRIS backbone (ICCV 2025 SOTA).

    Integration: grid cells produce a per-position mask correction added
    to DeRIS's predicted masks. Same additive gated pattern as OneRef/C3VG.
    """

    def __init__(self, deris_model, embed_dim=256, ablation_seg_only=False,
                 unfreeze_layers=0, grid_cfg=None):
        super().__init__()
        self.deris = deris_model
        self._ablation_seg_only = ablation_seg_only
        self._unfreeze_layers = unfreeze_layers
        self._embed_dim = embed_dim

        # Detect BEiT-3 output dim (1024 for Large, 768 for Base)
        vis_enc = self.deris.understanding_branch.vis_enc
        self._beit3_dim = vis_enc.beit3.args.encoder_embed_dim  # 1024 for L

        # FREEZE entire DeRIS model
        for param in self.deris.parameters():
            param.requires_grad = False

        # Multi-Scale Grid Cells at 256-dim (DeRIS hidden channels)
        if not ablation_seg_only:
            gc_heads = grid_cfg.get('num_heads', 8) if grid_cfg else 8
            gc_dropout = grid_cfg.get('dropout', 0.05) if grid_cfg else 0.05
            self.multi_grid = MultiScaleGridCells(
                embed_dim=embed_dim, num_heads=gc_heads, dropout=gc_dropout,
            )
            # Project cls_feat from BEiT-3 dim (1024) → hidden_channels (256)
            # cls_feat comes raw from BEiT-3, not projected like img_feat/text_feat
            if self._beit3_dim != embed_dim:
                self.cls_proj = nn.Linear(self._beit3_dim, embed_dim)
                # Zero-init → cls contribution starts at 0
                # Combined with zero-init MLP output, gc_mask = 0 at init
                # This guarantees EXACT baseline preservation (no random noise)
                nn.init.zeros_(self.cls_proj.weight)
                nn.init.zeros_(self.cls_proj.bias)
            else:
                self.cls_proj = nn.Identity()
        else:
            self.multi_grid = None
            self.cls_proj = None

        # Optionally unfreeze last N BEiT-3 encoder layers
        if unfreeze_layers > 0:
            self._unfreeze_backbone_layers(unfreeze_layers)

    def _unfreeze_backbone_layers(self, n_unfreeze):
        """Unfreeze last N encoder layers of BEiT-3 inside DeRIS."""
        vis_enc = self.deris.understanding_branch.vis_enc
        encoder_layers = vis_enc.beit3.encoder.layers
        n_total = len(encoder_layers)
        start_idx = n_total - n_unfreeze
        unfrozen_count = 0
        for i in range(start_idx, n_total):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True
                unfrozen_count += 1
        print(f"  DeRIS BEiT-3 layers {start_idx}-{n_total-1}: "
              f"UNFROZEN ({unfrozen_count} param tensors)")

        encoder = vis_enc.beit3.encoder
        if hasattr(encoder, 'layer_norm') and encoder.layer_norm is not None:
            for param in encoder.layer_norm.parameters():
                param.requires_grad = True

    def get_gate_value(self):
        if self.multi_grid is not None:
            return torch.tanh(self.multi_grid.scale).item()
        return 0.0

    def get_scale_weights(self):
        """Get softmax-normalized scale weights for 4x4, 8x8, 16x16 grids."""
        if self.multi_grid is not None:
            w = F.softmax(self.multi_grid.scale_weights, dim=0)
            return {'4x4': w[0].item(), '8x8': w[1].item(), '16x16': w[2].item()}
        return {'4x4': 0.33, '8x8': 0.33, '16x16': 0.33}

    def get_param_groups(self, lr_backbone=1e-5, lr_grid=3e-4, lr_gate=None):
        """Param groups with separate LR for gate/scale_weights (10x grid LR)."""
        backbone_params = []
        grid_params = []
        gate_params = []

        for name, param in self.deris.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)

        # Grid cell params: separate gate params from attention params
        if self.multi_grid is not None:
            for name, param in self.multi_grid.named_parameters():
                if name in ('scale', 'scale_weights'):
                    gate_params.append(param)
                else:
                    grid_params.append(param)
        if self.cls_proj is not None and not isinstance(self.cls_proj, nn.Identity):
            for param in self.cls_proj.parameters():
                grid_params.append(param)

        # Gate LR: 10x grid LR by default for faster gate learning
        gate_lr = lr_gate if lr_gate is not None else lr_grid * 10

        groups = []
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': lr_backbone,
                           'name': 'backbone'})
        if grid_params:
            groups.append({'params': grid_params, 'lr': lr_grid,
                           'name': 'grid_cells'})
        if gate_params:
            groups.append({'params': gate_params, 'lr': gate_lr,
                           'name': 'gate_params'})
        return groups

    def print_summary(self):
        deris_frozen = sum(p.numel() for p in self.deris.parameters()
                           if not p.requires_grad)
        deris_trainable = sum(p.numel() for p in self.deris.parameters()
                              if p.requires_grad)
        grid_p = sum(p.numel() for p in self.multi_grid.parameters()) \
            if self.multi_grid is not None else 0
        cls_proj_p = sum(p.numel() for p in self.cls_proj.parameters()) \
            if self.cls_proj is not None and not isinstance(self.cls_proj, nn.Identity) else 0
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n[Venice-H1 on DeRIS] Architecture:")
        print(f"  BEiT-3 dim:              {self._beit3_dim} → "
              f"hidden_channels: {self._embed_dim}")
        print(f"  DeRIS FROZEN:            {deris_frozen:,}")
        print(f"  DeRIS TRAINABLE:         {deris_trainable:,} "
              f"(last {self._unfreeze_layers} BEiT-3 layers)")
        print(f"  MultiScale Grid TRAINABLE: {grid_p:,}")
        print(f"  CLS Projection TRAINABLE:  {cls_proj_p:,}")
        print(f"  Total TRAINABLE:         {total:,}")

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def _grid_cell_mask(self, img_feat, text_feat, cls_feat):
        """
        Grid cell mask correction using 256-dim features from DeRIS.

        Args:
            img_feat: [B, 256, H, W] understanding branch image features
            text_feat: [B, L, 256] projected text features
            cls_feat: [B, 1024] raw BEiT-3 CLS token (projected to 256 here)

        Returns:
            gc_mask: [B, 1, H, W] grid cell mask correction
        """
        if self.multi_grid is None:
            return None

        B, C, H, W = img_feat.shape
        spatial_2d = img_feat.detach().permute(0, 2, 3, 1)  # [B, H, W, 256]
        spatial_norm = F.normalize(spatial_2d, dim=-1)

        # Per-position offset — SAME call as OneRef/C3VG
        query_offset = self.multi_grid(
            spatial_norm,
            text_feat.detach(),
        )  # [B, H, W, 256]

        # Project cls_feat from BEiT-3 dim (1024) → hidden_channels (256)
        cls_proj = self.cls_proj(cls_feat.detach())  # [B, 256]

        # Enhanced language per-position
        enhanced_lang = cls_proj.reshape(B, 1, 1, -1) + query_offset

        # Dot product → mask
        gc_mask = torch.mul(
            enhanced_lang.expand(-1, H, W, -1),
            spatial_norm
        ).sum(dim=-1).unsqueeze(1)  # [B, 1, H, W]

        return gc_mask

    def forward_train(self, img, ref_expr_inds, img_metas,
                      text_attention_mask=None, gt_bbox=None,
                      gt_mask_rle=None, gt_mask_parts_rle=None,
                      rescale=False, epoch=None, img_mini=None, **kwargs):
        """
        Training forward: frozen DeRIS + trainable grid cell mask correction.

        Gradient flow (SAME as C3VG pattern):
          loss_mask → pred_masks_enhanced → gc_mask_up → multi_grid (TRAINABLE)
          All other losses (loss_class, loss_det, etc.) are detached.
        """
        # ===== 1. Run full DeRIS forward (frozen) =====
        with torch.no_grad():
            feat_dict = self.deris.understanding_branch.pre_forward(
                img_mini, ref_expr_inds, text_attention_mask)

            perception_results = self.deris.perception_branch(
                img, feat_dict["img_feat"], feat_dict["text_feat"],
                text_attention_mask, feat_dict["query_feat"])

            understanding_results = self.deris.understanding_branch.post_forward(
                perception_results["query_feat"],
                perception_results,
                feat_dict["img_feat"],
                feat_dict["text_feat"],
                text_attention_mask,
                feat_dict["cls_feat"],
            )

        # ===== 2. Grid cell mask correction (TRAINABLE) =====
        gc_mask = self._grid_cell_mask(
            feat_dict["img_feat"],
            feat_dict["text_feat"],
            feat_dict["cls_feat"],
        )

        # ===== 3. Apply additive correction to pred_masks =====
        pred_masks = perception_results["pred_masks"]  # [B, N, H, W]

        if gc_mask is not None:
            gc_mask_up = F.interpolate(
                gc_mask, size=pred_masks.shape[-2:],
                mode='bilinear', align_corners=False)
            # Additive: detach head output, only gc_mask has gradients
            pred_masks_enhanced = pred_masks.detach() + gc_mask_up
        else:
            pred_masks_enhanced = pred_masks.detach()

        # ===== 4. Build dicts and compute loss through DeRIS head =====
        pred_global_mask = None
        if understanding_results.get("pred_global_mask") is not None:
            pred_global_mask = F.interpolate(
                understanding_results["pred_global_mask"].detach(),
                size=pred_masks_enhanced.shape[-2:],
                mode="bilinear",
            )

        # Detach aux_outputs to prevent gradient flow through frozen layers
        aux_refer = []
        for aux in perception_results.get("aux_outputs_refer", []):
            aux_refer.append({k: v.detach() if isinstance(v, torch.Tensor) else v
                              for k, v in aux.items()})
        aux_perception = []
        for aux in perception_results.get("aux_outputs_perception", []):
            aux_perception.append({k: v.detach() if isinstance(v, torch.Tensor) else v
                                   for k, v in aux.items()})

        pred_dict = {
            "pred_boxes": perception_results["pred_boxes"].detach(),
            "pred_masks": pred_masks_enhanced,
            "pred_logits": perception_results["pred_refer_logits"].detach(),
            "pred_existence": understanding_results["pred_existence"].detach()
                if understanding_results.get("pred_existence") is not None
                else None,
            "pred_global_mask": pred_global_mask,
            "aux_outputs": aux_refer,
        }

        perception_pred_dict = {
            "pred_boxes": perception_results["pred_boxes"].detach(),
            "pred_masks": pred_masks_enhanced,
            "pred_logits": perception_results["pred_logits"].detach(),
            "aux_outputs": aux_perception,
        }

        targets = {
            "mask": gt_mask_rle,
            "bbox": gt_bbox,
            "img_metas": img_metas,
            "epoch": epoch,
            "mask_parts": gt_mask_parts_rle,
        }

        losses_dict = self.deris.head.forward_train(
            predictions=pred_dict,
            perception_prediction=perception_pred_dict,
            targets=targets,
        )

        # Get predictions for metrics (no grad)
        with torch.no_grad():
            predictions = self.deris.get_predictions_parts(
                pred_dict, img_metas, rescale=rescale,
                with_bbox=True, with_mask=True)

        return losses_dict, predictions

    @torch.no_grad()
    def forward_test(self, img, ref_expr_inds, img_metas,
                     text_attention_mask=None, with_bbox=False,
                     with_mask=False, gt_bbox=None, gt_mask_rle=None,
                     gt_mask_parts_rle=None, rescale=False,
                     img_mini=None, **kwargs):

        # Understanding branch pre-forward
        feat_dict = self.deris.understanding_branch.pre_forward(
            img_mini, ref_expr_inds, text_attention_mask)

        # Perception branch
        perception_results = self.deris.perception_branch(
            img, feat_dict["img_feat"], feat_dict["text_feat"],
            text_attention_mask, feat_dict["query_feat"])

        # Grid cell mask correction
        gc_mask = self._grid_cell_mask(
            feat_dict["img_feat"],
            feat_dict["text_feat"],
            feat_dict["cls_feat"],
        )

        pred_masks = perception_results["pred_masks"]
        if gc_mask is not None:
            gc_mask_up = F.interpolate(
                gc_mask, size=pred_masks.shape[-2:],
                mode='bilinear', align_corners=False)
            pred_masks = pred_masks + gc_mask_up

        # Understanding branch post-forward
        understanding_results = self.deris.understanding_branch.post_forward(
            perception_results["query_feat"],
            perception_results,
            feat_dict["img_feat"],
            feat_dict["text_feat"],
            text_attention_mask,
            feat_dict["cls_feat"],
        )

        if understanding_results.get("pred_global_mask") is not None:
            pred_global_mask = F.interpolate(
                understanding_results["pred_global_mask"],
                size=pred_masks.shape[-2:],
                mode="bilinear",
            )
        else:
            pred_global_mask = None

        pred_dict = {
            "pred_boxes": perception_results["pred_boxes"],
            "pred_masks": pred_masks,
            "pred_logits": perception_results["pred_refer_logits"],
            "pred_existence": understanding_results.get("pred_existence"),
            "pred_global_mask": pred_global_mask,
            "aux_outputs": perception_results.get("aux_outputs_refer", []),
        }

        predictions = self.deris.get_predictions_parts(
            pred_dict, img_metas, rescale=rescale,
            with_bbox=with_bbox, with_mask=with_mask)

        return predictions
