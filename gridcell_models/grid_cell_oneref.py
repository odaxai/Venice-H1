"""
GridCellOneRef v5: Multi-Scale Grid Cells + Partial Backbone Unfreeze

Experimental results guiding this design:
    Baseline RES frozen:  mIoU=78.44%  (ceiling with everything frozen)
    v4 post-injection:    mIoU=79.04%  (+0.6%, saturates quickly)

Strategy: unfreeze last 2 BEiT-3 layers (LR=1e-5) + multi-scale grid cells
operating at 4x4, 8x8, 16x16 simultaneously for coarse-to-fine spatial reasoning.

Architecture:
    1. BEiT-3 layers 0-9 (FROZEN)
    2. BEiT-3 layers 10-11 (TRAINABLE, LR=1e-5) — adapt high-level features
    3. Seg head (FROZEN RES pretrained)
    4. MultiScaleGridCells (TRAINABLE, LR=2e-4) — post-seghead spatial queries
    5. mask = seg_features DOT (language_cls + multi_scale_query)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models import create_model


class SpatialLanguageQuery(nn.Module):
    """Single-scale spatial language query at a given grid resolution."""
    def __init__(self, embed_dim, grid_size=8, num_heads=8, dropout=0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        self.downsample_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        # Zero-init output layer → grid cell offset starts at 0
        # This guarantees gc_mask = 0 at init (exact baseline preservation)
        nn.init.zeros_(self.query_proj[-1].weight)
        nn.init.zeros_(self.query_proj[-1].bias)

    def forward(self, seg_features_2d, language_feat):
        """
        Args:
            seg_features_2d: [B, H, W, D]
            language_feat: [B, L, D]
        Returns:
            query_offset: [B, H, W, D]
        """
        B, H, W, D = seg_features_2d.shape
        gs = self.grid_size

        seg_2d = seg_features_2d.permute(0, 3, 1, 2)
        grid_feat = F.adaptive_avg_pool2d(seg_2d, (gs, gs))
        grid_feat = grid_feat.permute(0, 2, 3, 1).reshape(B, gs*gs, D)
        grid_feat = self.downsample_proj(grid_feat)

        attended, _ = self.cross_attn(grid_feat, language_feat, language_feat)
        grid_feat = self.norm1(grid_feat + attended)

        refined, _ = self.self_attn(grid_feat, grid_feat, grid_feat)
        grid_feat = self.norm2(grid_feat + refined)

        query_offset = self.query_proj(grid_feat)

        query_2d = query_offset.reshape(B, gs, gs, D).permute(0, 3, 1, 2)
        query_up = F.interpolate(query_2d, size=(H, W), mode='bilinear',
                                 align_corners=False)
        return query_up.permute(0, 2, 3, 1)


class MultiScaleGridCells(nn.Module):
    """
    Multi-scale grid cells operating at 4x4, 8x8, 16x16 simultaneously.

    - 4x4 (16 cells): coarse position ("left half", "the big one")
    - 8x8 (64 cells): medium position ("the cat near the table")
    - 16x16 (256 cells): fine position ("the left eye")

    Each scale produces its query offset independently, then they are
    AVERAGED (not concatenated) to avoid OOM on high-resolution features.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.05):
        super().__init__()
        self.scale_4 = SpatialLanguageQuery(embed_dim, grid_size=4,
                                             num_heads=num_heads, dropout=dropout)
        self.scale_8 = SpatialLanguageQuery(embed_dim, grid_size=8,
                                             num_heads=num_heads, dropout=dropout)
        self.scale_16 = SpatialLanguageQuery(embed_dim, grid_size=16,
                                              num_heads=num_heads, dropout=dropout)

        # Per-scale learnable weights
        self.scale_weights = nn.Parameter(torch.zeros(3))

        # Global scale: starts at 0 → pure baseline
        # For quick experiments, init to 0.5 (warm start) via command line
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, seg_features_2d, language_feat):
        """
        Args:
            seg_features_2d: [B, H, W, D]
            language_feat: [B, L, D]
        Returns:
            fused_query: [B, H, W, D] multi-scale spatial query offset
        """
        q4 = self.scale_4(seg_features_2d, language_feat)
        q8 = self.scale_8(seg_features_2d, language_feat)
        q16 = self.scale_16(seg_features_2d, language_feat)

        # Weighted average (memory efficient, no concat at full resolution)
        w = F.softmax(self.scale_weights, dim=0)
        fused = w[0] * q4 + w[1] * q8 + w[2] * q16

        return torch.tanh(self.scale) * fused


class GridCellOneRef(nn.Module):
    """
    Venice-H1 v5: Multi-Scale Grid Cells + Partial Backbone Unfreeze.

    Key changes from v4:
    - BEiT-3 last 2 layers UNFROZEN (LR=1e-5, conservative adaptation)
    - Multi-scale grid cells at 4x4 + 8x8 + 16x16 (coarse-to-fine)
    - Forward passes last 2 backbone layers WITH gradients
    - Seg head stays FROZEN (pretrained RES)

    Supports:
    - --ablation_seg_only: no grid cells, pure OneRef
    - --unfreeze_layers N: how many backbone layers to unfreeze (default 2)
    """
    def __init__(self, oneref_args, grid_cfg=None, ablation_seg_only=False):
        super().__init__()
        self._ablation_seg_only = ablation_seg_only
        self._grid_size = grid_cfg.get('grid_size', 8) if grid_cfg else 8
        self._unfreeze_layers = getattr(oneref_args, 'unfreeze_layers', 2)

        import sys
        oneref_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'OneRef')
        if oneref_path not in sys.path:
            sys.path.insert(0, oneref_path)
        import models.OneRef_model as OneRef_model  # noqa: F401
        import models.utils as beit3_utils  # noqa: F401

        oneref_args.use_mask_loss = True

        if not oneref_args.model.endswith('grounding'):
            model_config = f"{oneref_args.model}_grounding"
        else:
            model_config = oneref_args.model

        self.backbone = create_model(
            model_config, sys_args=oneref_args, pretrained=False,
            drop_path_rate=getattr(oneref_args, 'drop_path', 0.1),
            vocab_size=getattr(oneref_args, 'vocab_size', 64010),
            checkpoint_activations=getattr(oneref_args, 'checkpoint_activations', None),
        )

        embed_dim = self.backbone.beit3.args.encoder_embed_dim
        self._embed_dim = embed_dim
        self._num_patches = self.backbone.num_visu_token
        self._patch_grid = int(self._num_patches ** 0.5)

        # FREEZE entire backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Multi-Scale Grid Cells (our contribution)
        if not ablation_seg_only:
            gc_heads = min(grid_cfg.get('num_heads', 8), 8) if grid_cfg else 8
            gc_dropout = grid_cfg.get('dropout', 0.05) if grid_cfg else 0.05
            self.multi_grid = MultiScaleGridCells(
                embed_dim=embed_dim, num_heads=gc_heads, dropout=gc_dropout,
            )
        else:
            self.multi_grid = None

        self._has_seg = hasattr(self.backbone, 'seg_conv1')
        self.alpha = None  # compat

    def get_gate_value(self):
        if self.multi_grid is not None:
            return torch.tanh(self.multi_grid.scale).item()
        return 0.0

    def load_oneref_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"[Checkpoint] Loaded: {checkpoint_path}")
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")

        # Check seg head
        seg_keys = [k for k in missing if any(
            s in k for s in ['seg_conv', 'bn1', 'bn2', 'bn3'])]
        seg_is_pretrained = len(seg_keys) == 0

        if seg_is_pretrained:
            print(f"  Seg head: PRETRAINED (FROZEN)")
        else:
            print(f"  Seg head: RANDOM INIT ({len(seg_keys)} missing, UNFROZEN)")
            for name, param in self.backbone.named_parameters():
                if any(k in name for k in ['seg_conv1', 'seg_conv2', 'seg_conv3',
                                            'bn1', 'bn2', 'bn3']):
                    param.requires_grad = True

        # UNFREEZE last N encoder layers for adaptation
        n_unfreeze = self._unfreeze_layers
        if n_unfreeze > 0:
            encoder_layers = self.backbone.beit3.encoder.layers
            n_total = len(encoder_layers)
            start_idx = n_total - n_unfreeze
            unfrozen_count = 0
            for i in range(start_idx, n_total):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
            print(f"  Backbone layers {start_idx}-{n_total-1}: "
                  f"UNFROZEN ({unfrozen_count} param tensors)")

            # Also unfreeze layer_norm if exists (after last layer)
            if self.backbone.beit3.encoder.layer_norm is not None:
                for param in self.backbone.beit3.encoder.layer_norm.parameters():
                    param.requires_grad = True

        # Print summary
        backbone_frozen = sum(p.numel() for p in self.backbone.parameters()
                              if not p.requires_grad)
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters()
                                 if p.requires_grad)
        grid_p = sum(p.numel() for p in self.multi_grid.parameters()) \
            if self.multi_grid is not None else 0
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n[Venice-H1 v5] Architecture:")
        print(f"  Backbone FROZEN:         {backbone_frozen:,}")
        print(f"  Backbone TRAINABLE:      {backbone_trainable:,} "
              f"(last {n_unfreeze} layers)")
        print(f"  MultiScale Grid TRAINABLE: {grid_p:,}")
        print(f"  Total TRAINABLE:         {total:,}")

    def get_param_groups(self, lr_backbone=1e-5, lr_grid=2e-4):
        """Return param groups with different LRs for dual-LR optimizer."""
        backbone_params = []
        grid_params = []

        # Backbone trainable params (unfrozen layers)
        for param in self.backbone.parameters():
            if param.requires_grad:
                backbone_params.append(param)

        # Grid cell params
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

    @staticmethod
    def mask_to_bbox(mask_logits, threshold=0.5):
        B, device = mask_logits.shape[0], mask_logits.device
        mask_binary = (mask_logits.sigmoid() > threshold).float()
        boxes = []
        for b in range(B):
            m = mask_binary[b]
            nz = m.nonzero(as_tuple=False)
            if nz.shape[0] == 0:
                boxes.append(torch.tensor([0.5, 0.5, 0.1, 0.1], device=device))
            else:
                H, W = m.shape
                y1, y2 = nz[:, 0].min().float() / H, (nz[:, 0].max().float() + 1) / H
                x1, x2 = nz[:, 1].min().float() / W, (nz[:, 1].max().float() + 1) / W
                boxes.append(torch.stack([(x1+x2)/2, (y1+y2)/2,
                                          (x2-x1).clamp(0.01), (y2-y1).clamp(0.01)]))
        return torch.stack(boxes)

    def forward(self, image, img_mask, text, text_len=None,
                target_bbox=None, **kwargs):
        B = image.shape[0]
        device = image.device
        ps = self._patch_grid

        # ===== 1. BEiT-3 encoding =====
        # Split: frozen layers under no_grad, unfrozen layers with grad
        beit3 = self.backbone.beit3
        encoder = beit3.encoder
        n_total = len(encoder.layers)
        n_unfreeze = self._unfreeze_layers
        split_idx = n_total - n_unfreeze  # e.g., 10 for 2 unfrozen

        with torch.no_grad():
            text_tokens, padding_mask = \
                self.backbone._get_text_token_and_padding_mask(text, device)
            image_len = beit3.vision_embed.num_position_embeddings()

            # Embed
            from torchscale.component.multiway_network import set_split_position
            x1 = beit3.vision_embed(image, None)
            multiway_split_position = x1.size(1)
            x2 = beit3.text_embed(text_tokens)
            x = torch.cat([x1, x2], dim=1)

            if padding_mask is not None:
                encoder_padding_mask = torch.cat([
                    torch.zeros(x1.shape[:-1], device=device).bool(),
                    padding_mask,
                ], dim=1)
            else:
                encoder_padding_mask = None

            encoder.apply(set_split_position(multiway_split_position))

            # Forward embedding
            x, _ = encoder.forward_embedding(None, x, None)
            if encoder_padding_mask is not None:
                x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

            # Run FROZEN layers (0 to split_idx-1)
            for idx in range(split_idx):
                x, _ = encoder.layers[idx](
                    x, encoder_padding_mask=encoder_padding_mask,
                    attn_mask=None, rel_pos=None,
                    multiway_split_position=multiway_split_position,
                )

        # Detach and run UNFROZEN layers (split_idx to end) WITH gradients
        if n_unfreeze > 0:
            x = x.detach().requires_grad_(True)
            for idx in range(split_idx, n_total):
                x, _ = encoder.layers[idx](
                    x, encoder_padding_mask=encoder_padding_mask,
                    attn_mask=None, rel_pos=None,
                    multiway_split_position=multiway_split_position,
                )
            # Layer norm
            if encoder.layer_norm is not None:
                x = encoder.layer_norm(x)

            vision_feat = x[:, :image_len]
            language_feat = x[:, image_len:]
        else:
            # All frozen
            if encoder.layer_norm is not None:
                x = encoder.layer_norm(x)
            vision_feat = x[:, :image_len]
            language_feat = x[:, image_len:]

        # ===== Frozen heads (bbox, projections) =====
        with torch.no_grad():
            vision_norm = F.normalize(self.backbone.vision_head(
                vision_feat.detach()), dim=-1)
            language_norm = F.normalize(self.backbone.language_head(
                language_feat.detach()), dim=-1)
            language_cls = language_norm[:, 0].contiguous()
            visu_sim = torch.mul(
                language_cls.unsqueeze(1).expand(-1, image_len - 1, -1),
                vision_norm[:, 1:].contiguous()
            ).sum(dim=-1)
            visu_src = self.backbone.visu_proj(vision_feat.detach())[:, 1:]
            attn_base = visu_sim.softmax(dim=-1)
            vg_hs = torch.mul(
                attn_base.unsqueeze(-1).expand_as(visu_src), visu_src
            ).sum(dim=1)
            pred_box = self.backbone.bbox_embed(vg_hs).sigmoid()

        # Vision patches for seg head (with grad from unfrozen layers)
        vision_patches = vision_feat[:, 1:]

        # ===== 2. Seg head (FROZEN RES pretrained) =====
        seg_mask = None
        if self._has_seg:
            channel = vision_patches.shape[2]
            seg_feat = vision_patches.permute(0, 2, 1).reshape(B, channel, ps, ps)
            seg_feat = self.backbone.bn3(
                self.backbone.seg_conv3(
                    self.backbone.relu(
                        self.backbone.bn2(
                            self.backbone.seg_conv2(
                                self.backbone.relu(
                                    self.backbone.bn1(
                                        self.backbone.seg_conv1(seg_feat))))))))
            seg_feat_norm = F.normalize(seg_feat.permute(0, 2, 3, 1), dim=-1)

            # ===== 3. Multi-Scale Grid Cells (TRAINABLE) =====
            if self.multi_grid is not None:
                query_offset = self.multi_grid(
                    seg_feat_norm.detach(),
                    language_feat.detach(),
                )
                enhanced_lang = language_cls.detach().reshape(B, 1, 1, -1) + query_offset
                seg_mask = torch.mul(
                    enhanced_lang.expand(-1, seg_feat_norm.shape[1],
                                         seg_feat_norm.shape[2], -1),
                    seg_feat_norm
                ).sum(dim=-1).unsqueeze(1)
            else:
                seg_mask = torch.mul(
                    language_cls.detach().reshape(B, 1, 1, -1).expand(
                        -1, seg_feat_norm.shape[1], seg_feat_norm.shape[2], -1),
                    seg_feat_norm
                ).sum(dim=-1).unsqueeze(1)

        if seg_mask is None:
            seg_mask = torch.zeros(B, 1, ps * 4, ps * 4, device=device)

        with torch.no_grad():
            pred_box_mask = self.mask_to_bbox(seg_mask.squeeze(1))

        return {
            'pred_box': pred_box.detach(),
            'pred_box_mask': pred_box_mask,
            'seg_mask': seg_mask,
            'seg_mask_base': None,
            'gate_seg': torch.tensor(self.get_gate_value()),
            'visu_sim_base': visu_sim.detach(),
            'attn_base': attn_base.detach(),
        }
