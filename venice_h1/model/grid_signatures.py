"""
Multi-Scale Grid Signatures — the core spatial feature extractor of Venice-H1.

Pools mask probabilities onto 4×4, 8×8, and 16×16 grids to produce
compact 675-dimensional descriptors per candidate query.

Reference: Section 3.3 of the Venice-H1 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLanguageQuery(nn.Module):
    """Single-scale spatial language query at a fixed grid resolution."""

    def __init__(self, embed_dim: int, grid_size: int = 8,
                 num_heads: int = 8, dropout: float = 0.05):
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
        # Zero-init → grid offset starts at 0, preserving exact baseline
        nn.init.zeros_(self.query_proj[-1].weight)
        nn.init.zeros_(self.query_proj[-1].bias)

    def forward(self, seg_features_2d: torch.Tensor,
                language_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_features_2d: [B, H, W, D] spatial features
            language_feat:   [B, L, D]   language token features
        Returns:
            query_offset:    [B, H, W, D] upsampled spatial query offset
        """
        B, H, W, D = seg_features_2d.shape
        gs = self.grid_size

        seg_2d = seg_features_2d.permute(0, 3, 1, 2)          # [B, D, H, W]
        grid_feat = F.adaptive_avg_pool2d(seg_2d, (gs, gs))   # [B, D, gs, gs]
        grid_feat = grid_feat.permute(0, 2, 3, 1).reshape(B, gs * gs, D)
        grid_feat = self.downsample_proj(grid_feat)

        attended, _ = self.cross_attn(grid_feat, language_feat, language_feat)
        grid_feat = self.norm1(grid_feat + attended)

        refined, _ = self.self_attn(grid_feat, grid_feat, grid_feat)
        grid_feat = self.norm2(grid_feat + refined)

        query_offset = self.query_proj(grid_feat)
        query_2d = query_offset.reshape(B, gs, gs, D).permute(0, 3, 1, 2)
        query_up = F.interpolate(query_2d, size=(H, W),
                                 mode='bilinear', align_corners=False)
        return query_up.permute(0, 2, 3, 1)                   # [B, H, W, D]


class MultiScaleGridSignatures(nn.Module):
    """
    Multi-Scale Grid Signatures operating at 4×4, 8×8, 16×16 simultaneously.

    Each scale encodes complementary spatial information:
      - 4×4  (16 cells):  coarse global layout
      - 8×8  (64 cells):  medium-range positional structure
      - 16×16 (256 cells): fine-grained local shape and boundary detail

    The design is inspired by multi-scale grid-cell representations in the
    mammalian entorhinal cortex (Moser & Moser, 2014; Hafting et al., 2005).

    Total descriptor dimensionality: 675 (grid means + grid max + boundary energy).
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.05):
        super().__init__()
        self.scale_4 = SpatialLanguageQuery(
            embed_dim, grid_size=4, num_heads=num_heads, dropout=dropout)
        self.scale_8 = SpatialLanguageQuery(
            embed_dim, grid_size=8, num_heads=num_heads, dropout=dropout)
        self.scale_16 = SpatialLanguageQuery(
            embed_dim, grid_size=16, num_heads=num_heads, dropout=dropout)

        # Learnable per-scale combination weights
        self.scale_weights = nn.Parameter(torch.zeros(3))
        # Global gating scalar — starts at 0 (pure baseline at init)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, seg_features_2d: torch.Tensor,
                language_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_features_2d: [B, H, W, D] normalised spatial features
            language_feat:   [B, L, D]   language token features
        Returns:
            fused_query:     [B, H, W, D] gated multi-scale spatial query
        """
        q4  = self.scale_4(seg_features_2d, language_feat)
        q8  = self.scale_8(seg_features_2d, language_feat)
        q16 = self.scale_16(seg_features_2d, language_feat)

        w = F.softmax(self.scale_weights, dim=0)
        fused = w[0] * q4 + w[1] * q8 + w[2] * q16
        return torch.tanh(self.scale) * fused

    def get_scale_weights(self) -> dict:
        """Return softmax-normalised per-scale weights (for logging/inspection)."""
        w = F.softmax(self.scale_weights, dim=0)
        return {"4x4": w[0].item(), "8x8": w[1].item(), "16x16": w[2].item()}

    def get_gate_value(self) -> float:
        """Return the current global gate value tanh(scale)."""
        return torch.tanh(self.scale).item()
