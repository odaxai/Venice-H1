"""
Venice-H1 Re-Ranker: Failure Gate + Gain Predictor.

Architecture exactly matching the paper (Section 3.4):
  - Per-query feature: f_i = [q_i; s_i; μ_i; p̂_i; a_i; σ_i; g_i] ∈ R^936
  - Query encoder: 2-layer MLP → R^512 (hidden dim Hd=512)
  - Transformer: L=3 layers, A=8 heads, pre-norm, GELU FFN
  - Head 1 (Gain Predictor): 2-layer MLP → ΔIoU per query
  - Head 2 (Failure Gate): MLP on r=[h̄; ĥ; h_def; s; μ] → P_fail

Training (Section 3.5):
  - L = L_gate + λ·L_gain (λ=5)
  - L_gate: focal BCE (γ=2, auto w_pos)
  - L_gain: smooth-L1 on IoU regression (all samples)
  - AdamW, cosine + 3-epoch warmup, 20 epochs, batch 512, FP16

Inference (Algorithm 1):
  - if P_fail > τ: select argmax_i ĝ_i
  - else: retain Query-0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_GRID_DIM = 675   # 4×4 (33) + 8×8 (129) + 16×16 (513)
_MASK_STATS = 4   # μ, p̂, a, σ
_SCORE_DIM = 1    # detection score


class QueryEncoder(nn.Module):
    """Two-layer MLP projecting raw features into Hd=512."""

    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FailureGate(nn.Module):
    """
    Predicts P(failure) from global feature vector:
      r = [h_mean; h_max; h_def; s; μ] ∈ R^(3*Hd + 2*N)
    """

    def __init__(self, hidden_dim: int = 512, n_queries: int = 10):
        super().__init__()
        in_dim = 3 * hidden_dim + 2 * n_queries
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_mean: torch.Tensor, h_max: torch.Tensor,
                h_def: torch.Tensor, scores: torch.Tensor,
                mask_means: torch.Tensor) -> torch.Tensor:
        r = torch.cat([h_mean, h_max, h_def, scores, mask_means], dim=-1)
        return torch.sigmoid(self.net(r).squeeze(-1))


class GainPredictor(nn.Module):
    """Two-layer MLP predicting IoU gain per query (Eq. 8)."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class VeniceH1Reranker(nn.Module):
    """
    Venice-H1 Failure-Aware Re-Ranker.

    Matches the paper specification exactly:
      - Df = D + 680 = 936 (for D=256, DeRIS query dim)
      - Hd = 512 (hidden dimension)
      - L = 3 Transformer layers, A = 8 heads, pre-norm GELU
      - ~11.3M trainable parameters

    Args:
        query_feat_dim: D (query embedding dim from backbone), default 256
        hidden_dim:     Hd (Transformer hidden size), default 512
        n_layers:       L (Transformer encoder layers), default 3
        n_heads:        A (attention heads), default 8
        n_queries:      N (number of candidate queries), default 10
        dropout:        dropout rate, default 0.1
        use_grid:       use multi-scale grid signatures (675-dim)
        tau:            default Failure Gate threshold
    """

    def __init__(
        self,
        query_feat_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        n_queries: int = 10,
        dropout: float = 0.1,
        use_grid: bool = True,
        tau: float = 0.5,
    ):
        super().__init__()
        self.tau = tau
        self.use_grid = use_grid
        self.n_queries = n_queries
        self.hidden_dim = hidden_dim

        grid_d = _GRID_DIM if use_grid else 0
        feat_dim = query_feat_dim + _SCORE_DIM + _MASK_STATS + grid_d  # 936

        self.query_encoder = QueryEncoder(feat_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.failure_gate = FailureGate(hidden_dim, n_queries)
        self.gain_predictor = GainPredictor(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor,
                det_scores: torch.Tensor | None = None,
                mask_means: torch.Tensor | None = None) -> dict:
        """
        Args:
            features:    [B, N, Df] raw per-query features
            det_scores:  [B, N] detection scores (for gate input)
            mask_means:  [B, N] mean mask coverage (for gate input)

        Returns dict:
            p_fail:      [B]    failure probability
            gain_logits: [B, N] predicted IoU gain per query
        """
        B, N, _ = features.shape

        # Query encoder
        h = self.query_encoder(features)       # [B, N, Hd]

        # Transformer inter-query reasoning
        h = self.transformer(h)                # [B, N, Hd]

        # Gain Predictor (per-query)
        gain_logits = self.gain_predictor(h)   # [B, N]

        # Failure Gate: aggregate features
        h_mean = h.mean(dim=1)                 # [B, Hd]
        h_max  = h.max(dim=1).values           # [B, Hd]
        h_def  = h[:, 0, :]                    # [B, Hd] (Query-0)

        if det_scores is None:
            det_scores = features[:, :, -_GRID_DIM - _MASK_STATS]  # fallback
        if mask_means is None:
            mask_means = features[:, :, -_GRID_DIM - _MASK_STATS + 1]

        p_fail = self.failure_gate(
            h_mean, h_max, h_def, det_scores, mask_means)

        return {"p_fail": p_fail, "gain_logits": gain_logits}

    @torch.no_grad()
    def rerank(self, features: torch.Tensor,
               det_scores: torch.Tensor | None = None,
               mask_means: torch.Tensor | None = None,
               tau: float | None = None) -> torch.Tensor:
        """
        Inference re-ranking (Algorithm 1, steps 6-11).

        Returns:
            selected_idx [B] — index of the selected query.
        """
        tau = tau if tau is not None else self.tau
        out = self.forward(features, det_scores, mask_means)

        best_alt = out["gain_logits"].argmax(dim=1)
        default  = torch.zeros_like(best_alt)
        selected = torch.where(out["p_fail"] > tau, best_alt, default)
        return selected

    def param_groups(self, lr: float = 5e-4) -> list:
        """Single learning rate as in paper (AdamW, lr=5e-4)."""
        return [{"params": self.parameters(), "lr": lr}]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
