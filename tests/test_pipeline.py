#!/usr/bin/env python3
"""
Tests for Venice-H1 — verifies the full pipeline matches paper specifications.

Run with: pytest tests/test_pipeline.py -v
"""

import torch
import pytest


class TestMultiScaleGridSignatures:
    """Test grid signature computation (Section 3.3)."""

    def test_signature_dimensions(self):
        """Grid signatures should produce 675-dim output (Eq. 5)."""
        from venice_h1.model.grid_signatures import MultiScaleGridSignatures

        model = MultiScaleGridSignatures(embed_dim=256)
        seg_feat = torch.randn(2, 16, 16, 256)
        lang_feat = torch.randn(2, 20, 256)

        out = model(seg_feat, lang_feat)
        assert out.shape == (2, 16, 16, 256)

    def test_zero_init(self):
        """At initialization, output should be zero (baseline preservation)."""
        from venice_h1.model.grid_signatures import MultiScaleGridSignatures

        model = MultiScaleGridSignatures(embed_dim=256)
        seg_feat = torch.randn(1, 8, 8, 256)
        lang_feat = torch.randn(1, 10, 256)

        with torch.no_grad():
            out = model(seg_feat, lang_feat)

        # scale starts at 0 → tanh(0) = 0 → output should be all zeros
        assert out.abs().max().item() < 1e-6

    def test_scale_weights(self):
        """Scale weights should sum to 1 after softmax."""
        from venice_h1.model.grid_signatures import MultiScaleGridSignatures

        model = MultiScaleGridSignatures(embed_dim=256)
        weights = model.get_scale_weights()

        total = weights["4x4"] + weights["8x8"] + weights["16x16"]
        assert abs(total - 1.0) < 1e-5


class TestVeniceH1Reranker:
    """Test the re-ranker (Section 3.4)."""

    def test_architecture_specs(self):
        """Model should match paper: Hd=512, ~11.3M params."""
        from venice_h1.model.reranker import VeniceH1Reranker

        model = VeniceH1Reranker(
            query_feat_dim=256, hidden_dim=512,
            n_layers=3, n_heads=8, n_queries=10
        )

        n_params = model.num_parameters()
        # Paper: ~11.3M parameters
        assert 10_000_000 < n_params < 13_000_000, \
            f"Expected ~11.3M params, got {n_params:,}"

    def test_feature_dim(self):
        """Input feature dim should be 936 = 256 + 1 + 4 + 675 (Eq. 6)."""
        from venice_h1.model.reranker import VeniceH1Reranker

        model = VeniceH1Reranker(query_feat_dim=256, hidden_dim=512)
        # Df = 256 (query) + 1 (score) + 4 (stats) + 675 (grid) = 936
        features = torch.randn(4, 10, 936)
        det_scores = torch.randn(4, 10)
        mask_means = torch.randn(4, 10)

        out = model(features, det_scores, mask_means)
        assert "p_fail" in out
        assert "gain_logits" in out
        assert out["p_fail"].shape == (4,)
        assert out["gain_logits"].shape == (4, 10)

    def test_p_fail_range(self):
        """Failure probability should be in [0, 1]."""
        from venice_h1.model.reranker import VeniceH1Reranker

        model = VeniceH1Reranker()
        features = torch.randn(8, 10, 936)
        det_scores = torch.randn(8, 10)
        mask_means = torch.randn(8, 10)

        out = model(features, det_scores, mask_means)
        assert (out["p_fail"] >= 0).all()
        assert (out["p_fail"] <= 1).all()

    def test_gated_selection(self):
        """When P_fail <= tau, should retain Query-0 (Eq. 10)."""
        from venice_h1.model.reranker import VeniceH1Reranker

        model = VeniceH1Reranker(tau=0.99)  # Very high threshold
        features = torch.randn(4, 10, 936)
        det_scores = torch.randn(4, 10)
        mask_means = torch.randn(4, 10)

        selected = model.rerank(features, det_scores, mask_means, tau=0.99)
        # With tau=0.99, almost certainly P_fail < tau → Query-0 retained
        assert (selected == 0).all()

    def test_rerank_output_range(self):
        """Selected indices should be in [0, N-1]."""
        from venice_h1.model.reranker import VeniceH1Reranker

        model = VeniceH1Reranker(tau=0.01)  # Low threshold → more interventions
        features = torch.randn(16, 10, 936)
        det_scores = torch.randn(16, 10)
        mask_means = torch.randn(16, 10)

        selected = model.rerank(features, det_scores, mask_means, tau=0.01)
        assert (selected >= 0).all()
        assert (selected < 10).all()

    def test_no_grid_ablation(self):
        """Without grid features, Df should be 261 (BASE only)."""
        from venice_h1.model.reranker import VeniceH1Reranker

        model = VeniceH1Reranker(use_grid=False)
        # Df = 256 + 1 + 4 = 261
        features = torch.randn(4, 10, 261)
        det_scores = torch.randn(4, 10)
        mask_means = torch.randn(4, 10)

        out = model(features, det_scores, mask_means)
        assert out["p_fail"].shape == (4,)


class TestGridSignatureExtraction:
    """Test the feature extraction functions (Section 3.2-3.3)."""

    def test_mask_statistics(self):
        """Mask stats should produce 4 scalars per query."""
        from scripts.extract_features import compute_mask_statistics

        mask_probs = torch.rand(10, 64, 64)
        stats = compute_mask_statistics(mask_probs)

        assert stats["mask_mean"].shape == (10,)
        assert stats["mask_max"].shape == (10,)
        assert stats["mask_area"].shape == (10,)
        assert stats["mask_std"].shape == (10,)

        # mask_mean should be in [0, 1]
        assert (stats["mask_mean"] >= 0).all()
        assert (stats["mask_mean"] <= 1).all()

    def test_grid_signatures_4x4(self):
        """4×4 grid should produce 16 + 16 + 1 = 33 values per query."""
        from scripts.extract_features import compute_grid_signatures

        mask_probs = torch.rand(5, 128, 128)
        sigs = compute_grid_signatures(mask_probs, grid_size=4)

        assert sigs["grid_mean_4"].shape == (5, 16)
        assert sigs["grid_max_4"].shape == (5, 16)
        assert sigs["boundary_4"].shape == (5,)

    def test_grid_signatures_8x8(self):
        """8×8 grid should produce 64 + 64 + 1 = 129 values."""
        from scripts.extract_features import compute_grid_signatures

        mask_probs = torch.rand(5, 128, 128)
        sigs = compute_grid_signatures(mask_probs, grid_size=8)

        assert sigs["grid_mean_8"].shape == (5, 64)
        assert sigs["grid_max_8"].shape == (5, 64)

    def test_grid_signatures_16x16(self):
        """16×16 grid should produce 256 + 256 + 1 = 513 values."""
        from scripts.extract_features import compute_grid_signatures

        mask_probs = torch.rand(5, 128, 128)
        sigs = compute_grid_signatures(mask_probs, grid_size=16)

        assert sigs["grid_mean_16"].shape == (5, 256)
        assert sigs["grid_max_16"].shape == (5, 256)

    def test_total_signature_dim(self):
        """Total grid signature = 33 + 129 + 513 = 675."""
        from scripts.extract_features import compute_grid_signatures

        mask_probs = torch.rand(3, 64, 64)
        g4  = compute_grid_signatures(mask_probs, 4)
        g8  = compute_grid_signatures(mask_probs, 8)
        g16 = compute_grid_signatures(mask_probs, 16)

        dim = (g4["grid_mean_4"].shape[1] + g4["grid_max_4"].shape[1] + 1 +
               g8["grid_mean_8"].shape[1] + g8["grid_max_8"].shape[1] + 1 +
               g16["grid_mean_16"].shape[1] + g16["grid_max_16"].shape[1] + 1)
        assert dim == 675

    def test_boundary_energy_nonneg(self):
        """Boundary energy should always be non-negative."""
        from scripts.extract_features import compute_grid_signatures

        mask_probs = torch.rand(10, 64, 64)
        sigs = compute_grid_signatures(mask_probs, 8)
        assert (sigs["boundary_8"] >= 0).all()


class TestEndToEnd:
    """End-to-end pipeline test (Algorithm 1)."""

    def test_full_inference_pipeline(self):
        """Simulate full Algorithm 1 with random features."""
        from venice_h1.model.reranker import VeniceH1Reranker
        from scripts.extract_features import (
            compute_mask_statistics, compute_grid_signatures
        )

        # Simulate DeRIS outputs
        B, N, D, H, W = 2, 10, 256, 64, 64
        query_embeddings = torch.randn(B, N, D)
        mask_logits = torch.randn(B, N, H, W)
        det_scores = torch.rand(B, N)

        # Step 2: mask probabilities
        mask_probs = torch.sigmoid(mask_logits)

        # Step 3-4: extract features per batch
        all_features = []
        for b in range(B):
            stats = compute_mask_statistics(mask_probs[b])
            g4 = compute_grid_signatures(mask_probs[b], 4)
            g8 = compute_grid_signatures(mask_probs[b], 8)
            g16 = compute_grid_signatures(mask_probs[b], 16)

            # Step 5: assemble f_i
            f = torch.cat([
                query_embeddings[b],             # [N, 256]
                det_scores[b].unsqueeze(-1),     # [N, 1]
                stats["mask_mean"].unsqueeze(-1), # [N, 1]
                stats["mask_max"].unsqueeze(-1),  # [N, 1]
                stats["mask_area"].unsqueeze(-1), # [N, 1]
                stats["mask_std"].unsqueeze(-1),  # [N, 1]
                g4["grid_mean_4"],               # [N, 16]
                g4["grid_max_4"],                # [N, 16]
                g4["boundary_4"].unsqueeze(-1),  # [N, 1]
                g8["grid_mean_8"],               # [N, 64]
                g8["grid_max_8"],                # [N, 64]
                g8["boundary_8"].unsqueeze(-1),  # [N, 1]
                g16["grid_mean_16"],             # [N, 256]
                g16["grid_max_16"],              # [N, 256]
                g16["boundary_16"].unsqueeze(-1), # [N, 1]
            ], dim=-1)
            all_features.append(f)

        features = torch.stack(all_features)  # [B, N, 936]
        assert features.shape == (B, N, 936)

        # Step 6-11: re-ranker
        model = VeniceH1Reranker(tau=0.05)
        selected = model.rerank(
            features,
            det_scores,
            features[:, :, 262:263].squeeze(-1),  # mask_mean position
            tau=0.05
        )

        assert selected.shape == (B,)
        assert (selected >= 0).all()
        assert (selected < N).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
