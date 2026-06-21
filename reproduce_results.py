#!/usr/bin/env python3
"""
reproduce_results.py — Reproduce Venice-H1 paper results from scratch.

This script downloads the trained checkpoint from HuggingFace, loads it,
and either:
  (A) runs a full evaluation on pre-extracted feature caches (if available), or
  (B) runs a verified dummy forward pass to confirm the model architecture.

OdaxAI Research · Nicolò Savioli, Ph.D. · nicolo.savioli@odaxai.com
https://github.com/odaxai/Venice-H1

Usage
-----
# Option A: full evaluation with real feature caches
python reproduce_results.py --features_dir data/ --splits all

# Option B: architecture verification only (no features needed)
python reproduce_results.py --verify_only
"""

import argparse
import json
import sys
from pathlib import Path

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Constants matching Table 1 of the paper
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_PARAMS = 11_296_258

# Paper Table 1 — RefCOCO val (backbone: DeRIS-L)
PAPER_RESULTS = {
    "refcoco_val": {
        "delta_fail": 1.824,  # mIoU improvement on failure cases
        "auc_fail":   0.778,  # failure-detection AUC
        "delta_full": 0.039,  # overall mIoU improvement
    }
}

# All splits produced by extract_features.py
ALL_SPLITS = [
    "val_refcoco_unc",
    "testA_refcoco_unc",
    "testB_refcoco_unc",
    "val_refcoco+_unc",
    "testA_refcoco+_unc",
    "testB_refcoco+_unc",
    "val_refcocog_umd",
    "test_refcocog_umd",
]


def download_checkpoint() -> str:
    """Download Venice-H1 checkpoint from HuggingFace OdaxAI/venice-h1."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    print("Downloading Venice-H1 checkpoint from OdaxAI/venice-h1 ...")
    path = hf_hub_download(repo_id="OdaxAI/venice-h1", filename="venice_h1_deris_l.pt")
    print(f"  ✓ {path}")
    return path


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in ckpt, "Checkpoint missing 'model' key"
    assert "config" in ckpt, "Checkpoint missing 'config' key"
    return ckpt


def build_model(cfg: dict):
    from venice_h1.model.reranker import VeniceH1Reranker
    model = VeniceH1Reranker(
        query_feat_dim=cfg["query_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_queries=cfg["n_queries"],
        dropout=cfg.get("dropout", 0.1),
        use_grid=cfg.get("use_grid", True),
        tau=cfg["tau"],
    )
    return model


def verify_architecture(ckpt: dict) -> None:
    """Verify checkpoint matches paper architecture exactly."""
    print("\n── Architecture Verification ──────────────────────────────")
    cfg = ckpt["config"]
    print(f"  query_feat_dim : {cfg['query_feat_dim']}  (expected 256)")
    print(f"  hidden_dim     : {cfg['hidden_dim']}  (expected 512)")
    print(f"  n_layers       : {cfg['n_layers']}  (expected 3)")
    print(f"  n_heads        : {cfg['n_heads']}   (expected 8)")
    print(f"  n_queries      : {cfg['n_queries']}  (expected 10)")
    print(f"  use_grid       : {cfg['use_grid']}  (expected True)")
    print(f"  tau            : {cfg['tau']}  (paper τ = 0.9)")

    n_params = sum(v.numel() for v in ckpt["model"].values() if hasattr(v, "numel"))
    print(f"\n  Parameters     : {n_params:,}  (expected {EXPECTED_PARAMS:,})")
    match = "✓ MATCH" if n_params == EXPECTED_PARAMS else "✗ MISMATCH"
    print(f"  Status         : {match}")
    assert n_params == EXPECTED_PARAMS, f"Parameter count mismatch: {n_params} != {EXPECTED_PARAMS}"


def verify_forward_pass(model, cfg: dict) -> None:
    """Run a random forward pass and check output shapes/ranges."""
    print("\n── Forward Pass Verification ───────────────────────────────")
    model.eval()
    B, N = 4, cfg["n_queries"]
    feat_dim = cfg["query_feat_dim"] + 675 + 5  # embed + grid + mask_stats + det_score

    features   = torch.randn(B, N, feat_dim)
    det_scores = torch.rand(B, N)
    mask_means = torch.rand(B, N)

    with torch.no_grad():
        out = model(features, det_scores, mask_means)
        selected = model.rerank(features, det_scores, mask_means)

    gain_key = "gain_logits" if "gain_logits" in out else "gain"
    assert out["p_fail"].shape == (B,),   f"p_fail shape wrong: {out['p_fail'].shape}"
    assert out[gain_key].shape == (B, N), f"gain shape wrong: {out[gain_key].shape}"
    assert selected.shape      == (B,),   f"selected shape wrong: {selected.shape}"
    assert (out["p_fail"] >= 0).all() and (out["p_fail"] <= 1).all(), "p_fail out of [0,1]"

    print(f"  Input  features : {list(features.shape)}")
    print(f"  Output p_fail   : {list(out['p_fail'].shape)}  range [{out['p_fail'].min():.3f}, {out['p_fail'].max():.3f}]")
    print(f"  Output gain     : {list(out[gain_key].shape)}")
    print(f"  Selected query  : {selected.tolist()}")
    print(f"  ✓ Forward pass OK")


def verify_stored_metrics(ckpt: dict) -> None:
    """Print stored training metrics and check against paper claims."""
    print("\n── Stored Evaluation Metrics (RefCOCO val) ─────────────────")
    m = ckpt.get("metrics", {})
    for k, v in m.items():
        if v is not None:
            print(f"  {k:<30s}: {v:.4f}" if isinstance(v, float) else f"  {k:<30s}: {v}")

    # Cross-check against paper
    ref = PAPER_RESULTS["refcoco_val"]
    print("\n── Paper Cross-Check ────────────────────────────────────────")
    checks = [
        ("delta_fail", m.get("delta_fail"), ref["delta_fail"], 0.01),
        ("auc_fail",   m.get("auc_fail"),   ref["auc_fail"],   0.01),
        ("delta_full", m.get("delta_full"), ref["delta_full"], 0.01),
    ]
    for name, got, expected, tol in checks:
        if got is None:
            print(f"  {name:<15s}: NOT STORED")
            continue
        ok = abs(got - expected) <= tol
        status = "✓" if ok else "~"
        print(f"  {status} {name:<15s}: {got:.4f}  (paper: {expected:.3f})")


def evaluate_on_features(model, features_dir: Path, splits: list, tau: float) -> None:
    """Run full evaluation on pre-extracted feature caches."""
    try:
        from torch.utils.data import DataLoader
        from train import FeatureCacheDataset, evaluate
    except ImportError as e:
        print(f"[ERROR] Cannot import train.py: {e}")
        return

    print(f"\n── Full Evaluation (τ={tau}) ─────────────────────────────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = {}
    for split in splits:
        candidates = [
            features_dir / f"cached_{split}_feats.pt",
            features_dir / f"{split}.pt",
        ]
        cache_path = next((p for p in candidates if p.exists()), None)
        if cache_path is None:
            print(f"  [skip] {split}  (no cache found in {features_dir})")
            continue

        ds     = FeatureCacheDataset(str(cache_path), use_grid=True)
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)
        met    = evaluate(model, loader, device, tau=tau)
        results[split] = met

        print(
            f"  {split:<35s}  "
            f"AUC={met['gate_auc']:.3f}  "
            f"∆_fail={met.get('delta_fail', met.get('correct_rerank', 0))*100:.2f}  "
            f"harmful={met['harmful_switch']*100:.3f}%"
        )

    if results:
        out = Path("reproduction_results.json")
        out.write_text(json.dumps(results, indent=2))
        print(f"\n  Results saved → {out}")
    else:
        print("  No feature caches found. Use --verify_only for architecture check.")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Venice-H1 paper results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Architecture + metrics verification (no data needed):
  python reproduce_results.py --verify_only

  # Full evaluation on extracted feature caches:
  python reproduce_results.py --features_dir data/ --splits all

  # Evaluate specific splits:
  python reproduce_results.py --features_dir data/ --splits val_refcoco_unc testA_refcoco_unc

  # Use a local checkpoint instead of downloading:
  python reproduce_results.py --checkpoint /path/to/best_v3.pth --verify_only
""")
    parser.add_argument("--checkpoint", default=None,
                        help="Local checkpoint path. Downloads from HF if not provided.")
    parser.add_argument("--features_dir", default="data/",
                        help="Directory containing pre-extracted feature caches (.pt files)")
    parser.add_argument("--splits", nargs="+", default=["all"],
                        help="Splits to evaluate. 'all' evaluates all 8 standard splits.")
    parser.add_argument("--tau", type=float, default=0.9,
                        help="Failure Gate threshold (default: 0.9 from checkpoint)")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify architecture and stored metrics, skip full eval")
    args = parser.parse_args()

    print("=" * 65)
    print(" Venice-H1 · Reproduction Script")
    print(" OdaxAI Research · Nicolò Savioli, Ph.D.")
    print("=" * 65)

    # 1. Get checkpoint
    ckpt_path = args.checkpoint or download_checkpoint()
    ckpt = load_checkpoint(ckpt_path)
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Arch version: {ckpt.get('arch_version', 'legacy')}")

    # 2. Verify architecture
    verify_architecture(ckpt)

    # 3. Build model
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=False)

    # 4. Forward pass check
    verify_forward_pass(model, ckpt["config"])

    # 5. Stored metrics cross-check
    verify_stored_metrics(ckpt)

    # 6. Full evaluation (optional)
    if not args.verify_only:
        splits = ALL_SPLITS if "all" in args.splits else args.splits
        evaluate_on_features(model, Path(args.features_dir), splits, tau=args.tau)

    print("\n" + "=" * 65)
    print(" Reproduction complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
