#!/usr/bin/env python3
"""
Evaluate Venice-H1 on pre-extracted feature caches.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt \
                       --splits data/cached_testA_refcoco_unc_feats.pt \
                                data/cached_testB_refcoco_unc_feats.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from venice_h1.model import VeniceH1Reranker
from train import FeatureCacheDataset, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--splits", nargs="+", required=True,
                        help="One or more feature cache .pt files to evaluate")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="Failure Gate threshold")
    parser.add_argument("--no_grid", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    # Infer model config from weights
    use_grid = not args.no_grid
    model = VeniceH1Reranker(use_grid=use_grid).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"Loaded: {args.checkpoint}")
    print(f"Parameters: {model.num_parameters():,}  |  tau={args.tau}")
    print()

    results = {}
    for split_path in args.splits:
        if not Path(split_path).exists():
            print(f"  [skip] {split_path} not found")
            continue

        ds     = FeatureCacheDataset(split_path, use_grid=use_grid)
        loader = DataLoader(ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
        split_name = Path(split_path).stem.replace("cached_", "").replace("_feats", "")

        metrics = evaluate(model, loader, device, tau=args.tau)
        results[split_name] = metrics

        print(f"  {split_name:<30s} "
              f"AUC={metrics['gate_auc']:.3f}  "
              f"harmful={metrics['harmful_switch']*100:.3f}%  "
              f"correct_rerank={metrics['correct_rerank']*100:.1f}%  "
              f"failure_rate={metrics['failure_rate']*100:.1f}%")

    print()
    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results → {out_path}")


if __name__ == "__main__":
    main()
