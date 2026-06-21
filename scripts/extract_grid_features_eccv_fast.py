#!/usr/bin/env python3
"""
ECCV 2026: Fast Multi-Process Multi-GPU Feature Extraction (2x3090 optimized)
with Grid Multi-Scale Signatures.

Key speedups vs baseline:
- Vectorized IoU/oracle on GPU (no Python loops over BxN)
- No CPU transfer of mask_logits (we only keep what we need)
- AMP autocast for forward
- Better sharding: default shards_per_split = num_gpus * 4 (keeps both GPUs busy)
- DataLoader: persistent_workers + prefetch + pin_memory, non_blocking copies
- Each GPU in its own process via CUDA_VISIBLE_DEVICES

Outputs per split shard:
  query_feat [M,N,D] (fp16)
  det_scores [M,N] (fp16)
  mask_mean/max/area/std [M,N] (fp16)
  grid_mean_{4,8,16} [M,N,G*G] (fp16)
  grid_max_{4,8,16}  [M,N,G*G] (fp16)
  boundary_{4,8,16}  [M,N] (fp16)
  query_ious [M,N] (fp16)
  oracle_idx [M] (int64)
  q0_iou [M] (fp16)

Usage:
  # Use both 3090
  python scripts/extract_grid_features_eccv_fast.py --world_auto --gpus 0,1 --batch_size 128

  # Resume
  python scripts/extract_grid_features_eccv_fast.py --world_auto --gpus 0,1 --resume

  # Merge
  python scripts/extract_grid_features_eccv_fast.py --merge --cache_dir ./outputs/eccv_full_cache
"""

import os
import sys
import json
import time
import glob
import hashlib
import argparse
import multiprocessing as mp
from datetime import datetime
from functools import partial
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as maskUtils


# -------- Paths --------
VENICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIS_ROOT = "/home/bionick87/miccai_2026/code/DeRIS/DeRIS-main"
sys.path.insert(0, VENICE_ROOT)
sys.path.insert(0, DERIS_ROOT)

# -------- Splits --------
ALL_SPLITS = [
    "train",
    "val_refcoco_unc", "testA_refcoco_unc", "testB_refcoco_unc",
    "val_refcocoplus_unc", "testA_refcocoplus_unc", "testB_refcocoplus_unc",
    "val_refcocog_umd", "test_refcocog_umd",
]


# ============================================================
# Helpers
# ============================================================

def extract_data(inputs):
    """MMCV DataContainer → plain tensors."""
    from mmcv.parallel.data_container import DataContainer
    result = {}
    for k, v in inputs.items():
        if isinstance(v, DataContainer):
            data = v.data
            if isinstance(data, list) and len(data) == 1:
                data = data[0]
            result[k] = data
        else:
            result[k] = v
    return result


def unwrap_model(m):
    return m.module if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else m


def file_hash(path):
    """Quick hash of file for versioning."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def infer_split_name_from_filename(path: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r"\.pt$", "", base)
    base = re.sub(r"^cached_", "", base)
    base = re.sub(r"_feats$", "", base)
    return base


# ============================================================
# Indexed datasets for correct resume (global index tracking)
# ============================================================

class IndexedDataset(Dataset):
    """Wraps a dataset so __getitem__ returns (global_index, sample)."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return (idx, self.base[idx])


class ShardView(Dataset):
    """View of base IndexedDataset over a list of global indices."""
    def __init__(self, indexed_dataset, global_indices):
        self.base = indexed_dataset
        self.indices = list(global_indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        global_idx = self.indices[i]
        return self.base[global_idx]  # returns (global_idx, sample)


def indexed_collate(batch_of_pairs, batch_size):
    """Collate list of (index, sample) into (idxs_tensor, mmcv_batch)."""
    from mmcv.parallel import collate
    idxs = [p[0] for p in batch_of_pairs]
    samples = [p[1] for p in batch_of_pairs]
    batch = collate(samples, samples_per_gpu=batch_size)
    return (torch.tensor(idxs, dtype=torch.long), batch)


# ============================================================
# Grid Multi-Scale Feature Extraction
# ============================================================

@torch.no_grad()
def compute_grid_signatures(mask_probs, grid_sizes=(4, 8, 16)):
    """
    mask_probs: [B, N, H, W] float
    returns dict:
      grid_mean_G [B,N,G*G]
      grid_max_G  [B,N,G*G]
      boundary_G  [B,N]
    """
    B, N, H, W = mask_probs.shape
    x = mask_probs.view(B * N, 1, H, W)

    out = {}
    for G in grid_sizes:
        gm = F.adaptive_avg_pool2d(x, (G, G)).view(B, N, G * G)
        gx = F.adaptive_max_pool2d(x, (G, G)).view(B, N, G * G)

        # boundary energy on pooled avg map
        g2 = F.adaptive_avg_pool2d(x, (G, G))  # [B*N,1,G,G]
        grad_x = (g2[:, :, :, 1:] - g2[:, :, :, :-1]).abs()
        grad_y = (g2[:, :, 1:, :] - g2[:, :, :-1, :]).abs()
        boundary = 0.5 * (grad_x.mean(dim=(-2, -1)) + grad_y.mean(dim=(-2, -1)))  # [B*N,1]
        boundary = boundary.view(B, N)

        out[f"grid_mean_{G}"] = gm
        out[f"grid_max_{G}"] = gx
        out[f"boundary_{G}"] = boundary
    return out


@torch.no_grad()
def decode_gt_rles_to_tensor(gt_rles, device, out_hw):
    """
    gt_rles: list length B, each is RLE (pycocotools) or dict
    Returns gt mask tensor [B,1,outH,outW] float (0/1)
    """
    outH, outW = out_hw
    # maskUtils.decode supports list-of-RLE → (H,W,B) or (H,W) depending
    decoded = maskUtils.decode(gt_rles)
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]  # H,W,1
    # decoded: H,W,B
    decoded = np.transpose(decoded, (2, 0, 1))  # B,H,W
    gt = torch.from_numpy(decoded).to(device=device, dtype=torch.float32)  # [B,H,W]
    gt = gt.unsqueeze(1)  # [B,1,H,W]
    gt = F.interpolate(gt, size=(outH, outW), mode="nearest")
    return gt


@torch.no_grad()
def compute_oracle_and_ious_gpu(mask_probs, gt_rles, device):
    """
    Vectorized IoU on GPU.
    mask_probs: [B,N,H,W] float
    gt_rles: list length B
    Returns:
      oracle_idx: [B] long
      query_ious: [B,N] float
    """
    B, N, H, W = mask_probs.shape
    gt = decode_gt_rles_to_tensor(gt_rles, device=device, out_hw=(H, W))  # [B,1,H,W]

    pred = (mask_probs > 0.5).to(torch.float32)  # [B,N,H,W]
    gt_bin = (gt > 0.5).to(torch.float32)        # [B,1,H,W]

    # Broadcast gt across N queries
    gtN = gt_bin.expand(B, N, H, W)

    inter = (pred * gtN).sum(dim=(-1, -2))  # [B,N]
    union = pred.sum(dim=(-1, -2)) + gtN.sum(dim=(-1, -2)) - inter
    iou = inter / (union + 1e-6)

    oracle_idx = iou.argmax(dim=1)
    return oracle_idx, iou


# ============================================================
# DeRIS Model Loading
# ============================================================

def load_deris_model(checkpoint_path, config_path, device):
    config_path = os.path.abspath(config_path)
    checkpoint_path = os.path.abspath(checkpoint_path)

    os.chdir(DERIS_ROOT)
    from mmcv import Config
    from deris.models import build_model as deris_build_model

    cfg = Config.fromfile(config_path)
    cfg.model.mask_save_target_dir = None
    cfg.model.visualize_params = {"enable": False}

    model = deris_build_model(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Strip DDP "module." prefix if present (otherwise strict=False silently drops EVERY key)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if len(unexpected) > 0:
        print(f"  [warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    n_loaded = len(state) - len(unexpected)
    print(f"  [load] {n_loaded}/{len(state)} ckpt tensors mapped into model")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


def build_full_dataset(cfg, split_name):
    from deris.datasets import build_dataset as deris_build_dataset
    return deris_build_dataset(cfg.data[split_name])


# ============================================================
# Feature Extraction (FAST)
# ============================================================

@torch.no_grad()
def extract_features_batch_fast(model, inputs, gt_rles, device, use_amp=True):
    """
    Extract DeRIS features + grid signatures + oracle labels.
    Everything computed on GPU; we only move final tensors to CPU.
    """
    m = unwrap_model(model)

    img = inputs["img"].to(device, non_blocking=True)
    ref_expr_inds = inputs["ref_expr_inds"].to(device, non_blocking=True)
    img_metas = inputs["img_metas"]

    text_attention_mask = inputs.get("text_attention_mask")
    if text_attention_mask is not None:
        text_attention_mask = text_attention_mask.to(device, non_blocking=True)

    img_mini = inputs.get("img_mini")
    if img_mini is not None:
        img_mini = img_mini.to(device, non_blocking=True)

    # Set batch_input_shape
    batch_input_shape = tuple(img.shape[-2:])
    for meta in img_metas:
        meta["batch_input_shape"] = batch_input_shape

    amp_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"), dtype=torch.float16)

    with amp_ctx:
        feat_dict = m.understanding_branch.pre_forward(
            img_mini, ref_expr_inds, text_attention_mask
        )

        perception_results = m.perception_branch(
            img, feat_dict["img_feat"], feat_dict["text_feat"],
            text_attention_mask, feat_dict["query_feat"]
        )

        mask_logits = perception_results["pred_masks"]  # [B,N,H,W] (fp16 under AMP)
        query_feat = perception_results["query_feat"]   # [B,N,D]

        if "pred_refer_logits" in perception_results:
            # DeRIS convention: SEG_cls outputs (1+1) classes; channel 0 = referent / positive class.
            # See DeRIS get_predictions_parts (mix_gref_hiervg_mr_loopback.py:396):
            #   scores = F.softmax(box_cls, dim=-1)[:, :, 0]
            det_scores = perception_results["pred_refer_logits"].softmax(dim=-1)[..., 0]  # [B,N]
        else:
            det_scores = torch.full(
                (query_feat.shape[0], query_feat.shape[1]),
                1.0 / query_feat.shape[1],
                device=device,
                dtype=query_feat.dtype,
            )

    # For IoU and stats, use probs in float32 for stable sums
    mask_probs = mask_logits.float().sigmoid()  # [B,N,H,W] float32

    # Mask stats
    mask_mean = mask_probs.mean(dim=(-2, -1))               # [B,N]
    mask_max = mask_probs.amax(dim=(-2, -1))                # [B,N]
    mask_area = (mask_probs > 0.5).float().mean(dim=(-2, -1))  # [B,N]
    mask_std = mask_probs.std(dim=(-2, -1), unbiased=False) # [B,N]

    # Grid signatures
    grid_sigs = compute_grid_signatures(mask_probs, grid_sizes=(4, 8, 16))

    # Oracle IoUs on GPU (vectorized)
    oracle_idx, query_ious = compute_oracle_and_ious_gpu(mask_probs, gt_rles, device=device)
    q0_iou = query_ious[:, 0]

    # Move to CPU (fp16) only what we need
    out = {
        "query_feat": query_feat.detach().cpu().half(),
        "det_scores": det_scores.detach().cpu().half(),
        "mask_mean": mask_mean.detach().cpu().half(),
        "mask_max":  mask_max.detach().cpu().half(),
        "mask_area": mask_area.detach().cpu().half(),
        "mask_std":  mask_std.detach().cpu().half(),
        "query_ious": query_ious.detach().cpu().half(),
        "oracle_idx": oracle_idx.detach().cpu().long(),
        "q0_iou": q0_iou.detach().cpu().half(),
    }
    for k, v in grid_sigs.items():
        out[k] = v.detach().cpu().half()

    # Also return pred H,W for meta
    out["_pred_hw"] = (int(mask_probs.shape[-2]), int(mask_probs.shape[-1]))
    return out


def _chunk_path(split_dir, shard_id, chunk_id):
    return os.path.join(split_dir, f"shard_{shard_id:02d}_chunk_{chunk_id:04d}.pt")


def _write_chunk(split_dir, shard_id, chunk_id, chunk_feats, last_index):
    """Write a single chunk file (only new samples since last flush). Atomic."""
    path = _chunk_path(split_dir, shard_id, chunk_id)
    out = {k: torch.stack(v) for k, v in chunk_feats.items()}
    out["chunk_meta"] = {"shard_id": shard_id, "count": int(out["query_feat"].shape[0]), "last_index": int(last_index)}
    tmp = path + ".tmp"
    torch.save(out, tmp)
    os.rename(tmp, path)


def extract_shard(args, split_name, shard_id, num_shards, device):
    """Extract features for one shard of a split. Indexed iteration + chunked flush."""
    split_dir = os.path.join(args.cache_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    shard_path = os.path.join(split_dir, f"shard_{shard_id:02d}_of_{num_shards:02d}.pt")
    progress_path = os.path.join(split_dir, f"shard_{shard_id:02d}.progress.json")

    if os.path.exists(shard_path) and not args.force:
        try:
            data = torch.load(shard_path, weights_only=False)
            if data.get("complete", False):
                print(f"  [{split_name}] Shard {shard_id}/{num_shards} already complete, skipping")
                return
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  SHARD {shard_id}/{num_shards} for {split_name} on device {device}")
    print(f"{'='*60}")

    model, cfg = load_deris_model(args.deris_checkpoint, args.deris_config, device)
    dataset = build_full_dataset(cfg, split_name)
    total_size = len(dataset)
    indexed_ds = IndexedDataset(dataset)

    shard_size = (total_size + num_shards - 1) // num_shards
    start_idx = shard_id * shard_size
    end_idx = min(start_idx + shard_size, total_size)
    full_shard_indices = list(range(start_idx, end_idx))

    # Resume
    last_processed = -1
    chunk_id = 0
    last_flushed_count = 0

    feat_keys = [
        "query_feat", "det_scores",
        "mask_mean", "mask_max", "mask_area", "mask_std",
        "grid_mean_4", "grid_max_4", "boundary_4",
        "grid_mean_8", "grid_max_8", "boundary_8",
        "grid_mean_16", "grid_max_16", "boundary_16",
        "query_ious", "oracle_idx", "q0_iou",
    ]
    all_feats = {k: [] for k in feat_keys}

    if args.resume and os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            progress = json.load(f)
            last_processed = int(progress.get("last_index", -1))
        print(f"  Resuming from index {last_processed + 1}")

        # Load existing chunks
        existing = sorted(glob.glob(os.path.join(split_dir, f"shard_{shard_id:02d}_chunk_*.pt")))
        for path in existing:
            try:
                data = torch.load(path, weights_only=False)
                for k in feat_keys:
                    if k in data:
                        # append each sample row to list
                        v = data[k]
                        for i in range(v.shape[0]):
                            all_feats[k].append(v[i])
                last_flushed_count = len(all_feats["query_feat"])
                chunk_id += 1
            except Exception as e:
                print(f"  Warning: failed to load chunk {path}: {e}")

    shard_indices = [i for i in full_shard_indices if i > last_processed]
    if len(shard_indices) == 0:
        # If shards already done but chunks exist, merge them
        if last_flushed_count > 0:
            _finalize_from_chunks(args, split_dir, split_name, shard_id, num_shards, device, feat_keys)
            if os.path.exists(progress_path):
                os.remove(progress_path)
            print(f"  All indices already processed; merged {last_flushed_count} samples.")
        else:
            print(f"  All indices already processed")
        return

    print(f"  Processing indices {start_idx}..{end_idx} ({len(shard_indices)} remaining)")

    shard_view = ShardView(indexed_ds, shard_indices)
    loader = DataLoader(
        shard_view,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(indexed_collate, batch_size=args.batch_size),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    t0 = time.time()
    last_idx_seen = last_processed
    n_batches = len(loader)
    pred_hw = None

    for batch_idx, (idxs, inputs) in enumerate(loader):
        # Pull GT masks (RLE) before extract_data
        gt_rles = None
        if "gt_mask_rle" in inputs:
            mask_data = inputs.pop("gt_mask_rle")
            gt_rles = mask_data.data[0] if hasattr(mask_data, "data") else mask_data

        # Remove keys we don't need
        for key in ["is_crowd", "gt_bbox", "gt_mask_parts_rle"]:
            inputs.pop(key, None)

        inputs_clean = extract_data(inputs)
        if gt_rles is None:
            raise RuntimeError("gt_mask_rle missing in batch; cannot compute oracle/query_ious.")

        try:
            feat = extract_features_batch_fast(
                model, inputs_clean, gt_rles, device,
                use_amp=not args.no_amp
            )
        except Exception as e:
            print(f"  Warning: batch {batch_idx} failed: {e}")
            continue

        B = feat["query_feat"].shape[0]
        last_idx_seen = int(idxs.max().item())
        pred_hw = feat["_pred_hw"]

        for i in range(B):
            for k in feat_keys:
                all_feats[k].append(feat[k][i])

        n_total = len(all_feats["query_feat"])

        # Progress
        if (batch_idx + 1) % args.log_every == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.time() - t0
            speed = (n_total - last_flushed_count) / elapsed if elapsed > 0 else 0.0
            remaining = len(shard_indices) - (n_total - last_flushed_count)
            eta = remaining / max(speed, 1e-9)
            print(f"  [{split_name}] Shard {shard_id} | {batch_idx+1}/{n_batches} | "
                  f"{n_total} total | {speed:.2f} samp/s | ETA {eta/60:.1f} min")

        # Chunked flush
        if (batch_idx + 1) % args.flush_every == 0 and n_total > last_flushed_count:
            chunk_feats = {k: all_feats[k][last_flushed_count:n_total] for k in feat_keys}
            try:
                _write_chunk(split_dir, shard_id, chunk_id, chunk_feats, last_idx_seen)
                chunk_id += 1
                last_flushed_count = n_total
                with open(progress_path + ".tmp", "w") as f:
                    json.dump({"last_index": last_idx_seen, "timestamp": datetime.now().isoformat()}, f)
                os.rename(progress_path + ".tmp", progress_path)
            except Exception as e:
                print(f"  Warning: chunk write failed: {e}")

    # Finalize: merge chunks + tail into final shard
    _finalize_from_chunks(args, split_dir, split_name, shard_id, num_shards, device, feat_keys, all_feats=all_feats, pred_hw=pred_hw)

    if os.path.exists(progress_path):
        os.remove(progress_path)

    print(f"  Saved to {shard_path} ({os.path.getsize(shard_path) / 1e6:.1f} MB)")


def _finalize_from_chunks(args, split_dir, split_name, shard_id, num_shards, device, feat_keys, all_feats=None, pred_hw=None):
    """Merge chunk files + remaining in-memory tail into final shard file, then cleanup."""
    shard_path = os.path.join(split_dir, f"shard_{shard_id:02d}_of_{num_shards:02d}.pt")

    chunk_files = sorted(glob.glob(os.path.join(split_dir, f"shard_{shard_id:02d}_chunk_*.pt")))
    merged = {}
    n_from_chunks = 0

    if chunk_files:
        for path in chunk_files:
            data = torch.load(path, weights_only=False)
            for k, v in data.items():
                if k == "chunk_meta":
                    continue
                merged.setdefault(k, []).append(v)
        result = {k: torch.cat(vs, dim=0) for k, vs in merged.items()}
        n_from_chunks = int(result["query_feat"].shape[0])
    else:
        result = None

    # Append tail from memory if present
    if all_feats is not None:
        n_total = len(all_feats["query_feat"])
        if result is None:
            result = {k: torch.stack(all_feats[k]) for k in feat_keys}
        else:
            if n_from_chunks < n_total:
                for k in feat_keys:
                    tail = all_feats[k][n_from_chunks:]
                    if tail:
                        result[k] = torch.cat([result[k], torch.stack(tail, dim=0)], dim=0)

    if result is None:
        raise RuntimeError(f"No data to finalize for split={split_name}, shard={shard_id}")

    # Meta
    N = int(result["query_feat"].shape[1])
    D = int(result["query_feat"].shape[2])
    H = int(pred_hw[0]) if pred_hw else 0
    W = int(pred_hw[1]) if pred_hw else 0

    result["meta"] = {
        "split": split_name,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "num_samples": int(result["query_feat"].shape[0]),
        "timestamp": datetime.now().isoformat(),
        "deris_ckpt_hash": file_hash(args.deris_checkpoint),
        "config_hash": file_hash(args.deris_config),
        "H": H, "W": W, "N": N, "D": D,
        "amp": (not args.no_amp),
    }
    result["complete"] = True

    tmp_path = shard_path + ".tmp"
    torch.save(result, tmp_path)
    os.rename(tmp_path, shard_path)

    # cleanup chunks
    for path in chunk_files:
        try:
            os.remove(path)
        except OSError:
            pass


# ============================================================
# Merge Shards
# ============================================================

def merge_shards(args, split_name):
    """Merge shards for a split into single file."""
    split_dir = os.path.join(args.cache_dir, split_name)
    if not os.path.exists(split_dir):
        print(f"  {split_name}: no shard directory")
        return

    shard_files = sorted([
        f for f in os.listdir(split_dir)
        if f.startswith("shard_") and f.endswith(".pt") and "_chunk_" not in f
    ])
    if len(shard_files) == 0:
        print(f"  {split_name}: no shards to merge")
        return

    print(f"\n  Merging {len(shard_files)} shards for {split_name}...")

    all_data = {}
    total_samples = 0

    for shard_file in shard_files:
        shard_path = os.path.join(split_dir, shard_file)
        data = torch.load(shard_path, weights_only=False)
        if not data.get("complete", False):
            print(f"    Warning: {shard_file} not complete, skipping")
            continue

        for k, v in data.items():
            if k in ["meta", "complete"]:
                continue
            all_data.setdefault(k, []).append(v)

        total_samples += int(data["meta"]["num_samples"])

    merged = {k: torch.cat(vs, dim=0) for k, vs in all_data.items()}

    merged["meta"] = {
        "split": split_name,
        "num_samples": total_samples,
        "num_shards_merged": len(shard_files),
        "timestamp": datetime.now().isoformat(),
    }
    merged_path = os.path.join(args.cache_dir, f"{split_name}_feats.pt")
    print(f"    Merged {total_samples} samples, saving to {merged_path}...")
    torch.save(merged, merged_path)
    print(f"    Saved ({os.path.getsize(merged_path) / 1e6:.1f} MB)")


# ============================================================
# Worker Process
# ============================================================

def worker_process(gpu_rank, gpu_id, shard_ids_per_split, args):
    """
    Worker for one GPU.
    NOTE: we map CUDA_VISIBLE_DEVICES to a single GPU per process for clean isolation.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # prevent CPU oversubscription (2 processes)
    torch.set_num_threads(1)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"\n[GPU {gpu_id}] Starting worker (rank={gpu_rank})...")

    splits = args.splits.split(",") if args.splits != "all" else ALL_SPLITS

    for split in splits:
        shard_ids = shard_ids_per_split[split][gpu_rank]
        for shard_id in shard_ids:
            extract_shard(args, split, shard_id, args.shards_per_split, device)

    print(f"\n[GPU {gpu_id}] Worker done")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("ECCV Grid Feature Extraction (FAST 2x3090)")

    # Paths
    parser.add_argument("--deris_checkpoint", default="/home/bionick87/miccai_2026/model/DERIS/DeRIS-L-refcoco.pth")
    parser.add_argument("--deris_config", default="/home/bionick87/miccai_2026/code/DeRIS/DeRIS-main/configs/refcoco/DERIS-L-refcoco.py")
    parser.add_argument("--cache_dir", default="./outputs/eccv_full_cache")

    # Extraction params
    parser.add_argument("--splits", default="all", help="Comma-separated or 'all'")
    parser.add_argument("--batch_size", type=int, default=128, help="3090: 128 is typical sweet spot")
    parser.add_argument("--num_workers", type=int, default=8, help="per-process dataloader workers (2 GPUs => 2 processes)")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor")
    parser.add_argument("--flush_every", type=int, default=200, help="flush chunks every N batches")
    parser.add_argument("--log_every", type=int, default=50)

    # Multi-GPU
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU IDs")
    parser.add_argument("--shards_per_split", type=int, default=None,
                        help="Num shards per split. Default = num_gpus*4 (better load balance).")
    parser.add_argument("--world_auto", action="store_true", help="Spawn 1 process per GPU")
    parser.add_argument("--shard_id", type=int, default=None, help="Manual shard ID (single process)")

    # Control
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge", action="store_true", help="Merge shards only")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP autocast")

    args = parser.parse_args()
    args.cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    os.makedirs(args.cache_dir, exist_ok=True)

    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    num_gpus = len(gpu_ids)

    if args.shards_per_split is None:
        # IMPORTANT: more shards than GPUs helps keep both 3090 busy across uneven splits.
        args.shards_per_split = max(num_gpus * 4, num_gpus)

    splits = args.splits.split(",") if args.splits != "all" else ALL_SPLITS

    # === MERGE MODE ===
    if args.merge:
        print("\n=== MERGE MODE ===")
        for split in splits:
            merge_shards(args, split)
        print("\nMerge complete!")
        return

    # Manual shard mode (use only first GPU in list)
    if args.shard_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        torch.set_num_threads(1)
        device = torch.device("cuda:0")
        split = splits[0]
        extract_shard(args, split, args.shard_id, args.shards_per_split, device)
        return

    # Auto multi-process mode
    if args.world_auto:
        print(f"\n=== AUTO MULTI-PROCESS MODE ({num_gpus} GPUs) ===")
        print(f"GPUs: {gpu_ids} | shards_per_split={args.shards_per_split} | batch_size={args.batch_size} | num_workers={args.num_workers}")

        # Assign shards to GPUs (round-robin)
        shard_ids_per_split = {}
        for split in splits:
            shard_assignment = [[] for _ in range(num_gpus)]
            for shard_id in range(args.shards_per_split):
                gpu_idx = shard_id % num_gpus
                shard_assignment[gpu_idx].append(shard_id)
            shard_ids_per_split[split] = shard_assignment

        mp.set_start_method("spawn", force=True)

        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            p = mp.Process(target=worker_process, args=(i, gpu_id, shard_ids_per_split, args))
            p.start()
            processes.append(p)
            print(f"  Started worker on GPU {gpu_id} (PID {p.pid})")

        for p in processes:
            p.join()

        print("\n=== All workers done ===")

        # Auto merge
        print("\n=== Auto-merging shards ===")
        for split in splits:
            merge_shards(args, split)
        print("\nExtraction + merge complete!")
    else:
        # Single-process (still uses CUDA_VISIBLE_DEVICES to isolate)
        print("\n=== SINGLE PROCESS MODE ===")
        shard_ids_per_split = {}
        for split in splits:
            shard_ids_per_split[split] = [list(range(args.shards_per_split))]
        worker_process(0, gpu_ids[0], shard_ids_per_split, args)

        print("\n=== Merging shards ===")
        for split in splits:
            merge_shards(args, split)


if __name__ == "__main__":
    main()
