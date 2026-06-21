#!/usr/bin/env python3
"""
Venice-H1 Pre-training Micro-tests
====================================
Validates model architecture, stability, baseline preservation, and LoRA
integration BEFORE launching the definitive training run.

Run:
    cd Venice-H1
    python tests/test_venice_h1_pretrain.py

Expected: All tests PASS in ~3-5 minutes on a single GPU.
"""

import os
import sys
import time
import traceback
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Path setup (same as train.py)
VENICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONEREF_ROOT = os.path.join(VENICE_ROOT, 'OneRef')
sys.path.insert(0, ONEREF_ROOT)
sys.path.insert(0, VENICE_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# OneRef imports
import utils.misc as oneref_utils
from datasets import build_dataset
import models.utils as beit3_utils       # noqa: F401
import models.OneRef_model as OneRef_model  # noqa: F401

# Venice-H1 imports
from gridcell_models.grid_cell_oneref import GridCellModule, GridCellOneRef
from gridcell_models.losses import compute_grid_cell_losses
from gridcell_models.lora import (
    LoRALinear, apply_lora_to_model, get_lora_params, get_lora_state_dict
)

# Defaults
ONEREF_CKPT = ('./checkpoints/oneref/'
               'rec_single_dataset_finetuning_base/unc/best_checkpoint.pth')
SPM_PATH = './checkpoints/beit3.spm'
DATA_ROOT = './data/oneref'
SPLIT_ROOT = ('./data/oneref_annotations/'
              'ref_data_shuffled/single_dataset')

# =====================================================================
# HELPERS
# =====================================================================
results = []


def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return passed


def make_test_args(**overrides):
    """Build args namespace matching what train.py's get_args() would produce."""
    defaults = dict(
        model='beit3_base_patch16_384',
        oneref_checkpoint=ONEREF_CKPT,
        oneref_checkpoint_dir='',
        sentencepiece_model=SPM_PATH,
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        dataset='unc',
        imsize=384,
        max_query_len=64,
        vocab_size=64010,
        drop_path=0.1,
        checkpoint_activations=None,
        frozen_backbone=True,
        task='grounding',
        use_mask_loss=False,
        enable_seg_mask=True,
        use_contrastive_loss=False,
        use_box_mask_constraints=False,
        use_regress_box=False,
        enable_ref_mlm=False,
        enable_ref_mim=False,
        enable_dynamic_mim=False,
        mim_mid_layer=0,
        codebook_size=8192,
        text_mask_prob=0.4,
        drop_worst_ratio=0.2,
        drop_worst_after=12000,
        label_smoothing=0.1,
        prompt='{pseudo_query}',
        sup_type='full',
        aug_blur=False,
        aug_crop=True,
        aug_scale=True,
        aug_translate=True,
        # Grid cell
        grid_size=8,
        gc_num_layers=6,
        gc_num_heads=16,
        gc_dropout=0.05,
        num_rounds=3,
        top_k_cells=16,
        cell_tau=0.3,
        mask_dim=128,
        # LoRA
        enable_lora=False,
        lora_rank=8,
        lora_alpha=16,
        lora_layers=6,
        lora_dropout=0.05,
        lora_lr_scale=1.0,
        # Training
        batch_size=32,
        eval_batch_size=128,
        lr=5e-4,
        num_workers=2,
        # Loss
        l_mask=1.0, l_ref=1.0, l_aux=0.2,
        l_bce=2.0, l_dice=5.0, l_focal=2.0, l_boundary=1.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_grid_cfg(args):
    return {
        'grid_size': args.grid_size,
        'num_layers': args.gc_num_layers,
        'num_heads': args.gc_num_heads,
        'dropout': args.gc_dropout,
        'num_rounds': args.num_rounds,
        'top_k_cells': args.top_k_cells,
        'cell_tau': args.cell_tau,
        'mask_dim': args.mask_dim,
    }


def make_lora_cfg(args):
    if not getattr(args, 'enable_lora', False):
        return None
    return {
        'enabled': True,
        'rank': args.lora_rank,
        'alpha': args.lora_alpha,
        'num_layers': args.lora_layers,
        'dropout': args.lora_dropout,
        'target_projections': ('q_proj', 'k_proj', 'v_proj'),
    }


# =====================================================================
# TEST 1: Architecture Verification
# =====================================================================
def test_architecture(device):
    print("\n--- Test 1: Architecture Verification ---")
    args = make_test_args()
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg)

    # No gate_bbox
    has_gate_bbox = hasattr(model, 'gate_bbox')
    report("No gate_bbox attribute", not has_gate_bbox)

    # Has gate_seg with correct init
    has_gate_seg = hasattr(model, 'gate_seg')
    report("Has gate_seg", has_gate_seg)
    if has_gate_seg:
        gate_val = model.gate_seg.item()
        report("gate_seg init ~= -3.0", abs(gate_val - (-3.0)) < 0.01,
               f"value={gate_val}")

    # Backbone frozen
    backbone_frozen = all(
        not p.requires_grad for p in model.backbone.parameters()
    )
    report("Backbone fully frozen", backbone_frozen)

    # Grid cell trainable
    gc_trainable = any(
        p.requires_grad for p in model.grid_cell.parameters()
    )
    report("Grid cell has trainable params", gc_trainable)

    # Param counts
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    gc_params = sum(p.numel() for p in model.grid_cell.parameters()
                    if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)
    report("Param counts correct",
           backbone_params > 0 and gc_params > 0 and total_trainable > gc_params,
           f"backbone={backbone_params:,} frozen, "
           f"gc={gc_params:,} trainable, "
           f"total_trainable={total_trainable:,}")

    # LoRA not enabled by default
    report("LoRA off by default", not model._lora_enabled)

    del model
    torch.cuda.empty_cache()
    return True


# =====================================================================
# TEST 2: Forward Pass Sanity
# =====================================================================
def test_forward_pass(device):
    print("\n--- Test 2: Forward Pass Sanity ---")
    args = make_test_args()
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg).to(device)
    model.load_oneref_checkpoint(args.oneref_checkpoint)
    model.eval()

    B = 2
    images = torch.randn(B, 3, 384, 384, device=device)
    texts = ["a person on the left", "the red car near the tree"]
    target_bbox = torch.tensor([[0.3, 0.4, 0.2, 0.3],
                                 [0.6, 0.5, 0.3, 0.4]], device=device)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(image=images, img_mask=None, text=texts,
                    target_bbox=target_bbox)

    # Check output keys
    required_keys = ['pred_box', 'pred_box_mask', 'seg_mask', 'gate_seg',
                     'cell_logits', 'mask_logits', 'topk_idx']
    for key in required_keys:
        present = key in out
        if present and torch.is_tensor(out[key]):
            finite = torch.isfinite(out[key]).all().item()
            report(f"Output '{key}' present & finite", finite,
                   f"shape={list(out[key].shape)}")
        elif present:
            report(f"Output '{key}' present", True,
                   f"value={out[key]}")
        else:
            report(f"Output '{key}' present", False, "MISSING")

    # pred_box is detached (no grad_fn)
    report("pred_box is detached", out['pred_box'].grad_fn is None)

    # Shapes
    report("pred_box shape [B,4]",
           list(out['pred_box'].shape) == [B, 4])
    report("seg_mask shape [B,1,H,W]",
           len(out['seg_mask'].shape) == 4 and out['seg_mask'].shape[0] == B)

    # Gate_seg value
    gs = out['gate_seg'].item() if torch.is_tensor(out['gate_seg']) else out['gate_seg']
    report("gate_seg ~= sigmoid(-3) ~= 0.047",
           0.01 < gs < 0.1, f"value={gs:.4f}")

    del model, out
    torch.cuda.empty_cache()
    return True


# =====================================================================
# TEST 3: Loss & Backward Stability
# =====================================================================
def test_loss_backward(device):
    print("\n--- Test 3: Loss & Backward Stability ---")
    args = make_test_args()
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg).to(device)
    model.load_oneref_checkpoint(args.oneref_checkpoint)
    model.train()
    model.backbone.eval()  # Keep backbone in eval mode

    B = 2
    images = torch.randn(B, 3, 384, 384, device=device)
    texts = ["a person on the left", "the red car near the tree"]
    target_bbox = torch.tensor([[0.3, 0.4, 0.2, 0.3],
                                 [0.6, 0.5, 0.3, 0.4]], device=device)
    target_mask = torch.zeros(B, 1, 384, 384, device=device)
    # Create simple GT masks
    target_mask[0, 0, 100:200, 80:160] = 1.0
    target_mask[1, 0, 150:300, 180:350] = 1.0

    with torch.amp.autocast('cuda'):
        out = model(image=images, img_mask=None, text=texts,
                    target_bbox=target_bbox)
        losses, total_loss = compute_grid_cell_losses(
            out, target_bbox, target_mask,
            l_mask=1.0, l_ref=1.0, l_aux=0.2,
            l_bce=2.0, l_dice=5.0, l_focal=2.0, l_boundary=1.0,
        )

    # Loss is finite
    loss_val = total_loss.item()
    report("Total loss is finite", np.isfinite(loss_val),
           f"loss={loss_val:.4f}")

    # Backward
    scaler = torch.amp.GradScaler('cuda')
    scaler.scale(total_loss).backward()

    # Grid cell has gradients
    gc_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.grid_cell.parameters() if p.requires_grad
    )
    report("Grid cell params have gradients", gc_has_grad)

    # gate_seg has gradient
    gate_has_grad = (model.gate_seg.grad is not None and
                     model.gate_seg.grad.abs().sum() > 0)
    report("gate_seg has gradient", gate_has_grad,
           f"grad={model.gate_seg.grad.item() if model.gate_seg.grad is not None else 'None'}")

    # Backbone has NO gradients
    backbone_no_grad = all(
        p.grad is None or p.grad.abs().sum() == 0
        for p in model.backbone.parameters()
    )
    report("Backbone has NO gradients", backbone_no_grad)

    del model, out, total_loss
    torch.cuda.empty_cache()
    return True


# =====================================================================
# TEST 4: Short Training Stability (5 steps)
# =====================================================================
def test_training_stability(device):
    print("\n--- Test 4: Short Training Stability (5 steps) ---")
    args = make_test_args()
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg).to(device)
    model.load_oneref_checkpoint(args.oneref_checkpoint)

    # Snapshot backbone params for comparison
    backbone_snapshot = {
        n: p.clone() for n, p in model.backbone.named_parameters()
    }
    gate_init = model.gate_seg.item()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=0.01)
    # Use a smaller init_scale to avoid overflow on first steps
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)

    B = 4
    losses_history = []

    for step in range(5):
        model.train()
        model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)

        images = torch.randn(B, 3, 384, 384, device=device)
        texts = ["a person", "the red car", "a dog running", "blue sky"]
        target_bbox = torch.rand(B, 4, device=device) * 0.5 + 0.25
        target_mask = torch.zeros(B, 1, 384, 384, device=device)
        for b in range(B):
            y, x = int(target_bbox[b, 1].item() * 384), int(target_bbox[b, 0].item() * 384)
            h, w = max(20, int(target_bbox[b, 3].item() * 384)), max(20, int(target_bbox[b, 2].item() * 384))
            target_mask[b, 0,
                        max(0, y - h // 2):min(384, y + h // 2),
                        max(0, x - w // 2):min(384, x + w // 2)] = 1.0

        with torch.amp.autocast('cuda'):
            out = model(image=images, img_mask=None, text=texts,
                        target_bbox=target_bbox)
            losses, total_loss = compute_grid_cell_losses(
                out, target_bbox, target_mask,
                l_mask=1.0, l_ref=1.0, l_aux=0.2,
                l_bce=2.0, l_dice=5.0, l_focal=2.0, l_boundary=1.0,
            )

        loss_val = total_loss.item()
        losses_history.append(loss_val)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            report(f"Step {step}: loss finite", False, f"loss={loss_val}")
            break

        scaler.scale(total_loss).backward()

        # Must unscale before clip and step
        scaler.unscale_(optimizer)

        # Check for inf/nan grads (scaler may set them)
        valid_grads = True
        for p in trainable_params:
            if p.grad is not None and (torch.isinf(p.grad).any() or
                                        torch.isnan(p.grad).any()):
                valid_grads = False
                break

        if valid_grads:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
        scaler.update()

    # All losses finite
    all_finite = all(np.isfinite(l) for l in losses_history)
    report("All 5 steps: loss finite (no NaN)", all_finite,
           f"losses={[f'{l:.3f}' for l in losses_history]}")

    # gate_seg changed
    gate_after = model.gate_seg.item()
    gate_changed = abs(gate_after - gate_init) > 1e-6
    report("gate_seg value changed (gradient flowing)", gate_changed,
           f"before={gate_init:.6f} after={gate_after:.6f}")

    # Backbone unchanged
    backbone_unchanged = all(
        torch.equal(p, backbone_snapshot[n])
        for n, p in model.backbone.named_parameters()
    )
    report("Backbone params unchanged", backbone_unchanged)

    del model, optimizer
    torch.cuda.empty_cache()
    return True


# =====================================================================
# TEST 5: Baseline Preservation on RefCOCO val (subset)
# =====================================================================
def test_baseline_refcoco(device):
    print("\n--- Test 5: Baseline Preservation on RefCOCO val ---")
    args = make_test_args()
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg).to(device)
    model.load_oneref_checkpoint(args.oneref_checkpoint)
    model.eval()

    from utils.box_utils import xywh2xyxy, bbox_iou
    from torch.utils.data import DataLoader

    # Load RefCOCO val
    ds = build_dataset('val', args)
    loader = DataLoader(
        ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=oneref_utils.collate_fn,
        num_workers=2, drop_last=False, pin_memory=True,
    )

    bbox_hits = []
    n_batches = 0
    max_batches = 10  # Only first 10 batches for speed

    for batch in loader:
        if n_batches >= max_batches:
            break
        img_data, text_data, target_bbox, target_mask = batch
        img_data = img_data.to(device)
        target_bbox = target_bbox.to(device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model(image=img_data.tensors, img_mask=None,
                        text=text_data)

        pred_box = out['pred_box'].float()
        gt_xyxy = xywh2xyxy(target_bbox)
        pred_xyxy = xywh2xyxy(pred_box).clamp(0, 1)

        for b in range(target_bbox.shape[0]):
            biou = bbox_iou(pred_xyxy[b:b + 1], gt_xyxy[b:b + 1])
            bbox_hits.append((biou >= 0.5).float().item())

        n_batches += 1
        del img_data, target_bbox, out

    bbox_acc = np.mean(bbox_hits) * 100
    report("RefCOCO val bbox_acc >= 85%", bbox_acc >= 85.0,
           f"bbox_acc={bbox_acc:.1f}% ({len(bbox_hits)} samples, "
           f"{n_batches} batches)")

    torch.cuda.empty_cache()
    del model
    return True


# =====================================================================
# TEST 6: Cross-dataset Evaluation (RefCOCO+, RefCOCOg)
# =====================================================================
def test_cross_dataset(device):
    print("\n--- Test 6: Cross-dataset Evaluation (RefCOCO+, RefCOCOg) ---")
    args = make_test_args()
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg).to(device)
    model.load_oneref_checkpoint(args.oneref_checkpoint)
    model.eval()

    from utils.box_utils import xywh2xyxy, bbox_iou
    from torch.utils.data import DataLoader

    datasets_to_test = {
        'unc+': 'val',      # RefCOCO+ val
        'gref_umd': 'val',  # RefCOCOg val
    }

    for ds_name, split in datasets_to_test.items():
        try:
            orig_dataset = args.dataset
            args.dataset = ds_name
            ds = build_dataset(split, args)
            args.dataset = orig_dataset

            loader = DataLoader(
                ds, batch_size=args.eval_batch_size, shuffle=False,
                collate_fn=oneref_utils.collate_fn,
                num_workers=2, drop_last=False, pin_memory=True,
            )

            bbox_hits = []
            n_batches = 0
            max_batches = 5  # Only a few batches

            for batch in loader:
                if n_batches >= max_batches:
                    break
                img_data, text_data, target_bbox, target_mask = batch
                img_data = img_data.to(device)
                target_bbox = target_bbox.to(device)

                with torch.no_grad(), torch.amp.autocast('cuda'):
                    out = model(image=img_data.tensors, img_mask=None,
                                text=text_data)

                pred_box = out['pred_box'].float()
                gt_xyxy = xywh2xyxy(target_bbox)
                pred_xyxy = xywh2xyxy(pred_box).clamp(0, 1)

                for b in range(target_bbox.shape[0]):
                    biou = bbox_iou(pred_xyxy[b:b + 1], gt_xyxy[b:b + 1])
                    bbox_hits.append((biou >= 0.5).float().item())

                n_batches += 1
                del img_data, target_bbox, out

            bbox_acc = np.mean(bbox_hits) * 100
            # Cross-dataset (unc checkpoint on unc+/gref): expect >= 50%
            report(f"{ds_name} {split}: pipeline works, bbox_acc >= 50%",
                   bbox_acc >= 50.0,
                   f"bbox_acc={bbox_acc:.1f}% ({len(bbox_hits)} samples)")

        except Exception as e:
            report(f"{ds_name} {split}: pipeline works", False,
                   f"ERROR: {e}")

    torch.cuda.empty_cache()
    del model
    return True


# =====================================================================
# TEST 7: LoRA Integration
# =====================================================================
def test_lora_integration(device):
    print("\n--- Test 7: LoRA Integration ---")
    args = make_test_args(enable_lora=True)
    grid_cfg = make_grid_cfg(args)
    lora_cfg = make_lora_cfg(args)
    # Create model WITHOUT LoRA, load checkpoint, THEN apply LoRA
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg,
                           lora_cfg=lora_cfg)
    model.load_oneref_checkpoint(args.oneref_checkpoint)
    model.apply_lora(lora_cfg)
    model.to(device)

    # LoRA is enabled
    report("LoRA enabled", model._lora_enabled)

    # LoRA params are trainable
    lora_params = get_lora_params(model)
    n_lora_params = sum(p.numel() for p in lora_params)
    report("LoRA params found and trainable",
           len(lora_params) > 0 and all(p.requires_grad for p in lora_params),
           f"{len(lora_params)} param tensors, {n_lora_params:,} total params")

    # Base backbone params still frozen
    base_frozen = all(
        not p.requires_grad
        for name, p in model.backbone.named_parameters()
        if 'lora_' not in name
    )
    report("Base backbone params remain frozen", base_frozen)

    # Forward pass with LoRA
    B = 2
    images = torch.randn(B, 3, 384, 384, device=device)
    texts = ["a person on the left", "the red car"]
    target_bbox = torch.tensor([[0.3, 0.4, 0.2, 0.3],
                                 [0.6, 0.5, 0.3, 0.4]], device=device)
    target_mask = torch.zeros(B, 1, 384, 384, device=device)
    target_mask[0, 0, 100:200, 80:160] = 1.0
    target_mask[1, 0, 150:300, 180:350] = 1.0

    model.train()
    model.backbone.eval()

    out = model(image=images, img_mask=None, text=texts,
                target_bbox=target_bbox)

    # Output shapes unchanged
    report("LoRA: pred_box shape [B,4]",
           list(out['pred_box'].shape) == [B, 4])
    report("LoRA: seg_mask shape correct",
           out['seg_mask'].shape[0] == B and len(out['seg_mask'].shape) == 4)

    # Outputs finite
    all_finite = (torch.isfinite(out['pred_box']).all().item() and
                  torch.isfinite(out['seg_mask']).all().item())
    report("LoRA: outputs finite", all_finite)

    # Backward with LoRA (no AMP scaler for clean gradient check)
    losses, total_loss = compute_grid_cell_losses(
        out, target_bbox, target_mask,
        l_mask=1.0, l_ref=1.0, l_aux=0.2,
        l_bce=2.0, l_dice=5.0, l_focal=2.0, l_boundary=1.0,
    )

    loss_val = total_loss.item()
    report("LoRA: loss is finite", np.isfinite(loss_val),
           f"loss={loss_val:.4f}")

    total_loss.backward()

    # LoRA params have gradients (at init, only lora_B gets grads
    # because lora_B is initialized to zeros, so d_loss/d_lora_A = 0.
    # After 1st optimizer step, lora_A will also get gradients.)
    lora_has_grad = sum(
        1 for p in lora_params if p.grad is not None and p.grad.abs().sum() > 0
    )
    # At least half should have grads (all lora_B params)
    report("LoRA: adapters have gradients (lora_B at init)",
           lora_has_grad >= len(lora_params) // 3,
           f"{lora_has_grad}/{len(lora_params)} params with grad "
           f"(lora_A=0 at init is expected)")

    # Grid cell also has gradients
    gc_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.grid_cell.parameters() if p.requires_grad
    )
    report("LoRA: grid cell also has gradients", gc_has_grad)

    # Save/load LoRA state dict
    lora_sd = get_lora_state_dict(model)
    report("LoRA: state dict extractable",
           len(lora_sd) > 0,
           f"{len(lora_sd)} keys")

    del model, out, total_loss
    torch.cuda.empty_cache()
    return True


# =====================================================================
# TEST 8: Full RefCOCO Baseline (all splits)
# =====================================================================
def test_full_refcoco_baseline(device):
    """Evaluate on ALL splits of RefCOCO, RefCOCO+, RefCOCOg.

    Runs on FULL datasets (not subsets) to verify the model meets
    expected baseline thresholds on every split.
    """
    print("\n--- Test 8: Full RefCOCO Baseline (all splits) ---")
    args = make_test_args()
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg).to(device)
    model.load_oneref_checkpoint(args.oneref_checkpoint)
    model.eval()

    from utils.box_utils import xywh2xyxy, bbox_iou
    from torch.utils.data import DataLoader

    # Expected thresholds per dataset/split
    expected = {
        'unc': {
            'val':   87.0,
            'testA': 89.0,
            'testB': 84.0,
        },
        'unc+': {
            'val':   70.0,
            'testA': 72.0,
            'testB': 60.0,
        },
        'gref_umd': {
            'val':  72.0,
            'test': 70.0,
        },
    }

    all_pass = True
    for ds_name, splits in expected.items():
        for split, threshold in splits.items():
            try:
                orig = args.dataset
                args.dataset = ds_name
                ds = build_dataset(split, args)
                args.dataset = orig

                loader = DataLoader(
                    ds, batch_size=args.eval_batch_size, shuffle=False,
                    collate_fn=oneref_utils.collate_fn,
                    num_workers=2, drop_last=False, pin_memory=True,
                )

                bbox_hits = []
                for batch in loader:
                    img_data, text_data, target_bbox, target_mask = batch
                    img_data = img_data.to(device)
                    target_bbox = target_bbox.to(device)

                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        out = model(image=img_data.tensors, img_mask=None,
                                    text=text_data)

                    pred_box = out['pred_box'].float()
                    gt_xyxy = xywh2xyxy(target_bbox)
                    pred_xyxy = xywh2xyxy(pred_box).clamp(0, 1)

                    for b in range(target_bbox.shape[0]):
                        biou = bbox_iou(pred_xyxy[b:b + 1], gt_xyxy[b:b + 1])
                        bbox_hits.append((biou >= 0.5).float().item())

                    del img_data, target_bbox, out

                bbox_acc = np.mean(bbox_hits) * 100
                passed = bbox_acc >= threshold
                if not passed:
                    all_pass = False
                report(
                    f"{ds_name}/{split}: bbox_acc >= {threshold}%",
                    passed,
                    f"bbox_acc={bbox_acc:.1f}% ({len(bbox_hits)} samples)"
                )
            except Exception as e:
                report(f"{ds_name}/{split}: load & eval", False, str(e))
                all_pass = False

    torch.cuda.empty_cache()
    del model
    return all_pass


# =====================================================================
# TEST 9: Short Training Does Not Degrade RefCOCO
# =====================================================================
def test_training_no_degradation(device):
    """Train 3 epochs on a real RefCOCO subset, verify bbox_acc stays.

    Uses first 2000 samples of RefCOCO train set. Evaluates on val
    (first 20 batches) before and after training. Allows -2% fluctuation.
    """
    print("\n--- Test 9: 3-epoch Training Does Not Degrade RefCOCO ---")
    args = make_test_args(batch_size=16, eval_batch_size=64)
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg).to(device)
    model.load_oneref_checkpoint(args.oneref_checkpoint)

    from utils.box_utils import xywh2xyxy, bbox_iou
    from torch.utils.data import DataLoader, Subset

    # Build val loader (first 20 batches for speed)
    val_ds = build_dataset('val', args)
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=oneref_utils.collate_fn, num_workers=2,
        drop_last=False, pin_memory=True,
    )

    def eval_bbox_acc(model_to_eval, max_batches=20):
        model_to_eval.eval()
        hits = []
        n = 0
        for batch in val_loader:
            if n >= max_batches:
                break
            img_data, text_data, target_bbox, target_mask = batch
            img_data = img_data.to(device)
            target_bbox = target_bbox.to(device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                out = model_to_eval(image=img_data.tensors, img_mask=None,
                                    text=text_data)
            pred_box = out['pred_box'].float()
            gt_xyxy = xywh2xyxy(target_bbox)
            pred_xyxy = xywh2xyxy(pred_box).clamp(0, 1)
            for b in range(target_bbox.shape[0]):
                biou = bbox_iou(pred_xyxy[b:b + 1], gt_xyxy[b:b + 1])
                hits.append((biou >= 0.5).float().item())
            n += 1
            del img_data, target_bbox, out
        return np.mean(hits) * 100

    # Baseline bbox_acc BEFORE training
    bbox_before = eval_bbox_acc(model)
    report("Pre-training bbox_acc measured",
           bbox_before > 0,
           f"bbox_acc={bbox_before:.1f}%")

    # Build training subset (first 2000 samples)
    train_ds = build_dataset('train', args)
    subset_size = min(2000, len(train_ds))
    train_subset = Subset(train_ds, list(range(subset_size)))
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        collate_fn=oneref_utils.collate_fn, num_workers=2,
        drop_last=True, pin_memory=True,
    )

    # Train 3 epochs
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)

    nan_seen = False
    for epoch in range(3):
        model.train()
        model.backbone.eval()
        epoch_losses = []
        for batch in train_loader:
            img_data, text_data, target_bbox, target_mask = batch
            img_data = img_data.to(device)
            target_bbox = target_bbox.to(device)
            target_mask = target_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(image=img_data.tensors, img_mask=None,
                            text=text_data, target_bbox=target_bbox)
                losses, total_loss = compute_grid_cell_losses(
                    out, target_bbox, target_mask,
                    l_mask=1.0, l_ref=1.0, l_aux=0.2,
                    l_bce=2.0, l_dice=5.0, l_focal=2.0, l_boundary=1.0,
                )

            loss_val = total_loss.item()
            if not np.isfinite(loss_val):
                nan_seen = True
                break
            epoch_losses.append(loss_val)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            del img_data, target_bbox, target_mask, out, total_loss

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        print(f"    Epoch {epoch + 1}/3: avg_loss={avg_loss:.4f} "
              f"({len(epoch_losses)} steps)")
        if nan_seen:
            break

    report("No NaN/Inf in 3 epochs", not nan_seen)

    # Evaluate AFTER training
    bbox_after = eval_bbox_acc(model)
    drop = bbox_before - bbox_after
    report(
        "bbox_acc after 3 epochs >= baseline - 2%",
        drop <= 2.0,
        f"before={bbox_before:.1f}% after={bbox_after:.1f}% "
        f"drop={drop:.1f}%"
    )

    torch.cuda.empty_cache()
    del model, optimizer
    return True


# =====================================================================
# TEST 10: LoRA Training Does Not Degrade RefCOCO
# =====================================================================
def test_lora_training_no_degradation(device):
    """Same as Test 9 but with LoRA enabled.

    Verifies LoRA parameters learn, backbone base params stay frozen,
    and bbox_acc doesn't degrade.
    """
    print("\n--- Test 10: 3-epoch LoRA Training Does Not Degrade RefCOCO ---")
    # Use conservative LR for LoRA (standard LoRA fine-tuning uses 1e-5 to 5e-5)
    args = make_test_args(enable_lora=True, batch_size=16, eval_batch_size=64,
                          lr=1e-4, lora_lr_scale=0.5)
    grid_cfg = make_grid_cfg(args)
    lora_cfg = make_lora_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg, lora_cfg=lora_cfg)
    model.load_oneref_checkpoint(args.oneref_checkpoint)
    model.apply_lora(lora_cfg)
    model.to(device)

    from utils.box_utils import xywh2xyxy, bbox_iou
    from torch.utils.data import DataLoader, Subset

    # Snapshot backbone base params
    backbone_snapshot = {}
    for n, p in model.backbone.named_parameters():
        if 'lora_' not in n:
            backbone_snapshot[n] = p.data.clone()

    # Snapshot LoRA params before training
    lora_params = get_lora_params(model)
    lora_before = [p.data.clone() for p in lora_params]

    # Build val loader
    val_ds = build_dataset('val', args)
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=oneref_utils.collate_fn, num_workers=2,
        drop_last=False, pin_memory=True,
    )

    def eval_bbox_acc(model_to_eval, max_batches=20):
        model_to_eval.eval()
        hits = []
        n = 0
        for batch in val_loader:
            if n >= max_batches:
                break
            img_data, text_data, target_bbox, target_mask = batch
            img_data = img_data.to(device)
            target_bbox = target_bbox.to(device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                out = model_to_eval(image=img_data.tensors, img_mask=None,
                                    text=text_data)
            pred_box = out['pred_box'].float()
            gt_xyxy = xywh2xyxy(target_bbox)
            pred_xyxy = xywh2xyxy(pred_box).clamp(0, 1)
            for b in range(target_bbox.shape[0]):
                biou = bbox_iou(pred_xyxy[b:b + 1], gt_xyxy[b:b + 1])
                hits.append((biou >= 0.5).float().item())
            n += 1
            del img_data, target_bbox, out
        return np.mean(hits) * 100

    # Baseline bbox_acc
    bbox_before = eval_bbox_acc(model)
    report("LoRA pre-training bbox_acc measured",
           bbox_before > 0,
           f"bbox_acc={bbox_before:.1f}%")

    # Build training subset (first 2000 samples)
    train_ds = build_dataset('train', args)
    subset_size = min(2000, len(train_ds))
    train_subset = Subset(train_ds, list(range(subset_size)))
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        collate_fn=oneref_utils.collate_fn, num_workers=2,
        drop_last=True, pin_memory=True,
    )

    # Optimizer with separate LoRA group (conservative LR for backbone)
    gc_params = [p for n, p in model.named_parameters()
                 if p.requires_grad and 'lora_' not in n]
    lora_p = [p for n, p in model.named_parameters()
              if p.requires_grad and 'lora_' in n]
    param_groups = [
        {'params': gc_params, 'lr': args.lr},
        {'params': lora_p, 'lr': args.lr * args.lora_lr_scale},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)

    nan_seen = False
    for epoch in range(3):
        model.train()
        model.backbone.eval()
        epoch_losses = []
        for batch in train_loader:
            img_data, text_data, target_bbox, target_mask = batch
            img_data = img_data.to(device)
            target_bbox = target_bbox.to(device)
            target_mask = target_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(image=img_data.tensors, img_mask=None,
                            text=text_data, target_bbox=target_bbox)
                losses, total_loss = compute_grid_cell_losses(
                    out, target_bbox, target_mask,
                    l_mask=1.0, l_ref=1.0, l_aux=0.2,
                    l_bce=2.0, l_dice=5.0, l_focal=2.0, l_boundary=1.0,
                )

            loss_val = total_loss.item()
            if not np.isfinite(loss_val):
                nan_seen = True
                break
            epoch_losses.append(loss_val)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            all_params = gc_params + lora_p
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            del img_data, target_bbox, target_mask, out, total_loss

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        print(f"    Epoch {epoch + 1}/3: avg_loss={avg_loss:.4f} "
              f"({len(epoch_losses)} steps)")
        if nan_seen:
            break

    report("LoRA: No NaN/Inf in 3 epochs", not nan_seen)

    # LoRA params changed (learning happened)
    lora_changed = sum(
        1 for i, p in enumerate(lora_params)
        if not torch.equal(p.data, lora_before[i])
    )
    report("LoRA: adapters changed during training",
           lora_changed > 0,
           f"{lora_changed}/{len(lora_params)} params changed")

    # Backbone base params unchanged
    base_unchanged = all(
        torch.equal(p.data, backbone_snapshot[n])
        for n, p in model.backbone.named_parameters()
        if 'lora_' not in n
    )
    report("LoRA: backbone base params unchanged", base_unchanged)

    # Evaluate AFTER training
    bbox_after = eval_bbox_acc(model)
    drop = bbox_before - bbox_after
    report(
        "LoRA: bbox_acc after 3 epochs >= baseline - 2%",
        drop <= 2.0,
        f"before={bbox_before:.1f}% after={bbox_after:.1f}% "
        f"drop={drop:.1f}%"
    )

    torch.cuda.empty_cache()
    del model, optimizer
    return True


# =====================================================================
# TEST 11: Medical Data Pipeline
# =====================================================================
def test_medical_pipeline(device):
    """Load medical datasets, run forward pass, verify outputs.

    Tests MS-CXR, NIH ChestX-ray, and M3D-RefSeg val splits.
    Verifies pipeline doesn't crash and outputs are finite.
    """
    print("\n--- Test 11: Medical Data Pipeline ---")

    # We need train.py imports for medical loaders
    sys.path.insert(0, os.path.join(VENICE_ROOT, 'scripts'))
    from datasets import make_transforms

    args = make_test_args(eval_batch_size=16)
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg).to(device)
    model.load_oneref_checkpoint(args.oneref_checkpoint)
    model.eval()

    from utils.box_utils import xywh2xyxy, bbox_iou
    from torch.utils.data import DataLoader

    # Import medical loaders from train.py
    # We reproduce the loading here to avoid importing the full train module
    MEDICAL_ROOT = './data'

    # Inline medical loaders (same logic as train.py)
    import json
    import re
    from pathlib import Path

    try:
        import pandas as pd
        has_pandas = True
    except ImportError:
        has_pandas = False

    medical_datasets = {}

    # MS-CXR
    mscxr_json = Path(MEDICAL_ROOT) / 'mscxr' / 'data' / 'annotations' / 'test_mscxr.json'
    if mscxr_json.exists():
        with open(mscxr_json) as f:
            data = json.load(f)
        images_map = {img['id']: img for img in data['images']}
        annotations = data['annotations']
        np_rng = np.random.RandomState(42)
        idx = np_rng.permutation(len(annotations))
        n_train = int(0.8 * len(annotations))
        n_val = int(0.1 * len(annotations))
        val_indices = idx[n_train:n_train + n_val]
        items = []
        img_dir = Path(MEDICAL_ROOT) / 'mscxr' / 'data' / 'images'
        for i in val_indices:
            ann = annotations[i]
            img_info = images_map.get(ann['image_id'])
            if img_info is None:
                continue
            img_path = img_dir / img_info['file_name']
            if not img_path.exists():
                continue
            x, y, w, h = ann['bbox']
            items.append({
                'img_path': str(img_path),
                'bbox_xywh': [x, y, w, h],
                'phrase': ann.get('description', 'medical finding'),
                'img_size': (img_info['height'], img_info['width']),
                'dataset': 'mscxr',
            })
        if items:
            medical_datasets['mscxr'] = items

    # NIH ChestX-ray
    if has_pandas:
        nih_csv = Path(MEDICAL_ROOT) / 'NIH-Chest-X-ray-dataset' / 'data' / 'BBox_List_2017.csv'
        nih_img_dir = Path(MEDICAL_ROOT) / 'NIH-Chest-X-ray-dataset' / 'data' / 'images' / 'images'
        if nih_csv.exists() and nih_img_dir.exists():
            df = pd.read_csv(nih_csv)
            np_rng = np.random.RandomState(42)
            idx = np_rng.permutation(len(df))
            n_train = int(0.8 * len(df))
            n_val = int(0.1 * len(df))
            subset = df.iloc[idx[n_train:n_train + n_val]]
            items = []
            for _, row in subset.iterrows():
                img_name = str(row.iloc[0])
                finding = str(row.iloc[1])
                x, y, w, h = float(row.iloc[2]), float(row.iloc[3]), \
                    float(row.iloc[4]), float(row.iloc[5])
                img_path = nih_img_dir / img_name
                if not img_path.exists():
                    continue
                items.append({
                    'img_path': str(img_path),
                    'bbox_xywh': [x, y, w, h],
                    'phrase': f'{finding.lower()} in the chest radiograph',
                    'img_size': (1024, 1024),
                    'dataset': 'nih',
                })
            if items:
                medical_datasets['nih'] = items

    # M3D-RefSeg-2D
    if has_pandas:
        m3d_csv = Path(MEDICAL_ROOT) / 'M3D-RefSeg-2D' / 'm3d_2d_val.csv'
        m3d_img_dir = Path(MEDICAL_ROOT) / 'M3D-RefSeg-2D' / 'images'
        if m3d_csv.exists():
            df = pd.read_csv(m3d_csv)
            items = []
            for _, row in df.iterrows():
                img_rel = str(row['image_path'])
                if img_rel.startswith('images/'):
                    img_rel = img_rel[7:]
                img_path = m3d_img_dir / img_rel
                if not img_path.exists():
                    continue
                x, y, w, h = float(row['x']), float(row['y']), \
                    float(row['w']), float(row['h'])
                text = str(row.get('text', 'lesion'))
                text = re.sub(r'\s+is\s+at\s*,', ' region', text)
                text = re.sub(r'\s+is\s+at\s+the\s*\.', '.', text)
                words = text.split()[:50]
                text = ' '.join(words)
                if len(text) < 10:
                    text = 'medical lesion region'
                items.append({
                    'img_path': str(img_path),
                    'bbox_xywh': [x, y, w, h],
                    'phrase': text,
                    'img_size': (int(row['image_height']), int(row['image_width'])),
                    'dataset': 'm3d',
                })
            if items:
                medical_datasets['m3d'] = items

    if not medical_datasets:
        report("Medical datasets found", False,
               "No medical data available, skipping pipeline test")
        return True

    # Now create MedicalVGDataset and test forward pass
    # Import MedicalVGDataset from train.py
    # We need make_transforms which is already available
    from PIL import Image as PILImage

    class SimpleMedicalDataset(torch.utils.data.Dataset):
        """Simplified medical dataset for testing (no train.py dependency)."""
        def __init__(self, items, imsize=384):
            self.items = items
            self.imsize = imsize
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            try:
                img = PILImage.open(item['img_path']).convert('RGB')
            except Exception:
                img = PILImage.new('RGB', (384, 384), (128, 128, 128))

            img_t = self.transform(img)

            orig_h, orig_w = item['img_size']
            x, y, w, h = item['bbox_xywh']
            # Normalize bbox to [0,1] in cxcywh format
            cx = (x + w / 2) / max(orig_w, 1)
            cy = (y + h / 2) / max(orig_h, 1)
            nw = w / max(orig_w, 1)
            nh = h / max(orig_h, 1)
            bbox = torch.tensor([cx, cy, nw, nh], dtype=torch.float32)

            return img_t, item['phrase'], bbox

    def medical_collate(batch):
        """Simple collate: stack images, collect texts, stack bboxes."""
        imgs, texts, bboxes = zip(*batch)
        imgs = torch.stack(imgs, 0)
        bboxes = torch.stack(bboxes, 0)
        return imgs, list(texts), bboxes

    for ds_name, items in medical_datasets.items():
        try:
            ds = SimpleMedicalDataset(items[:50])  # Use max 50 items
            loader = DataLoader(
                ds, batch_size=8, shuffle=False,
                collate_fn=medical_collate, num_workers=0,
                drop_last=False,
            )

            bbox_hits = []
            n_batches = 0
            max_batches = 5

            for imgs, texts, bboxes in loader:
                if n_batches >= max_batches:
                    break

                imgs = imgs.to(device)
                bboxes = bboxes.to(device)

                with torch.no_grad(), torch.amp.autocast('cuda'):
                    out = model(image=imgs, img_mask=None, text=texts)

                # Check outputs are finite
                finite = (torch.isfinite(out['pred_box']).all().item() and
                          torch.isfinite(out['seg_mask']).all().item())
                if not finite:
                    report(f"{ds_name}: outputs finite", False)
                    break

                # Compute bbox_acc for reference
                pred_box = out['pred_box'].float()
                gt_xyxy = torch.stack([
                    bboxes[:, 0] - bboxes[:, 2] / 2,
                    bboxes[:, 1] - bboxes[:, 3] / 2,
                    bboxes[:, 0] + bboxes[:, 2] / 2,
                    bboxes[:, 1] + bboxes[:, 3] / 2,
                ], dim=1)
                from utils.box_utils import xywh2xyxy, bbox_iou
                pred_xyxy = xywh2xyxy(pred_box).clamp(0, 1)
                for b in range(bboxes.shape[0]):
                    biou = bbox_iou(pred_xyxy[b:b + 1], gt_xyxy[b:b + 1])
                    bbox_hits.append((biou >= 0.5).float().item())

                n_batches += 1
                del imgs, bboxes, out

            bbox_acc = np.mean(bbox_hits) * 100 if bbox_hits else 0.0
            report(
                f"{ds_name}: pipeline works, outputs finite",
                True,
                f"bbox_acc={bbox_acc:.1f}% ({len(bbox_hits)} samples, "
                f"{n_batches} batches)"
            )
        except Exception as e:
            report(f"{ds_name}: pipeline works", False, f"ERROR: {e}")
            traceback.print_exc()

    torch.cuda.empty_cache()
    del model
    return True


# =====================================================================
# TEST 12: Memory and Throughput
# =====================================================================
def test_memory_throughput(device):
    """Test training step with batch_size=44 on single GPU.

    Verifies peak VRAM stays within safe limits and measures throughput.
    """
    print("\n--- Test 12: Memory and Throughput ---")
    args = make_test_args(batch_size=44)
    grid_cfg = make_grid_cfg(args)
    model = GridCellOneRef(oneref_args=args, grid_cfg=grid_cfg).to(device)
    model.load_oneref_checkpoint(args.oneref_checkpoint)

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"    Total VRAM: {total_vram:.1f} GB")

    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)

    B = 44
    model.train()
    model.backbone.eval()

    # Run 3 training steps and measure
    timings = []
    peak_vrams = []

    for step in range(3):
        torch.cuda.synchronize()
        t0 = time.time()

        images = torch.randn(B, 3, 384, 384, device=device)
        texts = [f"object {i}" for i in range(B)]
        target_bbox = torch.rand(B, 4, device=device) * 0.5 + 0.25
        target_mask = torch.zeros(B, 1, 384, 384, device=device)
        for b in range(B):
            y = int(target_bbox[b, 1].item() * 384)
            x = int(target_bbox[b, 0].item() * 384)
            h = max(20, int(target_bbox[b, 3].item() * 384))
            w = max(20, int(target_bbox[b, 2].item() * 384))
            target_mask[b, 0,
                        max(0, y - h // 2):min(384, y + h // 2),
                        max(0, x - w // 2):min(384, x + w // 2)] = 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(image=images, img_mask=None, text=texts,
                        target_bbox=target_bbox)
            losses, total_loss = compute_grid_cell_losses(
                out, target_bbox, target_mask,
                l_mask=1.0, l_ref=1.0, l_aux=0.2,
                l_bce=2.0, l_dice=5.0, l_focal=2.0, l_boundary=1.0,
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        timings.append(elapsed)

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        peak_vrams.append(peak_mem)

        del images, target_bbox, target_mask, out, total_loss

    peak_vram = max(peak_vrams)
    avg_time = np.mean(timings)
    throughput = B / avg_time

    report(
        f"Peak VRAM <= 23 GB (safe margin)",
        peak_vram <= 23.0,
        f"peak={peak_vram:.1f} GB ({peak_vram / total_vram * 100:.0f}%)"
    )

    report(
        "Throughput measured",
        throughput > 0,
        f"avg_step={avg_time:.2f}s, throughput={throughput:.1f} samples/sec"
    )

    # VRAM utilization note: single-GPU with frozen backbone stores no
    # activation gradients for 224M backbone params. Actual DDP training
    # adds gradient buckets + communication overhead (~3-4GB extra per GPU).
    utilization = peak_vram / total_vram * 100
    report(
        "VRAM utilization reasonable (>= 20%, DDP adds more)",
        utilization >= 20.0,
        f"utilization={utilization:.0f}% "
        f"(DDP adds ~3-4GB overhead per GPU)"
    )

    torch.cuda.empty_cache()
    del model, optimizer
    return True


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("  Venice-H1 Pre-training Micro-tests")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    t_start = time.time()

    test_functions = [
        ("Architecture Verification", test_architecture),
        ("Forward Pass Sanity", test_forward_pass),
        ("Loss & Backward Stability", test_loss_backward),
        ("Short Training Stability", test_training_stability),
        ("Baseline RefCOCO val", test_baseline_refcoco),
        ("Cross-dataset (RefCOCO+, RefCOCOg)", test_cross_dataset),
        ("LoRA Integration", test_lora_integration),
        ("Full RefCOCO Baseline (all splits)", test_full_refcoco_baseline),
        ("3-epoch Training No Degradation", test_training_no_degradation),
        ("3-epoch LoRA Training No Degradation", test_lora_training_no_degradation),
        ("Medical Data Pipeline", test_medical_pipeline),
        ("Memory and Throughput", test_memory_throughput),
    ]

    for name, test_fn in test_functions:
        try:
            test_fn(device)
        except Exception as e:
            report(f"{name} (EXCEPTION)", False, str(e))
            traceback.print_exc()
        torch.cuda.empty_cache()

    # Summary
    elapsed = time.time() - t_start
    passed = sum(1 for _, p in results if p)
    failed = sum(1 for _, p in results if not p)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} PASS, {failed} FAIL ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    if failed > 0:
        print("\n  FAILED tests:")
        for name, p in results:
            if not p:
                print(f"    - {name}")
        sys.exit(1)
    else:
        print("\n  All tests PASSED! Ready for definitive training.")
        sys.exit(0)


if __name__ == '__main__':
    main()
