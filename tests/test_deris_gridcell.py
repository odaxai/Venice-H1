#!/usr/bin/env python3
"""
Venice-H1 DeRIS Micro-tests: Can Grid Cells Beat SOTA?
========================================================
Validates GridCellDeRIS architecture, stability, baseline preservation,
and actual improvement BEFORE launching the definitive training run.

Target: DeRIS-L baseline 85.72% mIoU → 86%+ with grid cells = NEW SOTA

Tests:
  1. Architecture verification (frozen/trainable params correct)
  2. Forward pass sanity (outputs finite, shapes correct)
  3. Loss & backward stability (gradients flow only to grid cells)
  4. Short training stability (5 steps, no NaN)
  5. Baseline preservation (DeRIS mIoU ≈ 85.72% with grid cells at init)
  6. Quick training improvement (200 steps, mIoU should improve)
  7. Ablation: seg-only baseline (no grid cells)
  8. Gate value evolution (alpha should move from 0)
  9. Memory & throughput (fits in GPU, reasonable speed)
 10. Grid cell adaptation sweep (try different hyperparams)

Run:
    cd Venice-H1
    python tests/test_deris_gridcell.py

    # Or run specific test:
    python tests/test_deris_gridcell.py --test 5

Expected: All tests PASS in ~10-15 minutes on a single GPU.
"""

import os
import sys
import gc
import copy
import time
import argparse
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# PATH SETUP (same as train_deris.py)
# =====================================================================
VENICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIS_ROOT = os.path.join(os.path.dirname(VENICE_ROOT), 'DeRIS', 'DeRIS-main')
sys.path.insert(0, VENICE_ROOT)
sys.path.insert(0, DERIS_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Change to DERIS_ROOT so relative data paths in config work
os.chdir(DERIS_ROOT)

from mmcv import Config

from deris.datasets import build_dataset as deris_build_dataset
from deris.datasets import build_dataloader as deris_build_dataloader
from deris.models import build_model as deris_build_model
from deris.apis.test import accuracy
from deris.utils import reduce_mean
from deris.datasets import extract_data

from gridcell_models.grid_cell_deris import GridCellDeRIS
from gridcell_models.paper_logger import DERIS_L_BASELINE

# =====================================================================
# DEFAULTS
# =====================================================================
DERIS_CKPT = './checkpoints/deris_l.pth'
DERIS_CONFIG = os.path.join(DERIS_ROOT, 'configs', 'refcoco',
                            'DERIS-L-refcoco.py')

# SOTA targets (from DERIS paper Table 1)
SOTA_MIOU = DERIS_L_BASELINE['val_refcoco_unc']['miou']  # 85.72


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


def load_cfg():
    """Load DeRIS config with safe defaults."""
    cfg = Config.fromfile(DERIS_CONFIG)
    cfg.seed = 42
    cfg.deterministic = True
    cfg.distributed = False
    cfg.launcher = 'none'
    cfg.rank = 0
    cfg.world_size = 1
    cfg.data.samples_per_gpu = 4
    cfg.data.workers_per_gpu = 2
    return cfg


def build_deris_model(cfg):
    """Build and load DeRIS-L model."""
    cfg.model.mask_save_target_dir = None
    cfg.model.visualize_params = {
        "enable": False, "row_columns": (2, 5),
        "train_interval": 9999, "val_interval": 9999,
    }

    # DeRIS build_model takes only cfg (no word_emb/num_token — uses BEiT-3)
    deris_model = deris_build_model(cfg.model)
    deris_model = deris_model.cuda()

    # Load checkpoint
    ckpt = torch.load(DERIS_CKPT,
                      map_location=lambda storage, loc: storage.cuda(),
                      weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    if list(state.keys())[0].startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    deris_model.load_state_dict(state, strict=False)

    return deris_model, None


def wrap_with_gridcells(deris_model, embed_dim=256, gc_heads=8,
                        gc_dropout=0.05, ablation_seg_only=False,
                        unfreeze_layers=0):
    """Wrap DeRIS model with GridCellDeRIS and move to CUDA."""
    grid_cfg = {'num_heads': gc_heads, 'dropout': gc_dropout}
    model = GridCellDeRIS(
        deris_model=deris_model,
        embed_dim=embed_dim,
        ablation_seg_only=ablation_seg_only,
        unfreeze_layers=unfreeze_layers,
        grid_cfg=grid_cfg,
    )
    model = model.cuda()
    return model


def build_eval_loader(cfg, split="val_refcoco_unc"):
    """Build a single eval dataloader."""
    ds_cfg = getattr(cfg.data, split)
    ds = deris_build_dataset(ds_cfg)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.data.samples_per_gpu = 8
    return deris_build_dataloader(eval_cfg, ds)


def build_train_loader(cfg):
    """Build training dataloader."""
    ds = deris_build_dataset(cfg.data.train)
    return deris_build_dataloader(cfg, ds), ds


def extract_gt(inputs):
    """Extract GT from mmcv-style batch."""
    gt_bbox, gt_mask, gt_mask_parts = None, None, None
    if "gt_bbox" in inputs:
        if isinstance(inputs["gt_bbox"], torch.Tensor):
            inputs["gt_bbox"] = [inputs["gt_bbox"][i]
                                 for i in range(inputs["gt_bbox"].shape[0])]
            gt_bbox = copy.deepcopy(inputs["gt_bbox"])
        else:
            gt_bbox = copy.deepcopy(inputs["gt_bbox"].data[0])
    if "gt_mask_rle" in inputs:
        gt_mask = inputs.pop("gt_mask_rle").data[0]
    if "gt_mask_parts_rle" in inputs:
        gt_mask_parts = inputs.pop("gt_mask_parts_rle").data[0]
    if "is_crowd" in inputs:
        inputs.pop("is_crowd")
    return inputs, gt_bbox, gt_mask, gt_mask_parts


@torch.no_grad()
def quick_eval(model, loader, max_batches=20):
    """Fast eval returning mIoU, oIoU, det_acc.

    DeRIS accuracy() returns 5 values:
      det_acc, mask_miou, mask_oiou, mask_acc_at_thrs, det_acc_at_thrs
    It computes miou/oiou internally, so we just accumulate per-batch.
    """
    model.eval()
    device = next(model.parameters()).device
    det_acc_list, miou_list, oiou_list = [], [], []

    for batch_idx, inputs in enumerate(loader):
        if batch_idx >= max_batches:
            break
        gt_bbox, gt_mask, is_crowd = None, None, None
        if "gt_bbox" in inputs:
            if isinstance(inputs["gt_bbox"], torch.Tensor):
                inputs["gt_bbox"] = [inputs["gt_bbox"][i]
                                     for i in range(inputs["gt_bbox"].shape[0])]
                gt_bbox = copy.deepcopy(inputs["gt_bbox"])
            else:
                gt_bbox = copy.deepcopy(inputs["gt_bbox"].data[0])
        if "gt_mask_rle" in inputs:
            gt_mask = inputs.pop("gt_mask_rle").data[0]
        if "is_crowd" in inputs:
            is_crowd = inputs.pop("is_crowd").data[0]

        inputs = extract_data(inputs)
        preds = model(**inputs, return_loss=False, gt_mask_rle=gt_mask,
                       rescale=False, with_bbox=True, with_mask=True)

        # DeRIS accuracy returns: det_acc, mask_miou, mask_oiou, mask_acc, det_accs
        det_acc, mask_miou, mask_oiou, _, _ = accuracy(
            preds.get("pred_bboxes"), gt_bbox,
            preds.get("pred_masks"), gt_mask,
            is_crowd=is_crowd, device=device)
        det_acc_list.append(det_acc.item() if torch.is_tensor(det_acc) else det_acc)
        miou_list.append(mask_miou if isinstance(mask_miou, (int, float)) else mask_miou.item())
        oiou_list.append(mask_oiou if isinstance(mask_oiou, (int, float)) else mask_oiou.item())

    miou = np.mean(miou_list) if miou_list else 0.0
    oiou = np.mean(oiou_list) if oiou_list else 0.0
    det_acc = np.mean(det_acc_list) if det_acc_list else 0.0
    return miou, oiou, det_acc


# =====================================================================
# TEST 1: Architecture Verification
# =====================================================================
def test_architecture(device):
    print("\n--- Test 1: Architecture Verification ---")
    cfg = load_cfg()
    deris_model, _ = build_deris_model(cfg)
    model = wrap_with_gridcells(deris_model)

    # DeRIS fully frozen
    deris_frozen = sum(p.numel() for p in model.deris.parameters()
                       if not p.requires_grad)
    deris_trainable = sum(p.numel() for p in model.deris.parameters()
                          if p.requires_grad)
    report("DeRIS fully frozen", deris_trainable == 0,
           f"frozen={deris_frozen:,}, trainable={deris_trainable:,}")

    # Grid cells trainable
    grid_p = sum(p.numel() for p in model.multi_grid.parameters())
    grid_trainable = sum(p.numel() for p in model.multi_grid.parameters()
                         if p.requires_grad)
    report("Grid cells all trainable", grid_p == grid_trainable,
           f"total={grid_p:,}, trainable={grid_trainable:,}")

    # Grid cells are small relative to DeRIS
    ratio = grid_p / deris_frozen * 100
    report("Grid cells < 5% of DeRIS params", ratio < 5.0,
           f"ratio={ratio:.2f}%")

    # Gate init at 0 (tanh(0) = 0 → no contribution at init)
    gate = model.get_gate_value()
    report("Gate init ≈ 0.0 (baseline preserved)", abs(gate) < 0.01,
           f"gate={gate:.4f}")

    # Correct embed_dim
    report("embed_dim=256 (DeRIS hidden_channels)",
           model._embed_dim == 256,
           f"embed_dim={model._embed_dim}")

    del model, deris_model
    torch.cuda.empty_cache()


# =====================================================================
# TEST 2: Forward Pass Sanity
# =====================================================================
def test_forward_pass(device):
    print("\n--- Test 2: Forward Pass Sanity (real data) ---")
    cfg = load_cfg()
    deris_model, _ = build_deris_model(cfg)
    model = wrap_with_gridcells(deris_model)
    model.eval()

    loader = build_eval_loader(cfg, "val_refcoco_unc")

    # Get one batch
    batch = next(iter(loader))
    inputs, gt_bbox, gt_mask, gt_mask_parts = extract_gt(batch)
    inputs = extract_data(inputs)

    # Forward test (inference)
    with torch.no_grad():
        preds = model(**inputs, return_loss=False, gt_mask_rle=gt_mask,
                       rescale=False, with_bbox=True, with_mask=True)

    report("forward_test returns dict", isinstance(preds, dict))
    report("pred_bboxes present", "pred_bboxes" in preds)
    report("pred_masks present", "pred_masks" in preds)

    # Forward train
    model.train()
    model.deris.understanding_branch.eval()
    model.deris.perception_branch.eval()

    # Reload batch (popped gt_mask)
    batch2 = next(iter(loader))
    inputs2, gt_bbox2, gt_mask2, gt_mask_parts2 = extract_gt(batch2)
    inputs2 = extract_data(inputs2)

    # No autocast — DeRIS head uses BCE which requires fp32
    losses, train_preds = model(
        **inputs2, gt_mask_rle=gt_mask2,
        gt_mask_parts_rle=gt_mask_parts2,
        epoch=0, rescale=False)

    report("forward_train returns losses dict", isinstance(losses, dict))
    report("loss_mask in losses", "loss_mask" in losses,
           f"keys={list(losses.keys())}")

    loss_mask = losses.get("loss_mask", torch.tensor(0.0))
    report("loss_mask is finite", torch.isfinite(loss_mask).item(),
           f"loss_mask={loss_mask.item():.4f}")

    del model, deris_model, loader
    torch.cuda.empty_cache()


# =====================================================================
# TEST 3: Loss & Backward Stability
# =====================================================================
def test_loss_backward(device):
    print("\n--- Test 3: Loss & Backward Stability ---")
    cfg = load_cfg()
    deris_model, _ = build_deris_model(cfg)
    model = wrap_with_gridcells(deris_model)
    model.train()
    model.deris.understanding_branch.eval()
    model.deris.perception_branch.eval()

    loader = build_eval_loader(cfg, "val_refcoco_unc")
    batch = next(iter(loader))
    inputs, gt_bbox, gt_mask, gt_mask_parts = extract_gt(batch)
    inputs = extract_data(inputs)

    losses, _ = model(**inputs, gt_mask_rle=gt_mask,
                      gt_mask_parts_rle=gt_mask_parts,
                      epoch=0, rescale=False)

    loss_mask = losses["loss_mask"]
    report("loss_mask has grad_fn", loss_mask.grad_fn is not None,
           f"requires_grad={loss_mask.requires_grad}")

    # Backward
    loss_mask.backward()

    # Grid cells have gradients
    gc_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.multi_grid.parameters() if p.requires_grad
    )
    report("Grid cell params have gradients", gc_has_grad)

    # DeRIS has NO gradients (fully frozen)
    deris_no_grad = all(
        p.grad is None or p.grad.abs().sum() == 0
        for p in model.deris.parameters()
    )
    report("DeRIS params have NO gradients", deris_no_grad)

    # Other losses should not have grad
    loss_class = losses.get("loss_class", None)
    if loss_class is not None:
        has_grad = loss_class.requires_grad
        report("loss_class is detached (no grad to grid cells)",
               not has_grad or loss_class.grad_fn is None,
               f"requires_grad={has_grad}")

    del model, deris_model
    torch.cuda.empty_cache()


# =====================================================================
# TEST 4: Short Training Stability (5 steps)
# =====================================================================
def test_training_stability(device):
    print("\n--- Test 4: Short Training Stability (5 steps) ---")
    cfg = load_cfg()
    deris_model, _ = build_deris_model(cfg)
    model = wrap_with_gridcells(deris_model)

    # Snapshot DeRIS params
    deris_snapshot = {n: p.clone() for n, p in model.deris.named_parameters()}
    gate_init = model.get_gate_value()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.05)

    train_loader, _ = build_train_loader(cfg)
    loader_iter = iter(train_loader)

    losses_history = []
    for step in range(5):
        model.train()
        model.deris.understanding_branch.eval()
        model.deris.perception_branch.eval()

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        inputs, gt_bbox, gt_mask, gt_mask_parts = extract_gt(batch)
        inputs = extract_data(inputs)

        optimizer.zero_grad(set_to_none=True)
        # DeRIS head uses binary_cross_entropy which is unsafe with
        # autocast fp16 — run without autocast (DeRIS default: fp16=False)
        losses, _ = model(**inputs, gt_mask_rle=gt_mask,
                          gt_mask_parts_rle=gt_mask_parts,
                          epoch=0, rescale=False)

        loss = losses["loss_mask"]
        loss_val = loss.item()
        losses_history.append(loss_val)

        if not np.isfinite(loss_val):
            report(f"Step {step}: loss finite", False, f"loss={loss_val}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.15)
        optimizer.step()

    all_finite = all(np.isfinite(l) for l in losses_history)
    report("All 5 steps: loss finite", all_finite,
           f"losses={[f'{l:.3f}' for l in losses_history]}")

    # Gate changed
    gate_after = model.get_gate_value()
    report("Gate value changed (gradient flowing)",
           abs(gate_after - gate_init) > 1e-7,
           f"before={gate_init:.6f} after={gate_after:.6f}")

    # DeRIS unchanged
    deris_unchanged = all(
        torch.equal(p, deris_snapshot[n])
        for n, p in model.deris.named_parameters()
    )
    report("DeRIS params unchanged after training", deris_unchanged)

    del model, deris_model, optimizer
    torch.cuda.empty_cache()


# =====================================================================
# TEST 5: Baseline Preservation (mIoU at init ≈ DeRIS-L SOTA)
# =====================================================================
def test_baseline_preservation(device):
    print("\n--- Test 5: Baseline Preservation (mIoU ≈ SOTA at init) ---")
    cfg = load_cfg()
    deris_model, _ = build_deris_model(cfg)
    model = wrap_with_gridcells(deris_model)
    model.eval()

    loader = build_eval_loader(cfg, "val_refcoco_unc")

    # Eval with grid cells at init (gate ≈ 0 → baseline preserved)
    miou, oiou, det_acc = quick_eval(model, loader, max_batches=30)

    print(f"    mIoU={miou:.2f}%  oIoU={oiou:.2f}%  DetAcc={det_acc:.2f}%")
    print(f"    SOTA={SOTA_MIOU:.2f}%")

    # At init, grid cells contribute nothing (gate=0), so mIoU ≈ baseline
    # Allow 1% margin for subset evaluation noise
    report("mIoU at init ≈ SOTA (within 1.5%)",
           abs(miou - SOTA_MIOU) < 1.5,
           f"mIoU={miou:.2f}%, SOTA={SOTA_MIOU}%, "
           f"delta={miou - SOTA_MIOU:+.2f}%")

    report("DetAcc > 90% (sane baseline)", det_acc > 90.0,
           f"det_acc={det_acc:.2f}%")

    del model, deris_model, loader
    torch.cuda.empty_cache()
    return miou


# =====================================================================
# TEST 6: Quick Training Improvement (200 steps)
# =====================================================================
def test_quick_training(device):
    print("\n--- Test 6: Quick Training Improvement (200 steps) ---")
    cfg = load_cfg()
    deris_model, _ = build_deris_model(cfg)
    model = wrap_with_gridcells(deris_model)

    eval_loader = build_eval_loader(cfg, "val_refcoco_unc")
    train_loader, _ = build_train_loader(cfg)

    # BEFORE
    miou_before, oiou_before, det_before = quick_eval(
        model, eval_loader, max_batches=20)
    print(f"    BEFORE: mIoU={miou_before:.2f}%  "
          f"oIoU={oiou_before:.2f}%  DetAcc={det_before:.2f}%")

    # Train 200 steps
    param_groups = model.get_param_groups(lr_backbone=1e-5, lr_grid=3e-4)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    model.train()
    model.deris.understanding_branch.eval()
    model.deris.perception_branch.eval()

    loader_iter = iter(train_loader)
    t0 = time.time()
    loss_first, loss_last = [], []

    for step in range(200):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        inputs, gt_bbox, gt_mask, gt_mask_parts = extract_gt(batch)
        inputs = extract_data(inputs)

        optimizer.zero_grad(set_to_none=True)
        losses, _ = model(**inputs, gt_mask_rle=gt_mask,
                          gt_mask_parts_rle=gt_mask_parts,
                          epoch=0, rescale=False)

        loss = losses["loss_mask"]
        if not np.isfinite(loss.item()):
            continue

        if step < 10:
            loss_first.append(loss.item())
        if step >= 190:
            loss_last.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 0.15)
        optimizer.step()

    elapsed = time.time() - t0
    gate_s = model.get_gate_value()

    # AFTER
    miou_after, oiou_after, det_after = quick_eval(
        model, eval_loader, max_batches=20)
    print(f"    AFTER:  mIoU={miou_after:.2f}%  "
          f"oIoU={oiou_after:.2f}%  DetAcc={det_after:.2f}%")

    d_miou = miou_after - miou_before
    avg_first = np.mean(loss_first) if loss_first else 0
    avg_last = np.mean(loss_last) if loss_last else 0

    print(f"    ΔmIoU: {d_miou:+.2f}%  Gate: {gate_s:.3f}  "
          f"Time: {elapsed:.0f}s")
    print(f"    Loss: {avg_first:.4f} → {avg_last:.4f}")

    # Loss should decrease
    report("Loss decreased during training",
           avg_last < avg_first * 0.95,
           f"first={avg_first:.4f} → last={avg_last:.4f}")

    # Gate should move away from 0
    report("Gate moved from 0 (grid cells contributing)",
           abs(gate_s) > 0.01,
           f"gate={gate_s:.4f}")

    # mIoU should not degrade significantly
    report("mIoU not degraded (drop < 1%)",
           d_miou > -1.0,
           f"delta={d_miou:+.2f}%")

    # mIoU should improve (this is the key test!)
    report("mIoU IMPROVED (grid cells helping!)",
           d_miou > 0,
           f"delta={d_miou:+.2f}% {'*** BEAT SOTA! ***' if miou_after > SOTA_MIOU else ''}")

    # DetAcc should be stable (frozen backbone)
    det_drop = det_before - det_after
    report("DetAcc stable (drop < 2%)",
           det_drop < 2.0,
           f"before={det_before:.2f}% after={det_after:.2f}%")

    del model, deris_model, optimizer
    torch.cuda.empty_cache()
    return d_miou, miou_after


# =====================================================================
# TEST 7: Ablation — seg-only vs grid cells
# =====================================================================
def test_ablation_seg_only(device):
    print("\n--- Test 7: Ablation — seg-only vs grid cells ---")
    cfg = load_cfg()
    eval_loader = build_eval_loader(cfg, "val_refcoco_unc")

    # 7a: Seg-only (no grid cells)
    deris_model, _ = build_deris_model(cfg)
    model_seg = wrap_with_gridcells(deris_model, ablation_seg_only=True)
    model_seg.eval()

    miou_seg, oiou_seg, det_seg = quick_eval(
        model_seg, eval_loader, max_batches=15)
    print(f"    Seg-only:    mIoU={miou_seg:.2f}%  oIoU={oiou_seg:.2f}%")

    report("Seg-only: no grid cells → pure DeRIS",
           model_seg.multi_grid is None)

    del model_seg, deris_model
    torch.cuda.empty_cache()

    # 7b: With grid cells (at init → should be same as seg-only)
    deris_model2, _ = build_deris_model(cfg)
    model_gc = wrap_with_gridcells(deris_model2)
    model_gc.eval()

    miou_gc, oiou_gc, det_gc = quick_eval(
        model_gc, eval_loader, max_batches=15)
    print(f"    Grid cells:  mIoU={miou_gc:.2f}%  oIoU={oiou_gc:.2f}%")

    # At init (gate=0), grid cells should give same result as seg-only
    diff = abs(miou_gc - miou_seg)
    report("At init: grid cells ≈ seg-only (diff < 0.5%)",
           diff < 0.5,
           f"seg_only={miou_seg:.2f}%, grid={miou_gc:.2f}%, "
           f"diff={diff:.2f}%")

    del model_gc, deris_model2, eval_loader
    torch.cuda.empty_cache()


# =====================================================================
# TEST 8: Gate Value Evolution
# =====================================================================
def test_gate_evolution(device):
    print("\n--- Test 8: Gate Value Evolution ---")
    cfg = load_cfg()
    deris_model, _ = build_deris_model(cfg)
    model = wrap_with_gridcells(deris_model)

    train_loader, _ = build_train_loader(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.05)
    model.train()
    model.deris.understanding_branch.eval()
    model.deris.perception_branch.eval()

    loader_iter = iter(train_loader)
    gate_values = [model.get_gate_value()]

    for step in range(50):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        inputs, gt_bbox, gt_mask, gt_mask_parts = extract_gt(batch)
        inputs = extract_data(inputs)

        optimizer.zero_grad(set_to_none=True)
        losses, _ = model(**inputs, gt_mask_rle=gt_mask,
                          gt_mask_parts_rle=gt_mask_parts,
                          epoch=0, rescale=False)

        loss = losses["loss_mask"]
        if not np.isfinite(loss.item()):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.15)
        optimizer.step()

        if (step + 1) % 10 == 0:
            gate_values.append(model.get_gate_value())

    print(f"    Gate values: {[f'{g:.4f}' for g in gate_values]}")

    # Gate should evolve (not stuck at 0)
    report("Gate evolved from init",
           abs(gate_values[-1] - gate_values[0]) > 0.001,
           f"init={gate_values[0]:.4f} → final={gate_values[-1]:.4f}")

    # Scale weights should have moved
    w = F.softmax(model.multi_grid.scale_weights, dim=0)
    print(f"    Scale weights: 4x4={w[0]:.3f}, "
          f"8x8={w[1]:.3f}, 16x16={w[2]:.3f}")
    report("Scale weights not uniform (learning multi-scale)",
           not (abs(w[0] - w[1]) < 0.01 and abs(w[1] - w[2]) < 0.01),
           f"weights={w.detach().cpu().tolist()}")

    del model, deris_model, optimizer
    torch.cuda.empty_cache()


# =====================================================================
# TEST 9: Memory & Throughput
# =====================================================================
def test_memory_throughput(device):
    print("\n--- Test 9: Memory & Throughput ---")
    cfg = load_cfg()
    deris_model, _ = build_deris_model(cfg)
    model = wrap_with_gridcells(deris_model)

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"    Total VRAM: {total_vram:.1f} GB")

    torch.cuda.reset_peak_memory_stats()

    train_loader, _ = build_train_loader(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-4)
    model.train()
    model.deris.understanding_branch.eval()
    model.deris.perception_branch.eval()

    loader_iter = iter(train_loader)
    timings = []

    for step in range(3):
        batch = next(loader_iter)
        inputs, gt_bbox, gt_mask, gt_mask_parts = extract_gt(batch)
        inputs = extract_data(inputs)

        torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        losses, _ = model(**inputs, gt_mask_rle=gt_mask,
                          gt_mask_parts_rle=gt_mask_parts,
                          epoch=0, rescale=False)
        loss = losses["loss_mask"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.15)
        optimizer.step()

        torch.cuda.synchronize()
        timings.append(time.time() - t0)

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    avg_time = np.mean(timings)
    bs = cfg.data.samples_per_gpu
    throughput = bs / avg_time

    report(f"Peak VRAM <= {total_vram:.0f} GB",
           peak_vram <= total_vram,
           f"peak={peak_vram:.1f} GB ({peak_vram / total_vram * 100:.0f}%)")

    report("Step time reasonable (< 30s)",
           avg_time < 30.0,
           f"avg_step={avg_time:.2f}s, throughput={throughput:.1f} "
           f"samples/s (bs={bs})")

    del model, deris_model, optimizer
    torch.cuda.empty_cache()


# =====================================================================
# TEST 10: Grid Cell Adaptation Sweep
# =====================================================================
def test_adaptation_sweep(device):
    """Try different hyperparameters to find best grid cell config."""
    print("\n--- Test 10: Grid Cell Adaptation Sweep ---")

    configs = [
        {"name": "Default (heads=8, drop=0.05)",
         "gc_heads": 8, "gc_dropout": 0.05, "lr": 3e-4},
        {"name": "More heads (heads=16)",
         "gc_heads": 16, "gc_dropout": 0.05, "lr": 3e-4},
        {"name": "Higher LR (lr=5e-4)",
         "gc_heads": 8, "gc_dropout": 0.05, "lr": 5e-4},
        {"name": "Lower LR (lr=1e-4)",
         "gc_heads": 8, "gc_dropout": 0.05, "lr": 1e-4},
        {"name": "Less dropout (drop=0.01)",
         "gc_heads": 8, "gc_dropout": 0.01, "lr": 3e-4},
    ]

    cfg = load_cfg()
    eval_loader = build_eval_loader(cfg, "val_refcoco_unc")
    train_loader, _ = build_train_loader(cfg)

    sweep_results = []
    N_STEPS = 100

    for ci, config in enumerate(configs):
        print(f"\n    [{ci+1}/{len(configs)}] {config['name']}")

        deris_model, _ = build_deris_model(cfg)
        model = wrap_with_gridcells(
            deris_model,
            gc_heads=config["gc_heads"],
            gc_dropout=config["gc_dropout"])

        # Baseline
        miou_before, _, _ = quick_eval(model, eval_loader, max_batches=10)

        # Train N steps
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=config["lr"], weight_decay=0.05)
        # No scaler — DeRIS uses fp32

        model.train()
        model.deris.understanding_branch.eval()
        model.deris.perception_branch.eval()

        loader_iter = iter(train_loader)
        loss_vals = []

        for step in range(N_STEPS):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            inputs, gt_bbox, gt_mask, gt_mask_parts = extract_gt(batch)
            inputs = extract_data(inputs)

            optimizer.zero_grad(set_to_none=True)
            losses, _ = model(**inputs, gt_mask_rle=gt_mask,
                              gt_mask_parts_rle=gt_mask_parts,
                              epoch=0, rescale=False)
            loss = losses["loss_mask"]
            if np.isfinite(loss.item()):
                loss_vals.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 0.15)
                optimizer.step()

        # After
        miou_after, _, _ = quick_eval(model, eval_loader, max_batches=10)
        gate = model.get_gate_value()
        d_miou = miou_after - miou_before
        loss_decrease = (np.mean(loss_vals[:5]) - np.mean(loss_vals[-5:])) \
            if len(loss_vals) >= 10 else 0

        result = {
            'name': config['name'],
            'miou_before': miou_before,
            'miou_after': miou_after,
            'd_miou': d_miou,
            'gate': gate,
            'loss_decrease': loss_decrease,
        }
        sweep_results.append(result)

        print(f"      mIoU: {miou_before:.2f}% → {miou_after:.2f}% "
              f"(Δ={d_miou:+.2f}%)  gate={gate:.4f}  "
              f"loss_Δ={loss_decrease:+.4f}")

        del model, deris_model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    print(f"\n    {'Config':<35} {'ΔmIoU':>8} {'Gate':>8} {'LossΔ':>8}")
    print(f"    {'-'*60}")
    best_config = None
    best_d = -999
    for r in sweep_results:
        marker = " ***" if r['d_miou'] > 0 else ""
        print(f"    {r['name']:<35} {r['d_miou']:>+7.2f}% "
              f"{r['gate']:>7.4f} {r['loss_decrease']:>+7.4f}{marker}")
        if r['d_miou'] > best_d:
            best_d = r['d_miou']
            best_config = r['name']

    report("At least one config improves mIoU",
           best_d > 0,
           f"best={best_config} (Δ={best_d:+.2f}%)")

    report("Best config beats SOTA",
           any(r['miou_after'] > SOTA_MIOU for r in sweep_results),
           f"SOTA={SOTA_MIOU}%, best={max(r['miou_after'] for r in sweep_results):.2f}%")

    del eval_loader, train_loader
    torch.cuda.empty_cache()


# =====================================================================
# MAIN
# =====================================================================
def main():
    global DERIS_CKPT

    parser = argparse.ArgumentParser("Venice-H1 DeRIS Micro-tests")
    parser.add_argument('--test', type=int, default=0,
                        help='Run specific test (0=all)')
    parser.add_argument('--deris_checkpoint', default=DERIS_CKPT, type=str)
    args = parser.parse_args()

    DERIS_CKPT = args.deris_checkpoint

    print("=" * 60)
    print("  Venice-H1 DeRIS Micro-tests: Can We Beat SOTA?")
    print(f"  Target: DeRIS-L {SOTA_MIOU}% mIoU → BEAT IT!")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    t_start = time.time()

    test_functions = [
        (1, "Architecture Verification", test_architecture),
        (2, "Forward Pass Sanity", test_forward_pass),
        (3, "Loss & Backward Stability", test_loss_backward),
        (4, "Short Training Stability", test_training_stability),
        (5, "Baseline Preservation", test_baseline_preservation),
        (6, "Quick Training Improvement", test_quick_training),
        (7, "Ablation: seg-only vs grid cells", test_ablation_seg_only),
        (8, "Gate Value Evolution", test_gate_evolution),
        (9, "Memory & Throughput", test_memory_throughput),
        (10, "Grid Cell Adaptation Sweep", test_adaptation_sweep),
    ]

    for test_id, name, test_fn in test_functions:
        if args.test > 0 and args.test != test_id:
            continue
        try:
            test_fn(device)
        except Exception as e:
            report(f"{name} (EXCEPTION)", False, str(e))
            traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()

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
        print("\n  Fix issues before launching definitive training.")
        sys.exit(1)
    else:
        print("\n  All tests PASSED!")
        print(f"  Grid cells are ready to beat DeRIS-L SOTA ({SOTA_MIOU}%).")
        print(f"\n  Next step: run full training:")
        print(f"    torchrun --nproc_per_node=2 scripts/train_deris.py train \\")
        print(f"        --deris_checkpoint {DERIS_CKPT} \\")
        print(f"        --epochs 20 --lr 3e-4 --batch_size 8")
        sys.exit(0)


if __name__ == '__main__':
    main()
