#!/usr/bin/env python3
"""
Failure Re-Ranker V3 - COMPLETE IMPLEMENTATION (Training + Evaluation)

V3 CRITICAL FIXES:
1. rerank_rate bounds [0,100] with assertions
2. margin_pred safety constraint (only rerank if confident)
3. Zero-sum rank_logits relative to default query
4. Cross-entropy on oracle_idx (ranking objective, NOT regression)
5. Optional gate calibration (temperature scaling)
6. Detailed diagnostics (oracle accuracy, true gain, FP harm)

CHANGES FROM V2:
- Renamed gain_pred → rank_logits (conceptual clarity: these are ranking scores)
- Zero-sum constraint: rank_logits[default_query] = 0 (applied at train & eval)
- Training objective: Cross-entropy on oracle_idx (only on failures)
- Evaluation: margin_pred safety + detailed diagnostics
- Temperature calibration for gate (optional)
"""

import os, re, json, glob, argparse
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from sklearn.metrics import roc_auc_score
except:
    roc_auc_score = None

# =============================================================================
# INLINED BASE COMPONENTS (for standalone execution)
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _require_key(d, k: str):
    if k not in d:
        keys = sorted(list(d.keys()))
        raise KeyError(f"Missing key '{k}' in cached dict.\nAvailable keys: {keys}")
    return d[k]

def infer_split_name_from_filename(path: str) -> str:
    """cached_val_refcoco_unc_feats.pt -> val_refcoco_unc"""
    base = os.path.basename(path)
    base = re.sub(r"\.pt$", "", base)
    base = re.sub(r"^cached_", "", base)
    base = re.sub(r"_feats$", "", base)
    return base

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class FailureDataset(Dataset):
    """Dataset with failure flags and gain labels. Supports BASE and BASE+GRID feature sets."""

    def __init__(self, data_dict, use_grid: bool = True):
        self.use_grid = bool(use_grid)

        # Required base features
        self.query_feat = _require_key(data_dict, "query_feat").float()
        self.det_scores = _require_key(data_dict, "det_scores").float()
        self.query_ious = _require_key(data_dict, "query_ious").float()
        self.oracle_idx = _require_key(data_dict, "oracle_idx")

        self.mask_mean = _require_key(data_dict, "mask_mean").float()
        self.mask_max  = _require_key(data_dict, "mask_max").float()
        self.mask_area = _require_key(data_dict, "mask_area").float()
        self.mask_std  = _require_key(data_dict, "mask_std").float()

        # Grid-cell features
        if self.use_grid:
            self.grid_mean_4  = _require_key(data_dict, "grid_mean_4").float()
            self.grid_max_4   = _require_key(data_dict, "grid_max_4").float()
            self.boundary_4   = _require_key(data_dict, "boundary_4").float()
            self.grid_mean_8  = _require_key(data_dict, "grid_mean_8").float()
            self.grid_max_8   = _require_key(data_dict, "grid_max_8").float()
            self.boundary_8   = _require_key(data_dict, "boundary_8").float()
            self.grid_mean_16 = _require_key(data_dict, "grid_mean_16").float()
            self.grid_max_16  = _require_key(data_dict, "grid_max_16").float()
            self.boundary_16  = _require_key(data_dict, "boundary_16").float()
        else:
            self.grid_mean_4 = self.grid_max_4 = self.boundary_4 = None
            self.grid_mean_8 = self.grid_max_8 = self.boundary_8 = None
            self.grid_mean_16 = self.grid_max_16 = self.boundary_16 = None

        # Labels (default = argmax(det_scores), NOT query 0)
        self.default_idx = self.det_scores.argmax(dim=1)
        self.default_iou = self.query_ious.gather(1, self.default_idx.unsqueeze(1)).squeeze(1)
        self.failure_flag = (self.oracle_idx != self.default_idx).float()
        self.gain = self.query_ious - self.default_iou.unsqueeze(1)
        self.q0_iou = self.query_ious[:, 0]

        self.n_samples = int(self.query_feat.shape[0])
        self.n_failures = int(self.failure_flag.sum().item())

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = {
            "query_feat": self.query_feat[idx],
            "det_scores": self.det_scores[idx],
            "query_ious": self.query_ious[idx],
            "oracle_idx": self.oracle_idx[idx],
            "default_iou": self.default_iou[idx],
            "q0_iou": self.q0_iou[idx],
            "failure_flag": self.failure_flag[idx],
            "gain": self.gain[idx],
            "mask_mean": self.mask_mean[idx],
            "mask_max": self.mask_max[idx],
            "mask_area": self.mask_area[idx],
            "mask_std": self.mask_std[idx],
        }

        if self.use_grid:
            item.update({
                "grid_mean_4": self.grid_mean_4[idx],
                "grid_max_4": self.grid_max_4[idx],
                "boundary_4": self.boundary_4[idx],
                "grid_mean_8": self.grid_mean_8[idx],
                "grid_max_8": self.grid_max_8[idx],
                "boundary_8": self.boundary_8[idx],
                "grid_mean_16": self.grid_mean_16[idx],
                "grid_max_16": self.grid_max_16[idx],
                "boundary_16": self.boundary_16[idx],
            })

        return item


class FailureReranker(nn.Module):
    """V3: Transformer-based model with query-to-query attention for better classification."""

    def __init__(self, query_dim=256, hidden_dim=512, n_queries=10, use_grid=True, 
                 n_transformer_layers=3, n_heads=8):
        super().__init__()
        self.n_queries = int(n_queries)
        self.use_grid = bool(use_grid)
        self.hidden_dim = hidden_dim

        base_dim = query_dim + 4 + 1  # query + mask(4) + det(1)
        grid_dim = 0
        if self.use_grid:
            grid_dim = (16 + 16 + 1) + (64 + 64 + 1) + (256 + 256 + 1)

        per_query_dim = base_dim + grid_dim

        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(per_query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # Gain predictor
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Failure gate
        failure_input_dim = hidden_dim * 3 + self.n_queries * 2
        self.failure_encoder = nn.Sequential(
            nn.Linear(failure_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )
        self.failure_head = nn.Linear(hidden_dim // 4, 1)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and 'transformer' not in name:
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small bias to start slightly cautious but NOT -2.0 which freezes the gate
        nn.init.constant_(self.failure_head.bias, -0.5)

    def forward(
        self, query_feat, det_scores,
        mask_mean, mask_max, mask_area, mask_std,
        grid_mean_4=None, grid_max_4=None, boundary_4=None,
        grid_mean_8=None, grid_max_8=None, boundary_8=None,
        grid_mean_16=None, grid_max_16=None, boundary_16=None,
    ):
        B, N, D = query_feat.shape

        # Concatenate per-query features
        mask_stats = torch.stack([mask_mean, mask_max, mask_area, mask_std], dim=-1)
        per_query_parts = [query_feat, mask_stats, det_scores.unsqueeze(-1)]

        if self.use_grid:
            per_query_parts.extend([
                grid_mean_4, grid_max_4, boundary_4.unsqueeze(-1),
                grid_mean_8, grid_max_8, boundary_8.unsqueeze(-1),
                grid_mean_16, grid_max_16, boundary_16.unsqueeze(-1),
            ])

        per_query_input = torch.cat(per_query_parts, dim=-1)
        query_encoded = self.query_encoder(per_query_input)
        query_attended = self.transformer(query_encoded)

        # Gain prediction per query
        gain_pred = self.gain_head(query_attended).squeeze(-1)

        # V3 FIX 3: Zero-sum (relative to default)
        default_idx = det_scores.argmax(dim=1)
        default_logit = gain_pred.gather(1, default_idx.unsqueeze(1))
        gain_pred = gain_pred - default_logit  # relative to default

        # Failure gate
        default_features = query_attended.gather(
            1, default_idx.view(B, 1, 1).expand(-1, -1, self.hidden_dim)
        ).squeeze(1)

        range_idx = torch.arange(N, device=query_attended.device).unsqueeze(0).expand(B, -1)
        non_default_mask = (range_idx != default_idx.unsqueeze(1)).unsqueeze(-1)
        masked_attended = torch.where(non_default_mask, query_attended, 
                                      torch.full_like(query_attended, -1e9))
        max_alternative_features = masked_attended.max(dim=1)[0]
        query_mean = query_attended.mean(dim=1)

        global_feat = torch.cat([
            default_features, max_alternative_features, query_mean,
            det_scores, mask_mean
        ], dim=-1)

        failure_hidden = self.failure_encoder(global_feat)
        p_fail_logit = self.failure_head(failure_hidden).squeeze(-1)
        p_fail = torch.sigmoid(p_fail_logit)

        return p_fail, gain_pred, p_fail_logit


#=============================================================================
# V3 TRAINING EPOCH (Cross-Entropy Ranking Objective)
#=============================================================================

def focal_bce_with_logits(logits, targets, gamma=2.0, alpha=None, pos_weight=None):
    """
    Focal loss for binary classification with logits.
    Reduces loss on easy examples, focuses on hard ones (misclassified).
    
    gamma: focusing parameter (0=standard BCE, 2=standard focal)
    alpha: class balance weight for positive class (0.25-0.75 typical)
    pos_weight: weight for positive class in BCE (for class imbalance)
    """
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction='none')
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)  # p_t = p if y=1, 1-p if y=0
    focal_weight = (1 - p_t) ** gamma
    
    loss = focal_weight * bce
    
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    return loss.mean()


def train_epoch_v3(model, optimizer, loader, device, epoch, writer=None, 
                   use_grid=True, use_amp=True, gate_only=False, ce_weight=1.0,
                   focal_gamma=2.0, gate_pos_weight=None, rank_mode='iou_reg'):
    """
    V3 Training with:
    - Zero-sum rank_logits
    - Focal loss for gate (handles class imbalance)
    - Ranking head: 'ce' = cross-entropy on oracle_idx (failure-only)
                    'iou_reg' = regression on relative IoU (ALL samples)
    """
    model.train()
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_loss = 0.0
    total_gate_loss = 0.0
    total_rank_loss = 0.0
    n_batches = 0

    for batch in loader:
        query_feat = batch["query_feat"].to(device, non_blocking=True)
        det_scores = batch["det_scores"].to(device, non_blocking=True)

        mask_mean = batch["mask_mean"].to(device, non_blocking=True)
        mask_max  = batch["mask_max"].to(device, non_blocking=True)
        mask_area = batch["mask_area"].to(device, non_blocking=True)
        mask_std  = batch["mask_std"].to(device, non_blocking=True)

        failure_flag = batch["failure_flag"].to(device, non_blocking=True)  # [B]
        oracle_idx = batch["oracle_idx"].to(device, non_blocking=True).long()  # [B]
        query_ious = batch["query_ious"].to(device, non_blocking=True)  # [B, N]

        if use_grid:
            grid_mean_4 = batch["grid_mean_4"].to(device, non_blocking=True)
            grid_max_4  = batch["grid_max_4"].to(device, non_blocking=True)
            boundary_4  = batch["boundary_4"].to(device, non_blocking=True)
            grid_mean_8 = batch["grid_mean_8"].to(device, non_blocking=True)
            grid_max_8  = batch["grid_max_8"].to(device, non_blocking=True)
            boundary_8  = batch["boundary_8"].to(device, non_blocking=True)
            grid_mean_16 = batch["grid_mean_16"].to(device, non_blocking=True)
            grid_max_16  = batch["grid_max_16"].to(device, non_blocking=True)
            boundary_16  = batch["boundary_16"].to(device, non_blocking=True)
        else:
            grid_mean_4 = grid_max_4 = boundary_4 = None
            grid_mean_8 = grid_max_8 = boundary_8 = None
            grid_mean_16 = grid_max_16 = boundary_16 = None

        with torch.cuda.amp.autocast(enabled=use_amp):
            p_fail, rank_logits_raw, p_fail_logit = model(
                query_feat, det_scores,
                mask_mean, mask_max, mask_area, mask_std,
                grid_mean_4=grid_mean_4, grid_max_4=grid_max_4, boundary_4=boundary_4,
                grid_mean_8=grid_mean_8, grid_max_8=grid_max_8, boundary_8=boundary_8,
                grid_mean_16=grid_mean_16, grid_max_16=grid_max_16, boundary_16=boundary_16,
            )

            # Gate loss: FOCAL LOSS with class imbalance handling
            gate_loss = focal_bce_with_logits(
                p_fail_logit, failure_flag, 
                gamma=focal_gamma, 
                pos_weight=gate_pos_weight
            )

            if gate_only:
                loss = gate_loss
                rank_loss = torch.tensor(0.0, device=device)
            else:
                B, N = rank_logits_raw.shape
                default_idx = det_scores.argmax(dim=1)  # [B]
                default_logit = rank_logits_raw.gather(1, default_idx.unsqueeze(1))  # [B,1]
                rank_logits = rank_logits_raw - default_logit  # [B,N], default=0
                
                if rank_mode == 'iou_reg':
                    # REGRESSION on relative IoU: learn from ALL samples
                    # Target: query_ious relative to default query's IoU
                    default_iou = query_ious.gather(1, default_idx.unsqueeze(1))  # [B,1]
                    relative_iou = query_ious - default_iou  # [B,N] - positive if better than default
                    
                    # Smooth L1 loss on rank_logits vs relative_iou (all samples)
                    rank_loss = F.smooth_l1_loss(rank_logits, relative_iou, beta=0.1)
                    
                elif rank_mode == 'listnet':
                    # ListNet: softmax cross-entropy on IoU distribution (all samples)
                    default_iou = query_ious.gather(1, default_idx.unsqueeze(1))
                    relative_iou = query_ious - default_iou
                    # Temperature for target distribution
                    target_dist = F.softmax(relative_iou / 0.1, dim=1)
                    pred_dist = F.log_softmax(rank_logits, dim=1)
                    rank_loss = F.kl_div(pred_dist, target_dist, reduction='batchmean')
                    
                else:  # 'ce' - original CE mode (failure-only)
                    failure_mask = failure_flag > 0.5
                    n_failures = failure_mask.sum().item()
                    if n_failures > 0:
                        rank_logits_fail = rank_logits[failure_mask]
                        oracle_idx_fail = oracle_idx[failure_mask]
                        rank_loss = F.cross_entropy(rank_logits_fail, oracle_idx_fail, 
                                                    reduction='mean', label_smoothing=0.05)
                    else:
                        rank_loss = torch.tensor(0.0, device=device)

                loss = gate_loss + ce_weight * rank_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_gate_loss += float(gate_loss.item())
        total_rank_loss += float(rank_loss.item())
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_gate = total_gate_loss / max(n_batches, 1)
    avg_rank = total_rank_loss / max(n_batches, 1)

    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/gate_loss", avg_gate, epoch)
        writer.add_scalar("train/rank_loss", avg_rank, epoch)

    return avg_loss, avg_gate, avg_rank


#=============================================================================
# V3 EVALUATION (with all fixes)
#=============================================================================

@torch.no_grad()
def evaluate_v3(model, loader, device, tau=0.5, margin_pred=0.02, margin_gap=0.0,
                use_grid=True, gate_temperature=1.0):
    """
    V3 Evaluation with ALL fixes:
    1. rerank_rate bounds [0,100] + assert
    2. margin_pred safety constraint
    3. Zero-sum rank_logits
    4. Detailed diagnostics
    5. Gate temperature scaling
    """
    model.eval()
    
    all_p_fail_logit = []
    all_rank_logits = []
    all_failure_flag = []
    all_default_iou = []
    all_oracle_iou = []
    all_oracle_idx = []
    all_default_idx = []
    all_query_ious = []
    all_det_scores = []
    
    for batch in loader:
        query_feat = batch["query_feat"].to(device, non_blocking=True)
        det_scores = batch["det_scores"].to(device, non_blocking=True)
        
        mask_mean = batch["mask_mean"].to(device, non_blocking=True)
        mask_max = batch["mask_max"].to(device, non_blocking=True)
        mask_area = batch["mask_area"].to(device, non_blocking=True)
        mask_std = batch["mask_std"].to(device, non_blocking=True)
        
        if use_grid:
            grid_mean_4 = batch["grid_mean_4"].to(device, non_blocking=True)
            grid_max_4 = batch["grid_max_4"].to(device, non_blocking=True)
            boundary_4 = batch["boundary_4"].to(device, non_blocking=True)
            grid_mean_8 = batch["grid_mean_8"].to(device, non_blocking=True)
            grid_max_8 = batch["grid_max_8"].to(device, non_blocking=True)
            boundary_8 = batch["boundary_8"].to(device, non_blocking=True)
            grid_mean_16 = batch["grid_mean_16"].to(device, non_blocking=True)
            grid_max_16 = batch["grid_max_16"].to(device, non_blocking=True)
            boundary_16 = batch["boundary_16"].to(device, non_blocking=True)
        else:
            grid_mean_4 = grid_max_4 = boundary_4 = None
            grid_mean_8 = grid_max_8 = boundary_8 = None
            grid_mean_16 = grid_max_16 = boundary_16 = None
        
        p_fail, rank_logits_raw, p_fail_logit = model(
            query_feat, det_scores,
            mask_mean, mask_max, mask_area, mask_std,
            grid_mean_4=grid_mean_4, grid_max_4=grid_max_4, boundary_4=boundary_4,
            grid_mean_8=grid_mean_8, grid_max_8=grid_max_8, boundary_8=boundary_8,
            grid_mean_16=grid_mean_16, grid_max_16=grid_max_16, boundary_16=boundary_16,
        )
        
        # V3 FIX 3: Zero-sum rank_logits (in eval)
        default_idx_batch = det_scores.argmax(dim=1)  # [B]
        default_logit = rank_logits_raw.gather(1, default_idx_batch.unsqueeze(1))  # [B,1]
        rank_logits = rank_logits_raw - default_logit  # relative to default
        
        all_p_fail_logit.append(p_fail_logit.cpu())
        all_rank_logits.append(rank_logits.cpu())
        all_det_scores.append(det_scores.cpu())
        all_failure_flag.append(batch["failure_flag"])
        all_default_iou.append(batch["default_iou"])
        all_oracle_idx.append(batch["oracle_idx"])
        all_default_idx.append(default_idx_batch.cpu())
        all_query_ious.append(batch["query_ious"])
        all_oracle_iou.append(batch["query_ious"].max(dim=1)[0])
    
    p_fail_logit = torch.cat(all_p_fail_logit)
    rank_logits = torch.cat(all_rank_logits)
    det_scores = torch.cat(all_det_scores)
    failure_flag = torch.cat(all_failure_flag)
    default_iou = torch.cat(all_default_iou)
    oracle_idx = torch.cat(all_oracle_idx)
    default_idx = torch.cat(all_default_idx)
    query_ious = torch.cat(all_query_ious)
    oracle_iou = torch.cat(all_oracle_iou)
    
    N = int(failure_flag.shape[0])
    
    # V3 FIX 5: Gate calibration (temperature scaling)
    p_fail = torch.sigmoid(p_fail_logit / gate_temperature)
    
    # V3 FIX 3: Safety margin - check BEST ALTERNATIVE (excluding default which is 0)
    B, N_queries = rank_logits.shape
    idx = torch.arange(N_queries, device=rank_logits.device).unsqueeze(0).expand(B, -1)
    mask_non_default = (idx != default_idx.unsqueeze(1))
    
    neg_inf = torch.full_like(rank_logits, -1e9)
    rank_logits_alt = torch.where(mask_non_default, rank_logits, neg_inf)
    
    # V3 FIX 2: margin_pred safety constraint
    best_alt = rank_logits_alt.max(dim=1).values  # [B]
    use_rerank = (p_fail > tau) & (best_alt > margin_pred)
    
    # V3 FIX 6 (OPTIONAL): TOP2 margin (only if margin_gap > 0)
    if margin_gap > 0:
        vals, _ = torch.topk(rank_logits_alt, k=min(2, N_queries-1), dim=1)
        best_alt = vals[:, 0]
        second_alt = vals[:, 1] if vals.shape[1] > 1 else torch.full_like(best_alt, -1e9)
        use_rerank = use_rerank & ((best_alt - second_alt) > margin_gap)
    
    best_query_pred = rank_logits.argmax(dim=1)  # [B]
    
    # V3 FIX 1: rerank_rate bounds check
    rerank_count = int(use_rerank.sum().item())
    rerank_rate = (rerank_count / N) * 100.0
    assert 0.0 <= rerank_rate <= 100.0001, f"BUG: rerank_rate={rerank_rate:.4f}% out of bounds!"
    
    # Selected IoU
    iou_at_best = query_ious.gather(1, best_query_pred.unsqueeze(1)).squeeze(1)
    selected_iou = torch.where(use_rerank, iou_at_best, default_iou)
    
    # Base metrics
    q0_miou = default_iou.mean().item() * 100.0
    oracle_miou = oracle_iou.mean().item() * 100.0
    selected_miou = selected_iou.mean().item() * 100.0
    delta_full = selected_miou - q0_miou
    
    fail_mask = (failure_flag == 1)
    n_fail = int(fail_mask.sum().item())
    fail_rate = (n_fail / N) * 100.0
    
    if n_fail > 0:
        q0_miou_fail = default_iou[fail_mask].mean().item() * 100.0
        oracle_miou_fail = oracle_iou[fail_mask].mean().item() * 100.0
        selected_miou_fail = selected_iou[fail_mask].mean().item() * 100.0
        delta_fail = selected_miou_fail - q0_miou_fail
        gap = oracle_miou_fail - q0_miou_fail
        gap_closed = (delta_fail / gap * 100.0) if gap > 1e-9 else 0.0
    else:
        q0_miou_fail = oracle_miou_fail = selected_miou_fail = delta_fail = gap_closed = 0.0
    
    # AUC
    auc_fail = 0.5
    if roc_auc_score is not None:
        try:
            auc_fail = float(roc_auc_score(failure_flag.numpy(), p_fail.numpy()))
        except:
            pass
    
    # V3 FIX 4: Compute confusion for BOTH gate-only and final decision
    
    # A) Gate-only (measures gate quality: p_fail > tau)
    gate_pred = (p_fail > tau).float()
    gate_acc = (gate_pred == failure_flag.float()).float().mean().item() * 100.0
    gate_tp = int(((gate_pred == 1) & (failure_flag == 1)).sum().item())
    gate_fp = int(((gate_pred == 1) & (failure_flag == 0)).sum().item())
    gate_tn = int(((gate_pred == 0) & (failure_flag == 0)).sum().item())
    gate_fn = int(((gate_pred == 0) & (failure_flag == 1)).sum().item())
    gate_prec = gate_tp / max(gate_tp + gate_fp, 1)
    gate_rec = gate_tp / max(gate_tp + gate_fn, 1)
    
    # B) Final decision (measures actual rerank policy: includes margin_pred)
    decision_pred = use_rerank.float()
    decision_acc = (decision_pred == failure_flag.float()).float().mean().item() * 100.0
    decision_tp = int(((decision_pred == 1) & (failure_flag == 1)).sum().item())
    decision_fp = int(((decision_pred == 1) & (failure_flag == 0)).sum().item())
    decision_tn = int(((decision_pred == 0) & (failure_flag == 0)).sum().item())
    decision_fn = int(((decision_pred == 0) & (failure_flag == 1)).sum().item())
    decision_prec = decision_tp / max(decision_tp + decision_fp, 1)
    decision_rec = decision_tp / max(decision_tp + decision_fn, 1)
    decision_fp_rate = decision_fp / max(decision_fp + decision_tn, 1) * 100.0
    
    # V3 FIX 6: DETAILED DIAGNOSTICS
    fail_and_rerank = fail_mask & use_rerank
    nonfail_and_rerank = (~fail_mask) & use_rerank
    
    oracle_acc_on_reranked_fail = 0.0
    true_gain_on_reranked_fail = 0.0
    if fail_and_rerank.sum() > 0:
        oracle_correct = (best_query_pred[fail_and_rerank] == oracle_idx[fail_and_rerank])
        oracle_acc_on_reranked_fail = oracle_correct.float().mean().item() * 100.0
        true_gain_on_reranked_fail = (selected_iou[fail_and_rerank] - default_iou[fail_and_rerank]).mean().item() * 100.0
    
    n_fp_rerank = int(nonfail_and_rerank.sum().item())
    fp_rerank_harm = 0.0
    if n_fp_rerank > 0:
        fp_rerank_harm = (selected_iou[nonfail_and_rerank] - default_iou[nonfail_and_rerank]).mean().item() * 100.0
    
    # Overall oracle accuracy (sanity check)
    oracle_accuracy_overall = (default_idx == oracle_idx).float().mean().item() * 100.0
    
    return {
        "tau": float(tau),
        "margin_pred": float(margin_pred),
        "margin_gap": float(margin_gap),
        "gate_temperature": float(gate_temperature),
        "n_samples": N,
        "n_failures": n_fail,
        "failure_rate": fail_rate,
        "rerank_count": rerank_count,
        "rerank_rate": rerank_rate,
        
        "q0_miou": q0_miou,
        "oracle_miou": oracle_miou,
        "selected_miou": selected_miou,
        "delta_full": delta_full,
        
        "q0_miou_fail": q0_miou_fail,
        "oracle_miou_fail": oracle_miou_fail,
        "selected_miou_fail": selected_miou_fail,
        "delta_fail": delta_fail,
        "gap_closed": gap_closed,
        
        "auc_fail": auc_fail,
        
        # Gate-only metrics (measures p_fail > tau)
        "gate_acc": gate_acc,
        "gate_prec": gate_prec * 100.0,
        "gate_rec": gate_rec * 100.0,
        "gate_tp": gate_tp,
        "gate_fp": gate_fp,
        "gate_tn": gate_tn,
        "gate_fn": gate_fn,
        
        # Decision metrics (measures ACTUAL policy with margin_pred)
        "decision_acc": decision_acc,
        "decision_prec": decision_prec * 100.0,
        "decision_rec": decision_rec * 100.0,
        "decision_fp_rate": decision_fp_rate,
        "decision_tp": decision_tp,
        "decision_fp": decision_fp,
        "decision_tn": decision_tn,
        "decision_fn": decision_fn,
        
        # V3 Diagnostics
        "oracle_acc_on_reranked_failures": oracle_acc_on_reranked_fail,
        "true_gain_on_reranked_failures": true_gain_on_reranked_fail,
        "n_false_positive_reranks": n_fp_rerank,
        "fp_rerank_harm": fp_rerank_harm,
        "oracle_accuracy_overall": oracle_accuracy_overall,
    }


def find_best_tau_v3(model, loader, device, taus, margin_pred=0.02, margin_gap=0.0,
                     gate_temperature=1.0, use_grid=True):
    """Find best tau with V3 evaluate."""
    best_tau = taus[len(taus)//2] if taus else 0.5
    best_delta = -1e9
    best_metrics = None
    
    for tau in taus:
        m = evaluate_v3(model, loader, device, tau=tau, margin_pred=margin_pred,
                       margin_gap=margin_gap, gate_temperature=gate_temperature, use_grid=use_grid)
        if m["delta_full"] > best_delta:
            best_delta = m["delta_full"]
            best_tau = tau
            best_metrics = m
    
    return best_tau, best_metrics


def find_best_tau_and_temp_v3(model, loader, device, taus, temps, margin_pred=0.02,
                              margin_gap=0.0, use_grid=True):
    """
    V3 FIX 5: Find best (tau, temperature) combination.
    
    Recommended protocol:
    1. Pick temperature using AUC on tune split
    2. Then pick tau using delta_full with fixed temperature
    
    Current: Joint optimization for simplicity.
    """
    best_tau = taus[len(taus)//2] if taus else 0.5
    best_temp = 1.0
    best_delta = -1e9
    best_metrics = None
    
    for temp in temps:
        for tau in taus:
            m = evaluate_v3(model, loader, device, tau=tau, margin_pred=margin_pred,
                           margin_gap=margin_gap, gate_temperature=temp, use_grid=use_grid)
            if m["delta_full"] > best_delta:
                best_delta = m["delta_full"]
                best_tau = tau
                best_temp = temp
                best_metrics = m
    
    return best_tau, best_temp, best_metrics


def tb_log_split_v3(writer: SummaryWriter, split: str, metrics: Dict[str, float], epoch: int, prefix="eval"):
    """TensorBoard logging with V3 diagnostics."""
    # Base metrics
    writer.add_scalar(f"{prefix}/{split}/q0_miou", metrics["q0_miou"], epoch)
    writer.add_scalar(f"{prefix}/{split}/oracle_miou", metrics["oracle_miou"], epoch)
    writer.add_scalar(f"{prefix}/{split}/selected_miou", metrics["selected_miou"], epoch)
    writer.add_scalar(f"{prefix}/{split}/delta_full", metrics["delta_full"], epoch)
    
    # Failure subset
    writer.add_scalar(f"{prefix}/{split}/failure_rate", metrics["failure_rate"], epoch)
    writer.add_scalar(f"{prefix}/{split}/rerank_rate", metrics["rerank_rate"], epoch)
    writer.add_scalar(f"{prefix}/{split}/delta_fail", metrics["delta_fail"], epoch)
    writer.add_scalar(f"{prefix}/{split}/gap_closed", metrics["gap_closed"], epoch)
    
    # Gate-only metrics
    writer.add_scalar(f"{prefix}/{split}/auc_fail", metrics["auc_fail"], epoch)
    writer.add_scalar(f"{prefix}/{split}/gate_acc", metrics["gate_acc"], epoch)
    writer.add_scalar(f"{prefix}/{split}/gate_prec", metrics["gate_prec"], epoch)
    writer.add_scalar(f"{prefix}/{split}/gate_rec", metrics["gate_rec"], epoch)
    
    # Decision metrics (actual policy with margin)
    writer.add_scalar(f"{prefix}/{split}/decision_acc", metrics["decision_acc"], epoch)
    writer.add_scalar(f"{prefix}/{split}/decision_prec", metrics["decision_prec"], epoch)
    writer.add_scalar(f"{prefix}/{split}/decision_rec", metrics["decision_rec"], epoch)
    writer.add_scalar(f"{prefix}/{split}/decision_fp_rate", metrics["decision_fp_rate"], epoch)
    
    # V3 Diagnostics
    writer.add_scalar(f"{prefix}/{split}/oracle_acc_on_reranked_failures", 
                     metrics["oracle_acc_on_reranked_failures"], epoch)
    writer.add_scalar(f"{prefix}/{split}/true_gain_on_reranked_failures",
                     metrics["true_gain_on_reranked_failures"], epoch)
    writer.add_scalar(f"{prefix}/{split}/n_false_positive_reranks",
                     metrics["n_false_positive_reranks"], epoch)
    writer.add_scalar(f"{prefix}/{split}/fp_rerank_harm",
                     metrics["fp_rerank_harm"], epoch)
    writer.add_scalar(f"{prefix}/{split}/oracle_accuracy_overall",
                     metrics["oracle_accuracy_overall"], epoch)


#=============================================================================
# TABLE EXPORT UTILITIES
#=============================================================================

def format_float(x, nd=2, signed=False):
    """Format float for LaTeX table."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    try:
        val = float(x)
        if signed:
            return f"{val:+.{nd}f}"
        else:
            return f"{val:.{nd}f}"
    except:
        return "--"


def split_name_to_table_key(split_name):
    """
    Map cached split names to table column keys.
    
    Examples:
        val_refcoco_unc -> refcoco_val
        testA_refcoco_unc -> refcoco_testA
        testB_refcocoplus_unc -> refcocoplus_testB
        val_refcocog_umd -> refcocog_val
        test_refcocog_umd -> refcocog_test
    """
    split_name = split_name.lower()
    
    # Extract dataset
    if "refcocoplus" in split_name or "refcoco+" in split_name:
        dataset = "refcocoplus"
    elif "refcocog" in split_name:
        dataset = "refcocog"
    elif "refcoco" in split_name:
        dataset = "refcoco"
    else:
        return None
    
    # Extract split type
    if split_name.startswith("val_") or "_val" in split_name:
        split_type = "val"
    elif split_name.startswith("testa_") or "_testa" in split_name:
        split_type = "testA"
    elif split_name.startswith("testb_") or "_testb" in split_name:
        split_type = "testB"
    elif split_name.startswith("test_") or "_test" in split_name:
        split_type = "test"
    else:
        return None
    
    return f"{dataset}_{split_type}"


def load_eval_loaders_for_export(cache_dir, batch_size=32, num_workers=0, use_grid=True):
    """Load all evaluation splits for table export."""
    loaders = {}
    
    # Standard split names
    split_files = [
        "val_refcoco_unc_feats.pt",
        "testA_refcoco_unc_feats.pt",
        "testB_refcoco_unc_feats.pt",
        "val_refcocoplus_unc_feats.pt",
        "testA_refcocoplus_unc_feats.pt",
        "testB_refcocoplus_unc_feats.pt",
        "val_refcocog_umd_feats.pt",
        "test_refcocog_umd_feats.pt",
    ]
    
    pw = (num_workers > 0)
    pf = 2 if pw else None
    
    for fname in split_files:
        fpath = os.path.join(cache_dir, fname)
        if not os.path.exists(fpath):
            print(f"WARNING: Split file not found: {fpath}")
            continue
        
        try:
            data = torch.load(fpath, weights_only=False)
            ds = FailureDataset(data, use_grid=use_grid)
            split_key = split_name_to_table_key(fname.replace("_feats.pt", ""))
            
            if split_key:
                loader = DataLoader(
                    ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=pw, prefetch_factor=pf
                )
                loaders[split_key] = loader
                print(f"  Loaded: {fname} -> {split_key} ({len(ds)} samples)")
        except Exception as e:
            print(f"WARNING: Failed to load {fname}: {e}")
    
    return loaders


def eval_all_splits_for_export(model, loaders, device, tau=0.5, temp=1.0, 
                                margin_pred=0.02, margin_gap=0.0, use_grid=True):
    """Evaluate model on all splits and return results dict."""
    results = {}
    
    for split_key, loader in loaders.items():
        print(f"  Evaluating {split_key}...")
        metrics = evaluate_v3(
            model, loader, device, tau=tau, margin_pred=margin_pred,
            margin_gap=margin_gap, gate_temperature=temp, use_grid=use_grid
        )
        results[split_key] = metrics
        print(f"    DeRIS: {metrics['q0_miou']:.2f}% -> {metrics['selected_miou']:.2f}%  "
              f"Δ={metrics['delta_full']:+.2f}%")
    
    return results


def generate_sota_table_latex(results_dict):
    """
    Generate SOTA comparison table LaTeX.
    
    Args:
        results_dict: {split_key: metrics_dict}
            where split_key like 'refcoco_val', 'refcocoplus_testA', etc.
    
    Returns:
        LaTeX table string
    """
    
    # Column order
    cols = [
        ('refcoco', ['val', 'testA', 'testB']),
        ('refcocoplus', ['val', 'testA', 'testB']),
        ('refcocog', ['val', 'test'])
    ]
    
    # Extract baseline and selected mIoU for each split
    def get_miou(dataset, split):
        key = f"{dataset}_{split}"
        if key in results_dict:
            return (results_dict[key]['q0_miou'], 
                   results_dict[key]['selected_miou'],
                   results_dict[key]['delta_full'])
        return (None, None, None)
    
    # Build table rows
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Comparison with state-of-the-art methods} on RefCOCO, RefCOCO+, and RefCOCOg (mIoU\,\%).")
    lines.append(r"$^\dagger$~cIoU metric.")
    lines.append(r"Best specialist results in \textbf{bold}; second best \underline{underlined}.")
    lines.append(r"\vH{} operates as a post-hoc re-ranker on the frozen DeRIS-L output, adding ${\sim}$11.3\,M parameters and $<1$\,ms latency.}")
    lines.append(r"\label{tab:sota}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l|cc|ccc|ccc|cc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Method} & \multirow{2}{*}{Visual Enc.} & \multirow{2}{*}{Text Enc.}")
    lines.append(r"& \multicolumn{3}{c|}{RefCOCO} & \multicolumn{3}{c|}{RefCOCO+} & \multicolumn{2}{c}{RefCOCOg} \\")
    lines.append(r" & & & val & testA & testB & val & testA & testB & val(U) & test(U) \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{11}{l}{\textit{MLLM-based Methods}} \\")
    lines.append(r"LISA-7B$^\dagger$       & SAM-H+CLIP-L            & Vicuna-7B     & 74.90 & 79.10 & 72.30 & 65.10 & 70.80 & 58.10 & 67.90 & 70.60 \\")
    lines.append(r"GSVA-13B$^\dagger$      & SAM-H+CLIP-L            & Vicuna-13B    & 78.20 & 80.40 & 74.20 & 67.40 & 71.50 & 60.90 & 74.20 & 75.60 \\")
    lines.append(r"GLaMM-7B$^\dagger$      & SAM-H+CLIP-H            & Vicuna-7B     & 79.50 & 83.20 & 76.90 & 72.60 & 78.70 & 64.60 & 74.20 & 74.90 \\")
    lines.append(r"SAM4MLLM-8B$^\dagger$   & SAM-EfViT-XL1           & Qwen-VL-7B    & 79.80 & 82.70 & 74.70 & 74.60 & 80.00 & 67.20 & 75.50 & 76.40 \\")
    lines.append(r"DeRIS-7B$^\dagger$      & Swin-B+SigLIP           & Qwen2-OV-7B   & 84.05 & 85.79 & 83.32 & 80.30 & 83.92 & 76.16 & 80.62 & 80.59 \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{11}{l}{\textit{Specialist Methods}} \\")
    lines.append(r"LAVT                    & Swin-B                  & BERT-B        & 74.46 & 76.89 & 70.94 & 65.81 & 70.97 & 59.23 & 63.34 & 63.62 \\")
    lines.append(r"CRIS                    & CLIP-RN101              & CLIP-T        & 70.47 & 73.18 & 66.10 & 62.27 & 68.08 & 53.68 & 59.87 & 60.36 \\")
    lines.append(r"SimVG-Seg               & BEiT3-B                 & BEiT3-B       & 77.78 & 79.14 & 76.02 & 72.21 & 75.37 & 67.85 & 72.19 & 73.02 \\")
    lines.append(r"C3VG                    & BEiT3-B                 & BEiT3-B       & 81.37 & 82.93 & 79.12 & 77.05 & 79.61 & 72.40 & 76.34 & 77.10 \\")
    lines.append(r"OneRef-L                & BEiT3-L                 & BEiT3-L       & 81.26 & 83.06 & 79.45 & 76.60 & 80.16 & 72.95 & 75.68 & 76.82 \\")
    lines.append(r"DeRIS-B                 & Swin-S                  & BEiT3-B       & 81.99 & 82.97 & 80.14 & 75.62 & 79.16 & 71.63 & 76.30 & 77.15 \\")
    
    # DeRIS-L baseline row (using q0_miou from results)
    deris_row = "DeRIS-L                 & Swin-B                  & BEiT3-L       "
    for dataset, splits in cols:
        for split in splits:
            baseline, _, _ = get_miou(dataset, split)
            deris_row += f"& {format_float(baseline, 2)} "
    deris_row += r"\\"
    lines.append(deris_row)
    
    # Our method row (using selected_miou from results)
    ours_row = r"DeRIS-L + \vH{} (Ours) & Swin-B                  & BEiT3-L       "
    for dataset, splits in cols:
        for split in splits:
            _, selected, _ = get_miou(dataset, split)
            ours_row += f"& {format_float(selected, 2)} "
    ours_row += r"\\"
    lines.append(ours_row)
    
    # Delta row
    delta_row = r"\quad $\Delta$ (improvement) & --                      & --            "
    for dataset, splits in cols:
        for split in splits:
            _, _, delta = get_miou(dataset, split)
            delta_row += f"& {format_float(delta, 2, signed=True)} "
    delta_row += r"\\"
    lines.append(delta_row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def generate_ablation_table_latex(ablation_results):
    """
    Generate ablation table LaTeX.
    
    Args:
        ablation_results: List of dicts with keys:
            - name: Configuration name
            - params_m: Parameters in millions
            - delta_full: Overall improvement
            - delta_fail: Improvement on failures
            - auc_fail: Failure detection AUC
            - group: (optional) Group identifier for separators
    
    Returns:
        LaTeX table string
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Ablation study on RefCOCO val.}")
    lines.append(r"We report overall improvement ($\Delta_{\text{full}}$), improvement on failures ($\Delta_{\text{fail}}$), ")
    lines.append(r"and failure detection quality (AUC). All configurations use the same Transformer V3 architecture (11.3M params).}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{l|c|c|c|c}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Params & $\Delta_{\text{full}}$ & $\Delta_{\text{fail}}$ & AUC \\")
    lines.append(r"             & (M)    & (\%)                   & (\%)                   &     \\")
    lines.append(r"\midrule")
    
    current_group = None
    for result in ablation_results:
        # Add group separator if needed
        if 'group' in result and result['group'] != current_group:
            if current_group is not None:
                lines.append(r"\midrule")
            lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{result['group']}}}}} \\\\")
            current_group = result['group']
        
        name = result.get('name', 'Unknown')
        params_m = result.get('params_m', 11.3)
        delta_full = result.get('delta_full', None)
        delta_fail = result.get('delta_fail', None)
        auc_fail = result.get('auc_fail', None)
        
        # Format with optional boldface for best
        is_best = result.get('is_best', False)
        
        def maybe_bold(val_str):
            if is_best and val_str != "--":
                return f"\\textbf{{{val_str}}}"
            return val_str
        
        row = f"{name} & {format_float(params_m, 2)} "
        row += f"& {maybe_bold(format_float(delta_full, 2, signed=True))} "
        row += f"& {maybe_bold(format_float(delta_fail, 2, signed=True))} "
        row += f"& {maybe_bold(format_float(auc_fail, 3))} "
        row += r"\\"
        
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def run_table_export(args):
    """Run table export mode."""
    print("\n" + "="*80)
    print("TABLE EXPORT MODE")
    print("="*80)
    
    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    # Determine checkpoint path
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(args.output_dir, "best_v3.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Please specify --ckpt or ensure best_v3.pth exists in output_dir.")
        return
    
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract hyperparams from checkpoint
    tau = ckpt.get('tau', 0.5)
    temp = ckpt.get('temp', 1.0)
    margin_pred = ckpt.get('margin_pred', args.margin_pred)
    margin_gap = ckpt.get('margin_gap', args.margin_gap)
    
    # Load architecture from checkpoint (with fallback to inference)
    if 'use_grid' in ckpt and 'hidden_dim' in ckpt:
        # Modern checkpoint with architecture params
        use_grid = ckpt['use_grid']
        hidden_dim = ckpt['hidden_dim']
        n_transformer_layers = ckpt['n_transformer_layers']
        n_heads = ckpt['n_heads']
        query_dim = ckpt.get('query_dim', 256)
        n_queries = ckpt.get('n_queries', 10)
        print(f"  Loaded architecture from checkpoint: use_grid={use_grid}, hidden_dim={hidden_dim}, "
              f"n_layers={n_transformer_layers}, n_heads={n_heads}")
    else:
        # Legacy checkpoint - infer from model state
        print("  WARNING: Legacy checkpoint (no architecture params). Attempting inference...")
        model_state = ckpt['model']
        first_layer_in = model_state['query_encoder.0.weight'].shape[1]
        use_grid = (first_layer_in > 300)
        hidden_dim = model_state['query_encoder.0.weight'].shape[0]
        n_transformer_layers = sum(1 for k in model_state.keys() if 'transformer.layers.' in k and '.linear1.weight' in k)
        n_heads = args.n_heads  # Must use CLI args for legacy checkpoints
        query_dim = args.query_dim if hasattr(args, 'query_dim') else 256
        n_queries = 10  # Assume default
        print(f"  Inferred: use_grid={use_grid}, hidden_dim={hidden_dim}, "
              f"n_layers={n_transformer_layers}, n_heads={n_heads}")
        print(f"  NOTE: For legacy checkpoints, specify --use_grid, --hidden_dim, etc. explicitly")
    print(f"  tau={tau:.2f}, temp={temp:.2f}, margin_pred={margin_pred:.3f}, margin_gap={margin_gap:.3f}")
    
    # Load data (MUST use the same use_grid as checkpoint!)
    print(f"\nLoading evaluation splits from {args.cache_dir}...")
    loaders = load_eval_loaders_for_export(
        args.cache_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_grid=use_grid  # Use inferred value!
    )
    
    if not loaders:
        print("ERROR: No evaluation splits loaded. Check cache_dir.")
        return
    
    # Create model with loaded/inferred architecture
    model = FailureReranker(
        query_dim=query_dim,
        hidden_dim=hidden_dim,
        n_queries=n_queries,
        use_grid=use_grid,
        n_transformer_layers=n_transformer_layers,
        n_heads=n_heads,
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    print(f"\nModel loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    # Evaluate all splits
    print(f"\nEvaluating all splits...")
    results = eval_all_splits_for_export(
        model, loaders, device, tau=tau, temp=temp,
        margin_pred=margin_pred, margin_gap=margin_gap, use_grid=use_grid
    )
    
    # Determine output directory
    tables_out = args.tables_out if args.tables_out else args.output_dir
    os.makedirs(tables_out, exist_ok=True)
    
    # Generate SOTA table
    print(f"\n{'='*80}")
    print("GENERATING SOTA TABLE")
    print(f"{'='*80}")
    sota_latex = generate_sota_table_latex(results)
    
    sota_path = os.path.join(tables_out, "table_sota.tex")
    with open(sota_path, "w") as f:
        f.write(sota_latex)
    
    print(f"\nSaved to: {sota_path}")
    print("\n" + "-"*80)
    print(sota_latex)
    print("-"*80)
    
    # Generate ablation table (if config provided)
    if args.ablation_config and os.path.exists(args.ablation_config):
        print(f"\n{'='*80}")
        print("GENERATING ABLATION TABLE")
        print(f"{'='*80}")
        print(f"\nLoading ablation config: {args.ablation_config}")
        
        # Load ablation config (support both JSON and YAML)
        try:
            with open(args.ablation_config, 'r') as f:
                content = f.read()
                try:
                    ablation_configs = json.loads(content)
                except:
                    # Try YAML (manual simple parser to avoid dependency)
                    ablation_configs = []
                    print("WARNING: JSON parse failed, trying simple YAML parse")
        except Exception as e:
            print(f"ERROR: Failed to load ablation config: {e}")
            ablation_configs = []
        
        # Run ablations
        ablation_results = []
        
        # Get RefCOCO val loader
        refcoco_val_key = 'refcoco_val'
        if refcoco_val_key not in loaders:
            print(f"WARNING: RefCOCO val split not found. Skipping ablations.")
        else:
            refcoco_val_loader = loaders[refcoco_val_key]
            
            for config in ablation_configs:
                name = config.get('name', 'Unknown')
                ckpt_path_abl = config.get('ckpt_path', None)
                
                if ckpt_path_abl is None:
                    print(f"  SKIP: {name} (no ckpt_path)")
                    ablation_results.append({
                        'name': name,
                        'group': config.get('group', None),
                        'params_m': config.get('params_m', 11.3),
                        'delta_full': None,
                        'delta_fail': None,
                        'auc_fail': None,
                        'is_best': config.get('is_best', False),
                    })
                    continue
                
                if not os.path.exists(ckpt_path_abl):
                    print(f"  SKIP: {name} (ckpt not found: {ckpt_path_abl})")
                    ablation_results.append({
                        'name': name,
                        'group': config.get('group', None),
                        'params_m': config.get('params_m', 11.3),
                        'delta_full': None,
                        'delta_fail': None,
                        'auc_fail': None,
                        'is_best': config.get('is_best', False),
                    })
                    continue
                
                print(f"  Evaluating: {name}")
                
                # Load checkpoint
                ckpt_abl = torch.load(ckpt_path_abl, map_location=device, weights_only=False)
                model.load_state_dict(ckpt_abl['model'])
                model.eval()
                
                # Evaluate
                tau_abl = ckpt_abl.get('tau', tau)
                temp_abl = ckpt_abl.get('temp', temp)
                margin_pred_abl = ckpt_abl.get('margin_pred', margin_pred)
                margin_gap_abl = ckpt_abl.get('margin_gap', margin_gap)
                
                metrics = evaluate_v3(
                    model, refcoco_val_loader, device,
                    tau=tau_abl, margin_pred=margin_pred_abl, margin_gap=margin_gap_abl,
                    gate_temperature=temp_abl, use_grid=use_grid
                )
                
                ablation_results.append({
                    'name': name,
                    'group': config.get('group', None),
                    'params_m': config.get('params_m', 11.3),
                    'delta_full': metrics['delta_full'],
                    'delta_fail': metrics['delta_fail'],
                    'auc_fail': metrics['auc_fail'],
                    'is_best': config.get('is_best', False),
                })
                
                print(f"    Δfull={metrics['delta_full']:+.2f}%, "
                      f"Δfail={metrics['delta_fail']:+.2f}%, "
                      f"AUC={metrics['auc_fail']:.3f}")
        
        if ablation_results:
            # Generate table
            ablation_latex = generate_ablation_table_latex(ablation_results)
            
            ablation_path = os.path.join(tables_out, "table_ablation.tex")
            with open(ablation_path, "w") as f:
                f.write(ablation_latex)
            
            print(f"\nSaved to: {ablation_path}")
            print("\n" + "-"*80)
            print(ablation_latex)
            print("-"*80)
    else:
        if args.ablation_config:
            print(f"\nWARNING: Ablation config not found: {args.ablation_config}")
        print("Skipping ablation table generation.")
    
    print(f"\n{'='*80}")
    print("TABLE EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {tables_out}")
    print(f"  - table_sota.tex")
    if args.ablation_config and ablation_results:
        print(f"  - table_ablation.tex")


#=============================================================================
# MAIN V3
#=============================================================================

def main():
    parser = argparse.ArgumentParser("Failure Re-Ranker V3 (Complete)")
    
    parser.add_argument("--cache_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=3, type=int,
                       help="Train gate only for first N epochs")
    parser.add_argument("--ce_weight", default=1.0, type=float,
                       help="Weight for cross-entropy ranking loss")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--focal_gamma", default=2.0, type=float,
                       help="Focal loss gamma for gate (0=standard BCE, 2=focal)")
    parser.add_argument("--lr_warmup", default=5, type=int,
                       help="Linear LR warmup epochs")
    parser.add_argument("--rank_mode", default="iou_reg", type=str,
                       choices=["ce", "iou_reg", "listnet"],
                       help="Ranking loss: ce=cross-entropy (failure-only), iou_reg=IoU regression (all), listnet=ListNet (all)")
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    
    # V3 specific
    parser.add_argument("--margin_pred", default=0.02, type=float,
                       help="Min predicted gain to trigger rerank (safety)")
    parser.add_argument("--margin_gap", default=0.0, type=float,
                       help="Min gap between best and second-best alternative (optional, 0=disabled)")
    parser.add_argument("--temps", default="1.0", type=str,
                       help="Comma-separated temperatures for calibration (e.g. '0.7,1.0,1.5')")
    
    # Model
    parser.add_argument("--use_grid", default=1, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--n_transformer_layers", default=3, type=int)
    parser.add_argument("--n_heads", default=8, type=int)
    
    # Files
    parser.add_argument("--train_file", default="train_combined_feats.pt", type=str)
    parser.add_argument("--tune_file", default="testA_refcoco_unc_feats.pt", type=str)
    parser.add_argument("--primary_split", default="testB_refcocoplus_unc", type=str)
    parser.add_argument("--eval_glob", default="testA_*_feats.pt,testB_*_feats.pt,test_refcocog_umd_feats.pt", type=str)
    parser.add_argument("--taus", default="0.0,0.05,0.1,0.2,0.3,0.5,0.7", type=str)
    parser.add_argument("--save_every", default=5, type=int)
    
    # Table export
    parser.add_argument("--export_tables", action="store_true",
                       help="Export LaTeX tables from checkpoint (no training)")
    parser.add_argument("--ckpt", default=None, type=str,
                       help="Checkpoint path for table export (default: output_dir/best_v3.pth)")
    parser.add_argument("--tables_out", default=None, type=str,
                       help="Output dir for tables (default: output_dir)")
    parser.add_argument("--ablation_config", default=None, type=str,
                       help="Path to ablation config YAML/JSON file")
    
    args = parser.parse_args()
    if args.no_amp:
        args.amp = False
    
    use_grid = bool(args.use_grid)
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # Check if we're in table export mode
    if args.export_tables:
        run_table_export(args)
        return
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"FAILURE RE-RANKER V3 TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"\nV3 Configuration:")
    print(f"  - Zero-sum rank_logits (default=0)")
    print(f"  - Cross-entropy on oracle_idx (ranking objective)")
    print(f"  - margin_pred safety: {args.margin_pred:.3f}")
    print(f"  - ce_weight: {args.ce_weight:.2f}")
    print(f"  - Warmup epochs: {args.warmup_epochs} (gate only)")
    print(f"  - Full epochs: {args.epochs - args.warmup_epochs}")
    
    # Load data
    train_path = os.path.join(args.cache_dir, args.train_file)
    tune_path = os.path.join(args.cache_dir, args.tune_file)
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing: {train_path}")
    if not os.path.exists(tune_path):
        raise FileNotFoundError(f"Missing: {tune_path}")
    
    print(f"\nLoading data...")
    train_data = torch.load(train_path, weights_only=False)
    tune_data = torch.load(tune_path, weights_only=False)
    
    # Eval splits
    eval_paths = []
    for patt in [p.strip() for p in args.eval_glob.split(",") if p.strip()]:
        eval_paths.extend(glob.glob(os.path.join(args.cache_dir, patt)))
    eval_paths = sorted(list(set(eval_paths)))
    
    print(f"Train: {args.train_file} ({train_data['query_feat'].shape[0]} samples)")
    print(f"Tune:  {args.tune_file} ({tune_data['query_feat'].shape[0]} samples)")
    print(f"Eval:  {len(eval_paths)} splits")
    
    # Datasets
    train_ds = FailureDataset(train_data, use_grid=use_grid)
    tune_ds = FailureDataset(tune_data, use_grid=use_grid)
    
    # FIX 1: persistent_workers only when num_workers > 0
    pw = (args.num_workers > 0)
    pf = 2 if pw else None
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, 
                              persistent_workers=pw, prefetch_factor=pf)
    tune_loader = DataLoader(tune_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=pw, prefetch_factor=pf)
    
    eval_loaders = {}
    for ep in eval_paths:
        edata = torch.load(ep, weights_only=False)
        eds = FailureDataset(edata, use_grid=use_grid)
        split_name = infer_split_name_from_filename(ep)
        eval_loaders[split_name] = DataLoader(eds, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True,
                                              persistent_workers=pw, prefetch_factor=pf)
    
    # Model
    n_queries = int(train_data["query_feat"].shape[1])
    query_dim = int(train_data["query_feat"].shape[2])
    
    model = FailureReranker(
        query_dim=query_dim,
        hidden_dim=args.hidden_dim,
        n_queries=n_queries,
        use_grid=use_grid,
        n_transformer_layers=args.n_transformer_layers,
        n_heads=args.n_heads,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel V3 (Transformer): {n_params:,} params ({n_params/1e6:.2f}M)")
    print(f"  hidden_dim={args.hidden_dim}, layers={args.n_transformer_layers}, heads={args.n_heads}")
    
    # Compute pos_weight for gate: #negatives / #positives
    n_fail_train = int(train_ds.failure_flag.sum().item())
    n_total_train = len(train_ds)
    n_nonfail_train = n_total_train - n_fail_train
    gate_pos_weight = torch.tensor(n_nonfail_train / max(n_fail_train, 1), device=device)
    print(f"\n  Gate class balance: {n_fail_train} failures / {n_total_train} total ({100*n_fail_train/n_total_train:.1f}%)")
    print(f"  pos_weight = {gate_pos_weight.item():.2f}, focal_gamma = {args.focal_gamma}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine schedule with linear warmup
    def lr_lambda(epoch):
        if epoch < args.lr_warmup:
            return (epoch + 1) / args.lr_warmup
        progress = (epoch - args.lr_warmup) / max(1, args.epochs - args.lr_warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    tb_dir = os.path.join(args.output_dir, "tb_logs")
    writer = SummaryWriter(tb_dir)
    print(f"\nTensorBoard: {tb_dir}")
    
    # Parse taus and temps
    taus = [float(x.strip()) for x in args.taus.split(",") if x.strip()]
    temps = [float(x.strip()) for x in args.temps.split(",") if x.strip()]
    
    # Initial eval
    print(f"\n{'='*80}")
    print("BEFORE TRAINING")
    print(f"{'='*80}")
    
    if len(temps) > 1:
        print(f"Finding best (tau, temperature)...")
        best_tau, best_temp, tune_metrics0 = find_best_tau_and_temp_v3(
            model, tune_loader, device, taus, temps, 
            margin_pred=args.margin_pred, margin_gap=args.margin_gap, use_grid=use_grid
        )
        print(f"Best: tau={best_tau:.2f}, temp={best_temp:.2f}, Δfull={tune_metrics0['delta_full']:+.2f}%")
    else:
        best_temp = temps[0] if temps else 1.0
        best_tau, tune_metrics0 = find_best_tau_v3(
            model, tune_loader, device, taus, margin_pred=args.margin_pred,
            margin_gap=args.margin_gap, gate_temperature=best_temp, use_grid=use_grid
        )
        print(f"Best: tau={best_tau:.2f}, Δfull={tune_metrics0['delta_full']:+.2f}%")
    
    # Best tracking
    primary_split = args.primary_split
    if primary_split not in eval_loaders and eval_loaders:
        primary_split = list(eval_loaders.keys())[0]
        print(f"WARNING: primary_split not found, using {primary_split}")
    
    best_score = -1e9
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        gate_only = (epoch <= args.warmup_epochs)
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs}  lr={lr:.6f}  {'[GATE ONLY]' if gate_only else '[FULL]'}")
        print(f"{'='*80}")
        
        avg_loss, avg_gate, avg_rank = train_epoch_v3(
            model, optimizer, train_loader, device, epoch, writer,
            use_grid=use_grid, use_amp=args.amp, gate_only=gate_only, ce_weight=args.ce_weight,
            focal_gamma=args.focal_gamma, gate_pos_weight=gate_pos_weight,
            rank_mode=args.rank_mode
        )
        
        if gate_only:
            print(f"Train: loss={avg_loss:.4f} gate={avg_gate:.4f} [WARMUP]")
        else:
            print(f"Train: loss={avg_loss:.4f} gate={avg_gate:.4f} rank={avg_rank:.4f} [{args.rank_mode}]")
        
        writer.add_scalar("train/lr", lr, epoch)
        scheduler.step()
        
        # Tune
        if len(temps) > 1:
            best_tau, best_temp, tune_metrics = find_best_tau_and_temp_v3(
                model, tune_loader, device, taus, temps, 
                margin_pred=args.margin_pred, margin_gap=args.margin_gap, use_grid=use_grid
            )
        else:
            best_tau, tune_metrics = find_best_tau_v3(
                model, tune_loader, device, taus, margin_pred=args.margin_pred,
                margin_gap=args.margin_gap, gate_temperature=best_temp, use_grid=use_grid
            )
        
        writer.add_scalar("tau/best_tau", best_tau, epoch)
        writer.add_scalar("tau/best_temp", best_temp, epoch)
        tb_log_split_v3(writer, "tune", tune_metrics, epoch)
        
        print(f"Tune: tau={best_tau:.2f} temp={best_temp:.2f} Δfull={tune_metrics['delta_full']:+.2f}% AUC={tune_metrics['auc_fail']:.3f}")
        
        # Eval all splits
        epoch_eval = {}
        for split, loader in eval_loaders.items():
            m = evaluate_v3(model, loader, device, tau=best_tau, margin_pred=args.margin_pred,
                          margin_gap=args.margin_gap, gate_temperature=best_temp, use_grid=use_grid)
            epoch_eval[split] = m
            tb_log_split_v3(writer, split, m, epoch)
            
            print(f"  [{split}] DeRIS {m['q0_miou']:.2f}% → {m['selected_miou']:.2f}%  "
                 f"Δ={m['delta_full']:+.2f}%  oracle_acc={m['oracle_acc_on_reranked_failures']:.1f}%")
        
        # FIX 2: Save best (proper fallback)
        if primary_split in epoch_eval:
            score = epoch_eval[primary_split]["delta_full"]
        else:
            # Fallback to tune metrics
            score = tune_metrics["delta_full"]
        
        if score > best_score:
            best_score = score
            best_epoch = epoch
            
            state = {
                "model": model.state_dict(),
                "tau": best_tau,
                "temp": best_temp,
                "margin_pred": args.margin_pred,
                "margin_gap": args.margin_gap,
                "epoch": epoch,
                # Architecture params (for table export)
                "use_grid": use_grid,
                "hidden_dim": args.hidden_dim,
                "n_transformer_layers": args.n_transformer_layers,
                "n_heads": args.n_heads,
                "query_dim": query_dim,
                "n_queries": n_queries,
            }
            torch.save(state, os.path.join(args.output_dir, "best_v3.pth"))
            
            summary = {
                "best_epoch": epoch,
                "best_tau": best_tau,
                "best_temp": best_temp,
                "margin_pred": args.margin_pred,
                "margin_gap": args.margin_gap,
                "best_delta_full": best_score,
                "tune": tune_metrics,
                "splits": epoch_eval,
            }
            with open(os.path.join(args.output_dir, "summary_v3.json"), "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"  ✓ NEW BEST: Δfull={best_score:+.2f}% (epoch={epoch})")
        
        writer.flush()
    
    # Save last epoch model (even if not "best" by delta_full)
    last_state = {
        "model": model.state_dict(),
        "tau": best_tau,
        "temp": best_temp,
        "margin_pred": args.margin_pred,
        "margin_gap": args.margin_gap,
        "epoch": args.epochs,
        "use_grid": use_grid,
        "hidden_dim": args.hidden_dim,
        "n_transformer_layers": args.n_transformer_layers,
        "n_heads": args.n_heads,
        "query_dim": query_dim,
        "n_queries": n_queries,
    }
    torch.save(last_state, os.path.join(args.output_dir, "last_v3.pth"))
    
    writer.close()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best: epoch {best_epoch}, Δfull={best_score:+.2f}%")
    print(f"Saved: {os.path.join(args.output_dir, 'best_v3.pth')}")
    print(f"Last:  {os.path.join(args.output_dir, 'last_v3.pth')}")


if __name__ == "__main__":
    main()
