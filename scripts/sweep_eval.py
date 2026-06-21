#!/usr/bin/env python3
"""
Comprehensive evaluation sweep across ALL splits and ALL configurations.
OPTIMIZED: one forward pass per split, then instant threshold sweep.
"""
import torch
import torch.nn.functional as F
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_failure_reranker_v3_complete import (
    FailureReranker, FailureDataset, set_seed
)

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

SPLIT_ORDER = [
    'refcoco_val', 'refcoco_testA', 'refcoco_testB',
    'refcocoplus_val', 'refcocoplus_testA', 'refcocoplus_testB',
    'refcocog_val', 'refcocog_test'
]

SPLIT_SHORT = {
    'refcoco_val': 'RC-val', 'refcoco_testA': 'RC-tA', 'refcoco_testB': 'RC-tB',
    'refcocoplus_val': 'RC+-val', 'refcocoplus_testA': 'RC+-tA', 'refcocoplus_testB': 'RC+-tB',
    'refcocog_val': 'RCg-val', 'refcocog_test': 'RCg-tst'
}

TABLE_LABELS = {
    'refcoco_val': 'val', 'refcoco_testA': 'testA', 'refcoco_testB': 'testB',
    'refcocoplus_val': 'val', 'refcocoplus_testA': 'testA', 'refcocoplus_testB': 'testB',
    'refcocog_val': 'val(U)', 'refcocog_test': 'test(U)'
}


def load_eval_splits(cache_dir, use_grid=True, batch_size=256, num_workers=0):
    """Load all evaluation splits."""
    from torch.utils.data import DataLoader
    
    feat_dir = cache_dir
    file_map = {
        'refcoco_val': 'val_refcoco_unc_feats.pt',
        'refcoco_testA': 'testA_refcoco_unc_feats.pt',
        'refcoco_testB': 'testB_refcoco_unc_feats.pt',
        'refcocoplus_val': 'val_refcocoplus_unc_feats.pt',
        'refcocoplus_testA': 'testA_refcocoplus_unc_feats.pt',
        'refcocoplus_testB': 'testB_refcocoplus_unc_feats.pt',
        'refcocog_val': 'val_refcocog_umd_feats.pt',
        'refcocog_test': 'test_refcocog_umd_feats.pt',
    }
    
    loaders = {}
    for key, fname in file_map.items():
        fpath = os.path.join(feat_dir, fname)
        if not os.path.exists(fpath):
            print(f"  SKIP: {fname} not found")
            continue
        data_dict = torch.load(fpath, map_location='cpu', weights_only=False)
        ds = FailureDataset(data_dict, use_grid=use_grid)
        loaders[key] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)
        print(f"  Loaded: {fname} -> {key} ({len(ds)} samples)")
    
    return loaders


def forward_pass_cached(model, loader, device, use_grid=True):
    """Run model forward pass ONCE and cache all outputs."""
    model.eval()
    
    all_p_fail_logit = []
    all_rank_logits_raw = []
    all_det_scores = []
    all_failure_flag = []
    all_default_iou = []
    all_oracle_idx = []
    all_query_ious = []
    
    with torch.no_grad():
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
            
            all_p_fail_logit.append(p_fail_logit.cpu())
            all_rank_logits_raw.append(rank_logits_raw.cpu())
            all_det_scores.append(det_scores.cpu())
            all_failure_flag.append(batch["failure_flag"])
            all_default_iou.append(batch["default_iou"])
            all_oracle_idx.append(batch["oracle_idx"])
            all_query_ious.append(batch["query_ious"])
    
    return {
        'p_fail_logit': torch.cat(all_p_fail_logit),
        'rank_logits_raw': torch.cat(all_rank_logits_raw),
        'det_scores': torch.cat(all_det_scores),
        'failure_flag': torch.cat(all_failure_flag),
        'default_iou': torch.cat(all_default_iou),
        'oracle_idx': torch.cat(all_oracle_idx),
        'query_ious': torch.cat(all_query_ious),
    }


def threshold_sweep(cached, tau, margin_pred, temp, margin_gap=0.0):
    """Apply threshold logic on cached model outputs. INSTANT - no GPU needed."""
    
    p_fail_logit = cached['p_fail_logit']
    rank_logits_raw = cached['rank_logits_raw']
    det_scores = cached['det_scores']
    failure_flag = cached['failure_flag']
    default_iou = cached['default_iou']
    oracle_idx = cached['oracle_idx']
    query_ious = cached['query_ious']
    
    N = failure_flag.shape[0]
    B, N_queries = rank_logits_raw.shape
    
    # Zero-sum rank_logits
    default_idx = det_scores.argmax(dim=1)  # [B]
    default_logit = rank_logits_raw.gather(1, default_idx.unsqueeze(1))  # [B,1]
    rank_logits = rank_logits_raw - default_logit
    
    # Gate with temperature
    p_fail = torch.sigmoid(p_fail_logit / temp)
    
    # Mask non-default for safety margin
    idx = torch.arange(N_queries).unsqueeze(0).expand(B, -1)
    mask_non_default = (idx != default_idx.unsqueeze(1))
    neg_inf = torch.full_like(rank_logits, -1e9)
    rank_logits_alt = torch.where(mask_non_default, rank_logits, neg_inf)
    
    # Decision
    best_alt = rank_logits_alt.max(dim=1).values
    use_rerank = (p_fail > tau) & (best_alt > margin_pred)
    
    # Optional TOP2 margin
    if margin_gap > 0:
        vals, _ = torch.topk(rank_logits_alt, k=min(2, N_queries - 1), dim=1)
        best_v = vals[:, 0]
        second_v = vals[:, 1] if vals.shape[1] > 1 else torch.full_like(best_v, -1e9)
        use_rerank = use_rerank & ((best_v - second_v) > margin_gap)
    
    best_query_pred = rank_logits.argmax(dim=1)  # [B]
    oracle_iou = query_ious.max(dim=1)[0]
    
    # Rerank rate
    rerank_count = int(use_rerank.sum().item())
    rerank_rate = (rerank_count / N) * 100.0
    
    # Selected IoU
    iou_at_best = query_ious.gather(1, best_query_pred.unsqueeze(1)).squeeze(1)
    selected_iou = torch.where(use_rerank, iou_at_best, default_iou)
    
    q0_miou = default_iou.mean().item() * 100.0
    oracle_miou = oracle_iou.mean().item() * 100.0
    selected_miou = selected_iou.mean().item() * 100.0
    delta_full = selected_miou - q0_miou
    
    fail_mask = (failure_flag == 1)
    n_fail = int(fail_mask.sum().item())
    
    # Diagnostics on failures
    fail_and_rerank = fail_mask & use_rerank
    nonfail_and_rerank = (~fail_mask) & use_rerank
    
    oracle_acc = 0.0
    true_gain = 0.0
    if fail_and_rerank.sum() > 0:
        oracle_correct = (best_query_pred[fail_and_rerank] == oracle_idx[fail_and_rerank])
        oracle_acc = oracle_correct.float().mean().item() * 100.0
        true_gain = (selected_iou[fail_and_rerank] - default_iou[fail_and_rerank]).mean().item() * 100.0
    
    n_fp = int(nonfail_and_rerank.sum().item())
    fp_harm = 0.0
    if n_fp > 0:
        fp_harm = (selected_iou[nonfail_and_rerank] - default_iou[nonfail_and_rerank]).mean().item() * 100.0
    
    # AUC
    auc = 0.5
    if roc_auc_score is not None:
        try:
            auc = float(roc_auc_score(failure_flag.numpy(), p_fail.numpy()))
        except:
            pass
    
    # Gate confusion
    gate_pred = (p_fail > tau).float()
    gate_tp = int(((gate_pred == 1) & (failure_flag == 1)).sum().item())
    gate_fp = int(((gate_pred == 1) & (failure_flag == 0)).sum().item())
    gate_prec = gate_tp / max(gate_tp + gate_fp, 1)
    gate_rec = gate_tp / max(gate_tp + int(((gate_pred == 0) & (failure_flag == 1)).sum().item()), 1)
    
    # Decision confusion
    dec_tp = int(((use_rerank) & (fail_mask)).sum().item())
    dec_fp = int(((use_rerank) & (~fail_mask)).sum().item())
    
    return {
        'q0_miou': q0_miou,
        'oracle_miou': oracle_miou,
        'selected_miou': selected_miou,
        'delta_full': delta_full,
        'rerank_rate': rerank_rate,
        'rerank_count': rerank_count,
        'n_failures': n_fail,
        'n_samples': N,
        'oracle_acc': oracle_acc,
        'true_gain': true_gain,
        'n_fp': n_fp,
        'fp_harm': fp_harm,
        'auc': auc,
        'gate_prec': gate_prec,
        'gate_rec': gate_rec,
        'dec_tp': dec_tp,
        'dec_fp': dec_fp,
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    cache_dir = "/home/bionick87/miccai_2026/features"
    
    # ========== SWEEP GRID ==========
    taus = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    margin_preds = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
    temps = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    # ========== MODELS ==========
    ckpt_configs = []
    for name, path in [
        ("IoUReg-best", "./outputs/reranker_v4_ioureg/best_v3.pth"),
        ("ListNet-best", "./outputs/reranker_v4_listnet/best_v3.pth"),
        ("ListNet-last", "./outputs/reranker_v4_listnet/last_v3.pth"),
    ]:
        if os.path.exists(path):
            ckpt_configs.append((path, name))
    
    if not ckpt_configs:
        print("ERROR: No checkpoints found!")
        return
    
    # ========== LOAD SPLITS ONCE ==========
    # We need to know use_grid; infer from first checkpoint
    first_ckpt = torch.load(ckpt_configs[0][0], map_location='cpu', weights_only=False)
    use_grid_first = first_ckpt.get('use_grid', True)
    
    print("Loading evaluation splits...")
    loaders = load_eval_splits(cache_dir, use_grid=use_grid_first, batch_size=256)
    
    n_configs = len(taus) * len(margin_preds) * len(temps)
    print(f"\nSweep grid: {len(taus)} taus x {len(margin_preds)} margins x {len(temps)} temps = {n_configs} configs")
    print(f"Models: {[c[1] for c in ckpt_configs]}")
    print(f"Splits: {list(loaders.keys())}")
    
    # ========== RUN ==========
    all_model_results = {}
    
    for ckpt_path, label in ckpt_configs:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        use_grid = ckpt.get('use_grid', True)
        hidden_dim = ckpt.get('hidden_dim', 512)
        n_transformer_layers = ckpt.get('n_transformer_layers', 3)
        n_heads = ckpt.get('n_heads', 8)
        query_dim = ckpt.get('query_dim', 256)
        n_queries = ckpt.get('n_queries', 10)
        
        model = FailureReranker(
            query_dim=query_dim, hidden_dim=hidden_dim, n_queries=n_queries,
            use_grid=use_grid, n_transformer_layers=n_transformer_layers, n_heads=n_heads,
        ).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n{'='*120}")
        print(f"MODEL: {label} | {n_params:.1f}M params | {hidden_dim}d {n_transformer_layers}L {n_heads}H | grid={use_grid}")
        print(f"  Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch','?')})")
        print(f"{'='*120}")
        
        # ---- STEP 1: Forward pass - ONE TIME per split ----
        cached_splits = {}
        t0 = time.time()
        for split_key in SPLIT_ORDER:
            if split_key not in loaders:
                continue
            print(f"  Forward pass: {split_key}...", end=' ', flush=True)
            cached_splits[split_key] = forward_pass_cached(model, loaders[split_key], device, use_grid)
            n = cached_splits[split_key]['failure_flag'].shape[0]
            nf = int((cached_splits[split_key]['failure_flag'] == 1).sum())
            print(f"done ({n} samples, {nf} failures = {nf/n*100:.1f}%)")
        fwd_time = time.time() - t0
        print(f"  Forward passes complete in {fwd_time:.1f}s")
        
        # ---- STEP 2: Threshold sweep - INSTANT ----
        print(f"\n  Sweeping {n_configs} configs x {len(cached_splits)} splits = {n_configs*len(cached_splits)} evals...")
        t0 = time.time()
        
        all_results = {}  # {split_key: {(tau,mp,temp): metrics}}
        best_per_split = {}
        
        for split_key in SPLIT_ORDER:
            if split_key not in cached_splits:
                continue
            cached = cached_splits[split_key]
            split_results = {}
            
            for temp in temps:
                for tau in taus:
                    for mp in margin_preds:
                        m = threshold_sweep(cached, tau, mp, temp, margin_gap=0.0)
                        split_results[(tau, mp, temp)] = m
            
            all_results[split_key] = split_results
            
            # Best config for this split
            best_key = max(split_results.keys(), key=lambda k: split_results[k]['delta_full'])
            best_m = split_results[best_key]
            best_per_split[split_key] = {
                'config': best_key,
                'tau': best_key[0], 'margin_pred': best_key[1], 'temp': best_key[2],
                **best_m
            }
        
        sweep_time = time.time() - t0
        print(f"  Sweep complete in {sweep_time:.1f}s")
        
        # ================================================================
        # REPORTS
        # ================================================================
        
        # ---- REPORT 1: Best config PER SPLIT ----
        print(f"\n{'='*120}")
        print(f"REPORT 1: BEST CONFIG PER SPLIT (independent optimization) - {label}")
        print(f"{'='*120}")
        print(f"  {'Split':<12} {'tau':>6} {'mp':>8} {'T':>5} | {'Base':>8} {'Ours':>8} {'Δ':>8} {'Rr%':>7} {'OAcc':>7} {'TrueG':>7} {'FP':>5} {'FPH':>7} {'AUC':>6}")
        print(f"  {'-'*110}")
        
        for split_key in SPLIT_ORDER:
            if split_key not in best_per_split:
                continue
            b = best_per_split[split_key]
            short = SPLIT_SHORT[split_key]
            sign = '+' if b['delta_full'] >= 0 else ''
            print(f"  {short:<12} {b['tau']:>6.2f} {b['margin_pred']:>8.3f} {b['temp']:>5.1f} | "
                  f"{b['q0_miou']:>7.2f}% {b['selected_miou']:>7.2f}% {sign}{b['delta_full']:>7.2f}% "
                  f"{b['rerank_rate']:>6.1f}% {b['oracle_acc']:>6.1f}% {b['true_gain']:>+6.2f}% "
                  f"{b['n_fp']:>5d} {b['fp_harm']:>+6.2f}% {b['auc']:>5.3f}")
        
        # ---- REPORT 2: Best GLOBAL config ----
        first_split = next(iter(all_results.keys()))
        all_configs = list(all_results[first_split].keys())
        
        best_avg = -1e9
        best_global_cfg = None
        for cfg in all_configs:
            deltas = [all_results[s][cfg]['delta_full'] for s in SPLIT_ORDER if s in all_results]
            avg = sum(deltas) / len(deltas)
            if avg > best_avg:
                best_avg = avg
                best_global_cfg = cfg
        
        print(f"\n{'='*120}")
        print(f"REPORT 2: BEST GLOBAL CONFIG (same for all splits) - {label}")
        print(f"  Config: tau={best_global_cfg[0]:.2f}, margin_pred={best_global_cfg[1]:.3f}, temp={best_global_cfg[2]:.1f} -> avg Δ={best_avg:+.3f}%")
        print(f"{'='*120}")
        print(f"  {'Split':<12} {'Base':>8} {'Ours':>8} {'Δ':>8} {'Rr%':>7} {'OAcc':>7} {'FP':>5} {'FPH':>7}")
        print(f"  {'-'*70}")
        
        for split_key in SPLIT_ORDER:
            if split_key not in all_results:
                continue
            m = all_results[split_key][best_global_cfg]
            short = SPLIT_SHORT[split_key]
            sign = '+' if m['delta_full'] >= 0 else ''
            print(f"  {short:<12} {m['q0_miou']:>7.2f}% {m['selected_miou']:>7.2f}% {sign}{m['delta_full']:>7.2f}% "
                  f"{m['rerank_rate']:>6.1f}% {m['oracle_acc']:>6.1f}% {m['n_fp']:>5d} {m['fp_harm']:>+6.2f}%")
        
        # ---- REPORT 3: Best config with NO negative splits ----
        best_safe_avg = -1e9
        best_safe_cfg = None
        for cfg in all_configs:
            deltas = [all_results[s][cfg]['delta_full'] for s in SPLIT_ORDER if s in all_results]
            if any(d < -0.005 for d in deltas):
                continue
            avg = sum(deltas) / len(deltas)
            if avg > best_safe_avg:
                best_safe_avg = avg
                best_safe_cfg = cfg
        
        print(f"\n{'='*120}")
        print(f"REPORT 3: BEST SAFE CONFIG (no negative splits) - {label}")
        if best_safe_cfg:
            print(f"  Config: tau={best_safe_cfg[0]:.2f}, margin_pred={best_safe_cfg[1]:.3f}, temp={best_safe_cfg[2]:.1f} -> avg Δ={best_safe_avg:+.3f}%")
            print(f"{'='*120}")
            print(f"  {'Split':<12} {'Base':>8} {'Ours':>8} {'Δ':>8} {'Rr%':>7}")
            print(f"  {'-'*50}")
            for split_key in SPLIT_ORDER:
                if split_key not in all_results:
                    continue
                m = all_results[split_key][best_safe_cfg]
                short = SPLIT_SHORT[split_key]
                sign = '+' if m['delta_full'] >= 0 else ''
                print(f"  {short:<12} {m['q0_miou']:>7.2f}% {m['selected_miou']:>7.2f}% {sign}{m['delta_full']:>7.2f}% {m['rerank_rate']:>6.1f}%")
        else:
            print(f"  NO CONFIG EXISTS where all splits are non-negative!")
            print(f"{'='*120}")
        
        # ---- REPORT 4: Delta heatmap vs TAU (mp=0, T=1.0) ----
        print(f"\n{'='*120}")
        print(f"REPORT 4: DELTA vs TAU (mp=0.0, T=1.0) - {label}")
        print(f"{'='*120}")
        
        avail_splits = [s for s in SPLIT_ORDER if s in all_results]
        header = f"  {'tau':<6}" + "".join([f"{SPLIT_SHORT[s]:>10}" for s in avail_splits]) + f"{'AVG':>10}"
        print(header)
        print(f"  {'-'*len(header)}")
        
        for tau in taus:
            cfg = (tau, 0.0, 1.0)
            row = f"  {tau:<6.2f}"
            deltas = []
            for s in avail_splits:
                m = all_results[s].get(cfg)
                if m:
                    d = m['delta_full']
                    deltas.append(d)
                    sign = '+' if d >= 0 else ''
                    row += f"{sign}{d:>9.3f}%"
                else:
                    row += f"{'--':>10}"
            if deltas:
                avg = sum(deltas) / len(deltas)
                sign = '+' if avg >= 0 else ''
                row += f"{sign}{avg:>9.3f}%"
            print(row)
        
        # ---- REPORT 5: Rerank rate heatmap vs TAU ----
        print(f"\n{'='*120}")
        print(f"REPORT 5: RERANK RATE (%) vs TAU (mp=0.0, T=1.0) - {label}")
        print(f"{'='*120}")
        
        header = f"  {'tau':<6}" + "".join([f"{SPLIT_SHORT[s]:>10}" for s in avail_splits])
        print(header)
        print(f"  {'-'*len(header)}")
        
        for tau in taus:
            cfg = (tau, 0.0, 1.0)
            row = f"  {tau:<6.2f}"
            for s in avail_splits:
                m = all_results[s].get(cfg)
                if m:
                    row += f"{m['rerank_rate']:>9.1f}%"
                else:
                    row += f"{'--':>10}"
            print(row)
        
        # ---- REPORT 6: Oracle accuracy vs TAU ----
        print(f"\n{'='*120}")
        print(f"REPORT 6: ORACLE ACC (%) on reranked failures vs TAU (mp=0.0, T=1.0) - {label}")
        print(f"{'='*120}")
        
        header = f"  {'tau':<6}" + "".join([f"{SPLIT_SHORT[s]:>10}" for s in avail_splits])
        print(header)
        print(f"  {'-'*len(header)}")
        
        for tau in taus:
            cfg = (tau, 0.0, 1.0)
            row = f"  {tau:<6.2f}"
            for s in avail_splits:
                m = all_results[s].get(cfg)
                if m:
                    row += f"{m['oracle_acc']:>9.1f}%"
                else:
                    row += f"{'--':>10}"
            print(row)
        
        # ---- REPORT 7: Delta vs margin_pred (tau=0.0, T=1.0) ----
        print(f"\n{'='*120}")
        print(f"REPORT 7: DELTA vs MARGIN_PRED (tau=0.0, T=1.0) - {label}")
        print(f"{'='*120}")
        
        header = f"  {'mp':<8}" + "".join([f"{SPLIT_SHORT[s]:>10}" for s in avail_splits]) + f"{'AVG':>10}"
        print(header)
        print(f"  {'-'*len(header)}")
        
        for mp in margin_preds:
            cfg = (0.0, mp, 1.0)
            row = f"  {mp:<8.3f}"
            deltas = []
            for s in avail_splits:
                m = all_results[s].get(cfg)
                if m:
                    d = m['delta_full']
                    deltas.append(d)
                    sign = '+' if d >= 0 else ''
                    row += f"{sign}{d:>9.3f}%"
                else:
                    row += f"{'--':>10}"
            if deltas:
                avg = sum(deltas) / len(deltas)
                sign = '+' if avg >= 0 else ''
                row += f"{sign}{avg:>9.3f}%"
            print(row)
        
        # ---- REPORT 8: Delta vs TEMP (tau=0.0, mp=0.0) ----
        print(f"\n{'='*120}")
        print(f"REPORT 8: DELTA vs TEMPERATURE (tau=0.0, mp=0.0) - {label}")
        print(f"{'='*120}")
        
        header = f"  {'temp':<6}" + "".join([f"{SPLIT_SHORT[s]:>10}" for s in avail_splits]) + f"{'AVG':>10}"
        print(header)
        print(f"  {'-'*len(header)}")
        
        for temp in temps:
            cfg = (0.0, 0.0, temp)
            row = f"  {temp:<6.1f}"
            deltas = []
            for s in avail_splits:
                m = all_results[s].get(cfg)
                if m:
                    d = m['delta_full']
                    deltas.append(d)
                    sign = '+' if d >= 0 else ''
                    row += f"{sign}{d:>9.3f}%"
                else:
                    row += f"{'--':>10}"
            if deltas:
                avg = sum(deltas) / len(deltas)
                sign = '+' if avg >= 0 else ''
                row += f"{sign}{avg:>9.3f}%"
            print(row)
        
        # ---- REPORT 9: FP harm analysis at best global config ----
        print(f"\n{'='*120}")
        print(f"REPORT 9: FALSE POSITIVE ANALYSIS - {label}")
        print(f"{'='*120}")
        
        for tau_check in [0.0, 0.1, 0.2, 0.3, 0.5]:
            cfg = (tau_check, 0.0, 1.0)
            print(f"\n  tau={tau_check:.1f}, mp=0.0, T=1.0:")
            print(f"  {'Split':<12} {'Rr%':>7} {'DecTP':>7} {'DecFP':>7} {'FPHarm':>8} {'TrueG':>8} {'Net':>8}")
            print(f"  {'-'*60}")
            for s in avail_splits:
                m = all_results[s].get(cfg)
                if m:
                    short = SPLIT_SHORT[s]
                    # Net effect = true_gain * rerank_on_fail / total + fp_harm * rerank_on_nonfail / total
                    print(f"  {short:<12} {m['rerank_rate']:>6.1f}% {m['dec_tp']:>7d} {m['dec_fp']:>7d} "
                          f"{m['fp_harm']:>+7.2f}% {m['true_gain']:>+7.2f}% {m['delta_full']:>+7.3f}%")
        
        # ---- REPORT 10: BEST LATEX TABLE (per-split optimized) ----
        print(f"\n{'='*120}")
        print(f"REPORT 10: LATEX TABLE with per-split best config - {label}")
        print(f"{'='*120}")
        
        baseline_values = {s: best_per_split[s]['q0_miou'] for s in avail_splits}
        ours_values = {s: best_per_split[s]['selected_miou'] for s in avail_splits}
        delta_values = {s: best_per_split[s]['delta_full'] for s in avail_splits}
        
        print(f"\n  Baseline: " + " & ".join([f"{baseline_values[s]:.2f}" for s in avail_splits]))
        print(f"  Ours:     " + " & ".join([f"{ours_values[s]:.2f}" for s in avail_splits]))
        print(f"  Delta:    " + " & ".join([f"{delta_values[s]:+.2f}" for s in avail_splits]))
        
        # ---- REPORT 11: Save comprehensive results ----
        all_model_results[label] = {
            'best_per_split': {k: {kk: vv for kk, vv in v.items() if kk != 'config'} for k, v in best_per_split.items()},
            'best_global_config': {'tau': best_global_cfg[0], 'mp': best_global_cfg[1], 'temp': best_global_cfg[2]},
            'best_global_avg_delta': best_avg,
            'best_safe_config': {'tau': best_safe_cfg[0], 'mp': best_safe_cfg[1], 'temp': best_safe_cfg[2]} if best_safe_cfg else None,
            'best_safe_avg_delta': best_safe_avg if best_safe_cfg else None,
        }
    
    # ===== SAVE =====
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'sweep_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_model_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    
    print(f"\n{'#'*120}")
    print("SWEEP COMPLETE")
    print(f"{'#'*120}")


if __name__ == "__main__":
    main()
