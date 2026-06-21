"""
PaperLogger: Unified logging for paper-ready outputs.
Shared across all three training scripts (OneRef, C3VG, DeRIS).

Baselines from published papers:
  OneRef-B (NeurIPS'24):  77.81% mIoU on RefCOCO val
  C3VG     (AAAI'25):     81.42% mIoU on RefCOCO val
  DeRIS-L  (ICCV'25):     85.72% mIoU on RefCOCO val  ← CURRENT SOTA
"""

import os
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# =====================================================================
# BASELINES — from published papers
# =====================================================================

# C3VG baseline results (AAAI'25, from their retrained model)
C3VG_BASELINE = {
    'val_refcoco_unc': {'det_acc': 92.40, 'miou': 81.42, 'oiou': 80.95},
    'testA_refcoco_unc': {'det_acc': 94.81, 'miou': 82.98, 'oiou': 82.91},
    'testB_refcoco_unc': {'det_acc': 89.63, 'miou': 79.86, 'oiou': 79.03},
    'val_refcocoplus_unc': {'det_acc': 87.21, 'miou': 77.00, 'oiou': 74.32},
    'testA_refcocoplus_unc': {'det_acc': 90.59, 'miou': 79.53, 'oiou': 77.84},
    'testB_refcocoplus_unc': {'det_acc': 81.61, 'miou': 72.90, 'oiou': 69.29},
    'val_refcocog_umd': {'det_acc': 87.85, 'miou': 76.20, 'oiou': 74.85},
    'test_refcocog_umd': {'det_acc': 88.19, 'miou': 77.05, 'oiou': 76.43},
}

# OneRef-B baseline (NeurIPS'24, our measured from RES checkpoint)
ONEREF_BASELINE = {
    'unc_val': {'miou': 77.81, 'bbox_acc': 6.78},
    'unc_testA': {'miou': 79.35, 'bbox_acc': 7.92},
    'unc_testB': {'miou': 75.19, 'bbox_acc': 6.56},
    'unc+_val': {'miou': 66.62, 'bbox_acc': 7.00},
    'unc+_testA': {'miou': 72.13, 'bbox_acc': 7.93},
    'unc+_testB': {'miou': 59.52, 'bbox_acc': 7.14},
    'gref_umd_val': {'miou': 68.16, 'bbox_acc': 8.03},
    'gref_umd_test': {'miou': 68.43, 'bbox_acc': 7.73},
}

# DeRIS-L baseline (ICCV'25, CURRENT SOTA — from their paper Table 1)
DERIS_L_BASELINE = {
    'val_refcoco_unc': {'miou': 85.72, 'oiou': 85.41},
    'testA_refcoco_unc': {'miou': 86.64, 'oiou': 86.49},
    'testB_refcoco_unc': {'miou': 84.52, 'oiou': 82.87},
    'val_refcocoplus_unc': {'miou': 81.28, 'oiou': 79.01},
    'testA_refcocoplus_unc': {'miou': 83.74, 'oiou': 82.34},
    'testB_refcocoplus_unc': {'miou': 78.59, 'oiou': 74.41},
    'val_refcocog_umd': {'miou': 80.01, 'oiou': 77.65},
    'test_refcocog_umd': {'miou': 81.32, 'oiou': 80.12},
}

# DeRIS-B baseline (ICCV'25, from their paper Table 1)
DERIS_B_BASELINE = {
    'val_refcoco_unc': {'miou': 81.99, 'oiou': 80.80},
    'testA_refcoco_unc': {'miou': 82.97, 'oiou': 82.68},
    'testB_refcoco_unc': {'miou': 80.14, 'oiou': 78.47},
    'val_refcocoplus_unc': {'miou': 75.62, 'oiou': 72.21},
    'testA_refcocoplus_unc': {'miou': 79.16, 'oiou': 77.26},
    'testB_refcocoplus_unc': {'miou': 71.63, 'oiou': 66.11},
    'val_refcocog_umd': {'miou': 76.30, 'oiou': 73.89},
    'test_refcocog_umd': {'miou': 77.15, 'oiou': 75.88},
}

# Display name mapping (used by all three scripts)
SPLIT_DISPLAY = {
    # OneRef format
    'unc_val': 'RefCOCO val', 'unc_testA': 'RefCOCO testA',
    'unc_testB': 'RefCOCO testB',
    'unc+_val': 'RefCOCO+ val', 'unc+_testA': 'RefCOCO+ testA',
    'unc+_testB': 'RefCOCO+ testB',
    'gref_umd_val': 'RefCOCOg val', 'gref_umd_test': 'RefCOCOg test',
    # C3VG / DeRIS format
    'val_refcoco_unc': 'RefCOCO val', 'testA_refcoco_unc': 'RefCOCO testA',
    'testB_refcoco_unc': 'RefCOCO testB',
    'val_refcocoplus_unc': 'RefCOCO+ val',
    'testA_refcocoplus_unc': 'RefCOCO+ testA',
    'testB_refcocoplus_unc': 'RefCOCO+ testB',
    'val_refcocog_umd': 'RefCOCOg val',
    'test_refcocog_umd': 'RefCOCOg test',
}


class PaperLogger:
    """Unified logging for paper-ready outputs."""

    def __init__(self, log_dir, output_dir, model_name):
        self.output_dir = output_dir
        self.model_name = model_name
        self.figures_dir = os.path.join(output_dir, 'paper_figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        self.writer = None
        if SummaryWriter is not None and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        self.all_results = {}  # {split: {epoch: miou}}
        self.best_results = {}  # {split: best_miou}

    def log_train(self, step, loss, lr, alpha, epoch=None,
                  scale_weights=None, loss_components=None):
        """Log training scalars with full Grid Cell diagnostics."""
        if self.writer:
            self.writer.add_scalar('Train/loss', loss, step)
            self.writer.add_scalar('Train/lr', lr, step)
            self.writer.add_scalar('Train/gate_alpha', alpha, step)
            
            # Log Grid Cell scale weights (soft competition between 4x4, 8x8, 16x16)
            if scale_weights is not None:
                for i, (name, w) in enumerate(scale_weights.items()):
                    self.writer.add_scalar(f'GridCell/scale_{name}', w, step)
            
            # Log individual loss components
            if loss_components is not None:
                for name, val in loss_components.items():
                    self.writer.add_scalar(f'Loss/{name}', val, step)

    def log_eval(self, epoch, split, det_acc, miou, oiou):
        """Log eval metrics with consistent tag names."""
        dname = SPLIT_DISPLAY.get(split, split)
        if self.writer:
            self.writer.add_scalar(f'mIoU/{dname}', miou, epoch)
            self.writer.add_scalar(f'DetAcc/{dname}', det_acc, epoch)
            self.writer.add_scalar(f'oIoU/{dname}', oiou, epoch)
        self.all_results.setdefault(split, {})[epoch] = miou
        if miou > self.best_results.get(split, 0):
            self.best_results[split] = miou

    def log_eval_delta(self, epoch, split, miou, baseline_dict):
        """Log delta vs SOTA baseline — shows improvement over time."""
        dname = SPLIT_DISPLAY.get(split, split)
        sota = baseline_dict.get(split, {}).get('miou', 0)
        if self.writer and sota > 0:
            delta = miou - sota
            self.writer.add_scalar(f'Delta_vs_SOTA/{dname}', delta, epoch)
            # Also log SOTA as horizontal line (constant)
            self.writer.add_scalar(f'SOTA_baseline/{dname}', sota, epoch)
            # Log our mIoU for direct comparison
            self.writer.add_scalar(f'GridCell_mIoU/{dname}', miou, epoch)

    def print_cycle_table(self, cycle, baseline_miou, current_miou, gate_value, 
                          scale_weights=None, elapsed_time=None):
        """Print a formatted comparison table after each cycle."""
        delta = current_miou - baseline_miou
        status = "✅ BEATING SOTA" if delta > 0 else "❌ BELOW SOTA"
        
        print("\n" + "=" * 60)
        print(f"  CYCLE {cycle} RESULTS — Grid Cells vs DeRIS-L Baseline")
        print("=" * 60)
        print(f"  │ DeRIS-L Baseline (SOTA):  {baseline_miou:.2f}%")
        print(f"  │ Grid Cells Current:       {current_miou:.2f}%")
        print(f"  │ Delta (improvement):      {delta:+.2f}%  {status}")
        print(f"  │ Gate Value (alpha):       {gate_value:.4f}")
        if scale_weights:
            print(f"  │ Scale Weights:")
            for name, w in scale_weights.items():
                bar = "█" * int(w * 20) + "░" * (20 - int(w * 20))
                print(f"  │   {name:>5}: {bar} {w:.3f}")
        if elapsed_time:
            print(f"  │ Training Time:            {elapsed_time:.0f}s")
        print("=" * 60 + "\n")

    def log_image_grid(self, epoch, split, images, pred_masks, gt_masks,
                       pred_bboxes=None, gt_bboxes=None, texts=None,
                       max_images=4):
        """Log prediction visualizations to TensorBoard."""
        if not self.writer:
            return
        try:
            import torchvision.utils as vutils
            from torchvision.transforms.functional import to_pil_image
            import io
            from PIL import Image, ImageDraw, ImageFont

            dname = SPLIT_DISPLAY.get(split, split)
            vis_images = []

            for i in range(min(max_images, len(images))):
                img = images[i]
                if isinstance(img, torch.Tensor):
                    if img.dim() == 3 and img.shape[0] == 3:
                        img = img.cpu()
                    else:
                        continue
                else:
                    continue

                # Denormalize if needed (ImageNet normalization)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = img.clamp(0, 1)

                pil_img = to_pil_image(img)
                draw = ImageDraw.Draw(pil_img)
                W, H = pil_img.size

                # Draw GT mask (green overlay)
                if gt_masks is not None and i < len(gt_masks):
                    gt_m = gt_masks[i]
                    if isinstance(gt_m, torch.Tensor):
                        gt_m = gt_m.cpu().numpy()
                    if gt_m.ndim == 2:
                        overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                        ov_draw = ImageDraw.Draw(overlay)
                        for y in range(min(gt_m.shape[0], H)):
                            for x in range(min(gt_m.shape[1], W)):
                                if gt_m[y, x] > 0.5:
                                    ov_draw.point((x, y), fill=(0, 255, 0, 60))
                        pil_img = Image.alpha_composite(
                            pil_img.convert('RGBA'), overlay).convert('RGB')
                        draw = ImageDraw.Draw(pil_img)

                # Draw pred mask (red overlay)
                if pred_masks is not None and i < len(pred_masks):
                    pred_m = pred_masks[i]
                    if isinstance(pred_m, torch.Tensor):
                        pred_m = pred_m.cpu().numpy()
                    if pred_m.ndim == 2:
                        overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                        ov_draw = ImageDraw.Draw(overlay)
                        for y in range(min(pred_m.shape[0], H)):
                            for x in range(min(pred_m.shape[1], W)):
                                if pred_m[y, x] > 0.5:
                                    ov_draw.point((x, y), fill=(255, 0, 0, 60))
                        pil_img = Image.alpha_composite(
                            pil_img.convert('RGBA'), overlay).convert('RGB')
                        draw = ImageDraw.Draw(pil_img)

                # Draw GT bbox (green)
                if gt_bboxes is not None and i < len(gt_bboxes):
                    bbox = gt_bboxes[i]
                    if isinstance(bbox, torch.Tensor):
                        bbox = bbox.cpu().numpy()
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        draw.rectangle([x1, y1, x2, y2],
                                       outline='green', width=2)

                # Draw pred bbox (red)
                if pred_bboxes is not None and i < len(pred_bboxes):
                    bbox = pred_bboxes[i]
                    if isinstance(bbox, torch.Tensor):
                        bbox = bbox.cpu().numpy()
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        draw.rectangle([x1, y1, x2, y2],
                                       outline='red', width=2)

                # Draw text query
                if texts is not None and i < len(texts):
                    txt = texts[i] if isinstance(texts[i], str) else str(texts[i])
                    draw.text((5, 5), txt[:60], fill='white')

                # Convert back to tensor
                img_t = torch.from_numpy(
                    np.array(pil_img)).permute(2, 0, 1).float() / 255.0
                vis_images.append(img_t)

            if vis_images:
                grid = vutils.make_grid(vis_images, nrow=2, padding=4,
                                         normalize=False)
                self.writer.add_image(f'Predictions/{dname}', grid, epoch)
        except Exception as e:
            print(f"  [TBVis] Could not log images: {e}")

    def log_early_stopping(self, epoch, best_miou, patience_counter):
        """Log early stopping state."""
        if self.writer:
            self.writer.add_scalar('EarlyStopping/best_miou', best_miou, epoch)
            self.writer.add_scalar('EarlyStopping/patience_counter',
                                   patience_counter, epoch)

    def save_paper_table(self, baseline_name, baseline_results,
                         gridcell_name, gridcell_results):
        """Generate LaTeX and Markdown comparison tables."""
        # Canonical order
        refcoco_splits = ['val', 'testA', 'testB']
        refcocop_splits = ['val', 'testA', 'testB']
        refcocog_splits = ['val', 'test']

        def _get_miou(results, dataset, split):
            """Try multiple key formats."""
            keys = [
                f'{dataset}_{split}',
                f'{split}_{dataset}',
                f'{dataset.replace("unc", "refcoco_unc")}',
            ]
            for k in keys:
                if k in results:
                    v = results[k]
                    if isinstance(v, dict):
                        return v.get('miou', 0)
                    return v
            return 0

        # Build rows
        rows = []
        for name, results in [(baseline_name, baseline_results),
                               (gridcell_name, gridcell_results)]:
            row = [name]
            for ds, splits in [('unc', refcoco_splits),
                                ('unc+', refcocop_splits),
                                ('gref_umd', refcocog_splits)]:
                for sp in splits:
                    row.append(_get_miou(results, ds, sp))
            rows.append(row)

        # LaTeX
        latex = []
        latex.append(r'\begin{table}[t]')
        latex.append(r'\caption{Cross-architecture generalization of '
                     r'Grid Cells on RefCOCO/+/g (mIoU).}')
        latex.append(r'\centering')
        latex.append(r'\begin{tabular}{l|ccc|ccc|cc}')
        latex.append(r'\toprule')
        latex.append(r'Method & \multicolumn{3}{c|}{RefCOCO} & '
                     r'\multicolumn{3}{c|}{RefCOCO+} & '
                     r'\multicolumn{2}{c}{RefCOCOg} \\')
        latex.append(r'       & val & testA & testB & '
                     r'val & testA & testB & val & test \\')
        latex.append(r'\midrule')
        for row in rows:
            name = row[0]
            vals = [f'{v:.2f}' if v > 0 else '-' for v in row[1:]]
            if 'GridCell' in name or '+GC' in name:
                name = r'\textbf{' + name + '}'
            latex.append(f'{name} & {" & ".join(vals)} \\\\')
        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\end{table}')

        latex_str = '\n'.join(latex)
        latex_path = os.path.join(self.output_dir, 'paper_table.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_str)

        # Markdown
        md = []
        md.append('| Method | RC val | RC tA | RC tB | '
                   'RC+ val | RC+ tA | RC+ tB | RCg val | RCg test |')
        md.append('|--------|--------|-------|-------|'
                   '---------|--------|--------|---------|----------|')
        for row in rows:
            vals = [f'{v:.2f}' if v > 0 else '-' for v in row[1:]]
            md.append(f'| {row[0]} | {" | ".join(vals)} |')

        md_str = '\n'.join(md)
        md_path = os.path.join(self.output_dir, 'results.md')
        with open(md_path, 'w') as f:
            f.write(md_str)

        print(f"\n{md_str}\n")
        print(f"  Tables saved: {latex_path}, {md_path}")

        return latex_str, md_str

    def flush(self):
        if self.writer:
            try:
                self.writer.flush()
            except Exception:
                pass

    def close(self):
        if self.writer:
            self.writer.close()

    def log_final_summary(self, baseline_dict, results_history):
        """Log final summary comparing all cycles to baseline."""
        print("\n" + "=" * 70)
        print("  FINAL SUMMARY — Grid Cells Performance Analysis")
        print("=" * 70)
        
        if not results_history:
            print("  No results recorded.")
            return
        
        for split, cycles in results_history.items():
            dname = SPLIT_DISPLAY.get(split, split)
            baseline = baseline_dict.get(split, {}).get('miou', 0)
            print(f"\n  {dname}:")
            print(f"    Baseline (SOTA): {baseline:.2f}%")
            
            best_miou = 0
            best_cycle = 0
            for cycle, miou in sorted(cycles.items()):
                delta = miou - baseline
                marker = "→" if delta > 0 else " "
                print(f"    Cycle {cycle:>2}: {miou:.2f}% (Δ {delta:+.2f}%) {marker}")
                if miou > best_miou:
                    best_miou = miou
                    best_cycle = cycle
            
            final_delta = best_miou - baseline
            if final_delta > 0:
                print(f"    ✅ BEST: Cycle {best_cycle} with {best_miou:.2f}% (+{final_delta:.2f}% vs SOTA)")
            else:
                print(f"    ❌ Best: Cycle {best_cycle} with {best_miou:.2f}% ({final_delta:.2f}% vs SOTA)")
        
        print("\n" + "=" * 70)
