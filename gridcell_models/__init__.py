# Venice-H1: Grid Cell enhanced Referring Image Segmentation
#
# Architecture-agnostic grid cells applied to 3 SOTA backbones:
#   OneRef (NeurIPS'24) → +1.11% mIoU
#   C3VG   (AAAI'25)   → target: +1.0% mIoU
#   DeRIS  (ICCV'25)   → target: BEAT 85.72% mIoU SOTA
#
from .grid_cell_oneref import GridCellOneRef, MultiScaleGridCells
from .grid_cell_c3vg import GridCellC3VG
from .grid_cell_deris import GridCellDeRIS
from .losses import compute_grid_cell_losses
from .paper_logger import PaperLogger