"""
Venice-H1: Failure-Aware Query Re-Ranking with Multi-Scale Grid Signatures
for Referring Image Segmentation.

OdaxAI Research — Nicolò Savioli, Ph.D.
https://arxiv.org/abs/2606.22546
"""

from venice_h1.model.grid_signatures import MultiScaleGridSignatures
from venice_h1.model.reranker import VeniceH1Reranker

__version__ = "1.0.0"
__author__ = "Nicolò Savioli"
__email__ = "nicolo.savioli@odaxai.com"
__all__ = ["MultiScaleGridSignatures", "VeniceH1Reranker"]
