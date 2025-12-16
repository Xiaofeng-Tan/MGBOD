"""
MGBOD: Fuzzy Granule Density-Based Outlier Detection with Multi-Scale Granular Balls

This package provides the implementation of the TKDE 2025 paper.
"""

from .core.frs_od import FRS_OD, FRS_OD_GB
from .core.granular_ball import GB, general_GB, get_newM
from .utils.data import load_data, downsample
from .utils.metrics import get_group_score, analyse
from .visualization.plot import plot_cir_p

__version__ = "1.0.0"
__all__ = [
    "FRS_OD",
    "FRS_OD_GB", 
    "GB",
    "general_GB",
    "get_newM",
    "load_data",
    "downsample",
    "get_group_score",
    "analyse",
    "plot_cir_p",
]
