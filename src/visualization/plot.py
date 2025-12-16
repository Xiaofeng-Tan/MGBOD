"""
Visualization utilities for granular balls.

This module provides functions for plotting data points and granular balls.
"""

import os
import matplotlib.pyplot as plt
import warnings
from typing import List, Union

warnings.filterwarnings('ignore')

# Configure matplotlib style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})


def plot_cir_p(
    X: Union[list, 'torch.Tensor'], 
    c: Union[int, List[List[float]]], 
    r: Union[int, List[float]], 
    k: int,
    output_dir: str = "../fig"
) -> None:
    """
    Plot data points and granular balls.
    
    This function visualizes 2D data points and optionally draws circles
    representing granular balls.
    
    Args:
        X: Data points, either as a list or tensor of shape (N, 2).
        c: Either 0 (to plot only points) or a list of circle centers.
        r: Either 0 (to plot only points) or a list of circle radii.
        k: Index for the output filename.
        output_dir: Directory to save output figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if c == 0:
        # Plot only points
        x = [p[0] for p in X]
        y = [p[1] for p in X]
        plt.scatter(x, y, s=10)
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, f"GB_scale_{k}.pdf"))
        plt.savefig(os.path.join(output_dir, f"GB_scale_{k}.png"), dpi=300)
        plt.cla()
    else:
        # Plot points and circles
        x = [p[0] for p in X]
        y = [p[1] for p in X]
        plt.scatter(x, y, s=10)
        
        # Draw circles for granular balls
        for i in range(len(c)):
            circle = plt.Circle(c[i], r[i], color='r', fill=False)
            plt.gcf().gca().add_artist(circle)
        
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, f"GB_scale_{k}.pdf"))
        plt.savefig(os.path.join(output_dir, f"GB_scale_{k}.png"), dpi=300)
        plt.cla()
