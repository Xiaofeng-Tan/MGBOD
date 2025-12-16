#!/usr/bin/env python
"""
Visualization script for granular balls.

This script generates visualizations of multi-scale granular balls.

Usage:
    python scripts/run_visualization.py
    # or from project root:
    python -m scripts.run_visualization
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.preprocessing import MinMaxScaler

from src.core.granular_ball import general_GB, get_newM
from src.utils.data import load_data
from src.visualization.plot import plot_cir_p


def run_GB(X: torch.Tensor, output_dir: str) -> None:
    """
    Generate and visualize multi-scale granular balls.
    
    Args:
        X: Tensor of shape (N, 2), the 2D data points.
        output_dir: Directory to save output figures.
    """
    M = torch.cdist(X, X, p=2.0)
    k = 0
    
    while True:
        GB_list, c, r = general_GB(X, M)
        plot_cir_p(X, c, r, k, output_dir)
        k += 1
        print(f"Scale {k}")
        r_1 = torch.tensor(r)
        X_1 = torch.tensor(c)
        X, M = get_newM(X_1, r_1)
        if M.max() == 0:
            break


def main():
    """Main function for visualization."""
    # Set paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'datasets', '2.npz')
    output_dir = os.path.join(project_root, 'figures')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    X, y = load_data(data_path)
    X = MinMaxScaler().fit_transform(X)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    y = torch.from_numpy(y).to(dtype=torch.float32)
    
    # Plot original data
    plot_cir_p(X, 0, 0, 10, output_dir)
    
    # Generate and visualize granular balls
    run_GB(X, output_dir)
    
    print(f"\nFigures saved to {output_dir}")


if __name__ == '__main__':
    main()
