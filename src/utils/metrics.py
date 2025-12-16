"""
Evaluation metrics and scoring utilities.

This module provides functions for computing outlier scores and 
evaluation metrics.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from typing import List


def get_group_score(
    data: torch.Tensor, 
    centers: List[List[float]], 
    radii: List[float], 
    score: np.ndarray
) -> torch.Tensor:
    """
    Compute group-level outlier scores based on granular ball membership.
    
    This function assigns outlier scores to individual samples based on 
    which granular ball they belong to.
    
    Args:
        data: Tensor of shape (N, M), the original data points.
        centers: List of granular ball centers.
        radii: List of granular ball radii.
        score: Array of outlier scores for each granular ball.
        
    Returns:
        Tensor of outlier scores for each sample.
    """
    score = torch.from_numpy(score)
    max_val, min_val = torch.max(score), torch.min(score)
    s = torch.ones(data.shape[0])
    
    if max_val != min_val:
        score = (score - min_val) / (max_val - min_val)
    else:
        return torch.zeros_like(s) + 0.5
    
    for i in range(len(centers)):
        center = centers[i]
        radius = radii[i]
        center_tensor = torch.tensor(center, device=data.device)
        dists = torch.norm(data - center_tensor, dim=1)
        indices = torch.where(dists <= radius)[0]
        indices = indices.to(device=s.device)
        s[indices] = torch.multiply(torch.tensor(score[i]), s[indices])
    
    return s


def analyse(score: np.ndarray, y: np.ndarray, path: str) -> pd.DataFrame:
    """
    Analyze detection performance at different thresholds.
    
    Computes precision and recall for the top a% samples (a = 5, 10, ..., 100)
    classified as outliers based on their scores.
    
    Args:
        score: Array of outlier scores for each sample.
        y: Array of true labels (0 for inlier, 1 for outlier).
        path: File path to save the results as Excel.
        
    Returns:
        A DataFrame containing precision and recall at different thresholds.
    """
    assert len(score) == len(y), "score和y的长度必须相同"
    score_ord = np.argsort(score)[::-1]
    
    results = []
    y_pred = np.array([0] * len(score))
    
    for a in range(5, 101, 5):
        num_outliers = int(len(score) * a / 100)
        y_pred[score_ord[0:num_outliers]] = 1
        TN, FP, FN, TP = confusion_matrix(y_true=y, y_pred=y_pred).ravel()
        DR = TP / (TP + FP)
        FAR = TP / (TP + FN)
        results.append([DR, FAR])
    
    results_df = pd.DataFrame(results, columns=["P", "R"])
    results_df.index = [f"{i}%" for i in range(5, 101, 5)]
    results_df.to_excel(path)
    return results_df
