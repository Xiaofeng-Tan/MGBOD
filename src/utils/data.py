"""
Data loading and preprocessing utilities.

This module provides functions for loading datasets in various formats
and performing data preprocessing operations.
"""

import numpy as np
import random
from typing import Tuple, List
from mat4py import loadmat


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from a given file path.
    
    Supports `.npz` and `.mat` formats from BElloney and ADBench datasets.
    
    Args:
        path: The file path of the dataset.
        
    Returns:
        A tuple of (X, y) where X is the feature matrix and y is the label vector.
    """
    try:
        print(path)
        data = loadmat(path)
    except:
        data = np.load(path, allow_pickle=True)
    
    try:
        data = np.array(data['trandata'])
    except:
        X = np.array(data['X'])
        y = np.array(data['y'])
        if type(y[0]) == np.ndarray:
            yy = []
            for i in range(len(y)):
                yy.append(y[i][0])
            y = np.array(yy)
        return X, y

    X = data[:, 0:-1]
    y = data[:, -1]
    if max(y) != 1:
        y -= min(y)
    if sum(y) > len(y) / 2:
        for i in range(len(y)):
            y[i] = 1 if y[i] == 0 else 0
    return X, y


def downsample(p: float, y: np.ndarray, n: int) -> Tuple[List[int], List[int]]:
    """
    Downsample the dataset by selecting a proportion of inlier samples.
    
    Args:
        p: The proportion of samples to select.
        y: The label array.
        n: Random seed.
        
    Returns:
        A tuple of (labels, indices) where labels is the new label list
        and indices are the selected sample indices.
    """
    if p == 0:
        return [], []
    
    random.seed(n)
    pos_sample = [i for i in range(len(y)) if y[i] == 0]
    n = int(p * len(y))
    n = 2 if n == 0 else n
    index = random.sample(pos_sample, n)
    labels = [0 for i in range(len(y))]
    labels = np.array(labels)
    labels[index] = 1
    return labels.tolist(), index
