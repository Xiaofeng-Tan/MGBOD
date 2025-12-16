#!/usr/bin/env python
"""
Main entry point for running MGBOD experiments.

This script reproduces the results reported in the TKDE 2025 paper.

Usage:
    python scripts/run_experiment.py
    # or from project root:
    python -m scripts.run_experiment
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import copy as cp
import pandas as pd
import pickle
from sklearn import svm
from sklearn.metrics import roc_auc_score

from src.utils.data import load_data
from src.detector import fit, run_FRS, join, get_uncertainty


def test(
    X: torch.Tensor, 
    y: torch.Tensor, 
    _l: float, 
    _d: float, 
    _k: float
) -> list:
    """
    Run experiments comparing different methods.
    
    Experiments include:
    - FRS + one view
    - FRS + multi-views  
    - Proposed method + one view
    - Proposed method (full)
    
    Args:
        X: Tensor of shape (N, M), where N and M are the number of samples 
           and features, respectively.
        y: Tensor of shape (N), the labels of all samples. 
           NOTE: THE LABELS ARE NOT USED FOR TRAINING.
        _d: The similarity threshold δ in the paper.
        _l: The weighted parameter λ in the paper.
        _k: The parameter Δ in the paper.
        
    Returns:
        List of ROC scores for different method variants.
    """
    # Obtain the outlier proportion
    p = sum(y) / len(y)
    
    # Execute the proposed method without weighted SVM
    score, e_list, weight_list = fit(X, y, _l, _d)
    
    # Evaluate the results
    FRS = roc_auc_score(y_score=score, y_true=y)
    sort_score = np.argsort(score)
    
    # Obtain indices of reliable inliers and outliers
    index_il = sort_score[0:int(len(score) * _k * (1 - p))]
    index_ol = sort_score[-int(len(score) * _k * p)::]

    # Obtain pseudo labels for training
    y_pseudo = cp.deepcopy(y)
    y_pseudo[index_ol] = 1
    y_pseudo[index_il] = 0
    index = np.concatenate((index_il, index_ol))
    index = torch.from_numpy(np.unique(index))
    X_train = X[index]
    y_train = y_pseudo[index]
    
    # Training SVM
    e = get_uncertainty(index, e_list, weight_list)
    clf = svm.SVC(probability=True, class_weight='balanced')
    clf.fit(X_train, y_train, sample_weight=e)
    score = clf.decision_function(X)
    SVM = roc_auc_score(y_score=score, y_true=y)

    # Original view + weighted SVM
    score = run_FRS(X, y, _d, _l)
    score, e_list, weight_list = join([score], y)
    sort_score = np.argsort(score)
    index_il = sort_score[0:int(len(score) * _k * (1 - p))]
    index_ol = sort_score[-int(len(score) * _k * p)::]

    y_pseudo = cp.deepcopy(y)
    y_pseudo[index_ol] = 1
    y_pseudo[index_il] = 0
    index = np.concatenate((index_il, index_ol))
    index = torch.from_numpy(np.unique(index))
    X_train = X[index]
    y_train = y_pseudo[index]
    
    e = get_uncertainty(index, e_list, weight_list)
    clf = svm.SVC(probability=True, class_weight='balanced')
    clf.fit(X_train, y_train, sample_weight=e)
    score = clf.decision_function(X)
    SVM_oriview = roc_auc_score(y_score=score, y_true=y)
    
    return [
        roc_auc_score(y_score=run_FRS(X, y, _d, _l), y_true=y),
        FRS, 
        SVM_oriview, 
        SVM
    ]


def main():
    """Main function to run all experiments."""
    # Set paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    np.random.seed(0)
    dir_path = os.path.join(project_root, 'datasets')
    config_path = os.path.join(project_root, 'configs', 'parameters.pkl')
    results_path = os.path.join(project_root, 'results', 'result.xlsx')
    
    files = os.listdir(dir_path)
    
    with open(config_path, 'rb') as pkl_file:
        parameters = pickle.load(pkl_file)
    
    rocs = []
    name = []
    
    for file in files:
        file_path = os.path.join(dir_path, file)
        if file.endswith('.npz') or file.endswith('.mat'):
            # Skip files without parameter configuration
            if (file, 'l') not in parameters:
                print(f"Skipping {file} (no parameters configured)")
                continue
            name.append(file)
            l, d = parameters[(file, 'l')], parameters[(file, 'd')]
            print(f"Processing {file}...")
            X, y = load_data(file_path)
            X = torch.from_numpy(X).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)
            
            # Calculate ROC scores for different method variants
            # Methods WITHOUT density (l = 0)
            FRS, GB, SVM, SVM_S = test(X, y, 0, d, 0.7)
            res = [FRS, GB, SVM, SVM_S]
            
            # Methods WITH density (l != 0)
            FRS, GB, SVM, SVM_S = test(X, y, l, d, 0.7)
            res.extend([FRS, GB, SVM, SVM_S])
            rocs.append(res)
    
    # Save results
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    pd.DataFrame(
        data=rocs,
        columns=['000', '010', '001', '011', '100', '110', '101', '111'],
        index=name
    ).to_excel(results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
