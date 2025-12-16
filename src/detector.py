"""
Main MGBOD detector module.

This module provides the high-level API for multi-scale granular ball 
outlier detection.
"""

import numpy as np
import torch
import copy as cp
from typing import List, Tuple

from .core.frs_od import FRS_OD, FRS_OD_GB
from .core.granular_ball import general_GB, get_newM
from .utils.metrics import get_group_score


def OD_GB(
    X: torch.Tensor, 
    y: torch.Tensor, 
    d: float, 
    l: float
) -> List[torch.Tensor]:
    """
    Outlier detection in Granular Ball views.
    
    Args:
        X: Tensor of shape (N, M), where N and M are the number of samples 
           and features, respectively.
        y: Tensor of shape (N), the labels of all samples. 
           NOTE: THE LABELS ARE NOT USED FOR TRAINING.
        d: The similarity threshold δ in the paper.
        l: The weighted parameter λ in the paper.
        
    Returns:
        A list of outlier scores for all views.
    """
    M = torch.cdist(X, X, p=2.0)
    score_list = []
    X_true = cp.deepcopy(X)
    
    while True:
        GB_list, c, r = general_GB(X, M)
        r_1 = torch.tensor(r)
        X_1 = torch.tensor(c)
        clf = FRS_OD_GB(_deta=d, _lambda=l, _density=True, r=r_1)
        clf.fit(X=X_1, y=y)
        score = clf.predict()
        score = get_group_score(X_true, c, r, score.cpu().numpy())
        score_list.append(score)
        X, M = get_newM(X_1, r_1)
        if M.max() == 0:
            break
    
    return score_list


def run_FRS(
    X: torch.Tensor, 
    y: torch.Tensor, 
    d: float, 
    l: float
) -> torch.Tensor:
    """
    Outlier detection in original view.
    
    Args:
        X: Tensor of shape (N, M), where N and M are the number of samples 
           and features, respectively.
        y: Tensor of shape (N), the labels of all samples. 
           NOTE: THE LABELS ARE NOT USED FOR TRAINING.
        d: The similarity threshold δ in the paper.
        l: The weighted parameter λ in the paper.
        
    Returns:
        Outlier scores.
    """
    clf = FRS_OD(_deta=d, _lambda=l, _density=True)
    clf.fit(X=X, y=y)
    score = clf.predict()
    return score


def join(
    score_list: List[torch.Tensor], 
    y: torch.Tensor
) -> Tuple[torch.Tensor, List[np.ndarray], List[float]]:
    """
    Calculate the refined outlier probability by combining multi-view scores.
    
    Args:
        score_list: A list of outlier scores for all views, including the original view.
        y: Tensor of shape (N), the labels of all samples. 
           NOTE: THE LABELS ARE NOT USED FOR TRAINING.
           
    Returns:
        A tuple of (ans, e_list, weight_list) where:
        - ans: Combined outlier scores
        - e_list: List of entropy for each view
        - weight_list: List of sample weights for all views
    """
    e_list, weight_list = [], []
    ans = 0
    
    for i in range(len(score_list)):
        score = torch.tensor(score_list[i])
        if score.min() != score.max():
            sort_score = torch.argsort(score)
            score_pos = score[sort_score[0:int(len(y) - sum(y))]]
            score_neg = score[sort_score[-int(sum(y))::]]
            
            if score_pos.max() != score_pos.min():
                score_pos = (score_pos - score_pos.min()) / (score_pos.max() - score_pos.min()) / 2
            else:
                score_pos = 1 / 4
            
            if score_neg.max() != score_neg.min():
                score_neg = (score_neg - score_neg.min()) / (score_neg.max() - score_neg.min()) / 2 + 1 / 2
            else:
                score_neg = 3 / 4
            
            score[sort_score[0:int(len(y) - sum(y))]] = score_pos
            score[sort_score[-int(sum(y))::]] = score_neg
        else:
            score[:] = 1 / 2
        
        e = -score * torch.log2(score) - (1 - score) * torch.log2(1 - score)
        e = e.nan_to_num(0)
        e_list.append(e)
        weight = 1 - e.mean()
        weight_list.append(weight)
        score_list[i] = score
    
    weight_list = torch.tensor(weight_list) / torch.tensor(weight_list).sum()
    
    for i in range(len(score_list)):
        ans += weight_list[i] * score_list[i].cpu()
    
    return ans, e_list, weight_list.numpy().tolist()


def fit(
    X: torch.Tensor, 
    y: torch.Tensor, 
    l: float, 
    d: float
) -> Tuple[torch.Tensor, List[np.ndarray], List[float]]:
    """
    Outlier detection in all views.
    
    Args:
        X: Tensor of shape (N, M), where N and M are the number of samples 
           and features, respectively.
        y: Tensor of shape (N), the labels of all samples. 
           NOTE: THE LABELS ARE NOT USED FOR TRAINING.
        d: The similarity threshold δ in the paper.
        l: The weighted parameter λ in the paper.
        
    Returns:
        See return value of join().
    """
    score_list = OD_GB(X, y, d, l)
    score_2 = run_FRS(X, y, d, l)
    score_list.append(score_2)
    return join(score_list, y)


def get_uncertainty(
    index: torch.Tensor, 
    e_list: List[np.ndarray], 
    weight_list: List[float]
) -> torch.Tensor:
    """
    Calculate uncertainty for each sample.
    
    Args:
        index: Indices of samples to calculate uncertainty for.
        e_list: List of entropy for each view.
        weight_list: List of sample weights for all views.
        
    Returns:
        Sample weights based on uncertainty.
    """
    ans = 0.0
    for i in range(len(e_list)):
        w, e = weight_list[i], e_list[i][index]
        ans += w * e.cpu()
    return 1 - ans
