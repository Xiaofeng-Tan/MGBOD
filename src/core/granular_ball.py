"""
Granular Ball generation and view update module.

This module implements the multi-scale view generation method based on 
granular-ball computing to collaboratively identify group outliers at 
different granularity levels.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


class GB:
    """
    Granular Ball class for representing data clusters.
    
    A granular ball is defined by its center and radius, representing a 
    cluster of data points in the feature space.
    
    Attributes:
        data (torch.Tensor): The data points contained in this granular ball.
        M (torch.Tensor): The distance matrix for the data points.
        center (torch.Tensor): The center of the granular ball.
        r (torch.Tensor): The radius of the granular ball.
    """
    
    def __init__(self, data: torch.Tensor, M: torch.Tensor):
        """
        Initialize a Granular Ball.
        
        Args:
            data: Tensor of shape (N, M) containing the data points.
            M: Distance matrix of shape (N, N).
        """
        self.data = data
        self.m, self.n = data.shape
        self.M = M
        max_indices = torch.argmax(M)
        self.p1_idx, self.p2_idx = divmod(int(max_indices), self.m)
        self.center = self.data.mean(dim=0)
        self.distances = torch.nn.functional.pairwise_distance(data, self.center)
        self.r, _ = self.distances.max(dim=0)
                
    def get_DM(self) -> torch.Tensor:
        """
        Get the mean distance from center.
        
        Returns:
            The mean distance of all points to the center.
        """
        return self.distances.mean()
    
    def split_1(self) -> Tuple[bool, Optional['GB'], Optional['GB']]:
        """
        Attempt to split the granular ball based on quality improvement.
        
        This method splits the ball if the weighted mean distance of the 
        resulting balls is less than the current mean distance.
        
        Returns:
            A tuple of (success, gb_1, gb_2) where success indicates if 
            the split was performed, and gb_1, gb_2 are the resulting balls.
        """
        if self.r <= 0.001:
            return False, [], []
        
        dist_to_p1 = self.M[:, self.p1_idx]
        dist_to_p2 = self.M[:, self.p2_idx]

        mask1 = dist_to_p1 < dist_to_p2
        mask2 = ~mask1
        data_1, data_2 = self.data[mask1], self.data[mask2]
        sub_M_1, sub_M_2 = self.M[:, mask1][mask1], self.M[:, mask2][mask2]
        
        if sub_M_1.shape[0] == 0 or sub_M_2.shape[0] == 0:
            return False, [], []
        if sub_M_1.max() == 0 or sub_M_2.max() == 0:
            return False, [], []
        
        gb_1, gb_2 = GB(data=data_1, M=sub_M_1), GB(data=data_2, M=sub_M_2)
        
        DM = self.get_DM()
        DM_1, DM_2 = gb_1.get_DM(), gb_2.get_DM()
        w_DM = (DM_1 * data_1.shape[0] + DM_2 * data_2.shape[0]) / self.data.shape[0]
        
        if w_DM < DM:
            return True, gb_1, gb_2 
        return False, [], []
    
    def split_2(self) -> Tuple['GB', 'GB']:
        """
        Force split the granular ball into two parts.
        
        This method always splits the ball regardless of quality improvement.
        
        Returns:
            A tuple of (gb_1, gb_2), the two resulting granular balls.
        """
        dist_to_p1 = self.M[:, self.p1_idx]
        dist_to_p2 = self.M[:, self.p2_idx]
        mask1 = dist_to_p1 < dist_to_p2
        mask2 = ~mask1
        data_1, data_2 = self.data[mask1], self.data[mask2]
        sub_M_1, sub_M_2 = self.M[:, mask1][mask1], self.M[:, mask2][mask2]
        gb_1, gb_2 = GB(data=data_1, M=sub_M_1), GB(data=data_2, M=sub_M_2)
        return gb_1, gb_2
    
    def get_circles(self) -> Tuple[List[float], float]:
        """
        Get the center and radius of the granular ball.
        
        Returns:
            A tuple of (center, radius).
        """
        return self.center.tolist(), self.r.item()
    
    def get_data(self) -> torch.Tensor:
        """
        Get the data points in this granular ball.
        
        Returns:
            The data tensor.
        """
        return self.data


def get_GB_r_c(GB_list: List[GB]) -> Tuple[List[List[float]], List[float]]:
    """
    Extract centers and radii from a list of granular balls.
    
    Args:
        GB_list: List of GB objects.
        
    Returns:
        A tuple of (centers, radii).
    """
    c, r = [], []
    for gb in GB_list:
        c.append(gb.get_circles()[0])
        r.append(gb.get_circles()[1])
    return c, r


def general_GB(X: torch.Tensor, M: torch.Tensor = None) -> Tuple[List[GB], List[List[float]], List[float]]:
    """
    Generate granular balls from data.
    
    This function implements the multi-scale view generation method based on 
    granular-ball computing.
    
    Args:
        X: Tensor of shape (N, M), the input data.
        M: Optional distance matrix. If None, it will be computed.
        
    Returns:
        A tuple of (GB_list, centers, radii).
    """
    if M is None:
        if len(X.shape) == 1:
            X = X.view(-1, 1)
        M = torch.cdist(X, X, p=2.0)
    
    GB_init = GB(data=X, M=M)
    stack = [GB_init]
    GB_list_1 = []
    
    while True:
        gb = stack.pop()
        flag, gb_1, gb_2 = gb.split_1()
        if flag:
            stack.append(gb_1)
            stack.append(gb_2)
        else:
            GB_list_1.append(gb)
        if len(stack) == 0:
            break
    
    c, r = get_GB_r_c(GB_list=GB_list_1)
    mean_r = sum(r) / len(r)
    middle_r = np.median(r)
    
    GB_list = []
    for i in range(len(GB_list_1)):
        gb = GB_list_1[i]
        if r[i] >= max(mean_r, middle_r):
            gb_1, gb_2 = gb.split_2()
            GB_list.append(gb_1)
            GB_list.append(gb_2)
        else:
            GB_list.append(gb)
    
    return GB_list, c, r


def get_newM(X: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the updated distance matrix for the next scale.
    
    This function adjusts distances based on granular ball radii to enable
    multi-scale analysis.
    
    Args:
        X: Tensor of granular ball centers.
        r: Tensor of granular ball radii.
        
    Returns:
        A tuple of (X, M) where M is the adjusted distance matrix.
    """
    def get_matrix(X: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        matrix = torch.cdist(X, X, p=2.0)
        matrix.fill_diagonal_(0)
        d_r = torch.cdist(r.view(n, 1), (-1 * r).view(n, 1))
        d_r = d_r - torch.diag_embed(torch.diag(d_r))
        matrix -= d_r
        return matrix

    M = get_matrix(X, r)
    M[M < 0] = 0
    return X, M
