"""
Fuzzy Rough Sets based Outlier Detection module.

This module implements the FRS-based outlier detection methods and their variants 
in Granular Ball views.
"""

import torch
from sklearn.metrics import roc_auc_score

device = torch.device('cpu')


class FRS_OD:
    """
    Fuzzy Rough Sets based Outlier Detection.
    
    This class implements the fuzzy rough sets approach that incorporates 
    relative fuzzy granule density to enhance the detection of local outliers.
    
    Attributes:
        deta (float): The similarity threshold δ in the paper.
        _lambda (float): The weighted parameter λ in the paper.
        _density (bool): Whether to use density-based approach.
    """
    
    def __init__(self, _deta: float = 0.8, _lambda: float = 1, _density: bool = True):
        """
        Initialize FRS_OD.
        
        Args:
            _deta: The similarity threshold δ in the paper.
            _lambda: The weighted parameter λ in the paper.
            _density: Whether to use density-based approach.
        """
        self.deta = _deta
        self._lambda = _lambda
        self._density = _density
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the model with training data.
        
        Args:
            X: Tensor of shape (N, M), where N and M are the number of samples 
               and features, respectively.
            y: Tensor of shape (N), the labels of all samples. 
               NOTE: THE LABELS ARE NOT USED FOR TRAINING.
        """
        self.X = X.to(device)
        self.y = y.to(device)
        self.X = torch.nan_to_num(self.X)
        
    def evaluation(self) -> float:
        """
        Evaluate the model using ROC AUC score.
        
        Returns:
            The ROC AUC score.
        """
        score, y = self.score, self.y
        return roc_auc_score(y_score=score, y_true=y)
    
    def get_score(self, attribute_sub: torch.Tensor) -> torch.Tensor:
        """
        Calculate outlier scores based on attribute subsets.
        
        Args:
            attribute_sub: Tensor of attribute indices sorted by significance.
            
        Returns:
            Outlier scores for all samples.
        """
        M = torch.ones((self.X.shape[0], self.X.shape[0]), device=device)
        score = torch.ones((self.X.shape[0]), device=device)
        for i in attribute_sub:
            M = torch.minimum(self.get_matrix(i), M)
            w = self.get_sig(M)
            score += M.mean(dim=1) * w
        score /= len(attribute_sub)
        return 1 - score

    def get_sig(self, M: torch.Tensor) -> float:
        """
        Calculate the significance of the fuzzy equivalence relation.
        
        Args:
            M: The fuzzy relation matrix.
            
        Returns:
            The significance value.
        """
        eq_class = M.sum(dim=1)
        entropy = torch.log(eq_class / self.X.shape[0])
        sig_unlabel = torch.mean(entropy)
        return -sig_unlabel

    def predict(self) -> torch.Tensor:
        """
        Predict outlier scores for all samples.
        
        Returns:
            Tensor of outlier scores.
        """
        attribute_sub = torch.tensor([
            self.get_sig(self.get_matrix(i)) for i in range(self.X.shape[1])
        ]).argsort(descending=True)
        self.score = self.get_score(attribute_sub)
        return self.score

    def get_matrix(self, i: int) -> torch.Tensor:
        """
        Compute the fuzzy relation matrix for attribute i.
        
        Args:
            i: The attribute index.
            
        Returns:
            The fuzzy relation matrix.
        """
        X = self.X[:, i]
        
        def is_categorical():
            unique_values = len(torch.unique(X))
            if unique_values < self.X.shape[0] / 50 and torch.equal(X, X.floor()):
                return True
            else:
                return False
        
        n = X.shape[0]
        if is_categorical():
            matrix = torch.eq(X.view(n, 1), X.view(1, n)).float()
        else:
            self.X[:, i] = (self.X[:, i] - self.X[:, i].min()) / (self.X[:, i].max() - self.X[:, i].min())
            X = (X - X.min()) / (X.max() - X.min())
            matrix = torch.cdist(X.view(n, 1), X.view(n, 1))
            std = torch.std(X)
            t = std / self.deta
            matrix[matrix > t] = 1
            matrix = 1 - matrix
        
        if self._density:
            den = matrix.mean(dim=1)
            diff = torch.cdist(den.view(n, 1), den.view(n, 1))
            rel_den = torch.exp(-1 * self._lambda * diff)
            matrix = torch.multiply(rel_den, matrix)
        
        return matrix
    

class FRS_OD_GB(FRS_OD):
    """
    FRS-based Outlier Detection in Granular Ball views.
    
    This class extends FRS_OD to work with granular ball representations,
    enabling multi-scale outlier detection.
    
    Attributes:
        r (torch.Tensor): The radii of granular balls.
    """
    
    def __init__(self, r: torch.Tensor, _deta: float = 0.8, _lambda: float = 1, _density: bool = True):
        """
        Initialize FRS_OD_GB.
        
        Args:
            r: Tensor of granular ball radii.
            _deta: The similarity threshold δ in the paper.
            _lambda: The weighted parameter λ in the paper.
            _density: Whether to use density-based approach.
        """
        super().__init__(_deta, _lambda, _density)
        self.r = r.to(device)

    def evaluation(self):
        """This method is disabled in GB mode."""
        raise NotImplementedError("This method has been disabled in the GB")
        
    def get_matrix(self, i: int) -> torch.Tensor:
        """
        Compute the fuzzy relation matrix for attribute i in GB view.
        
        Args:
            i: The attribute index.
            
        Returns:
            The fuzzy relation matrix adjusted for granular balls.
        """
        r = torch.clone(self.r)
        if (self.X[:, i].max() - self.X[:, i].min()) != 0:
            t = (self.X[:, i].max() - self.X[:, i].min())
            self.X[:, i] = (self.X[:, i] - self.X[:, i].min()) / t
            r /= t
        X = self.X[:, i]
        n = self.X.shape[0]
        matrix = torch.cdist(X.view(n, 1), X.view(n, 1))
        r = torch.divide(torch.pow(r, 1 / self.X.shape[1]), self.X.shape[1])
        n = self.r.shape[0]
        d_r = torch.cdist(r.view(n, 1), (-1 * r).view(n, 1))
        d_r = d_r - torch.diag_embed(torch.diag(d_r))
        matrix -= d_r
        matrix[matrix < 0] = 0
        
        std = torch.std(X)
        t = std / self.deta
        matrix[matrix > t] = 1
        matrix = 1 - matrix
        
        if self._density:
            den = matrix.mean(dim=1)
            diff = torch.cdist(den.view(n, 1), den.view(n, 1))
            rel_den = torch.exp(-1 * self._lambda * diff)
            matrix = torch.multiply(rel_den, matrix)

        return matrix
