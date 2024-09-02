import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import copy as cp

device = torch.device('cpu')

class FRS_OD():
    def __init__(self,_deta:float=0.8,_lambda:float=1,_density:bool=True):
        self.deta,self._lambda,self._density = _deta,_lambda,_density
    
    def fit(self,X:torch.tensor,y:torch.tensor):
        self.X,self.y = X.to(device),y.to(device)
        self.X = torch.nan_to_num(self.X)
        
    def evaluation(self):
        score,y = self.score,self.y
        return roc_auc_score(y_score=score,y_true=y)
    
    def get_score(self,attribute_sub):
        M = torch.ones((self.X.shape[0],self.X.shape[0]),device=device)
        score = torch.ones((self.X.shape[0]),device=device)
        for i in attribute_sub:
            M = torch.minimum(self.get_matrix(i),M)
            w = self.get_sig(M)
            score += M.mean(dim = 1) * w
        score /= len(attribute_sub)
        return 1 - score

    def get_sig(self,M):
        eq_class = M.sum(dim=1)
        entropy = torch.log(eq_class/self.X.shape[0])
        sig_unlabel = torch.mean(entropy)
        return -sig_unlabel

    def predict(self):
        attribute_sub = torch.tensor([self.get_sig(self.get_matrix(i)) for i in range(self.X.shape[1])]).argsort(descending=True)
        self.score = self.get_score(attribute_sub)
        return self.score

    def get_matrix(self,i):
        X = self.X[:,i]
        def is_categorical():
            unique_values = len(torch.unique(X))
            if unique_values < self.X.shape[0]/50 and torch.equal(X, X.floor()):
                return True
            else:
                return False
        n = X.shape[0]
        if is_categorical():
            matrix = torch.eq(X.view(n, 1), X.view(1, n)).float()
        else:
            self.X[:,i] = (self.X[:,i] - self.X[:,i].min()) / (self.X[:,i].max() - self.X[:,i].min())
            X = (X - X.min()) / (X.max() - X.min())
            matrix = torch.cdist(X.view(n, 1), X.view(n, 1))
            std = torch.std(X)
            t = std / self.deta
            matrix[matrix > t] = 1
            matrix = 1 - matrix
        if self._density:
            den = matrix.mean(dim = 1)
            diff = torch.cdist(den.view(n,1),den.view(n,1))
            rel_den = torch.exp(-1 * self._lambda * diff)
            matrix = torch.multiply(rel_den,matrix)
        return matrix
    

class FRS_OD_GB(FRS_OD):
    def __init__(self,r:torch.tensor,_deta:float=0.8,_lambda:float=1,_density:bool=True):
        super().__init__(_deta,_lambda,_density)
        self.r = r.to(device)

    def evaluation(self):
        raise NotImplementedError("This method has been disabled in the GB")
        
    def get_matrix(self,i):
        r = torch.clone(self.r)
        if (self.X[:,i].max() - self.X[:,i].min()) != 0:
            t = (self.X[:,i].max() - self.X[:,i].min())
            self.X[:,i] = (self.X[:,i] - self.X[:,i].min()) / t
            r /= t
        X = self.X[:,i]
        n = self.X.shape[0]
        matrix = torch.cdist(X.view(n, 1), X.view(n, 1))
        r = torch.divide(torch.pow(r,1 / self.X.shape[1]),self.X.shape[1])
        n = self.r.shape[0]
        d_r = torch.cdist(r.view(n, 1), (-1*r).view(n, 1))
        d_r = d_r - torch.diag_embed(torch.diag(d_r))
        matrix -= d_r
        matrix[matrix < 0] = 0
        
        std = torch.std(X)
        t = std / self.deta
        matrix[matrix > t] = 1
        matrix = 1 - matrix
        if self._density:
            den = matrix.mean(dim = 1)
            diff = torch.cdist(den.view(n,1),den.view(n,1))
            rel_den = torch.exp(-1 * self._lambda * diff)
            matrix = torch.multiply(rel_den,matrix)

        return matrix

