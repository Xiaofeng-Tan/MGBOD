
import numpy as np
import torch
import copy as cp
import pandas as pd
import os
import pickle
from units import load_data,get_group_score
from GB import general_GB,get_newM
from FRS_OD import FRS_OD_GB,FRS_OD
from sklearn import svm
from sklearn.metrics import roc_auc_score

def OD_GB(X,y,d,l):
    '''
        Outlier detection in GB views.
        Args:
            X (torch.Tensor): tensor of shape (N, M), where N and M are the number of samples and features, respectively.
            y (torch.Tensor): tensor of shape (N), the lables of all samples. [! NOTE THAT THE LABELS ARE NOT USED FOR TRAINING].
            d (float) : the similarity threshold named $\delta$ in paper.
            l (float) : the weighted parameter named $\lambda$ in paper.
        Returns:
            List[torch.Tensor]: A list of outlier scores for all views.
    '''
    M = torch.cdist(X, X,p=2.0)
    score_list = []
    X_true = cp.deepcopy(X)
    while True:
        GB_list,c,r = general_GB(X,M)
        r_1 = torch.tensor(r)
        X_1 = torch.tensor(c)
        clf = FRS_OD_GB(_deta = d,_lambda = l,_density=True,r=r_1)
        clf.fit(X=X_1,y=y)
        score = clf.predict()
        score = get_group_score(X_true,c,r,score.cpu().numpy())
        score_list.append(score)
        X,M = get_newM(X_1,r_1)
        if M.max()==0:
            break
    return score_list

def run_FRS(X,y,d,l):
    '''
        Outlier detection in original views.
        Args:
            X (torch.Tensor): tensor of shape (N, M), where N and M are the number of samples and features, respectively.
            y (torch.Tensor): tensor of shape (N), the lables of all samples. [! NOTE THAT THE LABELS ARE NOT USED FOR TRAINING].
            d (float) : the similarity threshold named $\delta$ in paper.
            l (float) : the weighted parameter named $\lambda$ in paper.
        Returns:
            torch.Tensor: outlier score
    '''
    clf = FRS_OD(_deta = d,_lambda = l,_density=True)
    clf.fit(X=X,y=y)
    score = clf.predict()
    return score

def join(score_list,y):
    '''
        Calaulate the refined outlier probability
        Args:
            score_list list[torch.Tensor]: A list of outlier scores for all views, including the original view.
            y (torch.Tensor): tensor of shape (N), the lables of all samples. [! NOTE THAT THE LABELS ARE NOT USED FOR TRAINING].
        Returns:
            ans: (torch.Tensor) outlier score
            e_list: list[numpy.array] a list of entropy
            weight_list: list[numpy.array] a list of sample weights for all views
    '''
    e_list,weight_list = [],[]
    ans = 0
    for i in range(len(score_list)):
        score = torch.tensor(score_list[i])
        if score.min() != score.max():
            sort_score = torch.argsort(score)
            score_pos,score_neg = score[sort_score[0:int(len(y)-sum(y))]],score[sort_score[-int(sum(y))::]]
            if score_pos.max() != score_pos.min():
                score_pos = (score_pos - score_pos.min())/(score_pos.max() - score_pos.min()) / 2
            else:
                score_pos = 1/4
            if score_neg.max() != score_neg.min():
                score_neg = (score_neg - score_neg.min())/(score_neg.max() - score_neg.min()) / 2 + 1 / 2
            else:
                score_neg = 3/4
            score[sort_score[0:int(len(y)-sum(y))]], score[sort_score[-int(sum(y))::]] = score_pos,score_neg
        else:
            score[:] = 1/2
        e = - score * torch.log2(score)  - (1-score) * torch.log2(1-score)
        e = e.nan_to_num(0)
        e_list.append(e)
        weight = 1 - e.mean()
        weight_list.append(weight)
        score_list[i] = score
    weight_list = torch.tensor(weight_list) / torch.tensor(weight_list).sum()
    for i in range(len(score_list)):
        ans += weight_list[i] * score_list[i].cpu()
    return ans,e_list, weight_list.numpy().tolist()

def fit(X,y,l,d):
    '''
        Outlier detection in all views.
        Args:
            X (torch.Tensor): tensor of shape (N, M), where N and M are the number of samples and features, respectively.
            y (torch.Tensor): tensor of shape (N), the lables of all samples. [! NOTE THAT THE LABELS ARE NOT USED FOR TRAINING].
            d (float) : the similarity threshold named $\delta$ in paper.
            l (float) : the weighted parameter named $\lambda$ in paper.
        Returns:
            See that of join(score_list,y)
    '''
    score_list,score_2 = OD_GB(X,y,d,l),run_FRS(X,y,d,l)
    score_list.append(score_2)
    return join(score_list,y)

def get_uncertainty(index,e_list,weight_list):
    '''
        Calaulate uncertainty for each samples
        Args:
            ans: (torch.Tensor) outlier score
            e_list: list[numpy.array] a list of entropy
            weight_list: list[numpy.array] a list of sample weights for all views
        Returns:
            (torch.Tensor) sample weights
    '''
    ans = 0.0
    for i in range(len(e_list)):
        w,e = weight_list[i],e_list[i][index]
        ans += w * e.cpu()
    return 1-ans

def test(X,y,_l,_d,_k):
    '''
        Outlier detection in original views.
        Args:
            X (torch.Tensor): tensor of shape (N, M), where N and M are the number of samples and features, respectively.
            y (torch.Tensor): tensor of shape (N), the lables of all samples. [! NOTE THAT THE LABELS ARE NOT USED FOR TRAINING].
            _d (float) : the similarity threshold named $\delta$ in paper.
            _l (float) : the weighted parameter named $\lambda$ in paper.
            _k (float) : the parameter named $\Delta$ in paper.
        Returns:
            list(float): roc scores of FRS + one view, FRS + multi-views, proposed method + one view, proposed method, respectively.
    '''
    # obtain the outlier proportion
    p = sum(y) / len(y)
    # execute the proposed without weighted SVM
    score,e_list,weight_list = fit(X,y,_l,_d)
    # evalue the results
    FRS = roc_auc_score(y_score = score,y_true=y)
    sort_score = np.argsort(score)
    # obtain indices of reliable inliers and outliers
    index_il,index_ol = sort_score[0:int(len(score) * _k * (1-p))],sort_score[-int(len(score) * _k * p)::]

    # obtain pseudo label for training
    y_pseudo = cp.deepcopy(y)
    y_pseudo[index_ol] = 1
    y_pseudo[index_il] = 0
    index = np.concatenate((index_il,index_ol))
    index = torch.from_numpy(np.unique(index))
    X_train = X[index]
    y_train = y_pseudo[index]
    
    # training SVM
    e = get_uncertainty(index,e_list,weight_list)
    clf = svm.SVC(probability = True,class_weight='balanced')
    clf.fit(X_train,y_train,sample_weight=e)
    score = clf.decision_function(X)
    SVM = roc_auc_score(y_score=score,y_true=y)

    # original view + weighted SVM
    score = run_FRS(X,y,_d,_l)
    score,e_list,weight_list = join([score],y)
    sort_score = np.argsort(score)
    index_il,index_ol = sort_score[0:int(len(score) * _k * (1-p))],sort_score[-int(len(score) * _k * p)::]

    y_pseudo = cp.deepcopy(y)
    y_pseudo[index_ol] = 1
    y_pseudo[index_il] = 0
    index = np.concatenate((index_il,index_ol))
    index = torch.from_numpy(np.unique(index))
    X_train = X[index]
    y_train = y_pseudo[index]
    
    e = get_uncertainty(index,e_list,weight_list)
    clf = svm.SVC(probability = True,class_weight='balanced')
    clf.fit(X_train,y_train,sample_weight=e)
    score = clf.decision_function(X)
    SVM_oriview = roc_auc_score(y_score=score,y_true=y)
    
    return roc_auc_score(y_score = run_FRS(X,y,_d,_l),y_true=y),FRS, SVM_oriview, SVM

if __name__ == '__main__':
    np.random.seed(0)
    dir_path = '../datasets/'
    files = os.listdir(dir_path)
    with open('./paramaters.pkl', 'rb') as pkl_file:
        parameters = pickle.load(pkl_file)
    rocs = []
    name = []
    for file in files:
        file_path = os.path.join(dir_path, file)
        if file.endswith('28_pendigits.npz') or file.endswith('xxxx.mat'):
            name.append(file)
            file = file.replace('../datasets/','')
            l,d = parameters[(file,'l')],parameters[(file,'d')]
            X,y = load_data(file_path)
            X, y = torch.from_numpy(X).to(dtype=torch.float32), torch.from_numpy(y).to( dtype=torch.float32)
            # calculate roc scores of FRS + one view, FRS + multi-views, proposed method + one view, proposed method. 
            # THE METHODS WITHOUT DENSITY (l = 0)
            FRS,GB, SVM, SVM_S = test(X,y,0,d,0.7)
            res = [FRS,GB,SVM, SVM_S]
            # THE METHODS WITH DENSITY (l != 0)
            FRS,GB, SVM, SVM_S = test(X,y,l,d,0.7)
            res.extend([FRS,GB,SVM, SVM_S])
            rocs.append(res)            
    #print(np.array(rocs))
    pd.DataFrame(data = rocs,columns=['000','010','001','011','100','110','101','111'],index=name).to_excel("../results/result.xlsx")    

