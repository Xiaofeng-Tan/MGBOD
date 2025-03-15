import numpy as np
import torch

class GB:
    def __init__(self, data, M) :
        self.data = data
        self.m,self.n = data.shape
        self.M = M
        max_indices = torch.argmax(M)
        self.p1_idx, self.p2_idx = divmod(int(max_indices), self.m)
        self.center = self.data.mean(dim=0)
        self.distances = torch.nn.functional.pairwise_distance(data, self.center)
        self.r, _ = self.distances.max(dim=0)
                
    def get_DM(self):
        return self.distances.mean()
    
    def split_1(self):
        if self.r <= 0.001:
            return False,[],[]
        dist_to_p1,dist_to_p2 = self.M[:, self.p1_idx],self.M[:, self.p2_idx]

        mask1 = dist_to_p1 < dist_to_p2
        mask2 = ~mask1
        data_1, data_2 = self.data[mask1],self.data[mask2]
        sub_M_1,sub_M_2 = self.M[:,mask1][mask1],self.M[:,mask2][mask2]
        if sub_M_1.shape[0] == 0 or sub_M_2.shape[0] == 0:
            return False,[],[]
        if sub_M_1.max() == 0 or sub_M_2.max() == 0:
            return False,[],[]
        gb_1,gb_2 = GB(data=data_1,M=sub_M_1),GB(data=data_2,M=sub_M_2)
        
        DM,DM_1,DM_2 = self.get_DM(),gb_1.get_DM(),gb_2.get_DM()
        w_DM = (DM_1 * data_1.shape[0] + DM_2 * data_2.shape[0]) / self.data.shape[0]
        if w_DM < DM:
            return True,gb_1,gb_2 
        #if gb_1.get_DM() < DM or gb_2.get_DM() < DM :
        #    return True,gb_1,gb_2
        return False,[],[]
    
    def split_2(self):
        dist_to_p1,dist_to_p2 = self.M[:, self.p1_idx],self.M[:, self.p2_idx]
        mask1 = dist_to_p1 < dist_to_p2
        mask2 = ~mask1
        data_1, data_2 = self.data[mask1],self.data[mask2]
        sub_M_1,sub_M_2 = self.M[:,mask1][mask1],self.M[:,mask2][mask2]
        gb_1,gb_2 = GB(data=data_1,M=sub_M_1),GB(data=data_2,M=sub_M_2)
        return gb_1,gb_2
    
    def get_circles(self):
        return self.center.tolist(),self.r.item()
    
    def get_data(self):
        return self.data
    
def get_GB_r_c(GB_list):
    c,r = [],[]
    for gb in GB_list:
        c.append(gb.get_circles()[0])
        r.append(gb.get_circles()[1])
    return c,r

def general_GB(X:torch.tensor,M:torch.tensor = None):
    if M == None:
        if len(X.shape) == 1:
            X = X.view(-1,1)
        M = torch.cdist(X, X,p=2.0)
    GB_init = GB(data=X,M=M)
    stack = [GB_init]
    GB_list_1 = []
    while True:
        gb = stack.pop()
        flag,gb_1,gb_2 = gb.split_1()
        if flag:
            stack.append(gb_1)
            stack.append(gb_2)
        else:
            GB_list_1.append(gb)
        if len(stack) == 0:
            break
    c,r = get_GB_r_c(GB_list=GB_list_1)
    mean_r,middle_r = sum(r) / len(r), np.median(r)
    GB_list = []
    for i in range(len(GB_list_1)):
        gb = GB_list_1[i]
        if r[i] >= max(mean_r,middle_r):
            gb_1,gb_2 = gb.split_2()
            GB_list.append(gb_1)
            GB_list.append(gb_2)
        else:
            GB_list.append(gb)
    return GB_list,c,r

def get_newM(X:torch.tensor,r:torch.tensor):
    def get_matrix(X:torch.tensor,r:torch.tensor):
        n = X.shape[0]
        matrix = torch.cdist(X, X,p=2.0)
        matrix.fill_diagonal_(0)
        d_r = torch.cdist(r.view(n, 1), (-1*r).view(n, 1))
        d_r = d_r - torch.diag_embed(torch.diag(d_r))
        matrix -= d_r
        return matrix

    M = get_matrix(X,r)
    M[M < 0] = 0
    return X,M