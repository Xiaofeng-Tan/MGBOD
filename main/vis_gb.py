import time
from GB import general_GB,get_newM
from sklearn.preprocessing import MinMaxScaler
import torch
from units import plot_cir_p,load_data
import copy as cp
def run_GB(X):
    M = torch.cdist(X, X,p=2.0)
    k = 0
    while True:
        GB_list,c,r = general_GB(X,M)
        plot_cir_p(X,c,r,k)
        k += 1
        print(k)
        r_1 = torch.tensor(r)
        X_1 = torch.tensor(c)
        X,M = get_newM(X_1,r_1)
        if M.max()==0:
            break

if __name__ == '__main__':
    X,y = load_data("../datasets/2.npz")
    X = MinMaxScaler().fit_transform(X)
    X, y = torch.from_numpy(X).to(dtype=torch.float32), torch.from_numpy(y).to( dtype=torch.float32)
    plot_cir_p(X,0,0,10)
    run_GB(X)
