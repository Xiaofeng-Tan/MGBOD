import numpy as np
import warnings
from mat4py import loadmat
import random
import torch
warnings.filterwarnings('ignore')

def load_data(path):
    try:
        print(path)
        data = loadmat(path)
    except:
        data = np.load(path, allow_pickle=True)
    try:
        data = np.array(data['trandata'])
    except:
        X = np.array(data['X'])
        #X = MinMaxScaler().fit_transform(X)
        y = np.array(data['y'])
        #print(y)
        if type(y[0]) == np.ndarray:
            yy = []
            for i in range(len(y)):
                yy.append(y[i][0])
        #input()
            y = np.array(yy)
        return X, y

    X = data[:,0:-1]
    y = data[:,-1]
    if max(y) != 1:
        y -= min(y)
    if sum(y) > len(y) / 2:
        for i in range(len(y)):
            y[i] = 1 if y[i] == 0 else 0
    return X, y

def downsample(p, y, n):
    if p == 0:
        return [],[]
    random.seed(n)
    pos_sample = [i for i in range(len(y)) if y[i] == 0]
    n = int(p * len(y))
    n = 2 if n == 0 else n
    index = random.sample(pos_sample,n)
    labels = [0 for i in range(len(y))]
    labels = np.array(labels)
    labels[index] = 1
    return labels.tolist(), index
    
def get_group_score(data, centers, radii, score):
    score = torch.from_numpy(score)
    max_val,min_val = torch.max(score),torch.min(score)
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
