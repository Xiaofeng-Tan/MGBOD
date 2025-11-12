import numpy as np
import matplotlib.pyplot as plt
import warnings
from mat4py import loadmat
import random
warnings.filterwarnings('ignore')
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})
def analyse(score, y, path):
    """
    计算固定离群分数前a%的样本为离群点的情况下，检出率和误检率的表格。
    
    参数：
        score: 一个代表每一个样本的离群分数的序列。
        y: 一个代表每一个样本是否为离群点的序列，0代表内点，1代表外点。
    
    返回：
        一个Pandas DataFrame对象，包含检出率和误检率表格。
    """
    assert len(score) == len(y), "score和y的长度必须相同"
    score_ord = np.argsort(score)[::-1]
    
    results = []
    y_pred = np.array([0] * len(score))
    for a in range(5, 101, 5):
        num_outliers = int(len(score) * a / 100)
        y_pred[score_ord[0:num_outliers]] = 1
        TN, FP, FN, TP = confusion_matrix(y_true=y, y_pred=y_pred).ravel()
        DR = TP / (TP + FP)
        FAR = TP / (TP + FN)
        results.append([DR, FAR])
    
    results_df = pd.DataFrame(results, columns=["P", "R"])
    results_df.index = [f"{i}%" for i in range(5, 101, 5)]
    #print(results_df.mean())
    results_df.to_excel(path)
    return results_df

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

def plot_cir_p(X, c, r,k):
    if c == 0:
        # 绘制点
        x = [p[0] for p in X]
        y = [p[1] for p in X]
        plt.scatter(x, y, s=10)
        #plt.axis('tight')
        plt.axis('equal')
        plt.savefig("../fig/O_"+str(k)+'.pdf')
        plt.savefig("../fig/O_"+str(k)+'.png',dpi = 1500)
        plt.cla()
    else:
        # 绘制点
        x = [p[0] for p in X]
        y = [p[1] for p in X]
        plt.scatter(x, y, s=10)
        
        # 绘制圆
        for i in range(len(c)):
            circle = plt.Circle(c[i], r[i], color='r', fill=False)
            plt.gcf().gca().add_artist(circle)
        #plt.axis('tight')
        plt.axis('equal')
        plt.savefig("../fig/O_"+str(k)+'.pdf')
        plt.savefig("../fig/O_"+str(k)+'.png',dpi = 1500)
        plt.cla()
    
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
