import copy
import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.spatial.distance import pdist, squareform


def init_Y_confidence(target):
    res = []
    p_target = copy.deepcopy(target)
    row, col = p_target.shape
    for i in range(row):
        sing_target = p_target[i, :]
        count = sing_target.sum()
        init_y = 1 / count
        sing_target[sing_target > 0] = init_y
        if sing_target.shape[0] == 1:
            x = sing_target.tolist()[0]
        else:
            x = sing_target.tolist()
        res.append(x)
    res = np.array(res)
    return res

def get_candidate(target):
    res = []
    p_target = copy.deepcopy(target)
    row, col = p_target.shape
    for i in range(row):
        p_target_single = p_target[i, :]
        if p_target_single.shape[0] == 1:
            sing_target = np.array(p_target_single[0])
            indexs = np.argwhere(sing_target == 1)
            indexs = [x[1] for x in indexs.tolist()]
            res.append(indexs)
        else:
            sing_target = np.array(p_target_single)
            indexs = np.argwhere(sing_target == 1).flatten().tolist()
            res.append(indexs)
    return res

def Get_K_Neighbors(features, k):
    pdist_martrix = pdist(features, metric='euclidean')
    
    martrix = squareform(pdist_martrix)
    topK_neighbors = np.argsort(martrix, axis = 0)[1: k + 1].T
    return topK_neighbors

def calc_acc(predict_Y, target_list):
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(predict_Y, target_list.flatten().tolist())
    # print(acc)
    return acc

# 获取真实标记的单标记版本
def get_single_target(target):
    res = []
    sample_size, label_size = target.shape
    for i in range(sample_size):
        tmp_single = target[i, :]
        tmp_single = tmp_single.tolist()
        if len(tmp_single) == 1:
            index = tmp_single[0].index(1)
        else:
            index = tmp_single.index(1)
        res.append([index])
    res = np.array(res)
    return res

'''
    KNN升级版本（使用Y—confidence）
'''
def KNN_disambiguation(features, partial_target, k, Q, candidate, Yconfidence):
    res = []
    M = features.shape[0]
    Neighbor_votes = np.zeros(M)
    top_k_neigs = Get_K_Neighbors(features, k)
    sample_size, k_size = top_k_neigs.shape
    wr = [k - i for i in range(k)]
    target_size = partial_target.shape[1]
    predict_Y = []
    for i in range(sample_size):
        k_neighbors = top_k_neigs[i, :]
        sumY = np.zeros(Q)
        count1 = np.zeros(Q)
        percandidate = candidate[i]
        sizecandidate = len(percandidate)
        
        for t in range(sizecandidate):
            indexlabel = percandidate[t]
            for j in range(k):
                indexneighbor = k_neighbors[j]
                sumY[indexlabel] = sumY[indexlabel] + Yconfidence[indexneighbor, indexlabel] * wr[j]
                if Yconfidence[indexneighbor, indexlabel] > 0:
                    count1[indexlabel] = count1[indexlabel] + 1
        col = np.argwhere(sumY == max(sumY)).tolist()
        col = [val[0] for val in col]
        count2 = len(col)
        count3 = count1[col]
        count4 = max(count3)
        count4_size = len(np.argwhere(count3 == count4).tolist())
        count5 = np.argwhere(count1 == count4).tolist()
        count5 = [val[0] for val in count5]
        # print(count5)
        res.append([])
        if (max(sumY) == 0):
            res[i] = candidate[i]
            Neighbor_votes[i] = 0
        elif count2 == 1:
            res[i] = col
            Neighbor_votes[i] = count3
        elif (count4_size == 1):
            Neighbor_votes[i] = count4
            res[i] = count5
        else:
            Neighbor_votes[i] = count4
            res[i] = count5

        res[i] = res[i][0]

    return res

'''
    基础KNN
'''
def KNN_disambiguation_simple(features, partial_target, k, Q, candidate):
    top_k_neigs = Get_K_Neighbors(features, k)
    predict_Y = []
    sample_size, _ = top_k_neigs.shape
    for i in range(sample_size):
        k_neighbors = top_k_neigs[i, :]
        percandidate = candidate[i]
        sizecandidate = len(percandidate)
        sumY = np.zeros(Q)
        for t in range(sizecandidate):
            indexlabel = percandidate[t]
            for j in range(k):
                indexneighbor = k_neighbors[j]
                sumY[indexlabel] = sumY[indexlabel] + partial_target[indexneighbor, indexlabel]
        # print(sumY)
        max_val = max(sumY)
        if max_val != 0:
            predict_label = np.argmax(sumY)
        else:
            predict_label = percandidate[0]
        # print(predict_label)
        predict_Y.append(predict_label)
    return predict_Y

# 处理稀疏矩阵
def process_csc_matrix(labels, partial_target):
    if isinstance(labels, csc_matrix):
        # print('labels 稀疏矩阵')
        target = labels.todense().T
    else:
        target = labels.T

    if isinstance(partial_target, csc_matrix):
        # print('partial_target 稀疏矩阵')
        partial_target = partial_target.todense().T
    else:
        partial_target = partial_target.T
    return target, partial_target