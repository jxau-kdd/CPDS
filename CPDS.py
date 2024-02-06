import copy
import random
import numpy as np
from tqdm import tqdm
from math import ceil
import scipy.io as sio
from scipy import spatial
from collections import Counter
from scipy.spatial.distance import pdist, squareform

def normalize(features):
    ConditionalLength = features.shape[1]
    for i in range(ConditionalLength):
        MaxValue = max(features[:, i])
        MinValue = min(features[:, i])
        for j in range(features.shape[0]):
            features[j, i] = (features[j, i] - MinValue) / (MaxValue - MinValue)
    return features

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

'''
    决策类编号(编号方案2)
'''
def get_number_label(ldl):
    sample_size, label_size = ldl.shape
    # print(sample_size, label_size)
    result = []
    for i in range(sample_size):
        tag_labels = np.zeros((label_size))
        ldl_label = ldl[i, :]
        idxs = np.where(ldl_label != 0)
        valid_tag = ldl_label[idxs].tolist()
        if len(valid_tag) == 0:
            pass
        else:
            valid_tag.sort(reverse=True)
            number = 1
            for tag in valid_tag:
                tag_idxs = np.where(ldl_label==tag)[0]
                for j in tag_idxs:
                    if tag_labels[j] == 0:
                        tag_labels[j] = number
                        number += 1
                        break
        result.append(tag_labels.tolist())
    return np.array(result)

'''
    决策类编号(编号方案2)
'''
def get_number_label_2(ldl, threshold=0.5):
    sample_size, label_size = ldl.shape
    # print(sample_size, label_size)
    result = []
    for i in range(sample_size):
        tag_labels = np.zeros((label_size))
        ldl_label = ldl[i, :].tolist()
        for j in range(label_size):
            tag = ldl_label[j]
            if tag >= threshold:
                tag_labels[j] = 1
            else:
                tag_labels[j] = 2
        result.append(tag_labels.tolist())
    return np.array(result)

'''
    返回整个实例集合下的邻域阈值
'''
def GetThreshold(features, ParameterOmega):
    Temp = 0
    cond_feature_len = features.shape[1]
    for i in range(cond_feature_len):
        std_fea = np.std(features[:, i])
        T = float(std_fea / ParameterOmega)
        Temp += T
    Result = float((Temp / cond_feature_len) / ParameterOmega)
    return Result

'''
    获取所有实例下的邻域集合
'''
def GetAllInstanceNeigborhoodList(features, ParameterOmega, indexs = 'ALL'):
    res = []

    if indexs != 'ALL':
        tmp_fearures = features[:, indexs]
    else:
        tmp_fearures = features

    sample_size = tmp_fearures.shape[0]

    Threshold = GetThreshold(tmp_fearures, ParameterOmega)
    Distances = squareform(pdist(features))
    neig_sorts = np.argsort(Distances)
    
    for i in range(sample_size):
        T = []
        # Vector_1 = tmp_fearures[i, :]
        neigs = neig_sorts[i, :]
        for neig in neigs:
            # Vector_2 = tmp_fearures[j, :]
            Distance = Distances[i, neig]
            if Distance <= Threshold:
                T.append(neig)
            else:
                break
        res.append(T)
    return res, Threshold

'''
    邻域等价类划分
'''
def neig_equival_class(features, Omega):
    
    Neigborhood_f_Matrix, Threshold = GetAllInstanceNeigborhoodList(features, Omega)
    # print(Neigborhood_f_Matrix)
    return Neigborhood_f_Matrix

'''
    决策类划分(划分方案1)
'''
def decision_class_1(ldl):
    result = []
    tag_labels = get_number_label(ldl)
    label_num = ldl.shape[1]
    for i in range(label_num):
        label = tag_labels[:, i]
        idxs = np.where(label == 1)[0]
        if len(idxs) > 0:
            result.append(idxs.tolist())
    return result

'''
    决策类划分(划分方案2)
'''
def decision_class_2(ldl):
    result = []
    tag_labels = get_number_label(ldl)
    label_num = ldl.shape[1]
    for i in range(label_num):
        label = tag_labels[:, i]
        min_label = min(label)
        idxs = np.where(label == min_label)[0]
        if len(idxs) > 0:
            result.append(idxs.tolist())
    return result

'''
    决策类划分(划分方案3)
'''
def decision_class_3(ldl):
    result = []
    tag_labels = get_number_label(ldl)
    label_num = ldl.shape[1]
    for i in range(label_num):
        label = tag_labels[:, i]
        cluters = dict(Counter(label.tolist()))
        for key in cluters.keys():
            idxs = np.where(label == key)[0]
            if len(idxs) > 0:
                result.append(idxs.tolist())
    return result

'''
    决策类划分(划分方案4)
'''
def decision_class_4(ldl):
    result = []
    threshold = get_threshold(ldl)
    tag_labels = get_number_label_2(ldl, threshold)
    label_num = ldl.shape[1]
    for i in range(label_num):
        label = tag_labels[:, i]
        idxs = np.where(label == 1)[0]
        if len(idxs) > 0:
            result.append(idxs.tolist())
    return result

'''
    计算标记阈值
'''
def get_threshold(ldl):
    res = 0
    rate = 0.1
    tmp = ldl.flatten().tolist()
    label_len = len(tmp)
    idx = ceil(label_len * rate)
    tmp.sort(reverse=True)
    res = tmp[idx - 1]
    return res

'''
    计算下近似
'''
def calc_Lower_approximation(equival_classes, dec_classes):
    sample_size = len(equival_classes)
    dec_size = len(dec_classes)
    count = 0
    for i in range(sample_size):
        equ_class = equival_classes[i]

        for j in range(dec_size):
            dec_class = dec_classes[j]
            if set(equ_class) <= set(dec_class):
                count += 1
    return count

'''
    计算依赖度
'''
def calc_dep(features, dec_classes, omega):
    equival_classes = neig_equival_class(features, omega)
    low_appr = calc_Lower_approximation(equival_classes, dec_classes)
    yilai = low_appr / features.shape[0]
    return yilai

'''
    启发式搜索方案（废弃）
'''
def select_best_feature(features, dec_classes, base, omega):
    best_feature_idx = -1
    max_score = -1
    
    for idx in base:
        tmp_feature_idx = copy.deepcopy(base)
        tmp_feature_idx.remove(idx)
        important_score = calc_dep(features[:, tmp_feature_idx], dec_classes, omega)
        if important_score > max_score:
            max_score = important_score
            best_feature_idx = idx
    return best_feature_idx

'''
    计算邻居
'''
def Get_K_Neighbors(features, k):
    pdist_martrix = pdist(features, metric='euclidean')
    
    martrix = squareform(pdist_martrix)
    topK_neighbors = np.argsort(martrix, axis = 0)[1: k + 1].T
    return topK_neighbors

'''
    计算余弦相似度
'''
def calc_cosine(vec_1, vec_2):
    res = 1 - spatial.distance.cosine(vec_1, vec_2)
    return res

'''
    计算same score（根据样本与邻居的预测标记是否一致进行计算）
'''
def calc_same_score_Neig(features, predict_Y, idx, same_k, Omega):
    select_features = [i for i in range(features.shape[1]) if i != idx]
    # select_features.append(idx)
    X = features[:, select_features]
    sample_size = X.shape[0]
    feature_len = X.shape[1]
    score_count = 0
    k = same_k
    neigs = Get_K_Neighbors(X, k)
    # if feature_len < 20:
    #     neigs = Get_K_Neighbors(X, k)
    # else:
    #     neigs, Threshold = GetAllInstanceNeigborhoodList(X, Omega)
    for i in range(sample_size):
        samples = neigs[i]
        # print(samples)
        base_target = predict_Y[i]
        count = 0
        for j in range(len(samples)):
            else_target = predict_Y[samples[j]]
            if base_target == else_target:
                count += 1
        score_count += count / k
    return score_count

'''
    计算same score（根据标记置信度的余弦相似度进行计算）
'''
def calc_same_score_Neig_2(features, partial_labels, idx, same_k, Omega):
    select_features = [i for i in range(features.shape[1]) if i != idx]
    # select_features.append(idx)
    X = features[:, select_features]
    sample_size = X.shape[0]
    feature_len = X.shape[1]
    score_count = 0
    k = same_k
    neigs = Get_K_Neighbors(X, k)

    for i in range(sample_size):
        samples = neigs[i]
        # print(samples)
        self_target = partial_labels[i, :]
        count = 0
        for j in range(len(samples)):
            else_target = partial_labels[samples[j], :]
            count += calc_cosine(self_target, else_target)
        score_count += count / k
    return score_count

'''
    计算same score（根据标记置信度的KL散度进行计算）
'''
def calc_same_score_Neig_3(features, predict_Y, idx, same_k, Omega):
    select_features = [i for i in range(features.shape[1]) if i != idx]
    # select_features.append(idx)
    X = features[:, select_features]
    sample_size = X.shape[0]
    feature_len = X.shape[1]
    score_count = 0
    k = same_k
    neigs = Get_K_Neighbors(X, k)
    # if feature_len < 20:
    #     neigs = Get_K_Neighbors(X, k)
    # else:
    #     neigs, Threshold = GetAllInstanceNeigborhoodList(X, Omega)
    for i in range(sample_size):
        samples = neigs[i]
        # print(samples)
        base_target = predict_Y[i]
        count = 0
        for j in range(len(samples)):
            else_target = predict_Y[samples[j]]
            if base_target == else_target:
                count += 1
        score_count += count / k
    return score_count

'''
    根据标记置信度得到预测标记
'''
def get_predict_Y(Y_confidence):
    res = []
    sample_size, label_size = Y_confidence.shape
    for i in range(sample_size):
        Y_conf = Y_confidence[i, :]
        max_Y = max(Y_conf)
        max_idx = np.where(Y_conf == max_Y)[0].tolist()
        if len(max_idx) == 1:
            res.append(max_idx[0])
        else:
            res.append(max_idx[random.randint(0, len(max_idx) - 1)])
    return res

'''
    计算特征的分数
'''
def get_feature_scores(features, partial_labels, dec_classes, base, omega, same_k):

    dep_scores = []
    same_scores = []

    for idx in tqdm(base):
        tmp_feature_idx = copy.deepcopy(base)
        tmp_feature_idx.remove(idx)

        important_score = calc_dep(features[:, tmp_feature_idx], dec_classes, omega)
        dep_scores.append(important_score)

        same_score = calc_same_score_Neig_2(features, partial_labels, idx, same_k, omega)
        same_scores.append(same_score)
        
        # print(idx + 1, (idx + 1) / len(base))
        
    return np.array(dep_scores), np.array(same_scores)

'''
    得到特征排序
'''
def get_feature_rank(features, partial_labels, omega, same_k=6, dep_weight=0.5, des_mode=1):

    feature_len = features.shape[1]
    
    base = [i for i in range(feature_len)]
    if des_mode == 1:
        dec_classes = decision_class_1(partial_labels)
    elif des_mode == 2:
        dec_classes = decision_class_2(partial_labels)
    elif des_mode == 3:
        dec_classes = decision_class_3(partial_labels)
    elif des_mode == 4:
        dec_classes = decision_class_4(partial_labels)
    # print(dec_classes)
    dep_scores, same_scores = get_feature_scores(features, partial_labels, dec_classes, base, omega, same_k)
    
    dep_scores = normalization(dep_scores)
    same_scores = normalization(same_scores)

    scores = (dep_weight * dep_scores) + ((1 - dep_weight) * same_scores)
    ranks = np.argsort(np.array(scores)) + 1

    return ranks

def demo():
    same_k = 5
    omega = 0.38
    dep_weight = 0.5

    dataset_path = './demo_fe.mat'
    datas = sio.loadmat(dataset_path)

    features = datas['data']
    partial_labels = datas['Yconfidence']
    features = normalize(features)

    ranks = get_feature_rank(features, partial_labels, omega, same_k, dep_weight, 4)
    ranks_strs = [str(rank) for rank in ranks.tolist()]
    ranks_str = ",".join(ranks_strs)
    print(ranks_str)


if __name__ == "__main__":
    demo()