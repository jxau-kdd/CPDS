import copy
import math
import scipy
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from tools import Get_K_Neighbors, init_Y_confidence, get_candidate, process_csc_matrix, get_single_target, KNN_disambiguation, KNN_disambiguation_simple

class Ball:
    def __init__(self, features, labels, partial_target, sample_indexs) -> None:
        self.features = features
        self.labels = labels
        self.partial_target = partial_target
        self.sample_indexs = sample_indexs
        self.sample_size = features.shape[0]

        self.ball_feature = self.get_ball_feature()
        self.ball_label = self.get_ball_label()
        # self.avg_cosine = self.calc_acg_cosine()
        pass

    def get_ball_label(self):
        labels = self.partial_target
        tmp_labels = labels.sum(axis=0).tolist()
        if len(tmp_labels) == 1:
            label = np.array(tmp_labels[0])
        else:
            label = np.array(tmp_labels)

        label = label / sum(label)
        return label

    def get_ball_feature(self):
        features = self.features
        feature = np.array(features.mean(axis=0))
        return feature

    def calc_avg_cosine(self):
        features = self.features
        pdist_martrix = pdist(features, metric='cosine')
    
        similar_martrix = squareform(pdist_martrix)
        avg_cosine = np.mean(similar_martrix)
        return avg_cosine
    
    def get_predict_labels(self, mode, k_limlit):
        if self.sample_size <= k_limlit:
            candidate = get_candidate(self.partial_target)
            target_list = get_single_target(self.labels)
            k = self.sample_size - 1
            Q = self.partial_target.shape[1]
            Y_confidence = init_Y_confidence(self.partial_target)
            if mode == 'no_Y':
                predict_Y = KNN_disambiguation_simple(self.features, self.partial_target, k, Q, candidate)
            else:
                predict_Y = KNN_disambiguation(self.features, self.partial_target, k, Q, candidate, Y_confidence)
            acc = calc_acc(predict_Y, target_list)
            return acc, acc * self.sample_size
        else:
            candidate = get_candidate(self.partial_target)
            target_list = get_single_target(self.labels)
            k = k_limlit
            Q = self.partial_target.shape[1]
            Y_confidence = init_Y_confidence(self.partial_target)
            if mode == 'no_Y':
                predict_Y = KNN_disambiguation_simple(self.features, self.partial_target, k, Q, candidate)
            else:
                predict_Y = KNN_disambiguation(self.features, self.partial_target, k, Q, candidate, Y_confidence)
            acc = calc_acc(predict_Y, target_list)
            return acc, acc * self.sample_size

    def get_predict_labels_no_limit(self, mode = "no_Y"):
        candidate = get_candidate(self.partial_target)
        target_list = get_single_target(self.labels)
        k = self.sample_size - 1
        Q = self.partial_target.shape[1]
        Y_confidence = init_Y_confidence(self.partial_target)
        if mode == 'no_Y':
            predict_Y = KNN_disambiguation_simple(self.features, self.partial_target, k, Q, candidate)
        else:
            predict_Y = KNN_disambiguation(self.features, self.partial_target, k, Q, candidate, Y_confidence)
        acc = calc_acc(predict_Y, target_list)
        return acc, acc * self.sample_size

    def get_Yconfidence(self, use_base_Y, k = 10):
        '''
            use_base_Y： 是否使用初始化的Yconfidence来进行标记增强
            k：KNN的邻居个数
        '''
        sample_size = self.features.shape[0]
        if k >= sample_size:
            k = sample_size - 1
        top_k_neigs = Get_K_Neighbors(self.features, k)
        base_Yconfidence = init_Y_confidence(self.partial_target)

        candidate = get_candidate(self.partial_target)
        
        wr = [k - i for i in range(k)]
        Q = self.partial_target.shape[1]
        result = []
        for i in range(sample_size):
            k_neighbors = top_k_neigs[i, :]
            sumY = np.zeros(Q)

            percandidate = candidate[i]
            sizecandidate = len(percandidate)
            
            for t in range(sizecandidate):
                indexlabel = percandidate[t]
                for j in range(k):
                    indexneighbor = k_neighbors[j]
                    sumY[indexlabel] = sumY[indexlabel] + self.partial_target[indexneighbor, indexlabel] * wr[j]
            if sum(sumY) == 0:
                sumY = base_Yconfidence[i, :]
            else:
                sumY = sumY / sum(sumY)
                if use_base_Y:
                    sumY = 0.5 * base_Yconfidence[i, :] + 0.5 * sumY
            result.append(sumY.tolist())
        return np.array(result)

class BallList:
    def __init__(self, features, labels, partial_target, ball_split_len) -> None:
        self.ball_split_len = ball_split_len
        self.all_sample_size = features.shape[0]
        self.label_size = labels.shape[1]
        # self.labels = labels
        # self.partial_target = partial_target
        self.balls = self.build_ball(features, labels, partial_target, ball_split_len)
        self.ball_size = len(self.balls)
    
        # 分裂球
    def build_ball(self, features, labels, partial_target, ball_split_len):
        sample_indexs = [i for i in range(features.shape[0])]
        balls = self.cluster_predict(features, labels, partial_target, sample_indexs)
        count = 0
        final_balls = []
        while True:
            ball_len = len(balls)
            update_balls = []
            for ball in balls:
                if ball.sample_size <= 6:
                    final_balls.append(ball)
                    continue
                else:
                    new_balls = self.cluster_predict(ball.features, ball.labels, ball.partial_target, ball.sample_indexs, ball_split_len)
                    if self.judge_split(new_balls):
                        for new_ball in new_balls:
                            update_balls.append(new_ball)
                    else:
                        final_balls.append(ball)
            new_ball_len = len(update_balls)
            if new_ball_len <= ball_len:
                for ball in update_balls:
                    final_balls.append(ball)
                break
            else:
                balls = update_balls
            count += 1
        return final_balls

    def build_ball_2(self, features, labels, partial_target, ball_split_len):
        balls = self.cluster_predict(features, labels, partial_target)
        count = 0
        final_balls = []
        while True:
            ball_len = len(balls)
            update_balls = []
            for ball in balls:
                if ball.sample_size <= 6:
                    final_balls.append(ball)
                    continue
                else:
                    new_balls = self.cluster_predict(ball.features, ball.labels, ball.partial_target, ball_split_len)
                    for new_ball in new_balls:
                        if new_ball.sample_size > 3:
                            update_balls.append(new_ball)
                        else:
                            final_balls.append(new_ball)
            new_ball_len = len(update_balls)
            if new_ball_len <= ball_len:
                for ball in update_balls:
                    final_balls.append(ball)
                break
            else:
                balls = update_balls
            count += 1
        return final_balls

    # 单纯使用球内个数进行判断
    def judge_split(self, new_balls):
        for new_ball in new_balls:
            if new_ball.sample_size < 3:
                return False
        return True

    # 加上余弦相似度的判断（废弃）
    def judge_split_with_cosine(self, orig_ball, new_balls):
        for new_ball in new_balls:
            if new_ball.sample_size < 3 or new_ball.avg_cosine > orig_ball.avg_cosine:
                return False
        return True

    # cluster
    def cluster_predict(self, features, labels, partial_target, sample_indexs, n=2):
        kmeans = KMeans(n_clusters=n, random_state=0).fit(features)
        predict = kmeans.predict(features)
        # can_division = judge(features, predict)
        balls = []
        for i in range(n):
            cluster_indexs = np.where(predict == i)
            clusrer_features = features[cluster_indexs]
            clusrer_labels = labels[cluster_indexs]
            clusrer_partial_target = partial_target[cluster_indexs]
            clusrer_sample_indexs = (np.array(sample_indexs)[cluster_indexs]).tolist()
            ball = Ball(clusrer_features, clusrer_labels, clusrer_partial_target, clusrer_sample_indexs)
            balls.append(ball)
        return balls

    def get_acc(self, mode, use_limit=None):
        '''
            mode： 使用哪种模式
                   模式no_Y：不使用Y_confidence进行标记预测
                   模式use_Y：使用Y_confidence进行标记预测
            use_limit：进行预测的时候是否对k进行限制，限制则调用get_predict_labels，
                       不限制则调用get_predict_labels_no_limit
        '''
        balls = self.balls
        # ball = balls[0]
        right_samples = 0
        samples = 0
        for ball in balls:
            if use_limit:
                ball_acc, ball_right_samples = ball.get_predict_labels(mode, use_limit)
            else:
                ball_acc, ball_right_samples = ball.get_predict_labels_no_limit(mode)
            samples += ball.sample_size
            right_samples += ball_right_samples
        acc = right_samples / self.all_sample_size
        return acc
    
    def get_Yconfidence(self, use_base_Y=False):
        '''
            use_base_Y： 是否使用初始化的Yconfidence
        '''
        balls = self.balls
        result = np.zeros((self.all_sample_size, self.label_size))
        for ball in balls:
            ball_Y = ball.get_Yconfidence(use_base_Y)
            ball_idxs = ball.sample_indexs
            for i in range(len(ball_idxs)):
                idx = ball_idxs[i]
                result[idx, :] = ball_Y[i, :]
        return result

    def get_balls_samples(self):
        balls = self.balls
        features = []
        labels = []
        for ball in balls:
            features.append(ball.ball_feature.tolist())
            labels.append(ball.ball_label.tolist())
            
        features = np.array(features)
        labels = np.array(labels)

        return features, labels

def Get_K_Neighbors(features, k):
    pdist_martrix = pdist(features, metric='euclidean')
    
    martrix = squareform(pdist_martrix)
    topK_neighbors = np.argsort(martrix, axis = 0)[1: k + 1].T
    return topK_neighbors

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

# 计算关联度矩阵
def calc_A(candidate, M_distances, top_k_neigs, label_size, ball_size):
    res = np.zeros((label_size, ball_size))
    sample_size = len(candidate)
    for i in range(sample_size):
        neigs = top_k_neigs[i, :]
        for neig in neigs:
            cands = candidate[neig]
            for cand in cands:
                res[cand] = res[cand] + M_distances[i]
    return res

def calc_edu(x, y):
    return np.sqrt(np.sum((x - y)**2))

# 计算出样本对粒球的隶属度矩阵
def calc_M(features, ball_features, labels, ball_labels):
    distances = scipy.spatial.distance.cdist(features, ball_features, metric='euclidean')
    distances_l = scipy.spatial.distance.cdist(labels, ball_labels, metric='euclidean')
    res = np.zeros(distances.shape)
    sample_size, ball_size = distances.shape
    for i in range(sample_size):
        dis_sum = sum(distances[i])
        l_dis_sum = sum(distances_l[i])
        for j in range(ball_size):
            if distances_l[i][j] != 0 and distances[i][j] != 0:
                tmp_dis = 0.5 * dis_sum / distances[i][j] + 0.5 * l_dis_sum / distances_l[i][j]
            if distances[i][j] == 0 and distances_l[i][j] != 0:
                tmp_dis = 0.5 * l_dis_sum / distances_l[i][j]
            if distances[i][j] != 0 and distances_l[i][j] == 0:
                tmp_dis = 0.5 * dis_sum / distances[i][j]
            res[i][j] = tmp_dis
    return res

# 根据模糊算子得到最终的置信度矩阵
def composition_operator(A, M, candidate):
    res = []
    res = np.matmul(A, M.T).T
    # res[np.where(res > 1)] = 1
    label_size = A.shape[0]
    for i in range(len(candidate)):
        cands = candidate[i]
        for j in range(label_size):
            if j not in cands:
                res[i][j] = 0
            else:
                res[i][j] = math.fabs(res[i][j])
        res[i] = res[i] / res[i].sum()
    return res

# 计算标记置信度
def calc_Y_confidence(features, labels, partial_labels, ball_split_len, k):
    candidate = get_candidate(partial_labels)
    balls = BallList(features, labels, partial_labels, ball_split_len)
    ball_features, ball_labels = balls.get_balls_samples()

    top_k_neigs = Get_K_Neighbors(features, k)

    candidate = get_candidate(partial_labels)
    
    # 计算出样本对粒球的隶属度
    M_distances = calc_M(features, ball_features, partial_labels, ball_labels)
    M_distances = np.array(M_distances)

    M_distances = StandardScaler().fit_transform(M_distances)
    

    A_matrix = calc_A(candidate, M_distances, top_k_neigs, partial_labels.shape[1], balls.ball_size)
    A_matrix = StandardScaler().fit_transform(A_matrix)
    
    confidence = composition_operator(A_matrix, M_distances, candidate)
    
    confidence = np.around(confidence, decimals=3)
    return confidence

def demo():
    
    datas = loadmat('./demo.mat')
    
    features = datas['data']
    labels = datas['target']
    partial_labels = datas['partial_target']

    labels, partial_labels = process_csc_matrix(labels, partial_labels)
    
    confidence = calc_Y_confidence(features, labels, partial_labels, 3, 5)
    print(confidence)

if __name__ == "__main__":
    demo()
