import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_acc(y_true, y_pred, return_ind=False):

    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)  #匈牙利算法最小成本  现在要求最大匹配  所以有第二个括号的内容来转化为目标问题
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
