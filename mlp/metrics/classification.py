import numpy as np

def confusion_matrix(y_true, y_pred):
    k = int(np.max(y_true)) + 1
    conf_mat = np.zeros((k, k))
    # Rows are actual, columns are predicted.
    # For example conf_mat[0, 2] is number of actual 0 class that are predicted as 2
    for i in range(k):
        actual_indices = np.where(y_true == i)
        for j in range(k):
            conf_mat[i, j] = np.sum(y_pred[actual_indices] == j)
    return conf_mat

def precision_recall_f1(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    k = conf_mat.shape[0]

    precision = np.zeros(k)
    recall = np.zeros(k)
    f1_score = np.zeros(k)
    for i in range(k):
        if np.sum(conf_mat[:, i]) == 0:
            precision[i] = float('nan')
        else:
            precision[i] = conf_mat[i, i] / np.sum(conf_mat[:, i])

        if np.sum(conf_mat[i, :]) == 0:
            recall[i] = float('nan')
        else:
            recall[i] = conf_mat[i, i] / np.sum(conf_mat[i, :])

        if precision[i] == 0 or precision[i] == float('nan') or recall[i] == 0 or recall[i] == float('nan'):
            f1_score[i] = float('nan')
        else:
            f1_score[i] = 2 / (1.0/precision[i] + 1.0/recall[i])
    return precision, recall, f1_score

def accuracy_score(y_true, y_pred):
    return np.sum(y_true.reshape(-1, 1) == y_pred.reshape(-1, 1))/(y_true.shape[0])

def cross_entropy(p, q):
    return -np.sum(np.multiply(p, np.log(q)))/p.shape[0]

def cross_entropy_bin(y, pred):
    return -np.sum(y*np.log(pred) + (1-y)*np.log(1-pred))/y.shape[0]
