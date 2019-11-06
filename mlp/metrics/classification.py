import numpy as np

def accuracy_score(y, pred):
    return np.sum(y==pred)/(y.shape[0])

def cross_entropy(Y, pred):
    return -np.sum(np.multiply(Y, np.log(pred)))/Y.shape[0]

def cross_entropy_bin(y, pred):
    return -np.sum(y*np.log(pred) + (1-y)*np.log(1-pred))/y.shape[0]
