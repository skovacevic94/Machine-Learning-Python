import numpy as np

def split_train_test(X, ratio=0.8):
    n = np.round(X.shape[0]*ratio)
    np.random.shuffle(X)
    train, test = X[:n, :], X[n:, :]
    return train, test