import numpy as np

def train_test_split(X, y, test_size=0.33, random_state=None):
    rnd = np.random.RandomState(random_state)
    n = np.round(X.shape[0]*(1-test_size))
    S = [X, y]
    rnd.shuffle(S)
    train, test = X[:n, :], X[n:, :]
    return train, test