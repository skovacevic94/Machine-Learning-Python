import numpy as np

class StandardScaler(object):
    def __init__(self):
        pass

    def fit_transform(self, X):
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

        return self.transform(X)

    def transform(self, X):
        return (X-self._mean)/self._std


class MinMaxScaler(object):
    def __init__(self):
        pass

    def fit_transform(self, X):
        self._min = np.min(X, axis=0)
        self._max = np.max(X, axis=0)

        return self.transform(X)

    def transform(self, X):
        return np.multiply((X - self._min), 1/(self._max - self._min))
