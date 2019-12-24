import numpy as np

class LogTransform(object):
    def __init__(self, columns=None):
        self._columns = columns

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        if self._columns is None:
            return np.log(X)
        result = X.copy()
        for col in self._columns:
            result[:, col] = np.log(X[:, col])
        return result