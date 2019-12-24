import numpy as np
import math


class LoessRegression(object):
    def __init__(self, bandwidth=1, use_matrix=True, learning_rate=0.01, max_iter=None):
        self._bandwith = bandwidth
        self._use_matrix = use_matrix
        self._max_iter = max_iter
        if not use_matrix and max_iter is None:
            self._max_iter = 400
        self._learning_rate = learning_rate
    
    def fit(self, X, y):
        X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
        self._P = X
        self._Y = y
        return self

    def _distance(self, x):
        D = self._P - x
        dist = np.linalg.norm(D, ord=2, axis=1)
        return dist

    def _get_weights(self, x):
        dist = self._distance(x)
        weights = np.exp(-dist/(2*np.square(self._bandwith)))
        return np.diag(weights)

    def _fit_normal(self, X):
        result = []
        for x in X:
            x = np.concatenate(([1], x), axis=0)
            x = np.reshape(x, newshape=(1, X.shape[1]+1))
            P = self._P
            Y = self._Y
            P_t = P.transpose()

            W = self._get_weights(x)

            P_d = np.linalg.pinv(np.matmul(np.matmul(P_t, W), P))
            theta = np.dot(np.matmul(np.matmul(P_d, P_t), W), Y)
            result.append(np.dot(x, theta))
        return np.reshape(np.array(result), newshape=(len(result), 1))

    def _fit_gdc(self, X):
        P = self._P
        P_t = np.transpose(P)
        Y = self._Y

        n = P.shape[0]
        m = P.shape[1]

        result = []
        for x in X:
            x = np.concatenate(([1], x), axis=0)
            x = np.reshape(x, newshape=(1, m))

            theta = np.random.normal(0, 0.2, size=(m,1))
            theta[m-1] = 1

            dist = self._distance(x)
            weights = np.reshape(np.exp(-dist/self._bandwith), newshape=(n, 1))
            for i in range(self._max_iter):
                error_term = (np.reshape(P @ theta, newshape=(n, 1)) - Y)
                cost_gradient = (1 / n) * P_t @ (np.multiply(error_term, weights))
                theta = theta - (self._learning_rate) * cost_gradient

            self.coef_ = theta[1:]
            self.intercept_ = theta[0]
            result.append(np.dot(x, theta))
        return np.reshape(np.array(result), newshape=(len(result), 1))

    def predict(self, X):
        if self._use_matrix:
            return self._fit_normal(X)
        else:
            return self._fit_gdc(X)