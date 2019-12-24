import numpy as np

class LinearRegression(object):
    def __init__(self, C=0, use_matrix=True, max_iter=None, learning_rate=0.01):
        self._lambda = C
        self._use_matrix = use_matrix
        self._max_iter = max_iter
        if not use_matrix and max_iter is None:
            self._max_iter = 400
        self._learning_rate = learning_rate

    def fit(self, X, y):
        if self._use_matrix:
            self._fit_normal(X, y)
        else:
            self._fit_gdc(X, y)

    def _fit_gdc(self, X, y):
        n = X.shape[0]
        m = X.shape[1]

        theta = np.random.normal(0, 0.2, size=(m+1,1))
        theta[m] = 1

        X = np.concatenate((np.ones(shape=(n, 1)), X), axis=1)
        for i in range(self._max_iter):
            cost_gradient = (1 / n) * np.transpose(X) @ (np.reshape(X @ theta, newshape=(n, 1)) - y)
            regularization_term = (self._lambda/n)*theta
            regularization_term[0] = 0 # Dont regularize bias
            theta = theta - (self._learning_rate) * (cost_gradient + regularization_term)

        self.coef_ = theta[1:]
        self.intercept_ = theta[0]

    def _fit_normal(self, X, y):
        X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
        X_t = X.transpose()
        E = np.identity(X.shape[1])
        E[0, 0] = 0 # Don't regularize bias (intercept) term

        X_d = np.linalg.pinv(np.matmul(X_t, X) + self._lambda*E)
        theta = np.dot(np.matmul(X_d, X_t), y)
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]
        return self

    def predict(self, X):
        # Check if coef and intercept are defined, meaning that model is trained
        try:
            getattr(self, "coef_")
            getattr(self, "intercept_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return np.dot(X, self.coef_) + self.intercept_
