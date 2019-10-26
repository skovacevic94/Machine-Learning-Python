from sklearn.base import BaseEstimator
import numpy as np

class LinearRegression(BaseEstimator):
    def __init__(self, C=0):
        self._lambda = C
        return

    def fit(self, X, y):
        X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
        X_t = X.transpose()
        E = np.identity(X.shape[1])
        E[0, 0] = 0 # Don't regularize bias (intercept) term

        X_d = np.linalg.inv(np.matmul(X_t, X) - self._lambda*E)
        theta = np.dot(np.matmul(X_d, X_t), y)
        self.coeff_ = theta[1:]
        self.intercept_ = theta[0]
        return self

    def predict(self, X):
        # Check if coeff and intercept are defined, meaning that model is trained
        try:
            getattr(self, "coeff_")
            getattr(self, "intercept_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return np.dot(X, self.coeff_) + self.intercept_
       