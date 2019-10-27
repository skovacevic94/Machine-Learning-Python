import numpy as np

class ElasticNetRegression(object):
    def __init__(self, C1=0, C2=0, learning_rate=1e-3, num_iter=100):
        self._lambda1 = C1
        self._lambda2 = C2
        self._learning_rate = learning_rate
        self._num_iter = num_iter
        return


    def fit(self, X, y):
        return

    def predict(self, X):
        # Check if coeff and intercept are defined, meaning that model is trained
        try:
            getattr(self, "coeff_")
            getattr(self, "intercept_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return np.dot(X, self.coef_) + self.intercept_
