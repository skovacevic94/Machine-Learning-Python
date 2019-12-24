import numpy as np

from mlp.model_selection import next_batch
from mlp.activations import sigmoid
from mlp.metrics.classification import cross_entropy_bin

class LogisticRegression(object):
    def __init__(self, C=0, learning_rate=0.01, max_iter=400, batch_size=None):
        self._lambda = C
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._batch_size = batch_size

    def fit(self, X, y):
        self.loss_ = []
        m = X.shape[1]

        theta = np.random.normal(0, 0.2, size=(m+1, 1))
        theta[0] = 1
        X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
        for iteration in range(self._max_iter):
            for X_batch, y_batch in next_batch(X, y, self._batch_size):
                n = X_batch.shape[0]

                h_theta = sigmoid(np.dot(X_batch, theta))
                cost_gradient = -(1 / n) * np.dot(np.transpose(X_batch), (np.reshape(y_batch, newshape=(n, 1)) - h_theta))
                regularization_term = (self._lambda/n)*theta
                regularization_term[0] = 0 # Dont regularize bias
                theta = theta - (self._learning_rate) * (cost_gradient + regularization_term)

                self.loss_.append(cross_entropy_bin(y_batch, h_theta))
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]
        
    def predict(self, X):
        # Check if coef and intercept are defined, meaning that model is trained
        try:
            getattr(self, "coef_")
            getattr(self, "intercept_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        z = np.dot(X, self.coef_) + self.intercept_ # This is faster to do than concatenate whole tensor X, just so you can write np.dot(X, self.coef_), and more importantly easier to write
        return sigmoid(z)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt
    
    X, y = make_classification(200, 2, 2, 0, weights=[.5, .5])

    model = LogisticRegression(batch_size=32)
    model.fit(X, y)

    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid)
    Z = (probs>0.5).reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
               cmap=plt.cm.Paired, vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.show()