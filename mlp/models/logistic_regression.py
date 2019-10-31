
import numpy as np
import matplotlib.pyplot as plt

from mlp.activations import sigmoid

class LogisticRegression(object):
    def __init__(self, C=0, learning_rate=0.01, max_iter=400, batch_size=None):
        self._lambda = C
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._batch_size = batch_size

    def _cost_function(self):
        pass

    def fit(self, X, y):
        n = X.shape[0]
        m = X.shape[1]

        theta = np.random.normal(0, 0.2, size=(m+1, 1))
        theta[0] = 1
        X = np.concatenate((np.ones(shape=(n, 1)), X), axis=1)
        for iteration in range(self._max_iter):
            h_theta = sigmoid(np.dot(X, theta))
            cost_gradient = (1 / n) * np.dot(np.transpose(X), (np.reshape(y, newshape=(n, 1)) - h_theta))
            regularization_term = (self._lambda/n)*theta
            regularization_term[0] = 0 # Dont regularize bias
            theta = theta + (self._learning_rate) * (cost_gradient + regularization_term)

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
    
    X, y = make_classification(200, 2, 2, 0, weights=[.5, .5])

    model = LogisticRegression()
    model.fit(X, y)

    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid).reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.show()