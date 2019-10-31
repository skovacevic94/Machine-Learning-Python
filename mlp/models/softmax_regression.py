import numpy as np
from mlp.activations import softmax

class SoftmaxRegression(object):
    """description of class"""
    def __init__(self, learning_rate=0.01, max_iter=400, batch_size=None):
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._batch_size = batch_size

    def fit(self, X, y):
        n = X.shape[0]
        m = X.shape[1]
        unique = np.unique(y)
        k = unique.shape[0]

        self._theta = np.random.normal(0, 0.2, size=(m+1, k))
        self._theta[:, k-1] = 0
        X = np.concatenate((np.ones(shape=(n, 1)), X), axis=1)
        X_t = X.T

        Y = np.zeros((n, k))
        Y[np.arange(n), y] = 1
        for iteration in range(self._max_iter):
            h_theta = softmax(np.dot(X, self._theta))
            E = Y - h_theta
            E[:, k-1] = 0

            #for l in range(k-1):
            #    delta_l = np.zeros((m+1,))
            #    for i in range(n):
            #        delta_l = delta_l + ((y[i]==l)-h_theta[i, l])*X[i, :]
            #    self._theta[:, l] = self._theta[:, l] + (self._learning_rate*delta_l)
            self._theta = self._theta - (self._learning_rate)*(-np.matmul(X_t, E))

    def predict(self, X):
        # Check if coef and intercept are defined, meaning that model is trained
        try:
            getattr(self, "_theta")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
        Z = np.matmul(X, self._theta)
        return softmax(Z)

if __name__=="__main__":
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt
    
    X, y = make_classification(200, 2, 2, 0, weights=[.3, .3, .4], n_classes=3, n_clusters_per_class=1, random_state=29)

    model = SoftmaxRegression(C=1)
    model.fit(X, y)

    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid)
    Z = np.argmax(probs, axis=1).reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
               cmap=plt.cm.Paired, vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.show()
