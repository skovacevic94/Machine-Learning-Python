import numpy as np

from mlp.statistics import multivariate_gaussian_pdf

class QuadraticDiscriminantAnalysis(object):
    """description of class"""
    
    def fit(self, X, y):
        n = X.shape[0]
        m = X.shape[1]
        k = np.max(y)+1
        
        self._phi = np.zeros(k)
        self._means = np.zeros(shape=(k, m))
        self._sigma_tensor = np.zeros(shape=(k, m, m))
        for i in range(k):
            class_i_indices = np.argwhere(y==i).ravel()
            n_i = class_i_indices.shape[0]

            self._phi[i] = n_i/n

            X_yi = np.take(X, class_i_indices, axis=0)
            self._means[i, :] = (1/n_i)*np.sum(X_yi, axis=0)

            D_yi = X_yi - self._means[i, :]
            # Assumption is data is gaussian so MLE covariance matrix (it's diagonal because of "Naive", and we store only it's diagonal elements-variances) estimation gives divition by n_i instead of n_i-1
            self._sigma_tensor[i] = (1/n_i)*np.matmul(D_yi.T, D_yi)

    def predict(self, X):
        # Check if coef and intercept are defined, meaning that model is trained
        try:
            getattr(self, "_phi")
            getattr(self, "_means")
            getattr(self, "_sigma_tensor")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        k = self._phi.shape[0]
        #Instead of computing products, we convert to log-probabilities to convert iterated product to sum. Reason: Numerical accuracy
        p_Xy = np.zeros((X.shape[0], k))
        for i in range(k):
            p_Xy[:, i] = np.reshape(self._phi[i] * multivariate_gaussian_pdf(X, self._means[i], self._sigma_tensor[i]), p_Xy[:, i].shape)
        S = np.sum(p_Xy, axis=1).reshape(X.shape[0], 1)
        return p_Xy/S


if __name__=="__main__":
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt
    
    X, y = make_classification(200, 2, 2, 0, weights=[.3, .3, .4], n_classes=3, n_clusters_per_class=1, random_state=None)

    model = QuadraticDiscriminantAnalysis()
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


