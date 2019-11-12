import numpy as np
from cvxopt import matrix, solvers

class SVM(object):
    def __init__(self, C, kernel='linear'):
        self._kernel = 'linear'
        self._C = C

    def _fit_linear_primal(self, X, y):
        n = X.shape[0]
        m = X.shape[1]

        X = np.concatenate((np.ones(shape=(n, 1)), X), axis=1)

        y_0 = np.where(y==0)
        y[y_0] = -1

        '''
        CVXOPT phrases problem as following:
        min (1/2)x.T@P@x + q.T@x
        s.t.    G@x <= h
                A@x = b

        Our problem is
        min (1/2)||w||^2 + C*(eps_1 + eps_2 + ... + eps_n)
        s.t. y(i)*(w@x(i) + b) >= 1-eps_i but we can combine bias if we extend our input space by concatenating 1 in front x = [1, x_1, x_2, ..., x_m] (typical thing we do for log of other ML algorithms)

        We will instead concatenate vector w and eps into 1 single vector z such that first m+1 elements are from w, and last n elements of z are from eps. 
        '''
        
        # P[0:m+1, 0:m+1] submatrix is identity matrix since that submatrix describes first part of optimization objective: ||w||^2 = (w_0)^2 + (w_1)^2 + (w_2)^2 + ... + (w_m)^2.
        # Since we don't have cofactors (w_i*w_j where i != j), and coefficients are all 1's, that part of the matrix (submatrix) is identity.
        P = np.identity(m+1 + n) # m+1 for ||w||^2 and n for epsilon regularization variable
        P[0, 0] = 0 # Ignore bias b since it's not part of minimization objective, but rather, linear constraint
        P[m+1:, m+1:] = 0 # Rest of the matrix is 0, since epsilon only have linear factors so it's ignored here

        q = self._C*np.ones(m+1 + n) # First m+1 for w (which will be 0 since w don't have linear factors), and n for epsilon (which will be ones)
        q[0:m+1] = 0 # w doesn't have any linear factors

        G = np.zeros((2*n, m+1+n))
        Constraints_LHS = np.multiply(X, np.tile(np.reshape(y, (n, 1)), (1, m+1))) # Functional margin for every example i=1, 2, ..., n
        G[:n, :m+1] = Constraints_LHS # Add Functional margin to G
        G[:n, m+1:] = np.identity(n) # Add eps_i to each functional margin
        G[n:, m+1:] = np.identity(n)
        G = (-1)*G # Invert sign to convert >= to <=

        h = -np.ones(2*n) # G@w <= vector(-1)
        h[n:] = 0 # -eps_i <= 0

        try:
            sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
            if 'optimal' not in sol['status']:
                raise RuntimeError("Optimal solution not found.")
        except ValueError:
            raise RuntimeError("Optimal solution not found.") 

        self._w = np.array(sol['x']).reshape((m+1+n, 1))[:m+1]

    def _fit_dual(self, X, y):
        pass

    def fit(self, X, y):
        self._fit_linear_primal(X, y)

    def _predict_linear_primal(self, X):
        #Check if _w is defined, meaning that model is trained
        try:
            getattr(self, "_w")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        n = X.shape[0]
        m = X.shape[1]
        X = np.concatenate((np.ones(shape=(n, 1)), X), axis=1)
        return np.sign(np.dot(X, self._w))

    def _predict_dual(self, X):
        pass

    def predict(self, X):
        return self._predict_linear_primal(X)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    
    #X, y = make_classification(200, 2, 2, 0, weights=[.5, .5])
    mean_0 = [-1, -1]
    cov_0 = [[0.1, 0], [0, 0.1]]
    mean_1 = [2, 2]
    cov_1 = [[0.1, 0], [0, 0.1]]
    mean_e = [0.5, 0.5]
    cov_e = [[0.1, 0], [0, 0.1]]
    n_0 = 200
    n_1 = 200
    n_e = 20

    np.random.seed(40)
    X_0 = np.random.multivariate_normal(mean_0, cov_0, n_0)
    y_0 = np.zeros(n_0)

    X_1 = np.random.multivariate_normal(mean_1, cov_1, n_1)
    y_1 = np.ones(n_1)

    X_e = np.random.multivariate_normal(mean_e, cov_e, n_e)
    y_e = np.zeros(n_e)

    X = np.concatenate((X_0, X_1, X_e), axis=0)
    y = np.concatenate((y_0, y_1, y_e), axis=0)


    model1 = SVM(C=100, kernel="linear")
    model1.fit(X, y)

    model2 = SVC(C=100, kernel='linear')
    model2.fit(X, y)

    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    #plt.subplot(2, 1, 1)
    probs = model1.predict(grid)
    Z = (probs>0.5).reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
               cmap=plt.cm.Paired, vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")

    #plt.subplot(2, 1, 2)
    probs = model2.predict(grid)
    Z = (probs>0.5).reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
               cmap=plt.cm.Paired, vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.show()