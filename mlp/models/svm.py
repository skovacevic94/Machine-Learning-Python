import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics import pairwise_distances

class SVM(object):
    def __init__(self, C=1, sigma=None, kernel='rbf'):
        self._C = C
        self._sigma = sigma
        self._kernel = kernel

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
            solvers.options['show_progress'] = False
            sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
            if 'optimal' not in sol['status']:
                raise RuntimeError("Optimal solution not found.")
        except ValueError:
            raise RuntimeError("Optimal solution not found.") 

        self._w = np.array(sol['x']).reshape((m+1+n, 1))[:m+1]

    # Running time O(n_1*n_2). Memory Complexity O(n_1 + n_2)
    # Maybe can be improved with some Computational Geometry techniques to compute pairwise distances
    def _rbf_kernel_matrix(self, X_1, X_2):
        D_pair = np.einsum('ijk->ij', (X_1[:, None, :] - X_2)**2)
        #D_pair = np.power(pairwise_distances(X_1, X_2), 2)
        return np.exp(-D_pair/(2*self._sigma**2))

    def _linear_kernel(self, X_1, X_2):
        return np.matmul(X_1, X_2.T)

    def _fit_rbf_dual(self, X, y):
        n = X.shape[0]

        y_0 = np.where(y==0)
        y[y_0] = -1

        y = np.reshape(y, (n, 1))
        K = self._rbf_kernel_matrix(X, X)
        P = np.outer(y, y) * K
        q = -np.ones(n) # Linear factors of alpha. <\alpha, q> = \alpha_1 + \alpha_2 + ... + \alpha_n

        G_0 = -np.identity(n) # Handle constraints \alpha_i >= 0  <==> -\alpha_i < 0
        h_0 = np.zeros((n, 1)) # Handle constraints \alpha_i >= 0  <==> -\alpha_i < 0
        
        G_c = np.identity(n) # Handle constraints \alpha_i <= C
        h_c = np.ones((n, 1)) * self._C # Handle constraints \alpha_i <= C
        G = np.vstack((G_0, G_c))
        h = np.vstack((h_0, h_c))

        # Handle constraint <y, \alpha> = 0
        A = y.reshape(1, -1)
        b = np.zeros(1)

        try:
            solvers.options['show_progress'] = False
            sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
            if 'optimal' not in sol['status']:
                raise RuntimeError("Optimal solution not found.")
        except ValueError:
            raise RuntimeError("Optimal solution not found.") 

        largrange_mul = np.array(sol['x'])
        support_vector_idx = np.where(largrange_mul >= 1e-5)[0]
        self.largrange_mul_ = largrange_mul[support_vector_idx]
        self._support_y = y[support_vector_idx]
        self.support_vectors_ = X[support_vector_idx, :]

        # Calculate intercept with first support vector
        self._intercept = self._support_y[0] + np.dot(K[support_vector_idx[0], support_vector_idx], -self.largrange_mul_*self._support_y)

    def fit(self, X, y):
        X_ = X.astype("double")
        y_ = y.astype("double")
        if self._sigma is None:
            self._sigma = np.sqrt(X_.shape[1]/2)*X_.std()

        if self._kernel == 'linear':
            self._fit_linear_primal(X_, y_)
        elif self._kernel == 'rbf':
            self._fit_rbf_dual(X_, y_)

    def _predict_linear_primal(self, X):
        #Check if _w is defined, meaning that model is trained
        try:
            getattr(self, "_w")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        n = X.shape[0]
        X = np.concatenate((np.ones(shape=(n, 1)), X), axis=1)
        return np.sign(np.dot(X, self._w))

    def _predict_rbf_dual(self, X):
        #Check if support_vectors_, _support_y, and largrange_mul_ are defined, meaning that model is trained
        try:
            getattr(self, "support_vectors_")
            getattr(self, "_support_y")
            getattr(self, "largrange_mul_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        K = self._rbf_kernel_matrix(X.astype('double'), self.support_vectors_)
        return np.sign(np.dot(K, self.largrange_mul_*self._support_y) + self._intercept)
        
    def predict(self, X):
        if self._kernel == 'linear':
            pred = self._predict_linear_primal(X.astype('double'))
        elif self._kernel == 'rbf':
            pred = self._predict_rbf_dual(X.astype('double'))

        pred_0 = np.where(pred == -1)
        pred[pred_0] = 0
        return pred

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt

    #X, y = make_classification(200, 2, 2, 0, weights=[.5, .5])
    mean_0 = [-1, -1]
    cov_0 = [[0.1, 0], [0, 0.1]]
    mean_1 = [2, 2]
    cov_1 = [[0.1, 0], [0, 0.1]]
    mean_e = [1, 1]
    cov_e = [[0.1, 0], [0, 0.1]]
    n_0 = 20
    n_1 = 20
    n_e = 3

    np.random.seed(432)
    X_0 = np.random.multivariate_normal(mean_0, cov_0, n_0)
    y_0 = np.zeros(n_0)

    X_1 = np.random.multivariate_normal(mean_1, cov_1, n_1)
    y_1 = np.ones(n_1)

    X_e = np.random.multivariate_normal(mean_e, cov_e, n_e)
    y_e = np.zeros(n_e)

    X = np.concatenate((X_0, X_1, X_e), axis=0)
    y = np.concatenate((y_0, y_1, y_e), axis=0)

    model = SVM(C=1, sigma=np.sqrt(0.02), kernel="rbf")
    model.fit(X, y)

    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    #plt.subplot(2, 1, 1)
    probs = model.predict(grid)
    Z = (probs>0).reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
               cmap=plt.cm.Paired, vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.show()