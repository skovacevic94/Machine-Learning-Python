import unittest
import numpy as np
import numpy.testing as npt

import mpl_toolkits.mplot3d.axes3d
import matplotlib.pyplot as plt

from mlp_tests.models.utils import generate_random_basis

import mlp.models.linear_regression as linear_regression
from sklearn.linear_model import LinearRegression

class Test_linear_regression(unittest.TestCase):
    
    def test_two_horizontal_close(self):
        X = np.array([[1.0],[1.000001]])
        y = np.array([2.0, 2.0])

        model = linear_regression.LinearRegression(C=1) #Use regularization to avoid singular matrix
        model.fit(X, y)

        npt.assert_array_almost_equal(model.coef_, [0])
        npt.assert_almost_equal(model.intercept_, 2)

        X_test = np.array([[4.0], [-4.0], [8.0], [0]])
        y_test = np.array([2]*X_test.shape[0])

        y_pred = model.predict(X_test)
        npt.assert_array_almost_equal(y_pred, y_test)

    # Hard numerical test
    def test_rectangle_45_degrees(self):
        # Define rectangle (1, 0), (0, 1), (100, 99), (99, 100)
        X = np.array([[1.0],[0.0], [100.0], [99.0]])
        y = np.array([0.0, 1.0, 99.0, 100.0])

        model = linear_regression.LinearRegression(C=1)
        model.fit(X, y)

        npt.assert_array_almost_equal(model.coef_, [1], decimal=3)
        npt.assert_almost_equal(model.intercept_, 0, decimal=1)

        X_test = np.array([[-1000.0], [-4.0], [1000.0], [0], [21], [535]])
        y_test = np.reshape(X_test, newshape=(X_test.shape[0],))

        y_pred = model.predict(X_test)
        npt.assert_array_almost_equal(y_pred, y_test, decimal=1)

    # Hard numerical test
    def test_rectangle_45_degrees_11_right(self):
        # Define rectangle (12, 0), (11, 1), (111, 99), (110, 100)
        X = np.array([[12.0],[11.0], [111.0], [110.0]])
        y = np.array([0.0, 1.0, 99.0, 100.0])

        model = linear_regression.LinearRegression(C=0)
        model.fit(X, y)

        npt.assert_array_almost_equal(model.coef_, [1], decimal=3)
        npt.assert_almost_equal(model.intercept_, -11, decimal=1)

        X_test = np.array([[-23.0], [-4.0], [1.0], [0], [21], [535]])
        y_test = np.reshape(X_test-11, newshape=(X_test.shape[0],))

        y_pred_1 = model.predict(X_test)
        npt.assert_array_almost_equal(y_pred_1, y_test, decimal=1)

        model = LinearRegression()
        model.fit(X, y)

        y_pred_2 = model.predict(X_test)
        npt.assert_array_almost_equal(y_pred_1, y_pred_2, decimal=1)

    def test_1000_coplanar_10d(self):
        dim = 10
        hyperplane_d_1 = np.random.uniform(-5, 5, (1000, dim))
        hyperplane_d_1[0:, dim-1] = 0

        m_v = generate_random_basis(dim, seed=3288)
        m_e = np.linalg.inv(m_v)

        offset_vector = np.random.uniform(-3, 3, dim)
        offset_vector[dim-1]=0 #dont offset dependent variable

        transformed = np.matmul(hyperplane_d_1, m_e.transpose())
        transformed = transformed + offset_vector
        X = transformed[0:, 0:(dim-1)]
        y = transformed[0:, dim-1]

        model = linear_regression.LinearRegression(C=0)
        model.fit(X, y)

        model_normal = np.concatenate((model.coef_, [-1]))
        npt.assert_almost_equal(np.dot(model_normal/np.linalg.norm(model_normal, ord=2), m_v[dim-1]), 1, decimal=6)
        npt.assert_almost_equal(model.intercept_/np.linalg.norm(model_normal), np.dot(-offset_vector, m_v[dim-1]), decimal=6)

        test_hyperplane = np.random.uniform(-15, 15, (50, dim))
        test_hyperplane[0:, (dim-1)] = 0
        test_transform = np.matmul(test_hyperplane, m_e.transpose())
        test_transform = test_transform + offset_vector
        X_test = test_transform[0:, 0:(dim-1)]
        y_test = test_transform[0:, dim-1]

        y_pred_1 = model.predict(X_test)
        npt.assert_array_almost_equal(y_test, y_pred_1, decimal=6)

    def test_gaussian_high_corr_3d(self):
        mean = np.array([-4, 2, -1])
        cov = np.array([
            [1, 200, 1000],
            [200, 1, 0],
            [1000, 0, 1]
        ])

        u, s, v = np.linalg.svd(cov)
        normal = -u[0:, 2]

        samples = np.random.multivariate_normal(mean, cov, 500)
        X = samples[0:, 0:2]
        y = samples[0:, 2]

        model = linear_regression.LinearRegression(C=0)
        model.fit(X, y)

        model_normal = np.concatenate((model.coef_, [-1]))
        npt.assert_almost_equal(np.dot(model_normal/np.linalg.norm(model_normal, ord=2), normal), 1, decimal=4)
        npt.assert_almost_equal(model.intercept_/np.linalg.norm(model_normal), np.dot(-mean, normal), decimal=1)

        modelSk = LinearRegression()
        modelSk.fit(X, y)
        npt.assert_array_almost_equal(model.coef_, modelSk.coef_, decimal=6)
        npt.assert_almost_equal(model.intercept_, modelSk.intercept_, decimal=6)

    def test_gaussian_high_corr_nd(self):
        dim = 5
        mean = np.zeros(dim)

        m_v = generate_random_basis(dim, seed=3288)
        eigen_vals = np.array([1000, 254, 50, 10, 1])

        cov = np.matmul(np.matmul(m_v, np.diag(eigen_vals)), m_v.T)

        normal = -m_v[dim-1]
        samples = np.random.multivariate_normal(mean, cov, 500)
        X = samples[0:, 0:(dim-1)]
        y = samples[0:, dim-1]

        model = linear_regression.LinearRegression(C=0)
        model.fit(X, y)

        model_normal = np.concatenate((model.coef_, [-1]))
        #npt.assert_almost_equal(np.dot(model_normal/np.linalg.norm(model_normal, ord=2), normal), 1, decimal=4)
        #npt.assert_almost_equal(model.intercept_/np.linalg.norm(model_normal), np.dot(-mean, normal), decimal=1)

        modelSk = LinearRegression()
        modelSk.fit(X, y)
        npt.assert_array_almost_equal(model.coef_, modelSk.coef_, decimal=6)
        npt.assert_almost_equal(model.intercept_, modelSk.intercept_, decimal=6)

if __name__ == '__main__':
    unittest.main()

