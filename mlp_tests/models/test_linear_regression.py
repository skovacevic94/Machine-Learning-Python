import unittest
import numpy as np
import numpy.testing as npt
import math

import mlp.models.linear_regression as linear_regression
from sklearn.linear_model import LinearRegression

class Test_linear_regression(unittest.TestCase):
    
    def test_two_horizontal_close(self):

        model = linear_regression.LinearRegression(C=1) #Use regularization to avoid singular matrix
        model.fit(X, y)

        npt.assert_array_almost_equal(model.coeff_, [0])
        npt.assert_almost_equal(model.intercept_, 2)

        X_test = np.array([[4.0], [-4.0], [8.0], [0]])
        y_test = np.array([2]*X_test.shape[0])

        y_pred = model.predict(X_test)
        npt.assert_array_almost_equal(y_test, y_pred)

    def test_rectangle_45_degrees(self):
        # Define rectangle (1, 0), (0, 1), (100, 99), (99, 100)
        X = np.array([[1.0],[0.0], [100.0], [99.0]])
        y = np.array([0.0, 1.0, 99.0, 100.0])

        model = linear_regression.LinearRegression(C=1) #Use regularization to avoid singular matrix
        model.fit(X, y)

        npt.assert_array_almost_equal(model.coeff_, [1], decimal=3)
        npt.assert_almost_equal(model.intercept_, 0, decimal=1)

        X_test = np.array([[-1000.0], [-4.0], [1000.0], [0], [21], [535]])
        y_test = np.reshape(X_test, newshape=(X_test.shape[0],))

        y_pred = model.predict(X_test)
        npt.assert_array_almost_equal(y_test, y_pred, decimal=1)

    def test_rectangle_45_degrees_11_right(self):
        # Define rectangle (12, 0), (11, 1), (111, 99), (110, 100)
        X = np.array([[12.0],[11.0], [111.0], [110.0]])
        y = np.array([0.0, 1.0, 99.0, 100.0])

        model = linear_regression.LinearRegression(C=0) #Use regularization to avoid singular matrix
        model.fit(X, y)

        npt.assert_array_almost_equal(model.coeff_, [1], decimal=3)
        npt.assert_almost_equal(model.intercept_, -11, decimal=1)

        X_test = np.array([[-23.0], [-4.0], [1.0], [0], [21], [535]])
        y_test = np.reshape(X_test-11, newshape=(X_test.shape[0],))

        y_pred_1 = model.predict(X_test)
        npt.assert_array_almost_equal(y_test, y_pred_1, decimal=1)

        model = LinearRegression()
        model.fit(X, y)

        y_pred_2 = model.predict(X_test)
        npt.assert_array_almost_equal(y_pred_1, y_pred_2)


    @staticmethod
    def _sample_nsphere(n):
        angles = np.random.normal(0, 1, n)
        angles_norm = np.linalg.norm(angles)
        nsphere_sample = angles / angles_norm
        return nsphere_sample


    def test_1000_coplanar(self):
        np.random.seed(42)

        n = 3 # Num of samples
        d = 3 # Dimensionality
         
        hyper_plane_n1 = np.random.uniform(-5, 5, (n, d-1)) # Generate hyperplane in (d-1)-dim (d-1 dim space)

        # Generate random basis for hyperplane in d-dim space. Time complexity O(d^2).
        # Algorithm Monte-Carlo Graham Smith Orthogonalization
        m_v = []
        up_vec = np.zeros(d)
        up_vec[d-1]=1
        for i in range(0, d-1):
            ei = self._sample_nsphere(d)

            while True:
                is_vertical = math.isclose(0.8, np.abs(np.dot(ei, up_vec))) # Avoid close-to-vertical hyperplane
                if not is_vertical:
                    is_orthogonalizable = True  # Graham Smith Orthogonalization
                    for j in range(0, i):
                        proj = np.dot(ei, m_v[j])
                        ei = ei - proj
                        if math.isclose(0, np.linalg.norm(ei)):
                            is_orthogonalizable = False
                            break
                        ei = ei / np.linalg.norm(ei)
                    if is_orthogonalizable is True:
                        break
                ei = self._sample_nsphere(d)
            m_v.append(ei)

        assert(1==1)
        return

    def test_gaussian_high_corr(self):
        return

if __name__ == '__main__':
    unittest.main()

