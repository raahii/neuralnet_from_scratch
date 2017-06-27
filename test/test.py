import unittest
import sys, os
sys.path.append(os.pardir)
import numpy as np
from lib.common_functions import numerical_gradient, gradient_descent

def function1(x):
    return x[0]**2 + x[1]**2

class TestGradMethods(unittest.TestCase):

    def test_numerical_diff(self):
        x1 = 3.0
        x2 = 4.0
        ans = np.array([2.0*x1, 2.0*x2])
        grad = numerical_gradient(function1, np.array([x1, x2]))
        np.testing.assert_allclose(grad, ans)

    def test_gradient_descent(self):
        init_params = np.array([-3.0, 4.0])
        params = gradient_descent(function1, init_params, 1e-2, 1000)
        ans = np.array([0.0, 0.0])

        np.testing.assert_allclose(params, ans, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
