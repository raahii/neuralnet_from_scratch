# coding : utf-8

import numpy as np
from lib.common_functions import sigmoid, softmax, relu

class Affine:
    def __init__(self, input_size, output_size, activate_function, weight_init_std = 0.01):
        self.W = weight_init_std * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)

        self.dW = None
        self.db = None
        self.x  = None
        self.activate_function = activate_function

    def forward(self, x):
        self.x = x
        y = np.dot(self.x, self.W) + self.b

        return self.activate_function.forward(y)

    def backward(self, dy):
        dy = self.activate_function.backward(dy)

        self.dW = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)

        dx = np.dot(dy, self.W.T)

        return dx, self.dW.copy(), self.db.copy()

    def set_params(self, W, b):
        self.W = W
        self.b = b
