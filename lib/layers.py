# coding : utf-8

import numpy as np
from lib.common_functions import sigmoid, softmax

class Affine:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)
        self.x = np.array([])
        self.b = np.array([])
        self.y = np.array([])

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.x, self.W) + self.b

        return self.y

    def set_params(self, W, b):
        self.W = W
        self.b = b

class Sigmoid:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])

    def forward(self, x):
        self.x = x
        self.y = sigmoid(x)

        return self.y

class Softmax:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])

    def forward(self, x):
        self.x = x
        self.y = softmax(self.x)

        return self.y
