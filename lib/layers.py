# coding : utf-8

import numpy as np
from lib.common_functions import sigmoid, softmax

class Affine:
    def __init__(self, input_size, output_size, activate_function):
        self.W = np.random.randn(input_size, output_size)
        self.x = np.array([])
        self.b = np.array([])
        self.y = np.array([])
        self.activate_function = activate_function

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.x, self.W) + self.b

        return self.activate_function.forward(self.y)

    def backward(self, dy):
        self.dy = self.activate_function.backward(dy)

        self.dW = np.dot(self.x.T, self.dy)
        self.db = np.sum(dy, axis=0)

        dx = np.dot(dy, self.W.T)

        return dx, self.dW, self.db

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
    
    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y
        
        return dx

class Softmax:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])

    def forward(self, x):
        self.x = x
        self.y = softmax(self.x)

        return self.y
    
    def backward(self, dy):
        batch_size = self.x.shape[0]
        dx = dy / batch_size
        
        return dx
