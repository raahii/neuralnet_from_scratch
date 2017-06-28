# coding : utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from lib.common_functions import *

class MyNeuralNet:
    def __init__(self, cost_function = cross_entropy_error):
        self.cost_function = cost_function
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def loss(self, y, t):

        return self.cost_function(y, t)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)

        return y

    def backward(self, y, t):
        dy = y - t

        grads = []
        for layer in reversed(self.layers):
            dy, dW, db = layer.backward(dy)
            grads.append({'dW': dW, 'db': db})

        return reversed(grads)

    def accuracy(self, x, t):
        y = np.argmax(self.forward(x), 1)
        _t = np.argmax(t, 1)

        return 1.0 * np.sum((_t == y)) / x.shape[0]
