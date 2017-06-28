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

    def loss(self, x, t):
        y = self.forward(x)

        return self.cost_function(y, t)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)

        return y

    def backword(self, x, t):

        return grads

    def accuracy(self, x, t):
        y = np.argmax(self.forward(x), 1)
        _t = np.argmax(t, 1)

        return 1.0 * np.sum((_t == y)) / x.shape[0]
