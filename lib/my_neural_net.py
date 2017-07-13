# coding : utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np

class MyNeuralNet:
    def __init__(self, cost_function):
        self.cost_function = cost_function
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def loss(self, y, t):

        return self.cost_function(y, t)

    def forward(self, x, train_flg=False):
        y = x
        for layer in self.layers:
            y = layer.forward(y, train_flg)

        return y

    def backward(self, y, t):
        dy = y - t

        for layer in reversed(self.layers):
            dy = layer.backward(dy)

    def accuracy(self, x, t):
        y = np.argmax(self.forward(x), 1)
        _t = np.argmax(t, 1)

        return 1.0 * np.sum((_t == y)) / x.shape[0]
