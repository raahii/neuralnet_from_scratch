# coding : utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from lib.common_functions import *
from dataset.mnist import load_mnist
from PIL import Image
import pickle

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                       cost_function = cross_entropy_error,
                       weight_init_std = 0.01):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.cost_function = cost_function
    
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(self.input_size, self.hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(self.hidden_size, self.output_size)
        self.params['b1'] = np.zeros(self.hidden_size, )
        self.params['b2'] = np.zeros(self.output_size, )

    def loss(self, x, t):
        y = self.forward(x)
        return self.cost_function(y, t)

    def forward(self, x):
        W1, W2, b1, b2= self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = softmax(a2)

        return z2

    def backword(self, x, t):
        W1, W2, b1, b2= self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        grads = {}

        batch_size = t.shape[0]
        dout = (y - t) / batch_size

        grads["W2"] = np.dot(z1.T, dout)
        grads["b2"] = np.sum(dout, axis = 0)
        dz1 = np.dot(dout, W2.T)

        da1 = dz1.copy()
        da1[ a1 <= 0 ] = 0
        grads["W1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis = 0)

        return grads

    def accuracy(self, x, t):
        y = np.argmax(self.forward(x), 1)
        _t = np.argmax(t, 1)

        return 1.0 * np.sum((_t == y)) / x.shape[0]
