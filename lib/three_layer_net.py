# coding : utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from lib.common_functions import *
from dataset.mnist import load_mnist
from PIL import Image
import pickle

class ThreeLayerNet:
    def __init__(self, input_num, hidden_num1, hidden_num2, output_num,\
                       cost_function = cross_entropy_error,
                       weight_init_std = 0.01):
        self.iun  = input_num
        self.hun1 = hidden_num1
        self.hun2 = hidden_num2
        self.oun  = output_num
        self.cost_function = cost_function
    
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(self.iun, self.hun1)
        self.params['W2'] = weight_init_std * np.random.randn(self.hun1, self.hun2)
        self.params['W3'] = weight_init_std * np.random.randn(self.hun2, self.oun)
        self.params['b1'] = np.zeros(self.hun1, )
        self.params['b2'] = np.zeros(self.hun2, )
        self.params['b3'] = np.zeros(self.oun, )

    def set_params(self, trained_params):
        for key in trained_params.keys():
            self.params[key] = trained_params[key]

    def loss(self, x, t):
        y = self.forward(x)
        return self.cost_function(y, t)

    def numerical_gradient(self, x, t):
        """
        common_functionsのnumerical_gradientを用いる。
        だがloss関数は本来x,tを引数であり、内部で暗黙的に
        インスタンス変数Wを用いていることに注意。もちろん
        lossを最小化するように更新していくのはW。
        ここはどうみても設計が悪い。lambdaのスコープもよく
        わからなくなっているし。
        """
        y = self.forward(x)
        loss_W = lambda W: self.cost_function(y, t)

        grads = {}
        for key in self.params.keys():
            grads[key] = numerical_gradient(loss_W, self.params[key])

        return grads

    def forward(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = softmax(a3)

        return z3

    def backword(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        if np.isinf(b1).any() or np.isnan(b1).any():
            import pdb; pdb.set_trace()

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)

        grads = {}

        batch_size = t.shape[0]
        dout = (y - t) / batch_size

        grads["W3"] = np.dot(z2.T, dout)
        grads["b3"] = np.sum(dout, axis = 0)
        dz2 = np.dot(dout, W3.T)
        
        da2 = dz2 * (1.0 - z2) * z2
        grads["W2"] = np.dot(z1.T, da2)
        grads["b2"] = np.sum(da2, axis = 0)
        dz1 = np.dot(da2, W2.T)

        da1 = dz1 * (1.0 - z1) * z1
        grads["W1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis = 0)

        return grads

    def accuracy(self, x, t):
        y = np.argmax(self.forward(x), 1)
        _t = np.argmax(t, 1)

        return 1.0 * np.sum((_t == y)) / x.shape[0]
