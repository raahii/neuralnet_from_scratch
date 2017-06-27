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
                       loss_function = cross_entropy_error):
        self.iun  = input_num
        self.hun1 = hidden_num1
        self.hun2 = hidden_num2
        self.oun  = output_num
        self.loss_function = loss_function
    
        self.params = {}
        self.params['W1'] = np.random.randn(self.iun, self.hun1)
        self.params['b1'] = np.random.randn(self.hun1, )
        self.params['W2'] = np.random.randn(self.hun1, self.hun2)
        self.params['b2'] = np.random.randn(self.hun2, )
        self.params['W3'] = np.random.randn(self.hun2, self.oun)
        self.params['b3'] = np.random.randn(self.oun, )

    def set_params(self, params):
        self.params['W1'] = params["W1"]
        self.params['b1'] = params["b1"]
        self.params['W2'] = params["W2"]
        self.params['b2'] = params["b2"]
        self.params['W3'] = params["W3"]
        self.params['b3'] = params["b3"]

    def loss(self, x, t):
        y = self.forward(x)
        return self.loss_function(y, t)

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
        loss_W = lambda W: self.loss_function(y, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
        grads["W3"] = numerical_gradient(loss_W, self.params["W3"])
        grads["b3"] = numerical_gradient(loss_W, self.params["b3"])

        return grads

    def train(self, x_train, t_train):
        pass

    def backword(self, x, t):
        pass

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
