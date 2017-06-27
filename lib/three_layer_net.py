# coding : utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from lib.common_functions import sigmoid, softmax, img_show
from dataset.mnist import load_mnist
from PIL import Image
import pickle

class ThreeLayerNet:
    def __init__(self, input_num, hidden_num1, hidden_num2, output_num):
        self.iun  = input_num
        self.hun1 = hidden_num1
        self.hun2 = hidden_num2
        self.oun  = output_num
    
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

    def train(self):
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
