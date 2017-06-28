# coding : utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from lib.common_functions import *

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

class Relu:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])

    def forward(self, x):
        self.x = x
        self.y = relu(self.x)

        return self.y

    def backward(self, dy):
        dx = dy.copy()
        dx[ self.x<=0 ] = 0

        return dx

class Softmax:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        y = softmax(self.x)

        return y
    
    def backward(self, dy):
        batch_size = self.x.shape[0]
        dx = dy / batch_size
        
        return dx
