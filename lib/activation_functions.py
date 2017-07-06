# coding : utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from lib.common_functions import *
from lib.utils import im2col, col2im

class Sigmoid:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, train_flg=True):
        self.x = x
        self.y = sigmoid(x)

        return self.y
    
    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y
        
        return dx

class Relu:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, train_flg=True):
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

    def forward(self, x, train_flg=True):
        self.x = x
        y = softmax(self.x)

        return y
    
    def backward(self, dy):
        batch_size = self.x.shape[0]
        dx = dy / batch_size
        
        return dx

class BatchNormalization:
    def __init__(self, lr=0.1):
        self.eps        = 1e-7
        self.batch_size = None
        self.lr         = lr

        self.x      = None
        self.x_norm = None
        self.x_c    = None
        self.mu     = None
        self.var    = None

        self.gamma  = None
        self.beta   = 0.0
        self.dbeta  = None
        self.dgamma = None

    def forward(self, x, train_flg=True):
        if x.ndim != 2:
            self.x = x.reshape(x.shape[0], -1)
        else:
            self.x = x

        self.batch_size = x.shape[0]
        self.input_dim = x.shape[1]

        if self.gamma is None:
            self.gamma = np.ones(self.input_dim, )

        self.mu = np.mean(x, axis = 0)
        self.std = np.std(x, axis = 0)
        self.xc = x - self.mu

        self.x_norm = self.xc / (self.std + self.eps)
        y = self.gamma * self.x_norm + self.beta

        return y

    def backward(self, dy):
        self.dbeta = np.sum(dy)
        self.dgamma = np.sum(self.x_norm * dy, axis = 0)

        dx_norm = dy * self.gamma

        dxc = dx_norm / self.std
        dstd = -np.sum(dx_norm * self.xc / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += 2.0 / self.batch_size * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        # dx = dxc * ( 1.0 - 1.0 / self.batch_size)

        # TODO: should refactor design
        self.gamma -= self.lr * self.dgamma
        self.beta  -= self.lr * self.dbeta

        return dx

class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate)

    def backward(self, dy):
        return dy * self.mask

class Pooling:
    def __init__(self, FH, FW, stride=1, padding=0):
        self.FH = FH
        self.FW = FW
        self.S = stride
        self.P = padding

    def forward(self, x):
        BS, C, IH, IW = x.shape
        OH = int( (IH+2*self.P-self.FH) / self.S + 1 )
        OW = int( (IW+2*self.P-self.FW) / self.S + 1 )

        col_x = im2col(x, self.FH, self.FW, self.S, self.P)
        col_x = col_x.reshape(-1, self.FW*self.FH)

        arg_max = np.argmax(col_x, axis=1)
        col_y = np.max(col_x, axis = 1)
        y = col_y.reshape(BS, OH, OW, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        return y

    def backward(self, dy):
        dy = dy.transpose(0, 2, 3, 1)
        
        pool_size = self.FH * self.FW
        dmax = np.zeros((dy.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dy.flatten()
        dmax = dmax.reshape(dy.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.FH, self.FW, self.S, self.P)
        
        return dx
