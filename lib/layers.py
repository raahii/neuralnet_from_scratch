# coding : utf-8

import numpy as np

class Affine:
    def __init__(self, input_size, output_size):

        self.x = np.array([])
        self.y = np.array([])
        self.W = np.random.randn(input_size, output_size)

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.x, self.W)
        return self.y
