# coding: utf-8

import sys
import numpy as np

def conv(img, kernel):
    h, w   = img.shape
    kh, kw = kernel.shape

    if kernel.ndim != 1:
        kernel = kernel.reshape(-1)

    nh = (h-kh) + 1
    nw = (w-kw) + 1

    c = np.zeros((nw, nh))
    
    for i in range(nh):
        for j in range(nw):
            p = img[i:i+kh, j:j+kw].reshape(-1)
            c[i, j] = np.dot(p, kernel)

    return c

img = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
    ])
kernel = np.array([
    [2, 0, 1],
    [0, 1, 2],
    [1, 0, 2]
    ])

conved = conv(img, kernel)
print(conved)
