# coding: utf-8

import sys
import numpy as np

def conv(img, kernel, bias = 0, padding = 0, stride = 1):
    p = padding
    s = stride
    h, w   = img.shape
    fh, fw = kernel.shape
    nh = int((h + 2*p - fh) / s) + 1
    nw = int((w + 2*p - fw) / s) + 1

    if kernel.ndim != 1:
        kernel = kernel.reshape(-1)

    if padding != 0:
        _h, _w = h, w
        h, w = h+2*p, w+2*p
        new_img = np.zeros((h, w))
        new_img[p:p+_h, p:p+_w] = img
        img = new_img

    c = np.zeros((nh, nw))
    
    for i in range(0, nh):
        for j in range(0, nw):
            t = img[s*i:s*i+fh, s*j:s*j+fw].reshape(-1)
            # import pdb; pdb.set_trace()
            c[i, j] = np.dot(t, kernel)

    return c + bias

# img = np.array([
#     [1, 2, 3, 0],
#     [0, 1, 2, 3],
#     [3, 0, 1, 2],
#     [2, 3, 0, 1]
#     ])
img = np.array([
    [1, 2, 3, 0, 1, 2, 3],
    [0, 1, 2, 3, 0, 1, 2],
    [3, 0, 1, 2, 3, 0, 1],
    [2, 3, 0, 1, 2, 3, 0],
    [1, 2, 3, 0, 1, 2, 3],
    [0, 1, 2, 3, 0, 1, 2],
    [3, 0, 1, 2, 3, 0, 1],
    ])
kernel = np.array([
    [2, 0, 1],
    [0, 1, 2],
    [1, 0, 2]
    ])

conved = conv(img, kernel, stride=2)
print(conved)
