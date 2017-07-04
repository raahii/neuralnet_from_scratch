# coding: utf-8

import sys
import numpy as np

def conv(img, kernel, bias = 0, padding = 0, stride = 1):
    # change ndim
    if img.ndim == 2:
        img = img.reshape(((1, 1)+img.shape)[-4:])
    if kernel.ndim == 2:
        kernel = kernel.reshape(((1, 1)+kernel.shape)[-4:])

    p = padding
    s = stride
    batch_num, rd, rh, rw = img.shape
    fn, fd, fh, fw = kernel.shape
    nh = int((rh + 2*p - fh) / s) + 1
    nw = int((rw + 2*p - fw) / s) + 1

    if padding != 0:
        _h, _w = rh, rw
        rh, rw = rh+2*rp, rw+2*rp
        new_img = np.zeros((rh, rw))
        new_img[p:p+_h, p:p+_w] = img
        img = new_img

    ret = np.zeros((batch_num, fn, nh, nw))
    for bn in range(batch_num):
        for i in range(nh):
            for j in range(nw):
                for c in range(fn):
                    for d in range(fd):
                        # import pdb; pdb.set_trace()
                        t = img[bn, d, s*i:s*i+fh, s*j:s*j+fw]
                        ret[bn, c, i, j] += np.sum(kernel[c, d] * t)

    return ret + bias

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
