# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from lib.utils import img_show

def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    
    # # m = 60000, n = 784(28x28)
    # print(x_train.shape)
    # print(t_train.shape)
    
    img_num = 20
    img_table = np.zeros((28*img_num, 28*img_num))
    for i in range(img_num):
        for j in range(img_num):
            img = x_train[img_num*i+j]
            img = img.reshape(28, 28)
            img_table[28*i:28*i+28, 28*j:28*j+28] = img

    img_show(img_table)

if __name__=="__main__":
    main()
