# coding:utf-8

import sys, os 
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from lib.my_neural_net import MyNeuralNet
from lib.common_functions import cross_entropy_error
from lib.layers import *
from lib.trainer import Trainer

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

# ネットワークを定義
network = MyNeuralNet(cost_function = cross_entropy_error)

conv1 = Conv( filter_shape = (96,  1, 3, 3) )
conv1 = Conv( filter_shape = (96, 1, 4, 4) )
conv2 = Conv( filter_shape = (256,  96, 3, 3) )
conv3 = Conv( filter_shape = (384, 256, 3, 3) )
conv4 = Conv( filter_shape = (384, 384, 3, 3) )
conv5 = Conv( filter_shape = (256, 384, 3, 3) )
out_size= 9
affine1 = Affine( 256*out_size**2, 256*out_size**2, "he" )
affine2 = Affine( 256*out_size**2, 10, "he" )

network.add_layer(conv1)
network.add_layer(Relu())
network.add_layer(LRN())
network.add_layer(Pooling(3, 3))

network.add_layer(conv2)
network.add_layer(Relu())
network.add_layer(LRN())
network.add_layer(Pooling(3, 3))

network.add_layer(conv3)
network.add_layer(conv4)
network.add_layer(Relu())
network.add_layer(LRN())
network.add_layer(Pooling(3, 3))

network.add_layer(conv5)
network.add_layer(Relu())
network.add_layer(LRN())
network.add_layer(Pooling(3, 3))

network.add_layer(affine1)
network.add_layer(Relu())

network.add_layer(affine2)
network.add_layer(Softmax())

# 学習
trainer = Trainer(network, x_train, t_train, x_test, t_test)
trainer.train(lr=0.1, epoch_num=20, batch_size=100, for_cnn=True)
trainer.savefig("aleexnet", "../data/alexnet.png")
