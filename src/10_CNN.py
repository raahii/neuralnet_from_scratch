# coding:utf-8

import sys, os 
sys.path.append(os.pardir)
import numpy as np
from tqdm import tqdm
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

# レイヤを追加
conv1 = Conv((30, 1, 5, 5))
pool1 = Pooling(2, 2, stride=2)
affine1 = Affine( 30*12*12, 100, "he" )
affine2 = Affine( 100, 10, "he" )

network.add_layer(conv1)
network.add_layer(Relu())
network.add_layer(pool1)

network.add_layer(affine1)
network.add_layer(Relu())

network.add_layer(affine2)
network.add_layer(Softmax())

# 学習
trainer = Trainer(network, x_train, t_train, x_test, t_test)
trainer.train(lr=0.1, epoch_num=100, batch_size=100)
trainer.savefig("dropout", "../data/dropout.png")
