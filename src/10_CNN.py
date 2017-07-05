# coding:utf-8

import sys, os 
sys.path.append(os.pardir)
import numpy as np

from dataset.mnist import load_mnist
from lib.my_neural_net import MyNeuralNet
from lib.common_functions import cross_entropy_error
from lib.layers import Affine, Conv
from lib.activation_functions import *


# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

# ネットワークを定義
network = MyNeuralNet(
            cost_function = cross_entropy_error
        )

# レイヤを追加
conv = Conv( activation_function = [Relu(), Pooling(2, 2, stride=2)],
              input_shape = (1, 28, 28),
              filter_shape = (30, 1, 5, 5) )
network.add_layer(conv)

x = x_train[:100]
# import pdb; pdb.set_trace()
hoge = network.forward(x)
print(hoge.shape)
