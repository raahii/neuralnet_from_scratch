#speed coding : utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from lib.my_neural_net import MyNeuralNet
from lib.layers import Affine

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import tqdm

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = MyNeuralNet()

# レイヤを追加
network.add_layer(Affine(28*28, 50))

print(network.layers)

# forward
x = x_train[0]
y = network.forward(x)
print(y.shape)
