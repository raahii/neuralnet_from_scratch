#speed coding : utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from lib.my_neural_net import MyNeuralNet
from lib.layers import Affine, Sigmoid, Softmax
import pickle

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import tqdm

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = MyNeuralNet()

# レイヤを追加
network.add_layer(Affine(28*28, 50))
network.add_layer(Sigmoid())
network.add_layer(Affine(50, 100))
network.add_layer(Sigmoid())
network.add_layer(Affine(100, 10))
network.add_layer(Softmax())

# 学習済みのパラメータをセット
with open("../dataset/sample_weight.pkl", 'rb') as f:
    params = pickle.load(f)
network.layers[0].set_params(params["W1"], params["b1"])
network.layers[2].set_params(params["W2"], params["b2"])
network.layers[4].set_params(params["W3"], params["b3"])

# 予測
batch_size = 100
correct_num = 0
data_size = x_test.shape[0]
for i in range(0, data_size, batch_size):
    x = x_test[i:i+batch_size]
    t = t_test[i:i+batch_size]
    y = np.argmax(network.forward(x), 1)
    correct_num += np.sum((t == y))

print("acc: {0:0.2f}%".format(100 * correct_num / data_size))

