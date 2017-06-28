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
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = MyNeuralNet()

# レイヤを追加
network.add_layer(Affine(28*28, 50, Sigmoid()))
network.add_layer(Affine(50, 100, Sigmoid()))
network.add_layer(Affine(100, 10, Softmax()))

# 学習済みのパラメータをセット
with open("../dataset/sample_weight.pkl", 'rb') as f:
    params = pickle.load(f)

network.layers[0].set_params(params["W1"], params["b1"])
network.layers[1].set_params(params["W2"], params["b2"])
network.layers[2].set_params(params["W3"], params["b3"])

# 予測
train_size = x_test.shape[0]
batch_size = 100

iters_num = 10000
learning_rate = 0.1

loss_list = []
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    
    x = x_train[batch_mask]
    t = t_train[batch_mask]

    y = network.forward(x)
    grads = network.backward(y, t)

    for layer, grad in zip(network.layers, grads):
        layer.dW -= learning_rate * grad["dW"]
        layer.db -= learning_rate * grad["db"]
    

    loss_list.append(network.loss(y, t))

    plt.clf()
    x = np.array(range(1, len(loss_list)+1))
    plt.plot(x, loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    # plt.xlim(1, epoch_num*train_size+1)
    plt.draw()
    plt.pause(0.05)
