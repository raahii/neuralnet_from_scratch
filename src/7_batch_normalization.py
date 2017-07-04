# coding : utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from lib.common_functions import cross_entropy_error
from lib.my_neural_net import MyNeuralNet
from lib.layers import Affine
from lib.activation_functions import Sigmoid, Softmax, Relu, BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
networks = {
            "normal": MyNeuralNet(cross_entropy_error),
            "batch_normalization": MyNeuralNet(cross_entropy_error),
           }

# レイヤを追加
LAYER_NUM = 6
network = networks["normal"]
network.add_layer(Affine(28*28, 100, Sigmoid(), "xavier"))
network.add_layer(Affine(100,   100, Sigmoid(), "xavier"))
network.add_layer(Affine(100,   100, Sigmoid(), "xavier"))
network.add_layer(Affine(100,   100, Sigmoid(), "xavier"))
network.add_layer(Affine(100,   100, Sigmoid(), "xavier"))
network.add_layer(Affine(100,   10,  Softmax(), "xavier"))

network = networks["batch_normalization"]
network.add_layer(Affine(28*28, 100, [BatchNormalization(), Sigmoid()], "xavier"))
network.add_layer(Affine(100,   100, [BatchNormalization(), Sigmoid()], "xavier"))
network.add_layer(Affine(100,   100, [BatchNormalization(), Sigmoid()], "xavier"))
network.add_layer(Affine(100,   100, [BatchNormalization(), Sigmoid()], "xavier"))
network.add_layer(Affine(100,   100, [BatchNormalization(), Sigmoid()], "xavier"))
network.add_layer(Affine(100,   10 , Softmax(),                         "xavier"))

# 学習
train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(int(train_size / batch_size), 1)

iters_num = iter_per_epoch * 50
learning_rate = 0.1

plt.figure(figsize=(12,8))
loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    
    x = x_train[batch_mask]
    t = t_train[batch_mask]

    y = network.forward(x)
    network.backward(y, t)

    for network in networks.values():
        y = network.forward(x)
        network.backward(y, t)

        for layer in network.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db

    if i % iter_per_epoch == 0:
        print("--- epoch {} ---".format(int(i/iter_per_epoch)))

        plt.clf()
        rows_num = len(networks.keys())
        columns_num = LAYER_NUM-1

        for row, n in enumerate(networks.items()):
            title, network = n

            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            print(title, ":", train_acc, test_acc)

            for column, layer in enumerate(network.layers[0:-1]):
                plt.subplot(rows_num, columns_num, columns_num*(row) + column+1)
                # plt.suptitle(title)
                plt.title("{}-layer".format(column+1))

                colors = ["r", "g", "b"]
                plt.hist(layer.activation_functions[-1].y.flatten(), bins=30, color=colors[row])

        plt.draw()
        plt.pause(0.05)
