# coding : utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from lib.common_functions import cross_entropy_error
from lib.my_neural_net import MyNeuralNet
from lib.layers import Affine
from lib.activation_functions import Sigmoid, Softmax, Relu, Dropout

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
networks = {
        "normal": MyNeuralNet(cross_entropy_error),
        "dropout": MyNeuralNet(cross_entropy_error),
        }

# レイヤを追加
network = networks["dropout"]
network.add_layer(Affine(28*28, 100 , [Relu()], "he"))
network.add_layer(Affine(100   , 100, [Relu()], "he"))
network.add_layer(Affine(100   , 100, [Relu()], "he"))
network.add_layer(Affine(100   , 100, [Relu()], "he"))
network.add_layer(Affine(100   , 100, [Relu()], "he"))
network.add_layer(Affine(100   , 100, [Relu()], "he"))
network.add_layer(Affine(100   , 100, [Relu(), Dropout(0.8)], "he"))
network.add_layer(Affine(100   , 10 , Softmax(), "he"))

network = networks["normal"]
network.add_layer(Affine(28*28, 100 , Relu(), "he"))
network.add_layer(Affine(100  , 100 , Relu(), "he"))
network.add_layer(Affine(100  , 100 , Relu(), "he"))
network.add_layer(Affine(100  , 100 , Relu(), "he"))
network.add_layer(Affine(100  , 100 , Relu(), "he"))
network.add_layer(Affine(100  , 100 , Relu(), "he"))
network.add_layer(Affine(100  , 100 , Relu(), "he"))
network.add_layer(Affine(100   , 10 , Softmax(), "he"))

# 学習
train_size = 100
batch_size = 20

iter_per_epoch = max(int(train_size / batch_size), 1)

iters_num = iter_per_epoch * 200
learning_rate = 0.1

plt.figure(figsize=(12,8))

loss_list      = {}
train_acc_list = {}
test_acc_list  = {}

# 結果を格納する変数を初期化
for name in networks.keys():
    loss_list[name] = []
    train_acc_list[name] = []
    test_acc_list[name] = []

# 学習
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    
    x = x_train[batch_mask]
    t = t_train[batch_mask]
    
    for name, network in networks.items():
        y = network.forward(x)
        network.backward(y, t)

        for layer in network.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db
        
        loss_list[name].append(network.loss(y, t))

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc  = network.accuracy(x_test, t_test)
            train_acc_list[name].append(train_acc)
            test_acc_list[name].append(test_acc)

    if i % iter_per_epoch == 0:
        plt.clf()
        l = len(networks.keys())
        plt.suptitle("Comparison parameter learning with dropout or not\n(top: with dropout, bottom: without dropout)")
        for i, name in enumerate(networks.keys()):
            plt.subplot(2, 2, l*i+1)
            plt.title(name)
            x = np.array(range(1, len(loss_list[name])+1))
            plt.plot(x, loss_list[name])
            plt.xlabel("epoch")
            plt.ylabel("loss")

            plt.title(name)
            plt.subplot(2, 2, l*i+l)
            x = np.array(range(1, len(train_acc_list[name])+1))
            plt.plot(x, train_acc_list[name], label="train_acc")
            plt.plot(x, test_acc_list[name], label="test_acc")
            plt.xlabel("epoch")
            plt.ylabel("acc")
            plt.legend()

        plt.draw()
        plt.pause(0.01)
