# coding : utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from lib.common_functions import cross_entropy_error
from lib.my_neural_net import MyNeuralNet
from lib.layers import *

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = MyNeuralNet(cross_entropy_error)

# レイヤを追加
network.add_layer(Affine(28*28, 50, "he"))
network.add_layer(Relu())
network.add_layer(Affine(50, 100, "he"))
network.add_layer(Relu())
network.add_layer(Affine(100, 100, "he"))
network.add_layer(Relu())
network.add_layer(Affine(100, 100, "he"))
network.add_layer(Relu())
network.add_layer(Affine(100, 100, "he"))
network.add_layer(Relu())
network.add_layer(Affine(100, 10, "he"))
network.add_layer(Softmax())

# 学習
train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(int(train_size / batch_size), 1)

epoch_num = 20
iters_num = iter_per_epoch * epoch_num
learning_rate = 0.1

loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    
    x = x_train[batch_mask]
    t = t_train[batch_mask]

    y = network.forward(x)
    network.backward(y, t)

    for layer in network.layers:
        layer.update(learning_rate)

    if i != 0 and i % iter_per_epoch == 0:
        loss_list.append(network.loss(y, t))
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        plt.clf()
        plt.subplot(1, 2, 1)
        x = np.array(range(1, len(loss_list)+1))
        plt.plot(x, loss_list)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim(1, epoch_num+1)

        plt.subplot(1, 2, 2)
        plt.plot(x, train_acc_list, "r", label="train_acc")
        plt.plot(x, test_acc_list, "b", label="test_acc")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.xlim(1, epoch_num+1)
        plt.legend()

        plt.draw()
        plt.pause(0.05)

plt.clf()
plt.subplot(1, 2, 1)
x = np.array(range(1, len(loss_list)+1))
plt.plot(x, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.xlim(1, epoch_num+1)

plt.subplot(1, 2, 2)
plt.plot(x, train_acc_list, "r", label="train_acc")
plt.plot(x, test_acc_list, "b", label="test_acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.xlim(1, epoch_num+1)
plt.legend()

plt.savefig("../data/deeper_network.png")
print("train acc: {}, test acc: {}".format(train_acc_list[-1], test_acc_list[-1]))
