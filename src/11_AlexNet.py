# coding:utf-8

import sys, os 
sys.path.append(os.pardir)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

conv1 = Conv( activation_function = [Relu()], filter_shape = (96, 1, 4, 4))
conv2 = Conv( activation_function = [Relu(), LRN(), Pooling(3, 3), Dropout(0.5)], filter_shape = (256,  96, 3, 3) )
conv3 = Conv( activation_function = [Relu(), LRN(), Pooling(3, 3), Dropout(0.6)], filter_shape = (384, 256, 3, 3) )
conv4 = Conv( activation_function = [Relu(), LRN(), Pooling(3, 3), Dropout(0.7)], filter_shape = (384, 384, 3, 3) )
conv5 = Conv( activation_function = [Relu(), LRN(), Pooling(3, 3), Dropout(0.8)], filter_shape = (256, 384, 3, 3) )
out_size= 9
affine1 = Affine( 256*out_size**2, 256*out_size**2, Relu(), "he" )
affine2 = Affine( 256*out_size**2, 10, Softmax(), "he" )

network.add_layer(conv1)
network.add_layer(conv2)
network.add_layer(conv3)
network.add_layer(conv4)
network.add_layer(conv5)
network.add_layer(affine1)
network.add_layer(affine2)

# 学習
train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(int(train_size / batch_size), 1)

iters_num = iter_per_epoch * 100
learning_rate = 0.1

plt.style.use("ggplot")
plt.figure(figsize=(12,8))
loss_list = []
train_acc_list = []
test_acc_list = []

for i in tqdm(range(iters_num)):
    batch_mask = np.random.choice(train_size, batch_size)

    x = x_train[batch_mask]
    t = t_train[batch_mask]

    y = network.forward(x)
    network.backward(y, t)

    for layer in network.layers:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db

    loss_list.append(network.loss(y, t))

    if i != 0 and i % iter_per_epoch == 0:
        plt.clf()

        # plt.subplot(1, 2, 1)
        x = np.array(range(1, len(loss_list)+1))
        plt.plot(x, loss_list)
        plt.xlabel("iter")
        plt.ylabel("loss")

        # train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        print(test_acc)
        # train_acc_list.append(train_acc)
        # test_acc_list.append(test_acc)
        # x = np.array(range(1, len(train_acc_list)+1))
        # plt.subplot(1, 2, 2)
        # plt.plot(x, train_acc_list)
        # plt.plot(x, test_acc_list)
        # plt.xlabel("iter")
        # plt.ylabel("acc")
        # plt.legend()

        plt.draw()
        plt.pause(0.01)

x = np.array(range(1, len(loss_list)+1))
plt.plot(x, loss_list)
plt.xlabel("iter")
plt.ylabel("loss")
plt.savefig("./power_of_cnn.png")

print("--------- last result -----------")
print("test_acc: {}".format(network.accuracy(x_test, t_test)))
# print("train_acc: {}".format(network.accuracy(x_train, t_train)))
print("---------------------------------")
